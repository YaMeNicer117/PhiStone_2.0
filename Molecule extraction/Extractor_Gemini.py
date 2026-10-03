import os
import shutil
import json
import time
import io
import re
import functools
import threading
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from openpyxl.styles import PatternFill
import pandas as pd
import concurrent.futures
import fitz  # PyMuPDF
from tqdm import tqdm
from py2opsin import py2opsin
import pubchempy as pcp
from rdkit import Chem
from google import genai
from google.genai import types
import signal 
import sys
import gc
import faulthandler
faulthandler.enable()
fitz.TOOLS.mupdf_display_errors(False)     

# ================= 1. 全局配置 (超参数与常量) =================

API_KEY = "sk-xxxx"

# --- 模型选择 ---
MODEL_FLASH_NAME = "gemini-3-flash-preview"
MODEL_PRO_NAME = "gemini-3.1-pro-preview"

# --- 并发与速率控制 ---
MAX_CONCURRENT_FOLDERS = 5        # 同时处理的文件夹数量 (多线程并发度)
API_CALLS_PER_MINUTE = 1000       # API 每分钟最大调用次数（全局共享，防止 429 报错）
API_MAX_RETRIES = 5               # API 请求失败时的最大重试次数
API_INITIAL_DELAY = 10            # API 重试的初始等待时间（秒，后续按指数退避递增）
PDF_TIMEOUT_SECONDS = 600         # ：单篇文献最大处理时间限制 (秒)，600秒即10分钟

# --- 运行逻辑与截断控制 ---
ABSOLUTE_MAX_PAGES = 36          # 单篇文献最多扫描提取的页面总数 (防止超长综述耗尽大模型上下文)
MAX_PAGES_TO_PRO = 12            # 单次发给 Pro 模型的最大图片数量 (多余的会分批续传)
TEXT_BATCH_SIZE = 6              # 文本分类时，每批处理的页面数
TEXT_MIN_LENGTH = 200            # 判断页面是否含有足够文本的字符数下限
MAX_CONTINUATIONS = 3            # 大模型输出 JSON 截断时，允许启动“断点续传”的最大次数
MAX_EXCEL_RETRIES = 10            # 遇到 Excel 文件被人工打开占用 (PermissionError) 时的最大等待重试次数

# --- 图像渲染参数 ---
IMG_DPI_FLASH = 72               # Flash 模型分类使用的低分辨率 DPI
IMG_DPI_PRO = 360                # Pro 模型精确提取使用的高分辨率 DPI
MAX_WORKERS_RENDER = 3          # PyMuPDF 并发渲染图片的最大工作线程数


# --- 路径与文件设置 ---
INPUT_BASE_FOLDER = "kept_pdfs"        # 存放多个子文件夹的根目录
OUTPUT_EXCEL_FOLDER = "output_excels"  # 存放生成的 xlsx 文件的目录
SUCCESS_FOLDER = "processed_success"   # 处理成功的 PDF 移动到此处
FAIL_FOLDER = "processed_fail"         # 处理失败或无数据的 PDF 移动到此处
REVIEW_FOLDER = "processed_reviews"
LOW_IF_FOLDER = "processed_low_if"     # 专门存放 IF < 1 或 N/A 的文件夹
TIMEOUT_FOLDER = "processed_timeout"   # 处理超时的 PDF 移动到此处
SHEET_NAME = "Extracted_Results"       # Excel 写入的 Sheet 名称
JCR_FILE_PATH = "JCR完整版.xlsx"       # 本地 JCR 影响因子文件路径 
DYNAMIC_IF_CACHE_FILE = "dynamic_if_cache.txt"

# ================= 2. 日志配置 =================
def silent_force_quit(signum, frame):
    os._exit(1)

signal.signal(signal.SIGINT, silent_force_quit)

class MuteAFCFillter(logging.Filter):
    """自定义过滤器：专门拦截并屏蔽 AFC 相关的废话日志"""
    def filter(self, record):
        return "AFC is enabled" not in record.getMessage()

# 基础配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

# 给所有的日志输出渠道装上我们的消音器
for handler in logging.root.handlers:
    handler.addFilter(MuteAFCFillter())

# 屏蔽其他底层网络库的常规唠叨
logging.getLogger("httpx").setLevel(logging.WARNING)


# ================= 3. 线程安全的全局速率限制器 =================

class RateLimiter:
    """令牌桶算法：控制全局 API 调用频率，线程安全"""
    def __init__(self, calls_per_minute: int):
        self._interval = 60.0 / max(calls_per_minute, 1)
        self._lock = threading.Lock()
        self._last_call = 0.0

    def acquire(self):
        with self._lock:
            now = time.monotonic()
            wait = self._interval - (now - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()

class APIConnectionError(Exception):
    pass

def retry_api():
    """API 请求重试装饰器，支持指数退避，遇到额度耗尽直接强杀进程"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            delay = API_INITIAL_DELAY
            for attempt in range(API_MAX_RETRIES):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    error_msg = str(e).lower()
                    
                    if "429" in error_msg or "quota" in error_msg or "exhausted" in error_msg:
                        print("\n") 
                        log.critical("=====================================================")
                        log.critical(f"❌ [FATAL ERROR] API 额度已耗尽或被服务商严重限流！")
                        sys.stdout.flush()
                        with open("CRASH_REPORT.txt", "w") as f:
                            f.write(f"Killed due to API Limit/429. Error: {e}")
                        os._exit(1) 
                    
                    # 正常的网络波动或超时，继续执行重试逻辑
                    log.warning(f"[API Retry {attempt+1}/{API_MAX_RETRIES}] {func.__name__}: {e}")
                    if attempt == API_MAX_RETRIES - 1:
                        log.error(f"[API Failed] {func.__name__} 达到最大重试次数。")
                        return None
                    time.sleep(delay)
                    delay *= 2
            return None
        return wrapper
    return decorator
# ================= 4. 初始化与离线词典加载 =================

client = genai.Client(
    api_key=API_KEY,
    http_options={
        'api_version': 'v1beta',
        'base_url': 'https://api.viviai.top',
    }
)

rate_limiter = RateLimiter(API_CALLS_PER_MINUTE)

for folder in [INPUT_BASE_FOLDER, OUTPUT_EXCEL_FOLDER, SUCCESS_FOLDER, FAIL_FOLDER, REVIEW_FOLDER, TIMEOUT_FOLDER, LOW_IF_FOLDER]: 
    os.makedirs(folder, exist_ok=True)

# --- 构建全局 JCR 离线词典 ---
JCR_DICT = {}
def load_jcr_dict():
    """从指定的列名加载 JCR 字典，支持简称和全称的混合匹配"""
    if os.path.exists(JCR_FILE_PATH):
        try:
            df = pd.read_excel(JCR_FILE_PATH, keep_default_na=False)

            target_col = "Journal Name"
            if_col = "JIF 2024"
            
            if target_col in df.columns and if_col in df.columns:
                keys = df[target_col].astype(str).str.upper().str.replace(r'[^A-Z0-9]', '', regex=True)
                values = df[if_col]
                
                for k, v in zip(keys, values):
                    if k and k != "NAN":
                        JCR_DICT[k] = v 
                log.info(f"✅ Successfully loaded {len(JCR_DICT)} journal mappings from '{JCR_FILE_PATH}'.")
            else:
                log.error(f"❌ Column names mismatch! Need '{target_col}' and '{if_col}'. Found: {df.columns.tolist()}")
        except Exception as e:
            log.error(f"❌ Failed to load JCR file: {e}")

# 初始化执行加载
load_jcr_dict()

# ================= 4.5 动态 IF 缓存与大模型查询 =================
DYNAMIC_IF_CACHE = {}

def load_dynamic_if_cache():
    """加载大模型历史查询过的大模型 IF 缓存"""
    global DYNAMIC_IF_CACHE
    if os.path.exists(DYNAMIC_IF_CACHE_FILE):
        try:
            with open(DYNAMIC_IF_CACHE_FILE, 'r', encoding='utf-8') as f:
                DYNAMIC_IF_CACHE = json.load(f)
            log.info(f"✅ Successfully loaded {len(DYNAMIC_IF_CACHE)} dynamic IF mappings from '{DYNAMIC_IF_CACHE_FILE}'.")
        except Exception as e:
            log.error(f"❌ Failed to load dynamic IF cache: {e}")

load_dynamic_if_cache()

def save_dynamic_if_cache():
    """保存最新查询的影响因子到本地 txt (采用 JSON 格式以保结构)"""
    try:
        with open(DYNAMIC_IF_CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(DYNAMIC_IF_CACHE, f, indent=4, ensure_ascii=False)
    except Exception as e:
        log.error(f"❌ Failed to save dynamic IF cache: {e}")

@retry_api()
def fetch_if_from_pro(journal_abbr, journal_full):
    """当本地 JCR 缺失时，呼叫 Pro 模型作为学术代理查询影响因子"""
    cache_key = f"{journal_abbr} | {journal_full}"
    
    # 优先查缓存
    if cache_key in DYNAMIC_IF_CACHE:
        return DYNAMIC_IF_CACHE[cache_key]
    
    prompt = f"""
    You are an expert academic librarian. I need the latest Clarivate Journal Impact Factor (IF) for this journal:
    Abbreviation: "{journal_abbr}"
    Full Name: "{journal_full}"
    
    - If you know the exact or approximate numerical Impact Factor, reply STRICTLY with the number only (e.g., 4.5, 12.3).
    - If this is a very obscure journal, a predatory journal, not indexed in SCI/JCR, or you cannot confidently determine the IF, reply STRICTLY with "NA". 
    - DO NOT hallucinate or guess. DO NOT output any other text or explanation.
    """
    
    rate_limiter.acquire()
    response = client.models.generate_content(
        model=MODEL_PRO_NAME,
        contents=[prompt],
        config=types.GenerateContentConfig(temperature=0.0) # 极低温度，锁死大模型的发散空间，防止幻觉
    )
    
    res_text = response.text.strip().upper()
    final_if = "NA"
    
    if "NA" not in res_text and "UNKNOWN" not in res_text:
        import re
        match = re.search(r'\d+(\.\d+)?', res_text)
        if match:
            final_if = float(match.group(0))
            
    # 写入内存并保存到文件
    DYNAMIC_IF_CACHE[cache_key] = final_if
    save_dynamic_if_cache()
    
    return final_if

# ================= 5. Prompt 库 =================
PROMPT_METADATA = """
Extract the DOI, the Journal's UPPERCASE ABBREVIATION, the Journal's FULL NAME, the Publication Date (Year), and the Article Type from this page.
Classify Article Type strictly as either "Review" or "Research".

### CRITICAL CLASSIFICATION RULES:
- **Research**: The article reports NEW, ORIGINAL experimental work. It typically contains or introduces sections like "Materials and Methods", "Experimental Section", "Synthesis", or primary biological assay data. Even if the introduction contains a literature review, if the paper aims to present original lab data or novel compounds, it MUST be "Research".
- **Review**: The article PURELY summarizes existing published literature (e.g., systematic reviews, overviews) and contains NO new laboratory experiments or chemical syntheses of its own. 
- **DEFAULT RULE**: When in doubt, or if you are not 100% sure it is a pure review, default to "Research".

Return STRICTLY valid JSON: 
{
  "doi": "10.1021/jmxxxx", 
  "journal_abbreviation": "J MED CHEM", 
  "journal_full_name": "Journal of Medicinal Chemistry",
  "publication_date": "2023",
  "article_type": "Research"
}
If any field is missing, use "Unknown".
"""

PROMPT_CLASSIFY_BATCH_TEXT = """
Analyze the text content of these document pages to identify pages containing PRIMARY DATA. 
Pay special attention to BOTH English and Chinese academic terminologies.
Classify each page into ONE of the following categories:

- **STRUCTURE**: Pages containing chemical structure definitions that are essential context for understanding compounds:
    - Synthetic Schemes ("Scheme 1", "路线 1", "合成路线") depicting scaffolds + R-groups / substituent tables ("取代基表").
    - Figures showing numbered chemical structures ("Figure 1", "图 1", "Compound 5a", "化合物 5a").
    - General synthetic procedures, Experimental sections, or Chemistry sections ("Experimental", "Synthesis", "合成步骤", "通法") that define the core scaffold or list formal IUPAC names and NMR data.
- **DATA**: Pages containing biological activity results:
    - Tables ("Table 1", "表 1", "附表") with headers like "MIC", "最小抑菌浓度", "IC50", "半数抑制浓度", "% Inhibition", "抑菌率", "Zone of Inhibition", "抑菌圈", "Activity", "活性", "In vitro", "体外".
    - Text paragraphs describing biological results (e.g., "Compound 5a exhibited potent activity...", "化合物5a表现出显著的抑菌活性...").
    - Reference drugs / Positive controls ("阳性对照", "参比药物") listed with activity data.
- **BOTH**: Pages containing BOTH structure definitions AND activity data.
- **IRRELEVANT**: Pages containing ONLY: Title Page, Abstract ("摘要"), Introduction ("前言", "引言"), Conclusion ("结论"), Acknowledgement ("致谢"), Funding ("基金资助"), Conflict of Interest, or References/Bibliography ("参考文献").

Return STRICTLY valid JSON: {"Page_1": "STRUCTURE", "Page_2": "DATA", "Page_3": "IRRELEVANT"}
"""

PROMPT_CLASSIFY_VISION = """
Look at this page image and classify it into ONE category:
- **STRUCTURE**: Contains chemical structure diagrams, Schemes, or R-group definition tables.
- **DATA**: Contains biological activity data tables or results text.
- **BOTH**: Contains both structure diagrams AND activity data.
- **IRRELEVANT**: Title page, text-only Abstract, Introduction, Conclusion, or Reference list.
Output JSON: {"category": "STRUCTURE"} or {"category": "DATA"} or {"category": "BOTH"} or {"category": "IRRELEVANT"}
"""

PROMPT_EXTRACT = """
You are an expert Medicinal Chemist. Extract anti-tuberculosis (anti-mycobacterial) compound data from these High-Res Page Images.

### SOURCE MATERIAL AWARENESS
- **Mixed Content**: Pages often contain Text, Tables, and Figures. Scan the *entire* image.
- **Table Continuations**: Be highly aware that tables often span multiple pages. You may receive an image of a table without headers (a continuation from a previous page). If you see a grid of numbers without headers, logically infer the columns (e.g., Compound ID, MIC, IC50) based on the typical structure of biological data tables.
- **Data Priority**:
  1. **Tables**: Primary source. Extract ALL metrics found (MIC, IC50, Inhibition).
  2. **Text**: Secondary source. Use if Table is missing.
  3. **Schemes**: Use these to resolve structure definitions (Scaffolds + R-groups).

### TASKS (Chain of Thought) - STRICT ORDER
1.  **Scan for Target & Activity (Primary Filter)**:
    - ONLY look for data tested against **Mycobacterium** species (e.g., *M. tuberculosis*, *M. smegmatis*, *M. bovis*, H37Rv).
    - Identify compound IDs that have explicitly reported NUMERICAL activity against these specific targets.
    - **CRITICAL**: You MUST extract the compound EVEN IF the activity result lacks a specific number and is reported as "NA","-", or with qualitative words like "inactive" or "no inhibition". 
    - In these cases, put the exact original text (e.g., "NA") into the value field (e.g., `mic_value`), AND you MUST set the flag `"is_qualitative_inactive": true` for this compound.
    - **IGNORE** compounds that only have data for other bacteria (e.g., Gram-negative/positive), fungi, or general cytotoxicity.

2.  **Resolve Structure (For Targeted IDs ONLY)**:
    - For the compound IDs identified in Step 1, resolve their chemical structures.
    - Match ID -> Scheme Image (Scaffold + R-groups) -> Text (IUPAC).
    - For Reference Drugs (e.g., Isoniazid, Rifampicin): Provide standard SMILES in `predicted_smiles`.

3.  **Extract Activity Data (Universal Target Lock)**:
    - **CRITICAL**: Extract numerical data ONLY if the specific test column/row is explicitly against a *Mycobacterium* strain in a WHOLE-CELL (in vitro) assay.
    - **NO DATA FILTERING (CRITICAL VETO)**: You MUST extract EVERY SINGLE COMPOUND listed in the table that has a numerical value against the target strain, REGARDLESS of how high or poor the MIC/IC50 value is (e.g., even if MIC is > 900 µM). DO NOT selectively extract only the "potent" or "active" compounds highlighted in the text. DO NOT act as a filter. If it is in the table with a number, it MUST be in your JSON output.
    - **IGNORE OFF-TARGET**: If a table contains columns for mammalian cell cytotoxicity (e.g., CC50, Vero, HeLa, RAW) or other pathogens (e.g., E. coli, S. aureus, Fungi), you MUST completely IGNORE those values. 
    - **IGNORE ENZYME ASSAYS**: DO NOT extract data from pure enzyme inhibition assays. If you see metrics like Ki (Inhibition constant) or target-specific enzyme tests (e.g., Mtb Ung, InhA, KatG), IGNORE THEM completely. We ONLY want whole-cell antimycobacterial efficacy.
    - **Extract exact values**: If you see inequalities (e.g., "<0.1", ">64") or ranges (e.g., "0.5-1.0"), extract them exactly as written in the text/table. If tested against multiple mycobacterial strains, create a SEPARATE JSON object for each strain.
    - **QSAR ALERT**: For QSAR papers, specifically watch for pMIC or log(1/MIC) columns and place them STRICTLY in the 'pmic_value' field.
    - **TIME-POINT RULES**: If a table reports MIC/IC50 values across multiple incubation time points (e.g., 7 days, 14 days, 21 days), you MUST STRICTLY extract the data for the 14-day period. If experiment time is unavailable, extract the data for the longest incubation period reported.
    - **DUAL UNITS / PARENTHESES RESOLUTION**: If a table reports dual values in a single cell (e.g., "2.5 (8.6)" where the primary unit is µg/mL and the parenthetical is µmol/L), you MUST STRIP OUT the parentheses. ONLY extract the primary number (e.g., extract "2.5") into the _value field. Never extract parentheses or the secondary number.
    
### STRAIN RECOGNITION & HAZARD HIERARCHY
Identify the specific strain for each activity measurement. Be highly aware of common abbreviations. Map them to the `hazard_level` field using this exact hierarchy (A is highest):
* **Level A (Highest)**: M. tuberculosis (M. tb, MTB, H37Rv, Erdman, MDR-TB, XDR-TB), M. bovis (BCG).
* **Level B**: M. abscessus (M. abs), M. avium complex (MAC).
* **Level C**: M. marinum, M. kansasii.
* **Level D (Lowest)**: M. smegmatis (M. smeg, mc2 155), M. aurum, M. phlei.

### OUTPUT FORMAT (JSON List)
[
  {
    "compound_id": "5a",                      // STRICTLY the local ID used in the paper (e.g., 5a, IV, 3).
    "target_strain": "M. tuberculosis H37Rv", // Extract written name (even if abbreviated)
    "hazard_level": "A",                     // Must be A, B, C, or D based on the hierarchy above
    "iupac_name": "Full IUPAC or Trivial name. Priority 1: Strictly check the Experimental/Synthesis section for the formal systematic name. Priority 2: IF not explicitly named in the text, you MAY autonomously infer the correct IUPAC name based on the core scaffold and R-groups in the Scheme.CRITICAL: For known Reference Drugs (e.g., 'Bedaquiline', 'Macozinone', 'Isoniazid'), you MUST put their common name here so they can be queried in databases. STRICTLY DO NOT put local IDs (e.g., 'Compound 5a', 'Derivative IV') or loose descriptive phrases (e.g., 'usnic acid arginine conjugate') here. Use trivial names only if formal names are completely unavailable. If you cannot confidently extract or infer a true chemical name, leave as empty string ''. But Do NOT let an empty name stop you from generating the SMILES in the next step.",
    "is_standard_iupac": false,  // CRITICAL FLAG: Set to true ONLY if the extracted 'iupac_name' is a strictly formal, machine-readable systematic IUPAC name (e.g., with proper brackets, numbers, and standard English suffixes like '-yl'). Set to false if it is a trivial name (e.g., 'dihydrosanguinarine'), contains typos/local suffixes (e.g., '-il'), or is a descriptive phrase.
    "predicted_smiles": "Standard SMILES. STRATEGY HIERARCHY: 1. If you extracted a known trivial name or formal IUPAC name, first attempt to generate the SMILES based purely on your chemical knowledge of that name. 2. If the name is completely unknown to you, ONLY THEN attempt to visually infer the SMILES from the structural Scheme images. Relax the '100% confident' rule for standard drug-like small molecules (like quinolines or pyridines). ONLY leave this blank if the molecule is a massive macrolide, complex peptide, or the image is completely unreadable.",
    // --- Data Fields (STRICTLY for Mycobacterium strains ONLY) ---
    // DO NOT extract any values here if they represent Cytotoxicity (Vero, etc.) or non-mycobacterial activity.
    // CRITICAL WARNING:  If you see MIC50, IC90, CC50, TC50, or any Cytotoxicity data, IGNORE IT. Do not confuse the type of data with the type of unit.Only extract IC50, MIC(90), PIC50, and several other non-specific activity values.
    // EXTRACT UNITS EXACTLY AS WRITTEN IN THE HEADER. Pay extreme attention to the prefix (nM vs µM vs mM). Do not assume defaults.
    "mic_value": "0.5",       // STRICTLY for standard MIC, MIC90, or MIC99. (NEVER extract MIC50 here).
    "mic_unit": "µg/mL",      // e.g., "µg/mL", "µM"
    "ic50_value": "",         // STRICTLY for IC50 against Mycobacterium. NEVER put MIC50 here! Keep empty if it's cytotoxicity.
    "ic50_unit": "",
    "inhibition_value": "95", // Just the number
    "inhibition_conc": "10µM",// Concentration used (e.g., "at 10 µg/mL")
    "pic50_value": "",        // For pIC50 or -log(IC50).Do not mix logarithmic values (pIC50/pMIC) with standard concentration values (IC50/MIC).
    "pmic_value": "",         // CRITICAL FOR QSAR: Extract here if the table reports pMIC, log(1/MIC), or -log(MIC)
    "zoi_value": "",
    "is_qualitative_inactive": false,  // CRITICAL FLAG: Set to true ONLY IF the activity is reported as "NA", "ND", "-", "inactive", etc. Otherwise false,
    "ignored_toxicity_or_other_data": "Put any CC50, TC50, MIC50, or off-target bacterial data here to keep the main fields clean.", 
    "extraction_source": "Table 1 & Scheme 1",
    "needs_check": false
  }
]

### CLASSIFICATION RUBRIC (For 'qualitative_level')
* **Excellent**: MIC/IC50 ≤ 1.0 µg/mL (or ≤ 2.0 µM) OR Inhibition ≥ 90%.
* **Good**: MIC/IC50 > 1.0 AND ≤ 10.0 µg/mL OR Inhibition ≥ 80% and < 90%.
* **Moderate**: MIC/IC50 > 10.0 AND ≤ 64.0 µg/mL OR Inhibition ≥ 20% and < 80%.
* **Inactive**: MIC/IC50 > 64.0 µg/mL OR Inhibition < 20% OR Text says "inactive/not active".

### STRICT EXCLUSION RULES (VETO)
1. **Target Specificity**: If a compound's extracted activity is NOT against a *Mycobacterium* species, DO NOT extract the compound.
2. **Missing Numerical Data**: (RELAXED) As long as a compound was explicitly tested against a targeted Mycobacterium strain, extract it. If the table reports "NA", "-", or "inactive", record that exact text and set the `is_qualitative_inactive` flag to true. Do not skip it.
3. **Data Precision Priority**: If the activity data of the same compound is extracted from multiple sources in the literature, then the more precise data takes higher priority.
"""

# ================= 6. 辅助函数 =================
def parse_numerical_value(val_str):
    """解析带有不等号或区间的数值为浮点数"""
    if not val_str: return None
    try:
        clean = str(val_str).replace('>', '').replace('<', '').replace('~', '').replace('=', '').strip()
        if '-' in clean:
            parts = clean.split('-')
            return (float(parts[0]) + float(parts[1])) / 2
        return float(clean)
    except (ValueError, IndexError): 
        return None
    


def standardize_smiles(smiles):
    """验证并返回 RDKit 标准化的 Canonical SMILES"""
    if not smiles or len(str(smiles)) < 2 or str(smiles).lower() in ["n/a", "none", "null"]: 
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return None
        # 保留原有的同位素限制逻辑
        isotope_count = sum(1 for atom in mol.GetAtoms() if atom.GetIsotope() > 0)
        if isotope_count >= 3: return None
        
        # 核心改动：输出 RDKit 标准化后的 SMILES，消除异构表达差异
        return Chem.MolToSmiles(mol) 
    except Exception: 
        return None


# 常见结核药物离线字典 (扩充版，包含新型靶点如 ATP 合成酶抑制剂等)
COMMON_TB_DRUGS = {
    # --- 经典一线/二线抗结核药物 ---
    "isoniazid": "NNC(C1C=CN=CC=1)=O",
    "rifampicin": "O=C(/C(C)=C\C=C\[C@H](C)[C@H](O)[C@@H](C)[C@@H](O)[C@H]([C@H](OC(C)=O)[C@H](C)[C@@H](OC)/C=C/O[C@@]1(C)O2)C)NC3=C(O)C4=C(O)C(C)=C2C(C1=O)=C4C(O)=C3/C=N/N5CCN(C)CC5",
    "ethambutol": "CC[C@@H](CO)NCCN[C@@H](CC)CO",
    "pyrazinamide": "NC(C1=NC=CN=C1)=O",
    "streptomycin": "O[C@H]1[C@H](O[C@@H]2[C@@H](O[C@@H]3[C@H](NC)[C@@H](O)[C@@H](O)[C@H](CO)O3)[C@](O)(C=O)[C@@H](C)O2)[C@@H](NC(N)=N)[C@H](O)[C@@H](NC(N)=N)[C@@H]1O",
    
    # --- 氟喹诺酮类 (Fluoroquinolones) ---
    "ciprofloxacin": "O=C(O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O",
    "ofloxacin": "CC1COc2c(N3CCN(C)CC3)c(F)cc3c(=O)c(C(=O)O)cn1c23",
    "levofloxacin": "C[C@H]1COc2c(N3CCN(C)CC3)c(F)cc3c(=O)c(C(=O)O)cn1c23",
    "moxifloxacin": "OC(C1=CN(C2=C(C(N3C[C@@]4([H])[C@@](NCCC4)([H])C3)=C(F)C=C2C1=O)OC)C5CC5)=O",
    
    # --- 新型抗结核明星药物 (ATP合成酶抑制剂、硝基咪唑类等) ---
    "bedaquiline": "CO[C@](CCN(C)C)(C1=C2C=CC=CC2=CC=C1)[C@@H](C3=CC4=CC(Br)=CC=C4N=C3OC)C5=CC=CC=C5", # ATP合成酶抑制剂
    "delamanid": "CC1(COc2ccc(N3CCC(Oc4ccc(OC(F)(F)F)cc4)CC3)cc2)Cn2cc([N+](=O)[O-])nc2O1", # 细胞壁合成抑制剂
    "pretomanid": "O=[N+]([O-])c1cn2c(n1)OC[C@@H](OCc1ccc(OC(F)(F)F)cc1)C2", # PA-824
    "macozinone": "C1CCC(CC1)CN2CCN(CC2)C3=NC(=O)C4=C(S3)C(=CC(=C4)C(F)(F)F)[N+](=O)[O-]", # PBTZ169 DprE1抑制剂
    "btz043": "C[C@H]1COC2(CCN(c3nc(=O)c4cc(C(F)(F)F)cc([N+](=O)[O-])c4s3)CC2)O1",
    "btz-043": "C[C@H]1COC2(CCN(c3nc(=O)c4cc(C(F)(F)F)cc([N+](=O)[O-])c4s3)CC2)O1",
    "linezolid": "CC(=O)NC[C@H]1CN(c2ccc(N3CCOCC3)c(F)c2)C(=O)O1", # 恶唑烷酮类
    "clofazimine": "CC(C)/N=c1\cc2n(-c3ccc(Cl)cc3)c3ccccc3nc-2cc1Nc1ccc(Cl)cc1", # 吩嗪类

    # === 经典大环内酯类 (Macrolides，抗非结核分枝杆菌 NTM 主力) ===
    "erythromycin": "CC[C@H]1OC(=O)[C@H](C)[C@@H](O[C@H]2CC(C)(OC)[C@@H](O)[C@H](C)O2)[C@H](C)[C@@H](O[C@@H]2O[C@H](C)C[C@H](N(C)C)[C@H]2O)[C@](C)(O)C[C@@H](C)C(=O)[C@H](C)[C@@H](O)[C@]1(C)O", # 红霉素
    "azithromycin": "CC[C@H]1OC(=O)[C@H](C)[C@@H](O[C@H]2CC(C)(OC)[C@@H](O)[C@H](C)O2)[C@H](C)[C@@H](O[C@@H]2O[C@H](C)C[C@H](N(C)C)[C@H]2O)[C@](C)(O)C[C@@H](C)CN(C)[C@@H](C)[C@@H](O)[C@]1(C)O", # 阿奇霉素 (15元环氮杂化物)
    "clarithromycin": "CC[C@H]1OC(=O)[C@H](C)[C@@H](O[C@H]2CC(C)(OC)[C@@H](O)[C@H](C)O2)[C@H](C)[C@@H](O[C@@H]2O[C@H](C)C[C@H](N(C)C)[C@H]2O)[C@@](C)(OC)C[C@@H](C)C(=O)[C@H](C)[C@@H](O)[C@]1(C)O", # 克拉霉素
    "roxithromycin": "CC[C@H]1OC(=O)[C@H](C)[C@@H](O[C@H]2C[C@@](C)(OC)[C@@H](O)[C@H](C)O2)[C@H](C)[C@@H](O[C@@H]2O[C@H](C)C[C@H](N(C)C)[C@H]2O)[C@@](C)(O)C[C@@H](C)/C(=N\OCOCCOC)[C@H](C)[C@@H](O)[C@]1(C)O", # 罗红霉素

    # === 安莎霉素类 / 大环内酰胺 (Ansamycins，抗结核核心) ===
    "rifapentine": "COC(=O)[C@H]1[C@H](C)[C@@H](OC)C=CO[C@@]2(C)Oc3c(C)c(O)c4c(O)c(/C=N/N5CCN(C6CCCC6)CC5)c(c(O)c4c3C2=O)NC(=O)C(C)=CC=C[C@H](C)[C@@H](O)[C@H](C)[C@@H](O)[C@@H]1C", # 利福喷丁
    "rifabutin": "CO[C@@H]1C=CO[C@@]2(C)Oc3c(C)c(O)c4c(c3C2=O)C2=NC3(CCN(CC(C)C)CC3)NC2=C(NC(=O)C(C)=CC=C[C@H](C)[C@H](O)[C@@H](C)[C@@H](O)[C@@H](C)[C@H](OC(C)=O)[C@@H]1C)C4=O", # 利福布汀

    # === 其他重要大环抗菌分子 (极易触发 API 404 或 AI 幻觉) ===
    "fidaxomicin": "CC[C@H]1/C=C(/[C@H](C/C=C/C=C(/C(=O)O[C@@H](C/C=C(/C=C(/[C@@H]1O[C@H]2[C@H]([C@H]([C@@H](C(O2)(C)C)OC(=O)C(C)C)O)O)\C)\C)[C@@H](C)O)\CO[C@H]3[C@H]([C@H]([C@@H]([C@H](O3)C)OC(=O)C4=C(C(=C(C(=C4O)Cl)O)Cl)CC)O)OC)O)\C", # 非达霉素 (18元大环内酯，主治艰难梭菌，但作为对照药常出现在文献中)
    "telithromycin": "CC[C@@H]1[C@@]2([C@@H]([C@H](C(=O)[C@@H](C[C@@]([C@@H]([C@H](C(=O)[C@H](C(=O)O1)C)C)O[C@H]3[C@@H]([C@H](C[C@H](O3)C)N(C)C)O)(C)OC)C)C)N(C(=O)O2)CCCCN4C=C(N=C4)C5=CN=CC=C5)C", # 泰利霉素 (酮内酯类代表)
    "ohmyungsamycin a": "C[C@@H]1[C@@H](C(=O)N([C@H](C(=O)N[C@H](C(=O)N([C@H](C(=O)N[C@H](C(=O)N([C@H](C(=O)N([C@H](C(=O)N[C@H](C(=O)N[C@H](C(=O)N[C@H](C(=O)O1)C(C)C)[C@@H](C2=CC=CC=C2)O)C(C)C)CC3=CNC4=C3C(=CC=C4)OC)C)C(C)C)C)C(C)C)CC(C)C)C)C(C)C)[C@@H](C)O)C)NC(=O)[C@H](C(C)C)NC(=O)[C@H](C(C)C)NC",

    # 一些补充
    "Celastrol": "CC1=C(C(=O)C=C2C1=CC=C3[C@]2(CC[C@@]4([C@@]3(CC[C@@]5([C@H]4C[C@](CC5)(C)C(=O)O)C)C)C)C)O",
    "Triclosan": "Oc1cc(Cl)ccc1Oc1ccc(Cl)cc1Cl",
    "Kanamycin": "C1[C@H]([C@@H]([C@H]([C@@H]([C@H]1N)O[C@@H]2[C@@H]([C@H]([C@@H]([C@H](O2)CN)O)O)O)O)O[C@@H]3[C@@H]([C@H]([C@@H]([C@H](O3)CO)O)N)O)N",
    "Vancomycin": "C[C@H]1[C@H]([C@@](C[C@@H](O1)O[C@@H]2[C@H]([C@@H]([C@H](O[C@H]2OC3=C4C=C5C=C3OC6=C(C=C(C=C6)[C@H]([C@H](C(=O)N[C@H](C(=O)N[C@H]5C(=O)N[C@@H]7C8=CC(=C(C=C8)O)C9=C(C=C(C=C9O)O)[C@H](NC(=O)[C@H]([C@@H](C1=CC(=C(O4)C=C1)Cl)O)NC7=O)C(=O)O)CC(=O)N)NC(=O)[C@@H](CC(C)C)NC)O)Cl)CO)O)O)(C)N)O",
    "Ethionamide": "CCC1=NC=CC(=C1)C(=S)N",
    "Gatifloxacin": "CC1CN(CCN1)C2=C(C=C3C(=C2OC)N(C=C(C3=O)C(=O)O)C4CC4)F",
    "Metronidazole": "CC1=NC=C(N1CCO)[N+](=O)[O-]",
    "Nitazoxanide": "CC(=O)OC1=CC=CC=C1C(=O)NC2=NC=C(S2)[N+](=O)[O-]",
    "Tizoxanide": " C1=CC=C(C(=C1)C(=O)NC2=NC=C(S2)[N+](=O)[O-])O",
    "rifampin":"C[C@H]1/C=C/C=C(\\C(=O)NC2=C(C(=C3C(=C2O)C(=C(C4=C3C(=O)[C@](O4)(O/C=C/[C@@H]([C@H]([C@H]([C@@H]([C@@H]([C@@H]([C@H]1O)C)O)C)OC(=O)C)C)OC)C)C)O)O)/C=N/N5CCN(CC5)C)/C",
    "Cycloserine":"C1[C@H](C(=O)NO1)N",
    "Amikacin":"C1[C@@H]([C@H]([C@@H]([C@H]([C@@H]1NC(=O)[C@H](CCN)O)O[C@@H]2[C@@H]([C@H]([C@@H]([C@H](O2)CO)O)N)O)O)O[C@@H]3[C@@H]([C@H]([C@@H]([C@H](O3)CN)O)O)O)N",
    "Pyrimethamine":"CCC1=C(C(=NC(=N1)N)N)C2=CC=C(C=C2)Cl",
}




def get_smiles_robust(iupac_name, ai_predicted_smiles=None, is_standard_iupac=False, doc_cache=None):
    """获取标准的 SMILES，加入离线常见库过滤与 PubChem 长度拦截"""
    if doc_cache is None:
        doc_cache = {}
        
    raw_name = str(iupac_name).strip().replace('\n', ' ')
    lower_name = raw_name.lower()
    
    # 1. 检查篇内缓存
    if raw_name and raw_name != "N/A" and raw_name in doc_cache:
        return doc_cache[raw_name]

    result_smi, result_source = "N/A", "None"
    
    if not raw_name or raw_name == "N/A":
        canon_ai = standardize_smiles(ai_predicted_smiles)
        if canon_ai: result_smi, result_source = canon_ai, "AI"
    else:
        # 2. 检查全局常见 TB 药物离线字典 (极速通道)
        if lower_name in COMMON_TB_DRUGS:
            canon_smi = standardize_smiles(COMMON_TB_DRUGS[lower_name])
            if canon_smi:
                result_smi, result_source = canon_smi, "Offline Dictionary"
                doc_cache[raw_name] = (result_smi, result_source)
                return result_smi, result_source

        import re
        if re.match(r'^(compound|compd|derivative)?\s*[\dIVX\-\_]+[A-Za-z]?$', raw_name, re.IGNORECASE) or len(raw_name) <= 3:
            canon_ai = standardize_smiles(ai_predicted_smiles)
            if canon_ai: result_smi, result_source = canon_ai, "AI (Local ID fallback)"
        else:
            canon_smi = None
            # 3. 尝试 Opsin 离线解析
            if is_standard_iupac:
                # 3. 尝试 Opsin 离线解析
                try:
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        smiles = py2opsin(raw_name)
                    canon_smi = standardize_smiles(smiles)
                    if canon_smi: result_smi, result_source = canon_smi, "Opsin"
                except Exception: 
                    pass
                    
                # 4. 尝试 PubChem 
                if not canon_smi:
                    is_too_complex = len(raw_name) > 35 or raw_name.count('-') > 3 or raw_name.count('(') > 1
                    if not is_too_complex:
                        try:
                            dynamic_sleep = (MAX_CONCURRENT_FOLDERS / 5.0) + 0.5
                            time.sleep(dynamic_sleep)
                            
                            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                                future = executor.submit(pcp.get_compounds, raw_name, 'name')
                                compounds = future.result(timeout=12.0)
                                
                            if compounds:
                                smi = compounds[0].canonical_smiles
                                canon_smi = standardize_smiles(smi)
                                if canon_smi: result_smi, result_source = canon_smi, "PubChem"
                                
                        except concurrent.futures.TimeoutError:
                            log.warning(f"  -> [PubChem Timeout] '{raw_name[:15]}...' skipped.")
                        except Exception:
                            pass 
                    else:
                        log.info(f"  -> [PubChem Skip] Name too complex ('{raw_name[:15]}...'), fallback to AI.")
            else:
                # 记录一下被拦截的俗名或错别字
                log.info(f"  -> [Non-Standard Name] '{raw_name[:15]}...' marked as non-standard by AI. Bypassing Opsin/PubChem.")
            
            # 5. 兜底 AI (如果非规范命名，会直接跳到这里执行)
            if not canon_smi:
                canon_ai = standardize_smiles(ai_predicted_smiles)
                if canon_ai: result_smi, result_source = canon_ai, "AI"

    # 写入篇内缓存
    if raw_name and raw_name != "N/A":
        doc_cache[raw_name] = (result_smi, result_source)
        
    return result_smi, result_source

def pdf_page_to_bytes(pdf_path, page_num, dpi):
    doc = fitz.open(pdf_path)
    try:
        page = doc.load_page(page_num)
        pix = page.get_pixmap(dpi=dpi)
        from PIL import Image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return buf.getvalue()
    finally: doc.close()

def render_pages_concurrent(pdf_path, page_nums, dpi):
    """
    重构为顺序渲染：复用同一个 doc 句柄，避免 Windows 底层多线程 I/O 冲突。
    同时能大幅降低峰值内存占用，防止 C 级别内存溢出崩溃。
    """
    results = {}
    doc = fitz.open(pdf_path)
    try:
        from PIL import Image
        import io
        for p in page_nums:
            page = doc.load_page(p)
            pix = page.get_pixmap(dpi=dpi)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            results[p] = buf.getvalue()
    except Exception as e:
        log.error(f"Image Render Error in {os.path.basename(pdf_path)}: {e}")
    finally:
        doc.close()
    gc.collect() 
    
    return results

def clean_json_text(text):
    if not text: return ""
    clean_text = text
    if "```json" in clean_text:
        parts = clean_text.split("```json")
        if len(parts) > 1: clean_text = parts[1].split("```")[0]
    elif "```" in clean_text:
        parts = clean_text.split("```")
        if len(parts) > 1: clean_text = parts[1].split("```")[0]
    match_list = re.search(r'\[.*\]', clean_text, re.DOTALL)
    match_obj = re.search(r'\{.*\}', clean_text, re.DOTALL)
    if match_list: return match_list.group(0)
    if match_obj: return match_obj.group(0)
    return clean_text.strip()

def robust_json_extract(text):
    """尝试解析 JSON，如果发现被截断，尝试强制闭合以抢救已有数据。返回 (提取数据List, 是否被截断Bool)"""
    if not text: return [], False
    clean_text = text.strip()
    
    if "```json" in clean_text:
        parts = clean_text.split("```json")
        if len(parts) > 1: clean_text = parts[1].split("```")[0].strip()
    elif "```" in clean_text:
        parts = clean_text.split("```")
        if len(parts) > 1: clean_text = parts[1].split("```")[0].strip()
    
    is_truncated = not clean_text.endswith(']')
    
    try:
        data = json.loads(clean_text)
        if isinstance(data, list): return data, False
        if isinstance(data, dict): return [data], False
    except json.JSONDecodeError:
        last_brace = clean_text.rfind('}')
        if last_brace != -1:
            fixed_text = clean_text[:last_brace+1] + ']'
            try:
                data = json.loads(fixed_text)
                if isinstance(data, list): return data, True
            except json.JSONDecodeError:
                pass
    return [], False

def calculate_final_level(item):
    if item.get("is_qualitative_inactive") is True:
        return "Inactive"
    def parse_val(val_str):
        if not val_str: return None
        try:
            s = str(val_str).strip()
            # 核心改进：通过极小的偏移量保留不等号的物理意义
            offset = 0.0001 if '>' in s else -0.0001 if '<' in s else 0.0
            
            clean = s.replace('>', '').replace('<', '').replace('~', '').replace('=', '').strip()
            if '-' in clean:
                parts = clean.split('-')
                return (float(parts[0]) + float(parts[1])) / 2
            
            return float(clean) + offset
        except (ValueError, IndexError): 
            return None
    def get_thresholds(unit_str):
        t_exc, t_good, t_mod = 1.0, 10.0, 64.0
        if not unit_str: return t_exc, t_good, t_mod
        u = str(unit_str).strip().replace(" ", "")
        if re.search(r'[munµ]?g/', u, re.IGNORECASE): return t_exc, t_good, t_mod
        if re.search(r'nM\b', u): t_exc, t_good, t_mod = 2000.0, 20000.0, 100000.0
        elif re.search(r'[uµμ]M\b', u): t_exc, t_good, t_mod = 2.0, 20.0, 100.0
        elif re.search(r'mM\b', u): t_exc, t_good, t_mod = 0.002, 0.02, 0.1
        elif re.search(r'(?<![munµμ])M\b', u): t_exc, t_good, t_mod = 0.000002, 0.00002, 0.0001
        return t_exc, t_good, t_mod
    for key in ['mic', 'ic50']:
        val = parse_val(item.get(f"{key}_value"))
        unit = item.get(f"{key}_unit", "")
        if val is not None:
            t_exc, t_good, t_mod = get_thresholds(unit)
            if val <= t_exc: return "Excellent"
            if val <= t_good: return "Good"
            if val <= t_mod: return "Moderate"
            return "Inactive"
    inh_val = parse_numerical_value(item.get("inhibition_value")) 
    if inh_val is None: 
        inh_val = parse_val(item.get("inhibition_value"))     
    if inh_val is not None:
        if inh_val >= 90: return "Excellent"
        if inh_val >= 80: return "Good"
        if inh_val >= 20: return "Moderate"
        return "Inactive"
    pic50 = parse_val(item.get("pic50_value"))
    if pic50 is not None:
        if pic50 >= 6.0: return "Excellent"
        if pic50 >= 5.0: return "Good"
        return "Moderate/Inactive"
    pmic = parse_val(item.get("pmic_value"))
    if pmic is not None:
        # 对数体系下，数值越大，活性越强
        if pmic >= 6.0: return "Excellent"   # 等效于 <= 1 uM
        if pmic >= 5.0: return "Good"        # 等效于 <= 10 uM
        if pmic >= 4.2: return "Moderate"    # 等效于约 <= 64 uM (近似值)
        return "Inactive"
    zoi = parse_numerical_value(item.get("zoi_value"))
    if zoi is None: zoi = parse_val(item.get("zoi_value"))
    if zoi is not None:
        # 锁死上限，并提高判定阈值
        if zoi >= 25.0: return "Good (ZOI)"      # 即使大于 25mm，也绝不给 Excellent
        if zoi >= 15.0: return "Moderate (ZOI)"
        return "Inactive"
    return item.get("qualitative_level", "")


@retry_api()
def extract_metadata(pdf_path):
    doi, if_val, display_name, pub_date, article_type = "Unknown_DOI", "JCR_MISS", "Unknown", "Unknown", "Unknown"
    img_bytes = pdf_page_to_bytes(pdf_path, 0, dpi=150)
    
    rate_limiter.acquire()
    response = client.models.generate_content(
        model=MODEL_FLASH_NAME,
        contents=[
            types.Part(text=PROMPT_METADATA),
            types.Part(inline_data=types.Blob(mime_type="image/jpeg", data=img_bytes))
        ],
        config=types.GenerateContentConfig(response_mime_type="application/json")
    )
    
    try:
        data = json.loads(clean_json_text(response.text))
        # 直接获取 AI 提取的原生 DOI，仅去除首尾多余空格
        doi = data.get("doi", "Unknown_DOI").strip()
        
        # 原始 AI 提取的名字 (保留用于展示和写入 Excel)
        raw_abbr = data.get("journal_abbreviation", "Unknown").strip().upper()
        raw_full = data.get("journal_full_name", "Unknown").strip().upper()
        pub_date = data.get("publication_date", "Unknown").strip() 
        display_name = data.get("journal_full_name") if data.get("journal_full_name") != "Unknown" else data.get("journal_abbreviation")
        article_type = data.get("article_type", "Research").strip()

        # 【强化模糊匹配】：构建剔除所有特殊字符的纯净检索词
        search_abbr = re.sub(r'[^A-Z0-9]', '', raw_abbr)
        search_full = re.sub(r'[^A-Z0-9]', '', raw_full)

        # 匹配逻辑 (使用纯净词检索)
        if search_abbr and search_abbr in JCR_DICT:
            if_val = JCR_DICT[search_abbr]
            log.info(f"   -> [JCR Hit-Abbr] '{raw_abbr}' | IF: {if_val}")
        elif search_full and search_full in JCR_DICT:
            if_val = JCR_DICT[search_full]
            log.info(f"   -> [JCR Hit-Full] '{raw_full}' | IF: {if_val}")
        else:
            log.info(f"   -> [JCR Miss] No match for '{raw_abbr}' or '{raw_full}'. Querying Pro model...")
            if_val = fetch_if_from_pro(raw_abbr, raw_full)
            log.info(f"   -> [Pro Model Result] IF: {if_val}")

    except Exception as e:
        log.error(f"Metadata Parse Error for {os.path.basename(pdf_path)}: {e}")
        doi = os.path.basename(pdf_path)

    return doi, if_val, display_name, pub_date, article_type

@retry_api()
def classify_batch_text(pdf_path, page_list):
    doc = fitz.open(pdf_path)
    combined_text = ""
    for p in page_list:
        try:
            text = doc.load_page(p).get_text().replace('\n', ' ')[:1500]
            combined_text += f"\n### Page_{p} ###\n{text}\n"
        except Exception as e:
            log.warning(f"   -> [Text Extract] Skipping damaged text on page {p}.")
            continue
    doc.close()
    if not combined_text.strip(): return {}
    rate_limiter.acquire()
    response = client.models.generate_content(
        model=MODEL_FLASH_NAME,
        contents=[f"{PROMPT_CLASSIFY_BATCH_TEXT}\n{combined_text}"],
        config=types.GenerateContentConfig(response_mime_type="application/json", temperature=1.0)
    )
    try:
        raw_dict = json.loads(clean_json_text(response.text))
        return {int(k.replace("Page_", "")): v for k, v in raw_dict.items() if k.replace("Page_", "").isdigit()}
    except Exception: return {}

@retry_api()
def classify_single_vision(pdf_path, page_num):
    img_bytes = pdf_page_to_bytes(pdf_path, page_num, IMG_DPI_FLASH)
    rate_limiter.acquire()
    response = client.models.generate_content(
        model=MODEL_FLASH_NAME,
        contents=[
            types.Part(text=PROMPT_CLASSIFY_VISION),
            types.Part(inline_data=types.Blob(mime_type="image/jpeg", data=img_bytes))
        ],
        config=types.GenerateContentConfig(response_mime_type="application/json", temperature=1.0)
    )
    try: return json.loads(clean_json_text(response.text)).get("category", "IRRELEVANT")
    except Exception: return "IRRELEVANT"

def analyze_pdf_structure_hybrid(pdf_path):
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    max_scan = min(total_pages, ABSOLUTE_MAX_PAGES)
    text_queue, vision_queue = [], []
    for p in range(max_scan):
        if len(doc.load_page(p).get_text().strip()) > TEXT_MIN_LENGTH: text_queue.append(p)
        else: vision_queue.append(p)
    doc.close()

    page_roles = {}
    if text_queue:
        for i in range(0, len(text_queue), TEXT_BATCH_SIZE):
            res = classify_batch_text(pdf_path, text_queue[i: i + TEXT_BATCH_SIZE])
            if res:
                for p_num, cat in res.items():
                    if cat != "IRRELEVANT": page_roles[p_num] = cat
    if vision_queue:
        for p in vision_queue:
            cat = classify_single_vision(pdf_path, p)
            if cat != "IRRELEVANT": page_roles[p] = cat
    
    relevant_pages = sorted(page_roles.keys())
    return relevant_pages, page_roles


# ================= 8. 核心提取模块 (含断点续传) =================

@retry_api()
def extract_data_pro_vision(pdf_path, page_nums):
    if not page_nums: return []
    
    rendered = render_pages_concurrent(pdf_path, page_nums, IMG_DPI_PRO)
    image_parts = []
    for p in page_nums:
        image_parts.append(types.Part(text=f"--- IMAGE OF PAGE {p} ---"))
        image_parts.append(types.Part(inline_data=types.Blob(mime_type="image/jpeg", data=rendered[p])))

    all_extracted_data = []
    
    base_prompt_part = types.Part(text=PROMPT_EXTRACT)
    rate_limiter.acquire()
    response = client.models.generate_content(
        model=MODEL_PRO_NAME,
        contents=[types.Content(parts=image_parts + [base_prompt_part])],
        config=types.GenerateContentConfig(
            temperature=1.0,
            thinking_config=types.ThinkingConfig(thinking_level="high"),
            media_resolution="media_resolution_high"
        )
    )
    
    parsed_data, is_truncated = robust_json_extract(response.text)
    if parsed_data:
        all_extracted_data.extend(parsed_data)

    continuations = 0
    while is_truncated and continuations < MAX_CONTINUATIONS and parsed_data:
        last_id = parsed_data[-1].get("compound_id", "")
        if not last_id:
            break
            
        log.warning(f"   -> [Truncated] Salvaged up to ID '{last_id}'. Requesting continuation ({continuations+1}/{MAX_CONTINUATIONS})...")
        
        cont_text = f"\n\n[SYSTEM ALERT]: The previous output was cut off. You stopped precisely at compound ID '{last_id}'. Please CONTINUE extracting the remaining compounds from the images, starting STRICTLY with the compound immediately after ID '{last_id}'. Do not repeat compounds already extracted. Return the result as a new JSON list."
        cont_prompt_part = types.Part(text=PROMPT_EXTRACT + cont_text)
        
        rate_limiter.acquire()
        cont_response = client.models.generate_content(
            model=MODEL_PRO_NAME,
            contents=[types.Content(parts=[cont_prompt_part] + image_parts)],
            config=types.GenerateContentConfig(
                temperature=1.0,
                thinking_config=types.ThinkingConfig(thinking_level="high"),
                media_resolution="media_resolution_high"
            )
        )
        
        parsed_data, is_truncated = robust_json_extract(cont_response.text)
        if parsed_data:
            all_extracted_data.extend(parsed_data)
        continuations += 1

    return all_extracted_data


# ================= 9. 纯追加写入 Excel 模块 (带防碰撞重试) =================

def append_to_excel(excel_path, doi, filename, if_val, journal_name, pub_date, extracted_data):
    if not extracted_data:
        return
    
    valid_items = []
    for item in extracted_data:
        # 检查所有数值字段是否至少有一个非空
        numerical_fields = [
            item.get("mic_value"), 
            item.get("ic50_value"), 
            item.get("inhibition_value"), 
            item.get("zoi_value"),
            item.get("pic50_value")
        ]
        is_inactive_flag = item.get("is_qualitative_inactive", False)
        has_any_data = any(v is not None and str(v).strip() != "" and str(v).lower() != "null" for v in numerical_fields)
        
        if is_inactive_flag or has_any_data:
            valid_items.append(item)
    
    if not valid_items:
        log.info(f"   -> [No Data] {doi} contains no activity data, skipping records.")
        return

    final_rows = []
    current_doc_smiles_cache = {}
    id_smiles_cache = {}  
    
    for item in valid_items: 
        raw_c_name = item.get("iupac_name", item.get("compound_name", "")).strip()
        import re
        if re.match(r'^(compound|compd|derivative)?\s*[\dIVX\-\_]+[A-Za-z]?$', raw_c_name, re.IGNORECASE):
            c_name = ""
        else:
            c_name = raw_c_name
            
        ai_smi = item.get("predicted_smiles", "N/A")
        c_id = str(item.get("compound_id", "")).strip().lower() 
        
        is_standard_iupac = item.get("is_standard_iupac", False)

        if c_id and c_id in id_smiles_cache:
            smi, source = id_smiles_cache[c_id]
            log.info(f"      -> [Cache Hit] Compound ID '{c_id}' SMILES duplicated from previous strain record.")
        else:
            smi, source = get_smiles_robust(c_name, ai_smi, is_standard_iupac, current_doc_smiles_cache)
            if c_id and smi and smi != "N/A":
                id_smiles_cache[c_id] = (smi, source)

        final_rows.append({
            "Source Filename": filename,
            "Filename/DOI": doi,
            "Journal Name": journal_name,
            "Publication Date": pub_date,
            "Impact Factor": if_val,
            "Compound ID": item.get("compound_id"),
            "Target Strain": item.get("target_strain", ""),
            "Hazard Level": item.get("hazard_level", ""),
            "IUPAC Name": c_name, 
            "SMILES": smi,
            "SMILES Source": source,
            "MIC Value": item.get("mic_value", ""),
            "MIC Unit": item.get("mic_unit", ""),
            "IC50 Value": item.get("ic50_value", ""),
            "IC50 Unit": item.get("ic50_unit", ""),
            "% Inhibition": item.get("inhibition_value", ""),
            "Inhibition Conc": item.get("inhibition_conc", ""),
            "pIC50": item.get("pic50_value", ""),
            "pMIC": item.get("pmic_value", ""),
            "ZOI (mm)": item.get("zoi_value", ""),
            "Activity Level": item.get("activity_level_override", ""),
            "Extraction Source": item.get("extraction_source", ""),
            "Extraction Time": time.strftime("%Y-%m-%d %H:%M:%S")
        })

    df = pd.DataFrame(final_rows)
    
    # 定义警告红色填充样式（浅红色，不会遮挡文字）
    red_fill = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")
    
    for attempt in range(MAX_EXCEL_RETRIES):
        try:
            mode = 'a' if os.path.exists(excel_path) else 'w'
            with pd.ExcelWriter(excel_path, engine='openpyxl', mode=mode,
                                if_sheet_exists='overlay' if mode == 'a' else None) as writer:
                
                # 计算新数据应该从哪一行开始追加
                start_row = writer.book[SHEET_NAME].max_row if mode == 'a' else 0
                if mode == 'a':
                    start_row += 1
                df.to_excel(writer, sheet_name=SHEET_NAME, index=False, header=(mode == 'w'), startrow=start_row)
                
                # === 动态样式标红逻辑 ===
                ws = writer.book[SHEET_NAME]
                
                # 获取表头列表（openpyxl 的行列索引从 1 开始）
                if mode == 'w':
                    headers = list(df.columns)
                    data_start_row = 2 
                else:
                    headers = [cell.value for cell in ws[1]]
                    data_start_row = start_row + 1

                try:
                    # 动态寻找 "Impact Factor" 所在的列索引
                    if_col_idx = headers.index("Impact Factor") + 1
                    data_end_row = data_start_row + len(df)
                    
                    # 遍历刚写入的这几行数据
                    for r in range(data_start_row, data_end_row):
                        cell = ws.cell(row=r, column=if_col_idx)
                        if str(cell.value) == "NA":
                            cell.fill = red_fill
                except ValueError:
                    pass # 如果因意外找不到 Impact Factor 列，静默跳过样式处理
            
            # 【新增】：统计独立的化合物数量 (基于 Compound ID 去重)
            unique_compounds = set(
                str(row.get("Compound ID", "")).strip().lower() 
                for row in final_rows 
                if row.get("Compound ID")
            )
            
            # 【修改】：更新日志输出格式
            log.info(f"      [{os.path.basename(excel_path)}] Appended {len(final_rows)} records (Unique compounds: {len(unique_compounds)}).")
            break
        except PermissionError:
            log.warning(f"⚠️ [File Locked] Please close {os.path.basename(excel_path)}. Retrying in 3s ({attempt+1}/{MAX_EXCEL_RETRIES})...")
            time.sleep(3)
    else:
        log.error(f"❌ [Fatal] Could not write to {excel_path} after {MAX_EXCEL_RETRIES} retries. Data for {doi} was not saved.")


# ================= 10. 流程控制 =================
def process_single_pdf(pdf_path, target_excel):
    filename = os.path.basename(pdf_path)
    log.info(f"Analyzing: {filename}")

    try:
        # 1. 提取元数据
        doi, if_val, journal_name, pub_date, article_type = extract_metadata(pdf_path)
        log.info(f"   -> Meta: DOI={doi}, Date={pub_date}, IF={if_val}")

        if str(article_type).strip().upper() == "REVIEW":
            log.info(f"📚 [Review Detected] {filename} is a Review article. Moving to '{REVIEW_FOLDER}' and skipping extraction.")
            shutil.move(pdf_path, os.path.join(REVIEW_FOLDER, filename))
            return False 

        # 【新增拦截哨卡】：如果 IF < 1 或为明确的 "N/A"，则原地跳过，不作处理
        skip_trigger = False
        skip_reason = ""  # 记录具体的跳过原因，方便看日志
        
        # 判断 1：无论是 Excel 里的 N/A，还是 Pro 模型查不到返回的 NA
        if str(if_val).strip().upper() in ["N/A", "NA", "NULL", "NONE"]:
            skip_trigger = True
            skip_reason = f"IF: {if_val}"
        # 判断 2：如果 IF 是数字类型，且小于 1
        elif isinstance(if_val, (int, float)) and if_val < 1:
            skip_trigger = True
            skip_reason = f"IF: {if_val}"
            
        # 判断 3：【新增】判断文献年份是否早于 2000 年
        if not skip_trigger and str(pub_date).strip().upper() != "UNKNOWN":
            try:
                # 使用正则安全提取出前 4 位连续数字作为年份 (兼容 "1999-05" 等格式)
                match = re.search(r'\d{4}', str(pub_date))
                if match:
                    year = int(match.group(0))
                    if year < 2000:
                        skip_trigger = True
                        skip_reason = f"Year: {year} < 2000"
            except Exception:
                pass 
            
        if skip_trigger:
            if not skip_reason: skip_reason = f"IF: {if_val}" 
                
            log.info(f"⏩ [Threshold-Skip] {filename} | Reason: {skip_reason}. Moving to '{LOW_IF_FOLDER}' and skipping extraction.")
            shutil.move(pdf_path, os.path.join(LOW_IF_FOLDER, filename)) # ✅ 改为 move
            return False 
        relevant_page_nums, page_roles = analyze_pdf_structure_hybrid(pdf_path)
        if not relevant_page_nums:
            log.info(f"   No relevant data pages found in {filename}")
            shutil.move(pdf_path, os.path.join(FAIL_FOLDER, filename))
            return False

        safe_page_nums = relevant_page_nums[:ABSOLUTE_MAX_PAGES]

        anchor_pages = sorted([p for p in safe_page_nums if page_roles.get(p) in ("STRUCTURE", "BOTH")])
        data_pages = [p for p in safe_page_nums if page_roles.get(p) in ("DATA", "BOTH")]
        if not data_pages and not anchor_pages: data_pages = safe_page_nums; anchor_pages = []
        elif not data_pages: data_pages = safe_page_nums; anchor_pages = []

        extracted_data = []
        data_batch_size = max(4, MAX_PAGES_TO_PRO - len(anchor_pages))

        # 3. 核心视觉提取
        for i in range(0, len(data_pages), data_batch_size):
            batch_data_pages = data_pages[i: i + data_batch_size]
            target_pages = sorted(set(anchor_pages + batch_data_pages))
            batch_data = extract_data_pro_vision(pdf_path, target_pages)
            if batch_data: extracted_data.extend(batch_data)

        # 4. 篇内去重合并
        if extracted_data:
            grouped_data = {}
            # 按 ID 分组
            for item in extracted_data:
                cid = str(item.get("compound_id", "")).strip().lower()
                if not cid: continue
                if cid not in grouped_data:
                    grouped_data[cid] = []
                grouped_data[cid].append(item)

            final_extracted_data = []

            for cid, items in grouped_data.items():
                # 过滤出有合法危害等级的记录
                valid_items = [i for i in items if str(i.get("hazard_level")).strip().upper() in ["A", "B", "C", "D"]]
                
                if not valid_items:
                    # 如果大模型全部分类失败，直接回退单算
                    for i in items:
                        i["activity_level_override"] = calculate_final_level(i)
                        final_extracted_data.append(i)
                    continue

                # 找最高危级别（A < B < C < D，字符串比对取 min 即为最高危）
                highest_hazard = min([str(i.get("hazard_level")).strip().upper() for i in valid_items])
                
                top_rows = [i for i in items if str(i.get("hazard_level")).strip().upper() == highest_hazard]
                lower_rows = [i for i in items if str(i.get("hazard_level")).strip().upper() != highest_hazard]

                # --- 寻找最优数据与主控等级定性 (全指标覆盖) ---
                target_key = None
                is_lower_better = True
                
                if any(r.get("mic_value") for r in top_rows):
                    target_key = "mic"
                    is_lower_better = True  
                elif any(r.get("ic50_value") for r in top_rows):
                    target_key = "ic50"
                    is_lower_better = True  
                elif any(r.get("pic50_value") for r in top_rows):
                    target_key = "pic50"
                    is_lower_better = False 
                elif any(r.get("pmic_value") for r in top_rows):
                    target_key = "pmic"
                    is_lower_better = False 
                elif any(r.get("inhibition_value") for r in top_rows):
                    target_key = "inhibition"
                    is_lower_better = False 
                elif any(r.get("zoi_value") for r in top_rows):
                    target_key = "zoi"
                    is_lower_better = False 

                best_row = None
                best_val = None

                if target_key:
                    for row in top_rows:
                        raw_val = row.get(f"{target_key}_value")
                        parsed_v = parse_numerical_value(raw_val)
                        if parsed_v is not None:
                            # 记录第一个有效数值
                            if best_val is None:
                                best_val = parsed_v
                                best_row = row
                            else:
                                # 根据指标类型比较，刷新最优行
                                if is_lower_better and parsed_v < best_val:
                                    best_val = parsed_v
                                    best_row = row
                                elif not is_lower_better and parsed_v > best_val:
                                    best_val = parsed_v
                                    best_row = row

                # --- 赋值与清空 ---
                for row in top_rows:
                    if best_row is not None and row is best_row:
                        row["activity_level_override"] = calculate_final_level(row)
                    elif best_row is None and row is top_rows[0]:
                        row["activity_level_override"] = calculate_final_level(row)
                    else:
                        row["activity_level_override"] = ""
                    
                    # 👇 这一步极其关键，确保同化合物的所有“最高危”测试记录都被完整保留进最终列表
                    final_extracted_data.append(row)
                
                for row in lower_rows:
                    # 陪跑行（低危菌株）仅清空 Activity Level
                    row["activity_level_override"] = ""
                    # 👇 确保同化合物的“低危”测试记录也被完整保留
                    final_extracted_data.append(row)

            extracted_data = final_extracted_data

        # 5. 追加写入目标 Excel
        if extracted_data:
            append_to_excel(target_excel, doi, filename, if_val, journal_name, pub_date, extracted_data)
            shutil.move(pdf_path, os.path.join(SUCCESS_FOLDER, filename))
            return True
        else:
            shutil.move(pdf_path, os.path.join(FAIL_FOLDER, filename))
            return False

    except APIConnectionError as e:
        log.error(f"[NETWORK ERROR] Skipping {filename}: {e}")
        return False
    except Exception as e:
        log.error(f"[CODE ERROR] {filename}: {e}")
        return False

def process_folder_task(folder_path):
    folder_name = os.path.basename(folder_path)
    target_excel = os.path.join(OUTPUT_EXCEL_FOLDER, f"{folder_name}.xlsx")
    
    pdfs = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
    pdfs.sort()
    
    if not pdfs:
        log.info(f"[{folder_name}] No PDFs found. Skipping.")
        return folder_name, 0, 0

    log.info(f"--- Starting Folder: {folder_name} ({len(pdfs)} PDFs) -> {folder_name}.xlsx ---")
    
    success_count = 0
    fail_count = 0
    
    for pdf_filename in pdfs:
        pdf_path = os.path.join(folder_path, pdf_filename)
        pdf_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = pdf_executor.submit(process_single_pdf, pdf_path, target_excel)
        
        try:
            success = future.result(timeout=PDF_TIMEOUT_SECONDS)
            if success:
                success_count += 1
            else:
                fail_count += 1
                
        except concurrent.futures.TimeoutError:
            log.error(f"⏰ [Timeout] {pdf_filename} exceeded {PDF_TIMEOUT_SECONDS}s limit. Skipping & moving to '{TIMEOUT_FOLDER}'.")
            
            try:
                shutil.move(pdf_path, os.path.join(TIMEOUT_FOLDER, pdf_filename))
            except Exception as move_e:
                log.error(f"   -> Failed to move timed-out file {pdf_filename}: {move_e}")
                
            fail_count += 1
            
        except Exception as e:
            log.error(f"❌ [Unhandled Error] {pdf_filename}: {e}")
            fail_count += 1
            
        finally:
            pdf_executor.shutdown(wait=False, cancel_futures=True)
            
    return folder_name, success_count, fail_count


def main():
    log.info(f"Gemini Drug Discovery Agent | Folder Concurrent Mode: {MAX_CONCURRENT_FOLDERS} Threads")

    subfolders = [os.path.join(INPUT_BASE_FOLDER, d) for d in os.listdir(INPUT_BASE_FOLDER) 
                  if os.path.isdir(os.path.join(INPUT_BASE_FOLDER, d))]

    if not subfolders:
        log.warning(f"No subfolders found in {INPUT_BASE_FOLDER}. Please organize your PDFs into subfolders.")
        return

    log.info(f"Found {len(subfolders)} subfolders to process.")

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_FOLDERS, thread_name_prefix="Dir") as executor:
        future_to_folder = {executor.submit(process_folder_task, path): path for path in subfolders}
        active_futures = set(future_to_folder.keys())
        try:
            with tqdm(total=len(subfolders), desc="Processing Folders") as pbar:
                
                # 使用带有 0.5 秒超时的 while 循环，让主线程保持“呼吸”
                while active_futures:
                    done, active_futures = concurrent.futures.wait(
                        active_futures, 
                        timeout=0.5, 
                        return_when=concurrent.futures.FIRST_COMPLETED
                    )
                    
                    # 处理在这 0.5 秒内完成的任务
                    for future in done:
                        folder_path = future_to_folder[future]
                        try:
                            folder_name, succ, fail = future.result()
                            log.info(f"✅ Folder Completed: {folder_name} | Success: {succ} | Fail: {fail}")
                        except Exception as e:
                            log.error(f"[Unhandled Folder Error] {os.path.basename(folder_path)}: {e}")
                        pbar.update(1)

            log.info("🎉 All folders processed completely.")
            
        except KeyboardInterrupt:
            # === 捕捉 Ctrl+C 信号 ===
            print("\n") # 换行，避免和 tqdm 进度条混在一起
            log.warning("⚠️ 收到手动中断信号 (Ctrl+C)！正在物理强制结束所有任务并退出...")
            
            # os._exit(1) 是 C 级别的强杀，无视 ThreadPoolExecutor 的挂起等待，瞬间结束
            os._exit(1)

if __name__ == "__main__":
    main()