import os
import shutil
import json
import time
import io
import queue
import re
import unicodedata
import base64
import hashlib
import functools
import tempfile
import threading
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from openpyxl.styles import PatternFill
import pandas as pd
import concurrent.futures
import fitz  # PyMuPDF
from tqdm import tqdm
from py2opsin import py2opsin
from rdkit import Chem
from rdkit import RDLogger
from google import genai
from google.genai import types
import json_repair
from uniparser_tools.api.clients import UniParserClient
from uniparser_tools.common.constant import ParseMode, ParseModeTextual
import signal 
import sys
import faulthandler

try:
    from PIL import Image as PILImage
except ImportError:
    PILImage = None

faulthandler.enable()
fitz.TOOLS.mupdf_display_errors(False)     
RDLogger.DisableLog('rdApp.*')

# ================= 1. 全局配置 (超参数与常量) =================

GEMINI_API_KEY = "sk-xxxx"
UNIPARSER_API_KEY = "up_xxxx"
API_KEY = GEMINI_API_KEY
REPAIR_GEMINI_API_KEY = GEMINI_API_KEY

# --- 模型选择 ---
MODEL_FLASH_NAME = "gemini-3-flash-preview"
MODEL_PRO_NAME = "gemini-3.1-pro-preview"

# --- 并发与速率控制 ---
MAX_CONCURRENT_FOLDERS = 4        # 同时处理的文件夹数量 (多线程并发度)
API_CALLS_PER_MINUTE = 1000       # API 每分钟最大调用次数（全局共享，防止 429 报错）
REPAIR_API_CALLS_PER_MINUTE = 100 # IUPAC/马库什结构解析沿用修复脚本速率
API_MAX_RETRIES = 5               # API 请求失败时的最大重试次数
API_INITIAL_DELAY = 10            # API 重试的初始等待时间（秒，后续按指数退避递增）
REPAIR_API_INITIAL_DELAY = 20     # UniParser/修复 Gemini 沿用修复脚本退避时间
PDF_TIMEOUT_SECONDS = 600         # ：单篇文献最大处理时间限制 (秒)，600秒即10分钟
INTERRUPT_COMMIT_GRACE_SECONDS = 20 # Ctrl+C 后只等待已进入最终提交的任务
GEMINI_REQUEST_TIMEOUT_SECONDS = 360 # 单次 Gemini 调用外层截止时间
UNIPARSER_REQUEST_TIMEOUT_SECONDS = 120 # UniParser 同步解析/取结果截止时间

# --- 运行逻辑与截断控制 ---
MAX_TABLE_PAGES_PER_BATCH = 12    # 单次发送给 Pro 的最大含表格页面数
MAX_TABLE_CONTEXT_CHARS = 100000  # 防止单批结构化表格文本过长；单个超大表格仍单独发送
MAX_EVIDENCE_FILTER_PAGES = 12    # IUPAC 证据预筛选每批最多覆盖的 UniParser 页面数
MAX_EVIDENCE_FILTER_CHARS = 90000 # 同页或单个超大记录不拆分，其余按字符数分批
EVIDENCE_NEIGHBOR_BLOCKS = 2      # IUPAC 命中块前后自动保留的原始文本块数量
IUPAC_NMR_PAGE_RADIUS = 1         # 同时含 NMR/Hz 的关键页向前、向后各保留一页
MAX_FIGURE_LONG_EDGE = 1600       # 发送结构解析 Gemini 前缩小 Figure
FIGURE_JPEG_QUALITY = 85          # JPEG Figure 缩放后的保存质量
VERBOSE_PIPELINE_REJECTIONS = False # 默认只输出每篇文献的阶段摘要
MAX_CONTINUATIONS = 3            # 大模型输出 JSON 截断时，允许启动“断点续传”的最大次数
MAX_EXCEL_RETRIES = 10            # 遇到 Excel 文件被人工打开占用 (PermissionError) 时的最大等待重试次数


# --- 路径与文件设置 ---
INPUT_BASE_FOLDER = "kept_pdfs"        # 存放多个子文件夹的根目录
OUTPUT_EXCEL_FOLDER = "output_excels-1"  # 存放生成的 xlsx 文件的目录
SUCCESS_FOLDER = "processed_success-1"   # 处理成功的 PDF 移动到此处
FAIL_FOLDER = "processed_fail-1"         # 处理失败或无数据的 PDF 移动到此处
REVIEW_FOLDER = "processed_reviews"
LOW_IF_FOLDER = "processed_low_if"     # 专门存放 IF < 1 或 N/A 的文件夹
TIMEOUT_FOLDER = "processed_timeout-1"   # 处理超时的 PDF 移动到此处
SHEET_NAME = "Extracted_Results"       # Excel 写入的 Sheet 名称
JCR_FILE_PATH = "JCR完整版.xlsx"       # 本地 JCR 影响因子文件路径 
DYNAMIC_IF_CACHE_FILE = "dynamic_if_cache.txt"

# --- 结果标记颜色 ---
CONFLICT_FILL = PatternFill(start_color="FFF2CC", end_color="FFF2CC", fill_type="solid")
METAL_FILL = PatternFill(start_color="FF0000", end_color="FF0000", fill_type="solid")

# ================= 2. 日志配置 =================
def silent_force_quit(signum, frame):
    """把 SIGINT 交给主流程，首次 Ctrl+C 获得短暂的原子提交窗口。"""
    raise KeyboardInterrupt

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
                except APIRequestCancelled:
                    return None
                except APIRequestDeadlineExceeded as exc:
                    log.error(f"[API Timeout] 不重复提交同一请求: {exc}")
                    return None
                except Exception as e:
                    error_msg = str(e).lower()

                    if "429" in error_msg or "quota" in error_msg or "exhausted" in error_msg:
                        print("\n")
                        log.critical("=====================================================")
                        log.critical("❌ [FATAL ERROR] API 额度已耗尽或被服务商严重限流！")
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
    api_key=GEMINI_API_KEY,
    http_options={
        'api_version': 'v1beta',
        'base_url': 'https://api.viviai.top',
    }
)

repair_client = genai.Client(
    api_key=GEMINI_API_KEY,
    http_options={
        'api_version': 'v1beta',
        'base_url': 'https://api.viviai.top',
    }
)

parser_client = UniParserClient(
    host="https://uniparser.dp.tech/",
    api_key=UNIPARSER_API_KEY
)

rate_limiter = RateLimiter(API_CALLS_PER_MINUTE)
repair_rate_limiter = RateLimiter(REPAIR_API_CALLS_PER_MINUTE)
GLOBAL_STOP_EVENT = threading.Event()
FINAL_COMMIT_CONDITION = threading.Condition(threading.Lock())
ACTIVE_FINAL_COMMITS = set()


def _request_global_stop():
    with FINAL_COMMIT_CONDITION:
        GLOBAL_STOP_EVENT.set()
        FINAL_COMMIT_CONDITION.notify_all()


def _begin_final_commit(commit_key):
    """中断发生后禁止处理一半的 PDF 新进入写入或移动阶段。"""
    with FINAL_COMMIT_CONDITION:
        if GLOBAL_STOP_EVENT.is_set():
            return False
        ACTIVE_FINAL_COMMITS.add(commit_key)
        return True


def _end_final_commit(commit_key):
    with FINAL_COMMIT_CONDITION:
        ACTIVE_FINAL_COMMITS.discard(commit_key)
        FINAL_COMMIT_CONDITION.notify_all()


def _move_pdf_as_final_commit(pdf_path, destination_folder, label):
    """文件分类移动也遵循中断边界；未开始移动的 PDF 留给下次。"""
    commit_key = f"move|{label}|{os.path.abspath(pdf_path)}"
    if not _begin_final_commit(commit_key):
        log.warning(
            f"[GLOBAL STOP] Skipping {label} move for {os.path.basename(pdf_path)}; "
            "the PDF remains for the next run."
        )
        return False
    try:
        os.makedirs(destination_folder, exist_ok=True)
        shutil.move(
            pdf_path,
            os.path.join(destination_folder, os.path.basename(pdf_path)),
        )
        return True
    finally:
        _end_final_commit(commit_key)


def _force_exit_after_commit_grace(reason, exit_code):
    """不等待网络线程；只给已登记的最终提交最多 10 秒。"""
    deadline = time.monotonic() + INTERRUPT_COMMIT_GRACE_SECONDS
    pending = 0
    try:
        with FINAL_COMMIT_CONDITION:
            while ACTIVE_FINAL_COMMITS:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                log.warning(
                    f"[{reason}] Waiting for {len(ACTIVE_FINAL_COMMITS)} final commit(s); "
                    f"at most {remaining:.1f}s remain."
                )
                FINAL_COMMIT_CONDITION.wait(timeout=min(0.5, remaining))
            pending = len(ACTIVE_FINAL_COMMITS)
    except KeyboardInterrupt:
        pending = len(ACTIVE_FINAL_COMMITS)
        log.warning("Second Ctrl+C received; forcing immediate exit.")
    if pending:
        log.error(
            f"Commit grace window ended; abandoning {pending} unfinished commit(s). "
            "Those PDFs will be processed again next run."
        )
    else:
        log.warning(
            "All already-started commits ended; forcing exit without waiting for "
            "network-blocked or partially processed PDFs."
        )
    os._exit(exit_code)


class APIRequestDeadlineExceeded(TimeoutError):
    pass


class APIRequestCancelled(Exception):
    pass


def _run_with_deadline(request_func, timeout_seconds, operation_name):
    """守护线程承载不可取消 SDK 调用，调用者可响应停止或截止时间。"""
    result_queue = queue.Queue(maxsize=1)

    def runner():
        try:
            result_queue.put((True, request_func()))
        except BaseException as exc:
            result_queue.put((False, exc))

    request_thread = threading.Thread(
        target=runner,
        name=f"API-{operation_name}",
        daemon=True,
    )
    request_thread.start()
    deadline = time.monotonic() + timeout_seconds
    while request_thread.is_alive():
        if GLOBAL_STOP_EVENT.is_set():
            raise APIRequestCancelled(f"{operation_name} cancelled by global stop")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise APIRequestDeadlineExceeded(
                f"{operation_name} exceeded {timeout_seconds} seconds"
            )
        request_thread.join(timeout=min(0.2, remaining))

    succeeded, payload = result_queue.get()
    if succeeded:
        return payload
    raise payload


def _generate_content_with_deadline(client_instance, **kwargs):
    return _run_with_deadline(
        lambda: client_instance.models.generate_content(**kwargs),
        GEMINI_REQUEST_TIMEOUT_SECONDS,
        "Gemini generate_content",
    )


def _normalize_status(status):
    """兼容 UniParser 返回的字符串状态和 Enum 状态。"""
    if hasattr(status, "value"):
        status = status.value
    return str(status or "").strip().lower()


def _uniparser_response_ok(result, health_check=False):
    if not isinstance(result, dict):
        return False

    status = _normalize_status(result.get("status"))
    if status in {"success", "ok", "healthy"}:
        return True

    if health_check and not status:
        http_status = result.get("http_status")
        has_error = any(result.get(field) for field in ("description", "message", "body"))
        try:
            http_ok = http_status is None or int(http_status) < 400
        except (TypeError, ValueError):
            http_ok = False
        return not has_error and http_ok

    return False


def _uniparser_error_text(result):
    if not isinstance(result, dict):
        return repr(result)

    fields = ("http_status", "status", "description", "message", "body")
    parts = [
        f"{field}={result.get(field)}"
        for field in fields
        if result.get(field) not in (None, "")
    ]
    return " | ".join(parts) or repr(result)


def _classify_uniparser_error(result):
    """将 UniParser 错误分为 fatal、retry、document。"""
    error_text = _uniparser_error_text(result).lower()
    try:
        http_status = (
            int(result.get("http_status"))
            if isinstance(result, dict) and result.get("http_status")
            else None
        )
    except (TypeError, ValueError):
        http_status = None

    fatal_hints = (
        "authentication required",
        "authentication failed",
        "insufficient balance",
        "missing permission",
        "administrator privileges required",
        "unauthorized",
        "forbidden",
    )
    retry_hints = (
        "rate limit",
        "too many requests",
        "429",
        "timeout",
        "timed out",
        "connectionerror",
        "connection error",
        "connection reset",
        "temporarily unavailable",
        "bad gateway",
        "service unavailable",
        "internal server error",
        "502",
        "503",
        "504",
        "process_failed",
        "status_not_found",
        "result_not_found",
    )

    if http_status in {401, 402, 403} or any(hint in error_text for hint in fatal_hints):
        return "fatal"
    if http_status == 429 or (http_status is not None and http_status >= 500):
        return "retry"
    if any(hint in error_text for hint in retry_hints):
        return "retry"
    return "document"


def _call_uniparser_with_retry(operation_name, request_func):
    """UniParser SDK 会用 dict 表示错误，因此必须同时检查 status。"""
    delay = REPAIR_API_INITIAL_DELAY

    for attempt in range(1, API_MAX_RETRIES + 1):
        if GLOBAL_STOP_EVENT.is_set():
            return None

        try:
            result = _run_with_deadline(
                request_func,
                UNIPARSER_REQUEST_TIMEOUT_SECONDS,
                f"UniParser {operation_name}",
            )
        except APIRequestCancelled:
            return None
        except APIRequestDeadlineExceeded as exc:
            log.error(
                f"[UniParser Timeout] {exc}; stopping this run without moving "
                "or persisting the current PDF."
            )
            _request_global_stop()
            return None
        except Exception as exc:
            result = {
                "status": "error",
                "description": f"{type(exc).__name__}: {exc}",
            }

        if _uniparser_response_ok(result):
            return result

        error_kind = _classify_uniparser_error(result)
        error_text = _uniparser_error_text(result)

        if error_kind == "fatal":
            log.critical(f"[FATAL] UniParser {operation_name} 鉴权、权限或余额检查失败: {error_text}")
            _request_global_stop()
            return None

        if error_kind == "document":
            log.error(f"[UniParser Document Error] {operation_name}: {error_text}")
            return None

        log.warning(f"[UniParser Retry {attempt}/{API_MAX_RETRIES}] {operation_name}: {error_text}")
        if attempt == API_MAX_RETRIES:
            return None
        if GLOBAL_STOP_EVENT.wait(delay):
            return None
        delay *= 2

    return None


def _is_transient_repair_network_error(exc):
    """只识别结构解析 Gemini 的临时传输故障。"""
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True

    exc_type = type(exc)
    type_name = f"{exc_type.__module__}.{exc_type.__name__}".lower()
    type_hints = (
        "timeout",
        "transporterror",
        "networkerror",
        "connectionerror",
        "connecterror",
        "socketerror",
        "proxyerror",
    )
    if any(hint in type_name for hint in type_hints):
        return True

    error_msg = str(exc).lower()
    message_hints = (
        "timed out",
        "timeout",
        "connection reset",
        "connection aborted",
        "connection refused",
        "connection closed",
        "server disconnected",
        "remote end closed",
        "broken pipe",
        "network is unreachable",
        "temporary failure in name resolution",
        "name or service not known",
        "dns lookup",
        "connect error",
        "proxy error",
    )
    return any(hint in error_msg for hint in message_hints)


def retry_repair_api():
    """脚本2的安全重试与熔断逻辑，供 IUPAC 映射和结构推理使用。"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if GLOBAL_STOP_EVENT.is_set():
                return None

            delay = REPAIR_API_INITIAL_DELAY
            for attempt in range(1, API_MAX_RETRIES + 1):
                if GLOBAL_STOP_EVENT.is_set():
                    return None

                try:
                    return func(*args, **kwargs)
                except APIRequestCancelled:
                    return None
                except APIRequestDeadlineExceeded as exc:
                    log.error(f"[Repair Timeout] 不重复提交同一请求: {exc}")
                    return None
                except Exception as exc:
                    error_msg = str(exc).lower()
                    is_rate_limited = any(keyword in error_msg for keyword in (
                        "429",
                        "resource_exhausted",
                        "rate limit",
                        "too many requests",
                    ))
                    is_context_error = "token" in error_msg and any(
                        keyword in error_msg for keyword in ("context", "maximum", "too long")
                    )
                    is_gemini_fatal = any(keyword in error_msg for keyword in (
                        "quota",
                        "billing",
                        "api key",
                        "authentication",
                        "unauthorized",
                        "forbidden",
                        "permission denied",
                    )) and not is_rate_limited

                    if is_gemini_fatal:
                        log.critical(f"[FATAL] 修复 Gemini API 额度、鉴权或权限错误: {exc}")
                        _request_global_stop()
                        return None

                    if is_context_error:
                        log.error(f"[Repair Context Error] 上下文过长，停止重复请求: {exc}")
                        return None

                    if not _is_transient_repair_network_error(exc):
                        log.error(
                            f"[Repair No Retry] 非网络错误，不重复请求 "
                            f"({func.__name__}): {exc}"
                        )
                        return None

                    log.warning(f"[Repair Retry {attempt}/{API_MAX_RETRIES}] {func.__name__}: {exc}")
                    if attempt == API_MAX_RETRIES:
                        return None
                    if GLOBAL_STOP_EVENT.wait(delay):
                        return None
                    delay *= 2

            return None
        return wrapper
    return decorator


def test_uniparser_connection():
    """运行时前置健康检查；静态验证阶段不会调用。"""
    log.info("正在进行前置检查: 验证 UniParser API 可用性...")
    try:
        result = _run_with_deadline(
            parser_client.health,
            30,
            "UniParser health check",
        )
    except Exception as exc:
        log.critical(f"UniParser API 健康检查调用失败: {type(exc).__name__}: {exc}")
        return False

    if _uniparser_response_ok(result, health_check=True):
        log.info("UniParser API 状态正常。")
        return True

    log.critical(f"UniParser API 健康检查失败: {_uniparser_error_text(result)}")
    return False


def test_repair_gemini_connection():
    """运行时验证负责 IUPAC/INFERENCE 的 Gemini 客户端。"""
    log.info("正在进行前置检查: 验证结构解析 Gemini API 可用性...")
    try:
        response = _generate_content_with_deadline(
            repair_client,
            model=MODEL_PRO_NAME,
            contents=["这是连通性检查。请只回复 Pong，不要输出其他内容。"],
            config=types.GenerateContentConfig(temperature=0.1)
        )
        if response.text and "pong" in response.text.lower():
            log.info("结构解析 Gemini API 状态正常。")
            return True
        log.critical("结构解析 Gemini API 返回了非预期的健康检查响应。")
        return False
    except Exception as exc:
        log.critical(f"结构解析 Gemini API 连接失败: {exc}")
        return False

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
    你是一名严谨的学术期刊信息检索专家。请判断下列期刊最新的 Clarivate Journal Impact Factor（IF）：
    期刊简称："{journal_abbr}"
    期刊全称："{journal_full}"

    - 如果你确信具体或近似的数值，只能回复该数值，例如 4.5 或 12.3。
    - 如果该期刊非常冷门、属于掠夺性期刊、未被 SCI/JCR 收录，或者你无法有把握地确定其 IF，只能回复 "NA"。
    - 禁止臆测或猜测；禁止输出任何解释、单位、Markdown 或其他文字。
    """
    
    rate_limiter.acquire()
    response = _generate_content_with_deadline(
        client,
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
请仅依据所提供的文献页面提取以下元数据：DOI、期刊大写简称、期刊全称、出版年份以及文章类型。
`article_type` 只能是固定值 "Review" 或 "Research"。

【文章类型关键判定规则】
- "Research"：文章报告新的原创实验工作，通常包含或引出 Materials and Methods、Experimental Section、Synthesis、原始化合物数据或原始生物活性实验。即使引言中包含文献综述，只要文章旨在报告新的实验数据或新化合物，就必须判为 "Research"。
- "Review"：文章只总结已经发表的文献，例如系统综述或概述，并且没有文章自身完成的新实验、新化合物合成或原始生物活性数据。
- 默认规则：只要存在疑问，或者无法 100% 确认文章是纯综述，就判为 "Research"。

【字段要求】
- `doi`：忠实抄录页面中的 DOI，不得补写或猜测。
- `journal_abbreviation`：使用期刊简称并转换为大写。
- `journal_full_name`：忠实抄录期刊全称，不要用简称替代。
- `publication_date`：只填写四位出版年份。
- 任一字段缺失时填写固定值 "Unknown"。

只返回严格有效的 JSON 对象，不要输出 Markdown、解释或其他文字：
{
  "doi": "10.1021/jmxxxx", 
  "journal_abbreviation": "J MED CHEM", 
  "journal_full_name": "Journal of Medicinal Chemistry",
  "publication_date": "2023",
  "article_type": "Research"
}
"""

PROMPT_TABLE_ACTIVITY = """
你是一名严谨的药物化学专家。请严格且仅从本指令之后提供的 UniParser 结构化表格块中，提取化合物的抗结核及抗分枝杆菌活性数据。

【绝对来源边界——关键否决规则】
- 输入内容只包括表格结构、表题、表注、脚注以及页码和表格编号。
- 只能使用这些表格块。严禁从摘要、引言、结果段落、结论、普通正文、分子节点、反应路线、化学图片或背景知识中补充任何活性值。
- 如果某项活性只出现在叙述性正文中，而没有明确出现在所提供的表格块中，则该信息不属于本任务，绝对不得提取。
- 表题或表注只能用于解释其所属表格中的列名、菌株、单位、时间点、缩写或实验条件，不得用于补充其他表格。
- 表格可能跨页延续，续表可能省略重复表头。只能使用输入中明确给出的前序表头或表题，不得臆造缺失表头。

【任务步骤——必须严格按顺序执行】
1. 扫描表格中的生物学对象：
   - 只接受针对 Mycobacterium 属物种的活性，包括但不限于 M. tuberculosis、M. tb、MTB、H37Rv、Erdman、MDR-TB、XDR-TB、M. bovis BCG、M. abscessus、M. avium complex、M. marinum、M. kansasii、M. smegmatis、mc2 155、M. aurum 或 M. phlei。
   - 实验必须代表全细胞体外抗分枝杆菌实验。不得把分离蛋白或纯酶实验当作全细胞活性。
   - 识别表格实际使用的化合物编号，优先保留表内局部编号，例如 5a、IV、3 或 RIF，必须忠实抄录且不得自行改号。
   - 如果表格没有独立编号，而是直接以化合物名或药物名作为行标签，则将该名称原样写入 `compound_id`。
   - 如果表格另有明确的化合物名称列，则将其原样写入 `compound_name`；否则填写空字符串。

2. 不按活性强弱筛选，保留所有被测试化合物：
   - 只要某一化合物行在合格的 Mycobacterium 目标列中有明确表格记录，就必须提取，不论结果强、弱、极差，或作者是否重点讨论。
   - 例如 ">900 µM" 仍是有效表格数据，必须保留。
   - 如果目标单元格明确写有 "NA"、"N/A"、"ND"、"-"、"inactive"、"not active" 或 "no inhibition"，也必须保留该化合物记录。
   - 将上述定性文字原样写入对应的活性数值字段，并将 `is_qualitative_inactive` 设为 JSON 布尔值 true。
   - 不得把真正的空白单元格推断为 inactive。目标单元格为空且没有明确实验结果时，不构成活性记录。

3. 执行全局目标锁定：
   - 忽略所有非分枝杆菌细菌、真菌、寄生虫、病毒以及泛抗微生物面板中的其他目标值。
   - 忽略哺乳动物细胞毒性和毒性字段，例如 CC50、TC50、Vero、HeLa、RAW、HepG2、selectivity index 和 therapeutic index。
   - 忽略纯酶或靶蛋白实验，包括 Ki，以及针对 InhA、KatG、Mtb Ung、DprE1、ATP synthase 或任何分离酶和蛋白的实验。本任务只提取全细胞抗分枝杆菌效果。
   - 如果同一表格同时含有有效的 Mycobacterium 全细胞列和应排除的列，只提取有效目标列。

4. 准确保留活性类型、数值和单位：
   - 不等式（如 "<0.1"、">64"）、近似符号和范围（如 "0.5-1.0"）必须按表格原样保留。
   - 单位必须从表头或表注中原样提取，特别注意 ng/mL、µg/mL、mg/L、nM、µM、mM 和 M；绝对不得猜测或默认单位。
   - 同一化合物如果针对多个 Mycobacterium 菌株进行测试，每个菌株必须分别建立一个 JSON 对象，不得合并。
   - `mic_value` 只能填写标准 MIC、MIC90 或 MIC99，绝对不得填写 MIC50。
   - `ic50_value` 只能填写 Mycobacterium 全细胞实验的 IC50，绝对不得填写 MIC50、IC90、CC50 或 TC50。
   - `pic50_value` 只能填写 pIC50 或 -log(IC50)。
   - `pmic_value` 只能填写 pMIC、log(1/MIC) 或 -log(MIC)，包括 QSAR 表格中的相应数据。
   - `inhibition_value` 只填写明确报告的抑制百分比；`inhibition_conc` 必须原样填写其明确给出的测试浓度及单位。
   - `zoi_value` 只填写抑菌圈（zone of inhibition）结果，保留原始数值、不等式或范围，不得将其他指标放入该字段。
   - 一个对象可以保留同一化合物、同一菌株在表格中明确给出的多个不同活性指标；不得把一种指标的值复制到另一字段。

5. 时间点和成对数值规则：
   - 如果 MIC 或 IC50 同时报告 7、14、21 天等多个培养时间，优先提取 14 天结果。
   - 如果没有 14 天结果，则提取最长培养时间的结果。
   - 如果单元格以 "2.5 (8.6)" 形式同时给出不同单位下的两个值，则去掉括号及括号内数值，只保留主要值 "2.5"。
   - 禁止暗中换算单位。

【菌株识别与危害等级】
将合格菌株按以下固定层级写入 `hazard_level`，其中 A 为最高级：
- A 级：M. tuberculosis（M. tb、MTB、H37Rv、Erdman、MDR-TB、XDR-TB）以及 M. bovis（BCG）。
- B 级：M. abscessus（M. abs）以及 M. avium complex（MAC）。
- C 级：M. marinum 以及 M. kansasii。
- D 级：M. smegmatis（M. smeg、mc2 155）、M. aurum 以及 M. phlei。
- 未列入以上层级的其他 Mycobacterium 物种仍可提取，但 `hazard_level` 必须为空字符串，不得自行分级。

【输出格式——严格 JSON 数组】
[
  {
    "compound_id": "5a",
    "compound_name": "",
    "target_strain": "M. tuberculosis H37Rv",
    "hazard_level": "A",
    "mic_value": "0.5",
    "mic_unit": "µg/mL",
    "ic50_value": "",
    "ic50_unit": "",
    "inhibition_value": "",
    "inhibition_conc": "",
    "pic50_value": "",
    "pmic_value": "",
    "zoi_value": "",
    "is_qualitative_inactive": false,
    "ignored_toxicity_or_other_data": "",
    "extraction_source": "第8页 | 表2",
    "needs_check": false
  }
]

【活性分类参考——只用于与后处理逻辑保持一致，绝对不得据此过滤记录】
- 定性 inactive、not active、no inhibition、NA、N/A、ND 或 "-"：Inactive。
- MIC/IC50（µg/mL 或等效的 mg/L）：≤1.0 为 Excellent；>1.0 且 ≤10.0 为 Good；>10.0 且 ≤64.0 为 Moderate；>64.0 为 Inactive。
- MIC/IC50（µM）：≤2.0 为 Excellent；>2.0 且 ≤20.0 为 Good；>20.0 且 ≤100.0 为 Moderate；>100.0 为 Inactive。
- inhibition：≥90% 为 Excellent；≥80% 且 <90% 为 Good；≥20% 且 <80% 为 Moderate；<20% 为 Inactive。
- pIC50：≥6.0 为 Excellent；≥5.0 且 <6.0 为 Good；<5.0 为 Moderate/Inactive。
- pMIC：≥6.0 为 Excellent；≥5.0 且 <6.0 为 Good；≥4.2 且 <5.0 为 Moderate；<4.2 为 Inactive。
- ZOI：≥25 为 Good (ZOI)；≥15 且 <25 为 Moderate (ZOI)；<15 为 Inactive；ZOI 永远不得判为 Excellent。
- 最终 Activity Level 由程序按危害等级及 MIC、IC50、pIC50、pMIC、inhibition、ZOI 的既定优先级计算。本阶段不要输出 `activity_level`，也不得删除低活性记录。

【最终严格排除与输出规则】
1. 如果活性不是在表格中明确针对 Mycobacterium 物种的全细胞实验结果，则不得提取。
2. 明确的 NA、ND、"-" 或 inactive 等目标记录必须保留并标记；空白单元格不得转换为 inactive。
3. 本阶段绝对不得生成 IUPAC 名称或 SMILES。
4. `extraction_source` 必须使用输入中给出的实际页码和表格编号；`ignored_toxicity_or_other_data` 固定为空字符串。
5. 只有在单元格确实存在但 OCR 字符无法可靠辨认时才将 `needs_check` 设为 true；不得用猜测值替代模糊字符。
6. 只能返回 JSON 数组，不得输出说明、Markdown 代码围栏或推理过程，也不得添加示例之外的字段。
7. 如果相同活性结果在多个已提供表格中重复出现，优先保留信息更精确的表格值；不同菌株、实验类型、单位或时间点绝对不得合并。
"""


PROMPT_IUPAC_EVIDENCE_FILTER = """
你是一名召回优先的药物化学文献证据筛选员。当前只执行低思考证据筛选：从本批 UniParser 原始文本块中选择可能帮助后续模型建立“目标化合物 ID—文献原文系统名称”关系的 Block ID。

【目标化合物 ID 或表格名称】
{target_ids}

【必须保留的证据类型】
1. Experimental、Chemistry、Synthesis、Preparation、General Procedure、Compound Characterization、Materials and Methods 等章节的标题和正文。
2. 含有目标编号、正式化学名称、产物标题、制备步骤、表征数据、NMR、HRMS、LCMS 或熔点的段落。
3. 目标编号附近的标题、前后段落、跨栏续写或跨页延续内容。
4. 如果某个块可能属于多个目标或暂时无法确定具体目标，把它放入 `global_block_ids`。

【筛选规则】
- 宁可多保留，也不能漏掉可能的系统名称或其上下文。
- 只能依据原文明确编号、小节标题、同一句或明确交叉引用建立目标映射。
- 不得提取、修复或输出化学名称，不得输出 SMILES，也不得解释筛选理由。
- 只能返回输入中真实存在的 Block ID。

【本批 UniParser 原始文本块】
{blocks_context}

只能返回一个 JSON 对象，不得输出 Markdown 或解释：
{{
  "global_block_ids": ["TXT-0001"],
  "target_blocks": [
    {{"compound_id": "5a", "block_ids": ["TXT-0002", "TXT-0003"]}}
  ]
}}
"""


PROMPT_STRUCTURE_EVIDENCE_FILTER = """
你是一名召回优先的药物化学结构证据筛选员。当前任务不是推导 SMILES；唯一任务是从本批 UniParser 原始 molecule、moleculeid、表格和反应路线记录中，为每个目标化合物选择后续高思考推理可能需要的 Evidence ID。

【待解决的目标化合物 ID】
{target_ids}

【必须保留的证据类型】
1. 明确出现目标 ID、相同系列编号、moleculeid、完整 Core、马库什母核、`<sep>`、`(*)`、R1/R2/R3 定义或取代基名称的记录。注意：`<sep>` 只是结构主体与锚点注释的分隔符，完整结构也可能带有它；筛选阶段不得仅因出现 `<sep>` 就把记录判为马库什或删除。
2. 含有目标行、表头、R 基列、跨页续表、表题、表注或占位符映射的完整表格记录。不得只选目标行而丢掉表头或同一完整表格。
3. 母核与取代基可能位于不同记录或不同页面；只要可能共同形成证据链，就都要保留。
4. 合成路线、反应物—产物关系、连接位置、键型或立体化学可能帮助确定目标结构时必须保留。
5. 如果记录可能服务多个目标，或者暂时无法确定归属，把它放入 `global_evidence_ids`，不得因映射不清而删除。

【筛选原则】
- 这是高召回预筛选。宁可多保留，也不能遗漏可能的母核、取代基、表格行或路线证据。
- 不得生成、修复或猜测 SMILES，不得判断最终结构是否正确。
- 只能返回输入中真实存在的 Evidence ID。

【本批 UniParser 原始结构证据】
{evidence_context}

只能返回一个 JSON 对象，不得输出 Markdown 或解释：
{{
  "global_evidence_ids": ["MOL-0001", "TAB-0001"],
  "target_evidence": [
    {{"compound_id": "5a", "evidence_ids": ["MOL-0002", "TAB-0002"]}}
  ]
}}
"""


PROMPT_IUPAC_MAPPING = """
你是一名药物化学名称证据提取专家。请在同一次高思考调用中，为下列多个目标查找文献原文明确写出的系统性 IUPAC 名称。

【目标化合物编号或表格名称】
{target_ids}

【证据规则】
1. 只能使用下方提供的 UniParser 正文片段和分子题注上下文，并且只能用于化合物编号与名称映射。
2. `raw_iupac_name` 必须忠实抄录原文；`iupac_name` 可以修复 PDF/OCR 断词、错字、字符混淆和排版问题，使其成为可被 OPSIN 解析的完整系统名称。
3. 目标编号与名称必须由原文编号、小节标题、同一句或明确交叉引用直接支持，不得只依据版面位置或邻近关系。
4. 可以使用化学知识修复明显 OCR 错误，但不得在原文不存在候选名称时根据结构反向生成名称。
5. 同一目标存在多个明确候选名称时分别返回，不得自行消除冲突。
6. `evidence_block_ids` 只能填写上下文中真实存在并直接支持名称及编号映射的 Evidence/Block ID。
7. 某一目标不存在明确候选名称时直接省略，不得输出失败原因或占位对象。

【UniParser 正文证据上下文】
{text_context}

【UniParser 分子及题注证据上下文】
{molecules_context}

只能返回以下结构的 JSON 数组，不得输出 Markdown 代码围栏、解释或推理过程：
[
  {{
    "compound_id": "5a",
    "raw_iupac_name": "原文中的名称",
    "iupac_name": "完成 OCR 与排版规范化后的系统名称",
    "evidence_block_ids": ["TXT-0002"]
  }}
]
"""


PROMPT_FIGURE_FILTER = """
你是一名极其谨慎的药物化学 Figure 证据筛选员。当前只判断这一张文献 Figure 图像（可能已等比例降采样）是否“可能”帮助后续模型为指定目标化合物确定结构、母核、连接关系或取代基；本阶段绝对不得生成、修复或猜测 SMILES。

【待解决的目标化合物 ID】
{target_ids}

【Figure 元数据】
页码：{page}
题注或周边说明：{caption}

【保守保留规则——避免漏掉 UniParser 未识别的取代基】
只要图片中存在下列任一可能性，就必须把 `is_relevant` 设为 JSON 布尔值 true：
1. 出现目标 ID、同系列化合物编号、编号与结构之间的对应关系，或可能包含目标 ID 的化学结构阵列。
2. 出现完整化学结构、马库什母核、R1/R2/R3 等变量位点、取代基定义、结构通式、构效关系结构图或取代基表。
3. 出现合成路线、反应物到产物的结构变化、试剂连接关系，且可能帮助确定目标化合物的最终结构。
4. UniParser 的文字、表格或 molecule OCR 可能无法正确读取图片中的小字号编号、键型、立体化学、取代基标签或结构。不得因为题注没有写出目标 ID，或者 OCR 上下文缺失，就否决一张视觉上包含化学结构信息的图片。
5. 即使你不能在本阶段确定图片中的结构，只要它可能与目标系列的母核或取代基拼装有关，也应保留给后续高思考模型复核。

【可以排除的图片】
只有在你能明确确认图片不包含任何可用于化合物结构确定的信息时，才可返回 false，例如纯生物学流程图、动物照片、显微图、单纯活性柱状图/折线图、药代动力学图、无化学结构的机制示意图或装饰性图片。

【绝对限制】
- 本阶段只是召回优先的筛选器，不负责判断最终结构是否正确。
- `is_relevant` 必须是 JSON 布尔值 true/false，不得使用字符串或数字。

只能返回一个 JSON 对象，不得输出 Markdown、解释或额外字段：
{{
  "is_relevant": true,
  "reason": "一句话说明图片中可能存在的结构证据类型"
}}
"""


PROMPT_STRUCTURE_INFERENCE = """
你是一位严谨的高级药物化学计算专家。
你的前置证据包括工业级多模态 OCR 引擎 UniParser 提取的结构化文本，以及经过低思考筛选后、可能包含结构或取代基信息的文献 Figure 图像。这些图像可能已经等比例降采样，但不会改变原始版面关系。
你的任务是：在同一次推理中综合分子母核、表格、反应路线和相关 Figure 图像，严格按照规则进行逻辑拼装，为指定的“目标化合物 ID”推导标准 SMILES。

【目标化合物 ID 清单】
{target_ids}
注意：你只需且只能输出该清单中的化合物，绝对不得输出文献中出现的其他化合物。

【编号清理与映射——关键规则】
文献原文中的化合物编号经常会粘连产率、剂型或多余符号（例如："1a (98%)", "1a, 1b", "Compound 1a"）。
你必须在有明确证据的前提下规范化编号，剥离 "Compound"、产率、括号和分隔符等附加信息；不得把相似但不同的编号（例如 1、1a、11a）模糊合并。
- 在最终 JSON 中，`compound_id` 必须严格等于目标清单中的原始纯净编号，绝对不能包含括号、产率或其他附加字符。
- 如果 `moleculeid` 原文类似 "8a: R1=Ph; R2=Me"，冒号后的内容不是噪声，而是必须保留并用于结构拼装的取代基定义。只能清理最终输出的 `compound_id`，推理过程中不得删除这些定义。

=======================================================
【UniParser 数据格式说明与跨区检索指南——防止幻觉】
模块 1：分子、母核、反应路线及相关题注区（分子上下文）
格式范例：`页码: 4 | 化合物编号（ID）: 4a | 结构母核（Core）: c1ccccc1 | 周边文字（Text）: Yield: 85%`
- `化合物编号（ID）`：UniParser 返回的原始 `moleculeid`。它可能同时含有化合物编号和 `R1=...` 等取代基定义；后者必须完整保留并用于拼装。如果该项为 `[Route]`，表示内容来自合成路线。
- `结构母核（Core）`：识别出的 UniParser 扩展 SMILES。包含 `>>` 表示合成路线。`<sep>` 只分隔化学结构主体与后续锚点注释，完整结构也可能出现 `<sep>`；只有结构主体含 `*`/`[*]` 等虚原子，或 `<sep>` 后的锚点明确标为 R、R1/R2、X、Ar、HetAr 等变量时，才能判定为带有未决取代基的马库什母核。例如 `c1(*)cc(*)ccc1<sep><a>1:R1</a>` 是马库什结构，而“不含虚原子且注释不含变量、结构主体本身可形成完整合法分子”的记录属于完整结构。
- `周边文字（Text）`：紧贴该结构的版面文字，只能作为编号、取代基或路线映射证据。

【核心推导逻辑——必须严格按优先级执行】
对每一个目标 ID 进行全文数据扫描，并严格按以下三个优先级处理：

【最高优先级冲突原则】
- 同一目标 ID 可能同时对应带占位符的通用母核和不含占位符的完整独立结构。只有当完整结构与该 ID 的映射明确、唯一且不存在冲突时，才优先采用完整结构并忽略通用母核。
- 如果同一目标 ID 对应两个或更多彼此不同的完整 `Core`，必须视为结构冲突；不得挑选其中之一，不得用版面位置或化学常识消除冲突，必须返回 `is_fully_confident: false` 和空的 `repaired_smiles`。
- 如果多个完整 SMILES 只是不同写法，但你无法 100% 确认它们代表完全相同的结构，也必须按冲突处理并留空。

优先级 1（完整结构直接采用）：扫描全部数据，确认目标 ID 是否明确对应一个完整、合法、结构主体不含虚原子或未决变量且不含 `>>` 的 SMILES `Core`。不得把 `<sep>` 单独当作不完整证据；如果 `<sep>` 前的结构主体完整可解析，且后续锚点注释不含 R/X/Ar/HetAr 等变量，它仍属于完整结构。
- 动作：只有映射唯一且没有冲突时，才可直接采用去除 UniParser `<sep>` 注释后的完整化学 SMILES；不需要跨区拼装。

优先级 2（马库什母核跨区组装）：如果全文中该目标 ID 的 `Core` 结构主体含 `*`/`[*]` 等虚原子，或者 `<sep>` 后锚点注释明确含 R、R1/R2、X、Ar、HetAr 等变量，说明没有可直接采用的完整结构。仅出现 `<sep>` 而没有虚原子或变量标签时，不得启动马库什拼装。
- 动作：启动下方的取代基交叉检索，只有在母核、占位位点和全部取代基都能唯一闭环时才允许拼装。

优先级 3（反应路线推导）：如果 `Core` 含有 `>>`，或者编号标识为 `[Route]`，才进入反应路线推导。
- 动作：只有当路线明确对应目标 ID，并且反应物、产物、连接变化及最终结构都能唯一确定时才可输出；不能唯一确定时必须留空。

【取代基交叉检索——仅用于优先级 2】
由于 PDF 排版复杂，目标化合物的母核与取代基（R1、R2 等）可能在物理位置上分离。当母核自身的周边文字缺少信息时，必须在以下区域交叉检索：
1. `化合物编号（ID）` 的原始内容，例如 `8a: R1=phenyl; R2=Me`。
2. 下方的“Markdown 表格区”（经常以 R1, R2 为表头，ID 为行首的构效关系表）。
3. 带有 `化合物编号（ID）: [Route]` 的合成路线周边文字。
4. 随本请求附带的相关 Figure 图像。Figure 中的小字号化合物编号、R 基标签、连接键、立体化学或取代基可能没有被 UniParser 正确识别，必须直接查看图像进行交叉核对。
- 禁止依据同系列相邻化合物、构效关系趋势、常见取代基或背景化学知识类推出目标化合物缺失的取代基。

模块 2：Markdown 表格区（表格上下文）
- 读取长表格中的取代基时，必须执行严格的行号锁定：逐行读取 `|` 分隔符，确保目标 ID、R1、R2 等来自同一行。
- 绝对不得发生行号偏移，例如不得把第 3 行的 R1 错误拼接到第 2 行化合物上。
- 如果表头、目标行、跨页续表或取代基列的对应关系不唯一，必须按不确定处理并留空。

模块 3：相关 Figure 图像（仅作为本次 INFERENCE 的视觉证据）
- Figure 不是新的独立 SMILES 来源，也不是 UniParser `IMAGE` 来源。它只与母核、moleculeid、表格和反应路线共同参与本次 `INFERENCE`。
- 低思考筛选结果只代表图片“可能相关”，绝不代表其中的目标编号、取代基或结构已经得到确认。你必须亲自读取原图并重新核验。
- 如果 Figure 清楚展示了 UniParser 漏识别的 R1/R2、取代基名称、键型、连接位点或完整目标结构，可以将该视觉信息用于补全证据链；但目标 ID 与该视觉结构之间必须存在明确且唯一的对应关系。
- 不得仅凭结构在图片中的左右顺序、上下位置、颜色、相似骨架或系列排列，把某个结构分配给目标 ID。
- Figure、UniParser 母核、表格行或反应路线之间出现任何实质冲突时，不得自行选择看似合理的一方；必须判定为不确定并留空。
- Figure 中任何原子、键、位点、取代基文字或立体化学标记无法清楚辨认时，必须视为证据不完整，严禁猜测。
=======================================================

【马库什母核（E-SMILES）组装安全协议——关键禁令】
当 `Core` 的结构主体出现 `*`/`[*]` 等虚原子，或 `<sep>` 后明确出现 R/X/Ar/HetAr 等变量锚点时：
- 严禁进行粗暴的文本级替换！
- 唯一允许的方式是在化学拓扑层面还原二维连接关系，将有明确证据的取代基通过正确键型连接到唯一占位位点，再从头生成一个完整、合法且不含 `*`、`<sep>` 或 `>>` 的 SMILES。
- 必须形成完整证据闭环：目标 ID 唯一对应母核；母核中的每个占位符唯一对应表头；目标 ID 唯一对应表格行；该行中的每个取代基均完整可读；每个取代基的连接位点与键型均唯一确定。任一环节缺失即视为不确定。
- 不得仅因结构含金属而删除金属、拆盐或拒绝输出。如果文献证据明确且完整结构能够 100% 确定，应保留金属结构，后续程序会单独标记。

【100% 确信度安全校验机制——最高级禁令】
你必须对你输出的化学结构负全责！
在输出前，请自我校验：你是否 100% 确定取代基完美且正确地连接到了母核上？
- 只有当你完全确定（100% 确信，没有任何疑问）时，才允许将 is_fully_confident 设为 JSON 布尔值 true，并输出标准 SMILES。
- 只要存在任何疑问、多个完整结构冲突、取代基信息缺失、ID 与母核映射不唯一、ID 与表格行映射不唯一、连接位点或键型不确定、文献要求的立体化学不确定、反应路线无法唯一闭环，或者任何证据只能依靠猜测，就必须将 is_fully_confident 设为 JSON 布尔值 false，同时将 repaired_smiles 设为空字符串。
- 严禁在 is_fully_confident 为 false、缺失或无法确认时输出非空 SMILES；严禁输出带有 '*'、'<sep>' 或其他占位符的残次品。
- is_fully_confident 必须是 JSON 布尔值 true/false，不能使用字符串 "true"/"false"、数字或其他替代写法。

【数据上下文输入】
[分子母核、moleculeid、反应路线及相关题注数据]：
{molecules_context}

[表格结构、占位映射及表注数据]：
{tables_context}

[随请求附带的相关 Figure 图像索引]：
{figures_context}
实际 Figure 图像以紧随本提示词之后的多模态图片 Part 提供；每张图片前均有与此索引一致的编号、页码和题注。

【输出格式约束】
必须为目标清单中的每一个 ID 返回且只返回一个对象；即使无法确定，也必须返回 `is_fully_confident: false` 和空 `repaired_smiles`，不得省略目标，也不得输出失败原因。
只能输出如下结构的 JSON 数组，不得输出 Markdown 代码围栏、解释或推理过程：
[
  {{
    "compound_id": "4a",
    "repaired_smiles": "完整的标准SMILES(如果非100%确定，必须填空字符串)",
    "is_fully_confident": true
  }}
]
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
RAW_COMMON_TB_DRUGS = {
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
    "clinafloxacin":"NC1CCN(c2cc3c(c(Cl)c2F)c(=O)c(C(=O)O)cn3C2CC2)C1",
    "Norfloxacin":"CCN1C=C(C(=O)C2=CC(=C(C=C21)N3CCNCC3)F)C(=O)O",
    "Sarafloxacin":"O=C(O)c1cn(-c2ccc(F)cc2)c2cc(N3CCNCC3)c(F)cc2c1=O",
    "Ethambutol":"CCC(CO)CNCCNC(CC)CO",
    "Ampicillin":"CC1([C@@H](N2[C@H](S1)[C@@H](C2=O)NC(=O)[C@@H](C3=CC=CC=C3)N)C(=O)O)C",
    "Novobiocin":"CC1=C(C=CC2=C1OC(=O)C(=C2O)NC(=O)C3=CC(=C(C=C3)O)CC=C(C)C)O[C@H]4[C@@H]([C@@H]([C@H](C(O4)(C)C)OC)OC(=O)N)O",
    "Ciprofloxacine":"C1CC1N2C=C(C(=O)C3=CC(=C(C=C32)N4CCNCC4)F)C(=O)O",
    "p-aminosalicylic":"C1=CC(=C(C=C1N)O)C(=O)O",
    "Clotrimazole":"C1=CC=C(C=C1)C(C2=CC=CC=C2)(C3=CC=CC=C3Cl)N4C=CN=C4",
    "Coumermycin A1":"CC1=CC=C(N1)C(=O)O[C@H]2[C@H]([C@@H](OC([C@@H]2OC)(C)C)OC3=C(C4=C(C=C3)C(=C(C(=O)O4)NC(=O)C5=CNC(=C5C)C(=O)NC6=C(C7=C(C(=C(C=C7)O[C@H]8[C@@H]([C@@H]([C@H](C(O8)(C)C)OC)OC(=O)C9=CC=C(N9)C)O)C)OC6=O)O)O)C)O",
    "tigecycline":"CC(C)(C)NCC(=O)NC1=CC(=C2C[C@H]3C[C@H]4[C@@H](C(=O)C(=C([C@]4(C(=O)C3=C(C2=C1O)O)O)O)C(=O)N)N(C)C)N(C)C",
    "Capreomycin":"C[C@H]1C(=O)N[C@H](C(=O)N/C(=C\\NC(=O)N)/C(=O)N[C@H](C(=O)NC[C@@H](C(=O)N1)N)C2CCN=C(N2)N)CNC(=O)CC(CCCN)N.C1CN=C(NC1[C@H]2C(=O)NC[C@@H](C(=O)N[C@H](C(=O)N[C@H](C(=O)N/C(=C\\NC(=O)N)/C(=O)N2)CNC(=O)CC(CCCN)N)CO)N)N.OS(=O)(=O)O.OS(=O)(=O)O",
    "Deoxyecumicin":"CC[C@@H](C)[C@@H](C(=O)N[C@@H]1C(=O)N(C)[C@@H]([C@@H](C)O)C(=O)N[C@@H](C(C)C)C(=O)N(C)[C@@H](CC(C)C)C(=O)N[C@@H](C(C)C)C(=O)N(C)[C@@H](C(C)C)C(=O)N(C)[C@@H](C(C)C)C(=O)N(C)[C@@H](Cc2c[nH]c3cccc(OC)c23)C(=O)N[C@@H](C(C)C)C(=O)N[C@@H](Cc2ccccc2)C(=O)N[C@@H](C(C)C)C(=O)O[C@@H]1C)N(C)C(=O)[C@@H](NC(=O)[C@H](C(C)C)N(C)C)C(C)C",
    "Ecumicin":"CC[C@@H](C)[C@@H](C(=O)N[C@@H]1C(=O)N(C)[C@@H]([C@@H](C)O)C(=O)N[C@@H](C(C)C)C(=O)N(C)[C@@H](CC(C)C)C(=O)N[C@@H](C(C)C)C(=O)N(C)[C@@H](C(C)C)C(=O)N(C)[C@@H](C(C)C)C(=O)N(C)[C@@H](Cc2c[nH]c3cccc(OC)c23)C(=O)N[C@@H](C(C)C)C(=O)N[C@@H]([C@H](O)c2ccccc2)C(=O)N[C@@H](C(C)C)C(=O)O[C@@H]1C)N(C)C(=O)[C@@H](NC(=O)[C@H](C(C)C)N(C)C)C(C)C",
    "TAM16":"CNC(=O)C1=C(OC2=C1C(=C(C=C2)O)CN3CCCCC3)C4=CC=C(C=C4)O",
    "Amphotericin B":"C[C@H]1/C=C/C=C/C=C/C=C/C=C/C=C/C=C/[C@@H](C[C@H]2[C@@H]([C@H](C[C@](O2)(C[C@H](C[C@H]([C@@H](CC[C@H](C[C@H](CC(=O)O[C@H]([C@@H]([C@@H]1O)C)C)O)O)O)O)O)O)O)C(=O)O)O[C@H]3[C@H]([C@H]([C@@H]([C@H](O3)C)O)N)O",
    "Thalidomide":"O=C1CCC(N2C(=O)c3ccccc3C2=O)C(=O)N1",
    "Epalrestat":"C/C(=C\\C1=CC=CC=C1)/C=C\\2/C(=O)N(C(=S)S2)CC(=O)O",
    "Antimycin":"CCCCCC[C@@H]1[C@H]([C@@H](OC(=O)[C@H]([C@H](OC1=O)C)NC(=O)C2=C(C(=CC=C2)NC=O)O)C)OC(=O)CC(C)C",
    "Tamoxifen":"CC/C(=C(\\C1=CC=CC=C1)/C2=CC=C(C=C2)OCCN(C)C)/C3=CC=CC=C3",
    "Paclitaxel":"CC1=C2[C@H](C(=O)[C@@]3([C@H](C[C@@H]4[C@]([C@H]3[C@@H]([C@@](C2(C)C)(C[C@@H]1OC(=O)[C@@H]([C@H](C5=CC=CC=C5)NC(=O)C6=CC=CC=C6)O)O)OC(=O)C7=CC=CC=C7)(CO4)OC(=O)C)O)C)OC(=O)C",
    "Butylated hydroxytoluene":"CC1=CC(=C(C(=C1)C(C)(C)C)O)C(C)(C)C",
    "Licochalcone A":"CC(C)(C=C)C1=C(C=C(C(=C1)/C=C/C(=O)C2=CC=C(C=C2)O)OC)O",
    "mucorisocoumarin C":"COC1=CC(=C2C(=C1)C=C(OC2=O)CC(C(=O)OC)O)O",
    "peyroisocoumarin D":"C[C@@H]([C@@H](C1=CC2=CC(=CC(=C2C(=O)O1)O)OC)O)O",
    "Imipenem":"C[C@H]([C@@H]1[C@H]2CC(=C(N2C1=O)C(=O)O)SCCN=CN)O",
    "SQ109":"CC(=CCC/C(=C/CNCCNC1C2CC3CC(C2)CC1C3)/C)C",
    "Nitroimidazopyran":"C1=COC2=NC(=NC2=C1)[N+](=O)[O-]",
    "Fluconazole":"C1=CC(=C(C=C1F)F)C(CN2C=NC=N2)(CN3C=NC=N3)O",
    "Econazole":"C1=CC(=CC=C1COC(CN2C=CN=C2)C3=C(C=C(C=C3)Cl)Cl)Cl",
    "mefloquine":"C1CCNC(C1)C(C2=CC(=NC3=C2C=CC=C3C(F)(F)F)C(F)(F)F)O",
    "Actinonin":"CCCCCC(CC(=O)NO)C(=O)NC(C(=O)N1CCCC1CO)C(C)C",
    "Ilomastat":"CNC(=O)[C@H](Cc1c[nH]c2ccccc12)NC(=O)[C@@H](CC(=O)NO)CC(C)C",
    "Batimastat":"CNC(=O)[C@H](Cc1ccccc1)NC(=O)[C@H](CC(C)C)[C@H](CSc1cccs1)C(=O)NO",
    "Pravastatin Sodium":"CC[C@H](C)C(=O)O[C@H]1C[C@H](O)C=C2C=C[C@H](C)[C@H](CC[C@@H](O)C[C@@H](O)CC(=O)[O][Na])[C@H]21",
    "Tinostamustine":"Cn1c(CCCCCCC(=O)NO)nc2cc(N(CCCl)CCCl)ccc21",
    "CUDC-101":"C#Cc1cccc(Nc2ncnc3cc(OC)c(OCCCCCCC(=O)NO)cc23)c1",
    "Ascorbic acid":"C([C@@H]([C@@H]1C(=C(C(=O)O1)O)O)O)O",
    "EDTA":"C(CN(CC(=O)O)CC(=O)O)N(CC(=O)O)CC(=O)O",
    "Gemifloxacin":"CO/N=C/1\\CN(CC1CN)C2=C(C=C3C(=O)C(=CN(C3=N2)C4CC4)C(=O)O)F",
    "PBTZ169":"C1CCC(CC1)CN2CCN(CC2)C3=NC(=O)C4=C(S3)C(=CC(=C4)C(F)(F)F)[N+](=O)[O-]",
    "Sulfamethoxazole":"CC1=CC(=NO1)NS(=O)(=O)C2=CC=C(C=C2)N",
    "Ebselen":"C1=CC=C(C=C1)N2C(=O)C3=CC=CC=C3[Se]2",
    "Gentamycin":"CC(C1CCC(C(O1)OC2C(CC(C(C2O)OC3C(C(C(CO3)(C)O)NC)O)N)N)N)NC",
    "Nystatin":"C[C@H]1/C=C/C=C/CC/C=C/C=C/C=C/C=C/C(CC2C(C(C[C@](O2)(CC(C(CCC(CC(CC(CC(=O)O[C@H]([C@@H]([C@@H]1O)C)C)O)O)O)O)O)O)O)C(=O)O)O[C@@H]3[C@H]([C@H]([C@@H]([C@H](O3)C)O)N)O",
    "Greseofulvin":"C[C@@H]1CC(=O)C=C([C@]12C(=O)C3=C(O2)C(=C(C=C3OC)OC)Cl)OC",
    "Thiolactomycin":"CC1=C([C@@](SC1=O)(C)/C=C(\\C)/C=C)O",
    "Fosmidomycin":"C(CN(C=O)O)CP(=O)(O)O",
    "Quercetin":"C1=CC(=C(C=C1C2=C(C(=O)C3=C(C=C(C=C3O2)O)O)O)O)O",
}

COMMON_TB_DRUGS = {key.lower(): value for key, value in RAW_COMMON_TB_DRUGS.items()}




def contains_metal(smiles):
    """检测标准化结构中是否含金属；金属结构保留但需要在 Excel 标红。"""
    if not smiles:
        return False
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        metal_symbols = {
            "Li", "Be", "Na", "Mg", "Al", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn",
            "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Rb", "Sr", "Y", "Zr", "Nb", "Mo",
            "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Cs", "Ba", "La", "Ce",
            "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
            "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb",
            "Bi", "Po", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm",
        }
        return any(atom.GetSymbol() in metal_symbols for atom in mol.GetAtoms())
    except Exception:
        return False


def normalize_compound_identifier(value):
    """用于跨来源精确对应的保守 ID 归一化，不推断或改写化合物编号。"""
    text = str(value or "").strip()
    text = re.sub(r'^(compound|compd|derivative)\s+', '', text, flags=re.IGNORECASE)
    return re.sub(r'\s+', '', text).casefold()


def stable_unique(items):
    seen = set()
    result = []
    for item in items:
        if item in (None, "", [], {}):
            continue
        normalized = item.strip() if isinstance(item, str) else item
        try:
            hash(normalized)
            unique_key = normalized
        except TypeError:
            unique_key = json.dumps(normalized, ensure_ascii=False, sort_keys=True)
        if unique_key not in seen:
            seen.add(unique_key)
            result.append(normalized)
    return result

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
    response = _generate_content_with_deadline(
        client,
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

# ================= 8. UniParser 正文解析与表格活性抽取 =================

def _node_text(node, fields=("str", "text", "plain", "caption", "smi")):
    for field in fields:
        value = node.get(field)
        if value not in (None, ""):
            return str(value).strip()
    return ""


def _node_children(node):
    children = []
    for field in ("items", "children"):
        value = node.get(field)
        if isinstance(value, list):
            children.extend(child for child in value if isinstance(child, dict))
    return children


def _context_value(value):
    if value in (None, "", [], {}):
        return ""
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, ensure_ascii=False)


def _table_context(node):
    """兼容 HQ structure，以及 SDK 可能返回的 placeholders/contents。"""
    structure = _context_value(node.get("structure"))
    placeholders = node.get("placeholders")
    contents = node.get("contents")

    parts = []
    if structure:
        parts.append(structure)
    placeholder_text = _context_value(placeholders)
    contents_text = _context_value(contents)
    if placeholder_text:
        parts.append(f"[占位符]\n{placeholder_text}")
    if contents_text:
        parts.append(f"[映射内容]\n{contents_text}")
    return "\n".join(parts)


def _deduplicate_records(records):
    seen = set()
    result = []
    for record in records:
        key = json.dumps(record, ensure_ascii=False, sort_keys=True)
        if key not in seen:
            seen.add(key)
            result.append(record)
    return result


def _assign_record_ids(records, prefix):
    """在去重后按文档原始顺序分配稳定证据 ID，供低思考模型只返回引用。"""
    for index, record in enumerate(records, start=1):
        record["record_id"] = f"{prefix}-{index:04d}"
    return records


def _detect_figure_mime(image_bytes, declared_mime=""):
    """优先依据文件头识别 MIME；UniParser 未声明时仍可安全构造图片 Part。"""
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if image_bytes.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
        return "image/webp"
    if image_bytes.startswith(b"BM"):
        return "image/bmp"
    if image_bytes.startswith((b"II*\x00", b"MM\x00*")):
        return "image/tiff"

    normalized = str(declared_mime or "").split(";", 1)[0].strip().lower()
    if normalized.startswith("image/"):
        return normalized
    return "image/png"


def _decode_figure_value(value, declared_mime=""):
    """兼容 UniParser Figure `source` 的裸 Base64、data URL 和嵌套字典。"""
    if isinstance(value, (bytes, bytearray)):
        image_bytes = bytes(value)
        if len(image_bytes) < 64:
            return None
        return image_bytes, _detect_figure_mime(image_bytes, declared_mime)

    if isinstance(value, dict):
        nested_mime = (
            value.get("mime_type")
            or value.get("mimeType")
            or value.get("content_type")
            or value.get("contentType")
            or declared_mime
        )
        for field in ("source", "base64", "image_base64", "data", "content"):
            if field in value:
                decoded = _decode_figure_value(value.get(field), nested_mime)
                if decoded:
                    return decoded
        return None

    if not isinstance(value, str):
        return None

    raw_value = value.strip()
    if not raw_value or raw_value.lower().startswith(("http://", "https://", "file://")):
        return None

    mime_type = declared_mime
    data_url_match = re.match(
        r"^data:(image/[^;,]+)(?:;[^,]*)?;base64,(.+)$",
        raw_value,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if data_url_match:
        mime_type = data_url_match.group(1)
        raw_value = data_url_match.group(2)

    compact = re.sub(r"\s+", "", raw_value)
    if len(compact) < 64:
        return None
    compact += "=" * (-len(compact) % 4)
    try:
        image_bytes = base64.b64decode(compact, validate=True)
    except Exception:
        return None
    if len(image_bytes) < 64:
        return None
    return image_bytes, _detect_figure_mime(image_bytes, mime_type)


def _extract_figure_payload(node):
    declared_mime = (
        node.get("mime_type")
        or node.get("mimeType")
        or node.get("content_type")
        or node.get("contentType")
        or ""
    )
    for field in ("source", "base64", "image_base64", "image", "data", "content"):
        if field in node:
            decoded = _decode_figure_value(node.get(field), declared_mime)
            if decoded:
                return decoded
    return None


def _prepare_figure_for_structure_api(image_bytes, mime_type):
    """缩小发送给结构解析 Gemini 的 Figure；失败时安全回退原图。"""
    if PILImage is None or MAX_FIGURE_LONG_EDGE <= 0:
        return image_bytes, mime_type, ""

    try:
        with PILImage.open(io.BytesIO(image_bytes)) as source_image:
            source_image.load()
            original_width, original_height = source_image.size
            if max(original_width, original_height) <= MAX_FIGURE_LONG_EDGE:
                return image_bytes, mime_type, ""

            resized_image = source_image.copy()
            resampling = getattr(PILImage, "Resampling", PILImage)
            resized_image.thumbnail(
                (MAX_FIGURE_LONG_EDGE, MAX_FIGURE_LONG_EDGE),
                resampling.LANCZOS,
            )

            output = io.BytesIO()
            normalized_mime = str(mime_type or "").lower()
            if normalized_mime == "image/jpeg":
                if "A" in resized_image.getbands():
                    rgba_image = resized_image.convert("RGBA")
                    flattened = PILImage.new("RGB", rgba_image.size, "white")
                    flattened.paste(rgba_image, mask=rgba_image.getchannel("A"))
                    resized_image = flattened
                elif resized_image.mode not in {"RGB", "L"}:
                    resized_image = resized_image.convert("RGB")
                resized_image.save(
                    output,
                    format="JPEG",
                    quality=FIGURE_JPEG_QUALITY,
                    optimize=True,
                )
                output_mime = "image/jpeg"
            else:
                if resized_image.mode == "CMYK":
                    resized_image = resized_image.convert("RGB")
                resized_image.save(output, format="PNG", optimize=True)
                output_mime = "image/png"

            new_width, new_height = resized_image.size
            return (
                output.getvalue(),
                output_mime,
                f"{original_width}x{original_height}->{new_width}x{new_height}",
            )
    except Exception as exc:
        log.warning(
            f"Figure resize failed; using original image: "
            f"{type(exc).__name__}: {exc}"
        )
        return image_bytes, mime_type, ""


def _figure_caption_text(node):
    captions = []
    for field in ("caption", "desc", "description", "title", "text", "str"):
        value = node.get(field)
        if value not in (None, ""):
            captions.append(str(value).strip())
    for child in _node_children(node):
        child_type = str(child.get("type", "")).strip().lower()
        if "caption" in child_type:
            child_text = _node_text(child)
            if child_text:
                captions.append(child_text)
    return " | ".join(dict.fromkeys(text for text in captions if text))


def _register_figure(figure_map, node, page_number, inherited_caption=""):
    payload = _extract_figure_payload(node)
    if not payload:
        return
    image_bytes, mime_type = payload
    digest = hashlib.sha256(image_bytes).hexdigest()
    caption = " | ".join(dict.fromkeys(
        text for text in (str(inherited_caption or "").strip(), _figure_caption_text(node))
        if text
    ))

    existing = figure_map.get(digest)
    if existing is None:
        api_image_bytes, api_mime_type, resize_note = _prepare_figure_for_structure_api(
            image_bytes, mime_type
        )
        figure_map[digest] = {
            "page": page_number,
            "caption": caption,
            "mime_type": api_mime_type,
            "image_bytes": api_image_bytes,
            "sha256": digest,
            "resize_note": resize_note,
        }
        return

    merged_caption = " | ".join(dict.fromkeys(
        text for text in (existing.get("caption", ""), caption) if text
    ))
    existing["caption"] = merged_caption


def call_uniparser_extract(pdf_path):
    """解析整篇文献，并准备降采样后的 Figure 供多模态结构推理使用。"""
    trigger_result = _call_uniparser_with_retry(
        "提交解析",
        lambda: parser_client.trigger_file(
            file_path=pdf_path,
            textual=ParseModeTextual.OCRFast,
            table=ParseMode.OCRHighQuality,
            molecule=ParseMode.OCRFast,
            chart=0,
            figure=ParseMode.DumpBase64,
            equation=0,
            expression=0,
            sync=True
        )
    )
    if not trigger_result:
        return None

    token = trigger_result.get("token")
    if not token:
        log.error(f"UniParser 成功响应中缺少 token: {trigger_result}")
        return None

    formatted_result = _call_uniparser_with_retry(
        "获取格式化结果",
        lambda: parser_client.get_formatted(
            token,
            content=False,
            pages_tree=True
        )
    )
    if not formatted_result:
        return None

    pages_tree = formatted_result.get("pages_tree") or []
    if not isinstance(pages_tree, list):
        log.error(f"UniParser pages_tree 格式异常: {type(pages_tree).__name__}")
        return None

    table_blocks = []
    molecule_records = []
    text_blocks = []
    figure_map = {}

    def add_text_block(page_number, node_type, text):
        clean = str(text or "").strip()
        if clean:
            text_blocks.append({
                "page": page_number,
                "type": node_type,
                "text": clean,
            })

    def traverse(node_list, page_number):
        if isinstance(node_list, dict):
            node_list = [node_list]
        if not isinstance(node_list, list):
            return

        for node in node_list:
            if not isinstance(node, dict):
                continue

            ntype = str(node.get("type", "")).strip().lower()
            children = _node_children(node)

            if ntype in {"figure", "image", "figuregroup", "imagegroup"}:
                group_captions = [
                    _node_text(child)
                    for child in children
                    if "caption" in str(child.get("type", "")).strip().lower()
                    and _node_text(child)
                ]
                group_caption = " | ".join(dict.fromkeys(group_captions))
                _register_figure(figure_map, node, page_number, group_caption)
                for child in children:
                    child_type = str(child.get("type", "")).strip().lower()
                    if child_type in {"figure", "image"}:
                        _register_figure(figure_map, child, page_number, group_caption)

            if ntype == "tablegroup":
                captions = []
                tables = []
                remaining_children = []
                for child in children:
                    child_type = str(child.get("type", "")).strip().lower()
                    if child_type == "table":
                        table_text = _table_context(child)
                        if table_text:
                            tables.append(table_text)
                    elif "caption" in child_type or child_type == "tablefootnote":
                        caption_text = _node_text(child)
                        if caption_text:
                            captions.append(caption_text)
                    else:
                        remaining_children.append(child)

                caption_text = " | ".join(stable_unique(captions))
                for table_text in stable_unique(tables):
                    table_blocks.append({
                        "page": page_number,
                        "caption": caption_text,
                        "text": table_text,
                    })
                traverse(remaining_children, page_number)
                continue

            child_types = [str(child.get("type", "")).strip().lower() for child in children]
            is_molecule_group = (
                "group" in ntype
                and any(child_type in {"molecule", "moleculeid"} for child_type in child_types)
            )

            if is_molecule_group:
                cores = []
                molecule_ids = []
                nearby_texts = []
                sequence = []
                consumed_indexes = set()

                for index, (child, child_type) in enumerate(zip(children, child_types)):
                    if child_type == "molecule":
                        core_text = _node_text(child)
                        if core_text:
                            cores.append(core_text)
                            sequence.append({"position": index + 1, "type": child_type, "value": core_text})
                        consumed_indexes.add(index)
                    elif child_type == "moleculeid":
                        id_text = _node_text(child)
                        if id_text:
                            molecule_ids.append(id_text)
                            sequence.append({"position": index + 1, "type": child_type, "value": id_text})
                        consumed_indexes.add(index)
                    elif child_type in {
                        "moleculecaption",
                        "imagecaption",
                        "expressioncaption",
                    }:
                        caption_text = _node_text(child)
                        if caption_text:
                            nearby_texts.append(caption_text)
                            sequence.append({"position": index + 1, "type": child_type, "value": caption_text})
                            add_text_block(page_number, child_type, caption_text)
                        consumed_indexes.add(index)

                if cores or molecule_ids or nearby_texts:
                    molecule_records.append({
                        "page": page_number,
                        "ids": stable_unique(molecule_ids),
                        "cores": stable_unique(cores),
                        "text": " ".join(stable_unique(nearby_texts)),
                        "kind": "molecule",
                        "sequence": sequence,
                    })

                remaining_children = [
                    child for index, child in enumerate(children)
                    if index not in consumed_indexes
                ]
                traverse(remaining_children, page_number)
                continue

            if ntype == "molecule":
                core_text = _node_text(node)
                if core_text:
                    molecule_records.append({
                        "page": page_number,
                        "ids": [],
                        "cores": [core_text],
                        "text": "",
                        "kind": "molecule",
                        "sequence": [{"position": 1, "type": "molecule", "value": core_text}],
                    })
            elif ntype == "moleculeid":
                id_text = _node_text(node)
                if id_text:
                    molecule_records.append({
                        "page": page_number,
                        "ids": [id_text],
                        "cores": [],
                        "text": "",
                        "kind": "molecule",
                        "sequence": [{"position": 1, "type": "moleculeid", "value": id_text}],
                    })
            elif "expression" in ntype:
                route_text = _node_text(node)
                if route_text:
                    molecule_records.append({
                        "page": page_number,
                        "ids": ["[Route]"],
                        "cores": [route_text],
                        "text": "[合成路线]",
                        "kind": "route",
                        "sequence": [{"position": 1, "type": ntype, "value": route_text}],
                    })
            elif ntype == "table":
                table_text = _table_context(node)
                if table_text:
                    table_blocks.append({
                        "page": page_number,
                        "caption": "",
                        "text": table_text,
                    })
            elif ntype in {"tablecaption", "tablefootnote"}:
                caption_text = _node_text(node)
                if caption_text:
                    table_blocks.append({
                        "page": page_number,
                        "caption": caption_text,
                        "text": "",
                    })
            elif (
                ntype in {
                    "text",
                    "paragraph",
                    "title",
                    "heading",
                    "section",
                    "list",
                    "listitem",
                    "moleculecaption",
                    "imagecaption",
                    "expressioncaption",
                }
                or "text" in ntype
            ):
                add_text_block(page_number, ntype, _node_text(node))

            traverse(children, page_number)

    for page_index, page_nodes in enumerate(pages_tree, start=1):
        traverse(page_nodes, page_index)

    document = {
        "table_blocks": _assign_record_ids(_deduplicate_records(table_blocks), "TAB"),
        "molecule_records": _assign_record_ids(_deduplicate_records(molecule_records), "MOL"),
        "text_blocks": _assign_record_ids(_deduplicate_records(text_blocks), "TXT"),
        "figure_records": _assign_record_ids(list(figure_map.values()), "FIG"),
    }
    resized_figure_count = sum(
        bool(record.get("resize_note")) for record in document["figure_records"]
    )
    if document["figure_records"] and PILImage is None:
        log.warning("Pillow is unavailable; Figure images keep their original resolution.")
    log.info(
        "   -> [UniParser] "
        f"Pages={len(pages_tree)}, Tables={len(document['table_blocks'])}, "
        f"Molecules={len(document['molecule_records'])}, Text blocks={len(document['text_blocks'])}, "
        f"Figures={len(document['figure_records'])}, Resized={resized_figure_count}"
    )
    return document


def _format_table_block(block, index):
    page = block.get("page", "未知")
    record_id = block.get("record_id", f"TAB-{index:04d}")
    caption = str(block.get("caption", "")).strip()
    table_text = str(block.get("text", "")).strip()
    return (
        f"===== Evidence ID {record_id} | 表格块 {index} | 页码 {page} =====\n"
        f"表题/表注：{caption}\n"
        f"{table_text}"
    ).strip()


def build_table_batches(table_blocks):
    """按整张表格分批，限制含表格页数与总字符数，不拆分单个表格。"""
    ordered = sorted(
        enumerate(table_blocks, start=1),
        key=lambda pair: (pair[1].get("page", 0), pair[0])
    )
    batches = []
    current_blocks = []
    current_pages = set()
    current_chars = 0

    for original_index, block in ordered:
        formatted = _format_table_block(block, original_index)
        page = block.get("page")
        next_pages = set(current_pages)
        next_pages.add(page)
        exceeds_pages = len(next_pages) > MAX_TABLE_PAGES_PER_BATCH
        exceeds_chars = current_chars + len(formatted) > MAX_TABLE_CONTEXT_CHARS

        if current_blocks and (exceeds_pages or exceeds_chars):
            batches.append(current_blocks)
            current_blocks = []
            current_pages = set()
            current_chars = 0

        current_blocks.append(formatted)
        current_pages.add(page)
        current_chars += len(formatted)

    if current_blocks:
        batches.append(current_blocks)
    return batches


@retry_api()
def call_table_activity_model(prompt):
    rate_limiter.acquire()
    response = _generate_content_with_deadline(
        client,
        model=MODEL_PRO_NAME,
        contents=[prompt],
        config=types.GenerateContentConfig(
            temperature=1.0,
            thinking_config=types.ThinkingConfig(thinking_level="high"),
            media_resolution="media_resolution_high"
        )
    )
    return response.text


def normalize_activity_item(item, batch_index):
    if not isinstance(item, dict):
        return None
    compound_id = str(item.get("compound_id", "") or "").strip()
    if not compound_id or compound_id.casefold() in {"nan", "none", "null"}:
        return None

    normalized = dict(item)
    normalized["compound_id"] = compound_id
    normalized["compound_name"] = str(item.get("compound_name", "") or "").strip()
    normalized["iupac_name"] = ""
    normalized["is_standard_iupac"] = False

    raw_inactive = item.get("is_qualitative_inactive", False)
    normalized["is_qualitative_inactive"] = (
        raw_inactive is True
        or (
            isinstance(raw_inactive, str)
            and raw_inactive.strip().casefold() == "true"
        )
    )
    if not normalized.get("extraction_source"):
        normalized["extraction_source"] = f"UniParser 表格批次 {batch_index}"
    return normalized


def extract_table_activity(table_blocks):
    """只向 Gemini 发送 UniParser 表格，保留原 JSON 截断续传行为。"""
    if not table_blocks:
        return []

    all_extracted_data = []
    table_batches = build_table_batches(table_blocks)

    for batch_index, batch_blocks in enumerate(table_batches, start=1):
        table_context = "\n\n".join(batch_blocks)
        base_prompt = (
            PROMPT_TABLE_ACTIVITY
            + f"\n\n【UniParser 表格批次 {batch_index}/{len(table_batches)}】\n"
            + table_context
        )
        response_text = call_table_activity_model(base_prompt)
        parsed_data, is_truncated = robust_json_extract(response_text)

        if parsed_data:
            for item in parsed_data:
                normalized_item = normalize_activity_item(item, batch_index)
                if normalized_item:
                    all_extracted_data.append(normalized_item)

        continuations = 0
        while is_truncated and continuations < MAX_CONTINUATIONS and parsed_data:
            last_id = str(parsed_data[-1].get("compound_id", "")).strip()
            if not last_id:
                break

            log.warning(
                f"   -> [Truncated] Table batch {batch_index} salvaged through "
                f"ID '{last_id}'. Continuation {continuations + 1}/{MAX_CONTINUATIONS}."
            )
            continuation_prompt = (
                base_prompt
                + "\n\n【系统提醒】上一次 JSON 输出被截断。"
                + f"你已输出到化合物编号 '{last_id}'。请从该编号所对应表格行之后的下一行继续，"
                + "不得重复此前记录，只返回一个新的 JSON 数组。"
            )
            continuation_text = call_table_activity_model(continuation_prompt)
            parsed_data, is_truncated = robust_json_extract(continuation_text)
            if parsed_data:
                for item in parsed_data:
                    normalized_item = normalize_activity_item(item, batch_index)
                    if normalized_item:
                        all_extracted_data.append(normalized_item)
            continuations += 1

    return all_extracted_data


# ================= 8.5 结构来源解析：DICTIONARY > IMAGE > IUPAC > INFERENCE =================

def _identifier_variants(raw_identifier):
    raw_text = str(raw_identifier or "").strip()
    if not raw_text:
        return set()
    variants = {normalize_compound_identifier(raw_text)}
    prefix = re.split(r'[:;,\(\[\{]', raw_text, maxsplit=1)[0]
    prefix_normalized = normalize_compound_identifier(prefix)
    if prefix_normalized:
        variants.add(prefix_normalized)
    return {variant for variant in variants if variant}


def _record_target_keys(record, target_keys):
    matches = set()
    for raw_identifier in record.get("ids", []):
        variants = _identifier_variants(raw_identifier)
        matches.update(variant for variant in variants if variant in target_keys)
    return matches


def _text_mentions_target(text, display_value):
    source = str(text or "")
    target = str(display_value or "").strip()
    if not source or not target:
        return False

    escaped = re.escape(target)
    if len(normalize_compound_identifier(target)) <= 2:
        patterns = (
            rf'\b(?:compound|compd|derivative)\s*{escaped}\b',
            rf'(?m)^\s*(?:compound\s*)?{escaped}(?=\s|[\.:;,\)\]\(\-])',
            rf'(?<![A-Za-z0-9])\({escaped}\)(?![A-Za-z0-9])',
            rf'(?<![A-Za-z0-9])\[{escaped}\](?![A-Za-z0-9])',
        )
        return any(re.search(pattern, source, re.IGNORECASE) for pattern in patterns)
    return re.search(rf'(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])', source, re.IGNORECASE) is not None


def build_target_index(activity_items):
    targets = {}
    for item in activity_items:
        display_id = str(item.get("compound_id", "")).strip()
        key = normalize_compound_identifier(display_id)
        if not key:
            continue
        target = targets.setdefault(key, {
            "display_id": display_id,
            "names": [],
        })
        compound_name = str(item.get("compound_name", "") or "").strip()
        if compound_name and compound_name.casefold() not in {"nan", "none", "null"}:
            target["names"].append(compound_name)

    for target in targets.values():
        target["names"] = stable_unique(target["names"])
    return targets


CHEMISTRY_SECTION_PATTERN = re.compile(
    r"\b(experimental(?:\s+section)?|chemistry|chemical\s+synthesis|synthesis|"
    r"preparation|general\s+procedure|compound\s+characteri[sz]ation|"
    r"materials\s+and\s+methods|synthetic\s+procedure)\b",
    re.IGNORECASE,
)
SECTION_NODE_TYPES = {"title", "heading", "section", "sectiontitle", "subtitle"}


def _target_search_labels(target):
    labels = [target.get("display_id", "")]
    labels.extend(target.get("names", []))
    labels.extend(target.get("dictionary_names", []))
    return [str(value).strip() for value in dict.fromkeys(labels) if str(value).strip()]


def _format_text_block_for_filter(block):
    return (
        f"[Block ID {block.get('record_id', '未知')} | 页码 {block.get('page', '未知')} | "
        f"节点类型 {block.get('type', '未知')}]\n{block.get('text', '')}"
    )


def _format_text_blocks(blocks):
    return "\n\n".join(_format_text_block_for_filter(block) for block in blocks)


IUPAC_DASH_TRANSLATION = str.maketrans({
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "−": "-", "﹘": "-", "－": "-",
})


def _normalize_iupac_unicode(value):
    """只统一排版字符；不做任何化学词根、位次或取代基纠错。"""
    text = unicodedata.normalize("NFKC", str(value or ""))
    return (
        text.replace("\u00ad", "")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\u2028", "\n")
        .replace("\u2029", "\n")
        .translate(IUPAC_DASH_TRANSLATION)
    )


def _cleanup_iupac_layout_variant(value):
    text = _normalize_iupac_unicode(value)
    text = re.sub(r"[ \t]*\n[ \t]*", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s+([,;:)\]\}])", r"\1", text)
    text = re.sub(r"([(\[\{,;:])\s+", r"\1", text)
    text = re.sub(r"\s*-\s*", "-", text)
    return re.sub(r"\s+", " ", text).strip()


def _iupac_evidence_match_key(value):
    text = _normalize_iupac_unicode(value)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*-\s*", "-", text)
    return text.strip().casefold()


def _raw_iupac_name_in_blocks(raw_name, blocks):
    needle = _iupac_evidence_match_key(raw_name)
    if not needle:
        return False
    texts = [str(block.get("text", "") or "") for block in blocks]
    haystacks = texts + ["\n".join(texts)]
    return any(needle in _iupac_evidence_match_key(text) for text in haystacks if text)


def _load_evidence_filter_object(response_text):
    """优先直接解析顶层对象，避免通用 JSON 清理器误截取对象内部数组。"""
    if response_text is None or not str(response_text).strip():
        return None
    raw_text = str(response_text).strip()
    candidates = [raw_text]
    fence = chr(96) * 3
    if fence in raw_text:
        for fenced_part in raw_text.split(fence)[1::2]:
            fenced_part = fenced_part.strip()
            if fenced_part.casefold().startswith("json"):
                fenced_part = fenced_part[4:].lstrip()
            if fenced_part:
                candidates.append(fenced_part)
    cleaned = clean_json_text(raw_text)
    if cleaned and cleaned != raw_text:
        candidates.append(cleaned)
    for candidate in candidates:
        try:
            data = json_repair.loads(candidate)
        except Exception:
            continue
        if isinstance(data, list) and len(data) == 1 and isinstance(data[0], dict):
            data = data[0]
        if isinstance(data, dict):
            return data
    return None


def _batch_records_for_evidence_filter(records, formatter):
    """按页面和字符数分批；同页记录及单个超大记录不拆分。"""
    batches, current, current_pages, current_chars = [], [], [], 0
    for record in records:
        rendered = formatter(record)
        page_key = str(record.get("page", "未知"))
        is_new_page = page_key not in current_pages
        should_split = bool(current) and is_new_page and (
            len(current_pages) >= MAX_EVIDENCE_FILTER_PAGES
            or current_chars + len(rendered) > MAX_EVIDENCE_FILTER_CHARS
        )
        if should_split:
            batches.append(current)
            current, current_pages, current_chars = [], [], 0
            is_new_page = True
        current.append(record)
        current_chars += len(rendered)
        if is_new_page:
            current_pages.append(page_key)
    if current:
        batches.append(current)
    return batches


def _parse_evidence_filter_selection(
    response_text,
    valid_record_ids,
    target_keys,
    global_field,
    target_field,
    record_ids_field,
):
    data = _load_evidence_filter_object(response_text)
    if data is None:
        return None

    def valid_ids(values):
        if isinstance(values, str):
            values = [values]
        if not isinstance(values, list):
            return set()
        return {
            str(value).strip()
            for value in values
            if str(value).strip() in valid_record_ids
        }

    global_ids = valid_ids(data.get(global_field, []))
    target_map = {key: set() for key in target_keys}
    target_entries = data.get(target_field, [])
    if isinstance(target_entries, dict):
        target_entries = [
            {"compound_id": compound_id, record_ids_field: record_ids}
            for compound_id, record_ids in target_entries.items()
        ]
    if not isinstance(target_entries, list):
        target_entries = []
    for entry in target_entries:
        if not isinstance(entry, dict):
            continue
        target_key = normalize_compound_identifier(entry.get("compound_id", ""))
        if target_key in target_map:
            target_map[target_key].update(valid_ids(entry.get(record_ids_field, [])))
    return global_ids, target_map


def _is_section_heading(block):
    node_type = str(block.get("type", "")).strip().lower()
    text = str(block.get("text", "")).strip()
    return node_type in SECTION_NODE_TYPES or (len(text) <= 180 and bool(CHEMISTRY_SECTION_PATTERN.search(text)))


def _expand_text_selection(selected_ids, text_blocks):
    """仅扩展有限邻域和最近标题，不再自动带入整个化学章节。"""
    index_by_id = {
        block.get("record_id"): index
        for index, block in enumerate(text_blocks)
        if block.get("record_id")
    }
    expanded = {record_id for record_id in selected_ids if record_id in index_by_id}
    for record_id in list(expanded):
        index = index_by_id[record_id]
        start = max(0, index - EVIDENCE_NEIGHBOR_BLOCKS)
        end = min(len(text_blocks), index + EVIDENCE_NEIGHBOR_BLOCKS + 1)
        expanded.update(
            block.get("record_id") for block in text_blocks[start:end] if block.get("record_id")
        )
        for heading_index in range(index - 1, -1, -1):
            if _is_section_heading(text_blocks[heading_index]):
                expanded.add(text_blocks[heading_index].get("record_id"))
                break
    return {record_id for record_id in expanded if record_id}


IUPAC_NMR_PATTERN = re.compile(r"NMR", re.IGNORECASE)
IUPAC_HZ_PATTERN = re.compile(r"Hz", re.IGNORECASE)


def _build_primary_iupac_evidence_blocks(text_blocks, targets):
    """构建有限主证据池，并记录不得被低思考筛选丢弃的目标邻域。"""
    page_order, page_texts = [], {}
    for block in text_blocks:
        page_key = str(block.get("page", "未知"))
        if page_key not in page_texts:
            page_order.append(page_key)
            page_texts[page_key] = []
        page_texts[page_key].append(str(block.get("text", "") or ""))

    key_pages = [
        page_key for page_key in page_order
        if IUPAC_NMR_PATTERN.search("\n".join(page_texts[page_key]))
        and IUPAC_HZ_PATTERN.search("\n".join(page_texts[page_key]))
    ]
    focus_pages = set()
    page_index = {page_key: index for index, page_key in enumerate(page_order)}
    for key_page in key_pages:
        center = page_index[key_page]
        start = max(0, center - IUPAC_NMR_PAGE_RADIUS)
        end = min(len(page_order), center + IUPAC_NMR_PAGE_RADIUS + 1)
        focus_pages.update(page_order[start:end])

    candidate_ids = {
        block.get("record_id")
        for block in text_blocks
        if str(block.get("page", "未知")) in focus_pages and block.get("record_id")
    }
    mandatory_by_target = {key: set() for key in targets}
    for index, block in enumerate(text_blocks):
        block_text = block.get("text", "")
        for key, target in targets.items():
            if any(
                _text_mentions_target(block_text, label)
                for label in _target_search_labels(target)
            ):
                direct_ids = _expand_text_selection(
                    {block.get("record_id")}, text_blocks
                )
                mandatory_by_target[key].update(direct_ids)
                candidate_ids.update(direct_ids)

        if (
            _is_section_heading(block)
            and CHEMISTRY_SECTION_PATTERN.search(str(block_text))
        ):
            for nearby in text_blocks[index:min(len(text_blocks), index + 3)]:
                if nearby.get("record_id"):
                    candidate_ids.add(nearby["record_id"])

    return (
        [block for block in text_blocks if block.get("record_id") in candidate_ids],
        mandatory_by_target,
    )


def select_iupac_evidence_blocks(document, targets):
    """低思考只筛选 Block ID；异常批次降级，并保留确定性目标邻域。"""
    stats = {"filter_calls": 0, "prompt_chars": 0, "invalid_json": 0}
    all_text_blocks = document.get("text_blocks", [])
    if not all_text_blocks:
        return {key: [] for key in targets}, stats

    text_blocks, mandatory_by_target = _build_primary_iupac_evidence_blocks(
        all_text_blocks, targets
    )
    if not text_blocks:
        return {key: [] for key in targets}, stats

    valid_ids = {block.get("record_id") for block in text_blocks if block.get("record_id")}
    target_keys = set(targets)
    global_ids = set()
    target_ids = {key: set() for key in targets}
    target_prompt = "\n".join(
        f"- {target['display_id']}" for target in targets.values()
    )

    for batch in _batch_records_for_evidence_filter(text_blocks, _format_text_block_for_filter):
        prompt = PROMPT_IUPAC_EVIDENCE_FILTER.format(
            target_ids=target_prompt,
            blocks_context=_format_text_blocks(batch),
        )
        stats["filter_calls"] += 1
        stats["prompt_chars"] += len(prompt)
        response_text = call_iupac_evidence_filter(prompt)
        parsed = _parse_evidence_filter_selection(
            response_text,
            valid_ids,
            target_keys,
            "global_block_ids",
            "target_blocks",
            "block_ids",
        )
        if parsed is None:
            stats["invalid_json"] += 1
            continue
        batch_global, batch_targets = parsed
        global_ids.update(batch_global)
        for key in target_ids:
            target_ids[key].update(batch_targets[key])

    global_ids = _expand_text_selection(global_ids, all_text_blocks)
    selected_by_target = {}
    for key in targets:
        selected_ids = set(mandatory_by_target[key])
        selected_ids.update(_expand_text_selection(target_ids[key], all_text_blocks))
        selected_ids.update(global_ids)
        selected_by_target[key] = [
            block for block in all_text_blocks if block.get("record_id") in selected_ids
        ]
    return selected_by_target, stats


def format_molecule_records(records):
    parts = []
    for record in records:
        ids = " ".join(record.get("ids", []))
        cores = " || ".join(record.get("cores", []))
        sequence = " -> ".join(
            f"{item.get('position')}:{item.get('type')}={item.get('value')}"
            for item in record.get("sequence", [])
        )
        parts.append(
            f"Evidence ID: {record.get('record_id', '未知')} | 页码: {record.get('page')} | "
            f"化合物编号（ID）: {ids} | 结构母核（Core）: {cores} | "
            f"周边文字（Text）: {record.get('text', '')} | 原始组内顺序: {sequence}"
        )
    return "\n".join(parts)


def format_all_tables(table_blocks):
    return "\n\n".join(
        _format_table_block(block, index)
        for index, block in enumerate(table_blocks, start=1)
    )


def format_relevant_figures(figure_records):
    if not figure_records:
        return "未筛选到相关 Figure 图像。"
    return "\n".join(
        f"相关 Figure {index} | 页码: {record.get('page', '未知')} | "
        f"题注: {record.get('caption', '') or '无'}"
        for index, record in enumerate(figure_records, start=1)
    )


@retry_repair_api()
def call_iupac_evidence_filter(prompt):
    repair_rate_limiter.acquire()
    response = _generate_content_with_deadline(
        repair_client,
        model=MODEL_PRO_NAME,
        contents=[prompt],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.0,
            thinking_config=types.ThinkingConfig(thinking_level="low")
        )
    )
    return response.text


@retry_repair_api()
def call_structure_evidence_filter(prompt):
    repair_rate_limiter.acquire()
    response = _generate_content_with_deadline(
        repair_client,
        model=MODEL_PRO_NAME,
        contents=[prompt],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.0,
            thinking_config=types.ThinkingConfig(thinking_level="low")
        )
    )
    return response.text


@retry_repair_api()
def call_iupac_mapping(prompt):
    repair_rate_limiter.acquire()
    response = _generate_content_with_deadline(
        repair_client,
        model=MODEL_PRO_NAME,
        contents=[prompt],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.0,
            thinking_config=types.ThinkingConfig(thinking_level="high")
        )
    )
    return response.text


@retry_repair_api()
def call_figure_filter(prompt, figure_record):
    """使用同一 Pro 模型的低思考模式逐张筛选 Figure，不在此阶段生成结构。"""
    repair_rate_limiter.acquire()
    response = _generate_content_with_deadline(
        repair_client,
        model=MODEL_PRO_NAME,
        contents=[
            types.Part(text=prompt),
            types.Part(inline_data=types.Blob(
                mime_type=figure_record["mime_type"],
                data=figure_record["image_bytes"],
            )),
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.0,
            thinking_config=types.ThinkingConfig(thinking_level="low")
        )
    )
    return response.text


@retry_repair_api()
def call_structure_inference(prompt, relevant_figures=None):
    """在一次高思考调用中综合全部未解决目标、结构证据和已筛选 Figure。"""
    repair_rate_limiter.acquire()
    contents = [types.Part(text=prompt)]
    for figure_index, figure_record in enumerate(relevant_figures or [], start=1):
        contents.extend([
            types.Part(text=(
                f"[相关 Figure {figure_index} | 页码 {figure_record.get('page', '未知')} | "
                f"题注 {figure_record.get('caption', '') or '无'}]"
            )),
            types.Part(inline_data=types.Blob(
                mime_type=figure_record["mime_type"],
                data=figure_record["image_bytes"],
            )),
        ])
    response = _generate_content_with_deadline(
        repair_client,
        model=MODEL_PRO_NAME,
        contents=contents,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.0,
            thinking_config=types.ThinkingConfig(thinking_level="high")
        )
    )
    return response.text


def _parse_repaired_json(response_text):
    """返回结构对象列表；None 表示响应缺失、无法解析或顶层类型错误。"""
    if not response_text:
        return None
    try:
        data = json_repair.loads(clean_json_text(response_text))
    except Exception as exc:
        log.warning(f"JSON repair failed: {exc}")
        return None
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    return None


def _parse_figure_filter_response(response_text):
    """返回严格布尔值；None 表示筛选响应失败，不能静默当作无关 Figure。"""
    if response_text is None or not str(response_text).strip():
        return None
    try:
        data = json_repair.loads(clean_json_text(response_text))
    except Exception as exc:
        log.warning(f"Figure filter JSON repair failed: {exc}")
        return None
    if isinstance(data, list) and len(data) == 1 and isinstance(data[0], dict):
        data = data[0]
    if not isinstance(data, dict):
        return None
    relevant = data.get("is_relevant")
    if relevant is True:
        return True
    if relevant is False:
        return False
    return None


def convert_verified_iupac_name(iupac_name):
    """标准 IUPAC 只走 OPSIN + RDKit，不使用 PubChem 或 AI SMILES 兜底。"""
    raw_name = str(iupac_name or "").strip()
    if not raw_name:
        return None
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw_smiles = py2opsin(raw_name)
        return standardize_smiles(raw_smiles)
    except Exception:
        return None


def _build_iupac_audit_records(text_blocks, molecule_records):
    """建立可核对的原文记录；molecule 只提供其明确 ID/题注文字，不使用 Core 反推名称。"""
    records = list(text_blocks or [])
    for record in molecule_records or []:
        parts = [str(value).strip() for value in record.get("ids", []) if str(value).strip()]
        text_value = str(record.get("text", "") or "").strip()
        if text_value:
            parts.append(text_value)
        if parts:
            records.append({
                "record_id": record.get("record_id"),
                "text": " | ".join(parts),
            })
    return records


def _parse_iupac_mapping_array(response_text):
    if response_text is None or not str(response_text).strip():
        return None
    try:
        data = json_repair.loads(clean_json_text(response_text))
    except Exception:
        return None
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    return None


def _validate_iupac_candidate(item, expected_keys, evidence_records):
    if not isinstance(item, dict):
        return None, "INVALID_ITEM"
    target_key = normalize_compound_identifier(item.get("compound_id", ""))
    if target_key not in expected_keys:
        return None, "UNKNOWN_TARGET"

    raw_name = str(item.get("raw_iupac_name", "") or "").strip()
    normalized_name = _cleanup_iupac_layout_variant(item.get("iupac_name", ""))
    if not raw_name or not normalized_name:
        return None, "EMPTY_NAME"

    records_by_id = {
        str(record.get("record_id", "")).strip(): record
        for record in evidence_records
        if str(record.get("record_id", "")).strip()
    }
    requested_ids = item.get("evidence_block_ids", [])
    if isinstance(requested_ids, str):
        requested_ids = [requested_ids]
    if not isinstance(requested_ids, list):
        return None, "INVALID_EVIDENCE_ID"
    requested_ids = [str(value).strip() for value in requested_ids if str(value).strip()]
    if not requested_ids or any(record_id not in records_by_id for record_id in requested_ids):
        return None, "INVALID_EVIDENCE_ID"

    cited_records = [records_by_id[record_id] for record_id in requested_ids]
    if not _raw_iupac_name_in_blocks(raw_name, cited_records):
        return None, "RAW_NOT_GROUNDED"
    return {
        "target_key": target_key,
        "raw_name": raw_name,
        "normalized_name": normalized_name,
    }, None


def extract_iupac_candidates(document, targets, source_label=""):
    if not targets:
        return {}, {}

    explicit_names = {key: [] for key in targets}
    valid_by_target = {key: {} for key in targets}
    molecule_records = document.get("molecule_records", [])
    stats = {
        "filter_calls": 0,
        "extract_calls": 0,
        "prompt_chars": 0,
        "invalid_json": 0,
        "conflicts": 0,
        "rejections": {},
    }

    def note_rejection(target_key, code):
        stats["rejections"][code] = stats["rejections"].get(code, 0) + 1
        if VERBOSE_PIPELINE_REJECTIONS:
            display_id = targets.get(target_key, {}).get("display_id", target_key or "UNKNOWN")
            log.info(f"   -> [IUPAC {code}] {source_label or '未知文献'} | {display_id}")

    selected_by_target, filter_stats = select_iupac_evidence_blocks(document, targets)
    stats["filter_calls"] = filter_stats["filter_calls"]
    stats["prompt_chars"] = filter_stats["prompt_chars"]
    stats["invalid_json"] = filter_stats["invalid_json"]

    selected_ids_by_target = {
        key: {
            block.get("record_id") for block in selected_by_target.get(key, [])
            if block.get("record_id")
        }
        for key in targets
    }
    union_ids = set().union(*selected_ids_by_target.values()) if selected_ids_by_target else set()
    union_blocks = [
        block for block in document.get("text_blocks", [])
        if block.get("record_id") in union_ids
    ]
    direct_molecules = {
        key: [
            record for record in molecule_records
            if key in _record_target_keys(record, {key})
        ]
        for key in targets
    }

    def process_extraction_batch(batch, batch_keys):
        if not batch_keys:
            return
        batch_key_set = set(batch_keys)
        selected_pages = {block.get("page") for block in batch}
        matching_molecules = [
            record for record in molecule_records
            if _record_target_keys(record, batch_key_set)
            or (record.get("page") in selected_pages and record.get("text"))
        ]
        target_prompt = "\n".join(
            f"- {targets[key]['display_id']}" for key in batch_keys
        )
        prompt = PROMPT_IUPAC_MAPPING.format(
            target_ids=target_prompt,
            text_context=_format_text_blocks(batch) or "无正文文本块",
            molecules_context=format_molecule_records(matching_molecules) or "无分子题注记录",
        )
        stats["extract_calls"] += 1
        stats["prompt_chars"] += len(prompt)
        parsed = _parse_iupac_mapping_array(call_iupac_mapping(prompt))
        if parsed is None:
            stats["invalid_json"] += 1
            return

        evidence_records = _build_iupac_audit_records(batch, matching_molecules)
        for item in parsed:
            validated, reject_code = _validate_iupac_candidate(
                item, batch_key_set, evidence_records
            )
            if validated is None:
                returned_key = (
                    normalize_compound_identifier(item.get("compound_id", ""))
                    if isinstance(item, dict) else None
                )
                note_rejection(returned_key, reject_code)
                continue

            target_key = validated["target_key"]
            normalized_name = validated["normalized_name"]
            explicit_names[target_key].append(normalized_name)
            canonical = convert_verified_iupac_name(normalized_name)
            if not canonical:
                note_rejection(target_key, "OPSIN_FAIL")
                continue
            valid_by_target[target_key].setdefault(canonical, []).append(normalized_name)

    called_keys = set()
    for batch in _batch_records_for_evidence_filter(
        union_blocks, _format_text_block_for_filter
    ):
        batch_ids = {block.get("record_id") for block in batch if block.get("record_id")}
        batch_keys = [
            key for key in targets
            if selected_ids_by_target[key].intersection(batch_ids)
        ]
        if batch_keys:
            process_extraction_batch(batch, batch_keys)
            called_keys.update(batch_keys)

    molecule_only_keys = [
        key for key in targets
        if key not in called_keys and direct_molecules[key]
    ]
    if molecule_only_keys:
        process_extraction_batch([], molecule_only_keys)

    accepted = {}
    for target_key, canonical_map in valid_by_target.items():
        if len(canonical_map) == 1:
            canonical, names = next(iter(canonical_map.items()))
            accepted[target_key] = {
                "smiles": canonical,
                "iupac_name": stable_unique(names)[0],
            }
        elif len(canonical_map) > 1:
            stats["conflicts"] += 1
            note_rejection(target_key, "CANONICAL_CONFLICT")

    for target_key in explicit_names:
        explicit_names[target_key] = stable_unique(explicit_names[target_key])

    label = source_label or "未知文献"
    log.info(
        f"   -> [{label}] IUPAC 摘要：目标={len(targets)}，"
        f"低筛调用={stats['filter_calls']}，高思考调用={stats['extract_calls']}，"
        f"输入字符={stats['prompt_chars']}，接受={len(accepted)}，"
        f"冲突={stats['conflicts']}，未解决={len(targets) - len(accepted)}，"
        f"无效JSON={stats['invalid_json']}。"
    )
    return accepted, explicit_names


UNIPARSER_ANCHOR_PATTERN = re.compile(
    r"<a(?:\s[^>]*)?>\s*[^:<]+?\s*:\s*([^<]+?)\s*</a>",
    re.IGNORECASE,
)
UNIPARSER_VARIABLE_LABEL_PATTERN = re.compile(
    r"^(?:R(?:[_^]?\d+|['′″]+)?|X\d*|Y\d*|Z\d*|Ar\d*|Het(?:Ar)?\d*)(?:=.*)?$",
    re.IGNORECASE,
)
UNIPARSER_WILDCARD_ATOMS = ("*", "∗", "﹡")


def _annotation_contains_variable_label(annotation):
    for label in UNIPARSER_ANCHOR_PATTERN.findall(str(annotation or "")):
        for token in re.split(r"[,;/|]+", label):
            normalized = re.sub(r"\s+", "", token).strip("()[]{}")
            if UNIPARSER_VARIABLE_LABEL_PATTERN.fullmatch(normalized):
                return True
    return False


def _classify_uniparser_core(raw_core):
    """区分可直接采用的完整分子、马库什母核、路线和不安全注释。

    `<sep>` 是 UniParser 结构主体与锚点注释之间的分隔符，本身不证明结构
    含有未决取代基。只有结构主体含虚原子，或注释明确声明 R/X/Ar 等变量
    时，才归入 Markush。返回的 complete SMILES 已去除 UniParser 注释。
    """
    raw_text = str(raw_core or "").strip()
    if not raw_text:
        return "invalid", None
    if ">>" in raw_text:
        return "route", None

    core_parts = re.split(r"<sep>", raw_text, maxsplit=1, flags=re.IGNORECASE)
    chemical_smiles = core_parts[0].strip()
    annotation = core_parts[1].strip() if len(core_parts) == 2 else ""
    if not chemical_smiles:
        return "invalid", None
    if any(marker in chemical_smiles for marker in UNIPARSER_WILDCARD_ATOMS):
        return "markush", None
    if annotation and _annotation_contains_variable_label(annotation):
        return "markush", None

    if annotation:
        # 只允许由 `<a>原子序号:标签</a>` 组成的注释；未知尾部不能被静默截断。
        annotation_residue = UNIPARSER_ANCHOR_PATTERN.sub("", annotation)
        annotation_residue = re.sub(
            r"<sep>", "", annotation_residue, flags=re.IGNORECASE
        )
        if annotation_residue.strip():
            return "annotated_unknown", None

    canonical = standardize_smiles(chemical_smiles)
    if canonical:
        return "complete", canonical
    return "invalid", None


def collect_image_candidates(document, targets):
    """收集 UniParser molecule OCR 的完整、明确映射结构；不依赖版面顺序。"""
    candidates = {key: set() for key in targets}
    target_keys = set(targets)
    classification_counts = {
        "complete": 0,
        "markush": 0,
        "route": 0,
        "annotated_unknown": 0,
        "invalid": 0,
    }

    for record in document.get("molecule_records", []):
        matched_keys = _record_target_keys(record, target_keys)
        if len(matched_keys) != 1:
            continue
        target_key = next(iter(matched_keys))

        for raw_core in record.get("cores", []):
            core_kind, canonical = _classify_uniparser_core(raw_core)
            classification_counts[core_kind] += 1
            if core_kind == "complete" and canonical:
                candidates[target_key].add(canonical)

    classified_total = sum(classification_counts.values())
    if classified_total:
        log.info(
            "   -> [UniParser molecule classification] "
            f"IMAGE complete={classification_counts['complete']}, "
            f"Markush={classification_counts['markush']}, "
            f"Route={classification_counts['route']}, "
            f"Unknown/invalid="
            f"{classification_counts['annotated_unknown'] + classification_counts['invalid']}"
        )

    accepted = {}
    for target_key, structures in candidates.items():
        if len(structures) == 1:
            accepted[target_key] = next(iter(structures))
        elif len(structures) > 1:
            log.warning(
                f"   -> [IMAGE Ambiguous] {targets[target_key]['display_id']} "
                f"has {len(structures)} distinct complete molecule structures."
            )
    return accepted


def collect_dictionary_candidates(targets, explicit_names):
    accepted = {}
    for target_key, target in targets.items():
        lookup_names = [target["display_id"]] + target["names"] + explicit_names.get(target_key, [])
        structures = set()
        for name in stable_unique(lookup_names):
            dictionary_smiles = COMMON_TB_DRUGS.get(str(name).strip().casefold())
            canonical = standardize_smiles(dictionary_smiles)
            if canonical:
                structures.add(canonical)
        if len(structures) == 1:
            accepted[target_key] = next(iter(structures))
        elif len(structures) > 1:
            log.warning(
                f"   -> [Dictionary Ambiguous] {target['display_id']} "
                f"matched {len(structures)} distinct dictionary structures."
            )
    return accepted


def _format_structure_evidence_for_filter(record):
    if str(record.get("record_id", "")).startswith("TAB-"):
        return _format_table_block(record, 1)
    return format_molecule_records([record])


def _evidence_page_sort_key(record):
    page = record.get("page")
    try:
        return 0, int(page), str(record.get("record_id", ""))
    except (TypeError, ValueError):
        return 1, str(page), str(record.get("record_id", ""))


def select_structure_evidence_records(document, targets, unresolved_keys):
    """低思考只选择原始 Evidence ID；目标无选择结果时回退到全部结构证据。"""
    molecule_records = document.get("molecule_records", [])
    table_blocks = document.get("table_blocks", [])
    evidence_records = sorted(
        molecule_records + table_blocks,
        key=_evidence_page_sort_key,
    )
    if not evidence_records:
        return {
            key: {"molecule_records": [], "table_blocks": []}
            for key in unresolved_keys
        }

    valid_ids = {
        record.get("record_id") for record in evidence_records if record.get("record_id")
    }
    target_keys = set(unresolved_keys)
    global_ids = set()
    target_ids = {key: set() for key in unresolved_keys}
    target_prompt = "\n".join(
        f"- {targets[key]['display_id']}" for key in unresolved_keys
    )

    for batch in _batch_records_for_evidence_filter(
        evidence_records, _format_structure_evidence_for_filter
    ):
        batch_context = "\n\n".join(
            _format_structure_evidence_for_filter(record) for record in batch
        )
        response_text = call_structure_evidence_filter(
            PROMPT_STRUCTURE_EVIDENCE_FILTER.format(
                target_ids=target_prompt,
                evidence_context=batch_context,
            )
        )
        parsed = _parse_evidence_filter_selection(
            response_text,
            valid_ids,
            target_keys,
            "global_evidence_ids",
            "target_evidence",
            "evidence_ids",
        )
        if parsed is None:
            return None
        batch_global, batch_targets = parsed
        global_ids.update(batch_global)
        for key in target_ids:
            target_ids[key].update(batch_targets[key])

    for record in molecule_records:
        matched_keys = _record_target_keys(record, target_keys)
        for key in matched_keys:
            target_ids[key].add(record.get("record_id"))
    for block in table_blocks:
        searchable_text = f"{block.get('caption', '')}\n{block.get('text', '')}"
        for key in unresolved_keys:
            if any(
                _text_mentions_target(searchable_text, label)
                for label in _target_search_labels(targets[key])
            ):
                target_ids[key].add(block.get("record_id"))

    records_by_id = {
        record.get("record_id"): record
        for record in evidence_records
        if record.get("record_id")
    }
    records_by_page = {}
    for record in evidence_records:
        records_by_page.setdefault(record.get("page"), set()).add(record.get("record_id"))

    selected_by_target = {}
    for key in unresolved_keys:
        selected_ids = set(target_ids[key]) | set(global_ids)
        for record_id in list(selected_ids):
            record = records_by_id.get(record_id)
            if record is not None:
                selected_ids.update(records_by_page.get(record.get("page"), set()))
        selected_ids.discard(None)
        if not selected_ids:
            selected_ids = set(valid_ids)
        selected_by_target[key] = {
            "molecule_records": [
                record for record in molecule_records
                if record.get("record_id") in selected_ids
            ],
            "table_blocks": [
                block for block in table_blocks
                if block.get("record_id") in selected_ids
            ],
        }
    return selected_by_target


def screen_relevant_figures(document, targets, unresolved_keys):
    """逐张低思考筛选；单张失败时跳过，避免整篇文献重新执行。"""
    figure_records = document.get("figure_records", [])
    if not unresolved_keys or not figure_records:
        return []

    target_ids = ", ".join(
        targets[target_key]["display_id"] for target_key in unresolved_keys
    )
    log.info(
        f"   -> [Figure Filter] Screening {len(figure_records)} deduplicated figures "
        f"for {len(unresolved_keys)} unresolved targets."
    )
    relevant_figures = []
    for figure_index, figure_record in enumerate(figure_records, start=1):
        if GLOBAL_STOP_EVENT.is_set():
            log.warning(
                "   -> [Figure Filter] Global stop active; keeping previously selected figures."
            )
            break
        prompt = PROMPT_FIGURE_FILTER.format(
            target_ids=target_ids,
            page=figure_record.get("page", "未知"),
            caption=figure_record.get("caption", "") or "无",
        )
        response_text = call_figure_filter(prompt, figure_record)
        is_relevant = _parse_figure_filter_response(response_text)
        if is_relevant is None:
            log.warning(
                f"   -> [Figure Filter Failed] Figure {figure_index} "
                f"(page {figure_record.get('page', 'unknown')}) returned no strict boolean; "
                "skipped without aborting the PDF."
            )
            continue
        if is_relevant:
            relevant_figures.append(figure_record)
            log.info(
                f"   -> [Figure Kept] {figure_index}/{len(figure_records)} | "
                f"page {figure_record.get('page', 'unknown')}"
            )

    log.info(
        f"   -> [Figure Filter] Kept {len(relevant_figures)}/{len(figure_records)} figures "
        "as auxiliary evidence for INFERENCE."
    )
    return relevant_figures


def infer_unresolved_structures(document, targets, unresolved_keys, source_label=""):
    if not unresolved_keys:
        return {}

    selected_evidence = select_structure_evidence_records(
        document, targets, unresolved_keys
    )
    if selected_evidence is None:
        log.error(
            "   -> [Structure Evidence Filter Failed] Invalid JSON; preserving earlier "
            "results and persisting remaining targets as unresolved."
        )
        return {}
    relevant_figures = screen_relevant_figures(document, targets, unresolved_keys)
    if relevant_figures is None:
        relevant_figures = []
    candidate_sets = {target_key: set() for target_key in unresolved_keys}
    stats = {
        "calls": 0,
        "no_evidence": 0,
        "not_confident": 0,
        "omitted": 0,
        "conflicts": 0,
    }

    def note_rejection(target_key, code):
        if code == "MODEL_NOT_CONFIDENT":
            stats["not_confident"] += 1
        elif code == "MODEL_OMITTED_TARGET":
            stats["omitted"] += 1
        if VERBOSE_PIPELINE_REJECTIONS:
            log.info(
                f"   -> [INFERENCE {code}] {source_label or '未知文献'} | "
                f"{targets[target_key]['display_id']}"
            )

    def consume_inference_items(items, expected_keys):
        returned_keys = set()
        for item in items:
            if not isinstance(item, dict):
                continue
            target_key = normalize_compound_identifier(item.get("compound_id", ""))
            if target_key not in expected_keys:
                continue
            returned_keys.add(target_key)
            if item.get("is_fully_confident") is not True:
                note_rejection(target_key, "MODEL_NOT_CONFIDENT")
                continue
            raw_smiles = str(item.get("repaired_smiles", "") or "").strip()
            if (
                not raw_smiles
                or "<sep>" in raw_smiles.casefold()
                or "*" in raw_smiles
                or ">>" in raw_smiles
            ):
                note_rejection(target_key, "EMPTY_OR_PLACEHOLDER")
                continue
            canonical = standardize_smiles(raw_smiles)
            if canonical:
                candidate_sets[target_key].add(canonical)
            else:
                note_rejection(target_key, "RDKIT_FAIL")
        return returned_keys

    eligible_keys = []
    for target_key in unresolved_keys:
        evidence = selected_evidence.get(target_key, {})
        if (
            evidence.get("molecule_records")
            or evidence.get("table_blocks")
            or relevant_figures
        ):
            eligible_keys.append(target_key)
        else:
            stats["no_evidence"] += 1
            note_rejection(target_key, "NO_EVIDENCE")

    def merge_selected_records(field_name):
        merged_records = []
        seen_records = set()
        for target_key in eligible_keys:
            evidence = selected_evidence.get(target_key, {})
            for record in evidence.get(field_name, []):
                record_id = str(record.get("record_id", "") or "").strip()
                dedup_key = record_id or json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                    default=str,
                )
                if dedup_key in seen_records:
                    continue
                seen_records.add(dedup_key)
                merged_records.append(record)
        return merged_records

    if eligible_keys and not GLOBAL_STOP_EVENT.is_set():
        molecule_records = merge_selected_records("molecule_records")
        table_blocks = merge_selected_records("table_blocks")
        target_prompt = "\n".join(
            f"- {targets[target_key]['display_id']}" for target_key in eligible_keys
        )
        prompt = PROMPT_STRUCTURE_INFERENCE.format(
            target_ids=target_prompt,
            molecules_context=format_molecule_records(molecule_records),
            tables_context=format_all_tables(table_blocks),
            figures_context=format_relevant_figures(relevant_figures),
        )
        stats["calls"] += 1
        response_text = call_structure_inference(prompt, relevant_figures)
        parsed = _parse_repaired_json(response_text)
        if parsed is None:
            log.error(
                "   -> [INFERENCE Failed] Aggregated response could not be parsed; "
                "no duplicate request will be sent, and unresolved targets will be persisted."
            )
        else:
            expected_keys = set(eligible_keys)
            returned_keys = consume_inference_items(parsed, expected_keys)
            for missing_key in expected_keys - returned_keys:
                note_rejection(missing_key, "MODEL_OMITTED_TARGET")

    accepted = {}
    for target_key, structures in candidate_sets.items():
        if len(structures) == 1:
            accepted[target_key] = next(iter(structures))
        elif len(structures) > 1:
            stats["conflicts"] += 1
            note_rejection(target_key, "CANONICAL_CONFLICT")
    log.info(
        f"   -> [{source_label or '未知文献'}] INFERENCE 摘要："
        f"目标={len(unresolved_keys)}，调用={stats['calls']}，接受={len(accepted)}，"
        f"模型不确定={stats['not_confident']}，无证据={stats['no_evidence']}，"
        f"模型遗漏={stats['omitted']}，冲突={stats['conflicts']}。"
    )
    return accepted


def resolve_structures(activity_items, document, source_label=""):
    targets = build_target_index(activity_items)
    if not targets:
        return activity_items

    empty_explicit_names = {target_key: [] for target_key in targets}
    dictionary_candidates = collect_dictionary_candidates(targets, empty_explicit_names)
    image_candidates = collect_image_candidates(document, targets)
    iupac_targets = {
        key: targets[key]
        for key in targets
        if key not in dictionary_candidates and key not in image_candidates
    }
    if iupac_targets:
        iupac_candidates, explicit_names = extract_iupac_candidates(
            document, iupac_targets, source_label=source_label
        )
        dictionary_candidates.update(
            collect_dictionary_candidates(iupac_targets, explicit_names)
        )
    else:
        iupac_candidates = {}
        log.info(
            f"   -> [{source_label or '未知文献'}] IUPAC 摘要：目标=0，"
            "低筛调用=0，高思考调用=0，输入字符=0，接受=0，"
            "冲突=0，未解决=0，无效JSON=0。"
        )

    unresolved_keys = [
        target_key
        for target_key in targets
        if target_key not in dictionary_candidates
        and target_key not in iupac_candidates
        and target_key not in image_candidates
    ]
    inference_candidates = infer_unresolved_structures(
        document, targets, unresolved_keys, source_label=source_label
    )
    if inference_candidates is None:
        log.error(
            f"   -> [{source_label or '未知文献'}] INFERENCE returned no result; "
            "preserving earlier sources and leaving remaining targets unresolved."
        )
        inference_candidates = {}

    resolved = {}
    for target_key, target in targets.items():
        smiles = ""
        source = ""
        iupac_name = ""
        style_flag = ""

        if target_key in iupac_candidates:
            iupac_name = iupac_candidates[target_key]["iupac_name"]

        if target_key in dictionary_candidates:
            smiles = dictionary_candidates[target_key]
            source = "DICTIONARY"
        elif target_key in image_candidates:
            smiles = image_candidates[target_key]
            source = "IMAGE"
        elif target_key in iupac_candidates:
            smiles = iupac_candidates[target_key]["smiles"]
            source = "IUPAC"
        elif target_key in inference_candidates:
            smiles = inference_candidates[target_key]
            source = "INFERENCE"

        if smiles and contains_metal(smiles):
            style_flag = "METAL"
            log.warning(
                f"   -> [Metal Detected] {target['display_id']} retained and marked red."
            )

        resolved[target_key] = {
            "smiles": smiles,
            "source": source,
            "iupac_name": iupac_name,
            "style_flag": style_flag,
        }

    for item in activity_items:
        target_key = normalize_compound_identifier(item.get("compound_id", ""))
        structure = resolved.get(target_key, {})
        item["iupac_name"] = structure.get("iupac_name", "")
        item["is_standard_iupac"] = bool(item["iupac_name"])
        item["_resolved_smiles"] = structure.get("smiles", "")
        item["_smiles_source"] = structure.get("source", "")
        item["_smiles_style"] = structure.get("style_flag", "")
    return activity_items


# ================= 9. 纯追加写入 Excel 模块 (带防碰撞重试) =================

def append_to_excel(excel_path, doi, filename, if_val, journal_name, pub_date, extracted_data):
    if not extracted_data:
        return False
    
    valid_items = []
    for item in extracted_data:
        numerical_fields = [
            item.get("mic_value"), 
            item.get("ic50_value"), 
            item.get("inhibition_value"), 
            item.get("zoi_value"),
            item.get("pic50_value"),
            item.get("pmic_value"),
        ]
        raw_inactive_flag = item.get("is_qualitative_inactive", False)
        is_inactive_flag = (
            raw_inactive_flag is True
            or (
                isinstance(raw_inactive_flag, str)
                and raw_inactive_flag.strip().casefold() == "true"
            )
        )
        has_any_data = any(v is not None and str(v).strip() != "" and str(v).lower() != "null" for v in numerical_fields)
        
        if is_inactive_flag or has_any_data:
            valid_items.append(item)
    
    if not valid_items:
        log.info(f"   -> [No Data] {doi} contains no activity data, skipping records.")
        return False

    final_rows = []
    row_style_flags = []
    
    for item in valid_items: 
        c_name = str(item.get("iupac_name", "") or "").strip()
        smi = str(item.get("_resolved_smiles", "") or "").strip()
        source = str(item.get("_smiles_source", "") or "").strip()
        if not smi:
            source = ""
        if source not in {"DICTIONARY", "IUPAC", "IMAGE", "INFERENCE", ""}:
            log.warning(f"Unexpected SMILES source '{source}' for {item.get('compound_id')}; cleared.")
            source = ""

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
        row_style_flags.append(str(item.get("_smiles_style", "") or "").strip().upper())

    df = pd.DataFrame(final_rows)
    
    red_fill = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")
    
    for attempt in range(MAX_EXCEL_RETRIES):
        temp_excel_path = None
        try:
            target_path = os.path.abspath(excel_path)
            target_dir = os.path.dirname(target_path)
            os.makedirs(target_dir, exist_ok=True)
            file_descriptor, temp_excel_path = tempfile.mkstemp(
                prefix=".extractor_commit_",
                suffix=".xlsx",
                dir=target_dir,
            )
            os.close(file_descriptor)
            if os.path.exists(target_path):
                shutil.copy2(target_path, temp_excel_path)

            mode = 'a' if os.path.getsize(temp_excel_path) > 0 else 'w'
            with pd.ExcelWriter(temp_excel_path, engine='openpyxl', mode=mode,
                                if_sheet_exists='overlay' if mode == 'a' else None) as writer:
                
                # 计算新数据应该从哪一行开始追加
                start_row = writer.book[SHEET_NAME].max_row if mode == 'a' else 0
                if mode == 'a':
                    start_row += 1
                df.to_excel(writer, sheet_name=SHEET_NAME, index=False, header=(mode == 'w'), startrow=start_row)
                
                ws = writer.book[SHEET_NAME]
                
                if mode == 'w':
                    headers = list(df.columns)
                    data_start_row = 2 
                else:
                    headers = [cell.value for cell in ws[1]]
                    data_start_row = start_row + 1

                try:
                    if_col_idx = headers.index("Impact Factor") + 1
                    data_end_row = data_start_row + len(df)
                    
                    for r in range(data_start_row, data_end_row):
                        cell = ws.cell(row=r, column=if_col_idx)
                        if str(cell.value) == "NA":
                            cell.fill = red_fill
                except ValueError:
                    pass

                try:
                    smiles_col_idx = headers.index("SMILES") + 1
                    for offset, style_flag in enumerate(row_style_flags):
                        smiles_cell = ws.cell(
                            row=data_start_row + offset,
                            column=smiles_col_idx,
                        )
                        if style_flag == "METAL":
                            smiles_cell.fill = METAL_FILL
                        elif style_flag == "CONFLICT":
                            smiles_cell.fill = CONFLICT_FILL
                except ValueError:
                    pass

            os.replace(temp_excel_path, target_path)
            temp_excel_path = None
             
            unique_compounds = set(
                str(row.get("Compound ID", "")).strip().lower() 
                for row in final_rows 
                if row.get("Compound ID")
            )
            log.info(f"      [{os.path.basename(excel_path)}] Appended {len(final_rows)} records (Unique compounds: {len(unique_compounds)}).")
            return True
        except PermissionError:
            log.warning(f"⚠️ [File Locked] Please close {os.path.basename(excel_path)}. Retrying in 3s ({attempt+1}/{MAX_EXCEL_RETRIES})...")
            if GLOBAL_STOP_EVENT.wait(3):
                log.warning(
                    f"[GLOBAL STOP] Excel commit for {os.path.basename(excel_path)} "
                    "will not be retried."
                )
                return False
        finally:
            if temp_excel_path and os.path.exists(temp_excel_path):
                try:
                    os.remove(temp_excel_path)
                except OSError:
                    pass

    log.error(f"❌ [Fatal] Could not write to {excel_path} after {MAX_EXCEL_RETRIES} retries. Data for {doi} was not saved.")
    return False


# ================= 10. 流程控制 =================
def process_single_pdf(pdf_path, target_excel):
    filename = os.path.basename(pdf_path)
    log.info(f"Analyzing: {filename}")

    try:
        # 1. 提取元数据
        doi, if_val, journal_name, pub_date, article_type = extract_metadata(pdf_path)
        log.info(f"   -> Meta: DOI={doi}, Date={pub_date}, IF={if_val}")
        if GLOBAL_STOP_EVENT.is_set():
            log.warning(f"[GLOBAL STOP] Leaving {filename} for the next run.")
            return False

        if str(article_type).strip().upper() == "REVIEW":
            log.info(f"📚 [Review Detected] {filename} is a Review article. Moving to '{REVIEW_FOLDER}' and skipping extraction.")
            _move_pdf_as_final_commit(pdf_path, REVIEW_FOLDER, "review")
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
            _move_pdf_as_final_commit(pdf_path, LOW_IF_FOLDER, "threshold-skip")
            return False 
        # 2. 整篇文献交给 UniParser；正文活性只允许来自返回的结构化表格。
        document = call_uniparser_extract(pdf_path)
        if GLOBAL_STOP_EVENT.is_set():
            log.error(f"[GLOBAL STOP] Leaving {filename} in place for a later restart.")
            return False

        if not document:
            log.info(f"   UniParser returned no usable document for {filename}")
            _move_pdf_as_final_commit(pdf_path, FAIL_FOLDER, "uniparser-failed")
            return False

        if not document.get("table_blocks"):
            log.info(f"   UniParser found no tables in {filename}")
            _move_pdf_as_final_commit(pdf_path, FAIL_FOLDER, "no-tables")
            return False

        # 3. 表格活性抽取；不会把正文、分子节点或反应路线发送给此阶段。
        extracted_data = extract_table_activity(document["table_blocks"])
        if GLOBAL_STOP_EVENT.is_set():
            log.error(f"[GLOBAL STOP] Leaving {filename} in place for a later restart.")
            return False

        if not extracted_data:
            log.info(f"   No qualifying antimycobacterial table activity found in {filename}")
            _move_pdf_as_final_commit(pdf_path, FAIL_FOLDER, "no-activity")
            return False

        # 4. 只对表格中出现的目标 ID 执行严格结构来源解析。
        resolved_data = resolve_structures(
            extracted_data, document, source_label=filename
        )
        if resolved_data is None:
            log.error(
                f"   Structure resolution returned no result for {filename}; "
                "activity rows will be persisted with unresolved structures."
            )
        else:
            extracted_data = resolved_data
        if GLOBAL_STOP_EVENT.is_set():
            log.warning(
                f"[GLOBAL STOP] Discarding partial results for {filename}; "
                "the PDF will remain for the next run."
            )
            return False

        # 5. 篇内分组与 Activity Level 计算（保留原逻辑）
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

        # 6. 仅在最终 Excel 成功落盘后移动 PDF
        if extracted_data:
            commit_key = f"extractor|{os.path.abspath(pdf_path)}"
            if not _begin_final_commit(commit_key):
                log.warning(
                    f"[GLOBAL STOP] {filename} had not entered final commit; "
                    "discarding this run and leaving the PDF for next time."
                )
                return False
            try:
                write_ok = append_to_excel(
                    target_excel,
                    doi,
                    filename,
                    if_val,
                    journal_name,
                    pub_date,
                    extracted_data,
                )
                if write_ok:
                    shutil.move(pdf_path, os.path.join(SUCCESS_FOLDER, filename))
                    return True
                log.error(f"   Final Excel write failed; leaving {filename} in input folder.")
                return False
            finally:
                _end_final_commit(commit_key)
        else:
            if GLOBAL_STOP_EVENT.is_set():
                log.warning(
                    f"[GLOBAL STOP] Leaving incomplete {filename} for the next run."
                )
                return False
            _move_pdf_as_final_commit(pdf_path, FAIL_FOLDER, "empty-final-data")
            return False

    except APIConnectionError as e:
        log.error(f"[NETWORK ERROR] Skipping {filename}: {e}")
        return False
    except Exception as e:
        log.error(f"[CODE ERROR] {filename}: {e}")
        return False


def load_processed_filenames(excel_path):
    """从单一最终 Excel 读取断点；读取失败时抛错并停止对应文件夹。"""
    if not os.path.exists(excel_path):
        return set()
    checkpoint_df = pd.read_excel(
        excel_path,
        sheet_name=SHEET_NAME,
        usecols=["Source Filename"],
        keep_default_na=False,
    )
    return {
        str(value).strip()
        for value in checkpoint_df["Source Filename"].tolist()
        if str(value).strip()
    }


def select_pending_subfolders(subfolders):
    """运行 API 健康检查前先完成本地断点过滤，确保无任务时不联网。"""
    pending_subfolders = []
    for folder_path in subfolders:
        folder_name = os.path.basename(folder_path)
        pdfs = [
            filename
            for filename in os.listdir(folder_path)
            if filename.lower().endswith(".pdf")
        ]
        if not pdfs:
            continue

        target_excel = os.path.join(OUTPUT_EXCEL_FOLDER, f"{folder_name}.xlsx")
        try:
            processed_filenames = load_processed_filenames(target_excel)
        except Exception as exc:
            log.error(
                f"[{folder_name}] Cannot read checkpoint Excel '{target_excel}'. "
                f"Folder excluded to prevent duplicate appends: {type(exc).__name__}: {exc}"
            )
            continue

        if any(filename not in processed_filenames for filename in pdfs):
            pending_subfolders.append(folder_path)
        else:
            log.info(f"[{folder_name}] All PDFs are already present in the final Excel.")
    return pending_subfolders


def process_folder_task(folder_path):
    folder_name = os.path.basename(folder_path)
    target_excel = os.path.join(OUTPUT_EXCEL_FOLDER, f"{folder_name}.xlsx")
    
    pdfs = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
    pdfs.sort()
    
    if not pdfs:
        log.info(f"[{folder_name}] No PDFs found. Skipping.")
        return folder_name, 0, 0

    try:
        processed_filenames = load_processed_filenames(target_excel)
    except Exception as exc:
        log.error(
            f"[{folder_name}] Cannot read checkpoint Excel '{target_excel}'. "
            f"Folder stopped to prevent duplicate appends: {type(exc).__name__}: {exc}"
        )
        return folder_name, 0, len(pdfs)

    if processed_filenames:
        original_count = len(pdfs)
        pdfs = [filename for filename in pdfs if filename not in processed_filenames]
        skipped_count = original_count - len(pdfs)
        if skipped_count:
            log.info(
                f"[{folder_name}] Excel checkpoint skipped {skipped_count} "
                "already persisted PDF(s)."
            )

    if not pdfs:
        log.info(f"[{folder_name}] All PDFs are already present in the final Excel.")
        return folder_name, 0, 0

    log.info(f"--- Starting Folder: {folder_name} ({len(pdfs)} PDFs) -> {folder_name}.xlsx ---")
    
    success_count = 0
    fail_count = 0
    
    for pdf_filename in pdfs:
        if GLOBAL_STOP_EVENT.is_set():
            log.warning(f"[{folder_name}] Global stop active; leaving remaining PDFs untouched.")
            break
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
            log.error(
                f"⏰ [Timeout] {pdf_filename} exceeded {PDF_TIMEOUT_SECONDS}s. "
                "The running thread cannot be cancelled safely; stopping the process "
                "without moving or persisting this PDF so it can be retried next run."
            )
            fail_count += 1
            _request_global_stop()
            future.cancel()
            pdf_executor.shutdown(wait=False, cancel_futures=True)
            _force_exit_after_commit_grace("PDF timeout", 124)
            
        except Exception as e:
            log.error(f"❌ [Unhandled Error] {pdf_filename}: {e}")
            fail_count += 1
            
        finally:
            pdf_executor.shutdown(wait=False, cancel_futures=True)
            
    return folder_name, success_count, fail_count


def main():
    GLOBAL_STOP_EVENT.clear()
    with FINAL_COMMIT_CONDITION:
        ACTIVE_FINAL_COMMITS.clear()
    log.info(f"Gemini Drug Discovery Agent | Folder Concurrent Mode: {MAX_CONCURRENT_FOLDERS} Threads")

    subfolders = [os.path.join(INPUT_BASE_FOLDER, d) for d in os.listdir(INPUT_BASE_FOLDER) 
                  if os.path.isdir(os.path.join(INPUT_BASE_FOLDER, d))]

    if not subfolders:
        log.warning(f"No subfolders found in {INPUT_BASE_FOLDER}. Please organize your PDFs into subfolders.")
        return

    log.info(f"Found {len(subfolders)} subfolders.")

    subfolders = select_pending_subfolders(subfolders)
    if not subfolders:
        log.info("No pending PDF tasks after Excel checkpoint filtering; no API was called.")
        return
    log.info(f"Found {len(subfolders)} subfolders with pending PDF tasks.")

    if not test_uniparser_connection():
        log.critical("UniParser 前置检查失败，未提交任何 PDF。")
        return
    if not test_repair_gemini_connection():
        log.critical("结构解析 Gemini 前置检查失败，未提交任何 PDF。")
        return
    log.info("UniParser 与结构解析 Gemini 前置检查均已通过。")

    executor = ThreadPoolExecutor(
        max_workers=MAX_CONCURRENT_FOLDERS,
        thread_name_prefix="Dir",
    )
    future_to_folder = {
        executor.submit(process_folder_task, path): path for path in subfolders
    }
    active_futures = set(future_to_folder.keys())
    try:
        with tqdm(total=len(subfolders), desc="Processing Folders") as pbar:
            while active_futures:
                done, active_futures = concurrent.futures.wait(
                    active_futures,
                    timeout=0.5,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                for future in done:
                    folder_path = future_to_folder[future]
                    try:
                        folder_name, succ, fail = future.result()
                        log.info(
                            f"✅ Folder Completed: {folder_name} | "
                            f"Success: {succ} | Fail: {fail}"
                        )
                    except Exception as exc:
                        log.error(
                            f"[Unhandled Folder Error] {os.path.basename(folder_path)}: {exc}"
                        )
                    pbar.update(1)

                if GLOBAL_STOP_EVENT.is_set():
                    log.warning(
                        "Global stop is active: cancelling queued folders and "
                        "waiting at most 10 seconds for final commits."
                    )
                    for future in active_futures:
                        future.cancel()
                    executor.shutdown(wait=False, cancel_futures=True)
                    _force_exit_after_commit_grace("Global stop", 1)

        log.info("🎉 All folders processed completely.")
    except KeyboardInterrupt:
        print("\n", flush=True)
        log.warning(
            "⚠️ 收到 Ctrl+C：停止新提交，放弃处理一半的 PDF；"
            "仅等待已开始的最终 Excel 提交，最多 10 秒。"
        )
        _request_global_stop()
        for future in active_futures:
            future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        _force_exit_after_commit_grace("Ctrl+C", 130)
    else:
        executor.shutdown(wait=False, cancel_futures=True)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n", flush=True)
        log.warning(
            "⚠️ 收到 Ctrl+C：放弃处理一半的 PDF；"
            "仅等待已开始的最终提交，最多 10 秒。"
        )
        _request_global_stop()
        _force_exit_after_commit_grace("Ctrl+C", 130)
