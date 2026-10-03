import os
import glob
import json
import time
import functools
import re
import unicodedata
import base64
import hashlib
import io
import queue
import tempfile
import threading
import pandas as pd
import concurrent.futures
from openpyxl import Workbook, load_workbook
from openpyxl.styles import PatternFill
from concurrent.futures import ThreadPoolExecutor
from rdkit import Chem
from py2opsin import py2opsin
from rdkit import RDLogger  # 新增导入
RDLogger.DisableLog('rdApp.*')
from google import genai
import json_repair
from google.genai import types
from uniparser_tools.api.clients import UniParserClient
from uniparser_tools.common.constant import ParseMode, ParseModeTextual

try:
    from PIL import Image as PILImage
except ImportError:
    PILImage = None


# ================= 1. 全局配置与常量 =================
UNIPARSER_API_KEY = "up_xxxx"

GEMINI_API_KEY = "sk-xxxx"

MODEL_PRO_NAME = "gemini-3.1-pro-preview"
# 目录配置
EXCEL_DIR = "EXCEL_DIR"              
PDF_DIR = "PDF_DIR"                  
FALLBACK_PDF_DIR = "processed-loss"  # PDF_DIR 缺失文献时使用的备用目录
OUTPUT_DIR = "OUTPUT_EXCEL_DIR"      

# 并发配置
MAX_WORKERS = 2                      
MAX_PDFS_PER_RUN = None               # 仅处理断点过滤后的新文献；0 或 None 表示不限制
API_CALLS_PER_MINUTE = 100           
API_MAX_RETRIES = 5                  
API_INITIAL_DELAY = 20               
INTERRUPT_COMMIT_GRACE_SECONDS = 10  # Ctrl+C 后只等待已进入最终落盘的任务
GEMINI_REQUEST_TIMEOUT_SECONDS = 540 # 单次 Gemini 调用的外层截止时间
UNIPARSER_REQUEST_TIMEOUT_SECONDS = 120 # UniParser 同步解析/取结果截止时间
MAX_EVIDENCE_FILTER_PAGES = 12       # IUPAC 证据预筛选每批最多覆盖的 UniParser 页面数
MAX_EVIDENCE_FILTER_CHARS = 90000    # 同页或单个超大记录不拆分，其余按字符数分批
EVIDENCE_NEIGHBOR_BLOCKS = 2         # IUPAC 命中块前后自动保留的原始文本块数量
IUPAC_NMR_PAGE_RADIUS = 1            # 同时含 NMR/Hz 的关键页向前、向后各保留一页
MAX_FIGURE_LONG_EDGE = 1600           # 发送 Gemini 前缩小 Figure，避免高分辨率原图重复消耗
FIGURE_JPEG_QUALITY = 85              # JPEG Figure 缩放后的保存质量
VERBOSE_PIPELINE_REJECTIONS = False  # 默认只输出每篇文献的阶段摘要

# 颜色配置
BLUE_FILL = PatternFill(start_color="B8CCE4", end_color="B8CCE4", fill_type="solid")
PALE_YELLOW_FILL = PatternFill(start_color="FFF2CC", end_color="FFF2CC", fill_type="solid")
PINK_FILL = PatternFill(start_color="F4CCCC", end_color="F4CCCC", fill_type="solid")
PALE_PURPLE_FILL = PatternFill(start_color="E4DFEC", end_color="E4DFEC", fill_type="solid")
RED_FILL = PatternFill(start_color="FF0000", end_color="FF0000", fill_type="solid")

PROCESSED_PDF_COUNT = 0
COUNT_LOCK = threading.Lock()

METAL_SYMBOLS = {
    "Li","Be","Na","Mg","Al","K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga",
    "Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn","Cs","Ba","La","Ce",
    "Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu","Hf","Ta","W","Re","Os",
    "Ir","Pt","Au","Hg","Tl","Pb","Bi","Po","Fr","Ra","Ac","Th","Pa","U","Np","Pu","Am","Cm"
}

def contains_metal(smiles):
    if not smiles: return False
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol: return False
        for atom in mol.GetAtoms():
            if atom.GetSymbol() in METAL_SYMBOLS:
                return True
        return False
    except Exception:
        return False
    
os.makedirs(EXCEL_DIR, exist_ok=True)
os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(FALLBACK_PDF_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= 1.5 离线真理字典 (Offline Ground Truth) =================
RAW_COMMON_TB_DRUGS = {
    "isoniazid": "NNC(C1C=CN=CC=1)=O",
    "rifampicin": "O=C(/C(C)=C\C=C\[C@H](C)[C@H](O)[C@@H](C)[C@@H](O)[C@H]([C@H](OC(C)=O)[C@H](C)[C@@H](OC)/C=C/O[C@@]1(C)O2)C)NC3=C(O)C4=C(O)C(C)=C2C(C1=O)=C4C(O)=C3/C=N/N5CCN(C)CC5",
    "ethambutol": "CC[C@@H](CO)NCCN[C@@H](CC)CO",
    "pyrazinamide": "NC(C1=NC=CN=C1)=O",
    "streptomycin": "O[C@H]1[C@H](O[C@@H]2[C@@H](O[C@@H]3[C@H](NC)[C@@H](O)[C@@H](O)[C@H](CO)O3)[C@](O)(C=O)[C@@H](C)O2)[C@@H](NC(N)=N)[C@H](O)[C@@H](NC(N)=N)[C@@H]1O",
    "ciprofloxacin": "O=C(O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O",
    "ofloxacin": "CC1COc2c(N3CCN(C)CC3)c(F)cc3c(=O)c(C(=O)O)cn1c23",
    "levofloxacin": "C[C@H]1COc2c(N3CCN(C)CC3)c(F)cc3c(=O)c(C(=O)O)cn1c23",
    "moxifloxacin": "OC(C1=CN(C2=C(C(N3C[C@@]4([H])[C@@](NCCC4)([H])C3)=C(F)C=C2C1=O)OC)C5CC5)=O",
    "bedaquiline": "CO[C@](CCN(C)C)(C1=C2C=CC=CC2=CC=C1)[C@@H](C3=CC4=CC(Br)=CC=C4N=C3OC)C5=CC=CC=C5",
    "delamanid": "CC1(COc2ccc(N3CCC(Oc4ccc(OC(F)(F)F)cc4)CC3)cc2)Cn2cc([N+](=O)[O-])nc2O1",
    "pretomanid": "O=[N+]([O-])c1cn2c(n1)OC[C@@H](OCc1ccc(OC(F)(F)F)cc1)C2",
    "macozinone": "C1CCC(CC1)CN2CCN(CC2)C3=NC(=O)C4=C(S3)C(=CC(=C4)C(F)(F)F)[N+](=O)[O-]",
    "btz043": "C[C@H]1COC2(CCN(c3nc(=O)c4cc(C(F)(F)F)cc([N+](=O)[O-])c4s3)CC2)O1",
    "btz-043": "C[C@H]1COC2(CCN(c3nc(=O)c4cc(C(F)(F)F)cc([N+](=O)[O-])c4s3)CC2)O1",
    "linezolid": "CC(=O)NC[C@H]1CN(c2ccc(N3CCOCC3)c(F)c2)C(=O)O1",
    "clofazimine": "CC(C)/N=c1\cc2n(-c3ccc(Cl)cc3)c3ccccc3nc-2cc1Nc1ccc(Cl)cc1",
    "erythromycin": "CC[C@H]1OC(=O)[C@H](C)[C@@H](O[C@H]2CC(C)(OC)[C@@H](O)[C@H](C)O2)[C@H](C)[C@@H](O[C@@H]2O[C@H](C)C[C@H](N(C)C)[C@H]2O)[C@](C)(O)C[C@@H](C)C(=O)[C@H](C)[C@@H](O)[C@]1(C)O",
    "azithromycin": "CC[C@H]1OC(=O)[C@H](C)[C@@H](O[C@H]2CC(C)(OC)[C@@H](O)[C@H](C)O2)[C@H](C)[C@@H](O[C@@H]2O[C@H](C)C[C@H](N(C)C)[C@H]2O)[C@](C)(O)C[C@@H](C)CN(C)[C@@H](C)[C@@H](O)[C@]1(C)O",
    "clarithromycin": "CC[C@H]1OC(=O)[C@H](C)[C@@H](O[C@H]2CC(C)(OC)[C@@H](O)[C@H](C)O2)[C@H](C)[C@@H](O[C@@H]2O[C@H](C)C[C@H](N(C)C)[C@H]2O)[C@@](C)(OC)C[C@@H](C)C(=O)[C@H](C)[C@@H](O)[C@]1(C)O",
    "roxithromycin": "CC[C@H]1OC(=O)[C@H](C)[C@@H](O[C@H]2C[C@@](C)(OC)[C@@H](O)[C@H](C)O2)[C@H](C)[C@@H](O[C@@H]2O[C@H](C)C[C@H](N(C)C)[C@H]2O)[C@@](C)(O)C[C@@H](C)/C(=N\OCOCCOC)[C@H](C)[C@@H](O)[C@]1(C)O",
    "rifapentine": "COC(=O)[C@H]1[C@H](C)[C@@H](OC)C=CO[C@@]2(C)Oc3c(C)c(O)c4c(O)c(/C=N/N5CCN(C6CCCC6)CC5)c(c(O)c4c3C2=O)NC(=O)C(C)=CC=C[C@H](C)[C@@H](O)[C@H](C)[C@@H](O)[C@@H]1C",
    "rifabutin": "CO[C@@H]1C=CO[C@@]2(C)Oc3c(C)c(O)c4c(c3C2=O)C2=NC3(CCN(CC(C)C)CC3)NC2=C(NC(=O)C(C)=CC=C[C@H](C)[C@H](O)[C@@H](C)[C@@H](O)[C@@H](C)[C@H](OC(C)=O)[C@@H]1C)C4=O",
    "fidaxomicin": "CC[C@H]1/C=C(/[C@H](C/C=C/C=C(/C(=O)O[C@@H](C/C=C(/C=C(/[C@@H]1O[C@H]2[C@H]([C@H]([C@@H](C(O2)(C)C)OC(=O)C(C)C)O)O)\C)\C)[C@@H](C)O)\CO[C@H]3[C@H]([C@H]([C@@H]([C@H](O3)C)OC(=O)C4=C(C(=C(C(=C4O)Cl)O)Cl)CC)O)OC)O)\C",
    "telithromycin": "CC[C@@H]1[C@@]2([C@@H]([C@H](C(=O)[C@@H](C[C@@]([C@@H]([C@H](C(=O)[C@H](C(=O)O1)C)C)O[C@H]3[C@@H]([C@H](C[C@H](O3)C)N(C)C)O)(C)OC)C)C)N(C(=O)O2)CCCCN4C=C(N=C4)C5=CN=CC=C5)C",
    "ohmyungsamycin a": "C[C@@H]1[C@@H](C(=O)N([C@H](C(=O)N[C@H](C(=O)N([C@H](C(=O)N[C@H](C(=O)N([C@H](C(=O)N([C@H](C(=O)N[C@H](C(=O)N[C@H](C(=O)N[C@H](C(=O)O1)C(C)C)[C@@H](C2=CC=CC=C2)O)C(C)C)CC3=CNC4=C3C(=CC=C4)OC)C)C(C)C)C)C(C)C)CC(C)C)C)C(C)C)[C@@H](C)O)C)NC(=O)[C@H](C(C)C)NC(=O)[C@H](C(C)C)NC",
    "celastrol": "CC1=C(C(=O)C=C2C1=CC=C3[C@]2(CC[C@@]4([C@@]3(CC[C@@]5([C@H]4C[C@](CC5)(C)C(=O)O)C)C)C)C)O",
    "triclosan": "Oc1cc(Cl)ccc1Oc1ccc(Cl)cc1Cl",
    "kanamycin": "C1[C@H]([C@@H]([C@H]([C@@H]([C@H]1N)O[C@@H]2[C@@H]([C@H]([C@@H]([C@H](O2)CN)O)O)O)O[C@@H]3[C@@H]([C@H]([C@@H]([C@H](O3)CO)O)N)O)N",
    "vancomycin": "C[C@H]1[C@H]([C@@](C[C@@H](O1)O[C@@H]2[C@H]([C@@H]([C@H](O[C@H]2OC3=C4C=C5C=C3OC6=C(C=C(C=C6)[C@H]([C@H](C(=O)N[C@H](C(=O)N[C@H]5C(=O)N[C@@H]7C8=CC(=C(C=C8)O)C9=C(C=C(C=C9O)O)[C@H](NC(=O)[C@H]([C@@H](C1=CC(=C(O4)C=C1)Cl)O)NC7=O)C(=O)O)CC(=O)N)NC(=O)[C@@H](CC(C)C)NC)O)Cl)CO)O)O)(C)N)O",
    "ethionamide": "CCC1=NC=CC(=C1)C(=S)N",
    "gatifloxacin": "CC1CN(CCN1)C2=C(C=C3C(=C2OC)N(C=C(C3=O)C(=O)O)C4CC4)F",
    "metronidazole": "CC1=NC=C(N1CCO)[N+](=O)[O-]",
    "nitazoxanide": "CC(=O)OC1=CC=CC=C1C(=O)NC2=NC=C(S2)[N+](=O)[O-]",
    "tizoxanide": " C1=CC=C(C(=C1)C(=O)NC2=NC=C(S2)[N+](=O)[O-])O",
    "rifampin":"C[C@H]1/C=C/C=C(\\C(=O)NC2=C(C(=C3C(=C2O)C(=C(C4=C3C(=O)[C@](O4)(O/C=C/[C@@H]([C@H]([C@H]([C@@H]([C@@H]([C@@H]([C@H]1O)C)O)C)OC(=O)C)C)OC)C)C)O)O)/C=N/N5CCN(CC5)C)/C",
    "cycloserine":"C1[C@H](C(=O)NO1)N",
    "amikacin":"C1[C@@H]([C@H]([C@@H]([C@H]([C@@H]1NC(=O)[C@H](CCN)O)O[C@@H]2[C@@H]([C@H]([C@@H]([C@H](O2)CO)O)N)O)O)O[C@@H]3[C@@H]([C@H]([C@@H]([C@H](O3)CN)O)O)O)N",
    "pyrimethamine":"CCC1=C(C(=NC(=N1)N)N)C2=CC=C(C=C2)Cl",
    "clinafloxacin":"NC1CCN(c2cc3c(c(Cl)c2F)c(=O)c(C(=O)O)cn3C2CC2)C1",
    "Norfloxacin":"CCn1cc(C(=O)O)c(=O)c2cc(F)c(N3CCNCC3)cc21",
    "Sarafloxacin":"O=C(O)c1cn(-c2ccc(F)cc2)c2cc(N3CCNCC3)c(F)cc2c1=O",
    "Ethambutol":"CCC(CO)CNCCNC(CC)CO",
    "Ampicillin":"CC1([C@@H](N2[C@H](S1)[C@@H](C2=O)NC(=O)[C@@H](C3=CC=CC=C3)N)C(=O)O)C",
    "Novobiocin":"CC1=C(C=CC2=C1OC(=O)C(=C2O)NC(=O)C3=CC(=C(C=C3)O)CC=C(C)C)O[C@H]4[C@@H]([C@@H]([C@H](C(O4)(C)C)OC)OC(=O)N)O",
    "Ciprofloxacine":"C1CC1N2C=C(C(=O)C3=CC(=C(C=C32)N4CCNCC4)F)C(=O)O",
    "p-aminosalicylic":"C1=CC(=C(C=C1N)O)C(=O)O",
    "Clotrimazole":"C1=CC=C(C=C1)C(C2=CC=CC=C2)(C3=CC=CC=C3Cl)N4C=CN=C4",
    "Coumermycin A1":"CC1=CC=C(N1)C(=O)O[C@H]2[C@H]([C@@H](OC([C@@H]2OC)(C)C)OC3=C(C4=C(C=C3)C(=C(C(=O)O4)NC(=O)C5=CNC(=C5C)C(=O)NC6=C(C7=C(C(=C(C=C7)O[C@H]8[C@@H]([C@@H]([C@H](C(O8)(C)C)OC)OC(=O)C9=CC=C(N9)C)O)C)OC6=O)O)O)C)O",
    "tigecycline":"CC(C)(C)NCC(=O)NC1=CC(=C2C[C@H]3C[C@H]4[C@@H](C(=O)C(=C([C@]4(C(=O)C3=C(C2=C1O)O)O)O)C(=O)N)N(C)C)N(C)C",
    "Capreomycin":"C[C@H]1C(=O)N[C@H](C(=O)N/C(=C/NC(=O)N)/C(=O)N[C@H](C(=O)NC[C@@H](C(=O)N1)N)[C@H]2CCN=C(N2)N)CNC(=O)C[C@H](CCCN)N.C1CN=C(N[C@H]1[C@H]2C(=O)NC[C@@H](C(=O)N[C@H](C(=O)N[C@H](C(=O)N/C(=C/NC(=O)N)/C(=O)N2)CNC(=O)C[C@H](CCCN)N)CO)N)N",
    "Deoxyecumicin":"CC[C@@H](C)[C@@H](C(=O)N[C@@H]1C(=O)N(C)[C@@H]([C@@H](C)O)C(=O)N[C@@H](C(C)C)C(=O)N(C)[C@@H](CC(C)C)C(=O)N[C@@H](C(C)C)C(=O)N(C)[C@@H](C(C)C)C(=O)N(C)[C@@H](C(C)C)C(=O)N(C)[C@@H](Cc2c[nH]c3cccc(OC)c23)C(=O)N[C@@H](C(C)C)C(=O)N[C@@H](Cc2ccccc2)C(=O)N[C@@H](C(C)C)C(=O)O[C@@H]1C)N(C)C(=O)[C@@H](NC(=O)[C@H](C(C)C)N(C)C)C(C)C",
    "Ecumicin":"CC[C@@H](C)[C@@H](C(=O)N[C@@H]1C(=O)N(C)[C@@H]([C@@H](C)O)C(=O)N[C@@H](C(C)C)C(=O)N(C)[C@@H](CC(C)C)C(=O)N[C@@H](C(C)C)C(=O)N(C)[C@@H](C(C)C)C(=O)N(C)[C@@H](C(C)C)C(=O)N(C)[C@@H](Cc2c[nH]c3cccc(OC)c23)C(=O)N[C@@H](C(C)C)C(=O)N[C@@H]([C@H](O)c2ccccc2)C(=O)N[C@@H](C(C)C)C(=O)O[C@@H]1C)N(C)C(=O)[C@@H](NC(=O)[C@H](C(C)C)N(C)C)C(C)C",
    "Kanamycin":"C1[C@H]([C@@H]([C@H]([C@@H]([C@H]1N)O[C@@H]2[C@@H]([C@H]([C@@H]([C@H](O2)CN)O)O)O)O)O[C@@H]3[C@@H]([C@H]([C@@H]([C@H](O3)CO)O)N)O)N",
    "Norfloxacin":"CCN1C=C(C(=O)C2=CC(=C(C=C21)N3CCNCC3)F)C(=O)O",
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
    "Capreomycin":"C[C@H]1C(=O)N[C@H](C(=O)N/C(=C\\NC(=O)N)/C(=O)N[C@H](C(=O)NC[C@@H](C(=O)N1)N)C2CCN=C(N2)N)CNC(=O)CC(CCCN)N.C1CN=C(NC1[C@H]2C(=O)NC[C@@H](C(=O)N[C@H](C(=O)N[C@H](C(=O)N/C(=C\\NC(=O)N)/C(=O)N2)CNC(=O)CC(CCCN)N)CO)N)N.OS(=O)(=O)O.OS(=O)(=O)O",
    "Imipenem":"C[C@H]([C@@H]1[C@H]2CC(=C(N2C1=O)C(=O)O)SCCN=CN)O",
    "SQ109":"CC(=CCC/C(=C/CNCCNC1C2CC3CC(C2)CC1C3)/C)C",
    "Ampicillin":"CC1([C@@H](N2[C@H](S1)[C@@H](C2=O)NC(=O)[C@@H](C3=CC=CC=C3)N)C(=O)O)C",
    "Nitroimidazopyran":"C1=COC2=NC(=NC2=C1)[N+](=O)[O-]",
    "Fluconazole":"C1=CC(=C(C=C1F)F)C(CN2C=NC=N2)(CN3C=NC=N3)O",
    "Econazole":"C1=CC(=CC=C1COC(CN2C=CN=C2)C3=C(C=C(C=C3)Cl)Cl)Cl",
    "mefloquine":"C1CCNC(C1)C(C2=CC(=NC3=C2C=CC=C3C(F)(F)F)C(F)(F)F)O",
    "Actinonin":"CCCCCC(CC(=O)NO)C(=O)NC(C(=O)N1CCCC1CO)C(C)C",
    "Nitroimidazopyran":"C1=COC2=NC(=NC2=C1)[N+](=O)[O-]",
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
    "ferulenol":"CC(=CCC/C(=C/CC/C(=C/CC1=C(C2=CC=CC=C2OC1=O)O)/C)/C)C",
    "suberosin":"CC(=CCC1=C(C=C2C(=C1)C=CC(=O)O2)OC)C",
    "Sparfloxacin":"C[C@@H]1CN(C[C@@H](N1)C)C2=C(C(=C3C(=C2F)N(C=C(C3=O)C(=O)O)C4CC4)N)F",
    "Tobramycin":"C1[C@@H]([C@H]([C@@H]([C@H]([C@@H]1N)O[C@@H]2[C@@H]([C@H]([C@@H]([C@H](O2)CO)O)N)O)O)O[C@@H]3[C@@H](C[C@@H]([C@H](O3)CN)O)N)N",
    "Spectinomycin":"C[C@@H]1CC(=O)[C@]2([C@@H](O1)O[C@@H]3[C@H]([C@@H]([C@@H]([C@@H]([C@H]3O2)NC)O)NC)O)O",
    "limocitrin":"COC1=C(C=CC(=C1)C2=C(C(=O)C3=C(O2)C(=C(C=C3O)O)OC)O)O",
    "Luteolin":"C1=CC(=C(C=C1C2=CC(=O)C3=C(C=C(C=C3O2)O)O)O)O",
    "Kanglemycin A":"C[C@@H]1[C@H](/C=C/O[C@]2(C(=O)C3=C(O2)C(=C(C4=C3C(=O)C=C(C4=O)NC(=O)/C(=C/C=C/[C@@H]([C@@H]([C@@H]([C@@H]([C@@H]([C@H]1OC(=O)C)C)O)C)O)[C@@H](C)OC(=O)C(C)(C)CC(=O)O)/C)O)C)C)O[C@@H]5C[C@H]6[C@@H]([C@@H](O5)C)OCO6",
    "Aurachin D":"CC1=C(C(=O)C2=CC=CC=C2N1)C/C=C(\\C)/CC/C=C(\\C)/CCC=C(C)C",
    "Trimethoprim":"COC1=CC(=CC(=C1OC)OC)CC2=CN=C(N=C2N)N",
    "Aditoprim":"CN(C)C1=C(C=C(C=C1OC)CC2=CN=C(N=C2N)N)OC",
    "Arbekacin":"C1C[C@H]([C@H](O[C@@H]1CN)O[C@@H]2[C@H](C[C@H]([C@@H]([C@H]2O)O[C@@H]3[C@@H]([C@H]([C@@H]([C@H](O3)CO)O)N)O)NC(=O)[C@H](CCN)O)N)N",
    "Gentamicin":"CC(C1CCC(C(O1)OC2C(CC(C(C2O)OC3C(C(C(CO3)(C)O)NC)O)N)N)N)NC", 
    "Methotrexate":"CN(CC1=CN=C2C(=N1)C(=NC(=N2)N)N)C3=CC=C(C=C3)C(=O)N[C@@H](CCC(=O)O)C(=O)O", 
    "Sansanmycin A":"CC(C(C(=O)N/C=C/1\C[C@H]([C@@H](O1)N2C=CC(=O)NC2=O)O)NC(=O)C(CCSC)NC(=O)NC(CC3=CNC4=CC=CC=C43)C(=O)O)N(C)C(=O)C(CC5=CC(=CC=C5)O)N",
    "Nitrofurazone":"C1=C(OC(=C1)[N+](=O)[O-])/C=N/NC(=O)N",
    "Nitrofurantoin":"C1C(=O)NC(=O)N1/N=C/C2=CC=C(O2)[N+](=O)[O-]",
    "Penicillin":"CC1(C(N2C(S1)C(C2=O)NC(=O)CC3=CC=CC=C3)C(=O)O)C"
}
COMMON_TB_DRUGS = {k.lower(): v for k, v in RAW_COMMON_TB_DRUGS.items()}

# ================= 2. 客户端与稳健性保障 =================
genai_client = genai.Client(
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

class RateLimiter:
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

gemini_rate_limiter = RateLimiter(API_CALLS_PER_MINUTE)
GLOBAL_STOP_EVENT = threading.Event()
FINAL_COMMIT_CONDITION = threading.Condition(threading.Lock())
ACTIVE_FINAL_COMMITS = set()


def _request_global_stop():
    """与最终提交登记使用同一把锁，确保中断边界没有竞态。"""
    with FINAL_COMMIT_CONDITION:
        GLOBAL_STOP_EVENT.set()
        FINAL_COMMIT_CONDITION.notify_all()


def _begin_final_commit(commit_key):
    """仅允许中断发生前已经进入最终提交阶段的任务继续落盘。"""
    with FINAL_COMMIT_CONDITION:
        if GLOBAL_STOP_EVENT.is_set():
            return False
        ACTIVE_FINAL_COMMITS.add(commit_key)
        return True


def _end_final_commit(commit_key):
    with FINAL_COMMIT_CONDITION:
        ACTIVE_FINAL_COMMITS.discard(commit_key)
        FINAL_COMMIT_CONDITION.notify_all()


def _force_exit_after_commit_grace(reason, exit_code):
    """最多等待已登记的最终提交；不等待阻塞在网络调用中的普通线程。"""
    deadline = time.monotonic() + INTERRUPT_COMMIT_GRACE_SECONDS
    pending = 0
    try:
        with FINAL_COMMIT_CONDITION:
            while ACTIVE_FINAL_COMMITS:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                pending = len(ACTIVE_FINAL_COMMITS)
                print(
                    f"[!] {reason}：等待 {pending} 个已开始的落盘任务，"
                    f"最多再等待 {remaining:.1f} 秒...",
                    flush=True,
                )
                FINAL_COMMIT_CONDITION.wait(timeout=min(0.5, remaining))
            pending = len(ACTIVE_FINAL_COMMITS)
    except KeyboardInterrupt:
        pending = len(ACTIVE_FINAL_COMMITS)
        print("[!] 再次收到 Ctrl+C，立即强制退出。", flush=True)
    if pending:
        print(
            f"[!] 10 秒落盘窗口已结束；放弃 {pending} 个未完成提交并强制退出。"
            "这些文献下次将重新处理。",
            flush=True,
        )
    else:
        print(
            "[!] 已开始的落盘任务均已结束；现在强制退出，"
            "其余处理一半的文献不会写入。",
            flush=True,
        )
    os._exit(exit_code)


class APIRequestDeadlineExceeded(TimeoutError):
    pass


class APIRequestCancelled(Exception):
    pass


def _run_with_deadline(request_func, timeout_seconds, operation_name):
    """在守护线程中运行不可取消的 SDK 调用，避免其阻止主进程退出。"""
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


def _generate_content_with_deadline(client, **kwargs):
    return _run_with_deadline(
        lambda: client.models.generate_content(**kwargs),
        GEMINI_REQUEST_TIMEOUT_SECONDS,
        "Gemini generate_content",
    )


def _normalize_status(status):
    """兼容字符串状态和 Enum 状态。"""
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
    """返回 fatal、retry 或 document，避免依赖模糊的单词 token。"""
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
    """UniParser SDK 以 dict 返回错误，因此必须检查 status，而不是只捕获异常。"""
    delay = API_INITIAL_DELAY

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
            print(
                f"[-] Uni-Parser 请求超时: {exc}；本轮停止，"
                "当前文献不落盘并留待下次重新处理。"
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
            print(f"\n[FATAL] 🛑 Uni-Parser {operation_name} 鉴权、权限或余额检查失败: {error_text}")
            _request_global_stop()
            return None

        if error_kind == "document":
            print(f"[-] Uni-Parser {operation_name} 请求不可重试: {error_text}")
            return None

        print(f"[Uni-Parser Retry {attempt}/{API_MAX_RETRIES}] {operation_name}: {error_text}")
        if attempt == API_MAX_RETRIES:
            return None
        if GLOBAL_STOP_EVENT.wait(delay):
            return None
        delay *= 2

    return None


def _is_transient_gemini_network_error(exc):
    """只识别传输层临时故障；限流、模型输出和请求参数错误均不重试。"""
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


def retry_gemini_api():
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if GLOBAL_STOP_EVENT.is_set():
                return None

            delay = API_INITIAL_DELAY
            for attempt in range(1, API_MAX_RETRIES + 1):
                if GLOBAL_STOP_EVENT.is_set():
                    return None

                try:
                    return func(*args, **kwargs)
                except APIRequestCancelled:
                    return None
                except APIRequestDeadlineExceeded as exc:
                    print(f"[-] Gemini 单次请求超时，不重复提交同一请求: {exc}")
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
                        print(f"\n[FATAL] 🛑 检测到 Gemini API 额度、鉴权或权限错误！引发报错: {exc}")
                        print("[FATAL] 已点亮全局停止红灯，正在通知所有线程安全撤退...")
                        _request_global_stop()
                        return None

                    if is_context_error:
                        print(f"[-] Gemini 上下文过长，重复请求不会解决问题: {exc}")
                        return None

                    if not _is_transient_gemini_network_error(exc):
                        print(
                            f"[-] Gemini 非网络错误，不进行重复请求 "
                            f"({func.__name__}): {exc}"
                        )
                        return None

                    print(f"[Gemini Retry {attempt}/{API_MAX_RETRIES}] {func.__name__}: {exc}")
                    if attempt == API_MAX_RETRIES:
                        return None
                    if GLOBAL_STOP_EVENT.wait(delay):
                        return None
                    delay *= 2

            return None
        return wrapper
    return decorator


def test_uniparser_connection():
    print("\n[*] 正在进行前置检查: 验证 Uni-Parser API 可用性...")
    try:
        result = _run_with_deadline(
            parser_client.health,
            30,
            "UniParser health check",
        )
    except Exception as exc:
        print(f"[FATAL] ❌ Uni-Parser API 健康检查调用失败: {type(exc).__name__}: {exc}")
        return False

    if _uniparser_response_ok(result, health_check=True):
        print("[+] Uni-Parser API 状态正常。")
        return True

    print(f"[FATAL] ❌ Uni-Parser API 健康检查失败: {_uniparser_error_text(result)}")
    return False


def test_gemini_connection():
    print("\n[*] 正在进行前置检查: 验证 Gemini API 可用性...")
    try:
        response = _generate_content_with_deadline(
            genai_client,
            model=MODEL_PRO_NAME,
            contents=["这是连通性检查。请只回复 Pong，不要输出其他内容。"],
            config=types.GenerateContentConfig(temperature=0.1)
        )
        if response.text and "pong" in response.text.lower():
            print("[+] Gemini API 状态正常，准备执行主任务。")
            return True
        print("[FATAL] ❌ Gemini API 返回了非预期的健康检查响应。")
        return False
    except Exception as exc:
        print("\n[FATAL] ❌ Gemini API 连接失败！请检查额度或网络代理。")
        print(f"报错详情: {exc}")
        return False

# ================= 3. 化学结构辅助函数与提示词 =================
PDF_PROCESS_FAILED = object()
VALID_EMPTY_TEXT = {"", "n/a", "nan", "none", "null"}


def standardize_smiles(smiles):
    text = str(smiles or "").strip()
    if text.casefold() in VALID_EMPTY_TEXT:
        return None
    try:
        mol = Chem.MolFromSmiles(text)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def smiles_stereo_compatible_for_coloring(left_smiles, right_smiles):
    """连接关系相同且不存在明确立体冲突时，颜色判断视为兼容。"""
    left_canonical = standardize_smiles(left_smiles)
    right_canonical = standardize_smiles(right_smiles)
    if not left_canonical or not right_canonical:
        return False
    if left_canonical == right_canonical:
        return True
    try:
        left_mol = Chem.MolFromSmiles(left_canonical)
        right_mol = Chem.MolFromSmiles(right_canonical)
        if left_mol is None or right_mol is None:
            return False

        left_without_stereo = Chem.Mol(left_mol)
        right_without_stereo = Chem.Mol(right_mol)
        Chem.RemoveStereochemistry(left_without_stereo)
        Chem.RemoveStereochemistry(right_without_stereo)
        left_connectivity = Chem.MolToSmiles(
            left_without_stereo, isomericSmiles=True
        )
        right_connectivity = Chem.MolToSmiles(
            right_without_stereo, isomericSmiles=True
        )
        if left_connectivity != right_connectivity:
            return False

        match_parameters = Chem.SubstructMatchParameters()
        match_parameters.useChirality = True
        return (
            left_mol.HasSubstructMatch(right_mol, match_parameters)
            or right_mol.HasSubstructMatch(left_mol, match_parameters)
        )
    except Exception:
        return False


def clean_excel_text(value):
    text = str(value or "").strip()
    return "" if text.casefold() in VALID_EMPTY_TEXT else text


def normalize_compound_identifier(value):
    """保守归一化 Compound ID，只清理固定前缀和空白，不猜测编号。"""
    text = clean_excel_text(value)
    text = re.sub(r"^(compound|compd|derivative)\s+", "", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", "", text).casefold()


def _stable_unique(items):
    seen = set()
    result = []
    for item in items:
        if item in (None, "", [], {}):
            continue
        normalized = item.strip() if isinstance(item, str) else item
        try:
            key = normalized
            hash(key)
        except TypeError:
            key = json.dumps(normalized, ensure_ascii=False, sort_keys=True)
        if key not in seen:
            seen.add(key)
            result.append(normalized)
    return result


def clean_json_text(text):
    if not text:
        return ""
    clean_text = str(text).strip()
    fence = chr(96) * 3
    json_fence = fence + "json"
    if json_fence in clean_text:
        parts = clean_text.split(json_fence)
        if len(parts) > 1:
            clean_text = parts[1].split(fence)[0].strip()
    elif fence in clean_text:
        parts = clean_text.split(fence)
        if len(parts) > 1:
            clean_text = parts[1].split(fence)[0].strip()
    return clean_text


def _parse_json_array(response_text):
    """返回字典列表；None 表示模型响应失败或无法解析，由调用阶段决定降级策略。"""
    if response_text is None or not str(response_text).strip():
        return None
    try:
        data = json_repair.loads(clean_json_text(response_text))
    except Exception as exc:
        print(f"[-] Gemini JSON 响应无法修复: {exc}")
        return None
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    print(f"[-] Gemini JSON 顶层类型异常: {type(data).__name__}")
    return None


def _parse_figure_filter_response(response_text):
    """返回严格布尔值；None 表示筛选响应失败，不能静默当作无关 Figure。"""
    if response_text is None or not str(response_text).strip():
        return None
    try:
        data = json_repair.loads(clean_json_text(response_text))
    except Exception as exc:
        print(f"[-] Figure 筛选 JSON 响应无法修复: {exc}")
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


def build_targets(rows):
    targets = {}
    for row in rows:
        display_id = clean_excel_text(row.get("Compound ID", ""))
        target_key = normalize_compound_identifier(display_id)
        if not target_key:
            continue
        target = targets.setdefault(target_key, {
            "display_id": display_id,
            "dictionary_names": [],
        })
        old_iupac = clean_excel_text(row.get("IUPAC Name", ""))
        if old_iupac:
            target["dictionary_names"].append(old_iupac)
    for target in targets.values():
        target["dictionary_names"] = _stable_unique(target["dictionary_names"])
    return targets


def collect_dictionary_candidates(targets, explicit_names=None):
    """对 ID、Excel 原名称和文献明确名称做大小写归一化后的精确匹配。"""
    explicit_names = explicit_names or {}
    accepted = {}
    for target_key, target in targets.items():
        lookup_names = (
            [target["display_id"]]
            + target.get("dictionary_names", [])
            + explicit_names.get(target_key, [])
        )
        structures = set()
        for name in _stable_unique(lookup_names):
            raw_smiles = COMMON_TB_DRUGS.get(str(name).strip().casefold())
            canonical = standardize_smiles(raw_smiles)
            if canonical:
                structures.add(canonical)
        if len(structures) == 1:
            accepted[target_key] = next(iter(structures))
        elif len(structures) > 1:
            print(
                f"   [词典冲突] {target['display_id']} 精确命中了 "
                f"{len(structures)} 个不同结构，本轮不采用词典。"
            )
    return accepted


def rows_require_online_api(rows):
    targets = build_targets(rows)
    if not targets:
        return False
    dictionary_candidates = collect_dictionary_candidates(targets)
    return any(key not in dictionary_candidates for key in targets)



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
- 不得从图片中输出 SMILES、IUPAC Name、活性值或任何结构结论。
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
# ================= 4. UniParser 全文文档结构 =================
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


def _prepare_figure_for_gemini(image_bytes, mime_type):
    """缩小发送给 Gemini 的图片；失败时保留原始数据，不影响文档处理。"""
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
        print(f"[-] Figure 缩放失败，继续使用原图: {type(exc).__name__}: {exc}")
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
        api_image_bytes, api_mime_type, resize_note = _prepare_figure_for_gemini(
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
    """整篇解析文献，并准备降采样后的 Figure 供多模态推理使用。"""
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
        print(f"[-] UniParser 成功响应中缺少 token: {trigger_result}")
        return None
    formatted_result = _call_uniparser_with_retry(
        "获取格式化结果",
        lambda: parser_client.get_formatted(token, content=False, pages_tree=True)
    )
    if not formatted_result:
        return None
    pages_tree = formatted_result.get("pages_tree") or []
    if not isinstance(pages_tree, list):
        print(f"[-] UniParser pages_tree 格式异常: {type(pages_tree).__name__}")
        return None

    table_blocks, molecule_records, text_blocks = [], [], []
    figure_map = {}

    def add_text_block(page_number, node_type, text):
        clean = str(text or "").strip()
        if clean:
            text_blocks.append({"page": page_number, "type": node_type, "text": clean})

    def traverse(node_list, page_number):
        if isinstance(node_list, dict):
            node_list = [node_list]
        if not isinstance(node_list, list):
            return
        for node in node_list:
            if not isinstance(node, dict):
                continue
            node_type = str(node.get("type", "")).strip().lower()
            children = _node_children(node)

            if node_type in {"figure", "image", "figuregroup", "imagegroup"}:
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

            if node_type == "tablegroup":
                captions, tables, remaining_children = [], [], []
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
                caption_text = " | ".join(_stable_unique(captions))
                for table_text in _stable_unique(tables):
                    table_blocks.append({
                        "page": page_number,
                        "caption": caption_text,
                        "text": table_text,
                    })
                traverse(remaining_children, page_number)
                continue

            child_types = [
                str(child.get("type", "")).strip().lower()
                for child in children
            ]
            is_molecule_group = (
                "group" in node_type
                and any(t in {"molecule", "moleculeid"} for t in child_types)
            )
            if is_molecule_group:
                cores, molecule_ids, nearby_texts, sequence = [], [], [], []
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
                        "ids": _stable_unique(molecule_ids),
                        "cores": _stable_unique(cores),
                        "text": " ".join(_stable_unique(nearby_texts)),
                        "kind": "molecule",
                        "sequence": sequence,
                    })
                traverse(
                    [child for i, child in enumerate(children) if i not in consumed_indexes],
                    page_number
                )
                continue

            if node_type == "molecule":
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
            elif node_type == "moleculeid":
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
            elif "expression" in node_type:
                route_text = _node_text(node)
                if route_text:
                    molecule_records.append({
                        "page": page_number,
                        "ids": ["[Route]"],
                        "cores": [route_text],
                        "text": "[合成路线]",
                        "kind": "route",
                        "sequence": [{"position": 1, "type": node_type, "value": route_text}],
                    })
            elif node_type == "table":
                table_text = _table_context(node)
                if table_text:
                    table_blocks.append({
                        "page": page_number,
                        "caption": "",
                        "text": table_text,
                    })
            elif node_type in {"tablecaption", "tablefootnote"}:
                caption_text = _node_text(node)
                if caption_text:
                    table_blocks.append({
                        "page": page_number,
                        "caption": caption_text,
                        "text": "",
                    })
            elif (
                node_type in {
                    "text", "paragraph", "title", "heading", "section",
                    "list", "listitem", "moleculecaption",
                    "imagecaption", "expressioncaption",
                }
                or "text" in node_type
            ):
                add_text_block(page_number, node_type, _node_text(node))
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
        print("[-] 当前环境未安装 Pillow，Figure 将使用原始分辨率。")
    print(
        "   [UniParser] "
        f"页数={len(pages_tree)}，表格={len(document['table_blocks'])}，"
        f"分子记录={len(document['molecule_records'])}，"
        f"正文块={len(document['text_blocks'])}，"
        f"Figure={len(document['figure_records'])}，"
        f"已缩放={resized_figure_count}"
    )
    return document


# ================= 5. 结构证据映射与来源判定 =================
def _identifier_variants(raw_identifier):
    raw_text = clean_excel_text(raw_identifier)
    if not raw_text:
        return set()
    variants = {normalize_compound_identifier(raw_text)}
    prefix = re.split(r"[:\(\[\{]", raw_text, maxsplit=1)[0].strip()
    if "," not in prefix and ";" not in prefix:
        normalized_prefix = normalize_compound_identifier(prefix.rstrip(",;"))
        if normalized_prefix:
            variants.add(normalized_prefix)
    return {value for value in variants if value}


def _record_target_keys(record, target_keys):
    matches = set()
    for raw_identifier in record.get("ids", []):
        matches.update(
            value
            for value in _identifier_variants(raw_identifier)
            if value in target_keys
        )
    return matches


def _text_mentions_target(text, display_value):
    source = str(text or "")
    target = clean_excel_text(display_value)
    if not source or not target:
        return False
    escaped = re.escape(target)
    if len(normalize_compound_identifier(target)) <= 2:
        patterns = (
            rf"\b(?:compound|compd|derivative)\s*{escaped}\b",
            rf"(?m)^\s*(?:compound\s*)?{escaped}(?=\s|[\.:;,\)\]\(\-])",
            rf"(?<![A-Za-z0-9])\({escaped}\)(?![A-Za-z0-9])",
            rf"(?<![A-Za-z0-9])\[{escaped}\](?![A-Za-z0-9])",
        )
        return any(re.search(pattern, source, re.IGNORECASE) for pattern in patterns)
    return re.search(
        rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])",
        source,
        re.IGNORECASE
    ) is not None


def _format_table_block(block, index):
    return (
        f"===== Evidence ID {block.get('record_id', f'TAB-{index:04d}')} | "
        f"表格块 {index} | 页码 {block.get('page', '未知')} =====\n"
        f"表题/表注：{str(block.get('caption', '')).strip()}\n"
        f"{str(block.get('text', '')).strip()}"
    ).strip()


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


def format_molecule_records(records):
    parts = []
    for record in records:
        sequence = " -> ".join(
            f"{item.get('position')}:{item.get('type')}={item.get('value')}"
            for item in record.get("sequence", [])
        )
        parts.append(
            f"Evidence ID: {record.get('record_id', '未知')} | 页码: {record.get('page')} | "
            f"化合物编号（ID）: {' '.join(record.get('ids', []))} | "
            f"结构母核（Core）: {' || '.join(record.get('cores', []))} | "
            f"周边文字（Text）: {record.get('text', '')} | 原始组内顺序: {sequence}"
        )
    return "\n".join(parts)


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


@retry_gemini_api()
def call_iupac_evidence_filter(prompt):
    gemini_rate_limiter.acquire()
    response = _generate_content_with_deadline(
        genai_client,
        model=MODEL_PRO_NAME,
        contents=[prompt],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.0,
            thinking_config=types.ThinkingConfig(thinking_level="low")
        )
    )
    return response.text


@retry_gemini_api()
def call_structure_evidence_filter(prompt):
    gemini_rate_limiter.acquire()
    response = _generate_content_with_deadline(
        genai_client,
        model=MODEL_PRO_NAME,
        contents=[prompt],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.0,
            thinking_config=types.ThinkingConfig(thinking_level="low")
        )
    )
    return response.text


@retry_gemini_api()
def call_iupac_mapping(prompt):
    gemini_rate_limiter.acquire()
    response = _generate_content_with_deadline(
        genai_client,
        model=MODEL_PRO_NAME,
        contents=[prompt],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.0,
            thinking_config=types.ThinkingConfig(thinking_level="high")
        )
    )
    return response.text


@retry_gemini_api()
def call_figure_filter(prompt, figure_record):
    """使用同一 Pro 模型的低思考模式逐张筛选 Figure，不在此阶段生成结构。"""
    gemini_rate_limiter.acquire()
    response = _generate_content_with_deadline(
        genai_client,
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


@retry_gemini_api()
def call_structure_inference(prompt, relevant_figures=None):
    """在一次高思考调用中综合全部未解决目标、结构证据和已筛选 Figure。"""
    gemini_rate_limiter.acquire()
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
        genai_client,
        model=MODEL_PRO_NAME,
        contents=contents,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.0,
            thinking_config=types.ThinkingConfig(thinking_level="high")
        )
    )
    return response.text


def convert_verified_iupac_name(iupac_name):
    raw_name = clean_excel_text(iupac_name)
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
    """IUPAC 批次解析失败时静默降级，由文献摘要统一计数。"""
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
            print(f"   [IUPAC {code}] {source_label or '未知文献'} | {display_id}")

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
                "iupac_name": _stable_unique(names)[0],
            }
        elif len(canonical_map) > 1:
            stats["conflicts"] += 1
            note_rejection(target_key, "CANONICAL_CONFLICT")

    for target_key in explicit_names:
        explicit_names[target_key] = _stable_unique(explicit_names[target_key])

    label = source_label or "未知文献"
    print(
        f"[*] [{label}] IUPAC 摘要：目标={len(targets)}，"
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
    """安全剥离 UniParser 注释，并区分完整结构与需要推理的结构证据。"""
    raw_text = clean_excel_text(raw_core)
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
        # `<sep>` 本身只是分隔符；但无法识别的尾部不能被当作普通注释丢弃。
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
        print(
            "   [UniParser molecule 分类] "
            f"IMAGE 完整结构={classification_counts['complete']}，"
            f"马库什={classification_counts['markush']}，"
            f"路线={classification_counts['route']}，"
            f"未知注释/无效="
            f"{classification_counts['annotated_unknown'] + classification_counts['invalid']}。"
        )

    accepted = {}
    for target_key, structures in candidates.items():
        if len(structures) == 1:
            accepted[target_key] = next(iter(structures))
        elif len(structures) > 1:
            print(
                f"   [IMAGE 冲突] {targets[target_key]['display_id']} "
                f"存在 {len(structures)} 个不同完整结构，已拒绝该来源。"
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
    print(
        f"   [Figure 筛选] 待逐张检查 {len(figure_records)} 张去重图片，"
        f"目标 ID 共 {len(unresolved_keys)} 个。"
    )
    relevant_figures = []
    for figure_index, figure_record in enumerate(figure_records, start=1):
        if GLOBAL_STOP_EVENT.is_set():
            print("[-] Gemini 已停止，保留此前通过筛选的 Figure 并结束本阶段。")
            break
        prompt = PROMPT_FIGURE_FILTER.format(
            target_ids=target_ids,
            page=figure_record.get("page", "未知"),
            caption=figure_record.get("caption", "") or "无",
        )
        response_text = call_figure_filter(prompt, figure_record)
        is_relevant = _parse_figure_filter_response(response_text)
        if is_relevant is None:
            print(
                f"[-] Figure {figure_index}（页码 {figure_record.get('page', '未知')}）"
                "未返回严格布尔筛选结果，已跳过该图并继续。"
            )
            continue
        if is_relevant:
            relevant_figures.append(figure_record)
            print(
                f"   [Figure 保留] {figure_index}/{len(figure_records)} | "
                f"页码 {figure_record.get('page', '未知')}"
            )

    print(
        f"   [Figure 筛选] 共保留 {len(relevant_figures)}/{len(figure_records)} 张，"
        "只作为 INFERENCE 的辅助证据。"
    )
    return relevant_figures


def infer_unresolved_structures(document, targets, unresolved_keys, source_label=""):
    if not unresolved_keys:
        return {}

    selected_evidence = select_structure_evidence_records(
        document, targets, unresolved_keys
    )
    if selected_evidence is None:
        print(
            "[-] 结构低思考证据召回返回无效 JSON；"
            "保留当前已有结果，其余目标按未解决状态落盘。"
        )
        return {}
    relevant_figures = screen_relevant_figures(document, targets, unresolved_keys)
    if relevant_figures is None:
        relevant_figures = []
    candidate_sets = {key: set() for key in unresolved_keys}
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
            print(
                f"   [INFERENCE {code}] {source_label or '未知文献'} | "
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
            raw_smiles = clean_excel_text(item.get("repaired_smiles", ""))
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
        parsed = _parse_json_array(response_text)
        if parsed is None:
            print(
                "[-] 汇总结构推断响应无法解析；"
                "不重复调用，剩余目标按未解决状态落盘。"
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
    print(
        f"[*] [{source_label or '未知文献'}] INFERENCE 摘要："
        f"目标={len(unresolved_keys)}，调用={stats['calls']}，接受={len(accepted)}，"
        f"模型不确定={stats['not_confident']}，无证据={stats['no_evidence']}，"
        f"模型遗漏={stats['omitted']}，冲突={stats['conflicts']}。"
    )
    return accepted


def resolve_document_structures(document, targets, initial_dictionary, source_label=""):
    if not targets:
        return {}

    dictionary_candidates = dict(initial_dictionary)
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
        print(
            f"[*] [{source_label or '未知文献'}] IUPAC 摘要：目标=0，"
            "低筛调用=0，高思考调用=0，输入字符=0，接受=0，"
            "冲突=0，未解决=0，无效JSON=0。"
        )

    results = {}
    for key in targets:
        if key in dictionary_candidates:
            results[key] = {
                "smiles": dictionary_candidates[key],
                "source": "DICTIONARY",
                "iupac_name": "",
                "smiles_style": None,
            }
            continue
        image_smiles = image_candidates.get(key)
        if image_smiles:
            results[key] = {
                "smiles": image_smiles,
                "source": "IMAGE",
                "iupac_name": "",
                "smiles_style": None,
            }
            continue
        iupac_item = iupac_candidates.get(key)
        if iupac_item:
            results[key] = {
                "smiles": iupac_item["smiles"],
                "source": "IUPAC",
                "iupac_name": iupac_item["iupac_name"],
                "smiles_style": None,
            }

    inference_keys = [key for key in targets if key not in results]
    inference_candidates = infer_unresolved_structures(
        document, targets, inference_keys, source_label=source_label
    )
    if inference_candidates is None:
        print(
            f"[-] [{source_label or '未知文献'}] 结构推断阶段未返回结果；"
            "保留前序结果，其余目标按未解决状态落盘。"
        )
        inference_candidates = {}
    for key, smiles in inference_candidates.items():
        results[key] = {
            "smiles": smiles,
            "source": "INFERENCE",
            "iupac_name": "",
            "smiles_style": None,
        }
    return results


# ================= 6. 单篇 PDF 核验与 Excel 写入 =================
def locate_pdf_path(pdf_filename):
    """优先返回 PDF_DIR 文件；仅主目录缺失时使用 processed-loss。"""
    primary_path = os.path.join(PDF_DIR, pdf_filename)
    if os.path.isfile(primary_path):
        return primary_path, "PRIMARY"

    fallback_path = os.path.join(FALLBACK_PDF_DIR, pdf_filename)
    if os.path.isfile(fallback_path):
        return fallback_path, "FALLBACK"
    return None, None


def _mark_preserved_metal(row):
    if contains_metal(clean_excel_text(row.get("SMILES", ""))):
        row["_smiles_style"] = "RED"


def process_single_pdf_task(pdf_filename, rows_data):
    print("\n" + "=" * 60)
    print(f"[*] 正在核验: {pdf_filename}")
    pdf_path, pdf_location = locate_pdf_path(pdf_filename)

    if pdf_path is None:
        print(
            f"[-] 在 {PDF_DIR} 和 {FALLBACK_PDF_DIR} 均找不到 PDF，"
            "保留原内容并将文件名标蓝。"
        )
        for row in rows_data:
            row["_filename_style"] = "BLUE"
            _mark_preserved_metal(row)
        return rows_data
    if pdf_location == "FALLBACK":
        print(
            f"[*] {PDF_DIR} 中未找到该文献，已改用备用目录 "
            f"{FALLBACK_PDF_DIR}。"
        )

    targets = build_targets(rows_data)
    initial_dictionary = collect_dictionary_candidates(targets)

    if targets:
        print(
            f"[*] 目标 ID 共 {len(targets)} 个；首次词典命中 "
            f"{len(initial_dictionary)} 个。"
        )
        if len(initial_dictionary) == len(targets):
            print(f"[*] [{pdf_filename}] 全部目标由离线词典解决，跳过 UniParser 与 Gemini。")
            print(
                f"[*] [{pdf_filename}] IUPAC 摘要：目标=0，"
                "低筛调用=0，高思考调用=0，输入字符=0，接受=0，"
                "冲突=0，未解决=0，无效JSON=0。"
            )
            results = {
                key: {
                    "smiles": initial_dictionary[key],
                    "source": "DICTIONARY",
                    "iupac_name": "",
                    "smiles_style": None,
                }
                for key in targets
            }
        else:
            document = call_uniparser_extract(pdf_path)
            if document is None:
                if GLOBAL_STOP_EVENT.is_set():
                    return None
                print(f"[-] {pdf_filename} 文档解析失败，本次不落盘。")
                return PDF_PROCESS_FAILED
            results = resolve_document_structures(
                document, targets, initial_dictionary, source_label=pdf_filename
            )
            if results is None:
                print(
                    f"[-] {pdf_filename} 模型阶段未返回结果；"
                    "本篇按未解决状态落盘，后续不自动重跑。"
                )
                results = {}
    else:
        print("[*] 当前 PDF 没有有效 Compound ID，不调用文献解析 API。")
        results = {}

    counts = {
        "DICTIONARY": 0, "IUPAC": 0, "IMAGE": 0,
        "INFERENCE": 0, "UNRESOLVED": 0,
    }
    for row in rows_data:
        target_key = normalize_compound_identifier(row.get("Compound ID", ""))
        resolution = results.get(target_key) if target_key else None
        if resolution:
            old_raw = clean_excel_text(row.get("SMILES", ""))
            old_canonical = standardize_smiles(old_raw)
            new_canonical = resolution["smiles"]
            stereo_compatible = smiles_stereo_compatible_for_coloring(
                old_raw, new_canonical
            )
            if old_canonical != new_canonical:
                row["SMILES"] = new_canonical
            row["SMILES Source"] = resolution["source"]
            if resolution.get("iupac_name"):
                row["IUPAC Name"] = resolution["iupac_name"]
            row["_smiles_style"] = resolution.get("smiles_style")
            # 仍写入更完整的立体 SMILES；仅当连接关系或明确立体信息冲突时标粉色。
            if resolution["source"] == "INFERENCE" and not stereo_compatible:
                row["_smiles_style"] = "PINK"
            counts[resolution["source"]] += 1
        else:
            row["_smiles_style"] = "PURPLE"
            counts["UNRESOLVED"] += 1

        if contains_metal(clean_excel_text(row.get("SMILES", ""))):
            row["_smiles_style"] = "RED"

    print(
        f"[✓] {pdf_filename} 核验完成："
        f"DICTIONARY={counts['DICTIONARY']}，IUPAC={counts['IUPAC']}，"
        f"IMAGE={counts['IMAGE']}，INFERENCE={counts['INFERENCE']}，"
        f"未确定={counts['UNRESOLVED']}。"
    )
    return rows_data


def _atomic_save_workbook(workbook, output_path):
    """先完整写入同目录临时文件，再原子替换最终断点文件。"""
    target_path = os.path.abspath(output_path)
    target_dir = os.path.dirname(target_path)
    os.makedirs(target_dir, exist_ok=True)
    file_descriptor, temp_path = tempfile.mkstemp(
        prefix=".repair_commit_",
        suffix=".xlsx",
        dir=target_dir,
    )
    os.close(file_descriptor)
    try:
        workbook.save(temp_path)
        os.replace(temp_path, target_path)
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


def worker_routine(chunk_idx, task_dict, sheet_name, output_headers):
    safe_sheet = "".join(c for c in sheet_name if c.isalnum() or c in " _-")
    output_path = os.path.join(OUTPUT_DIR, f"Repaired_{safe_sheet}_Part_{chunk_idx}.xlsx")
    expected_headers = _stable_unique(output_headers)

    if os.path.exists(output_path):
        wb = load_workbook(output_path)
        ws = wb.active
        headers = [
            ws.cell(row=1, column=column).value
            for column in range(1, ws.max_column + 1)
        ]
        while headers and headers[-1] in (None, ""):
            headers.pop()
        for header in expected_headers:
            if header not in headers:
                headers.append(header)
                ws.cell(row=1, column=len(headers), value=header)
    else:
        wb = Workbook()
        ws = wb.active
        ws.title = "Repaired_Results"
        headers = expected_headers
        ws.append(headers)

    fills = {
        "YELLOW": PALE_YELLOW_FILL,
        "PINK": PINK_FILL,
        "PURPLE": PALE_PURPLE_FILL,
        "RED": RED_FILL,
    }
    for pdf_name, rows in task_dict.items():
        if GLOBAL_STOP_EVENT.is_set():
            break
        processed_rows = process_single_pdf_task(pdf_name, rows)
        if processed_rows is None:
            break
        if processed_rows is PDF_PROCESS_FAILED:
            continue

        commit_key = f"repair|{sheet_name}|{chunk_idx}|{pdf_name}"
        if not _begin_final_commit(commit_key):
            print(
                f"[!] [{pdf_name}] 中断已生效，本篇尚未进入最终落盘，"
                "本轮结果已放弃，下次重新处理。"
            )
            break

        try:
            for row in processed_rows:
                filename_style = row.pop("_filename_style", None)
                smiles_style = row.pop("_smiles_style", None)
                ws.append([row.get(column, "") if column else "" for column in headers])
                row_index = ws.max_row
                if filename_style == "BLUE":
                    try:
                        column = headers.index("Source Filename") + 1
                        ws.cell(row=row_index, column=column).fill = BLUE_FILL
                    except ValueError:
                        pass
                fill = fills.get(smiles_style)
                if fill is not None:
                    try:
                        column = headers.index("SMILES") + 1
                        ws.cell(row=row_index, column=column).fill = fill
                    except ValueError:
                        pass

            if headers and processed_rows:
                ws.append([])
            _atomic_save_workbook(wb, output_path)

            global PROCESSED_PDF_COUNT
            with COUNT_LOCK:
                PROCESSED_PDF_COUNT += 1
        finally:
            _end_final_commit(commit_key)

    print(f"\n[+] 线程 {chunk_idx} 在工作表 [{sheet_name}] 的任务已安全退出。")



# ================= 5. 主函数 =================
def main():
    global PROCESSED_PDF_COUNT

    GLOBAL_STOP_EVENT.clear()
    with FINAL_COMMIT_CONDITION:
        ACTIVE_FINAL_COMMITS.clear()
    PROCESSED_PDF_COUNT = 0
    print(">>> 开始执行文献 SMILES 核验工具（支持断点续传与全文结构解析）<<<")

    # 先完成所有本地检查；没有待处理任务时，不调用任何 API。
    excel_files = sorted(glob.glob(os.path.join(EXCEL_DIR, "*.xlsx")))
    if not excel_files:
        print(f"错误: 在 {EXCEL_DIR} 未找到 Excel 文件。")
        return

    target_excel = excel_files[0]
    print(f"[*] 发现目标 Excel: {target_excel}")

    try:
        all_dfs = pd.read_excel(target_excel, sheet_name=None, keep_default_na=False)
    except Exception as exc:
        print(f"[FATAL] 无法读取目标 Excel: {type(exc).__name__}: {exc}")
        return

    prepared_sheets = []
    required_columns = {"Source Filename", "Compound ID"}

    for sheet_name, df in all_dfs.items():
        missing_columns = sorted(required_columns - set(df.columns))
        if missing_columns:
            print(f"[-] 工作表 [{sheet_name}] 缺少必要列 {missing_columns}，已跳过。")
            continue

        processed_pdfs = set()
        safe_sheet = "".join(c for c in sheet_name if c.isalnum() or c in " _-")
        out_files = glob.glob(os.path.join(OUTPUT_DIR, f"Repaired_{safe_sheet}_Part_*.xlsx"))
        for out_file in out_files:
            try:
                df_out = pd.read_excel(out_file)
                if "Source Filename" in df_out.columns:
                    processed_pdfs.update(
                        df_out["Source Filename"].dropna().astype(str).str.strip().tolist()
                    )
            except Exception as exc:
                print(f"[警告] 无法读取断点文件 {out_file}: {type(exc).__name__}: {exc}")

        pdf_tasks = {}
        for row in df.to_dict(orient="records"):
            pdf_name = str(row.get("Source Filename", "")).strip()
            if not pdf_name or pdf_name.lower() in {"nan", "none"}:
                continue
            if pdf_name in processed_pdfs:
                continue
            pdf_tasks.setdefault(pdf_name, []).append(row)

        if not pdf_tasks:
            print(f">>> 当前工作表 {sheet_name} 已全部处理完毕或没有有效任务。 <<<")
            continue

        output_headers = list(df.columns)
        for required_output_column in ("IUPAC Name", "SMILES", "SMILES Source"):
            if required_output_column not in output_headers:
                output_headers.append(required_output_column)

        prepared_sheets.append({
            "sheet_name": sheet_name,
            "pdf_tasks": pdf_tasks,
            "output_headers": output_headers,
            "processed_count": len(processed_pdfs),
        })

    if not prepared_sheets:
        print(">>> 没有待处理任务，未调用任何 API。 <<<")
        return

    total_unprocessed = sum(
        len(sheet_spec["pdf_tasks"])
        for sheet_spec in prepared_sheets
    )

    try:
        run_limit = 0 if MAX_PDFS_PER_RUN is None else int(MAX_PDFS_PER_RUN)
    except (TypeError, ValueError):
        print(f"[FATAL] MAX_PDFS_PER_RUN 必须是非负整数、0 或 None，当前值: {MAX_PDFS_PER_RUN!r}")
        return

    if run_limit < 0:
        print(f"[FATAL] MAX_PDFS_PER_RUN 不能小于 0，当前值: {run_limit}")
        return

    if run_limit > 0:
        remaining_slots = run_limit
        limited_sheets = []

        # 此处的任务已经完成断点过滤，只从真正未处理的 PDF 中选择。
        for sheet_spec in prepared_sheets:
            if remaining_slots <= 0:
                break

            selected_tasks = {}
            for pdf_name, rows in sheet_spec["pdf_tasks"].items():
                if remaining_slots <= 0:
                    break
                selected_tasks[pdf_name] = rows
                remaining_slots -= 1

            if selected_tasks:
                limited_spec = dict(sheet_spec)
                limited_spec["pdf_tasks"] = selected_tasks
                limited_sheets.append(limited_spec)

        prepared_sheets = limited_sheets
        selected_count = run_limit - remaining_slots
        print(
            f"[*] 本次运行上限: {run_limit} 篇；"
            f"断点过滤后共有 {total_unprocessed} 篇未处理文献，实际选择 {selected_count} 篇。"
        )
    else:
        print(f"[*] 本次运行不限制篇数；断点过滤后共有 {total_unprocessed} 篇未处理文献。")

    # 仅根据本次实际选择的未处理任务判断是否需要在线 API。
    requires_online_api = any(
        locate_pdf_path(pdf_name)[0] is not None
        and rows_require_online_api(rows)
        for sheet_spec in prepared_sheets
        for pdf_name, rows in sheet_spec["pdf_tasks"].items()
    )

    if requires_online_api:
        # UniParser health 不提交文档；先检查它，失败时可避免 Gemini 的最小生成调用。
        if not test_uniparser_connection():
            print(">>> 双 API 前置检查未通过，任务已终止，未提交任何 PDF。 <<<")
            return
        if not test_gemini_connection():
            print(">>> 双 API 前置检查未通过，任务已终止，未提交任何 PDF。 <<<")
            return
        print("[+] 双 API 前置检查全部通过，开始正式处理。")
    else:
        print("[*] 待处理任务仅涉及词典、无有效 ID 或缺失 PDF，无需调用在线 API。")

    # 各 Sheet 依次处理，每个 Sheet 内部使用线程池并发。
    for sheet_spec in prepared_sheets:
        if GLOBAL_STOP_EVENT.is_set():
            break

        sheet_name = sheet_spec["sheet_name"]
        pdf_tasks = sheet_spec["pdf_tasks"]
        output_headers = sheet_spec["output_headers"]
        processed_count = sheet_spec["processed_count"]
        pdf_names = list(pdf_tasks.keys())

        print("\n" + "=" * 60)
        print(f"[*] 🚀 开始处理工作表 (Sheet): {sheet_name}")
        print("=" * 60)
        if processed_count:
            print(f"[*] 断点续传：发现 {processed_count} 篇已处理文献，已自动跳过。")
        print(f"[*] 当前工作表有 {len(pdf_names)} 篇剩余文献任务。")

        num_workers = min(MAX_WORKERS, len(pdf_names))
        chunks = {worker_id: {} for worker_id in range(num_workers)}
        for index, pdf_name in enumerate(pdf_names):
            worker_id = index % num_workers
            chunks[worker_id][pdf_name] = pdf_tasks[pdf_name]

        executor = ThreadPoolExecutor(max_workers=num_workers)
        futures = [
            executor.submit(
                worker_routine,
                worker_id,
                chunks[worker_id],
                sheet_name,
                output_headers
            )
            for worker_id in range(num_workers)
        ]

        try:
            while futures:
                done, pending = concurrent.futures.wait(futures, timeout=1.0)
                futures = list(pending)
                for future in done:
                    exc = future.exception()
                    if exc:
                        print(f"\n[FATAL] 🚨 线程内部发生严重异常，已崩溃退出: {exc}")
                        import traceback
                        traceback.print_exception(type(exc), exc, exc.__traceback__)
                        _request_global_stop()
                if GLOBAL_STOP_EVENT.is_set():
                    for future in futures:
                        future.cancel()
                    break
        except KeyboardInterrupt:
            print("\n[!] 🚨 收到 Ctrl+C 手动中断指令！", flush=True)
            print(
                "[!] 已停止新落盘；处理一半的文献将放弃。"
                "仅等待已开始的最终提交，最多 10 秒...",
                flush=True,
            )
            _request_global_stop()
            for future in futures:
                future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            _force_exit_after_commit_grace("Ctrl+C", 130)
        else:
            executor.shutdown(wait=False, cancel_futures=True)
            if GLOBAL_STOP_EVENT.is_set():
                print(
                    "[!] 全局停止已生效：取消未开始任务，"
                    "仅等待已开始的最终提交，最多 10 秒。",
                    flush=True,
                )
                _force_exit_after_commit_grace("全局停止", 1)

    if GLOBAL_STOP_EVENT.is_set():
        print("\n>>> ⚠️ 任务已安全中止！<<<")
        print(">>> 原因: API 额度、鉴权、运行错误，或接收到人工 Ctrl+C 中断。")
    else:
        print("\n>>> 🎉 全部工作表的文献结构核验已完成！<<<")

    print(f"\n>>> 📊 【本次运行统计】：共成功处理了 {PROCESSED_PDF_COUNT} 篇文献。")
    print(">>> 现有处理进度已全部安全落盘，下次启动将自动断点续传。请查看 OUTPUT_EXCEL_DIR 目录。\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[!] 🚨 收到 Ctrl+C 手动中断指令！", flush=True)
        print(
            "[!] 已停止新落盘；处理一半的文献将放弃。"
            "仅等待已开始的最终提交，最多 10 秒...",
            flush=True,
        )
        _request_global_stop()
        _force_exit_after_commit_grace("Ctrl+C", 130)
