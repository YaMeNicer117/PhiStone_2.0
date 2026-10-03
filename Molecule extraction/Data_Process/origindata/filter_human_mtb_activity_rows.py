"""Filter human M. tuberculosis rows and recalculate Activity Level.

The script scans every worksheet in every direct-child XLSX file under
``tian_datas``. A row is retained when its complete ``Target Strain`` value,
after Unicode/whitespace/case normalization, occurs in
``hazard_A_target_strains.txt``.

Each retained row is classified independently. The existing Activity Level
has the highest priority. Empty or unclassified activities are separated
into their own respective workbooks to avoid Compound ID numbering.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
import tempfile
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    from openpyxl import Workbook, load_workbook
except ImportError:  # Report a short, actionable error from main().
    Workbook = None
    load_workbook = None


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = SCRIPT_DIR
DEFAULT_STRAIN_FILE = SCRIPT_DIR / "hazard_A_target_strains.txt"
DEFAULT_OUTPUT_FILE = SCRIPT_DIR / "human_mtb_activity_reclassified.xlsx"

DEFAULT_HEADER_SCAN_ROWS = 50
MAX_HEADER_COLUMNS = 256
DEFAULT_MAX_SHEET_ROWS = 20_000

TARGET_STRAIN_KEY = "target strain"
ACTIVITY_LEVEL_KEY = "activity level"

ACTIVITY_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "mic_value": ("mic value", "mic"),
    "mic_unit": ("mic unit",),
    "ic50_value": ("ic50 value", "ic50"),
    "ic50_unit": ("ic50 unit",),
    "inhibition_value": ("% inhibition", "inhibition", "inhibition value"),
    "pic50_value": ("pic50", "pic50 value"),
    "pmic_value": ("pmic", "pmic value"),
    "zoi_value": ("zoi (mm)", "zoi", "zoi value"),
    "activity_level": (ACTIVITY_LEVEL_KEY,),
}

# 变更为全小写关键词，用于子串包含匹配
QUALITATIVE_INACTIVE_KEYWORDS = {
    "--",
    "na",
    "n/a",
    "n.a",
    "not determined",
    "not detected",
    "inactive",
    "not active",
    "no activity",
    "no inhibition",
    "poor",
    "resistant",
    "ns",
    "weak",
    "NMA",
    "low activity",
    "no effective",
    "less active",
    "lack",
    "loss of activity",
    "ineffective"
}


@dataclass(frozen=True)
class SheetSpec:
    """Information required to reread one source worksheet."""

    workbook_path: Path
    sheet_name: str
    header_row: int
    column_by_key: dict[str, int]


@dataclass
class RunStats:
    source_workbooks: int = 0
    readable_workbooks: int = 0
    failed_workbooks: int = 0
    source_sheets: int = 0
    recognized_sheets: int = 0
    skipped_sheets: int = 0
    failed_sheets: int = 0
    scanned_data_rows: int = 0
    matched_rows: int = 0
    empty_rows: int = 0
    unclassified_rows: int = 0
    output_sheets: int = 0
    level_counts: Counter[int] = field(default_factory=Counter)
    metric_counts: Counter[str] = field(default_factory=Counter)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "按菌株 TXT 筛选 tian_datas 中所有 XLSX 行，并逐行重算数值型 Activity Level。"
            "原始级别的优先级最高。活性为空和无法分类的数据将被独立拆分出输出。"
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"源 XLSX 目录（默认：{DEFAULT_INPUT_DIR}）",
    )
    parser.add_argument(
        "--strain-file",
        type=Path,
        default=DEFAULT_STRAIN_FILE,
        help=f"允许的 Target Strain TXT（默认：{DEFAULT_STRAIN_FILE}）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help=f"主输出 XLSX（默认：{DEFAULT_OUTPUT_FILE}）",
    )
    parser.add_argument(
        "--header-scan-rows",
        type=int,
        default=DEFAULT_HEADER_SCAN_ROWS,
        help=f"每个 sheet 前多少行内查找表头（默认：{DEFAULT_HEADER_SCAN_ROWS}）",
    )
    parser.add_argument(
        "--max-rows-per-sheet",
        type=int,
        default=DEFAULT_MAX_SHEET_ROWS,
        help=(
            "每个输出 sheet 的最大总行数，包含表头 "
            f"（默认：{DEFAULT_MAX_SHEET_ROWS}）"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="允许覆盖已经存在的输出 XLSX。",
    )
    return parser.parse_args()


def normalize_header(value: Any) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value)).replace("\ufeff", "")
    return " ".join(text.split()).casefold()


def normalize_strain(value: Any) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value)).replace("\ufeff", "")
    return " ".join(text.split()).casefold()


def display_header(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).replace("\ufeff", "").split())


def get_first_value(row: Mapping[str, Any], aliases: Sequence[str]) -> Any:
    for alias in aliases:
        key = normalize_header(alias)
        value = row.get(key)
        if value is not None and str(value).strip() != "":
            return value
    return None


def parse_numeric(value: Any, preserve_inequality: bool = True) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None

    text = unicodedata.normalize("NFKC", str(value)).strip()
    if not text:
        return None

    text = (
        text.replace("≥", ">=")
        .replace("≤", "<=")
        .replace("＞", ">")
        .replace("＜", "<")
        .replace("～", "~")
    )
    
    text = re.split(r"±|\+/-", text)[0].strip()

    offset = 0.0
    if preserve_inequality:
        if ">" in text:
            offset = 0.0001
        elif "<" in text:
            offset = -0.0001

    cleaned = re.sub(r"[<>=~]", "", text).strip()
    cleaned = cleaned.removesuffix("%").strip()

    number_pattern = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
    range_match = re.fullmatch(
        rf"({number_pattern})\s*[-–—]\s*({number_pattern})", cleaned
    )
    try:
        if range_match:
            left = float(range_match.group(1))
            right = float(range_match.group(2))
            number = (left + right) / 2.0
        elif re.fullmatch(number_pattern, cleaned):
            number = float(cleaned)
        else:
            return None
    except (TypeError, ValueError, OverflowError):
        return None

    number += offset
    return number if math.isfinite(number) else None


def concentration_thresholds(unit_value: Any) -> tuple[float, float, float]:
    excellent, good, moderate = 1.0, 10.0, 64.0
    if unit_value is None or str(unit_value).strip() == "":
        return excellent, good, moderate

    unit = unicodedata.normalize("NFKC", str(unit_value)).replace(" ", "")
    if re.search(r"[munuμµ]?g/", unit, re.IGNORECASE):
        return excellent, good, moderate
    if re.search(r"nM\b", unit, re.IGNORECASE):
        return 2000.0, 20000.0, 100000.0
    if re.search(r"[uμµ]M\b", unit, re.IGNORECASE):
        return 2.0, 20.0, 100.0
    if re.search(r"mM\b", unit, re.IGNORECASE):
        return 0.002, 0.02, 0.1
    if re.search(r"(?<![munuμµ])M\b", unit, re.IGNORECASE):
        return 0.000002, 0.00002, 0.0001
    return excellent, good, moderate


def contains_qualitative_inactive(values: Iterable[Any]) -> bool:
    for value in values:
        if value is None:
            continue
        normalized = " ".join(
            unicodedata.normalize("NFKC", str(value)).strip().casefold().split()
        )
        # 变更为只要包含了集合中的任意一个词，即视为触发
        for keyword in QUALITATIVE_INACTIVE_KEYWORDS:
            if keyword in normalized:
                return True
    return False


def map_existing_level(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        if math.isfinite(number) and number.is_integer() and 0 <= number <= 3:
            return int(number)

    text = unicodedata.normalize("NFKC", str(value)).strip().casefold()
    if not text:
        return None
    if "excellent" in text:
        return 3
    if "good" in text:
        return 2
    if "moderate" in text:
        return 1
    if "inactive" in text:
        return 0
    if text in {"0", "1", "2", "3"}:
        return int(text)
    return None


def classify_activity(row: Mapping[str, Any]) -> tuple[int | None, str]:
    values = {
        name: get_first_value(row, aliases)
        for name, aliases in ACTIVITY_FIELD_ALIASES.items()
    }

    existing = map_existing_level(values["activity_level"])
    if existing is not None:
        return existing, "existing_level_first"

    if all(val is None for val in values.values()):
        return None, "empty"

    for value_key, unit_key, metric_name in (
        ("mic_value", "mic_unit", "mic"),
        ("ic50_value", "ic50_unit", "ic50"),
    ):
        number = parse_numeric(values[value_key], preserve_inequality=True)
        if number is None:
            continue
        excellent, good, moderate = concentration_thresholds(values[unit_key])
        if number <= excellent:
            return 3, metric_name
        if number <= good:
            return 2, metric_name
        if number <= moderate:
            return 1, metric_name
        return 0, metric_name

    inhibition = parse_numeric(
        values["inhibition_value"], preserve_inequality=False
    )
    if inhibition is not None:
        if inhibition >= 90.0:
            return 3, "inhibition"
        if inhibition >= 80.0:
            return 2, "inhibition"
        if inhibition >= 20.0:
            return 1, "inhibition"
        return 0, "inhibition"

    pic50 = parse_numeric(values["pic50_value"], preserve_inequality=True)
    if pic50 is not None:
        if pic50 >= 6.0:
            return 3, "pic50"
        if pic50 >= 5.0:
            return 2, "pic50"
        if pic50 >= 4.2:
            return 1, "pic50"
        return 0, "pic50"

    pmic = parse_numeric(values["pmic_value"], preserve_inequality=True)
    if pmic is not None:
        if pmic >= 6.0:
            return 3, "pmic"
        if pmic >= 5.0:
            return 2, "pmic"
        if pmic >= 4.2:
            return 1, "pmic"
        return 0, "pmic"

    zoi = parse_numeric(values["zoi_value"], preserve_inequality=False)
    if zoi is not None:
        if zoi >= 25.0:
            return 2, "zoi"
        if zoi >= 15.0:
            return 1, "zoi"
        return 0, "zoi"

    measurement_values = [
        values["mic_value"],
        values["ic50_value"],
        values["inhibition_value"],
        values["pic50_value"],
        values["pmic_value"],
        values["zoi_value"],
    ]
    if contains_qualitative_inactive(measurement_values):
        return 0, "qualitative_inactive"

    return None, "unclassified"


def load_allowed_strains(path: Path) -> tuple[set[str], int]:
    strains: set[str] = set()
    nonempty_lines = 0
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            nonempty_lines += 1
            normalized = normalize_strain(line)
            if normalized:
                strains.add(normalized)
    return strains, nonempty_lines


def list_source_workbooks(input_dir: Path, output_path: Path) -> list[Path]:
    output_resolved = output_path.resolve()
    return sorted(
        (
            path.resolve()
            for path in input_dir.glob("*.xlsx")
            if path.is_file()
            and not path.name.startswith("~$")
            and path.resolve() != output_resolved
        ),
        key=lambda path: path.name.casefold(),
    )


def find_header(worksheet: Any, scan_rows: int) -> tuple[int, dict[str, int], list[tuple[str, str]]] | None:
    activity_keys = {
        normalize_header(alias)
        for aliases in ACTIVITY_FIELD_ALIASES.values()
        for alias in aliases
    }
    for row_number, row in enumerate(
        worksheet.iter_rows(
            min_row=1,
            max_row=scan_rows,
            max_col=MAX_HEADER_COLUMNS,
            values_only=True,
        ),
        start=1,
    ):
        keys = [normalize_header(value) for value in row]
        if TARGET_STRAIN_KEY not in keys:
            continue
        if not any(key in activity_keys for key in keys):
            continue

        column_by_key: dict[str, int] = {}
        ordered_headers: list[tuple[str, str]] = []
        for column_index, value in enumerate(row):
            key = normalize_header(value)
            if not key or key in column_by_key:
                continue
            column_by_key[key] = column_index
            ordered_headers.append((key, display_header(value)))
        return row_number, column_by_key, ordered_headers
    return None


def discover_sheets(
    workbook_paths: Sequence[Path],
    header_scan_rows: int,
    stats: RunStats,
) -> tuple[dict[Path, list[SheetSpec]], dict[str, str]]:
    specs_by_workbook: dict[Path, list[SheetSpec]] = defaultdict(list)
    output_headers: dict[str, str] = {}

    for workbook_path in workbook_paths:
        try:
            workbook = load_workbook(
                filename=workbook_path,
                read_only=True,
                data_only=True,
            )
        except Exception as exc:
            stats.failed_workbooks += 1
            print(
                f"[错误] 无法读取工作簿 {workbook_path.name}：{exc}",
                file=sys.stderr,
            )
            continue

        stats.readable_workbooks += 1
        try:
            for worksheet in workbook.worksheets:
                stats.source_sheets += 1
                try:
                    header = find_header(worksheet, header_scan_rows)
                except Exception as exc:
                    stats.failed_sheets += 1
                    print(
                        f"[错误] 检查 {workbook_path.name} / {worksheet.title} 失败：{exc}",
                        file=sys.stderr,
                    )
                    continue

                if header is None:
                    stats.skipped_sheets += 1
                    print(
                        f"[警告] 跳过 {workbook_path.name} / {worksheet.title}："
                        f"前 {header_scan_rows} 行未找到 Target Strain 及活性字段表头。",
                        file=sys.stderr,
                    )
                    continue

                header_row, column_by_key, ordered_headers = header
                stats.recognized_sheets += 1
                specs_by_workbook[workbook_path].append(
                    SheetSpec(
                        workbook_path=workbook_path,
                        sheet_name=worksheet.title,
                        header_row=header_row,
                        column_by_key=column_by_key,
                    )
                )
                for key, header_text in ordered_headers:
                    output_headers.setdefault(key, header_text)
        finally:
            workbook.close()

    if ACTIVITY_LEVEL_KEY in output_headers:
        output_headers[ACTIVITY_LEVEL_KEY] = "Activity Level"
    else:
        output_headers[ACTIVITY_LEVEL_KEY] = "Activity Level"
    return specs_by_workbook, output_headers


class StreamingOutput:
    """Write a plain XLSX while enforcing a total-row limit per sheet."""
    def __init__(
        self,
        output_headers: Mapping[str, str],
        max_rows_per_sheet: int,
    ) -> None:
        self.workbook = Workbook(write_only=True)
        self.header_keys = list(output_headers.keys())
        self.header_values = [output_headers[key] for key in self.header_keys]
        self.max_rows_per_sheet = max_rows_per_sheet
        self.current_sheet: Any | None = None
        self.current_total_rows = 0
        self.sheet_count = 0
        self.sheet_row_counts: list[int] = []

    def _finish_current_sheet(self) -> None:
        if self.current_sheet is None:
            return
        self.sheet_row_counts.append(self.current_total_rows)

    def _start_sheet(self) -> None:
        self._finish_current_sheet()
        self.sheet_count += 1
        worksheet = self.workbook.create_sheet(f"Results_{self.sheet_count:03d}")
        
        # 仅追加默认格式表头
        worksheet.append(self.header_values)

        self.current_sheet = worksheet
        self.current_total_rows = 1

    def append(self, values_by_key: Mapping[str, Any]) -> None:
        if (
            self.current_sheet is None
            or self.current_total_rows >= self.max_rows_per_sheet
        ):
            self._start_sheet()

        row_cells = [values_by_key.get(key) for key in self.header_keys]
        self.current_sheet.append(row_cells)
        self.current_total_rows += 1

    def append_blank_row(self) -> None:
        """为不同文献记录插入一个空行以作区分"""
        if self.current_sheet is None:
            return
        if self.current_total_rows >= self.max_rows_per_sheet:
            self._start_sheet()
        
        self.current_sheet.append([None] * len(self.header_keys))
        self.current_total_rows += 1

    def ensure_sheet(self) -> None:
        if self.current_sheet is None:
            self._start_sheet()

    def save(self, output_path: Path) -> None:
        self.ensure_sheet()
        self._finish_current_sheet()

        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{output_path.stem}_",
            suffix=".tmp.xlsx",
            dir=output_path.parent,
        )
        os.close(file_descriptor)
        temporary_path = Path(temporary_name)
        try:
            self.workbook.save(temporary_path)
            os.replace(temporary_path, output_path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()


def process_sources(
    specs_by_workbook: Mapping[Path, Sequence[SheetSpec]],
    allowed_strains: set[str],
    output_headers: Mapping[str, str],
    max_rows_per_sheet: int,
    stats: RunStats,
) -> tuple[StreamingOutput, StreamingOutput, dict[str, set[str]]]:
    writer_valid = StreamingOutput(output_headers, max_rows_per_sheet)
    
    # 为 unclassified 数据扩展特殊的表头列
    unclass_headers = dict(output_headers)
    unclass_headers["_source_location"] = "Source Excel & Sheet"
    writer_unclass = StreamingOutput(unclass_headers, max_rows_per_sheet)
    
    empty_records: dict[str, set[str]] = defaultdict(set)
    
    # 用于追踪无法分类数据来源的变化
    previous_unclass_workbook: Path | None = None

    for workbook_path in sorted(
        specs_by_workbook, key=lambda path: path.name.casefold()
    ):
        specs = specs_by_workbook[workbook_path]
        try:
            workbook = load_workbook(
                filename=workbook_path,
                read_only=True,
                data_only=True,
            )
        except Exception as exc:
            stats.failed_workbooks += 1
            print(
                f"[错误] 第二次读取工作簿 {workbook_path.name} 失败：{exc}",
                file=sys.stderr,
            )
            continue

        unclass_appended_this_wb = False

        try:
            for spec in specs:
                try:
                    worksheet = workbook[spec.sheet_name]
                    for row in worksheet.iter_rows(
                        min_row=spec.header_row + 1,
                        values_only=True,
                    ):
                        stats.scanned_data_rows += 1
                        target_index = spec.column_by_key[TARGET_STRAIN_KEY]
                        target_value = (
                            row[target_index] if target_index < len(row) else None
                        )
                        if normalize_strain(target_value) not in allowed_strains:
                            continue

                        values_by_key = {
                            key: row[column_index]
                            if column_index < len(row)
                            else None
                            for key, column_index in spec.column_by_key.items()
                        }
                        
                        smiles_key = next((k for k in values_by_key if "smiles" in k.lower()), None)
                        if smiles_key and values_by_key[smiles_key] is not None:
                            raw_smiles = str(values_by_key[smiles_key])
                            if "<sep>" in raw_smiles:
                                values_by_key[smiles_key] = raw_smiles.split("<sep>")[0].strip()

                        level, metric = classify_activity(values_by_key)

                        stats.matched_rows += 1
                        stats.metric_counts[metric] += 1
                        
                        is_empty = (level is None and metric == "empty")
                        is_unclassified = (level is None and metric != "empty")
                        
                        if is_empty:
                            stats.empty_rows += 1
                            
                            sm_val = values_by_key.get(smiles_key) if smiles_key else None
                            smiles_val = str(sm_val).strip() if sm_val is not None and str(sm_val).strip() else "Unknown_SMILES"
                                
                            source_key = next((k for k in values_by_key if "source filename" in k.lower() or "filename/doi" in k.lower() or "source" in k.lower()), None)
                            src_val = values_by_key.get(source_key) if source_key else None
                            source_val = str(src_val).strip() if src_val is not None and str(src_val).strip() else "Unknown_Source"
                                
                            empty_records[smiles_val].add(source_val)

                        elif is_unclassified:
                            # 判定如果来源于不同的文献记录，执行空行提行
                            if previous_unclass_workbook is not None and previous_unclass_workbook != workbook_path and not unclass_appended_this_wb:
                                writer_unclass.append_blank_row()
                            
                            unclass_appended_this_wb = True
                            previous_unclass_workbook = workbook_path

                            stats.unclassified_rows += 1
                            values_by_key[ACTIVITY_LEVEL_KEY] = level
                            values_by_key["_source_location"] = f"{workbook_path.name} / {spec.sheet_name}"
                            writer_unclass.append(values_by_key)
                        else:
                            stats.level_counts[level] += 1
                            values_by_key[ACTIVITY_LEVEL_KEY] = level
                            writer_valid.append(values_by_key)
                except Exception as exc:
                    stats.failed_sheets += 1
                    print(
                        f"[错误] 处理 {workbook_path.name} / {spec.sheet_name} 失败：{exc}",
                        file=sys.stderr,
                    )
        finally:
            workbook.close()

    stats.output_sheets = writer_valid.sheet_count
    return writer_valid, writer_unclass, empty_records


def validate_args(args: argparse.Namespace) -> tuple[Path, Path, Path] | None:
    input_dir = args.input_dir.expanduser().resolve()
    strain_file = args.strain_file.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    if load_workbook is None or Workbook is None:
        print(
            "[错误] 当前 Python 环境缺少 openpyxl，无法处理 XLSX。"
            "请确认 PDF_Extractor 环境已安装该依赖。",
            file=sys.stderr,
        )
        return None
    if not input_dir.is_dir():
        print(f"[错误] 输入目录不存在：{input_dir}", file=sys.stderr)
        return None
    if not strain_file.is_file():
        print(f"[错误] 菌株 TXT 不存在：{strain_file}", file=sys.stderr)
        return None
    if args.header_scan_rows < 1:
        print("[错误] --header-scan-rows 必须大于或等于 1。", file=sys.stderr)
        return None
    if args.max_rows_per_sheet < 2:
        print(
            "[错误] --max-rows-per-sheet 至少为 2（1 行表头 + 1 行数据）。",
            file=sys.stderr,
        )
        return None
    if output_path.exists() and not args.overwrite:
        print(
            f"[错误] 输出文件已存在：{output_path}\n"
            "如需覆盖，请显式添加 --overwrite。",
            file=sys.stderr,
        )
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return input_dir, strain_file, output_path


def print_summary(
    stats: RunStats,
    strain_nonempty_lines: int,
    allowed_strain_count: int,
    output_path: Path,
    unclass_path: Path,
    empty_path: Path,
    sheet_row_counts: Sequence[int],
) -> None:
    print(
        f"菌株 TXT：{strain_nonempty_lines} 个非空字段，"
        f"规范化后 {allowed_strain_count} 个唯一字段"
    )
    print(
        f"源工作簿：{stats.source_workbooks} 个；"
        f"首次读取成功 {stats.readable_workbooks}，读取失败 {stats.failed_workbooks}"
    )
    print(
        f"源工作表：{stats.source_sheets} 个；识别 {stats.recognized_sheets}，"
        f"跳过 {stats.skipped_sheets}，处理失败 {stats.failed_sheets}"
    )
    print(f"扫描数据行：{stats.scanned_data_rows}")
    print(f"匹配总行数：{stats.matched_rows}")
    print(
        "有效 Activity Level："
        f"0={stats.level_counts[0]}，1={stats.level_counts[1]}，"
        f"2={stats.level_counts[2]}，3={stats.level_counts[3]}"
    )
    print(f"独立提取：活性为空={stats.empty_rows}，无法被明确分类={stats.unclassified_rows}")
    print(
        "有效数据判定依据："
        + "，".join(
            f"{name}={count}"
            for name, count in sorted(stats.metric_counts.items())
            if name not in ("empty", "unclassified")
        )
    )
    print(
        f"有效输出 sheet：{len(sheet_row_counts)} 个；总行数（含各 sheet 表头）："
        + ", ".join(str(count) for count in sheet_row_counts)
    )
    print(f"主输出文件 (级别 0123)：{output_path}")
    print(f"副输出文件 (无法分类)：{unclass_path}")
    print(f"副输出文件 (活性为空)：{empty_path}")


def main() -> int:
    args = parse_args()
    paths = validate_args(args)
    if paths is None:
        return 2
    input_dir, strain_file, output_path = paths

    allowed_strains, strain_nonempty_lines = load_allowed_strains(strain_file)
    if not allowed_strains:
        print(f"[错误] 菌株 TXT 中没有有效字段：{strain_file}", file=sys.stderr)
        return 1

    workbook_paths = list_source_workbooks(input_dir, output_path)
    if not workbook_paths:
        print(f"[错误] 输入目录中没有可处理的 XLSX：{input_dir}", file=sys.stderr)
        return 1

    stats = RunStats(source_workbooks=len(workbook_paths))
    specs_by_workbook, output_headers = discover_sheets(
        workbook_paths=workbook_paths,
        header_scan_rows=args.header_scan_rows,
        stats=stats,
    )
    if not specs_by_workbook:
        print("[错误] 没有找到包含所需表头的工作表。", file=sys.stderr)
        return 1

    writer_valid, writer_unclass, empty_records = process_sources(
        specs_by_workbook=specs_by_workbook,
        allowed_strains=allowed_strains,
        output_headers=output_headers,
        max_rows_per_sheet=args.max_rows_per_sheet,
        stats=stats,
    )
    
    # 1. 保存主输出数据
    writer_valid.save(output_path)
    
    # 2. 保存无法分类数据
    unclass_path = output_path.with_name(output_path.stem + "_unclassified.xlsx")
    if writer_unclass.sheet_count > 0 or writer_unclass.current_total_rows > 1:
        writer_unclass.save(unclass_path)
        
    # 3. 保存并聚合活性为空数据
    empty_path = output_path.with_name(output_path.stem + "_empty.xlsx")
    if empty_records:
        wb = Workbook(write_only=True)
        ws = wb.create_sheet("Empty Activity")
        
        headers = ["SMILES", "Source Filenames"]
        ws.append(headers)
        
        for sm, srcs in empty_records.items():
            ws.append([sm, " | ".join(sorted(srcs))])
            
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{empty_path.stem}_",
            suffix=".tmp.xlsx",
            dir=empty_path.parent,
        )
        os.close(file_descriptor)
        temporary_path = Path(temporary_name)
        try:
            wb.save(temporary_path)
            os.replace(temporary_path, empty_path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

    print_summary(
        stats=stats,
        strain_nonempty_lines=strain_nonempty_lines,
        allowed_strain_count=len(allowed_strains),
        output_path=output_path,
        unclass_path=unclass_path,
        empty_path=empty_path,
        sheet_row_counts=writer_valid.sheet_row_counts,
    )

    if stats.failed_workbooks or stats.failed_sheets:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())