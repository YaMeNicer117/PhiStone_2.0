"""Extract unique Target Strain values from Hazard Level A rows in XLSX files.

By default, the script scans every worksheet in every ``.xlsx`` file directly
under the ``tian_datas`` directory next to this script. Results are written as
UTF-8 text with one unique Target Strain value per line.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from openpyxl import load_workbook
except ImportError:  # Give a concise error from main instead of a traceback.
    load_workbook = None


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = SCRIPT_DIR
DEFAULT_OUTPUT_NAME = SCRIPT_DIR / "hazard_A_target_strains.txt"
DEFAULT_HEADER_SCAN_ROWS = 50

HAZARD_HEADER = "hazard level"
TARGET_STRAIN_HEADER = "target strain"


@dataclass
class SheetResult:
    """Summary of one worksheet scan."""

    header_found: bool = False
    matched_rows: int = 0
    empty_target_rows: int = 0
    new_target_count: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "提取所有 XLSX 工作表中 Hazard Level 为 A 的行，并将不重复的 "
            "Target Strain 写入 TXT。"
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"XLSX 所在目录（默认：{DEFAULT_INPUT_DIR}）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "输出 TXT 路径（默认：输入目录下的 "
            f"{DEFAULT_OUTPUT_NAME}）"
        ),
    )
    parser.add_argument(
        "--header-scan-rows",
        type=int,
        default=DEFAULT_HEADER_SCAN_ROWS,
        help=f"在每个工作表前多少行内查找表头（默认：{DEFAULT_HEADER_SCAN_ROWS}）",
    )
    return parser.parse_args()


def normalize_header(value: Any) -> str:
    """Normalize a possible header without changing data-cell contents."""

    if value is None:
        return ""
    text = str(value).replace("\ufeff", "")
    return " ".join(text.split()).casefold()


def clean_cell_text(value: Any) -> str:
    """Convert a cell value to text and remove only leading/trailing spaces."""

    if value is None:
        return ""
    return str(value).strip()


def process_worksheet(
    worksheet: Any,
    seen_targets: set[str],
    ordered_targets: list[str],
    header_scan_rows: int,
) -> SheetResult:
    """Scan one worksheet once, preserving first-seen order across all files."""

    result = SheetResult()
    hazard_index: int | None = None
    target_index: int | None = None

    for row_number, row in enumerate(
        worksheet.iter_rows(values_only=True), start=1
    ):
        if not result.header_found:
            if row_number > header_scan_rows:
                break

            normalized_headers = [normalize_header(value) for value in row]
            if HAZARD_HEADER in normalized_headers and TARGET_STRAIN_HEADER in normalized_headers:
                hazard_index = normalized_headers.index(HAZARD_HEADER)
                target_index = normalized_headers.index(TARGET_STRAIN_HEADER)
                result.header_found = True
            continue

        # Both indexes are assigned at the moment header_found becomes True.
        assert hazard_index is not None and target_index is not None
        hazard_value = row[hazard_index] if hazard_index < len(row) else None
        if clean_cell_text(hazard_value) != "A":
            continue

        result.matched_rows += 1
        target_value = row[target_index] if target_index < len(row) else None
        target_strain = clean_cell_text(target_value)
        if not target_strain:
            result.empty_target_rows += 1
            continue

        if target_strain not in seen_targets:
            seen_targets.add(target_strain)
            ordered_targets.append(target_strain)
            result.new_target_count += 1

    return result


def list_workbooks(input_dir: Path) -> list[Path]:
    """Return direct-child XLSX files in stable filename order."""

    return sorted(
        (
            path
            for path in input_dir.glob("*.xlsx")
            if path.is_file() and not path.name.startswith("~$")
        ),
        key=lambda path: path.name.casefold(),
    )


def write_targets(output_path: Path, targets: list[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(targets)
    if content:
        content += "\n"
    output_path.write_text(content, encoding="utf-8")


def main() -> int:
    args = parse_args()

    if load_workbook is None:
        print(
            "[错误] 当前 Python 环境缺少 openpyxl，无法读取 XLSX。"
            "请先确认 PDF_Extractor 环境已安装该依赖。",
            file=sys.stderr,
        )
        return 2

    if args.header_scan_rows < 1:
        print("[错误] --header-scan-rows 必须大于或等于 1。", file=sys.stderr)
        return 2

    input_dir = args.input_dir.expanduser().resolve()
    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else input_dir / DEFAULT_OUTPUT_NAME
    )

    if not input_dir.is_dir():
        print(f"[错误] 输入目录不存在：{input_dir}", file=sys.stderr)
        return 2

    workbooks = list_workbooks(input_dir)
    if not workbooks:
        print(f"[错误] 输入目录中没有找到 XLSX 文件：{input_dir}", file=sys.stderr)
        return 1

    seen_targets: set[str] = set()
    ordered_targets: list[str] = []

    processed_workbooks = 0
    failed_workbooks = 0
    total_sheets = 0
    sheets_with_header = 0
    failed_sheets = 0
    matched_rows = 0
    empty_target_rows = 0

    for workbook_path in workbooks:
        try:
            workbook = load_workbook(
                filename=workbook_path,
                read_only=True,
                data_only=True,
            )
        except Exception as exc:
            failed_workbooks += 1
            print(f"[错误] 无法读取工作簿 {workbook_path.name}：{exc}", file=sys.stderr)
            continue

        processed_workbooks += 1
        try:
            for worksheet in workbook.worksheets:
                total_sheets += 1
                try:
                    result = process_worksheet(
                        worksheet=worksheet,
                        seen_targets=seen_targets,
                        ordered_targets=ordered_targets,
                        header_scan_rows=args.header_scan_rows,
                    )
                except Exception as exc:
                    failed_sheets += 1
                    print(
                        f"[错误] 读取 {workbook_path.name} / {worksheet.title} 失败：{exc}",
                        file=sys.stderr,
                    )
                    continue

                if not result.header_found:
                    print(
                        f"[警告] {workbook_path.name} / {worksheet.title}："
                        f"前 {args.header_scan_rows} 行未同时找到 "
                        "Hazard Level 和 Target Strain 表头。",
                        file=sys.stderr,
                    )
                    continue

                sheets_with_header += 1
                matched_rows += result.matched_rows
                empty_target_rows += result.empty_target_rows
        finally:
            workbook.close()

    write_targets(output_path, ordered_targets)

    print(f"扫描工作簿：{len(workbooks)} 个（成功 {processed_workbooks}，失败 {failed_workbooks}）")
    print(f"扫描工作表：{total_sheets} 个（识别表头 {sheets_with_header}，失败 {failed_sheets}）")
    print(f"Hazard Level 为 A 的行：{matched_rows} 行")
    print(f"其中 Target Strain 为空：{empty_target_rows} 行")
    print(f"提取的不重复 Target Strain：{len(ordered_targets)} 个")
    print(f"输出文件：{output_path}")

    return 1 if failed_workbooks or failed_sheets else 0


if __name__ == "__main__":
    raise SystemExit(main())
