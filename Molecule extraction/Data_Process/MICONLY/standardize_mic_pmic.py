"""Standardize MIC measurements and calculate 0-1 activity scores.

The script reads every data worksheet from an input workbook, keeps rows whose
``MIC Value`` is not blank, converts supported MIC units to micromolar, and
writes the converted value back to ``MIC Value``. The output workbook contains
only the selected source fields plus ``Standardized pMIC`` and an orange-review
reason. Recognized inactivity keywords use 64 uM as the cutoff representative;
unconvertible yellow rows retain their original ``MIC Value`` for diagnosis.

Rows that need manual review are highlighted in the output workbook:

* yellow: unrecognized MIC text, invalid/missing SMILES, unknown units, or a
  failed/non-positive numeric conversion;
* light pink: a valid SMILES whose total formal charge is not zero;
* orange: a value explicitly marked as approximate with
  ``~``/``about``/``approx``.

Whitespace inside ``MIC Value`` is ignored during parsing. ``±`` and everything
after it are discarded without review marking, while ``≈`` is treated as exact
equality. Inequalities whose converted boundary is 6.25-32 uM (inclusive) are
excluded; the comparator is ignored for every other inequality. Numeric ranges
use their midpoint without orange highlighting. Within each output worksheet,
yellow rows are placed after ordinary/orange rows and immediately before pink
rows.

Dataset-specific unit aliases treat ``uM/ml`` as a typo for ``uM`` (x1), while
``umol/ml`` is scaled by 1000 and ``umol/L`` remains x1. ``mmol`` is treated as
mM. Ranges may use ``-``, ``~``, ``to``, ``->``, or ``→``; repeated
``<<``/``>>`` comparators are treated as ``<``/``>``. Explicit not-tested or
not-determined codes are discarded before any other processing.

The standardized score uses three logarithmic MIC regions. The ranges
0.1-6.25 uM, 6.25-32 uM, and 32-64 uM each occupy one third of the final
0-1 score span so that medium- and low-activity compounds retain sufficient
resolution during model training.

The input workbook is never modified.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
import tempfile
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime
from numbers import Real
from pathlib import Path
from typing import Any, Literal, Sequence

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.worksheet import Worksheet
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors


DEFAULT_INPUT = Path("human_mtb_activity_reclassified_cleaned.xlsx")
DEFAULT_OUTPUT = Path("human_mtb_activity_reclassified_cleaned_pMIC.xlsx")

HEADER_SCAN_ROWS = 20
MIC_MIN_UM = 0.1
MIC_MAX_UM = 64.0
ACTIVITY_BOUNDARY_HIGH_UM = 6.25
ACTIVITY_BOUNDARY_LOW_UM = 32.0
STANDARDIZED_SCORE_MAX = 1.0
STANDARDIZED_SCORE_MIN = 0.0
STANDARDIZED_SCORE_ONE_THIRD = 1.0 / 3.0
STANDARDIZED_SCORE_TWO_THIRDS = 2.0 / 3.0
PMIC_DECIMALS = 6
HIGH_ACTIVITY_LOG_SPAN = math.log10(
    ACTIVITY_BOUNDARY_HIGH_UM / MIC_MIN_UM
)
MEDIUM_ACTIVITY_LOG_SPAN = math.log10(
    ACTIVITY_BOUNDARY_LOW_UM / ACTIVITY_BOUNDARY_HIGH_UM
)
LOW_ACTIVITY_LOG_SPAN = math.log10(
    MIC_MAX_UM / ACTIVITY_BOUNDARY_LOW_UM
)
INEQUALITY_EXCLUDE_MIN_UM = ACTIVITY_BOUNDARY_HIGH_UM
INEQUALITY_EXCLUDE_MAX_UM = ACTIVITY_BOUNDARY_LOW_UM

PRESERVED_COLUMNS = (
    "Source Filename",
    "Filename/DOI",
    "Publication Date",
    "Impact Factor",
    "Compound ID",
    "Target Strain",
    "SMILES",
    "MIC Value",
)
INTERNAL_COLUMNS = ("MIC Unit",)
REQUIRED_INPUT_COLUMNS = PRESERVED_COLUMNS + INTERNAL_COLUMNS
OUTPUT_COLUMNS = PRESERVED_COLUMNS + (
    "Standardized pMIC",
    "Orange Flag Reason",
)

YELLOW_FILL = PatternFill(fill_type="solid", fgColor="FFF2CC")
PINK_FILL = PatternFill(fill_type="solid", fgColor="F4CCCC")
ORANGE_FILL = PatternFill(fill_type="solid", fgColor="F4B183")
HEADER_FILL = PatternFill(fill_type="solid", fgColor="1F4E78")
HEADER_FONT = Font(color="FFFFFF", bold=True)

COLUMN_WIDTHS = {
    "Source Filename": 28,
    "Filename/DOI": 36,
    "Publication Date": 16,
    "Impact Factor": 14,
    "Compound ID": 18,
    "Target Strain": 24,
    "SMILES": 50,
    "MIC Value": 24,
    "Standardized pMIC": 20,
    "Orange Flag Reason": 36,
}

Comparator = Literal["eq", "lt", "le", "gt", "ge"]
UnitDimension = Literal["molar", "mass"]


class ProcessingError(RuntimeError):
    """Raised when workbook validation or processing cannot safely continue."""


@dataclass(frozen=True)
class UnitSpec:
    dimension: UnitDimension
    multiplier: float
    canonical: str


@dataclass(frozen=True)
class ParsedMic:
    numeric_value: float | None
    comparator: Comparator = "eq"
    approximate: bool = False
    range_used: bool = False
    keyword_inactive: bool = False

    @property
    def is_numeric(self) -> bool:
        return self.numeric_value is not None


@dataclass(frozen=True)
class SheetSpec:
    name: str
    header_row: int
    columns: dict[str, int]


@dataclass(frozen=True)
class SmilesAnalysis:
    valid: bool
    molecular_weight: float | None
    total_charge: int | None


@dataclass(frozen=True)
class OutputRow:
    values: list[Any]
    yellow: bool
    charged: bool
    orange_reason: str


@dataclass
class SheetStats:
    name: str
    source_rows: int = 0
    removed_blank: int = 0
    output_rows: int = 0
    calculated: int = 0
    keyword_rows: int = 0
    excluded_keyword_rows: int = 0
    excluded_inequality_rows: int = 0
    range_rows: int = 0
    pink_rows: int = 0
    orange_rows: int = 0
    yellow_rows: int = 0


def _unit_aliases() -> dict[str, UnitSpec]:
    aliases: dict[str, UnitSpec] = {}

    def add(
        names: Sequence[str],
        dimension: UnitDimension,
        multiplier: float,
        canonical: str,
    ) -> None:
        spec = UnitSpec(dimension, multiplier, canonical)
        for name in names:
            aliases[name] = spec

    # For molar units, multiplier converts the source value directly to uM.
    add(("m", "mol/l", "moll-1", "molar"), "molar", 1_000_000.0, "M")
    add(
        ("mm", "mmol", "mmol/l", "mmoll-1", "millimolar"),
        "molar",
        1_000.0,
        "mM",
    )
    add(
        (
            "um",
            "um/ml",
            "umol/l",
            "umoll-1",
            "micromolar",
            "micromol/l",
            "micromoll-1",
        ),
        "molar",
        1.0,
        "uM",
    )
    # uM/ml is handled above as a documented typo for uM. In contrast,
    # micromoles per mL is a true amount-per-volume unit equal to 1000 uM.
    add(("umol/ml", "umolml-1"), "molar", 1_000.0, "umol/mL")
    add(
        ("nm", "nmol/l", "nmoll-1", "nanomolar"),
        "molar",
        0.001,
        "nM",
    )
    add(
        ("pm", "pmol/l", "pmoll-1", "picomolar"),
        "molar",
        0.000001,
        "pM",
    )

    # For mass units, multiplier converts one source unit to g/L.
    add(("g/l", "gl-1", "gram/l", "grams/l"), "mass", 1.0, "g/L")
    add(("mg/ml", "mgml-1"), "mass", 1.0, "mg/mL")
    add(("mg/l", "mgl-1"), "mass", 0.001, "mg/L")
    add(
        ("ug/ml", "ugml-1", "mcg/ml", "mcgml-1"),
        "mass",
        0.001,
        "ug/mL",
    )
    add(
        ("ug/l", "ugl-1", "mcg/l", "mcgl-1"),
        "mass",
        0.000001,
        "ug/L",
    )
    add(("ng/ml", "ngml-1"), "mass", 0.000001, "ng/mL")
    return aliases


UNIT_ALIASES = _unit_aliases()

UNSIGNED_NUMBER_PATTERN = (
    r"(?:(?:\d+(?:[.,]\d*)?)|(?:[.,]\d+))(?:[eE][+-]?\d+)?"
)
NUMBER_RE = re.compile(rf"^[+-]?{UNSIGNED_NUMBER_PATTERN}")
RANGE_RE = re.compile(
    rf"^(?P<start>{UNSIGNED_NUMBER_PATTERN})"
    rf"(?:->|→|~|to|-)"
    rf"(?P<end>{UNSIGNED_NUMBER_PATTERN})",
    re.IGNORECASE,
)
MIC_PREFIX_RE = re.compile(r"^mic(?:value)?(?:[:=])?", re.IGNORECASE)
APPROX_PREFIX_RE = re.compile(
    r"^(?:~|about|approx(?:imately)?\.?)",
    re.IGNORECASE,
)
COMPARATOR_PREFIX_RE = re.compile(
    r"^(?:"
    r"(?P<le><=|=<|≤|lessthanorequalto|nomorethan|atmost)|"
    r"(?P<ge>>=|=>|≥|greaterthanorequalto|morethanorequalto|atleast)|"
    r"(?P<lt>(?:<<|<|lessthan|below|under))|"
    r"(?P<gt>(?:>>|>|greaterthan|morethan|above|over))|"
    r"(?P<eq>=|≈)"
    r")",
    re.IGNORECASE,
)

SHORT_INACTIVE_PATTERNS = (
    re.compile(r"(?<![a-z0-9])na(?![a-z0-9])", re.IGNORECASE),
    re.compile(r"(?<![a-z0-9])n\s*[/\.]\s*a\.?(?![a-z0-9])", re.IGNORECASE),
)
EXCLUDED_MIC_PATTERNS = (
    re.compile(r"\*"),
    re.compile(r"--"),
    re.compile(
        r"(?<![a-z0-9])n\s*(?:[/\.]\s*)?t\.?(?![a-z0-9])",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?<![a-z0-9])n\s*(?:[/\.]\s*)?d\.?(?![a-z0-9])",
        re.IGNORECASE,
    ),
    re.compile(r"(?<![a-z0-9])nma(?![a-z0-9])", re.IGNORECASE),
    re.compile(r"(?<![a-z0-9])res(?![a-z0-9])", re.IGNORECASE),
)
INACTIVE_KEYWORDS_COMPACT = (
    "notdetermined",
    "inactive",
    "notactive",
    "noactivity",
    "noinhibition",
    "poor",
    "resistant",
    "weak",
    "lowactivity",
    "noeffective",
    "less",
    "lack",
    "loss",
    "ineffective",
)


def normalize_text(value: Any) -> str:
    """Return normalized text without changing its semantic content."""

    if value is None:
        return ""
    return unicodedata.normalize("NFKC", str(value)).strip()


def normalize_header(value: Any) -> str:
    text = normalize_text(value)
    return re.sub(r"\s+", " ", text).casefold()


def normalize_unit_token(value: Any) -> str:
    """Normalize common Unicode and textual unit spellings for lookup."""

    text = normalize_text(value).casefold()
    if not text:
        return ""

    text = text.replace("μ", "u").replace("µ", "u")
    text = text.replace("−", "-").replace("–", "-").replace("—", "-")
    text = text.replace("⁻", "-").replace("¹", "1")
    text = text.replace("·", "").replace("⋅", "").replace("*", "")
    text = text.replace("\\", "/").replace("per", "/")
    text = text.replace("^", "").replace(".", "")
    text = re.sub(r"\s+", "", text)
    return text.strip("()[]{};,:")


def parse_unit(value: Any) -> UnitSpec | None:
    return UNIT_ALIASES.get(normalize_unit_token(value))


def is_blank_mic(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _discard_uncertainty(text: str) -> str:
    positions = [position for marker in ("±", "+/-") if (position := text.find(marker)) >= 0]
    if not positions:
        return text
    return text[: min(positions)]


def _consume_approximation(text: str) -> tuple[str, bool]:
    match = APPROX_PREFIX_RE.match(text)
    if not match:
        return text, False
    return text[match.end() :], True


def _consume_comparator(text: str) -> tuple[str, Comparator]:
    match = COMPARATOR_PREFIX_RE.match(text)
    if not match:
        return text, "eq"

    comparator: Comparator = "eq"
    for candidate in ("le", "ge", "lt", "gt", "eq"):
        if match.group(candidate) is not None:
            comparator = candidate  # type: ignore[assignment]
            break
    return text[match.end() :], comparator


def _parse_numeric_token(token: str) -> float | None:
    mantissa = token
    exponent = ""
    exponent_match = re.search(r"[eE][+-]?\d+$", token)
    if exponent_match:
        mantissa = token[: exponent_match.start()]
        exponent = exponent_match.group(0)

    if "," in mantissa and "." in mantissa:
        return None
    mantissa = mantissa.replace(",", ".")
    try:
        result = float(mantissa + exponent)
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def _numeric_string_parse(value: str) -> ParsedMic | None:
    text = re.sub(r"\s+", "", normalize_text(value))
    text = text.replace("−", "-").replace("–", "-").replace("—", "-")
    text = _discard_uncertainty(text)
    text = MIC_PREFIX_RE.sub("", text, count=1).strip()

    text, approx_before = _consume_approximation(text)
    text, comparator = _consume_comparator(text)
    text, approx_after = _consume_approximation(text)
    approximate = approx_before or approx_after

    range_match = RANGE_RE.match(text)
    if range_match:
        start_value = _parse_numeric_token(range_match.group("start"))
        end_value = _parse_numeric_token(range_match.group("end"))
        if start_value is None or end_value is None:
            return None
        if start_value <= 0 or end_value <= 0:
            return None

        suffix = text[range_match.end() :].strip().rstrip(";,.").strip()
        if suffix:
            stripped_suffix = suffix.strip("()[]{}")
            if parse_unit(stripped_suffix) is None:
                return None

        return ParsedMic(
            numeric_value=(start_value + end_value) / 2.0,
            comparator=comparator,
            approximate=approximate,
            range_used=True,
        )

    match = NUMBER_RE.match(text)
    if not match:
        return None

    numeric_value = _parse_numeric_token(match.group(0))
    if numeric_value is None:
        return None

    suffix = text[match.end() :].strip().rstrip(";,.").strip()
    if suffix:
        stripped_suffix = suffix.strip("()[]{}")
        if parse_unit(stripped_suffix) is None:
            return None

    return ParsedMic(
        numeric_value=numeric_value,
        comparator=comparator,
        approximate=approximate,
    )


def contains_inactive_keyword(value: Any) -> bool:
    text = normalize_text(value)
    if not text:
        return False
    if any(pattern.search(text) for pattern in SHORT_INACTIVE_PATTERNS):
        return True
    compact = re.sub(r"\s+", "", text).casefold()
    return any(keyword in compact for keyword in INACTIVE_KEYWORDS_COMPACT)


def contains_excluded_mic_keyword(value: Any) -> bool:
    """Return whether MIC text contains a code that must not be extracted."""

    text = normalize_text(value)
    if not text:
        return False
    return any(pattern.search(text) for pattern in EXCLUDED_MIC_PATTERNS)


def parse_mic_value(value: Any) -> ParsedMic:
    """Parse a MIC value, giving valid numeric expressions precedence."""

    if isinstance(value, Real) and not isinstance(value, bool):
        numeric_value = float(value)
        if math.isfinite(numeric_value):
            return ParsedMic(numeric_value=numeric_value)
        return ParsedMic(numeric_value=None)

    if isinstance(value, str):
        parsed_numeric = _numeric_string_parse(value)
        if parsed_numeric is not None:
            return parsed_numeric
        if contains_inactive_keyword(value):
            return ParsedMic(numeric_value=None, comparator="ge", keyword_inactive=True)

    return ParsedMic(numeric_value=None)


def analyze_smiles(
    smiles_value: Any,
    cache: dict[str, SmilesAnalysis],
) -> SmilesAnalysis:
    """Return validity, average MolWt, and total formal charge for a SMILES."""

    smiles = normalize_text(smiles_value)
    if not smiles:
        return SmilesAnalysis(False, None, None)
    if smiles in cache:
        return cache[smiles]

    try:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            result = SmilesAnalysis(False, None, None)
        else:
            molecular_weight = float(Descriptors.MolWt(molecule))
            if not math.isfinite(molecular_weight) or molecular_weight <= 0:
                result = SmilesAnalysis(False, None, None)
            else:
                total_charge = int(Chem.GetFormalCharge(molecule))
                result = SmilesAnalysis(True, molecular_weight, total_charge)
    except Exception:
        result = SmilesAnalysis(False, None, None)

    cache[smiles] = result
    return result


def convert_to_micromolar(
    numeric_value: float,
    unit: UnitSpec,
    molecular_weight: float | None,
) -> float | None:
    if not math.isfinite(numeric_value) or numeric_value <= 0:
        return None

    if unit.dimension == "molar":
        micromolar = numeric_value * unit.multiplier
    else:
        if molecular_weight is None or molecular_weight <= 0:
            return None
        grams_per_liter = numeric_value * unit.multiplier
        micromolar = grams_per_liter * 1_000_000.0 / molecular_weight

    if not math.isfinite(micromolar) or micromolar <= 0:
        return None
    return micromolar


def calculate_standardized_pmic(micromolar: float) -> float:
    """Map MIC to a clipped three-region logarithmic activity score."""

    if micromolar <= MIC_MIN_UM:
        return round(STANDARDIZED_SCORE_MAX, PMIC_DECIMALS)
    if micromolar >= MIC_MAX_UM:
        return round(STANDARDIZED_SCORE_MIN, PMIC_DECIMALS)

    if micromolar <= ACTIVITY_BOUNDARY_HIGH_UM:
        within_region = math.log10(
            ACTIVITY_BOUNDARY_HIGH_UM / micromolar
        ) / HIGH_ACTIVITY_LOG_SPAN
        score = (
            STANDARDIZED_SCORE_TWO_THIRDS
            + STANDARDIZED_SCORE_ONE_THIRD * within_region
        )
    elif micromolar <= ACTIVITY_BOUNDARY_LOW_UM:
        within_region = math.log10(
            ACTIVITY_BOUNDARY_LOW_UM / micromolar
        ) / MEDIUM_ACTIVITY_LOG_SPAN
        score = (
            STANDARDIZED_SCORE_ONE_THIRD
            + STANDARDIZED_SCORE_ONE_THIRD * within_region
        )
    else:
        within_region = math.log10(
            MIC_MAX_UM / micromolar
        ) / LOW_ACTIVITY_LOG_SPAN
        score = STANDARDIZED_SCORE_ONE_THIRD * within_region

    score = max(STANDARDIZED_SCORE_MIN, min(STANDARDIZED_SCORE_MAX, score))
    return round(score, PMIC_DECIMALS)


def orange_reasons(parsed: ParsedMic) -> list[str]:
    """Describe parse conditions that require orange review highlighting."""

    reasons: list[str] = []
    if parsed.approximate:
        reasons.append("MIC含近似标记")
    return reasons


def should_exclude_inequality(parsed: ParsedMic, micromolar: float) -> bool:
    return (
        parsed.comparator != "eq"
        and INEQUALITY_EXCLUDE_MIN_UM
        <= micromolar
        <= INEQUALITY_EXCLUDE_MAX_UM
    )


def _header_positions(sheet: Worksheet, row_number: int) -> dict[str, list[int]]:
    positions: dict[str, list[int]] = {}
    for column_number in range(1, sheet.max_column + 1):
        key = normalize_header(sheet.cell(row=row_number, column=column_number).value)
        if key:
            positions.setdefault(key, []).append(column_number)
    return positions


def find_header(sheet: Worksheet) -> tuple[int, dict[str, list[int]]] | None:
    mic_key = normalize_header("MIC Value")
    last_scan_row = min(HEADER_SCAN_ROWS, sheet.max_row)
    for row_number in range(1, last_scan_row + 1):
        positions = _header_positions(sheet, row_number)
        if mic_key in positions:
            return row_number, positions
    return None


def preflight_workbook(workbook: Any) -> tuple[list[SheetSpec], list[str]]:
    specs: list[SheetSpec] = []
    skipped: list[str] = []
    errors: list[str] = []

    for sheet in workbook.worksheets:
        header = find_header(sheet)
        if header is None:
            skipped.append(sheet.title)
            continue

        header_row, positions = header
        columns: dict[str, int] = {}
        sheet_errors: list[str] = []
        for column_name in REQUIRED_INPUT_COLUMNS:
            matches = positions.get(normalize_header(column_name), [])
            if not matches:
                sheet_errors.append(f"缺少列 {column_name!r}")
            elif len(matches) > 1:
                sheet_errors.append(f"列 {column_name!r} 重复 {len(matches)} 次")
            else:
                columns[column_name] = matches[0]

        if sheet_errors:
            errors.append(f"Sheet {sheet.title!r}: " + "；".join(sheet_errors))
        else:
            specs.append(SheetSpec(sheet.title, header_row, columns))

    if errors:
        raise ProcessingError("工作簿预检失败：\n- " + "\n- ".join(errors))
    if not specs:
        raise ProcessingError(
            f"前 {HEADER_SCAN_ROWS} 行内未找到包含 'MIC Value' 的数据 Sheet。"
        )
    return specs, skipped


def style_output_sheet(sheet: Worksheet) -> None:
    sheet.freeze_panes = "A2"
    sheet.sheet_view.showGridLines = False
    sheet.auto_filter.ref = f"A1:{get_column_letter(len(OUTPUT_COLUMNS))}{sheet.max_row}"

    for cell in sheet[1]:
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    sheet.row_dimensions[1].height = 28

    for index, column_name in enumerate(OUTPUT_COLUMNS, start=1):
        sheet.column_dimensions[get_column_letter(index)].width = COLUMN_WIDTHS[column_name]


def apply_row_fill(sheet: Worksheet, row_number: int, fill: PatternFill) -> None:
    for column_number in range(1, len(OUTPUT_COLUMNS) + 1):
        sheet.cell(row=row_number, column=column_number).fill = fill


def copy_output_values(
    source_sheet: Worksheet,
    source_row: int,
    spec: SheetSpec,
    standardized_mic_value: Any,
    pmic: float | None,
    orange_reason: str,
) -> list[Any]:
    values: list[Any] = []
    for column_name in PRESERVED_COLUMNS:
        if column_name == "MIC Value":
            values.append(standardized_mic_value)
        else:
            values.append(
                source_sheet.cell(
                    row=source_row,
                    column=spec.columns[column_name],
                ).value
            )
    values.append(pmic)
    values.append(orange_reason)
    return values


def process_sheet(
    source_sheet: Worksheet,
    output_sheet: Worksheet,
    spec: SheetSpec,
    smiles_cache: dict[str, SmilesAnalysis],
) -> SheetStats:
    stats = SheetStats(
        name=source_sheet.title,
        source_rows=max(source_sheet.max_row - spec.header_row, 0),
    )
    output_sheet.append(list(OUTPUT_COLUMNS))

    mic_column = spec.columns["MIC Value"]
    unit_column = spec.columns["MIC Unit"]
    smiles_column = spec.columns["SMILES"]
    regular_rows: list[OutputRow] = []
    yellow_rows: list[OutputRow] = []
    pink_rows: list[OutputRow] = []

    for source_row in range(spec.header_row + 1, source_sheet.max_row + 1):
        mic_value = source_sheet.cell(row=source_row, column=mic_column).value
        if is_blank_mic(mic_value):
            stats.removed_blank += 1
            continue
        if contains_excluded_mic_keyword(mic_value):
            stats.excluded_keyword_rows += 1
            continue

        parsed = parse_mic_value(mic_value)
        unit = parse_unit(source_sheet.cell(row=source_row, column=unit_column).value)
        smiles_analysis = analyze_smiles(
            source_sheet.cell(row=source_row, column=smiles_column).value,
            smiles_cache,
        )

        charged = (
            smiles_analysis.valid
            and smiles_analysis.total_charge is not None
            and smiles_analysis.total_charge != 0
            and (unit is None or unit.dimension != "molar")
        )
        yellow = False
        standardized_mic_value: Any = mic_value
        pmic: float | None = None
        reasons: list[str] = []

        if parsed.keyword_inactive:
            # Keyword rows are explicitly defined as inactive. Their score does
            # not depend on MIC Unit or SMILES validity and is never yellow for
            # those two reasons. A valid charged SMILES may still make the row
            # pink when its source unit is not a recognized molar unit.
            stats.keyword_rows += 1
            standardized_mic_value = MIC_MAX_UM
            pmic = round(STANDARDIZED_SCORE_MIN, PMIC_DECIMALS)
        elif parsed.is_numeric:
            numeric_value = parsed.numeric_value
            assert numeric_value is not None

            micromolar = None
            if unit is not None and smiles_analysis.valid:
                micromolar = convert_to_micromolar(
                    numeric_value,
                    unit,
                    smiles_analysis.molecular_weight,
                )
            elif unit is not None and unit.dimension == "molar":
                # Molar units do not require molecular weight, so an invalid or
                # absent SMILES does not prevent conversion; the row stays yellow.
                micromolar = convert_to_micromolar(numeric_value, unit, None)

            yellow = unit is None or not smiles_analysis.valid
            if micromolar is None:
                yellow = True
            else:
                # The converted uM value is the single source for exclusion,
                # scoring, and the output MIC Value.
                standardized_mic_value = micromolar
                if should_exclude_inequality(parsed, micromolar):
                    stats.excluded_inequality_rows += 1
                    continue
                pmic = calculate_standardized_pmic(micromolar)
                reasons = orange_reasons(parsed)
                if parsed.range_used:
                    stats.range_rows += 1
        else:
            yellow = True

        orange_reason = "；".join(reasons)
        if pmic is not None:
            stats.calculated += 1

        output_row_data = OutputRow(
            values=copy_output_values(
                source_sheet,
                source_row,
                spec,
                standardized_mic_value,
                pmic,
                orange_reason,
            ),
            yellow=yellow,
            charged=charged,
            orange_reason=orange_reason,
        )
        if yellow:
            yellow_rows.append(output_row_data)
            stats.yellow_rows += 1
        elif charged:
            pink_rows.append(output_row_data)
            stats.pink_rows += 1
        else:
            regular_rows.append(output_row_data)
            if orange_reason:
                stats.orange_rows += 1

    mic_value_column = OUTPUT_COLUMNS.index("MIC Value") + 1
    pmic_column = OUTPUT_COLUMNS.index("Standardized pMIC") + 1
    publication_date_column = OUTPUT_COLUMNS.index("Publication Date") + 1
    for output_row_data in regular_rows + yellow_rows + pink_rows:
        output_sheet.append(output_row_data.values)
        output_row = output_sheet.max_row
        stats.output_rows += 1

        mic_value_cell = output_sheet.cell(row=output_row, column=mic_value_column)
        mic_value_cell.number_format = "0.######"
        pmic_cell = output_sheet.cell(row=output_row, column=pmic_column)
        pmic_cell.number_format = "0.000000"
        publication_date = output_sheet.cell(
            row=output_row,
            column=publication_date_column,
        )
        if isinstance(publication_date.value, (date, datetime)):
            publication_date.number_format = "yyyy-mm-dd"

        for cell in output_sheet[output_row]:
            cell.alignment = Alignment(vertical="top", wrap_text=False)

        if output_row_data.yellow:
            apply_row_fill(output_sheet, output_row, YELLOW_FILL)
        elif output_row_data.charged:
            apply_row_fill(output_sheet, output_row, PINK_FILL)
        elif output_row_data.orange_reason:
            apply_row_fill(output_sheet, output_row, ORANGE_FILL)

    style_output_sheet(output_sheet)
    return stats


def _same_path(left: Path, right: Path) -> bool:
    left_normalized = os.path.normcase(str(left.resolve(strict=False)))
    right_normalized = os.path.normcase(str(right.resolve(strict=False)))
    return left_normalized == right_normalized


def save_workbook_atomically(
    workbook: Workbook,
    output_path: Path,
    overwrite: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{output_path.stem}.",
            suffix=".tmp.xlsx",
            dir=output_path.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
        workbook.save(temporary_path)
        if output_path.exists() and not overwrite:
            raise ProcessingError(
                f"输出文件在处理期间出现，未执行覆盖：{output_path}\n"
                "如确认覆盖，请显式添加 --overwrite。"
            )
        os.replace(temporary_path, output_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def standardize_workbook(
    input_path: Path,
    output_path: Path,
    overwrite: bool,
) -> tuple[list[SheetStats], list[str]]:
    if not input_path.is_file():
        raise ProcessingError(f"输入文件不存在：{input_path}")
    if input_path.suffix.casefold() != ".xlsx":
        raise ProcessingError("输入文件必须是 .xlsx 格式。")
    if output_path.suffix.casefold() != ".xlsx":
        raise ProcessingError("输出文件必须使用 .xlsx 扩展名。")
    if _same_path(input_path, output_path):
        raise ProcessingError("输出路径不能与输入路径相同，源工作簿不会被覆盖。")
    if output_path.exists() and not overwrite:
        raise ProcessingError(
            f"输出文件已存在：{output_path}\n"
            "如确认覆盖，请显式添加 --overwrite。"
        )

    source_workbook = load_workbook(input_path, data_only=True, read_only=False)
    try:
        specs, skipped = preflight_workbook(source_workbook)
        output_workbook = Workbook()
        try:
            output_workbook.remove(output_workbook.active)
            smiles_cache: dict[str, SmilesAnalysis] = {}
            stats: list[SheetStats] = []

            for spec in specs:
                source_sheet = source_workbook[spec.name]
                output_sheet = output_workbook.create_sheet(spec.name)
                stats.append(
                    process_sheet(source_sheet, output_sheet, spec, smiles_cache)
                )

            save_workbook_atomically(output_workbook, output_path, overwrite)
            return stats, skipped
        finally:
            output_workbook.close()
    finally:
        source_workbook.close()


def print_summary(
    stats: Sequence[SheetStats],
    skipped_sheets: Sequence[str],
    output_path: Path,
) -> None:
    for sheet_name in skipped_sheets:
        print(
            f"[跳过] Sheet {sheet_name!r}: 前 {HEADER_SCAN_ROWS} 行未找到 'MIC Value'。"
        )
    for item in stats:
        print(
            f"[{item.name}] 原始行={item.source_rows}, "
            f"删除空MIC={item.removed_blank}, 输出={item.output_rows}, "
            f"成功计算={item.calculated}, 关键词归类={item.keyword_rows}, "
            f"关键词删除={item.excluded_keyword_rows}, "
            f"排除不等式={item.excluded_inequality_rows}, "
            f"范围中值={item.range_rows}, "
            f"粉色={item.pink_rows}, 橙色={item.orange_rows}, "
            f"黄色={item.yellow_rows}"
        )
    print(f"处理完成：{output_path.resolve(strict=False)}")


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将多 Sheet MIC 数据统一换算为 uM，并计算 0-1 标准化活性评分。"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"输入工作簿路径（默认：{DEFAULT_INPUT}）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"输出工作簿路径（默认：{DEFAULT_OUTPUT}）",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="显式允许覆盖已存在的输出工作簿；永远不会覆盖输入文件。",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    RDLogger.DisableLog("rdApp.error")
    try:
        stats, skipped = standardize_workbook(
            arguments.input,
            arguments.output,
            arguments.overwrite,
        )
    except (ProcessingError, OSError, ValueError) as error:
        print(f"错误：{error}", file=sys.stderr)
        return 1

    print_summary(stats, skipped, arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
