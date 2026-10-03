"""用统一换算为 μM 的七档 Activity Level 评估 PhiSGATv2 活性趋势。"""

from __future__ import annotations

import argparse
import math
import multiprocessing
import os
import re
import sys
import tempfile
import unicodedata
import uuid
from dataclasses import dataclass
from pathlib import Path

# 必须在导入数值库和结构处理模块前限制底层线程竞争。
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
CHEMCUT_DIR = os.path.join(PROJECT_ROOT, "ChemCut")
if CHEMCUT_DIR not in sys.path:
    sys.path.insert(0, CHEMCUT_DIR)

if __package__:
    from . import PhiSGATv2_config as config
    from . import PhiSGATv2_predict as predictor
else:  # 支持直接运行 GATv2 目录中的脚本
    import PhiSGATv2_config as config
    import PhiSGATv2_predict as predictor

import PhiSSeparator_2D_work_data_process as work_data  # noqa: E402


EVALUATION_RESULT_FILENAME = "activity_curve_evaluation.xlsx"
EVALUATION_PLOT_FILENAME = "activity_curve_comparison.png"
PREDICTION_RESULT_FILENAME = "prediction_results.csv"
PREDICTION_ERROR_FILENAME = "prediction_errors.csv"
PROCESSED_DIRNAME = "processed_2d"
REQUIRED_COLUMNS = (
    "compound id",
    "smiles",
    "activity",
    "mic unit",
)
ACTIVITY_LEVEL_BANDS = (
    (2.0, 1.0),
    (12.5, 0.8),
    (32.0, 0.6),
    (64.0, 0.4),
)
INACTIVE_MIC_UM = 256.0
LOW_ACTIVITY_LEVEL = 0.1
ACTIVITY_LEVEL_MIN = 0.0
ACTIVITY_LEVEL_MAX = 1.0
ACTIVITY_LEVEL_GUIDES = (0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0)
MAX_PLOT_X_TICKS = 30
PLOT_SMOOTHING_FRACTION = 0.002
PLOT_SMOOTHING_MIN_BANDWIDTH = 1.0
PLOT_SMOOTH_CURVE_MIN_POINTS = 300
PLOT_SMOOTH_CURVE_MAX_POINTS = 3000
DUPLICATE_SMILES_COLUMNS = (
    "canonical_smiles",
    "representative_compound_id",
    "source_compound_id",
    "input_smiles",
    "source_sheet",
    "source_row",
    "mic_value",
    "mic_unit",
    "mic_value_um",
    "true_activity",
    "group_measurement_count",
    "group_true_activity_mean",
)
ACTIVITY_EVALUATION_INPUT_DIR = os.path.join(
    config.DATA_ROOT,
    "work_datas",
    "test_unknownactivity",
)
ACTIVITY_EVALUATION_DIR = os.path.join(
    config.DATA_ROOT,
    "work_datas",
    "activity_evaluation",
)


@dataclass(frozen=True)
class MicUnitSpec:
    """描述输入 MIC 单位及其到统一单位所需的换算参数。"""

    dimension: str
    multiplier: float
    canonical: str


def _build_mic_unit_aliases():
    aliases = {}

    def add(names, dimension, multiplier, canonical):
        spec = MicUnitSpec(
            dimension=dimension,
            multiplier=float(multiplier),
            canonical=canonical,
        )
        for name in names:
            aliases[name] = spec

    # 摩尔浓度的 multiplier 直接把原值换算为 μM。
    add(("m", "mol/l", "moll-1", "molar"), "molar", 1_000_000.0, "M")
    add(
        ("mm", "mmol/l", "mmoll-1", "millimolar"),
        "molar",
        1_000.0,
        "mM",
    )
    add(
        (
            "um",
            "umol/l",
            "umoll-1",
            "micromolar",
            "micromol/l",
            "micromoll-1",
        ),
        "molar",
        1.0,
        "μM",
    )
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

    # 质量浓度的 multiplier 先把原值换算为 g/L。
    add(("g/l", "gl-1", "gram/l", "grams/l"), "mass", 1.0, "g/L")
    add(("mg/ml", "mgml-1"), "mass", 1.0, "mg/mL")
    add(("mg/l", "mgl-1"), "mass", 0.001, "mg/L")
    add(
        ("ug/ml", "ugml-1", "mcg/ml", "mcgml-1"),
        "mass",
        0.001,
        "μg/mL",
    )
    add(
        ("ug/l", "ugl-1", "mcg/l", "mcgl-1"),
        "mass",
        0.000001,
        "μg/L",
    )
    add(("ng/ml", "ngml-1"), "mass", 0.000001, "ng/mL")
    return aliases


MIC_UNIT_ALIASES = _build_mic_unit_aliases()


def _validate_evaluation_settings():
    numeric_settings = {
        "INACTIVE_MIC_UM": INACTIVE_MIC_UM,
        "LOW_ACTIVITY_LEVEL": LOW_ACTIVITY_LEVEL,
        "ACTIVITY_LEVEL_MIN": ACTIVITY_LEVEL_MIN,
        "ACTIVITY_LEVEL_MAX": ACTIVITY_LEVEL_MAX,
    }
    for index, (upper_bound, activity_level) in enumerate(
        ACTIVITY_LEVEL_BANDS,
        start=1,
    ):
        numeric_settings[f"ACTIVITY_LEVEL_BAND_{index}_UPPER"] = upper_bound
        numeric_settings[f"ACTIVITY_LEVEL_BAND_{index}_LEVEL"] = activity_level
    for name, value in numeric_settings.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise TypeError(f"{name} 必须是有限数值")
    upper_bounds = [item[0] for item in ACTIVITY_LEVEL_BANDS]
    if not (
        upper_bounds
        and upper_bounds[0] > 0.0
        and all(
            left < right
            for left, right in zip(upper_bounds, upper_bounds[1:])
        )
        and upper_bounds[-1] < INACTIVE_MIC_UM
    ):
        raise ValueError("MIC μM 七档映射边界顺序无效")
    band_levels = [item[1] for item in ACTIVITY_LEVEL_BANDS]
    if not (
        ACTIVITY_LEVEL_MAX == band_levels[0]
        and all(
            higher > lower
            for higher, lower in zip(band_levels, band_levels[1:])
        )
        and band_levels[-1] > LOW_ACTIVITY_LEVEL > ACTIVITY_LEVEL_MIN
    ):
        raise ValueError("七档 Activity Level 顺序无效")
    if not (
        math.isclose(
            float(config.TRAIN_ACTIVITY_MIN),
            ACTIVITY_LEVEL_MIN,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and math.isclose(
            float(config.TRAIN_ACTIVITY_MAX),
            ACTIVITY_LEVEL_MAX,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise ValueError(
            "评估真实值范围与训练标签范围不一致: "
            f"评估=[{ACTIVITY_LEVEL_MIN}, {ACTIVITY_LEVEL_MAX}], "
            f"训练=[{config.TRAIN_ACTIVITY_MIN}, "
            f"{config.TRAIN_ACTIVITY_MAX}]"
        )
    for unit in set(MIC_UNIT_ALIASES.values()):
        if unit.dimension not in {"molar", "mass"}:
            raise ValueError(f"未知 MIC 单位维度: {unit.dimension}")
        if not math.isfinite(unit.multiplier) or unit.multiplier <= 0.0:
            raise ValueError(f"{unit.canonical} 的换算系数必须大于 0")


def _column_lookup(table, source_label):
    cleaned_columns = [str(column).strip() for column in table.columns]
    casefolded = [column.casefold() for column in cleaned_columns]
    if len(set(casefolded)) != len(casefolded):
        raise ValueError(f"{source_label}: 清理后存在大小写无关的重名列")
    table = table.copy()
    table.columns = cleaned_columns
    return table, {
        column.casefold(): column for column in cleaned_columns
    }


def _positive_finite_mic(value, source_label):
    if isinstance(value, bool):
        raise TypeError(f"{source_label}: MIC 值必须是正数")
    try:
        mic = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{source_label}: MIC 值不是纯数字: {value!r}"
        ) from exc
    if not math.isfinite(mic) or mic <= 0.0:
        raise ValueError(f"{source_label}: MIC 值必须是大于 0 的有限数值")
    return mic


def _normalize_mic_unit_token(value):
    """统一 Unicode、空白和常见文本单位写法，供别名表查询。"""
    text = unicodedata.normalize("NFKC", str(value)).strip().casefold()
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


def _parse_mic_unit(value, source_label):
    if value is None or isinstance(value, bool):
        raise ValueError(f"{source_label}: MIC Unit 不能为空")
    key = _normalize_mic_unit_token(value)
    if not key:
        raise ValueError(f"{source_label}: MIC Unit 不能为空")
    unit = MIC_UNIT_ALIASES.get(key)
    if unit is None:
        raise ValueError(
            f"{source_label}: 不支持的 MIC Unit {value!r}；"
            "支持 M、mM、μM、nM、pM、g/L、mg/mL、mg/L、"
            "μg/mL、μg/L、ng/mL 及其常见等价写法"
        )
    return unit


def _analyze_smiles(smiles, source_label):
    """返回保留立体信息的规范 SMILES 和 RDKit 平均分子量。"""
    try:
        molecule = Chem.MolFromSmiles(smiles, sanitize=True)
    except Exception as exc:
        raise ValueError(f"{source_label}: SMILES 解析失败") from exc
    if molecule is None or molecule.GetNumAtoms() == 0:
        raise ValueError(f"{source_label}: SMILES 无法解析")
    try:
        canonical_smiles = Chem.MolToSmiles(
            molecule,
            canonical=True,
            isomericSmiles=True,
        )
    except Exception as exc:
        raise ValueError(
            f"{source_label}: 无法生成保留立体信息的规范 SMILES"
        ) from exc
    if not canonical_smiles:
        raise ValueError(
            f"{source_label}: 无法生成保留立体信息的规范 SMILES"
        )
    try:
        molecular_weight = float(Descriptors.MolWt(molecule))
    except Exception as exc:
        raise ValueError(f"{source_label}: 分子量计算失败") from exc
    if not math.isfinite(molecular_weight) or molecular_weight <= 0.0:
        raise ValueError(f"{source_label}: 无法计算有效分子量")
    return canonical_smiles, molecular_weight


def convert_to_micromolar(mic, unit, molecular_weight):
    """先将任一受支持的 MIC 单位统一换算为 μM。"""
    mic = _positive_finite_mic(mic, "MIC")
    if not isinstance(unit, MicUnitSpec):
        raise TypeError("MIC 单位描述必须是 MicUnitSpec")
    if unit.dimension == "molar":
        micromolar = mic * unit.multiplier
    elif unit.dimension == "mass":
        if (
            isinstance(molecular_weight, bool)
            or not isinstance(molecular_weight, (int, float))
            or not math.isfinite(float(molecular_weight))
            or molecular_weight <= 0.0
        ):
            raise ValueError("质量浓度换算需要大于 0 的有限分子量")
        grams_per_liter = mic * unit.multiplier
        micromolar = grams_per_liter * 1_000_000.0 / molecular_weight
    else:
        raise ValueError(f"未知 MIC 单位维度: {unit.dimension}")
    if not math.isfinite(micromolar) or micromolar <= 0.0:
        raise ValueError("MIC 换算为 μM 后必须是大于 0 的有限数值")
    return float(micromolar)


def calculate_activity_level(micromolar):
    """把纯数值 μM MIC 直接映射为七档 Activity Level。"""
    micromolar = _positive_finite_mic(micromolar, "MIC (μM)")
    for upper_bound, activity_level in ACTIVITY_LEVEL_BANDS:
        if micromolar <= upper_bound:
            return activity_level
    if micromolar < 128.0:
        return 0.2
    if micromolar < INACTIVE_MIC_UM:
        return LOW_ACTIVITY_LEVEL
    return ACTIVITY_LEVEL_MIN


def _format_mic_observation(mic, mic_unit):
    return f"{float(mic):.12g} {mic_unit}"


def _format_micromolar_observation(micromolar):
    return f"{float(micromolar):.12g} μM"


def _read_activity_records(input_xlsx):
    try:
        sheets = pd.read_excel(
            input_xlsx,
            sheet_name=None,
            dtype=object,
            keep_default_na=False,
        )
    except Exception as exc:
        raise RuntimeError(f"无法读取评估表格: {input_xlsx}") from exc

    records = []
    for sheet_name, original_table in sheets.items():
        if original_table is None or len(original_table.columns) == 0:
            continue
        source_label = f"{input_xlsx}::{sheet_name}"
        table, lookup = _column_lookup(original_table, source_label)
        missing = [name for name in REQUIRED_COLUMNS if name not in lookup]
        if missing:
            raise ValueError(
                f"{source_label}: 缺少必需列 {missing}，"
                "需要 Compound ID、SMILES、Activity、MIC Unit"
            )

        compound_column = lookup["compound id"]
        smiles_column = lookup["smiles"]
        activity_column = lookup["activity"]
        unit_column = lookup["mic unit"]
        for row_number, (_, row) in enumerate(
            table.iterrows(),
            start=2,
        ):
            row_label = f"{source_label}:row{row_number}"
            compound_id = str(row[compound_column]).strip()
            smiles = str(row[smiles_column]).strip()
            if not compound_id:
                raise ValueError(f"{row_label}: Compound ID 不能为空")
            if not smiles:
                raise ValueError(f"{row_label}: SMILES 不能为空")
            mic = _positive_finite_mic(row[activity_column], row_label)
            mic_unit = _parse_mic_unit(row[unit_column], row_label)
            canonical_smiles, molecular_weight = _analyze_smiles(
                smiles,
                row_label,
            )
            micromolar = convert_to_micromolar(
                mic,
                mic_unit,
                molecular_weight,
            )
            records.append({
                "compound_id": compound_id,
                "input_smiles": smiles,
                "canonical_smiles": canonical_smiles,
                "mic_value": mic,
                "mic_unit": mic_unit.canonical,
                "mic_value_um": micromolar,
                "true_activity": calculate_activity_level(micromolar),
                "source_sheet": str(sheet_name),
                "source_row": row_number,
            })

    if not records:
        raise ValueError("评估表格中没有可读取的数据行")
    return records


def _build_evaluation_records(input_xlsx):
    source_records = _read_activity_records(input_xlsx)
    grouped = {}
    for source_record in source_records:
        canonical_smiles = source_record["canonical_smiles"]
        group = grouped.get(canonical_smiles)
        if group is None:
            group = {
                "compound_id": source_record["compound_id"],
                "source_compound_ids": [],
                "source_compound_id_keys": set(),
                "canonical_smiles": canonical_smiles,
                "mic_observations": [],
                "micromolar_observations": [],
                "true_activities": [],
                "source_records": [],
            }
            grouped[canonical_smiles] = group
        source_id = source_record["compound_id"]
        source_id_key = source_id.casefold()
        if source_id_key not in group["source_compound_id_keys"]:
            group["source_compound_ids"].append(source_id)
            group["source_compound_id_keys"].add(source_id_key)
        group["mic_observations"].append(
            _format_mic_observation(
                source_record["mic_value"],
                source_record["mic_unit"],
            )
        )
        group["micromolar_observations"].append(
            _format_micromolar_observation(
                source_record["mic_value_um"],
            )
        )
        group["true_activities"].append(source_record["true_activity"])
        group["source_records"].append(source_record)

    evaluation_records = []
    duplicate_records = []
    for index, group in enumerate(grouped.values(), start=1):
        true_activities = group["true_activities"]
        measurement_count = len(true_activities)
        mean_true_activity = float(
            math.fsum(true_activities) / measurement_count
        )
        evaluation_records.append({
            "compound_id": group["compound_id"],
            "source_compound_ids": "; ".join(
                group["source_compound_ids"]
            ),
            "canonical_smiles": group["canonical_smiles"],
            "measurement_count": measurement_count,
            "mic_observations": "; ".join(group["mic_observations"]),
            "micromolar_observations": "; ".join(
                group["micromolar_observations"]
            ),
            "true_activity": mean_true_activity,
            "processed_compound_id": f"activity_eval_{index:06d}",
        })
        if measurement_count > 1:
            for source_record in group["source_records"]:
                duplicate_records.append({
                    "canonical_smiles": group["canonical_smiles"],
                    "representative_compound_id": group["compound_id"],
                    "source_compound_id": source_record["compound_id"],
                    "input_smiles": source_record["input_smiles"],
                    "source_sheet": source_record["source_sheet"],
                    "source_row": source_record["source_row"],
                    "mic_value": source_record["mic_value"],
                    "mic_unit": source_record["mic_unit"],
                    "mic_value_um": source_record["mic_value_um"],
                    "true_activity": source_record["true_activity"],
                    "group_measurement_count": measurement_count,
                    "group_true_activity_mean": mean_true_activity,
                })
    return evaluation_records, duplicate_records, len(source_records)


def _resolve_input_xlsx(input_xlsx):
    if input_xlsx is not None:
        path = os.path.abspath(os.fspath(input_xlsx))
        if not os.path.isfile(path):
            raise FileNotFoundError(f"评估 XLSX 不存在: {path}")
        if os.path.splitext(path)[1].lower() != ".xlsx":
            raise ValueError(f"评估输入必须是 .xlsx 文件: {path}")
        return path

    input_dir = Path(ACTIVITY_EVALUATION_INPUT_DIR)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"评估测试数据目录不存在: {input_dir}")
    candidates = sorted(
        (
            str(path.resolve())
            for path in input_dir.glob("*.xlsx")
            if path.is_file() and not path.name.startswith("~$")
        ),
        key=lambda value: value.casefold(),
    )
    if len(candidates) != 1:
        raise RuntimeError(
            "未指定 --input-xlsx 时，输入目录必须恰好包含一个 XLSX；"
            f"实际找到 {len(candidates)} 个: {candidates}"
        )
    return candidates[0]


def _write_unique_work_input(evaluation_records, path):
    table = pd.DataFrame({
        "Compound ID": [
            record["processed_compound_id"]
            for record in evaluation_records
        ],
        "SMILES": [
            record["canonical_smiles"]
            for record in evaluation_records
        ],
    })
    table.to_csv(path, index=False, encoding="utf-8-sig")


def _safe_correlation(first, second):
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if first.size < 2:
        return float("nan")
    if np.all(first == first[0]) or np.all(second == second[0]):
        return float("nan")
    return float(np.corrcoef(first, second)[0, 1])


def _calculate_metrics(
    results,
    input_measurement_count,
    duplicate_smiles_group_count,
):
    true_values = results["true_activity"].to_numpy(dtype=np.float64)
    predictions = results["predicted_activity"].to_numpy(dtype=np.float64)
    errors = predictions - true_values
    true_ranks = pd.Series(true_values).rank(method="average").to_numpy()
    prediction_ranks = pd.Series(predictions).rank(
        method="average"
    ).to_numpy()
    molecule_count = int(len(results))
    return {
        "input_measurement_count": int(input_measurement_count),
        "molecule_count": molecule_count,
        "duplicate_measurement_count": int(
            input_measurement_count - molecule_count
        ),
        "duplicate_smiles_group_count": int(
            duplicate_smiles_group_count
        ),
        "spearman": _safe_correlation(true_ranks, prediction_ranks),
        "mae": float(np.mean(np.abs(errors))),
    }


def _atomic_write_workbook(
    results,
    duplicate_results,
    summary_rows,
    path,
):
    path = os.path.abspath(os.fspath(path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    suffix = os.path.splitext(path)[1] or ".xlsx"
    temp_path = (
        f"{os.path.splitext(path)[0]}."
        f"{os.getpid()}-{uuid.uuid4().hex}.tmp{suffix}"
    )
    try:
        with pd.ExcelWriter(temp_path, engine="openpyxl") as writer:
            results.to_excel(writer, sheet_name="results", index=False)
            duplicate_results.to_excel(
                writer,
                sheet_name="duplicate_smiles",
                index=False,
            )
            pd.DataFrame(summary_rows).to_excel(
                writer,
                sheet_name="summary",
                index=False,
            )
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def _smooth_plot_curve(x_values, y_values):
    """用高斯核回归生成仅用于绘图的平滑曲线。"""
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    if (
        x_values.ndim != 1
        or y_values.ndim != 1
        or len(x_values) != len(y_values)
    ):
        raise ValueError("平滑曲线的横纵坐标必须是一维且长度一致")
    if len(x_values) < 3:
        return None
    if not (
        np.all(np.isfinite(x_values))
        and np.all(np.isfinite(y_values))
    ):
        raise ValueError("平滑曲线的横纵坐标必须全部为有限数值")
    if not np.all(np.diff(x_values) > 0.0):
        raise ValueError("平滑曲线的横坐标必须严格递增")

    x_span = float(x_values[-1] - x_values[0])
    bandwidth = max(
        PLOT_SMOOTHING_MIN_BANDWIDTH,
        x_span * PLOT_SMOOTHING_FRACTION,
    )
    dense_point_count = min(
        max(len(x_values) * 12, PLOT_SMOOTH_CURVE_MIN_POINTS),
        PLOT_SMOOTH_CURVE_MAX_POINTS,
    )
    smooth_x = np.linspace(
        float(x_values[0]),
        float(x_values[-1]),
        num=dense_point_count,
    )
    smooth_y = np.empty_like(smooth_x)

    # 分块计算以限制样本较多时的临时权重矩阵大小。
    chunk_size = max(1, min(256, 1_000_000 // len(x_values)))
    for start in range(0, dense_point_count, chunk_size):
        end = min(start + chunk_size, dense_point_count)
        scaled_distances = (
            smooth_x[start:end, np.newaxis]
            - x_values[np.newaxis, :]
        ) / bandwidth
        weights = np.exp(-0.5 * np.square(scaled_distances))
        smooth_y[start:end] = (
            weights @ y_values
        ) / weights.sum(axis=1)

    return smooth_x, smooth_y


def _atomic_write_plot(results, metrics, path):
    path = os.path.abspath(os.fspath(path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temp_path = (
        f"{os.path.splitext(path)[0]}."
        f"{os.getpid()}-{uuid.uuid4().hex}.tmp.png"
    )
    figure = None
    try:
        curve_order = results["curve_order"].to_numpy()
        true_values = results["true_activity"].to_numpy()
        predictions = results["predicted_activity"].to_numpy()
        compound_ids = results["compound_id"].astype(str).to_numpy()
        figure, axis = plt.subplots(figsize=(14, 7))
        true_curve = _smooth_plot_curve(curve_order, true_values)
        prediction_curve = _smooth_plot_curve(curve_order, predictions)
        axis.scatter(
            curve_order,
            true_values,
            color="#1f77b4",
            s=11.0,
            alpha=0.28,
            edgecolors="none",
            label=(
                "Activity Level (true)"
                if true_curve is None
                else None
            ),
            zorder=2,
        )
        axis.scatter(
            curve_order,
            predictions,
            color="#d62728",
            s=11.0,
            alpha=0.25,
            edgecolors="none",
            label="Model prediction" if prediction_curve is None else None,
            zorder=2,
        )
        if true_curve is not None:
            axis.plot(
                true_curve[0],
                true_curve[1],
                color="#1f77b4",
                linewidth=2.0,
                label="Activity Level (true)",
                zorder=3,
            )
        if prediction_curve is not None:
            axis.plot(
                prediction_curve[0],
                prediction_curve[1],
                color="#d62728",
                linewidth=1.6,
                alpha=0.85,
                label="Model prediction",
                zorder=3,
            )
        for activity_level in ACTIVITY_LEVEL_GUIDES:
            axis.axhline(
                float(activity_level),
                color="#777777",
                linewidth=0.7,
                linestyle="--",
                alpha=0.65,
            )
        tick_count = min(len(curve_order), MAX_PLOT_X_TICKS)
        tick_indices = np.linspace(
            0,
            len(curve_order) - 1,
            num=tick_count,
            dtype=int,
        )
        axis.set_xticks(curve_order[tick_indices])
        axis.set_xticklabels(
            compound_ids[tick_indices],
            rotation=60,
            ha="right",
            fontsize=8,
        )
        axis.set_xlabel("Compound ID (ordered by true activity, low to high)")
        axis.set_ylabel("Activity Level (0-1; higher is better)")
        axis.set_title("Activity Level Trend Comparison")
        axis.grid(axis="y", alpha=0.2)
        axis.legend()
        axis.text(
            0.015,
            0.98,
            (
                f"Spearman={metrics['spearman']:.4f}  "
                f"MAE={metrics['mae']:.4f}"
            ),
            transform=axis.transAxes,
            va="top",
        )
        axis.set_ylim(float(config.OUTPUT_MIN), float(config.OUTPUT_MAX))
        axis.margins(x=0.01)
        figure.tight_layout()
        figure.savefig(temp_path, dpi=200, format="png")
        os.replace(temp_path, path)
    finally:
        if figure is not None:
            plt.close(figure)
        if os.path.exists(temp_path):
            os.remove(temp_path)


def evaluate_activity_curve(
    input_xlsx=None,
    output_dir=ACTIVITY_EVALUATION_DIR,
    checkpoint_path=config.BEST_CHECKPOINT_PATH,
    device=None,
    num_workers=None,
):
    """完成表格校验、二维处理、预测、排序、统计和曲线输出。"""
    config.validate_config()
    _validate_evaluation_settings()
    input_xlsx = _resolve_input_xlsx(input_xlsx)
    output_dir = os.path.abspath(os.fspath(output_dir))
    os.makedirs(output_dir, exist_ok=True)
    pyg_dir = os.path.join(output_dir, PROCESSED_DIRNAME)
    prediction_result_path = os.path.join(
        output_dir,
        PREDICTION_RESULT_FILENAME,
    )
    prediction_error_path = os.path.join(
        output_dir,
        PREDICTION_ERROR_FILENAME,
    )
    evaluation_result_path = os.path.join(
        output_dir,
        EVALUATION_RESULT_FILENAME,
    )
    evaluation_plot_path = os.path.join(
        output_dir,
        EVALUATION_PLOT_FILENAME,
    )

    evaluation_records, duplicate_records, input_measurement_count = (
        _build_evaluation_records(input_xlsx)
    )
    with tempfile.TemporaryDirectory(
        prefix="activity_evaluation_input_",
        dir=output_dir,
    ) as temporary_input_dir:
        unique_input_path = os.path.join(
            temporary_input_dir,
            "unique_compounds.csv",
        )
        _write_unique_work_input(evaluation_records, unique_input_path)
        processing_summary = work_data.process_work_data(
            input_dir=unique_input_path,
            output_dir=pyg_dir,
            num_workers=num_workers,
        )
    expected_count = len(evaluation_records)
    processing_failed = int(processing_summary.get("processing_failed", 0))
    save_failed = int(processing_summary.get("save_failed", 0))
    error_count = int(processing_summary.get("error_count", 0))
    saved_pyg = int(processing_summary.get("saved_pyg", 0))
    if (
        processing_failed
        or save_failed
        or error_count
        or saved_pyg != expected_count
    ):
        raise RuntimeError(
            "评估数据没有全部成功转换为 PyG: "
            f"期望={expected_count}, 保存={saved_pyg}, "
            f"处理失败={processing_failed}, 保存失败={save_failed}, "
            f"错误={error_count}"
        )

    expected_files = [
        f"{record['processed_compound_id']}.pt"
        for record in evaluation_records
    ]
    missing_files = [
        file_name for file_name in expected_files
        if not os.path.isfile(os.path.join(pyg_dir, file_name))
    ]
    if missing_files:
        raise RuntimeError(f"评估 PyG 文件缺失: {missing_files}")

    prediction_records = predictor.predict_activity(
        data_dir=pyg_dir,
        file_names=expected_files,
        checkpoint_path=checkpoint_path,
        device=device,
        result_path=prediction_result_path,
        error_path=prediction_error_path,
    )
    prediction_by_id = {}
    for prediction_record in prediction_records:
        compound_id = str(prediction_record["compound_id"])
        if compound_id in prediction_by_id:
            raise RuntimeError(f"预测结果 Compound ID 重复: {compound_id}")
        prediction_by_id[compound_id] = float(
            prediction_record["predicted_activity"]
        )

    for record in evaluation_records:
        processed_id = record["processed_compound_id"]
        if processed_id not in prediction_by_id:
            raise RuntimeError(f"缺少预测结果: {processed_id}")
        record["predicted_activity"] = prediction_by_id[processed_id]
    if len(prediction_by_id) != expected_count:
        raise RuntimeError(
            "预测结果数量与评估分子数量不一致: "
            f"{len(prediction_by_id)} != {expected_count}"
        )

    results = pd.DataFrame(evaluation_records)
    results["prediction_error"] = (
        results["predicted_activity"] - results["true_activity"]
    )
    results["absolute_error"] = results["prediction_error"].abs()
    results = results.sort_values(
        "true_activity",
        ascending=True,
        kind="mergesort",
    ).reset_index(drop=True)
    results.insert(0, "curve_order", np.arange(1, len(results) + 1))
    results = results[[
        "curve_order",
        "compound_id",
        "source_compound_ids",
        "canonical_smiles",
        "measurement_count",
        "mic_observations",
        "micromolar_observations",
        "true_activity",
        "predicted_activity",
        "prediction_error",
        "absolute_error",
    ]]

    duplicate_results = pd.DataFrame(
        duplicate_records,
        columns=DUPLICATE_SMILES_COLUMNS,
    )
    duplicate_smiles_group_count = int(
        sum(record["measurement_count"] > 1 for record in evaluation_records)
    )
    metrics = _calculate_metrics(
        results,
        input_measurement_count,
        duplicate_smiles_group_count,
    )
    summary_rows = [
        {"item": "input_xlsx", "value": input_xlsx},
        {"item": "checkpoint", "value": os.path.abspath(checkpoint_path)},
        {
            "item": "mapping",
            "value": (
                "convert every supported MIC unit to μM first; "
                "map MIC to fixed Activity Levels: 0<MIC<=2 to 1.0, "
                "2<MIC<=12.5 to 0.8, 12.5<MIC<=32 to 0.6, "
                "32<MIC<=64 to 0.4, 64<MIC<128 to 0.2, "
                "128<=MIC<256 to 0.1, and MIC>=256 μM to 0.0"
            ),
        },
        {
            "item": "deduplication",
            "value": (
                "direct arithmetic mean of per-measurement Activity "
                "Levels by canonical isomeric SMILES"
            ),
        },
    ]
    summary_rows.extend(
        {"item": key, "value": value}
        for key, value in metrics.items()
    )
    _atomic_write_workbook(
        results,
        duplicate_results,
        summary_rows,
        evaluation_result_path,
    )
    _atomic_write_plot(results, metrics, evaluation_plot_path)

    return {
        **metrics,
        "input_xlsx": input_xlsx,
        "pyg_dir": pyg_dir,
        "prediction_result_path": prediction_result_path,
        "prediction_error_path": prediction_error_path,
        "evaluation_result_path": evaluation_result_path,
        "evaluation_plot_path": evaluation_plot_path,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "将已知 MIC 统一换算为 μM 和七档 Activity Level，"
            "并生成真实/预测活性排序曲线"
        ),
    )
    parser.add_argument(
        "--input-xlsx",
        default=None,
        help=(
            "包含 Compound ID、SMILES、Activity、MIC Unit 的 XLSX；"
            "省略时要求 test_unknownactivity 根目录恰好有一个 XLSX"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=ACTIVITY_EVALUATION_DIR,
        help="评估结果、预测结果和临时 PyG 的独立输出目录",
    )
    parser.add_argument(
        "--checkpoint",
        default=config.BEST_CHECKPOINT_PATH,
        help="用于外部评估的最佳检查点",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="覆盖默认推理设备，例如 cpu 或 cuda:0",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="请求的二维切割进程数",
    )
    return parser.parse_args()


def main():
    multiprocessing.freeze_support()
    args = parse_args()
    summary = evaluate_activity_curve(
        input_xlsx=args.input_xlsx,
        output_dir=args.output_dir,
        checkpoint_path=args.checkpoint,
        device=args.device,
        num_workers=args.workers,
    )
    print("\n--- MIC 活性曲线评估完成 ---")
    print(f"输入 MIC 记录: {summary['input_measurement_count']}")
    print(f"唯一化合物: {summary['molecule_count']}")
    print(f"合并重复记录: {summary['duplicate_measurement_count']}")
    print(f"重复 SMILES 组: {summary['duplicate_smiles_group_count']}")
    print(f"Spearman: {summary['spearman']:.6f}")
    print(f"MAE: {summary['mae']:.6f}")
    print(f"结果表格: {summary['evaluation_result_path']}")
    print(f"曲线图片: {summary['evaluation_plot_path']}")


if __name__ == "__main__":
    main()
