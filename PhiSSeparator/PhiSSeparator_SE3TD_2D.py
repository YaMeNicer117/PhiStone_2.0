# =============================================================================
# 纯 SMILES 表格 -> SE3TD 2D PyG 数据
# =============================================================================

import os
import gc
import json
import math
import multiprocessing
import sys
from pathlib import Path

# 防止多进程 worker 与底层数值库争抢线程。
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import torch

from rdkit import Chem, RDLogger
from torch_geometric.data import Data
from tqdm import tqdm

_PROJECT_IMPORT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_IMPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_IMPORT_ROOT))

from PhiSSeparator import utils


torch.set_num_threads(1)

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# =============================================================================
# 配置
# =============================================================================

NUM_COMPOUNDS_TO_PROCESS = None
TASK_TIMEOUT_SECONDS = 120

MAX_ALLOWED_WORKERS = 24
DETECTED_CORES = int(
    os.environ.get("SLURM_CPUS_PER_TASK", multiprocessing.cpu_count())
)
NUM_WORKERS = min(DETECTED_CORES, MAX_ALLOWED_WORKERS)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_ROOT = os.path.join(PROJECT_ROOT, "Datas")
if not os.path.isdir(DATA_ROOT):
    raise RuntimeError(f"无法定位项目数据目录: {DATA_ROOT}")

# CSV 读取单表；Excel 读取全部工作表。后续切换数据集时只需修改该路径。
INPUT_FILE = os.path.join(
    DATA_ROOT,
    "Pre_datas",
    "Antitubercular Molecules",
    "activity_mic_level_mean.xlsx",
)

PROCESSED_DATA_ROOT = os.path.join(DATA_ROOT, "Processed_datas")
OUTPUT_DIR = os.path.join(
    PROCESSED_DATA_ROOT,
    "SE3TD_2D_128_Pygdata",
)
PYG_DATA_DIR = os.path.join(OUTPUT_DIR, "pyg_pending")
SUMMARY_FILE = os.path.join(OUTPUT_DIR, "processing_summary.json")
ERROR_LOG_FILE = os.path.join(OUTPUT_DIR, "processing_errors.csv")

# 归一化统计量必须预先存在；两类共享粗语料可由本流程首次创建。
SHARED_DATA_DIR = os.path.join(
    PROCESSED_DATA_ROOT,
    "SE3TD_128_Shared_data",
)
NORM_STATS_FILE = os.path.join(
    SHARED_DATA_DIR,
    "normalization_stats.json",
)
RAW_CORPUS_FILE = os.path.join(
    SHARED_DATA_DIR,
    "fragment_raw_corpus.npz",
)
RAW_CORPUS_METADATA_FILE = os.path.join(
    SHARED_DATA_DIR,
    "fragment_raw_corpus_metadata.json",
)
RAW_ATOM_CORPUS_FILE = os.path.join(
    SHARED_DATA_DIR,
    "fragment_atom_raw_corpus.npz",
)
RAW_ATOM_CORPUS_METADATA_FILE = os.path.join(
    SHARED_DATA_DIR,
    "fragment_atom_raw_corpus_metadata.json",
)

FEATURE_COLUMNS_2D = (
    "avg_ecc",
    "ecc_range",
    "avg_logp",
    "tpsa",
    "avg_charge",
    "HBA",
    "HBD",
    "Aromatic",
    "Hydrophobe",
    "PosIonizable",
    "NegIonizable",
)
COUNT_FEATURES = {
    "HBA",
    "HBD",
    "Aromatic",
    "Hydrophobe",
    "PosIonizable",
    "NegIonizable",
}

WORKER_STAT_KEYS = (
    "fragmentation_success",
    "missing_compound_id",
    "empty_smiles",
    "invalid_smiles",
    "fragmentation_failed",
    "port_failed",
    "dropped_fragments",
    "fragment_warnings",
    "removed_ion_fragments",
    "reconstruction_failed",
    "retained_metal_nodes",
)

ERROR_LOG_COLUMNS = (
    "compound_id",
    "sheet_name",
    "row_number",
    "error_type",
    "error",
)

WINDOWS_INVALID_FILENAME_CHARS = set('<>:"/\\|?*')
WINDOWS_RESERVED_FILENAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}


# =============================================================================
# 输入表格
# =============================================================================

def _normalize_text(value):
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value).strip()


def _parse_activity_value(value):
    if value is None:
        raise ValueError("值为空")
    if isinstance(value, (list, tuple, dict, set, np.ndarray)):
        raise ValueError("值不是单个数值")
    try:
        is_missing = pd.isna(value)
    except (TypeError, ValueError):
        is_missing = False
    if isinstance(is_missing, (bool, np.bool_)) and bool(is_missing):
        raise ValueError("值为空")

    if isinstance(value, (bool, np.bool_)):
        raise ValueError("布尔值不是有效活性值")

    text = str(value).strip()
    if not text:
        raise ValueError("值为空")
    try:
        numeric_value = float(text)
    except (TypeError, ValueError):
        raise ValueError(f"无法解析为数值: {value!r}")
    if not math.isfinite(numeric_value):
        raise ValueError("值必须是有限数值")
    if not 0.0 <= numeric_value <= 3.0:
        raise ValueError(
            f"值超出允许范围 [0, 3]: {numeric_value}"
        )
    return numeric_value


def _assign_unique_compound_ids(records):
    """为重复 Compound ID 分配稳定且不覆盖原始 ID 的数字后缀。"""
    reserved_ids = {
        record["compound_id"].casefold()
        for record in records
        if record["compound_id"]
    }
    assigned_ids = set()
    next_suffix_by_id = {}
    renamed_count = 0

    for record in records:
        compound_id = record["compound_id"]
        if not compound_id:
            continue

        normalized_id = compound_id.casefold()
        if normalized_id not in assigned_ids:
            assigned_ids.add(normalized_id)
            continue

        suffix = next_suffix_by_id.get(normalized_id, 2)
        while True:
            candidate = f"{compound_id}_{suffix}"
            normalized_candidate = candidate.casefold()
            if (
                normalized_candidate not in reserved_ids
                and normalized_candidate not in assigned_ids
            ):
                record["compound_id"] = candidate
                assigned_ids.add(normalized_candidate)
                next_suffix_by_id[normalized_id] = suffix + 1
                renamed_count += 1
                break
            suffix += 1

    return renamed_count


def _read_csv_table(input_path):
    try:
        return pd.read_csv(
            input_path,
            dtype=object,
            keep_default_na=False,
            encoding="utf-8-sig",
        )
    except UnicodeDecodeError:
        return pd.read_csv(
            input_path,
            dtype=object,
            keep_default_na=False,
            encoding="gb18030",
        )


def load_input_records(input_path):
    """读取单个 CSV 或 Excel，并返回全部工作表中的 2D 记录。"""
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"未找到输入表格: {input_path}")

    extension = os.path.splitext(input_path)[1].lower()
    if extension == ".csv":
        tables = {
            os.path.basename(input_path): _read_csv_table(input_path)
        }
    elif extension in {".xls", ".xlsx"}:
        tables = pd.read_excel(
            input_path,
            sheet_name=None,
            dtype=object,
            keep_default_na=False,
        )
    else:
        raise ValueError(
            f"不支持的输入格式 {extension!r}；仅支持 CSV/XLS/XLSX"
        )

    records = []
    for sheet_name, original_table in tables.items():
        if original_table is None:
            continue

        table = original_table.copy()
        normalized_columns = [
            str(column).strip() for column in table.columns
        ]
        if len(set(normalized_columns)) != len(normalized_columns):
            raise RuntimeError(
                f"工作表 {sheet_name!r} 存在清理空格后重名的列"
            )
        table.columns = normalized_columns

        # 完全空白的工作表不视为输入表。
        if len(table.columns) == 0:
            continue

        required_columns = {
            "Compound ID",
            "Canonical SMILES",
            "Mean Activity Level",
        }
        missing_columns = required_columns - set(table.columns)
        if missing_columns:
            raise RuntimeError(
                f"工作表 {sheet_name!r} 缺少必需列: "
                f"{sorted(missing_columns)}"
            )

        for row_index, row in table.iterrows():
            row_number = int(row_index) + 2
            try:
                activity = _parse_activity_value(
                    row["Mean Activity Level"]
                )
            except ValueError as exc:
                raise RuntimeError(
                    f"工作表 {sheet_name!r} 第 {row_number} 行 "
                    f"Mean Activity Level 非法: {exc}"
                ) from exc
            records.append({
                "compound_id": _normalize_text(row["Compound ID"]),
                "smiles": _normalize_text(row["Canonical SMILES"]),
                "sheet_name": str(sheet_name),
                "row_number": row_number,
                "activity": activity,
            })

    if not records:
        raise RuntimeError("输入文件中没有可处理的数据行")

    return records


# =============================================================================
# 归一化参数
# =============================================================================

def load_2d_normalization_params(stats_file):
    if not os.path.isfile(stats_file):
        raise FileNotFoundError(
            f"未找到 3D 归一化统计文件: {stats_file}"
        )

    try:
        with open(stats_file, "r", encoding="utf-8") as file_obj:
            stats = json.load(file_obj)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"归一化统计文件无法读取: {stats_file}: {exc}"
        ) from exc

    if not isinstance(stats, dict) or not stats:
        raise RuntimeError(f"归一化统计文件为空或格式非法: {stats_file}")

    params = {}
    for feature_name in FEATURE_COLUMNS_2D:
        feature_stats = stats.get(feature_name)
        if not isinstance(feature_stats, dict):
            raise RuntimeError(
                f"归一化统计文件缺少特征: {feature_name}"
            )
        if not {"count", "mean", "M2"} <= set(feature_stats):
            raise RuntimeError(
                f"特征 {feature_name} 的归一化统计格式非法"
            )

        count = feature_stats["count"]
        mean = feature_stats["mean"]
        m2 = feature_stats["M2"]
        if (
            isinstance(count, bool)
            or not isinstance(count, (int, float))
            or count <= 0
            or isinstance(mean, bool)
            or not isinstance(mean, (int, float))
            or not math.isfinite(float(mean))
            or isinstance(m2, bool)
            or not isinstance(m2, (int, float))
            or not math.isfinite(float(m2))
            or m2 < 0
        ):
            raise RuntimeError(
                f"特征 {feature_name} 的归一化统计数值非法"
            )

        std = math.sqrt(float(m2) / float(count)) if count >= 2 else 0.0
        params[feature_name] = {
            "mean": float(mean),
            "std": float(std),
        }

    return params


def normalize_shard_features(shard, normalization_params):
    feature_values = []
    for feature_name in FEATURE_COLUMNS_2D:
        raw_value = shard.get(feature_name, 0.0)
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"节点特征 {feature_name} 不是数值: {raw_value!r}"
            ) from exc
        if not math.isfinite(value):
            raise ValueError(
                f"节点特征 {feature_name} 不是有限数值: {value}"
            )

        params = normalization_params[feature_name]
        mean = params["mean"]
        std = params["std"]
        if feature_name in COUNT_FEATURES:
            safe_std = max(std, 1.0)
        else:
            safe_std = std if std > 1e-5 else 1.0
        normalized_value = (value - mean) / safe_std
        feature_values.append(
            float(np.clip(normalized_value, -6.0, 6.0))
        )
    return feature_values


# =============================================================================
# 2D 分片
# =============================================================================

def build_2d_metal_shard_record(metal_mol):
    """构造不依赖三维坐标的金属 shard。"""
    clean_mol = Chem.Mol(metal_mol)
    try:
        clean_mol = Chem.RemoveHs(clean_mol)
    except Exception:
        pass

    standardized_smiles = utils.get_standardized_metal_smiles(clean_mol)
    if standardized_smiles is None:
        try:
            standardized_smiles = Chem.MolToSmiles(
                clean_mol,
                canonical=True,
                isomericSmiles=True,
            )
        except Exception:
            return None
    if not standardized_smiles:
        return None

    try:
        feature_charge = utils.get_metal_feature_charge(clean_mol)
    except Exception:
        return None

    return {
        "mol": clean_mol,
        "clean_mol": clean_mol,
        "smiles": standardized_smiles,
        "avg_ecc": 0.0,
        "ecc_range": 0.0,
        "avg_logp": 0.0,
        "tpsa": 0.0,
        "avg_charge": float(feature_charge),
        "HBA": 0.0,
        "HBD": 0.0,
        "Aromatic": 0.0,
        "Hydrophobe": 0.0,
        "PosIonizable": 0.0,
        "NegIonizable": 0.0,
        "ports": [],
        "hac": utils.calculate_heavy_atom_count(clean_mol),
        "ring_count": utils.calculate_ring_count(clean_mol),
        "is_metal": True,
    }


def process_single_fragment_2d(
    fragment,
    compound_id,
    source_fragment_id,
    id_start_offset,
):
    fragment_stats = {
        "removed_ion_fragments": 0,
        "reconstruction_failed": 0,
        "retained_metal_nodes": 0,
    }

    try:
        prepared = utils.prepare_molecule_for_cutting(
            fragment,
            source_fragment_id=source_fragment_id,
            origin_type="ligand",
        )
        fragment_stats["removed_ion_fragments"] = prepared[
            "removed_fragment_count"
        ]
        fragment_stats["reconstruction_failed"] = prepared[
            "reconstruction_failed_count"
        ]
        fragment_stats["retained_metal_nodes"] = len(
            prepared["metal_fragments"]
        )

        try:
            original_smiles = Chem.MolToSmiles(
                fragment,
                canonical=True,
                isomericSmiles=True,
            )
        except Exception:
            original_smiles = "<ERROR>"

        dropped_count = 0
        temporary_results = []

        for component in prepared["organic_components"]:
            main_mol = component["main_mol_for_matching"]
            final_fragments = utils.execute_fragmentation_pipeline(
                main_mol,
                component["bond_info"],
                track_ports=True,
            )
            result_storage = []
            dropped_count += utils.calculate_descriptors_and_store(
                storage=result_storage,
                mol_data={
                    "IDs": compound_id,
                    "Smiles": original_smiles,
                },
                original_mol_with_conf=fragment,
                main_mol_for_matching=main_mol,
                final_covalent_fragments=final_fragments,
                original_eccentricities=component[
                    "original_eccentricities"
                ],
                original_logp_contribs=component[
                    "original_logp_contribs"
                ],
                original_tpsa_contribs=component[
                    "original_tpsa_contribs"
                ],
                original_charges=component["original_charges"],
                origin_type="ligand",
            )
            if result_storage:
                temporary_results.extend(result_storage[0]["results"])

        for metal_fragment in prepared["metal_fragments"]:
            metal_record = build_2d_metal_shard_record(metal_fragment)
            if metal_record is None:
                dropped_count += 1
            else:
                temporary_results.append(metal_record)

        shards = []
        for local_index, shard in enumerate(temporary_results):
            shard["origin"] = "ligand"
            shard["Shard ID"] = id_start_offset + local_index
            shards.append(shard)

        if shards:
            return shards, None, dropped_count, fragment_stats
        return [], "未生成有效 shard", dropped_count, fragment_stats
    except (utils.PortValidationError, utils.LigandSecondaryCleaningError):
        raise
    except Exception as exc:
        return [], str(exc), 1, fragment_stats


def _empty_worker_result(compound_info):
    return {
        "success": False,
        "compound_info": compound_info,
        "processed_groups": None,
        "edge_pairs": None,
        "error_type": "",
        "error": "",
        "stats": {key: 0 for key in WORKER_STAT_KEYS},
    }


def worker_process_compound(compound_info):
    """在独立进程中完成一个 SMILES 的切割和共价建边。"""
    RDLogger.logger().setLevel(RDLogger.CRITICAL)
    result = _empty_worker_result(compound_info)

    compound_id = compound_info["compound_id"]
    smiles = compound_info["smiles"]
    if not compound_id:
        result["error_type"] = "missing_compound_id"
        result["error"] = "Compound ID 为空"
        result["stats"]["missing_compound_id"] = 1
        return result
    if not smiles:
        result["error_type"] = "empty_smiles"
        result["error"] = "SMILES 为空"
        result["stats"]["empty_smiles"] = 1
        return result

    try:
        molecule = Chem.MolFromSmiles(smiles, sanitize=True)
    except Exception as exc:
        molecule = None
        result["error"] = f"SMILES 解析异常: {exc}"
    if molecule is None or molecule.GetNumAtoms() == 0:
        result["error_type"] = "invalid_smiles"
        if not result["error"]:
            result["error"] = "RDKit 无法解析 SMILES"
        result["stats"]["invalid_smiles"] = 1
        return result

    try:
        original_fragments = list(
            Chem.GetMolFrags(
                molecule,
                asMols=True,
                sanitizeFrags=False,
            )
        )
    except Exception as exc:
        result["error_type"] = "fragmentation_failed"
        result["error"] = f"无法拆分不连通组分: {exc}"
        result["stats"]["fragmentation_failed"] = 1
        return result
    if not original_fragments:
        result["error_type"] = "fragmentation_failed"
        result["error"] = "SMILES 未产生分子组分"
        result["stats"]["fragmentation_failed"] = 1
        return result

    processed_groups = []
    shard_id_counter = 0
    total_dropped_fragments = 0
    fragment_warnings = []

    try:
        for fragment_index, original_fragment in enumerate(
            original_fragments
        ):
            source_fragment_id = (
                f"{compound_id}:ligand:{fragment_index}"
            )
            shards, warning, dropped_count, fragment_stats = (
                process_single_fragment_2d(
                    original_fragment,
                    compound_id,
                    source_fragment_id,
                    shard_id_counter,
                )
            )
            total_dropped_fragments += dropped_count
            for stat_name, stat_value in fragment_stats.items():
                result["stats"][stat_name] += int(stat_value)
            if warning:
                fragment_warnings.append(
                    f"组分 {fragment_index}: {warning}"
                )
            if shards:
                processed_groups.append({
                    "original_fragment": original_fragment,
                    "shards": shards,
                })
                shard_id_counter += len(shards)

        result["stats"]["dropped_fragments"] = (
            total_dropped_fragments
        )
        result["stats"]["fragment_warnings"] = len(fragment_warnings)
        if total_dropped_fragments > 10:
            raise utils.PortValidationError("切割质量控制未通过")

        all_shards = [
            shard
            for group in processed_groups
            for shard in group["shards"]
        ]
        if not all_shards:
            result["error_type"] = "fragmentation_failed"
            result["error"] = (
                "; ".join(fragment_warnings)
                if fragment_warnings
                else "未生成有效 shard"
            )
            result["stats"]["fragmentation_failed"] = 1
            return result

        # 必须在移除 original_fragment 和 shard mol 前建立共价连接。
        legacy_edges = utils.build_legacy_covalent_edges(
            processed_groups
        )
        edge_pairs = sorted({
            tuple(sorted((int(first), int(second))))
            for first, second in legacy_edges
            if int(first) != int(second)
        })

    except utils.PortValidationError as exc:
        result["error_type"] = "port_failed"
        result["error"] = str(exc)
        result["stats"]["port_failed"] = 1
        return result
    except utils.LigandSecondaryCleaningError as exc:
        result["error_type"] = "fragmentation_failed"
        result["error"] = str(exc)
        result["stats"]["fragmentation_failed"] = 1
        return result
    except Exception as exc:
        result["error_type"] = "fragmentation_failed"
        result["error"] = str(exc)
        result["stats"]["fragmentation_failed"] = 1
        return result

    for group in processed_groups:
        group.pop("original_fragment", None)
        for shard in group["shards"]:
            shard.pop("mol", None)
            shard.pop("clean_mol", None)
            shard.pop("ports", None)

    result["processed_groups"] = processed_groups
    result["edge_pairs"] = edge_pairs
    result["success"] = True
    result["stats"]["fragmentation_success"] = 1
    return result


# =============================================================================
# PyG 组装与保存
# =============================================================================

def validate_compound_id_for_filename(compound_id):
    if not compound_id:
        raise ValueError("Compound ID 为空")
    if compound_id in {".", ".."}:
        raise ValueError("Compound ID 不能是相对路径标记")
    if compound_id.endswith((" ", ".")):
        raise ValueError(
            "Compound ID 不能以空格或句点结尾"
        )
    invalid_chars = sorted(
        set(compound_id).intersection(WINDOWS_INVALID_FILENAME_CHARS)
    )
    if invalid_chars:
        raise ValueError(
            f"Compound ID 含 Windows 文件名非法字符: {invalid_chars}"
        )
    stem = compound_id.split(".", 1)[0].upper()
    if stem in WINDOWS_RESERVED_FILENAMES:
        raise ValueError(
            f"Compound ID 是 Windows 保留文件名: {compound_id}"
        )


def build_output_path(compound_id):
    validate_compound_id_for_filename(compound_id)
    output_root = os.path.abspath(PYG_DATA_DIR)
    target_dir = os.path.abspath(
        os.path.join(output_root, compound_id[0])
    )
    target_path = os.path.abspath(
        os.path.join(target_dir, f"{compound_id}.pt")
    )
    if os.path.commonpath([output_root, target_path]) != output_root:
        raise ValueError("Compound ID 导致输出路径越界")
    return target_path


def assemble_pyg_data(
    worker_result,
    normalization_params,
    *,
    include_activity=True,
):
    compound_info = worker_result["compound_info"]
    all_shards = [
        shard
        for group in worker_result["processed_groups"]
        for shard in group["shards"]
    ]
    all_shards.sort(key=lambda shard: shard["Shard ID"])

    shard_ids = [int(shard["Shard ID"]) for shard in all_shards]
    if shard_ids != list(range(len(all_shards))):
        raise ValueError("Shard ID 必须从 0 开始连续编号")

    node_features = []
    metal_mask_values = []
    hac_values = []
    ring_count_values = []
    fragment_smiles = []

    for shard in all_shards:
        node_features.append(
            normalize_shard_features(shard, normalization_params)
        )
        metal_mask_values.append(
            bool(shard.get("is_metal", False))
        )
        hac_values.append(int(shard["hac"]))
        ring_count_values.append(int(shard["ring_count"]))

        shard_smiles = shard.get("smiles")
        if not isinstance(shard_smiles, str) or not shard_smiles:
            raise ValueError("节点缺少化学规范 SMILES")
        if shard_smiles in utils.RAW_FRAGMENT_SPECIAL_TOKENS:
            raise ValueError(
                f"真实节点不能使用特殊 SMILES: {shard_smiles}"
            )
        fragment_smiles.append(shard_smiles)

    num_nodes = len(all_shards)
    x = torch.tensor(node_features, dtype=torch.float)
    metal_mask = torch.tensor(metal_mask_values, dtype=torch.bool)
    hac = torch.tensor(hac_values, dtype=torch.long)
    ring_count = torch.tensor(ring_count_values, dtype=torch.long)

    undirected_edges = []
    for first, second in worker_result["edge_pairs"]:
        first = int(first)
        second = int(second)
        if (
            first < 0
            or second < 0
            or first >= num_nodes
            or second >= num_nodes
        ):
            raise ValueError(
                f"共价边节点越界: {(first, second)}"
            )
        if first == second:
            continue
        undirected_edges.append([first, second])

    if undirected_edges:
        edge_index = torch.tensor(
            undirected_edges,
            dtype=torch.long,
        ).t().contiguous()
        edge_index = torch.cat(
            [edge_index, edge_index.flip(0)],
            dim=1,
        )
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    data_fields = {
        "x": x,
        "edge_index": edge_index,
        "compound_id": compound_info["compound_id"],
        "smiles": compound_info["smiles"],
        "fragment_smiles": fragment_smiles,
        "metal_mask": metal_mask,
        "hac": hac,
        "ring_count": ring_count,
    }
    if include_activity:
        data_fields["y"] = torch.tensor(
            [float(compound_info["activity"])],
            dtype=torch.float,
        )
    data = Data(**data_fields)

    return data, fragment_smiles


def save_pyg_data_atomic(data, target_path):
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    temp_path = f"{target_path}.tmp"
    try:
        torch.save(data, temp_path)
        os.replace(temp_path, target_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# =============================================================================
# 报告与共享粗语料
# =============================================================================

def make_error_record(compound_info, error_type, error):
    return {
        "compound_id": compound_info.get("compound_id", ""),
        "sheet_name": compound_info.get("sheet_name", ""),
        "row_number": compound_info.get("row_number", ""),
        "error_type": error_type,
        "error": str(error),
    }


def save_reports(summary, errors):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(SUMMARY_FILE, "w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, ensure_ascii=False, indent=2)

    error_table = pd.DataFrame(errors, columns=ERROR_LOG_COLUMNS)
    error_table.to_csv(
        ERROR_LOG_FILE,
        index=False,
        encoding="utf-8-sig",
    )


def update_shared_raw_corpora(successful_vocabulary):
    fragment_corpus, fragment_metadata, merge_stats = (
        utils.merge_raw_fragment_corpus(
            successful_vocabulary,
            RAW_CORPUS_FILE,
            RAW_CORPUS_METADATA_FILE,
        )
    )
    if fragment_corpus is None:
        return {
            **merge_stats,
            "atom_corpus_num_smiles": 0,
            "atom_corpus_num_atoms": 0,
            "atom_corpus_num_bonds": 0,
            "atom_corpus_resolver_added_smiles": 0,
            "atom_corpus_rewritten": False,
        }

    final_fragment_smiles = (
        fragment_corpus["smiles"].astype(str).tolist()
    )

    atom_metadata = utils.save_raw_fragment_atom_corpus(
        final_fragment_smiles,
        RAW_ATOM_CORPUS_FILE,
        RAW_ATOM_CORPUS_METADATA_FILE,
        source_fragment_corpus_sha256=fragment_metadata[
            "corpus_sha256"
        ],
        preserve_resolver_entries=True,
    )
    _, verified_atom_metadata = (
        utils.load_raw_fragment_atom_corpus(
            RAW_ATOM_CORPUS_FILE,
            RAW_ATOM_CORPUS_METADATA_FILE,
        )
    )
    if verified_atom_metadata["corpus_sha256"] != (
        atom_metadata["corpus_sha256"]
    ):
        raise RuntimeError("共享原子粗语料保存后的校验值发生变化")
    if atom_metadata.get("source_fragment_corpus_sha256") != (
        fragment_metadata["corpus_sha256"]
    ):
        raise RuntimeError("共享原子粗语料未关联当前片段粗语料")

    return {
        **merge_stats,
        "atom_corpus_num_smiles": atom_metadata["num_smiles"],
        "atom_corpus_num_atoms": atom_metadata["num_atoms"],
        "atom_corpus_num_bonds": atom_metadata["num_bonds"],
        "atom_corpus_resolver_added_smiles": atom_metadata[
            "resolver_added_smiles_count"
        ],
        "atom_corpus_rewritten": True,
    }


# =============================================================================
# 主程序
# =============================================================================

def main():
    RDLogger.DisableLog("rdApp.*")
    multiprocessing.freeze_support()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PYG_DATA_DIR, exist_ok=True)
    os.makedirs(SHARED_DATA_DIR, exist_ok=True)

    print(f"输入文件: {os.path.abspath(INPUT_FILE)}")
    print(f"PyG 输出目录: {os.path.abspath(PYG_DATA_DIR)}")
    print(f"共享数据目录: {os.path.abspath(SHARED_DATA_DIR)}")
    print(f"进程数: {NUM_WORKERS}")

    normalization_params = load_2d_normalization_params(
        NORM_STATS_FILE
    )
    print(
        f"已从 3D 统计文件加载 "
        f"{len(normalization_params)} 个 2D 特征参数。"
    )

    raw_corpus_exists = os.path.isfile(RAW_CORPUS_FILE)
    raw_corpus_metadata_exists = os.path.isfile(
        RAW_CORPUS_METADATA_FILE
    )
    if raw_corpus_exists != raw_corpus_metadata_exists:
        missing_path = (
            RAW_CORPUS_METADATA_FILE
            if raw_corpus_exists
            else RAW_CORPUS_FILE
        )
        raise FileNotFoundError(
            "共享片段粗语料 NPZ 与元数据必须同时存在或同时不存在；"
            f"缺少: {missing_path}"
        )

    existing_vocabulary = set()
    shared_vocab_size_before = 0
    if raw_corpus_exists:
        shared_corpus, shared_corpus_metadata = (
            utils.load_raw_fragment_corpus(
                RAW_CORPUS_FILE,
                RAW_CORPUS_METADATA_FILE,
            )
        )
        existing_vocabulary = set(
            shared_corpus["smiles"].astype(str).tolist()
        )
        shared_vocab_size_before = int(
            shared_corpus_metadata["num_smiles"]
        )
        del shared_corpus
        gc.collect()
        print(
            f"共享片段粗语料校验通过，现有词数: "
            f"{shared_vocab_size_before}"
        )
    else:
        print("未发现共享片段粗语料，将从空词表开始。")

    records = load_input_records(INPUT_FILE)
    if (
        isinstance(NUM_COMPOUNDS_TO_PROCESS, int)
        and NUM_COMPOUNDS_TO_PROCESS > 0
    ):
        records = records[:NUM_COMPOUNDS_TO_PROCESS]
    if not records:
        raise RuntimeError("没有待处理记录")

    duplicate_compound_ids_renamed = _assign_unique_compound_ids(
        records
    )
    print(
        f"共读取 {len(records)} 条记录；活性字段: y；"
        f"重复 ID 重命名: {duplicate_compound_ids_renamed}"
    )

    summary = {
        "input_file": os.path.abspath(INPUT_FILE),
        "planned_records": len(records),
        "duplicate_compound_ids_renamed": (
            duplicate_compound_ids_renamed
        ),
        "saved_pyg": 0,
        "save_failed": 0,
        "timeout": 0,
        "process_exception": 0,
        "activity_field": "y",
        "activity_range": [0.0, 3.0],
        "feature_columns": list(FEATURE_COLUMNS_2D),
        "feature_dimension": len(FEATURE_COLUMNS_2D),
        "normalization_source": os.path.abspath(NORM_STATS_FILE),
        "shared_corpus_source": os.path.abspath(RAW_CORPUS_FILE),
        "shared_atom_corpus_source": os.path.abspath(
            RAW_ATOM_CORPUS_FILE
        ),
        "shared_atom_corpus_metadata_source": os.path.abspath(
            RAW_ATOM_CORPUS_METADATA_FILE
        ),
        "shared_vocab_size_before": shared_vocab_size_before,
        "new_vocab_candidates": 0,
        "new_vocab_added": 0,
        "shared_vocab_size_after": shared_vocab_size_before,
        "corpus_created": False,
        "corpus_rewritten": False,
        "corpus_update_failed": 0,
        "atom_corpus_num_smiles": 0,
        "atom_corpus_num_atoms": 0,
        "atom_corpus_num_bonds": 0,
        "atom_corpus_resolver_added_smiles": 0,
        "atom_corpus_rewritten": False,
        "embedding_status": "pending_offline_128d",
        "atom_embedding_status": "pending_offline_64d",
        "future_3d_rerun_warning": (
            "当前 3D 脚本重跑时会从空 vocabulary 重写共享粗语料，"
            "可能移除本次 2D 新增的片段及原子粗语料词汇。"
        ),
    }
    for stat_key in WORKER_STAT_KEYS:
        summary[stat_key] = 0

    errors = []
    successful_vocabulary = set()

    def handle_worker_result(worker_result):
        for stat_key in WORKER_STAT_KEYS:
            summary[stat_key] += int(
                worker_result["stats"].get(stat_key, 0)
            )

        compound_info = worker_result["compound_info"]
        if not worker_result["success"]:
            errors.append(make_error_record(
                compound_info,
                worker_result["error_type"] or "processing_failed",
                worker_result["error"] or "未知处理错误",
            ))
            return

        try:
            data, fragment_smiles = assemble_pyg_data(
                worker_result,
                normalization_params,
            )
            target_path = build_output_path(
                compound_info["compound_id"]
            )
            save_pyg_data_atomic(data, target_path)
        except Exception as exc:
            summary["save_failed"] += 1
            errors.append(make_error_record(
                compound_info,
                "save_failed",
                exc,
            ))
            return

        summary["saved_pyg"] += 1
        successful_vocabulary.update(fragment_smiles)

    print("开始切割与共价建边...")
    if NUM_WORKERS <= 1:
        for record in tqdm(records, desc="2D 处理进度"):
            try:
                worker_result = worker_process_compound(record)
            except Exception as exc:
                summary["process_exception"] += 1
                errors.append(make_error_record(
                    record,
                    "process_exception",
                    exc,
                ))
                continue
            handle_worker_result(worker_result)
    else:
        from concurrent.futures import TimeoutError
        from pebble import ProcessPool

        with ProcessPool(max_workers=NUM_WORKERS) as pool:
            future = pool.map(
                worker_process_compound,
                records,
                timeout=TASK_TIMEOUT_SECONDS,
            )
            iterator = future.result()
            progress = tqdm(total=len(records), desc="2D 处理进度")
            task_index = 0

            while True:
                try:
                    worker_result = next(iterator)
                except StopIteration:
                    break
                except TimeoutError:
                    record = records[task_index]
                    summary["timeout"] += 1
                    errors.append(make_error_record(
                        record,
                        "timeout",
                        f"处理时间超过 {TASK_TIMEOUT_SECONDS} 秒",
                    ))
                except Exception as exc:
                    record = records[task_index]
                    summary["process_exception"] += 1
                    errors.append(make_error_record(
                        record,
                        "process_exception",
                        exc,
                    ))
                else:
                    handle_worker_result(worker_result)

                progress.update(1)
                task_index += 1

            progress.close()

    corpus_update_error = None
    try:
        corpus_update_stats = update_shared_raw_corpora(
            successful_vocabulary,
        )
        summary.update(corpus_update_stats)
    except Exception as exc:
        corpus_update_error = exc
        summary["corpus_update_failed"] = 1
        summary["new_vocab_candidates"] = len(
            successful_vocabulary - existing_vocabulary
        )
        errors.append({
            "compound_id": "",
            "sheet_name": "",
            "row_number": "",
            "error_type": "corpus_update_failed",
            "error": str(exc),
        })

    save_reports(summary, errors)

    print(
        f"处理完成：保存 {summary['saved_pyg']} 个 PyG，"
        f"失败 {len(errors)} 项。"
    )
    print(
        f"共享粗语料新增 {summary['new_vocab_added']} 个词。"
    )
    print(
        "共享原子粗语料："
        f"片段={summary['atom_corpus_num_smiles']}，"
        f"原子={summary['atom_corpus_num_atoms']}，"
        f"键={summary['atom_corpus_num_bonds']}。"
    )
    if os.path.isfile(RAW_CORPUS_FILE):
        print(f"共享片段粗语料: {RAW_CORPUS_FILE}")
    else:
        print("共享片段粗语料未创建：本次没有可保存的有效片段。")
    if os.path.isfile(RAW_ATOM_CORPUS_FILE):
        print(f"共享原子粗语料: {RAW_ATOM_CORPUS_FILE}")
    else:
        print("共享原子粗语料未创建：当前没有片段词表。")
    print(f"汇总: {SUMMARY_FILE}")
    print(f"错误日志: {ERROR_LOG_FILE}")

    if corpus_update_error is not None:
        raise RuntimeError(
            f"PyG 已处理，但共享粗语料更新失败: "
            f"{corpus_update_error}"
        ) from corpus_update_error


if __name__ == "__main__":
    main()
