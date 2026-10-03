"""将复合物 PDB 和可选 SDF 转换为 PhiSSE3TD 推理条件数据。

此模块保留 notebook 的计算流程；网页的文件上传、选择和三维展示由调用方处理。
默认读取 PhiStone_2.0_online/Datas 下的数据，并写入 work_datas/processed。
"""

# =============================================================================
# 1. 配置、依赖与训练归一化参数
# =============================================================================
from pathlib import Path
from itertools import combinations
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import sys
import tempfile

import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data import Data


SEPARATOR_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SEPARATOR_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PhiSSeparator import utils
from PhiSSeparator import PhiSSeparator_vocabulary_sync as vocabulary_sync

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

WORK_ROOT = PROJECT_ROOT / "Datas" / "work_datas"
INPUT_PDB_DIR = WORK_ROOT / "input_pdb"
PROCESSED_DIR = WORK_ROOT / "processed"
SHARED_DATA_DIR = (
    PROJECT_ROOT / "Datas" / "Processed_datas" / "SE3TD_128_Shared_data"
)
NORM_STATS_FILE = SHARED_DATA_DIR / "normalization_stats.json"

TASKA_POCKET_RADIUS = 10.0
TASKB_POCKET_RADIUS = 12.0
LL_PP_CENTROID_CUTOFF = 5.0
LP_CENTROID_CUTOFF = 7.0
MAX_DROPPED_FRAGMENTS = 10

FEATURE_COLUMNS = [
    "avg_ecc", "ecc_range",
    "avg_logp", "tpsa", "avg_charge",
    "HBA", "HBD", "Aromatic", "Hydrophobe",
    "PosIonizable", "NegIonizable",
    "max_dist_3d", "max_angle_3d", "max_plane_angle_3d",
]
ANGLE_FEATURES = {
    "max_angle_3d": 180.0,
    "max_plane_angle_3d": 180.0,
}
COUNT_FEATURES = {
    "HBA", "HBD", "Aromatic", "Hydrophobe",
    "PosIonizable", "NegIonizable",
}

if not NORM_STATS_FILE.is_file():
    raise FileNotFoundError(
        f"缺少训练归一化统计文件：{NORM_STATS_FILE}"
    )
with NORM_STATS_FILE.open("r", encoding="utf-8") as file_obj:
    norm_stats = json.load(file_obj)

missing_features = set(FEATURE_COLUMNS) - set(norm_stats)
if missing_features:
    raise RuntimeError(
        f"归一化统计缺少特征：{sorted(missing_features)}"
    )

FINAL_NORM_PARAMS = {}
for feature_name in FEATURE_COLUMNS:
    stats = norm_stats[feature_name]
    if not isinstance(stats, dict) or not {"count", "mean", "M2"} <= set(stats):
        raise RuntimeError(f"特征 {feature_name} 的统计格式非法")
    count = stats["count"]
    mean = stats["mean"]
    m2 = stats["M2"]
    if (
        not isinstance(count, (int, float)) or count <= 0
        or not isinstance(mean, (int, float)) or not math.isfinite(float(mean))
        or not isinstance(m2, (int, float)) or not math.isfinite(float(m2))
        or m2 < 0
    ):
        raise RuntimeError(f"特征 {feature_name} 的统计数值非法")
    std = math.sqrt(float(m2) / float(count)) if count >= 2 else 0.0
    FINAL_NORM_PARAMS[feature_name] = {
        "mean": float(mean),
        "std": float(std),
    }

# =============================================================================
# 2. PDB 解析、HETATM 清理、口袋截取与中心化写出
# =============================================================================

# 明确排除的水、常见溶剂、缓冲剂和无关非金属离子。
# 金属优先通过元素判断保留，不会因残基名命中而删除。
REMOVABLE_HETATM_RESIDUES = {
    "HOH", "WAT", "H2O", "DOD", "D2O",
    "SO4", "PO4", "NO3", "CO3", "CL", "BR", "IOD", "FLC",
    "GOL", "EDO", "PEG", "PGE", "MPD", "DMS", "ACE", "NH2",
    "BME", "MES", "TRS", "EPE", "FMT", "ACT", "IMD", "IPA",
}
SINGLE_ATOM_NONMETAL_IONS = {"CL", "BR", "I", "F"}


def _record_name(line):
    return line[:6].strip().upper()


def _infer_element(padded_line):
    element = padded_line[76:78].strip()
    if element:
        return element.upper()
    atom_name = padded_line[12:16].strip()
    letters = "".join(char for char in atom_name if char.isalpha())
    if not letters:
        return ""
    if len(letters) >= 2 and letters[:2].upper() in utils.METAL_SYMBOLS:
        return letters[:2].upper()
    return letters[0].upper()


def _parse_coordinate_line(line, line_index):
    padded = line.rstrip("\r\n").ljust(80)
    try:
        serial = int(padded[6:11])
        x = float(padded[30:38])
        y = float(padded[38:46])
        z = float(padded[46:54])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"PDB 第 {line_index + 1} 行坐标或原子编号非法"
        ) from exc

    record = _record_name(padded)
    res_name = padded[17:20].strip().upper()
    chain_id = padded[21:22].strip()
    res_seq = padded[22:26].strip()
    insertion_code = padded[26:27].strip()
    group_key = (
        record, chain_id, res_name, res_seq, insertion_code
    )
    return {
        "line": line if line.endswith("\n") else line + "\n",
        "line_index": int(line_index),
        "record": record,
        "serial": serial,
        "atom_name": padded[12:16].strip(),
        "alt_loc": padded[16:17].strip(),
        "res_name": res_name,
        "chain_id": chain_id,
        "res_seq": res_seq,
        "insertion_code": insertion_code,
        "element": _infer_element(padded),
        "coord": np.asarray([x, y, z], dtype=np.float64),
        "group_key": group_key,
    }


def parse_first_pdb_model(pdb_path):
    pdb_path = Path(pdb_path).resolve()
    raw_lines = pdb_path.read_text(
        encoding="utf-8", errors="replace"
    ).splitlines(keepends=True)
    model_count = sum(
        _record_name(line) == "MODEL" for line in raw_lines
    )
    has_models = model_count > 0
    active_model = not has_models
    current_model = 0

    events = []
    groups = {}
    conect_lines = []
    warnings_list = []

    for line_index, line in enumerate(raw_lines):
        record = _record_name(line)
        if record == "MODEL":
            current_model += 1
            active_model = current_model == 1
            continue
        if record == "ENDMDL":
            if active_model:
                active_model = False
            continue
        if record == "CONECT":
            conect_lines.append(line if line.endswith("\n") else line + "\n")
            continue
        if has_models and not active_model:
            continue

        if record in {"ATOM", "HETATM"}:
            entry = _parse_coordinate_line(line, line_index)
            events.append(("coord", entry))
            group = groups.setdefault(
                entry["group_key"],
                {
                    "key": entry["group_key"],
                    "record": entry["record"],
                    "chain_id": entry["chain_id"],
                    "res_name": entry["res_name"],
                    "res_seq": entry["res_seq"],
                    "insertion_code": entry["insertion_code"],
                    "entries": [],
                },
            )
            group["entries"].append(entry)
        elif record == "TER":
            events.append(("ter", line if line.endswith("\n") else line + "\n"))

    if not groups:
        raise ValueError("PDB 首个模型中没有可读取的 ATOM/HETATM")
    if model_count > 1:
        warnings_list.append(
            f"检测到 {model_count} 个 MODEL；仅处理第一个模型"
        )

    return {
        "path": pdb_path,
        "events": events,
        "groups": groups,
        "conect_lines": conect_lines,
        "warnings": warnings_list,
        "model_count": int(model_count),
    }


def group_is_metal(group):
    elements = {
        entry["element"].upper()
        for entry in group["entries"]
        if entry["element"]
    }
    return bool(elements) and any(
        element in utils.METAL_SYMBOLS for element in elements
    )


def group_is_all_metal(group):
    elements = [
        entry["element"].upper()
        for entry in group["entries"]
        if entry["element"]
    ]
    return bool(elements) and all(
        element in utils.METAL_SYMBOLS for element in elements
    )


def classify_hetatm_group(group):
    if group_is_metal(group):
        return "retain_metal"
    res_name = group["res_name"].upper()
    if res_name in REMOVABLE_HETATM_RESIDUES:
        return "remove_known_interference"
    if len(group["entries"]) == 1:
        element = group["entries"][0]["element"].upper()
        if element in SINGLE_ATOM_NONMETAL_IONS:
            return "remove_nonmetal_ion"
    return "retain_unknown_or_cofactor"


def describe_group(group):
    return {
        "record": group["record"],
        "res_name": group["res_name"],
        "chain_id": group["chain_id"],
        "res_seq": group["res_seq"],
        "insertion_code": group["insertion_code"],
        "atom_count": len(group["entries"]),
        "elements": sorted({
            entry["element"] for entry in group["entries"]
            if entry["element"]
        }),
        "is_metal": bool(group_is_metal(group)),
        "label": (
            f"{group['res_name']}:{group['chain_id'] or '-'}:"
            f"{group['res_seq']}{group['insertion_code'] or ''}"
            f" ({len(group['entries'])} atoms)"
        ),
    }


def prepare_receptor_groups(parsed, selected_ligand_key=None):
    retained_keys = set()
    retained_hetatm = []
    removed_hetatm = []

    for key, group in parsed["groups"].items():
        if group["record"] == "ATOM":
            retained_keys.add(key)
            continue

        descriptor = describe_group(group)
        if selected_ligand_key is not None and key == selected_ligand_key:
            descriptor["reason"] = "selected_reference_ligand"
            removed_hetatm.append(descriptor)
            continue

        decision = classify_hetatm_group(group)
        descriptor["reason"] = decision
        if decision.startswith("retain_"):
            retained_keys.add(key)
            retained_hetatm.append(descriptor)
        else:
            removed_hetatm.append(descriptor)

    return retained_keys, retained_hetatm, removed_hetatm


def prepare_sdf_receptor_groups(parsed):
    retained_keys, retained_hetatm, removed_hetatm = (
        prepare_receptor_groups(parsed)
    )
    location_only_keys = {
        group["key"] for group in list_ligand_candidates(parsed)
    }
    retained_keys -= location_only_keys
    retained_hetatm = [
        item for item in retained_hetatm
        if (
            item["record"], item["chain_id"], item["res_name"],
            item["res_seq"], item["insertion_code"],
        ) not in location_only_keys
    ]
    for key in sorted(location_only_keys):
        descriptor = describe_group(parsed["groups"][key])
        descriptor["reason"] = "pdb_ligand_excluded_from_receptor"
        removed_hetatm.append(descriptor)
    return retained_keys, retained_hetatm, removed_hetatm


def list_ligand_candidates(parsed):
    retained_keys, _, _ = prepare_receptor_groups(parsed)
    candidates = []
    for key, group in parsed["groups"].items():
        if (
            key in retained_keys
            and group["record"] == "HETATM"
            and len(group["entries"]) >= 2
            and not group_is_all_metal(group)
        ):
            candidates.append(group)
    candidates.sort(
        key=lambda group: (
            group["chain_id"],
            group["res_seq"],
            group["insertion_code"],
            group["res_name"],
        )
    )
    return candidates


def group_coordinates(group):
    return np.stack([
        entry["coord"] for entry in group["entries"]
    ]).astype(np.float64, copy=False)


def coordinates_for_keys(parsed, group_keys):
    coords = [
        entry["coord"]
        for key in group_keys
        for entry in parsed["groups"][key]["entries"]
    ]
    if not coords:
        return np.empty((0, 3), dtype=np.float64)
    return np.stack(coords).astype(np.float64, copy=False)


def select_groups_within_radius(
    parsed, receptor_keys, reference_points, radius
):
    reference_points = np.asarray(
        reference_points, dtype=np.float64
    ).reshape(-1, 3)
    if len(reference_points) == 0:
        raise ValueError("参考坐标为空")
    selected = set()
    for key in receptor_keys:
        coords = group_coordinates(parsed["groups"][key])
        distances = np.linalg.norm(
            coords[:, None, :] - reference_points[None, :, :],
            axis=2,
        )
        if float(distances.min()) < float(radius):
            selected.add(key)
    return selected


def _translate_coordinate_line(line, center):
    if center is None:
        return line if line.endswith("\n") else line + "\n"
    padded = line.rstrip("\r\n").ljust(80)
    try:
        xyz = np.asarray([
            float(padded[30:38]),
            float(padded[38:46]),
            float(padded[46:54]),
        ], dtype=np.float64)
    except ValueError as exc:
        raise ValueError("写出PDB时遇到非法坐标行") from exc
    shifted = xyz - np.asarray(center, dtype=np.float64)
    suffix = padded[54:].rstrip()
    return (
        f"{padded[:30]}{shifted[0]:8.3f}{shifted[1]:8.3f}"
        f"{shifted[2]:8.3f}{suffix}\n"
    )


def _filtered_conect_lines(conect_lines, kept_serials):
    output = []
    for line in conect_lines:
        if _record_name(line) != "CONECT":
            continue
        try:
            serials = [
                int(line[start:start + 5])
                for start in range(6, 31, 5)
                if line[start:start + 5].strip()
            ]
        except ValueError:
            continue
        if len(serials) < 2:
            continue
        center_serial = serials[0]
        if center_serial not in kept_serials:
            continue
        neighbors = [
            serial for serial in serials[1:]
            if serial in kept_serials
        ]
        if not neighbors:
            continue
        output.append(
            "CONECT"
            + f"{center_serial:5d}"
            + "".join(f"{serial:5d}" for serial in neighbors)
            + "\n"
        )
    return output


def serialize_pdb(parsed, keep_keys, center=None):
    keep_keys = set(keep_keys)
    kept_serials = {
        entry["serial"]
        for key in keep_keys
        for entry in parsed["groups"][key]["entries"]
    }
    output = []
    emitted_since_ter = False

    for event_type, payload in parsed["events"]:
        if event_type == "coord":
            if payload["group_key"] not in keep_keys:
                continue
            output.append(
                _translate_coordinate_line(payload["line"], center)
            )
            emitted_since_ter = True
        elif event_type == "ter" and emitted_since_ter:
            output.append(payload)
            emitted_since_ter = False

    output.extend(
        _filtered_conect_lines(
            parsed["conect_lines"], kept_serials
        )
    )
    output.append("END\n")
    return output


def pdb_text(parsed, keep_keys, center=None):
    return "".join(serialize_pdb(parsed, keep_keys, center))


def build_receptor_mol(parsed, keep_keys):
    block = pdb_text(parsed, keep_keys, center=None)
    mol = Chem.MolFromPDBBlock(
        block, removeHs=True, sanitize=False
    )
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError("无法从清理后的PDB构建受体分子")
    mol = utils.advanced_rescue_valence_errors(mol)
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError("受体分子化合价抢救失败")
    return mol


def build_reference_ligand_mol(parsed, group):
    ligand_lines = [
        entry["line"] for entry in group["entries"]
    ]
    ligand_lines.extend(_filtered_conect_lines(
        parsed["conect_lines"],
        {entry["serial"] for entry in group["entries"]},
    ))
    mol = utils.build_ligand_mol_from_pdb_lines(ligand_lines)
    if (
        mol is None
        or mol.GetNumAtoms() == 0
        or mol.GetNumConformers() == 0
    ):
        raise ValueError("无法从所选HETATM构建带3D坐标的参考配体")
    return mol


def load_sdf_ligand(path):
    # 只读取所选 SDF 的首个分子，不从坐标重新推断键级。
    supplier = Chem.SDMolSupplier(str(path), removeHs=True)
    mol = next(iter(supplier), None)
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError("SDF 首个分子为空或无法解析")
    if mol.GetNumConformers() == 0 or not mol.GetConformer(0).Is3D():
        raise ValueError("SDF 配体需要与 PDB 同一坐标系下的三维构象")
    if not np.all(np.isfinite(mol.GetConformer(0).GetPositions())):
        raise ValueError("SDF 配体坐标含 NaN 或 Inf")
    return mol


def file_sha256(path, block_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file_obj:
        while True:
            block = file_obj.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


# =============================================================================
# 3. 当前 utils 接口的碎片化、构图、节点过滤与严格校验
# =============================================================================

def process_single_fragment(
    fragment,
    origin_type,
    sample_id,
    source_fragment_id,
    id_start_offset,
):
    fragment_stats = {
        "removed_ion_fragments": 0,
        "reconstruction_failed": 0,
        "retained_metal_nodes": 0,
    }
    try:
        pre_calc_data = utils.prepare_molecule_for_cutting(
            fragment, source_fragment_id, origin_type
        )
        fragment_stats["removed_ion_fragments"] = (
            pre_calc_data["removed_fragment_count"]
        )
        fragment_stats["reconstruction_failed"] = (
            pre_calc_data["reconstruction_failed_count"]
        )
        fragment_stats["retained_metal_nodes"] = len(
            pre_calc_data["metal_fragments"]
        )

        try:
            original_smiles = Chem.MolToSmiles(fragment)
        except Exception:
            original_smiles = "<ERROR>"

        dropped_count = 0
        temporary_results = []
        for component in pre_calc_data["organic_components"]:
            main_mol = component["main_mol_for_matching"]
            final_fragments = utils.execute_fragmentation_pipeline(
                main_mol,
                component["bond_info"],
                track_ports=(origin_type == "ligand"),
            )
            results_storage = []
            dropped_count += utils.calculate_descriptors_and_store(
                storage=results_storage,
                mol_data={
                    "IDs": sample_id,
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
                origin_type=origin_type,
            )
            if results_storage:
                temporary_results.extend(
                    results_storage[0]["results"]
                )

        for metal_fragment in pre_calc_data["metal_fragments"]:
            metal_record = utils.build_metal_shard_record(
                metal_fragment
            )
            if metal_record is None:
                dropped_count += 1
            else:
                temporary_results.append(metal_record)

        shards = []
        for index, shard in enumerate(temporary_results):
            shard["origin"] = origin_type
            shard["Shard ID"] = id_start_offset + index
            shards.append(shard)

        if shards:
            return shards, None, dropped_count, fragment_stats
        return [], "未生成有效 shard", dropped_count, fragment_stats
    except (
        utils.PortValidationError,
        utils.LigandSecondaryCleaningError,
    ):
        raise
    except Exception as exc:
        return [], str(exc), 1, fragment_stats


def fragmentize_molecule(
    mol,
    origin_type,
    sample_id,
    id_start_offset=0,
):
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError(f"{origin_type} 分子为空")

    fragments = list(Chem.GetMolFrags(
        mol, asMols=True, sanitizeFrags=False
    ))
    if not fragments:
        raise ValueError(f"{origin_type} 不包含可处理组分")

    all_shards = []
    next_offset = int(id_start_offset)
    dropped_total = 0
    aggregate = {
        "removed_ion_fragments": 0,
        "reconstruction_failed": 0,
        "retained_metal_nodes": 0,
    }
    warnings_list = []

    for fragment_index, fragment in enumerate(fragments):
        source_fragment_id = (
            f"{sample_id}:{origin_type}:{fragment_index}"
        )
        shards, error, dropped, stats = process_single_fragment(
            fragment,
            origin_type,
            sample_id,
            source_fragment_id,
            next_offset,
        )
        dropped_total += int(dropped)
        for key in aggregate:
            aggregate[key] += int(stats.get(key, 0))
        if error:
            warnings_list.append(
                f"{source_fragment_id}: {error}"
            )
        if shards:
            all_shards.extend(shards)
            next_offset += len(shards)

    return {
        "shards": all_shards,
        "next_offset": next_offset,
        "dropped_count": dropped_total,
        "stats": aggregate,
        "warnings": warnings_list,
    }


def normalize_shard_features(shard):
    values = []
    for feature_name in FEATURE_COLUMNS:
        value = float(shard.get(feature_name, 0.0))
        params = FINAL_NORM_PARAMS[feature_name]
        mean = params["mean"]
        std = params["std"]

        if feature_name in ANGLE_FEATURES:
            normalized = float(np.clip(
                value / ANGLE_FEATURES[feature_name],
                0.0,
                1.0,
            ))
        elif feature_name in COUNT_FEATURES:
            safe_std = max(std, 1.0)
            normalized = float(np.clip(
                (value - mean) / safe_std,
                -6.0,
                6.0,
            ))
        else:
            safe_std = std if std > 1e-5 else 1.0
            normalized = float(np.clip(
                (value - mean) / safe_std,
                -6.0,
                6.0,
            ))
        values.append(normalized)
    return values


def _build_full_directed_edges(all_shards):
    base_edges = []
    for first, second in combinations(all_shards, 2):
        first_center = first.get("centroid_3d")
        second_center = second.get("centroid_3d")
        if first_center is None or second_center is None:
            continue
        distance = float(np.linalg.norm(
            np.asarray(first_center, dtype=np.float64)
            - np.asarray(second_center, dtype=np.float64)
        ))
        same_origin = first["origin"] == second["origin"]
        cutoff = (
            LL_PP_CENTROID_CUTOFF
            if same_origin
            else LP_CENTROID_CUTOFF
        )
        if distance >= cutoff:
            continue

        origins = sorted((first["origin"], second["origin"]))
        edge_type = (
            0 if origins == ["ligand", "ligand"]
            else 1 if origins == ["ligand", "pocket"]
            else 2
        )
        attr = [0.0, 0.0, 0.0, 0.0]
        attr[edge_type] = 1.0
        base_edges.append((
            int(first["Shard ID"]),
            int(second["Shard ID"]),
            attr,
        ))

    global_old_index = len(all_shards)
    for shard in all_shards:
        if shard["origin"] != "ligand":
            continue
        base_edges.append((
            global_old_index,
            int(shard["Shard ID"]),
            [0.0, 0.0, 0.0, 1.0],
        ))

    directed = []
    for first, second, attr in base_edges:
        directed.append((first, second, list(attr)))
        directed.append((second, first, list(attr)))
    return directed, global_old_index


def assemble_filtered_pyg_data(
    ligand_shards,
    pocket_shards,
    kept_ligand_ids,
    center,
    sample_id,
    source,
):
    all_shards = sorted(
        list(ligand_shards) + list(pocket_shards),
        key=lambda shard: int(shard["Shard ID"]),
    )
    shard_ids = [int(shard["Shard ID"]) for shard in all_shards]
    if shard_ids != list(range(len(all_shards))):
        raise ValueError(
            "完整 shard 的 Shard ID 必须从0连续编号"
        )

    if any(
        shard.get("centroid_3d") is None
        or shard.get("ref_coord_3d") is None
        for shard in all_shards
    ):
        raise ValueError("存在缺少3D质心或参考坐标的 shard")

    full_connection_edge_index, full_connection_atom_ids, (
        full_connection_distance
    ) = utils.build_fragment_connections(all_shards)

    directed_edges, global_old_index = (
        _build_full_directed_edges(all_shards)
    )

    kept_ligand_ids = {
        int(value) for value in kept_ligand_ids
    }
    valid_ligand_ids = {
        int(shard["Shard ID"]) for shard in ligand_shards
    }
    unknown_ids = kept_ligand_ids - valid_ligand_ids
    if unknown_ids:
        raise ValueError(
            f"选择了不存在的 ligand shard：{sorted(unknown_ids)}"
        )

    kept_ligand = [
        shard for shard in sorted(
            ligand_shards,
            key=lambda shard: int(shard["Shard ID"]),
        )
        if int(shard["Shard ID"]) in kept_ligand_ids
    ]
    kept_pocket = sorted(
        pocket_shards,
        key=lambda shard: int(shard["Shard ID"]),
    )
    kept_real_shards = kept_ligand + kept_pocket
    kept_old_ids = [
        int(shard["Shard ID"]) for shard in kept_real_shards
    ]
    old_to_new = {
        old_id: new_id
        for new_id, old_id in enumerate(kept_old_ids)
    }
    new_global_index = len(kept_real_shards)
    old_to_new[global_old_index] = new_global_index

    center = np.asarray(center, dtype=np.float64).reshape(3)
    node_features = []
    node_types = []
    metal_mask = []
    hac_values = []
    ring_values = []
    positions = []
    reference_coords = []
    fragment_smiles = []

    for shard in kept_real_shards:
        node_features.append(normalize_shard_features(shard))
        if shard["origin"] == "ligand":
            node_types.append([1.0, 0.0, 0.0])
        elif shard["origin"] == "pocket":
            node_types.append([0.0, 1.0, 0.0])
        else:
            raise ValueError(
                f"未知 shard origin：{shard['origin']}"
            )
        metal_mask.append(bool(shard.get("is_metal", False)))
        hac_values.append(int(shard["hac"]))
        ring_values.append(int(shard["ring_count"]))
        centroid_3d = np.asarray(
            shard["centroid_3d"], dtype=np.float64
        )
        positions.append((centroid_3d - center).tolist())
        reference_frame_3d = np.asarray(
            shard["ref_coord_3d"], dtype=np.float64
        )
        relative_reference_frame = (
            reference_frame_3d - centroid_3d
        )
        restored_centered_frame = (
            centroid_3d - center
        )[None, :] + relative_reference_frame
        if not np.allclose(
            restored_centered_frame,
            reference_frame_3d - center,
            rtol=0.0, atol=1.0e-8,
        ):
            raise RuntimeError(
                "相对框架坐标无法恢复中心化绝对坐标"
            )
        reference_coords.append(
            relative_reference_frame.tolist()
        )
        smiles = shard.get("smiles")
        if not isinstance(smiles, str) or not smiles:
            raise ValueError("真实节点缺少规范 SMILES")
        fragment_smiles.append(smiles)

    node_features.append([0.0] * len(FEATURE_COLUMNS))
    node_types.append([0.0, 0.0, 1.0])
    metal_mask.append(False)
    hac_values.append(0)
    ring_values.append(0)
    positions.append([0.0, 0.0, 0.0])
    reference_coords.append(
        np.zeros((3, 3), dtype=np.float64).tolist()
    )
    fragment_smiles.append("<GLOBAL>")

    filtered_edges = [
        (
            old_to_new[first],
            old_to_new[second],
            attr,
        )
        for first, second, attr in directed_edges
        if first in old_to_new and second in old_to_new
    ]
    if filtered_edges:
        edge_index = torch.tensor(
            [
                [record[0] for record in filtered_edges],
                [record[1] for record in filtered_edges],
            ],
            dtype=torch.long,
        )
        edge_attr = torch.tensor(
            [record[2] for record in filtered_edges],
            dtype=torch.float,
        )
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 4), dtype=torch.float)

    kept_connections = []
    for connection_index, (first, second) in enumerate(
        zip(
            full_connection_edge_index[0],
            full_connection_edge_index[1],
        )
    ):
        first = int(first)
        second = int(second)
        if first not in old_to_new or second not in old_to_new:
            continue
        kept_connections.append({
            "nodes": (
                old_to_new[first],
                old_to_new[second],
            ),
            "atoms": full_connection_atom_ids[
                connection_index
            ],
            "distance": full_connection_distance[
                connection_index
            ],
        })

    if kept_connections:
        connection_edge_index = torch.tensor(
            [
                [record["nodes"][0] for record in kept_connections],
                [record["nodes"][1] for record in kept_connections],
            ],
            dtype=torch.long,
        ).reshape(2, -1)
        connection_atom_ids = torch.tensor(
            [record["atoms"] for record in kept_connections],
            dtype=torch.long,
        ).reshape(-1, 2)
        connection_distance = torch.tensor(
            [record["distance"] for record in kept_connections],
            dtype=torch.float,
        ).reshape(-1)
    else:
        connection_edge_index = torch.empty(
            (2, 0), dtype=torch.long
        )
        connection_atom_ids = torch.empty(
            (0, 2), dtype=torch.long
        )
        connection_distance = torch.empty(
            (0,), dtype=torch.float
        )

    data = Data(
        x=torch.tensor(node_features, dtype=torch.float),
        node_type=torch.tensor(node_types, dtype=torch.float),
        metal_mask=torch.tensor(metal_mask, dtype=torch.bool),
        hac=torch.tensor(hac_values, dtype=torch.long),
        ring_count=torch.tensor(ring_values, dtype=torch.long),
        pos=torch.tensor(positions, dtype=torch.float),
        ref_coords=torch.tensor(
            np.asarray(reference_coords), dtype=torch.float
        ),
        ref_coords_mode="relative_to_node_pos",
        fragment_smiles=fragment_smiles,
        edge_index=edge_index,
        edge_attr=edge_attr,
        sample_id=str(sample_id),
        pdb_id=str(sample_id).lower(),
        source=str(source),
        connection_edge_index=connection_edge_index,
        connection_atom_ids=connection_atom_ids,
        connection_distance=connection_distance,
    )
    validate_pyg_data(data)
    return data, {
        "initial_ligand_nodes": len(ligand_shards),
        "kept_ligand_nodes": len(kept_ligand),
        "pocket_nodes": len(kept_pocket),
        "global_nodes": 1,
        "total_nodes": int(data.x.size(0)),
        "directed_edges": int(data.edge_index.size(1)),
        "connections": int(data.connection_edge_index.size(1)),
    }


def validate_pyg_data(data):
    required_fields = (
        "x", "node_type", "metal_mask", "hac", "ring_count",
        "pos", "ref_coords", "fragment_smiles",
        "ref_coords_mode",
        "edge_index", "edge_attr",
        "sample_id", "pdb_id", "source",
        "connection_edge_index", "connection_atom_ids",
        "connection_distance",
    )
    missing = [
        name for name in required_fields
        if not hasattr(data, name)
    ]
    if missing:
        raise ValueError(f"PyG缺少字段：{missing}")
    if data.ref_coords_mode != "relative_to_node_pos":
        raise ValueError(
            "ref_coords_mode 必须为 relative_to_node_pos"
        )

    num_nodes = int(data.x.size(0))
    expected_shapes = {
        "x": (num_nodes, len(FEATURE_COLUMNS)),
        "node_type": (num_nodes, 3),
        "metal_mask": (num_nodes,),
        "hac": (num_nodes,),
        "ring_count": (num_nodes,),
        "pos": (num_nodes, 3),
        "ref_coords": (num_nodes, 3, 3),
    }
    for name, expected_shape in expected_shapes.items():
        actual_shape = tuple(getattr(data, name).shape)
        if actual_shape != expected_shape:
            raise ValueError(
                f"{name} 形状错误：{actual_shape} != {expected_shape}"
            )
    if len(data.fragment_smiles) != num_nodes:
        raise ValueError("fragment_smiles 与节点数不一致")
    if data.fragment_smiles[-1] != "<GLOBAL>":
        raise ValueError("最后一个节点必须是 <GLOBAL>")
    if not torch.equal(
        data.node_type[-1],
        torch.tensor([0.0, 0.0, 1.0]),
    ):
        raise ValueError("global 节点类型错误")
    if bool(data.metal_mask[-1]) or int(data.hac[-1]) != 0:
        raise ValueError("global 节点标量属性错误")
    if not torch.all(data.x[-1] == 0):
        raise ValueError("global 节点 x 必须全零")
    if not torch.all(data.pos[-1] == 0):
        raise ValueError("global 节点 pos 必须全零")
    if not torch.all(data.ref_coords[-1] == 0):
        raise ValueError("global 节点 ref_coords 必须全零")

    for tensor_name in ("x", "pos", "ref_coords"):
        if not torch.isfinite(getattr(data, tensor_name)).all():
            raise ValueError(f"{tensor_name} 包含 NaN 或 Inf")
    absolute_ref_coords = data.pos[:, None, :] + data.ref_coords
    if not torch.isfinite(absolute_ref_coords).all():
        raise ValueError("pos + ref_coords 包含 NaN 或 Inf")

    for smiles in data.fragment_smiles[:-1]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"节点 SMILES 无法解析：{smiles}")
        canonical = Chem.MolToSmiles(
            mol, canonical=True, isomericSmiles=True
        )
        if canonical != smiles:
            raise ValueError(
                f"节点 SMILES 不是稳定规范形式：{smiles} -> {canonical}"
            )

    if data.edge_index.dim() != 2 or tuple(
        data.edge_index.shape[:1]
    ) != (2,):
        raise ValueError("edge_index 必须为 [2,E]")
    edge_count = int(data.edge_index.size(1))
    if tuple(data.edge_attr.shape) != (edge_count, 4):
        raise ValueError("edge_attr 必须为 [E,4]")
    if edge_count > 0:
        if (
            int(data.edge_index.min()) < 0
            or int(data.edge_index.max()) >= num_nodes
        ):
            raise ValueError("edge_index 含越界节点")
        if bool((
            data.edge_index[0] == data.edge_index[1]
        ).any()):
            raise ValueError("edge_index 包含自环")
        if not torch.allclose(
            data.edge_attr.sum(dim=1),
            torch.ones(edge_count),
        ):
            raise ValueError("edge_attr 不是四类 one-hot")

        records = Counter(
            (
                int(data.edge_index[0, index]),
                int(data.edge_index[1, index]),
                tuple(float(value) for value in data.edge_attr[index]),
            )
            for index in range(edge_count)
        )
        for (first, second, attr), count in records.items():
            if records[(second, first, attr)] != count:
                raise ValueError(
                    f"边 ({first},{second}) 缺少对应反向边"
                )

    connection_count = int(
        data.connection_edge_index.size(1)
    )
    if tuple(data.connection_edge_index.shape) != (
        2, connection_count
    ):
        raise ValueError(
            "connection_edge_index 必须为 [2,C]"
        )
    if tuple(data.connection_atom_ids.shape) != (
        connection_count, 2
    ):
        raise ValueError(
            "connection_atom_ids 必须为 [C,2]"
        )
    if tuple(data.connection_distance.shape) != (
        connection_count,
    ):
        raise ValueError(
            "connection_distance 必须为 [C]"
        )
    if connection_count > 0:
        if (
            int(data.connection_edge_index.min()) < 0
            or int(data.connection_edge_index.max()) >= num_nodes - 1
        ):
            raise ValueError(
                "connection_edge_index 含越界真实节点"
            )
        if bool((
            data.connection_edge_index[0]
            == data.connection_edge_index[1]
        ).any()):
            raise ValueError(
                "connection_edge_index 包含自环"
            )
        if bool((data.connection_atom_ids < 0).any()):
            raise ValueError(
                "connection_atom_ids 包含负原子编号"
            )
        if (
            not torch.isfinite(data.connection_distance).all()
            or bool((data.connection_distance < 0).any())
        ):
            raise ValueError(
                "connection_distance 包含非法值"
            )

        pair_counts = Counter(
            tuple(sorted((int(first), int(second))))
            for first, second in (
                data.connection_edge_index.t().tolist()
            )
        )
        if any(count > 2 for count in pair_counts.values()):
            raise ValueError(
                "同一节点对包含超过两个成键原子对"
            )


# =============================================================================
# 4. TASKA/TASKB 执行、原子写出与元数据
# =============================================================================

def atomic_write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(
        f".{path.name}.{os.getpid()}.tmp"
    )
    try:
        temp_path.write_text(text, encoding="utf-8")
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def atomic_torch_save(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(
        f".{path.name}.{os.getpid()}.tmp"
    )
    try:
        torch.save(data, temp_path)
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def atomic_write_sdf(mol, center, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(
        f".{path.stem}.{os.getpid()}.tmp.sdf"
    )
    normalized = Chem.Mol(mol)
    if normalized.GetNumConformers() == 0:
        raise ValueError("参考配体缺少3D构象，无法保存SDF")
    conf = normalized.GetConformer(0)
    center = np.asarray(center, dtype=np.float64).reshape(3)
    for atom_index in range(normalized.GetNumAtoms()):
        position = np.asarray(
            conf.GetAtomPosition(atom_index),
            dtype=np.float64,
        )
        shifted = position - center
        conf.SetAtomPosition(
            atom_index,
            (
                float(shifted[0]),
                float(shifted[1]),
                float(shifted[2]),
            ),
        )
    try:
        writer = Chem.SDWriter(str(temp_path))
        if writer is None:
            raise RuntimeError("无法创建SDF写入器")
        writer.write(normalized)
        writer.close()
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def shard_summary(shards):
    return [
        {
            "shard_id": int(shard["Shard ID"]),
            "smiles": str(shard["smiles"]),
            "origin": str(shard["origin"]),
            "is_metal": bool(shard.get("is_metal", False)),
            "hac": int(shard["hac"]),
            "ring_count": int(shard["ring_count"]),
        }
        for shard in sorted(
            shards, key=lambda item: int(item["Shard ID"])
        )
    ]


def _base_metadata(state, task_mode, center, radius):
    parsed = state["parsed"]
    return {
        "status": "processing",
        "ref_coords_mode": "relative_to_node_pos",
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "task_mode": task_mode,
        "sample_id": state["sample_id"],
        "pdb_id": state["sample_id"].lower(),
        "source": (
            "work_data_taska"
            if task_mode == "TASKA"
            else "work_data_taskb"
        ),
        "input_pdb": str(parsed["path"]),
        "input_sdf": str(state["sdf_path"]) if state.get("sdf_path") else None,
        "ligand_source": (
            "sdf" if task_mode == "TASKA" and state.get("sdf_path")
            else "pdb" if task_mode == "TASKA" else None
        ),
        "pocket_location_source": state.get(
            "pocket_location_source",
            "pdb_ligand" if task_mode == "TASKA" else "coordinates",
        ),
        "input_pdb_sha256": file_sha256(parsed["path"]),
        "input_model_count": parsed["model_count"],
        "reference_center_original": [
            float(value) for value in np.asarray(center).reshape(3)
        ],
        "pocket_radius_angstrom": float(radius),
        "embedding_status": "pending_vocabulary_synchronization",
        "warnings": list(parsed["warnings"]),
    }


def _write_common_outputs(
    state,
    task_mode,
    center,
    data,
    metadata,
    full_receptor_keys,
    pocket_keys,
    ligand_mol=None,
):
    sample_id = state["sample_id"]
    output_dir = state.get("output_root", PROCESSED_DIR) / sample_id
    output_dir.mkdir(parents=True, exist_ok=True)

    pt_path = output_dir / f"{sample_id}.pt"
    pocket_path = output_dir / "pocket.pdb"
    full_receptor_path = (
        output_dir / "full_receptor_normalized.pdb"
    )
    metadata_path = output_dir / "processing_metadata.json"
    ligand_path = (
        output_dir / f"{sample_id}_ligand_normalized.sdf"
    )

    pocket_text = pdb_text(
        state["parsed"], pocket_keys, center=center
    )
    full_receptor_text = pdb_text(
        state["parsed"],
        full_receptor_keys,
        center=center,
    )

    atomic_write_text(pocket_path, pocket_text)
    atomic_write_text(
        full_receptor_path, full_receptor_text
    )
    if ligand_mol is not None:
        atomic_write_sdf(ligand_mol, center, ligand_path)
    elif task_mode == "TASKA":
        raise ValueError("TASKA缺少参考配体SDF源分子")
    elif ligand_path.exists():
        # 只清理本脚本定义的已知 TASKA 旧产物。
        ligand_path.unlink()

    sync_report_path = (
        output_dir / "vocabulary_sync_report.json"
    )
    metadata["outputs"] = {
        "pyg": str(pt_path),
        "pocket_pdb": str(pocket_path),
        "full_receptor_pdb": str(full_receptor_path),
        "reference_ligand_sdf": (
            str(ligand_path) if ligand_mol is not None else None
        ),
        "vocabulary_sync_report": str(sync_report_path),
    }
    utils.atomic_write_json(metadata_path, metadata)
    atomic_torch_save(data, pt_path)

    try:
        sync_report = (
            vocabulary_sync.synchronize_condition_vocabularies(
                [pt_path],
                device="auto",
                report_path=sync_report_path,
            )
        )
    except Exception as exc:
        metadata["status"] = "failed"
        metadata["embedding_status"] = (
            "vocabulary_synchronization_failed"
        )
        metadata["failure_stage"] = (
            "post_write_vocabulary_sync"
        )
        metadata["error_type"] = type(exc).__name__
        metadata["error"] = str(exc)
        metadata["vocabulary_sync"] = {
            "status": "failed",
            "report_path": str(sync_report_path),
        }
        utils.atomic_write_json(metadata_path, metadata)
        raise

    metadata["status"] = "complete"
    metadata["embedding_status"] = (
        "synchronized_during_processing"
    )
    metadata["vocabulary_sync"] = {
        "status": sync_report["status"],
        "report_path": str(sync_report_path),
        "unique_smiles_count": sync_report[
            "unique_smiles_count"
        ],
        "fragment_cache_hits": sync_report["fragment"][
            "cache_hits"
        ],
        "fragment_added_count": len(
            sync_report["fragment"][
                "newly_added_to_standard_table"
            ]
        ),
        "atom_cache_hits": sync_report["atom"][
            "cache_hits"
        ],
        "atom_added_count": len(
            sync_report["atom"][
                "newly_added_to_standard_table"
            ]
        ),
    }
    utils.atomic_write_json(metadata_path, metadata)
    return output_dir


def run_taska(state, kept_ligand_ids):
    """按参考配体定位口袋并处理 TASKA 的配体片段。"""
    parsed = state["parsed"]
    sample_id = state["sample_id"]
    task_mode = state["task_mode"]
    selected_group = state.get("selected_ligand_group")
    radius = TASKA_POCKET_RADIUS
    ligand_mol = state["ligand_mol"]
    ligand_result = state["ligand_result"]
    ligand_shards = ligand_result["shards"]
    center = np.asarray(
        state["reference_center"], dtype=np.float64
    )

    if state.get("pocket_location_source") == "sdf_ligand":
        full_receptor_keys, retained_hetatm, removed_hetatm = (
            prepare_sdf_receptor_groups(parsed)
        )
        location_coords = state["sdf_mol"].GetConformer(0).GetPositions()
    else:
        full_receptor_keys, retained_hetatm, removed_hetatm = (
            prepare_receptor_groups(
                parsed,
                selected_ligand_key=selected_group["key"],
            )
        )
        location_coords = group_coordinates(selected_group)
    pocket_keys = select_groups_within_radius(
        parsed,
        full_receptor_keys,
        location_coords,
        radius,
    )
    if not pocket_keys:
        raise ValueError(f"{radius:g} Å范围内没有可用受体口袋")

    pocket_mol = build_receptor_mol(parsed, pocket_keys)
    pocket_result = fragmentize_molecule(
        pocket_mol,
        "pocket",
        sample_id,
        id_start_offset=len(ligand_shards),
    )
    if not pocket_result["shards"]:
        raise ValueError("未生成有效 pocket shard")

    total_dropped = (
        ligand_result["dropped_count"]
        + pocket_result["dropped_count"]
    )
    if total_dropped > MAX_DROPPED_FRAGMENTS:
        raise ValueError(
            f"切割质量控制失败：共丢弃 {total_dropped} 个片段"
        )

    data, graph_stats = assemble_filtered_pyg_data(
        ligand_shards,
        pocket_result["shards"],
        kept_ligand_ids,
        center,
        sample_id,
        source="work_data_taska" if task_mode == "TASKA" else "work_data_taskb",
    )

    kept_ligand_ids = {
        int(value) for value in kept_ligand_ids
    }
    kept_ligand_shards = [
        shard for shard in ligand_shards
        if int(shard["Shard ID"]) in kept_ligand_ids
    ]
    removed_ligand_shards = [
        shard for shard in ligand_shards
        if int(shard["Shard ID"]) not in kept_ligand_ids
    ]
    metadata = _base_metadata(
        state, task_mode, center, radius
    )
    metadata.update({
        "reference_ligand": (
            describe_group(selected_group) if selected_group is not None else None
        ),
        "hetatm_retained": retained_hetatm,
        "hetatm_removed": removed_hetatm,
        "ligand_shards_initial": shard_summary(ligand_shards),
        "ligand_shards_kept": shard_summary(
            kept_ligand_shards
        ),
        "ligand_shards_removed": shard_summary(
            removed_ligand_shards
        ),
        "ligand_shard_ids_kept": sorted(kept_ligand_ids),
        "ligand_shard_ids_removed": sorted(
            int(shard["Shard ID"])
            for shard in ligand_shards
            if int(shard["Shard ID"]) not in kept_ligand_ids
        ),
        "pocket_shards": shard_summary(
            pocket_result["shards"]
        ),
        "fragmentation": {
            "dropped_count": int(total_dropped),
            "ligand": ligand_result["stats"],
            "pocket": pocket_result["stats"],
        },
        "graph": graph_stats,
        "pocket_group_count": len(pocket_keys),
        "full_receptor_group_count": len(
            full_receptor_keys
        ),
    })
    metadata["warnings"].extend(
        ligand_result["warnings"]
        + pocket_result["warnings"]
    )

    output_dir = _write_common_outputs(
        state,
        task_mode,
        center,
        data,
        metadata,
        full_receptor_keys,
        pocket_keys,
        ligand_mol=ligand_mol,
    )
    return data, metadata, output_dir


def run_taskb(state, center):
    parsed = state["parsed"]
    sample_id = state["sample_id"]
    center = np.asarray(center, dtype=np.float64).reshape(3)

    location_source = state["pocket_location_source"]
    selected_group = state.get("selected_ligand_group")
    if location_source == "pdb_ligand":
        full_receptor_keys, retained_hetatm, removed_hetatm = (
            prepare_receptor_groups(
                parsed, selected_ligand_key=selected_group["key"]
            )
        )
        location_coords = group_coordinates(selected_group)
        radius = TASKA_POCKET_RADIUS
    else:
        full_receptor_keys, retained_hetatm, removed_hetatm = (
            prepare_sdf_receptor_groups(parsed)
        )
        location_coords = (
            state["sdf_mol"].GetConformer(0).GetPositions()
            if location_source == "sdf_ligand" else center.reshape(1, 3)
        )
        radius = (
            TASKA_POCKET_RADIUS if location_source == "sdf_ligand"
            else TASKB_POCKET_RADIUS
        )
    pocket_keys = select_groups_within_radius(
        parsed,
        full_receptor_keys,
        location_coords,
        radius,
    )
    if not pocket_keys:
        raise ValueError(f"{radius:g} Å范围内没有可用受体口袋")

    pocket_mol = build_receptor_mol(parsed, pocket_keys)
    pocket_result = fragmentize_molecule(
        pocket_mol,
        "pocket",
        sample_id,
        id_start_offset=0,
    )
    if not pocket_result["shards"]:
        raise ValueError("未生成有效 pocket shard")
    if (
        pocket_result["dropped_count"]
        > MAX_DROPPED_FRAGMENTS
    ):
        raise ValueError(
            "切割质量控制失败："
            f"共丢弃 {pocket_result['dropped_count']} 个片段"
        )

    data, graph_stats = assemble_filtered_pyg_data(
        [],
        pocket_result["shards"],
        kept_ligand_ids=set(),
        center=center,
        sample_id=sample_id,
        source="work_data_taskb",
    )
    metadata = _base_metadata(state, "TASKB", center, radius)
    metadata.update({
        "reference_ligand": (
            describe_group(selected_group) if selected_group is not None else None
        ),
        "hetatm_retained": retained_hetatm,
        "hetatm_removed": removed_hetatm,
        "ligand_shards_initial": [],
        "ligand_shards_kept": [],
        "ligand_shards_removed": [],
        "ligand_shard_ids_kept": [],
        "ligand_shard_ids_removed": [],
        "pocket_shards": shard_summary(
            pocket_result["shards"]
        ),
        "fragmentation": {
            "dropped_count": int(
                pocket_result["dropped_count"]
            ),
            "ligand": None,
            "pocket": pocket_result["stats"],
        },
        "graph": graph_stats,
        "pocket_group_count": len(pocket_keys),
        "full_receptor_group_count": len(
            full_receptor_keys
        ),
    })
    metadata["warnings"].extend(
        pocket_result["warnings"]
    )

    output_dir = _write_common_outputs(
        state,
        "TASKB",
        center,
        data,
        metadata,
        full_receptor_keys,
        pocket_keys,
        ligand_mol=None,
    )
    return data, metadata, output_dir


# =============================================================================
# 5. 供后台调用的输入预览、参考位置选择与条件数据生成
# =============================================================================

def load_work_input(pdb_path, task_mode="TASKA", sdf_path=None, output_root=None):
    """读取输入文件，返回后续步骤共用的处理状态。"""
    task_mode = str(task_mode).upper()
    if task_mode not in {"TASKA", "TASKB"}:
        raise ValueError("task_mode 必须为 TASKA 或 TASKB")

    pdb_path = Path(pdb_path).resolve()
    state = {
        "pdb_path": pdb_path,
        "parsed": parse_first_pdb_model(pdb_path),
        "sample_id": pdb_path.stem,
        "task_mode": task_mode,
        "sdf_path": Path(sdf_path).resolve() if sdf_path else None,
    }
    if state["sdf_path"] is not None:
        state["sdf_mol"] = load_sdf_ligand(state["sdf_path"])
    if output_root is not None:
        state["output_root"] = Path(output_root).resolve()
    return state


def describe_work_input(state):
    """提供可直接转成 JSON 的结构预览和定位选项。"""
    parsed = state["parsed"]
    task_mode = state["task_mode"]
    groups = list_ligand_candidates(parsed)
    if task_mode == "TASKA" and not groups and not state.get("sdf_path"):
        raise ValueError("清理干扰物后没有可供选择的多原子HETATM配体")
    candidates = [
        {
            **describe_group(group),
            "key": list(group["key"]),
            "center": [
                float(value) for value in group_coordinates(group).mean(axis=0)
            ],
        }
        for group in groups
    ]
    suggested_center = None
    receptor_keys, _, _ = (
        prepare_sdf_receptor_groups(parsed)
        if state.get("sdf_path") else prepare_receptor_groups(parsed)
    )
    if state.get("sdf_path") or task_mode == "TASKB":
        receptor_coords = coordinates_for_keys(parsed, receptor_keys)
        if len(receptor_coords) == 0:
            raise ValueError("清理后的完整受体没有坐标")
        location_coords = (
            state["sdf_mol"].GetConformer(0).GetPositions()
            if state.get("sdf_path") else receptor_coords
        )
        center, _ = utils.calculate_minimum_enclosing_ball(location_coords)
        suggested_center = [float(value) for value in center]

    return {
        "sample_id": state["sample_id"],
        "task_mode": task_mode,
        "input_pdb": str(state["pdb_path"]),
        "input_sdf": str(state["sdf_path"]) if state.get("sdf_path") else None,
        "structure_pdb": pdb_text(parsed, receptor_keys),
        "ligand_candidates": candidates,
        "suggested_center": suggested_center,
        "reference_sdf_mol_block": (
            Chem.MolToMolBlock(state["sdf_mol"])
            if state.get("sdf_path") else None
        ),
        "pocket_radius_angstrom": (
            TASKA_POCKET_RADIUS if task_mode == "TASKA" or state.get("sdf_path")
            else TASKB_POCKET_RADIUS
        ),
        "warnings": list(parsed["warnings"]),
    }


def prepare_ligand_shards(state, ligand_mol, reference_center=None):
    """沿用 notebook 的配体碎片化和参考中心计算。"""
    ligand_result = fragmentize_molecule(
        ligand_mol, "ligand", state["sample_id"], id_start_offset=0
    )
    if not ligand_result["shards"]:
        raise ValueError("参考配体未生成有效 shard")
    if ligand_result["dropped_count"] > MAX_DROPPED_FRAGMENTS:
        raise ValueError(
            "参考配体质量控制失败："
            f"丢弃 {ligand_result['dropped_count']} 个片段"
        )
    ligand_centers = np.asarray(
        [shard["centroid_3d"] for shard in ligand_result["shards"]],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(ligand_centers)):
        raise ValueError("参考配体 shard 质心含 NaN 或 Inf")
    if reference_center is None:
        reference_center, _ = utils.calculate_minimum_enclosing_ball(
            ligand_centers
        )
    state.update({
        "ligand_mol": ligand_mol,
        "ligand_result": ligand_result,
        "reference_center": np.asarray(reference_center, dtype=np.float64),
    })


def prepare_reference(state, ligand_key=None, center=None):
    """按上传的 SDF、所选 PDB 配体或 TASKB 手动坐标定位口袋。"""
    state.pop("selected_ligand_group", None)
    if state.get("sdf_path"):
        reference_center, _ = utils.calculate_minimum_enclosing_ball(
            state["sdf_mol"].GetConformer(0).GetPositions()
        )
        state["pocket_location_source"] = "sdf_ligand"
        if state["task_mode"] == "TASKA":
            prepare_ligand_shards(
                state, Chem.Mol(state["sdf_mol"]), reference_center
            )
        else:
            state["reference_center"] = np.asarray(
                reference_center, dtype=np.float64
            )
    elif ligand_key is not None:
        selected_group = next(
            (
                group for group in list_ligand_candidates(state["parsed"])
                if group["key"] == tuple(ligand_key)
            ),
            None,
        )
        if selected_group is None:
            raise ValueError("所选 PDB 配体不在候选列表中")
        state["selected_ligand_group"] = selected_group
        state["pocket_location_source"] = "pdb_ligand"
        if state["task_mode"] == "TASKA":
            ligand_mol = build_reference_ligand_mol(
                state["parsed"], selected_group
            )
            prepare_ligand_shards(state, ligand_mol)
        else:
            reference_center, _ = utils.calculate_minimum_enclosing_ball(
                group_coordinates(selected_group)
            )
            state["reference_center"] = np.asarray(
                reference_center, dtype=np.float64
            )
    else:
        if state["task_mode"] == "TASKA":
            raise ValueError("TASKA 需要选择 PDB 配体或上传参考 SDF")
        if center is None:
            raise ValueError("TASKB 需要选择 PDB 配体或输入三维口袋中心坐标")
        center = np.asarray(center, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(center)):
            raise ValueError("TASKB 口袋中心坐标含 NaN 或 Inf")
        state["reference_center"] = center
        state["pocket_location_source"] = "coordinates"
    return state


def describe_ligand_shards(state):
    """返回片段编号、属性和供 3Dmol.js 展示的 MolBlock。"""
    if "ligand_result" not in state:
        return []
    result = []
    for shard in sorted(
        state["ligand_result"]["shards"],
        key=lambda item: int(item["Shard ID"]),
    ):
        mol = shard.get("mol") or shard.get("clean_mol")
        result.append({
            **shard_summary([shard])[0],
            "center": [float(value) for value in shard["centroid_3d"]],
            "mol_block": (
                Chem.MolToMolBlock(mol)
                if mol is not None and mol.GetNumConformers() > 0 else None
            ),
        })
    return result


def generate_condition(state, kept_ligand_ids=None):
    """写出条件 .pt、中心化结构文件、元数据和词汇表同步结果。"""
    if "reference_center" not in state:
        raise ValueError("请先调用 prepare_reference 选择参考位置")
    if state["task_mode"] == "TASKA":
        if kept_ligand_ids is None:
            kept_ligand_ids = {
                int(shard["Shard ID"])
                for shard in state["ligand_result"]["shards"]
            }
        else:
            kept_ligand_ids = {int(value) for value in kept_ligand_ids}
        if not kept_ligand_ids:
            raise ValueError("TASKA 至少需要保留一个配体片段")
        data, metadata, output_dir = run_taska(state, kept_ligand_ids)
    else:
        data, metadata, output_dir = run_taskb(
            state, state["reference_center"]
        )
    return {
        "sample_id": state["sample_id"],
        "task_mode": state["task_mode"],
        "output_dir": str(output_dir),
        "outputs": metadata["outputs"],
        "node_count": int(data.x.size(0)),
        "directed_edge_count": int(data.edge_index.size(1)),
        "metadata": metadata,
    }


def process_work_data(
    pdb_path,
    task_mode="TASKA",
    sdf_path=None,
    ligand_key=None,
    center=None,
    kept_ligand_ids=None,
    output_root=None,
):
    """一次完成解析、参考位置选择和条件数据生成。"""
    state = load_work_input(pdb_path, task_mode, sdf_path, output_root)
    prepare_reference(state, ligand_key=ligand_key, center=center)
    return generate_condition(state, kept_ligand_ids=kept_ligand_ids)
