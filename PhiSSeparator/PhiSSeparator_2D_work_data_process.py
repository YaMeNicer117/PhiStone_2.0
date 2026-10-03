"""将未知活性输入转换为无标签二维 PyG，并同步完整片段/原子词表。"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import re
import sys
import uuid
from pathlib import Path

# 必须在导入 pandas、RDKit 和 PyTorch 前限制底层数值库的线程竞争。
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import pandas as pd
from rdkit import Chem, RDLogger
from tqdm import tqdm


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from PhiSGATv2 import PhiSGATv2_config as config  # noqa: E402
from PhiSSeparator import PhiSSeparator_SE3TD_2D as separator_2d  # noqa: E402


TABLE_EXTENSIONS = {".csv", ".xls", ".xlsx"}
SDF_EXTENSION = ".sdf"
SUMMARY_FILENAME = "processing_summary.json"
ERROR_FILENAME = "processing_errors.csv"
VOCABULARY_SYNC_REPORT_FILENAME = "vocabulary_sync_report.json"
ERROR_COLUMNS = (
    "compound_id",
    "source_file",
    "location",
    "error_type",
    "error",
)
NON_TETRAHEDRAL_CHIRAL_TAGS = {
    Chem.ChiralType.CHI_SQUAREPLANAR,
    Chem.ChiralType.CHI_TRIGONALBIPYRAMIDAL,
    Chem.ChiralType.CHI_OCTAHEDRAL,
}


def _remove_nonisotopic_hydrogens_for_activity(molecule):
    """删除活性预测输入中的普通显式氢，并保留同位素氢。"""

    parameters = Chem.RemoveHsParameters()
    parameters.removeAndTrackIsotopes = False
    parameters.removeIsotopes = False
    parameters.removeDefiningBondStereo = True
    parameters.removeDegreeZero = True
    parameters.removeDummyNeighbors = True
    parameters.removeHigherDegrees = True
    parameters.removeHydrides = True
    parameters.removeInSGroups = True
    parameters.removeMapped = True
    parameters.removeNontetrahedralNeighbors = True
    parameters.removeOnlyHNeighbors = True
    parameters.removeWithQuery = True
    parameters.removeWithWedgedBond = True
    parameters.showWarnings = False
    cleaned = Chem.RemoveHs(
        Chem.Mol(molecule),
        parameters,
        sanitize=True,
    )
    remaining_indices = [
        atom.GetIdx()
        for atom in cleaned.GetAtoms()
        if atom.GetAtomicNum() == 1 and atom.GetIsotope() == 0
    ]
    if remaining_indices:
        raise ValueError(
            "活性预测分子清理后仍包含普通氢原子: "
            + ", ".join(str(index) for index in remaining_indices)
        )
    return cleaned


def _normalize_activity_fragment_smiles(smiles):
    """规范化活性预测片段，并移除不稳定的非四面体手性标记。"""
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError("片段 SMILES 必须是非空字符串")
    molecule = Chem.MolFromSmiles(smiles.strip(), sanitize=True)
    if molecule is None or molecule.GetNumAtoms() == 0:
        raise ValueError(f"无法解析片段 SMILES: {smiles}")

    molecule = Chem.Mol(molecule)
    # RDKit 不能保证 @SP/@TB/@OH SMILES 的规范写出稳定；
    # 普通四面体手性标签继续保留。
    for atom in molecule.GetAtoms():
        if atom.GetChiralTag() in NON_TETRAHEDRAL_CHIRAL_TAGS:
            atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    molecule = _remove_nonisotopic_hydrogens_for_activity(molecule)

    normalized = Chem.MolToSmiles(
        molecule,
        canonical=True,
        isomericSmiles=True,
    )
    if not normalized:
        raise ValueError(f"无法规范化片段 SMILES: {smiles}")
    return normalized


def _normalize_text(value):
    return separator_2d._normalize_text(value)


def _sanitize_compound_id(value, fallback):
    """把来源 ID 转换成可安全用于 Windows 文件名的稳定文本。"""
    candidate = _normalize_text(value) or str(fallback)
    candidate = "".join(
        "_"
        if char in separator_2d.WINDOWS_INVALID_FILENAME_CHARS
        or ord(char) < 32
        else char
        for char in candidate
    )
    candidate = re.sub(r"\s+", "_", candidate).strip(" ._")
    if not candidate:
        candidate = str(fallback)
    candidate = candidate[:config.WORK_OUTPUT_ID_MAX_LENGTH].rstrip(" ._")
    if not candidate:
        candidate = "work_molecule"
    stem = candidate.split(".", 1)[0].upper()
    if stem in separator_2d.WINDOWS_RESERVED_FILENAMES:
        candidate = f"work_{candidate}"
    separator_2d.validate_compound_id_for_filename(candidate)
    return candidate


def _allocate_unique_id(base_id, id_state):
    """按 Windows 大小写不敏感规则分配本次运行内唯一的文件 ID。"""
    base_key = base_id.casefold()
    count = id_state["base_counts"].get(base_key, 0)
    while True:
        count += 1
        if count == 1:
            candidate = base_id
        else:
            suffix = f"_{count}"
            trimmed = base_id[
                :config.WORK_OUTPUT_ID_MAX_LENGTH - len(suffix)
            ].rstrip(" ._")
            candidate = f"{trimmed}{suffix}"
        candidate_key = candidate.casefold()
        if candidate_key not in id_state["allocated_ids"]:
            id_state["base_counts"][base_key] = count
            id_state["allocated_ids"].add(candidate_key)
            return candidate


def _error_record(
    *,
    compound_id="",
    source_file="",
    location="",
    error_type,
    error,
):
    return {
        "compound_id": str(compound_id),
        "source_file": str(source_file),
        "location": str(location),
        "error_type": str(error_type),
        "error": str(error),
    }


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


def _read_table_file(path):
    extension = os.path.splitext(path)[1].lower()
    if extension == ".csv":
        return {os.path.basename(path): separator_2d._read_csv_table(path)}
    return pd.read_excel(
        path,
        sheet_name=None,
        dtype=object,
        keep_default_na=False,
    )


def _append_table_records(path, records, errors, id_state):
    try:
        tables = _read_table_file(path)
    except Exception as exc:
        errors.append(_error_record(
            source_file=path,
            error_type="table_load_failed",
            error=exc,
        ))
        return

    for sheet_name, original_table in tables.items():
        if original_table is None or len(original_table.columns) == 0:
            continue
        source_label = f"{path}::{sheet_name}"
        try:
            table, lookup = _column_lookup(original_table, source_label)
        except Exception as exc:
            errors.append(_error_record(
                source_file=path,
                location=sheet_name,
                error_type="invalid_columns",
                error=exc,
            ))
            continue

        smiles_column = lookup.get("smiles")
        compound_id_column = lookup.get("compound id")
        if smiles_column is None:
            errors.append(_error_record(
                source_file=path,
                location=sheet_name,
                error_type="missing_smiles_column",
                error="表格缺少 SMILES 列",
            ))
            continue

        source_stem = Path(path).stem
        safe_sheet = _sanitize_compound_id(sheet_name, "sheet")
        for row_index, row in table.iterrows():
            display_row = int(row_index) + 2
            fallback_id = (
                f"{source_stem}_{safe_sheet}_row{display_row:06d}"
            )
            raw_id = (
                row[compound_id_column]
                if compound_id_column is not None
                else fallback_id
            )
            try:
                base_id = _sanitize_compound_id(raw_id, fallback_id)
                compound_id = _allocate_unique_id(base_id, id_state)
            except Exception as exc:
                errors.append(_error_record(
                    source_file=path,
                    location=f"{sheet_name}:row{display_row}",
                    error_type="invalid_compound_id",
                    error=exc,
                ))
                continue
            records.append({
                "compound_id": compound_id,
                "smiles": _normalize_text(row[smiles_column]),
                "sheet_name": str(sheet_name),
                "row_number": display_row,
                "source_file": os.path.abspath(path),
            })


def _append_sdf_records(path, records, errors, id_state):
    try:
        supplier = Chem.SDMolSupplier(
            path,
            removeHs=False,
            sanitize=True,
            strictParsing=True,
        )
    except Exception as exc:
        errors.append(_error_record(
            source_file=path,
            error_type="sdf_load_failed",
            error=exc,
        ))
        return

    source_stem = Path(path).stem
    try:
        iterator = enumerate(supplier, start=1)
        for record_index, molecule in iterator:
            fallback_id = f"{source_stem}_mol{record_index:06d}"
            if molecule is None or molecule.GetNumAtoms() == 0:
                errors.append(_error_record(
                    compound_id=fallback_id,
                    source_file=path,
                    location=f"record{record_index}",
                    error_type="invalid_sdf_record",
                    error="RDKit 无法解析该 SDF 记录",
                ))
                continue
            raw_id = (
                molecule.GetProp("_Name")
                if molecule.HasProp("_Name")
                else fallback_id
            )
            try:
                molecule = _remove_nonisotopic_hydrogens_for_activity(
                    molecule
                )
                smiles = Chem.MolToSmiles(
                    molecule,
                    canonical=True,
                    isomericSmiles=True,
                )
                if not smiles:
                    raise ValueError("无法生成规范 SMILES")
                base_id = _sanitize_compound_id(raw_id, fallback_id)
                compound_id = _allocate_unique_id(base_id, id_state)
            except Exception as exc:
                errors.append(_error_record(
                    compound_id=fallback_id,
                    source_file=path,
                    location=f"record{record_index}",
                    error_type="sdf_to_smiles_failed",
                    error=exc,
                ))
                continue
            records.append({
                "compound_id": compound_id,
                "smiles": smiles,
                "sheet_name": Path(path).name,
                "row_number": record_index,
                "source_file": os.path.abspath(path),
            })
    except Exception as exc:
        errors.append(_error_record(
            source_file=path,
            error_type="sdf_iteration_failed",
            error=exc,
        ))


def collect_input_records(input_dir, manual_smiles=None):
    """从目录或单个受支持文件收集记录，并保留逐记录错误。"""
    records = []
    errors = []
    id_state = {"base_counts": {}, "allocated_ids": set()}
    input_files = []

    input_dir = os.path.abspath(os.fspath(input_dir))
    if os.path.isdir(input_dir):
        input_files = sorted(
            (
                str(path.resolve())
                for path in Path(input_dir).rglob("*")
                if path.is_file()
                and path.suffix.lower() in TABLE_EXTENSIONS | {SDF_EXTENSION}
            ),
            key=lambda value: value.casefold(),
        )
    elif os.path.isfile(input_dir):
        extension = os.path.splitext(input_dir)[1].lower()
        if extension not in TABLE_EXTENSIONS | {SDF_EXTENSION}:
            raise ValueError(f"不支持的输入文件类型: {input_dir}")
        input_files = [input_dir]
    elif not manual_smiles:
        raise FileNotFoundError(f"未知活性输入路径不存在: {input_dir}")

    for path in input_files:
        extension = os.path.splitext(path)[1].lower()
        if extension in TABLE_EXTENSIONS:
            _append_table_records(path, records, errors, id_state)
        elif extension == SDF_EXTENSION:
            _append_sdf_records(path, records, errors, id_state)

    if manual_smiles is None:
        manual_values = []
    elif isinstance(manual_smiles, str):
        manual_values = [manual_smiles]
    else:
        manual_values = list(manual_smiles)
    for manual_index, smiles in enumerate(manual_values, start=1):
        fallback_id = f"manual_{manual_index:06d}"
        compound_id = _allocate_unique_id(fallback_id, id_state)
        records.append({
            "compound_id": compound_id,
            "smiles": _normalize_text(smiles),
            "sheet_name": "manual",
            "row_number": manual_index,
            "source_file": "<manual>",
        })
    return records, errors, input_files


def _atomic_write_json(payload, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temp_path = f"{path}.{os.getpid()}-{uuid.uuid4().hex}.tmp"
    try:
        with open(temp_path, "w", encoding="utf-8") as file_obj:
            json.dump(payload, file_obj, ensure_ascii=False, indent=2)
            file_obj.write("\n")
            file_obj.flush()
            os.fsync(file_obj.fileno())
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def _atomic_write_error_csv(errors, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temp_path = f"{path}.{os.getpid()}-{uuid.uuid4().hex}.tmp"
    try:
        pd.DataFrame(errors, columns=ERROR_COLUMNS).to_csv(
            temp_path,
            index=False,
            encoding="utf-8-sig",
        )
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def _worker_error(worker_result):
    info = worker_result["compound_info"]
    return _error_record(
        compound_id=info.get("compound_id", ""),
        source_file=info.get("source_file", ""),
        location=(
            f"{info.get('sheet_name', '')}:"
            f"{info.get('row_number', '')}"
        ),
        error_type=worker_result.get("error_type") or "processing_failed",
        error=worker_result.get("error") or "未知处理错误",
    )


def process_work_data(
    input_dir=config.WORK_INPUT_DIR,
    output_dir=config.WORK_PYG_DIR,
    manual_smiles=None,
    num_workers=None,
) -> dict:
    """处理未知活性输入目录或单个文件，并返回汇总字典。"""
    config.validate_config()
    RDLogger.DisableLog("rdApp.*")
    output_dir = os.path.abspath(os.fspath(output_dir))
    os.makedirs(output_dir, exist_ok=True)

    requested_workers = (
        config.WORK_PROCESS_WORKERS
        if num_workers is None
        else num_workers
    )
    effective_workers = config.clamp_cpu_count(
        requested_workers,
        config.MAX_WORK_PROCESS_WORKERS,
        "WORK_PROCESS_WORKERS",
    )
    print(
        "工作数据切割进程: "
        f"请求={requested_workers}, "
        f"硬上限={config.MAX_WORK_PROCESS_WORKERS}, "
        f"可用={config.allocated_cpu_count()}, "
        f"实际={effective_workers}"
    )

    records, errors, input_files = collect_input_records(
        input_dir,
        manual_smiles=manual_smiles,
    )
    if not records:
        summary = {
            "input_dir": os.path.abspath(os.fspath(input_dir)),
            "output_dir": output_dir,
            "input_files": input_files,
            "planned_records": 0,
            "saved_pyg": 0,
            "processing_failed": 0,
            "save_failed": 0,
            "overwritten_existing": 0,
            "input_error_count": len(errors),
            "error_count": len(errors),
            "requested_workers": requested_workers,
            "effective_workers": effective_workers,
            "task_timeout_seconds": config.WORK_TASK_TIMEOUT_SECONDS,
        }
        _atomic_write_json(summary, os.path.join(output_dir, SUMMARY_FILENAME))
        _atomic_write_error_csv(errors, os.path.join(output_dir, ERROR_FILENAME))
        raise RuntimeError("没有从输入文件或手工参数中读取到可处理记录")

    normalization_params = separator_2d.load_2d_normalization_params(
        separator_2d.NORM_STATS_FILE
    )
    summary = {
        "input_dir": os.path.abspath(os.fspath(input_dir)),
        "output_dir": output_dir,
        "input_files": input_files,
        "planned_records": len(records),
        "saved_pyg": 0,
        "processing_failed": 0,
        "save_failed": 0,
        "overwritten_existing": 0,
        "input_error_count": len(errors),
        "requested_workers": requested_workers,
        "effective_workers": effective_workers,
        "task_timeout_seconds": config.WORK_TASK_TIMEOUT_SECONDS,
        "feature_dimension": len(separator_2d.FEATURE_COLUMNS_2D),
        "normalization_source": os.path.abspath(
            separator_2d.NORM_STATS_FILE
        ),
        "fragment_embedding_status": (
            "pending_vocabulary_synchronization"
        ),
    }
    for stat_key in separator_2d.WORKER_STAT_KEYS:
        summary[stat_key] = 0
    saved_condition_paths = []

    def handle_result(worker_result):
        for stat_key in separator_2d.WORKER_STAT_KEYS:
            summary[stat_key] += int(
                worker_result["stats"].get(stat_key, 0)
            )
        info = worker_result["compound_info"]
        if not worker_result["success"]:
            summary["processing_failed"] += 1
            errors.append(_worker_error(worker_result))
            return

        try:
            data, _ = separator_2d.assemble_pyg_data(
                worker_result,
                normalization_params,
                include_activity=False,
            )
            data.fragment_smiles = [
                _normalize_activity_fragment_smiles(smiles)
                for smiles in data.fragment_smiles
            ]
            if "y" in data:
                raise RuntimeError("无标签工作数据不应包含 y")
            for unwanted_field in ("frag_embeds", "edge_attr"):
                if unwanted_field in data:
                    del data[unwanted_field]
            target_path = os.path.abspath(os.path.join(
                output_dir,
                f"{info['compound_id']}.pt",
            ))
            if os.path.commonpath([output_dir, target_path]) != output_dir:
                raise ValueError("输出路径越界")
            if os.path.exists(target_path):
                summary["overwritten_existing"] += 1
            separator_2d.save_pyg_data_atomic(data, target_path)
        except Exception as exc:
            summary["save_failed"] += 1
            errors.append(_error_record(
                compound_id=info.get("compound_id", ""),
                source_file=info.get("source_file", ""),
                location=(
                    f"{info.get('sheet_name', '')}:"
                    f"{info.get('row_number', '')}"
                ),
                error_type="save_failed",
                error=exc,
            ))
            return
        summary["saved_pyg"] += 1
        saved_condition_paths.append(target_path)

    if effective_workers == 1:
        for record in tqdm(records, desc="未知活性2D处理"):
            try:
                result = separator_2d.worker_process_compound(record)
            except Exception as exc:
                summary["processing_failed"] += 1
                errors.append(_error_record(
                    compound_id=record.get("compound_id", ""),
                    source_file=record.get("source_file", ""),
                    location=(
                        f"{record.get('sheet_name', '')}:"
                        f"{record.get('row_number', '')}"
                    ),
                    error_type="process_exception",
                    error=exc,
                ))
                continue
            handle_result(result)
    else:
        from concurrent.futures import TimeoutError
        from pebble import ProcessPool

        with ProcessPool(max_workers=effective_workers) as pool:
            future = pool.map(
                separator_2d.worker_process_compound,
                records,
                timeout=config.WORK_TASK_TIMEOUT_SECONDS,
            )
            iterator = future.result()
            progress = tqdm(total=len(records), desc="未知活性2D处理")
            task_index = 0
            while True:
                try:
                    result = next(iterator)
                except StopIteration:
                    break
                except TimeoutError:
                    record = records[task_index]
                    summary["processing_failed"] += 1
                    errors.append(_error_record(
                        compound_id=record.get("compound_id", ""),
                        source_file=record.get("source_file", ""),
                        location=(
                            f"{record.get('sheet_name', '')}:"
                            f"{record.get('row_number', '')}"
                        ),
                        error_type="timeout",
                        error=(
                            "处理时间超过 "
                            f"{config.WORK_TASK_TIMEOUT_SECONDS} 秒"
                        ),
                    ))
                except Exception as exc:
                    record = records[task_index]
                    summary["processing_failed"] += 1
                    errors.append(_error_record(
                        compound_id=record.get("compound_id", ""),
                        source_file=record.get("source_file", ""),
                        location=(
                            f"{record.get('sheet_name', '')}:"
                            f"{record.get('row_number', '')}"
                        ),
                        error_type="process_exception",
                        error=exc,
                    ))
                else:
                    handle_result(result)
                progress.update(1)
                task_index += 1
            progress.close()

    synchronization_error = None
    sync_report_path = os.path.join(
        output_dir, VOCABULARY_SYNC_REPORT_FILENAME
    )
    if saved_condition_paths:
        from PhiSSeparator import PhiSSeparator_vocabulary_sync as vocabulary_sync

        try:
            sync_report = (
                vocabulary_sync.synchronize_condition_vocabularies(
                    saved_condition_paths,
                    device="auto",
                    report_path=sync_report_path,
                )
            )
        except Exception as exc:
            synchronization_error = exc
            summary["fragment_embedding_status"] = (
                "vocabulary_synchronization_failed"
            )
            summary["vocabulary_sync"] = {
                "status": "failed",
                "report_path": os.path.abspath(sync_report_path),
                "condition_file_count": len(saved_condition_paths),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            errors.append(_error_record(
                source_file=os.path.abspath(sync_report_path),
                location="post_processing_vocabulary_sync",
                error_type="vocabulary_sync_failed",
                error=exc,
            ))
        else:
            summary["fragment_embedding_status"] = (
                "synchronized_during_processing"
            )
            summary["vocabulary_sync"] = {
                "status": sync_report["status"],
                "report_path": os.path.abspath(sync_report_path),
                "condition_file_count": sync_report[
                    "condition_file_count"
                ],
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
    else:
        summary["fragment_embedding_status"] = "not_run_no_saved_pyg"
        summary["vocabulary_sync"] = {
            "status": "not_run",
            "reason": "no_saved_condition_files",
            "condition_file_count": 0,
        }

    summary["error_count"] = len(errors)
    _atomic_write_json(summary, os.path.join(output_dir, SUMMARY_FILENAME))
    _atomic_write_error_csv(errors, os.path.join(output_dir, ERROR_FILENAME))
    if synchronization_error is not None:
        raise RuntimeError(
            "条件 .pt 已保留，但完整片段/原子词表同步失败；"
            f"详情见 {sync_report_path}"
        ) from synchronization_error
    print(
        f"工作数据处理完成: 保存={summary['saved_pyg']}, "
        f"处理失败={summary['processing_failed']}, "
        f"保存失败={summary['save_failed']}, 错误总数={len(errors)}"
    )
    print(f"PyG 输出目录: {output_dir}")
    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="把未知活性 CSV/Excel/SDF/手工 SMILES 转换为二维 PyG",
    )
    parser.add_argument(
        "--input-dir",
        default=config.WORK_INPUT_DIR,
        help="CSV、Excel、SDF 的递归输入目录或单个文件",
    )
    parser.add_argument(
        "--output-dir",
        default=config.WORK_PYG_DIR,
        help="无标签二维 .pt 输出目录",
    )
    parser.add_argument(
        "--smiles",
        action="append",
        dest="manual_smiles",
        help="手工输入 SMILES，可重复传入；会与输入目录内容一起处理",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="请求的切割进程数；仍受配置中的硬上限和可用CPU限制",
    )
    return parser.parse_args()


def main():
    multiprocessing.freeze_support()
    args = parse_args()
    process_work_data(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        manual_smiles=args.manual_smiles,
        num_workers=args.workers,
    )


if __name__ == "__main__":
    main()
