"""使用完整最佳检查点预测未知活性二维 PyG 数据。"""

from __future__ import annotations

import argparse
import glob
import os
import uuid

import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from torch_geometric.loader import DataLoader

if __package__:
    from . import PhiSGATv2_config as config
    from . import PhiSGATv2_utils as utils
    from .PhiSGATv2_dataset import (
        MoleculeDataset,
        build_fragment_embedding_lookup,
        discover_pyg_files,
        validate_prediction_files,
    )
    from .PhiSGATv2_model import GATv2Model
else:  # 支持直接运行 PhiSGATv2 目录中的脚本
    import PhiSGATv2_config as config
    import PhiSGATv2_utils as utils
    from PhiSGATv2_dataset import (
        MoleculeDataset,
        build_fragment_embedding_lookup,
        discover_pyg_files,
        validate_prediction_files,
    )
    from PhiSGATv2_model import GATv2Model


RESULT_COLUMNS = (
    "file_path",
    "compound_id",
    "smiles",
    "predicted_activity",
)
ERROR_COLUMNS = ("file_path", "error")


def _atomic_write_csv(records, columns, path):
    path = os.path.abspath(os.fspath(path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temp_path = f"{path}.{os.getpid()}-{uuid.uuid4().hex}.tmp"
    try:
        pd.DataFrame(records, columns=columns).to_csv(
            temp_path,
            index=False,
            encoding="utf-8-sig",
        )
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def _write_prediction_images(records, image_dir):
    """按预测结果顺序绘制分子结构、名称和预测活性。"""
    image_dir = os.path.abspath(os.fspath(image_dir))
    os.makedirs(image_dir, exist_ok=True)

    old_image_pattern = os.path.join(
        image_dir,
        f"{config.PREDICTION_IMAGE_FILENAME_PREFIX}_*.png",
    )
    for old_image_path in glob.glob(old_image_pattern):
        if os.path.isfile(old_image_path):
            os.remove(old_image_path)

    drawable_mols = []
    legends = []
    # 图片按照预测活性从大到小排列；分数相同时保持 CSV 中的原始顺序。
    ordered_records = sorted(
        enumerate(records, start=1),
        key=lambda item: (-float(item[1]["predicted_activity"]), item[0]),
    )
    for original_index, record in ordered_records:
        smiles = str(record.get("smiles", "")).strip()
        compound_id = str(
            record.get("compound_id") or f"molecule_{original_index:04d}"
        )

        if not smiles:
            print(f"[绘图警告] {compound_id} 缺少 SMILES，已跳过图片。")
            continue

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError("RDKit 无法解析 SMILES")
            for atom in mol.GetAtoms():
                atom.SetAtomMapNum(0)
            AllChem.Compute2DCoords(mol)
        except Exception as exc:
            print(f"[绘图警告] {compound_id} 无法生成二维结构: {exc}")
            continue

        predicted_activity = float(record["predicted_activity"])
        drawable_mols.append(mol)
        legends.append(
            f"{compound_id}\nPredicted activity: {predicted_activity:.4f}"
        )

    image_paths = []
    for batch_index, start in enumerate(
        range(0, len(drawable_mols), config.PREDICTION_MAX_MOLS_PER_IMAGE),
        start=1,
    ):
        end = start + config.PREDICTION_MAX_MOLS_PER_IMAGE
        batch_mols = drawable_mols[start:end]
        batch_legends = legends[start:end]
        image = Draw.MolsToGridImage(
            batch_mols,
            molsPerRow=min(config.PREDICTION_MOLS_PER_ROW, len(batch_mols)),
            subImgSize=config.PREDICTION_IMAGE_SUB_SIZE,
            legends=batch_legends,
            useSVG=False,
            returnPNG=False,
        )
        image_path = os.path.join(
            image_dir,
            f"{config.PREDICTION_IMAGE_FILENAME_PREFIX}_{batch_index:03d}.png",
        )
        image.save(image_path)
        image_paths.append(image_path)

    return image_paths


def _resolve_requested_files(data_dir, file_names):
    all_files = discover_pyg_files(data_dir)
    if file_names is None:
        return all_files

    if isinstance(file_names, (str, os.PathLike)):
        file_names = [file_names]
    else:
        file_names = list(file_names)
    basename_index = {}
    for path in all_files:
        basename_index.setdefault(os.path.basename(path), []).append(path)

    resolved = []
    seen = set()
    for requested in file_names:
        requested = os.fspath(requested)
        direct_path = (
            os.path.abspath(requested)
            if os.path.isabs(requested)
            else os.path.abspath(os.path.join(data_dir, requested))
        )
        if os.path.isfile(direct_path):
            selected = direct_path
        else:
            matches = basename_index.get(os.path.basename(requested), [])
            if not matches:
                raise FileNotFoundError(f"找不到指定 PyG 文件: {requested}")
            if len(matches) > 1:
                raise ValueError(
                    f"文件名 {requested!r} 在目录中不唯一，请传入相对路径: "
                    f"{matches}"
                )
            selected = matches[0]
        try:
            common_root = os.path.commonpath([data_dir, selected])
        except ValueError as exc:
            raise ValueError(
                f"指定 PyG 文件不在 data_dir 内: {requested}"
            ) from exc
        if os.path.normcase(common_root) != os.path.normcase(data_dir):
            raise ValueError(f"指定 PyG 文件不在 data_dir 内: {requested}")
        normalized_key = os.path.normcase(os.path.abspath(selected))
        if normalized_key not in seen:
            seen.add(normalized_key)
            resolved.append(os.path.abspath(selected))
    return resolved


def _batch_strings(batch, field_name, count, fallback_values=None):
    value = getattr(batch, field_name, None)
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, (list, tuple)):
        values = [str(item) for item in value]
    elif value is None and fallback_values is not None:
        values = list(fallback_values)
    else:
        raise ValueError(f"预测 batch 的 {field_name} 不是字符串列表")
    if len(values) != count:
        raise ValueError(
            f"预测 batch 的 {field_name} 数量 {len(values)} "
            f"与图数量 {count} 不一致"
        )
    return values


def predict_activity(
    data_dir=config.WORK_PYG_DIR,
    file_names=None,
    checkpoint_path=config.BEST_CHECKPOINT_PATH,
    device=None,
    result_path=config.PREDICTION_RESULT_PATH,
    error_path=config.PREDICTION_ERROR_PATH,
    image_dir=None,
    data_loader_workers=None,
) -> list[dict]:
    """预测全部或指定 PyG，并按需输出结果表和分子结构图。"""
    config.validate_config()
    data_dir = os.path.abspath(os.fspath(data_dir))
    selected_files = _resolve_requested_files(data_dir, file_names)
    if not selected_files:
        raise RuntimeError(f"预测目录中没有 .pt 文件: {data_dir}")

    valid_files, errors = validate_prediction_files(selected_files)
    if not valid_files:
        _atomic_write_csv(
            errors,
            ERROR_COLUMNS,
            error_path,
        )
        raise RuntimeError("没有通过接口校验的预测 PyG 文件")

    prediction_device = (
        config.DEVICE if device is None else torch.device(device)
    )
    checkpoint = utils.load_checkpoint_payload(
        checkpoint_path,
        map_location="cpu",
    )
    model = GATv2Model()
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(prediction_device)
    model.eval()
    del checkpoint

    embedding_lookup = build_fragment_embedding_lookup(
        valid_files,
        cpu_threads=config.RESOLVER_CPU_THREADS,
    )
    dataset = MoleculeDataset(
        valid_files,
        embedding_lookup,
        require_y=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=(
            config.DATA_LOADER_WORKERS
            if data_loader_workers is None else data_loader_workers
        ),
        pin_memory=prediction_device.type == "cuda",
        exclude_keys=config.PREDICT_EXCLUDE_KEYS,
    )

    results = []
    with torch.no_grad():
        for batch in loader:
            num_graphs = int(batch.num_graphs)
            file_paths = _batch_strings(
                batch,
                "pyg_file_path",
                num_graphs,
            )
            fallback_ids = [
                os.path.splitext(os.path.basename(path))[0]
                for path in file_paths
            ]
            try:
                compound_ids = _batch_strings(
                    batch,
                    "compound_id",
                    num_graphs,
                    fallback_values=fallback_ids,
                )
            except ValueError:
                compound_ids = fallback_ids
            try:
                smiles_values = _batch_strings(
                    batch,
                    "smiles",
                    num_graphs,
                    fallback_values=[""] * num_graphs,
                )
            except ValueError:
                smiles_values = [""] * num_graphs

            try:
                batch = batch.to(prediction_device)
                predictions = model(batch).detach().cpu().reshape(-1)
                if predictions.numel() != num_graphs:
                    raise RuntimeError(
                        "预测数量与图数量不一致: "
                        f"{predictions.numel()} != {num_graphs}"
                    )
                if not torch.isfinite(predictions).all():
                    raise RuntimeError("预测结果包含 NaN 或 Inf")
            except Exception as exc:
                errors.extend({
                    "file_path": path,
                    "error": f"batch_prediction_failed: {exc}",
                } for path in file_paths)
                continue

            for path, compound_id, smiles, prediction in zip(
                file_paths,
                compound_ids,
                smiles_values,
                predictions.tolist(),
            ):
                results.append({
                    "file_path": path,
                    "compound_id": compound_id,
                    "smiles": smiles,
                    "predicted_activity": float(prediction),
                })

    _atomic_write_csv(
        results,
        RESULT_COLUMNS,
        result_path,
    )
    _atomic_write_csv(
        errors,
        ERROR_COLUMNS,
        error_path,
    )
    if not results:
        raise RuntimeError(
            f"所有预测 batch 均失败，详见 {os.path.abspath(error_path)}"
        )
    if image_dir is not None:
        image_paths = _write_prediction_images(results, image_dir)
        print(f"分子结构图片: {len(image_paths)} 张")
    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="预测 processed_2d 中全部或指定二维 PyG 的连续活性",
    )
    parser.add_argument(
        "--data-dir",
        default=config.WORK_PYG_DIR,
        help="待预测 .pt 根目录；默认使用 Datas/work_datas/processed_2d",
    )
    parser.add_argument(
        "--file",
        action="append",
        dest="file_names",
        help="只预测指定文件；可重复传入，省略时递归预测全部 .pt",
    )
    parser.add_argument(
        "--checkpoint",
        default=config.BEST_CHECKPOINT_PATH,
        help="完整最佳检查点路径",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="覆盖默认推理设备，例如 cpu 或 cuda:0",
    )
    parser.add_argument(
        "--image-dir",
        default=config.PREDICTION_IMAGE_DIR,
        help="二维分子结构图输出目录",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    results = predict_activity(
        data_dir=args.data_dir,
        file_names=args.file_names,
        checkpoint_path=args.checkpoint,
        device=args.device,
        image_dir=args.image_dir,
    )
    print("\n--- 连续活性预测结果 ---")
    for record in results:
        print(
            f"{record['compound_id']}: "
            f"{record['predicted_activity']}"
        )
    print(f"结果文件: {config.PREDICTION_RESULT_PATH}")
    print(f"错误文件: {config.PREDICTION_ERROR_PATH}")
    print(f"分子图片目录: {os.path.abspath(args.image_dir)}")


if __name__ == "__main__":
    main()
