"""PhiSGATv2 单任务二维活性回归训练入口。"""

import math
import os
import random
import time

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter

if __package__:
    from . import PhiSGATv2_config as config
    from . import PhiSGATv2_utils as utils
    from .PhiSGATv2_dataset import (
        MoleculeDataset,
        build_fragment_embedding_lookup,
        load_pyg_file,
        select_labeled_pyg_files,
        validate_graph_data,
    )
    from .PhiSGATv2_model import GATv2Model
else:  # 支持直接运行 PhiSGATv2 目录中的脚本
    import PhiSGATv2_config as config
    import PhiSGATv2_utils as utils
    from PhiSGATv2_dataset import (
        MoleculeDataset,
        build_fragment_embedding_lookup,
        load_pyg_file,
        select_labeled_pyg_files,
        validate_graph_data,
    )
    from PhiSGATv2_model import GATv2Model


def set_random_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _relative_to_training_root(path):
    relative = os.path.relpath(
        os.path.abspath(path),
        os.path.abspath(config.TRAIN_PYG_DIR),
    )
    return relative.replace("\\", "/")


def build_random_split_manifest(data_files):
    """以稳定文件列表和独立随机生成器按配置比例构建清单。"""
    num_files = len(data_files)
    if num_files < 2:
        raise ValueError("训练/验证划分至少需要两个有效有标签图")
    train_size = int(math.floor(config.TRAIN_RATIO * num_files))
    train_size = min(max(train_size, 1), num_files - 1)

    generator = torch.Generator().manual_seed(config.RANDOM_SEED)
    permutation = torch.randperm(num_files, generator=generator).tolist()
    train_files = [data_files[index] for index in permutation[:train_size]]
    val_files = [data_files[index] for index in permutation[train_size:]]
    return {
        "data_root": os.path.abspath(config.TRAIN_PYG_DIR),
        "random_seed": int(config.RANDOM_SEED),
        "train_ratio": float(config.TRAIN_RATIO),
        "val_ratio": float(config.VAL_RATIO),
        "train_files": [
            _relative_to_training_root(path) for path in train_files
        ],
        "val_files": [
            _relative_to_training_root(path) for path in val_files
        ],
    }


def resolve_split_manifest(split_manifest):
    """验证检查点中的划分清单并恢复绝对路径。"""
    if not isinstance(split_manifest, dict):
        raise ValueError("数据划分清单必须是字典")

    resolved = {}
    for split_name in ("train_files", "val_files"):
        values = split_manifest.get(split_name)
        if not isinstance(values, list) or not values:
            raise ValueError(f"划分清单 {split_name} 不能为空")
        absolute_paths = []
        for relative_path in values:
            if not isinstance(relative_path, str) or not relative_path:
                raise ValueError(f"{split_name} 包含非法路径")
            absolute_path = os.path.abspath(
                os.path.join(config.TRAIN_PYG_DIR, relative_path)
            )
            if os.path.commonpath([
                os.path.abspath(config.TRAIN_PYG_DIR),
                absolute_path,
            ]) != os.path.abspath(config.TRAIN_PYG_DIR):
                raise ValueError(f"划分路径越界: {relative_path}")
            if not os.path.isfile(absolute_path):
                raise FileNotFoundError(
                    f"断点续训所需 PyG 文件缺失: {absolute_path}"
                )
            absolute_paths.append(absolute_path)
        resolved[split_name] = absolute_paths

    overlap = set(resolved["train_files"]) & set(resolved["val_files"])
    if overlap:
        raise ValueError(f"训练集与验证集存在重复文件: {sorted(overlap)}")
    return resolved["train_files"], resolved["val_files"]


def validate_training_files(paths):
    for path in paths:
        data = load_pyg_file(path)
        validate_graph_data(
            data,
            path,
            require_y=True,
            enforce_train_y_range=True,
        )
        del data


def main():
    config.validate_config()
    set_random_seeds(config.RANDOM_SEED)
    os.makedirs(config.MODEL_OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.TENSORBOARD_LOG_DIR, exist_ok=True)

    print(f"训练设备: {config.DEVICE}")
    print(f"训练 PyG 根目录: {os.path.abspath(config.TRAIN_PYG_DIR)}")
    print(
        "模型结构: "
        f"输入={config.COMBINED_INPUT_DIM}, "
        f"GATv2={config.NUM_LAYERS}层×{config.HEADS}头×"
        f"{config.HIDDEN_CHANNELS}维, "
        f"节点输出={config.FINAL_NODE_DIM}, 单任务回归"
    )

    resume_payload = None
    if config.RESUME_TRAINING:
        resume_payload = utils.load_checkpoint_payload(
            config.RESUME_CHECKPOINT_PATH,
            map_location="cpu",
        )
        split_manifest = resume_payload["split_manifest"]
        train_files, val_files = resolve_split_manifest(split_manifest)
        validate_training_files(train_files + val_files)
        utils.atomic_write_json(
            split_manifest,
            config.SPLIT_MANIFEST_PATH,
        )
        print(
            f"断点续训复用原划分: 训练={len(train_files)}, "
            f"验证={len(val_files)}"
        )
    else:
        labeled_files, missing_label_files = select_labeled_pyg_files(
            config.TRAIN_PYG_DIR
        )
        if missing_label_files:
            print(
                f"缺少 y、已排除的 PyG: {len(missing_label_files)} 个"
            )
        if len(labeled_files) < 2:
            raise RuntimeError(
                "有效有标签 PyG 少于两个，无法划分训练集和验证集"
            )
        split_manifest = build_random_split_manifest(labeled_files)
        train_files, val_files = resolve_split_manifest(split_manifest)
        utils.atomic_write_json(
            split_manifest,
            config.SPLIT_MANIFEST_PATH,
        )
        print(
            f"已按种子 {config.RANDOM_SEED} 随机划分: "
            f"训练={len(train_files)}, 验证={len(val_files)}"
        )

    all_split_files = train_files + val_files
    embedding_lookup = build_fragment_embedding_lookup(
        all_split_files,
        cpu_threads=config.RESOLVER_CPU_THREADS,
    )
    print(f"本次内存片段嵌入映射: {len(embedding_lookup)} 个键")

    # Resolver 是否需要编码缺词不应改变新训练的模型初始化与批次随机顺序。
    set_random_seeds(config.RANDOM_SEED)

    train_dataset = MoleculeDataset(
        train_files,
        embedding_lookup,
        require_y=True,
    )
    val_dataset = MoleculeDataset(
        val_files,
        embedding_lookup,
        require_y=True,
    )
    pin_memory = config.DEVICE.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.DATA_LOADER_WORKERS,
        pin_memory=pin_memory,
        exclude_keys=config.TRAIN_EXCLUDE_KEYS,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA_LOADER_WORKERS,
        pin_memory=pin_memory,
        exclude_keys=config.TRAIN_EXCLUDE_KEYS,
    )

    model = GATv2Model().to(config.DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.EPOCHS,
        eta_min=config.MIN_LEARNING_RATE,
    )
    print(
        "可训练参数量: "
        f"{sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad):,}"
    )

    start_epoch = 1
    best_val_loss = float("inf")
    early_stopping_counter = 0
    if resume_payload is not None:
        restored = utils.restore_training_state(
            resume_payload,
            model,
            optimizer,
            scheduler,
            config.DEVICE,
        )
        start_epoch = restored["start_epoch"]
        best_val_loss = restored["best_val_loss"]
        early_stopping_counter = restored[
            "early_stopping_counter"
        ]
        print(
            f"已恢复 Epoch {start_epoch - 1}，下一轮={start_epoch}，"
            f"最佳验证混合损失={best_val_loss:.6f}"
        )

    if start_epoch > config.EPOCHS:
        print(
            f"检查点已完成 {start_epoch - 1} 轮，"
            f"不低于当前 EPOCHS={config.EPOCHS}，无需继续训练。"
        )
        return

    run_type = "resume" if resume_payload is not None else "train"
    run_timestamp = time.strftime("%Y%m%d-%H%M%S")
    tensorboard_run_dir = os.path.join(
        config.TENSORBOARD_LOG_DIR,
        f"{run_type}-{run_timestamp}-pid{os.getpid()}",
    )
    writer = SummaryWriter(log_dir=tensorboard_run_dir)
    print(f"TensorBoard 日志: {tensorboard_run_dir}")

    try:
        training_start = time.time()
        for epoch in range(start_epoch, config.EPOCHS + 1):
            epoch_start = time.time()
            train_loss = utils.train_epoch(
                model,
                train_loader,
                optimizer,
                config.DEVICE,
            )
            validation = utils.evaluate(model, val_loader, config.DEVICE)
            scheduler.step()

            improved = validation["composite_loss"] < (
                best_val_loss - config.EARLY_STOPPING_MIN_DELTA
            )
            if improved:
                best_val_loss = validation["composite_loss"]
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1

            checkpoint_payload = utils.build_checkpoint_payload(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                best_val_loss=best_val_loss,
                early_stopping_counter=early_stopping_counter,
                split_manifest=split_manifest,
            )
            utils.atomic_torch_save(
                checkpoint_payload,
                config.LAST_CHECKPOINT_PATH,
            )
            if improved:
                utils.atomic_torch_save(
                    checkpoint_payload,
                    config.BEST_CHECKPOINT_PATH,
                )

            current_lr = float(optimizer.param_groups[0]["lr"])
            epoch_seconds = time.time() - epoch_start
            writer.add_scalar("Loss/train_composite", train_loss, epoch)
            writer.add_scalar(
                "Loss/validation_composite",
                validation["composite_loss"],
                epoch,
            )
            writer.add_scalar(
                "Metrics/validation_smooth_l1",
                validation["smooth_l1"],
                epoch,
            )
            writer.add_scalar("Metrics/validation_mse", validation["mse"], epoch)
            writer.add_scalar(
                "Metrics/validation_rmse",
                validation["rmse"],
                epoch,
            )
            writer.add_scalar("Metrics/validation_mae", validation["mae"], epoch)
            writer.flush()

            print(
                f"Epoch {epoch:03d}/{config.EPOCHS} | "
                f"Time={epoch_seconds:.2f}s | "
                f"LR={current_lr:.7g} | "
                f"Train Mixed={train_loss:.4f} | "
                f"Val Mixed={validation['composite_loss']:.4f} | "
                f"Val Smooth L1={validation['smooth_l1']:.4f} | "
                f"Val MSE={validation['mse']:.4f} | "
                f"RMSE={validation['rmse']:.4f} | "
                f"MAE={validation['mae']:.4f} | "
                f"EarlyStop={early_stopping_counter}/"
                f"{config.EARLY_STOPPING_PATIENCE}"
            )
            if improved:
                print(
                    "    -> 最佳检查点已更新: "
                    f"{config.BEST_CHECKPOINT_PATH}"
                )

            if early_stopping_counter >= config.EARLY_STOPPING_PATIENCE:
                print(
                    f"验证损失连续 {early_stopping_counter} 轮未达到最小改善量，"
                    "触发早停。"
                )
                break

        print(
            f"训练结束，总耗时={(time.time() - training_start) / 60.0:.2f}分钟，"
            f"最佳验证混合损失={best_val_loss:.6f}"
        )
        print(f"最佳检查点: {config.BEST_CHECKPOINT_PATH}")
        print(f"最近检查点: {config.LAST_CHECKPOINT_PATH}")
    finally:
        writer.close()


if __name__ == "__main__":
    main()
