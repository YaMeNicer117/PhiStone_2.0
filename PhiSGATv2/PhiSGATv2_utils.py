"""PhiSGATv2 单任务训练、评估和完整检查点工具。"""

import json
import os
import random
import uuid

import numpy as np
import torch
import torch.nn.functional as F

if __package__:
    from . import PhiSGATv2_config as config
else:  # 支持直接运行 GATv2 目录中的脚本
    import PhiSGATv2_config as config


def prepare_single_task_labels(labels, predictions):
    """将 PyG 拼批后的连续 y 整理为与模型输出相同的 [B, 1]。"""
    if not torch.is_tensor(labels):
        raise TypeError("batch.y 必须是张量")
    if predictions.dim() != 2 or predictions.size(1) != 1:
        raise ValueError(
            f"模型预测必须为 [B, 1]，实际为 {tuple(predictions.shape)}"
        )
    labels = labels.to(predictions.device).float().reshape(-1, 1)
    if labels.size(0) != predictions.size(0):
        raise ValueError(
            "y 数量与图数量不一致: "
            f"{labels.size(0)} != {predictions.size(0)}"
        )
    if not torch.isfinite(labels).all():
        raise ValueError("batch.y 包含 NaN 或 Inf")
    if bool(
        (
            (labels < config.TRAIN_ACTIVITY_MIN)
            | (labels > config.TRAIN_ACTIVITY_MAX)
        ).any()
    ):
        raise ValueError(
            "batch.y 超出训练闭区间 "
            f"[{config.TRAIN_ACTIVITY_MIN}, {config.TRAIN_ACTIVITY_MAX}]"
        )
    return labels


def single_task_composite_loss(predictions, labels):
    labels = prepare_single_task_labels(labels, predictions)
    smooth_l1 = F.smooth_l1_loss(
        predictions,
        labels,
        beta=config.SMOOTH_L1_BETA,
        reduction="none",
    )
    mse = F.mse_loss(predictions, labels, reduction="none")
    mse_weight = config.MSE_AUX_WEIGHT
    composite_loss = (
        (1.0 - mse_weight) * smooth_l1
        + mse_weight * mse
    )
    return composite_loss.mean()


def train_epoch(model, loader, optimizer, device):
    """执行一个训练轮次，并进行非有限值防护与梯度裁剪。"""
    if len(loader.dataset) == 0:
        raise ValueError("训练数据集为空")
    model.train()
    weighted_loss_sum = 0.0
    total_graphs = 0
    skipped_loss_batches = 0
    skipped_gradient_batches = 0

    for batch_index, batch in enumerate(loader):
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        predictions = model(batch)
        loss = single_task_composite_loss(predictions, batch.y)

        if not torch.isfinite(loss):
            skipped_loss_batches += 1
            optimizer.zero_grad(set_to_none=True)
            print(
                f"    [警告] Batch {batch_index:04d}/{len(loader)} "
                "损失不是有限值，已跳过"
            )
            continue

        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=config.GRAD_CLIP_MAX_NORM,
        )
        if not torch.isfinite(gradient_norm):
            skipped_gradient_batches += 1
            optimizer.zero_grad(set_to_none=True)
            print(
                f"    [警告] Batch {batch_index:04d}/{len(loader)} "
                "梯度范数不是有限值，已跳过"
            )
            continue

        optimizer.step()
        num_graphs = int(batch.num_graphs)
        weighted_loss_sum += float(loss.item()) * num_graphs
        total_graphs += num_graphs

        if (
            config.TRAIN_LOG_INTERVAL > 0
            and batch_index % config.TRAIN_LOG_INTERVAL == 0
        ):
            print(
                f"    [Batch {batch_index:04d}/{len(loader)}] "
                f"Loss={loss.item():.4f} | "
                f"GradNorm(before clip)={gradient_norm.item():.4f} | "
                f"ClipMax={config.GRAD_CLIP_MAX_NORM:.2f}"
            )

    if skipped_loss_batches or skipped_gradient_batches:
        print(
            "    [警告] 本轮跳过批次: "
            f"非有限损失={skipped_loss_batches}, "
            f"非有限梯度={skipped_gradient_batches}"
        )
    if total_graphs == 0:
        raise RuntimeError("本轮没有成功完成任何训练 batch")
    return weighted_loss_sum / total_graphs


def evaluate(model, loader, device):
    """计算单任务验证混合损失及其诊断指标。"""
    if len(loader.dataset) == 0:
        raise ValueError("验证数据集为空")
    model.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            predictions = model(batch)
            labels = prepare_single_task_labels(batch.y, predictions)
            if not torch.isfinite(predictions).all():
                raise RuntimeError("验证预测包含 NaN 或 Inf")
            all_predictions.append(
                predictions.detach().cpu().numpy().reshape(-1)
            )
            all_labels.append(labels.detach().cpu().numpy().reshape(-1))

    if not all_labels:
        raise RuntimeError("验证阶段没有产生任何结果")
    labels_array = np.concatenate(all_labels).astype(np.float64, copy=False)
    predictions_array = np.concatenate(all_predictions).astype(
        np.float64,
        copy=False,
    )
    errors = predictions_array - labels_array
    absolute_errors = np.abs(errors)
    beta = float(config.SMOOTH_L1_BETA)
    smooth_l1_values = np.where(
        absolute_errors < beta,
        0.5 * np.square(errors) / beta,
        absolute_errors - 0.5 * beta,
    )
    squared_errors = np.square(errors)
    smooth_l1 = float(np.mean(smooth_l1_values))
    mse = float(np.mean(squared_errors))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(absolute_errors))
    mse_weight = float(config.MSE_AUX_WEIGHT)
    composite_loss_values = (
        (1.0 - mse_weight) * smooth_l1_values
        + mse_weight * squared_errors
    )
    composite_loss = float(np.mean(composite_loss_values))
    return {
        "composite_loss": composite_loss,
        "smooth_l1": smooth_l1,
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
    }


def model_config_snapshot():
    """返回决定检查点结构兼容性的模型配置。"""
    return {
        "input_dim_numeric": int(config.INPUT_DIM_NUMERIC),
        "embedding_dim": int(config.EMBEDDING_DIM),
        "hidden_channels": int(config.HIDDEN_CHANNELS),
        "heads": int(config.HEADS),
        "num_layers": int(config.NUM_LAYERS),
        "final_node_dim": int(config.FINAL_NODE_DIM),
        "mlp_hidden_dim": int(config.MLP_HIDDEN_DIM),
        "dropout": float(config.DROPOUT),
        "use_virtual_node": config.USE_VIRTUAL_NODE,
        "virtual_node_gate_bias": float(
            config.VIRTUAL_NODE_GATE_BIAS
        ),
        "output_min": float(config.OUTPUT_MIN),
        "output_max": float(config.OUTPUT_MAX),
    }


def training_config_snapshot():
    """保存便于复现实验但不决定权重张量形状的训练配置。"""
    return {
        "random_seed": int(config.RANDOM_SEED),
        "train_ratio": float(config.TRAIN_RATIO),
        "val_ratio": float(config.VAL_RATIO),
        "batch_size": int(config.BATCH_SIZE),
        "epochs": int(config.EPOCHS),
        "learning_rate": float(config.LEARNING_RATE),
        "min_learning_rate": float(config.MIN_LEARNING_RATE),
        "weight_decay": float(config.WEIGHT_DECAY),
        "smooth_l1_beta": float(config.SMOOTH_L1_BETA),
        "mse_aux_weight": float(config.MSE_AUX_WEIGHT),
        "grad_clip_max_norm": float(config.GRAD_CLIP_MAX_NORM),
        "train_activity_min": float(config.TRAIN_ACTIVITY_MIN),
        "train_activity_max": float(config.TRAIN_ACTIVITY_MAX),
        "early_stopping_patience": int(
            config.EARLY_STOPPING_PATIENCE
        ),
        "early_stopping_min_delta": float(
            config.EARLY_STOPPING_MIN_DELTA
        ),
    }


def capture_rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": None,
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state):
    if not isinstance(state, dict):
        raise ValueError("检查点随机状态格式非法")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def atomic_torch_save(payload, path):
    """使用同目录临时文件原子保存 PyTorch 对象。"""
    path = os.path.abspath(os.fspath(path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temp_path = f"{path}.{os.getpid()}-{uuid.uuid4().hex}.tmp"
    try:
        torch.save(payload, temp_path)
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def atomic_write_json(payload, path):
    """原子写入 UTF-8 JSON。"""
    path = os.path.abspath(os.fspath(path))
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


def build_checkpoint_payload(
    *,
    epoch,
    model,
    optimizer,
    scheduler,
    best_val_loss,
    early_stopping_counter,
    split_manifest,
):
    return {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_loss": float(best_val_loss),
        "early_stopping_counter": int(early_stopping_counter),
        "model_config": model_config_snapshot(),
        "training_config": training_config_snapshot(),
        "split_manifest": split_manifest,
        "rng_state": capture_rng_state(),
    }


def load_checkpoint_payload(path, map_location):
    path = os.path.abspath(os.fspath(path))
    if not os.path.isfile(path):
        raise FileNotFoundError(f"检查点不存在: {path}")
    payload = torch.load(
        path,
        map_location=map_location,
        weights_only=False,
    )
    if not isinstance(payload, dict):
        raise ValueError(f"检查点不是字典格式: {path}")
    required = {
        "epoch",
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "best_val_loss",
        "early_stopping_counter",
        "model_config",
        "training_config",
        "split_manifest",
        "rng_state",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"检查点缺少字段: {sorted(missing)}")
    # 加入开关前的模型始终启用虚拟节点；仅补齐内存中的配置，不改写权重文件。
    payload["model_config"].setdefault("use_virtual_node", True)
    expected_model_config = model_config_snapshot()
    if payload["model_config"] != expected_model_config:
        raise ValueError(
            "检查点模型配置与当前代码不一致。"
            f"\n检查点: {payload['model_config']}"
            f"\n当前配置: {expected_model_config}"
        )
    return payload


def _move_optimizer_state_to_device(optimizer, device):
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def restore_training_state(
    payload,
    model,
    optimizer,
    scheduler,
    device,
):
    """恢复完整训练状态并返回下一轮编号及早停状态。"""
    model.load_state_dict(payload["model_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    _move_optimizer_state_to_device(optimizer, device)
    scheduler.load_state_dict(payload["scheduler_state_dict"])
    restore_rng_state(payload["rng_state"])
    return {
        "start_epoch": int(payload["epoch"]) + 1,
        "best_val_loss": float(payload["best_val_loss"]),
        "early_stopping_counter": int(
            payload["early_stopping_counter"]
        ),
    }
