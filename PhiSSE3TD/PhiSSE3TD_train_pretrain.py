import contextlib
import math
import os
import sys

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import PhiSSE3TD_config as config
from PhiSSE3TD_dataset import (
    build_training_data_profile,
    get_train_val_dataloaders,
    save_training_data_profile,
)
from PhiSSE3TD_diffusion import DiffusionProcess
from PhiSSE3TD_model import E3NNTransformerDiffusion
from PhiSSE3TD_utils_geom import (
    build_covalent_edge_targets,
    build_dynamic_edge_targets,
    compute_covalent_edge_loss,
    compute_edge_topology_loss,
    compute_embedding_cosine_loss,
    compute_feature_noise_losses,
    compute_frame_atom_geometry_losses,
    select_dynamic_edge_samples,
)


TENSORBOARD_WEIGHTED_LOSS_NAMES = (
    'embed_noise',
    'chem_noise',
    'pos',
    'frame',
    'edge',
    'covalent',
    'frame_ll',
    'frame_lp',
    'frame_null',
    'frame_gl',
    'embed_cos',
    'hac',
    'ring',
)
TENSORBOARD_LOSS_METRIC_NAMES = ('total',) + tuple(
    f'weighted/{name}' for name in TENSORBOARD_WEIGHTED_LOSS_NAMES
)
TENSORBOARD_VALIDATION_QUALITY_NAMES = (
    'hac_macro_f1',
    'ring_macro_f1',
    'embedding_cosine',
    'edge_ll_recall',
    'edge_lp_recall',
    'edge_null_recall',
    'covalent_f1',
)


def build_loss_weight_snapshot():
    """记录所有参与总损失加权的 W_* 配置，供断点恢复时比较。"""
    return {
        name: float(getattr(config, name))
        for name in sorted(vars(config))
        if name.startswith('W_')
    }


def describe_loss_weight_changes(checkpoint_snapshot, current_snapshot):
    """返回检查点与当前损失权重之间的可读差异。"""
    if not isinstance(checkpoint_snapshot, dict):
        return ['检查点未记录 loss_weight_snapshot']

    changes = []
    for name in sorted(set(checkpoint_snapshot) | set(current_snapshot)):
        previous = checkpoint_snapshot.get(name, '<缺失>')
        current = current_snapshot.get(name, '<缺失>')
        if previous != current:
            changes.append(f'{name}: {previous} -> {current}')
    return changes


def build_tensorboard_custom_layout():
    """将标量按训练损失、验证损失和低噪声验证质量分区展示。"""
    return {
        'Train Losses': {
            metric_name.split('/', 1)[-1]: [
                'Multiline',
                [f'Train/{metric_name}'],
            ]
            for metric_name in TENSORBOARD_LOSS_METRIC_NAMES
        },
        'Validation Losses': {
            metric_name.split('/', 1)[-1]: [
                'Multiline',
                [f'Validation/{metric_name}'],
            ]
            for metric_name in TENSORBOARD_LOSS_METRIC_NAMES
        },
        'Validation Quality - Low Noise': {
            metric_name: [
                'Multiline',
                [f'Validation/low_noise_metric/{metric_name}'],
            ]
            for metric_name in TENSORBOARD_VALIDATION_QUALITY_NAMES
        },
    }


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def reduce_metric_dict(metrics, device):
    if not dist.is_initialized():
        return metrics
    keys = sorted(metrics)
    values = torch.tensor([metrics[key] for key in keys], device=device, dtype=torch.float64)
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    values /= dist.get_world_size()
    return {key: float(value) for key, value in zip(keys, values.tolist())}


def low_noise_aux_schedule(timesteps, num_timesteps):
    """高噪声保持小权重，在最后低噪声阶段用余弦曲线升至目标权重。"""
    start_ratio = float(config.LOW_NOISE_AUX_START_RATIO)
    minimum_ratio = float(config.HIGH_NOISE_AUX_WEIGHT_RATIO)
    if not 0.0 < start_ratio <= 1.0:
        raise ValueError("LOW_NOISE_AUX_START_RATIO 必须位于 (0,1]")
    if not 0.0 <= minimum_ratio <= 1.0:
        raise ValueError("HIGH_NOISE_AUX_WEIGHT_RATIO 必须位于 [0,1]")

    normalized_t = timesteps.float() / max(1, num_timesteps - 1)
    progress = torch.clamp(
        (start_ratio - normalized_t) / start_ratio,
        min=0.0,
        max=1.0,
    )
    smooth_progress = 0.5 - 0.5 * torch.cos(math.pi * progress)
    return minimum_ratio + (1.0 - minimum_ratio) * smooth_progress


def encode_hac_targets(hac, profile):
    cap = int(profile['hac_cap'])
    return torch.where(hac <= cap, hac - 1, torch.full_like(hac, cap)).long()


def encode_ring_targets(ring_count, profile):
    cap = int(profile['ring_cap'])
    return torch.where(
        ring_count <= cap, ring_count, torch.full_like(ring_count, cap + 1)
    ).long()


def weighted_cross_entropy(logits, targets, class_weights, sample_weights):
    if logits.shape[0] == 0:
        return logits.sum() * 0.0
    weights = torch.as_tensor(
        class_weights, device=logits.device, dtype=logits.dtype
    )
    per_sample = F.cross_entropy(logits, targets, weight=weights, reduction='none')
    valid = weights[targets] > 0
    if not valid.any():
        return logits.sum() * 0.0
    scheduled = sample_weights.to(per_sample.dtype)
    # 除以有效样本数而非时间权重之和，保留低噪声调度的绝对强度。
    return (per_sample[valid] * scheduled[valid]).sum() / valid.sum()


def binary_precision_recall_f1(logits, targets, sample_mask):
    if logits.numel() == 0 or not sample_mask.any():
        return 0.0, 0.0, 0.0
    predictions = (
        torch.sigmoid(logits[sample_mask]) >= config.COVALENT_PROB_THRESHOLD
    )
    positives = targets[sample_mask].bool()
    true_positive = (predictions & positives).sum().float()
    false_positive = (predictions & ~positives).sum().float()
    false_negative = (~predictions & positives).sum().float()
    precision = true_positive / (true_positive + false_positive).clamp_min(1.0)
    recall = true_positive / (true_positive + false_negative).clamp_min(1.0)
    f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1.0e-8)
    return float(precision), float(recall), float(f1)


def macro_f1(logits, targets, num_classes):
    if logits.shape[0] == 0:
        return 0.0
    predictions = logits.argmax(dim=-1)
    scores = []
    for class_index in range(num_classes):
        true_positive = ((predictions == class_index) & (targets == class_index)).sum()
        false_positive = ((predictions == class_index) & (targets != class_index)).sum()
        false_negative = ((predictions != class_index) & (targets == class_index)).sum()
        denominator = 2 * true_positive + false_positive + false_negative
        if denominator.item() > 0:
            scores.append((2 * true_positive.float() / denominator.float()).item())
    return sum(scores) / len(scores) if scores else 0.0


def class_recall(logits, targets, sample_mask, class_index):
    class_mask = sample_mask & (targets == class_index)
    if not class_mask.any():
        return 0.0
    return (
        (logits.argmax(dim=-1)[class_mask] == class_index).float().mean().item()
    )


def build_fixed_ligand_mask(ligand_batch_indices):
    """40% Creative；其余样本保留 10%~60% ligand 作为干净锚点。"""
    fixed = torch.zeros(
        ligand_batch_indices.shape[0],
        dtype=torch.bool,
        device=ligand_batch_indices.device,
    )
    for graph_id in torch.unique(ligand_batch_indices):
        graph_nodes = torch.nonzero(
            ligand_batch_indices == graph_id, as_tuple=False
        ).flatten()
        num_nodes = graph_nodes.numel()
        if num_nodes <= 1 or torch.rand((), device=ligand_batch_indices.device) < 0.4:
            continue
        keep_ratio = torch.empty(
            (), device=ligand_batch_indices.device
        ).uniform_(0.1, 0.6)
        num_fixed = max(1, min(int(num_nodes * float(keep_ratio)), num_nodes - 1))
        permutation = torch.randperm(num_nodes, device=ligand_batch_indices.device)
        fixed[graph_nodes[permutation[:num_fixed]]] = True
    return fixed


def compute_batch_objective(model, batch, diffusion, profile, amp_dtype):
    device = batch.pos.device
    is_ligand = batch.is_ligand
    ligand_batch = batch.batch[is_ligand]
    x0_feat = torch.cat(
        [batch.frag_embeds[is_ligand], batch.x[is_ligand]], dim=-1
    )
    x0_pos = batch.pos[is_ligand]
    x0_ref = batch.ref_coords[is_ligand]
    ligand_hac = batch.hac[is_ligand]
    ligand_ring = batch.ring_count[is_ligand]
    hac_targets = encode_hac_targets(ligand_hac, profile)
    ring_targets = encode_ring_targets(ligand_ring, profile)

    is_fixed_ligand = build_fixed_ligand_mask(ligand_batch)
    is_free_ligand = ~is_fixed_ligand
    timestep = torch.randint(
        0, diffusion.num_timesteps, (batch.num_graphs,), device=device
    )
    (xt_feat, xt_pos, xt_ref), (
        true_noise_feat,
        true_noise_pos,
        true_noise_ref,
    ) = diffusion.q_sample(x0_feat, x0_pos, x0_ref, timestep, ligand_batch)

    xt_feat[is_fixed_ligand] = x0_feat[is_fixed_ligand]
    xt_pos[is_fixed_ligand] = x0_pos[is_fixed_ligand]
    xt_ref[is_fixed_ligand] = x0_ref[is_fixed_ligand]

    noisy_batch = batch.clone()
    noisy_batch.frag_embeds[is_ligand] = xt_feat[:, :config.EMBEDDING_DIM_IN]
    noisy_batch.x[is_ligand] = xt_feat[:, config.EMBEDDING_DIM_IN:]
    noisy_batch.pos[is_ligand] = xt_pos
    noisy_batch.ref_coords[is_ligand] = xt_ref
    noisy_batch = diffusion.rebuild_graph_with_dynamic_edges(noisy_batch)

    amp_enabled = device.type == 'cuda'
    with autocast(device_type=device.type, enabled=amp_enabled, dtype=amp_dtype):
        output = model(noisy_batch, timestep, return_aux=True)
        pred_noise_feat = output['pred_noise_feat']
        pred_noise_pos = output['pred_noise_pos']
        pred_noise_ref = output['pred_noise_frame']

        free_hac_targets = hac_targets[is_free_ligand]
        loss_embed_noise, loss_chem_noise = compute_feature_noise_losses(
            pred_noise_feat[is_free_ligand],
            true_noise_feat[is_free_ligand],
            free_hac_targets,
            profile['hac_class_weights'],
            beta=config.SMOOTH_L1_BETA,
        )
        loss_pos = F.smooth_l1_loss(
            pred_noise_pos[is_free_ligand],
            true_noise_pos[is_free_ligand],
            beta=config.SMOOTH_L1_BETA,
        )
        # 基础框架损失继续监督全部三个槽，包括补造的方向点。
        loss_frame = F.smooth_l1_loss(
            pred_noise_ref[is_free_ligand],
            true_noise_ref[is_free_ligand],
            beta=config.SMOOTH_L1_BETA,
        )

        x0_pred_feat, x0_pred_pos, x0_pred_ref = diffusion.predict_x0_from_noise(
            (xt_feat, xt_pos, xt_ref),
            (pred_noise_feat, pred_noise_pos, pred_noise_ref),
            timestep,
            ligand_batch,
        )
        x0_pred_feat = x0_pred_feat.clone()
        x0_pred_pos = x0_pred_pos.clone()
        x0_pred_ref = x0_pred_ref.clone()
        x0_pred_feat[is_fixed_ligand] = x0_feat[is_fixed_ligand]
        x0_pred_pos[is_fixed_ligand] = x0_pos[is_fixed_ligand]
        x0_pred_ref[is_fixed_ligand] = x0_ref[is_fixed_ligand]

        ligand_aux_weights = low_noise_aux_schedule(
            timestep[ligand_batch], diffusion.num_timesteps
        )

        loss_embed_cos = compute_embedding_cosine_loss(
            x0_pred_feat[is_free_ligand, :config.EMBEDDING_DIM_IN],
            x0_feat[is_free_ligand, :config.EMBEDDING_DIM_IN],
            ligand_aux_weights[is_free_ligand],
        )
        loss_hac = weighted_cross_entropy(
            output['hac_logits'][is_free_ligand],
            hac_targets[is_free_ligand],
            profile['hac_class_weights'],
            ligand_aux_weights[is_free_ligand],
        )
        loss_ring = weighted_cross_entropy(
            output['ring_logits'][is_free_ligand],
            ring_targets[is_free_ligand],
            profile['ring_class_weights'],
            ligand_aux_weights[is_free_ligand],
        )

        candidate_edges = output['candidate_edge_index']
        candidate_types = output['candidate_edge_type']
        edge_targets = build_dynamic_edge_targets(
            candidate_edges,
            candidate_types,
            batch.gt_dynamic_edge_index,
            batch.gt_dynamic_edge_type,
            batch.num_nodes,
        )
        fixed_global = torch.zeros(
            batch.num_nodes, dtype=torch.bool, device=device
        )
        fixed_global[is_ligand] = is_fixed_ligand
        edge_sample_mask = select_dynamic_edge_samples(
            edge_targets,
            candidate_edges,
            candidate_types,
            batch.batch,
            batch.num_nodes,
            fixed_node_mask=fixed_global,
        )
        edge_time_weights = low_noise_aux_schedule(
            timestep[batch.batch[candidate_edges[0]]],
            diffusion.num_timesteps,
        )
        loss_edge = compute_edge_topology_loss(
            output['edge_logits'],
            edge_targets,
            sample_mask=edge_sample_mask,
            sample_weights=edge_time_weights,
            class_weights=config.EDGE_CLASS_WEIGHTS,
            gamma=config.FOCAL_LOSS_GAMMA,
        )

        covalent_candidate_edges = output['covalent_candidate_edge_index']
        covalent_targets = build_covalent_edge_targets(
            covalent_candidate_edges,
            batch.gt_covalent_edge_index,
            batch.num_nodes,
        )
        if covalent_candidate_edges.shape[1]:
            covalent_sample_mask = ~(
                fixed_global[covalent_candidate_edges[0]]
                & fixed_global[covalent_candidate_edges[1]]
            )
            covalent_time_weights = low_noise_aux_schedule(
                timestep[batch.batch[covalent_candidate_edges[0]]],
                diffusion.num_timesteps,
            )
        else:
            covalent_sample_mask = torch.empty(
                (0,), dtype=torch.bool, device=device
            )
            covalent_time_weights = torch.empty(
                (0,), dtype=x0_feat.dtype, device=device
            )
        loss_covalent = compute_covalent_edge_loss(
            output['covalent_logits'],
            covalent_targets,
            pos_weight=profile['covalent_pos_weight'],
            sample_mask=covalent_sample_mask,
            sample_weights=covalent_time_weights,
        )

        pred_pos_full = batch.pos.clone()
        pred_ref_full = batch.ref_coords.clone()
        pred_pos_full[is_ligand] = x0_pred_pos
        pred_ref_full[is_ligand] = x0_pred_ref
        pred_abs_ref = pred_pos_full[:, None, :] + pred_ref_full
        true_abs_ref = batch.pos[:, None, :] + batch.ref_coords

        free_global = torch.zeros(
            batch.num_nodes, dtype=torch.bool, device=device
        )
        free_global[is_ligand] = is_free_ligand
        dynamic_node_weights = torch.zeros(
            batch.num_nodes, dtype=pred_abs_ref.dtype, device=device
        )
        gl_node_weights = torch.zeros_like(dynamic_node_weights)
        dynamic_node_weights[is_ligand] = ligand_aux_weights
        gl_node_weights[is_ligand] = ligand_aux_weights
        loss_frame_ll, loss_frame_lp, loss_frame_null, loss_frame_gl = (
            compute_frame_atom_geometry_losses(
                pred_abs_ref=pred_abs_ref,
                true_abs_ref=true_abs_ref,
                ref_atom_mask=batch.ref_atom_mask,
                candidate_edge_index=candidate_edges,
                target_classes=edge_targets,
                candidate_edge_type=candidate_types,
                is_ligand=batch.is_ligand,
                is_global=batch.is_global,
                batch=batch.batch,
                global_pos=batch.pos,
                free_ligand_mask=free_global,
                dynamic_node_weights=dynamic_node_weights,
                gl_node_weights=gl_node_weights,
                null_sample_mask=edge_sample_mask,
                beta=config.SMOOTH_L1_BETA,
            )
        )

        weighted = {
            'embed_noise': config.W_EMBED_NOISE * loss_embed_noise,
            'chem_noise': config.W_CHEM_NOISE * loss_chem_noise,
            'pos': config.W_POS_LOSS * loss_pos,
            'frame': config.W_FRAME_LOSS * loss_frame,
            'edge': config.W_EDGE_LOSS * loss_edge,
            'covalent': config.W_COVALENT_LOSS * loss_covalent,
            'frame_ll': config.W_FRAME_LL * loss_frame_ll,
            'frame_lp': config.W_FRAME_LP * loss_frame_lp,
            'frame_null': config.W_FRAME_NULL * loss_frame_null,
            'frame_gl': config.W_FRAME_GL * loss_frame_gl,
            'embed_cos': config.W_EMBED_COS * loss_embed_cos,
            'hac': config.W_HAC * loss_hac,
            'ring': config.W_RING * loss_ring,
        }
        total_loss = sum(weighted.values())

    raw_losses = {
        'embed_noise': loss_embed_noise,
        'chem_noise': loss_chem_noise,
        'pos': loss_pos,
        'frame': loss_frame,
        'edge': loss_edge,
        'covalent': loss_covalent,
        'frame_ll': loss_frame_ll,
        'frame_lp': loss_frame_lp,
        'frame_null': loss_frame_null,
        'frame_gl': loss_frame_gl,
        'embed_cos': loss_embed_cos,
        'hac': loss_hac,
        'ring': loss_ring,
    }

    with torch.no_grad():
        low_noise_threshold = int(
            (diffusion.num_timesteps - 1)
            * float(config.LOW_NOISE_AUX_START_RATIO)
        )
        low_noise_graph_mask = timestep <= low_noise_threshold
        low_noise_free_ligand = is_free_ligand & (
            low_noise_graph_mask[ligand_batch]
        )

        eval_hac_logits = output['hac_logits'][low_noise_free_ligand]
        eval_ring_logits = output['ring_logits'][low_noise_free_ligand]
        eval_hac = hac_targets[low_noise_free_ligand]
        eval_ring = ring_targets[low_noise_free_ligand]
        hac_prediction = eval_hac_logits.argmax(dim=-1)
        ring_prediction = eval_ring_logits.argmax(dim=-1)
        low_hac_mask = ligand_hac[low_noise_free_ligand] <= 3

        if low_noise_free_ligand.any():
            embedding_cosine = F.cosine_similarity(
                x0_pred_feat[
                    low_noise_free_ligand, :config.EMBEDDING_DIM_IN
                ].float(),
                x0_feat[
                    low_noise_free_ligand, :config.EMBEDDING_DIM_IN
                ].float(),
                dim=-1,
            ).mean().item()
        else:
            embedding_cosine = 0.0

        edge_eval_mask = edge_sample_mask & low_noise_graph_mask[
            batch.batch[candidate_edges[0]]
        ]
        covalent_eval_mask = covalent_sample_mask & low_noise_graph_mask[
            batch.batch[covalent_candidate_edges[0]]
        ]
        (
            covalent_precision,
            covalent_recall,
            covalent_f1,
        ) = binary_precision_recall_f1(
            output['covalent_logits'],
            covalent_targets,
            covalent_eval_mask,
        )
        metrics = {
            'hac_accuracy': (
                (hac_prediction == eval_hac).float().mean().item()
                if eval_hac.numel() else 0.0
            ),
            'hac_macro_f1': macro_f1(
                eval_hac_logits, eval_hac, int(profile['hac_num_classes'])
            ),
            'ring_accuracy': (
                (ring_prediction == eval_ring).float().mean().item()
                if eval_ring.numel() else 0.0
            ),
            'ring_macro_f1': macro_f1(
                eval_ring_logits, eval_ring, int(profile['ring_num_classes'])
            ),
            'low_hac_accuracy': (
                (hac_prediction[low_hac_mask] == eval_hac[low_hac_mask])
                .float()
                .mean()
                .item()
                if low_hac_mask.any() else 0.0
            ),
            'embedding_cosine': embedding_cosine,
            'edge_ll_recall': class_recall(
                output['edge_logits'], edge_targets, edge_eval_mask, 0
            ),
            'edge_lp_recall': class_recall(
                output['edge_logits'], edge_targets, edge_eval_mask, 1
            ),
            'edge_null_recall': class_recall(
                output['edge_logits'], edge_targets, edge_eval_mask, 2
            ),
            'covalent_precision': covalent_precision,
            'covalent_recall': covalent_recall,
            'covalent_f1': covalent_f1,
        }

    return total_loss, raw_losses, weighted, metrics


def run_epoch(
    model,
    loader,
    diffusion,
    profile,
    device,
    amp_dtype,
    epoch_num,
    optimizer=None,
    scaler=None,
    sampler=None,
    accumulation_steps=1,
):
    training = optimizer is not None
    if training:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        if sampler is not None:
            sampler.set_epoch(epoch_num)
    else:
        model.eval()

    description = (
        f"Epoch {epoch_num + 1}/{config.PRETRAIN_EPOCHS} "
        f"[{'Train' if training else 'Validation'}]"
    )
    progress = tqdm(loader, desc=description, leave=False) if is_main_process() else loader
    totals = {}
    max_raw_grad = 0.0
    num_batches = len(loader)

    grad_context = contextlib.nullcontext if training else torch.no_grad
    with grad_context():
        for batch_index, batch in enumerate(progress):
            batch = batch.to(device)
            update_step = (
                (batch_index + 1) % accumulation_steps == 0
                or (batch_index + 1) == num_batches
            )
            sync_context = (
                model.no_sync
                if training and not update_step and isinstance(model, DDP)
                else contextlib.nullcontext
            )
            with sync_context():
                loss, raw, weighted, batch_metrics = compute_batch_objective(
                    model, batch, diffusion, profile, amp_dtype
                )
                if training:
                    remainder = num_batches % accumulation_steps
                    in_final_partial_group = (
                        remainder != 0
                        and batch_index >= num_batches - remainder
                    )
                    divisor = (
                        remainder if in_final_partial_group else accumulation_steps
                    )
                    scaler.scale(loss / divisor).backward()

            if training and update_step:
                scaler.unscale_(optimizer)
                hold_epochs = config.PRETRAIN_HOLD_EPOCHS
                max_norm = (
                    config.GRAD_CLIP_WARMUP_NORM
                    if epoch_num < config.PRETRAIN_WARMUP_EPOCHS + hold_epochs
                    else config.GRAD_CLIP_STABLE_NORM
                )
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=max_norm
                )
                max_raw_grad = max(max_raw_grad, float(grad_norm))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            totals['total'] = totals.get('total', 0.0) + float(loss.detach())
            for name, value in raw.items():
                totals[f'raw/{name}'] = totals.get(f'raw/{name}', 0.0) + float(
                    value.detach()
                )
            for name, value in weighted.items():
                totals[f'weighted/{name}'] = totals.get(
                    f'weighted/{name}', 0.0
                ) + float(value.detach())
            for name, value in batch_metrics.items():
                totals[f'metric/{name}'] = totals.get(
                    f'metric/{name}', 0.0
                ) + float(value)

    averages = {name: value / max(1, num_batches) for name, value in totals.items()}
    averages = reduce_metric_dict(averages, device)
    if training:
        if dist.is_initialized():
            grad_tensor = torch.tensor(max_raw_grad, device=device)
            dist.all_reduce(grad_tensor, op=dist.ReduceOp.MAX)
            max_raw_grad = float(grad_tensor)
        averages['max_raw_grad'] = max_raw_grad
    return averages


def get_lr_multiplier(current_epoch):
    warmup = config.PRETRAIN_WARMUP_EPOCHS
    hold = config.PRETRAIN_HOLD_EPOCHS
    if current_epoch < warmup:
        return config.WARMUP_START_LR_RATIO + (
            1.0 - config.WARMUP_START_LR_RATIO
        ) * (current_epoch / max(1, warmup))
    if current_epoch < warmup + hold:
        return 1.0
    decay_epochs = config.PRETRAIN_EPOCHS - warmup - hold
    if decay_epochs <= 0:
        return 1.0
    decay_epoch = current_epoch - warmup - hold
    cosine = 0.5 * (1.0 + math.cos(math.pi * decay_epoch / decay_epochs))
    return 0.01 + 0.99 * cosine


def main():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    if 'LOCAL_RANK' not in os.environ:
        raise RuntimeError("主训练入口要求使用 torchrun/DDP 启动，并提供 LOCAL_RANK")
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=config.DIST_BACKEND)
    device = torch.device(f'cuda:{local_rank}')

    if is_main_process():
        os.makedirs(config.CHECKPOINT_DIR_PRETRAIN, exist_ok=True)
        os.makedirs(config.LOG_DIR_PRETRAIN, exist_ok=True)
        writer = SummaryWriter(config.LOG_DIR_PRETRAIN)
        writer.add_custom_scalars(build_tensorboard_custom_layout())
    else:
        writer = None

    train_loader, val_loader, train_sampler = get_train_val_dataloaders(
        distributed=True
    )
    profile_box = [None]
    if is_main_process():
        try:
            profile_box[0] = build_training_data_profile(
                train_loader.dataset, config.PYG_DATA_DIR, config.SEED
            )
        except Exception as exc:
            profile_box[0] = {
                '__profile_error__': f'{type(exc).__name__}: {exc}'
            }
    dist.broadcast_object_list(profile_box, src=0)
    profile = profile_box[0]
    if '__profile_error__' in profile:
        message = f"训练画像生成失败: {profile['__profile_error__']}"
        if is_main_process():
            print(message)
        dist.destroy_process_group()
        raise RuntimeError(message)
    if is_main_process():
        save_training_data_profile(profile)
        print(
            "训练画像: "
            f"HAC cap={profile['hac_cap']}, ring cap={profile['ring_cap']}, "
            f"LL P95={profile['max_ll_neighbors']}, "
            f"LP P95={profile['max_lp_neighbors']}, "
            f"共价正/负={profile['covalent_positive_pairs']}/"
            f"{profile['covalent_negative_pairs']}, "
            f"pos_weight={profile['covalent_pos_weight']:.3f}"
        )

    model = E3NNTransformerDiffusion(
        hac_num_classes=profile['hac_num_classes'],
        ring_num_classes=profile['ring_num_classes'],
    ).to(device)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    diffusion = DiffusionProcess(
        model=model,
        device=device,
        max_ll_neighbors=profile['max_ll_neighbors'],
        max_lp_neighbors=profile['max_lp_neighbors'],
    )
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.PRETRAIN_LR,
        weight_decay=config.PRETRAIN_WEIGHT_DECAY,
    )
    scheduler = LambdaLR(optimizer, lr_lambda=get_lr_multiplier)

    supports_bf16 = torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if supports_bf16 else torch.float16
    scaler = GradScaler('cuda', enabled=(amp_dtype == torch.float16))

    start_epoch = 0
    best_val_loss = float('inf')
    current_loss_weight_snapshot = build_loss_weight_snapshot()
    if os.path.exists(config.BEST_MODEL_PATH_PRETRAIN):
        checkpoint = torch.load(
            config.BEST_MODEL_PATH_PRETRAIN, map_location=device, weights_only=False
        )
        checkpoint_profile = checkpoint.get('training_data_profile')
        if profile != checkpoint_profile:
            if is_main_process():
                print("检查点训练画像与当前训练数据画像不一致，已明确终止。")
            dist.destroy_process_group()
            sys.exit(1)
        model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = int(checkpoint['epoch']) + 1
        loss_weight_changes = describe_loss_weight_changes(
            checkpoint.get('loss_weight_snapshot'),
            current_loss_weight_snapshot,
        )
        reset_best_val_loss = (
            config.RESET_BEST_VAL_LOSS_ON_WEIGHT_CHANGE
            and bool(loss_weight_changes)
        )
        if not reset_best_val_loss:
            best_val_loss = float(
                checkpoint.get('best_val_loss', float('inf'))
            )
        if is_main_process():
            print(f"已恢复检查点，将从 Epoch {start_epoch + 1} 继续。")
            if reset_best_val_loss:
                print(
                    "检测到损失权重配置变化，best_val_loss 已重置；"
                    "将以恢复后的首轮验证损失建立新基准。"
                )
                for change in loss_weight_changes:
                    print(f"  - {change}")
            elif loss_weight_changes:
                print(
                    "检测到损失权重配置变化，但自动重置开关已关闭；"
                    "将继续沿用检查点中的 best_val_loss。"
                )
    elif is_main_process():
        print("未发现预训练检查点，将从头训练。")

    for epoch in range(start_epoch, config.PRETRAIN_EPOCHS):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            diffusion=diffusion,
            profile=profile,
            device=device,
            amp_dtype=amp_dtype,
            epoch_num=epoch,
            optimizer=optimizer,
            scaler=scaler,
            sampler=train_sampler,
            accumulation_steps=config.GRAD_ACCUMULATION_STEPS,
        )
        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            diffusion=diffusion,
            profile=profile,
            device=device,
            amp_dtype=amp_dtype,
            epoch_num=epoch,
        )
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        if is_main_process():
            print(
                f"Epoch {epoch + 1}/{config.PRETRAIN_EPOCHS} | "
                f"LR {current_lr:.2e} | Grad {train_metrics['max_raw_grad']:.2e}"
            )
            print(
                f"  Train total={train_metrics['total']:.4f}, "
                f"embed/chem="
                f"{train_metrics['weighted/embed_noise']:.4f}/"
                f"{train_metrics['weighted/chem_noise']:.4f}, "
                f"pos={train_metrics['weighted/pos']:.4f}, "
                f"frame={train_metrics['weighted/frame']:.4f}, "
                f"edge/covalent="
                f"{train_metrics['weighted/edge']:.4f}/"
                f"{train_metrics['weighted/covalent']:.4f}, "
                f"geom(LL/LP/Null/GL)="
                f"{train_metrics['weighted/frame_ll']:.4f}/"
                f"{train_metrics['weighted/frame_lp']:.4f}/"
                f"{train_metrics['weighted/frame_null']:.4f}/"
                f"{train_metrics['weighted/frame_gl']:.4f}"
            )
            print(
                f"  Valid total={val_metrics['total']:.4f}, "
                f"Low-noise HAC acc/F1={val_metrics['metric/hac_accuracy']:.3f}/"
                f"{val_metrics['metric/hac_macro_f1']:.3f}, "
                f"Ring acc/F1={val_metrics['metric/ring_accuracy']:.3f}/"
                f"{val_metrics['metric/ring_macro_f1']:.3f}, "
                f"Cov P/R/F1="
                f"{val_metrics['metric/covalent_precision']:.3f}/"
                f"{val_metrics['metric/covalent_recall']:.3f}/"
                f"{val_metrics['metric/covalent_f1']:.3f}"
            )

            if writer:
                writer.add_scalar('Meta/learning_rate', current_lr, epoch)
                writer.add_scalar(
                    'Meta/max_raw_grad_norm',
                    train_metrics['max_raw_grad'],
                    epoch,
                )
                for split_name, metrics in (
                    ('Train', train_metrics),
                    ('Validation', val_metrics),
                ):
                    for metric_name in TENSORBOARD_LOSS_METRIC_NAMES:
                        writer.add_scalar(
                            f'{split_name}/{metric_name}',
                            metrics[metric_name],
                            epoch,
                        )

                for metric_name in TENSORBOARD_VALIDATION_QUALITY_NAMES:
                    writer.add_scalar(
                        f'Validation/low_noise_metric/{metric_name}',
                        val_metrics[f'metric/{metric_name}'],
                        epoch,
                    )

            if val_metrics['total'] < best_val_loss:
                best_val_loss = val_metrics['total']
                torch.save(
                    {
                        'epoch': epoch,
                        'model_state_dict': model.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'best_val_loss': best_val_loss,
                        'loss_weight_snapshot': current_loss_weight_snapshot,
                        'training_data_profile': profile,
                        'hac_class_mapping': profile['hac_class_mapping'],
                        'ring_class_mapping': profile['ring_class_mapping'],
                    },
                    config.BEST_MODEL_PATH_PRETRAIN,
                )
                print(f"  -> 已保存最佳模型: {best_val_loss:.4f}")

    if writer:
        writer.close()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
