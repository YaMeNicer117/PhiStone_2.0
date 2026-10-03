from __future__ import annotations

import contextlib
import copy
import math
import os
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Subset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
from tqdm import tqdm

import PhiSSE3TD_config as config
from PhiSSE3TD_activity_guidance import (
    ActivityGuidanceRuntime,
    build_activity_guidance_metadata,
    load_activity_guidance,
    score_activity,
    validate_activity_guidance_metadata,
)
from PhiSSE3TD_dataset import (
    FragmentEmbeddingLookup,
    LazyGraphDataset,
    PreloadedGraphDataset,
    PreprocessNodeFeatures,
    build_training_data_profile,
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


SURVIVAL_STRATEGIES = frozenset({"survival", "survival_modify"})
LINKER_PRIORITY_STRATEGIES = frozenset({"activity", "survival_modify"})
FINETUNE_STRATEGIES = frozenset(
    {"activity", "creative", "survival_linker"}
) | SURVIVAL_STRATEGIES

TENSORBOARD_WEIGHTED_LOSS_NAMES = (
    "embed_noise",
    "chem_noise",
    "pos",
    "frame",
    "edge",
    "covalent",
    "frame_ll",
    "frame_lp",
    "frame_null",
    "frame_gl",
    "embed_cos",
    "hac",
    "ring",
)
TENSORBOARD_LOSS_METRIC_NAMES = ("total",) + tuple(
    f"weighted/{name}" for name in TENSORBOARD_WEIGHTED_LOSS_NAMES
)
TENSORBOARD_VALIDATION_QUALITY_NAMES = (
    "hac_macro_f1",
    "ring_macro_f1",
    "embedding_cosine",
    "edge_ll_recall",
    "edge_lp_recall",
    "edge_null_recall",
    "covalent_f1",
)
TENSORBOARD_LINKER_PRIORITY_NAMES = (
    "linker_free_fraction",
    "linker_covalent_recall",
    "linker_covalent_f1",
)


@dataclass(frozen=True)
class FineTuneSettings:
    strategy: str
    data_dir: Path
    source_checkpoint_path: Path
    checkpoint_dir: Path
    best_checkpoint_path: Path
    last_checkpoint_path: Path
    training_profile_path: Path
    tensorboard_dir: Path
    embedding_table_path: Path
    embedding_metadata_path: Path

    epochs: int
    learning_rate: float
    weight_decay: float
    batch_size: int
    grad_accumulation_steps: int
    train_val_split: float
    seed: int
    use_lazy_dataset: bool
    num_workers: int
    preload_workers: int
    pin_memory: bool
    dist_backend: str

    warmup_epochs: int
    hold_epochs: int
    warmup_start_lr_ratio: float
    grad_clip_warmup_norm: float
    grad_clip_stable_norm: float
    smooth_l1_beta: float

    loss_weights: Mapping[str, float]
    low_noise_aux_start_ratio: float
    high_noise_aux_weight_ratio: float
    edge_class_weights: tuple[float, float, float]
    focal_loss_gamma: float
    covalent_pos_weight_max: float
    covalent_probability_threshold: float

    survival_keep_ratio_min: float
    survival_keep_ratio_max: float
    linker_max_hac: int
    linker_min_covalent_neighbors: int
    scalar_dropout: float = config.SCALAR_DROPOUT
    survival_max_hac: int = 7
    linker_full_prediction_probability: float = 0.2
    linker_predict_ratio_min: float = 0.1
    linker_predict_ratio_max: float = 0.5

    activity_checkpoint_path: Path | None = None
    activity_target: float = 1.0
    activity_loss_weight: float = 0.2
    activity_numeric_dim: int = 11
    activity_require_linker_topology: bool = False
    activity_covalent_probability_threshold: float = 0.2
    activity_linker_priority_max_weight: float = 1.0
    survival_linker_priority_max_weight: float = 1.0
    reset_best_val_loss_on_resume: bool = False
    save_all_best_checkpoints: bool = False
    activity_smooth_l1_beta: float = 0.25

    def validate(self) -> None:
        if self.strategy not in FINETUNE_STRATEGIES:
            raise ValueError(
                f"未知微调策略 {self.strategy!r}；允许值为 "
                f"{sorted(FINETUNE_STRATEGIES)}"
            )
        expected_loss_names = set(TENSORBOARD_WEIGHTED_LOSS_NAMES)
        actual_loss_names = set(self.loss_weights)
        if actual_loss_names != expected_loss_names:
            missing = sorted(expected_loss_names - actual_loss_names)
            extra = sorted(actual_loss_names - expected_loss_names)
            raise ValueError(
                f"loss_weights 字段不完整：missing={missing}, extra={extra}"
            )
        if any(float(value) < 0.0 for value in self.loss_weights.values()):
            raise ValueError("所有损失分量权重必须为非负数")
        if self.epochs < 1:
            raise ValueError("epochs 必须至少为 1")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("learning_rate 必须为正，weight_decay 不允许为负")
        if (
            not math.isfinite(float(self.scalar_dropout))
            or not 0.0 <= float(self.scalar_dropout) < 1.0
        ):
            raise ValueError("scalar_dropout 必须是位于 [0,1) 的有限数值")
        if self.batch_size < 1 or self.grad_accumulation_steps < 1:
            raise ValueError("batch_size 和 grad_accumulation_steps 必须为正整数")
        if not 0.0 < self.train_val_split < 1.0:
            raise ValueError("train_val_split 必须位于 (0,1)")
        if self.num_workers < 0 or self.preload_workers < 1:
            raise ValueError("num_workers 不允许为负，preload_workers 必须为正")
        if self.warmup_epochs < 0 or self.hold_epochs < 0:
            raise ValueError("warmup_epochs 和 hold_epochs 不允许为负")
        if self.warmup_epochs + self.hold_epochs > self.epochs:
            raise ValueError("warmup_epochs + hold_epochs 不得超过 epochs")
        if not 0.0 < self.warmup_start_lr_ratio <= 1.0:
            raise ValueError("warmup_start_lr_ratio 必须位于 (0,1]")
        if self.grad_clip_warmup_norm <= 0.0 or self.grad_clip_stable_norm <= 0.0:
            raise ValueError("梯度裁剪阈值必须为正数")
        if self.smooth_l1_beta <= 0.0:
            raise ValueError("smooth_l1_beta 必须为正数")
        if not 0.0 < self.low_noise_aux_start_ratio <= 1.0:
            raise ValueError("low_noise_aux_start_ratio 必须位于 (0,1]")
        if not 0.0 <= self.high_noise_aux_weight_ratio <= 1.0:
            raise ValueError("high_noise_aux_weight_ratio 必须位于 [0,1]")
        if len(self.edge_class_weights) != 3 or any(
            float(value) < 0.0 for value in self.edge_class_weights
        ):
            raise ValueError("edge_class_weights 必须包含三个非负值")
        if self.focal_loss_gamma < 0.0 or self.covalent_pos_weight_max < 1.0:
            raise ValueError("focal_loss_gamma 不允许为负，pos_weight 上限至少为 1")
        if not 0.0 <= self.covalent_probability_threshold <= 1.0:
            raise ValueError("covalent_probability_threshold 必须位于 [0,1]")
        if not (
            0.0 <= self.survival_keep_ratio_min
            <= self.survival_keep_ratio_max
            <= 1.0
        ):
            raise ValueError(
                "activity/survival 固定比例必须满足 0 <= min <= max <= 1"
            )
        if self.survival_max_hac < 1:
            raise ValueError(
                "activity/survival 候选节点 HAC 上限必须为正整数"
            )
        if self.linker_max_hac < 1 or self.linker_min_covalent_neighbors < 1:
            raise ValueError("linker HAC 上限和最小共价邻居数必须为正整数")
        if not 0.0 <= self.linker_full_prediction_probability <= 1.0:
            raise ValueError("linker 完整预测概率必须位于 [0,1]")
        if not (
            0.0
            < self.linker_predict_ratio_min
            <= self.linker_predict_ratio_max
            <= 1.0
        ):
            raise ValueError("linker 部分模式预测比例必须满足 0 < min <= max <= 1")
        if not self.dist_backend:
            raise ValueError("dist_backend 不允许为空")
        if not isinstance(self.reset_best_val_loss_on_resume, bool):
            raise TypeError("reset_best_val_loss_on_resume 必须是 bool")
        if not isinstance(self.save_all_best_checkpoints, bool):
            raise TypeError("save_all_best_checkpoints 必须是 bool")

        if self.strategy == "activity":
            if not isinstance(self.activity_checkpoint_path, Path):
                raise TypeError("activity 策略必须提供 PhiSGATv2 checkpoint Path")
            if not math.isfinite(self.activity_target):
                raise ValueError("activity_target 必须是有限数值")
            if (
                not math.isfinite(self.activity_smooth_l1_beta)
                or self.activity_smooth_l1_beta <= 0.0
            ):
                raise ValueError(
                    "activity_smooth_l1_beta 必须是大于 0 的有限数值"
                )
            if (
                not math.isfinite(self.activity_loss_weight)
                or self.activity_loss_weight < 0.0
            ):
                raise ValueError("activity_loss_weight 必须是有限非负数")
            if self.activity_numeric_dim != 11:
                raise ValueError("当前 PhiSGATv2 要求 activity_numeric_dim=11")
            if not isinstance(self.activity_require_linker_topology, bool):
                raise TypeError(
                    "activity_require_linker_topology 必须是 bool"
                )
            if not (
                0.0
                <= self.activity_covalent_probability_threshold
                <= 1.0
            ):
                raise ValueError(
                    "activity_covalent_probability_threshold 必须位于 [0,1]"
                )
            if (
                not math.isfinite(self.activity_linker_priority_max_weight)
                or self.activity_linker_priority_max_weight < 1.0
            ):
                raise ValueError(
                    "activity_linker_priority_max_weight 必须是至少为 1 的"
                    "有限数值"
                )
        elif self.activity_checkpoint_path is not None:
            raise ValueError("非 activity 策略不应配置活性指导 checkpoint")

        if self.strategy == "survival_modify" and (
            not math.isfinite(self.survival_linker_priority_max_weight)
            or self.survival_linker_priority_max_weight < 1.0
        ):
            raise ValueError(
                "survival_linker_priority_max_weight 必须是至少为 1 的"
                "有限数值"
            )

        path_fields = (
            self.data_dir,
            self.source_checkpoint_path,
            self.checkpoint_dir,
            self.best_checkpoint_path,
            self.last_checkpoint_path,
            self.training_profile_path,
            self.tensorboard_dir,
            self.embedding_table_path,
            self.embedding_metadata_path,
        )
        if any(not isinstance(path, Path) for path in path_fields):
            raise TypeError("FineTuneSettings 中的路径字段必须全部为 pathlib.Path")
        checkpoint_dir = self.checkpoint_dir.resolve()
        for output_path in (
            self.best_checkpoint_path,
            self.last_checkpoint_path,
            self.training_profile_path,
        ):
            resolved = output_path.resolve()
            if checkpoint_dir != resolved.parent:
                raise ValueError(
                    f"输出文件必须直接位于 checkpoint_dir 下: {resolved}"
                )

    def as_checkpoint_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key, value in list(payload.items()):
            if isinstance(value, Path):
                payload[key] = str(value.resolve())
            elif isinstance(value, tuple):
                payload[key] = list(value)
        payload["loss_weights"] = {
            str(key): float(value) for key, value in self.loss_weights.items()
        }
        return payload


def _loss_metric_names(settings: FineTuneSettings) -> tuple[str, ...]:
    names = ("total",) + tuple(
        f"weighted/{name}" for name in TENSORBOARD_WEIGHTED_LOSS_NAMES
    )
    if settings.strategy == "activity":
        names = names + ("raw/activity", "weighted/activity")
    return names


def _tensorboard_layout_label(metric_name: str) -> str:
    if metric_name in {"raw/activity", "weighted/activity"}:
        return metric_name.replace("/", "_")
    return metric_name.split("/", 1)[-1]


def _validation_quality_names(
    settings: FineTuneSettings,
) -> tuple[str, ...]:
    names = TENSORBOARD_VALIDATION_QUALITY_NAMES
    if settings.strategy in LINKER_PRIORITY_STRATEGIES:
        names = names + TENSORBOARD_LINKER_PRIORITY_NAMES
    return names


def build_tensorboard_custom_layout(
    settings: FineTuneSettings,
) -> dict[str, Any]:
    loss_metric_names = _loss_metric_names(settings)
    quality_names = _validation_quality_names(settings)
    layout = {
        "Train Losses": {
            _tensorboard_layout_label(metric_name): [
                "Multiline",
                [f"Train/{metric_name}"],
            ]
            for metric_name in loss_metric_names
        },
        "Validation Losses": {
            _tensorboard_layout_label(metric_name): [
                "Multiline",
                [f"Validation/{metric_name}"],
            ]
            for metric_name in loss_metric_names
        },
        "Validation Quality - Low Noise": {
            metric_name: [
                "Multiline",
                [f"Validation/low_noise_metric/{metric_name}"],
            ]
            for metric_name in quality_names
        },
    }
    if settings.strategy == "activity":
        layout["Activity Guidance"] = {
            "score_mean": [
                "Multiline",
                [
                    "Train/activity/score_mean",
                    "Validation/activity/score_mean",
                ],
            ],
            "target_gap": [
                "Multiline",
                [
                    "Train/activity/target_gap",
                    "Validation/activity/target_gap",
                ],
            ],
        }
    return layout


def is_main_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def reduce_metric_dict(
    metrics: Mapping[str, float], device: torch.device
) -> dict[str, float]:
    if not dist.is_initialized():
        return dict(metrics)
    keys = sorted(metrics)
    values = torch.tensor(
        [metrics[key] for key in keys], device=device, dtype=torch.float64
    )
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    values /= dist.get_world_size()
    return {key: float(value) for key, value in zip(keys, values.tolist())}


def _strip_module_prefix(
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    keys = list(state_dict)
    if not keys:
        raise ValueError("checkpoint 的 model_state_dict 为空")
    prefixed = [key.startswith("module.") for key in keys]
    if all(prefixed):
        return {
            key[len("module.") :]: value for key, value in state_dict.items()
        }
    if any(prefixed):
        raise ValueError("checkpoint state_dict 混用了 module. 前缀")
    return dict(state_dict)


def _load_source_checkpoint(settings: FineTuneSettings) -> dict[str, Any]:
    checkpoint_path = settings.source_checkpoint_path.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"预训练 checkpoint 不存在: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError("预训练 checkpoint 必须是字典")
    required = {"model_state_dict", "training_data_profile"}
    missing = sorted(required.difference(checkpoint))
    if missing:
        raise ValueError(f"预训练 checkpoint 缺少字段: {missing}")
    profile = checkpoint["training_data_profile"]
    if not isinstance(profile, dict):
        raise TypeError("预训练 checkpoint 的 training_data_profile 必须是字典")
    for key in (
        "hac_cap",
        "ring_cap",
        "hac_num_classes",
        "ring_num_classes",
        "hac_class_mapping",
        "ring_class_mapping",
        "fragment_embedding_table",
    ):
        if key not in profile:
            raise ValueError(f"预训练训练画像缺少字段 {key!r}")
    return checkpoint


def _unique_covalent_degree(data: Any) -> torch.Tensor:
    num_nodes = int(data.num_nodes)
    edge_index = data.gt_covalent_edge_index.long()
    degree = torch.zeros(
        num_nodes, dtype=torch.long, device=edge_index.device
    )
    if edge_index.numel() == 0:
        return degree
    low = torch.minimum(edge_index[0], edge_index[1])
    high = torch.maximum(edge_index[0], edge_index[1])
    pair_ids = torch.unique(low * num_nodes + high)
    first = torch.div(pair_ids, num_nodes, rounding_mode="floor")
    second = pair_ids.remainder(num_nodes)
    ones = torch.ones_like(first, dtype=torch.long)
    degree.index_add_(0, first, ones)
    degree.index_add_(0, second, ones)
    return degree


def build_survival_candidate_mask(
    data: Any,
    max_hac: int,
) -> torch.Tensor:
    return data.is_ligand.bool() & (data.hac.long() <= int(max_hac))


def build_linker_candidate_mask(
    data: Any,
    max_hac: int,
    min_covalent_neighbors: int,
) -> torch.Tensor:
    degree = _unique_covalent_degree(data)
    return (
        data.is_ligand.bool()
        & (data.hac.long() <= int(max_hac))
        & (degree >= int(min_covalent_neighbors))
    )


def build_activity_candidate_mask(
    data: Any,
    settings: FineTuneSettings,
) -> torch.Tensor:
    """按 activity 入口开关构造 HAC 或 HAC+linker 拓扑候选。"""
    if settings.activity_require_linker_topology:
        return build_linker_candidate_mask(
            data,
            settings.survival_max_hac,
            settings.linker_min_covalent_neighbors,
        )
    return build_survival_candidate_mask(data, settings.survival_max_hac)


def _linker_priority_max_weight(settings: FineTuneSettings) -> float:
    if settings.strategy == "activity":
        return float(settings.activity_linker_priority_max_weight)
    if settings.strategy == "survival_modify":
        return float(settings.survival_linker_priority_max_weight)
    return 1.0


def graph_filter_reason(data: Any, settings: FineTuneSettings) -> str | None:
    num_ligand = int(data.is_ligand.long().sum().item())
    if settings.strategy == "creative":
        return None
    if settings.strategy == "activity":
        candidate_mask = build_activity_candidate_mask(data, settings)
        return (
            "no_activity_candidate"
            if not bool(candidate_mask.any())
            else None
        )
    if settings.strategy in SURVIVAL_STRATEGIES:
        candidate_mask = build_survival_candidate_mask(
            data, settings.survival_max_hac
        )
        return (
            "no_survival_candidate"
            if not bool(candidate_mask.any())
            else None
        )

    candidate_mask = build_linker_candidate_mask(
        data,
        settings.linker_max_hac,
        settings.linker_min_covalent_neighbors,
    )
    num_candidates = int(candidate_mask.long().sum().item())
    if num_candidates == 0:
        return "no_linker_candidate"
    if num_candidates >= num_ligand:
        return "all_ligand_nodes_are_linker_candidates"
    return None


def build_fixed_ligand_mask(
    batch: Any,
    ligand_batch_indices: torch.Tensor,
    settings: FineTuneSettings,
) -> torch.Tensor:
    fixed = torch.zeros(
        ligand_batch_indices.shape[0],
        dtype=torch.bool,
        device=ligand_batch_indices.device,
    )
    if settings.strategy == "creative":
        return fixed

    if settings.strategy == "survival_linker":
        linker_candidates = build_linker_candidate_mask(
            batch,
            settings.linker_max_hac,
            settings.linker_min_covalent_neighbors,
        )[batch.is_ligand]
        fixed = torch.ones_like(linker_candidates)
        for graph_id in torch.unique(ligand_batch_indices):
            graph_nodes = torch.nonzero(
                ligand_batch_indices == graph_id, as_tuple=False
            ).flatten()
            candidate_nodes = graph_nodes[linker_candidates[graph_nodes]]
            num_candidates = int(candidate_nodes.numel())
            if num_candidates < 1:
                raise RuntimeError(
                    "survival_linker 过滤失效：图中没有 linker 候选节点"
                )

            predict_all = float(
                torch.rand((), device=ligand_batch_indices.device)
            ) < settings.linker_full_prediction_probability
            if predict_all:
                predicted_nodes = candidate_nodes
            else:
                min_predicted = max(
                    1,
                    min(
                        num_candidates,
                        math.ceil(
                            num_candidates * settings.linker_predict_ratio_min
                        ),
                    ),
                )
                max_predicted = max(
                    min_predicted,
                    min(
                        num_candidates,
                        math.floor(
                            num_candidates * settings.linker_predict_ratio_max
                        ),
                    ),
                )
                num_predicted = int(
                    torch.randint(
                        min_predicted,
                        max_predicted + 1,
                        (),
                        device=ligand_batch_indices.device,
                    )
                )
                permutation = torch.randperm(
                    num_candidates, device=ligand_batch_indices.device
                )
                predicted_nodes = candidate_nodes[
                    permutation[:num_predicted]
                ]
            fixed[predicted_nodes] = False
    elif (
        settings.strategy == "activity"
        or settings.strategy in SURVIVAL_STRATEGIES
    ):
        if settings.strategy == "activity":
            candidate_mask = build_activity_candidate_mask(batch, settings)
        else:
            candidate_mask = build_survival_candidate_mask(
                batch, settings.survival_max_hac
            )
        candidate_mask = candidate_mask[batch.is_ligand]
        fixed = torch.ones_like(candidate_mask)
        for graph_id in torch.unique(ligand_batch_indices):
            graph_nodes = torch.nonzero(
                ligand_batch_indices == graph_id, as_tuple=False
            ).flatten()
            candidate_nodes = graph_nodes[
                candidate_mask[graph_nodes]
            ]
            num_candidates = int(candidate_nodes.numel())
            if num_candidates < 1:
                raise RuntimeError(
                    f"{settings.strategy} 过滤失效："
                    "图中没有符合当前 HAC/拓扑条件的候选节点"
                )

            fixed[candidate_nodes] = False
            min_fixed = max(
                0,
                min(
                    num_candidates - 1,
                    math.ceil(
                        num_candidates * settings.survival_keep_ratio_min
                    ),
                ),
            )
            max_fixed = max(
                min_fixed,
                min(
                    num_candidates - 1,
                    math.floor(
                        num_candidates * settings.survival_keep_ratio_max
                    ),
                ),
            )
            num_fixed = int(
                torch.randint(
                    min_fixed,
                    max_fixed + 1,
                    (),
                    device=ligand_batch_indices.device,
                )
            )
            if num_fixed:
                permutation = torch.randperm(
                    num_candidates, device=ligand_batch_indices.device
                )
                fixed[
                    candidate_nodes[permutation[:num_fixed]]
                ] = True
    else:
        raise RuntimeError(f"未实现固定节点策略: {settings.strategy!r}")

    for graph_id in torch.unique(ligand_batch_indices):
        graph_mask = ligand_batch_indices == graph_id
        graph_fixed = fixed[graph_mask]
        if (
            settings.strategy == "activity"
            or settings.strategy in SURVIVAL_STRATEGIES
        ):
            if bool(graph_fixed.all()):
                raise RuntimeError(
                    f"{settings.strategy} 固定节点策略违反每图至少一个自由 "
                    "ligand 节点的约束"
                )
            continue
        if not bool(graph_fixed.any()) or bool(graph_fixed.all()):
            raise RuntimeError(
                f"{settings.strategy} 固定节点策略违反每图至少一个固定、"
                "一个自由 ligand 节点的约束"
            )
    return fixed


def _build_filtered_dataloaders(
    settings: FineTuneSettings,
) -> tuple[DataLoader, DataLoader, DistributedSampler, dict[str, Any]]:
    transform = PreprocessNodeFeatures(
        FragmentEmbeddingLookup(
            npz_path=settings.embedding_table_path,
            metadata_path=settings.embedding_metadata_path,
        )
    )
    if settings.use_lazy_dataset:
        full_dataset = LazyGraphDataset(
            data_dir=os.fspath(settings.data_dir), transform=transform
        )
    else:
        full_dataset = PreloadedGraphDataset(
            data_dir=os.fspath(settings.data_dir),
            transform=transform,
            max_workers=settings.preload_workers,
        )

    exclusion_counts: Counter[str] = Counter()
    if settings.strategy == "creative":
        eligible_indices = list(range(len(full_dataset)))
    else:
        eligible_indices: list[int] = []
        for index in range(len(full_dataset)):
            reason = graph_filter_reason(full_dataset[index], settings)
            if reason is None:
                eligible_indices.append(index)
            else:
                exclusion_counts[reason] += 1

    filter_stats = {
        "strategy": settings.strategy,
        "num_source_graphs": int(len(full_dataset)),
        "num_eligible_graphs": int(len(eligible_indices)),
        "num_excluded_graphs": int(len(full_dataset) - len(eligible_indices)),
        "exclusion_counts": {
            key: int(value) for key, value in sorted(exclusion_counts.items())
        },
    }
    if len(eligible_indices) < 2:
        raise RuntimeError(
            "策略过滤后不足两个有效图，无法创建非空训练集和验证集: "
            f"{filter_stats}"
        )

    eligible_dataset = Subset(full_dataset, eligible_indices)
    num_train = int(len(eligible_dataset) * settings.train_val_split)
    num_train = min(max(1, num_train), len(eligible_dataset) - 1)
    num_val = len(eligible_dataset) - num_train
    train_dataset, val_dataset = torch.utils.data.random_split(
        eligible_dataset,
        [num_train, num_val],
        generator=torch.Generator().manual_seed(settings.seed),
    )

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    train_loader = DataLoader(
        train_dataset,
        batch_size=settings.batch_size,
        shuffle=False,
        sampler=train_sampler,
        num_workers=settings.num_workers,
        pin_memory=settings.pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=settings.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=settings.num_workers,
        pin_memory=settings.pin_memory,
    )
    filter_stats["num_training_graphs"] = int(num_train)
    filter_stats["num_validation_graphs"] = int(num_val)
    return train_loader, val_loader, train_sampler, filter_stats


def _balanced_class_weights(frequencies: list[int]) -> list[float]:
    frequency_tensor = torch.as_tensor(frequencies, dtype=torch.float64)
    weights = torch.zeros_like(frequency_tensor)
    observed = frequency_tensor > 0
    if observed.any():
        weights[observed] = frequency_tensor[observed].rsqrt()
        weights[observed] /= weights[observed].mean()
        weights[observed] = weights[observed].clamp(0.5, 4.0)
    return [float(value) for value in weights.tolist()]


_APPEND_ONLY_EMBEDDING_FIELDS = frozenset(
    {
        "num_smiles",
        "table_sha256",
        "array_sha256",
        "npz_path",
        "metadata_path",
    }
)


def _validate_append_only_embedding_profile(
    previous: Mapping[str, Any],
    current: Mapping[str, Any],
    *,
    context: str,
) -> None:
    previous_stable = {
        key: value
        for key, value in previous.items()
        if key not in _APPEND_ONLY_EMBEDDING_FIELDS
    }
    current_stable = {
        key: value
        for key, value in current.items()
        if key not in _APPEND_ONLY_EMBEDDING_FIELDS
    }
    if previous_stable != current_stable:
        raise ValueError(
            f"{context}片段词嵌入表的格式、维度或其他稳定字段不一致"
        )

    previous_size = previous.get("num_smiles")
    current_size = current.get("num_smiles")
    if (
        not isinstance(previous_size, int)
        or isinstance(previous_size, bool)
        or previous_size < 0
        or not isinstance(current_size, int)
        or isinstance(current_size, bool)
        or current_size < 0
    ):
        raise ValueError(f"{context}片段词嵌入表缺少有效的 num_smiles")
    if current_size < previous_size:
        raise ValueError(
            f"{context}片段词嵌入表发生缩减: "
            f"{current_size} < {previous_size}"
        )


def _validate_resume_training_profile(
    stored_profile: Mapping[str, Any],
    current_profile: Mapping[str, Any],
) -> None:
    stored = copy.deepcopy(dict(stored_profile))
    current = copy.deepcopy(dict(current_profile))
    stored_embedding = stored.get("fragment_embedding_table")
    current_embedding = current.get("fragment_embedding_table")
    if not isinstance(stored_embedding, Mapping) or not isinstance(
        current_embedding, Mapping
    ):
        raise ValueError("微调断点训练画像缺少 fragment_embedding_table")

    _validate_append_only_embedding_profile(
        stored_embedding,
        current_embedding,
        context="微调断点与当前数据的",
    )
    stored["fragment_embedding_table"] = {
        key: value
        for key, value in stored_embedding.items()
        if key not in _APPEND_ONLY_EMBEDDING_FIELDS
    }
    current["fragment_embedding_table"] = {
        key: value
        for key, value in current_embedding.items()
        if key not in _APPEND_ONLY_EMBEDDING_FIELDS
    }
    if stored != current:
        raise ValueError("微调断点训练画像与当前 Refined 数据画像不一致")


def _locked_finetune_profile(
    raw_profile: Mapping[str, Any],
    source_profile: Mapping[str, Any],
    settings: FineTuneSettings,
    filter_stats: Mapping[str, Any],
) -> dict[str, Any]:
    profile = copy.deepcopy(dict(raw_profile))
    source_embedding = source_profile.get("fragment_embedding_table")
    current_embedding = profile.get("fragment_embedding_table")
    if not isinstance(source_embedding, Mapping) or not isinstance(
        current_embedding, Mapping
    ):
        raise ValueError("训练画像缺少 fragment_embedding_table")
    _validate_append_only_embedding_profile(
        source_embedding,
        current_embedding,
        context="Refined 数据与来源 checkpoint 的",
    )

    source_hac_cap = int(source_profile["hac_cap"])
    source_ring_cap = int(source_profile["ring_cap"])
    if int(source_profile["hac_num_classes"]) != source_hac_cap + 1:
        raise ValueError("预训练 HAC cap 与类别数不一致")
    if int(source_profile["ring_num_classes"]) != source_ring_cap + 2:
        raise ValueError("预训练 Ring cap 与类别数不一致")

    profile["dataset_derived_hac_cap"] = int(profile["hac_cap"])
    profile["dataset_derived_ring_cap"] = int(profile["ring_cap"])
    profile["hac_cap"] = source_hac_cap
    profile["hac_num_classes"] = source_hac_cap + 1
    profile["hac_class_mapping"] = copy.deepcopy(
        source_profile["hac_class_mapping"]
    )
    hac_frequencies = [0] * (source_hac_cap + 1)
    for value_text, count in profile["hac_histogram"].items():
        value = int(value_text)
        class_index = value - 1 if value <= source_hac_cap else source_hac_cap
        hac_frequencies[class_index] += int(count)
    profile["hac_frequencies"] = hac_frequencies
    profile["hac_class_weights"] = _balanced_class_weights(hac_frequencies)

    profile["ring_cap"] = source_ring_cap
    profile["ring_num_classes"] = source_ring_cap + 2
    profile["ring_class_mapping"] = copy.deepcopy(
        source_profile["ring_class_mapping"]
    )
    ring_frequencies = [0] * (source_ring_cap + 2)
    for value_text, count in profile["ring_histogram"].items():
        value = int(value_text)
        class_index = value if value <= source_ring_cap else source_ring_cap + 1
        ring_frequencies[class_index] += int(count)
    profile["ring_frequencies"] = ring_frequencies
    profile["ring_class_weights"] = _balanced_class_weights(ring_frequencies)

    positive = int(profile["covalent_positive_pairs"])
    negative = int(profile["covalent_negative_pairs"])
    if positive <= 0:
        raise ValueError("微调训练子集没有共价正样本")
    profile["covalent_pos_weight"] = min(
        float(settings.covalent_pos_weight_max),
        max(1.0, negative / positive),
    )
    profile["finetune"] = {
        "training_stage": "finetune",
        "strategy": settings.strategy,
        "source_checkpoint_path": str(
            settings.source_checkpoint_path.resolve()
        ),
        "filter": copy.deepcopy(dict(filter_stats)),
    }
    return profile


def low_noise_aux_schedule(
    timesteps: torch.Tensor,
    num_timesteps: int,
    settings: FineTuneSettings,
) -> torch.Tensor:
    normalized_t = timesteps.float() / max(1, num_timesteps - 1)
    progress = torch.clamp(
        (settings.low_noise_aux_start_ratio - normalized_t)
        / settings.low_noise_aux_start_ratio,
        min=0.0,
        max=1.0,
    )
    smooth_progress = 0.5 - 0.5 * torch.cos(math.pi * progress)
    return settings.high_noise_aux_weight_ratio + (
        1.0 - settings.high_noise_aux_weight_ratio
    ) * smooth_progress


def _normalize_selected_priority_weights(
    weights: torch.Tensor,
    sample_mask: torch.Tensor,
) -> torch.Tensor:
    if weights.shape != sample_mask.shape:
        raise ValueError("优先权重与样本掩码形状不一致")
    normalized = torch.ones_like(weights)
    if sample_mask.any():
        selected = weights[sample_mask]
        normalized[sample_mask] = selected / selected.mean().clamp_min(1.0e-8)
    return normalized


def _linker_node_priority_weights(
    linker_mask: torch.Tensor,
    free_mask: torch.Tensor,
    node_timesteps: torch.Tensor,
    num_timesteps: int,
    settings: FineTuneSettings,
) -> torch.Tensor:
    raw = torch.ones_like(node_timesteps, dtype=torch.float32)
    max_weight = _linker_priority_max_weight(settings)
    if settings.strategy in LINKER_PRIORITY_STRATEGIES and max_weight > 1.0:
        normalized_t = node_timesteps.float() / max(1, num_timesteps - 1)
        prioritized = linker_mask & free_mask
        raw[prioritized] = 1.0 + (max_weight - 1.0) * normalized_t[
            prioritized
        ]
    return _normalize_selected_priority_weights(raw, free_mask)


def _edge_priority_weights(
    edge_index: torch.Tensor,
    node_priority_weights: torch.Tensor,
    sample_mask: torch.Tensor,
) -> torch.Tensor:
    if edge_index.shape[1] == 0:
        return torch.empty(
            (0,),
            dtype=node_priority_weights.dtype,
            device=node_priority_weights.device,
        )
    raw = torch.maximum(
        node_priority_weights[edge_index[0]],
        node_priority_weights[edge_index[1]],
    )
    return _normalize_selected_priority_weights(raw, sample_mask)


def _weighted_smooth_l1_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    node_weights: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    if prediction.shape[0] == 0:
        return prediction.sum() * 0.0
    if node_weights.shape != (prediction.shape[0],):
        raise ValueError("SmoothL1 节点权重形状不一致")
    per_node = F.smooth_l1_loss(
        prediction,
        target,
        reduction="none",
        beta=beta,
    ).reshape(prediction.shape[0], -1).mean(dim=-1)
    weights = node_weights.to(per_node.dtype)
    return (per_node * weights).sum() / weights.sum().clamp_min(1.0e-8)


def encode_hac_targets(hac: torch.Tensor, profile: Mapping[str, Any]) -> torch.Tensor:
    cap = int(profile["hac_cap"])
    return torch.where(
        hac <= cap, hac - 1, torch.full_like(hac, cap)
    ).long()


def encode_ring_targets(
    ring_count: torch.Tensor, profile: Mapping[str, Any]
) -> torch.Tensor:
    cap = int(profile["ring_cap"])
    return torch.where(
        ring_count <= cap,
        ring_count,
        torch.full_like(ring_count, cap + 1),
    ).long()


def weighted_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weights: list[float],
    sample_weights: torch.Tensor,
) -> torch.Tensor:
    if logits.shape[0] == 0:
        return logits.sum() * 0.0
    weights = torch.as_tensor(
        class_weights, device=logits.device, dtype=logits.dtype
    )
    per_sample = F.cross_entropy(
        logits, targets, weight=weights, reduction="none"
    )
    valid = weights[targets] > 0
    if not valid.any():
        return logits.sum() * 0.0
    scheduled = sample_weights.to(per_sample.dtype)
    return (per_sample[valid] * scheduled[valid]).sum() / valid.sum()


def binary_precision_recall_f1(
    logits: torch.Tensor,
    targets: torch.Tensor,
    sample_mask: torch.Tensor,
    threshold: float,
) -> tuple[float, float, float]:
    if logits.numel() == 0 or not sample_mask.any():
        return 0.0, 0.0, 0.0
    predictions = torch.sigmoid(logits[sample_mask]) >= float(threshold)
    positives = targets[sample_mask].bool()
    true_positive = (predictions & positives).sum().float()
    false_positive = (predictions & ~positives).sum().float()
    false_negative = (~predictions & positives).sum().float()
    precision = true_positive / (true_positive + false_positive).clamp_min(1.0)
    recall = true_positive / (true_positive + false_negative).clamp_min(1.0)
    f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1.0e-8)
    return float(precision), float(recall), float(f1)


def macro_f1(
    logits: torch.Tensor, targets: torch.Tensor, num_classes: int
) -> float:
    if logits.shape[0] == 0:
        return 0.0
    predictions = logits.argmax(dim=-1)
    scores: list[float] = []
    for class_index in range(num_classes):
        true_positive = (
            (predictions == class_index) & (targets == class_index)
        ).sum()
        false_positive = (
            (predictions == class_index) & (targets != class_index)
        ).sum()
        false_negative = (
            (predictions != class_index) & (targets == class_index)
        ).sum()
        denominator = 2 * true_positive + false_positive + false_negative
        if denominator.item() > 0:
            scores.append(
                (2 * true_positive.float() / denominator.float()).item()
            )
    return sum(scores) / len(scores) if scores else 0.0


def class_recall(
    logits: torch.Tensor,
    targets: torch.Tensor,
    sample_mask: torch.Tensor,
    class_index: int,
) -> float:
    class_mask = sample_mask & (targets == class_index)
    if not class_mask.any():
        return 0.0
    return (
        (logits.argmax(dim=-1)[class_mask] == class_index)
        .float()
        .mean()
        .item()
    )


def compute_batch_objective(
    model: DDP,
    batch: Any,
    diffusion: DiffusionProcess,
    profile: Mapping[str, Any],
    settings: FineTuneSettings,
    amp_dtype: torch.dtype,
    activity_runtime: ActivityGuidanceRuntime | None = None,
) -> tuple[
    torch.Tensor,
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    dict[str, float],
]:
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
    if settings.strategy in LINKER_PRIORITY_STRATEGIES:
        priority_linker_global = build_linker_candidate_mask(
            batch,
            settings.survival_max_hac,
            settings.linker_min_covalent_neighbors,
        )
    else:
        priority_linker_global = torch.zeros(
            batch.num_nodes, dtype=torch.bool, device=device
        )
    priority_linker_ligand = priority_linker_global[is_ligand]

    is_fixed_ligand = build_fixed_ligand_mask(
        batch, ligand_batch, settings
    )
    is_free_ligand = ~is_fixed_ligand
    if not bool(is_free_ligand.any()):
        raise RuntimeError("当前批次没有自由 ligand 节点，无法计算去噪损失")

    timestep = torch.randint(
        0, diffusion.num_timesteps, (batch.num_graphs,), device=device
    )
    ligand_priority_weights = _linker_node_priority_weights(
        priority_linker_ligand,
        is_free_ligand,
        timestep[ligand_batch],
        diffusion.num_timesteps,
        settings,
    )
    free_priority_weights = ligand_priority_weights[is_free_ligand]
    global_node_priority_weights = torch.ones(
        batch.num_nodes, dtype=torch.float32, device=device
    )
    global_node_priority_weights[is_ligand] = ligand_priority_weights
    (xt_feat, xt_pos, xt_ref), (
        true_noise_feat,
        true_noise_pos,
        true_noise_ref,
    ) = diffusion.q_sample(x0_feat, x0_pos, x0_ref, timestep, ligand_batch)

    xt_feat[is_fixed_ligand] = x0_feat[is_fixed_ligand]
    xt_pos[is_fixed_ligand] = x0_pos[is_fixed_ligand]
    xt_ref[is_fixed_ligand] = x0_ref[is_fixed_ligand]

    noisy_batch = batch.clone()
    noisy_batch.frag_embeds[is_ligand] = xt_feat[
        :, : config.EMBEDDING_DIM_IN
    ]
    noisy_batch.x[is_ligand] = xt_feat[:, config.EMBEDDING_DIM_IN :]
    noisy_batch.pos[is_ligand] = xt_pos
    noisy_batch.ref_coords[is_ligand] = xt_ref
    noisy_batch = diffusion.rebuild_graph_with_dynamic_edges(noisy_batch)

    amp_enabled = device.type == "cuda"
    with autocast(
        device_type=device.type, enabled=amp_enabled, dtype=amp_dtype
    ):
        output = model(noisy_batch, timestep, return_aux=True)
        pred_noise_feat = output["pred_noise_feat"]
        pred_noise_pos = output["pred_noise_pos"]
        pred_noise_ref = output["pred_noise_frame"]

        free_hac_targets = hac_targets[is_free_ligand]
        loss_embed_noise, loss_chem_noise = compute_feature_noise_losses(
            pred_noise_feat[is_free_ligand],
            true_noise_feat[is_free_ligand],
            free_hac_targets,
            profile["hac_class_weights"],
            node_weights=free_priority_weights,
            beta=settings.smooth_l1_beta,
        )
        loss_pos = _weighted_smooth_l1_loss(
            pred_noise_pos[is_free_ligand],
            true_noise_pos[is_free_ligand],
            free_priority_weights,
            beta=settings.smooth_l1_beta,
        )
        loss_frame = _weighted_smooth_l1_loss(
            pred_noise_ref[is_free_ligand],
            true_noise_ref[is_free_ligand],
            free_priority_weights,
            beta=settings.smooth_l1_beta,
        )

        x0_pred_feat, x0_pred_pos, x0_pred_ref = (
            diffusion.predict_x0_from_noise(
                (xt_feat, xt_pos, xt_ref),
                (pred_noise_feat, pred_noise_pos, pred_noise_ref),
                timestep,
                ligand_batch,
            )
        )
        x0_pred_feat = x0_pred_feat.clone()
        x0_pred_pos = x0_pred_pos.clone()
        x0_pred_ref = x0_pred_ref.clone()
        x0_pred_feat[is_fixed_ligand] = x0_feat[is_fixed_ligand]
        x0_pred_pos[is_fixed_ligand] = x0_pos[is_fixed_ligand]
        x0_pred_ref[is_fixed_ligand] = x0_ref[is_fixed_ligand]

        ligand_aux_weights = low_noise_aux_schedule(
            timestep[ligand_batch], diffusion.num_timesteps, settings
        ) * ligand_priority_weights
        loss_embed_cos = compute_embedding_cosine_loss(
            x0_pred_feat[
                is_free_ligand, : config.EMBEDDING_DIM_IN
            ],
            x0_feat[is_free_ligand, : config.EMBEDDING_DIM_IN],
            ligand_aux_weights[is_free_ligand],
        )
        loss_hac = weighted_cross_entropy(
            output["hac_logits"][is_free_ligand],
            hac_targets[is_free_ligand],
            profile["hac_class_weights"],
            ligand_aux_weights[is_free_ligand],
        )
        loss_ring = weighted_cross_entropy(
            output["ring_logits"][is_free_ligand],
            ring_targets[is_free_ligand],
            profile["ring_class_weights"],
            ligand_aux_weights[is_free_ligand],
        )

        candidate_edges = output["candidate_edge_index"]
        candidate_types = output["candidate_edge_type"]
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
            settings,
        )
        edge_time_weights = edge_time_weights * _edge_priority_weights(
            candidate_edges,
            global_node_priority_weights,
            edge_sample_mask,
        )
        loss_edge = compute_edge_topology_loss(
            output["edge_logits"],
            edge_targets,
            sample_mask=edge_sample_mask,
            sample_weights=edge_time_weights,
            class_weights=settings.edge_class_weights,
            gamma=settings.focal_loss_gamma,
        )

        covalent_candidate_edges = output["covalent_candidate_edge_index"]
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
                settings,
            )
        else:
            covalent_sample_mask = torch.empty(
                (0,), dtype=torch.bool, device=device
            )
            covalent_time_weights = torch.empty(
                (0,), dtype=x0_feat.dtype, device=device
            )
        covalent_time_weights = (
            covalent_time_weights
            * _edge_priority_weights(
                covalent_candidate_edges,
                global_node_priority_weights,
                covalent_sample_mask,
            )
        )
        loss_covalent = compute_covalent_edge_loss(
            output["covalent_logits"],
            covalent_targets,
            pos_weight=profile["covalent_pos_weight"],
            sample_mask=covalent_sample_mask,
            sample_weights=covalent_time_weights,
        )

        activity_scores: torch.Tensor | None = None
        loss_activity: torch.Tensor | None = None
        if settings.strategy == "activity":
            if activity_runtime is None:
                raise RuntimeError(
                    "activity 策略缺少已加载的 PhiSGATv2 指导模型"
                )
            activity_scores, _ = score_activity(
                runtime=activity_runtime,
                x0_pred_feat=x0_pred_feat,
                ligand_batch=ligand_batch,
                candidate_edge_index=covalent_candidate_edges,
                covalent_logits=output["covalent_logits"],
                gt_covalent_edge_index=batch.gt_covalent_edge_index,
                is_ligand=is_ligand,
                is_fixed_ligand=is_fixed_ligand,
                num_graphs=int(batch.num_graphs),
                embedding_dim=config.EMBEDDING_DIM_IN,
                numeric_dim=settings.activity_numeric_dim,
                probability_threshold=(
                    settings.activity_covalent_probability_threshold
                ),
            )
            activity_scores_flat = activity_scores.squeeze(-1)
            activity_targets = torch.full_like(
                activity_scores_flat, settings.activity_target
            )
            activity_per_graph = F.smooth_l1_loss(
                activity_scores_flat,
                activity_targets,
                reduction="none",
                beta=settings.activity_smooth_l1_beta,
            )
            activity_time_weights = low_noise_aux_schedule(
                timestep, diffusion.num_timesteps, settings
            ).to(activity_per_graph.dtype)
            loss_activity = (
                activity_per_graph * activity_time_weights
            ).sum() / max(1, int(batch.num_graphs))

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
        (
            loss_frame_ll,
            loss_frame_lp,
            loss_frame_null,
            loss_frame_gl,
        ) = compute_frame_atom_geometry_losses(
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
            beta=settings.smooth_l1_beta,
        )

        raw_losses = {
            "embed_noise": loss_embed_noise,
            "chem_noise": loss_chem_noise,
            "pos": loss_pos,
            "frame": loss_frame,
            "edge": loss_edge,
            "covalent": loss_covalent,
            "frame_ll": loss_frame_ll,
            "frame_lp": loss_frame_lp,
            "frame_null": loss_frame_null,
            "frame_gl": loss_frame_gl,
            "embed_cos": loss_embed_cos,
            "hac": loss_hac,
            "ring": loss_ring,
        }
        if loss_activity is not None:
            raw_losses["activity"] = loss_activity
        weighted = {
            name: float(settings.loss_weights[name]) * loss_value
            for name, loss_value in raw_losses.items()
            if name != "activity"
        }
        if loss_activity is not None:
            weighted["activity"] = (
                float(settings.activity_loss_weight) * loss_activity
            )
        total_loss = sum(weighted.values())

    with torch.no_grad():
        low_noise_threshold = int(
            (diffusion.num_timesteps - 1)
            * settings.low_noise_aux_start_ratio
        )
        low_noise_graph_mask = timestep <= low_noise_threshold
        low_noise_free_ligand = is_free_ligand & (
            low_noise_graph_mask[ligand_batch]
        )

        eval_hac_logits = output["hac_logits"][low_noise_free_ligand]
        eval_ring_logits = output["ring_logits"][low_noise_free_ligand]
        eval_hac = hac_targets[low_noise_free_ligand]
        eval_ring = ring_targets[low_noise_free_ligand]
        hac_prediction = eval_hac_logits.argmax(dim=-1)
        ring_prediction = eval_ring_logits.argmax(dim=-1)
        low_hac_mask = ligand_hac[low_noise_free_ligand] <= 3

        if low_noise_free_ligand.any():
            embedding_cosine = F.cosine_similarity(
                x0_pred_feat[
                    low_noise_free_ligand, : config.EMBEDDING_DIM_IN
                ].float(),
                x0_feat[
                    low_noise_free_ligand, : config.EMBEDDING_DIM_IN
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
        linker_covalent_eval_mask = covalent_eval_mask & (
            priority_linker_global[covalent_candidate_edges[0]]
            | priority_linker_global[covalent_candidate_edges[1]]
        )
        (
            covalent_precision,
            covalent_recall,
            covalent_f1,
        ) = binary_precision_recall_f1(
            output["covalent_logits"],
            covalent_targets,
            covalent_eval_mask,
            settings.covalent_probability_threshold,
        )
        (
            _,
            linker_covalent_recall,
            linker_covalent_f1,
        ) = binary_precision_recall_f1(
            output["covalent_logits"],
            covalent_targets,
            linker_covalent_eval_mask,
            settings.covalent_probability_threshold,
        )
        num_priority_linkers = int(priority_linker_ligand.sum().item())
        linker_free_fraction = (
            float(
                (
                    priority_linker_ligand & is_free_ligand
                ).sum().item()
            )
            / num_priority_linkers
            if num_priority_linkers
            else 0.0
        )
        metrics = {
            "hac_accuracy": (
                (hac_prediction == eval_hac).float().mean().item()
                if eval_hac.numel()
                else 0.0
            ),
            "hac_macro_f1": macro_f1(
                eval_hac_logits,
                eval_hac,
                int(profile["hac_num_classes"]),
            ),
            "ring_accuracy": (
                (ring_prediction == eval_ring).float().mean().item()
                if eval_ring.numel()
                else 0.0
            ),
            "ring_macro_f1": macro_f1(
                eval_ring_logits,
                eval_ring,
                int(profile["ring_num_classes"]),
            ),
            "low_hac_accuracy": (
                (
                    hac_prediction[low_hac_mask]
                    == eval_hac[low_hac_mask]
                )
                .float()
                .mean()
                .item()
                if low_hac_mask.any()
                else 0.0
            ),
            "embedding_cosine": embedding_cosine,
            "edge_ll_recall": class_recall(
                output["edge_logits"], edge_targets, edge_eval_mask, 0
            ),
            "edge_lp_recall": class_recall(
                output["edge_logits"], edge_targets, edge_eval_mask, 1
            ),
            "edge_null_recall": class_recall(
                output["edge_logits"], edge_targets, edge_eval_mask, 2
            ),
            "covalent_precision": covalent_precision,
            "covalent_recall": covalent_recall,
            "covalent_f1": covalent_f1,
            "linker_free_fraction": linker_free_fraction,
            "linker_covalent_recall": linker_covalent_recall,
            "linker_covalent_f1": linker_covalent_f1,
        }
        if activity_scores is not None:
            detached_activity_scores = activity_scores.detach().float()
            metrics["activity_score_mean"] = float(
                detached_activity_scores.mean()
            )
            metrics["activity_target_gap"] = float(
                (
                    detached_activity_scores
                    - float(settings.activity_target)
                )
                .abs()
                .mean()
            )

    return total_loss, raw_losses, weighted, metrics


def run_epoch(
    model: DDP,
    loader: DataLoader,
    diffusion: DiffusionProcess,
    profile: Mapping[str, Any],
    settings: FineTuneSettings,
    device: torch.device,
    amp_dtype: torch.dtype,
    epoch_num: int,
    optimizer: optim.Optimizer | None = None,
    scaler: GradScaler | None = None,
    sampler: DistributedSampler | None = None,
    activity_runtime: ActivityGuidanceRuntime | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    if settings.strategy == "activity" and activity_runtime is None:
        raise RuntimeError("activity 策略必须提供 PhiSGATv2 指导模型")
    if activity_runtime is not None:
        activity_runtime.model.eval()
    if training:
        if scaler is None:
            raise ValueError("训练阶段必须提供 GradScaler")
        model.train()
        optimizer.zero_grad(set_to_none=True)
        if sampler is not None:
            sampler.set_epoch(epoch_num)
    else:
        model.eval()

    description = (
        f"Epoch {epoch_num + 1}/{settings.epochs} "
        f"[{'Train' if training else 'Validation'}]"
    )
    progress = (
        tqdm(loader, desc=description, leave=False)
        if is_main_process()
        else loader
    )
    totals: dict[str, float] = {}
    max_raw_grad = 0.0
    num_batches = len(loader)
    if num_batches == 0:
        raise RuntimeError("DataLoader 不包含任何 batch")

    grad_context = contextlib.nullcontext if training else torch.no_grad
    with grad_context():
        for batch_index, batch in enumerate(progress):
            batch = batch.to(device)
            update_step = (
                (batch_index + 1) % settings.grad_accumulation_steps == 0
                or (batch_index + 1) == num_batches
            )
            sync_context = (
                model.no_sync
                if training and not update_step and isinstance(model, DDP)
                else contextlib.nullcontext
            )
            with sync_context():
                loss, raw, weighted, batch_metrics = compute_batch_objective(
                    model,
                    batch,
                    diffusion,
                    profile,
                    settings,
                    amp_dtype,
                    activity_runtime,
                )
                if training:
                    remainder = num_batches % settings.grad_accumulation_steps
                    in_final_partial_group = (
                        remainder != 0
                        and batch_index >= num_batches - remainder
                    )
                    divisor = (
                        remainder
                        if in_final_partial_group
                        else settings.grad_accumulation_steps
                    )
                    scaler.scale(loss / divisor).backward()

            if training and update_step:
                scaler.unscale_(optimizer)
                max_norm = (
                    settings.grad_clip_warmup_norm
                    if epoch_num
                    < settings.warmup_epochs + settings.hold_epochs
                    else settings.grad_clip_stable_norm
                )
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=max_norm
                )
                max_raw_grad = max(max_raw_grad, float(grad_norm))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            totals["total"] = totals.get("total", 0.0) + float(
                loss.detach()
            )
            for name, value in raw.items():
                key = f"raw/{name}"
                totals[key] = totals.get(key, 0.0) + float(value.detach())
            for name, value in weighted.items():
                key = f"weighted/{name}"
                totals[key] = totals.get(key, 0.0) + float(value.detach())
            for name, value in batch_metrics.items():
                key = f"metric/{name}"
                totals[key] = totals.get(key, 0.0) + float(value)

    averages = {
        name: value / num_batches for name, value in totals.items()
    }
    averages = reduce_metric_dict(averages, device)
    if training:
        if dist.is_initialized():
            grad_tensor = torch.tensor(max_raw_grad, device=device)
            dist.all_reduce(grad_tensor, op=dist.ReduceOp.MAX)
            max_raw_grad = float(grad_tensor)
        averages["max_raw_grad"] = max_raw_grad
    return averages


def get_lr_multiplier(
    current_epoch: int, settings: FineTuneSettings
) -> float:
    if current_epoch < settings.warmup_epochs:
        return settings.warmup_start_lr_ratio + (
            1.0 - settings.warmup_start_lr_ratio
        ) * (current_epoch / max(1, settings.warmup_epochs))
    if current_epoch < settings.warmup_epochs + settings.hold_epochs:
        return 1.0
    decay_epochs = (
        settings.epochs - settings.warmup_epochs - settings.hold_epochs
    )
    if decay_epochs <= 0:
        return 1.0
    decay_epoch = current_epoch - settings.warmup_epochs - settings.hold_epochs
    cosine = 0.5 * (
        1.0 + math.cos(math.pi * decay_epoch / decay_epochs)
    )
    return 0.01 + 0.99 * cosine


def _validate_resume_checkpoint(
    checkpoint: Mapping[str, Any],
    profile: Mapping[str, Any],
    settings: FineTuneSettings,
    activity_guidance_metadata: Mapping[str, Any] | None,
) -> None:
    required = {
        "epoch",
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "scaler_state_dict",
        "best_val_loss",
        "training_data_profile",
        "hac_class_mapping",
        "ring_class_mapping",
        "training_stage",
        "finetune_strategy",
        "source_pretrain_checkpoint",
    }
    missing = sorted(required.difference(checkpoint))
    if missing:
        raise ValueError(f"微调 last_model.pt 缺少字段: {missing}")
    if checkpoint["training_stage"] != "finetune":
        raise ValueError("断点不是 diffusion finetune checkpoint")
    if checkpoint["finetune_strategy"] != settings.strategy:
        raise ValueError(
            "断点策略与当前入口不一致: "
            f"{checkpoint['finetune_strategy']!r} != {settings.strategy!r}"
        )
    stored_profile = checkpoint["training_data_profile"]
    if not isinstance(stored_profile, Mapping):
        raise TypeError("微调 last_model.pt 的 training_data_profile 必须是字典")
    _validate_resume_training_profile(stored_profile, profile)
    if Path(checkpoint["source_pretrain_checkpoint"]).resolve() != (
        settings.source_checkpoint_path.resolve()
    ):
        raise ValueError("微调断点的来源预训练 checkpoint 与当前配置不一致")
    if settings.strategy == "activity":
        stored_activity_metadata = checkpoint.get("activity_guidance")
        if not isinstance(stored_activity_metadata, Mapping):
            raise ValueError("activity 微调断点缺少 activity_guidance 元数据")
        if activity_guidance_metadata is None:
            raise RuntimeError("当前 activity 运行缺少指导模型元数据")
        validate_activity_guidance_metadata(
            stored_activity_metadata, activity_guidance_metadata
        )


def _build_checkpoint_payload(
    *,
    epoch: int,
    model: DDP,
    optimizer: optim.Optimizer,
    scheduler: LambdaLR,
    scaler: GradScaler,
    best_val_loss: float,
    profile: Mapping[str, Any],
    settings: FineTuneSettings,
    source_checkpoint: Mapping[str, Any],
    filter_stats: Mapping[str, Any],
    activity_guidance_metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    payload = {
        "epoch": int(epoch),
        "model_state_dict": model.module.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_val_loss": float(best_val_loss),
        "training_data_profile": copy.deepcopy(dict(profile)),
        "hac_class_mapping": copy.deepcopy(profile["hac_class_mapping"]),
        "ring_class_mapping": copy.deepcopy(profile["ring_class_mapping"]),
        "training_stage": "finetune",
        "finetune_strategy": settings.strategy,
        "source_pretrain_checkpoint": str(
            settings.source_checkpoint_path.resolve()
        ),
        "source_pretrain_epoch": source_checkpoint.get("epoch"),
        "filter_stats": copy.deepcopy(dict(filter_stats)),
        "finetune_settings": settings.as_checkpoint_dict(),
    }
    if activity_guidance_metadata is not None:
        payload["activity_guidance"] = copy.deepcopy(
            dict(activity_guidance_metadata)
        )
    return payload


def _build_best_checkpoint_history_path(
    best_checkpoint_path: Path,
    *,
    epoch: int,
    val_loss: float,
) -> Path:
    history_name = (
        f"{best_checkpoint_path.stem}_epoch_{epoch + 1:04d}"
        f"_val_{val_loss:.6f}{best_checkpoint_path.suffix}"
    )
    return best_checkpoint_path.with_name(history_name)


def _print_epoch_summary(
    epoch: int,
    settings: FineTuneSettings,
    current_lr: float,
    train_metrics: Mapping[str, float],
    val_metrics: Mapping[str, float],
) -> None:
    print(
        f"Epoch {epoch + 1}/{settings.epochs} | "
        f"LR {current_lr:.2e} | Grad {train_metrics['max_raw_grad']:.2e}"
    )
    print(
        f"  Train total={train_metrics['total']:.4f}, "
        "embed/chem="
        f"{train_metrics['weighted/embed_noise']:.4f}/"
        f"{train_metrics['weighted/chem_noise']:.4f}, "
        f"pos={train_metrics['weighted/pos']:.4f}, "
        f"frame={train_metrics['weighted/frame']:.4f}, "
        "edge/covalent="
        f"{train_metrics['weighted/edge']:.4f}/"
        f"{train_metrics['weighted/covalent']:.4f}, "
        "geom(LL/LP/Null/GL)="
        f"{train_metrics['weighted/frame_ll']:.4f}/"
        f"{train_metrics['weighted/frame_lp']:.4f}/"
        f"{train_metrics['weighted/frame_null']:.4f}/"
        f"{train_metrics['weighted/frame_gl']:.4f}"
    )
    print(
        f"  Valid total={val_metrics['total']:.4f}, "
        "Low-noise HAC acc/F1="
        f"{val_metrics['metric/hac_accuracy']:.3f}/"
        f"{val_metrics['metric/hac_macro_f1']:.3f}, "
        "Ring acc/F1="
        f"{val_metrics['metric/ring_accuracy']:.3f}/"
        f"{val_metrics['metric/ring_macro_f1']:.3f}, "
        "Cov P/R/F1="
        f"{val_metrics['metric/covalent_precision']:.3f}/"
        f"{val_metrics['metric/covalent_recall']:.3f}/"
        f"{val_metrics['metric/covalent_f1']:.3f}"
    )
    if settings.strategy == "activity":
        print(
            "  Activity weighted(train/valid)="
            f"{train_metrics['weighted/activity']:.4f}/"
            f"{val_metrics['weighted/activity']:.4f}, "
            "valid score/target-gap="
            f"{val_metrics['metric/activity_score_mean']:.3f}/"
            f"{val_metrics['metric/activity_target_gap']:.3f}"
        )
    if settings.strategy in LINKER_PRIORITY_STRATEGIES:
        print(
            "  Linker valid free-fraction/covalent-R/F1="
            f"{val_metrics['metric/linker_free_fraction']:.3f}/"
            f"{val_metrics['metric/linker_covalent_recall']:.3f}/"
            f"{val_metrics['metric/linker_covalent_f1']:.3f}"
        )


def _write_tensorboard(
    writer: SummaryWriter,
    epoch: int,
    current_lr: float,
    train_metrics: Mapping[str, float],
    val_metrics: Mapping[str, float],
    settings: FineTuneSettings,
) -> None:
    writer.add_scalar("Meta/learning_rate", current_lr, epoch)
    writer.add_scalar(
        "Meta/max_raw_grad_norm", train_metrics["max_raw_grad"], epoch
    )
    for split_name, metrics in (
        ("Train", train_metrics),
        ("Validation", val_metrics),
    ):
        for metric_name in _loss_metric_names(settings):
            writer.add_scalar(
                f"{split_name}/{metric_name}",
                metrics[metric_name],
                epoch,
            )
    for metric_name in _validation_quality_names(settings):
        writer.add_scalar(
            f"Validation/low_noise_metric/{metric_name}",
            val_metrics[f"metric/{metric_name}"],
            epoch,
        )
    if settings.strategy == "activity":
        for split_name, metrics in (
            ("Train", train_metrics),
            ("Validation", val_metrics),
        ):
            writer.add_scalar(
                f"{split_name}/activity/score_mean",
                metrics["metric/activity_score_mean"],
                epoch,
            )
            writer.add_scalar(
                f"{split_name}/activity/target_gap",
                metrics["metric/activity_target_gap"],
                epoch,
            )


def run_finetuning(settings: FineTuneSettings) -> None:
    settings.validate()
    os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
    )
    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError(
            "微调入口要求使用 torchrun/DDP 启动，并提供 LOCAL_RANK"
        )

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=settings.dist_backend)
    device = torch.device(f"cuda:{local_rank}")
    rank = dist.get_rank()
    torch.manual_seed(settings.seed + rank)
    torch.cuda.manual_seed_all(settings.seed + rank)

    writer: SummaryWriter | None = None
    try:
        if is_main_process():
            settings.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            settings.tensorboard_dir.mkdir(parents=True, exist_ok=True)
            writer = SummaryWriter(os.fspath(settings.tensorboard_dir))
            writer.add_custom_scalars(
                build_tensorboard_custom_layout(settings)
            )
            print(
                f"启动 {settings.strategy} 微调 | epochs={settings.epochs}, "
                f"lr={settings.learning_rate:.2e}"
            )

        source_checkpoint = _load_source_checkpoint(settings)
        source_profile = source_checkpoint["training_data_profile"]

        activity_runtime: ActivityGuidanceRuntime | None = None
        activity_guidance_metadata: dict[str, Any] | None = None
        if settings.strategy == "activity":
            if settings.activity_checkpoint_path is None:
                raise RuntimeError("activity 策略缺少 PhiSGATv2 checkpoint 路径")
            activity_runtime = load_activity_guidance(
                settings.activity_checkpoint_path, device
            )
            activity_guidance_metadata = build_activity_guidance_metadata(
                activity_runtime,
                target=settings.activity_target,
                smooth_l1_beta=settings.activity_smooth_l1_beta,
                loss_weight=settings.activity_loss_weight,
                numeric_dim=settings.activity_numeric_dim,
                probability_threshold=(
                    settings.activity_covalent_probability_threshold
                ),
                low_noise_aux_start_ratio=(
                    settings.low_noise_aux_start_ratio
                ),
                high_noise_aux_weight_ratio=(
                    settings.high_noise_aux_weight_ratio
                ),
            )
            if is_main_process():
                print(
                    "已加载冻结 PhiSGATv2 活性指导模型: "
                    f"{activity_runtime.checkpoint_path} "
                    f"(sha256={activity_runtime.checkpoint_sha256})"
                )

        train_loader, val_loader, train_sampler, filter_stats = (
            _build_filtered_dataloaders(settings)
        )
        if is_main_process():
            print(
                "策略过滤: "
                f"source={filter_stats['num_source_graphs']}, "
                f"eligible={filter_stats['num_eligible_graphs']}, "
                f"excluded={filter_stats['num_excluded_graphs']}, "
                f"reasons={filter_stats['exclusion_counts']}"
            )

        profile_box: list[Any] = [None]
        if is_main_process():
            try:
                raw_profile = build_training_data_profile(
                    train_loader.dataset,
                    os.fspath(settings.data_dir),
                    settings.seed,
                )
                profile_box[0] = _locked_finetune_profile(
                    raw_profile,
                    source_profile,
                    settings,
                    filter_stats,
                )
            except Exception as exc:
                profile_box[0] = {
                    "__profile_error__": f"{type(exc).__name__}: {exc}"
                }
        dist.broadcast_object_list(profile_box, src=0)
        profile = profile_box[0]
        if "__profile_error__" in profile:
            raise RuntimeError(
                f"微调训练画像生成失败: {profile['__profile_error__']}"
            )

        if is_main_process():
            save_training_data_profile(
                profile, os.fspath(settings.training_profile_path)
            )
            print(
                "微调训练画像: "
                f"HAC cap={profile['hac_cap']} "
                f"(data={profile['dataset_derived_hac_cap']}), "
                f"ring cap={profile['ring_cap']} "
                f"(data={profile['dataset_derived_ring_cap']}), "
                f"LL P95={profile['max_ll_neighbors']}, "
                f"LP P95={profile['max_lp_neighbors']}, "
                f"pos_weight={profile['covalent_pos_weight']:.3f}"
            )

        model = E3NNTransformerDiffusion(
            hac_num_classes=profile["hac_num_classes"],
            ring_num_classes=profile["ring_num_classes"],
            scalar_dropout=settings.scalar_dropout,
        )
        model.load_state_dict(
            _strip_module_prefix(source_checkpoint["model_state_dict"]),
            strict=True,
        )
        model = model.to(device)
        model = DDP(
            model,
            device_ids=[local_rank],
            find_unused_parameters=False,
        )
        diffusion = DiffusionProcess(
            model=model,
            device=device,
            max_ll_neighbors=profile["max_ll_neighbors"],
            max_lp_neighbors=profile["max_lp_neighbors"],
        )
        optimizer = optim.AdamW(
            model.parameters(),
            lr=settings.learning_rate,
            weight_decay=settings.weight_decay,
        )
        scheduler = LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: get_lr_multiplier(epoch, settings),
        )
        amp_dtype = (
            torch.bfloat16
            if torch.cuda.is_bf16_supported()
            else torch.float16
        )
        scaler = GradScaler("cuda", enabled=(amp_dtype == torch.float16))

        start_epoch = 0
        best_val_loss = float("inf")
        if settings.last_checkpoint_path.is_file():
            resume_checkpoint = torch.load(
                settings.last_checkpoint_path,
                map_location=device,
                weights_only=False,
            )
            if not isinstance(resume_checkpoint, Mapping):
                raise TypeError("微调 last_model.pt 必须是字典")
            _validate_resume_checkpoint(
                resume_checkpoint,
                profile,
                settings,
                activity_guidance_metadata,
            )
            model.module.load_state_dict(
                _strip_module_prefix(resume_checkpoint["model_state_dict"]),
                strict=True,
            )
            optimizer.load_state_dict(
                resume_checkpoint["optimizer_state_dict"]
            )
            scheduler.load_state_dict(
                resume_checkpoint["scheduler_state_dict"]
            )
            scaler.load_state_dict(resume_checkpoint["scaler_state_dict"])
            start_epoch = int(resume_checkpoint["epoch"]) + 1
            if settings.reset_best_val_loss_on_resume:
                best_val_loss = float("inf")
            else:
                best_val_loss = float(resume_checkpoint["best_val_loss"])
            if is_main_process():
                print(
                    f"已恢复 {settings.strategy} 微调断点，将从 "
                    f"Epoch {start_epoch + 1} 继续。"
                )
                if settings.reset_best_val_loss_on_resume:
                    print(
                        "已按配置重置 best_val_loss；下一轮验证总损失将建立"
                        "新的最佳模型比较基准。"
                    )
        elif is_main_process():
            print(
                "未发现本策略 last_model.pt；已严格加载预训练权重，"
                "优化器和调度器从头初始化。"
            )

        for epoch in range(start_epoch, settings.epochs):
            train_metrics = run_epoch(
                model=model,
                loader=train_loader,
                diffusion=diffusion,
                profile=profile,
                settings=settings,
                device=device,
                amp_dtype=amp_dtype,
                epoch_num=epoch,
                optimizer=optimizer,
                scaler=scaler,
                sampler=train_sampler,
                activity_runtime=activity_runtime,
            )
            val_metrics = run_epoch(
                model=model,
                loader=val_loader,
                diffusion=diffusion,
                profile=profile,
                settings=settings,
                device=device,
                amp_dtype=amp_dtype,
                epoch_num=epoch,
                activity_runtime=activity_runtime,
            )
            current_lr = scheduler.get_last_lr()[0]
            scheduler.step()

            if is_main_process():
                _print_epoch_summary(
                    epoch,
                    settings,
                    current_lr,
                    train_metrics,
                    val_metrics,
                )
                if writer is not None:
                    _write_tensorboard(
                        writer,
                        epoch,
                        current_lr,
                        train_metrics,
                        val_metrics,
                        settings,
                    )

                is_best = val_metrics["total"] < best_val_loss
                if is_best:
                    best_val_loss = val_metrics["total"]
                checkpoint_payload = _build_checkpoint_payload(
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    best_val_loss=best_val_loss,
                    profile=profile,
                    settings=settings,
                    source_checkpoint=source_checkpoint,
                    filter_stats=filter_stats,
                    activity_guidance_metadata=(
                        activity_guidance_metadata
                    ),
                )
                torch.save(
                    checkpoint_payload, settings.last_checkpoint_path
                )
                if is_best:
                    torch.save(
                        checkpoint_payload, settings.best_checkpoint_path
                    )
                    history_checkpoint_path: Path | None = None
                    if settings.save_all_best_checkpoints:
                        history_checkpoint_path = (
                            _build_best_checkpoint_history_path(
                                settings.best_checkpoint_path,
                                epoch=epoch,
                                val_loss=best_val_loss,
                            )
                        )
                        torch.save(
                            checkpoint_payload, history_checkpoint_path
                        )
                    save_message = (
                        "  -> 已保存最佳 "
                        f"{settings.strategy} 模型: {best_val_loss:.4f}"
                    )
                    if history_checkpoint_path is not None:
                        save_message += (
                            f" | 历史断点: {history_checkpoint_path.name}"
                        )
                    print(save_message)

        if is_main_process() and start_epoch >= settings.epochs:
            print(
                f"断点 epoch 已达到配置的 {settings.epochs} 轮，无需继续训练。"
            )
    finally:
        if writer is not None:
            writer.close()
        if dist.is_initialized():
            dist.destroy_process_group()


__all__ = [
    "FINETUNE_STRATEGIES",
    "FineTuneSettings",
    "build_activity_candidate_mask",
    "build_fixed_ligand_mask",
    "build_linker_candidate_mask",
    "build_survival_candidate_mask",
    "graph_filter_reason",
    "run_finetuning",
]
