from __future__ import annotations

import json
import math
import os
import sys
import uuid
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from PhiSSE3TD_dataset import FragmentEmbeddingLookup
from PhiSSE3TD_model import E3NNTransformerDiffusion

try:
    from tqdm.auto import tqdm as _tqdm
except ImportError:  # Keep DDIM progress usable without an extra dependency.
    _tqdm = None


ACTIVITY_MODEL_VARIANTS = frozenset(
    {"activity_modify", "activity_linker"}
)
FINETUNE_STRATEGY_BY_MODEL_VARIANT = {
    "creative": "creative",
    "survival_linker": "survival_linker",
    "survival_modify": "survival_modify",
    "activity_modify": "activity",
    "activity_linker": "activity",
}
ACTIVITY_LINKER_REQUIREMENT_BY_MODEL_VARIANT = {
    "activity_modify": False,
    "activity_linker": True,
}


@dataclass(frozen=True)
class InferenceSettings:
    """All inference-time controls are supplied by the main inference script."""

    model_variant: str
    checkpoint_path: Path
    condition_root: Path
    output_root: Path
    embedding_table_path: Path
    embedding_metadata_path: Path
    input_selection: str
    num_samples_per_condition: int
    training_num_timesteps: int
    beta_schedule: str
    ddim_steps: int
    ddim_eta: float
    ll_radius: float
    lp_radius: float
    edge_min_probability: float
    covalent_probability_threshold: float
    embedding_dim: int
    chem_dim: int
    node_type_dim: int
    edge_attr_dim: int
    embedding_safety_clamp: float
    chem_safety_clamp: float
    ref_coords_safety_clamp: float
    position_abs_safety_clamp: float
    position_clamp_buffer: float
    position_soft_clamp_ratio: float
    random_seed_min: int
    random_seed_max: int
    device: str
    continue_on_error: bool
    progress_interval: int
    manual_total_ligand_nodes: int
    creative_extra_generated_ligand_nodes: int
    taska_added_ligand_nodes: int
    taska_extra_generated_ligand_nodes: int

    @property
    def ligand_feature_dim(self) -> int:
        return self.embedding_dim + self.chem_dim

    def validate(self) -> None:
        allowed_variants = {"pretrain", *FINETUNE_STRATEGY_BY_MODEL_VARIANT}
        if self.model_variant not in allowed_variants:
            raise ValueError(
                f"MODEL_VARIANT 必须是 {sorted(allowed_variants)} 之一"
            )
        if self.input_selection not in {"first", "all"}:
            raise ValueError("INPUT_SELECTION 必须是 'first' 或 'all'")
        if self.num_samples_per_condition < 1:
            raise ValueError("NUM_SAMPLES_PER_CONDITION 必须至少为 1")
        if self.manual_total_ligand_nodes < 1:
            raise ValueError("creative 基础节点数量必须至少为 1")
        if self.creative_extra_generated_ligand_nodes < 0:
            raise ValueError("creative 额外节点数量不能为负数")
        if self.taska_added_ligand_nodes < 1:
            raise ValueError("TASKA 新增节点数量必须至少为 1")
        if self.taska_extra_generated_ligand_nodes < 0:
            raise ValueError("TASKA 额外节点数量不能为负数")
        if self.random_seed_min < 0:
            raise ValueError("RANDOM_SEED_MIN 不允许为负数")
        if self.random_seed_max < self.random_seed_min:
            raise ValueError("RANDOM_SEED_MAX 不得小于 RANDOM_SEED_MIN")
        if self.training_num_timesteps < 2:
            raise ValueError("TRAINING_NUM_TIMESTEPS 必须至少为 2")
        if self.beta_schedule != "cosine":
            raise ValueError("当前第一阶段推理只支持 cosine beta schedule")
        if not 2 <= self.ddim_steps <= self.training_num_timesteps:
            raise ValueError(
                "DDIM_STEPS 必须位于 [2, TRAINING_NUM_TIMESTEPS]"
            )
        if self.ddim_eta < 0:
            raise ValueError("DDIM_ETA 不允许为负数")
        if self.ll_radius <= 0 or self.lp_radius <= 0:
            raise ValueError("LL_RADIUS 和 LP_RADIUS 必须为正数")
        for name, value in (
            ("EDGE_MIN_PROBABILITY", self.edge_min_probability),
            (
                "COVALENT_PROBABILITY_THRESHOLD",
                self.covalent_probability_threshold,
            ),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} 必须位于 [0, 1]")
        expected_dims = (128, 14, 3, 4)
        actual_dims = (
            self.embedding_dim,
            self.chem_dim,
            self.node_type_dim,
            self.edge_attr_dim,
        )
        if actual_dims != expected_dims:
            raise ValueError(
                "当前 checkpoint schema 要求维度为 "
                f"embedding/chem/node_type/edge_attr={expected_dims}，"
                f"实际为 {actual_dims}"
            )
        for name, value in (
            ("EMBEDDING_SAFETY_CLAMP", self.embedding_safety_clamp),
            ("CHEM_SAFETY_CLAMP", self.chem_safety_clamp),
            ("REF_COORDS_SAFETY_CLAMP", self.ref_coords_safety_clamp),
            ("POSITION_ABS_SAFETY_CLAMP", self.position_abs_safety_clamp),
            ("POSITION_CLAMP_BUFFER", self.position_clamp_buffer),
            ("POSITION_SOFT_CLAMP_RATIO", self.position_soft_clamp_ratio),
        ):
            if value <= 0:
                raise ValueError(f"{name} 必须为正数")
        if self.progress_interval < 0:
            raise ValueError("PROGRESS_INTERVAL 不允许为负数")

    def as_manifest_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key, value in list(payload.items()):
            if isinstance(value, Path):
                payload[key] = str(value.resolve())
        payload["ligand_feature_dim"] = self.ligand_feature_dim
        return payload


@dataclass
class CheckpointBundle:
    model: E3NNTransformerDiffusion
    profile: dict[str, Any]
    hac_mapping: dict[str, Any]
    ring_mapping: dict[str, Any]
    checkpoint_path: Path
    epoch: int | None
    max_ll_neighbors: int
    max_lp_neighbors: int
    training_stage: str
    finetune_strategy: str | None
    source_pretrain_checkpoint: str | None
    source_pretrain_epoch: int | None
    activity_guidance: dict[str, Any] | None


@dataclass
class PreparedCondition:
    data: Data
    source_path: Path
    relative_path: Path
    sample_id: str
    mode: str
    num_fixed_ligand_nodes: int
    global_index: int


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def discover_condition_files(
    condition_root: Path, input_selection: str
) -> list[Path]:
    root = condition_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"条件数据目录不存在: {root}")
    files = sorted(
        (path.resolve() for path in root.rglob("*.pt") if path.is_file()),
        key=lambda path: path.as_posix().casefold(),
    )
    if not files:
        raise FileNotFoundError(f"条件数据目录中没有 .pt 文件: {root}")
    if input_selection == "first":
        return files[:1]
    if input_selection == "all":
        return files
    raise ValueError("input_selection 必须是 'first' 或 'all'")


def resolve_device(device_spec: str) -> torch.device:
    device = torch.device(device_spec)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"推理设备设置为 {device_spec!r}，但当前 PyTorch 未检测到 CUDA"
        )
    return device


def create_embedding_lookup(settings: InferenceSettings) -> FragmentEmbeddingLookup:
    return FragmentEmbeddingLookup(
        npz_path=settings.embedding_table_path,
        metadata_path=settings.embedding_metadata_path,
    )


def _strip_module_prefix(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    keys = list(state_dict)
    if not keys:
        raise ValueError("checkpoint 的 model_state_dict 为空")
    prefixed = [key.startswith("module.") for key in keys]
    if all(prefixed):
        return {key[len("module.") :]: value for key, value in state_dict.items()}
    if any(prefixed):
        raise ValueError("checkpoint state_dict 混用了带/不带 module. 前缀的参数名")
    return dict(state_dict)


def _validate_embedding_profile(
    checkpoint_profile: Mapping[str, Any],
    embedding_lookup: FragmentEmbeddingLookup,
) -> None:
    if not isinstance(embedding_lookup, FragmentEmbeddingLookup):
        raise TypeError("embedding_lookup 必须是 FragmentEmbeddingLookup")
    lookup_profile = embedding_lookup.profile
    stored = checkpoint_profile.get("fragment_embedding_table")
    if not isinstance(stored, Mapping):
        raise ValueError("checkpoint 训练画像缺少 fragment_embedding_table")
    if "embedding_dim" not in stored or "embedding_dim" not in lookup_profile:
        raise ValueError("词嵌入表画像缺少必需字段 'embedding_dim'")
    if stored["embedding_dim"] != lookup_profile["embedding_dim"]:
        raise ValueError(
            "当前标准片段词嵌入表与 checkpoint 训练时不一致: "
            f"embedding_dim={lookup_profile['embedding_dim']!r}, "
            f"checkpoint={stored['embedding_dim']!r}"
        )

    raw_stored_num_smiles = stored.get("num_smiles")
    if raw_stored_num_smiles is None:
        # 兼容缺少词数的旧 checkpoint；无法证明追加关系时保持原有严格行为。
        for key in ("table_sha256", "array_sha256"):
            if key not in stored or key not in lookup_profile:
                raise ValueError(f"词嵌入表画像缺少必需字段 {key!r}")
            if stored[key] != lookup_profile[key]:
                raise ValueError(
                    "旧 checkpoint 缺少 num_smiles，当前标准片段词嵌入表"
                    "必须与训练时完全一致: "
                    f"不一致字段={key!r}"
                )
        return
    if isinstance(raw_stored_num_smiles, bool) or not isinstance(
        raw_stored_num_smiles, int
    ):
        raise ValueError("checkpoint 词嵌入表画像的 num_smiles 必须是整数")
    stored_num_smiles = int(raw_stored_num_smiles)
    if stored_num_smiles < 1:
        raise ValueError("checkpoint 词嵌入表画像的 num_smiles 必须为正数")

    raw_current_num_smiles = lookup_profile.get("num_smiles")
    if isinstance(raw_current_num_smiles, bool) or not isinstance(
        raw_current_num_smiles, int
    ):
        raise ValueError("当前词嵌入表画像的 num_smiles 必须是整数")
    current_num_smiles = int(raw_current_num_smiles)
    if current_num_smiles < stored_num_smiles:
        raise ValueError(
            "当前标准片段词嵌入表少于 checkpoint 训练词表，"
            "不属于追加式扩展: "
            f"current={current_num_smiles}, checkpoint={stored_num_smiles}"
        )

    prefix_profile = embedding_lookup.profile_for_prefix(stored_num_smiles)
    for key in ("array_sha256", "table_sha256"):
        if key not in stored or key not in prefix_profile:
            raise ValueError(f"词嵌入表画像缺少必需字段 {key!r}")
        if stored[key] != prefix_profile[key]:
            raise ValueError(
                "当前标准片段词嵌入表不是 checkpoint 训练词表的"
                "追加式扩展: "
                f"前 {stored_num_smiles} 个词的 {key} 不一致"
            )

    if current_num_smiles > stored_num_smiles:
        warnings.warn(
            "当前标准片段词嵌入表包含 checkpoint 训练后追加的词汇，"
            "训练词表前缀校验通过，允许继续推理: "
            f"checkpoint={stored_num_smiles}, current={current_num_smiles}",
            RuntimeWarning,
            stacklevel=2,
        )


def _validate_class_mapping(
    name: str, mapping: Mapping[str, Any], expected_num_classes: int
) -> dict[str, Any]:
    classes = mapping.get("classes")
    overflow = mapping.get("overflow_class")
    if not isinstance(classes, Mapping) or overflow is None:
        raise ValueError(f"checkpoint 的 {name}_class_mapping 结构非法")
    exact_indices = {int(key) for key in classes}
    overflow_index = int(overflow)
    if overflow_index < 0 or overflow_index >= expected_num_classes:
        raise ValueError(f"{name} overflow class 越界: {overflow_index}")
    if exact_indices | {overflow_index} != set(range(expected_num_classes)):
        raise ValueError(f"{name} 类映射没有覆盖全部类别")
    normalized_classes = {
        str(int(key)): int(value) for key, value in classes.items()
    }
    exact_values = list(normalized_classes.values())
    if not exact_values or any(value < 0 for value in exact_values):
        raise ValueError(f"{name} 精确类别映射必须包含非负实际值")
    return {
        "classes": normalized_classes,
        "overflow_class": overflow_index,
        "overflow_min_value": max(exact_values) + 1,
    }


def _validate_checkpoint_variant(
    model_variant: str, payload: Mapping[str, Any]
) -> tuple[str, str | None]:
    raw_training_stage = payload.get("training_stage")
    raw_finetune_strategy = payload.get("finetune_strategy")
    if model_variant == "pretrain":
        if raw_training_stage not in (None, "pretrain") or (
            raw_finetune_strategy is not None
        ):
            raise ValueError(
                "MODEL_VARIANT='pretrain'，但所选 checkpoint 不是预训练模型"
            )
        training_stage = (
            str(raw_training_stage)
            if raw_training_stage is not None
            else "pretrain"
        )
        return training_stage, None

    if raw_training_stage != "finetune":
        raise ValueError(
            f"MODEL_VARIANT={model_variant!r} 要求 "
            "training_stage='finetune' 的 checkpoint"
        )
    expected_finetune_strategy = FINETUNE_STRATEGY_BY_MODEL_VARIANT.get(
        model_variant
    )
    if expected_finetune_strategy is None:
        raise ValueError(f"未知 MODEL_VARIANT={model_variant!r}")
    if raw_finetune_strategy != expected_finetune_strategy:
        raise ValueError(
            "所选微调 checkpoint 策略与 MODEL_VARIANT 不一致: "
            f"checkpoint={raw_finetune_strategy!r}, "
            f"requested={model_variant!r}, "
            f"expected_strategy={expected_finetune_strategy!r}"
        )

    if model_variant in ACTIVITY_MODEL_VARIANTS:
        raw_finetune_settings = payload.get("finetune_settings")
        if not isinstance(raw_finetune_settings, Mapping):
            raise ValueError(
                "activity 微调 checkpoint 缺少 finetune_settings 元数据"
            )
        expected_linker_requirement = (
            ACTIVITY_LINKER_REQUIREMENT_BY_MODEL_VARIANT[model_variant]
        )
        raw_linker_requirement = raw_finetune_settings.get(
            "activity_require_linker_topology"
        )
        if raw_linker_requirement is not expected_linker_requirement:
            raise ValueError(
                "activity 微调 checkpoint 与 MODEL_VARIANT 的 linker "
                "拓扑要求不一致: "
                f"checkpoint={raw_linker_requirement!r}, "
                f"requested={model_variant!r}, "
                f"expected={expected_linker_requirement!r}"
            )
    return "finetune", str(raw_finetune_strategy)


def _validate_activity_guidance_manifest(
    raw_metadata: Any,
) -> dict[str, Any]:
    if not isinstance(raw_metadata, Mapping):
        raise ValueError(
            "activity 微调 checkpoint 缺少 activity_guidance 元数据"
        )
    required = {
        "checkpoint_path",
        "checkpoint_sha256",
        "checkpoint_epoch",
        "model_config",
        "target",
        "loss_weight",
        "numeric_dim",
        "probability_threshold",
        "low_noise_aux_start_ratio",
        "high_noise_aux_weight_ratio",
        "edge_policy",
        "covalent_gradient_from_activity",
    }
    missing = sorted(required.difference(raw_metadata))
    if missing:
        raise ValueError(
            f"activity_guidance 元数据缺少字段: {missing}"
        )

    metadata = dict(raw_metadata)
    checkpoint_path = metadata["checkpoint_path"]
    checkpoint_sha256 = metadata["checkpoint_sha256"]
    model_config = metadata["model_config"]
    if not isinstance(checkpoint_path, str) or not checkpoint_path:
        raise ValueError("activity_guidance.checkpoint_path 非法")
    if (
        not isinstance(checkpoint_sha256, str)
        or len(checkpoint_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in checkpoint_sha256
        )
    ):
        raise ValueError("activity_guidance.checkpoint_sha256 非法")
    if not isinstance(model_config, Mapping):
        raise ValueError("activity_guidance.model_config 非法")

    try:
        checkpoint_epoch = int(metadata["checkpoint_epoch"])
        numeric_dim = int(metadata["numeric_dim"])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "activity_guidance 的 epoch 或 numeric_dim 非法"
        ) from exc
    if checkpoint_epoch < 0:
        raise ValueError("activity_guidance.checkpoint_epoch 不得为负")
    if numeric_dim != 11:
        raise ValueError("activity_guidance.numeric_dim 必须为 11")
    if int(model_config.get("input_dim_numeric", -1)) != numeric_dim:
        raise ValueError("activity_guidance 的 GAT 数值输入维度不一致")
    if int(model_config.get("embedding_dim", -1)) != 128:
        raise ValueError("activity_guidance 的 GAT 片段嵌入维度必须为 128")

    numeric_values: dict[str, float] = {}
    for name in ("target", "loss_weight"):
        value = float(metadata[name])
        if not math.isfinite(value):
            raise ValueError(f"activity_guidance.{name} 必须为有限数")
        numeric_values[name] = value
    if float(metadata["loss_weight"]) < 0.0:
        raise ValueError("activity_guidance.loss_weight 不得为负")
    for name in (
        "probability_threshold",
        "high_noise_aux_weight_ratio",
    ):
        value = float(metadata[name])
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"activity_guidance.{name} 必须位于 [0,1]")
        numeric_values[name] = value
    low_noise_ratio = float(metadata["low_noise_aux_start_ratio"])
    if not math.isfinite(low_noise_ratio) or not 0.0 < low_noise_ratio <= 1.0:
        raise ValueError(
            "activity_guidance.low_noise_aux_start_ratio 必须位于 (0,1]"
        )
    if metadata["edge_policy"] != (
        "gt_fixed_fixed_plus_predicted_non_fixed_bidirectional"
    ):
        raise ValueError("activity_guidance.edge_policy 与当前策略不兼容")
    if metadata["covalent_gradient_from_activity"] is not False:
        raise ValueError(
            "activity_guidance 必须声明活性梯度不进入 covalent logits"
        )
    metadata.update(numeric_values)
    metadata["checkpoint_epoch"] = checkpoint_epoch
    metadata["numeric_dim"] = numeric_dim
    metadata["low_noise_aux_start_ratio"] = low_noise_ratio
    metadata["model_config"] = dict(model_config)
    return metadata


def load_checkpoint_bundle(
    settings: InferenceSettings,
    embedding_lookup: FragmentEmbeddingLookup,
    device: torch.device,
) -> CheckpointBundle:
    checkpoint_path = settings.checkpoint_path.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"主扩散 checkpoint 不存在: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise TypeError("第一阶段推理要求 checkpoint 为包含训练画像的字典")

    training_stage, finetune_strategy = _validate_checkpoint_variant(
        settings.model_variant, payload
    )
    activity_guidance = None
    if settings.model_variant in ACTIVITY_MODEL_VARIANTS:
        activity_guidance = _validate_activity_guidance_manifest(
            payload.get("activity_guidance")
        )
    raw_source_pretrain = payload.get("source_pretrain_checkpoint")
    source_pretrain_checkpoint = (
        str(raw_source_pretrain)
        if isinstance(raw_source_pretrain, (str, os.PathLike))
        and os.fspath(raw_source_pretrain)
        else None
    )
    raw_source_pretrain_epoch = payload.get("source_pretrain_epoch")
    source_pretrain_epoch = (
        int(raw_source_pretrain_epoch)
        if raw_source_pretrain_epoch is not None
        else None
    )
    if settings.model_variant in ACTIVITY_MODEL_VARIANTS and (
        source_pretrain_checkpoint is None
    ):
        raise ValueError(
            "activity 微调 checkpoint 缺少 source_pretrain_checkpoint"
        )

    state_dict = payload.get("model_state_dict")
    profile = payload.get("training_data_profile")
    if not isinstance(state_dict, Mapping):
        raise ValueError("checkpoint 缺少 model_state_dict")
    if not isinstance(profile, Mapping):
        raise ValueError("checkpoint 缺少 training_data_profile")
    profile = dict(profile)
    _validate_embedding_profile(profile, embedding_lookup)

    required_profile_keys = (
        "hac_num_classes",
        "ring_num_classes",
        "hac_cap",
        "ring_cap",
        "max_ll_neighbors",
        "max_lp_neighbors",
    )
    missing = [key for key in required_profile_keys if key not in profile]
    if missing:
        raise ValueError(f"checkpoint 训练画像缺少字段: {missing}")

    hac_num_classes = int(profile["hac_num_classes"])
    ring_num_classes = int(profile["ring_num_classes"])
    raw_hac_mapping = payload.get(
        "hac_class_mapping", profile.get("hac_class_mapping")
    )
    raw_ring_mapping = payload.get(
        "ring_class_mapping", profile.get("ring_class_mapping")
    )
    if not isinstance(raw_hac_mapping, Mapping) or not isinstance(
        raw_ring_mapping, Mapping
    ):
        raise ValueError("checkpoint 缺少 HAC/Ring 类映射")
    hac_mapping = _validate_class_mapping(
        "hac", raw_hac_mapping, hac_num_classes
    )
    ring_mapping = _validate_class_mapping(
        "ring", raw_ring_mapping, ring_num_classes
    )

    max_ll_neighbors = int(profile["max_ll_neighbors"])
    max_lp_neighbors = int(profile["max_lp_neighbors"])
    if max_ll_neighbors < 1 or max_lp_neighbors < 1:
        raise ValueError("checkpoint 的 LL/LP 最大邻居数必须为正数")

    model = E3NNTransformerDiffusion(
        hac_num_classes=hac_num_classes,
        ring_num_classes=ring_num_classes,
    )
    model.load_state_dict(_strip_module_prefix(state_dict), strict=True)
    model.to(device)
    model.eval()

    epoch = payload.get("epoch")
    epoch = int(epoch) if epoch is not None else None
    return CheckpointBundle(
        model=model,
        profile=profile,
        hac_mapping=hac_mapping,
        ring_mapping=ring_mapping,
        checkpoint_path=checkpoint_path,
        epoch=epoch,
        max_ll_neighbors=max_ll_neighbors,
        max_lp_neighbors=max_lp_neighbors,
        training_stage=training_stage,
        finetune_strategy=finetune_strategy,
        source_pretrain_checkpoint=source_pretrain_checkpoint,
        source_pretrain_epoch=source_pretrain_epoch,
        activity_guidance=activity_guidance,
    )


def _require_tensor(data: Any, name: str) -> torch.Tensor:
    if not hasattr(data, name):
        raise ValueError(f"条件 .pt 缺少字段 {name!r}")
    value = getattr(data, name)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"条件字段 {name!r} 必须是 Tensor")
    return value


def _deduplicate_edges(
    edge_index: torch.Tensor, edge_attr: torch.Tensor, num_nodes: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if edge_index.numel() == 0:
        return edge_index, edge_attr
    edge_ids = edge_index[0] * num_nodes + edge_index[1]
    order = torch.argsort(edge_ids)
    sorted_ids = edge_ids[order]
    keep = torch.ones_like(sorted_ids, dtype=torch.bool)
    keep[1:] = sorted_ids[1:] != sorted_ids[:-1]
    selected = order[keep]
    return edge_index[:, selected], edge_attr[selected]


def prepare_condition(
    source_path: Path,
    condition_root: Path,
    lookup: FragmentEmbeddingLookup,
    settings: InferenceSettings,
    device: torch.device,
) -> PreparedCondition:
    source_path = source_path.resolve()
    raw = torch.load(source_path, map_location="cpu", weights_only=False)
    if not hasattr(raw, "x"):
        raise TypeError(f"条件文件不是有效的 PyG Data: {source_path}")
    raw = lookup.resolve(raw)

    x = _require_tensor(raw, "x")
    frag_embeds = _require_tensor(raw, "frag_embeds")
    node_type = _require_tensor(raw, "node_type")
    pos = _require_tensor(raw, "pos")
    ref_coords_relative = _require_tensor(raw, "ref_coords")
    edge_index = _require_tensor(raw, "edge_index")
    edge_attr = _require_tensor(raw, "edge_attr")
    hac = _require_tensor(raw, "hac")
    ring_count = _require_tensor(raw, "ring_count")

    num_nodes = int(x.shape[0])
    expected_shapes = {
        "x": (num_nodes, settings.chem_dim),
        "frag_embeds": (num_nodes, settings.embedding_dim),
        "node_type": (num_nodes, settings.node_type_dim),
        "pos": (num_nodes, 3),
        "ref_coords": (num_nodes, 3, 3),
        "hac": (num_nodes,),
        "ring_count": (num_nodes,),
    }
    actual_values = {
        "x": x,
        "frag_embeds": frag_embeds,
        "node_type": node_type,
        "pos": pos,
        "ref_coords": ref_coords_relative,
        "hac": hac,
        "ring_count": ring_count,
    }
    for name, expected in expected_shapes.items():
        if tuple(actual_values[name].shape) != expected:
            raise ValueError(
                f"条件字段 {name} 形状应为 {expected}，"
                f"实际为 {tuple(actual_values[name].shape)}"
            )
    if (
        edge_index.dtype != torch.long
        or edge_index.ndim != 2
        or edge_index.shape[0] != 2
    ):
        raise ValueError("条件 edge_index 必须是 int64 [2,E]")
    if tuple(edge_attr.shape) != (edge_index.shape[1], settings.edge_attr_dim):
        raise ValueError(
            f"条件 edge_attr 应为 ({edge_index.shape[1]},"
            f"{settings.edge_attr_dim})"
        )
    if edge_index.numel() and (
        bool((edge_index < 0).any()) or bool((edge_index >= num_nodes).any())
    ):
        raise ValueError("条件 edge_index 含越界节点")

    ref_coords_mode = getattr(raw, "ref_coords_mode", None)
    if ref_coords_mode != "relative_to_node_pos":
        raise ValueError(
            "条件 .pt 必须使用 ref_coords_mode='relative_to_node_pos'；请重新运行 "
            "PhiSSeparator_work_data_process.ipynb 生成条件数据"
        )

    floating = (x, frag_embeds, node_type, pos, ref_coords_relative, edge_attr)
    if any(tensor.dtype != torch.float32 for tensor in floating):
        raise TypeError("条件浮点字段必须全部为 float32")
    if any(not bool(torch.isfinite(tensor).all()) for tensor in floating):
        raise ValueError("条件浮点字段包含 NaN/Inf")
    integer_dtypes = {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }
    if hac.dtype not in integer_dtypes or ring_count.dtype not in integer_dtypes:
        raise TypeError("条件 hac 和 ring_count 必须是整数 Tensor")
    if bool((hac < 0).any()) or bool((ring_count < 0).any()):
        raise ValueError("条件 hac 和 ring_count 不允许负值")
    if not bool(((node_type == 0) | (node_type == 1)).all()):
        raise ValueError("条件 node_type 必须是严格 0/1 one-hot")
    if not bool((node_type.sum(dim=-1) == 1).all()):
        raise ValueError("条件 node_type 每行必须且只能有一个类别")
    if edge_attr.numel():
        if not bool(((edge_attr == 0) | (edge_attr == 1)).all()):
            raise ValueError("条件 edge_attr 必须是严格 0/1 one-hot")
        if not bool((edge_attr.sum(dim=-1) == 1).all()):
            raise ValueError("条件 edge_attr 每行必须且只能有一个类别")

    is_ligand = node_type[:, 0].bool()
    is_protein = node_type[:, 1].bool()
    is_global = node_type[:, 2].bool()
    if not bool(is_protein.any()):
        raise ValueError("条件图必须至少包含一个 P 节点")
    global_indices = torch.nonzero(is_global, as_tuple=False).flatten()
    if global_indices.numel() != 1:
        raise ValueError(
            f"条件图必须恰好包含一个 Global 节点，实际为 {global_indices.numel()}"
        )
    if bool((hac[is_ligand] < 1).any()):
        raise ValueError("固定 L 节点的 HAC 必须至少为 1")
    if not bool((ref_coords_relative[is_global] == 0).all()):
        raise ValueError("Global 节点的相对 ref_coords 必须全零")

    # Training removes raw LL/LP supervision edges. Inference likewise keeps
    # only PP here; GL is rebuilt for every fixed/generated ligand node.
    pp_mask = edge_attr[:, 2].bool()
    pp_edge_index = edge_index[:, pp_mask].clone()
    pp_edge_attr = edge_attr[pp_mask].clone()
    pp_edge_index, pp_edge_attr = _deduplicate_edges(
        pp_edge_index, pp_edge_attr, num_nodes
    )

    data = Data(
        x=x.clone(),
        frag_embeds=frag_embeds.clone(),
        node_type=node_type.clone(),
        pos=pos.clone(),
        ref_coords=ref_coords_relative.clone(),
        hac=hac.long().clone(),
        ring_count=ring_count.long().clone(),
        edge_index=pp_edge_index,
        edge_attr=pp_edge_attr,
        is_ligand=is_ligand.clone(),
        is_protein=is_protein.clone(),
        is_global=is_global.clone(),
        ref_coords_mode="relative_to_node_pos",
        batch=torch.zeros(num_nodes, dtype=torch.long),
        num_nodes=num_nodes,
    ).to(device)

    try:
        relative_path = source_path.relative_to(condition_root.resolve())
    except ValueError:
        relative_path = Path(source_path.name)
    sample_id = str(getattr(raw, "sample_id", source_path.stem))
    num_fixed = int(is_ligand.sum().item())
    return PreparedCondition(
        data=data,
        source_path=source_path,
        relative_path=relative_path,
        sample_id=sample_id,
        mode="survival" if num_fixed else "creative",
        num_fixed_ligand_nodes=num_fixed,
        global_index=int(global_indices.item()),
    )


def _empty_edges(
    device: torch.device, edge_attr_dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.empty((2, 0), dtype=torch.long, device=device),
        torch.empty((0, edge_attr_dim), dtype=torch.float32, device=device),
    )


def _build_gl_edges(
    ligand_indices: torch.Tensor,
    global_index: int,
    edge_attr_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = ligand_indices.device
    if ligand_indices.numel() == 0:
        return _empty_edges(device, edge_attr_dim)
    global_nodes = torch.full_like(ligand_indices, global_index)
    edge_index = torch.cat(
        [
            torch.stack([global_nodes, ligand_indices], dim=0),
            torch.stack([ligand_indices, global_nodes], dim=0),
        ],
        dim=1,
    )
    edge_attr = torch.zeros(
        (edge_index.shape[1], edge_attr_dim),
        dtype=torch.float32,
        device=device,
    )
    edge_attr[:, 3] = 1.0
    return edge_index, edge_attr


def _build_dynamic_edges(
    pos: torch.Tensor,
    is_ligand: torch.Tensor,
    is_protein: torch.Tensor,
    batch: torch.Tensor,
    settings: InferenceSettings,
    max_ll_neighbors: int,
    max_lp_neighbors: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = pos.device
    ll_chunks: list[torch.Tensor] = []
    lp_chunks: list[torch.Tensor] = []
    for graph_id in torch.unique(batch):
        ligand_indices = torch.nonzero(
            is_ligand & (batch == graph_id), as_tuple=False
        ).flatten()
        protein_indices = torch.nonzero(
            is_protein & (batch == graph_id), as_tuple=False
        ).flatten()

        if ligand_indices.numel() > 1:
            distances = torch.cdist(
                pos[ligand_indices].detach().float(),
                pos[ligand_indices].detach().float(),
            )
            distances.fill_diagonal_(float("inf"))
            for local_idx in range(ligand_indices.numel()):
                row = distances[local_idx]
                candidates = torch.nonzero(
                    row < settings.ll_radius, as_tuple=False
                ).flatten()
                if candidates.numel() > max_ll_neighbors:
                    nearest = torch.topk(
                        row[candidates], max_ll_neighbors, largest=False
                    ).indices
                    candidates = candidates[nearest]
                if candidates.numel():
                    src = ligand_indices[local_idx].expand(candidates.numel())
                    dst = ligand_indices[candidates]
                    ll_chunks.append(torch.stack([src, dst], dim=0))

        if ligand_indices.numel() and protein_indices.numel():
            distances = torch.cdist(
                pos[ligand_indices].detach().float(),
                pos[protein_indices].detach().float(),
            )
            for local_idx in range(ligand_indices.numel()):
                row = distances[local_idx]
                candidates = torch.nonzero(
                    row < settings.lp_radius, as_tuple=False
                ).flatten()
                if candidates.numel() == 0:
                    candidates = torch.argmin(row).reshape(1)
                elif candidates.numel() > max_lp_neighbors:
                    nearest = torch.topk(
                        row[candidates], max_lp_neighbors, largest=False
                    ).indices
                    candidates = candidates[nearest]
                ligand_nodes = ligand_indices[local_idx].expand(
                    candidates.numel()
                )
                protein_nodes = protein_indices[candidates]
                lp_chunks.append(
                    torch.cat(
                        [
                            torch.stack([ligand_nodes, protein_nodes], dim=0),
                            torch.stack([protein_nodes, ligand_nodes], dim=0),
                        ],
                        dim=1,
                    )
                )

    empty_index, empty_attr = _empty_edges(device, settings.edge_attr_dim)
    ll_index = torch.cat(ll_chunks, dim=1) if ll_chunks else empty_index
    ll_attr = (
        torch.zeros(
            (ll_index.shape[1], settings.edge_attr_dim),
            dtype=torch.float32,
            device=device,
        )
        if ll_index.numel()
        else empty_attr
    )
    if ll_attr.numel():
        ll_attr[:, 0] = 1.0

    lp_index = (
        torch.cat(lp_chunks, dim=1) if lp_chunks else empty_index.clone()
    )
    lp_attr = (
        torch.zeros(
            (lp_index.shape[1], settings.edge_attr_dim),
            dtype=torch.float32,
            device=device,
        )
        if lp_index.numel()
        else empty_attr.clone()
    )
    if lp_attr.numel():
        lp_attr[:, 1] = 1.0
    return (
        torch.cat([ll_index, lp_index], dim=1),
        torch.cat([ll_attr, lp_attr], dim=0),
    )


def build_sampling_graph(
    condition: PreparedCondition,
    free_feat: torch.Tensor,
    free_pos: torch.Tensor,
    free_ref_coords: torch.Tensor,
    settings: InferenceSettings,
    max_ll_neighbors: int,
    max_lp_neighbors: int,
) -> tuple[Data, torch.Tensor, torch.Tensor]:
    num_free = int(free_pos.shape[0])
    expected_shapes = (
        (num_free, settings.ligand_feature_dim),
        (num_free, 3),
        (num_free, 3, 3),
    )
    actual_shapes = (
        tuple(free_feat.shape),
        tuple(free_pos.shape),
        tuple(free_ref_coords.shape),
    )
    if actual_shapes != expected_shapes:
        raise ValueError(
            f"自由 L 状态形状非法: {actual_shapes}，应为 {expected_shapes}"
        )

    base = condition.data
    device = base.pos.device
    num_condition_nodes = int(base.num_nodes)
    generated_node_type = torch.zeros(
        (num_free, settings.node_type_dim),
        dtype=torch.float32,
        device=device,
    )
    generated_node_type[:, 0] = 1.0

    frag_embeds = torch.cat(
        [base.frag_embeds, free_feat[:, : settings.embedding_dim]], dim=0
    )
    x = torch.cat(
        [base.x, free_feat[:, settings.embedding_dim :]], dim=0
    )
    node_type = torch.cat([base.node_type, generated_node_type], dim=0)
    pos = torch.cat([base.pos, free_pos], dim=0)
    ref_coords = torch.cat([base.ref_coords, free_ref_coords], dim=0)
    is_ligand = node_type[:, 0].bool()
    is_protein = node_type[:, 1].bool()
    is_global = node_type[:, 2].bool()
    num_nodes = int(x.shape[0])
    batch = torch.zeros(num_nodes, dtype=torch.long, device=device)

    ligand_indices = torch.nonzero(is_ligand, as_tuple=False).flatten()
    gl_index, gl_attr = _build_gl_edges(
        ligand_indices, condition.global_index, settings.edge_attr_dim
    )
    dynamic_index, dynamic_attr = _build_dynamic_edges(
        pos,
        is_ligand,
        is_protein,
        batch,
        settings,
        max_ll_neighbors,
        max_lp_neighbors,
    )
    edge_index = torch.cat(
        [base.edge_index, gl_index, dynamic_index], dim=1
    )
    edge_attr = torch.cat([base.edge_attr, gl_attr, dynamic_attr], dim=0)
    edge_index, edge_attr = _deduplicate_edges(
        edge_index, edge_attr, num_nodes
    )

    graph = Data(
        x=x,
        frag_embeds=frag_embeds,
        node_type=node_type,
        pos=pos,
        ref_coords=ref_coords,
        edge_index=edge_index,
        edge_attr=edge_attr,
        is_ligand=is_ligand,
        is_protein=is_protein,
        is_global=is_global,
        batch=batch,
        num_nodes=num_nodes,
    )
    generated_ligand_mask = ligand_indices >= num_condition_nodes
    if int(generated_ligand_mask.sum().item()) != num_free:
        raise RuntimeError("生成 L 节点到模型 L 输出的索引映射失败")
    return graph, ligand_indices, generated_ligand_mask


def _cosine_alphas_cumprod(
    num_timesteps: int, device: torch.device, s: float = 0.008
) -> torch.Tensor:
    steps = torch.arange(
        num_timesteps + 1, device=device, dtype=torch.float32
    )
    f_t = torch.cos(
        ((steps / num_timesteps) + s) / (1 + s) * math.pi * 0.5
    ) ** 2
    raw_cumprod = f_t / f_t[0]
    raw_prev = F.pad(raw_cumprod[:-1], (1, 0), value=1.0)
    betas = 1.0 - (raw_cumprod / raw_prev)
    betas = torch.clip(betas[1:], 0.0001, 0.9999)
    return torch.cumprod(1.0 - betas, dim=0)


def _ddim_timesteps(settings: InferenceSettings, device: torch.device) -> torch.Tensor:
    timesteps = torch.linspace(
        settings.training_num_timesteps - 1,
        0,
        settings.ddim_steps,
        device=device,
        dtype=torch.float32,
    ).round().long()
    timesteps = torch.unique_consecutive(timesteps)
    if timesteps[-1].item() != 0:
        timesteps[-1] = 0
    if timesteps.numel() > 1 and not bool((timesteps[:-1] > timesteps[1:]).all()):
        raise RuntimeError("DDIM 时间步没有保持严格递减")
    return timesteps


def _hard_radial_clamp(pos: torch.Tensor, radius: torch.Tensor | None) -> torch.Tensor:
    if radius is None or pos.numel() == 0:
        return pos
    radius = torch.clamp(radius.to(device=pos.device, dtype=pos.dtype), min=1e-6)
    norm = torch.linalg.vector_norm(pos, dim=-1, keepdim=True)
    scale = torch.clamp(radius / (norm + 1e-8), max=1.0)
    return pos * scale


def _apply_position_boundary(
    pos: torch.Tensor,
    radius: torch.Tensor | None,
    soft_ratio: float | None,
) -> torch.Tensor:
    if radius is None:
        return pos
    radius = torch.clamp(radius.to(device=pos.device, dtype=pos.dtype), min=1e-6)
    if soft_ratio is None:
        pos = _hard_radial_clamp(pos, radius)
    else:
        soft_radius = radius * max(float(soft_ratio), 1e-6)
        norm = torch.linalg.vector_norm(pos, dim=-1, keepdim=True)
        scale = torch.where(
            norm > soft_radius,
            soft_radius * torch.tanh(norm / soft_radius) / (norm + 1e-8),
            torch.ones_like(norm),
        )
        pos = pos * scale
    center = pos.mean(dim=0, keepdim=True) if pos.numel() else pos
    if pos.numel():
        center_norm = torch.linalg.vector_norm(center, dim=-1, keepdim=True)
        center_scale = torch.clamp(radius / (center_norm + 1e-8), max=1.0)
        pos = pos + (center * center_scale - center)
    return _hard_radial_clamp(pos, radius)


def _predict_x0(
    free_feat: torch.Tensor,
    free_pos: torch.Tensor,
    free_ref: torch.Tensor,
    pred_noise_feat: torch.Tensor,
    pred_noise_pos: torch.Tensor,
    pred_noise_ref: torch.Tensor,
    alpha_t: torch.Tensor,
    position_radius: torch.Tensor | None,
    settings: InferenceSettings,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sqrt_alpha = torch.sqrt(alpha_t)
    sqrt_one_minus = torch.sqrt(torch.clamp(1.0 - alpha_t, min=0.0))
    safe_denom = sqrt_alpha + 1e-8
    x0_feat = (free_feat - sqrt_one_minus * pred_noise_feat) / safe_denom
    x0_pos = (free_pos - sqrt_one_minus * pred_noise_pos) / safe_denom
    x0_ref = (free_ref - sqrt_one_minus * pred_noise_ref) / safe_denom

    embed = x0_feat[:, : settings.embedding_dim].clamp(
        -settings.embedding_safety_clamp,
        settings.embedding_safety_clamp,
    )
    chem = x0_feat[:, settings.embedding_dim :].clamp(
        -settings.chem_safety_clamp,
        settings.chem_safety_clamp,
    )
    x0_feat = torch.cat([embed, chem], dim=-1)
    if position_radius is None:
        x0_pos = x0_pos.clamp(
            -settings.position_abs_safety_clamp,
            settings.position_abs_safety_clamp,
        )
    else:
        x0_pos = _apply_position_boundary(x0_pos, position_radius, None)
    x0_ref = x0_ref.clamp(
        -settings.ref_coords_safety_clamp,
        settings.ref_coords_safety_clamp,
    )
    return x0_feat, x0_pos, x0_ref


def _randn_like(
    tensor: torch.Tensor, generator: torch.Generator
) -> torch.Tensor:
    return torch.randn(
        tensor.shape,
        dtype=tensor.dtype,
        device=tensor.device,
        generator=generator,
    )


def _ddim_progress(
    timesteps: torch.Tensor,
    *,
    progress_interval: int,
):
    """Yield indexed DDIM timesteps with one in-place progress display."""

    total = int(timesteps.numel())
    if progress_interval <= 0:
        yield from enumerate(timesteps)
        return

    if _tqdm is not None:
        progress = _tqdm(
            enumerate(timesteps),
            total=total,
            desc="      DDIM",
            unit="step",
            dynamic_ncols=True,
            file=sys.stdout,
            miniters=max(1, int(progress_interval)),
        )
        try:
            for step_index, t_scalar in progress:
                progress.set_postfix(
                    t=int(t_scalar.item()),
                    refresh=False,
                )
                yield step_index, t_scalar
        finally:
            progress.close()
        return

    try:
        for step_index, t_scalar in enumerate(timesteps):
            yield step_index, t_scalar
            completed = step_index + 1
            if (
                step_index == 0
                or completed == total
                or completed % int(progress_interval) == 0
            ):
                percentage = 100 if total == 0 else int(100 * completed / total)
                print(
                    f"\r      DDIM: {completed}/{total} step "
                    f"({percentage:3d}%) t={int(t_scalar.item())}",
                    end="",
                    flush=True,
                )
    finally:
        print()


@torch.no_grad()
def run_ddim_sampling(
    bundle: CheckpointBundle,
    condition: PreparedCondition,
    settings: InferenceSettings,
    num_generated_nodes: int,
    seed: int,
    progress_callback=None,
) -> tuple[Data, dict[str, torch.Tensor]]:
    device = condition.data.pos.device
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    free_feat = torch.randn(
        (num_generated_nodes, settings.ligand_feature_dim),
        device=device,
        dtype=torch.float32,
        generator=generator,
    )
    free_pos = torch.randn(
        (num_generated_nodes, 3),
        device=device,
        dtype=torch.float32,
        generator=generator,
    )
    free_ref = torch.randn(
        (num_generated_nodes, 3, 3),
        device=device,
        dtype=torch.float32,
        generator=generator,
    )

    alphas_cumprod = _cosine_alphas_cumprod(
        settings.training_num_timesteps, device
    )
    timesteps = _ddim_timesteps(settings, device)
    protein_pos = condition.data.pos[condition.data.is_protein]
    position_radius = (
        torch.linalg.vector_norm(protein_pos, dim=-1).max()
        + settings.position_clamp_buffer
        if protein_pos.numel()
        else None
    )

    for step_index, t_scalar in _ddim_progress(
        timesteps,
        progress_interval=settings.progress_interval,
    ):
        graph, _, generated_ligand_mask = build_sampling_graph(
            condition,
            free_feat,
            free_pos,
            free_ref,
            settings,
            bundle.max_ll_neighbors,
            bundle.max_lp_neighbors,
        )
        time_tensor = t_scalar.reshape(1)
        outputs = bundle.model(graph, time_tensor, return_aux=False)
        if not isinstance(outputs, tuple) or len(outputs) != 5:
            raise RuntimeError("主模型默认输出不再是预期的五项 tuple")
        pred_noise_feat, pred_noise_pos, pred_noise_ref = outputs[:3]
        pred_noise_feat = pred_noise_feat[generated_ligand_mask]
        pred_noise_pos = pred_noise_pos[generated_ligand_mask]
        pred_noise_ref = pred_noise_ref[generated_ligand_mask]
        if tuple(pred_noise_feat.shape) != tuple(free_feat.shape):
            raise RuntimeError("模型自由 L feature 噪声输出形状不匹配")
        if tuple(pred_noise_pos.shape) != tuple(free_pos.shape):
            raise RuntimeError("模型自由 L pos 噪声输出形状不匹配")
        if tuple(pred_noise_ref.shape) != tuple(free_ref.shape):
            raise RuntimeError("模型自由 L ref_coords 噪声输出形状不匹配")

        alpha_t = alphas_cumprod[int(t_scalar.item())]
        x0_feat, x0_pos, x0_ref = _predict_x0(
            free_feat,
            free_pos,
            free_ref,
            pred_noise_feat,
            pred_noise_pos,
            pred_noise_ref,
            alpha_t,
            position_radius,
            settings,
        )
        is_last = step_index == timesteps.numel() - 1
        if is_last:
            free_feat, free_pos, free_ref = x0_feat, x0_pos, x0_ref
        else:
            t_prev = int(timesteps[step_index + 1].item())
            alpha_prev = alphas_cumprod[t_prev]
            sigma = settings.ddim_eta * torch.sqrt(
                torch.clamp(
                    (1.0 - alpha_prev)
                    / (1.0 - alpha_t)
                    * (1.0 - alpha_t / alpha_prev),
                    min=0.0,
                )
            )
            direction = torch.sqrt(
                torch.clamp(1.0 - alpha_prev - sigma.square(), min=0.0)
            )
            sqrt_alpha_prev = torch.sqrt(alpha_prev)
            feat_noise = (
                sigma * _randn_like(x0_feat, generator)
                if settings.ddim_eta > 0
                else 0.0
            )
            pos_noise = (
                sigma * _randn_like(x0_pos, generator)
                if settings.ddim_eta > 0
                else 0.0
            )
            ref_noise = (
                sigma * _randn_like(x0_ref, generator)
                if settings.ddim_eta > 0
                else 0.0
            )
            free_feat = (
                sqrt_alpha_prev * x0_feat
                + direction * pred_noise_feat
                + feat_noise
            )
            free_pos = (
                sqrt_alpha_prev * x0_pos
                + direction * pred_noise_pos
                + pos_noise
            )
            free_pos = _apply_position_boundary(
                free_pos,
                position_radius,
                settings.position_soft_clamp_ratio,
            )
            free_ref = (
                sqrt_alpha_prev * x0_ref
                + direction * pred_noise_ref
                + ref_noise
            )
        if progress_callback is not None:
            progress_callback(step_index + 1, int(timesteps.numel()))

    final_graph, _, _ = build_sampling_graph(
        condition,
        free_feat,
        free_pos,
        free_ref,
        settings,
        bundle.max_ll_neighbors,
        bundle.max_lp_neighbors,
    )
    final_aux = bundle.model(
        final_graph,
        torch.zeros(1, dtype=torch.long, device=device),
        return_aux=True,
    )
    if not isinstance(final_aux, dict):
        raise RuntimeError("主模型 return_aux=True 未返回字典")
    return final_graph, final_aux


def _decode_classes(
    classes: torch.Tensor, mapping: Mapping[str, Any]
) -> tuple[torch.Tensor, torch.Tensor]:
    classes = classes.detach().cpu().long()
    values = torch.full_like(classes, -1)
    covered = torch.zeros_like(classes, dtype=torch.bool)
    for class_key, value in mapping["classes"].items():
        class_index = int(class_key)
        mask = classes == class_index
        values[mask] = int(value)
        covered |= mask
    overflow = classes == int(mapping["overflow_class"])
    values[overflow] = int(mapping["overflow_min_value"])
    covered |= overflow
    if not bool(covered.all()):
        invalid = classes[~covered].unique().tolist()
        raise ValueError(f"模型预测了类映射之外的类别: {invalid}")
    return values, overflow


def _as_edge_index(pairs: list[tuple[int, int]]) -> torch.Tensor:
    if not pairs:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(pairs, dtype=torch.long).t().contiguous()


def _aggregate_dynamic_pairs(
    candidate_edge_index: torch.Tensor,
    candidate_edge_type: torch.Tensor,
    probabilities: torch.Tensor,
    graph: Data,
    ligand_graph_indices: torch.Tensor,
    requested_type: int,
    edge_min_probability: float,
) -> dict[str, torch.Tensor]:
    edge_index = candidate_edge_index.detach().cpu().long()
    edge_type = candidate_edge_type.detach().cpu().long()
    probabilities = probabilities.detach().cpu().float()
    is_ligand = graph.is_ligand.detach().cpu().bool()
    is_protein = graph.is_protein.detach().cpu().bool()
    ligand_indices = ligand_graph_indices.detach().cpu().long()
    graph_to_ligand = {
        int(graph_idx): local_idx
        for local_idx, graph_idx in enumerate(ligand_indices.tolist())
    }

    grouped: dict[tuple[int, int], list[torch.Tensor]] = {}
    selected_positions = torch.nonzero(
        edge_type == requested_type, as_tuple=False
    ).flatten()
    for edge_position in selected_positions.tolist():
        src = int(edge_index[0, edge_position])
        dst = int(edge_index[1, edge_position])
        if requested_type == 0:
            if not bool(is_ligand[src] and is_ligand[dst]):
                raise ValueError("Candidate-LL 包含非 L 节点")
            key = (min(src, dst), max(src, dst))
        else:
            if bool(is_ligand[src] and is_protein[dst]):
                key = (src, dst)
            elif bool(is_protein[src] and is_ligand[dst]):
                key = (dst, src)
            else:
                raise ValueError("Candidate-LP 端点类型非法")
        grouped.setdefault(key, []).append(probabilities[edge_position])

    pair_keys = sorted(grouped)
    mean_probabilities = (
        torch.stack(
            [torch.stack(grouped[key], dim=0).mean(dim=0) for key in pair_keys],
            dim=0,
        )
        if pair_keys
        else torch.empty((0, 3), dtype=torch.float32)
    )
    real_class = requested_type
    predicted_mask = (
        (mean_probabilities[:, real_class] > mean_probabilities[:, 2])
        & (mean_probabilities[:, real_class] >= edge_min_probability)
        if pair_keys
        else torch.empty((0,), dtype=torch.bool)
    )
    predicted_pairs = [
        pair_keys[index]
        for index in torch.nonzero(predicted_mask, as_tuple=False).flatten().tolist()
    ]
    predicted_probability = (
        mean_probabilities[predicted_mask, real_class]
        if pair_keys
        else torch.empty((0,), dtype=torch.float32)
    )

    if requested_type == 0:
        candidate_local = [
            (graph_to_ligand[first], graph_to_ligand[second])
            for first, second in pair_keys
        ]
        predicted_local = [
            (graph_to_ligand[first], graph_to_ligand[second])
            for first, second in predicted_pairs
        ]
        return {
            "candidate_pair_index_graph": _as_edge_index(pair_keys),
            "candidate_pair_index_ligand": _as_edge_index(candidate_local),
            "candidate_pair_probabilities": mean_probabilities,
            "edge_index_graph": _as_edge_index(predicted_pairs),
            "edge_index_ligand": _as_edge_index(predicted_local),
            "edge_probability": predicted_probability,
        }

    candidate_ligand_local = torch.tensor(
        [graph_to_ligand[first] for first, _ in pair_keys], dtype=torch.long
    )
    candidate_protein_graph = torch.tensor(
        [second for _, second in pair_keys], dtype=torch.long
    )
    predicted_ligand_local = torch.tensor(
        [graph_to_ligand[first] for first, _ in predicted_pairs],
        dtype=torch.long,
    )
    predicted_protein_graph = torch.tensor(
        [second for _, second in predicted_pairs], dtype=torch.long
    )
    return {
        "candidate_pair_index_graph": _as_edge_index(pair_keys),
        "candidate_ligand_index": candidate_ligand_local,
        "candidate_protein_graph_index": candidate_protein_graph,
        "candidate_pair_probabilities": mean_probabilities,
        "edge_index_graph": _as_edge_index(predicted_pairs),
        "ligand_index": predicted_ligand_local,
        "protein_graph_index": predicted_protein_graph,
        "edge_probability": predicted_probability,
    }


def _build_covalent_output(
    final_aux: Mapping[str, torch.Tensor],
    ligand_graph_indices: torch.Tensor,
    threshold: float,
) -> dict[str, torch.Tensor]:
    candidate_graph = final_aux["covalent_candidate_edge_index"].detach().cpu().long()
    logits = final_aux["covalent_logits"].detach().cpu().float()
    if (
        candidate_graph.ndim != 2
        or candidate_graph.shape[0] != 2
        or logits.ndim != 1
        or candidate_graph.shape[1] != logits.numel()
    ):
        raise ValueError("共价候选索引与 logits 形状不匹配")
    graph_to_ligand = {
        int(graph_idx): local_idx
        for local_idx, graph_idx in enumerate(
            ligand_graph_indices.detach().cpu().tolist()
        )
    }
    local_pairs = []
    for first, second in candidate_graph.t().tolist():
        if first not in graph_to_ligand or second not in graph_to_ligand:
            raise ValueError("共价候选包含非 L 节点")
        local_pairs.append((graph_to_ligand[first], graph_to_ligand[second]))
    candidate_local = _as_edge_index(local_pairs)
    probabilities = torch.sigmoid(logits)
    selected = probabilities >= threshold
    return {
        "candidate_edge_index_graph": candidate_graph,
        "candidate_edge_index_ligand": candidate_local,
        "logits": logits,
        "probabilities": probabilities,
        "edge_index_graph": candidate_graph[:, selected],
        "edge_index_ligand": candidate_local[:, selected],
        "edge_probability": probabilities[selected],
        "threshold": torch.tensor(float(threshold), dtype=torch.float32),
    }


def build_stage1_result(
    bundle: CheckpointBundle,
    condition: PreparedCondition,
    settings: InferenceSettings,
    sample_index: int,
    attempt_index: int,
    seed: int,
    num_generated_nodes: int,
    final_graph: Data,
    final_aux: Mapping[str, torch.Tensor],
    *,
    minimum_connected_component_ligand_nodes: int | None = None,
) -> dict[str, Any]:
    required_aux = (
        "edge_logits",
        "candidate_edge_index",
        "candidate_edge_type",
        "hac_logits",
        "ring_logits",
        "covalent_logits",
        "covalent_candidate_edge_index",
    )
    missing = [key for key in required_aux if key not in final_aux]
    if missing:
        raise ValueError(f"主模型辅助输出缺少字段: {missing}")

    ligand_graph_indices = torch.nonzero(
        final_graph.is_ligand, as_tuple=False
    ).flatten()
    ligand_indices_cpu = ligand_graph_indices.detach().cpu().long()
    num_condition_nodes = int(condition.data.num_nodes)
    is_generated = ligand_indices_cpu >= num_condition_nodes
    is_fixed = ~is_generated
    if int(is_generated.sum().item()) != num_generated_nodes:
        raise RuntimeError("最终生成 L 节点数量与索引映射不一致")

    ligand_pos = final_graph.pos[ligand_graph_indices].detach().cpu().float()
    ligand_ref_relative = (
        final_graph.ref_coords[ligand_graph_indices].detach().cpu().float()
    )
    ligand_ref_absolute = ligand_pos[:, None, :] + ligand_ref_relative
    feature_128 = (
        final_graph.frag_embeds[ligand_graph_indices].detach().cpu().float()
    )
    feature_128_normalized = F.normalize(feature_128, p=2, dim=-1, eps=1e-8)

    hac_logits = final_aux["hac_logits"].detach().cpu().float()
    ring_logits = final_aux["ring_logits"].detach().cpu().float()
    num_ligand_nodes = int(ligand_graph_indices.numel())
    if minimum_connected_component_ligand_nodes is None:
        minimum_connected_component_ligand_nodes = num_ligand_nodes
    if (
        isinstance(minimum_connected_component_ligand_nodes, bool)
        or not isinstance(minimum_connected_component_ligand_nodes, int)
    ):
        raise TypeError(
            "minimum_connected_component_ligand_nodes 必须为整数"
        )
    if not (
        1
        <= minimum_connected_component_ligand_nodes
        <= num_ligand_nodes
    ):
        raise ValueError(
            "minimum_connected_component_ligand_nodes 必须位于 "
            f"[1, {num_ligand_nodes}]"
        )
    if hac_logits.ndim != 2 or hac_logits.shape[0] != num_ligand_nodes:
        raise ValueError("HAC logits 与最终 L 节点数量不匹配")
    if ring_logits.ndim != 2 or ring_logits.shape[0] != num_ligand_nodes:
        raise ValueError("Ring logits 与最终 L 节点数量不匹配")
    hac_model_class = hac_logits.argmax(dim=-1)
    ring_model_class = ring_logits.argmax(dim=-1)
    hac_model_value, hac_model_overflow = _decode_classes(
        hac_model_class, bundle.hac_mapping
    )
    ring_model_value, ring_model_overflow = _decode_classes(
        ring_model_class, bundle.ring_mapping
    )
    hac_value = hac_model_value.clone()
    ring_value = ring_model_value.clone()
    fixed_graph_indices = ligand_indices_cpu[is_fixed]
    if fixed_graph_indices.numel():
        hac_value[is_fixed] = condition.data.hac[
            fixed_graph_indices.to(condition.data.hac.device)
        ].detach().cpu()
        ring_value[is_fixed] = condition.data.ring_count[
            fixed_graph_indices.to(condition.data.ring_count.device)
        ].detach().cpu()
    hac_overflow = hac_model_overflow & is_generated
    ring_overflow = ring_model_overflow & is_generated

    candidate_edge_index = (
        final_aux["candidate_edge_index"].detach().cpu().long()
    )
    candidate_edge_type = final_aux["candidate_edge_type"].detach().cpu().long()
    edge_logits = final_aux["edge_logits"].detach().cpu().float()
    if (
        candidate_edge_index.ndim != 2
        or candidate_edge_index.shape[0] != 2
        or candidate_edge_type.shape != (candidate_edge_index.shape[1],)
        or edge_logits.shape != (candidate_edge_index.shape[1], 3)
    ):
        raise ValueError("动态边 logits 与候选边数量不匹配")
    edge_probabilities = torch.softmax(edge_logits, dim=-1)
    edge_predicted_class = edge_probabilities.argmax(dim=-1)
    ll_output = _aggregate_dynamic_pairs(
        candidate_edge_index,
        candidate_edge_type,
        edge_probabilities,
        final_graph,
        ligand_graph_indices,
        requested_type=0,
        edge_min_probability=settings.edge_min_probability,
    )
    lp_output = _aggregate_dynamic_pairs(
        candidate_edge_index,
        candidate_edge_type,
        edge_probabilities,
        final_graph,
        ligand_graph_indices,
        requested_type=1,
        edge_min_probability=settings.edge_min_probability,
    )
    covalent_output = _build_covalent_output(
        final_aux,
        ligand_graph_indices,
        settings.covalent_probability_threshold,
    )

    return {
        "metadata": {
            "created_utc": utc_now_iso(),
            "source_pt": str(condition.source_path),
            "condition_relative_path": condition.relative_path.as_posix(),
            "sample_id": condition.sample_id,
            "sample_index": int(sample_index),
            "attempt_index": int(attempt_index),
            "seed": int(seed),
            "mode": condition.mode,
            "coordinate_frame": "condition_centered",
            "condition_ref_coords_mode": "relative_to_node_pos",
            "ref_coords_absolute_definition": "pos + ref_coords_relative",
            "checkpoint_path": str(bundle.checkpoint_path),
            "checkpoint_epoch": bundle.epoch,
            "embedding_table_sha256": bundle.profile[
                "fragment_embedding_table"
            ]["table_sha256"],
            "num_fixed_ligand_nodes": int(is_fixed.sum().item()),
            "num_generated_ligand_nodes": int(is_generated.sum().item()),
            "num_ligand_nodes": int(ligand_graph_indices.numel()),
            "minimum_connected_component_ligand_nodes": int(
                minimum_connected_component_ligand_nodes
            ),
            "max_ll_neighbors": bundle.max_ll_neighbors,
            "max_lp_neighbors": bundle.max_lp_neighbors,
            "hac_cap": int(bundle.profile["hac_cap"]),
            "ring_cap": int(bundle.profile["ring_cap"]),
        },
        "nodes": {
            "graph_index": ligand_indices_cpu,
            "is_fixed": is_fixed,
            "is_generated": is_generated,
            "pos": ligand_pos,
            "ref_coords_relative": ligand_ref_relative,
            "ref_coords_absolute": ligand_ref_absolute,
            "feature_128": feature_128,
            "feature_128_normalized": feature_128_normalized,
            "hac_value": hac_value,
            "hac_overflow": hac_overflow,
            "ring_value": ring_value,
            "ring_overflow": ring_overflow,
        },
        "hac_diagnostics": {
            "logits": hac_logits,
            "predicted_class": hac_model_class,
            "decoded_value": hac_model_value,
            "overflow": hac_model_overflow,
            "class_mapping": bundle.hac_mapping,
        },
        "ring_diagnostics": {
            "logits": ring_logits,
            "predicted_class": ring_model_class,
            "decoded_value": ring_model_value,
            "overflow": ring_model_overflow,
            "class_mapping": bundle.ring_mapping,
        },
        "dynamic_edges": {
            "class_mapping": {"LL": 0, "LP": 1, "Null": 2},
            "candidate_edge_index_graph": candidate_edge_index,
            "candidate_edge_type": candidate_edge_type,
            "logits": edge_logits,
            "probabilities": edge_probabilities,
            "predicted_class": edge_predicted_class,
        },
        "ll_edges": ll_output,
        "lp_edges": lp_output,
        "covalent_edges": covalent_output,
    }


def atomic_torch_save(payload: Any, output_path: Path) -> None:
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(
        f".{output_path.name}.{uuid.uuid4().hex}.tmp"
    )
    try:
        torch.save(payload, temp_path)
        os.replace(temp_path, output_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def atomic_json_save(payload: Any, output_path: Path) -> None:
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(
        f".{output_path.name}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temp_path, output_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()
