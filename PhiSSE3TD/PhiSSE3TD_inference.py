from __future__ import annotations

import math
import multiprocessing as mp
import os
import secrets
import shutil
import sys
import traceback
from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ProcessPoolExecutor,
    wait,
)
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import PhiSSE3TD_nodes_inference as node_count_inference

PHISLINKER_DIR = PROJECT_ROOT / "PhiSLinker"
if str(PHISLINKER_DIR) not in sys.path:
    sys.path.insert(0, str(PHISLINKER_DIR))

from PhiSLinker_atom_pair_compare import (
    FatalConditionError,
    RetryableReconstructionError,
    load_atom_pair_inference_runtime,
    load_reconstruction_vocabulary,
    preflight_fixed_condition_fragments,
    reconstruct_stage1_result,
)

from PhiSSE3TD_inference_utils import (
    InferenceSettings,
    atomic_json_save,
    atomic_torch_save,
    build_stage1_result,
    create_embedding_lookup,
    discover_condition_files,
    load_checkpoint_bundle,
    prepare_condition,
    resolve_device,
    run_ddim_sampling,
    utc_now_iso,
)


class WebInferenceCancelled(RuntimeError):
    """当前网页生成任务已收到终止请求。"""


# =============================================================================
# 第一阶段推理超参数
# =============================================================================

# 直接运行本脚本时使用的默认模式；网页调用会单独传入所选模式。
MODEL_VARIANT = "activity_modify"
# 各模型模式对应的 checkpoint；通常只需调整 MODEL_VARIANT，无需改动本映射。
CHECKPOINT_PATHS = {
    "pretrain": (
        SCRIPT_DIR
        / "training_artifacts"
        / "checkpoints"
        / "diffusion_pretrain"
        / "best_model.pt"
    ),
    "creative": (
        SCRIPT_DIR
        / "training_artifacts"
        / "checkpoints"
        / "diffusion_finetune"
        / "creative"
        / "best_model_creative_epoch_0097_val_1.564923.pt"
    ),
    "survival_linker": (
        SCRIPT_DIR
        / "training_artifacts"
        / "checkpoints"
        / "diffusion_finetune"
        / "survival_linker"
        / "best_model_survival_linker_epoch_0032_val_1.312337.pt"
    ),
    "survival_modify": (
        SCRIPT_DIR
        / "training_artifacts"
        / "checkpoints"
        / "diffusion_finetune"
        / "survival_modify"
        / "best_model_survival_modify_epoch_0082_val_1.562644.pt"
    ),
    "activity_modify": (
        SCRIPT_DIR
        / "training_artifacts"
        / "checkpoints"
        / "diffusion_finetune"
        / "activity"
        / "best_model_activity_modify_epoch_0048_val_1.432674.pt"
    ),
    "activity_linker": (
        SCRIPT_DIR
        / "training_artifacts"
        / "checkpoints"
        / "diffusion_finetune"
        / "activity"
        / "best_model_activity_linker.pt"
    ),
}
# 第一阶段条件 .pt 的输入根目录。
CONDITION_ROOT = PROJECT_ROOT / "Datas" / "work_datas" / "processed"
# 第一阶段推理结果和运行 manifest 的输出根目录。
OUTPUT_ROOT = PROJECT_ROOT / "Datas" / "work_datas" / "inference_stage"
# 标准片段嵌入表；必须与训练 checkpoint 使用的词表一致。
EMBEDDING_TABLE_PATH = (
    PROJECT_ROOT
    / "Datas"
    / "Processed_datas"
    / "SE3TD_128_Shared_data"
    / "fragment_embeddings_128d.npz"
)
# 标准片段嵌入表的元数据，用于校验词表画像和张量维度。
EMBEDDING_METADATA_PATH = (
    PROJECT_ROOT
    / "Datas"
    / "Processed_datas"
    / "SE3TD_128_Shared_data"
    / "fragment_embeddings_128d_metadata.json"
)

# =============================================================================
# 三维重建、原子匹配与复合物优化超参数
# =============================================================================

# 是否在第一阶段采样后继续执行片段解码、原子匹配和三维复合物重建。
ENABLE_3D_RECONSTRUCTION = True
# 将生成片段的 128 维嵌入解析为 SMILES 时使用的自定义词表目录。
CUSTOM_FRAGMENT_VOCAB_DIR = (
    PROJECT_ROOT
    / "Datas"
    / "Processed_datas"
    / "SE3TD_128_Shared_data"
    / "fragment_custom_embedding_128d"
)
# survival_linker 和 survival_modify 使用各自子集；其余模型使用普通子集。
CUSTOM_FRAGMENT_VOCAB_PATHS = {
    "pretrain": CUSTOM_FRAGMENT_VOCAB_DIR / "fragment_embeddings_128d.npz",
    "creative": CUSTOM_FRAGMENT_VOCAB_DIR / "fragment_embeddings_128d.npz",
    "survival_linker": (
        CUSTOM_FRAGMENT_VOCAB_DIR / "fragment_survival_linker_embeddings_128d.npz"
    ),
    "survival_modify": (
        CUSTOM_FRAGMENT_VOCAB_DIR / "fragment_survival_modify_embeddings_128d.npz"
    ),
    "activity_modify": (
        CUSTOM_FRAGMENT_VOCAB_DIR / "fragment_activity_modify_embeddings_128d.npz"
    ),
    "activity_linker": (
        CUSTOM_FRAGMENT_VOCAB_DIR / "fragment_activity_modify_embeddings_128d.npz"
    ),
}
if set(CUSTOM_FRAGMENT_VOCAB_PATHS) != set(CHECKPOINT_PATHS):
    raise RuntimeError("checkpoint 与重建词汇表的 MODEL_VARIANT 映射不一致")
# 设为 None 时，从 PhiSLinker/training_artifacts/checkpoints/*/best_model.pt
# 中选择最近修改的 checkpoint。
ATOM_PAIR_CHECKPOINT_PATH: Path | None = None
# 原子匹配模型使用的片段嵌入词表，必须与其 checkpoint 训练配置一致。
ATOM_PAIR_FRAGMENT_VOCAB_PATH = (
    PROJECT_ROOT
    / "Datas"
    / "Processed_datas"
    / "SE3TD_128_Shared_data"
    / "fragment_embeddings_128d.npz"
)
# 原子匹配模型使用的原子嵌入词表，必须与其 checkpoint 训练配置一致。
ATOM_PAIR_ATOM_VOCAB_PATH = (
    PROJECT_ROOT
    / "Datas"
    / "Processed_datas"
    / "SE3TD_128_Shared_data"
    / "fragment_atom_embeddings_64d.npz"
)
# 每得到一个成功样本所允许的最大尝试倍数，用于限制失败重采样次数。
MAX_ATTEMPTS_PER_SUCCESS = 20
# 每个 GPU 推理进程预留的显存预算（GiB）。动态并发数按当前可用显存
# 除以该值向下取整；实际单进程峰值应始终低于此预算。
GPU_MEMORY_PER_PROCESS_GIB = 12
MAX_WEB_INFERENCE_PROCESSES = 8

# 直接运行本脚本时的条件文件选择；网页调用始终指定单个 .pt。
INPUT_SELECTION = "first"
# 直接运行本脚本时的默认生成数量；网页调用会单独传入。
NUM_SAMPLES_PER_CONDITION = 10

# 训练扩散总步数，必须与 checkpoint 的训练配置保持一致。
TRAINING_NUM_TIMESTEPS = 1000
# 噪声调度类型，当前实现和 checkpoint 仅支持 cosine。
BETA_SCHEDULE = "cosine"
# DDIM 反向采样步数；越小越快，但可能降低生成质量或稳定性。
# 有效范围为 [2, TRAINING_NUM_TIMESTEPS]。
DDIM_STEPS = 600
# DDIM 随机性系数；0 为确定性路径，大于 0 时会引入额外采样噪声。
DDIM_ETA = 0.0

# 配体—配体候选动态图半径，单位为埃；建议与训练配置一致。
LL_RADIUS = 5.0
# 配体—蛋白候选动态图半径，单位为埃；建议与训练配置一致。
LP_RADIUS = 7.0
# LL/LP 动态边被保留所需的最低类别概率，范围为 [0, 1]。
EDGE_MIN_PROBABILITY = 0.4
# 将配体节点对判定为共价连接的最低概率，范围为 [0, 1]。
COVALENT_PROBABILITY_THRESHOLD = 0.3

# 以下四项由 checkpoint schema 固定；除非模型重新训练，否则不要调整。
# 片段嵌入维度。
EMBEDDING_DIM = 128
# 节点化学特征维度。
CHEM_DIM = 14
# 节点类型 one-hot 维度（配体、蛋白、全局节点）。
NODE_TYPE_DIM = 3
# 边属性维度。
EDGE_ATTR_DIM = 4

# 反向扩散中片段嵌入特征的绝对值安全截断上限。
EMBEDDING_SAFETY_CLAMP = 20.0
# 反向扩散中化学特征的绝对值安全截断上限。
CHEM_SAFETY_CLAMP = 20.0
# 反向扩散中局部参考坐标的绝对值安全截断上限。
REF_COORDS_SAFETY_CLAMP = 20.0
# 生成节点绝对坐标的数值安全截断上限，单位为埃。
POSITION_ABS_SAFETY_CLAMP = 50.0
# 位置边界相对最远蛋白节点额外扩展的距离，单位为埃。
POSITION_CLAMP_BUFFER = 20.0
# 软位置边界相对硬边界半径的比例；越小越早压缩远端坐标。
POSITION_SOFT_CLAMP_RATIO = 1.0

# 每次运行将该闭区间的种子随机打乱；跨条件和失败重试均不重复分配。
RANDOM_SEED_MIN = 1
RANDOM_SEED_MAX = 20480
# 推理设备；有可用 CUDA 时默认使用 GPU，否则使用 CPU。
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# 单个条件失败后是否继续处理后续条件；False 表示遇到终止性错误即停止。
CONTINUE_ON_ERROR = False
# DDIM 单行进度条的最小刷新步数；设为 0 可关闭进度显示。
PROGRESS_INTERVAL = 50


def build_settings(
    model_variant: str | None = None,
    num_samples_per_condition: int | None = None,
    output_root: Path | None = None,
    manual_total_ligand_nodes: int | None = None,
    creative_extra_generated_ligand_nodes: int | None = None,
    taska_added_ligand_nodes: int | None = None,
    taska_extra_generated_ligand_nodes: int | None = None,
) -> InferenceSettings:
    selected_variant = MODEL_VARIANT if model_variant is None else model_variant
    if selected_variant not in CHECKPOINT_PATHS:
        raise ValueError(
            f"未知 MODEL_VARIANT={selected_variant!r}；"
            f"允许值为 {sorted(CHECKPOINT_PATHS)}"
        )
    settings = InferenceSettings(
        model_variant=selected_variant,
        checkpoint_path=CHECKPOINT_PATHS[selected_variant],
        condition_root=CONDITION_ROOT,
        output_root=OUTPUT_ROOT if output_root is None else Path(output_root),
        embedding_table_path=EMBEDDING_TABLE_PATH,
        embedding_metadata_path=EMBEDDING_METADATA_PATH,
        input_selection=INPUT_SELECTION,
        num_samples_per_condition=(
            NUM_SAMPLES_PER_CONDITION
            if num_samples_per_condition is None else num_samples_per_condition
        ),
        training_num_timesteps=TRAINING_NUM_TIMESTEPS,
        beta_schedule=BETA_SCHEDULE,
        ddim_steps=DDIM_STEPS,
        ddim_eta=DDIM_ETA,
        ll_radius=LL_RADIUS,
        lp_radius=LP_RADIUS,
        edge_min_probability=EDGE_MIN_PROBABILITY,
        covalent_probability_threshold=COVALENT_PROBABILITY_THRESHOLD,
        embedding_dim=EMBEDDING_DIM,
        chem_dim=CHEM_DIM,
        node_type_dim=NODE_TYPE_DIM,
        edge_attr_dim=EDGE_ATTR_DIM,
        embedding_safety_clamp=EMBEDDING_SAFETY_CLAMP,
        chem_safety_clamp=CHEM_SAFETY_CLAMP,
        ref_coords_safety_clamp=REF_COORDS_SAFETY_CLAMP,
        position_abs_safety_clamp=POSITION_ABS_SAFETY_CLAMP,
        position_clamp_buffer=POSITION_CLAMP_BUFFER,
        position_soft_clamp_ratio=POSITION_SOFT_CLAMP_RATIO,
        random_seed_min=RANDOM_SEED_MIN,
        random_seed_max=RANDOM_SEED_MAX,
        device=DEVICE,
        continue_on_error=CONTINUE_ON_ERROR,
        progress_interval=PROGRESS_INTERVAL,
        manual_total_ligand_nodes=(
            node_count_inference.CREATIVE_NODE_PRESETS[
                node_count_inference.DEFAULT_CREATIVE_NODE_PRESET
            ]["base_nodes"]
            if manual_total_ligand_nodes is None else manual_total_ligand_nodes
        ),
        creative_extra_generated_ligand_nodes=(
            node_count_inference.CREATIVE_NODE_PRESETS[
                node_count_inference.DEFAULT_CREATIVE_NODE_PRESET
            ]["extra_nodes"]
            if creative_extra_generated_ligand_nodes is None
            else creative_extra_generated_ligand_nodes
        ),
        taska_added_ligand_nodes=(
            node_count_inference.TASKA_DEFAULT_ADDED_LIGAND_NODES
            if taska_added_ligand_nodes is None else taska_added_ligand_nodes
        ),
        taska_extra_generated_ligand_nodes=(
            node_count_inference.TASKA_EXTRA_GENERATED_LIGAND_NODES
            if taska_extra_generated_ligand_nodes is None
            else taska_extra_generated_ligand_nodes
        ),
    )
    settings.validate()
    _validate_reconstruction_settings()
    return settings


def get_web_inference_options(condition_pt: str | Path) -> dict[str, Any]:
    """根据单个条件文件返回网页允许展示的扩散模型模式。"""
    condition_path = Path(condition_pt).resolve()
    if not condition_path.is_file():
        raise FileNotFoundError(f"条件文件不存在: {condition_path}")
    condition = torch.load(
        condition_path, map_location="cpu", weights_only=False
    )
    source = str(getattr(condition, "source", "")).lower()
    node_type = getattr(condition, "node_type", None)
    if (
        not isinstance(node_type, torch.Tensor)
        or node_type.ndim != 2
        or node_type.shape[1] != 3
    ):
        raise ValueError("条件文件缺少有效的 node_type")
    fixed_ligand_nodes = int((node_type[:, 0] == 1).sum().item())

    if source == "work_data_taskb":
        if fixed_ligand_nodes:
            raise ValueError("TASKB 条件不能保留配体节点；请重新生成条件文件")
        model_variants = ["creative"]
    elif source == "work_data_taska":
        if fixed_ligand_nodes == 0:
            raise ValueError("TASKA 条件至少需要一个固定配体节点")
        model_variants = ["survival_linker", "survival_modify", "activity_modify"]
    else:
        raise ValueError("网页推理需要 PhiSSeparator 生成的 TASKA/TASKB 条件文件")

    return {
        "condition_pt": str(condition_path),
        "task_mode": "TASKB" if source == "work_data_taskb" else "TASKA",
        "fixed_ligand_nodes": fixed_ligand_nodes,
        "model_variants": model_variants,
        "default_model_variant": (
            MODEL_VARIANT if MODEL_VARIANT in model_variants
            else model_variants[0]
        ),
        "default_num_samples_per_condition": NUM_SAMPLES_PER_CONDITION,
        "creative_node_presets": node_count_inference.CREATIVE_NODE_PRESETS,
        "default_creative_node_preset": (
            node_count_inference.DEFAULT_CREATIVE_NODE_PRESET
        ),
        "taska_default_added_ligand_nodes": (
            node_count_inference.TASKA_DEFAULT_ADDED_LIGAND_NODES
        ),
        "taska_extra_generated_ligand_nodes": (
            node_count_inference.TASKA_EXTRA_GENERATED_LIGAND_NODES
        ),
    }


def _validate_reconstruction_settings() -> None:
    if (
        isinstance(MAX_ATTEMPTS_PER_SUCCESS, bool)
        or not isinstance(MAX_ATTEMPTS_PER_SUCCESS, int)
        or MAX_ATTEMPTS_PER_SUCCESS <= 0
    ):
        raise ValueError("MAX_ATTEMPTS_PER_SUCCESS 必须为正整数")
    if (
        isinstance(GPU_MEMORY_PER_PROCESS_GIB, bool)
        or not isinstance(GPU_MEMORY_PER_PROCESS_GIB, (int, float))
        or not math.isfinite(float(GPU_MEMORY_PER_PROCESS_GIB))
        or GPU_MEMORY_PER_PROCESS_GIB <= 0
    ):
        raise ValueError("GPU_MEMORY_PER_PROCESS_GIB 必须为有限正数")


def _condition_output_dir(
    output_root: Path, condition_relative_path: Path
) -> Path:
    return (
        output_root
        / condition_relative_path.parent
        / condition_relative_path.stem
    )


def _safe_remove_output_directory(path: Path, parent: Path) -> None:
    """Remove only a generated child directory beneath one condition output."""

    resolved_parent = parent.resolve()
    resolved_path = path.resolve()
    if resolved_path == resolved_parent or resolved_parent not in (
        resolved_path.parents
    ):
        raise RuntimeError(
            f"拒绝删除条件输出目录之外的路径: {resolved_path}"
        )
    if resolved_path.exists():
        shutil.rmtree(resolved_path)


def _discard_extra_success(success: dict[str, Any], output_dir: Path) -> None:
    """Remove the output of an attempt completed after the web target was met."""

    sample_index = int(success["sample_index"])
    expected_pt = (output_dir / f"sample_{sample_index:04d}.pt").resolve()
    if Path(success["output_pt"]).resolve() != expected_pt:
        raise RuntimeError(f"拒绝删除非当前尝试的结果文件: {success['output_pt']}")
    if expected_pt.is_file():
        expected_pt.unlink()
    _safe_remove_output_directory(
        output_dir / f"sample_{sample_index:04d}_3d", output_dir
    )


def _new_manifest(settings: InferenceSettings) -> dict:
    return {
        "started_utc": utc_now_iso(),
        "finished_utc": None,
        "settings": settings.as_manifest_dict(),
        "parallelism": {
            "mode": "process",
            "start_method": "spawn",
            "gpu_memory_per_process_gib": float(
                GPU_MEMORY_PER_PROCESS_GIB
            ),
            "runtime": None,
        },
        "selected_condition_files": [],
        "checkpoint": None,
        "node_count_control": {
            "mode": "manual",
            "creative_total_ligand_nodes": settings.manual_total_ligand_nodes,
            "creative_extra_generated_ligand_nodes": (
                settings.creative_extra_generated_ligand_nodes
            ),
            "taska_added_ligand_nodes": settings.taska_added_ligand_nodes,
            "taska_extra_generated_ligand_nodes": (
                settings.taska_extra_generated_ligand_nodes
            ),
            "conditions": [],
        },
        "reconstruction": {
            "enabled": bool(ENABLE_3D_RECONSTRUCTION),
            "custom_fragment_vocabulary": str(
                CUSTOM_FRAGMENT_VOCAB_PATHS[
                    settings.model_variant
                ].resolve()
            ),
            "atom_pair_checkpoint_requested": (
                str(ATOM_PAIR_CHECKPOINT_PATH.resolve())
                if ATOM_PAIR_CHECKPOINT_PATH is not None
                else None
            ),
            "atom_pair_fragment_vocabulary": str(
                ATOM_PAIR_FRAGMENT_VOCAB_PATH.resolve()
            ),
            "atom_pair_atom_vocabulary": str(
                ATOM_PAIR_ATOM_VOCAB_PATH.resolve()
            ),
            "max_attempts_per_success": int(MAX_ATTEMPTS_PER_SUCCESS),
        },
        "condition_preflights": [],
        "successes": [],
        "errors": [],
    }


def _record_error(
    manifest: dict,
    *,
    source_pt: Path | None,
    sample_index: int | None,
    attempt_index: int | None = None,
    exc: BaseException,
    stage: str | None = None,
    retryable: bool = False,
) -> None:
    error_stage = getattr(exc, "stage", None) or stage
    diagnostics = getattr(exc, "diagnostics", None)
    manifest["errors"].append(
        {
            "source_pt": str(source_pt) if source_pt is not None else None,
            "sample_index": sample_index,
            "attempt_index": attempt_index,
            "stage": error_stage,
            "retryable": bool(retryable),
            "error_type": type(exc).__name__,
            "message": str(exc),
            "diagnostics": (
                dict(diagnostics) if isinstance(diagnostics, dict) else None
            ),
            "traceback": (
                traceback.format_exc()
                if sys.exc_info()[0] is not None
                else None
            ),
        }
    )


@dataclass
class _WorkerRuntime:
    settings: InferenceSettings
    device: torch.device
    lookup: Any
    bundle: Any
    custom_vocabulary: Any
    atom_pair_runtime: Any
    condition_source: str | None = None
    condition: Any = None
    node_count_plan: Any = None


_WORKER_RUNTIME: _WorkerRuntime | None = None


def _resolve_parallel_runtime(
    device: torch.device,
    target_success_count: int,
    max_attempts: int,
    web_mode: bool = False,
) -> dict[str, Any]:
    """Resolve a safe process count for the selected logical CUDA device."""

    if target_success_count < 1 or max_attempts < 1:
        raise ValueError("并发规划要求成功目标和最大尝试次数均为正数")

    runtime: dict[str, Any] = {
        "requested_device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "target_success_count": int(target_success_count),
        "max_attempts_per_condition": int(max_attempts),
        "gpu_memory_per_process_gib": float(
            GPU_MEMORY_PER_PROCESS_GIB
        ),
    }
    if device.type != "cuda":
        runtime.update(
            {
                "device_type": device.type,
                "logical_cuda_index": None,
                "device_name": None,
                "total_memory_gib": None,
                "free_memory_gib": None,
                "memory_query_source": None,
                "process_capacity_by_memory": 1,
                "selected_processes": 1,
                "under_memory_budget": False,
            }
        )
        return runtime

    logical_index = (
        int(device.index)
        if device.index is not None
        else int(torch.cuda.current_device())
    )
    properties = torch.cuda.get_device_properties(logical_index)
    total_bytes = int(properties.total_memory)
    memory_query_source = "torch.cuda.mem_get_info"
    try:
        with torch.cuda.device(logical_index):
            free_bytes, reported_total_bytes = torch.cuda.mem_get_info()
        free_bytes = int(free_bytes)
        total_bytes = int(reported_total_bytes)
    except (AttributeError, RuntimeError):
        # Older PyTorch builds may not expose mem_get_info. Total memory still
        # provides a deterministic fallback, while the manifest records it.
        free_bytes = total_bytes
        memory_query_source = "device_total_memory_fallback"

    gib = float(1024**3)
    memory_budget_bytes = float(GPU_MEMORY_PER_PROCESS_GIB) * gib
    raw_memory_capacity = int(free_bytes // memory_budget_bytes)
    process_capacity = max(1, raw_memory_capacity)
    selected_processes = min(
        MAX_WEB_INFERENCE_PROCESSES if web_mode else target_success_count,
        max_attempts,
        process_capacity,
    )
    runtime.update(
        {
            "device_type": "cuda",
            "logical_cuda_index": logical_index,
            "device_name": properties.name,
            "total_memory_gib": total_bytes / gib,
            "free_memory_gib": free_bytes / gib,
            "memory_query_source": memory_query_source,
            "process_capacity_by_memory": raw_memory_capacity,
            "selected_processes": int(selected_processes),
            "under_memory_budget": raw_memory_capacity < 1,
        }
    )
    return runtime


def _checkpoint_manifest(bundle: Any, settings: InferenceSettings) -> dict:
    return {
        "path": str(bundle.checkpoint_path),
        "epoch": bundle.epoch,
        "model_variant": settings.model_variant,
        "training_stage": bundle.training_stage,
        "finetune_strategy": bundle.finetune_strategy,
        "source_pretrain_checkpoint": bundle.source_pretrain_checkpoint,
        "source_pretrain_epoch": bundle.source_pretrain_epoch,
        "activity_guidance": bundle.activity_guidance,
        "max_ll_neighbors": bundle.max_ll_neighbors,
        "max_lp_neighbors": bundle.max_lp_neighbors,
        "hac_cap": int(bundle.profile["hac_cap"]),
        "ring_cap": int(bundle.profile["ring_cap"]),
    }


def _serialize_worker_error(
    *,
    source_pt: Path | None,
    sample_index: int | None,
    attempt_index: int | None,
    exc: BaseException,
    stage: str | None,
    retryable: bool = False,
) -> dict[str, Any]:
    error_stage = getattr(exc, "stage", None) or stage
    diagnostics = getattr(exc, "diagnostics", None)
    return {
        "source_pt": str(source_pt) if source_pt is not None else None,
        "sample_index": sample_index,
        "attempt_index": attempt_index,
        "stage": error_stage,
        "retryable": bool(retryable),
        "error_type": type(exc).__name__,
        "message": str(exc),
        "diagnostics": (
            dict(diagnostics) if isinstance(diagnostics, dict) else None
        ),
        "traceback": traceback.format_exc(),
    }


def _create_worker_runtime(settings: InferenceSettings) -> _WorkerRuntime:
    device = resolve_device(settings.device)
    if device.type == "cuda":
        logical_index = (
            int(device.index)
            if device.index is not None
            else int(torch.cuda.current_device())
        )
        torch.cuda.set_device(logical_index)

    lookup = create_embedding_lookup(settings)
    bundle = load_checkpoint_bundle(settings, lookup, device)
    custom_vocabulary = None
    atom_pair_runtime = None
    if ENABLE_3D_RECONSTRUCTION:
        custom_vocabulary = load_reconstruction_vocabulary(
            CUSTOM_FRAGMENT_VOCAB_PATHS[settings.model_variant]
        )
        atom_pair_runtime = load_atom_pair_inference_runtime(
            checkpoint_path=ATOM_PAIR_CHECKPOINT_PATH,
            fragment_vocab_path=ATOM_PAIR_FRAGMENT_VOCAB_PATH,
            atom_vocab_path=ATOM_PAIR_ATOM_VOCAB_PATH,
            device=device,
            trusted_fragment_vocab_profile=bundle.profile[
                "fragment_embedding_table"
            ],
            trusted_fragment_vocab_path=lookup.npz_path,
        )
    return _WorkerRuntime(
        settings=settings,
        device=device,
        lookup=lookup,
        bundle=bundle,
        custom_vocabulary=custom_vocabulary,
        atom_pair_runtime=atom_pair_runtime,
    )


def _get_worker_runtime(settings: InferenceSettings) -> _WorkerRuntime:
    global _WORKER_RUNTIME

    if _WORKER_RUNTIME is None:
        _WORKER_RUNTIME = _create_worker_runtime(settings)
    elif _WORKER_RUNTIME.settings != settings:
        raise RuntimeError("同一工作进程收到了不一致的推理设置")
    return _WORKER_RUNTIME


def _worker_runtime_summary(settings: InferenceSettings) -> dict[str, Any]:
    try:
        runtime = _get_worker_runtime(settings)
        reconstruction_runtime = None
        if ENABLE_3D_RECONSTRUCTION:
            reconstruction_runtime = {
                "custom_fragment_vocabulary_resolved": str(
                    runtime.custom_vocabulary.source_path
                ),
                "atom_pair_checkpoint_resolved": str(
                    runtime.atom_pair_runtime.checkpoint_path
                ),
                "atom_pair_checkpoint_epoch": (
                    runtime.atom_pair_runtime.checkpoint_epoch
                ),
                "atom_pair_vocabulary_fingerprints": dict(
                    runtime.atom_pair_runtime.vocabulary_fingerprints
                ),
            }
        return {
            "status": "success",
            "worker_pid": os.getpid(),
            "checkpoint": _checkpoint_manifest(runtime.bundle, settings),
            "reconstruction_runtime": reconstruction_runtime,
        }
    except Exception as exc:
        return {
            "status": "error",
            "worker_pid": os.getpid(),
            "error": _serialize_worker_error(
                source_pt=None,
                sample_index=None,
                attempt_index=None,
                exc=exc,
                stage="worker_runtime_initialization",
            ),
        }


def _load_worker_condition(
    runtime: _WorkerRuntime, source_path: str
) -> tuple[Any, Any]:
    resolved_source = Path(source_path).resolve()
    source_key = str(resolved_source)
    if runtime.condition_source == source_key:
        return runtime.condition, runtime.node_count_plan

    # Only the active condition is cached so INPUT_SELECTION='all' does not
    # accumulate condition tensors on the GPU.
    runtime.condition_source = None
    runtime.condition = None
    runtime.node_count_plan = None
    condition = prepare_condition(
        resolved_source,
        runtime.settings.condition_root,
        runtime.lookup,
        runtime.settings,
        runtime.device,
    )
    plan = node_count_inference.resolve_generated_node_count(
        condition.num_fixed_ligand_nodes,
        manual_total_ligand_nodes=(
            runtime.settings.manual_total_ligand_nodes
        ),
        creative_extra_generated_ligand_nodes=(
            runtime.settings.creative_extra_generated_ligand_nodes
        ),
        taska_added_ligand_nodes=(
            runtime.settings.taska_added_ligand_nodes
        ),
        taska_extra_generated_ligand_nodes=(
            runtime.settings.taska_extra_generated_ligand_nodes
        ),
    )
    runtime.condition_source = source_key
    runtime.condition = condition
    runtime.node_count_plan = plan
    return condition, plan


def _worker_condition_summary(
    settings: InferenceSettings, source_path: str
) -> dict[str, Any]:
    resolved_source = Path(source_path).resolve()
    try:
        runtime = _get_worker_runtime(settings)
        condition, plan = _load_worker_condition(
            runtime, str(resolved_source)
        )
        preflight_report = None
        if ENABLE_3D_RECONSTRUCTION:
            if runtime.atom_pair_runtime is None:
                raise RuntimeError(
                    "原子匹配运行时尚未初始化，无法预检查固定片段"
                )
            preflight_report = preflight_fixed_condition_fragments(
                condition.source_path,
                atom_pair_runtime=runtime.atom_pair_runtime,
            )
        return {
            "status": "success",
            "worker_pid": os.getpid(),
            "source_pt": str(condition.source_path),
            "condition_relative_path": condition.relative_path.as_posix(),
            "sample_id": condition.sample_id,
            "mode": condition.mode,
            "num_fixed_ligand_nodes": condition.num_fixed_ligand_nodes,
            "node_count_plan": plan.as_manifest_dict(),
            "minimum_connected_ligand_nodes": plan.minimum_connected_ligand_nodes,
            "generated_ligand_nodes": plan.generated_ligand_nodes,
            "preflight_report": preflight_report,
        }
    except Exception as exc:
        fatal = isinstance(
            exc,
            (
                FatalConditionError,
                node_count_inference.FatalNodeCountError,
            ),
        )
        return {
            "status": "fatal_error" if fatal else "error",
            "worker_pid": os.getpid(),
            "error": _serialize_worker_error(
                source_pt=resolved_source,
                sample_index=None,
                attempt_index=None,
                exc=exc,
                stage="condition_preparation",
            ),
        }


def _cleanup_failed_reconstruction(
    reconstruction_dir: Path | None, output_dir: Path | None
) -> list[str]:
    if reconstruction_dir is None or output_dir is None:
        return []
    try:
        _safe_remove_output_directory(reconstruction_dir, output_dir)
    except Exception as cleanup_exc:
        return [
            "清理失败尝试目录时出现警告: "
            f"{type(cleanup_exc).__name__}: {cleanup_exc}"
        ]
    return []


def _worker_attempt(
    settings: InferenceSettings,
    source_path: str,
    sample_index: int,
    attempt_index: int,
    seed: int,
    expected_generated_nodes: int,
    expected_minimum_connected_nodes: int,
    web_progress_path: str | None = None,
    web_progress: dict[str, Any] | None = None,
    cancel_path: str | None = None,
    completion_stop_path: str | None = None,
) -> dict[str, Any]:
    resolved_source = Path(source_path).resolve()
    current_stage = "condition_preparation"
    output_dir: Path | None = None
    reconstruction_dir: Path | None = None

    def check_cancel() -> None:
        if completion_stop_path is not None and Path(completion_stop_path).exists():
            raise WebInferenceCancelled("已达到目标分子数量")
        if cancel_path is not None and Path(cancel_path).exists():
            raise WebInferenceCancelled("用户已终止本次生成任务")

    def report(stage: str, step: int = 0, total_steps: int = 0) -> None:
        check_cancel()
        if web_progress_path is not None and web_progress is not None:
            atomic_json_save(
                {
                    **web_progress,
                    "stage": stage,
                    "step": step,
                    "total_steps": total_steps,
                },
                Path(web_progress_path),
            )

    try:
        check_cancel()
        runtime = _get_worker_runtime(settings)
        condition, plan = _load_worker_condition(
            runtime, str(resolved_source)
        )
        if plan.generated_ligand_nodes != expected_generated_nodes:
            raise RuntimeError(
                "不同工作进程的生成节点数配置不一致: "
                f"expected={expected_generated_nodes}, "
                f"actual={plan.generated_ligand_nodes}"
            )
        if (
            plan.minimum_connected_ligand_nodes
            != expected_minimum_connected_nodes
        ):
            raise RuntimeError(
                "不同工作进程的连接分量下限配置不一致: "
                f"expected={expected_minimum_connected_nodes}, "
                f"actual={plan.minimum_connected_ligand_nodes}"
            )

        output_dir = _condition_output_dir(
            settings.output_root, condition.relative_path
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"sample_{sample_index:04d}.pt"

        current_stage = "diffusion_sampling"
        report("diffusion", total_steps=settings.ddim_steps)

        def report_step(completed: int, total: int) -> None:
            check_cancel()
            if completed == total or completed % max(1, total // 40) == 0:
                report("diffusion", completed, total)

        final_graph, final_aux = run_ddim_sampling(
            runtime.bundle,
            condition,
            settings,
            expected_generated_nodes,
            seed,
            progress_callback=(
                report_step
                if web_progress_path is not None or cancel_path is not None
                else None
            ),
        )
        current_stage = "stage1_result_build"
        report("assembling", settings.ddim_steps, settings.ddim_steps)
        result = build_stage1_result(
            runtime.bundle,
            condition,
            settings,
            sample_index,
            attempt_index,
            seed,
            expected_generated_nodes,
            final_graph,
            final_aux,
            minimum_connected_component_ligand_nodes=(
                expected_minimum_connected_nodes
            ),
        )

        reconstruction_outputs = None
        complex_optimization = None
        ligand_component_selection = None
        if ENABLE_3D_RECONSTRUCTION:
            if (
                runtime.custom_vocabulary is None
                or runtime.atom_pair_runtime is None
            ):
                raise RuntimeError(
                    "3D reconstruction dependencies were not initialized"
                )
            current_stage = "three_dimensional_reconstruction"
            report("reconstructing", settings.ddim_steps, settings.ddim_steps)
            reconstruction_dir = (
                output_dir / f"sample_{sample_index:04d}_3d"
            )
            if reconstruction_dir.exists():
                _safe_remove_output_directory(
                    reconstruction_dir, output_dir
                )
            receptor_path = (
                condition.source_path.parent
                / "full_receptor_normalized.pdb"
            )
            reconstruction_report = reconstruct_stage1_result(
                result,
                condition_pt_path=condition.source_path,
                receptor_pdb_path=receptor_path,
                output_dir=reconstruction_dir,
                custom_vocabulary=runtime.custom_vocabulary,
                atom_pair_runtime=runtime.atom_pair_runtime,
                fixed_preflight_completed=True,
            )
            reconstruction_outputs = dict(
                reconstruction_report["outputs"]
            )
            complex_optimization = dict(
                reconstruction_report["complex_optimization"]
            )
            ligand_component_selection = dict(
                reconstruction_report["ligand_component_selection"]
            )

        current_stage = "stage1_result_save"
        report("saving", settings.ddim_steps, settings.ddim_steps)
        atomic_torch_save(result, output_path)
        return {
            "status": "success",
            "worker_pid": os.getpid(),
            "success": {
                "source_pt": str(condition.source_path),
                "condition_relative_path": (
                    condition.relative_path.as_posix()
                ),
                "sample_index": sample_index,
                "attempt_index": attempt_index,
                "seed": seed,
                "mode": condition.mode,
                "num_fixed_ligand_nodes": (
                    condition.num_fixed_ligand_nodes
                ),
                "num_generated_ligand_nodes": expected_generated_nodes,
                "minimum_connected_component_ligand_nodes": (
                    expected_minimum_connected_nodes
                ),
                "output_pt": str(output_path.resolve()),
                "reconstruction_outputs": reconstruction_outputs,
                "complex_optimization": complex_optimization,
                "ligand_component_selection": ligand_component_selection,
            },
            "warnings": [],
        }
    except WebInferenceCancelled as exc:
        return {
            "status": "cancelled",
            "worker_pid": os.getpid(),
            "message": str(exc),
            "warnings": _cleanup_failed_reconstruction(
                reconstruction_dir, output_dir
            ),
        }
    except RetryableReconstructionError as exc:
        return {
            "status": "retryable_error",
            "worker_pid": os.getpid(),
            "error": _serialize_worker_error(
                source_pt=resolved_source,
                sample_index=sample_index,
                attempt_index=attempt_index,
                exc=exc,
                stage=current_stage,
                retryable=True,
            ),
            "warnings": _cleanup_failed_reconstruction(
                reconstruction_dir, output_dir
            ),
        }
    except (
        FatalConditionError,
        node_count_inference.FatalNodeCountError,
    ) as exc:
        return {
            "status": "fatal_error",
            "worker_pid": os.getpid(),
            "error": _serialize_worker_error(
                source_pt=resolved_source,
                sample_index=sample_index,
                attempt_index=attempt_index,
                exc=exc,
                stage=current_stage,
            ),
            "warnings": _cleanup_failed_reconstruction(
                reconstruction_dir, output_dir
            ),
        }
    except Exception as exc:
        return {
            "status": "error",
            "worker_pid": os.getpid(),
            "error": _serialize_worker_error(
                source_pt=resolved_source,
                sample_index=sample_index,
                attempt_index=attempt_index,
                exc=exc,
                stage=current_stage,
            ),
            "warnings": _cleanup_failed_reconstruction(
                reconstruction_dir, output_dir
            ),
        }


def main(
    settings: InferenceSettings | None = None,
    condition_file: str | Path | None = None,
    web_progress_path: str | Path | None = None,
    cancel_path: str | Path | None = None,
) -> dict[str, Any]:
    settings = build_settings() if settings is None else settings
    device = resolve_device(settings.device)
    manifest = _new_manifest(settings)
    settings.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = settings.output_root / "manifest.json"
    cancel_path = Path(cancel_path).resolve() if cancel_path is not None else None

    def cancellation_requested() -> bool:
        return cancel_path is not None and cancel_path.exists()

    if condition_file is None:
        condition_files = discover_condition_files(
            settings.condition_root, settings.input_selection
        )
    else:
        selected_condition = Path(condition_file).resolve()
        if not selected_condition.is_file():
            raise FileNotFoundError(f"条件文件不存在: {selected_condition}")
        condition_files = [selected_condition]
    manifest["selected_condition_files"] = [
        str(path) for path in condition_files
    ]
    condition_order = {
        str(path): index for index, path in enumerate(condition_files)
    }
    remaining_seeds = list(
        range(settings.random_seed_min, settings.random_seed_max + 1)
    )
    secrets.SystemRandom().shuffle(remaining_seeds)

    target_success_count = settings.num_samples_per_condition
    max_attempts = target_success_count * MAX_ATTEMPTS_PER_SUCCESS
    if web_progress_path is not None:
        web_progress_path = Path(web_progress_path).resolve()
    web_progress = {
        "stage": "initializing",
        "attempts_started": 0,
        "attempts_completed": 0,
        "success_count": 0,
        "target_success_count": target_success_count,
        "max_attempts": max_attempts,
        "worker_count": 0,
        "attempts": [],
        "last_attempt": None,
    }

    def save_web_progress(**changes) -> None:
        if web_progress_path is not None:
            web_progress.update(changes)
            atomic_json_save(web_progress, web_progress_path)

    save_web_progress()
    if cancellation_requested():
        manifest["cancelled"] = True
        manifest["finished_utc"] = utc_now_iso()
        atomic_json_save(manifest, manifest_path)
        save_web_progress(stage="cancelled")
        return manifest
    parallel_runtime = _resolve_parallel_runtime(
        device, target_success_count, max_attempts,
        web_mode=web_progress_path is not None,
    )
    worker_count = int(parallel_runtime["selected_processes"])
    save_web_progress(worker_count=worker_count)
    worker_progress_interval = (
        0 if worker_count > 1 else settings.progress_interval
    )
    worker_settings = replace(
        settings, progress_interval=worker_progress_interval
    )
    parallel_runtime["effective_worker_progress_interval"] = (
        worker_progress_interval
    )
    manifest["parallelism"]["runtime"] = parallel_runtime
    atomic_json_save(manifest, manifest_path)

    if device.type == "cuda":
        print(
            "并发配置: "
            f"CUDA_VISIBLE_DEVICES="
            f"{parallel_runtime['cuda_visible_devices']!r}, "
            f"device={device}, logical_cuda_index="
            f"{parallel_runtime['logical_cuda_index']}, "
            f"name={parallel_runtime['device_name']}, "
            f"total={parallel_runtime['total_memory_gib']:.2f} GiB, "
            f"free={parallel_runtime['free_memory_gib']:.2f} GiB, "
            f"budget/process={GPU_MEMORY_PER_PROCESS_GIB:.2f} GiB, "
            f"processes={worker_count}"
        )
        if parallel_runtime["under_memory_budget"]:
            print(
                "并发配置警告: 当前空闲显存低于单进程预算，"
                "为保持原有单进程能力仍尝试启动 1 个进程。"
            )
    else:
        print(f"并发配置: device={device}, processes=1")

    process_context = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=process_context,
    ) as executor:
        print("[1/2] 在首个工作进程中初始化推理运行时...")
        try:
            runtime_summary = executor.submit(
                _worker_runtime_summary, worker_settings
            ).result()
        except Exception as exc:
            _record_error(
                manifest,
                source_pt=None,
                sample_index=None,
                exc=exc,
                stage="worker_runtime_transport",
            )
            atomic_json_save(manifest, manifest_path)
            raise
        if runtime_summary["status"] != "success":
            manifest["errors"].append(runtime_summary["error"])
            atomic_json_save(manifest, manifest_path)
            error = runtime_summary["error"]
            raise RuntimeError(
                "工作进程运行时初始化失败: "
                f"{error['error_type']}: {error['message']}"
            )

        manifest["checkpoint"] = runtime_summary["checkpoint"]
        if runtime_summary["reconstruction_runtime"] is not None:
            manifest["reconstruction"].update(
                runtime_summary["reconstruction_runtime"]
            )
        atomic_json_save(manifest, manifest_path)
        print(
            "[2/2] 首个工作进程初始化完成: "
            f"pid={runtime_summary['worker_pid']}；开始处理 "
            f"{len(condition_files)} 个条件文件，每个生成 "
            f"{target_success_count} 份结果，每个条件累计尝试上限 "
            f"{target_success_count}×{MAX_ATTEMPTS_PER_SUCCESS}="
            f"{max_attempts}。"
        )

        for condition_index, source_path in enumerate(condition_files):
            if cancellation_requested():
                break
            if not remaining_seeds:
                seed_error = RuntimeError(
                    "本次运行的随机种子已耗尽，停止处理后续条件；"
                    "已完成结果保留。"
                )
                _record_error(
                    manifest,
                    source_pt=source_path,
                    sample_index=None,
                    exc=seed_error,
                    stage="random_seed_exhausted",
                )
                atomic_json_save(manifest, manifest_path)
                print(f"    {seed_error}")
                break
            try:
                condition_summary = executor.submit(
                    _worker_condition_summary,
                    worker_settings,
                    str(source_path),
                ).result()
            except Exception as exc:
                _record_error(
                    manifest,
                    source_pt=source_path,
                    sample_index=None,
                    exc=exc,
                    stage="condition_worker_transport",
                )
                atomic_json_save(manifest, manifest_path)
                raise

            if cancellation_requested():
                break
            if condition_summary["status"] != "success":
                error = condition_summary["error"]
                manifest["errors"].append(error)
                atomic_json_save(manifest, manifest_path)
                print(
                    "    条件初始化失败: "
                    f"{error['error_type']}: {error['message']}"
                )
                if (
                    condition_summary["status"] == "fatal_error"
                    or not settings.continue_on_error
                ):
                    raise RuntimeError(
                        "条件初始化失败: "
                        f"{error['error_type']}: {error['message']}"
                    )
                continue

            plan_record = {
                "source_pt": condition_summary["source_pt"],
                "condition_relative_path": condition_summary[
                    "condition_relative_path"
                ],
                "sample_id": condition_summary["sample_id"],
                **condition_summary["node_count_plan"],
            }
            manifest["node_count_control"]["conditions"].append(
                plan_record
            )
            if condition_summary["preflight_report"] is not None:
                manifest["condition_preflights"].append(
                    condition_summary["preflight_report"]
                )
            atomic_json_save(manifest, manifest_path)

            num_generated = int(
                condition_summary["generated_ligand_nodes"]
            )
            minimum_connected_nodes = int(
                condition_summary["minimum_connected_ligand_nodes"]
            )
            print(
                f"  [{condition_index + 1}/{len(condition_files)}] "
                f"{condition_summary['condition_relative_path']} | "
                f"mode={condition_summary['mode']}, fixed_L="
                f"{condition_summary['num_fixed_ligand_nodes']}, "
                f"node_count_source=manual, "
                f"resolved_total={minimum_connected_nodes}, "
                f"generated_L={num_generated}, "
                f"prepare_pid={condition_summary['worker_pid']}"
            )

            output_dir = _condition_output_dir(
                settings.output_root,
                Path(condition_summary["condition_relative_path"]),
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            completion_stop_path = (
                output_dir / "target_reached.stop"
                if web_progress_path is not None else None
            )
            pending: dict[Future, dict[str, int]] = {}
            success_count = 0
            attempt_index = 0
            next_sample_index = 0
            completion_reached = False
            condition_aborted = False
            abort_error: dict[str, Any] | None = None
            abort_requires_raise = False

            def finish_web_attempt(
                task: dict[str, int], status: str, message: str
            ) -> None:
                if web_progress_path is None:
                    return
                attempt = web_progress["attempts"][task["attempt_index"]]
                attempt.update(
                    status=status,
                    stage="attempt_complete",
                    step=settings.ddim_steps,
                    message=message,
                )
                save_web_progress(
                    attempts_completed=web_progress["attempts_completed"] + 1,
                    success_count=success_count,
                    last_attempt={
                        "number": attempt["number"],
                        "status": status,
                        "message": message,
                    },
                )

            def submit_attempt(sample_index: int) -> bool:
                nonlocal attempt_index
                nonlocal abort_error, abort_requires_raise
                nonlocal condition_aborted
                if (
                    condition_aborted
                    or completion_reached
                    or attempt_index >= max_attempts
                    or not remaining_seeds or cancellation_requested()
                ):
                    return False
                current_attempt_index = attempt_index
                seed = remaining_seeds.pop()
                attempt_progress_path = (
                    web_progress_path.parent
                    / f"attempt_{current_attempt_index:04d}.json"
                    if web_progress_path is not None else None
                )
                attempt_progress = {
                    "number": current_attempt_index + 1,
                    "sample_index": sample_index,
                    "seed": seed,
                    "status": "running",
                    "stage": "preparing",
                    "step": 0,
                    "total_steps": settings.ddim_steps,
                }
                try:
                    if attempt_progress_path is not None:
                        atomic_json_save(attempt_progress, attempt_progress_path)
                    future = executor.submit(
                        _worker_attempt,
                        worker_settings,
                        str(source_path),
                        sample_index,
                        current_attempt_index,
                        seed,
                        num_generated,
                        minimum_connected_nodes,
                        (
                            str(attempt_progress_path)
                            if attempt_progress_path is not None else None
                        ),
                        (
                            attempt_progress
                            if attempt_progress_path is not None else None
                        ),
                        str(cancel_path) if cancel_path is not None else None,
                        (
                            str(completion_stop_path)
                            if completion_stop_path is not None else None
                        ),
                    )
                except Exception as exc:
                    _record_error(
                        manifest,
                        source_pt=source_path,
                        sample_index=sample_index,
                        attempt_index=current_attempt_index,
                        exc=exc,
                        stage="worker_submission",
                    )
                    condition_aborted = True
                    abort_requires_raise = True
                    if abort_error is None:
                        abort_error = manifest["errors"][-1]
                    cancel_queued_attempts()
                    atomic_json_save(manifest, manifest_path)
                    save_web_progress(
                        last_attempt={
                            "number": current_attempt_index + 1,
                            "status": "failed",
                            "message": str(exc),
                        },
                    )
                    return False
                attempt_index += 1
                pending[future] = {
                    "sample_index": sample_index,
                    "attempt_index": current_attempt_index,
                    "seed": seed,
                }
                if web_progress_path is not None:
                    web_progress["attempts"].append(attempt_progress)
                    save_web_progress(
                        stage="running",
                        attempts_started=current_attempt_index + 1,
                    )
                print(
                    f"    已提交 sample={sample_index:04d}, "
                    f"attempt={current_attempt_index:04d}, seed={seed}, "
                    f"device={device}"
                )
                return True

            def cancel_queued_attempts() -> None:
                for queued_future in list(pending):
                    if queued_future.cancel():
                        task = pending.pop(queued_future)
                        finish_web_attempt(
                            task, "cancelled", "尚未开始的推理已取消"
                        )

            if web_progress_path is not None:
                while len(pending) < worker_count:
                    if not submit_attempt(next_sample_index):
                        break
                    next_sample_index += 1
            else:
                for sample_index in range(target_success_count):
                    submit_attempt(sample_index)

            while pending:
                completed, _ = wait(
                    tuple(pending), return_when=FIRST_COMPLETED
                )
                ordered_completed = sorted(
                    completed,
                    key=lambda future: pending[future]["attempt_index"],
                )
                retry_sample_indices: list[int] = []
                for future in ordered_completed:
                    task = pending.pop(future)
                    try:
                        outcome = future.result()
                    except Exception as exc:
                        if completion_reached:
                            finish_web_attempt(
                                task, "cancelled", "已达到目标分子数量"
                            )
                            continue
                        _record_error(
                            manifest,
                            source_pt=source_path,
                            sample_index=task["sample_index"],
                            attempt_index=task["attempt_index"],
                            exc=exc,
                            stage="worker_process_transport",
                        )
                        error = manifest["errors"][-1]
                        print(
                            "      工作进程通信失败: "
                            f"{error['error_type']}: {error['message']}"
                        )
                        condition_aborted = True
                        abort_requires_raise = True
                        if abort_error is None:
                            abort_error = error
                        cancel_queued_attempts()
                        atomic_json_save(manifest, manifest_path)
                        finish_web_attempt(task, "failed", str(exc))
                        continue

                    if completion_reached:
                        if outcome.get("status") == "success":
                            try:
                                _discard_extra_success(
                                    outcome["success"], output_dir
                                )
                            except Exception as cleanup_exc:
                                print(f"      清理超额结果失败: {cleanup_exc}")
                        finish_web_attempt(
                            task, "cancelled", "已达到目标分子数量"
                        )
                        continue

                    for warning in outcome.get("warnings", []):
                        print(f"      {warning}")
                    status = outcome.get("status")
                    if status == "success":
                        success = outcome["success"]
                        manifest["successes"].append(success)
                        manifest["successes"].sort(
                            key=lambda record: (
                                condition_order.get(
                                    record["source_pt"],
                                    len(condition_order),
                                ),
                                int(record["sample_index"]),
                            )
                        )
                        success_count += 1
                        if (
                            web_progress_path is not None
                            and success_count >= target_success_count
                        ):
                            completion_reached = True
                            completion_stop_path.touch()
                            cancel_queued_attempts()
                        print(
                            "      已完成: "
                            f"sample={success['sample_index']:04d}, "
                            f"attempt={success['attempt_index']:04d}, "
                            f"pid={outcome['worker_pid']}, "
                            f"output={success['output_pt']}"
                        )
                        if success["reconstruction_outputs"] is not None:
                            print(
                                "      三维重建完成: "
                                f"{success['reconstruction_outputs']}"
                            )
                    elif status == "retryable_error":
                        error = outcome["error"]
                        manifest["errors"].append(error)
                        print(
                            "      本次尝试失败，将使用新 seed 重新推理: "
                            f"sample={task['sample_index']:04d}, "
                            f"attempt={task['attempt_index']:04d}, "
                            f"pid={outcome['worker_pid']}, "
                            f"{error['error_type']}: {error['message']}"
                        )
                        retry_sample_indices.append(task["sample_index"])
                    elif status == "cancelled":
                        condition_aborted = True
                        cancel_queued_attempts()
                    elif status in {"fatal_error", "error"}:
                        error = outcome["error"]
                        manifest["errors"].append(error)
                        print(
                            "      条件不可继续: "
                            f"sample={task['sample_index']:04d}, "
                            f"attempt={task['attempt_index']:04d}, "
                            f"pid={outcome['worker_pid']}, "
                            f"{error['error_type']}: {error['message']}"
                        )
                        condition_aborted = True
                        if abort_error is None:
                            abort_error = error
                        if (
                            status == "fatal_error"
                            or not settings.continue_on_error
                        ):
                            abort_requires_raise = True
                        cancel_queued_attempts()
                    else:
                        protocol_error = RuntimeError(
                            "工作进程返回未知状态: " f"{status!r}"
                        )
                        _record_error(
                            manifest,
                            source_pt=source_path,
                            sample_index=task["sample_index"],
                            attempt_index=task["attempt_index"],
                            exc=protocol_error,
                            stage="worker_protocol",
                        )
                        condition_aborted = True
                        abort_requires_raise = True
                        if abort_error is None:
                            abort_error = manifest["errors"][-1]
                        cancel_queued_attempts()
                    atomic_json_save(manifest, manifest_path)
                    completion_status = (
                        "success" if status == "success" else
                        "cancelled" if status == "cancelled" else "failed"
                    )
                    if status == "success":
                        completion_message = "候选分子生成成功"
                    elif status == "cancelled":
                        completion_message = outcome.get(
                            "message", "用户已终止本次生成"
                        )
                    else:
                        completion_message = str(
                            (outcome.get("error") or {}).get(
                                "message", "本次推理失败"
                            )
                        )
                    finish_web_attempt(
                        task, completion_status, completion_message
                    )
                if (
                    not condition_aborted
                    and not completion_reached
                    and not cancellation_requested()
                ):
                    if web_progress_path is None:
                        for sample_index in retry_sample_indices:
                            submit_attempt(sample_index)
                    else:
                        while len(pending) < worker_count:
                            if not submit_attempt(next_sample_index):
                                break
                            next_sample_index += 1

            if cancellation_requested():
                break
            if condition_aborted:
                if abort_requires_raise:
                    if abort_error is None:
                        raise RuntimeError("并发推理因未知工作进程错误终止")
                    raise RuntimeError(
                        "并发推理终止: "
                        f"{abort_error['error_type']}: "
                        f"{abort_error['message']}"
                    )
                continue

            if success_count < target_success_count:
                if not remaining_seeds:
                    limit_error = RuntimeError(
                        "本次运行的随机种子已耗尽，仍未生成足够的成功配体；"
                        "已提交任务均已结束，已完成结果保留: "
                        f"success={success_count}, target={target_success_count}, "
                        f"attempts={attempt_index}"
                    )
                    limit_stage = "random_seed_exhausted"
                else:
                    limit_error = RuntimeError(
                        "达到条件最大尝试次数仍未生成足够的成功配体: "
                        f"success={success_count}, target={target_success_count}, "
                        f"attempts={attempt_index}, limit={max_attempts}"
                    )
                    limit_stage = "retry_limit_exhausted"
                _record_error(
                    manifest,
                    source_pt=source_path,
                    sample_index=success_count,
                    attempt_index=attempt_index,
                    exc=limit_error,
                    stage=limit_stage,
                )
                atomic_json_save(manifest, manifest_path)
                print(f"    {limit_error}")
                if not remaining_seeds:
                    break
                if not settings.continue_on_error and web_progress_path is None:
                    raise limit_error

    manifest["cancelled"] = cancellation_requested()
    manifest["finished_utc"] = utc_now_iso()
    atomic_json_save(manifest, manifest_path)
    save_web_progress(
        stage="cancelled" if manifest["cancelled"] else "finished",
        success_count=len(manifest["successes"]),
    )
    retry_count = sum(
        bool(record.get("retryable")) for record in manifest["errors"]
    )
    terminal_error_count = len(manifest["errors"]) - retry_count
    print(
        "推理完成: "
        f"成功 {len(manifest['successes'])}，重试 {retry_count}，"
        f"终止性错误 {terminal_error_count}。"
    )
    print(f"运行清单: {manifest_path.resolve()}")
    return manifest


def run_web_inference(
    condition_pt: str | Path,
    model_variant: str,
    num_samples_per_condition: int,
    output_root: str | Path,
    progress_path: str | Path | None = None,
    cancel_path: str | Path | None = None,
    manual_total_ligand_nodes: int | None = None,
    creative_extra_generated_ligand_nodes: int | None = None,
    taska_added_ligand_nodes: int | None = None,
    taska_extra_generated_ligand_nodes: int | None = None,
) -> dict[str, Any]:
    """网页后台入口：单个条件文件、所选模式和生成数量。"""
    options = get_web_inference_options(condition_pt)
    if model_variant not in options["model_variants"]:
        raise ValueError(
            f"当前条件不可使用 {model_variant!r}；"
            f"可选模式为 {options['model_variants']}"
        )
    if (
        isinstance(num_samples_per_condition, bool)
        or not isinstance(num_samples_per_condition, int)
        or num_samples_per_condition < 1
    ):
        raise ValueError("NUM_SAMPLES_PER_CONDITION 必须为正整数")
    settings = build_settings(
        model_variant=model_variant,
        num_samples_per_condition=num_samples_per_condition,
        output_root=Path(output_root).resolve(),
        manual_total_ligand_nodes=manual_total_ligand_nodes,
        creative_extra_generated_ligand_nodes=(
            creative_extra_generated_ligand_nodes
        ),
        taska_added_ligand_nodes=taska_added_ligand_nodes,
        taska_extra_generated_ligand_nodes=(
            taska_extra_generated_ligand_nodes
        ),
    )
    return main(
        settings=settings,
        condition_file=options["condition_pt"],
        web_progress_path=progress_path,
        cancel_path=cancel_path,
    )


if __name__ == "__main__":
    main()
