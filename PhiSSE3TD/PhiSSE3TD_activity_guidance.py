from __future__ import annotations

import hashlib
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch_geometric.data import Data


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PhiSGATv2.PhiSGATv2_model import GATv2Model  # noqa: E402
from PhiSGATv2.PhiSGATv2_utils import load_checkpoint_payload  # noqa: E402


@dataclass
class ActivityGuidanceRuntime:
    model: GATv2Model
    checkpoint_path: Path
    checkpoint_sha256: str
    checkpoint_epoch: int
    model_config: dict[str, Any]

    def checkpoint_metadata(self) -> dict[str, Any]:
        return {
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint_epoch": self.checkpoint_epoch,
            "model_config": dict(self.model_config),
        }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_activity_guidance(
    checkpoint_path: Path,
    device: torch.device,
) -> ActivityGuidanceRuntime:
    resolved_path = checkpoint_path.resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(
            f"PhiSGATv2 活性指导 checkpoint 不存在: {resolved_path}"
        )

    checkpoint_sha256 = _file_sha256(resolved_path)
    payload = load_checkpoint_payload(resolved_path, map_location="cpu")
    model = GATv2Model()
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    return ActivityGuidanceRuntime(
        model=model,
        checkpoint_path=resolved_path,
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_epoch=int(payload["epoch"]),
        model_config=dict(payload["model_config"]),
    )


def _canonical_bidirectional_edges(
    edge_index: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    if edge_index.numel() == 0:
        return torch.empty(
            (2, 0), dtype=torch.long, device=edge_index.device
        )

    low = torch.minimum(edge_index[0], edge_index[1])
    high = torch.maximum(edge_index[0], edge_index[1])
    non_self = low != high
    if not bool(non_self.any()):
        return torch.empty(
            (2, 0), dtype=torch.long, device=edge_index.device
        )

    pair_ids = torch.unique(
        low[non_self] * num_nodes + high[non_self], sorted=True
    )
    first = torch.div(pair_ids, num_nodes, rounding_mode="floor")
    second = pair_ids.remainder(num_nodes)
    undirected = torch.stack([first, second], dim=0)
    return torch.cat([undirected, undirected.flip(0)], dim=1)


def build_activity_edge_index(
    *,
    candidate_edge_index: torch.Tensor,
    covalent_logits: torch.Tensor,
    gt_covalent_edge_index: torch.Tensor,
    is_ligand: torch.Tensor,
    is_fixed_ligand: torch.Tensor,
    ligand_batch: torch.Tensor,
    probability_threshold: float,
) -> torch.Tensor:
    """
    固定—固定节点使用真实共价边，其余节点对使用 detach 后的预测硬边。
    返回以 ligand 局部编号表示、去重后的双向 GAT 边。
    """
    if candidate_edge_index.ndim != 2 or candidate_edge_index.shape[0] != 2:
        raise ValueError("covalent_candidate_edge_index 必须为 [2,E]")
    if covalent_logits.ndim != 1 or (
        covalent_logits.shape[0] != candidate_edge_index.shape[1]
    ):
        raise ValueError("covalent_logits 必须与候选共价节点对一一对应")
    if gt_covalent_edge_index.ndim != 2 or (
        gt_covalent_edge_index.shape[0] != 2
    ):
        raise ValueError("gt_covalent_edge_index 必须为 [2,E]")
    if not 0.0 <= float(probability_threshold) <= 1.0:
        raise ValueError("活性指导共价概率阈值必须位于 [0,1]")

    ligand_indices = torch.nonzero(
        is_ligand.bool(), as_tuple=False
    ).flatten()
    num_ligand = int(ligand_indices.numel())
    if tuple(is_fixed_ligand.shape) != (num_ligand,):
        raise ValueError("is_fixed_ligand 必须与 ligand 节点数量一致")
    if tuple(ligand_batch.shape) != (num_ligand,):
        raise ValueError("ligand_batch 必须与 ligand 节点数量一致")

    num_full_nodes = int(is_ligand.numel())
    global_to_ligand = torch.full(
        (num_full_nodes,),
        -1,
        dtype=torch.long,
        device=is_ligand.device,
    )
    global_to_ligand[ligand_indices] = torch.arange(
        num_ligand, dtype=torch.long, device=is_ligand.device
    )
    fixed_global = torch.zeros(
        num_full_nodes, dtype=torch.bool, device=is_ligand.device
    )
    fixed_global[ligand_indices] = is_fixed_ligand.bool()

    candidate_src, candidate_dst = candidate_edge_index.long()
    candidate_fixed_fixed = (
        fixed_global[candidate_src] & fixed_global[candidate_dst]
    )
    predicted_positive = (
        torch.sigmoid(covalent_logits.detach().float())
        >= float(probability_threshold)
    )
    predicted_selected = candidate_edge_index[
        :, predicted_positive & ~candidate_fixed_fixed
    ].long()

    gt_src, gt_dst = gt_covalent_edge_index.long()
    gt_fixed_fixed = fixed_global[gt_src] & fixed_global[gt_dst]
    fixed_selected = gt_covalent_edge_index[:, gt_fixed_fixed].long()
    selected_global = torch.cat(
        [fixed_selected, predicted_selected], dim=1
    )
    if selected_global.numel() == 0:
        return torch.empty(
            (2, 0), dtype=torch.long, device=is_ligand.device
        )

    selected_local = global_to_ligand[selected_global]
    if bool((selected_local < 0).any()):
        raise ValueError("活性指导共价边包含非 ligand 节点")
    directed = _canonical_bidirectional_edges(selected_local, num_ligand)
    if directed.numel() and not torch.equal(
        ligand_batch[directed[0]], ligand_batch[directed[1]]
    ):
        raise ValueError("活性指导共价边跨越了不同图")
    return directed


def score_activity(
    *,
    runtime: ActivityGuidanceRuntime,
    x0_pred_feat: torch.Tensor,
    ligand_batch: torch.Tensor,
    candidate_edge_index: torch.Tensor,
    covalent_logits: torch.Tensor,
    gt_covalent_edge_index: torch.Tensor,
    is_ligand: torch.Tensor,
    is_fixed_ligand: torch.Tensor,
    num_graphs: int,
    embedding_dim: int,
    numeric_dim: int,
    probability_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    expected_dim = int(embedding_dim) + int(numeric_dim)
    if int(num_graphs) < 1:
        raise ValueError("num_graphs 必须为正整数")
    if x0_pred_feat.ndim != 2 or x0_pred_feat.shape[1] < expected_dim:
        raise ValueError(
            "x0_pred_feat 无法提供 GATv2 所需特征: "
            f"实际={tuple(x0_pred_feat.shape)}, 至少需要={expected_dim}"
        )

    edge_index = build_activity_edge_index(
        candidate_edge_index=candidate_edge_index,
        covalent_logits=covalent_logits,
        gt_covalent_edge_index=gt_covalent_edge_index,
        is_ligand=is_ligand,
        is_fixed_ligand=is_fixed_ligand,
        ligand_batch=ligand_batch,
        probability_threshold=probability_threshold,
    )
    graph = Data(
        x=x0_pred_feat[
            :, embedding_dim : embedding_dim + numeric_dim
        ].float(),
        frag_embeds=x0_pred_feat[:, :embedding_dim].float(),
        edge_index=edge_index,
    )
    graph.batch = ligand_batch.long()

    observed_graphs = torch.unique(ligand_batch.long(), sorted=True)
    expected_graph_ids = torch.arange(
        int(num_graphs), device=ligand_batch.device, dtype=torch.long
    )
    if not torch.equal(observed_graphs, expected_graph_ids):
        raise ValueError(
            "每个批次图都必须至少包含一个 ligand 节点，且图编号必须连续"
        )

    runtime.model.eval()
    with torch.amp.autocast(
        device_type=x0_pred_feat.device.type, enabled=False
    ):
        scores = runtime.model(graph)
    expected_graphs = int(num_graphs)
    if tuple(scores.shape) != (expected_graphs, 1):
        raise RuntimeError(
            "PhiSGATv2 活性分数形状异常: "
            f"{tuple(scores.shape)} != ({expected_graphs},1)"
        )
    if not bool(torch.isfinite(scores).all()):
        raise RuntimeError("PhiSGATv2 活性分数包含 NaN 或 Inf")
    return scores, edge_index


def build_activity_guidance_metadata(
    runtime: ActivityGuidanceRuntime,
    *,
    target: float,
    smooth_l1_beta: float,
    loss_weight: float,
    numeric_dim: int,
    probability_threshold: float,
    low_noise_aux_start_ratio: float,
    high_noise_aux_weight_ratio: float,
) -> dict[str, Any]:
    target = float(target)
    smooth_l1_beta = float(smooth_l1_beta)
    try:
        output_min = float(runtime.model_config["output_min"])
        output_max = float(runtime.model_config["output_max"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "PhiSGATv2 checkpoint 缺少有效的活性输出边界"
        ) from exc
    if not (
        math.isfinite(target)
        and math.isfinite(output_min)
        and math.isfinite(output_max)
        and output_min <= target <= output_max
    ):
        raise ValueError(
            "活性指导目标必须是 PhiSGATv2 软输出边界内的有限数值: "
            f"target={target}, output=({output_min}, {output_max})"
        )
    if not math.isfinite(smooth_l1_beta) or smooth_l1_beta <= 0.0:
        raise ValueError("活性指导 SmoothL1 beta 必须是大于 0 的有限数值")

    payload = runtime.checkpoint_metadata()
    payload.update(
        {
            "target": target,
            "activity_smooth_l1_beta": smooth_l1_beta,
            "loss_weight": float(loss_weight),
            "numeric_dim": int(numeric_dim),
            "probability_threshold": float(probability_threshold),
            "low_noise_aux_start_ratio": float(low_noise_aux_start_ratio),
            "high_noise_aux_weight_ratio": float(
                high_noise_aux_weight_ratio
            ),
            "edge_policy": (
                "gt_fixed_fixed_plus_predicted_non_fixed_bidirectional"
            ),
            "covalent_gradient_from_activity": False,
        }
    )
    return payload


def validate_activity_guidance_metadata(
    stored: Mapping[str, Any], expected: Mapping[str, Any]
) -> None:
    if dict(stored) != dict(expected):
        raise ValueError(
            "微调断点的活性指导配置或 PhiSGATv2 checkpoint 与当前运行不一致"
        )


__all__ = [
    "ActivityGuidanceRuntime",
    "build_activity_edge_index",
    "build_activity_guidance_metadata",
    "load_activity_guidance",
    "score_activity",
    "validate_activity_guidance_metadata",
]
