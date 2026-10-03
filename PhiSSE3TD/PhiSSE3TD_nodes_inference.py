"""手动节点规模配置与单个生成任务的节点数计算。"""

from __future__ import annotations

from dataclasses import asdict, dataclass


# TASKB 三个档位：基础节点数决定最终分子的最低节点数；额外节点只增加生成量。
# 修改这里的数值后，网页展示和后台推理会同时更新。
CREATIVE_NODE_PRESETS = {
    "small": {
        "title": "节点较少",
        "description": "分子量可能较小，生成速度较快",
        "base_nodes": 3,
        "extra_nodes": 0,
    },
    "medium": {
        "title": "节点适中",
        "description": "分子量适中，生成速度适中",
        "base_nodes": 4,
        "extra_nodes": 1,
    },
    "large": {
        "title": "节点较多",
        "description": "分子量可能较大，生成速度较慢",
        "base_nodes": 5,
        "extra_nodes": 2,
    },
}
DEFAULT_CREATIVE_NODE_PRESET = "medium"

# TASKA：用户填写新增的基础节点数；额外节点由后台统一配置。
TASKA_DEFAULT_ADDED_LIGAND_NODES = 2
TASKA_EXTRA_GENERATED_LIGAND_NODES = 0


class FatalNodeCountError(RuntimeError):
    """节点规模配置无效，当前生成任务不能继续。"""


@dataclass(frozen=True)
class NodeCountPlan:
    mode: str
    fixed_ligand_nodes: int
    base_generated_ligand_nodes: int
    extra_generated_ligand_nodes: int
    generated_ligand_nodes: int
    minimum_connected_ligand_nodes: int

    def as_manifest_dict(self) -> dict:
        return asdict(self)


def resolve_generated_node_count(
    fixed_ligand_nodes: int,
    *,
    manual_total_ligand_nodes: int = CREATIVE_NODE_PRESETS[DEFAULT_CREATIVE_NODE_PRESET]["base_nodes"],
    creative_extra_generated_ligand_nodes: int = CREATIVE_NODE_PRESETS[DEFAULT_CREATIVE_NODE_PRESET]["extra_nodes"],
    taska_added_ligand_nodes: int = TASKA_DEFAULT_ADDED_LIGAND_NODES,
    taska_extra_generated_ligand_nodes: int = TASKA_EXTRA_GENERATED_LIGAND_NODES,
) -> NodeCountPlan:
    """额外节点增加采样数量，但不提高最终连通分子的最低节点数。"""
    fixed_ligand_nodes = int(fixed_ligand_nodes)
    if fixed_ligand_nodes < 0:
        raise FatalNodeCountError("固定配体节点数量不能为负数")

    if fixed_ligand_nodes == 0:
        base_nodes = int(manual_total_ligand_nodes)
        extra_nodes = int(creative_extra_generated_ligand_nodes)
    else:
        base_nodes = int(taska_added_ligand_nodes)
        extra_nodes = int(taska_extra_generated_ligand_nodes)
    if base_nodes < 1 or extra_nodes < 0:
        raise FatalNodeCountError("基础节点至少为 1，额外节点不能为负数")

    return NodeCountPlan(
        mode="manual",
        fixed_ligand_nodes=fixed_ligand_nodes,
        base_generated_ligand_nodes=base_nodes,
        extra_generated_ligand_nodes=extra_nodes,
        generated_ligand_nodes=base_nodes + extra_nodes,
        minimum_connected_ligand_nodes=fixed_ligand_nodes + base_nodes,
    )
