"""面向二维片段图的单任务 GATv2 连续活性回归模型。"""

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATv2Conv, GlobalAttention, global_mean_pool

if __package__:
    from . import PhiSGATv2_config as config
else:  # 支持直接运行 GATv2 目录中的脚本
    import PhiSGATv2_config as config


class VirtualNodeBlock(nn.Module):
    """以门控残差方式在图级虚拟状态与配体节点之间交换信息。"""

    def __init__(self, node_dim, hidden_dim, dropout, gate_bias):
        super().__init__()
        self.virtual_update = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, node_dim),
        )
        self.virtual_norm = nn.LayerNorm(node_dim)
        self.node_gate = nn.Linear(node_dim * 2, 1)
        self.node_norm = nn.LayerNorm(node_dim)
        self.broadcast_dropout = nn.Dropout(p=dropout)

        # 初始时仅小幅引入全局信息，随后由训练自行调整每个节点的门控。
        nn.init.zeros_(self.node_gate.weight)
        nn.init.constant_(self.node_gate.bias, gate_bias)

    def forward(self, node_features, batch, virtual_features):
        graph_summary = global_mean_pool(node_features, batch)
        if graph_summary.shape != virtual_features.shape:
            raise RuntimeError(
                "虚拟节点与图级摘要形状不一致: "
                f"{tuple(virtual_features.shape)} != "
                f"{tuple(graph_summary.shape)}"
            )

        virtual_delta = self.virtual_update(
            graph_summary + virtual_features
        )
        virtual_features = self.virtual_norm(
            virtual_features + virtual_delta
        )

        broadcast_features = virtual_features[batch]
        gate = torch.sigmoid(
            self.node_gate(
                torch.cat(
                    [node_features, broadcast_features],
                    dim=-1,
                )
            )
        )
        node_features = self.node_norm(
            node_features
            + self.broadcast_dropout(gate * broadcast_features)
        )
        return node_features, virtual_features


class GATv2Model(nn.Module):
    """多层 GATv2、可选辅助虚拟节点、注意力池化与单值回归头。"""

    def __init__(self):
        super().__init__()
        config.validate_config()

        self.dropout = float(config.DROPOUT)
        self.use_virtual_node = config.USE_VIRTUAL_NODE
        self.embedding_norm = nn.LayerNorm(config.EMBEDDING_DIM)
        self.convs = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.virtual_blocks = nn.ModuleList()
        if self.use_virtual_node:
            self.virtual_node_embedding = nn.Parameter(
                torch.zeros(1, config.FINAL_NODE_DIM)
            )

        input_dim = config.COMBINED_INPUT_DIM
        for _ in range(config.NUM_LAYERS):
            self.convs.append(
                GATv2Conv(
                    in_channels=input_dim,
                    out_channels=config.HIDDEN_CHANNELS,
                    heads=config.HEADS,
                    concat=True,
                )
            )
            self.layer_norms.append(
                nn.LayerNorm(config.FINAL_NODE_DIM)
            )
            if self.use_virtual_node:
                self.virtual_blocks.append(
                    VirtualNodeBlock(
                        node_dim=config.FINAL_NODE_DIM,
                        hidden_dim=config.MLP_HIDDEN_DIM,
                        dropout=self.dropout,
                        gate_bias=config.VIRTUAL_NODE_GATE_BIAS,
                    )
                )
            input_dim = config.FINAL_NODE_DIM

        gate_nn = nn.Linear(config.FINAL_NODE_DIM, 1)
        self.pool = GlobalAttention(gate_nn=gate_nn)
        if self.use_virtual_node:
            self.graph_fusion_gate = nn.Linear(config.FINAL_NODE_DIM * 2, 1)
            self.graph_fusion_dropout = nn.Dropout(p=self.dropout)
            nn.init.zeros_(self.graph_fusion_gate.weight)
            nn.init.constant_(
                self.graph_fusion_gate.bias,
                config.VIRTUAL_NODE_GATE_BIAS,
            )
        self.regression_head = nn.Sequential(
            nn.Linear(
                config.FINAL_NODE_DIM,
                config.MLP_HIDDEN_DIM,
            ),
            nn.SiLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(config.MLP_HIDDEN_DIM, 1),
        )

    @staticmethod
    def _validate_input(data):
        required = ("x", "edge_index", "frag_embeds")
        missing = [name for name in required if not hasattr(data, name)]
        if missing:
            raise ValueError(f"图数据缺少必需字段: {missing}")

        x = data.x
        frag_embeds = data.frag_embeds
        edge_index = data.edge_index
        if not torch.is_tensor(x) or x.dim() != 2:
            raise ValueError("data.x 必须是二维张量")
        if x.size(1) != config.INPUT_DIM_NUMERIC:
            raise ValueError(
                f"data.x 必须为 [N, {config.INPUT_DIM_NUMERIC}]，"
                f"实际为 {tuple(x.shape)}"
            )
        if not torch.is_tensor(frag_embeds) or frag_embeds.dim() != 2:
            raise ValueError("data.frag_embeds 必须是二维张量")
        if frag_embeds.size(1) != config.EMBEDDING_DIM:
            raise ValueError(
                f"data.frag_embeds 必须为 [N, {config.EMBEDDING_DIM}]，"
                f"实际为 {tuple(frag_embeds.shape)}"
            )
        if frag_embeds.size(0) != x.size(0):
            raise ValueError(
                "data.x 与 data.frag_embeds 的节点数不一致: "
                f"{x.size(0)} != {frag_embeds.size(0)}"
            )
        if not torch.isfinite(x).all():
            raise ValueError("data.x 包含 NaN 或 Inf")
        if not torch.isfinite(frag_embeds).all():
            raise ValueError("data.frag_embeds 包含 NaN 或 Inf")
        if (
            not torch.is_tensor(edge_index)
            or edge_index.dim() != 2
            or edge_index.size(0) != 2
        ):
            actual_shape = (
                tuple(edge_index.shape)
                if torch.is_tensor(edge_index)
                else type(edge_index).__name__
            )
            raise ValueError(
                f"data.edge_index 必须为 [2, E]，实际为 {actual_shape}"
            )

    def forward(self, data):
        self._validate_input(data)

        numeric_features = data.x.float()
        fragment_embeddings = self.embedding_norm(
            data.frag_embeds.float()
        )
        edge_index = data.edge_index.long()
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(
                numeric_features.size(0),
                dtype=torch.long,
                device=numeric_features.device,
            )
        else:
            batch = batch.long()
        if batch.dim() != 1 or batch.numel() != numeric_features.size(0):
            raise ValueError(
                "data.batch 必须为每个节点提供一个图编号，"
                f"实际形状为 {tuple(batch.shape)}"
            )

        node_features = torch.cat(
            [numeric_features, fragment_embeddings],
            dim=-1,
        )
        if node_features.size(1) != config.COMBINED_INPUT_DIM:
            raise RuntimeError(
                "拼接后的节点维度与配置不一致: "
                f"{node_features.size(1)} != {config.COMBINED_INPUT_DIM}"
            )

        if self.use_virtual_node:
            num_graphs = int(batch.max().item()) + 1
            virtual_features = self.virtual_node_embedding.expand(
                num_graphs,
                -1,
            )

        for layer_index, (conv, layer_norm) in enumerate(
            zip(self.convs, self.layer_norms)
        ):
            node_features = conv(node_features, edge_index)
            node_features = layer_norm(node_features)
            node_features = F.silu(node_features)
            node_features = F.dropout(
                node_features,
                p=self.dropout,
                training=self.training,
            )
            if self.use_virtual_node:
                node_features, virtual_features = self.virtual_blocks[layer_index](
                    node_features,
                    batch,
                    virtual_features,
                )

        attention_graph_features = self.pool(node_features, batch)
        graph_features = attention_graph_features
        if self.use_virtual_node:
            graph_fusion_gate = torch.sigmoid(
                self.graph_fusion_gate(
                    torch.cat(
                        [attention_graph_features, virtual_features],
                        dim=-1,
                    )
                )
            )
            graph_features = (
                attention_graph_features
                + self.graph_fusion_dropout(
                    graph_fusion_gate * virtual_features
                )
            )
        raw_prediction = self.regression_head(graph_features)
        prediction = (
            config.OUTPUT_MIN
            + config.OUTPUT_RANGE * torch.sigmoid(raw_prediction)
        )
        if prediction.dim() != 2 or prediction.size(1) != 1:
            raise RuntimeError(
                f"模型输出必须为 [B, 1]，实际为 {tuple(prediction.shape)}"
            )
        return prediction
