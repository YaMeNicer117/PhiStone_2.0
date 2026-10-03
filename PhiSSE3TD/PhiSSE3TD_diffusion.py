import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
import math

# Local Imports
import PhiSSE3TD_config as config

class DiffusionProcess:
    def __init__(self, model: nn.Module, device,
                num_timesteps=config.NUM_TIMESTEPS,
                beta_schedule=config.BETA_SCHEDULE,
                max_ll_neighbors=1, max_lp_neighbors=1):
        self.model = model
        self.num_timesteps = num_timesteps
        self.device = device
        self.max_ll_neighbors = max(1, int(max_ll_neighbors))
        self.max_lp_neighbors = max(1, int(max_lp_neighbors))
        
        if beta_schedule == 'linear':
            self.betas = self._get_linear_schedule_betas(num_timesteps)
        elif beta_schedule == 'cosine':
            self.betas = self._get_cosine_schedule_betas(num_timesteps)
        else:
            raise ValueError(f"不支持的 beta_schedule 类型: {beta_schedule}")
        
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

    def _get_linear_schedule_betas(self, num_timesteps, beta_start=1e-4, beta_end=0.02):
        return torch.linspace(beta_start, beta_end, num_timesteps, device='cpu').to(self.device)

    def _get_cosine_schedule_betas(self, num_timesteps, s=0.008):
        steps = torch.arange(num_timesteps + 1, device='cpu', dtype=torch.float32)
        f_t = torch.cos(((steps / num_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = f_t / f_t[0]
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        betas = 1 - (alphas_cumprod / alphas_cumprod_prev)
        return torch.clip(betas[1:], 0.0001, 0.9999).to(self.device)

    # =========================================================================
    # 核心优化：动态图构建 (Optimized Dynamic Graph Construction)
    # =========================================================================

    @staticmethod
    def _deduplicate_edges(edge_index, edge_attr, num_nodes):
        if edge_index.numel() == 0:
            return edge_index, edge_attr
        edge_ids = edge_index[0] * num_nodes + edge_index[1]
        order = torch.argsort(edge_ids)
        sorted_ids = edge_ids[order]
        keep = torch.ones_like(sorted_ids, dtype=torch.bool)
        keep[1:] = sorted_ids[1:] != sorted_ids[:-1]
        selected = order[keep]
        return edge_index[:, selected], edge_attr[selected]

    @staticmethod
    def _empty_edges(device):
        return (
            torch.empty((2, 0), dtype=torch.long, device=device),
            torch.empty((0, config.EDGE_ATTR_DIM), dtype=torch.float32, device=device),
        )

    def rebuild_graph_with_dynamic_edges(self, graph_data: Data) -> Data:
        """
        从当前（已加噪）质心坐标创建 Candidate-LL/Candidate-LP。
        PP/GL 完全复用预处理数据；Null 只作为监督标签，永不注入消息边属性。
        """
        device = graph_data.pos.device
        if graph_data.edge_attr.shape[-1] != config.EDGE_ATTR_DIM:
            raise ValueError(
                f"动态图输入 edge_attr 必须为 {config.EDGE_ATTR_DIM} 维，"
                f"实际为 {graph_data.edge_attr.shape[-1]}"
            )
        ligand_mask = graph_data.is_ligand
        protein_mask = graph_data.is_protein
        batch_index = graph_data.batch

        fixed_mask = graph_data.edge_attr[:, 2:].sum(dim=-1).bool()
        fixed_edge_index = graph_data.edge_index[:, fixed_mask]
        fixed_edge_attr = graph_data.edge_attr[fixed_mask]
        fixed_edge_index, fixed_edge_attr = self._deduplicate_edges(
            fixed_edge_index, fixed_edge_attr, graph_data.num_nodes
        )

        ll_edge_chunks = []
        lp_edge_chunks = []
        for graph_id in torch.unique(batch_index):
            ligand_indices = torch.nonzero(
                ligand_mask & (batch_index == graph_id), as_tuple=False
            ).flatten()
            protein_indices = torch.nonzero(
                protein_mask & (batch_index == graph_id), as_tuple=False
            ).flatten()

            if ligand_indices.numel() > 1:
                ligand_distances = torch.cdist(
                    graph_data.pos[ligand_indices].detach().float(),
                    graph_data.pos[ligand_indices].detach().float(),
                )
                ligand_distances.fill_diagonal_(float('inf'))
                for local_ligand in range(ligand_indices.numel()):
                    row = ligand_distances[local_ligand]
                    candidates = torch.nonzero(
                        row < config.DYNAMIC_LL_RADIUS, as_tuple=False
                    ).flatten()
                    if candidates.numel() > self.max_ll_neighbors:
                        nearest = torch.topk(
                            row[candidates], self.max_ll_neighbors, largest=False
                        ).indices
                        candidates = candidates[nearest]
                    if candidates.numel():
                        src = ligand_indices[local_ligand].expand(candidates.numel())
                        dst = ligand_indices[candidates]
                        ll_edge_chunks.append(torch.stack([src, dst], dim=0))

            if ligand_indices.numel() and protein_indices.numel():
                lp_distances = torch.cdist(
                    graph_data.pos[ligand_indices].detach().float(),
                    graph_data.pos[protein_indices].detach().float(),
                )
                for local_ligand in range(ligand_indices.numel()):
                    row = lp_distances[local_ligand]
                    candidates = torch.nonzero(
                        row < config.DYNAMIC_LP_RADIUS, as_tuple=False
                    ).flatten()
                    if candidates.numel() == 0:
                        # 噪声导致半径内无蛋白时，只补最近一个候选；它仍可被标为 Null。
                        candidates = torch.argmin(row).reshape(1)
                    elif candidates.numel() > self.max_lp_neighbors:
                        nearest = torch.topk(
                            row[candidates], self.max_lp_neighbors, largest=False
                        ).indices
                        candidates = candidates[nearest]

                    ligand_global = ligand_indices[local_ligand].expand(candidates.numel())
                    protein_global = protein_indices[candidates]
                    lp_edge_chunks.append(
                        torch.cat(
                            [
                                torch.stack([ligand_global, protein_global], dim=0),
                                torch.stack([protein_global, ligand_global], dim=0),
                            ],
                            dim=1,
                        )
                    )

        empty_index, empty_attr = self._empty_edges(device)
        edge_index_ll = (
            torch.cat(ll_edge_chunks, dim=1) if ll_edge_chunks else empty_index
        )
        attr_ll = (
            torch.zeros(
                (edge_index_ll.shape[1], config.EDGE_ATTR_DIM),
                device=device,
                dtype=torch.float32,
            )
            if edge_index_ll.numel() else empty_attr
        )
        if attr_ll.numel():
            attr_ll[:, 0] = 1.0

        edge_index_lp = (
            torch.cat(lp_edge_chunks, dim=1) if lp_edge_chunks else empty_index.clone()
        )
        attr_lp = (
            torch.zeros(
                (edge_index_lp.shape[1], config.EDGE_ATTR_DIM),
                device=device,
                dtype=torch.float32,
            )
            if edge_index_lp.numel() else empty_attr.clone()
        )
        if attr_lp.numel():
            attr_lp[:, 1] = 1.0

        final_edge_index = torch.cat(
            [fixed_edge_index, edge_index_ll, edge_index_lp], dim=1
        )
        final_edge_attr = torch.cat([fixed_edge_attr, attr_ll, attr_lp], dim=0)
        final_edge_index, final_edge_attr = self._deduplicate_edges(
            final_edge_index, final_edge_attr, graph_data.num_nodes
        )
        graph_data.edge_index = final_edge_index
        graph_data.edge_attr = final_edge_attr
        return graph_data

    # =========================================================================
    # 扩散采样逻辑 (Sampling Logic)
    # =========================================================================
    def q_sample(self, x0_feat, x0_pos, x0_ref_coords, t, batch_indices):
        noise_feat = torch.randn_like(x0_feat)
        noise_pos = torch.randn_like(x0_pos)
        noise_ref_coords = torch.randn_like(x0_ref_coords)
        t_nodes = t[batch_indices]
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t_nodes]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t_nodes]
        
        xt_feat = sqrt_alphas_cumprod_t.unsqueeze(-1) * x0_feat + sqrt_one_minus_alphas_cumprod_t.unsqueeze(-1) * noise_feat
        xt_pos = sqrt_alphas_cumprod_t.unsqueeze(-1) * x0_pos + sqrt_one_minus_alphas_cumprod_t.unsqueeze(-1) * noise_pos
        xt_ref_coords = sqrt_alphas_cumprod_t.view(-1, 1, 1) * x0_ref_coords + sqrt_one_minus_alphas_cumprod_t.view(-1, 1, 1) * noise_ref_coords
        return (xt_feat, xt_pos, xt_ref_coords), (noise_feat, noise_pos, noise_ref_coords)

    def predict_x0_from_noise(self, xt_tuple, pred_noise_tuple, t, batch_indices):
        xt_feat, xt_pos, xt_ref_coords = xt_tuple
        pred_noise_feat, pred_noise_pos, pred_noise_frame = pred_noise_tuple
        t_nodes = t[batch_indices]
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t_nodes]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t_nodes]
        
        safe_denom_t_node = sqrt_alphas_cumprod_t + 1e-8
        x0_pred_feat = (xt_feat - sqrt_one_minus_alphas_cumprod_t.unsqueeze(-1) * pred_noise_feat) / safe_denom_t_node.unsqueeze(-1)
        x0_pred_pos = (xt_pos - sqrt_one_minus_alphas_cumprod_t.unsqueeze(-1) * pred_noise_pos) / safe_denom_t_node.unsqueeze(-1)
        x0_pred_ref_coords = (xt_ref_coords - sqrt_one_minus_alphas_cumprod_t.view(-1, 1, 1) * pred_noise_frame) / safe_denom_t_node.view(-1, 1, 1)        
       
        x0_pred_pos = torch.clamp(
            x0_pred_pos,
            -config.POS_ABS_SAFETY_CLAMP,
            config.POS_ABS_SAFETY_CLAMP
        )
        
        # --- 修复：区分 Embedding 和 Chem Props 的截断策略 ---
        # 1. 切片提取 Embedding (前 128 维) 和 化学属性 (后 14 维)
        pred_embeds = x0_pred_feat[:, :config.EMBEDDING_DIM_IN]
        pred_chem_props = x0_pred_feat[:, config.EMBEDDING_DIM_IN : config.EMBEDDING_DIM_IN + config.CHEM_PROPS_DIM_IN]
        
        # 2. 对 Embedding 放宽限制 (仅防 Inf 数值崩溃，不改变语义方向，这里设为 20.0 作为绝对安全网)
        pred_embeds = torch.clamp(
            pred_embeds,
            -config.EMBEDDING_SAFETY_CLAMP,
            config.EMBEDDING_SAFETY_CLAMP
        )
        
        # 3. 仅对 化学属性 执行严格的 FEAT_CLAMP_BUFFER (如 4.0)
        pred_chem_props = torch.clamp(pred_chem_props, -config.FEAT_CLAMP_BUFFER, config.FEAT_CLAMP_BUFFER)
        
        # 4. 重新拼接
        x0_pred_feat = torch.cat([pred_embeds, pred_chem_props], dim=-1)
        
        x0_pred_ref_coords = torch.clamp(x0_pred_ref_coords, -config.REF_COORDS_CLAMP_BUFFER, config.REF_COORDS_CLAMP_BUFFER)

        return x0_pred_feat, x0_pred_pos, x0_pred_ref_coords
