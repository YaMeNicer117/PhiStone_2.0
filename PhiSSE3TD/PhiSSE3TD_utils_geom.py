import torch
import os
import PhiSSE3TD_config as config
import torch.nn.functional as F

DEBUG_TENSOR_HEALTH = os.environ.get('DEBUG_TENSOR_HEALTH', '0') == '1'

def _canonical_pair_ids(edge_index, num_nodes):
    """将无向节点对编码为与方向无关的唯一整数。"""
    low = torch.minimum(edge_index[0], edge_index[1])
    high = torch.maximum(edge_index[0], edge_index[1])
    return low * num_nodes + high


def _canonical_edge_ids(edge_index, edge_type, num_nodes):
    """方向无关的节点对 + 边族编码；LL/LP 的双向消息边共享同一标签。"""
    return _canonical_pair_ids(edge_index, num_nodes) * 2 + edge_type.long()


def build_dynamic_edge_targets(
    candidate_edge_index,
    candidate_edge_type,
    gt_edge_index,
    gt_edge_type,
    num_nodes,
):
    """把 Candidate-LL/LP 匹配到干净图，返回 [LL=0, LP=1, Null=2]。"""
    num_candidates = candidate_edge_index.shape[1]
    targets = torch.full(
        (num_candidates,), 2, dtype=torch.long, device=candidate_edge_index.device
    )
    if num_candidates == 0 or gt_edge_index.numel() == 0:
        return targets

    candidate_ids = _canonical_edge_ids(
        candidate_edge_index, candidate_edge_type, num_nodes
    )
    gt_ids = _canonical_edge_ids(gt_edge_index, gt_edge_type, num_nodes)
    sorted_gt_ids, order = torch.sort(gt_ids)
    sorted_gt_types = gt_edge_type[order].long()
    locations = torch.searchsorted(sorted_gt_ids, candidate_ids)
    in_bounds = locations < sorted_gt_ids.numel()
    safe_locations = locations.clamp(max=max(0, sorted_gt_ids.numel() - 1))
    matched = in_bounds & (sorted_gt_ids[safe_locations] == candidate_ids)
    targets[matched] = sorted_gt_types[safe_locations[matched]]
    return targets


def build_covalent_edge_targets(candidate_edge_index, gt_edge_index, num_nodes):
    """将全部无序 LL 候选对标为非共价(0)或共价(1)。"""
    num_candidates = candidate_edge_index.shape[1]
    targets = torch.zeros(
        num_candidates,
        dtype=torch.float32,
        device=candidate_edge_index.device,
    )
    if num_candidates == 0 or gt_edge_index.numel() == 0:
        return targets

    candidate_ids = _canonical_pair_ids(candidate_edge_index, num_nodes)
    gt_ids = torch.unique(_canonical_pair_ids(gt_edge_index, num_nodes), sorted=True)
    locations = torch.searchsorted(gt_ids, candidate_ids)
    in_bounds = locations < gt_ids.numel()
    safe_locations = locations.clamp(max=max(0, gt_ids.numel() - 1))
    matched = in_bounds & (gt_ids[safe_locations] == candidate_ids)
    targets[matched] = 1.0
    return targets


def select_dynamic_edge_samples(
    target_classes,
    candidate_edge_index,
    candidate_edge_type,
    batch,
    num_nodes,
    fixed_node_mask=None,
):
    """
    保留全部真实边；每图最多选择 max(1,真实动态边数) 个唯一 Null 节点对。
    同一节点对的两个消息方向会共同保留或共同丢弃。
    """
    device = target_classes.device
    selected = torch.zeros_like(target_classes, dtype=torch.bool)
    if target_classes.numel() == 0:
        return selected

    allowed = torch.ones_like(selected)
    if fixed_node_mask is not None:
        src, dst = candidate_edge_index
        allowed &= ~(fixed_node_mask[src] & fixed_node_mask[dst])

    canonical_ids = _canonical_edge_ids(
        candidate_edge_index, candidate_edge_type, num_nodes
    )
    edge_graph = batch[candidate_edge_index[0]]
    for graph_id in torch.unique(edge_graph):
        graph_mask = (edge_graph == graph_id) & allowed
        real_mask = graph_mask & (target_classes != 2)
        selected |= real_mask

        real_unique = canonical_ids[real_mask].unique().numel()
        null_ids = canonical_ids[graph_mask & (target_classes == 2)].unique()
        max_null = max(1, int(real_unique))
        if null_ids.numel() > max_null:
            permutation = torch.randperm(null_ids.numel(), device=device)[:max_null]
            null_ids = null_ids[permutation]
        if null_ids.numel():
            selected |= graph_mask & (target_classes == 2) & torch.isin(
                canonical_ids, null_ids
            )
    return selected


def compute_edge_topology_loss(
    edge_logits,
    target_classes,
    sample_mask=None,
    sample_weights=None,
    class_weights=None,
    gamma=2.0,
):
    """三分类、类别平衡且支持时间步权重的 Focal Loss。"""
    if sample_mask is None:
        sample_mask = torch.ones(
            target_classes.shape[0], dtype=torch.bool, device=target_classes.device
        )
    if edge_logits.shape[0] == 0 or not sample_mask.any():
        return edge_logits.sum() * 0.0

    logits = edge_logits[sample_mask]
    targets = target_classes[sample_mask]
    ce_loss = F.cross_entropy(logits, targets, reduction='none')
    pt = torch.exp(-ce_loss).clamp(min=0.0, max=0.9999)
    alpha = torch.as_tensor(
        class_weights if class_weights is not None else config.EDGE_CLASS_WEIGHTS,
        device=edge_logits.device,
        dtype=ce_loss.dtype,
    )
    focal = alpha[targets] * ((1.0 - pt) ** gamma) * ce_loss
    if sample_weights is not None:
        weights = sample_weights[sample_mask].to(focal.dtype)
        return (focal * weights).sum() / max(1, focal.numel())
    return focal.mean()


def compute_covalent_edge_loss(
    covalent_logits,
    covalent_targets,
    pos_weight,
    sample_mask=None,
    sample_weights=None,
):
    """类别加权 BCE；时间权重使用固定样本数归一化以保留绝对调度强度。"""
    if sample_mask is None:
        sample_mask = torch.ones_like(covalent_targets, dtype=torch.bool)
    if covalent_logits.numel() == 0 or not sample_mask.any():
        return covalent_logits.sum() * 0.0

    logits = covalent_logits[sample_mask]
    targets = covalent_targets[sample_mask].to(logits.dtype)
    positive_weight = torch.as_tensor(
        pos_weight, device=logits.device, dtype=logits.dtype
    )
    per_pair = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction='none',
        pos_weight=positive_weight,
    )
    if sample_weights is not None:
        weights = sample_weights[sample_mask].to(per_pair.dtype)
        per_pair = per_pair * weights
    return per_pair.sum() / max(1, per_pair.numel())


def compute_feature_noise_losses(
    pred_noise_feat,
    true_noise_feat,
    hac_class_targets,
    hac_class_weights,
    embedding_dim=config.EMBEDDING_DIM_IN,
    beta=1.0,
    node_weights=None,
):
    """分别计算 embedding 与化学属性噪声，并支持逐节点优先权重。"""
    if pred_noise_feat.shape[0] == 0:
        zero = pred_noise_feat.sum() * 0.0
        return zero, zero
    raw = F.smooth_l1_loss(
        pred_noise_feat, true_noise_feat, reduction='none', beta=beta
    )
    embed_node_loss = raw[:, :embedding_dim].mean(dim=-1)
    chem_node_loss = raw[:, embedding_dim:].mean(dim=-1)
    hac_weights = torch.as_tensor(
        hac_class_weights, device=pred_noise_feat.device, dtype=embed_node_loss.dtype
    )[hac_class_targets]
    if node_weights is None:
        priority_weights = torch.ones_like(embed_node_loss)
    else:
        priority_weights = node_weights.to(embed_node_loss.dtype)
        if priority_weights.shape != embed_node_loss.shape:
            raise ValueError(
                "feature noise 的 node_weights 形状与节点损失不一致"
            )
    combined_embed_weights = hac_weights * priority_weights
    denominator = max(1, embed_node_loss.numel())
    embed_loss = (embed_node_loss * combined_embed_weights).sum() / denominator
    chem_loss = (chem_node_loss * priority_weights).sum() / denominator
    return embed_loss, chem_loss


def compute_embedding_cosine_loss(pred_x0_embed, true_embed, node_weights=None):
    if pred_x0_embed.shape[0] == 0:
        return pred_x0_embed.sum() * 0.0
    node_loss = 1.0 - F.cosine_similarity(
        pred_x0_embed.float(), true_embed.float(), dim=-1, eps=1.0e-8
    )
    if node_weights is None:
        return node_loss.mean()
    weights = node_weights.to(node_loss.dtype)
    return (node_loss * weights).sum() / max(1, node_loss.numel())


def _weighted_group_mean(values, weights, mask, zero):
    if not mask.any():
        return zero
    selected_weights = weights[mask].to(values.dtype)
    denominator = mask.sum().clamp_min(1).to(dtype=values.dtype)
    return (values[mask] * selected_weights).sum() / denominator


def compute_frame_atom_geometry_losses(
    pred_abs_ref,
    true_abs_ref,
    ref_atom_mask,
    candidate_edge_index,
    target_classes,
    candidate_edge_type,
    is_ligand,
    is_global,
    batch,
    global_pos,
    free_ligand_mask,
    dynamic_node_weights,
    gl_node_weights,
    null_sample_mask=None,
    beta=1.0,
):
    """
    以真实框架槽掩码计算节点边级平均损失。每条节点边先平均其有效原子
    笛卡尔积，再在边之间平均，因此 1x1、2x3、3x3 不会产生天然权重差。
    """
    zero = pred_abs_ref.sum() * 0.0
    num_edges = candidate_edge_index.shape[1]
    losses = torch.empty((0,), device=pred_abs_ref.device, dtype=pred_abs_ref.dtype)
    edge_weights = torch.empty_like(losses)
    edge_targets = torch.empty((0,), device=pred_abs_ref.device, dtype=torch.long)
    edge_selected_null = torch.empty(
        (0,), device=pred_abs_ref.device, dtype=torch.bool
    )

    if num_edges:
        canonical_ids = _canonical_edge_ids(
            candidate_edge_index, candidate_edge_type, pred_abs_ref.shape[0]
        )
        order = torch.argsort(canonical_ids)
        sorted_ids = canonical_ids[order]
        keep = torch.ones_like(sorted_ids, dtype=torch.bool)
        keep[1:] = sorted_ids[1:] != sorted_ids[:-1]
        unique_indices = order[keep]

        src = candidate_edge_index[0, unique_indices]
        dst = candidate_edge_index[1, unique_indices]
        atom_pair_mask = (
            ref_atom_mask[src].unsqueeze(2) & ref_atom_mask[dst].unsqueeze(1)
        )
        valid_pair_count = atom_pair_mask.sum(dim=(1, 2))
        has_free_ligand = (
            (is_ligand[src] & free_ligand_mask[src])
            | (is_ligand[dst] & free_ligand_mask[dst])
        )
        valid_edge = (valid_pair_count > 0) & has_free_ligand

        pred_distance = torch.cdist(
            pred_abs_ref[src].float(), pred_abs_ref[dst].float()
        )
        true_distance = torch.cdist(
            true_abs_ref[src].float(), true_abs_ref[dst].float()
        )
        atom_losses = F.smooth_l1_loss(
            pred_distance, true_distance, reduction='none', beta=beta
        )
        per_edge = (
            (atom_losses * atom_pair_mask).sum(dim=(1, 2))
            / valid_pair_count.clamp_min(1)
        ).to(pred_abs_ref.dtype)

        free_src = is_ligand[src] & free_ligand_mask[src]
        free_dst = is_ligand[dst] & free_ligand_mask[dst]
        weight_sum = (
            dynamic_node_weights[src] * free_src
            + dynamic_node_weights[dst] * free_dst
        )
        weight_count = free_src.long() + free_dst.long()
        per_edge_weights = weight_sum / weight_count.clamp_min(1)

        losses = per_edge[valid_edge]
        edge_weights = per_edge_weights[valid_edge]
        edge_targets = target_classes[unique_indices][valid_edge]
        if null_sample_mask is None:
            edge_selected_null = torch.ones_like(edge_targets, dtype=torch.bool)
        else:
            edge_selected_null = null_sample_mask[unique_indices][valid_edge]

    loss_ll = _weighted_group_mean(
        losses, edge_weights, edge_targets == 0, zero
    )
    loss_lp = _weighted_group_mean(
        losses, edge_weights, edge_targets == 1, zero
    )
    loss_null = _weighted_group_mean(
        losses,
        edge_weights,
        (edge_targets == 2) & edge_selected_null,
        zero,
    )

    # 每个自由 ligand 只计算一次到本图 Global 中心的径向距离。
    free_indices = torch.nonzero(
        is_ligand & free_ligand_mask, as_tuple=False
    ).flatten()
    loss_gl = zero
    if free_indices.numel():
        global_indices = torch.nonzero(is_global, as_tuple=False).flatten()
        num_graphs = int(batch.max().item()) + 1
        global_by_graph = torch.empty(
            num_graphs, dtype=torch.long, device=batch.device
        )
        global_by_graph[batch[global_indices]] = global_indices
        matched_global = global_by_graph[batch[free_indices]]
        valid_slots = ref_atom_mask[free_indices]
        pred_radius = torch.linalg.vector_norm(
            pred_abs_ref[free_indices].float()
            - global_pos[matched_global, None, :].float(),
            dim=-1,
        )
        true_radius = torch.linalg.vector_norm(
            true_abs_ref[free_indices].float()
            - global_pos[matched_global, None, :].float(),
            dim=-1,
        )
        slot_losses = F.smooth_l1_loss(
            pred_radius, true_radius, reduction='none', beta=beta
        )
        per_node = (
            (slot_losses * valid_slots).sum(dim=-1)
            / valid_slots.sum(dim=-1).clamp_min(1)
        ).to(pred_abs_ref.dtype)
        weights = gl_node_weights[free_indices].to(per_node.dtype)
        loss_gl = (per_node * weights).sum() / max(1, per_node.numel())

    return loss_ll, loss_lp, loss_null, loss_gl


def check_tensor_health(tensor, variable_name: str, location_tag: str):
    """
    检查一个张量是否包含 NaN 或 inf。
    警告：此操作会触发 CPU-GPU 同步，严重拖慢训练速度。仅限 Debug 时使用。
    """
    # 极速短路：如果未开启调试模式，直接返回，开销为 0
    if not DEBUG_TENSOR_HEALTH:
        return

    # 只有在 DEBUG 模式下，才执行会阻塞线程的张量检查
    if not torch.all(torch.isfinite(tensor)):
        has_nan = torch.isnan(tensor).any()
        has_inf = torch.isinf(tensor).any()
        
        rank = os.environ.get('RANK', '0')
        
        print("\n" + "="*80)
        print(f" 严重诊断错误：在 [{location_tag}] (Rank {rank}) 发现不健康的张量! ")
        print(f"变量名: {variable_name}")
        print(f"形状: {tensor.shape}")
        print(f"数据类型: {tensor.dtype}")
        print(f"设备: {tensor.device}")
        print(f"包含 NaN: {has_nan.item()}")
        print(f"包含 Inf: {has_inf.item()}")
        
        finite_vals = tensor[torch.isfinite(tensor)]
        if finite_vals.numel() > 0:
            print(f"有限值的最大值: {finite_vals.max().item()}")
            print(f"有限值的最小值: {finite_vals.min().item()}")
            print(f"有限值的平均值: {finite_vals.mean().item()}")
        else:
            print("张量中不包含任何有限值。")
            
        print("="*80 + "\n")
        
        raise RuntimeError(f"程序已终止 (Rank {rank})，以防止 NaN/Inf 传播。请检查上面的报告。")
    
