import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
from torch_geometric.data import Batch
from torch_geometric.nn.aggr import AttentionalAggregation
from torch_geometric.utils import subgraph
from torch_scatter import scatter
from torch.utils.checkpoint import checkpoint

# =============================================================================
# [ 导入你的本地配置与组件 ]
# =============================================================================
import PhiSSE3TD_config as config
from PhiSSE3TD_model import EquivariantTransformerLayer
from e3nn import o3
from e3nn.math import soft_one_hot_linspace

# =============================================================================
# [ 超参数配置区 ] (Hyperparameters)
# =============================================================================
SIZE_PRED_EPOCHS = 120                # 训练轮数
SIZE_PRED_LR = 1e-3                  # 初始学习率
SIZE_PRED_WEIGHT_DECAY = 2e-5        # 权重衰减 (L2 正则化)
SIZE_PRED_NUM_LAYERS = 4             # Transformer 层数

# MLP 维度设置
INVARIANT_DIM = config.HIDDEN_SCALAR_CHANNELS + config.TOTAL_HIDDEN_VEC_CHANNELS
ATTN_GATE_HIDDEN = 128               
REGRESSION_HIDDEN = [256, 128, 64]   
REGRESSION_DROPOUT = 0.15             

SCRIPT_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = (
    SCRIPT_DIR
    / "training_artifacts"
    / "checkpoints"
    / "ligand_size_predictor"
)
BEST_CHECKPOINT_PATH = CHECKPOINT_DIR / "best.pt"
LAST_CHECKPOINT_PATH = CHECKPOINT_DIR / "last.pt"

# =============================================================================
# [ DDP 辅助函数 ]
# =============================================================================
def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0

def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt

# =============================================================================
# 1. 核心模型：口袋到配体大小预测器
# =============================================================================
class PocketToLigandSizePredictor(nn.Module):
    def __init__(self, num_layers=SIZE_PRED_NUM_LAYERS): 
        super().__init__()
        
        self.TOTAL_HIDDEN_VEC = config.TOTAL_HIDDEN_VEC_CHANNELS
        self.REF_COORDS_HIDDEN = config.REF_COORDS_HIDDEN_CHANNELS
        self.FREE_HIDDEN = self.TOTAL_HIDDEN_VEC - self.REF_COORDS_HIDDEN
        
        self.irreps_hidden = o3.Irreps(f"{config.HIDDEN_SCALAR_CHANNELS}x0e + {self.TOTAL_HIDDEN_VEC}x1o")
        
        self.irreps_ref_coords_out = o3.Irreps(f"{self.REF_COORDS_HIDDEN}x1o")
        self.irreps_free_vectors = o3.Irreps(f"{self.FREE_HIDDEN}x1o")
        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax=2)
        
        scalar_input_dim = config.EMBEDDING_DIM_IN + config.CHEM_PROPS_DIM_IN + config.TYPE_ENCODING_DIM_IN
        irreps_scalar_in = o3.Irreps(f"{scalar_input_dim}x0e")
        irreps_scalar_out = o3.Irreps(f"{config.HIDDEN_SCALAR_CHANNELS}x0e") + self.irreps_free_vectors
        
        self.scalar_embedding = o3.Linear(irreps_scalar_in, irreps_scalar_out)
        self.embed_norm = nn.LayerNorm(config.EMBEDDING_DIM_IN)

        irreps_ref_coords_in = o3.Irreps("3x1o") 
        self.ref_coords_embedding = o3.Linear(irreps_ref_coords_in, self.irreps_ref_coords_out)

        self.transformer_layers = nn.ModuleList([
            EquivariantTransformerLayer(self.irreps_hidden, self.irreps_hidden, self.irreps_sh)
            for _ in range(num_layers)
        ])

        # [优化] 口袋坐标是固定的，所有层都以 is_last_layer=True 运行，
        # coord_update_mlp 和 frame_update_mlp 永远不参与前向计算。
        # 删除它们以节省显存、参数量，并允许 DDP 关闭 find_unused_parameters。
        for layer in self.transformer_layers:
            if hasattr(layer, 'coord_update_mlp'):
                del layer.coord_update_mlp
            if hasattr(layer, 'frame_update_mlp'):
                del layer.frame_update_mlp

        self.attn_gate_nn = nn.Sequential(
            nn.Linear(INVARIANT_DIM, ATTN_GATE_HIDDEN),
            nn.SiLU(),
            nn.Linear(ATTN_GATE_HIDDEN, 1) 
        )
        self.global_pool = AttentionalAggregation(gate_nn=self.attn_gate_nn)

        layers = []
        current_dim = INVARIANT_DIM
        for h_dim in REGRESSION_HIDDEN:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.SiLU())
            layers.append(nn.Dropout(p=REGRESSION_DROPOUT))
            current_dim = h_dim
        layers.append(nn.Linear(current_dim, 1))
        
        self.regression_head = nn.Sequential(*layers)

    def _compute_geometry(self, pos, edge_index):
        pos_f32 = pos.float()
        edge_src, edge_dst = edge_index
        edge_vec = pos_f32[edge_dst] - pos_f32[edge_src]
        edge_squared = torch.sum(edge_vec ** 2, dim=-1)
        edge_length_raw = torch.sqrt(edge_squared + 1e-8)
        
        edge_sh = torch.zeros(edge_vec.shape[0], self.irreps_sh.dim, device=edge_vec.device, dtype=torch.float32)
        valid_edges_mask = edge_squared > 1e-6
        if valid_edges_mask.any():
            edge_sh[valid_edges_mask] = o3.spherical_harmonics(
                self.irreps_sh, edge_vec[valid_edges_mask], normalize=True, normalization='component'
            )
            
        edge_length_embedding = soft_one_hot_linspace(
            edge_length_raw, start=0.0, end=config.MAX_EDGE_LENGTH, 
            number=config.NUM_BASIS, basis='smooth_finite', cutoff=True
        ).mul(config.NUM_BASIS**0.5)
        
        return edge_vec, edge_length_raw, edge_sh, edge_length_embedding

    def forward(self, pocket_data):
        pos, ref_coords = pocket_data.pos, pocket_data.ref_coords
        
        raw_embeds = self.embed_norm(pocket_data.frag_embeds)
        features_in = torch.cat(
            [raw_embeds, pocket_data.x, pocket_data.node_type], dim=1
        )
        scalar_embedded = self.scalar_embedding(features_in)
        
        ref_coords_in = ref_coords.reshape(-1, 9)
        ref_coords_embedded = self.ref_coords_embedding(ref_coords_in)
        
        num_scalar_hidden = config.HIDDEN_SCALAR_CHANNELS
        hidden_scalars = scalar_embedded[:, :num_scalar_hidden]
        free_vectors = scalar_embedded[:, num_scalar_hidden:]
        
        f = torch.cat([hidden_scalars, ref_coords_embedded, free_vectors], dim=-1)
        
        edge_vec, edge_length_raw, edge_sh, edge_length_embedding = self._compute_geometry(pos, pocket_data.edge_index)
        edge_attr = pocket_data.edge_attr 

        # --- E. E(3) 等变消息传递 ---
        for layer in self.transformer_layers:
            if self.training:
                # 训练模式：使用梯度检查点极致压缩显存
                f, pos, ref_coords = checkpoint(
                    layer, 
                    f, pos, ref_coords, pocket_data.edge_index, edge_attr,
                    edge_vec, edge_length_raw, edge_sh, edge_length_embedding,
                    True, # 对应 is_last_layer=True
                    use_reentrant=False # 兼容 DDP 的最佳实践
                )
            else:
                # 验证/推理模式：不需要存图算梯度，直接前向传播
                f, pos, ref_coords = layer(
                    f, pos, ref_coords, pocket_data.edge_index, edge_attr,
                    edge_vec, edge_length_raw, edge_sh, edge_length_embedding,
                    is_last_layer=True 
                )

        node_scalars = f[:, :config.HIDDEN_SCALAR_CHANNELS]
        node_vectors = f[:, config.HIDDEN_SCALAR_CHANNELS:]
        node_vectors_view = node_vectors.view(-1, config.TOTAL_HIDDEN_VEC_CHANNELS, 3)
        vector_norms = torch.sqrt(torch.sum(node_vectors_view ** 2, dim=-1) + 1e-8) 

        invariant_features = torch.cat([node_scalars, vector_norms], dim=-1)

        graph_features = self.global_pool(invariant_features, index=pocket_data.batch, dim_size=pocket_data.num_graphs)
        predicted_size = self.regression_head(graph_features).squeeze(-1)
        
        return predicted_size

# =============================================================================
# 2. 数据处理：防泄露子图切割与目标提取
# =============================================================================
def extract_protein_pocket_subgraph(batch):
    """构造训练与推理共用的 protein-only、PP-edge 口袋子图。"""

    # 只保留 protein，显式排除 ligand 与预处理产生的 Global。
    protein_mask = batch.is_protein.bool()
    if not bool(protein_mask.any()):
        raise RuntimeError("节点数量预测器输入图中没有 protein 节点")

    pp_edge_mask = batch.edge_attr[:, 2].bool()
    
    # 输入子图严格只含 PP 边。
    subset_edge_index, subset_edge_attr = subgraph(
        protein_mask, 
        batch.edge_index[:, pp_edge_mask],
        edge_attr=batch.edge_attr[pp_edge_mask],
        relabel_nodes=True, 
        num_nodes=batch.num_nodes
    )
    
    # 只组装模型实际需要的字段，避免原 Batch 中的 ligand 标签或边残留。
    pocket_batch = Batch(
        frag_embeds=batch.frag_embeds[protein_mask],
        x=batch.x[protein_mask],
        node_type=batch.node_type[protein_mask],
        pos=batch.pos[protein_mask],
        ref_coords=batch.ref_coords[protein_mask],
        batch=batch.batch[protein_mask],
        is_ligand=batch.is_ligand[protein_mask],
        is_protein=batch.is_protein[protein_mask],
        is_global=batch.is_global[protein_mask],
        edge_index=subset_edge_index,
        edge_attr=subset_edge_attr,
    )
    pocket_batch._num_graphs = batch.num_graphs

    if pocket_batch.is_ligand.any() or pocket_batch.is_global.any():
        raise RuntimeError("节点数量预测器的 pocket 子图泄漏了 ligand 或 Global 节点")
    if pocket_batch.edge_attr.numel() and not pocket_batch.edge_attr[:, 2].bool().all():
        raise RuntimeError("节点数量预测器的 pocket 子图含非 PP 边")

    return pocket_batch


def extract_pocket_subgraph_and_target(batch):
    ligand_mask = batch.is_ligand.long()
    # 统计每个图真实的配体节点数量作为连续回归目标。
    target_sizes = scatter(
        ligand_mask, batch.batch, dim=0, reduce='sum'
    ).float()
    pocket_batch = extract_protein_pocket_subgraph(batch)

    return pocket_batch, target_sizes

# =============================================================================
# 3. 分布式训练循环 (DDP Training Loop 最终版)
# =============================================================================
def train_size_predictor_ddp():
    import contextlib

    import torch.optim as optim
    from torch.amp import GradScaler, autocast
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.tensorboard import SummaryWriter
    from tqdm import tqdm

    from PhiSSE3TD_dataset import get_train_val_dataloaders

    # 仅训练入口设置 CUDA allocator，导入模型供推理使用时不产生环境副作用。
    os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
    )

    # --- 1. DDP 初始化环境 ---
    is_distributed = "LOCAL_RANK" in os.environ
    cuda_available = torch.cuda.is_available()
    if is_distributed:
        if not cuda_available:
            raise RuntimeError(
                "节点数量预测器仅支持 CUDA DDP；CPU 环境请使用普通单进程启动"
            )
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')
        device = torch.device(f"cuda:{local_rank}")
        world_size = dist.get_world_size()
    else:
        device = torch.device('cuda' if cuda_available else 'cpu')
        world_size = 1

    if is_main_process():
        print(f"[{'='*60}]")
        execution_mode = "CUDA DDP" if is_distributed else str(device)
        print(f"初始化【轻量级口袋容量预测器】 ({execution_mode})")
        print(f"[{'='*60}]")
        print(f"GPUs Total   : {world_size}")
        print(f"Global Attn  : On | Vector Norms: On | Dropout: {REGRESSION_DROPOUT}")
        print(f"Batch Size   : {config.BATCH_SIZE} | Grad Accum: {config.GRAD_ACCUMULATION_STEPS}")
        print(f"Effective BS : {config.BATCH_SIZE * config.GRAD_ACCUMULATION_STEPS * world_size}")

    # --- 2. 加载 DataLoader ---
    train_loader, val_loader, train_sampler = get_train_val_dataloaders(
        distributed=is_distributed
    )
    
    # --- 3. 初始化模型与 DDP 包裹 ---
    model = PocketToLigandSizePredictor().to(device)
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
        
    optimizer = optim.AdamW(model.parameters(), lr=SIZE_PRED_LR, weight_decay=SIZE_PRED_WEIGHT_DECAY)
    loss_fn = nn.SmoothL1Loss(beta=1.0) 
    
    # 【新增：余弦退火调度器】
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=SIZE_PRED_EPOCHS, 
        eta_min= 5e-6
    )
    
    # CUDA 使用 AMP；CPU 保持完整 float32 路径。
    amp_enabled = device.type == 'cuda'
    supports_bf16 = amp_enabled and torch.cuda.is_bf16_supported()
    amp_dtype = (
        torch.bfloat16 if supports_bf16 else torch.float16
    ) if amp_enabled else None
    scaler = GradScaler(
        enabled=bool(amp_enabled and amp_dtype == torch.float16)
    )

    def amp_context():
        if not amp_enabled:
            return contextlib.nullcontext()
        return autocast(
            device_type='cuda', enabled=True, dtype=amp_dtype
        )

    if is_main_process():
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        # 使用配置中的统一 TensorBoard 目录。
        log_dir = Path(config.LOG_DIR_SIZE_PREDICTOR)
        log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(str(log_dir))
        
        amp_label = str(amp_dtype) if amp_enabled else "disabled (float32)"
        print(f"AMP 精度模式: {amp_label}")
    else:
        writer = None
        
    best_val_loss = float('inf')
    grad_clip_threshold = 10.0  

    # =====================================================================
    # 断点续训：只从 last.pt 恢复，best.pt 专供推理与最优权重保留。
    # =====================================================================
    start_epoch = 0
    if LAST_CHECKPOINT_PATH.is_file():
        if is_main_process():
            print(
                f"发现 last checkpoint，尝试从 "
                f"{LAST_CHECKPOINT_PATH} 恢复训练..."
            )

        checkpoint_dict = torch.load(
            LAST_CHECKPOINT_PATH,
            map_location=device,
            weights_only=False,
        )
        
        # 剥离 DDP 外壳以匹配权重键值对
        model_to_load = model.module if hasattr(model, 'module') else model

        required_keys = {
            'epoch',
            'model_state_dict',
            'optimizer_state_dict',
            'scheduler_state_dict',
            'scaler_state_dict',
            'best_val_loss',
            'embedding_dim',
        }
        if not isinstance(checkpoint_dict, dict):
            raise RuntimeError(
                "节点数量 last checkpoint 不是完整训练状态字典"
            )
        missing_keys = sorted(required_keys.difference(checkpoint_dict))
        if missing_keys:
            raise RuntimeError(
                "节点数量 last checkpoint 缺少字段: "
                + ", ".join(missing_keys)
            )
        if checkpoint_dict.get('embedding_dim') != config.EMBEDDING_DIM_IN:
            raise RuntimeError(
                "节点数量 last checkpoint 的嵌入维度与当前模型不一致"
            )

        model_to_load.load_state_dict(checkpoint_dict['model_state_dict'])
        optimizer.load_state_dict(checkpoint_dict['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint_dict['scheduler_state_dict'])
        scaler_state_dict = checkpoint_dict['scaler_state_dict']
        if scaler.is_enabled() and scaler_state_dict:
            scaler.load_state_dict(scaler_state_dict)
        elif scaler.is_enabled() and is_main_process():
            print(
                "last.pt 未包含可恢复的 FP16 GradScaler 状态，"
                "本次将重新初始化 scaler"
            )
        start_epoch = int(checkpoint_dict['epoch']) + 1
        best_val_loss = float(checkpoint_dict['best_val_loss'])
        if is_main_process():
            print(
                f"成功恢复状态，将从 Epoch {start_epoch + 1} 继续训练 "
                f"(历史最优 Val Loss: {best_val_loss:.4f})"
            )
    else:
        if is_main_process():
            print("未找到 last.pt，将从头开始随机初始化训练。")
    
    for epoch in range(start_epoch, SIZE_PRED_EPOCHS):
        if is_distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)
            
        model.train()
        total_loss = 0.0
        epoch_max_grad = 0.0
        num_batches = len(train_loader)
        grad_accum_steps = int(config.GRAD_ACCUMULATION_STEPS)
        if grad_accum_steps <= 0:
            raise ValueError("GRAD_ACCUMULATION_STEPS 必须为正整数")
        tail_accum_steps = num_batches % grad_accum_steps
        tail_group_start = (
            num_batches - tail_accum_steps
            if tail_accum_steps
            else num_batches
        )
        
        if is_main_process():
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{SIZE_PRED_EPOCHS} [Train]", leave=False)
        else:
            pbar = train_loader
            
        optimizer.zero_grad(set_to_none=True)
            
        for i, batch in enumerate(pbar):
            batch = batch.to(device)
            
            # 判断当前步是否需要更新权重
            is_update_step = (
                (i + 1) % grad_accum_steps == 0
                or (i + 1) == num_batches
            )
            # 尾部不足一个完整累积组时，该组内每个 micro-batch 必须使用
            # 相同的实际累积数量，避免只有最后一批被按余数缩放。
            current_accum_steps = (
                tail_accum_steps
                if tail_accum_steps and i >= tail_group_start
                else grad_accum_steps
            )
            
            # 【核心逻辑：DDP 梯度累积上下文管理器】
            sync_context = model.no_sync if (not is_update_step) and is_distributed else contextlib.nullcontext

            with sync_context():
                pocket_batch, true_sizes = extract_pocket_subgraph_and_target(batch)
                
                with amp_context():
                    pred_sizes = model(pocket_batch)
                    loss = loss_fn(pred_sizes, true_sizes)
                    loss_scaled = loss / current_accum_steps
                
                scaler.scale(loss_scaled).backward()
            
            # --- 仅在到达累积阈值时更新权重 ---
            if is_update_step:
                scaler.unscale_(optimizer)
                raw_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_threshold)
                current_grad_norm = raw_grad_norm.item() if isinstance(raw_grad_norm, torch.Tensor) else raw_grad_norm
                epoch_max_grad = max(epoch_max_grad, current_grad_norm)
                
                if current_grad_norm > grad_clip_threshold and is_main_process():
                    if isinstance(pbar, tqdm):
                        pbar.write(f"⚠️ [警告] 梯度 Norm 异常: {current_grad_norm:.2f} (已截断)")
                    else:
                        print(f"⚠️ [警告] 梯度 Norm 异常: {current_grad_norm:.2f}")

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            
            total_loss += loss.item()
            if is_main_process() and isinstance(pbar, tqdm) and is_update_step:
                pbar.set_postfix({'loss': f"{loss.item():.4f}", 'grad': f"{current_grad_norm:.1f}"})
                
        # 同步各卡训练 Loss
        train_avg_loss = total_loss / num_batches
        if is_distributed:
            metrics_t = torch.tensor([train_avg_loss, epoch_max_grad], device=device)
            metrics_t = reduce_mean(metrics_t, world_size)
            train_avg_loss = metrics_t[0].item()
            epoch_max_grad = metrics_t[1].item() 
        
        # =====================================================================
        # Validation
        # =====================================================================
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pocket_batch, true_sizes = extract_pocket_subgraph_and_target(batch)
                with amp_context():
                    pred_sizes = model(pocket_batch)
                    v_loss = loss_fn(pred_sizes, true_sizes)
                val_loss += v_loss.item()
                
        # 同步各卡验证 Loss
        val_avg_loss = val_loss / len(val_loader)
        if is_distributed:
            val_t = torch.tensor([val_avg_loss], device=device)
            val_t = reduce_mean(val_t, world_size)
            val_avg_loss = val_t[0].item()
        
        # 【新增：步进学习率调度器】
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()
        
        if is_main_process():
            print(f"Epoch {epoch+1:02d} | LR: {current_lr:.2e} | Max Grad: {epoch_max_grad:.1f} | Train SmoothL1: {train_avg_loss:.4f} | Val SmoothL1: {val_avg_loss:.4f}")
            
            # 【新增】：将关键指标写入 TensorBoard
            if writer:
                writer.add_scalar('Loss/Train', train_avg_loss, epoch)
                writer.add_scalar('Loss/Validation', val_avg_loss, epoch)
                writer.add_scalar('Metrics/Learning_Rate', current_lr, epoch)
                writer.add_scalar('Metrics/Max_Grad_Norm', epoch_max_grad, epoch)
            
            is_best_epoch = val_avg_loss < best_val_loss
            if is_best_epoch:
                best_val_loss = val_avg_loss

            model_to_save = (
                model.module if hasattr(model, 'module') else model
            )
            checkpoint_dict = {
                'epoch': epoch,
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'best_val_loss': best_val_loss,
                'embedding_dim': config.EMBEDDING_DIM_IN,
            }
            # last.pt 始终代表最近完成的 epoch，是唯一断点续训来源。
            torch.save(checkpoint_dict, LAST_CHECKPOINT_PATH)

            if is_best_epoch:
                torch.save(checkpoint_dict, BEST_CHECKPOINT_PATH)
                print(
                    "   -> 新的最佳模型已保存至 best.pt "
                    f"(Val Loss: {best_val_loss:.4f})"
                )

    if is_main_process() and writer:
        writer.close()

    if is_distributed:
        dist.destroy_process_group()

if __name__ == '__main__':
    train_size_predictor_ddp()
