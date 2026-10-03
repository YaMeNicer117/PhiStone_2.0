from pathlib import Path

from PhiSSE3TD_train_finetune_common import (
    FineTuneSettings,
    run_finetuning,
)


# =============================================================================
# 活性候选模式与路径：由本入口独立控制
# =============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
STRATEGY = "activity"

# 基础候选必须是 HAC 不超过该上限的 ligand 节点。
ACTIVITY_CANDIDATE_MAX_HAC = 6

# False：仅按 ligand + HAC 筛选；True：还要求候选节点至少具有下方数量的
# 唯一共价邻居，即启用 survival_linker 使用的 linker 拓扑条件。
ACTIVITY_REQUIRE_LINKER_TOPOLOGY = False
ACTIVITY_MIN_COVALENT_NEIGHBORS = 2

# 即使候选模式不强制 linker 拓扑，也将满足上述邻居阈值的节点作为优先节点。
# 高噪声阶段的最大相对权重从该值开始，随时间步降低逐渐回到 1.0；
# 设为 1.0 可关闭优先加权，不影响固定节点的随机采样。
ACTIVITY_LINKER_PRIORITY_MAX_WEIGHT = 1.0

# 每张图从合格候选节点中随机转为固定节点的比例范围。实际节点数按整数范围
# 采样，并始终至少保留一个候选节点用于预测。
ACTIVITY_FIXED_RATIO_MIN = 0.0
ACTIVITY_FIXED_RATIO_MAX = 0.6

ACTIVITY_CANDIDATE_MODE = (
    "activity_linker"
    if ACTIVITY_REQUIRE_LINKER_TOPOLOGY
    else "activity_modify"
)
#此处调整训练集
DATA_DIR = (
    PROJECT_ROOT
    / "Datas"
    / "Processed_datas"
    / "SE3TD_HiQBind_PDBbind_128_Pygdata"
    / "pyg_pending"
)
SOURCE_CHECKPOINT_PATH = (
    SCRIPT_DIR
    / "training_artifacts"
    / "checkpoints"
    / "diffusion_pretrain"
    / "best_model.pt"
)
ACTIVITY_CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "PhiSGATv2"
    / "training_artifacts"
    / "checkpoints"
    / "best_checkpoint.pt"
)
CHECKPOINT_DIR = (
    SCRIPT_DIR
    / "training_artifacts"
    / "checkpoints"
    / "diffusion_finetune"
    / STRATEGY
)
BEST_CHECKPOINT_PATH = (
    CHECKPOINT_DIR / f"best_model_{ACTIVITY_CANDIDATE_MODE}.pt"
)
LAST_CHECKPOINT_PATH = (
    CHECKPOINT_DIR / f"last_model_{ACTIVITY_CANDIDATE_MODE}.pt"
)
TRAINING_PROFILE_PATH = (
    CHECKPOINT_DIR / f"training_data_profile_{ACTIVITY_CANDIDATE_MODE}.json"
)
TENSORBOARD_DIR = (
    SCRIPT_DIR
    / "training_artifacts"
    / "tensorboard"
    / "diffusion_finetune"
    / STRATEGY
    / ACTIVITY_CANDIDATE_MODE
)
EMBEDDING_TABLE_PATH = (
    PROJECT_ROOT
    / "Datas"
    / "Processed_datas"
    / "SE3TD_128_Shared_data"
    / "fragment_embeddings_128d.npz"
)
EMBEDDING_METADATA_PATH = (
    PROJECT_ROOT
    / "Datas"
    / "Processed_datas"
    / "SE3TD_128_Shared_data"
    / "fragment_embeddings_128d_metadata.json"
)


# =============================================================================
# 训练超参数
# =============================================================================
# 基础训练与数据加载。BATCH_SIZE 是每个 DDP rank 的微批大小；全局有效
# batch 还需乘以 GRAD_ACCUMULATION_STEPS 和 DDP rank 数。
EPOCHS = 60  # 微调总轮数；从 last_model.pt 恢复时包含已完成轮数。
LEARNING_RATE = 1.0e-4  # AdamW 的基准/峰值学习率。
WEIGHT_DECAY = 2.0e-5  # AdamW 权重衰减系数。
SCALAR_DROPOUT = 0.15  # Transformer 标量通道及 HAC/Ring 预测头的 Dropout 率。
BATCH_SIZE = 8  # 每个 DDP rank、每次前向的图数量上限。
GRAD_ACCUMULATION_STEPS = 6  # 累积多少个微批后执行一次优化器更新。
TRAIN_VAL_SPLIT = 0.85  # 可用图中训练集比例，其余作为验证集。
SEED = 42  # 数据划分、固定节点采样和扩散时间步等随机过程的基础种子。
USE_LAZY_DATASET = False  # False 时启动阶段将处理后的图预载入内存。
NUM_WORKERS = 96  # 每个 DDP rank 的 DataLoader 子进程数。
PRELOAD_WORKERS = 16  # 非 lazy 模式下并行预载图数据的线程/进程数。
PIN_MEMORY = True  # 固定 CPU 内存，以加快主机到 CUDA 的异步拷贝。
DIST_BACKEND = "nccl"  # 多 GPU DDP 通信后端。

# True：恢复 last_model 时不沿用旧的最佳验证总损失，调参后的首个完整验证轮
# 将建立新的 best_val_loss 基准。若训练进程之后还可能再次重启，应在首轮成功
# 保存新断点后改回 False，避免每次恢复都再次重置比较基准。
RESET_BEST_VAL_LOSS_ON_RESUME = False

# True：除固定名称的最新最佳模型外，额外保留每次刷新最佳值的历史断点。
SAVE_ALL_BEST_CHECKPOINTS = True

# 学习率调度与数值稳定性。
WARMUP_EPOCHS = 4  # 从起始学习率比例升到基准学习率所用轮数。
HOLD_EPOCHS = 0  # warmup 后保持基准学习率的轮数。
WARMUP_START_LR_RATIO = 0.6  # warmup 起点相对 LEARNING_RATE 的比例。
GRAD_CLIP_WARMUP_NORM = 5.0  # warmup/hold 阶段的全局梯度范数上限。
GRAD_CLIP_STABLE_NORM = 5.0  # 后续余弦衰减阶段的全局梯度范数上限。
SMOOTH_L1_BETA = 1.0  # SmoothL1 从二次区间切换到线性区间的阈值。

# 13 项基础目标的总损失权重；活性损失在下方单独配置。
LOSS_WEIGHTS = {
    "embed_noise": 3.0,  # 128维片段嵌入噪声回归。
    "chem_noise": 0.3,  # 14维化学数值特征噪声回归。
    "pos": 2.0,  # ligand 节点中心坐标噪声回归。
    "frame": 2.0,  # ligand 局部参考原子坐标噪声回归。
    "edge": 2.0,  # 动态 LL/LP/Null 空间关系分类。
    "covalent": 3.0,  # ligand 候选原子对的共价 BCE。
    "frame_ll": 1.0,  # LL 边两端参考原子的距离几何约束。
    "frame_lp": 1.0,  # LP 边两端参考原子的距离几何约束。
    "frame_null": 1.0,  # 采样 Null 边两端参考原子的几何约束。
    "frame_gl": 0.5,  # 自由 ligand 参考原子到本图 Global 中心的径向约束。
    "embed_cos": 1.5,  # 重建 x0 片段嵌入与真值嵌入的余弦一致性。
    "hac": 0.5,  # 重建节点的 HAC 分类损失。
    "ring": 0.5,  # 重建节点的环数量分类损失。
}

# 低噪声辅助目标在 t/T<=0.4 后按余弦曲线从最低权重逐步升到 1。
LOW_NOISE_AUX_START_RATIO = 0.4  # 开始增强辅助目标的归一化时间步边界。
HIGH_NOISE_AUX_WEIGHT_RATIO = 0.05  # 高噪声阶段保留的辅助损失权重下限。
EDGE_CLASS_WEIGHTS = (4.0, 4.0, 4.0)  # LL、LP、Null 三类的 Focal 权重。
FOCAL_LOSS_GAMMA = 2.0  # 动态边 Focal Loss 对容易样本的抑制强度。
COVALENT_POS_WEIGHT_MAX = 12.0  # 共价 BCE 正样本权重的最大截断值。
COVALENT_PROBABILITY_THRESHOLD = 0.6  # 共价验证 P/R/F1 的概率阈值，不参与 BCE。

# 活性项不裁剪评分；硬边阈值只决定 GAT 图，不向 covalent logits 回传梯度。
ACTIVITY_TARGET = 1.05  # 活性标签的最高等级。
ACTIVITY_SMOOTH_L1_BETA = 0.2  # 仅控制活性指导，不影响空间与几何损失。
ACTIVITY_LOSS_WEIGHT = 0.2  # 活性损失加入基础总损失时的系数。
ACTIVITY_NUMERIC_DIM = 11  # GAT 使用14维化学特征中的前11维。
ACTIVITY_COVALENT_PROBABILITY_THRESHOLD = 0.6  # 构造 GAT 预测硬边的概率阈值。


def build_settings() -> FineTuneSettings:
    return FineTuneSettings(
        strategy=STRATEGY,
        data_dir=DATA_DIR,
        source_checkpoint_path=SOURCE_CHECKPOINT_PATH,
        checkpoint_dir=CHECKPOINT_DIR,
        best_checkpoint_path=BEST_CHECKPOINT_PATH,
        last_checkpoint_path=LAST_CHECKPOINT_PATH,
        training_profile_path=TRAINING_PROFILE_PATH,
        tensorboard_dir=TENSORBOARD_DIR,
        embedding_table_path=EMBEDDING_TABLE_PATH,
        embedding_metadata_path=EMBEDDING_METADATA_PATH,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        scalar_dropout=SCALAR_DROPOUT,
        batch_size=BATCH_SIZE,
        grad_accumulation_steps=GRAD_ACCUMULATION_STEPS,
        train_val_split=TRAIN_VAL_SPLIT,
        seed=SEED,
        use_lazy_dataset=USE_LAZY_DATASET,
        num_workers=NUM_WORKERS,
        preload_workers=PRELOAD_WORKERS,
        pin_memory=PIN_MEMORY,
        dist_backend=DIST_BACKEND,
        warmup_epochs=WARMUP_EPOCHS,
        hold_epochs=HOLD_EPOCHS,
        warmup_start_lr_ratio=WARMUP_START_LR_RATIO,
        grad_clip_warmup_norm=GRAD_CLIP_WARMUP_NORM,
        grad_clip_stable_norm=GRAD_CLIP_STABLE_NORM,
        smooth_l1_beta=SMOOTH_L1_BETA,
        loss_weights=LOSS_WEIGHTS,
        low_noise_aux_start_ratio=LOW_NOISE_AUX_START_RATIO,
        high_noise_aux_weight_ratio=HIGH_NOISE_AUX_WEIGHT_RATIO,
        edge_class_weights=EDGE_CLASS_WEIGHTS,
        focal_loss_gamma=FOCAL_LOSS_GAMMA,
        covalent_pos_weight_max=COVALENT_POS_WEIGHT_MAX,
        covalent_probability_threshold=COVALENT_PROBABILITY_THRESHOLD,
        survival_keep_ratio_min=ACTIVITY_FIXED_RATIO_MIN,
        survival_keep_ratio_max=ACTIVITY_FIXED_RATIO_MAX,
        linker_max_hac=ACTIVITY_CANDIDATE_MAX_HAC,
        linker_min_covalent_neighbors=(
            ACTIVITY_MIN_COVALENT_NEIGHBORS
        ),
        survival_max_hac=ACTIVITY_CANDIDATE_MAX_HAC,
        activity_checkpoint_path=ACTIVITY_CHECKPOINT_PATH,
        activity_target=ACTIVITY_TARGET,
        activity_smooth_l1_beta=ACTIVITY_SMOOTH_L1_BETA,
        activity_loss_weight=ACTIVITY_LOSS_WEIGHT,
        activity_numeric_dim=ACTIVITY_NUMERIC_DIM,
        activity_require_linker_topology=(
            ACTIVITY_REQUIRE_LINKER_TOPOLOGY
        ),
        activity_covalent_probability_threshold=(
            ACTIVITY_COVALENT_PROBABILITY_THRESHOLD
        ),
        activity_linker_priority_max_weight=(
            ACTIVITY_LINKER_PRIORITY_MAX_WEIGHT
        ),
        reset_best_val_loss_on_resume=(
            RESET_BEST_VAL_LOSS_ON_RESUME
        ),
        save_all_best_checkpoints=SAVE_ALL_BEST_CHECKPOINTS,
    )


if __name__ == "__main__":
    run_finetuning(build_settings())
