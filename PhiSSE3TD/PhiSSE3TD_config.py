import os
import torch
import torch.nn.functional as F

# =============================================================================
# 1. 系统与环境设置 (System & Environment)
# =============================================================================
# 获取当前脚本所在的绝对路径 (即 PhiStone_2.0_online/PhiSSE3TD/)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 定义项目根目录 (即 PhiStone_2.0_online/)
GNN_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, '..'))

# 随机种子
SEED = 42

# 计算设备
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

#  分布式后端配置 (通常使用 nccl)
DIST_BACKEND = 'nccl'

# =============================================================================
# 2. 数据与维度定义 (Data & Dimensions)
# =============================================================================
# 原始数据中的维度定义
EMBEDDING_DIM_IN = 128     # 新词汇表片段嵌入维度
CHEM_PROPS_DIM_IN = 14      # 化学属性维度 (如原子质量、价键数等物理属性)
TYPE_ENCODING_DIM_IN = 3    # One-hot: [1,0,0]=ligand, [0,1,0]=protein, [0,0,1]=Global

# 核心特征维度 (用于扩散和生成)
# 作用: 扩散模型预测的主要特征部分，包含嵌入和化学属性，不含类型编码
LIGAND_FEATURE_DIM = EMBEDDING_DIM_IN + CHEM_PROPS_DIM_IN 

# =============================================================================
# 3. E(3)NN 模型超参数 (Model Hyperparameters)
# =============================================================================
# 边特征相关
EDGE_ATTR_DIM = 4           # 边属性 One-hot: [Candidate-LL, Candidate-LP, PP, GL]
NUM_BASIS = 24              # 径向基函数 (RBF) 的数量，用于将连续的距离值离散化编码
MAX_EDGE_LENGTH = 12        # Transformer 编码边特征时的最大截断距离 (Angstrom)，超过此距离的相互作用在 Transformer 层被忽略

# 网络架构相关
NUM_TRANSFORMER_LAYERS = 4           # 等变 Transformer 的层数 (深度)，层数越深感受野越大
FC_NEURONS = [96, 64]               # 全连接层 (MLP) 的隐藏层神经元数量，用于处理边权和标量特征
EDGE_PREDICTOR_NEURONS = [192, 96]  # 三分类边头的隐藏层
COVALENT_PREDICTOR_NEURONS = [192, 96]  # LL 节点对共价关系二分类头
TIME_EMB_DIM = 64             # 扩散时间步 (t) 的正弦波嵌入维度
ACTIVATION_FUNCTION = F.silu  # 激活函数 (SiLU / Swish)，在深度学习中表现优于 ReLU

# Irreps (不可约表示) 配置 - 核心等变性参数
# 定义了网络隐藏层中包含多少个标量通道(0e)和向量通道(1o)
TOTAL_HIDDEN_VEC_CHANNELS = 128  # 隐藏层总共分配的向量通道数 (Vector Channels)
REF_COORDS_HIDDEN_CHANNELS = 64  # 其中专门用于编码参考坐标系 (Ref Coords) 的向量通道数
# 剩余 (128 - 32 = 96) 个通道作为"自由向量"，用于学习不依赖特定几何约束的抽象方向特征

HIDDEN_SCALAR_CHANNELS = 224     # 隐藏层标量通道数 

# 正则化
SCALAR_DROPOUT = 0.15        # 标量特征的 Dropout 率，防止过拟合

# =============================================================================
# 4. 扩散过程与动态图参数 (Diffusion & Graph)
# =============================================================================
# 扩散参数
NUM_TIMESTEPS = 1000        # 总扩散步数 T。训练时将数据加噪 T 步，生成时去噪 T 步
BETA_SCHEDULE = 'cosine'    # 噪声调度策略。'cosine' 比 'linear' 能在中间步数保留更多信息，生成质量通常更高

# 裁剪/截断参数 (Clamping)
# 作用: 在采样过程中防止预测值数值爆炸 (Exploding)，保证数值稳定性
FEAT_CLAMP_BUFFER = 20.0         # 特征值裁剪：限制预测的特征值范围
REF_COORDS_CLAMP_BUFFER = 20.0   # 相对参考系裁剪：限制局部坐标系的轴长不超过此值
POS_ABS_SAFETY_CLAMP = 50.0     # 坐标绝对安全截断：仅在没有动态半径边界时防止 x0_pred_pos 数值爆炸
EMBEDDING_SAFETY_CLAMP = 20.0   # Embedding 绝对安全截断：仅防止 x0_pred_feat embedding 维度数值爆炸

# 动态构图参数 (Dynamic Graph Construction)
# 作用: 扩散过程中原子位置在变，需要在每一步重新构建图连接
DYNAMIC_LL_RADIUS = 5.0
DYNAMIC_LP_RADIUS = 7.0

# =============================================================================
# 5. 训练超参数 (Training Hyperparameters)
# =============================================================================
GRAD_ACCUMULATION_STEPS = 3   # 每个优化器更新所累积的 mini-batch 数
BATCH_SIZE = 8                # 批处理大小 (受限于显存，E3NN 计算量较大，通常设较小值)
PIN_MEMORY = True             # 开启锁页内存，加速 CPU 到 GPU 的传输
SMOOTH_L1_BETA = 1.0          # SmoothL1Loss 的 Beta 参数，控制 L1 和 L2 损失的过渡点

# Dataset
TRAIN_VAL_SPLIT = 0.9           # 训练集占比

# --- 数据集加载策略 (Dataset Loading Strategy) ---
# True:  懒加载 (Lazy Loading) - 节省内存，每次从硬盘实时读取，但受限于磁盘 I/O 速度。
# False: 预加载 (Preloaded)    - 消耗大量内存，一次性读入物理内存，训练极快，消除 I/O 瓶颈。
USE_LAZY_DATASET = False
# 智能调整 Worker 数量
NUM_WORKERS = 96  # DataLoader worker 数；需结合 CPU、内存与磁盘吞吐调整

# 阶段一：预训练 (Pre-training) - 几何重建
# ---  Warmup 与梯度裁剪参数 ---
PRETRAIN_EPOCHS = 200            # 预训练轮数
PRETRAIN_WARMUP_EPOCHS = 6      # 阶段 1: 预热期 (LR 从小爬升到 100%)
PRETRAIN_HOLD_EPOCHS = 0        # 阶段 2: 稳定期 (LR 保持 100% 满血输出)
GRAD_CLIP_WARMUP_NORM = 10      # Warmup 期的最大梯度范数（容忍原子散开的剧烈变化）
GRAD_CLIP_STABLE_NORM = 10     # 稳定期的最大梯度范数（保护脆弱的化学几何结构）
WARMUP_START_LR_RATIO = 0.4      # Warmup 从目标学习率的 40% 开始爬升

PRETRAIN_LR = 1e-4              # 初始学习率       
PRETRAIN_WEIGHT_DECAY = 1e-5    # 权重衰减 (L2 正则化)

# --- 当前预训练实际使用的总损失分量权重 ---
# weighted/<name> = W_* × raw/<name>，total_loss 为 13 个 weighted 分量之和。
# 断点恢复时若检测到任一 W_* 权重变化，则保留模型/优化器/学习率进度，
# 仅重置 best_val_loss，使新权重下的验证损失从恢复后的首轮重新建立基准。
RESET_BEST_VAL_LOSS_ON_WEIGHT_CHANGE = True
#
# A. 基础扩散噪声监督：不经过低噪声辅助调度，在全部时间步保持完整强度。
W_EMBED_NOISE = 2.0   # 128 维片段 embedding 噪声预测
W_CHEM_NOISE = 0.2    # 14 维化学属性噪声预测
W_POS_LOSS = 4.0      # 配体节点坐标噪声预测
W_FRAME_LOSS = 2.0    # 三槽局部参考系噪声预测

# B. 低噪声辅助监督：下列数值表示低噪声阶段达到的完整目标权重。
# 每个样本的原始损失还会乘以 low_noise_aux_schedule(t)，高噪声时仅保留
# HIGH_NOISE_AUX_WEIGHT_RATIO，随后在低噪声区间平滑升至 1.0。
W_EDGE_LOSS = 4.0      # 动态 LL/LP/Null 拓扑三分类 Focal Loss
W_COVALENT_LOSS = 1.0  # LL 候选节点对共价关系二分类 BCE
W_FRAME_LL = 1.0       # LL 真边参考原子几何约束
W_FRAME_LP = 1.0       # LP 真边参考原子几何约束
W_FRAME_NULL = 2.0     # Null 候选边参考原子分离约束
W_FRAME_GL = 0.5       # Global-Ligand 参考系几何约束
W_EMBED_COS = 0.5     # 去噪后 128 维 embedding 余弦一致性
W_HAC = 0.5           # 重原子数分类
W_RING = 0.5          # 环数量分类

# 所有辅助任务共用同一低噪声增强曲线：
# 高噪声阶段保持最终目标权重的 5%，在最低噪声的最后 40% 内平滑升至 100%。
LOW_NOISE_AUX_START_RATIO = 0.4
HIGH_NOISE_AUX_WEIGHT_RATIO = 0.05

# --- 当前训练实际使用的损失内部参数（不是总损失分量权重）---
# 动态拓扑类别顺序: [LL, LP, Null]
EDGE_CLASS_WEIGHTS = [4.0, 4.0, 6.0]
FOCAL_LOSS_GAMMA = 2.0          # Focal Loss 难样本聚焦指数，不改变总损失分量名称
COVALENT_POS_WEIGHT_MAX = 12.0 # 数据画像计算共价 BCE 正样本权重时的上限
COVALENT_PROB_THRESHOLD = 0.4  # 仅用于共价 precision/recall/F1 的二值判定

# =============================================================================
# 6. 路径配置 (Path Configurations)
# =============================================================================

# --- A. 输入数据路径（统一位于 Datas）---
PROCESSED_DATA_ROOT = os.path.join(GNN_ROOT, 'Datas', 'Processed_datas')
WORK_DATA_ROOT = os.path.join(GNN_ROOT, 'Datas', 'work_datas')

# 128 维片段词嵌入表。PyGData 只保存逐节点规范 SMILES，训练加载时在
# 单图 Transform 中查询该表并生成运行时 frag_embeds。
SE3TD_SHARED_DATA_DIR = os.path.join(
    PROCESSED_DATA_ROOT, 'SE3TD_128_Shared_data'
)
FRAGMENT_EMBEDDING_TABLE_PATH = os.path.join(
    SE3TD_SHARED_DATA_DIR, 'fragment_embeddings_128d.npz'
)
FRAGMENT_EMBEDDING_METADATA_PATH = os.path.join(
    SE3TD_SHARED_DATA_DIR, 'fragment_embeddings_128d_metadata.json'
)

# 训练用的全量数据 (包含配体和蛋白图)。目录名 pyg_pending 表示尚未
# 离线回填 embedding；这些数据在训练加载时会完成 SMILES -> 128 维解析。
PYG_DATA_DIR = os.path.join(
    PROCESSED_DATA_ROOT,
    'SE3TD_HiQBind_PDBbind_128_Pygdata', 'pyg_pending'
)
# 测试用的独立数据
PYG_TEST_DATA_DIR = os.path.join(PROCESSED_DATA_ROOT, 'SE3TD_test_datas')

# [生成任务] 各类输入数据路径
# Survival 模式: 从 TASK_A 目录读取 complex.pt + pocket.pdb
# Creative 模式: 从 TASK_B 目录读取 {id}.pt + pocket.pdb
TASK_A_DATA_DIR = os.path.join(WORK_DATA_ROOT, 'TASK_A')
TASK_B_DATA_DIR = os.path.join(WORK_DATA_ROOT, 'TASK_B')

# 定义各个词汇表的路径
VOCAB_BASE_DIR = SE3TD_SHARED_DATA_DIR
VOCAB_FILES_MAP = {
    'puppy':  os.path.join(VOCAB_BASE_DIR, 'fragment_vocabulary_puppy.json'),
    'linker': os.path.join(VOCAB_BASE_DIR, 'fragment_vocabulary_linker.json'),
    'frame':  os.path.join(VOCAB_BASE_DIR, 'fragment_vocabulary_frame.json'),
    'filter': os.path.join(VOCAB_BASE_DIR, 'fragment_vocabulary_filter.json'),
    'valid': os.path.join(VOCAB_BASE_DIR, 'fragment_vocabulary_valid.json'),
    'ol': os.path.join(VOCAB_BASE_DIR, 'fragment_vocabulary_64_pca.json')
}

# --- B. 训练产物路径（统一位于 PhiSSE3TD/training_artifacts）---
TRAINING_ARTIFACTS_ROOT = os.path.join(PROJECT_ROOT, 'training_artifacts')
CHECKPOINTS_ROOT = os.path.join(TRAINING_ARTIFACTS_ROOT, 'checkpoints')
TENSORBOARD_ROOT = os.path.join(TRAINING_ARTIFACTS_ROOT, 'tensorboard')

# 扩散模型预训练
CHECKPOINT_DIR_PRETRAIN = os.path.join(CHECKPOINTS_ROOT, 'diffusion_pretrain')
LOG_DIR_PRETRAIN = os.path.join(TENSORBOARD_ROOT, 'diffusion_pretrain')
BEST_MODEL_PATH_PRETRAIN = os.path.join(CHECKPOINT_DIR_PRETRAIN, 'best_model.pt')
TRAINING_PROFILE_PATH = os.path.join(CHECKPOINT_DIR_PRETRAIN, 'training_data_profile.json')

# 扩散模型微调（当前训练流程尚未启用，先保留规范路径）
CHECKPOINT_DIR_FINETUNE = os.path.join(CHECKPOINTS_ROOT, 'diffusion_finetune')
LOG_DIR_FINETUNE_SA = os.path.join(TENSORBOARD_ROOT, 'diffusion_finetune')
BEST_MODEL_PATH_FINETUNE_SA = os.path.join(CHECKPOINT_DIR_FINETUNE, 'best_model.pt')

# 配体节点数量预测器
CHECKPOINT_DIR_SIZE_PREDICTOR = os.path.join(
    CHECKPOINTS_ROOT, 'ligand_size_predictor'
)
LOG_DIR_SIZE_PREDICTOR = os.path.join(
    TENSORBOARD_ROOT, 'ligand_size_predictor'
)
SIZE_PREDICTOR_TRAIN_MODEL_PATH = os.path.join(
    CHECKPOINT_DIR_SIZE_PREDICTOR, 'best_model.pt'
)

# --- C. 旧推理与生成路径（保留，等待新推理脚本迁移）---
SAVE_ROOT = os.path.join(PROJECT_ROOT, 'processed_data')

# 本阶段推理脚本尚未迁移到新 PyG 接口，因此继续路由旧 64 维权重。
LEGACY_BEST_MODEL_PATH_PRETRAIN = os.path.join(
    SAVE_ROOT, 'checkpoints_pretrained_64', 'best_pretrained_model_64.pt'
)
LEGACY_BEST_MODEL_PATH_FINETUNE_SA = os.path.join(
    SAVE_ROOT, 'checkpoints_finetuned_64', 'best_finetuned_SA_model_64.pt'
)
MODEL_WEIGHT_ROUTES = {
    'PRETRAIN':    LEGACY_BEST_MODEL_PATH_PRETRAIN,
    'FINETUNE_SA': LEGACY_BEST_MODEL_PATH_FINETUNE_SA,
}

# 旧推理入口继续读取 64 维节点数量预测器。
SIZE_PREDICTOR_MODEL_PATH = os.path.join(
    SAVE_ROOT, 'size_predictor_64', 'pocket_size_predictor_64.pt'
)

# 生成结果 (SDF/PDB) 输出路径
OUTPUT_DIR_GENERATION = os.path.join(SAVE_ROOT, 'generated_molecules')
OUTPUT_DIR_GENERATION2 = os.path.join(SAVE_ROOT, 'generated_molecules')

# --- C. 外部指导模型路径 (PhiSGATv2) ---
# 位于 PhiStone_2.0_online/PhiSGATv2
PHISGAT_ROOT = os.path.join(GNN_ROOT, 'PhiSGATv2')
# 如果模型权重在 PhiSGATv2/processed_data 下，微调脚本中需引用此路径
PHISGAT_DATA_DIR = os.path.join(PHISGAT_ROOT, 'processed_data')
