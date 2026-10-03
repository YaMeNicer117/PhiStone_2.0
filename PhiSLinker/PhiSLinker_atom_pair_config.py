"""Configuration for the PhiStone inter-fragment atom-pair predictor.

The values in the user-tunable section are deliberately plain module-level
constants so they can be edited on the training server without changing the
model, dataset, or training code.  Checkpoints also contain a complete
snapshot of the resolved dataclasses below.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


# =============================================================================
# Paths
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
PROCESSED_DATA_ROOT = PROJECT_ROOT / "Datas" / "Processed_datas"
SHARED_DATA_DIR = PROCESSED_DATA_ROOT / "SE3TD_128_Shared_data"

PYG_DATA_DIR = (
    PROCESSED_DATA_ROOT
    / "SE3TD_HiQBind_PDBbind_128_Pygdata"
    / "pyg_pending"
)
FRAGMENT_VOCAB_PATH = SHARED_DATA_DIR / "fragment_embeddings_128d.npz"
ATOM_VOCAB_PATH = SHARED_DATA_DIR / "fragment_atom_embeddings_64d.npz"

OUTPUT_ROOT = SCRIPT_DIR / "training_artifacts"
CHECKPOINT_DIR = OUTPUT_ROOT / "checkpoints"
TENSORBOARD_DIR = OUTPUT_ROOT / "tensorboard"
SPLIT_MANIFEST_PATH = OUTPUT_ROOT / "split_manifest.json"
PREFLIGHT_REPORT_PATH = OUTPUT_ROOT / "preflight_report.json"

ATOM_PAIR_SUPERVISION_POLICY = "single_atom_pair_with_global_capacity"


# =============================================================================
# User-tunable model and objective weights
# =============================================================================

# 节点对内部单原子对排序损失权重。
LOCAL_LOSS_WEIGHT = 1.0
# 合法错误全局组合与真实组合之间的排序损失权重。
VALID_COMBINATION_LOSS_WEIGHT = 1.5
# 价态冲突组合与真实组合之间的排序损失权重。
CONFLICT_COMBINATION_LOSS_WEIGHT = 1.0
# 合法错误组合的基础间隔。
VALID_COMBINATION_MARGIN = 0.5
# 每溢出一次原子容量追加的间隔。
CONFLICT_OVERFLOW_MARGIN = 1.0
# 训练时按 query 整体屏蔽候选原子几何特征的概率。
GEOMETRY_DROPOUT = 0.4

# 固定片段词表嵌入的维度。
FRAGMENT_EMBEDDING_DIM = 128
# 固定原子词表嵌入的维度。
ATOM_EMBEDDING_DIM = 64
# 节点类型 one-hot 向量的维度（ligand/pocket/global）。
NODE_TYPE_DIM = 3
# 图神经网络中片段节点隐藏状态的维度。
NODE_HIDDEN_DIM = 192
# 边特征经过边编码器后的隐藏维度。
EDGE_HIDDEN_DIM = 64
# 原子嵌入与片段上下文融合后的隐藏维度。
ATOM_HIDDEN_DIM = 128
# 边感知图消息传递层数。
NUM_GNN_LAYERS = 3
# 模型 MLP 和残差分支使用的 dropout 概率。
DROPOUT = 0.15

# 距离高斯径向基函数的中心数量。
RBF_BINS = 16
# RBF 中心覆盖范围及标量距离截断上限，单位为埃。
RBF_MAX_DISTANCE = 12.0
# 缓存的片段 3D 构象模板上限；设为 0 时不执行容量淘汰。
CONFORMER_TEMPLATE_CACHE_SIZE = 4096


# =============================================================================
# User-tunable inference parameters
# =============================================================================

# 非固定原子对进入全局组合搜索所需的最低 Softmax 概率。
ATOM_PAIR_MIN_PROBABILITY = 0.3
# 对满足最低概率的全局组合，优先让同一节点使用更多不同原子。
GLOBAL_PREFER_DISTINCT_NODE_ATOMS = True
# 每个非固定 query 进入全局组合搜索的局部候选数。
GLOBAL_TOP_K = 12
# 全局组合搜索 Beam 宽度。
GLOBAL_BEAM_WIDTH = 128
# 每张训练图保留的合法错误组合上限。
GLOBAL_MAX_LEGAL_NEGATIVES = 24
# 每张训练图保留的价态冲突组合上限。
GLOBAL_MAX_CONFLICT_NEGATIVES = 24
# 推理时输出并正式验证的合法全局组合数。
GLOBAL_INFERENCE_COMBINATIONS = 16
# 全局微调中启用固定 query 场景的概率。
FIXED_QUERY_SCENARIO_PROBABILITY = 0.6
FIXED_QUERY_MIN_FRACTION = 0.30
FIXED_QUERY_MAX_FRACTION = 0.80


# =============================================================================
# User-tunable training parameters
# =============================================================================

# Default stage used by ``python PhiSLinker_atom_pair_train.py``.
# Use "local" for node-pair pretraining or "global" for global fine-tuning.
TRAINING_STAGE = "global"

# 数据划分、批次打乱和模型初始化使用的随机种子。
RANDOM_SEED = 42
# 图文件中用于训练集的比例，其余用于验证集。
TRAIN_FRACTION = 0.85
# 最大训练轮数。
EPOCHS = 200
# AdamW 优化器的初始学习率。
LEARNING_RATE = 1.0e-4
# AdamW 优化器的权重衰减系数。
WEIGHT_DECAY = 1.0e-5
# 反向传播后允许的最大梯度范数。
GRADIENT_CLIP_NORM = 5.0
# 验证总损失连续未改善多少轮后提前停止。
EARLY_STOPPING_PATIENCE = 16
# 全局组合微调的最大训练轮数、学习率和早停耐心。
GLOBAL_EPOCHS = 60
GLOBAL_LEARNING_RATE = 2.0e-5
GLOBAL_EARLY_STOPPING_PATIENCE = 16
# ReduceLROnPlateau 触发时的学习率乘数。
SCHEDULER_FACTOR = 0.72
# 验证总损失连续未改善多少轮后降低学习率。
SCHEDULER_PATIENCE = 4
# 学习率调度器允许降低到的最小学习率。
MIN_LEARNING_RATE = 1.0e-6

# 单个批次最多包含的完整图数量。
MAX_GRAPHS_PER_BATCH = 128
# 单个批次的候选原子对预算；不会截断单个超预算图。
MAX_CANDIDATES_PER_BATCH = 10240
# DataLoader 工作进程数；0 表示在主进程加载。
NUM_WORKERS = 16
# 使用 CUDA 时是否启用 DataLoader pinned memory。
PIN_MEMORY = True
# 使用 CUDA 时是否启用自动混合精度训练。
USE_AMP = True


@dataclass(frozen=True)
class PathConfig:
    pyg_data_dir: Path = PYG_DATA_DIR
    fragment_vocab_path: Path = FRAGMENT_VOCAB_PATH
    atom_vocab_path: Path = ATOM_VOCAB_PATH
    output_root: Path = OUTPUT_ROOT
    checkpoint_dir: Path = CHECKPOINT_DIR
    tensorboard_dir: Path = TENSORBOARD_DIR
    split_manifest_path: Path = SPLIT_MANIFEST_PATH
    preflight_report_path: Path = PREFLIGHT_REPORT_PATH

    def validate_inputs(self) -> None:
        if not self.pyg_data_dir.is_dir():
            raise FileNotFoundError(f"PyG data directory not found: {self.pyg_data_dir}")
        for label, path in (
            ("fragment vocabulary", self.fragment_vocab_path),
            ("atom vocabulary", self.atom_vocab_path),
        ):
            if not path.is_file():
                raise FileNotFoundError(f"{label} not found: {path}")

    def create_output_directories(self) -> None:
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.tensorboard_dir.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class ModelConfig:
    fragment_embedding_dim: int = FRAGMENT_EMBEDDING_DIM
    atom_embedding_dim: int = ATOM_EMBEDDING_DIM
    node_type_dim: int = NODE_TYPE_DIM
    node_hidden_dim: int = NODE_HIDDEN_DIM
    edge_hidden_dim: int = EDGE_HIDDEN_DIM
    atom_hidden_dim: int = ATOM_HIDDEN_DIM
    num_gnn_layers: int = NUM_GNN_LAYERS
    dropout: float = DROPOUT
    rbf_bins: int = RBF_BINS
    rbf_max_distance: float = RBF_MAX_DISTANCE
    geometry_dropout: float = GEOMETRY_DROPOUT

    @property
    def node_input_dim(self) -> int:
        return self.fragment_embedding_dim + self.node_type_dim

    @property
    def edge_input_dim(self) -> int:
        # LL / LP / PP / KNOWN_CONNECTION plus centroid-distance RBF.
        return 4 + self.rbf_bins

    @property
    def pair_geometry_dim(self) -> int:
        return self.rbf_bins + 2

    @property
    def pair_input_dim(self) -> int:
        return 3 * self.atom_hidden_dim + self.pair_geometry_dim

    def validate(self) -> None:
        integer_values = {
            "fragment_embedding_dim": self.fragment_embedding_dim,
            "atom_embedding_dim": self.atom_embedding_dim,
            "node_type_dim": self.node_type_dim,
            "node_hidden_dim": self.node_hidden_dim,
            "edge_hidden_dim": self.edge_hidden_dim,
            "atom_hidden_dim": self.atom_hidden_dim,
            "num_gnn_layers": self.num_gnn_layers,
            "rbf_bins": self.rbf_bins,
        }
        for name, value in integer_values.items():
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not 0.0 <= self.geometry_dropout < 1.0:
            raise ValueError("geometry_dropout must be in [0, 1)")
        if self.rbf_max_distance <= 0.0:
            raise ValueError("rbf_max_distance must be positive")


@dataclass(frozen=True)
class TrainConfig:
    random_seed: int = RANDOM_SEED
    train_fraction: float = TRAIN_FRACTION
    epochs: int = EPOCHS
    learning_rate: float = LEARNING_RATE
    global_epochs: int = GLOBAL_EPOCHS
    global_learning_rate: float = GLOBAL_LEARNING_RATE
    weight_decay: float = WEIGHT_DECAY
    gradient_clip_norm: float = GRADIENT_CLIP_NORM
    early_stopping_patience: int = EARLY_STOPPING_PATIENCE
    global_early_stopping_patience: int = GLOBAL_EARLY_STOPPING_PATIENCE
    scheduler_factor: float = SCHEDULER_FACTOR
    scheduler_patience: int = SCHEDULER_PATIENCE
    min_learning_rate: float = MIN_LEARNING_RATE
    max_graphs_per_batch: int = MAX_GRAPHS_PER_BATCH
    max_candidates_per_batch: int = MAX_CANDIDATES_PER_BATCH
    num_workers: int = NUM_WORKERS
    pin_memory: bool = PIN_MEMORY
    use_amp: bool = USE_AMP
    local_loss_weight: float = LOCAL_LOSS_WEIGHT
    valid_combination_loss_weight: float = VALID_COMBINATION_LOSS_WEIGHT
    conflict_combination_loss_weight: float = CONFLICT_COMBINATION_LOSS_WEIGHT
    valid_combination_margin: float = VALID_COMBINATION_MARGIN
    conflict_overflow_margin: float = CONFLICT_OVERFLOW_MARGIN
    global_top_k: int = GLOBAL_TOP_K
    global_beam_width: int = GLOBAL_BEAM_WIDTH
    global_max_legal_negatives: int = GLOBAL_MAX_LEGAL_NEGATIVES
    global_max_conflict_negatives: int = GLOBAL_MAX_CONFLICT_NEGATIVES
    global_inference_combinations: int = GLOBAL_INFERENCE_COMBINATIONS
    fixed_query_scenario_probability: float = FIXED_QUERY_SCENARIO_PROBABILITY
    fixed_query_min_fraction: float = FIXED_QUERY_MIN_FRACTION
    fixed_query_max_fraction: float = FIXED_QUERY_MAX_FRACTION

    def validate(self) -> None:
        if not 0.0 < self.train_fraction < 1.0:
            raise ValueError("train_fraction must be in (0, 1)")
        for name in (
            "epochs",
            "global_epochs",
            "early_stopping_patience",
            "global_early_stopping_patience",
            "scheduler_patience",
            "max_graphs_per_batch",
            "max_candidates_per_batch",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.global_learning_rate <= 0.0:
            raise ValueError("global_learning_rate must be positive")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay cannot be negative")
        if self.gradient_clip_norm <= 0.0:
            raise ValueError("gradient_clip_norm must be positive")
        if not 0.0 < self.scheduler_factor < 1.0:
            raise ValueError("scheduler_factor must be in (0, 1)")
        if self.min_learning_rate <= 0.0:
            raise ValueError("min_learning_rate must be positive")
        loss_weights = (
            self.local_loss_weight,
            self.valid_combination_loss_weight,
            self.conflict_combination_loss_weight,
        )
        if any(value < 0.0 for value in loss_weights):
            raise ValueError("loss weights cannot be negative")
        if self.local_loss_weight <= 0.0:
            raise ValueError("local_loss_weight must be positive")
        if self.valid_combination_margin < 0.0:
            raise ValueError("valid_combination_margin cannot be negative")
        if self.conflict_overflow_margin < 0.0:
            raise ValueError("conflict_overflow_margin cannot be negative")
        for name in (
            "global_top_k",
            "global_beam_width",
            "global_max_legal_negatives",
            "global_max_conflict_negatives",
            "global_inference_combinations",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 <= self.fixed_query_scenario_probability <= 1.0:
            raise ValueError(
                "fixed_query_scenario_probability must be in [0, 1]"
            )
        if not (
            0.0
            <= self.fixed_query_min_fraction
            <= self.fixed_query_max_fraction
            <= 1.0
        ):
            raise ValueError(
                "fixed query fractions must satisfy 0 <= min <= max <= 1"
            )


DEFAULT_PATH_CONFIG = PathConfig()
DEFAULT_MODEL_CONFIG = ModelConfig()
DEFAULT_TRAIN_CONFIG = TrainConfig()


def configuration_snapshot(
    path_config: PathConfig,
    model_config: ModelConfig,
    train_config: TrainConfig,
) -> dict[str, Any]:
    """Return a JSON-serializable checkpoint configuration snapshot."""

    paths = {key: str(value) for key, value in asdict(path_config).items()}
    return {
        "paths": paths,
        "model": asdict(model_config),
        "training": asdict(train_config),
        "derived_dimensions": {
            "node_input_dim": model_config.node_input_dim,
            "edge_input_dim": model_config.edge_input_dim,
            "pair_input_dim": model_config.pair_input_dim,
        },
    }


DEFAULT_MODEL_CONFIG.validate()
DEFAULT_TRAIN_CONFIG.validate()
