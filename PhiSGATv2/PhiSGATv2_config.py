"""PhiSGATv2 单任务二维活性回归的集中配置。"""

import math
import os
import torch


# =============================================================================
# 路径配置
# =============================================================================

# 当前 GATv2 代码目录与项目根目录。所有默认路径都从脚本位置推导，
# 因此从任意工作目录启动训练或预测都使用同一套数据位置。
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(BASE_DIR, ".."))
DATA_ROOT = os.path.join(PROJECT_ROOT, "Datas")
PROCESSED_DATA_ROOT = os.path.join(DATA_ROOT, "Processed_datas")

# PhiSSeparator_SE3TD_2D.py 生成的有标签 PyG 根目录。Dataset 会递归读取
# 其所有子目录中的 .pt 文件，而不是只读取根目录的直接子文件。
TRAIN_PYG_DIR = os.path.join(
    PROCESSED_DATA_ROOT,
    "SE3TD_2D_128_Pygdata",
    "pyg_pending",
)

# 未知活性原始文件目录。工作数据脚本会递归读取其中的 CSV、Excel 和 SDF。
WORK_INPUT_DIR = os.path.join(
    DATA_ROOT,
    "work_datas",
    "input_unknownactivity",
)

# 未知活性 PyG 输出目录。不会自动清空；预测脚本默认递归预测其中全部 .pt。
WORK_PYG_DIR = os.path.join(
    DATA_ROOT,
    "work_datas",
    "processed_2d",
)

# 模型训练产物目录。检查点与数据划分清单均放在数据区，避免与源码混放。
MODEL_OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "training_artifacts",
)
CHECKPOINT_DIR = os.path.join(MODEL_OUTPUT_DIR, "checkpoints")
BEST_CHECKPOINT_PATH = os.path.join(
    CHECKPOINT_DIR,
    "best_checkpoint.pt",
)
LAST_CHECKPOINT_PATH = os.path.join(
    CHECKPOINT_DIR,
    "last_checkpoint.pt",
)
SPLIT_MANIFEST_PATH = os.path.join(
    MODEL_OUTPUT_DIR,
    "train_val_split.json",
)

# TensorBoard 日志根目录。每次训练会在此目录下创建独立运行目录，避免不同
# 训练过程的事件文件相互混写。
TENSORBOARD_LOG_DIR = os.path.join(MODEL_OUTPUT_DIR, "tensorboard")

# 预测脚本直接运行时覆盖写入的完整结果与逐文件错误报告。
PREDICTION_OUTPUT_DIR = os.path.join(
    DATA_ROOT,
    "work_datas",
    "PhiSGATv2_prediction_results",
)
PREDICTION_RESULT_PATH = os.path.join(
    PREDICTION_OUTPUT_DIR,
    "prediction_results.csv",
)
PREDICTION_ERROR_PATH = os.path.join(
    PREDICTION_OUTPUT_DIR,
    "prediction_errors.csv",
)

# 直接运行预测脚本时生成的二维分子结构图。每张图最多 36 个分子，
# 默认按 6×6 排列；单个分子图片尺寸越大，文字和复杂结构越清晰，
# 但整张图片的尺寸与内存占用也会相应增加。
PREDICTION_IMAGE_DIR = os.path.join(
    PREDICTION_OUTPUT_DIR,
    "molecule_images",
)
PREDICTION_IMAGE_FILENAME_PREFIX = "prediction_molecules"
PREDICTION_MAX_MOLS_PER_IMAGE = 36
PREDICTION_MOLS_PER_ROW = 6
PREDICTION_IMAGE_SUB_SIZE = (400, 400)


# =============================================================================
# 数据与随机性配置
# =============================================================================

# 完全随机划分和模型初始化使用的非负整数种子。修改后会改变新训练的数据划分、
# 初始权重与批次顺序；断点续训始终恢复检查点中的原划分和随机状态。
RANDOM_SEED = 42

# 训练/验证比例，有效范围均为 (0, 1) 且总和必须为 1；训练样本数最终会限制在
# [1, N-1]。
# 增大 TRAIN_RATIO 会提供更多训练样本，但减少用于监测泛化误差的验证样本。
TRAIN_RATIO = 0.85
VAL_RATIO = 0.15

# PyG DataLoader 每批图数量，必须是正整数。增大通常提高吞吐量，但会增加
# 显存/内存占用；它不改变模型结构或检查点兼容性。
BATCH_SIZE = 224

# DataLoader 子进程数，有效范围为大于等于 0 的整数。Windows 下复制大嵌入
# 映射的代价较高，默认 0 表示在主进程加载；调高可能增加吞吐量和内存占用，
# 但 Resolver 仍只允许在主进程预先执行。
DATA_LOADER_WORKERS = 16

# 训练拼批时不需要进入模型的字段；它们仍保留在原始 .pt 文件中。
TRAIN_EXCLUDE_KEYS = [
    "edge_attr",
    "compound_id",
    "smiles",
    "fragment_smiles",
    "metal_mask",
    "hac",
    "ring_count",
    "pyg_file_path",
]

# 预测时保留 compound_id、smiles 与 pyg_file_path 便于生成结果表，其余未参与
# 模型计算的节点元数据不进入批对象。
PREDICT_EXCLUDE_KEYS = [
    "edge_attr",
    "fragment_smiles",
    "metal_mask",
    "hac",
    "ring_count",
    "y",
]

# 训练标签是有限连续值，必须位于闭区间 [0, 1]。这两个边界只用于
# 训练数据校验，不限制推理结果；模型输出仍使用下方更宽的软边界。
TRAIN_ACTIVITY_MIN = 0.0
TRAIN_ACTIVITY_MAX = 1.0

# 验证训练/验证比例之和时使用的浮点容差，必须为非负数；通常无需修改。
RATIO_SUM_TOLERANCE = 1e-8


# =============================================================================
# 模型结构超参数
# =============================================================================

# PhiSSeparator 生成的归一化数值节点特征维度，必须为正整数并与 data.x
# 最后一维一致；改变后会使旧检查点不兼容。
INPUT_DIM_NUMERIC = 11

# 片段编码器的冻结输出维度，必须为正整数并与 Resolver 一致；改变后会使旧
# 检查点不兼容。
EMBEDDING_DIM = 128

# 每个注意力头的输出通道数。与 HEADS 相乘得到每个 GATv2 层的最终维度；
# 增大可提高容量，但会增加参数量、显存占用和过拟合风险。
HIDDEN_CHANNELS = 64

# 多头注意力头数。当前使用 concat=True，因此最终节点维度为 64*4=256。
# 增大头数可表达更多邻域权重模式，但单一共价边图通常不需要过多头。
HEADS = 4

# 真实化学边上的局部消息传递层数。启用虚拟节点时，每层会补充全图信息；
# 关闭虚拟节点时，节点信息仅沿真实边传播。增加层数可能引入过平滑。
NUM_LAYERS = 3

# 图级回归 MLP 的中间维度，必须为正整数。增大可提高回归头容量和参数量，但
# 不改变图消息传递能力；改变后会使旧检查点不兼容。
MLP_HIDDEN_DIM = 128

# 每层激活后的随机失活比例，有效范围 [0, 1)。增大可缓解过拟合，但过高会
# 降低小数据集的收敛速度。
DROPOUT = 0.15

# 是否启用完整虚拟节点分支。False 用于消融：不创建虚拟节点及融合门控，
# 跳过全局均值汇聚、节点广播和最终虚拟特征融合，保留 GATv2 与注意力池化。
USE_VIRTUAL_NODE = True

# 仅在 USE_VIRTUAL_NODE=True 时参与模型计算。
# 虚拟节点向配体节点及最终图表示注入全局信息时的门控初始偏置，必须是有限
# 数值。-2 经 sigmoid 后约为 0.12，使模型初期仍以真实边和注意力池化为主。
VIRTUAL_NODE_GATE_BIAS = -2.0

# 连续预测的软边界。模型使用 MIN + RANGE*sigmoid，因此输出严格位于
# (-0.1, 1.1)，而训练标签 0~1 不落在饱和边界上。
OUTPUT_MIN = -0.05
OUTPUT_MAX = 1.05

# 由上述参数推导，不应独立手动修改。
COMBINED_INPUT_DIM = INPUT_DIM_NUMERIC + EMBEDDING_DIM
FINAL_NODE_DIM = HIDDEN_CHANNELS * HEADS
OUTPUT_RANGE = OUTPUT_MAX - OUTPUT_MIN


# =============================================================================
# 训练超参数
# =============================================================================

# 最大训练轮数，必须为正整数。余弦学习率调度器以该值作为完整下降周期；增大
# 会延长下降周期和最长训练时间。
EPOCHS = 160

# AdamW 初始学习率，必须大于 0。增大可加快前期学习，但过大容易产生不稳定
# 或非有限梯度，并且必须不小于 MIN_LEARNING_RATE。
# 或非有限梯度，并且必须不小于 MIN_LEARNING_RATE。
LEARNING_RATE = 3e-4

# 余弦退火最低学习率，必须小于等于初始学习率；过低会使训练后期几乎停止。
MIN_LEARNING_RATE = 1e-5

# AdamW 权重衰减，有效范围为大于等于 0。增大可增强正则化，但过大会限制
# 模型拟合能力。
WEIGHT_DECAY = 1.5e-5

# Smooth L1 损失由二次区间切换到线性区间的误差阈值，必须大于 0。
# 设为 0.15 可降低跨文献异常误差对梯度的影响，同时保留小误差处的平滑优化。
SMOOTH_L1_BETA = 0.2

# 混合损失中辅助 MSE 的权重，有效范围为 [0, 1)。可适度增强大误差
# 样本的梯度，同时仍以 Smooth L1 的稳健性为主。
MSE_AUX_WEIGHT = 0.65

# 梯度 L2 范数上限，必须大于 0。超过后执行裁剪；降低可增强稳定性，但过低
# 会减慢学习。
GRAD_CLIP_MAX_NORM = 5.0

# 验证损失连续多少轮没有达到最小改善量后停止。增大更不容易过早停止，但会
# 延长无收益训练时间。
EARLY_STOPPING_PATIENCE = 15

# 判断验证损失真正改善所需的非负绝对差值；调大可过滤波动，但也更容易提前
# 触发早停。它与 EARLY_STOPPING_PATIENCE 共同决定停止条件。
EARLY_STOPPING_MIN_DELTA = 1e-5

# 是否从完整检查点续训。为 True 时使用 RESUME_CHECKPOINT_PATH，并恢复原划分、
# 优化器、调度器、早停计数和随机状态。
RESUME_TRAINING = False
RESUME_CHECKPOINT_PATH = LAST_CHECKPOINT_PATH

# 每隔多少个训练 batch 打印一次损失和裁剪前梯度范数，有效范围为大于等于 0
# 的整数；设为 0 可关闭批次日志，不影响训练结果。
TRAIN_LOG_INTERVAL = 100


# =============================================================================
# 片段嵌入 Resolver 与工作数据处理配置
# =============================================================================

# Resolver 固定在 CPU 运行，避免缺词编码模型与 GATv2 争用 GPU 显存。
RESOLVER_DEVICE = "cpu"

# Resolver CPU 推理请求线程数与硬上限，均必须为正整数。增大可能提高缺词编码
# 吞吐量，也会增加 CPU 竞争；实际值还受 Slurm 分配核数或系统可用核数限制。
RESOLVER_CPU_THREADS = 12
MAX_RESOLVER_CPU_THREADS = 24

# 未知活性结构切割的请求进程数和硬上限，均必须为正整数。它们独立于 Resolver
# 的内部线程数；增大可加快大量分子切割，但会近似线性增加 RDKit 进程内存。
WORK_PROCESS_WORKERS = 12
MAX_WORK_PROCESS_WORKERS = 24

# 单个未知活性分子的切割超时秒数，必须大于 0。增大可容忍复杂结构，过大会
# 延迟异常任务回收。
WORK_TASK_TIMEOUT_SECONDS = 240

# 未知活性输出文件名中 Compound ID 的最大字符数，必须大于 8。增大可保留更长
# ID，但会增加触发 Windows 总路径长度限制的风险。
WORK_OUTPUT_ID_MAX_LENGTH = 120


# =============================================================================
# 运行设备
# =============================================================================

# GATv2 训练/预测设备；有可用 CUDA 时默认使用 GPU，否则退回 CPU。
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def allocated_cpu_count():
    """返回调度器分配核数或本机可用逻辑核数。"""
    slurm_value = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_value is not None:
        try:
            parsed = int(slurm_value)
        except (TypeError, ValueError):
            parsed = 0
        if parsed > 0:
            return parsed
    if hasattr(os, "sched_getaffinity"):
        try:
            affinity_count = len(os.sched_getaffinity(0))
        except (OSError, TypeError):
            affinity_count = 0
        if affinity_count > 0:
            return affinity_count
    return max(int(os.cpu_count() or 1), 1)


def clamp_cpu_count(requested, hard_max, parameter_name):
    """将用户请求的线程/进程数限制到配置上限与实际可用核数。"""
    if isinstance(requested, bool) or not isinstance(requested, int):
        raise TypeError(f"{parameter_name} 必须是正整数")
    if requested < 1:
        raise ValueError(f"{parameter_name} 必须至少为 1")
    if isinstance(hard_max, bool) or not isinstance(hard_max, int):
        raise TypeError(f"{parameter_name} 的硬上限必须是正整数")
    if hard_max < 1:
        raise ValueError(f"{parameter_name} 的硬上限必须至少为 1")
    return min(requested, hard_max, allocated_cpu_count())


def validate_config():
    """在训练或预测启动时检查相互依赖的关键超参数。"""
    if not isinstance(USE_VIRTUAL_NODE, bool):
        raise TypeError("USE_VIRTUAL_NODE 必须是布尔值")
    integer_parameters = {
        "RANDOM_SEED": RANDOM_SEED,
        "BATCH_SIZE": BATCH_SIZE,
        "DATA_LOADER_WORKERS": DATA_LOADER_WORKERS,
        "INPUT_DIM_NUMERIC": INPUT_DIM_NUMERIC,
        "EMBEDDING_DIM": EMBEDDING_DIM,
        "HIDDEN_CHANNELS": HIDDEN_CHANNELS,
        "HEADS": HEADS,
        "NUM_LAYERS": NUM_LAYERS,
        "MLP_HIDDEN_DIM": MLP_HIDDEN_DIM,
        "EPOCHS": EPOCHS,
        "EARLY_STOPPING_PATIENCE": EARLY_STOPPING_PATIENCE,
        "TRAIN_LOG_INTERVAL": TRAIN_LOG_INTERVAL,
        "WORK_OUTPUT_ID_MAX_LENGTH": WORK_OUTPUT_ID_MAX_LENGTH,
    }
    for parameter_name, parameter_value in integer_parameters.items():
        if isinstance(parameter_value, bool) or not isinstance(
            parameter_value,
            int,
        ):
            raise TypeError(f"{parameter_name} 必须是整数")
    if RANDOM_SEED < 0:
        raise ValueError("RANDOM_SEED 必须是非负整数")
    if not (0.0 < TRAIN_RATIO < 1.0 and 0.0 < VAL_RATIO < 1.0):
        raise ValueError("TRAIN_RATIO 和 VAL_RATIO 必须位于 (0, 1)")
    if RATIO_SUM_TOLERANCE < 0.0:
        raise ValueError("RATIO_SUM_TOLERANCE 不能为负数")
    if abs((TRAIN_RATIO + VAL_RATIO) - 1.0) > RATIO_SUM_TOLERANCE:
        raise ValueError("TRAIN_RATIO 与 VAL_RATIO 之和必须为 1")
    if INPUT_DIM_NUMERIC < 1 or EMBEDDING_DIM < 1:
        raise ValueError("输入特征维度必须为正整数")
    if (
        HIDDEN_CHANNELS < 1
        or HEADS < 1
        or NUM_LAYERS < 1
        or MLP_HIDDEN_DIM < 1
    ):
        raise ValueError("GATv2 隐藏维度、头数、层数和 MLP 维度必须为正整数")
    if not 0.0 <= DROPOUT < 1.0:
        raise ValueError("DROPOUT 必须位于 [0, 1)")
    if OUTPUT_MAX <= OUTPUT_MIN:
        raise ValueError("OUTPUT_MAX 必须大于 OUTPUT_MIN")
    if EPOCHS < 1 or BATCH_SIZE < 1:
        raise ValueError("EPOCHS 和 BATCH_SIZE 必须为正整数")
    if DATA_LOADER_WORKERS < 0:
        raise ValueError("DATA_LOADER_WORKERS 不能为负数")
    if LEARNING_RATE <= 0.0 or MIN_LEARNING_RATE <= 0.0:
        raise ValueError("学习率必须为正数")
    if MIN_LEARNING_RATE > LEARNING_RATE:
        raise ValueError("MIN_LEARNING_RATE 不能大于 LEARNING_RATE")
    if WEIGHT_DECAY < 0.0:
        raise ValueError("WEIGHT_DECAY 不能为负数")
    if GRAD_CLIP_MAX_NORM <= 0.0:
        raise ValueError("GRAD_CLIP_MAX_NORM 必须大于 0")
    if EARLY_STOPPING_PATIENCE < 1:
        raise ValueError("EARLY_STOPPING_PATIENCE 必须至少为 1")
    if EARLY_STOPPING_MIN_DELTA < 0.0:
        raise ValueError("EARLY_STOPPING_MIN_DELTA 不能为负数")
    if TRAIN_LOG_INTERVAL < 0:
        raise ValueError("TRAIN_LOG_INTERVAL 不能为负数")
    for parameter_name, parameter_value in {
        "TRAIN_ACTIVITY_MIN": TRAIN_ACTIVITY_MIN,
        "TRAIN_ACTIVITY_MAX": TRAIN_ACTIVITY_MAX,
        "VIRTUAL_NODE_GATE_BIAS": VIRTUAL_NODE_GATE_BIAS,
        "SMOOTH_L1_BETA": SMOOTH_L1_BETA,
        "MSE_AUX_WEIGHT": MSE_AUX_WEIGHT,
    }.items():
        if (
            isinstance(parameter_value, bool)
            or not isinstance(parameter_value, (int, float))
            or not math.isfinite(float(parameter_value))
        ):
            raise TypeError(f"{parameter_name} 必须是有限数值")
    if TRAIN_ACTIVITY_MAX <= TRAIN_ACTIVITY_MIN:
        raise ValueError("TRAIN_ACTIVITY_MAX 必须大于 TRAIN_ACTIVITY_MIN")
    if SMOOTH_L1_BETA <= 0.0:
        raise ValueError("SMOOTH_L1_BETA 必须大于 0")
    if not 0.0 <= MSE_AUX_WEIGHT < 1.0:
        raise ValueError("MSE_AUX_WEIGHT 必须位于 [0, 1)")
    if WORK_TASK_TIMEOUT_SECONDS <= 0:
        raise ValueError("WORK_TASK_TIMEOUT_SECONDS 必须大于 0")
    if WORK_OUTPUT_ID_MAX_LENGTH <= 8:
        raise ValueError("WORK_OUTPUT_ID_MAX_LENGTH 必须大于 8")
    clamp_cpu_count(
        RESOLVER_CPU_THREADS,
        MAX_RESOLVER_CPU_THREADS,
        "RESOLVER_CPU_THREADS",
    )
    clamp_cpu_count(
        WORK_PROCESS_WORKERS,
        MAX_WORK_PROCESS_WORKERS,
        "WORK_PROCESS_WORKERS",
    )
