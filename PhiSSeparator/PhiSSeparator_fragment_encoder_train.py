"""
训练片段编码器，并用最佳 checkpoint 生成完整标准128维片段词嵌入表。

本脚本只负责：
1. 从 SE3TD_128_Shared_data 加载并校验固定粗语料；
2. 构建教师相似度监督；
3. 训练可配置输出维度的片段编码器；
4. 原子保存最佳 checkpoint；
5. 重新加载最佳 checkpoint，将完整粗语料导出为标准片段词嵌入表；
6. 以 NPZ 为权威表，同时生成一片段一行的完整可读 JSON 镜像。

本脚本不会读取、修改或回填任何 PyGData。
"""

import hashlib
import json
import os
import random
import uuid
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

try:
    import utils
except ImportError:  # 支持作为 PhiSSeparator 子模块导入
    from . import utils


# =============================================================================
# 用户配置区：模型维度与相似度权重
# =============================================================================

# 输出词向量维度
OUTPUT_DIM = 128

# 教师总相似度由 ECFP、FCFP 和固有属性三部分加权得到
TEACHER_ECFP_WEIGHT = 0.3
TEACHER_FCFP_WEIGHT = 0.3
TEACHER_INTRINSIC_WEIGHT = 0.4

# 固有属性内部权重，五项必须非负且总和为 1：
# HAC：重原子数量接近程度；RING：环数量接近程度；
# CHARGE：特征电荷是否相同；ELEMENT：重元素组成余弦相似度；
# METAL：两者是否同属金属片段或同属非金属片段。
INTRINSIC_HAC_WEIGHT = 0.3
INTRINSIC_RING_WEIGHT = 0.3
INTRINSIC_ELEMENT_WEIGHT = 0.2
INTRINSIC_METAL_WEIGHT = 0.1
INTRINSIC_CHARGE_WEIGHT = 0.1

# 金属片段与非金属片段不匹配时，是否强制总相似度为 0。
# True 是较强的领域约束，会让两类片段在目标空间中尽量分离。
ZERO_METAL_MISMATCH_SIMILARITY = True

# SIMILARITY_LOSS_WEIGHT 控制模型拟合教师绝对相似度的强度。
# RANK_LOSS_WEIGHT 控制“相似片段应排在不相似片段之前”的相对排序强度。
# 两者至少一个必须大于 0；提高排序权重可能改善近邻顺序，但会牺牲教师
# 相似度数值的精确拟合。
SIMILARITY_LOSS_WEIGHT = 1.0
RANK_LOSS_WEIGHT = 0.2

# 排序损失要求正样本余弦相似度至少比负样本高出的间隔。
# 过大可能使排序目标难以满足，过小则约束较弱。
RANK_MARGIN = 0.1

# 仅当教师正负样本分差达到该阈值时才构建排序三元组。
# 提高该值可减少含糊排序监督，但会减少可用三元组数量。
MIN_TEACHER_RANK_GAP = 0.05


# =============================================================================
# 用户配置区：网络与训练超参数
# =============================================================================
RANDOM_SEED = 42
TRAIN_FRACTION = 0.90
EPOCHS = 200
# 每个优化步骤使用的相似度配对数量
PAIR_BATCH_SIZE = 1024
# AdamW 的初始学习率
LEARNING_RATE = 2e-5
# AdamW 权重衰减
WEIGHT_DECAY = 2e-5
# 验证损失连续多少轮没有改善后停止训练
EARLY_STOPPING_PATIENCE = 15
# 两层隐藏层宽度。增大可提高容量，但会增加参数量、显存和计算时间。
HIDDEN_DIM_1 = 512
HIDDEN_DIM_2 = 256
DROPOUT = 0.15
# checkpoint 推理和标准词表导出的批大小，不参与训练优化步骤。
ENCODE_BATCH_SIZE = 512
# 设为正数时启用梯度裁剪；None 表示不裁剪。
# 若训练出现梯度爆炸或非有限损失，可尝试设置为 1.0 或 5.0。
GRADIENT_CLIP_NORM = None
# 每个锚点最多计算教师相似度的候选总数。
# 增大可提高候选覆盖率，但会增加配对构建耗时。
CANDIDATES_PER_ANCHOR = 256
# 候选总数中优先从同金属类别、同电荷且 HAC/环数接近的属性桶抽取的
# 最大数量，其余名额从当前训练集或验证集进行全局随机抽样。
NEAR_CANDIDATE_COUNT = 128

# 从每个锚点的候选中分别保留多少个高分、中段和低分相似度配对。
# 高低配对强调边界与排序，中段配对帮助拟合连续的相似度尺度。
HIGH_PAIRS_PER_ANCHOR = 4
MIDDLE_PAIRS_PER_ANCHOR = 3
LOW_PAIRS_PER_ANCHOR = 3

# Top-k 近邻验证最多使用的验证 SMILES 数量。
# 教师两两比较开销近似随该值平方增长。
EVALUATION_MAX_SMILES = 512

# 近邻验证中的 k；例如 5 表示比较预测 Top-5 与教师 Top-5 的重合率。
NEIGHBOR_K = 5

# 默认拒绝覆盖已有 checkpoint，避免误删既有训练结果。
# 若确实需要复用相同输出路径，需显式改为 True。
OVERWRITE_CHECKPOINT = True

# utils.save_raw_fragment_corpus 会把无效 SMILES 记录在 metadata 中。
# 默认要求来源语料没有无效条目。
# 改为 False 只会允许使用已过滤后的有效条目，不会恢复无效 SMILES。
REQUIRE_ZERO_INVALID_SMILES = True


# =============================================================================
# 固定路径与派生配置
# =============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
SHARED_DATA_DIR = os.path.join(
    PROJECT_ROOT,
    "Datas",
    "Processed_datas",
    "SE3TD_128_Shared_data",
)
RAW_CORPUS_PATH = os.path.join(
    SHARED_DATA_DIR,
    "fragment_raw_corpus.npz",
)
RAW_CORPUS_METADATA_PATH = os.path.join(
    SHARED_DATA_DIR,
    "fragment_raw_corpus_metadata.json",
)
CHECKPOINT_PATH = os.path.join(
    SCRIPT_DIR,
    "training_artifacts",
    f"fragment_encoder_{OUTPUT_DIM}.pth",
)
EMBEDDING_TABLE_PATH = os.path.join(
    SHARED_DATA_DIR,
    f"fragment_embeddings_{OUTPUT_DIM}d.npz",
)
EMBEDDING_JSON_PATH = utils.derive_embedding_json_mirror_path(
    EMBEDDING_TABLE_PATH
)
EMBEDDING_METADATA_PATH = os.path.join(
    SHARED_DATA_DIR,
    f"fragment_embeddings_{OUTPUT_DIM}d_metadata.json",
)
TRAINING_HISTORY_PATH = os.path.join(
    os.path.dirname(CHECKPOINT_PATH),
    "fragment_encoder_training_history.json",
)

EMBEDDING_TABLE_FIELDS = ("smiles", "fragment_embeddings")

SCALAR_FEATURE_ORDER = ("hac", "ring_count", "formal_charge")
NUMERIC_CLIP_SIGMA = 6.0
EXPECTED_INPUT_DIM = (
    2 * utils.RAW_FRAGMENT_FINGERPRINT_BITS
    + len(SCALAR_FEATURE_ORDER)
    + utils.RAW_FRAGMENT_ELEMENT_DIM
    + 1
)

# 不同用途使用独立 RNG，避免训练轮数改变评估抽样。
RNG_OFFSETS = {
    "split": 0,
    "train_pairs": 1,
    "validation_pairs": 2,
    "training": 3,
    "evaluation": 4,
}


class FragmentEncoder(nn.Module):
    """将固定粗特征映射为单位归一化的可配置维度向量。"""

    def __init__(
        self,
        input_dim,
        hidden_dim_1,
        hidden_dim_2,
        output_dim,
        dropout,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, int(hidden_dim_1)),
            nn.LayerNorm(int(hidden_dim_1)),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim_1), int(hidden_dim_2)),
            nn.LayerNorm(int(hidden_dim_2)),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim_2), self.output_dim),
        )

    def forward(self, features):
        return F.normalize(self.net(features), p=2, dim=-1, eps=1e-12)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path, block_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as file_obj:
        while True:
            block = file_obj.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def torch_load_compat(path, map_location):
    try:
        return torch.load(
            path,
            map_location=map_location,
            weights_only=False,
        )
    except TypeError:
        return torch.load(path, map_location=map_location)


def select_device(device_spec="auto"):
    if isinstance(device_spec, torch.device):
        device = device_spec
    elif device_spec == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        device = torch.device(device_spec)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求使用 CUDA，但当前环境没有可用 CUDA")
    return device


def validate_checkpoint_payload(checkpoint, expected_dim=OUTPUT_DIM):
    if not isinstance(checkpoint, dict):
        raise ValueError("片段编码器 checkpoint 顶层必须是字典")
    required = {
        "model_class",
        "model_state_dict",
        "model_config",
        "normalization",
        "feature_config",
        "training_corpus_sha256",
    }
    missing = required - set(checkpoint)
    if missing:
        raise ValueError(
            f"片段编码器 checkpoint 缺少字段: {sorted(missing)}"
        )
    if checkpoint["model_class"] != "FragmentEncoder":
        raise ValueError("片段编码器 checkpoint 模型类型不兼容")

    model_config = checkpoint["model_config"]
    if not isinstance(model_config, dict):
        raise ValueError("checkpoint 的 model_config 必须是字典")
    if model_config.get("input_dim") != EXPECTED_INPUT_DIM:
        raise ValueError("checkpoint 输入维度不兼容")
    if model_config.get("output_dim") != int(expected_dim):
        raise ValueError("checkpoint 输出维度不兼容")
    for name in ("hidden_dim_1", "hidden_dim_2"):
        value = model_config.get(name)
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"checkpoint 的 {name} 非法")
    dropout = model_config.get("dropout")
    if (
        not isinstance(dropout, (int, float))
        or not 0.0 <= float(dropout) < 1.0
    ):
        raise ValueError("checkpoint 的 dropout 非法")

    validate_scalar_normalization(checkpoint["normalization"])
    feature_config = checkpoint["feature_config"]
    if not isinstance(feature_config, dict):
        raise ValueError("checkpoint 的 feature_config 必须是字典")
    training_sha256 = checkpoint["training_corpus_sha256"]
    if (
        not isinstance(training_sha256, str)
        or len(training_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in training_sha256.lower()
        )
    ):
        raise ValueError("checkpoint 的训练语料 SHA-256 非法")


def validate_checkpoint_against_corpus_metadata(checkpoint, metadata):
    feature_config = checkpoint["feature_config"]
    if feature_config.get("fingerprint") != metadata.get("fingerprint"):
        raise ValueError("checkpoint 与粗语料的指纹参数不一致")
    if feature_config.get("intrinsic_features") != metadata.get(
        "intrinsic_features"
    ):
        raise ValueError("checkpoint 与粗语料的固有属性结构不一致")


def load_checkpoint_payload(
    checkpoint_path=CHECKPOINT_PATH,
    expected_dim=OUTPUT_DIM,
):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"找不到片段编码器 checkpoint: {checkpoint_path}"
        )
    checkpoint = torch_load_compat(
        checkpoint_path, map_location="cpu"
    )
    validate_checkpoint_payload(checkpoint, expected_dim=expected_dim)
    return checkpoint


def build_model_from_checkpoint(checkpoint, device):
    model = FragmentEncoder(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model


def _array_sha256(values, is_string=False):
    digest = hashlib.sha256()
    values = np.asarray(values)
    if is_string:
        for value in values.astype(str).tolist():
            encoded = value.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "little"))
            digest.update(encoded)
    else:
        contiguous = np.ascontiguousarray(values)
        digest.update(contiguous.dtype.str.encode("ascii"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _embedding_table_checksum(arrays):
    digest = hashlib.sha256()
    for key in EMBEDDING_TABLE_FIELDS:
        values = np.asarray(arrays[key])
        digest.update(key.encode("utf-8"))
        digest.update(json.dumps(list(values.shape)).encode("ascii"))
        digest.update(
            _array_sha256(values, is_string=(key == "smiles")).encode(
                "ascii"
            )
        )
    return digest.hexdigest()


def _build_embedding_metadata(
    arrays,
    *,
    checkpoint_sha256,
    checkpoint,
    current_raw_corpus_sha256,
    training_smiles_count,
    previous_metadata=None,
):
    previous_metadata = previous_metadata or {}
    num_smiles = len(arrays["smiles"])
    return {
        "embedding_dim": OUTPUT_DIM,
        "num_smiles": int(num_smiles),
        "training_smiles_count": int(training_smiles_count),
        "resolver_added_smiles_count": int(
            num_smiles - int(training_smiles_count)
        ),
        "checkpoint_sha256": str(checkpoint_sha256),
        "training_raw_corpus_sha256": checkpoint[
            "training_corpus_sha256"
        ],
        "current_raw_corpus_sha256": str(
            current_raw_corpus_sha256
        ),
        "model_config": checkpoint["model_config"],
        "normalization": checkpoint["normalization"],
        "feature_config": checkpoint["feature_config"],
        "created_utc": previous_metadata.get(
            "created_utc", utc_now()
        ),
        "updated_utc": utc_now(),
        "array_shapes": {
            key: list(np.asarray(value).shape)
            for key, value in arrays.items()
        },
        "array_dtypes": {
            key: np.asarray(value).dtype.str
            for key, value in arrays.items()
        },
        "array_sha256": {
            key: _array_sha256(
                value, is_string=(key == "smiles")
            )
            for key, value in arrays.items()
        },
        "table_sha256": _embedding_table_checksum(arrays),
    }


def _validate_embedding_table(arrays, metadata):
    missing = set(EMBEDDING_TABLE_FIELDS) - set(arrays)
    if missing:
        raise ValueError(
            f"标准片段词嵌入表缺少字段: {sorted(missing)}"
        )
    if metadata.get("embedding_dim") != OUTPUT_DIM:
        raise ValueError("标准片段词嵌入表维度不兼容")

    smiles = np.asarray(arrays["smiles"]).astype(str)
    embeddings = np.asarray(arrays["fragment_embeddings"])
    if embeddings.dtype != np.float32:
        raise ValueError("fragment_embeddings 必须为 float32")
    if embeddings.shape != (len(smiles), OUTPUT_DIM):
        raise ValueError("fragment_embeddings 形状错误")
    if len(set(smiles.tolist())) != len(smiles):
        raise ValueError("标准片段词嵌入表包含重复 SMILES")
    if utils.RAW_FRAGMENT_SPECIAL_TOKENS.intersection(
        smiles.tolist()
    ):
        raise ValueError("标准片段词嵌入表不允许保存特殊标记")
    if not np.all(np.isfinite(embeddings)):
        raise ValueError("标准片段词嵌入表包含 NaN 或 Inf")
    if len(embeddings) > 0:
        norms = np.linalg.norm(embeddings, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-4, rtol=0.0):
            raise ValueError("标准片段词嵌入未保持单位归一化")

    expected_shapes = {
        key: list(np.asarray(value).shape)
        for key, value in arrays.items()
    }
    expected_dtypes = {
        key: np.asarray(value).dtype.str
        for key, value in arrays.items()
    }
    expected_checksums = {
        key: _array_sha256(
            value, is_string=(key == "smiles")
        )
        for key, value in arrays.items()
    }
    if metadata.get("array_shapes") != expected_shapes:
        raise ValueError("标准片段词嵌入表数组形状元数据不一致")
    if metadata.get("array_dtypes") != expected_dtypes:
        raise ValueError("标准片段词嵌入表数组类型元数据不一致")
    if metadata.get("array_sha256") != expected_checksums:
        raise ValueError("标准片段词嵌入表逐数组校验值不一致")
    if metadata.get("table_sha256") != (
        _embedding_table_checksum(arrays)
    ):
        raise ValueError("标准片段词嵌入表整体校验值不一致")
    if metadata.get("num_smiles") != len(smiles):
        raise ValueError("标准片段词嵌入表片段数量不一致")
    training_count = metadata.get("training_smiles_count")
    resolver_count = metadata.get("resolver_added_smiles_count")
    if (
        not isinstance(training_count, int)
        or not isinstance(resolver_count, int)
        or training_count < 1
        or resolver_count < 0
        or training_count + resolver_count != len(smiles)
    ):
        raise ValueError("标准表训练/动态片段计数不一致")
    json_mirror = metadata.get("json_mirror")
    if json_mirror is not None:
        expected_json_metadata = {
            "source_table_sha256": metadata["table_sha256"],
            "table_kind": "fragment",
            "layout": "json_array_one_fragment_per_line",
            "authoritative": False,
            "num_smiles": len(smiles),
            "embedding_dim": OUTPUT_DIM,
        }
        if (
            not isinstance(json_mirror, dict)
            or any(
                json_mirror.get(key) != value
                for key, value in expected_json_metadata.items()
            )
            or not isinstance(json_mirror.get("file_name"), str)
            or not isinstance(json_mirror.get("sha256"), str)
            or len(json_mirror["sha256"]) != 64
        ):
            raise ValueError("标准片段词嵌入 JSON 镜像元数据不一致")


def _atomic_write_embedding_table(
    arrays,
    metadata,
    npz_path,
    metadata_path,
):
    npz_path = os.path.abspath(os.fspath(npz_path))
    json_path = utils.derive_embedding_json_mirror_path(npz_path)
    metadata_path = os.path.abspath(os.fspath(metadata_path))
    os.makedirs(os.path.dirname(npz_path), exist_ok=True)
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    token = f"{os.getpid()}-{uuid.uuid4().hex}"
    temp_npz = f"{npz_path}.{token}.tmp.npz"
    temp_json = f"{json_path}.{token}.tmp"
    temp_metadata = f"{metadata_path}.{token}.tmp"
    try:
        metadata.pop("json_mirror", None)
        _validate_embedding_table(arrays, metadata)
        np.savez_compressed(temp_npz, **arrays)
        with np.load(temp_npz, allow_pickle=False) as loaded:
            reloaded = {
                key: np.array(loaded[key], copy=True)
                for key in EMBEDDING_TABLE_FIELDS
        }
        reloaded["smiles"] = reloaded["smiles"].astype(str)
        _validate_embedding_table(reloaded, metadata)
        metadata["json_mirror"] = (
            utils.write_embedding_json_mirror_file(
                temp_json,
                reloaded,
                table_kind="fragment",
                source_table_sha256=metadata["table_sha256"],
                mirror_filename=os.path.basename(json_path),
            )
        )
        _validate_embedding_table(reloaded, metadata)
        with open(temp_metadata, "w", encoding="utf-8") as file_obj:
            json.dump(metadata, file_obj, ensure_ascii=False, indent=2)
            file_obj.write("\n")
            file_obj.flush()
            os.fsync(file_obj.fileno())
        os.replace(temp_npz, npz_path)
        os.replace(temp_json, json_path)
        os.replace(temp_metadata, metadata_path)
    finally:
        for temp_path in (temp_npz, temp_json, temp_metadata):
            if os.path.exists(temp_path):
                os.remove(temp_path)


def save_fragment_embedding_table(
    corpus,
    embeddings,
    *,
    checkpoint_path,
    checkpoint,
    raw_corpus_sha256,
    npz_path=EMBEDDING_TABLE_PATH,
    metadata_path=EMBEDDING_METADATA_PATH,
):
    if checkpoint["training_corpus_sha256"] != raw_corpus_sha256:
        raise ValueError(
            "最佳 checkpoint 不是由当前片段粗语料训练得到，"
            "禁止生成混合标准表"
        )
    arrays = {
        "smiles": np.asarray(corpus["smiles"], dtype=np.str_),
        "fragment_embeddings": np.asarray(
            embeddings, dtype=np.float32
        ),
    }
    checkpoint_sha256 = file_sha256(checkpoint_path)
    metadata = _build_embedding_metadata(
        arrays,
        checkpoint_sha256=checkpoint_sha256,
        checkpoint=checkpoint,
        current_raw_corpus_sha256=raw_corpus_sha256,
        training_smiles_count=len(arrays["smiles"]),
    )
    _atomic_write_embedding_table(
        arrays, metadata, npz_path, metadata_path
    )
    return metadata


def load_fragment_embedding_table(
    npz_path=EMBEDDING_TABLE_PATH,
    metadata_path=EMBEDDING_METADATA_PATH,
):
    with open(metadata_path, "r", encoding="utf-8") as file_obj:
        metadata = json.load(file_obj)
    with np.load(npz_path, allow_pickle=False) as loaded:
        missing = set(EMBEDDING_TABLE_FIELDS) - set(loaded.files)
        if missing:
            raise ValueError(
                f"标准片段词嵌入表缺少字段: {sorted(missing)}"
            )
        arrays = {
            key: np.array(loaded[key], copy=True)
            for key in EMBEDDING_TABLE_FIELDS
        }
    arrays["smiles"] = arrays["smiles"].astype(str)
    _validate_embedding_table(arrays, metadata)
    return arrays, metadata


def append_fragment_embedding_records(
    records,
    *,
    npz_path,
    metadata_path,
    checkpoint,
    current_raw_corpus_sha256,
):
    arrays, metadata = load_fragment_embedding_table(
        npz_path, metadata_path
    )
    smiles_list = arrays["smiles"].astype(str).tolist()
    existing = set(smiles_list)
    pending = []
    seen_pending = set()
    for record in records:
        canonical_smiles = str(record["smiles"])
        if (
            canonical_smiles in existing
            or canonical_smiles in seen_pending
        ):
            continue
        if canonical_smiles in utils.RAW_FRAGMENT_SPECIAL_TOKENS:
            raise ValueError("特殊标记不能追加到标准片段词嵌入表")
        embedding = np.asarray(
            record["fragment_embedding"], dtype=np.float32
        )
        if embedding.shape != (OUTPUT_DIM,):
            raise ValueError(
                f"片段 {canonical_smiles} 的新增词嵌入形状错误"
            )
        if not np.all(np.isfinite(embedding)):
            raise ValueError(
                f"片段 {canonical_smiles} 的新增词嵌入包含 NaN 或 Inf"
            )
        if not np.isclose(
            np.linalg.norm(embedding),
            1.0,
            atol=1e-4,
            rtol=0.0,
        ):
            raise ValueError(
                f"片段 {canonical_smiles} 的新增词嵌入未单位归一化"
            )
        pending.append((canonical_smiles, embedding))
        seen_pending.add(canonical_smiles)

    if not pending:
        return [], metadata

    updated = {
        "smiles": np.concatenate([
            arrays["smiles"],
            np.asarray(
                [smiles for smiles, _ in pending],
                dtype=np.str_,
            ),
        ]),
        "fragment_embeddings": np.concatenate([
            arrays["fragment_embeddings"],
            np.stack([
                embedding for _, embedding in pending
            ]).astype(np.float32),
        ], axis=0).astype(np.float32, copy=False),
    }
    updated_metadata = _build_embedding_metadata(
        updated,
        checkpoint_sha256=metadata["checkpoint_sha256"],
        checkpoint=checkpoint,
        current_raw_corpus_sha256=current_raw_corpus_sha256,
        training_smiles_count=int(metadata["training_smiles_count"]),
        previous_metadata=metadata,
    )
    _atomic_write_embedding_table(
        updated, updated_metadata, npz_path, metadata_path
    )
    return [smiles for smiles, _ in pending], updated_metadata


def append_fragment_embedding_record(
    canonical_smiles,
    embedding,
    *,
    npz_path,
    metadata_path,
    checkpoint,
    current_raw_corpus_sha256,
):
    added, metadata = append_fragment_embedding_records(
        [{
            "smiles": canonical_smiles,
            "fragment_embedding": embedding,
        }],
        npz_path=npz_path,
        metadata_path=metadata_path,
        checkpoint=checkpoint,
        current_raw_corpus_sha256=current_raw_corpus_sha256,
    )
    return bool(added), metadata


def repair_fragment_embedding_metadata(
    npz_path,
    metadata_path,
    *,
    checkpoint,
    current_raw_corpus_sha256,
):
    with open(metadata_path, "r", encoding="utf-8") as file_obj:
        previous_metadata = json.load(file_obj)
    with np.load(npz_path, allow_pickle=False) as loaded:
        arrays = {
            key: np.array(loaded[key], copy=True)
            for key in EMBEDDING_TABLE_FIELDS
        }
    arrays["smiles"] = arrays["smiles"].astype(str)
    repaired = _build_embedding_metadata(
        arrays,
        checkpoint_sha256=previous_metadata["checkpoint_sha256"],
        checkpoint=checkpoint,
        current_raw_corpus_sha256=current_raw_corpus_sha256,
        training_smiles_count=int(
            previous_metadata["training_smiles_count"]
        ),
        previous_metadata=previous_metadata,
    )
    _validate_embedding_table(arrays, repaired)
    json_path = utils.derive_embedding_json_mirror_path(npz_path)
    repaired["json_mirror"] = (
        utils.atomic_write_embedding_json_mirror(
            json_path,
            arrays,
            table_kind="fragment",
            source_table_sha256=repaired["table_sha256"],
        )
    )
    _validate_embedding_table(arrays, repaired)
    utils.atomic_write_json(metadata_path, repaired)
    return arrays, repaired


def _validate_normalized_weights(name, values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} 必须是一组有限数值")
    if np.any(values < 0.0):
        raise ValueError(f"{name} 不能包含负权重")
    if not np.isclose(values.sum(), 1.0, atol=1e-8, rtol=0.0):
        raise ValueError(
            f"{name} 的总和必须为 1，当前为 {values.sum():.12g}"
        )


def validate_configuration():
    """在加载大型语料和启动训练前检查全部用户配置。"""
    if (
        not isinstance(RANDOM_SEED, int)
        or isinstance(RANDOM_SEED, bool)
        or RANDOM_SEED < 0
    ):
        raise ValueError("RANDOM_SEED 必须是非负整数")

    boolean_values = {
        "ZERO_METAL_MISMATCH_SIMILARITY": (
            ZERO_METAL_MISMATCH_SIMILARITY
        ),
        "OVERWRITE_CHECKPOINT": OVERWRITE_CHECKPOINT,
        "REQUIRE_ZERO_INVALID_SMILES": REQUIRE_ZERO_INVALID_SMILES,
    }
    for name, value in boolean_values.items():
        if not isinstance(value, bool):
            raise ValueError(f"{name} 必须是布尔值")

    integer_positive = {
        "OUTPUT_DIM": OUTPUT_DIM,
        "EPOCHS": EPOCHS,
        "PAIR_BATCH_SIZE": PAIR_BATCH_SIZE,
        "ENCODE_BATCH_SIZE": ENCODE_BATCH_SIZE,
        "EARLY_STOPPING_PATIENCE": EARLY_STOPPING_PATIENCE,
        "HIDDEN_DIM_1": HIDDEN_DIM_1,
        "HIDDEN_DIM_2": HIDDEN_DIM_2,
        "CANDIDATES_PER_ANCHOR": CANDIDATES_PER_ANCHOR,
        "EVALUATION_MAX_SMILES": EVALUATION_MAX_SMILES,
        "NEIGHBOR_K": NEIGHBOR_K,
    }
    for name, value in integer_positive.items():
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} 必须是正整数")

    nonnegative_integer = {
        "NEAR_CANDIDATE_COUNT": NEAR_CANDIDATE_COUNT,
        "HIGH_PAIRS_PER_ANCHOR": HIGH_PAIRS_PER_ANCHOR,
        "MIDDLE_PAIRS_PER_ANCHOR": MIDDLE_PAIRS_PER_ANCHOR,
        "LOW_PAIRS_PER_ANCHOR": LOW_PAIRS_PER_ANCHOR,
    }
    for name, value in nonnegative_integer.items():
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name} 必须是非负整数")

    if NEAR_CANDIDATE_COUNT > CANDIDATES_PER_ANCHOR:
        raise ValueError(
            "NEAR_CANDIDATE_COUNT 不能大于 CANDIDATES_PER_ANCHOR"
        )
    if (
        HIGH_PAIRS_PER_ANCHOR
        + MIDDLE_PAIRS_PER_ANCHOR
        + LOW_PAIRS_PER_ANCHOR
        <= 0
    ):
        raise ValueError("每个锚点至少需要选择一个相似度配对")

    finite_positive = {
        "LEARNING_RATE": LEARNING_RATE,
    }
    for name, value in finite_positive.items():
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} 必须是有限正数")

    finite_nonnegative = {
        "WEIGHT_DECAY": WEIGHT_DECAY,
        "SIMILARITY_LOSS_WEIGHT": SIMILARITY_LOSS_WEIGHT,
        "RANK_LOSS_WEIGHT": RANK_LOSS_WEIGHT,
        "RANK_MARGIN": RANK_MARGIN,
        "MIN_TEACHER_RANK_GAP": MIN_TEACHER_RANK_GAP,
    }
    for name, value in finite_nonnegative.items():
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} 必须是有限非负数")

    if SIMILARITY_LOSS_WEIGHT == 0.0 and RANK_LOSS_WEIGHT == 0.0:
        raise ValueError("相似度损失和排序损失不能同时关闭")
    if not 0.0 < TRAIN_FRACTION < 1.0:
        raise ValueError("TRAIN_FRACTION 必须位于 (0, 1)")
    if not 0.0 <= DROPOUT < 1.0:
        raise ValueError("DROPOUT 必须位于 [0, 1)")
    if MIN_TEACHER_RANK_GAP > 1.0:
        raise ValueError("MIN_TEACHER_RANK_GAP 不能大于 1")
    if GRADIENT_CLIP_NORM is not None:
        if not np.isfinite(GRADIENT_CLIP_NORM) or GRADIENT_CLIP_NORM <= 0.0:
            raise ValueError("GRADIENT_CLIP_NORM 必须为正数或 None")

    _validate_normalized_weights(
        "教师总相似度权重",
        (
            TEACHER_ECFP_WEIGHT,
            TEACHER_FCFP_WEIGHT,
            TEACHER_INTRINSIC_WEIGHT,
        ),
    )
    _validate_normalized_weights(
        "固有属性内部权重",
        (
            INTRINSIC_HAC_WEIGHT,
            INTRINSIC_RING_WEIGHT,
            INTRINSIC_CHARGE_WEIGHT,
            INTRINSIC_ELEMENT_WEIGHT,
            INTRINSIC_METAL_WEIGHT,
        ),
    )


def validate_source_paths():
    for path, description in (
        (RAW_CORPUS_PATH, "粗语料 NPZ"),
        (RAW_CORPUS_METADATA_PATH, "粗语料 metadata"),
    ):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"找不到{description}: {path}")

    input_paths = {
        os.path.normcase(os.path.abspath(RAW_CORPUS_PATH)),
        os.path.normcase(os.path.abspath(RAW_CORPUS_METADATA_PATH)),
    }
    output_paths = {
        os.path.normcase(os.path.abspath(path))
        for path in (
            CHECKPOINT_PATH,
            EMBEDDING_TABLE_PATH,
            EMBEDDING_METADATA_PATH,
            TRAINING_HISTORY_PATH,
        )
    }
    if len(output_paths) != 4:
        raise ValueError("片段编码器的四个输出路径不能重复")
    if input_paths.intersection(output_paths):
        raise ValueError("片段编码器输出路径不能覆盖粗语料输入")
    if os.path.exists(CHECKPOINT_PATH) and not OVERWRITE_CHECKPOINT:
        raise FileExistsError(
            "checkpoint 已存在且 OVERWRITE_CHECKPOINT=False: "
            f"{CHECKPOINT_PATH}"
        )


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def validate_loaded_corpus(corpus, metadata):
    """补充检查数值内容；结构、类型和 SHA256 已由 utils 严格校验。"""
    num_smiles = len(corpus["smiles"])
    if num_smiles < 8:
        raise ValueError("离线训练至少需要 8 个不同的有效 SMILES")

    invalid_count = metadata.get("invalid_smiles_count")
    if not isinstance(invalid_count, int) or invalid_count < 0:
        raise ValueError("粗语料 metadata 的 invalid_smiles_count 非法")
    if REQUIRE_ZERO_INVALID_SMILES and invalid_count != 0:
        raise ValueError(
            "粗语料记录了无效 SMILES，当前配置拒绝训练: "
            f"invalid_smiles_count={invalid_count}"
        )

    for name in SCALAR_FEATURE_ORDER:
        if not np.all(np.isfinite(corpus[name])):
            raise ValueError(f"粗语料字段 {name} 包含 NaN 或 Inf")
    if np.any(corpus["hac"] < 0):
        raise ValueError("粗语料 hac 包含负数")
    if np.any(corpus["ring_count"] < 0):
        raise ValueError("粗语料 ring_count 包含负数")
    if np.any(corpus["element_counts"] < 0):
        raise ValueError("粗语料 element_counts 包含负数")

    for name in ("ecfp", "fcfp", "is_metal"):
        values = corpus[name]
        if values.min() < 0 or values.max() > 1:
            raise ValueError(f"粗语料字段 {name} 不是二值数据")


def calculate_scalar_normalization(corpus, row_indices):
    scalar_matrix = np.column_stack([
        corpus[name][row_indices].astype(np.float32)
        for name in SCALAR_FEATURE_ORDER
    ])
    means = scalar_matrix.mean(axis=0)
    stds = scalar_matrix.std(axis=0)
    stds = np.where(stds > 1e-6, stds, 1.0)
    return {
        "feature_order": list(SCALAR_FEATURE_ORDER),
        "mean": means.astype(np.float32).tolist(),
        "std": stds.astype(np.float32).tolist(),
    }


def validate_scalar_normalization(normalization):
    if not isinstance(normalization, dict):
        raise ValueError("标量标准化统计必须是字典")
    if normalization.get("feature_order") != list(SCALAR_FEATURE_ORDER):
        raise ValueError("标量标准化字段顺序不兼容")
    means = np.asarray(normalization.get("mean"), dtype=np.float32)
    stds = np.asarray(normalization.get("std"), dtype=np.float32)
    expected_shape = (len(SCALAR_FEATURE_ORDER),)
    if means.shape != expected_shape or stds.shape != expected_shape:
        raise ValueError("标量标准化统计形状不兼容")
    if not np.all(np.isfinite(means)) or not np.all(np.isfinite(stds)):
        raise ValueError("标量标准化统计包含 NaN 或 Inf")
    if np.any(stds <= 0.0):
        raise ValueError("标量标准差必须大于 0")


def build_encoder_input_rows(corpus, row_indices, normalization):
    """仅为当前批次构造稠密输入，避免一次展开整个大型语料。"""
    validate_scalar_normalization(normalization)
    row_indices = np.asarray(row_indices, dtype=np.int64)

    scalar_matrix = np.column_stack([
        corpus[name][row_indices].astype(np.float32)
        for name in SCALAR_FEATURE_ORDER
    ])
    means = np.asarray(normalization["mean"], dtype=np.float32)
    stds = np.asarray(normalization["std"], dtype=np.float32)
    scalar_matrix = (scalar_matrix - means) / stds
    # 仅裁剪标准化输入，保留原始计数供元素比例等计算使用。
    scalar_matrix = np.clip(
        scalar_matrix, -NUMERIC_CLIP_SIGMA, NUMERIC_CLIP_SIGMA
    )

    hac = corpus["hac"][row_indices].astype(np.float32).reshape(-1, 1)
    safe_hac = np.maximum(hac, 1.0)
    element_fractions = (
        corpus["element_counts"][row_indices].astype(np.float32)
        / safe_hac
    )
    is_metal = (
        corpus["is_metal"][row_indices].astype(np.float32).reshape(-1, 1)
    )

    inputs = np.concatenate(
        [
            corpus["ecfp"][row_indices].astype(np.float32),
            corpus["fcfp"][row_indices].astype(np.float32),
            scalar_matrix,
            element_fractions,
            is_metal,
        ],
        axis=1,
    )
    if inputs.shape[1] != EXPECTED_INPUT_DIM:
        raise ValueError(
            f"编码器输入维度错误: {inputs.shape[1]} != {EXPECTED_INPUT_DIM}"
        )
    if not np.all(np.isfinite(inputs)):
        raise ValueError("编码器输入包含 NaN 或 Inf")
    return inputs


def _bit_tanimoto(first, second):
    intersection = np.logical_and(first, second).sum(axis=1).astype(np.float32)
    union = np.logical_or(first, second).sum(axis=1).astype(np.float32)
    return np.divide(
        intersection,
        union,
        out=np.ones_like(intersection),
        where=union > 0,
    )


def teacher_similarity_for_pairs(corpus, first_indices, second_indices):
    """根据顶部配置的权重计算一批教师相似度。"""
    first_indices = np.asarray(first_indices, dtype=np.int64)
    second_indices = np.asarray(second_indices, dtype=np.int64)
    if first_indices.shape != second_indices.shape:
        raise ValueError("教师相似度的两组索引形状必须一致")

    ecfp_similarity = _bit_tanimoto(
        corpus["ecfp"][first_indices],
        corpus["ecfp"][second_indices],
    )
    fcfp_similarity = _bit_tanimoto(
        corpus["fcfp"][first_indices],
        corpus["fcfp"][second_indices],
    )

    hac_first = corpus["hac"][first_indices].astype(np.float32)
    hac_second = corpus["hac"][second_indices].astype(np.float32)
    hac_max = np.maximum(hac_first, hac_second)
    hac_similarity = np.divide(
        np.minimum(hac_first, hac_second),
        hac_max,
        out=np.ones_like(hac_max),
        where=hac_max > 0,
    )

    ring_similarity = 1.0 / (
        1.0
        + np.abs(
            corpus["ring_count"][first_indices].astype(np.float32)
            - corpus["ring_count"][second_indices].astype(np.float32)
        )
    )
    charge_similarity = np.isclose(
        corpus["formal_charge"][first_indices],
        corpus["formal_charge"][second_indices],
        atol=1e-6,
        rtol=0.0,
    ).astype(np.float32)

    element_first = corpus["element_counts"][first_indices].astype(np.float32)
    element_second = corpus["element_counts"][second_indices].astype(np.float32)
    element_dot = (element_first * element_second).sum(axis=1)
    element_first_norm = np.linalg.norm(element_first, axis=1)
    element_second_norm = np.linalg.norm(element_second, axis=1)
    element_denom = element_first_norm * element_second_norm
    element_similarity = np.divide(
        element_dot,
        element_denom,
        out=np.zeros_like(element_dot),
        where=element_denom > 0,
    )
    both_element_empty = (
        (element_first_norm == 0.0) & (element_second_norm == 0.0)
    )
    element_similarity[both_element_empty] = 1.0

    metal_first = corpus["is_metal"][first_indices].astype(bool)
    metal_second = corpus["is_metal"][second_indices].astype(bool)
    metal_similarity = (metal_first == metal_second).astype(np.float32)

    intrinsic_similarity = (
        INTRINSIC_HAC_WEIGHT * hac_similarity
        + INTRINSIC_RING_WEIGHT * ring_similarity
        + INTRINSIC_CHARGE_WEIGHT * charge_similarity
        + INTRINSIC_ELEMENT_WEIGHT * element_similarity
        + INTRINSIC_METAL_WEIGHT * metal_similarity
    )
    teacher_similarity = (
        TEACHER_ECFP_WEIGHT * ecfp_similarity
        + TEACHER_FCFP_WEIGHT * fcfp_similarity
        + TEACHER_INTRINSIC_WEIGHT * intrinsic_similarity
    )
    if ZERO_METAL_MISMATCH_SIMILARITY:
        teacher_similarity[metal_first != metal_second] = 0.0
    return np.clip(teacher_similarity, 0.0, 1.0).astype(np.float32)


def split_indices(num_samples, train_fraction, rng):
    indices = np.arange(num_samples, dtype=np.int64)
    rng.shuffle(indices)
    split_point = int(round(num_samples * train_fraction))
    split_point = min(max(split_point, 4), num_samples - 4)
    return np.sort(indices[:split_point]), np.sort(indices[split_point:])


def _build_coarse_buckets(corpus, allowed_indices):
    buckets = {}
    for idx in allowed_indices:
        # formal_charge 在 v2 粗语料中是 float32，金属可出现 1.5、2.5 等值；
        # 这里必须保留浮点值，不能像旧脚本一样转为 int。
        key = (
            int(corpus["is_metal"][idx]),
            float(corpus["formal_charge"][idx]),
            int(corpus["hac"][idx]),
            int(corpus["ring_count"][idx]),
        )
        buckets.setdefault(key, []).append(int(idx))
    return buckets


def _near_candidates(
    corpus,
    anchor_idx,
    buckets,
    max_candidates,
    rng,
):
    """从相近属性桶中直接进行有界抽样，避免先展开超大候选列表。"""
    if max_candidates <= 0:
        return []

    metal = int(corpus["is_metal"][anchor_idx])
    charge = float(corpus["formal_charge"][anchor_idx])
    hac = int(corpus["hac"][anchor_idx])
    ring_count = int(corpus["ring_count"][anchor_idx])
    candidate_buckets = []
    for candidate_hac in range(max(0, hac - 2), hac + 3):
        for candidate_ring in range(max(0, ring_count - 1), ring_count + 2):
            key = (metal, charge, candidate_hac, candidate_ring)
            values = buckets.get(key)
            if values:
                candidate_buckets.append(values)

    total_count = sum(len(values) for values in candidate_buckets)
    available_count = max(0, total_count - 1)
    target_count = min(int(max_candidates), available_count)
    if target_count == 0:
        return []

    if total_count <= 2 * target_count + 1:
        candidates = [
            int(value)
            for values in candidate_buckets
            for value in values
            if int(value) != anchor_idx
        ]
        if len(candidates) <= target_count:
            return candidates
        selected_positions = rng.choice(
            len(candidates),
            size=target_count,
            replace=False,
        )
        return [
            candidates[int(position)]
            for position in selected_positions
        ]

    cumulative_lengths = np.cumsum(
        [len(values) for values in candidate_buckets],
        dtype=np.int64,
    )
    selected = []
    selected_set = set()
    for _ in range(20):
        needed = target_count - len(selected)
        if needed <= 0:
            break
        flat_positions = rng.integers(
            0,
            total_count,
            size=max(32, needed * 4),
        )
        for flat_position in flat_positions:
            bucket_position = int(
                np.searchsorted(
                    cumulative_lengths,
                    flat_position,
                    side="right",
                )
            )
            previous_end = (
                0
                if bucket_position == 0
                else int(cumulative_lengths[bucket_position - 1])
            )
            local_position = int(flat_position) - previous_end
            candidate = int(
                candidate_buckets[bucket_position][local_position]
            )
            if candidate == anchor_idx or candidate in selected_set:
                continue
            selected.append(candidate)
            selected_set.add(candidate)
            if len(selected) == target_count:
                break

    if len(selected) < target_count:
        for values in candidate_buckets:
            for value in values:
                candidate = int(value)
                if candidate == anchor_idx or candidate in selected_set:
                    continue
                selected.append(candidate)
                selected_set.add(candidate)
                if len(selected) == target_count:
                    break
            if len(selected) == target_count:
                break

    if len(selected) != target_count:
        raise RuntimeError("无法完成相近属性候选抽样")
    return selected


def _sample_global_candidates(
    allowed_indices,
    excluded_indices,
    sample_size,
    rng,
):
    """
    不构造完整差集地抽取全局候选。

    常规路径的开销与需要抽取的候选数近似线性；只有数据集本身很小或
    可选元素已接近目标数量时才扫描一次 allowed_indices。
    """
    if sample_size <= 0:
        return []

    allowed_indices = np.asarray(allowed_indices, dtype=np.int64)
    excluded = {int(value) for value in excluded_indices}
    available_count = len(allowed_indices) - len(excluded)
    target_count = min(int(sample_size), max(0, available_count))
    if target_count == 0:
        return []

    if len(allowed_indices) <= 512 or available_count <= 2 * target_count:
        available = [
            int(value)
            for value in allowed_indices
            if int(value) not in excluded
        ]
        if len(available) <= target_count:
            return available
        selected_positions = rng.choice(
            len(available),
            size=target_count,
            replace=False,
        )
        return [available[int(position)] for position in selected_positions]

    selected = []
    selected_set = set()
    for _ in range(20):
        needed = target_count - len(selected)
        if needed <= 0:
            break
        draw_count = max(32, needed * 4)
        draw_positions = rng.integers(
            0,
            len(allowed_indices),
            size=draw_count,
        )
        for position in draw_positions:
            candidate = int(allowed_indices[int(position)])
            if candidate in excluded or candidate in selected_set:
                continue
            selected.append(candidate)
            selected_set.add(candidate)
            if len(selected) == target_count:
                break

    # 极端高排除率下使用确定长度的循环扫描兜底，不分配大型差集。
    if len(selected) < target_count:
        start = int(rng.integers(0, len(allowed_indices)))
        for offset in range(len(allowed_indices)):
            candidate = int(
                allowed_indices[(start + offset) % len(allowed_indices)]
            )
            if candidate in excluded or candidate in selected_set:
                continue
            selected.append(candidate)
            selected_set.add(candidate)
            if len(selected) == target_count:
                break

    if len(selected) != target_count:
        raise RuntimeError("无法完成全局候选抽样")
    return selected


def build_stratified_pairs(
    corpus,
    allowed_indices,
    rng,
    description,
    require_triplets,
):
    """
    为每个锚点抽取相近属性候选和全局候选，再保留高、中、低相似度对。
    """
    allowed_indices = np.asarray(allowed_indices, dtype=np.int64)
    if len(allowed_indices) < 2:
        raise ValueError(f"{description}至少需要两个 SMILES")
    if len(np.unique(allowed_indices)) != len(allowed_indices):
        raise ValueError(f"{description}索引包含重复值")

    buckets = _build_coarse_buckets(corpus, allowed_indices)
    pair_first = []
    pair_second = []
    pair_targets = []
    triplets = []
    skipped_small_gap = 0

    for anchor_idx in tqdm(
        allowed_indices,
        desc=f"构建{description}配对",
    ):
        anchor_idx = int(anchor_idx)
        near = _near_candidates(
            corpus,
            anchor_idx,
            buckets,
            NEAR_CANDIDATE_COUNT,
            rng,
        )

        remaining_count = max(
            0,
            CANDIDATES_PER_ANCHOR - len(near),
        )
        global_candidates = _sample_global_candidates(
            allowed_indices,
            {anchor_idx, *near},
            remaining_count,
            rng,
        )
        candidates = list(dict.fromkeys(near + global_candidates))
        if not candidates:
            continue

        anchor_values = np.full(
            (len(candidates),),
            anchor_idx,
            dtype=np.int64,
        )
        candidate_values = np.asarray(candidates, dtype=np.int64)
        scores = teacher_similarity_for_pairs(
            corpus,
            anchor_values,
            candidate_values,
        )
        order = np.argsort(scores, kind="stable")

        low_positions = order[
            : min(LOW_PAIRS_PER_ANCHOR, len(order))
        ]
        high_positions = order[
            max(0, len(order) - HIGH_PAIRS_PER_ANCHOR):
        ]
        middle_start = len(order) // 3
        middle_end = max(middle_start + 1, 2 * len(order) // 3)
        middle_pool = order[middle_start:middle_end]
        if len(middle_pool) > MIDDLE_PAIRS_PER_ANCHOR:
            middle_positions = rng.choice(
                middle_pool,
                size=MIDDLE_PAIRS_PER_ANCHOR,
                replace=False,
            )
        else:
            middle_positions = middle_pool

        selected_positions = np.unique(
            np.concatenate(
                [low_positions, middle_positions, high_positions]
            )
        )
        for position in selected_positions:
            pair_first.append(anchor_idx)
            pair_second.append(int(candidate_values[position]))
            pair_targets.append(float(scores[position]))

        best_position = int(order[-1])
        worst_position = int(order[0])
        teacher_gap = float(
            scores[best_position] - scores[worst_position]
        )
        if (
            best_position != worst_position
            and teacher_gap >= MIN_TEACHER_RANK_GAP
        ):
            triplets.append(
                (
                    anchor_idx,
                    int(candidate_values[best_position]),
                    int(candidate_values[worst_position]),
                )
            )
        else:
            skipped_small_gap += 1

    if not pair_first:
        raise ValueError(f"无法为{description}构建相似度配对")
    if require_triplets and not triplets:
        raise ValueError(
            f"无法为{description}构建满足最小教师分差的排序三元组"
        )

    targets = np.asarray(pair_targets, dtype=np.float32)
    triplet_array = np.asarray(triplets, dtype=np.int64).reshape(-1, 3)
    return {
        "first": np.asarray(pair_first, dtype=np.int64),
        "second": np.asarray(pair_second, dtype=np.int64),
        "target": targets,
        "triplets": triplet_array,
        "stats": {
            "num_anchors": int(len(allowed_indices)),
            "num_pairs": int(len(pair_first)),
            "num_triplets": int(len(triplets)),
            "triplets_skipped_small_gap": int(skipped_small_gap),
            "target_min": float(targets.min()),
            "target_mean": float(targets.mean()),
            "target_max": float(targets.max()),
        },
    }


def _model_rows(model, corpus, indices, normalization, device):
    inputs = build_encoder_input_rows(corpus, indices, normalization)
    tensor = torch.from_numpy(inputs).to(
        device=device,
        dtype=torch.float32,
    )
    return model(tensor)


def encode_corpus_rows(
    corpus,
    row_indices,
    model,
    normalization,
    device,
    batch_size=ENCODE_BATCH_SIZE,
    description="编码片段词嵌入",
):
    row_indices = np.asarray(row_indices, dtype=np.int64)
    if row_indices.ndim != 1:
        raise ValueError("待编码行索引必须是一维数组")
    if len(row_indices) == 0:
        return np.empty((0, model.output_dim), dtype=np.float32)
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("编码批大小必须是正整数")

    blocks = []
    model.eval()
    with torch.no_grad():
        for start in tqdm(
            range(0, len(row_indices), batch_size),
            desc=description,
            leave=False,
        ):
            indices = row_indices[start:start + batch_size]
            blocks.append(
                _model_rows(
                    model,
                    corpus,
                    indices,
                    normalization,
                    device,
                ).detach().cpu().numpy().astype(np.float32)
            )
    embeddings = np.concatenate(blocks, axis=0)
    if not np.all(np.isfinite(embeddings)):
        raise ValueError("片段编码结果包含 NaN 或 Inf")
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-4, rtol=0.0):
        raise ValueError("片段编码结果未保持单位归一化")
    return embeddings


def records_to_corpus(records):
    if not records:
        raise ValueError("待编码片段粗特征记录不能为空")
    return {
        "smiles": np.asarray(
            [record["smiles"] for record in records],
            dtype=np.str_,
        ),
        "ecfp": np.stack([
            record["ecfp"] for record in records
        ]).astype(np.uint8),
        "fcfp": np.stack([
            record["fcfp"] for record in records
        ]).astype(np.uint8),
        "hac": np.asarray(
            [record["hac"] for record in records],
            dtype=np.int16,
        ),
        "ring_count": np.asarray(
            [record["ring_count"] for record in records],
            dtype=np.int16,
        ),
        "formal_charge": np.asarray(
            [record["formal_charge"] for record in records],
            dtype=np.float32,
        ),
        "element_counts": np.stack([
            record["element_counts"] for record in records
        ]).astype(np.int16),
        "is_metal": np.asarray(
            [record["is_metal"] for record in records],
            dtype=np.uint8,
        ),
    }


def encode_feature_records(
    records,
    model,
    normalization,
    device,
    batch_size=ENCODE_BATCH_SIZE,
):
    mini_corpus = records_to_corpus(records)
    return encode_corpus_rows(
        mini_corpus,
        np.arange(len(records), dtype=np.int64),
        model,
        normalization,
        device,
        batch_size=batch_size,
        description="编码新增片段",
    )


def _model_index_groups(
    model,
    corpus,
    index_groups,
    normalization,
    device,
):
    """
    同一批次内的重复 SMILES 只编码一次，并让相同索引共享同一 dropout 结果。
    """
    lengths = [len(group) for group in index_groups]
    if not lengths or any(length == 0 for length in lengths):
        raise ValueError("模型索引组不能为空")
    combined = np.concatenate(index_groups).astype(np.int64, copy=False)
    unique_indices, inverse = np.unique(combined, return_inverse=True)
    unique_outputs = _model_rows(
        model,
        corpus,
        unique_indices,
        normalization,
        device,
    )
    inverse_tensor = torch.from_numpy(inverse).to(
        device=device,
        dtype=torch.long,
    )
    expanded = unique_outputs[inverse_tensor]
    return list(torch.split(expanded, lengths, dim=0))


def evaluate_model(model, corpus, pair_data, normalization, device):
    model.eval()
    pair_predictions = []
    with torch.no_grad():
        for start in range(
            0,
            len(pair_data["first"]),
            PAIR_BATCH_SIZE,
        ):
            end = min(
                start + PAIR_BATCH_SIZE,
                len(pair_data["first"]),
            )
            first = pair_data["first"][start:end]
            second = pair_data["second"][start:end]
            first_out, second_out = _model_index_groups(
                model,
                corpus,
                [first, second],
                normalization,
                device,
            )
            pair_predictions.append(
                (first_out * second_out).sum(dim=1).cpu().numpy()
            )

    predictions = np.concatenate(pair_predictions)
    targets = pair_data["target"]
    absolute_errors = np.abs(predictions - targets)
    similarity_loss = float(np.mean(
        np.where(
            absolute_errors < 1.0,
            0.5 * np.square(predictions - targets),
            absolute_errors - 0.5,
        )
    ))
    mae = float(np.mean(absolute_errors))

    triplets = pair_data["triplets"]
    rank_loss = 0.0
    rank_accuracy = 0.0
    if len(triplets) > 0:
        rank_correct = 0
        rank_losses = []
        with torch.no_grad():
            for start in range(0, len(triplets), PAIR_BATCH_SIZE):
                batch = triplets[start:start + PAIR_BATCH_SIZE]
                anchor_out, positive_out, negative_out = _model_index_groups(
                    model,
                    corpus,
                    [batch[:, 0], batch[:, 1], batch[:, 2]],
                    normalization,
                    device,
                )
                positive_sim = (anchor_out * positive_out).sum(dim=1)
                negative_sim = (anchor_out * negative_out).sum(dim=1)
                rank_correct += int(
                    (positive_sim > negative_sim).sum().item()
                )
                rank_losses.append(
                    F.relu(
                        RANK_MARGIN - positive_sim + negative_sim
                    ).cpu().numpy()
                )
        rank_loss = float(np.mean(np.concatenate(rank_losses)))
        rank_accuracy = float(rank_correct / len(triplets))

    return {
        "loss": float(
            SIMILARITY_LOSS_WEIGHT * similarity_loss
            + RANK_LOSS_WEIGHT * rank_loss
        ),
        "similarity_loss": similarity_loss,
        "rank_loss": rank_loss,
        "mae": mae,
        "rank_accuracy": rank_accuracy,
        "num_pairs": int(len(targets)),
        "num_triplets": int(len(triplets)),
    }


def calculate_neighbor_quality(
    model,
    corpus,
    validation_indices,
    normalization,
    device,
):
    """在固定验证子集上检查教师 Top-k 近邻召回率。"""
    evaluation_rng = np.random.default_rng(
        RANDOM_SEED + RNG_OFFSETS["evaluation"]
    )
    validation_indices = np.asarray(
        validation_indices,
        dtype=np.int64,
    )
    if len(validation_indices) > EVALUATION_MAX_SMILES:
        validation_indices = np.sort(
            evaluation_rng.choice(
                validation_indices,
                size=EVALUATION_MAX_SMILES,
                replace=False,
            )
        )
    if len(validation_indices) < 2:
        return {
            "evaluated_smiles": int(len(validation_indices)),
            "neighbor_k": 0,
            "topk_recall": 0.0,
            "predicted_topk_teacher_similarity": 0.0,
        }

    model.eval()
    embeddings = []
    with torch.no_grad():
        for start in range(
            0,
            len(validation_indices),
            PAIR_BATCH_SIZE,
        ):
            rows = validation_indices[start:start + PAIR_BATCH_SIZE]
            embeddings.append(
                _model_rows(
                    model,
                    corpus,
                    rows,
                    normalization,
                    device,
                ).cpu().numpy()
            )
    embeddings = np.concatenate(embeddings, axis=0)
    predicted_matrix = embeddings @ embeddings.T
    np.fill_diagonal(predicted_matrix, -np.inf)

    teacher_matrix = np.zeros(
        (len(validation_indices), len(validation_indices)),
        dtype=np.float32,
    )
    for row, anchor_idx in enumerate(validation_indices):
        anchor_values = np.full(
            (len(validation_indices),),
            anchor_idx,
            dtype=np.int64,
        )
        teacher_matrix[row] = teacher_similarity_for_pairs(
            corpus,
            anchor_values,
            validation_indices,
        )
    np.fill_diagonal(teacher_matrix, -np.inf)

    neighbor_k = min(NEIGHBOR_K, len(validation_indices) - 1)
    recalls = []
    selected_teacher_scores = []
    for row in range(len(validation_indices)):
        predicted_top = np.argpartition(
            predicted_matrix[row],
            -neighbor_k,
        )[-neighbor_k:]
        teacher_top = np.argpartition(
            teacher_matrix[row],
            -neighbor_k,
        )[-neighbor_k:]
        recalls.append(
            len(set(predicted_top) & set(teacher_top)) / neighbor_k
        )
        selected_teacher_scores.extend(
            teacher_matrix[row, predicted_top].tolist()
        )
    return {
        "evaluated_smiles": int(len(validation_indices)),
        "neighbor_k": int(neighbor_k),
        "topk_recall": float(np.mean(recalls)),
        "predicted_topk_teacher_similarity": float(
            np.mean(selected_teacher_scores)
        ),
    }


def _configuration_snapshot():
    return {
        "random_seed": RANDOM_SEED,
        "rng_offsets": dict(RNG_OFFSETS),
        "train_fraction": TRAIN_FRACTION,
        "epochs": EPOCHS,
        "pair_batch_size": PAIR_BATCH_SIZE,
        "encode_batch_size": ENCODE_BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "gradient_clip_norm": GRADIENT_CLIP_NORM,
        "similarity_loss_weight": SIMILARITY_LOSS_WEIGHT,
        "rank_loss_weight": RANK_LOSS_WEIGHT,
        "rank_margin": RANK_MARGIN,
        "min_teacher_rank_gap": MIN_TEACHER_RANK_GAP,
        "candidates_per_anchor": CANDIDATES_PER_ANCHOR,
        "near_candidate_count": NEAR_CANDIDATE_COUNT,
        "high_pairs_per_anchor": HIGH_PAIRS_PER_ANCHOR,
        "middle_pairs_per_anchor": MIDDLE_PAIRS_PER_ANCHOR,
        "low_pairs_per_anchor": LOW_PAIRS_PER_ANCHOR,
        "evaluation_max_smiles": EVALUATION_MAX_SMILES,
        "neighbor_k": NEIGHBOR_K,
        "require_zero_invalid_smiles": REQUIRE_ZERO_INVALID_SMILES,
    }


def train_encoder(corpus, corpus_metadata, device):
    split_rng = np.random.default_rng(
        RANDOM_SEED + RNG_OFFSETS["split"]
    )
    train_pair_rng = np.random.default_rng(
        RANDOM_SEED + RNG_OFFSETS["train_pairs"]
    )
    validation_pair_rng = np.random.default_rng(
        RANDOM_SEED + RNG_OFFSETS["validation_pairs"]
    )
    training_rng = np.random.default_rng(
        RANDOM_SEED + RNG_OFFSETS["training"]
    )

    train_indices, validation_indices = split_indices(
        len(corpus["smiles"]),
        TRAIN_FRACTION,
        split_rng,
    )
    normalization = calculate_scalar_normalization(
        corpus,
        train_indices,
    )
    validate_scalar_normalization(normalization)

    train_pairs = build_stratified_pairs(
        corpus,
        train_indices,
        train_pair_rng,
        "训练",
        require_triplets=RANK_LOSS_WEIGHT > 0.0,
    )
    validation_pairs = build_stratified_pairs(
        corpus,
        validation_indices,
        validation_pair_rng,
        "验证",
        require_triplets=False,
    )

    model_config = {
        "input_dim": EXPECTED_INPUT_DIM,
        "hidden_dim_1": HIDDEN_DIM_1,
        "hidden_dim_2": HIDDEN_DIM_2,
        "output_dim": OUTPUT_DIM,
        "dropout": DROPOUT,
    }
    model = FragmentEncoder(**model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    print(
        f"[数据划分] 训练 SMILES={len(train_indices)}, "
        f"验证 SMILES={len(validation_indices)}"
    )
    print(
        f"[训练配对] pairs={train_pairs['stats']['num_pairs']}, "
        f"valid_triplets={train_pairs['stats']['num_triplets']}, "
        "跳过弱排序="
        f"{train_pairs['stats']['triplets_skipped_small_gap']}"
    )
    print(
        f"[验证配对] pairs={validation_pairs['stats']['num_pairs']}, "
        f"valid_triplets={validation_pairs['stats']['num_triplets']}, "
        "跳过弱排序="
        f"{validation_pairs['stats']['triplets_skipped_small_gap']}"
    )

    best_state = None
    best_metrics = None
    best_epoch = 0
    stale_epochs = 0
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        pair_order = training_rng.permutation(
            len(train_pairs["first"])
        )
        triplets = train_pairs["triplets"]
        weighted_loss_sum = 0.0
        weighted_similarity_loss_sum = 0.0
        weighted_rank_loss_sum = 0.0
        sample_count = 0
        # 排序三元组按批次有放回抽样；这里统计本轮实际抽取次数，
        # 不等同于配对阶段生成的唯一有效三元组数量。
        rank_sample_count = 0
        rank_correct_count = 0

        progress = tqdm(
            range(0, len(pair_order), PAIR_BATCH_SIZE),
            desc=f"Epoch {epoch:03d}/{EPOCHS}",
            leave=False,
        )
        for start in progress:
            selected = pair_order[start:start + PAIR_BATCH_SIZE]
            first = train_pairs["first"][selected]
            second = train_pairs["second"][selected]
            targets = torch.from_numpy(
                train_pairs["target"][selected]
            ).to(device=device, dtype=torch.float32)

            optimizer.zero_grad(set_to_none=True)
            if RANK_LOSS_WEIGHT > 0.0:
                triplet_count = min(len(selected), len(triplets))
                triplet_rows = triplets[
                    training_rng.integers(
                        0,
                        len(triplets),
                        size=triplet_count,
                    )
                ]
                (
                    first_out,
                    second_out,
                    anchor_out,
                    positive_out,
                    negative_out,
                ) = _model_index_groups(
                    model,
                    corpus,
                    [
                        first,
                        second,
                        triplet_rows[:, 0],
                        triplet_rows[:, 1],
                        triplet_rows[:, 2],
                    ],
                    normalization,
                    device,
                )
            else:
                first_out, second_out = _model_index_groups(
                    model,
                    corpus,
                    [first, second],
                    normalization,
                    device,
                )

            predicted_similarity = (first_out * second_out).sum(dim=1)
            similarity_loss = F.smooth_l1_loss(
                predicted_similarity,
                targets,
            )

            if RANK_LOSS_WEIGHT > 0.0:
                positive_similarity = (
                    anchor_out * positive_out
                ).sum(dim=1)
                negative_similarity = (
                    anchor_out * negative_out
                ).sum(dim=1)
                rank_loss = F.relu(
                    RANK_MARGIN
                    - positive_similarity
                    + negative_similarity
                ).mean()
                batch_rank_correct = int(
                    (
                        positive_similarity > negative_similarity
                    ).sum().item()
                )
            else:
                triplet_count = 0
                rank_loss = similarity_loss.new_zeros(())
                batch_rank_correct = 0

            loss = (
                SIMILARITY_LOSS_WEIGHT * similarity_loss
                + RANK_LOSS_WEIGHT * rank_loss
            )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(
                    f"Epoch {epoch} 出现非有限训练损失"
                )
            loss.backward()
            if GRADIENT_CLIP_NORM is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=GRADIENT_CLIP_NORM,
                )
            optimizer.step()

            batch_count = len(selected)
            weighted_loss_sum += float(loss.item()) * batch_count
            weighted_similarity_loss_sum += (
                float(similarity_loss.item()) * batch_count
            )
            sample_count += batch_count
            if triplet_count > 0:
                weighted_rank_loss_sum += (
                    float(rank_loss.item()) * triplet_count
                )
                rank_sample_count += triplet_count
                rank_correct_count += batch_rank_correct

            progress.set_postfix(
                total=f"{loss.item():.4f}",
                sim=f"{similarity_loss.item():.4f}",
                rank=f"{rank_loss.item():.4f}",
            )

        validation_metrics = evaluate_model(
            model,
            corpus,
            validation_pairs,
            normalization,
            device,
        )
        record = {
            "epoch": epoch,
            "train_loss": float(weighted_loss_sum / sample_count),
            "train_similarity_loss": float(
                weighted_similarity_loss_sum / sample_count
            ),
            "train_rank_loss": float(
                weighted_rank_loss_sum / rank_sample_count
                if rank_sample_count > 0
                else 0.0
            ),
            "train_rank_accuracy": float(
                rank_correct_count / rank_sample_count
                if rank_sample_count > 0
                else 0.0
            ),
            "train_rank_samples": int(rank_sample_count),
            "validation_loss": validation_metrics["loss"],
            "validation_similarity_loss": (
                validation_metrics["similarity_loss"]
            ),
            "validation_rank_loss": validation_metrics["rank_loss"],
            "validation_mae": validation_metrics["mae"],
            "validation_rank_accuracy": (
                validation_metrics["rank_accuracy"]
            ),
            "validation_rank_samples": int(
                validation_metrics["num_triplets"]
            ),
        }
        history.append(record)

        improved = (
            best_metrics is None
            or record["validation_loss"]
            < best_metrics["validation_loss"]
        )
        if improved:
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            best_metrics = dict(record)
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1

        print(
            f"Epoch {epoch:03d} | "
            f"train_total={record['train_loss']:.5f} | "
            f"train_sim={record['train_similarity_loss']:.5f} | "
            f"train_rank={record['train_rank_loss']:.5f} | "
            f"train_rank_acc={record['train_rank_accuracy']:.3f} | "
            f"rank_draws={record['train_rank_samples']}"
        )
        print(
            "          | "
            f"val_total={record['validation_loss']:.5f} | "
            f"val_sim={record['validation_similarity_loss']:.5f} | "
            f"val_rank={record['validation_rank_loss']:.5f} | "
            f"val_rank_acc={record['validation_rank_accuracy']:.3f} | "
            f"val_triplets={record['validation_rank_samples']} | "
            f"val_MAE={record['validation_mae']:.5f}"
        )
        if stale_epochs >= EARLY_STOPPING_PATIENCE:
            print(
                f"[早停] 连续 {EARLY_STOPPING_PATIENCE} 轮未改善，"
                f"最佳轮次为 {best_epoch}"
            )
            break

    if best_state is None:
        raise RuntimeError("训练未得到有效模型权重")
    model.load_state_dict(best_state, strict=True)

    neighbor_quality = calculate_neighbor_quality(
        model,
        corpus,
        validation_indices,
        normalization,
        device,
    )
    print(
        f"[近邻验证] Top-{neighbor_quality['neighbor_k']} "
        f"召回率={neighbor_quality['topk_recall']:.4f}, "
        "预测近邻平均教师相似度="
        f"{neighbor_quality['predicted_topk_teacher_similarity']:.4f}"
    )

    teacher_config = {
        "ecfp_weight": TEACHER_ECFP_WEIGHT,
        "fcfp_weight": TEACHER_FCFP_WEIGHT,
        "intrinsic_weight": TEACHER_INTRINSIC_WEIGHT,
        "intrinsic_component_weights": {
            "hac": INTRINSIC_HAC_WEIGHT,
            "ring_count": INTRINSIC_RING_WEIGHT,
            "formal_charge": INTRINSIC_CHARGE_WEIGHT,
            "element_composition": INTRINSIC_ELEMENT_WEIGHT,
            "is_metal": INTRINSIC_METAL_WEIGHT,
        },
        "zero_metal_mismatch_similarity": (
            ZERO_METAL_MISMATCH_SIMILARITY
        ),
    }
    checkpoint = {
        "model_class": "FragmentEncoder",
        "model_state_dict": best_state,
        "model_config": model_config,
        "normalization": normalization,
        "feature_config": {
            "fingerprint": corpus_metadata["fingerprint"],
            "intrinsic_features": corpus_metadata["intrinsic_features"],
        },
        "teacher_config": teacher_config,
        "training_config": _configuration_snapshot(),
        "source": {
            "raw_corpus_path": os.path.abspath(RAW_CORPUS_PATH),
            "raw_corpus_metadata_path": os.path.abspath(
                RAW_CORPUS_METADATA_PATH
            ),
        },
        "corpus_metadata": corpus_metadata,
        "training_corpus_sha256": corpus_metadata["corpus_sha256"],
        "data_split": {
            "method": "random_disjoint_smiles",
            "train_indices": torch.from_numpy(train_indices.copy()),
            "validation_indices": torch.from_numpy(
                validation_indices.copy()
            ),
            "train_count": int(len(train_indices)),
            "validation_count": int(len(validation_indices)),
        },
        "pair_sampling": {
            "train": train_pairs["stats"],
            "validation": validation_pairs["stats"],
        },
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "neighbor_quality": neighbor_quality,
        "history": history,
    }
    return checkpoint


def save_checkpoint_atomic(checkpoint, path, overwrite):
    path = os.path.abspath(path)
    if os.path.exists(path) and not overwrite:
        raise FileExistsError(
            f"checkpoint 已存在且不允许覆盖: {path}"
        )

    os.makedirs(os.path.dirname(path), exist_ok=True)
    temp_path = (
        f"{path}.{os.getpid()}-{uuid.uuid4().hex}.tmp"
    )
    try:
        torch.save(checkpoint, temp_path)
        reloaded = torch_load_compat(
            temp_path, map_location="cpu"
        )
        validate_checkpoint_payload(
            reloaded, expected_dim=OUTPUT_DIM
        )
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def main():
    validate_configuration()
    validate_source_paths()
    set_random_seed(RANDOM_SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"[训练启动] 使用设备: {device}")
    print(f"[训练启动] 输出维度: {OUTPUT_DIM}")
    print(f"[训练启动] 粗语料: {RAW_CORPUS_PATH}")
    print(f"[训练启动] checkpoint: {CHECKPOINT_PATH}")
    print(f"[训练启动] 标准词表: {EMBEDDING_TABLE_PATH}")
    print(f"[训练启动] 可读 JSON 镜像: {EMBEDDING_JSON_PATH}")

    corpus, corpus_metadata = utils.load_raw_fragment_corpus(
        RAW_CORPUS_PATH,
        RAW_CORPUS_METADATA_PATH,
    )
    validate_loaded_corpus(corpus, corpus_metadata)
    print(
        f"[粗语料校验完成] 唯一 SMILES={len(corpus['smiles'])}, "
        f"SHA256={corpus_metadata['corpus_sha256']}"
    )

    checkpoint = train_encoder(
        corpus,
        corpus_metadata,
        device,
    )
    save_checkpoint_atomic(
        checkpoint,
        CHECKPOINT_PATH,
        overwrite=OVERWRITE_CHECKPOINT,
    )
    print(
        f"[训练完成] 最佳轮次={checkpoint['best_epoch']}, "
        f"checkpoint 已保存: {CHECKPOINT_PATH}"
    )

    best_checkpoint = load_checkpoint_payload(
        CHECKPOINT_PATH,
        expected_dim=OUTPUT_DIM,
    )
    validate_checkpoint_against_corpus_metadata(
        best_checkpoint, corpus_metadata
    )
    export_model = build_model_from_checkpoint(
        best_checkpoint, device
    )
    embeddings = encode_corpus_rows(
        corpus,
        np.arange(len(corpus["smiles"]), dtype=np.int64),
        export_model,
        best_checkpoint["normalization"],
        device,
        batch_size=ENCODE_BATCH_SIZE,
        description="导出完整标准128维片段词表",
    )
    table_metadata = save_fragment_embedding_table(
        corpus,
        embeddings,
        checkpoint_path=CHECKPOINT_PATH,
        checkpoint=best_checkpoint,
        raw_corpus_sha256=corpus_metadata["corpus_sha256"],
        npz_path=EMBEDDING_TABLE_PATH,
        metadata_path=EMBEDDING_METADATA_PATH,
    )
    utils.atomic_write_json(
        TRAINING_HISTORY_PATH,
        {
            "checkpoint_sha256": table_metadata[
                "checkpoint_sha256"
            ],
            "training_raw_corpus_sha256": corpus_metadata[
                "corpus_sha256"
            ],
            "embedding_table_sha256": table_metadata[
                "table_sha256"
            ],
            "embedding_json_mirror_sha256": table_metadata[
                "json_mirror"
            ]["sha256"],
            "best_epoch": best_checkpoint["best_epoch"],
            "best_metrics": best_checkpoint["best_metrics"],
            "neighbor_quality": best_checkpoint["neighbor_quality"],
            "pair_sampling": best_checkpoint["pair_sampling"],
            "training_config": best_checkpoint["training_config"],
            "history": best_checkpoint["history"],
        },
    )
    print(
        "[标准词表完成] "
        f"片段={table_metadata['num_smiles']}, "
        f"维度={table_metadata['embedding_dim']}"
    )
    print(f"[标准词表] {EMBEDDING_TABLE_PATH}")
    print(f"[完整可读 JSON 镜像] {EMBEDDING_JSON_PATH}")
    print(f"[标准词表元数据] {EMBEDDING_METADATA_PATH}")
    print(f"[训练历史] {TRAINING_HISTORY_PATH}")
    print("[任务结束] 未读取或修改任何 PyGData。")


if __name__ == "__main__":
    main()
