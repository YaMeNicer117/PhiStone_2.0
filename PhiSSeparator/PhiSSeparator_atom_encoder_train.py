import argparse
import hashlib
import json
import math
import os
import random
import uuid

import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Subset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv

try:
    import utils
except ImportError:  # 支持作为 PhiSSeparator 子模块导入
    from . import utils


# =============================================================================
# 用户配置区：网络与训练超参数
# =============================================================================

# 固定输出为 64 维；该维度与原子词表格式及下游读取逻辑绑定。
RANDOM_SEED = 42
ATOM_EMBEDDING_DIM = 64

# 数据划分与训练轮次
VALIDATION_FRACTION = 0.10
EPOCHS = 200
TRAIN_BATCH_SIZE = 256
ENCODE_BATCH_SIZE = 128

# AdamW 优化器
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5

# 早停设置
EARLY_STOPPING_PATIENCE = 15
EARLY_STOPPING_MIN_DELTA = 1e-6

# 自监督遮蔽与模型正则化
MASK_PROBABILITY = 0.2
DROPOUT = 0.1

# 原子、键重建损失的相对权重
ATOM_LOSS_WEIGHT = 1.0
BOND_LOSS_WEIGHT = 1.0

# 设为正数时启用梯度裁剪；None 表示不裁剪。
GRADIENT_CLIP_NORM = 5.0

# DataLoader 与计算设备
NUM_WORKERS = 0
DEVICE = "auto"


# =============================================================================
# 固定特征列与输出路径
# =============================================================================

ATOM_CATEGORICAL_COLUMNS = (0, 5, 7, 8, 9)
ATOM_NUMERIC_COLUMNS = (1, 2, 3, 4, 6)
NUMERIC_CLIP_SIGMA = 6.0
BOND_CATEGORICAL_COLUMNS = (0, 4)
BOND_BOOLEAN_COLUMNS = (1, 2, 3)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
SHARED_DATA_DIR = os.path.join(
    PROJECT_ROOT,
    "Datas",
    "Processed_datas",
    "SE3TD_128_Shared_data",
)
DEFAULT_RAW_CORPUS = os.path.join(
    SHARED_DATA_DIR, "fragment_atom_raw_corpus.npz"
)
DEFAULT_RAW_METADATA = os.path.join(
    SHARED_DATA_DIR, "fragment_atom_raw_corpus_metadata.json"
)
DEFAULT_CHECKPOINT = os.path.join(
    SCRIPT_DIR, "training_artifacts", "atom_encoder_64.pth"
)
DEFAULT_EMBEDDING_TABLE = os.path.join(
    SHARED_DATA_DIR, "fragment_atom_embeddings_64d.npz"
)
DEFAULT_EMBEDDING_JSON = utils.derive_embedding_json_mirror_path(
    DEFAULT_EMBEDDING_TABLE
)
DEFAULT_EMBEDDING_METADATA = os.path.join(
    SHARED_DATA_DIR, "fragment_atom_embeddings_64d_metadata.json"
)
DEFAULT_TRAINING_HISTORY = os.path.join(
    os.path.dirname(DEFAULT_CHECKPOINT), "atom_encoder_training_history.json"
)

_ATOM_EMBEDDING_TABLE_FIELDS = (
    "smiles",
    "atom_offsets",
    "canonical_atom_id",
    "atom_embeddings",
)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def file_sha256(path, block_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as file_obj:
        while True:
            block = file_obj.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _embedding_table_checksum(arrays):
    digest = hashlib.sha256()
    for key in _ATOM_EMBEDDING_TABLE_FIELDS:
        values = np.asarray(arrays[key])
        digest.update(key.encode("utf-8"))
        digest.update(json.dumps(list(values.shape)).encode("ascii"))
        if key == "smiles":
            for smiles in values.astype(str).tolist():
                encoded = smiles.encode("utf-8")
                digest.update(len(encoded).to_bytes(8, "little"))
                digest.update(encoded)
        else:
            contiguous = np.ascontiguousarray(values)
            digest.update(contiguous.dtype.str.encode("ascii"))
            digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _embedding_table_array_checksums(arrays):
    checksums = {}
    for key in _ATOM_EMBEDDING_TABLE_FIELDS:
        values = np.asarray(arrays[key])
        digest = hashlib.sha256()
        if key == "smiles":
            for smiles in values.astype(str).tolist():
                encoded = smiles.encode("utf-8")
                digest.update(len(encoded).to_bytes(8, "little"))
                digest.update(encoded)
        else:
            contiguous = np.ascontiguousarray(values)
            digest.update(contiguous.dtype.str.encode("ascii"))
            digest.update(contiguous.tobytes())
        checksums[key] = digest.hexdigest()
    return checksums


def _build_embedding_table_metadata(
    arrays,
    checkpoint_sha256,
    training_raw_corpus_sha256,
    current_raw_corpus_sha256,
    training_smiles_count,
    model_config,
    observed_atomic_numbers,
):
    num_smiles = len(arrays["smiles"])
    return {
        "embedding_dim": ATOM_EMBEDDING_DIM,
        "checkpoint_sha256": str(checkpoint_sha256),
        "training_raw_corpus_sha256": str(training_raw_corpus_sha256),
        "current_raw_corpus_sha256": str(current_raw_corpus_sha256),
        "training_smiles_count": int(training_smiles_count),
        "resolver_added_smiles_count": int(
            num_smiles - int(training_smiles_count)
        ),
        "num_smiles": int(num_smiles),
        "num_atoms": int(len(arrays["canonical_atom_id"])),
        "model_config": model_config,
        "observed_atomic_numbers": [
            int(value) for value in observed_atomic_numbers
        ],
        "array_shapes": {
            key: list(np.asarray(value).shape)
            for key, value in arrays.items()
        },
        "array_dtypes": {
            key: np.asarray(value).dtype.str
            for key, value in arrays.items()
        },
        "array_sha256": _embedding_table_array_checksums(arrays),
        "table_sha256": _embedding_table_checksum(arrays),
    }


def _validate_embedding_table(arrays, metadata):
    missing = set(_ATOM_EMBEDDING_TABLE_FIELDS) - set(arrays)
    if missing:
        raise ValueError(f"标准原子词嵌入表缺少字段: {sorted(missing)}")
    if metadata.get("embedding_dim") != ATOM_EMBEDDING_DIM:
        raise ValueError("标准原子词嵌入维度不兼容")

    smiles = np.asarray(arrays["smiles"]).astype(str)
    atom_offsets = np.asarray(arrays["atom_offsets"])
    canonical_atom_id = np.asarray(arrays["canonical_atom_id"])
    atom_embeddings = np.asarray(arrays["atom_embeddings"])
    if atom_offsets.dtype != np.int64:
        raise ValueError("标准词表 atom_offsets 必须为 int64")
    if canonical_atom_id.dtype != np.int32:
        raise ValueError("标准词表 canonical_atom_id 必须为 int32")
    if atom_embeddings.dtype != np.float32:
        raise ValueError("标准词表 atom_embeddings 必须为 float32")
    if atom_offsets.shape != (len(smiles) + 1,):
        raise ValueError("标准词表 atom_offsets 形状错误")
    if atom_offsets[0] != 0 or atom_offsets[-1] != len(canonical_atom_id):
        raise ValueError("标准词表 atom_offsets 起止值错误")
    if np.any(np.diff(atom_offsets) < 0):
        raise ValueError("标准词表 atom_offsets 必须单调不减")
    if atom_embeddings.shape != (
        len(canonical_atom_id), ATOM_EMBEDDING_DIM
    ):
        raise ValueError("标准原子词嵌入矩阵形状错误")
    if not np.all(np.isfinite(atom_embeddings)):
        raise ValueError("标准原子词嵌入包含 NaN 或 Inf")
    if len(set(smiles.tolist())) != len(smiles):
        raise ValueError("标准原子词嵌入表包含重复 SMILES")
    for fragment_index, smiles_value in enumerate(smiles.tolist()):
        atom_start = int(atom_offsets[fragment_index])
        atom_end = int(atom_offsets[fragment_index + 1])
        expected_ids = np.arange(atom_end - atom_start, dtype=np.int32)
        if not np.array_equal(
            canonical_atom_id[atom_start:atom_end], expected_ids
        ):
            raise ValueError(
                f"标准词表片段 {smiles_value} 的规范原子编号不连续"
            )

    expected_shapes = {
        key: list(np.asarray(value).shape)
        for key, value in arrays.items()
    }
    expected_dtypes = {
        key: np.asarray(value).dtype.str
        for key, value in arrays.items()
    }
    if metadata.get("array_shapes") != expected_shapes:
        raise ValueError("标准词表元数据中的数组形状不一致")
    if metadata.get("array_dtypes") != expected_dtypes:
        raise ValueError("标准词表元数据中的数组类型不一致")
    if metadata.get("array_sha256") != (
        _embedding_table_array_checksums(arrays)
    ):
        raise ValueError("标准词表元数据中的逐数组校验值不一致")
    if metadata.get("num_smiles") != len(smiles):
        raise ValueError("标准词表元数据中的片段数量不一致")
    if metadata.get("num_atoms") != len(canonical_atom_id):
        raise ValueError("标准词表元数据中的原子数量不一致")
    if metadata.get("table_sha256") != _embedding_table_checksum(arrays):
        raise ValueError("标准原子词嵌入表校验值不一致")
    training_count = metadata.get("training_smiles_count")
    resolver_count = metadata.get("resolver_added_smiles_count")
    if (
        not isinstance(training_count, int)
        or not isinstance(resolver_count, int)
        or training_count < 0
        or resolver_count < 0
        or training_count + resolver_count != len(smiles)
    ):
        raise ValueError("标准词表训练/动态词汇计数不一致")
    json_mirror = metadata.get("json_mirror")
    if json_mirror is not None:
        expected_json_metadata = {
            "source_table_sha256": metadata["table_sha256"],
            "table_kind": "atom",
            "layout": "json_array_one_fragment_per_line",
            "authoritative": False,
            "num_smiles": len(smiles),
            "num_atoms": len(canonical_atom_id),
            "embedding_dim": ATOM_EMBEDDING_DIM,
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
            raise ValueError("标准原子词嵌入 JSON 镜像元数据不一致")


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
                for key in _ATOM_EMBEDDING_TABLE_FIELDS
            }
        reloaded["smiles"] = reloaded["smiles"].astype(str)
        _validate_embedding_table(reloaded, metadata)
        metadata["json_mirror"] = (
            utils.write_embedding_json_mirror_file(
                temp_json,
                reloaded,
                table_kind="atom",
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


def save_atom_embedding_table(
    raw_corpus,
    atom_embeddings,
    npz_path,
    metadata_path,
    checkpoint_sha256,
    training_raw_corpus_sha256,
    current_raw_corpus_sha256,
    model_config,
    observed_atomic_numbers,
):
    atom_embeddings = np.asarray(atom_embeddings, dtype=np.float32)
    arrays = {
        "smiles": np.asarray(raw_corpus["smiles"], dtype=np.str_),
        "atom_offsets": np.asarray(
            raw_corpus["atom_offsets"], dtype=np.int64
        ),
        "canonical_atom_id": np.asarray(
            raw_corpus["canonical_atom_id"], dtype=np.int32
        ),
        "atom_embeddings": atom_embeddings,
    }
    metadata = _build_embedding_table_metadata(
        arrays,
        checkpoint_sha256=checkpoint_sha256,
        training_raw_corpus_sha256=training_raw_corpus_sha256,
        current_raw_corpus_sha256=current_raw_corpus_sha256,
        training_smiles_count=len(arrays["smiles"]),
        model_config=model_config,
        observed_atomic_numbers=observed_atomic_numbers,
    )
    _atomic_write_embedding_table(
        arrays, metadata, npz_path, metadata_path
    )
    return metadata


def load_atom_embedding_table(npz_path, metadata_path):
    with open(metadata_path, "r", encoding="utf-8") as file_obj:
        metadata = json.load(file_obj)
    with np.load(npz_path, allow_pickle=False) as loaded:
        missing = set(_ATOM_EMBEDDING_TABLE_FIELDS) - set(loaded.files)
        if missing:
            raise ValueError(
                f"标准原子词嵌入表缺少字段: {sorted(missing)}"
            )
        arrays = {
            key: np.array(loaded[key], copy=True)
            for key in _ATOM_EMBEDDING_TABLE_FIELDS
        }
    arrays["smiles"] = arrays["smiles"].astype(str)
    _validate_embedding_table(arrays, metadata)
    return arrays, metadata


def repair_atom_embedding_metadata(
    npz_path,
    metadata_path,
    current_raw_corpus_sha256,
):
    """
    仅用于持锁事务恢复：根据完整 NPZ 与旧元数据重建匹配的元数据。
    """
    with open(metadata_path, "r", encoding="utf-8") as file_obj:
        previous_metadata = json.load(file_obj)
    with np.load(npz_path, allow_pickle=False) as loaded:
        missing = set(_ATOM_EMBEDDING_TABLE_FIELDS) - set(loaded.files)
        if missing:
            raise ValueError(
                f"待恢复标准原子词表 NPZ 缺少字段: {sorted(missing)}"
            )
        arrays = {
            key: np.array(loaded[key], copy=True)
            for key in _ATOM_EMBEDDING_TABLE_FIELDS
        }
    arrays["smiles"] = arrays["smiles"].astype(str)
    repaired_metadata = _build_embedding_table_metadata(
        arrays,
        checkpoint_sha256=previous_metadata["checkpoint_sha256"],
        training_raw_corpus_sha256=previous_metadata[
            "training_raw_corpus_sha256"
        ],
        current_raw_corpus_sha256=current_raw_corpus_sha256,
        training_smiles_count=int(
            previous_metadata["training_smiles_count"]
        ),
        model_config=previous_metadata["model_config"],
        observed_atomic_numbers=previous_metadata[
            "observed_atomic_numbers"
        ],
    )
    _validate_embedding_table(arrays, repaired_metadata)
    json_path = utils.derive_embedding_json_mirror_path(npz_path)
    repaired_metadata["json_mirror"] = (
        utils.atomic_write_embedding_json_mirror(
            json_path,
            arrays,
            table_kind="atom",
            source_table_sha256=repaired_metadata["table_sha256"],
        )
    )
    _validate_embedding_table(arrays, repaired_metadata)
    utils.atomic_write_json(metadata_path, repaired_metadata)
    return arrays, repaired_metadata


def append_atom_embedding_record(
    canonical_smiles,
    canonical_atom_id,
    atom_embeddings,
    npz_path,
    metadata_path,
    current_raw_corpus_sha256,
):
    """调用方持有写锁时，向标准词表尾部追加一个片段。"""
    arrays, metadata = load_atom_embedding_table(npz_path, metadata_path)
    smiles_list = arrays["smiles"].astype(str).tolist()
    if canonical_smiles in smiles_list:
        return smiles_list.index(canonical_smiles), metadata

    canonical_atom_id = np.asarray(canonical_atom_id, dtype=np.int32)
    atom_embeddings = np.asarray(atom_embeddings, dtype=np.float32)
    if canonical_atom_id.shape != (len(atom_embeddings),):
        raise ValueError("新增标准词表记录的原子编号形状错误")
    if atom_embeddings.shape != (
        len(canonical_atom_id), ATOM_EMBEDDING_DIM
    ):
        raise ValueError("新增标准词表记录的嵌入形状错误")
    expected_ids = np.arange(len(canonical_atom_id), dtype=np.int32)
    if not np.array_equal(canonical_atom_id, expected_ids):
        raise ValueError("新增标准词表记录的规范原子编号不连续")
    if not np.all(np.isfinite(atom_embeddings)):
        raise ValueError("新增标准词表记录包含 NaN 或 Inf")

    updated = {
        "smiles": np.concatenate([
            arrays["smiles"],
            np.asarray([canonical_smiles], dtype=np.str_),
        ]),
        "atom_offsets": np.concatenate([
            arrays["atom_offsets"],
            np.asarray([
                int(arrays["atom_offsets"][-1]) + len(canonical_atom_id)
            ], dtype=np.int64),
        ]),
        "canonical_atom_id": np.concatenate([
            arrays["canonical_atom_id"], canonical_atom_id
        ]).astype(np.int32, copy=False),
        "atom_embeddings": np.concatenate([
            arrays["atom_embeddings"], atom_embeddings
        ], axis=0).astype(np.float32, copy=False),
    }
    updated_metadata = _build_embedding_table_metadata(
        updated,
        checkpoint_sha256=metadata["checkpoint_sha256"],
        training_raw_corpus_sha256=metadata[
            "training_raw_corpus_sha256"
        ],
        current_raw_corpus_sha256=current_raw_corpus_sha256,
        training_smiles_count=int(metadata["training_smiles_count"]),
        model_config=metadata["model_config"],
        observed_atomic_numbers=metadata["observed_atomic_numbers"],
    )
    _atomic_write_embedding_table(
        updated, updated_metadata, npz_path, metadata_path
    )
    return len(smiles_list), updated_metadata


class FragmentAtomDataset(torch.utils.data.Dataset):
    def __init__(self, corpus):
        self.corpus = corpus

    def __len__(self):
        return len(self.corpus["smiles"])

    def __getitem__(self, index):
        record = utils.extract_fragment_atom_record(self.corpus, index)
        atom_features = torch.tensor(
            record["atom_features"], dtype=torch.long
        )
        bond_atom_ids = torch.tensor(
            record["bond_atom_ids"], dtype=torch.long
        ).reshape(-1, 2)
        bond_index = bond_atom_ids.t().contiguous()
        bond_features = torch.tensor(
            record["bond_features"], dtype=torch.long
        ).reshape(-1, len(utils.RAW_BOND_FEATURE_COLUMNS))
        return Data(
            atom_features=atom_features,
            bond_index=bond_index,
            bond_features=bond_features,
            num_nodes=atom_features.size(0),
            fragment_index=torch.tensor([int(index)], dtype=torch.long),
        )


def build_data_from_atom_record(record):
    atom_features = torch.tensor(
        record["atom_features"], dtype=torch.long
    )
    bond_atom_ids = torch.tensor(
        record["bond_atom_ids"], dtype=torch.long
    ).reshape(-1, 2)
    return Data(
        atom_features=atom_features,
        bond_index=bond_atom_ids.t().contiguous(),
        bond_features=torch.tensor(
            record["bond_features"], dtype=torch.long
        ).reshape(-1, len(utils.RAW_BOND_FEATURE_COLUMNS)),
        num_nodes=atom_features.size(0),
    )


def _fragment_atom_rows(corpus, fragment_indices):
    blocks = []
    for fragment_index in fragment_indices:
        atom_start = int(corpus["atom_offsets"][fragment_index])
        atom_end = int(corpus["atom_offsets"][fragment_index + 1])
        blocks.append(corpus["atom_features"][atom_start:atom_end])
    if not blocks:
        raise ValueError("用于统计训练归一化参数的片段集合为空")
    return np.concatenate(blocks, axis=0)


def build_model_config(
    corpus,
    normalization_fragment_indices=None,
    dropout=DROPOUT,
):
    atom_features = np.asarray(corpus["atom_features"])
    bond_features = np.asarray(corpus["bond_features"])
    if normalization_fragment_indices is None:
        normalization_atom_features = atom_features
        normalization_smiles_count = len(corpus["smiles"])
    else:
        normalization_fragment_indices = [
            int(index) for index in normalization_fragment_indices
        ]
        normalization_atom_features = _fragment_atom_rows(
            corpus, normalization_fragment_indices
        )
        normalization_smiles_count = len(
            normalization_fragment_indices
        )
    numeric_values = normalization_atom_features[
        :, ATOM_NUMERIC_COLUMNS
    ].astype(np.float32)
    numeric_mean = numeric_values.mean(axis=0)
    numeric_std = numeric_values.std(axis=0)
    numeric_std = np.where(numeric_std < 1e-6, 1.0, numeric_std)

    atom_target_values = []
    for column_index in range(len(utils.RAW_ATOM_FEATURE_COLUMNS)):
        if column_index == 0:
            values = list(range(1, 119))
        elif column_index == 5:
            values = list(range(
                len(utils.RAW_ATOM_HYBRIDIZATION_CATEGORIES)
            ))
        elif column_index == 7:
            values = list(range(len(utils.RAW_ATOM_CHIRAL_CATEGORIES)))
        elif column_index in (8, 9):
            values = [0, 1]
        else:
            values = sorted({
                int(value) for value in atom_features[:, column_index]
            })
        atom_target_values.append(values)

    bond_target_values = []
    for column_index in range(len(utils.RAW_BOND_FEATURE_COLUMNS)):
        if column_index == 0:
            values = list(range(len(utils.RAW_BOND_TYPE_CATEGORIES)))
        elif column_index == 4:
            values = list(range(len(utils.RAW_BOND_STEREO_CATEGORIES)))
        else:
            values = [0, 1]
        bond_target_values.append(values)

    return {
        "gnn_operator": "GINEConv",
        "embedding_dim": ATOM_EMBEDDING_DIM,
        "num_gnn_layers": 3,
        "dropout": float(dropout),
        "atom_feature_columns": list(
            utils.RAW_ATOM_FEATURE_COLUMNS
        ),
        "bond_feature_columns": list(
            utils.RAW_BOND_FEATURE_COLUMNS
        ),
        "normalization_smiles_count": normalization_smiles_count,
        "atom_numeric_mean": numeric_mean.astype(float).tolist(),
        "atom_numeric_std": numeric_std.astype(float).tolist(),
        "atom_target_values": atom_target_values,
        "bond_target_values": bond_target_values,
    }


class AtomContextEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.get("gnn_operator") != "GINEConv":
            raise ValueError("AtomContextEncoder GNN 算子不兼容")
        if int(config["embedding_dim"]) != ATOM_EMBEDDING_DIM:
            raise ValueError("AtomContextEncoder 仅支持 64 维")
        if int(config["num_gnn_layers"]) != 3:
            raise ValueError("AtomContextEncoder 必须使用三层 GINE")
        if not 0.0 <= float(config["dropout"]) < 1.0:
            raise ValueError("AtomContextEncoder dropout 配置非法")
        if config.get("atom_feature_columns") != list(
            utils.RAW_ATOM_FEATURE_COLUMNS
        ):
            raise ValueError("AtomContextEncoder 原子特征列不兼容")
        if config.get("bond_feature_columns") != list(
            utils.RAW_BOND_FEATURE_COLUMNS
        ):
            raise ValueError("AtomContextEncoder 键特征列不兼容")
        numeric_mean = np.asarray(
            config.get("atom_numeric_mean"), dtype=np.float32
        )
        numeric_std = np.asarray(
            config.get("atom_numeric_std"), dtype=np.float32
        )
        if (
            numeric_mean.shape != (len(ATOM_NUMERIC_COLUMNS),)
            or numeric_std.shape != (len(ATOM_NUMERIC_COLUMNS),)
            or not np.all(np.isfinite(numeric_mean))
            or not np.all(np.isfinite(numeric_std))
            or np.any(numeric_std <= 0.0)
        ):
            raise ValueError("AtomContextEncoder 数值归一化参数非法")
        atom_targets = config.get("atom_target_values")
        bond_targets = config.get("bond_target_values")
        if (
            not isinstance(atom_targets, list)
            or len(atom_targets) != len(utils.RAW_ATOM_FEATURE_COLUMNS)
            or any(not values for values in atom_targets)
        ):
            raise ValueError("AtomContextEncoder 原子预测头配置非法")
        if (
            not isinstance(bond_targets, list)
            or len(bond_targets) != len(utils.RAW_BOND_FEATURE_COLUMNS)
            or any(not values for values in bond_targets)
        ):
            raise ValueError("AtomContextEncoder 键预测头配置非法")
        self.config = config
        self.dropout_probability = float(config["dropout"])

        self.atomic_number_embedding = nn.Embedding(120, 64)
        self.hybridization_embedding = nn.Embedding(
            len(utils.RAW_ATOM_HYBRIDIZATION_CATEGORIES) + 1, 64
        )
        self.chiral_embedding = nn.Embedding(
            len(utils.RAW_ATOM_CHIRAL_CATEGORIES) + 1, 64
        )
        self.aromatic_embedding = nn.Embedding(3, 64)
        self.ring_embedding = nn.Embedding(3, 64)
        self.atom_numeric_projection = nn.Sequential(
            nn.Linear(10, 64),
            nn.SiLU(),
        )
        self.atom_input_norm = nn.LayerNorm(64)

        self.bond_type_embedding = nn.Embedding(
            len(utils.RAW_BOND_TYPE_CATEGORIES) + 1, 64
        )
        self.bond_stereo_embedding = nn.Embedding(
            len(utils.RAW_BOND_STEREO_CATEGORIES) + 1, 64
        )
        self.bond_boolean_projection = nn.Linear(6, 64)
        self.bond_input_norm = nn.LayerNorm(64)

        self.gnn_layers = nn.ModuleList()
        self.gnn_norms = nn.ModuleList()
        for _ in range(3):
            message_mlp = nn.Sequential(
                nn.Linear(64, 128),
                nn.SiLU(),
                nn.Linear(128, 64),
            )
            self.gnn_layers.append(
                GINEConv(message_mlp, edge_dim=64, train_eps=True)
            )
            self.gnn_norms.append(nn.LayerNorm(64))
        self.dropout = nn.Dropout(self.dropout_probability)
        self.jumping_projection = nn.Linear(256, 64)
        self.output_norm = nn.LayerNorm(64)

        self.atom_prediction_heads = nn.ModuleList([
            nn.Linear(64, len(values))
            for values in config["atom_target_values"]
        ])
        self.bond_prediction_heads = nn.ModuleList([
            nn.Linear(128, len(values))
            for values in config["bond_target_values"]
        ])

        self.register_buffer(
            "atom_numeric_mean",
            torch.tensor(config["atom_numeric_mean"], dtype=torch.float),
        )
        self.register_buffer(
            "atom_numeric_std",
            torch.tensor(config["atom_numeric_std"], dtype=torch.float),
        )

    def encode_atom_inputs(self, atom_features, atom_feature_mask=None):
        if atom_feature_mask is None:
            atom_feature_mask = torch.zeros_like(
                atom_features, dtype=torch.bool
            )
        atomic_number = atom_features[:, 0].clone()
        hybridization = atom_features[:, 5].clone()
        chiral_tag = atom_features[:, 7].clone()
        is_aromatic = atom_features[:, 8].clone()
        is_in_ring = atom_features[:, 9].clone()

        atomic_number[atom_feature_mask[:, 0]] = 119
        hybridization[atom_feature_mask[:, 5]] = (
            len(utils.RAW_ATOM_HYBRIDIZATION_CATEGORIES)
        )
        chiral_tag[atom_feature_mask[:, 7]] = (
            len(utils.RAW_ATOM_CHIRAL_CATEGORIES)
        )
        is_aromatic[atom_feature_mask[:, 8]] = 2
        is_in_ring[atom_feature_mask[:, 9]] = 2

        numeric_values = atom_features[
            :, ATOM_NUMERIC_COLUMNS
        ].float()
        numeric_mask = atom_feature_mask[
            :, ATOM_NUMERIC_COLUMNS
        ]
        numeric_values = (
            numeric_values - self.atom_numeric_mean
        ) / self.atom_numeric_std
        # 在浮点标准化空间裁剪，训练、导出和推理共用同一边界。
        numeric_values = numeric_values.clamp(
            min=-NUMERIC_CLIP_SIGMA, max=NUMERIC_CLIP_SIGMA
        )
        numeric_values = numeric_values.masked_fill(numeric_mask, 0.0)
        numeric_input = torch.cat([
            numeric_values,
            numeric_mask.float(),
        ], dim=1)

        atom_embedding = (
            self.atomic_number_embedding(atomic_number)
            + self.hybridization_embedding(hybridization)
            + self.chiral_embedding(chiral_tag)
            + self.aromatic_embedding(is_aromatic)
            + self.ring_embedding(is_in_ring)
            + self.atom_numeric_projection(numeric_input)
        )
        return self.atom_input_norm(atom_embedding)

    def encode_bond_inputs(self, bond_features, bond_feature_mask=None):
        if bond_feature_mask is None:
            bond_feature_mask = torch.zeros_like(
                bond_features, dtype=torch.bool
            )
        bond_type = bond_features[:, 0].clone()
        stereo = bond_features[:, 4].clone()
        bond_type[bond_feature_mask[:, 0]] = (
            len(utils.RAW_BOND_TYPE_CATEGORIES)
        )
        stereo[bond_feature_mask[:, 4]] = (
            len(utils.RAW_BOND_STEREO_CATEGORIES)
        )
        boolean_values = bond_features[:, BOND_BOOLEAN_COLUMNS].float()
        boolean_mask = bond_feature_mask[:, BOND_BOOLEAN_COLUMNS]
        boolean_values = boolean_values.masked_fill(boolean_mask, 0.0)
        boolean_input = torch.cat([
            boolean_values,
            boolean_mask.float(),
        ], dim=1)
        bond_embedding = (
            self.bond_type_embedding(bond_type)
            + self.bond_stereo_embedding(stereo)
            + self.bond_boolean_projection(boolean_input)
        )
        return self.bond_input_norm(bond_embedding)

    def encode(
        self,
        atom_features,
        bond_index,
        bond_features,
        atom_feature_mask=None,
        bond_feature_mask=None,
    ):
        h0 = self.encode_atom_inputs(
            atom_features, atom_feature_mask
        )
        undirected_edge_attr = self.encode_bond_inputs(
            bond_features, bond_feature_mask
        )
        directed_edge_index = torch.cat([
            bond_index,
            bond_index.flip(0),
        ], dim=1)
        directed_edge_attr = torch.cat([
            undirected_edge_attr,
            undirected_edge_attr,
        ], dim=0)

        states = [h0]
        hidden = h0
        for gnn_layer, layer_norm in zip(
            self.gnn_layers, self.gnn_norms
        ):
            update = gnn_layer(
                hidden, directed_edge_index, directed_edge_attr
            )
            update = self.dropout(F.silu(update))
            hidden = layer_norm(hidden + update)
            states.append(hidden)
        context = F.silu(self.jumping_projection(torch.cat(states, dim=1)))
        return self.output_norm(h0 + context)

    def predict_atom_features(self, atom_embeddings):
        return [
            head(atom_embeddings) for head in self.atom_prediction_heads
        ]

    def predict_bond_features(self, atom_embeddings, bond_index):
        if bond_index.size(1) == 0:
            return [
                atom_embeddings.new_empty((0, head.out_features))
                for head in self.bond_prediction_heads
            ]
        first = atom_embeddings[bond_index[0]]
        second = atom_embeddings[bond_index[1]]
        pair_representation = torch.cat([
            first + second,
            torch.abs(first - second),
        ], dim=1)
        return [
            head(pair_representation)
            for head in self.bond_prediction_heads
        ]


def _random_choice(candidates, generator):
    position = int(torch.randint(
        len(candidates), (1,), generator=generator
    ).item())
    return int(candidates[position])


def generate_feature_masks(batch, mask_probability, generator=None):
    atom_shape = tuple(batch.atom_features.shape)
    bond_shape = tuple(batch.bond_features.shape)
    atom_mask = (
        torch.rand(atom_shape, generator=generator)
        < float(mask_probability)
    )
    bond_mask = (
        torch.rand(bond_shape, generator=generator)
        < float(mask_probability)
    )

    ptr = batch.ptr.detach().cpu().tolist()
    fallback_atom_fields = []
    for graph_index in range(len(ptr) - 1):
        start, end = int(ptr[graph_index]), int(ptr[graph_index + 1])
        num_atoms = end - start
        if num_atoms <= 0:
            continue
        if num_atoms == 1:
            atom_mask[start, 0] = False
            fallback_atom_fields.extend(
                (start, column_index) for column_index in range(1, 10)
            )
        elif num_atoms <= 3 and bool(atom_mask[start:end, 0].all()):
            atom_mask[start, 0] = False
            fallback_atom_fields.extend(
                (atom_index, column_index)
                for atom_index in range(start, end)
                for column_index in range(10)
            )
        else:
            fallback_atom_fields.extend(
                (atom_index, column_index)
                for atom_index in range(start, end)
                for column_index in range(10)
            )

    if not bool(atom_mask.any()):
        selected_position = _random_choice(
            list(range(len(fallback_atom_fields))), generator
        )
        atom_index, column_index = fallback_atom_fields[
            selected_position
        ]
        atom_mask[atom_index, column_index] = True

    if batch.bond_index.size(1) > 0 and not bool(bond_mask.any()):
        bond_position = _random_choice(
            list(range(batch.bond_index.size(1))), generator
        )
        column_index = _random_choice(list(range(5)), generator)
        bond_mask[bond_position, column_index] = True

    return (
        atom_mask.to(batch.atom_features.device),
        bond_mask.to(batch.bond_features.device),
    )


def _target_indices(values, vocabulary, device):
    mapping = {int(value): index for index, value in enumerate(vocabulary)}
    mapped = []
    for value in values.detach().cpu().tolist():
        value = int(value)
        if value not in mapping:
            raise ValueError(f"自监督目标类别未配置: {value}")
        mapped.append(mapping[value])
    return torch.tensor(mapped, dtype=torch.long, device=device)


def reconstruction_loss(
    model,
    batch,
    atom_mask,
    bond_mask,
    atom_loss_weight=ATOM_LOSS_WEIGHT,
    bond_loss_weight=BOND_LOSS_WEIGHT,
):
    atom_embeddings = model.encode(
        batch.atom_features,
        batch.bond_index,
        batch.bond_features,
        atom_feature_mask=atom_mask,
        bond_feature_mask=bond_mask,
    )
    atom_logits = model.predict_atom_features(atom_embeddings)
    atom_losses = []
    for column_index, logits in enumerate(atom_logits):
        selected = atom_mask[:, column_index]
        if not bool(selected.any()):
            continue
        targets = _target_indices(
            batch.atom_features[selected, column_index],
            model.config["atom_target_values"][column_index],
            logits.device,
        )
        atom_losses.append(F.cross_entropy(logits[selected], targets))
    if not atom_losses:
        raise RuntimeError("当前批次没有可计算的原子遮蔽目标")
    atom_loss = torch.stack(atom_losses).mean()

    bond_losses = []
    if batch.bond_index.size(1) > 0:
        bond_logits = model.predict_bond_features(
            atom_embeddings, batch.bond_index
        )
        for column_index, logits in enumerate(bond_logits):
            selected = bond_mask[:, column_index]
            if not bool(selected.any()):
                continue
            targets = _target_indices(
                batch.bond_features[selected, column_index],
                model.config["bond_target_values"][column_index],
                logits.device,
            )
            bond_losses.append(F.cross_entropy(logits[selected], targets))
    if bond_losses:
        bond_loss = torch.stack(bond_losses).mean()
    else:
        bond_loss = atom_loss.new_zeros(())
    total_loss = (
        float(atom_loss_weight) * atom_loss
        + float(bond_loss_weight) * bond_loss
    )
    return total_loss, atom_loss.detach(), bond_loss.detach()


def run_epoch(
    model,
    loader,
    device,
    optimizer=None,
    mask_probability=MASK_PROBABILITY,
    validation_seed=None,
    atom_loss_weight=ATOM_LOSS_WEIGHT,
    bond_loss_weight=BOND_LOSS_WEIGHT,
    gradient_clip_norm=GRADIENT_CLIP_NORM,
):
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "atom_loss": 0.0, "bond_loss": 0.0}
    num_batches = 0
    for batch_index, batch in enumerate(loader):
        batch = batch.to(device)
        generator = None
        if validation_seed is not None:
            generator = torch.Generator()
            generator.manual_seed(int(validation_seed) + batch_index)
        atom_mask, bond_mask = generate_feature_masks(
            batch, mask_probability, generator=generator
        )
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            loss, atom_loss, bond_loss = reconstruction_loss(
                model,
                batch,
                atom_mask,
                bond_mask,
                atom_loss_weight=atom_loss_weight,
                bond_loss_weight=bond_loss_weight,
            )
            if training:
                loss.backward()
                if gradient_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        float(gradient_clip_norm),
                    )
                optimizer.step()
        totals["loss"] += float(loss.detach().cpu())
        totals["atom_loss"] += float(atom_loss.cpu())
        totals["bond_loss"] += float(bond_loss.cpu())
        num_batches += 1
    if num_batches == 0:
        raise RuntimeError("原子编码器 DataLoader 没有生成批次")
    return {
        key: value / num_batches for key, value in totals.items()
    }


def _atomic_torch_save(payload, path):
    path = os.path.abspath(os.fspath(path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temp_path = f"{path}.{os.getpid()}-{uuid.uuid4().hex}.tmp"
    try:
        torch.save(payload, temp_path)
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def _torch_load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_atom_encoder_checkpoint(checkpoint_path, device):
    payload = _torch_load_checkpoint(checkpoint_path, device)
    required = {
        "model_config",
        "model_state_dict",
        "training_raw_corpus_sha256",
        "observed_atomic_numbers",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(
            f"原子编码器 checkpoint 缺少字段: {sorted(missing)}"
        )
    observed_atomic_numbers = payload["observed_atomic_numbers"]
    if (
        not isinstance(observed_atomic_numbers, list)
        or len(set(observed_atomic_numbers)) != len(
            observed_atomic_numbers
        )
        or any(
            not isinstance(value, int) or not 1 <= value <= 118
            for value in observed_atomic_numbers
        )
    ):
        raise ValueError("checkpoint 的已见元素集合非法")
    training_sha256 = payload["training_raw_corpus_sha256"]
    if (
        not isinstance(training_sha256, str)
        or len(training_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in training_sha256.lower()
        )
    ):
        raise ValueError("checkpoint 的训练语料 SHA-256 非法")
    model = AtomContextEncoder(payload["model_config"]).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    return model, payload


def export_full_embedding_table(
    model,
    corpus,
    device,
    batch_size,
    num_workers,
):
    dataset = FragmentAtomDataset(corpus)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    embedding_blocks = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            embeddings = model.encode(
                batch.atom_features,
                batch.bond_index,
                batch.bond_features,
            )
            embedding_blocks.append(
                embeddings.detach().cpu().numpy().astype(np.float32)
            )
    if embedding_blocks:
        atom_embeddings = np.concatenate(embedding_blocks, axis=0)
    else:
        atom_embeddings = np.empty(
            (0, ATOM_EMBEDDING_DIM), dtype=np.float32
        )
    if len(atom_embeddings) != len(corpus["canonical_atom_id"]):
        raise RuntimeError("导出的原子嵌入数量与原子粗语料不一致")
    if not np.all(np.isfinite(atom_embeddings)):
        raise RuntimeError("导出的原子嵌入包含 NaN 或 Inf")
    return atom_embeddings


def resolve_device(device_name):
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def _optional_positive_float(value):
    normalized = str(value).strip().lower()
    if normalized in {"none", "null", "off"}:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "必须是正数或 none"
        ) from exc
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("必须是有限正数或 none")
    return parsed


def parse_args():
    parser = argparse.ArgumentParser(
        description="训练三层原子环境编码器并导出标准64维原子词表"
    )
    parser.add_argument("--raw-corpus", default=DEFAULT_RAW_CORPUS)
    parser.add_argument("--raw-metadata", default=DEFAULT_RAW_METADATA)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--embedding-table", default=DEFAULT_EMBEDDING_TABLE
    )
    parser.add_argument(
        "--embedding-metadata", default=DEFAULT_EMBEDDING_METADATA
    )
    parser.add_argument(
        "--training-history", default=DEFAULT_TRAINING_HISTORY
    )
    parser.add_argument(
        "--batch-size", type=int, default=TRAIN_BATCH_SIZE
    )
    parser.add_argument(
        "--encode-batch-size", type=int, default=ENCODE_BATCH_SIZE
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument(
        "--patience", type=int, default=EARLY_STOPPING_PATIENCE
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=EARLY_STOPPING_MIN_DELTA,
    )
    parser.add_argument(
        "--learning-rate", type=float, default=LEARNING_RATE
    )
    parser.add_argument(
        "--weight-decay", type=float, default=WEIGHT_DECAY
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=VALIDATION_FRACTION,
    )
    parser.add_argument(
        "--mask-probability", type=float, default=MASK_PROBABILITY
    )
    parser.add_argument("--dropout", type=float, default=DROPOUT)
    parser.add_argument(
        "--atom-loss-weight", type=float, default=ATOM_LOSS_WEIGHT
    )
    parser.add_argument(
        "--bond-loss-weight", type=float, default=BOND_LOSS_WEIGHT
    )
    parser.add_argument(
        "--gradient-clip-norm",
        type=_optional_positive_float,
        default=GRADIENT_CLIP_NORM,
        help="正数启用梯度裁剪；none、null 或 off 表示禁用",
    )
    parser.add_argument(
        "--num-workers", type=int, default=NUM_WORKERS
    )
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def validate_training_configuration(args):
    positive_integer_fields = {
        "batch_size": "batch-size",
        "encode_batch_size": "encode-batch-size",
        "epochs": "epochs",
        "patience": "patience",
    }
    for field_name, display_name in positive_integer_fields.items():
        value = getattr(args, field_name)
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value <= 0
        ):
            raise ValueError(f"{display_name} 必须是正整数")

    if (
        not isinstance(args.num_workers, int)
        or isinstance(args.num_workers, bool)
        or args.num_workers < 0
    ):
        raise ValueError("num-workers 必须是非负整数")
    if (
        not isinstance(args.seed, int)
        or isinstance(args.seed, bool)
        or args.seed < 0
    ):
        raise ValueError("seed 必须是非负整数")

    finite_ranges = (
        ("learning-rate", args.learning_rate, 0.0, None, False),
        ("weight-decay", args.weight_decay, 0.0, None, True),
        (
            "early-stopping-min-delta",
            args.early_stopping_min_delta,
            0.0,
            None,
            True,
        ),
        (
            "validation-fraction",
            args.validation_fraction,
            0.0,
            1.0,
            False,
        ),
        (
            "mask-probability",
            args.mask_probability,
            0.0,
            1.0,
            False,
        ),
        ("dropout", args.dropout, 0.0, 1.0, True),
        (
            "atom-loss-weight",
            args.atom_loss_weight,
            0.0,
            None,
            False,
        ),
        (
            "bond-loss-weight",
            args.bond_loss_weight,
            0.0,
            None,
            True,
        ),
    )
    for name, value, lower, upper, include_lower in finite_ranges:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"{name} 必须是数值")
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"{name} 必须是有限数值")
        lower_valid = value >= lower if include_lower else value > lower
        upper_valid = upper is None or value < upper
        if not lower_valid or not upper_valid:
            lower_symbol = "[" if include_lower else "("
            upper_text = "∞)" if upper is None else f"{upper})"
            raise ValueError(
                f"{name} 必须位于 {lower_symbol}{lower}, {upper_text}"
            )

    if args.gradient_clip_norm is not None:
        if (
            not isinstance(args.gradient_clip_norm, (int, float))
            or isinstance(args.gradient_clip_norm, bool)
            or not math.isfinite(float(args.gradient_clip_norm))
            or float(args.gradient_clip_norm) <= 0.0
        ):
            raise ValueError("gradient-clip-norm 必须是有限正数或 None")
    if not isinstance(args.device, str) or not args.device.strip():
        raise ValueError("device 必须是非空字符串")


def main():
    args = parse_args()
    validate_training_configuration(args)

    set_random_seed(args.seed)
    device = resolve_device(args.device)
    corpus, raw_metadata = utils.load_raw_fragment_atom_corpus(
        args.raw_corpus, args.raw_metadata
    )

    dataset = FragmentAtomDataset(corpus)
    num_fragments = len(dataset)
    if num_fragments == 1:
        train_indices = [0]
        validation_indices = [0]
    else:
        generator = torch.Generator().manual_seed(args.seed)
        permutation = torch.randperm(
            num_fragments, generator=generator
        ).tolist()
        validation_count = max(
            1,
            int(round(num_fragments * args.validation_fraction)),
        )
        validation_count = min(validation_count, num_fragments - 1)
        validation_indices = permutation[:validation_count]
        train_indices = permutation[validation_count:]

    model_config = build_model_config(
        corpus,
        normalization_fragment_indices=train_indices,
        dropout=args.dropout,
    )
    training_atom_features = _fragment_atom_rows(
        corpus, train_indices
    )
    observed_atomic_numbers = sorted({
        int(value) for value in training_atom_features[:, 0]
    })

    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    validation_loader = DataLoader(
        Subset(dataset, validation_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = AtomContextEncoder(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    history = []
    best_validation_loss = math.inf
    epochs_without_improvement = 0
    best_epoch = 0
    checkpoint_saved_this_run = False
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            optimizer=optimizer,
            mask_probability=args.mask_probability,
            atom_loss_weight=args.atom_loss_weight,
            bond_loss_weight=args.bond_loss_weight,
            gradient_clip_norm=args.gradient_clip_norm,
        )
        validation_metrics = run_epoch(
            model,
            validation_loader,
            device,
            optimizer=None,
            mask_probability=args.mask_probability,
            validation_seed=args.seed + 100000,
            atom_loss_weight=args.atom_loss_weight,
            bond_loss_weight=args.bond_loss_weight,
            gradient_clip_norm=args.gradient_clip_norm,
        )
        epoch_record = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": validation_metrics,
        }
        history.append(epoch_record)
        print(
            f"Epoch {epoch:03d} | "
            f"train={train_metrics['loss']:.6f} | "
            f"val={validation_metrics['loss']:.6f}"
        )

        if (
            validation_metrics["loss"]
            < best_validation_loss - args.early_stopping_min_delta
        ):
            best_validation_loss = validation_metrics["loss"]
            best_epoch = epoch
            epochs_without_improvement = 0
            checkpoint_payload = {
                "model_config": model_config,
                "model_state_dict": {
                    key: value.detach().cpu()
                    for key, value in model.state_dict().items()
                },
                "training_raw_corpus_sha256": raw_metadata[
                    "corpus_sha256"
                ],
                "observed_atomic_numbers": observed_atomic_numbers,
                "best_epoch": best_epoch,
                "best_validation_loss": best_validation_loss,
                "training_parameters": {
                    "seed": args.seed,
                    "validation_fraction": args.validation_fraction,
                    "batch_size": args.batch_size,
                    "encode_batch_size": args.encode_batch_size,
                    "max_epochs": args.epochs,
                    "patience": args.patience,
                    "early_stopping_min_delta": (
                        args.early_stopping_min_delta
                    ),
                    "learning_rate": args.learning_rate,
                    "weight_decay": args.weight_decay,
                    "mask_probability": args.mask_probability,
                    "dropout": args.dropout,
                    "atom_loss_weight": args.atom_loss_weight,
                    "bond_loss_weight": args.bond_loss_weight,
                    "gradient_clip_norm": args.gradient_clip_norm,
                    "num_workers": args.num_workers,
                    "requested_device": args.device,
                    "resolved_device": str(device),
                },
            }
            _atomic_torch_save(checkpoint_payload, args.checkpoint)
            checkpoint_saved_this_run = True
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(
                    f"验证损失连续 {args.patience} 轮未改善，提前停止。"
                )
                break

    if (
        not checkpoint_saved_this_run
        or not os.path.isfile(args.checkpoint)
    ):
        raise RuntimeError("训练结束但没有生成最佳 checkpoint")
    best_model, checkpoint_payload = load_atom_encoder_checkpoint(
        args.checkpoint, device
    )
    checkpoint_sha = file_sha256(args.checkpoint)
    atom_embeddings = export_full_embedding_table(
        best_model,
        corpus,
        device,
        batch_size=args.encode_batch_size,
        num_workers=args.num_workers,
    )
    embedding_metadata = save_atom_embedding_table(
        corpus,
        atom_embeddings,
        args.embedding_table,
        args.embedding_metadata,
        checkpoint_sha256=checkpoint_sha,
        training_raw_corpus_sha256=checkpoint_payload[
            "training_raw_corpus_sha256"
        ],
        current_raw_corpus_sha256=raw_metadata["corpus_sha256"],
        model_config=checkpoint_payload["model_config"],
        observed_atomic_numbers=checkpoint_payload[
            "observed_atomic_numbers"
        ],
    )
    utils.atomic_write_json(
        args.training_history,
        {
            "best_epoch": best_epoch,
            "best_validation_loss": best_validation_loss,
            "num_train_smiles": len(train_indices),
            "num_validation_smiles": len(validation_indices),
            "training_parameters": checkpoint_payload[
                "training_parameters"
            ],
            "checkpoint_sha256": checkpoint_sha,
            "embedding_table_sha256": embedding_metadata["table_sha256"],
            "embedding_json_mirror_sha256": embedding_metadata[
                "json_mirror"
            ]["sha256"],
            "history": history,
        },
    )
    print(f"最佳 checkpoint: {args.checkpoint}")
    print(f"标准64维原子词表: {args.embedding_table}")
    print(
        "完整可读 JSON 镜像: "
        f"{utils.derive_embedding_json_mirror_path(args.embedding_table)}"
    )
    print(f"标准词表元数据: {args.embedding_metadata}")


if __name__ == "__main__":
    main()
