"""PhiSGATv2 的 PyG 数据发现、校验与片段嵌入装配。"""

import gc
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

if __package__:
    from . import PhiSGATv2_config as config
else:  # 支持直接运行 GATv2 目录中的脚本
    import PhiSGATv2_config as config


INTEGER_DTYPES = {
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
}


def discover_pyg_files(data_dir):
    """递归返回目录中稳定排序的全部 .pt 文件。"""
    root = Path(data_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"PyG 数据目录不存在: {root}")
    files = [path.resolve() for path in root.rglob("*.pt") if path.is_file()]
    return sorted((str(path) for path in files), key=lambda value: value.casefold())


def load_pyg_file(path):
    """显式加载可信的本地 PyG Data 文件。"""
    path = os.path.abspath(os.fspath(path))
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise RuntimeError(f"无法加载 PyG 文件 {path}: {exc}") from exc


def _require_node_vector(data, field_name, num_nodes, path):
    value = getattr(data, field_name, None)
    if not torch.is_tensor(value):
        raise ValueError(f"{path}: 缺少张量字段 {field_name}")
    if value.numel() != num_nodes:
        raise ValueError(
            f"{path}: {field_name} 必须包含 {num_nodes} 个节点值，"
            f"实际形状为 {tuple(value.shape)}"
        )
    return value.reshape(num_nodes)


def read_activity_y(data, path, required, enforce_train_range=False):
    """读取单个连续活性值，并按需校验训练区间。"""
    value = getattr(data, "y", None)
    if value is None:
        if required:
            raise ValueError(f"{path}: 训练数据缺少 y")
        return None

    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError(
                f"{path}: y 必须只有一个数值，实际形状为 {tuple(value.shape)}"
            )
        numeric_value = float(value.detach().cpu().reshape(-1)[0])
    else:
        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path}: y 不是数值: {value!r}") from exc

    if not np.isfinite(numeric_value):
        raise ValueError(f"{path}: y 不是有限数值")
    if enforce_train_range and not (
        config.TRAIN_ACTIVITY_MIN
        <= numeric_value
        <= config.TRAIN_ACTIVITY_MAX
    ):
        raise ValueError(
            f"{path}: 训练标签 y={numeric_value} 超出闭区间 "
            f"[{config.TRAIN_ACTIVITY_MIN}, {config.TRAIN_ACTIVITY_MAX}]"
        )
    return numeric_value


def validate_graph_data(
    data,
    path,
    require_y=False,
    enforce_train_y_range=False,
):
    """验证新二维接口，并返回节点 SMILES 与可选连续标签 y。"""
    path = os.path.abspath(os.fspath(path))
    x = getattr(data, "x", None)
    if not torch.is_tensor(x):
        raise ValueError(f"{path}: 缺少张量字段 x")
    if x.dim() != 2 or x.size(1) != config.INPUT_DIM_NUMERIC:
        raise ValueError(
            f"{path}: x 必须为 [N, {config.INPUT_DIM_NUMERIC}]，"
            f"实际为 {tuple(x.shape)}"
        )
    if x.size(0) < 1:
        raise ValueError(f"{path}: 图至少需要一个节点")
    if not torch.isfinite(x).all():
        raise ValueError(f"{path}: x 包含 NaN 或 Inf")
    num_nodes = int(x.size(0))

    edge_index = getattr(data, "edge_index", None)
    if not torch.is_tensor(edge_index):
        raise ValueError(f"{path}: 缺少张量字段 edge_index")
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError(
            f"{path}: edge_index 必须为 [2, E]，实际为 {tuple(edge_index.shape)}"
        )
    if edge_index.dtype not in INTEGER_DTYPES:
        raise ValueError(f"{path}: edge_index 必须是整数张量")
    if edge_index.numel() > 0:
        minimum = int(edge_index.min())
        maximum = int(edge_index.max())
        if minimum < 0 or maximum >= num_nodes:
            raise ValueError(
                f"{path}: edge_index 节点编号越界，范围为 [{minimum}, {maximum}]，"
                f"节点数为 {num_nodes}"
            )

    fragment_smiles = getattr(data, "fragment_smiles", None)
    if not isinstance(fragment_smiles, (list, tuple)):
        raise ValueError(f"{path}: fragment_smiles 必须是字符串列表")
    if len(fragment_smiles) != num_nodes:
        raise ValueError(
            f"{path}: fragment_smiles 长度 {len(fragment_smiles)} "
            f"与节点数 {num_nodes} 不一致"
        )
    normalized_smiles = []
    for node_index, smiles in enumerate(fragment_smiles):
        if not isinstance(smiles, str) or not smiles.strip():
            raise ValueError(
                f"{path}: 第 {node_index} 个 fragment_smiles 不是非空字符串"
            )
        normalized_smiles.append(smiles.strip())

    metal_mask = _require_node_vector(data, "metal_mask", num_nodes, path)
    if metal_mask.dtype != torch.bool:
        raise ValueError(f"{path}: metal_mask 必须是 bool 张量")

    for field_name in ("hac", "ring_count"):
        values = _require_node_vector(data, field_name, num_nodes, path)
        if values.dtype not in INTEGER_DTYPES:
            raise ValueError(f"{path}: {field_name} 必须是整数张量")
        if bool((values < 0).any()):
            raise ValueError(f"{path}: {field_name} 不能包含负数")

    activity = read_activity_y(
        data,
        path,
        required=require_y,
        enforce_train_range=enforce_train_y_range,
    )
    return normalized_smiles, activity


def select_labeled_pyg_files(data_dir):
    """返回有效有标签文件，并单独列出缺失 y 的文件。"""
    valid_files = []
    missing_label_files = []
    for path in discover_pyg_files(data_dir):
        data = load_pyg_file(path)
        _, activity = validate_graph_data(
            data,
            path,
            require_y=False,
            enforce_train_y_range=True,
        )
        if activity is None:
            missing_label_files.append(path)
        else:
            # 上一步已经完成训练区间检查；这里保持清晰的训练文件清单。
            valid_files.append(path)
        del data
    return valid_files, missing_label_files


def validate_prediction_files(paths):
    """逐文件校验预测数据，返回有效路径与不中断批次的错误记录。"""
    valid_files = []
    errors = []
    for path in paths:
        data = None
        try:
            data = load_pyg_file(path)
            validate_graph_data(data, path, require_y=False)
        except Exception as exc:
            errors.append({"file_path": os.path.abspath(path), "error": str(exc)})
        else:
            valid_files.append(os.path.abspath(path))
        finally:
            if data is not None:
                del data
    return valid_files, errors


def build_fragment_embedding_lookup(data_files, cpu_threads=None):
    """在主进程批量解析所有唯一片段，返回内存中的 128 维映射。"""
    requested_threads = (
        config.RESOLVER_CPU_THREADS
        if cpu_threads is None
        else cpu_threads
    )
    effective_threads = config.clamp_cpu_count(
        requested_threads,
        config.MAX_RESOLVER_CPU_THREADS,
        "RESOLVER_CPU_THREADS",
    )
    print(
        "Resolver CPU线程: "
        f"请求={requested_threads}, "
        f"硬上限={config.MAX_RESOLVER_CPU_THREADS}, "
        f"可用={config.allocated_cpu_count()}, "
        f"实际={effective_threads}"
    )

    unique_smiles = set()
    for path in data_files:
        data = load_pyg_file(path)
        fragment_smiles, _ = validate_graph_data(
            data,
            path,
            require_y=False,
        )
        unique_smiles.update(fragment_smiles)
        del data
    ordered_smiles = sorted(unique_smiles)
    if not ordered_smiles:
        raise RuntimeError("没有可供片段编码器解析的 fragment_smiles")

    if config.PROJECT_ROOT not in sys.path:
        sys.path.insert(0, config.PROJECT_ROOT)
    from PhiSSeparator.PhiSSeparator_fragment_encoder_resolver import (  # noqa: E402
        FragmentEmbeddingResolver,
    )

    previous_torch_threads = torch.get_num_threads()
    resolver = None
    try:
        torch.set_num_threads(effective_threads)
        resolver = FragmentEmbeddingResolver(
            device_spec=config.RESOLVER_DEVICE,
        )
        vectors = resolver.resolve_many(ordered_smiles)
        canonical_smiles = list(
            resolver.last_resolution["canonical_smiles"]
        )
    finally:
        if resolver is not None:
            del resolver
        torch.set_num_threads(previous_torch_threads)
        gc.collect()

    vectors = np.asarray(vectors, dtype=np.float32)
    expected_shape = (len(ordered_smiles), config.EMBEDDING_DIM)
    if tuple(vectors.shape) != expected_shape:
        raise ValueError(
            f"Resolver 输出形状应为 {expected_shape}，实际为 {tuple(vectors.shape)}"
        )
    if not np.isfinite(vectors).all():
        raise ValueError("Resolver 输出包含 NaN 或 Inf")

    lookup = {}
    for raw_smiles, canonical_value, vector in zip(
        ordered_smiles,
        canonical_smiles,
        vectors,
    ):
        vector_copy = np.array(vector, dtype=np.float32, copy=True)
        lookup[raw_smiles] = vector_copy
        lookup.setdefault(str(canonical_value), vector_copy)
    return lookup


class MoleculeDataset(Dataset):
    """加载已经切割的二维分子图，并在内存中附加片段嵌入。"""

    def __init__(self, data_files, embedding_lookup, require_y=True):
        self.data_files = [os.path.abspath(path) for path in data_files]
        self.embedding_lookup = embedding_lookup
        self.require_y = bool(require_y)
        if not self.data_files:
            raise ValueError("MoleculeDataset 至少需要一个 PyG 文件")
        if not isinstance(self.embedding_lookup, dict):
            raise TypeError("embedding_lookup 必须是字典")

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, index):
        path = self.data_files[index]
        data = load_pyg_file(path)
        fragment_smiles, activity = validate_graph_data(
            data,
            path,
            require_y=self.require_y,
            enforce_train_y_range=self.require_y,
        )

        missing = [
            smiles for smiles in fragment_smiles
            if smiles not in self.embedding_lookup
        ]
        if missing:
            raise KeyError(
                f"{path}: 片段嵌入映射缺少 {sorted(set(missing))}"
            )
        vectors = np.stack([
            self.embedding_lookup[smiles]
            for smiles in fragment_smiles
        ]).astype(np.float32, copy=False)
        expected_shape = (len(fragment_smiles), config.EMBEDDING_DIM)
        if tuple(vectors.shape) != expected_shape:
            raise ValueError(
                f"{path}: frag_embeds 应为 {expected_shape}，"
                f"实际为 {tuple(vectors.shape)}"
            )
        if not np.isfinite(vectors).all():
            raise ValueError(f"{path}: frag_embeds 包含 NaN 或 Inf")

        data.x = data.x.float()
        data.edge_index = data.edge_index.long()
        data.frag_embeds = torch.from_numpy(np.array(vectors, copy=True))
        data.pyg_file_path = path
        if activity is not None:
            data.y = torch.tensor([activity], dtype=torch.float)
        return data
