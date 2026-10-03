import os
import glob
import hashlib
import json
import math
from collections import Counter
import numpy as np
import torch
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from torch.utils.data import Dataset as TorchDataset
import concurrent.futures

# Local Imports
import PhiSSE3TD_config as config

def is_main_process():
    return int(os.environ.get('RANK', 0)) == 0


class FragmentEmbeddingLookup:
    """加载并严格校验标准 SMILES -> 128 维片段词嵌入表。"""

    TABLE_FIELDS = ('smiles', 'fragment_embeddings')

    def __init__(
        self,
        npz_path=config.FRAGMENT_EMBEDDING_TABLE_PATH,
        metadata_path=config.FRAGMENT_EMBEDDING_METADATA_PATH,
    ):
        self.npz_path = os.path.abspath(os.fspath(npz_path))
        self.metadata_path = os.path.abspath(os.fspath(metadata_path))
        arrays, metadata = self._load_and_validate()

        smiles = arrays['smiles'].tolist()
        self._smiles = tuple(str(smiles_value) for smiles_value in smiles)
        self.smiles_to_index = {
            smiles_value: row_index
            for row_index, smiles_value in enumerate(self._smiles)
        }
        # 拷贝为独立、连续的 CPU Tensor，供预加载线程安全地只读索引。
        self.embedding_tensor = torch.from_numpy(
            np.ascontiguousarray(arrays['fragment_embeddings']).copy()
        )
        self._profile = {
            'embedding_dim': int(metadata['embedding_dim']),
            'num_smiles': int(metadata['num_smiles']),
            'table_sha256': metadata['table_sha256'],
            'array_sha256': dict(metadata['array_sha256']),
            'npz_path': self.npz_path,
            'metadata_path': self.metadata_path,
        }

    @staticmethod
    def _array_sha256(values, is_string=False):
        digest = hashlib.sha256()
        values = np.asarray(values)
        if is_string:
            for value in values.astype(str).tolist():
                encoded = value.encode('utf-8')
                digest.update(len(encoded).to_bytes(8, 'little'))
                digest.update(encoded)
        else:
            contiguous = np.ascontiguousarray(values)
            digest.update(contiguous.dtype.str.encode('ascii'))
            digest.update(contiguous.tobytes())
        return digest.hexdigest()

    @classmethod
    def _table_sha256(cls, arrays):
        digest = hashlib.sha256()
        for key in cls.TABLE_FIELDS:
            values = np.asarray(arrays[key])
            digest.update(key.encode('utf-8'))
            digest.update(json.dumps(list(values.shape)).encode('ascii'))
            digest.update(
                cls._array_sha256(
                    values, is_string=(key == 'smiles')
                ).encode('ascii')
            )
        return digest.hexdigest()

    def _load_and_validate(self):
        if not os.path.isfile(self.npz_path):
            raise FileNotFoundError(f"未找到片段词嵌入表: {self.npz_path}")
        if not os.path.isfile(self.metadata_path):
            raise FileNotFoundError(
                f"未找到片段词嵌入表 metadata: {self.metadata_path}"
            )

        with open(self.metadata_path, 'r', encoding='utf-8') as handle:
            metadata = json.load(handle)
        if not isinstance(metadata, dict):
            raise ValueError("片段词嵌入表 metadata 必须是 JSON 对象")

        with np.load(self.npz_path, allow_pickle=False) as loaded:
            missing = set(self.TABLE_FIELDS) - set(loaded.files)
            if missing:
                raise ValueError(
                    f"片段词嵌入表缺少字段: {sorted(missing)}"
                )
            arrays = {
                key: np.array(loaded[key], copy=True)
                for key in self.TABLE_FIELDS
            }
        arrays['smiles'] = arrays['smiles'].astype(str)

        if metadata.get('embedding_dim') != config.EMBEDDING_DIM_IN:
            raise ValueError(
                "片段词嵌入表维度与模型配置不一致: "
                f"{metadata.get('embedding_dim')} != {config.EMBEDDING_DIM_IN}"
            )

        smiles = arrays['smiles']
        embeddings = arrays['fragment_embeddings']
        if embeddings.dtype != np.float32:
            raise TypeError(
                f"fragment_embeddings 必须是 float32，实际为 {embeddings.dtype}"
            )
        expected_shape = (len(smiles), config.EMBEDDING_DIM_IN)
        if embeddings.shape != expected_shape:
            raise ValueError(
                f"fragment_embeddings 形状应为 {expected_shape}，"
                f"实际为 {embeddings.shape}"
            )
        if len(set(smiles.tolist())) != len(smiles):
            raise ValueError("片段词嵌入表包含重复 SMILES")
        if not np.all(np.isfinite(embeddings)):
            raise ValueError("片段词嵌入表包含 NaN 或 Inf")
        if len(embeddings):
            norms = np.linalg.norm(embeddings, axis=1)
            if not np.allclose(norms, 1.0, atol=1e-4, rtol=0.0):
                raise ValueError("片段词嵌入未保持单位归一化")

        expected_shapes = {
            key: list(np.asarray(value).shape)
            for key, value in arrays.items()
        }
        expected_dtypes = {
            key: np.asarray(value).dtype.str
            for key, value in arrays.items()
        }
        expected_array_sha256 = {
            key: self._array_sha256(
                value, is_string=(key == 'smiles')
            )
            for key, value in arrays.items()
        }
        if metadata.get('array_shapes') != expected_shapes:
            raise ValueError("片段词嵌入表数组形状与 metadata 不一致")
        if metadata.get('array_dtypes') != expected_dtypes:
            raise ValueError("片段词嵌入表数组类型与 metadata 不一致")
        if metadata.get('array_sha256') != expected_array_sha256:
            raise ValueError("片段词嵌入表逐数组校验值不一致")
        if metadata.get('table_sha256') != self._table_sha256(arrays):
            raise ValueError("片段词嵌入表整体校验值不一致")
        if metadata.get('num_smiles') != len(smiles):
            raise ValueError("片段词嵌入表 SMILES 数量与 metadata 不一致")
        return arrays, metadata

    @property
    def profile(self):
        # 返回深拷贝，防止训练画像构建过程意外修改共享解析器状态。
        return json.loads(json.dumps(self._profile))

    def profile_for_prefix(self, num_smiles):
        """计算指定词数前缀的稳定画像，供追加式兼容校验使用。"""
        if isinstance(num_smiles, bool) or not isinstance(num_smiles, int):
            raise TypeError("词嵌入表前缀词数必须是整数")
        current_num_smiles = len(self._smiles)
        if not 1 <= num_smiles <= current_num_smiles:
            raise ValueError(
                "词嵌入表前缀词数必须位于 "
                f"[1, {current_num_smiles}]，实际为 {num_smiles}"
            )

        arrays = {
            'smiles': np.asarray(self._smiles[:num_smiles], dtype=np.str_),
            'fragment_embeddings': (
                self.embedding_tensor[:num_smiles].numpy()
            ),
        }
        return {
            'embedding_dim': self._profile['embedding_dim'],
            'num_smiles': int(num_smiles),
            'table_sha256': self._table_sha256(arrays),
            'array_sha256': {
                key: self._array_sha256(
                    value, is_string=(key == 'smiles')
                )
                for key, value in arrays.items()
            },
        }

    def resolve(self, data):
        if not hasattr(data, 'fragment_smiles'):
            raise ValueError("缺少必需字段 'fragment_smiles'")
        if not hasattr(data, 'x') or not isinstance(data.x, torch.Tensor):
            raise ValueError("解析 frag_embeds 前需要 Tensor 字段 'x'")
        if not hasattr(data, 'node_type') or not isinstance(
            data.node_type, torch.Tensor
        ):
            raise ValueError("解析 frag_embeds 前需要 Tensor 字段 'node_type'")

        num_nodes = int(data.x.shape[0])
        if tuple(data.node_type.shape) != (
            num_nodes, config.TYPE_ENCODING_DIM_IN
        ):
            raise ValueError(
                "node_type 形状与节点数不一致，无法解析 frag_embeds"
            )
        raw_smiles = data.fragment_smiles
        if isinstance(raw_smiles, (str, bytes)):
            raise TypeError("fragment_smiles 必须是逐节点字符串序列")
        try:
            smiles_list = [str(value) for value in raw_smiles]
        except TypeError as exc:
            raise TypeError("fragment_smiles 必须是逐节点字符串序列") from exc
        if len(smiles_list) != num_nodes:
            raise ValueError(
                f"fragment_smiles 数量应为 {num_nodes}，实际为 {len(smiles_list)}"
            )

        global_mask = data.node_type[:, 2].bool()
        global_indices = torch.nonzero(
            global_mask, as_tuple=False
        ).flatten().tolist()
        if len(global_indices) != 1:
            raise ValueError(
                f"每图必须恰好一个 Global 节点，实际为 {len(global_indices)}"
            )
        global_idx = global_indices[0]
        if smiles_list[global_idx] != '<GLOBAL>':
            raise ValueError(
                "Global 节点的 fragment_smiles 必须为 '<GLOBAL>'，"
                f"实际为 {smiles_list[global_idx]!r}"
            )

        node_indices = []
        table_indices = []
        missing_smiles = set()
        for node_idx, fragment_smiles in enumerate(smiles_list):
            if node_idx == global_idx:
                continue
            table_idx = self.smiles_to_index.get(fragment_smiles)
            if table_idx is None:
                missing_smiles.add(fragment_smiles)
                continue
            node_indices.append(node_idx)
            table_indices.append(table_idx)

        if missing_smiles:
            sample_id = getattr(data, 'sample_id', '<unknown>')
            pdb_id = getattr(data, 'pdb_id', '<unknown>')
            missing_preview = sorted(missing_smiles)[:20]
            raise ValueError(
                "存在无法在标准片段词嵌入表中命中的非 Global SMILES: "
                f"sample_id={sample_id}, pdb_id={pdb_id}, "
                f"缺失唯一值={len(missing_smiles)}, 前20项={missing_preview}"
            )

        frag_embeds = torch.zeros(
            (num_nodes, config.EMBEDDING_DIM_IN),
            dtype=torch.float32,
            device=data.x.device,
        )
        if node_indices:
            table_rows = self.embedding_tensor.index_select(
                0, torch.tensor(table_indices, dtype=torch.long)
            ).to(data.x.device)
            node_index_tensor = torch.tensor(
                node_indices, dtype=torch.long, device=data.x.device
            )
            frag_embeds[node_index_tensor] = table_rows
        data.frag_embeds = frag_embeds
        return data


# =============================================================================
# 1. 核心数据变换 (Transform)
# =============================================================================
class LazyGraphDataset(TorchDataset):
    """
    针对大规模 3D 分子图数据的懒加载数据集。
    只保存文件路径列表，在 __getitem__ 时才实时读取和预处理数据。
    """
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.embedding_table_profile = (
            transform.embedding_table_profile
            if transform is not None
            and hasattr(transform, 'embedding_table_profile')
            else None
        )
        
        # 扫描目录下所有的 .pt 文件
        search_pattern = os.path.join(data_dir, '**', '*.pt')
        self.file_list = sorted(glob.glob(search_pattern, recursive=True))
        
        if not self.file_list:
            raise FileNotFoundError(f"在 {data_dir} 中未找到任何 .pt 文件。")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        f_path = self.file_list[idx]
        try:
            data = torch.load(f_path, weights_only=False)
            if self.transform is not None:
                data = self.transform(data)
            return data
        except Exception as exc:
            raise RuntimeError(f"PyGData 校验或读取失败: {f_path}: {exc}") from exc

class PreprocessNodeFeatures(T.BaseTransform):
    """
    新 PyGData 契约适配器。

    磁盘字段保持解耦；这里只创建掩码、物理框架槽掩码，把绝对框架
    转为相对框架，并把 LL/LP 边移为监督标签。训练消息图只留下 PP/GL。
    """
    _FIELD_SHAPES = {
        'frag_embeds': (config.EMBEDDING_DIM_IN,),
        'x': (config.CHEM_PROPS_DIM_IN,),
        'node_type': (config.TYPE_ENCODING_DIM_IN,),
        'pos': (3,),
        'ref_coords': (3, 3),
    }

    def __init__(self, embedding_lookup):
        self.embedding_lookup = embedding_lookup

    @property
    def embedding_table_profile(self):
        return self.embedding_lookup.profile

    @staticmethod
    def _require_tensor(data, name):
        if not hasattr(data, name):
            raise ValueError(f"缺少必需字段 '{name}'")
        value = getattr(data, name)
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"字段 '{name}' 必须是 Tensor，实际为 {type(value).__name__}")
        return value

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

    def _validate_contract(self, data):
        tensors = {name: self._require_tensor(data, name) for name in self._FIELD_SHAPES}
        hac = self._require_tensor(data, 'hac')
        ring_count = self._require_tensor(data, 'ring_count')
        edge_index = self._require_tensor(data, 'edge_index')
        edge_attr = self._require_tensor(data, 'edge_attr')
        connection_edge_index = self._require_tensor(data, 'connection_edge_index')
        connection_atom_ids = self._require_tensor(data, 'connection_atom_ids')
        connection_distance = self._require_tensor(data, 'connection_distance')

        num_nodes = tensors['x'].shape[0]
        if num_nodes == 0:
            raise ValueError("图中没有节点")
        for name, trailing_shape in self._FIELD_SHAPES.items():
            expected = (num_nodes,) + trailing_shape
            if tuple(tensors[name].shape) != expected:
                raise ValueError(f"{name} 形状应为 {expected}，实际为 {tuple(tensors[name].shape)}")
            if tensors[name].dtype != torch.float32:
                raise TypeError(f"{name} 必须是 float32，实际为 {tensors[name].dtype}")
            if not torch.isfinite(tensors[name]).all():
                raise ValueError(f"{name} 含 NaN/Inf")

        for name, value in (('hac', hac), ('ring_count', ring_count)):
            if tuple(value.shape) != (num_nodes,):
                raise ValueError(f"{name} 形状应为 ({num_nodes},)，实际为 {tuple(value.shape)}")
            if value.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                raise TypeError(f"{name} 必须是整数 Tensor，实际为 {value.dtype}")
            if (value < 0).any():
                raise ValueError(f"{name} 不允许负值")

        if edge_index.dtype != torch.long or edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index 必须是 int64 [2,E]")
        if edge_index.numel() and ((edge_index < 0).any() or (edge_index >= num_nodes).any()):
            raise ValueError("edge_index 含越界节点索引")
        if tuple(edge_attr.shape) != (edge_index.shape[1], config.EDGE_ATTR_DIM):
            raise ValueError(
                f"edge_attr 形状应为 ({edge_index.shape[1]},{config.EDGE_ATTR_DIM})，"
                f"实际为 {tuple(edge_attr.shape)}"
            )
        if edge_attr.dtype != torch.float32:
            raise TypeError(f"edge_attr 必须是 float32，实际为 {edge_attr.dtype}")
        if not torch.isfinite(edge_attr).all():
            raise ValueError("edge_attr 含 NaN/Inf")

        node_type = tensors['node_type'].float()
        if not torch.all((node_type == 0) | (node_type == 1)):
            raise ValueError("node_type 必须是严格 0/1 one-hot")
        if not torch.all(node_type.sum(dim=-1) == 1):
            raise ValueError("node_type 每行必须且只能有一个类别")

        edge_attr_f = edge_attr.float()
        if edge_attr_f.numel():
            if not torch.all((edge_attr_f == 0) | (edge_attr_f == 1)):
                raise ValueError("edge_attr 必须是严格 0/1 one-hot")
            if not torch.all(edge_attr_f.sum(dim=-1) == 1):
                raise ValueError("edge_attr 每行必须且只能有一个类别")

        is_ligand = node_type[:, 0].bool()
        is_protein = node_type[:, 1].bool()
        is_global = node_type[:, 2].bool()
        if not is_ligand.any():
            raise ValueError("训练复合物图必须至少含一个 ligand 节点")
        if not is_protein.any():
            raise ValueError("训练复合物图必须至少含一个 protein 节点")
        if (hac[is_ligand] < 1).any():
            raise ValueError("ligand 节点的 hac 必须至少为 1")
        global_indices = torch.nonzero(is_global, as_tuple=False).flatten()
        if global_indices.numel() != 1:
            raise ValueError(f"每图必须恰好一个 Global 节点，实际为 {global_indices.numel()}")
        global_idx = int(global_indices.item())
        if int(hac[global_idx]) != 0 or int(ring_count[global_idx]) != 0:
            raise ValueError("Global 节点的 hac 和 ring_count 必须均为 0")

        if (
            connection_edge_index.dtype != torch.long
            or connection_edge_index.ndim != 2
            or connection_edge_index.shape[0] != 2
        ):
            raise ValueError("connection_edge_index 必须是 int64 [2,C]")
        num_connections = connection_edge_index.shape[1]
        if tuple(connection_atom_ids.shape) != (num_connections, 2):
            raise ValueError(
                f"connection_atom_ids 形状应为 ({num_connections},2)，"
                f"实际为 {tuple(connection_atom_ids.shape)}"
            )
        if connection_atom_ids.dtype != torch.long:
            raise TypeError("connection_atom_ids 必须是 int64")
        if tuple(connection_distance.shape) != (num_connections,):
            raise ValueError(
                f"connection_distance 形状应为 ({num_connections},)，"
                f"实际为 {tuple(connection_distance.shape)}"
            )
        if connection_distance.dtype != torch.float32:
            raise TypeError("connection_distance 必须是 float32")
        if not torch.isfinite(connection_distance).all():
            raise ValueError("connection_distance 含 NaN/Inf")
        if (connection_distance < 0).any():
            raise ValueError("connection_distance 不允许负值")
        if (connection_atom_ids < 0).any():
            raise ValueError("connection_atom_ids 不允许负值")

        if num_connections:
            conn_src, conn_dst = connection_edge_index
            if (
                (connection_edge_index < 0).any()
                or (connection_edge_index >= num_nodes).any()
            ):
                raise ValueError("connection_edge_index 含越界节点索引")
            if (conn_src == conn_dst).any():
                raise ValueError("connection_edge_index 不允许节点自环")
            if not (
                is_ligand[conn_src] & is_ligand[conn_dst]
            ).all():
                raise ValueError("共价连接端点必须均为 ligand 节点")

            seen_atom_pairs = set()
            node_pair_counts = Counter()
            for connection_idx in range(num_connections):
                first_node = int(conn_src[connection_idx])
                second_node = int(conn_dst[connection_idx])
                first_atom = int(connection_atom_ids[connection_idx, 0])
                second_atom = int(connection_atom_ids[connection_idx, 1])
                if first_node > second_node:
                    first_node, second_node = second_node, first_node
                    first_atom, second_atom = second_atom, first_atom
                atom_pair = (
                    first_node, second_node, first_atom, second_atom
                )
                if atom_pair in seen_atom_pairs:
                    raise ValueError("检测到重复的共价连接原子对")
                seen_atom_pairs.add(atom_pair)
                node_pair = (first_node, second_node)
                node_pair_counts[node_pair] += 1
                if node_pair_counts[node_pair] > 2:
                    raise ValueError("同一 LL 节点对包含超过两个共价原子对")

        src, dst = edge_index
        edge_classes = torch.argmax(edge_attr_f, dim=-1)
        ll_mask = edge_classes == 0
        lp_mask = edge_classes == 1
        pp_mask = edge_classes == 2
        if ll_mask.any() and not (
            is_ligand[src[ll_mask]] & is_ligand[dst[ll_mask]]
        ).all():
            raise ValueError("LL 边端点必须均为 ligand")
        valid_lp = (
            (is_ligand[src[lp_mask]] & is_protein[dst[lp_mask]])
            | (is_protein[src[lp_mask]] & is_ligand[dst[lp_mask]])
        )
        if lp_mask.any() and not valid_lp.all():
            raise ValueError("LP 边必须连接 ligand 与 protein")
        if pp_mask.any() and not (
            is_protein[src[pp_mask]] & is_protein[dst[pp_mask]]
        ).all():
            raise ValueError("PP 边端点必须均为 protein")

        # GL 必须严格双向覆盖每个 ligand，且不能重复。
        gl_mask = edge_attr_f[:, 3].bool()
        gl_edges = edge_index[:, gl_mask]
        gl_ids = gl_edges[0] * num_nodes + gl_edges[1]
        if gl_ids.unique().numel() != gl_ids.numel():
            raise ValueError("检测到重复 GL 边")
        expected = set()
        for ligand_idx in torch.nonzero(is_ligand, as_tuple=False).flatten().tolist():
            expected.add((global_idx, ligand_idx))
            expected.add((ligand_idx, global_idx))
        actual = set(map(tuple, gl_edges.t().cpu().tolist()))
        if actual != expected:
            missing = len(expected - actual)
            extra = len(actual - expected)
            raise ValueError(f"GL 双向覆盖非法：缺失 {missing} 条，多余 {extra} 条")

    def __call__(self, data):
        data = self.embedding_lookup.resolve(data)
        self._validate_contract(data)
        num_nodes = data.x.shape[0]
        data.frag_embeds = data.frag_embeds.float()
        data.x = data.x.float()
        data.node_type = data.node_type.float()
        data.pos = data.pos.float()
        data.ref_coords = data.ref_coords.float() - data.pos[:, None, :]
        data.hac = data.hac.long()
        data.ring_count = data.ring_count.long()
        data.connection_edge_index = data.connection_edge_index.long()
        data.connection_atom_ids = data.connection_atom_ids.long()
        data.connection_distance = data.connection_distance.float()

        data.is_ligand = data.node_type[:, 0].bool()
        data.is_protein = data.node_type[:, 1].bool()
        data.is_global = data.node_type[:, 2].bool()

        slot_ids = torch.arange(3, device=data.hac.device).unsqueeze(0)
        physical_slots = torch.clamp(data.hac, min=0, max=3).unsqueeze(1)
        data.ref_atom_mask = slot_ids < physical_slots
        data.ref_atom_mask[data.is_global] = False

        dynamic_mask = data.edge_attr[:, :2].sum(dim=-1).bool()
        data.gt_dynamic_edge_index = data.edge_index[:, dynamic_mask].clone()
        data.gt_dynamic_edge_type = torch.argmax(data.edge_attr[dynamic_mask, :2], dim=-1).long()

        # 共价头只预测“节点对是否连接”，因此将少量双原子对记录折叠为一个
        # 无向正样本；完成监督标签构造后不再把原始端口原子字段带入运行时 Batch。
        if data.connection_edge_index.shape[1]:
            connection_low = torch.minimum(
                data.connection_edge_index[0],
                data.connection_edge_index[1],
            )
            connection_high = torch.maximum(
                data.connection_edge_index[0],
                data.connection_edge_index[1],
            )
            connection_ids = connection_low * num_nodes + connection_high
            order = torch.argsort(connection_ids)
            sorted_ids = connection_ids[order]
            keep = torch.ones_like(sorted_ids, dtype=torch.bool)
            keep[1:] = sorted_ids[1:] != sorted_ids[:-1]
            selected = order[keep]
            data.gt_covalent_edge_index = torch.stack(
                [connection_low[selected], connection_high[selected]], dim=0
            )
        else:
            data.gt_covalent_edge_index = torch.empty(
                (2, 0), dtype=torch.long, device=data.edge_index.device
            )
        del data.connection_edge_index
        del data.connection_atom_ids
        del data.connection_distance

        fixed_mask = data.edge_attr[:, 2:].sum(dim=-1).bool()
        fixed_edge_index = data.edge_index[:, fixed_mask]
        fixed_edge_attr = data.edge_attr[fixed_mask].float()
        data.edge_index, data.edge_attr = self._deduplicate_edges(
            fixed_edge_index, fixed_edge_attr, num_nodes
        )
        # 原始 .pt 继续保留逐节点 SMILES；运行时完成解析后不让变长字符串
        # 序列进入 PyG Batch，避免非张量字段产生不必要的拼接行为。
        if hasattr(data, 'fragment_smiles'):
            del data.fragment_smiles
        data.num_nodes = num_nodes
        return data


def build_preprocess_transform():
    """在数据集创建时加载一次标准词表，避免模块导入产生数据 I/O。"""
    return PreprocessNodeFeatures(FragmentEmbeddingLookup())

class PreloadedGraphDataset(TorchDataset):
    """
    [极速全内存版] 将所有数据及预处理结果一次性加载到内存中。
    彻底消除每个 Epoch 训练时的磁盘 I/O 瓶颈和 CPU 预处理开销。
    """
    def __init__(self, data_dir, transform=None, max_workers=16):
        self.data_dir = data_dir
        self.transform = transform
        self.embedding_table_profile = (
            transform.embedding_table_profile
            if transform is not None
            and hasattr(transform, 'embedding_table_profile')
            else None
        )
        
        # 扫描目录下所有的 .pt 文件
        search_pattern = os.path.join(data_dir, '**', '*.pt')
        self.file_list = sorted(glob.glob(search_pattern, recursive=True))
        
        if not self.file_list:
            raise FileNotFoundError(f"在 {data_dir} 中未找到任何 .pt 文件。")

        if is_main_process():
            print(f"\n🚀 开始将 {len(self.file_list)} 个图数据预加载到物理内存...")
            print("注意：多卡 DDP 模式下，每个 GPU 进程会持有一份内存副本。")

        # 定义单文件加载与预处理函数
        def load_and_transform(f_path):
            try:
                # 1. 从硬盘读取
                data = torch.load(f_path, weights_only=False)
                # 2. 提前在 CPU 上完成 Transform (特征拼接、相对坐标计算等)
                if self.transform is not None:
                    data = self.transform(data)
                return data, None
            except Exception as e:
                return None, (f_path, str(e))

        # 使用多线程极速并发读取
        disable_tqdm = not is_main_process()
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务并显示进度条
            results = list(tqdm(
                executor.map(load_and_transform, self.file_list),
                total=len(self.file_list),
                desc="[RAM Loading]",
                disable=disable_tqdm
            ))
            
        failures = [failure for _, failure in results if failure is not None]
        if failures:
            preview = "\n".join(f"  - {path}: {error}" for path, error in failures[:20])
            suffix = "" if len(failures) <= 20 else f"\n  ...另有 {len(failures) - 20} 个失败样本"
            raise RuntimeError(
                f"发现 {len(failures)} 个非法/损坏 PyGData，已终止而非静默跳过：\n"
                f"{preview}{suffix}"
            )
        self.data_list = [data for data, _ in results]
        
        if is_main_process():
            print(f"✅ 预加载完成！成功驻留 {len(self.data_list)} 个图数据至内存。\n")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # 训练时，直接以 O(1) 的光速从内存返回已经预处理好的对象
        return self.data_list[idx]     


# =============================================================================
# 2. 训练子集画像
# =============================================================================
def _coverage_cap(values, coverage=0.95):
    if not values:
        raise ValueError("无法从空分布计算覆盖率阈值")
    counts = Counter(int(v) for v in values)
    target = math.ceil(len(values) * coverage)
    cumulative = 0
    for value in sorted(counts):
        cumulative += counts[value]
        if cumulative >= target:
            return int(value)
    return int(max(counts))


def _p95_integer(values):
    if not values:
        return 1
    ordered = sorted(int(v) for v in values)
    index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return max(1, ordered[index])


def _balanced_class_weights(frequencies):
    frequencies = torch.as_tensor(frequencies, dtype=torch.float64)
    weights = torch.zeros_like(frequencies)
    observed = frequencies > 0
    if observed.any():
        weights[observed] = frequencies[observed].rsqrt()
        weights[observed] /= weights[observed].mean()
        weights[observed] = weights[observed].clamp(0.5, 4.0)
    return [float(v) for v in weights.tolist()]


def _unwrap_subset(dataset):
    indices = list(range(len(dataset)))
    base = dataset
    while isinstance(base, torch.utils.data.Subset):
        indices = [base.indices[i] for i in indices]
        base = base.dataset
    return base, [int(i) for i in indices]


def _dataset_manifest(base_dataset, selected_indices, data_dir):
    file_list = getattr(base_dataset, 'file_list', None)
    if file_list is None:
        selected_paths = [f"dataset-index:{idx}" for idx in selected_indices]
        all_paths = selected_paths
    else:
        all_paths = [os.path.abspath(path) for path in file_list]
        selected_paths = [all_paths[idx] for idx in selected_indices]

    manifest_lines = []
    for path in all_paths:
        if os.path.isfile(path):
            stat = os.stat(path)
            rel = os.path.relpath(path, data_dir).replace('\\', '/')
            manifest_lines.append(f"{rel}\t{stat.st_size}\t{stat.st_mtime_ns}")
        else:
            manifest_lines.append(path)
    split_lines = [
        os.path.relpath(path, data_dir).replace('\\', '/')
        if os.path.isabs(path) else path
        for path in selected_paths
    ]
    return (
        hashlib.sha256("\n".join(manifest_lines).encode('utf-8')).hexdigest(),
        hashlib.sha256("\n".join(split_lines).encode('utf-8')).hexdigest(),
        len(all_paths),
    )


def build_training_data_profile(train_dataset, data_dir=config.PYG_DATA_DIR, seed=config.SEED):
    """
    仅统计固定划分后的训练子集。调用方负责在 DDP Rank 0 执行并广播结果。
    """
    base_dataset, selected_indices = _unwrap_subset(train_dataset)
    embedding_table_profile = getattr(
        base_dataset, 'embedding_table_profile', None
    )
    if not isinstance(embedding_table_profile, dict):
        raise ValueError("训练数据集缺少片段词嵌入表画像")
    hac_values, ring_values = [], []
    ll_degrees, lp_degrees = [], []
    covalent_positive_pairs = 0
    covalent_total_pairs = 0

    for base_idx in selected_indices:
        data = base_dataset[base_idx]
        ligand_indices = torch.nonzero(data.is_ligand, as_tuple=False).flatten()
        hac_values.extend(int(v) for v in data.hac[ligand_indices].tolist())
        ring_values.extend(int(v) for v in data.ring_count[ligand_indices].tolist())
        num_ligand_nodes = int(ligand_indices.numel())
        covalent_total_pairs += num_ligand_nodes * (num_ligand_nodes - 1) // 2
        covalent_positive_pairs += int(data.gt_covalent_edge_index.shape[1])

        ll_neighbors = {int(idx): set() for idx in ligand_indices.tolist()}
        lp_neighbors = {int(idx): set() for idx in ligand_indices.tolist()}
        gt_edges = data.gt_dynamic_edge_index
        gt_types = data.gt_dynamic_edge_type
        for edge_pos in range(gt_edges.shape[1]):
            src = int(gt_edges[0, edge_pos])
            dst = int(gt_edges[1, edge_pos])
            edge_type = int(gt_types[edge_pos])
            if edge_type == 0 and data.is_ligand[src] and data.is_ligand[dst] and src != dst:
                ll_neighbors[src].add(dst)
                ll_neighbors[dst].add(src)
            elif edge_type == 1:
                if data.is_ligand[src] and data.is_protein[dst]:
                    lp_neighbors[src].add(dst)
                elif data.is_ligand[dst] and data.is_protein[src]:
                    lp_neighbors[dst].add(src)

        ll_degrees.extend(len(ll_neighbors[int(idx)]) for idx in ligand_indices.tolist())
        lp_degrees.extend(len(lp_neighbors[int(idx)]) for idx in ligand_indices.tolist())

    if not hac_values:
        raise ValueError("训练子集中没有 ligand 节点，无法生成训练画像")
    if covalent_positive_pairs == 0:
        raise ValueError("训练子集中没有共价 LL 节点对，无法训练共价预测头")
    if covalent_positive_pairs > covalent_total_pairs:
        raise ValueError("共价 LL 正样本数超过全部无向 LL 节点对数量")

    hac_cap = max(1, _coverage_cap(hac_values))
    ring_cap = max(0, _coverage_cap(ring_values))
    covalent_negative_pairs = covalent_total_pairs - covalent_positive_pairs
    covalent_pos_weight = min(
        float(config.COVALENT_POS_WEIGHT_MAX),
        max(1.0, covalent_negative_pairs / covalent_positive_pairs),
    )

    # HAC: class 0..cap-1 对应 HAC 1..cap，class cap 为 overflow。
    hac_frequencies = [0] * (hac_cap + 1)
    for value in hac_values:
        class_index = value - 1 if value <= hac_cap else hac_cap
        hac_frequencies[class_index] += 1

    # Ring: class 0..cap 对应真实环数，class cap+1 为 overflow。
    ring_frequencies = [0] * (ring_cap + 2)
    for value in ring_values:
        class_index = value if value <= ring_cap else ring_cap + 1
        ring_frequencies[class_index] += 1

    manifest_hash, split_hash, num_files = _dataset_manifest(
        base_dataset, selected_indices, os.path.abspath(data_dir)
    )
    return {
        'data_dir': os.path.abspath(data_dir),
        'fragment_embedding_table': embedding_table_profile,
        'split_seed': int(seed),
        'train_val_split': float(config.TRAIN_VAL_SPLIT),
        'num_dataset_files': int(num_files),
        'num_training_graphs': len(selected_indices),
        'file_manifest_sha256': manifest_hash,
        'training_split_sha256': split_hash,
        'hac_cap': hac_cap,
        'hac_num_classes': hac_cap + 1,
        'hac_histogram': {
            str(key): int(value) for key, value in sorted(Counter(hac_values).items())
        },
        'hac_class_mapping': {
            'classes': {str(i): i + 1 for i in range(hac_cap)},
            'overflow_class': hac_cap,
        },
        'hac_frequencies': hac_frequencies,
        'hac_class_weights': _balanced_class_weights(hac_frequencies),
        'ring_cap': ring_cap,
        'ring_num_classes': ring_cap + 2,
        'ring_histogram': {
            str(key): int(value) for key, value in sorted(Counter(ring_values).items())
        },
        'ring_class_mapping': {
            'classes': {str(i): i for i in range(ring_cap + 1)},
            'overflow_class': ring_cap + 1,
        },
        'ring_frequencies': ring_frequencies,
        'ring_class_weights': _balanced_class_weights(ring_frequencies),
        'covalent_positive_pairs': covalent_positive_pairs,
        'covalent_negative_pairs': covalent_negative_pairs,
        'covalent_pos_weight': float(covalent_pos_weight),
        'max_ll_neighbors': _p95_integer(ll_degrees),
        'max_lp_neighbors': _p95_integer(lp_degrees),
        'll_degree_histogram': {
            str(key): int(value) for key, value in sorted(Counter(ll_degrees).items())
        },
        'lp_degree_histogram': {
            str(key): int(value) for key, value in sorted(Counter(lp_degrees).items())
        },
    }


def save_training_data_profile(profile, output_path=config.TRAINING_PROFILE_PATH):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    temporary_path = output_path + '.tmp'
    with open(temporary_path, 'w', encoding='utf-8') as handle:
        json.dump(profile, handle, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(temporary_path, output_path)

# =============================================================================
# 3. 数据集加载与划分 (DataLoader Factory)
# =============================================================================
def get_train_val_dataloaders(data_dir=config.PYG_DATA_DIR, 
                              batch_size=config.BATCH_SIZE, 
                              seed=config.SEED, 
                              distributed=False):
    if is_main_process():
        mode_str = "Lazy Loading (硬盘动态读取)" if config.USE_LAZY_DATASET else "Preloaded (内存常驻加速)"
        print(f"\n--- 开始初始化数据集 [{mode_str}] ---")
        print(f"数据源: {data_dir}")

    preprocess_transform = build_preprocess_transform()
    if is_main_process():
        embedding_profile = preprocess_transform.embedding_table_profile
        print(
            "片段词嵌入表: "
            f"{embedding_profile['num_smiles']} 个 SMILES, "
            f"{embedding_profile['embedding_dim']} 维, "
            f"SHA256={embedding_profile['table_sha256']}"
        )
    
    # 1. 根据全局配置，实例化对应的数据集
    if config.USE_LAZY_DATASET:
        full_dataset = LazyGraphDataset(data_dir=data_dir, transform=preprocess_transform)
    else:
        full_dataset = PreloadedGraphDataset(data_dir=data_dir, transform=preprocess_transform)
    
    if is_main_process():
        print(f"数据集初始化完成，共扫描到 {len(full_dataset)} 个复合物文件。")

    # 2. 按比例划分训练集和验证集
    num_data = len(full_dataset)
    num_train = int(num_data * config.TRAIN_VAL_SPLIT)
    num_val = num_data - num_train

    # 使用固定的 Generator 确保所有 GPU 上的划分一致
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [num_train, num_val],
        generator=torch.Generator().manual_seed(seed)
    )

    if is_main_process():
        print(f"数据集划分完成：训练集 {len(train_dataset)} 样本，验证集 {len(val_dataset)} 样本。")

    # =========================================================================
    # DDP Sampler 设置
    # =========================================================================
    if distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
        train_shuffle = False
    else:
        train_sampler = None
        val_sampler = None
        train_shuffle = True

    # =========================================================================
    # 创建 DataLoader
    # =========================================================================
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=train_shuffle, 
        sampler=train_sampler, 
        num_workers=config.NUM_WORKERS, 
        pin_memory=config.PIN_MEMORY,
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        sampler=val_sampler,   
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )
    
    return train_loader, val_loader, train_sampler


def get_test_dataloader(data_dir=config.PYG_TEST_DATA_DIR, 
                        batch_size=config.BATCH_SIZE, 
                        distributed=False):
    if not os.path.exists(data_dir):
        if is_main_process():
            print(f"提示: 测试数据目录不存在: {data_dir}")
        return None

    if is_main_process():
        print(f"\n--- 开始初始化测试数据集 ---")

    preprocess_transform = build_preprocess_transform()

    try:
        if config.USE_LAZY_DATASET:
            test_dataset = LazyGraphDataset(data_dir=data_dir, transform=preprocess_transform)
        else:
            test_dataset = PreloadedGraphDataset(data_dir=data_dir, transform=preprocess_transform)
    except FileNotFoundError:
        if is_main_process():
            print(f"提示: 测试数据目录为空: {data_dir}")
        return None

    if is_main_process():
        print(f"测试集初始化完成，共 {len(test_dataset)} 个样本。")

    if distributed:
        test_sampler = DistributedSampler(test_dataset, shuffle=False)
    else:
        test_sampler = None

    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        sampler=test_sampler, 
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )
    
    return test_loader

if __name__ == '__main__':
    # 简单的测试逻辑
    try:
        train_l, val_l, _ = get_train_val_dataloaders()
        batch = next(iter(train_l))
        print(f"\nBatch Test Success:")
        print(f"Fragment Embeddings: {batch.frag_embeds.shape}")
        print(f"Chemical Features: {batch.x.shape}")
        print(f"Node Types: {batch.node_type.shape}")
        print(f"Positions: {batch.pos.shape}")
        print(f"Is Ligand: {batch.is_ligand.sum()}")
    except Exception as e:
        print(f"Test failed (Expected if data path is empty): {e}")
