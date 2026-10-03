# =============================================================================
#
#                               utils.py
#
# =============================================================================

# ----------------------------------------
# 导入所需的模块
# ----------------------------------------

# RDKit
from rdkit import Chem, RDConfig
from rdkit.Chem import AllChem, rdMolDescriptors, rdFingerprintGenerator
from rdkit.Chem import ChemicalFeatures as Feat
from rdkit.Chem.Draw import MolsToGridImage
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.rdmolops import GetDistanceMatrix
from rdkit import Chem, RDConfig, RDLogger

# Python Standard & Third-Party Libraries
import os
import json
import math
import hashlib
import socket
import sys
import time
import uuid
import warnings
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import DataStructs

_PHISTONE_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PHISTONE_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PHISTONE_PROJECT_ROOT))

from PhiSLinker.PhiSLinker_atom_pair_smiles_decoder import (
    calculate_reference_coordinate_system as _decoder_reference_frame,
)

# 忽略 NumPy 的空切片警告，保持控制台绝对清爽
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")

# IPython (可选，用于Jupyter环境)
from IPython.display import display, HTML, clear_output


# ----------------------------------------
# 定义辅助函数
# ----------------------------------------

def get_num_non_dummy_atoms(mol):
    """计算一个片段中“真实”原子的数量（即不包括切割产生的虚拟原子*）。"""
    return sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() != 0)

def calculate_heavy_atom_count(mol):
    """显式计算非氢重原子数量，同时排除 dummy 原子。"""
    if mol is None:
        return 0
    return sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1)

def calculate_ring_count(mol):
    """计算 RDKit 感知到的环数量。"""
    if mol is None:
        return 0
    return int(rdMolDescriptors.CalcNumRings(mol))

def reconstruct_mol_with_coords(original_mol):
    """
    [严格模式] 通过 SMILES 重建分子拓扑。
    如果 SMILES 生成失败、重建失败或子结构匹配失败，直接返回 None。
    """
    if not original_mol:
        return None

    try:
        # 1. 尝试生成规范化 SMILES
        smiles = Chem.MolToSmiles(original_mol, isomericSmiles=True, canonical=True)
        if not smiles: 
            return None
            
        new_mol = Chem.MolFromSmiles(smiles)
        if not new_mol:
            return None

        # --- 场景 A: 3D 分子 ---
        if original_mol.GetNumConformers() > 0:
            if new_mol.GetNumAtoms() != original_mol.GetNumAtoms():
                return None # 原子数不匹配，视为失败
            
            # 必须匹配成功
            match = original_mol.GetSubstructMatch(new_mol, useChirality=True)
            if not match or len(match) != new_mol.GetNumAtoms():
                match = original_mol.GetSubstructMatch(new_mol, useChirality=False)
                if not match or len(match) != new_mol.GetNumAtoms():
                    return None # 匹配失败，直接舍弃

            # 移植坐标与属性
            conf = Chem.Conformer(new_mol.GetNumAtoms())
            original_conf = original_mol.GetConformer(0)
            
            for new_idx, old_idx in enumerate(match):
                pos = original_conf.GetAtomPosition(old_idx)
                conf.SetAtomPosition(new_idx, pos)
                
                old_atom = original_mol.GetAtomWithIdx(old_idx)
                new_atom = new_mol.GetAtomWithIdx(new_idx)
                if old_atom.HasProp("_original_index"):
                    new_atom.SetIntProp("_original_index", old_atom.GetIntProp("_original_index"))
                if old_atom.HasProp("_source_fragment_id"):
                    new_atom.SetProp("_source_fragment_id", old_atom.GetProp("_source_fragment_id"))
                if old_atom.HasProp("_GasteigerCharge"):
                    new_atom.SetProp("_GasteigerCharge", old_atom.GetProp("_GasteigerCharge"))

            new_mol.AddConformer(conf)
            return new_mol

        # --- 场景 B: 2D 分子 ---
        else:
             # 对于无3D坐标的情况，只要原子数一致且能匹配，就返回新拓扑
             if new_mol.GetNumAtoms() == original_mol.GetNumAtoms():
                 match = original_mol.GetSubstructMatch(new_mol, useChirality=True)
                 if match:
                     for new_idx, old_idx in enumerate(match):
                         old_atom = original_mol.GetAtomWithIdx(old_idx)
                         new_atom = new_mol.GetAtomWithIdx(new_idx)
                         if old_atom.HasProp("_original_index"):
                             new_atom.SetIntProp("_original_index", old_atom.GetIntProp("_original_index"))
                         if old_atom.HasProp("_source_fragment_id"):
                             new_atom.SetProp("_source_fragment_id", old_atom.GetProp("_source_fragment_id"))
                     return new_mol
             return None

    except Exception as e:
        # 发生任何异常，视为失败，返回 None
        return None

  
def would_create_isolated_atom(mol, bonds_to_cut_indices):
    """
    预测性检查：判断同时切割指定的键列表是否会导致某个 *碳* 原子被完全孤立。
    一个孤立的碳原子是指，它在切割后形成的片段里，是唯一的真实原子，且与其他两个或以上的虚拟原子相连。
    """
    temp_mol = Chem.FragmentOnBonds(mol, bonds_to_cut_indices, addDummies=True)
    for frag in Chem.GetMolFrags(temp_mol, asMols=True):
        # 至少需要3个原子才能构成 "1个真实原子 + 2个虚拟原子" 的情况
        if frag.GetNumAtoms() >= 3:
            non_dummy_atoms = [atom for atom in frag.GetAtoms() if atom.GetAtomicNum() != 0]

            # 检查片段是否只含一个真实原子，且虚拟原子数>=2
            if len(non_dummy_atoms) == 1 and (frag.GetNumAtoms() - 1) >= 2:
                single_real_atom = non_dummy_atoms[0]

                if single_real_atom.GetAtomicNum() == 6:
                    return True
    return False

def identify_carbon_a_atoms(fragment):
    """
    (V2.0 修订版) 识别并返回片段中所有 "Carbon 'a'" 原子的索引集合。
    Carbon 'a' 新定义: 连接了任何杂原子 或 参与了不饱和键的碳。
    """
    carbon_a_indices = set()
    for atom in fragment.GetAtoms():
        if atom.GetAtomicNum() != 6: continue

        # 条件1: 邻居中是否存在杂原子 (非碳、非氢、非虚拟原子)
        hetero_neighbors = sum(1 for n in atom.GetNeighbors() if n.GetAtomicNum() not in [6, 1, 0])
        # 条件2: 自身是否参与了任何非单键
        has_non_single_bond = any(b.GetBondType() != Chem.BondType.SINGLE for b in atom.GetBonds())

        if hetero_neighbors > 0 or has_non_single_bond:
            carbon_a_indices.add(atom.GetIdx())

    return carbon_a_indices

def get_ring_bond_info(mol, eccentricities):
    """
    分析分子中的环系，并为每个键标记其在环系统中的拓扑角色。
    这是切割复杂环状结构（如稠环、桥环）的核心预处理步骤。
    """
    bond_info = {}
    ri = mol.GetRingInfo()

    # 建立一个从原子索引到其所属环索引的映射
    atom_to_rings_map = {i: [] for i in range(mol.GetNumAtoms())}
    for ring_idx, atom_indices in enumerate(ri.AtomRings()):
        for atom_idx in atom_indices:
            atom_to_rings_map[atom_idx].append(ring_idx)

    # 临时存储键的属性，并找出所有桥头原子
    temp_bond_props = {}
    b_bond_atom_indices = set() # 存储所有'b'类键的端点原子（桥头原子）

    for bond in mol.GetBonds():
        idx = bond.GetIdx()
        # 'a'类键: 任何环中的键
        is_a = bond.IsInRing()
        if not is_a:
            temp_bond_props[idx] = {'is_a': False, 'is_b': False, 'is_c': False}
            continue

        begin_idx, end_idx = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        # 'b'类键: 桥头键，同时属于两个或以上环系的键
        is_b = len(set(atom_to_rings_map[begin_idx]).intersection(set(atom_to_rings_map[end_idx]))) >= 2
        # 'c'类键: 芳香键
        is_c = bond.GetIsAromatic()

        temp_bond_props[idx] = {'is_a': is_a, 'is_b': is_b, 'is_c': is_c}
        if is_b:
            b_bond_atom_indices.add(begin_idx)
            b_bond_atom_indices.add(end_idx)

    # 再次遍历，根据桥头原子信息确定'd'类键
    for bond in mol.GetBonds():
        idx = bond.GetIdx()
        props = temp_bond_props.get(idx, {'is_a': False})
        is_d = False
        # 'd'类键: 连接桥头原子的非桥头、非芳香的环键。是切割复杂环系的理想目标。
        if props.get('is_a') and not props.get('is_b') and not props.get('is_c'):
            begin_idx, end_idx = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if begin_idx in b_bond_atom_indices or end_idx in b_bond_atom_indices:
                is_d = True

        bond_info[idx] = {
            'is_a': props.get('is_a'), 'is_b': props.get('is_b'), 'is_c': props.get('is_c'), 'is_d': is_d,
            'ecc': (eccentricities[bond.GetBeginAtomIdx()] + eccentricities[bond.GetEndAtomIdx()]) / 2.0 if props.get('is_a') else -1.0
        }
    return bond_info

def find_type_a_rings(frag, main_mol_for_matching, match_indices, bond_info):
    """规则5的辅助函数：寻找含有两个或以上'd'类键的环。"""
    type_a_rings = []
    # 遍历片段中的每一个环
    for ring_atoms in frag.GetRingInfo().AtomRings():
        # 将片段环中的原子映射回原始分子，找到对应的化学键
        orig_bonds = [b.GetIdx() for i in range(len(ring_atoms)) if (b := main_mol_for_matching.GetBondBetweenAtoms(match_indices[ring_atoms[i]], match_indices[ring_atoms[(i + 1) % len(ring_atoms)]]))]
        
        # 检查这些原始化学键中有多少个是 'd' 类键
        d_bonds = [b_idx for b_idx in orig_bonds if bond_info.get(b_idx, {}).get('is_d')]
        
        # 如果一个环中'd'类键的数量大于等于2，则将其作为候选环
        if len(d_bonds) >= 2:
            type_a_rings.append({'atom_indices': ring_atoms, 'orig_d_bond_indices': d_bonds})
            
    return type_a_rings


def get_atom_coordinates(mol):
    """
    [已修改] 从一个RDKit分子中提取所有原子的3D坐标到一个numpy数组中。
    用于替代原Notebook中的get_atom_coords，并支持scipy的距离计算。
    """
    if not mol or mol.GetNumConformers() == 0:
        return np.array([])
    
    conf = mol.GetConformer(0)
    # 直接返回 (N, 3) 的 NumPy 数组
    return np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])

def calculate_centroid(mol_with_conf):
    """
    (增强版) 计算一个分子/碎片中所有重原子的几何中心（质心）坐标。
    
    改进点：
    显式忽略氢原子 (AtomicNum <= 1)。
    防止因去氢不彻底或虚原子残留导致的坐标偏差 (即忽略坐标为 0,0,0 的虚假原子)。
    """
    if not mol_with_conf or mol_with_conf.GetNumConformers() == 0:
        return None
        
    conf = mol_with_conf.GetConformer(0)
    
    # --- [关键修改] 筛选重原子 ---
    heavy_atom_indices = [a.GetIdx() for a in mol_with_conf.GetAtoms() if a.GetAtomicNum() > 1]
    num_heavy_atoms = len(heavy_atom_indices)
    
    if num_heavy_atoms == 0:
        return None    
        
    centroid = np.array([0.0, 0.0, 0.0])
    
    # 只累加重原子的坐标
    for i in heavy_atom_indices:
        pos = conf.GetAtomPosition(i)
        centroid += np.array([pos.x, pos.y, pos.z])
        
    # 除以重原子数量
    return tuple(centroid / num_heavy_atoms)


def calculate_reference_coordinate_system(mol_with_conf):
    """使用 PhiSLinker 的规范锚点规则生成三点参考框架。"""

    return _decoder_reference_frame(mol_with_conf)


def calculate_max_distance(mol_with_conf):
    """计算分子中任意两个原子间的最大3D距离。"""
    if not mol_with_conf or mol_with_conf.GetNumConformers() == 0: return 0.0
    return AllChem.Get3DDistanceMatrix(mol_with_conf).max()

def calculate_max_angle(mol_with_conf):
    """计算分子中任意三个原子形成的最大键角。"""
    if not mol_with_conf or mol_with_conf.GetNumConformers() == 0: return 0.0
    conf, num_atoms = mol_with_conf.GetConformer(), mol_with_conf.GetNumAtoms()
    if num_atoms < 3: return 0.0
    positions, max_angle = [conf.GetAtomPosition(i) for i in range(num_atoms)], 0.0
    for p_a, p_b, p_c in combinations(positions, 3):
        v_ba, v_bc = p_a - p_b, p_c - p_b
        len_ba, len_bc = v_ba.Length(), v_bc.Length()
        if len_ba < 1e-6 or len_bc < 1e-6: continue
        cos_angle = v_ba.DotProduct(v_bc) / (len_ba * len_bc)
        angle = math.degrees(math.acos(max(-1.0, min(1.0, cos_angle))))
        if angle > max_angle: max_angle = angle
    return max_angle

def calculate_max_plane_angle(
    mol_with_conf,
    min_triangle_quality=0.05,
    block_size=256,
):
    """
    计算任意两个有效三原子平面之间的最大夹角。

    三点近共线时不构成稳定平面，使用无量纲三角形质量过滤。
    平面不区分法向量正反方向，因此最终角度范围为 [0, 90] 度。
    """
    if (
        mol_with_conf is None
        or mol_with_conf.GetNumConformers() == 0
    ):
        return 0.0

    # 与其他三维特征口径一致，只使用重原子。
    atom_indices = [
        atom.GetIdx()
        for atom in mol_with_conf.GetAtoms()
        if atom.GetAtomicNum() > 1
    ]
    if len(atom_indices) < 4:
        return 0.0

    conf = mol_with_conf.GetConformer(0)
    coordinates = np.asarray(
        [
            list(conf.GetAtomPosition(atom_index))
            for atom_index in atom_indices
        ],
        dtype=np.float64,
    )

    unit_normals = []

    for first, second, third in combinations(
        range(len(coordinates)), 3
    ):
        p1 = coordinates[first]
        p2 = coordinates[second]
        p3 = coordinates[third]

        vector_12 = p2 - p1
        vector_13 = p3 - p1
        vector_23 = p3 - p2

        side_sq_12 = float(np.dot(vector_12, vector_12))
        side_sq_13 = float(np.dot(vector_13, vector_13))
        side_sq_23 = float(np.dot(vector_23, vector_23))
        max_side_sq = max(side_sq_12, side_sq_13, side_sq_23)

        # 原子重合或三角形尺度极小。
        if max_side_sq < 1e-12:
            continue

        normal = np.cross(vector_12, vector_13)
        normal_length = float(np.linalg.norm(normal))

        # 约等于三角形高度与最长边的比值。
        triangle_quality = normal_length / max_side_sq
        if triangle_quality < min_triangle_quality:
            continue

        unit_normals.append(normal / normal_length)

    if len(unit_normals) < 2:
        return 0.0

    unit_normals = np.asarray(unit_normals, dtype=np.float64)

    # 最大平面夹角对应最小的 |n1·n2|。
    # 分块计算，避免一次构造过大的 M×M 矩阵。
    min_abs_dot = 1.0

    for start in range(0, len(unit_normals), block_size):
        normal_block = unit_normals[start:start + block_size]
        abs_dots = np.abs(normal_block @ unit_normals.T)
        min_abs_dot = min(min_abs_dot, float(abs_dots.min()))

    min_abs_dot = float(np.clip(min_abs_dot, 0.0, 1.0))
    return float(np.degrees(np.arccos(min_abs_dot)))


PORT_CUT_IDS_PROP = "_port_cut_ids"
SOURCE_FRAGMENT_PROP = "_source_fragment_id"


class PortValidationError(RuntimeError):
    """端口记录无法形成完整、无歧义配对时抛出的异常。"""


class ConnectionTopologyError(PortValidationError):
    """连接出现自环或同一节点对包含过多成键原子对时抛出的异常。"""


class LigandSecondaryCleaningError(RuntimeError):
    """配体片段在二次清洗中仍无法通过净化时抛出的异常。"""


def _read_atom_port_cut_ids(atom):
    if not atom.HasProp(PORT_CUT_IDS_PROP):
        return []
    try:
        values = json.loads(atom.GetProp(PORT_CUT_IDS_PROP))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise PortValidationError(
            f"原子 {atom.GetIdx()} 的端口记录无法解析"
        ) from exc
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        raise PortValidationError(f"原子 {atom.GetIdx()} 的端口记录格式非法")
    return values


def _append_atom_port_cut_id(atom, cut_id):
    values = _read_atom_port_cut_ids(atom)
    if cut_id in values:
        raise PortValidationError(f"切断标识重复写入同一锚点原子: {cut_id}")
    values.append(cut_id)
    atom.SetProp(PORT_CUT_IDS_PROP, json.dumps(values, ensure_ascii=True))


def fragment_on_bonds_stable(mol, bond_indices):
    """
    通过手动移除化学键并添加带唯一、持久化标签的虚原子来切割分子。
    切断标识由原始组分和原始键端点组成，并同时写入两个真实锚点原子。
    因此 dummy 后续转氢或删除时，端口关系仍可继续传递到最终片段。
    """
    if not bond_indices:
        return Chem.Mol(mol)

    # 必须在删键前一次性冻结端点信息，避免 RWMol 删除键后 bond index 重排。
    cut_specs = []
    for bond_idx in sorted(set(bond_indices)):
        bond = mol.GetBondWithIdx(int(bond_idx))
        if bond is None:
            raise PortValidationError(f"找不到待切断键: {bond_idx}")
        if bond.GetBondType() != Chem.BondType.SINGLE:
            raise PortValidationError(
                f"端口机制只允许切断单键，键 {bond_idx} 的类型为 {bond.GetBondType()}"
            )

        begin_atom_idx = bond.GetBeginAtomIdx()
        end_atom_idx = bond.GetEndAtomIdx()
        begin_atom = mol.GetAtomWithIdx(begin_atom_idx)
        end_atom = mol.GetAtomWithIdx(end_atom_idx)
        if begin_atom.GetAtomicNum() == 0 or end_atom.GetAtomicNum() == 0:
            raise PortValidationError(f"禁止再次切断 dummy 连接键: {bond_idx}")
        if begin_atom.GetAtomicNum() == 1 or end_atom.GetAtomicNum() == 1:
            raise PortValidationError(f"禁止切断含氢端点的连接键: {bond_idx}")
        if not begin_atom.HasProp(SOURCE_FRAGMENT_PROP) or not end_atom.HasProp(SOURCE_FRAGMENT_PROP):
            raise PortValidationError("待切断原子缺少来源片段标识")

        begin_source = begin_atom.GetProp(SOURCE_FRAGMENT_PROP)
        end_source = end_atom.GetProp(SOURCE_FRAGMENT_PROP)
        if begin_source != end_source:
            raise PortValidationError("一根待切断键的两个端点来自不同原始片段")

        begin_original = (
            begin_atom.GetIntProp("_original_index")
            if begin_atom.HasProp("_original_index") else begin_atom_idx
        )
        end_original = (
            end_atom.GetIntProp("_original_index")
            if end_atom.HasProp("_original_index") else end_atom_idx
        )
        low, high = sorted((begin_original, end_original))
        unique_label = f"{begin_source}:{low}:{high}"
        cut_specs.append((begin_atom_idx, end_atom_idx, begin_original, end_original, unique_label))

    rw_mol = Chem.RWMol(mol)

    for begin_atom_idx, end_atom_idx, begin_original, end_original, unique_label in cut_specs:
        begin_atom = rw_mol.GetAtomWithIdx(begin_atom_idx)
        end_atom = rw_mol.GetAtomWithIdx(end_atom_idx)
        _append_atom_port_cut_id(begin_atom, unique_label)
        _append_atom_port_cut_id(end_atom, unique_label)
        
        # 1. 移除原始的化学键
        rw_mol.RemoveBond(begin_atom_idx, end_atom_idx)
        
        # 2. 添加第一个虚原子，连接到起始原子，并打上标签
        dummy1_idx = rw_mol.AddAtom(Chem.Atom(0))
        dummy1 = rw_mol.GetAtomWithIdx(dummy1_idx)
        dummy1.SetProp("_cut_id", unique_label)
        dummy1.SetIntProp("_anchor_original_index", begin_original)
        dummy1.SetProp(SOURCE_FRAGMENT_PROP, begin_atom.GetProp(SOURCE_FRAGMENT_PROP))
        rw_mol.AddBond(begin_atom_idx, dummy1_idx, Chem.BondType.SINGLE)
        
        # 3. 添加第二个虚原子，连接到结束原子，并打上相同的标签
        dummy2_idx = rw_mol.AddAtom(Chem.Atom(0))
        dummy2 = rw_mol.GetAtomWithIdx(dummy2_idx)
        dummy2.SetProp("_cut_id", unique_label)
        dummy2.SetIntProp("_anchor_original_index", end_original)
        dummy2.SetProp(SOURCE_FRAGMENT_PROP, end_atom.GetProp(SOURCE_FRAGMENT_PROP))
        rw_mol.AddBond(end_atom_idx, dummy2_idx, Chem.BondType.SINGLE)
        
    final_mol = rw_mol.GetMol()
    # 使用SANITIZE_NONE来避免RDKit对我们手动创建的结构进行化学合理性检查
    try:
        Chem.SanitizeMol(final_mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_NONE)
    except Exception:
        pass # 容忍可能出现的净化错误
        
    return final_mol


def fragment_on_bonds_without_ports(mol, bond_indices):
    """切断 pocket 化学键，不记录 ligand 连接端口。"""
    if not bond_indices:
        return Chem.Mol(mol)
    try:
        frozen_bond_indices = sorted(set(int(idx) for idx in bond_indices))
        for bond_idx in frozen_bond_indices:
            bond = mol.GetBondWithIdx(bond_idx)
            if bond is None:
                raise PortValidationError(f"找不到待切断键: {bond_idx}")
        return Chem.FragmentOnBonds(
            mol,
            frozen_bond_indices,
            addDummies=True,
        )
    except Exception as exc:
        raise RuntimeError(f"pocket 片段切割失败: {exc}") from exc


def fragment_on_bonds_by_origin(mol, bond_indices, track_ports):
    """按来源选择 ligand 端口切割或 pocket 普通切割。"""
    if track_ports:
        return fragment_on_bonds_stable(mol, bond_indices)
    return fragment_on_bonds_without_ports(mol, bond_indices)


def _copy_atom_trace_properties(source_atom, target_atom):
    """复制端口映射与原始原子追踪所需的最小属性集合。"""
    for prop_name in (SOURCE_FRAGMENT_PROP, PORT_CUT_IDS_PROP):
        if source_atom.HasProp(prop_name):
            target_atom.SetProp(prop_name, source_atom.GetProp(prop_name))
    for prop_name in ("_original_index", "_internal_index"):
        if source_atom.HasProp(prop_name):
            target_atom.SetIntProp(prop_name, source_atom.GetIntProp(prop_name))


def canonicalize_fragment_with_ports(mol):
    """
    将最终片段转换为规范 SMILES 的原子顺序，并同步映射端口锚点和三维坐标。

    返回 ``(canonical_mol, canonical_smiles, ports)``。ports 中每项只包含
    ``cut_id`` 和规范化后的 ``anchor_index``。
    """
    try:
        canonical_smiles = Chem.MolToSmiles(
            mol, canonical=True, isomericSmiles=True
        )
        canonical_mol = Chem.MolFromSmiles(canonical_smiles)
    except Exception as exc:
        raise PortValidationError(f"片段规范化失败: {exc}") from exc
    if canonical_mol is None:
        raise PortValidationError("规范 SMILES 无法重建片段")

    # 规范 SMILES 往返时，参与双键立体定义的显式氢可能被折叠为隐式氢。
    # 因此只严格比较非氢原子的元素组成；dummy 原子（原子序数 0）仍参与比较。
    source_non_hydrogen = sorted(
        atom.GetAtomicNum()
        for atom in mol.GetAtoms()
        if atom.GetAtomicNum() != 1
    )
    canonical_non_hydrogen = sorted(
        atom.GetAtomicNum()
        for atom in canonical_mol.GetAtoms()
        if atom.GetAtomicNum() != 1
    )
    if canonical_non_hydrogen != source_non_hydrogen:
        raise PortValidationError(
            "规范 SMILES 无法保持片段非氢原子组成"
        )

    matches = mol.GetSubstructMatches(
        canonical_mol,
        uniquify=False,
        useChirality=True,
        maxMatches=4096,
    )
    if not matches:
        matches = mol.GetSubstructMatches(
            canonical_mol,
            uniquify=False,
            useChirality=False,
            maxMatches=4096,
        )
    if not matches:
        raise PortValidationError("规范片段无法映射回带端口的原始片段")

    # 对称匹配时优先把端口锚点映射到字典序最小的位置，保证输出稳定。
    def match_key(match):
        anchor_positions = []
        for canonical_idx, source_idx in enumerate(match):
            source_atom = mol.GetAtomWithIdx(source_idx)
            anchor_count = len(_read_atom_port_cut_ids(source_atom))
            anchor_positions.extend([canonical_idx] * anchor_count)
        return tuple(anchor_positions), tuple(match)

    selected_match = min(matches, key=match_key)

    # 允许规范化过程中仅省略显式氢；任何未映射的非氢原子仍视为结构损坏。
    matched_source_indices = set(selected_match)
    unmatched_source_indices = (
        set(range(mol.GetNumAtoms())) - matched_source_indices
    )
    if any(
        mol.GetAtomWithIdx(index).GetAtomicNum() != 1
        for index in unmatched_source_indices
    ):
        raise PortValidationError("规范片段映射遗漏了非氢原子")

    canonical_mol = Chem.Mol(canonical_mol)

    if mol.GetNumConformers() > 0:
        source_conf = mol.GetConformer(0)
        canonical_conf = Chem.Conformer(canonical_mol.GetNumAtoms())
        for canonical_idx, source_idx in enumerate(selected_match):
            canonical_conf.SetAtomPosition(
                canonical_idx, source_conf.GetAtomPosition(source_idx)
            )
        canonical_mol.AddConformer(canonical_conf, assignId=True)

    ports = []
    for canonical_idx, source_idx in enumerate(selected_match):
        source_atom = mol.GetAtomWithIdx(source_idx)
        target_atom = canonical_mol.GetAtomWithIdx(canonical_idx)
        _copy_atom_trace_properties(source_atom, target_atom)
        for cut_id in _read_atom_port_cut_ids(source_atom):
            ports.append({"cut_id": cut_id, "anchor_index": canonical_idx})

    ports.sort(key=lambda item: (item["anchor_index"], item["cut_id"]))
    try:
        canonical_smiles = Chem.MolToSmiles(
            canonical_mol,
            canonical=True,
            isomericSmiles=True,
        )
    except Exception as exc:
        raise PortValidationError(
            f"最终规范片段无法生成稳定 SMILES: {exc}"
        ) from exc
    if not canonical_smiles:
        raise PortValidationError("最终规范片段生成了空 SMILES")
    return canonical_mol, canonical_smiles, ports


def build_fragment_connections(shards):
    """
    将 ligand shard 的内部端口记录转换为直接的节点、原子与距离信息。

    返回 ``(connection_edge_index, connection_atom_ids, connection_distance)``：
    ``connection_edge_index`` 形状为 ``[2, E]``，每列是一对节点编号；
    ``connection_atom_ids`` 形状为 ``[E, 2]``，每行是对应节点内的规范原子编号；
    ``connection_distance`` 长度为 ``E``，记录对应原子对的三维距离。
    """
    ordered_shards = sorted(shards, key=lambda shard: shard["Shard ID"])
    shard_ids = [shard["Shard ID"] for shard in ordered_shards]
    if shard_ids != list(range(len(ordered_shards))):
        raise PortValidationError("Shard ID 必须从 0 开始连续编号")

    endpoints_by_cut = {}

    for shard in ordered_shards:
        shard_id = shard["Shard ID"]
        ports = sorted(
            shard.get("ports", []),
            key=lambda item: (item["anchor_index"], item["cut_id"]),
        )
        if shard.get("origin") != "ligand" and ports:
            raise PortValidationError(
                f"非 ligand Shard {shard_id} 不应包含重建端口"
            )

        seen_local_records = set()
        shard_mol = shard.get("mol") or shard.get("clean_mol")
        if ports and shard_mol is None:
            raise PortValidationError(
                f"Shard {shard_id} 缺少用于提取连接原子的分子对象"
            )
        if ports and shard_mol.GetNumConformers() == 0:
            raise PortValidationError(
                f"Shard {shard_id} 的连接原子缺少三维构象"
            )
        conformer = shard_mol.GetConformer(0) if ports else None

        for port in ports:
            cut_id = port.get("cut_id")
            anchor_index = port.get("anchor_index")
            if not isinstance(cut_id, str) or not cut_id:
                raise PortValidationError(f"Shard {shard_id} 存在无效 cut_id")
            if not isinstance(anchor_index, int) or anchor_index < 0:
                raise PortValidationError(f"Shard {shard_id} 存在无效锚点编号")
            if anchor_index >= shard_mol.GetNumAtoms():
                raise PortValidationError(
                    f"Shard {shard_id} 的端口锚点 {anchor_index} 超出片段原子范围"
                )
            local_record = (cut_id, anchor_index)
            if local_record in seen_local_records:
                raise PortValidationError(
                    f"Shard {shard_id} 重复记录端口 {cut_id}"
                )
            seen_local_records.add(local_record)

            position = conformer.GetAtomPosition(anchor_index)
            coordinates = np.asarray(
                [position.x, position.y, position.z], dtype=np.float64
            )
            if not np.all(np.isfinite(coordinates)):
                raise PortValidationError(
                    f"Shard {shard_id} 的连接原子 {anchor_index} 坐标不是有限值"
                )
            endpoints_by_cut.setdefault(cut_id, []).append({
                "node_id": shard_id,
                "atom_id": anchor_index,
                "coordinates": coordinates,
            })

    connection_records = []
    connection_counts_by_node_pair = {}
    seen_connection_atom_pairs = set()
    for cut_id, endpoints in sorted(endpoints_by_cut.items()):
        if len(endpoints) != 2:
            raise PortValidationError(
                f"切断 {cut_id} 应有两个端口，实际得到 {len(endpoints)} 个"
            )
        first, second = sorted(
            endpoints,
            key=lambda endpoint: (endpoint["node_id"], endpoint["atom_id"]),
        )
        if first["node_id"] == second["node_id"]:
            raise ConnectionTopologyError(
                f"切断 {cut_id} 在节点 {first['node_id']} 内形成自环"
            )

        node_pair = (first["node_id"], second["node_id"])
        connection_atom_pair = (
            node_pair,
            (first["atom_id"], second["atom_id"]),
        )
        if connection_atom_pair in seen_connection_atom_pairs:
            raise ConnectionTopologyError(
                f"切断 {cut_id} 重复形成节点对 {node_pair} 的同一原子对"
            )
        seen_connection_atom_pairs.add(connection_atom_pair)
        pair_count = connection_counts_by_node_pair.get(node_pair, 0) + 1
        if pair_count > 2:
            raise ConnectionTopologyError(
                f"节点对 {node_pair} 包含超过两个成键原子对"
            )
        connection_counts_by_node_pair[node_pair] = pair_count

        distance = float(np.linalg.norm(
            first["coordinates"] - second["coordinates"]
        ))
        if not math.isfinite(distance):
            raise PortValidationError(
                f"切断 {cut_id} 的原子对距离不是有限值"
            )
        connection_records.append({
            "nodes": node_pair,
            "atoms": (first["atom_id"], second["atom_id"]),
            "distance": distance,
        })

    connection_edge_index = [
        [record["nodes"][0] for record in connection_records],
        [record["nodes"][1] for record in connection_records],
    ]
    connection_atom_ids = [
        list(record["atoms"]) for record in connection_records
    ]
    connection_distance = [
        record["distance"] for record in connection_records
    ]
    return connection_edge_index, connection_atom_ids, connection_distance


def build_legacy_covalent_edges(processed_groups):
    """
    保留旧版片段级共价边构建能力，供二维分子任务复用。

    当前 SE3TD 三维消息传递图不应把本函数结果加入 ``edge_index``。
    """
    all_shards = [
        shard for group in processed_groups for shard in group.get("shards", [])
    ]
    shard_map = {shard["Shard ID"]: shard for shard in all_shards}
    edge_attributes = {}

    for group in processed_groups:
        original_fragment = group.get("original_fragment")
        if original_fragment is None:
            continue
        atom_to_shard = {}
        for shard in group.get("shards", []):
            shard_mol = shard.get("mol")
            if shard_mol is None:
                continue
            for atom in shard_mol.GetAtoms():
                if atom.GetAtomicNum() != 0 and atom.HasProp("_original_index"):
                    atom_to_shard[atom.GetIntProp("_original_index")] = shard["Shard ID"]

        main_mol = Chem.RemoveHs(original_fragment)
        for bond in main_mol.GetBonds():
            first = atom_to_shard.get(bond.GetBeginAtomIdx())
            second = atom_to_shard.get(bond.GetEndAtomIdx())
            if first is None or second is None or first == second:
                continue
            edge = tuple(sorted((first, second)))
            if edge not in edge_attributes:
                edge_attributes[edge] = (
                    0 if shard_map[first].get("origin") == "ligand" else 1
                )
    return edge_attributes

# ----------------------------------------
#  分子预处理与属性计算
# ----------------------------------------
def prepare_molecule_for_cutting(mol, source_fragment_id=None, origin_type=None):
    """过滤无关离子、分离金属，并严格准备每个有机组分。"""
    if mol is None or mol.GetNumAtoms() == 0:
        return {
            "organic_components": [],
            "metal_fragments": [],
            "removed_fragment_count": 0,
            "reconstruction_failed_count": 0,
        }

    for atom in mol.GetAtoms():
        if not atom.HasProp("_original_index"):
            atom.SetIntProp("_original_index", atom.GetIdx())

    if source_fragment_id is not None:
        source_fragment_id = str(source_fragment_id)
        for atom in mol.GetAtoms():
            atom.SetProp(SOURCE_FRAGMENT_PROP, source_fragment_id)

    retained_fragments, removed_fragment_count = rule1_filter_ions_and_complexing_agents(
        mol, origin_type=origin_type
    )
    organic_fragments, metal_fragments = rule2_separate_metals(retained_fragments)

    result_data = {
        "organic_components": [],
        "metal_fragments": metal_fragments,
        "removed_fragment_count": removed_fragment_count,
        "reconstruction_failed_count": 0,
    }

    for organic_fragment in organic_fragments:
        component_data = {
            "main_mol_for_matching": None,
            "original_eccentricities": [],
            "original_logp_contribs": [],
            "original_tpsa_contribs": [],
            "original_charges": [],
            "bond_info": {},
        }
        try:
            main_mol_for_matching = Chem.RemoveHs(organic_fragment)
            reconstructed_mol = reconstruct_mol_with_coords(main_mol_for_matching)
            if reconstructed_mol is None:
                result_data["reconstruction_failed_count"] += 1
                continue
            main_mol_for_matching = reconstructed_mol

            for atom in main_mol_for_matching.GetAtoms():
                atom.SetIntProp("_internal_index", atom.GetIdx())
                if not atom.HasProp("_original_index"):
                    atom.SetIntProp("_original_index", atom.GetIdx())

            component_data["main_mol_for_matching"] = main_mol_for_matching

            try:
                distance_matrix = GetDistanceMatrix(main_mol_for_matching)
                component_data["original_eccentricities"] = distance_matrix.max(axis=1)
            except Exception:
                pass

            try:
                logp_contribs = rdMolDescriptors._CalcCrippenContribs(main_mol_for_matching)
                component_data["original_logp_contribs"] = [x[0] for x in logp_contribs]
            except Exception:
                pass

            try:
                tpsa_contribs = rdMolDescriptors._CalcTPSAContribs(main_mol_for_matching)
                component_data["original_tpsa_contribs"] = list(tpsa_contribs)
            except Exception:
                pass

            try:
                AllChem.ComputeGasteigerCharges(main_mol_for_matching)
                charges_raw = [
                    atom.GetProp("_GasteigerCharge")
                    for atom in main_mol_for_matching.GetAtoms()
                ]
                component_data["original_charges"] = [float(value) for value in charges_raw]
            except Exception:
                pass

            try:
                if len(component_data["original_eccentricities"]) > 0:
                    component_data["bond_info"] = get_ring_bond_info(
                        main_mol_for_matching,
                        component_data["original_eccentricities"],
                    )
            except Exception:
                pass

            result_data["organic_components"].append(component_data)
        except Exception:
            result_data["reconstruction_failed_count"] += 1

    return result_data
    
def calculate_minimum_enclosing_ball(points: np.ndarray):
    """
    使用 Badoiu 和 Clarkson 的迭代算法计算点云的最小包围球。

    Args:
        points (np.ndarray): 一个形状为 (N, 3) 的numpy数组，代表N个点的3D坐标。

    Returns:
        tuple[np.ndarray, float]: 返回一个元组，包含：
            - center (np.ndarray): 球心的3D坐标。
            - radius (float): 球的半径。
    """
    # --- 边缘情况处理 ---
    num_points = points.shape[0]
    if num_points == 0:
        return np.zeros(3), 0.0
    if num_points == 1:
        return points[0], 0.0

    # --- 使用迭代算法 ---
    # 1. 初始化: 使用平均中心作为一个好的起点
    center = np.mean(points, axis=0)
    
    # 2. 迭代更新中心点
    for i in range(1, 151): # 150 次迭代对于典型的口袋来说绰绰有余
        # 找到当前距离中心最远的点
        dists_sq = np.sum((points - center)**2, axis=1)
        furthest_idx = np.argmax(dists_sq)
        p_i = points[furthest_idx]
        
        # 将中心点向最远点移动一小步
        step = 1.0 / (i + 1)
        center = (1.0 - step) * center + step * p_i

    # 3. 确定最终半径
    #    在找到最终的中心后，半径就是该中心到所有点的最大距离
    final_dists = np.linalg.norm(points - center, axis=1)
    radius = np.max(final_dists)
    
    return center, radius


# --- 数据汇总与描述符计算函数 (2D/3D 兼容版) ---

METAL_ATOMIC_NUMBERS = {
    3, 4, 11, 12, 13, 19, 20,
    *range(21, 32), *range(37, 51), *range(55, 84),
    84, 87, 88, *range(89, 104),
}

METAL_SYMBOLS = {
    Chem.GetPeriodicTable().GetElementSymbol(atomic_num).upper()
    for atomic_num in METAL_ATOMIC_NUMBERS
}

METAL_DEFAULT_CHARGES = {
    # +1
    'LI': 1.0, 'NA': 1.0, 'K': 1.0, 'RB': 1.0, 'CS': 1.0,
    'FR': 1.0, 'AG': 1.0, 'TL': 1.0,
    # +1.5 / +2 / +2.5
    'CU': 1.5,
    'BE': 2.0, 'MG': 2.0, 'CA': 2.0, 'SR': 2.0, 'BA': 2.0,
    'RA': 2.0, 'ZN': 2.0, 'CD': 2.0, 'HG': 2.0, 'PD': 2.0,
    'PT': 2.0, 'SN': 2.0, 'PB': 2.0, 'PO': 2.0, 'NO': 2.0,
    'FE': 2.5, 'CO': 2.5, 'NI': 2.5,
    # +3
    'AL': 3.0, 'SC': 3.0, 'CR': 3.0, 'MN': 3.0, 'GA': 3.0,
    'Y': 3.0, 'LA': 3.0, 'CE': 3.0, 'PR': 3.0, 'ND': 3.0,
    'PM': 3.0, 'SM': 3.0, 'EU': 3.0, 'GD': 3.0, 'TB': 3.0,
    'DY': 3.0, 'HO': 3.0, 'ER': 3.0, 'TM': 3.0, 'YB': 3.0,
    'LU': 3.0, 'RU': 3.0, 'RH': 3.0, 'IR': 3.0, 'AU': 3.0,
    'IN': 3.0, 'BI': 3.0, 'AC': 3.0, 'AM': 3.0, 'CM': 3.0,
    'BK': 3.0, 'CF': 3.0, 'ES': 3.0, 'FM': 3.0, 'MD': 3.0,
    'LR': 3.0,
    # +4 / +5 / +6 / +7
    'TI': 4.0, 'V': 4.0, 'ZR': 4.0, 'HF': 4.0, 'RE': 4.0,
    'OS': 4.0, 'TH': 4.0, 'PU': 4.0,
    'NB': 5.0, 'MO': 5.0, 'TA': 5.0, 'PA': 5.0, 'NP': 5.0,
    'W': 6.0, 'U': 6.0,
    'TC': 7.0,
}

_missing_metal_charge_symbols = METAL_SYMBOLS - set(METAL_DEFAULT_CHARGES)
_extra_metal_charge_symbols = set(METAL_DEFAULT_CHARGES) - METAL_SYMBOLS
if _missing_metal_charge_symbols or _extra_metal_charge_symbols:
    raise RuntimeError(
        "金属经验电荷表与金属元素集合不一致: "
        f"missing={sorted(_missing_metal_charge_symbols)}, "
        f"extra={sorted(_extra_metal_charge_symbols)}"
    )

RAW_FRAGMENT_FINGERPRINT_RADIUS = 2
RAW_FRAGMENT_FINGERPRINT_BITS = 2048
RAW_FRAGMENT_INCLUDE_CHIRALITY = True
RAW_FRAGMENT_ELEMENT_DIM = 118
RAW_FRAGMENT_SPECIAL_TOKENS = {"<UNK>", "<GLOBAL>"}
RAW_FRAGMENT_CHARGE_POLICY = "metal_empirical_through_lr_else_formal"
RAW_FRAGMENT_HYDROGEN_POLICY = (
    "dummy_to_h_then_rdkit_remove_hs"
)

_RAW_ECFP_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(
    radius=RAW_FRAGMENT_FINGERPRINT_RADIUS,
    fpSize=RAW_FRAGMENT_FINGERPRINT_BITS,
    includeChirality=RAW_FRAGMENT_INCLUDE_CHIRALITY,
)
_RAW_FCFP_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(
    radius=RAW_FRAGMENT_FINGERPRINT_RADIUS,
    fpSize=RAW_FRAGMENT_FINGERPRINT_BITS,
    includeChirality=RAW_FRAGMENT_INCLUDE_CHIRALITY,
    atomInvariantsGenerator=(
        rdFingerprintGenerator.GetMorganFeatureAtomInvGen()
    ),
)


def is_metal_atom(atom):
    return atom.GetAtomicNum() in METAL_ATOMIC_NUMBERS


def get_atom_feature_charge(atom):
    """
    返回用于节点属性和粗语料的统一原子特征电荷。

    金属原子始终使用经验电荷表；非金属原子使用 RDKit 形式电荷。
    对经验表未覆盖的金属直接报错，避免静默写入不一致的回退值。
    """
    if is_metal_atom(atom):
        symbol = atom.GetSymbol().upper()
        if symbol not in METAL_DEFAULT_CHARGES:
            raise ValueError(f"金属元素 {symbol} 缺少经验电荷配置")
        return float(METAL_DEFAULT_CHARGES[symbol])
    return float(atom.GetFormalCharge())


def is_metal_only_fragment(mol_obj):
    real_atoms = [atom for atom in mol_obj.GetAtoms() if atom.GetAtomicNum() != 0]
    return bool(real_atoms) and all(is_metal_atom(atom) for atom in real_atoms)


def _fragment_fingerprint_to_array(fingerprint):
    values = np.zeros((RAW_FRAGMENT_FINGERPRINT_BITS,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fingerprint, values)
    return values


def calculate_fragment_vocab_features(smiles):
    """
    根据孤立片段的规范 SMILES 计算离线词汇训练所需的固定粗特征。

    返回 ECFP、FCFP、HAC、总环数、统一特征电荷、118维重元素计数和
    纯金属标记。该函数不读取片段所在母体的电荷、3D构象或理化环境。
    """
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError("SMILES 必须是非空字符串")
    smiles = smiles.strip()
    if smiles in RAW_FRAGMENT_SPECIAL_TOKENS:
        raise ValueError(f"特殊标记不能进入粗语料: {smiles}")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError(f"无法解析片段 SMILES: {smiles}")

    canonical_smiles = Chem.MolToSmiles(
        mol, canonical=True, isomericSmiles=True
    )
    canonical_mol = Chem.MolFromSmiles(canonical_smiles)
    if canonical_mol is None or canonical_mol.GetNumAtoms() == 0:
        raise ValueError(f"规范 SMILES 无法重建: {smiles}")

    ecfp = _RAW_ECFP_GENERATOR.GetFingerprint(canonical_mol)
    fcfp = _RAW_FCFP_GENERATOR.GetFingerprint(canonical_mol)

    element_counts = np.zeros((RAW_FRAGMENT_ELEMENT_DIM,), dtype=np.int16)
    for atom in canonical_mol.GetAtoms():
        atomic_number = atom.GetAtomicNum()
        if atomic_number <= 1:
            continue
        if atomic_number > RAW_FRAGMENT_ELEMENT_DIM:
            raise ValueError(
                f"原子序数 {atomic_number} 超出元素组成向量范围"
            )
        element_counts[atomic_number - 1] += 1

    feature_charge = sum(
        get_atom_feature_charge(atom) for atom in canonical_mol.GetAtoms()
    )
    return {
        "smiles": canonical_smiles,
        "ecfp": _fragment_fingerprint_to_array(ecfp),
        "fcfp": _fragment_fingerprint_to_array(fcfp),
        "hac": np.int16(calculate_heavy_atom_count(canonical_mol)),
        "ring_count": np.int16(calculate_ring_count(canonical_mol)),
        # 保留既有字段名以降低下游改动范围；v2 中该字段允许经验小数电荷。
        "formal_charge": np.float32(feature_charge),
        "element_counts": element_counts,
        "is_metal": np.uint8(is_metal_only_fragment(canonical_mol)),
    }


def _raw_fragment_corpus_checksum(corpus_arrays):
    """对粗语料全部字段计算与NPZ压缩方式无关的稳定校验值。"""
    digest = hashlib.sha256()
    field_order = (
        "smiles", "ecfp", "fcfp", "hac", "ring_count",
        "formal_charge", "element_counts", "is_metal",
    )
    for key in field_order:
        values = np.asarray(corpus_arrays[key])
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


def save_raw_fragment_corpus(
    smiles_collection,
    npz_path,
    metadata_path,
    preserve_resolver_entries=True,
    preserve_input_order=False,
):
    """
    将一组片段 SMILES 规范化、去重并保存为压缩粗语料。

    无效 SMILES 不写入 NPZ，而是记录到 JSON 元数据并通过返回值报告。
    检测到片段 Resolver 扩展标记时，默认保留既有表中的额外词汇；
    基础词汇若发生变化，标准词表仍会通过顺序或哈希检查要求重新训练。
    ``preserve_input_order=True`` 时保留首次出现的规范 SMILES 顺序，供
    需要保持既有词索引并只在尾部追加新词的流程使用。默认行为不变。
    写入使用临时文件和原子替换，避免中断后留下半成品。
    """
    npz_path = os.path.abspath(os.fspath(npz_path))
    metadata_path = os.path.abspath(os.fspath(metadata_path))
    os.makedirs(os.path.dirname(npz_path), exist_ok=True)
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)

    preserved_smiles = []
    preserved_metadata_fields = {}
    if (
        preserve_resolver_entries
        and os.path.isfile(npz_path)
        and os.path.isfile(metadata_path)
    ):
        try:
            with open(
                metadata_path, "r", encoding="utf-8"
            ) as file_obj:
                existing_metadata_probe = json.load(file_obj)
        except (OSError, ValueError):
            existing_metadata_probe = {}
        resolver_count = existing_metadata_probe.get(
            "fragment_embedding_resolver_added_smiles_count", 0
        )
        if (
            existing_metadata_probe.get(
                "fragment_embedding_checkpoint_sha256"
            )
            and isinstance(resolver_count, int)
            and resolver_count > 0
        ):
            existing_corpus, existing_metadata = (
                load_raw_fragment_corpus(npz_path, metadata_path)
            )
            preserved_smiles = (
                existing_corpus["smiles"].astype(str).tolist()
            )
            for key in (
                "fragment_embedding_training_smiles_count",
                "fragment_embedding_resolver_added_smiles_count",
                "fragment_embedding_checkpoint_sha256",
                "last_fragment_embedding_extension_utc",
            ):
                if key in existing_metadata:
                    preserved_metadata_fields[key] = (
                        existing_metadata[key]
                    )

    if preserve_input_order:
        raw_smiles_values = list(smiles_collection)
    else:
        raw_smiles_values = sorted(
            set(smiles_collection),
            key=lambda value: str(value),
        )

    records_by_smiles = {}
    invalid_entries = []
    for raw_smiles in raw_smiles_values:
        if raw_smiles in RAW_FRAGMENT_SPECIAL_TOKENS:
            continue
        try:
            record = calculate_fragment_vocab_features(raw_smiles)
            records_by_smiles.setdefault(record["smiles"], record)
        except Exception as exc:
            invalid_entries.append({
                "smiles": str(raw_smiles),
                "error": str(exc),
            })

    base_smiles = (
        list(records_by_smiles)
        if preserve_input_order
        else sorted(records_by_smiles)
    )
    if not base_smiles:
        raise ValueError("粗语料中没有可保存的有效片段 SMILES")

    preserved_records = []
    seen_smiles = set(base_smiles)
    for existing_smiles in preserved_smiles:
        if existing_smiles in seen_smiles:
            continue
        try:
            record = calculate_fragment_vocab_features(existing_smiles)
        except Exception as exc:
            raise ValueError(
                "无法保留 Resolver 已追加的片段 "
                f"{existing_smiles}: {exc}"
            ) from exc
        canonical_smiles = record["smiles"]
        if canonical_smiles != existing_smiles:
            raise ValueError(
                "Resolver 已追加片段的规范 SMILES 发生变化: "
                f"{existing_smiles} -> {canonical_smiles}"
            )
        preserved_records.append(record)
        seen_smiles.add(existing_smiles)

    sorted_smiles = base_smiles + [
        record["smiles"] for record in preserved_records
    ]
    records = [
        records_by_smiles[smiles] for smiles in base_smiles
    ] + preserved_records
    corpus_arrays = {
        "smiles": np.asarray(sorted_smiles, dtype=np.str_),
        "ecfp": np.stack([record["ecfp"] for record in records]).astype(np.uint8),
        "fcfp": np.stack([record["fcfp"] for record in records]).astype(np.uint8),
        "hac": np.asarray([record["hac"] for record in records], dtype=np.int16),
        "ring_count": np.asarray(
            [record["ring_count"] for record in records], dtype=np.int16
        ),
        "formal_charge": np.asarray(
            [record["formal_charge"] for record in records], dtype=np.float32
        ),
        "element_counts": np.stack(
            [record["element_counts"] for record in records]
        ).astype(np.int16),
        "is_metal": np.asarray(
            [record["is_metal"] for record in records], dtype=np.uint8
        ),
    }

    metadata = {
        "fingerprint": {
            "radius": RAW_FRAGMENT_FINGERPRINT_RADIUS,
            "n_bits": RAW_FRAGMENT_FINGERPRINT_BITS,
            "include_chirality": RAW_FRAGMENT_INCLUDE_CHIRALITY,
            "ecfp_use_features": False,
            "fcfp_use_features": True,
        },
        "intrinsic_features": {
            "scalar_order": ["hac", "ring_count", "formal_charge"],
            "element_dimension": RAW_FRAGMENT_ELEMENT_DIM,
            "metal_definition": "all_non_dummy_atoms_are_metals",
            "charge_policy": RAW_FRAGMENT_CHARGE_POLICY,
            "hydrogen_policy": RAW_FRAGMENT_HYDROGEN_POLICY,
        },
        "num_smiles": len(sorted_smiles),
        "base_smiles_count": len(base_smiles),
        "preserved_resolver_smiles_count": len(preserved_records),
        "corpus_sha256": _raw_fragment_corpus_checksum(corpus_arrays),
        "array_shapes": {
            key: list(value.shape) for key, value in corpus_arrays.items()
        },
        "invalid_smiles_count": len(invalid_entries),
        "invalid_entries": invalid_entries,
    }
    metadata.update(preserved_metadata_fields)

    temp_npz_path = f"{npz_path}.tmp.npz"
    temp_metadata_path = f"{metadata_path}.tmp"
    try:
        np.savez_compressed(temp_npz_path, **corpus_arrays)
        with open(temp_metadata_path, "w", encoding="utf-8") as file_obj:
            json.dump(metadata, file_obj, ensure_ascii=False, indent=2)
        os.replace(temp_npz_path, npz_path)
        os.replace(temp_metadata_path, metadata_path)
    finally:
        for temp_path in (temp_npz_path, temp_metadata_path):
            if os.path.exists(temp_path):
                os.remove(temp_path)

    return metadata


def load_raw_fragment_corpus(npz_path, metadata_path):
    """加载并严格校验由 :func:`save_raw_fragment_corpus` 生成的粗语料。"""
    npz_path = os.path.abspath(os.fspath(npz_path))
    metadata_path = os.path.abspath(os.fspath(metadata_path))
    with open(metadata_path, "r", encoding="utf-8") as file_obj:
        metadata = json.load(file_obj)

    fingerprint_meta = metadata.get("fingerprint", {})
    expected_fingerprint_meta = {
        "radius": RAW_FRAGMENT_FINGERPRINT_RADIUS,
        "n_bits": RAW_FRAGMENT_FINGERPRINT_BITS,
        "include_chirality": RAW_FRAGMENT_INCLUDE_CHIRALITY,
        "ecfp_use_features": False,
        "fcfp_use_features": True,
    }
    if fingerprint_meta != expected_fingerprint_meta:
        raise ValueError("粗语料指纹参数与当前代码不一致")
    intrinsic_meta = metadata.get("intrinsic_features", {})
    expected_intrinsic_meta = {
        "scalar_order": ["hac", "ring_count", "formal_charge"],
        "element_dimension": RAW_FRAGMENT_ELEMENT_DIM,
        "metal_definition": "all_non_dummy_atoms_are_metals",
    }
    if not isinstance(intrinsic_meta, dict) or any(
        intrinsic_meta.get(key) != value
        for key, value in expected_intrinsic_meta.items()
    ):
        raise ValueError("粗语料固有属性结构与当前代码不一致")

    required_arrays = {
        "smiles", "ecfp", "fcfp", "hac", "ring_count",
        "formal_charge", "element_counts", "is_metal",
    }
    with np.load(npz_path, allow_pickle=False) as loaded:
        missing = required_arrays - set(loaded.files)
        if missing:
            raise ValueError(f"粗语料 NPZ 缺少字段: {sorted(missing)}")
        corpus = {key: np.array(loaded[key], copy=True) for key in required_arrays}

    corpus["smiles"] = corpus["smiles"].astype(str)
    num_smiles = len(corpus["smiles"])
    expected_shapes = {
        "ecfp": (num_smiles, RAW_FRAGMENT_FINGERPRINT_BITS),
        "fcfp": (num_smiles, RAW_FRAGMENT_FINGERPRINT_BITS),
        "hac": (num_smiles,),
        "ring_count": (num_smiles,),
        "formal_charge": (num_smiles,),
        "element_counts": (num_smiles, RAW_FRAGMENT_ELEMENT_DIM),
        "is_metal": (num_smiles,),
    }
    for key, expected_shape in expected_shapes.items():
        if tuple(corpus[key].shape) != expected_shape:
            raise ValueError(
                f"粗语料字段 {key} 形状错误: "
                f"{tuple(corpus[key].shape)} != {expected_shape}"
            )
    metadata_shapes = metadata.get("array_shapes")
    actual_shapes = {
        key: list(value.shape) for key, value in corpus.items()
    }
    if metadata_shapes != actual_shapes:
        raise ValueError("粗语料元数据中的字段形状与NPZ不一致")

    expected_dtypes = {
        "ecfp": np.dtype(np.uint8),
        "fcfp": np.dtype(np.uint8),
        "hac": np.dtype(np.int16),
        "ring_count": np.dtype(np.int16),
        "formal_charge": np.dtype(np.float32),
        "element_counts": np.dtype(np.int16),
        "is_metal": np.dtype(np.uint8),
    }
    for key, expected_dtype in expected_dtypes.items():
        if corpus[key].dtype != expected_dtype:
            raise ValueError(
                f"粗语料字段 {key} 类型错误: "
                f"{corpus[key].dtype} != {expected_dtype}"
            )

    smiles_list = corpus["smiles"].tolist()
    if len(set(smiles_list)) != num_smiles:
        raise ValueError("粗语料包含重复的规范 SMILES")
    if RAW_FRAGMENT_SPECIAL_TOKENS.intersection(smiles_list):
        raise ValueError("粗语料中不允许出现特殊标记")
    if metadata.get("num_smiles") != num_smiles:
        raise ValueError("粗语料元数据中的 SMILES 数量不一致")
    if metadata.get("corpus_sha256") != _raw_fragment_corpus_checksum(corpus):
        raise ValueError("粗语料内容校验值不一致")

    return corpus, metadata


def merge_raw_fragment_corpus(
    smiles_collection,
    npz_path,
    metadata_path,
):
    """
    将新片段规范化后追加到可选的既有粗语料末尾。

    NPZ 与元数据必须同时存在或同时不存在。既有词保持原顺序；新词使用
    保留立体构型的规范 SMILES 去重后按字典序追加。若旧语料含有非规范
    写法或规范化后重复项，则保留其首次出现位置并在本次保存时修正。

    返回 ``(corpus, metadata, merge_stats)``。当既无旧语料也没有新词时，
    前两项为 ``None``，且不会创建空文件。
    """
    npz_path = os.path.abspath(os.fspath(npz_path))
    metadata_path = os.path.abspath(os.fspath(metadata_path))
    npz_exists = os.path.isfile(npz_path)
    metadata_exists = os.path.isfile(metadata_path)
    if npz_exists != metadata_exists:
        missing_path = metadata_path if npz_exists else npz_path
        raise FileNotFoundError(
            "片段粗语料 NPZ 与元数据必须同时存在或同时不存在；"
            f"缺少: {missing_path}"
        )

    existing_corpus = None
    existing_metadata = None
    stored_smiles = []
    if npz_exists:
        existing_corpus, existing_metadata = load_raw_fragment_corpus(
            npz_path,
            metadata_path,
        )
        stored_smiles = existing_corpus["smiles"].astype(str).tolist()

    normalized_existing_smiles = []
    existing_smiles_set = set()
    normalized_existing_count = 0
    for stored_smiles_value in stored_smiles:
        record = calculate_fragment_vocab_features(stored_smiles_value)
        canonical_smiles = record["smiles"]
        entry_needs_normalization = (
            canonical_smiles != stored_smiles_value
            or canonical_smiles in existing_smiles_set
        )
        if entry_needs_normalization:
            normalized_existing_count += 1
        if canonical_smiles in existing_smiles_set:
            continue
        normalized_existing_smiles.append(canonical_smiles)
        existing_smiles_set.add(canonical_smiles)

    candidate_records = {}
    for raw_smiles in sorted(
        set(smiles_collection),
        key=lambda value: str(value),
    ):
        record = calculate_fragment_vocab_features(raw_smiles)
        candidate_records.setdefault(record["smiles"], record)

    new_smiles = sorted(
        canonical_smiles
        for canonical_smiles in candidate_records
        if canonical_smiles not in existing_smiles_set
    )
    final_smiles = normalized_existing_smiles + new_smiles
    existing_needs_normalization = (
        normalized_existing_smiles != stored_smiles
    )
    corpus_rewritten = bool(
        final_smiles
        and (
            not npz_exists
            or existing_needs_normalization
            or new_smiles
        )
    )

    merge_stats = {
        "existing_vocab_size": len(stored_smiles),
        "existing_vocab_normalized": normalized_existing_count,
        "new_vocab_candidates": len(new_smiles),
        "new_vocab_added": len(new_smiles),
        "shared_vocab_size_after": len(final_smiles),
        "corpus_created": bool(not npz_exists and corpus_rewritten),
        "corpus_rewritten": corpus_rewritten,
    }

    if not final_smiles:
        return None, None, merge_stats

    if corpus_rewritten:
        saved_metadata = save_raw_fragment_corpus(
            final_smiles,
            npz_path,
            metadata_path,
            preserve_resolver_entries=False,
            preserve_input_order=True,
        )
        corpus, metadata = load_raw_fragment_corpus(
            npz_path,
            metadata_path,
        )
        if metadata["corpus_sha256"] != saved_metadata["corpus_sha256"]:
            raise RuntimeError("片段粗语料保存后的校验值发生变化")
    else:
        corpus = existing_corpus
        metadata = existing_metadata

    verified_smiles = corpus["smiles"].astype(str).tolist()
    if verified_smiles != final_smiles:
        raise RuntimeError("片段粗语料词序与预期追加顺序不一致")
    if metadata.get("invalid_smiles_count", 0) != 0:
        raise RuntimeError("片段粗语料包含无效 SMILES")
    if metadata.get("num_smiles") != len(final_smiles):
        raise RuntimeError("片段粗语料词数与预期不一致")

    return corpus, metadata, merge_stats


RAW_ATOM_FEATURE_COLUMNS = (
    "atomic_number",
    "formal_charge",
    "degree",
    "total_valence",
    "total_hs",
    "hybridization",
    "radical_electrons",
    "chiral_tag",
    "is_aromatic",
    "is_in_ring",
)
RAW_BOND_FEATURE_COLUMNS = (
    "bond_type",
    "is_conjugated",
    "is_aromatic",
    "is_in_ring",
    "stereo",
)
RAW_ATOM_HYBRIDIZATION_CATEGORIES = (
    "UNSPECIFIED",
    "S",
    "SP",
    "SP2",
    "SP3",
    "SP2D",
    "SP3D",
    "SP3D2",
    "OTHER",
)
RAW_ATOM_CHIRAL_CATEGORIES = (
    "CHI_UNSPECIFIED",
    "CHI_TETRAHEDRAL_CW",
    "CHI_TETRAHEDRAL_CCW",
    "CHI_TETRAHEDRAL",
    "CHI_ALLENE",
    "CHI_SQUAREPLANAR",
    "CHI_TRIGONALBIPYRAMIDAL",
    "CHI_OCTAHEDRAL",
    "CHI_OTHER",
)
RAW_BOND_TYPE_CATEGORIES = (
    "UNSPECIFIED",
    "SINGLE",
    "DOUBLE",
    "TRIPLE",
    "AROMATIC",
    "OTHER",
)
RAW_BOND_STEREO_CATEGORIES = (
    "STEREONONE",
    "STEREOANY",
    "STEREOZ",
    "STEREOE",
    "STEREOCIS",
    "STEREOTRANS",
    "STEREOATROPCW",
    "STEREOATROPCCW",
    "OTHER",
)

_RAW_ATOM_HYBRIDIZATION_TO_ID = {
    name: index
    for index, name in enumerate(RAW_ATOM_HYBRIDIZATION_CATEGORIES)
}
_RAW_ATOM_CHIRAL_TO_ID = {
    name: index for index, name in enumerate(RAW_ATOM_CHIRAL_CATEGORIES)
}
_RAW_BOND_TYPE_TO_ID = {
    name: index for index, name in enumerate(RAW_BOND_TYPE_CATEGORIES)
}
_RAW_BOND_STEREO_TO_ID = {
    name: index for index, name in enumerate(RAW_BOND_STEREO_CATEGORIES)
}

_RAW_ATOM_CORPUS_FIELDS = (
    "smiles",
    "atom_offsets",
    "canonical_atom_id",
    "atom_features",
    "bond_offsets",
    "bond_atom_ids",
    "bond_features",
)


def _stable_enum_id(value, mapping, fallback_name):
    """把 RDKit 枚举转换为代码中固定的类别编号。"""
    label = str(value)
    return mapping.get(label, mapping[fallback_name])


def canonicalize_fragment_smiles(smiles):
    """返回与连接原子编号使用同一规范顺序的 SMILES 和 RDKit Mol。"""
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError("SMILES 必须是非空字符串")
    smiles = smiles.strip()
    if smiles in RAW_FRAGMENT_SPECIAL_TOKENS:
        raise ValueError(f"特殊标记不能进入原子粗语料: {smiles}")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError(f"无法解析片段 SMILES: {smiles}")
    canonical_smiles = Chem.MolToSmiles(
        mol, canonical=True, isomericSmiles=True
    )
    canonical_mol = Chem.MolFromSmiles(canonical_smiles)
    if canonical_mol is None or canonical_mol.GetNumAtoms() == 0:
        raise ValueError(f"规范 SMILES 无法重建: {smiles}")
    roundtrip_smiles = Chem.MolToSmiles(
        canonical_mol, canonical=True, isomericSmiles=True
    )
    if roundtrip_smiles != canonical_smiles:
        raise ValueError(
            "规范 SMILES 往返结果不稳定: "
            f"{canonical_smiles} != {roundtrip_smiles}"
        )
    return canonical_smiles, canonical_mol


def calculate_fragment_atom_record(smiles):
    """构建一个规范 SMILES 的原子属性表和无向键属性表。"""
    canonical_smiles, canonical_mol = canonicalize_fragment_smiles(smiles)
    num_atoms = canonical_mol.GetNumAtoms()
    atom_features = np.empty(
        (num_atoms, len(RAW_ATOM_FEATURE_COLUMNS)), dtype=np.int16
    )
    int16_info = np.iinfo(np.int16)

    for atom in canonical_mol.GetAtoms():
        atom_index = atom.GetIdx()
        feature_values = (
            int(atom.GetAtomicNum()),
            int(atom.GetFormalCharge()),
            int(atom.GetDegree()),
            int(atom.GetTotalValence()),
            int(atom.GetTotalNumHs(True)),
            _stable_enum_id(
                atom.GetHybridization(),
                _RAW_ATOM_HYBRIDIZATION_TO_ID,
                "OTHER",
            ),
            int(atom.GetNumRadicalElectrons()),
            _stable_enum_id(
                atom.GetChiralTag(),
                _RAW_ATOM_CHIRAL_TO_ID,
                "CHI_OTHER",
            ),
            int(atom.GetIsAromatic()),
            int(atom.IsInRing()),
        )
        if any(
            value < int16_info.min or value > int16_info.max
            for value in feature_values
        ):
            raise ValueError(
                f"片段 {canonical_smiles} 的原子 {atom_index} "
                "属性超出 int16 范围"
            )
        atom_features[atom_index] = feature_values

    bond_records = []
    for bond in canonical_mol.GetBonds():
        first_atom, second_atom = sorted((
            int(bond.GetBeginAtomIdx()),
            int(bond.GetEndAtomIdx()),
        ))
        bond_records.append((
            first_atom,
            second_atom,
            (
                _stable_enum_id(
                    bond.GetBondType(),
                    _RAW_BOND_TYPE_TO_ID,
                    "OTHER",
                ),
                int(bond.GetIsConjugated()),
                int(bond.GetIsAromatic()),
                int(bond.IsInRing()),
                _stable_enum_id(
                    bond.GetStereo(),
                    _RAW_BOND_STEREO_TO_ID,
                    "OTHER",
                ),
            ),
        ))
    bond_records.sort(key=lambda item: (item[0], item[1], item[2]))

    if bond_records:
        bond_atom_ids = np.asarray(
            [[item[0], item[1]] for item in bond_records], dtype=np.int32
        )
        bond_features = np.asarray(
            [item[2] for item in bond_records], dtype=np.int16
        )
    else:
        bond_atom_ids = np.empty((0, 2), dtype=np.int32)
        bond_features = np.empty(
            (0, len(RAW_BOND_FEATURE_COLUMNS)), dtype=np.int16
        )

    return {
        "smiles": canonical_smiles,
        "canonical_atom_id": np.arange(num_atoms, dtype=np.int32),
        "atom_features": atom_features,
        "bond_atom_ids": bond_atom_ids,
        "bond_features": bond_features,
    }


def _raw_fragment_atom_corpus_checksum(corpus_arrays):
    """计算原子粗语料中全部数组的稳定校验值。"""
    digest = hashlib.sha256()
    for key in _RAW_ATOM_CORPUS_FIELDS:
        values = np.asarray(corpus_arrays[key])
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


def _raw_fragment_atom_smiles_checksum(smiles_values):
    digest = hashlib.sha256()
    for smiles in np.asarray(smiles_values).astype(str).tolist():
        encoded = smiles.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def _raw_fragment_atom_array_checksums(corpus_arrays):
    checksums = {}
    for key in _RAW_ATOM_CORPUS_FIELDS:
        values = np.asarray(corpus_arrays[key])
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


def _build_raw_fragment_atom_arrays(smiles_collection):
    """按输入顺序构建扁平变长原子粗语料数组。"""
    canonical_smiles_list = []
    records = []
    seen_smiles = set()
    for smiles in smiles_collection:
        record = calculate_fragment_atom_record(smiles)
        canonical_smiles = record["smiles"]
        if canonical_smiles in seen_smiles:
            raise ValueError(
                f"原子粗语料包含重复规范 SMILES: {canonical_smiles}"
            )
        if str(smiles).strip() != canonical_smiles:
            raise ValueError(
                "原子粗语料输入必须已经规范化: "
                f"{smiles} -> {canonical_smiles}"
            )
        seen_smiles.add(canonical_smiles)
        canonical_smiles_list.append(canonical_smiles)
        records.append(record)

    if not records:
        raise ValueError("原子粗语料中没有可保存的规范 SMILES")

    atom_offsets = [0]
    bond_offsets = [0]
    for record in records:
        atom_offsets.append(
            atom_offsets[-1] + len(record["canonical_atom_id"])
        )
        bond_offsets.append(
            bond_offsets[-1] + len(record["bond_atom_ids"])
        )

    atom_features = np.concatenate(
        [record["atom_features"] for record in records], axis=0
    ).astype(np.int16, copy=False)
    canonical_atom_id = np.concatenate(
        [record["canonical_atom_id"] for record in records], axis=0
    ).astype(np.int32, copy=False)
    if bond_offsets[-1] > 0:
        bond_atom_ids = np.concatenate(
            [record["bond_atom_ids"] for record in records], axis=0
        ).astype(np.int32, copy=False)
        bond_features = np.concatenate(
            [record["bond_features"] for record in records], axis=0
        ).astype(np.int16, copy=False)
    else:
        bond_atom_ids = np.empty((0, 2), dtype=np.int32)
        bond_features = np.empty(
            (0, len(RAW_BOND_FEATURE_COLUMNS)), dtype=np.int16
        )

    return {
        "smiles": np.asarray(canonical_smiles_list, dtype=np.str_),
        "atom_offsets": np.asarray(atom_offsets, dtype=np.int64),
        "canonical_atom_id": canonical_atom_id,
        "atom_features": atom_features,
        "bond_offsets": np.asarray(bond_offsets, dtype=np.int64),
        "bond_atom_ids": bond_atom_ids,
        "bond_features": bond_features,
    }


def _raw_fragment_atom_category_metadata():
    return {
        "atomic_number": {
            "encoding": "identity",
            "minimum": 1,
            "maximum": 118,
        },
        "hybridization": {
            name: index
            for index, name in enumerate(
                RAW_ATOM_HYBRIDIZATION_CATEGORIES
            )
        },
        "chiral_tag": {
            name: index
            for index, name in enumerate(RAW_ATOM_CHIRAL_CATEGORIES)
        },
        "bond_type": {
            name: index
            for index, name in enumerate(RAW_BOND_TYPE_CATEGORIES)
        },
        "stereo": {
            name: index
            for index, name in enumerate(RAW_BOND_STEREO_CATEGORIES)
        },
        "boolean": {"false": 0, "true": 1},
    }


def _build_raw_fragment_atom_metadata(
    corpus_arrays,
    source_fragment_corpus_sha256,
    base_smiles_count,
):
    num_smiles = len(corpus_arrays["smiles"])
    if not 0 <= int(base_smiles_count) <= num_smiles:
        raise ValueError("base_smiles_count 超出原子粗语料范围")
    return {
        "hydrogen_policy": RAW_FRAGMENT_HYDROGEN_POLICY,
        "atom_feature_columns": list(RAW_ATOM_FEATURE_COLUMNS),
        "bond_feature_columns": list(RAW_BOND_FEATURE_COLUMNS),
        "category_mappings": _raw_fragment_atom_category_metadata(),
        "source_fragment_corpus_sha256": str(
            source_fragment_corpus_sha256
        ),
        "base_smiles_count": int(base_smiles_count),
        "base_smiles_sha256": _raw_fragment_atom_smiles_checksum(
            corpus_arrays["smiles"][:int(base_smiles_count)]
        ),
        "resolver_added_smiles_count": int(
            num_smiles - int(base_smiles_count)
        ),
        "num_smiles": int(num_smiles),
        "num_atoms": int(len(corpus_arrays["canonical_atom_id"])),
        "num_bonds": int(len(corpus_arrays["bond_atom_ids"])),
        "smiles_sha256": _raw_fragment_atom_smiles_checksum(
            corpus_arrays["smiles"]
        ),
        "corpus_sha256": _raw_fragment_atom_corpus_checksum(
            corpus_arrays
        ),
        "array_shapes": {
            key: list(np.asarray(value).shape)
            for key, value in corpus_arrays.items()
        },
        "array_dtypes": {
            key: np.asarray(value).dtype.str
            for key, value in corpus_arrays.items()
        },
        "array_sha256": _raw_fragment_atom_array_checksums(
            corpus_arrays
        ),
    }


def _validate_raw_fragment_atom_corpus(
    corpus,
    metadata,
    validate_smiles=True,
):
    if metadata.get("atom_feature_columns") != list(
        RAW_ATOM_FEATURE_COLUMNS
    ):
        raise ValueError("原子粗语料的原子特征列顺序不兼容")
    if metadata.get("bond_feature_columns") != list(
        RAW_BOND_FEATURE_COLUMNS
    ):
        raise ValueError("原子粗语料的键特征列顺序不兼容")
    if metadata.get("category_mappings") != (
        _raw_fragment_atom_category_metadata()
    ):
        raise ValueError("原子粗语料的类别映射不兼容")
    missing = set(_RAW_ATOM_CORPUS_FIELDS) - set(corpus)
    if missing:
        raise ValueError(f"原子粗语料缺少字段: {sorted(missing)}")

    smiles = np.asarray(corpus["smiles"]).astype(str)
    atom_offsets = np.asarray(corpus["atom_offsets"])
    canonical_atom_id = np.asarray(corpus["canonical_atom_id"])
    atom_features = np.asarray(corpus["atom_features"])
    bond_offsets = np.asarray(corpus["bond_offsets"])
    bond_atom_ids = np.asarray(corpus["bond_atom_ids"])
    bond_features = np.asarray(corpus["bond_features"])

    expected_dtypes = {
        "atom_offsets": np.dtype(np.int64),
        "canonical_atom_id": np.dtype(np.int32),
        "atom_features": np.dtype(np.int16),
        "bond_offsets": np.dtype(np.int64),
        "bond_atom_ids": np.dtype(np.int32),
        "bond_features": np.dtype(np.int16),
    }
    for key, expected_dtype in expected_dtypes.items():
        actual_dtype = np.asarray(corpus[key]).dtype
        if actual_dtype != expected_dtype:
            raise ValueError(
                f"原子粗语料字段 {key} 类型错误: "
                f"{actual_dtype} != {expected_dtype}"
            )

    num_smiles = len(smiles)
    if len(set(smiles.tolist())) != num_smiles:
        raise ValueError("原子粗语料包含重复规范 SMILES")
    if RAW_FRAGMENT_SPECIAL_TOKENS.intersection(smiles.tolist()):
        raise ValueError("原子粗语料中不允许出现特殊标记")
    if atom_offsets.shape != (num_smiles + 1,):
        raise ValueError("atom_offsets 形状错误")
    if bond_offsets.shape != (num_smiles + 1,):
        raise ValueError("bond_offsets 形状错误")
    if atom_features.shape != (
        len(canonical_atom_id), len(RAW_ATOM_FEATURE_COLUMNS)
    ):
        raise ValueError("atom_features 形状错误")
    if bond_atom_ids.ndim != 2 or bond_atom_ids.shape[1] != 2:
        raise ValueError("bond_atom_ids 形状必须为 [B, 2]")
    if bond_features.shape != (
        len(bond_atom_ids), len(RAW_BOND_FEATURE_COLUMNS)
    ):
        raise ValueError("bond_features 形状错误")

    for offsets, total, name in (
        (atom_offsets, len(canonical_atom_id), "atom_offsets"),
        (bond_offsets, len(bond_atom_ids), "bond_offsets"),
    ):
        if offsets[0] != 0 or offsets[-1] != total:
            raise ValueError(f"{name} 起止值错误")
        if np.any(np.diff(offsets) < 0):
            raise ValueError(f"{name} 必须单调不减")

    hybridization_max = len(RAW_ATOM_HYBRIDIZATION_CATEGORIES)
    chiral_max = len(RAW_ATOM_CHIRAL_CATEGORIES)
    bond_type_max = len(RAW_BOND_TYPE_CATEGORIES)
    stereo_max = len(RAW_BOND_STEREO_CATEGORIES)
    if len(atom_features) > 0:
        if np.any((atom_features[:, 0] < 1) | (atom_features[:, 0] > 118)):
            raise ValueError("原子粗语料包含非法原子序数")
        if np.any(
            (atom_features[:, 5] < 0)
            | (atom_features[:, 5] >= hybridization_max)
        ):
            raise ValueError("原子粗语料包含非法杂化类别")
        if np.any(
            (atom_features[:, 7] < 0)
            | (atom_features[:, 7] >= chiral_max)
        ):
            raise ValueError("原子粗语料包含非法手性类别")
        for column_index, column_name in ((8, "芳香"), (9, "环")):
            if not np.all(np.isin(atom_features[:, column_index], (0, 1))):
                raise ValueError(f"原子粗语料包含非法{column_name}标记")
    if len(bond_features) > 0:
        if np.any(
            (bond_features[:, 0] < 0)
            | (bond_features[:, 0] >= bond_type_max)
        ):
            raise ValueError("原子粗语料包含非法键类型")
        if np.any(
            (bond_features[:, 4] < 0)
            | (bond_features[:, 4] >= stereo_max)
        ):
            raise ValueError("原子粗语料包含非法键立体类别")
        for column_index, column_name in (
            (1, "共轭"), (2, "芳香"), (3, "环")
        ):
            if not np.all(np.isin(bond_features[:, column_index], (0, 1))):
                raise ValueError(f"原子粗语料包含非法键{column_name}标记")

    for fragment_index, smiles_value in enumerate(smiles.tolist()):
        atom_start = int(atom_offsets[fragment_index])
        atom_end = int(atom_offsets[fragment_index + 1])
        bond_start = int(bond_offsets[fragment_index])
        bond_end = int(bond_offsets[fragment_index + 1])
        expected_ids = np.arange(atom_end - atom_start, dtype=np.int32)
        if not np.array_equal(
            canonical_atom_id[atom_start:atom_end], expected_ids
        ):
            raise ValueError(
                f"片段 {smiles_value} 的规范原子编号不是从 0 连续递增"
            )
        fragment_bonds = bond_atom_ids[bond_start:bond_end]
        if len(fragment_bonds) > 0:
            if np.any(fragment_bonds < 0) or np.any(
                fragment_bonds >= atom_end - atom_start
            ):
                raise ValueError(
                    f"片段 {smiles_value} 的键端点超出原子范围"
                )
            if np.any(fragment_bonds[:, 0] >= fragment_bonds[:, 1]):
                raise ValueError(
                    f"片段 {smiles_value} 的无向键端点顺序非法"
                )
            if len({
                (int(first), int(second))
                for first, second in fragment_bonds.tolist()
            }) != len(fragment_bonds):
                raise ValueError(
                    f"片段 {smiles_value} 重复保存了同一根无向键"
                )
        if validate_smiles:
            rebuilt = calculate_fragment_atom_record(smiles_value)
            if rebuilt["smiles"] != smiles_value:
                raise ValueError(
                    f"原子粗语料包含非规范 SMILES: {smiles_value}"
                )
            if not np.array_equal(
                canonical_atom_id[atom_start:atom_end],
                rebuilt["canonical_atom_id"],
            ):
                raise ValueError(
                    f"片段 {smiles_value} 的规范原子编号与重建结果不一致"
                )
            if not np.array_equal(
                atom_features[atom_start:atom_end],
                rebuilt["atom_features"],
            ):
                raise ValueError(
                    f"片段 {smiles_value} 的原子属性与重建结果不一致"
                )
            if not np.array_equal(
                fragment_bonds, rebuilt["bond_atom_ids"]
            ) or not np.array_equal(
                bond_features[bond_start:bond_end],
                rebuilt["bond_features"],
            ):
                raise ValueError(
                    f"片段 {smiles_value} 的键记录与重建结果不一致"
                )

    expected_shapes = {
        key: list(np.asarray(value).shape)
        for key, value in corpus.items()
    }
    expected_dtype_strings = {
        key: np.asarray(value).dtype.str
        for key, value in corpus.items()
    }
    if metadata.get("array_shapes") != expected_shapes:
        raise ValueError("原子粗语料元数据中的数组形状不一致")
    if metadata.get("array_dtypes") != expected_dtype_strings:
        raise ValueError("原子粗语料元数据中的数组类型不一致")
    if metadata.get("array_sha256") != (
        _raw_fragment_atom_array_checksums(corpus)
    ):
        raise ValueError("原子粗语料元数据中的逐数组校验值不一致")
    if metadata.get("num_smiles") != num_smiles:
        raise ValueError("原子粗语料元数据中的片段数量不一致")
    if metadata.get("num_atoms") != len(canonical_atom_id):
        raise ValueError("原子粗语料元数据中的原子数量不一致")
    if metadata.get("num_bonds") != len(bond_atom_ids):
        raise ValueError("原子粗语料元数据中的键数量不一致")
    if metadata.get("smiles_sha256") != (
        _raw_fragment_atom_smiles_checksum(smiles)
    ):
        raise ValueError("原子粗语料 SMILES 校验值不一致")
    if metadata.get("corpus_sha256") != (
        _raw_fragment_atom_corpus_checksum(corpus)
    ):
        raise ValueError("原子粗语料内容校验值不一致")
    base_count = metadata.get("base_smiles_count")
    resolver_count = metadata.get("resolver_added_smiles_count")
    if (
        not isinstance(base_count, int)
        or not isinstance(resolver_count, int)
        or base_count < 0
        or resolver_count < 0
        or base_count + resolver_count != num_smiles
    ):
        raise ValueError("原子粗语料基础/动态词汇计数不一致")
    if metadata.get("base_smiles_sha256") != (
        _raw_fragment_atom_smiles_checksum(smiles[:base_count])
    ):
        raise ValueError("原子粗语料的初始片段词汇校验值不一致")


def atomic_write_json(path, payload):
    """使用唯一临时文件和原子替换写入 JSON。"""
    path = os.path.abspath(os.fspath(path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    token = f"{os.getpid()}-{uuid.uuid4().hex}"
    temp_path = f"{path}.{token}.tmp"
    try:
        with open(temp_path, "w", encoding="utf-8") as file_obj:
            json.dump(payload, file_obj, ensure_ascii=False, indent=2)
            file_obj.write("\n")
            file_obj.flush()
            os.fsync(file_obj.fileno())
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def derive_embedding_json_mirror_path(npz_path):
    """由权威 NPZ 路径派生同名的完整可读 JSON 镜像路径。"""
    npz_path = os.path.abspath(os.fspath(npz_path))
    stem, extension = os.path.splitext(npz_path)
    if extension.lower() == ".npz":
        return f"{stem}.json"
    return f"{npz_path}.json"


def _prepare_embedding_json_mirror_arrays(arrays, table_kind):
    table_kind = str(table_kind)
    if table_kind == "fragment":
        required = {"smiles", "fragment_embeddings"}
    elif table_kind == "atom":
        required = {
            "smiles",
            "atom_offsets",
            "canonical_atom_id",
            "atom_embeddings",
        }
    else:
        raise ValueError(
            "table_kind 必须为 'fragment' 或 'atom'"
        )

    missing = required - set(arrays)
    if missing:
        raise ValueError(
            f"JSON 镜像源数组缺少字段: {sorted(missing)}"
        )

    smiles = np.asarray(arrays["smiles"]).astype(str)
    if smiles.ndim != 1:
        raise ValueError("JSON 镜像的 smiles 必须为一维数组")

    if table_kind == "fragment":
        embeddings = np.asarray(arrays["fragment_embeddings"])
        if embeddings.ndim != 2 or embeddings.shape[0] != len(smiles):
            raise ValueError("片段 JSON 镜像的嵌入矩阵形状错误")
        if not np.all(np.isfinite(embeddings)):
            raise ValueError("片段 JSON 镜像包含 NaN 或 Inf")
        return {
            "table_kind": table_kind,
            "smiles": smiles,
            "embeddings": embeddings.astype(
                np.float32, copy=False
            ),
            "embedding_dim": int(embeddings.shape[1]),
            "num_smiles": int(len(smiles)),
            "num_atoms": None,
        }

    atom_offsets = np.asarray(arrays["atom_offsets"])
    canonical_atom_id = np.asarray(arrays["canonical_atom_id"])
    embeddings = np.asarray(arrays["atom_embeddings"])
    if atom_offsets.shape != (len(smiles) + 1,):
        raise ValueError("原子 JSON 镜像的 atom_offsets 形状错误")
    if (
        len(atom_offsets) == 0
        or int(atom_offsets[0]) != 0
        or int(atom_offsets[-1]) != len(canonical_atom_id)
        or np.any(np.diff(atom_offsets) < 0)
    ):
        raise ValueError("原子 JSON 镜像的 atom_offsets 非法")
    if (
        embeddings.ndim != 2
        or embeddings.shape[0] != len(canonical_atom_id)
    ):
        raise ValueError("原子 JSON 镜像的嵌入矩阵形状错误")
    if not np.all(np.isfinite(embeddings)):
        raise ValueError("原子 JSON 镜像包含 NaN 或 Inf")
    return {
        "table_kind": table_kind,
        "smiles": smiles,
        "atom_offsets": atom_offsets.astype(
            np.int64, copy=False
        ),
        "canonical_atom_id": canonical_atom_id.astype(
            np.int32, copy=False
        ),
        "embeddings": embeddings.astype(np.float32, copy=False),
        "embedding_dim": int(embeddings.shape[1]),
        "num_smiles": int(len(smiles)),
        "num_atoms": int(len(canonical_atom_id)),
    }


def _embedding_json_mirror_record(prepared, fragment_index):
    record = {
        "vocab_id": int(fragment_index),
        "smiles": str(prepared["smiles"][fragment_index]),
    }
    if prepared["table_kind"] == "fragment":
        record["embedding"] = prepared["embeddings"][
            fragment_index
        ].tolist()
        return record

    atom_start = int(prepared["atom_offsets"][fragment_index])
    atom_end = int(prepared["atom_offsets"][fragment_index + 1])
    record["atoms"] = [
        {
            "canonical_atom_id": int(atom_id),
            "embedding": embedding.tolist(),
        }
        for atom_id, embedding in zip(
            prepared["canonical_atom_id"][atom_start:atom_end],
            prepared["embeddings"][atom_start:atom_end],
        )
    ]
    return record


def _validate_embedding_json_mirror_record(
    record,
    prepared,
    fragment_index,
):
    if not isinstance(record, dict):
        raise ValueError(
            f"JSON 镜像第 {fragment_index} 条记录不是对象"
        )
    expected_keys = (
        {"vocab_id", "smiles", "embedding"}
        if prepared["table_kind"] == "fragment"
        else {"vocab_id", "smiles", "atoms"}
    )
    if set(record) != expected_keys:
        raise ValueError(
            f"JSON 镜像第 {fragment_index} 条记录字段错误"
        )
    if (
        not isinstance(record["vocab_id"], int)
        or isinstance(record["vocab_id"], bool)
        or record["vocab_id"] != fragment_index
    ):
        raise ValueError(
            f"JSON 镜像第 {fragment_index} 条记录 vocab_id 错误"
        )
    expected_smiles = str(prepared["smiles"][fragment_index])
    if record["smiles"] != expected_smiles:
        raise ValueError(
            f"JSON 镜像第 {fragment_index} 条记录 SMILES 错误"
        )

    if prepared["table_kind"] == "fragment":
        embedding = np.asarray(
            record["embedding"], dtype=np.float32
        )
        expected = prepared["embeddings"][fragment_index]
        if (
            embedding.shape != expected.shape
            or not np.array_equal(embedding, expected)
        ):
            raise ValueError(
                f"JSON 镜像片段 {expected_smiles} 的向量不一致"
            )
        return

    atoms = record["atoms"]
    if not isinstance(atoms, list):
        raise ValueError(
            f"JSON 镜像片段 {expected_smiles} 的 atoms 不是列表"
        )
    atom_start = int(prepared["atom_offsets"][fragment_index])
    atom_end = int(prepared["atom_offsets"][fragment_index + 1])
    if len(atoms) != atom_end - atom_start:
        raise ValueError(
            f"JSON 镜像片段 {expected_smiles} 的原子数不一致"
        )
    for local_index, atom_record in enumerate(atoms):
        if (
            not isinstance(atom_record, dict)
            or set(atom_record)
            != {"canonical_atom_id", "embedding"}
        ):
            raise ValueError(
                f"JSON 镜像片段 {expected_smiles} 的原子字段错误"
            )
        flat_index = atom_start + local_index
        expected_atom_id = int(
            prepared["canonical_atom_id"][flat_index]
        )
        if (
            not isinstance(atom_record["canonical_atom_id"], int)
            or isinstance(atom_record["canonical_atom_id"], bool)
            or atom_record["canonical_atom_id"] != expected_atom_id
        ):
            raise ValueError(
                f"JSON 镜像片段 {expected_smiles} 的规范原子编号错误"
            )
        embedding = np.asarray(
            atom_record["embedding"], dtype=np.float32
        )
        expected = prepared["embeddings"][flat_index]
        if (
            embedding.shape != expected.shape
            or not np.array_equal(embedding, expected)
        ):
            raise ValueError(
                f"JSON 镜像片段 {expected_smiles} 的原子向量不一致"
            )


def validate_embedding_json_mirror(path, arrays, table_kind):
    """
    逐行验证完整 JSON 镜像。

    文件是合法 JSON 数组，同时强制每个片段记录独占一个物理行。
    """
    path = os.path.abspath(os.fspath(path))
    prepared = _prepare_embedding_json_mirror_arrays(
        arrays, table_kind
    )
    with open(path, "r", encoding="utf-8") as file_obj:
        if file_obj.readline().strip() != "[":
            raise ValueError("JSON 镜像必须以独立的 '[' 行开始")
        for fragment_index in range(prepared["num_smiles"]):
            line = file_obj.readline()
            if not line:
                raise ValueError("JSON 镜像记录数量不足")
            payload = line.strip()
            has_comma = payload.endswith(",")
            expected_comma = (
                fragment_index + 1 < prepared["num_smiles"]
            )
            if has_comma != expected_comma:
                raise ValueError(
                    f"JSON 镜像第 {fragment_index} 条记录逗号错误"
                )
            if has_comma:
                payload = payload[:-1]
            if not payload:
                raise ValueError(
                    f"JSON 镜像第 {fragment_index} 条记录为空"
                )
            record = json.loads(payload)
            _validate_embedding_json_mirror_record(
                record, prepared, fragment_index
            )
        if file_obj.readline().strip() != "]":
            raise ValueError("JSON 镜像必须以独立的 ']' 行结束")
        if any(line.strip() for line in file_obj):
            raise ValueError("JSON 镜像结束后包含多余内容")
    return {
        "num_smiles": prepared["num_smiles"],
        "num_atoms": prepared["num_atoms"],
        "embedding_dim": prepared["embedding_dim"],
    }


def _embedding_json_mirror_file_sha256(path, block_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as file_obj:
        while True:
            block = file_obj.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def write_embedding_json_mirror_file(
    path,
    arrays,
    table_kind,
    source_table_sha256,
    mirror_filename=None,
):
    """
    将完整词表写到指定文件并回读校验。

    该函数不执行原子替换，便于调用方将 NPZ、JSON 和 metadata 纳入
    同一组临时文件提交。
    """
    path = os.path.abspath(os.fspath(path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    prepared = _prepare_embedding_json_mirror_arrays(
        arrays, table_kind
    )
    with open(path, "w", encoding="utf-8", newline="\n") as file_obj:
        file_obj.write("[\n")
        for fragment_index in range(prepared["num_smiles"]):
            record = _embedding_json_mirror_record(
                prepared, fragment_index
            )
            line = json.dumps(
                record,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
            if fragment_index + 1 < prepared["num_smiles"]:
                line += ","
            file_obj.write(line)
            file_obj.write("\n")
        file_obj.write("]\n")
        file_obj.flush()
        os.fsync(file_obj.fileno())

    validated = validate_embedding_json_mirror(
        path, arrays, table_kind
    )
    metadata = {
        "file_name": str(
            mirror_filename
            if mirror_filename is not None
            else os.path.basename(path)
        ),
        "sha256": _embedding_json_mirror_file_sha256(path),
        "source_table_sha256": str(source_table_sha256),
        "table_kind": str(table_kind),
        "layout": "json_array_one_fragment_per_line",
        "authoritative": False,
        "num_smiles": int(validated["num_smiles"]),
        "embedding_dim": int(validated["embedding_dim"]),
    }
    if validated["num_atoms"] is not None:
        metadata["num_atoms"] = int(validated["num_atoms"])
    return metadata


def atomic_write_embedding_json_mirror(
    path,
    arrays,
    table_kind,
    source_table_sha256,
):
    """原子重建可读 JSON 镜像，并返回可写入主元数据的校验信息。"""
    path = os.path.abspath(os.fspath(path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    token = f"{os.getpid()}-{uuid.uuid4().hex}"
    temp_path = f"{path}.{token}.tmp"
    try:
        metadata = write_embedding_json_mirror_file(
            temp_path,
            arrays,
            table_kind,
            source_table_sha256,
            mirror_filename=os.path.basename(path),
        )
        os.replace(temp_path, path)
        return metadata
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def embedding_json_mirror_needs_refresh(
    path,
    arrays,
    metadata,
    table_kind,
    verify_sha256=False,
):
    """
    判断可读镜像是否缺失或明显过期。

    默认不扫描大型 JSON 内容；写入阶段已经完成逐条校验。显式要求时
    可进一步核对完整文件 SHA-256。
    """
    path = os.path.abspath(os.fspath(path))
    prepared = _prepare_embedding_json_mirror_arrays(
        arrays, table_kind
    )
    mirror = metadata.get("json_mirror")
    if not isinstance(mirror, dict) or not os.path.isfile(path):
        return True
    expected = {
        "file_name": os.path.basename(path),
        "source_table_sha256": metadata.get("table_sha256"),
        "table_kind": str(table_kind),
        "layout": "json_array_one_fragment_per_line",
        "authoritative": False,
        "num_smiles": prepared["num_smiles"],
        "embedding_dim": prepared["embedding_dim"],
    }
    if any(mirror.get(key) != value for key, value in expected.items()):
        return True
    if (
        table_kind == "atom"
        and mirror.get("num_atoms") != prepared["num_atoms"]
    ):
        return True
    checksum = mirror.get("sha256")
    if (
        not isinstance(checksum, str)
        or len(checksum) != 64
    ):
        return True
    if verify_sha256:
        return _embedding_json_mirror_file_sha256(path) != checksum
    return False


def _atomic_write_raw_fragment_atom_corpus(
    corpus_arrays,
    metadata,
    npz_path,
    metadata_path,
):
    npz_path = os.path.abspath(os.fspath(npz_path))
    metadata_path = os.path.abspath(os.fspath(metadata_path))
    os.makedirs(os.path.dirname(npz_path), exist_ok=True)
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    token = f"{os.getpid()}-{uuid.uuid4().hex}"
    temp_npz_path = f"{npz_path}.{token}.tmp.npz"
    temp_metadata_path = f"{metadata_path}.{token}.tmp"
    try:
        _validate_raw_fragment_atom_corpus(
            corpus_arrays, metadata, validate_smiles=False
        )
        np.savez_compressed(temp_npz_path, **corpus_arrays)
        with np.load(temp_npz_path, allow_pickle=False) as loaded:
            temp_corpus = {
                key: np.array(loaded[key], copy=True)
                for key in _RAW_ATOM_CORPUS_FIELDS
            }
        _validate_raw_fragment_atom_corpus(
            temp_corpus, metadata, validate_smiles=False
        )
        with open(temp_metadata_path, "w", encoding="utf-8") as file_obj:
            json.dump(metadata, file_obj, ensure_ascii=False, indent=2)
            file_obj.flush()
            os.fsync(file_obj.fileno())
        os.replace(temp_npz_path, npz_path)
        os.replace(temp_metadata_path, metadata_path)
    finally:
        for temp_path in (temp_npz_path, temp_metadata_path):
            if os.path.exists(temp_path):
                os.remove(temp_path)


def save_raw_fragment_atom_corpus(
    canonical_smiles_collection,
    npz_path,
    metadata_path,
    source_fragment_corpus_sha256,
    preserve_resolver_entries=True,
):
    """
    保存初始原子粗语料；重新生成时可保留 resolver 追加的词汇。
    """
    base_smiles = [str(value) for value in canonical_smiles_collection]
    if len(set(base_smiles)) != len(base_smiles):
        raise ValueError("基础原子粗语料包含重复 SMILES")

    final_smiles = list(base_smiles)
    if (
        preserve_resolver_entries
        and os.path.isfile(npz_path)
        and os.path.isfile(metadata_path)
    ):
        existing_corpus, existing_metadata = (
            load_raw_fragment_atom_corpus(npz_path, metadata_path)
        )
        existing_base_count = int(
            existing_metadata["base_smiles_count"]
        )
        base_set = set(base_smiles)
        for smiles in existing_corpus["smiles"][
            existing_base_count:
        ].astype(str).tolist():
            if smiles not in base_set:
                final_smiles.append(smiles)
                base_set.add(smiles)

    corpus_arrays = _build_raw_fragment_atom_arrays(final_smiles)
    metadata = _build_raw_fragment_atom_metadata(
        corpus_arrays,
        source_fragment_corpus_sha256,
        base_smiles_count=len(base_smiles),
    )
    _atomic_write_raw_fragment_atom_corpus(
        corpus_arrays, metadata, npz_path, metadata_path
    )
    return metadata


def load_raw_fragment_atom_corpus(npz_path, metadata_path):
    """加载并严格校验原子粗语料。"""
    npz_path = os.path.abspath(os.fspath(npz_path))
    metadata_path = os.path.abspath(os.fspath(metadata_path))
    with open(metadata_path, "r", encoding="utf-8") as file_obj:
        metadata = json.load(file_obj)
    with np.load(npz_path, allow_pickle=False) as loaded:
        missing = set(_RAW_ATOM_CORPUS_FIELDS) - set(loaded.files)
        if missing:
            raise ValueError(
                f"原子粗语料 NPZ 缺少字段: {sorted(missing)}"
            )
        corpus = {
            key: np.array(loaded[key], copy=True)
            for key in _RAW_ATOM_CORPUS_FIELDS
        }
    corpus["smiles"] = corpus["smiles"].astype(str)
    _validate_raw_fragment_atom_corpus(
        corpus, metadata, validate_smiles=True
    )
    return corpus, metadata


def repair_raw_fragment_atom_metadata(npz_path, metadata_path):
    """
    仅用于持锁事务恢复：根据完整 NPZ 与旧元数据重建匹配的元数据。
    """
    with open(metadata_path, "r", encoding="utf-8") as file_obj:
        previous_metadata = json.load(file_obj)
    with np.load(npz_path, allow_pickle=False) as loaded:
        missing = set(_RAW_ATOM_CORPUS_FIELDS) - set(loaded.files)
        if missing:
            raise ValueError(
                f"待恢复原子粗语料 NPZ 缺少字段: {sorted(missing)}"
            )
        corpus = {
            key: np.array(loaded[key], copy=True)
            for key in _RAW_ATOM_CORPUS_FIELDS
        }
    corpus["smiles"] = corpus["smiles"].astype(str)
    repaired_metadata = _build_raw_fragment_atom_metadata(
        corpus,
        previous_metadata["source_fragment_corpus_sha256"],
        base_smiles_count=int(previous_metadata["base_smiles_count"]),
    )
    _validate_raw_fragment_atom_corpus(
        corpus, repaired_metadata, validate_smiles=True
    )
    atomic_write_json(metadata_path, repaired_metadata)
    return corpus, repaired_metadata


def extract_fragment_atom_record(corpus, fragment_index):
    """从已加载的扁平原子粗语料中提取一个片段记录。"""
    fragment_index = int(fragment_index)
    num_smiles = len(corpus["smiles"])
    if not 0 <= fragment_index < num_smiles:
        raise IndexError("原子粗语料片段编号越界")
    atom_start = int(corpus["atom_offsets"][fragment_index])
    atom_end = int(corpus["atom_offsets"][fragment_index + 1])
    bond_start = int(corpus["bond_offsets"][fragment_index])
    bond_end = int(corpus["bond_offsets"][fragment_index + 1])
    return {
        "smiles": str(corpus["smiles"][fragment_index]),
        "canonical_atom_id": np.array(
            corpus["canonical_atom_id"][atom_start:atom_end], copy=True
        ),
        "atom_features": np.array(
            corpus["atom_features"][atom_start:atom_end], copy=True
        ),
        "bond_atom_ids": np.array(
            corpus["bond_atom_ids"][bond_start:bond_end], copy=True
        ),
        "bond_features": np.array(
            corpus["bond_features"][bond_start:bond_end], copy=True
        ),
    }


def append_raw_fragment_atom_record(
    smiles,
    npz_path,
    metadata_path,
):
    """
    向原子粗语料尾部追加一个规范片段。

    调用方负责持有跨进程写锁。
    """
    record = calculate_fragment_atom_record(smiles)
    corpus, metadata = load_raw_fragment_atom_corpus(
        npz_path, metadata_path
    )
    smiles_list = corpus["smiles"].astype(str).tolist()
    if record["smiles"] in smiles_list:
        return {
            "added": False,
            "fragment_index": smiles_list.index(record["smiles"]),
            "record": record,
        }, metadata

    updated = {
        "smiles": np.concatenate([
            corpus["smiles"],
            np.asarray([record["smiles"]], dtype=np.str_),
        ]),
        "atom_offsets": np.concatenate([
            corpus["atom_offsets"],
            np.asarray([
                int(corpus["atom_offsets"][-1])
                + len(record["canonical_atom_id"])
            ], dtype=np.int64),
        ]),
        "canonical_atom_id": np.concatenate([
            corpus["canonical_atom_id"],
            record["canonical_atom_id"],
        ]).astype(np.int32, copy=False),
        "atom_features": np.concatenate([
            corpus["atom_features"],
            record["atom_features"],
        ], axis=0).astype(np.int16, copy=False),
        "bond_offsets": np.concatenate([
            corpus["bond_offsets"],
            np.asarray([
                int(corpus["bond_offsets"][-1])
                + len(record["bond_atom_ids"])
            ], dtype=np.int64),
        ]),
        "bond_atom_ids": np.concatenate([
            corpus["bond_atom_ids"],
            record["bond_atom_ids"],
        ], axis=0).astype(np.int32, copy=False),
        "bond_features": np.concatenate([
            corpus["bond_features"],
            record["bond_features"],
        ], axis=0).astype(np.int16, copy=False),
    }
    updated_metadata = _build_raw_fragment_atom_metadata(
        updated,
        metadata["source_fragment_corpus_sha256"],
        base_smiles_count=int(metadata["base_smiles_count"]),
    )
    _atomic_write_raw_fragment_atom_corpus(
        updated, updated_metadata, npz_path, metadata_path
    )
    return {
        "added": True,
        "fragment_index": len(smiles_list),
        "record": record,
    }, updated_metadata


class AtomicFileLock:
    """仅依赖标准库的跨进程独占锁，支持超时和陈旧锁清理。"""

    def __init__(
        self,
        path,
        timeout_seconds=120.0,
        stale_seconds=600.0,
        poll_seconds=0.1,
    ):
        self.path = os.path.abspath(os.fspath(path))
        self.timeout_seconds = float(timeout_seconds)
        self.stale_seconds = float(stale_seconds)
        self.poll_seconds = float(poll_seconds)
        self.token = uuid.uuid4().hex
        self.acquired = False

    def _lock_payload(self):
        return {
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "created_at": time.time(),
            "token": self.token,
        }

    @staticmethod
    def _is_local_process_alive(pid):
        try:
            pid = int(pid)
        except (TypeError, ValueError):
            return None
        if pid <= 0:
            return False
        if pid == os.getpid():
            return True

        if os.name == "nt":
            try:
                import ctypes

                kernel32 = ctypes.WinDLL(
                    "kernel32", use_last_error=True
                )
                open_process = kernel32.OpenProcess
                open_process.argtypes = (
                    ctypes.c_uint32,
                    ctypes.c_int,
                    ctypes.c_uint32,
                )
                open_process.restype = ctypes.c_void_p
                get_exit_code = kernel32.GetExitCodeProcess
                get_exit_code.argtypes = (
                    ctypes.c_void_p,
                    ctypes.POINTER(ctypes.c_uint32),
                )
                get_exit_code.restype = ctypes.c_int
                close_handle = kernel32.CloseHandle
                close_handle.argtypes = (ctypes.c_void_p,)
                close_handle.restype = ctypes.c_int

                process_query_limited_information = 0x1000
                still_active = 259
                handle = open_process(
                    process_query_limited_information, 0, pid
                )
                if not handle:
                    # 拒绝访问时不能据此判断进程已经退出。
                    return None if ctypes.get_last_error() == 5 else False
                try:
                    exit_code = ctypes.c_uint32()
                    if not get_exit_code(
                        handle, ctypes.byref(exit_code)
                    ):
                        return None
                    return exit_code.value == still_active
                finally:
                    close_handle(handle)
            except Exception:
                return None

        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return None
        return True

    def _remove_if_stale(self):
        try:
            modified_at = os.path.getmtime(self.path)
        except FileNotFoundError:
            return True
        lock_info = {}
        try:
            with open(self.path, "r", encoding="utf-8") as file_obj:
                lock_info = json.load(file_obj)
        except (OSError, ValueError):
            pass
        created_at = lock_info.get("created_at", modified_at)
        try:
            created_at = float(created_at)
        except (TypeError, ValueError):
            created_at = modified_at
        age_seconds = time.time() - min(created_at, modified_at)
        owner_alive = None
        if lock_info.get("host") == socket.gethostname():
            owner_alive = self._is_local_process_alive(
                lock_info.get("pid")
            )
        if owner_alive is True:
            return False
        if owner_alive is None and age_seconds <= self.stale_seconds:
            return False
        try:
            os.remove(self.path)
            return True
        except FileNotFoundError:
            return True
        except OSError:
            return False

    def __enter__(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        deadline = time.monotonic() + self.timeout_seconds
        payload = json.dumps(
            self._lock_payload(), ensure_ascii=True
        ).encode("utf-8")
        while True:
            try:
                descriptor = os.open(
                    self.path,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                )
            except FileExistsError:
                self._remove_if_stale()
                if time.monotonic() >= deadline:
                    owner_message = ""
                    try:
                        with open(
                            self.path, "r", encoding="utf-8"
                        ) as file_obj:
                            owner = json.load(file_obj)
                        owner_message = (
                            f"；持有者 PID={owner.get('pid')}, "
                            f"host={owner.get('host')}, "
                            f"created_at={owner.get('created_at')}"
                        )
                    except (OSError, ValueError):
                        pass
                    raise TimeoutError(
                        f"等待共享数据写锁超时: {self.path}"
                        f"{owner_message}"
                    )
                time.sleep(self.poll_seconds)
                continue
            try:
                os.write(descriptor, payload)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            self.acquired = True
            return self

    def __exit__(self, exc_type, exc_value, traceback):
        if not self.acquired:
            return False
        try:
            with open(self.path, "r", encoding="utf-8") as file_obj:
                payload = json.load(file_obj)
            if payload.get("token") == self.token:
                os.remove(self.path)
        except FileNotFoundError:
            pass
        finally:
            self.acquired = False
        return False


def get_metal_feature_charge(mol_obj):
    """使用统一规则计算金属片段的经验特征电荷。"""
    real_atoms = [atom for atom in mol_obj.GetAtoms() if atom.GetAtomicNum() != 0]
    return float(sum(
        get_atom_feature_charge(atom)
        for atom in real_atoms
    ))


def get_standardized_metal_smiles(mol_obj):
    """
    (V1.1 已修正大小写)
    检查一个分子片段是否为单原子金属。
    如果是，则返回其不带电荷和方括号的、保持正确大小写的标准SMILES（例如 "[Mg]"）。
    否则返回 None。
    """
    if mol_obj.GetNumAtoms() == 1 and mol_obj.GetAtomWithIdx(0).GetAtomicNum() != 0:
        atom = mol_obj.GetAtomWithIdx(0)
        
        # --- 修正点: 先获取正确的大小写，再进行不区分大小写的比较 ---
        symbol_correct_case = atom.GetSymbol() # 例如 "Mg", "Fe"
        
        # 使用 .upper() 只是为了在 METAL_SYMBOLS 集合中进行查找
        if symbol_correct_case.upper() in METAL_SYMBOLS:
            # 返回时，使用原始的、大小写正确的符号
            return f"[{symbol_correct_case}]"
            
    return None


def build_metal_shard_record(metal_mol):
    """将一个金属原子重建成无键、无电荷、无氢的标准单原子 shard。"""
    if metal_mol is None:
        return None

    source_atoms = [
        atom for atom in metal_mol.GetAtoms()
        if atom.GetAtomicNum() != 0
    ]
    if len(source_atoms) != 1 or not is_metal_atom(source_atoms[0]):
        raise ValueError("金属 shard 输入必须恰好包含一个金属原子")

    source_atom = source_atoms[0]
    clean_atom = Chem.Atom(source_atom.GetAtomicNum())
    clean_atom.SetFormalCharge(0)
    clean_atom.SetNumExplicitHs(0)
    clean_atom.SetNoImplicit(True)
    clean_atom.SetIsotope(0)
    clean_atom.SetAtomMapNum(0)
    clean_atom.SetNumRadicalElectrons(0)
    _copy_atom_trace_properties(source_atom, clean_atom)

    rw_clean = Chem.RWMol()
    rw_clean.AddAtom(clean_atom)
    clean_mol = rw_clean.GetMol()
    clean_mol.UpdatePropertyCache(strict=False)
    Chem.GetSymmSSSR(clean_mol)

    if metal_mol.GetNumConformers() > 0:
        source_position = metal_mol.GetConformer(0).GetAtomPosition(
            source_atom.GetIdx()
        )
        clean_conformer = Chem.Conformer(1)
        clean_conformer.SetAtomPosition(0, source_position)
        clean_mol.AddConformer(clean_conformer, assignId=True)

    standardized_smiles = get_standardized_metal_smiles(clean_mol)
    if standardized_smiles is None:
        raise ValueError("重建后的金属原子无法生成标准 [X] SMILES")

    centroid = calculate_centroid(clean_mol)
    ref_coord = calculate_reference_coordinate_system(clean_mol)
    if centroid is None or ref_coord is None:
        return None

    return {
        "mol": clean_mol,
        "clean_mol": clean_mol,
        "smiles": standardized_smiles,
        "avg_ecc": 0.0,
        "ecc_range": 0.0,
        "avg_logp": 0.0,
        "tpsa": 0.0,
        "avg_charge": get_metal_feature_charge(clean_mol),
        "HBA": 0.0,
        "HBD": 0.0,
        "Aromatic": 0.0,
        "Hydrophobe": 0.0,
        "PosIonizable": 0.0,
        "NegIonizable": 0.0,
        "max_dist_3d": 0.0,
        "max_angle_3d": 0.0,
        "max_plane_angle_3d": 0.0,
        "centroid_3d": centroid,
        "ref_coord_3d": ref_coord,
        "ports": [],
        "hac": calculate_heavy_atom_count(clean_mol),
        "ring_count": calculate_ring_count(clean_mol),
        "is_metal": True,
    }

def _bfs_find_nearest_single_bond(mol, start_atom_idx, blocked_parent_idx):
    """
    辅助函数：BFS 搜索最近的单键。
    已包含对虚原子的检查，防止搜索穿过虚原子。
    """
    queue = [(start_atom_idx, 0)]
    visited = {start_atom_idx, blocked_parent_idx}
    MAX_SEARCH_DEPTH = 5 

    while queue:
        curr_idx, depth = queue.pop(0)
        if depth >= MAX_SEARCH_DEPTH: continue

        curr_atom = mol.GetAtomWithIdx(curr_idx)
        
        # 遇到虚原子直接停止该路径
        if curr_atom.GetAtomicNum() == 0:
            continue

        if curr_atom.IsInRing() and curr_idx != start_atom_idx:
            continue

        for bond in curr_atom.GetBonds():
            neighbor = bond.GetOtherAtom(curr_atom)
            nbr_idx = neighbor.GetIdx()
            
            # 忽略虚原子邻居
            if neighbor.GetAtomicNum() == 0:
                continue

            if nbr_idx in visited: continue

            if bond.GetBondType() == Chem.BondType.SINGLE and not bond.IsInRing():
                return bond.GetIdx()

            if not neighbor.IsInRing():
                visited.add(nbr_idx)
                queue.append((nbr_idx, depth + 1))
    return None

def is_fragment_valid(mol):
    """
    检查片段是否符合保留条件：
    1. 重原子数 <= 24 (新增: 且必须 > 0)
    2. 环数量 < 6
    3. 单个环的键数量 < 12 (无特大环)
    """
    # ====== 新增修复：过滤掉没有真实重原子的“幽灵”碎片 ======
    if mol.GetNumHeavyAtoms() == 0:
        return False
        
    # 1. 重原子上限检查
    if mol.GetNumHeavyAtoms() > 24:
        return False
    
    ri = mol.GetRingInfo()
    # 2. 环数量检查
    if ri.NumRings() >= 6:
        return False
        
    # 3. 大环检查 (AtomRings 返回每个环的原子索引元组，长度即为环大小)
    for ring in ri.AtomRings():
        if len(ring) >= 12:
            return False
            
    return True

def advanced_rescue_valence_errors(mol):
    """
    [增强版] 结晶学伪影精确抢救机制：
    直接捕获 RDKit 抛出的 Explicit valence 异常，通过正则解析出具体出问题的原子 ID。
    然后针对性地切断该原子连接的最长的一根键。循环直到分子通过检查。
    """
    if not mol or mol.GetNumConformers() == 0:
        return mol

    import re
    rw_mol = Chem.RWMol(mol)
    conf = rw_mol.GetConformer(0)
    
    max_attempts = 50  # 设置最大抢救次数，防止死循环
    
    for attempt in range(max_attempts):
        try:
            # 每次测试都需要在一个全新的拷贝上进行，防止 RDKit 内部状态在失败时被彻底破坏
            test_mol = rw_mol.GetMol()
            test_mol.UpdatePropertyCache(strict=False)
            Chem.SanitizeMol(test_mol, sanitizeOps=Chem.SANITIZE_SYMMRINGS|Chem.SANITIZE_SETCONJUGATION|Chem.SANITIZE_SETHYBRIDIZATION)
            # 如果能顺利走到这里，说明分子已经完全健康了！
            return test_mol
            
        except Exception as e:
            error_msg = str(e)
            # 尝试从错误信息中抓取出问题的原子 ID
            # 典型的报错如: "Explicit valence for atom # 1061 C, 5, is greater than permitted"
            match = re.search(r"atom #\s*(\d+)", error_msg)
            
            if match:
                bad_atom_idx = int(match.group(1))
                bad_atom = rw_mol.GetAtomWithIdx(bad_atom_idx)
                
                max_len = -1.0
                bond_to_break = None
                
                # 遍历这个错误原子的所有键，找到 3D 距离最长的那一根 (最可疑的虚假键)
                for bond in bad_atom.GetBonds():
                    neighbor = bond.GetOtherAtom(bad_atom)
                    pos1 = np.array(conf.GetAtomPosition(bad_atom.GetIdx()))
                    pos2 = np.array(conf.GetAtomPosition(neighbor.GetIdx()))
                    dist = np.linalg.norm(pos1 - pos2)
                    
                    if dist > max_len:
                        max_len = dist
                        bond_to_break = (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
                
                # 切断这根最长的键
                if bond_to_break:
                    rw_mol.RemoveBond(bond_to_break[0], bond_to_break[1])
                else:
                    # 如果找不到键可以切（极端情况），只能放弃抢救
                    break
            else:
                # 如果抛出的不是化合价溢出异常，或者正则没匹配上，停止抢救
                break
                
    # 返回尽力抢救后的结果
    return rw_mol.GetMol()


_PHARMACOPHORE_FACTORY = None
_PHARMACOPHORE_FACTORY_PID = None
_UNCHARGER = None
_UNCHARGER_PID = None


def _get_process_local_pharmacophore_factory():
    """
    为当前进程延迟创建并复用 RDKit 药效团特征工厂。

    FeatureFactory 是 RDKit C++ 对象。按片段反复构建会产生大量原生内存
    分配；在多进程 worker 中则应确保对象在 fork 之后由各进程各自创建。
    """
    global _PHARMACOPHORE_FACTORY
    global _PHARMACOPHORE_FACTORY_PID

    current_pid = os.getpid()
    if (
        _PHARMACOPHORE_FACTORY is None
        or _PHARMACOPHORE_FACTORY_PID != current_pid
    ):
        fdef_name = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
        _PHARMACOPHORE_FACTORY = Feat.BuildFeatureFactory(fdef_name)
        _PHARMACOPHORE_FACTORY_PID = current_pid
    return _PHARMACOPHORE_FACTORY


def _get_process_local_uncharger():
    """为当前 worker 进程延迟创建并复用 RDKit Uncharger。"""
    global _UNCHARGER
    global _UNCHARGER_PID

    current_pid = os.getpid()
    if _UNCHARGER is None or _UNCHARGER_PID != current_pid:
        _UNCHARGER = rdMolStandardize.Uncharger()
        _UNCHARGER_PID = current_pid
    return _UNCHARGER


def _heavy_atom_graph_signature(mol_obj):
    """返回不含电荷和氢数的重原子顺序及重原子键图签名。"""
    heavy_indices = [
        atom.GetIdx()
        for atom in mol_obj.GetAtoms()
        if atom.GetAtomicNum() > 1
    ]
    index_to_rank = {
        atom_index: rank
        for rank, atom_index in enumerate(heavy_indices)
    }
    atom_numbers = tuple(
        mol_obj.GetAtomWithIdx(atom_index).GetAtomicNum()
        for atom_index in heavy_indices
    )
    heavy_bonds = []
    for bond in mol_obj.GetBonds():
        begin_index = bond.GetBeginAtomIdx()
        end_index = bond.GetEndAtomIdx()
        if begin_index not in index_to_rank or end_index not in index_to_rank:
            continue
        first, second = sorted((
            index_to_rank[begin_index],
            index_to_rank[end_index],
        ))
        heavy_bonds.append((
            first,
            second,
            str(bond.GetBondType()),
            bool(bond.GetIsAromatic()),
        ))
    return atom_numbers, tuple(sorted(heavy_bonds))


def _heavy_atom_charge_hydrogen_signature(mol_obj):
    """返回用于判断中和是否实际改变片段的重原子电荷/氢数签名。"""
    return tuple(
        (
            atom.GetFormalCharge(),
            atom.GetTotalNumHs(includeNeighbors=True),
        )
        for atom in mol_obj.GetAtoms()
        if atom.GetAtomicNum() > 1
    )


def neutralize_pocket_fragment_if_possible(mol_obj, origin_type):
    """
    尝试中和至少含一个 C/N 且不含金属的 pocket 片段。

    仅在重原子顺序和重原子键图保持不变且候选结构可完整净化时提交；
    任何失败均静默返回原片段。
    """
    if origin_type != "pocket" or mol_obj is None:
        return mol_obj, False

    atoms = list(mol_obj.GetAtoms())
    if any(is_metal_atom(atom) for atom in atoms):
        return mol_obj, False
    if not any(atom.GetAtomicNum() in (6, 7) for atom in atoms):
        return mol_obj, False

    source_graph_signature = _heavy_atom_graph_signature(mol_obj)
    source_charge_h_signature = _heavy_atom_charge_hydrogen_signature(
        mol_obj
    )

    try:
        candidate = _get_process_local_uncharger().uncharge(
            Chem.Mol(mol_obj)
        )
        candidate = Chem.RemoveHs(candidate)
        candidate.UpdatePropertyCache(strict=False)
        Chem.SanitizeMol(candidate)
    except Exception:
        return mol_obj, False

    if _heavy_atom_graph_signature(candidate) != source_graph_signature:
        return mol_obj, False

    source_heavy_atoms = [
        atom for atom in mol_obj.GetAtoms()
        if atom.GetAtomicNum() > 1
    ]
    candidate_heavy_atoms = [
        atom for atom in candidate.GetAtoms()
        if atom.GetAtomicNum() > 1
    ]
    if len(source_heavy_atoms) != len(candidate_heavy_atoms):
        return mol_obj, False

    for source_atom, candidate_atom in zip(
        source_heavy_atoms,
        candidate_heavy_atoms,
    ):
        _copy_atom_trace_properties(source_atom, candidate_atom)

    if (
        mol_obj.GetNumConformers() > 0
        and candidate.GetNumConformers() == 0
    ):
        source_conformer = mol_obj.GetConformer(0)
        candidate_conformer = Chem.Conformer(candidate.GetNumAtoms())
        for source_atom, candidate_atom in zip(
            source_heavy_atoms,
            candidate_heavy_atoms,
        ):
            candidate_conformer.SetAtomPosition(
                candidate_atom.GetIdx(),
                source_conformer.GetAtomPosition(source_atom.GetIdx()),
            )
        candidate.AddConformer(candidate_conformer, assignId=True)

    try:
        Chem.MolToSmiles(
            candidate,
            canonical=True,
            isomericSmiles=True,
        )
    except Exception:
        return mol_obj, False

    changed = (
        _heavy_atom_charge_hydrogen_signature(candidate)
        != source_charge_h_signature
    )
    return candidate, changed


def calculate_descriptors_and_store(
    storage, mol_data, original_mol_with_conf, main_mol_for_matching,
    final_covalent_fragments, original_eccentricities,
    original_logp_contribs, original_tpsa_contribs, original_charges,
    origin_type,
):
    """
    (V3.0 修复版)
    - 逻辑修正: 二次精修队列判定改为检测虚原子，彻底解决 [5*] 等开环残留问题。
    - 兼容性: 支持仅处理离子片段（当主分子重构失败时）。
    """
    factory = _get_process_local_pharmacophore_factory()

    # === 新增：特征硬截断边界与辅助函数 ===
    FEATURE_BOUNDS = {
        'avg_ecc': (0.0, 100.0),
        'ecc_range': (0.0, 50.0),
        'avg_logp': (-15.0, 15.0),
        'tpsa': (0.0, 500.0),
        'avg_charge': (-5.0, 5.0),
        'HBA': (0.0, 20.0),
        'HBD': (0.0, 20.0),
        'Aromatic': (0.0, 10.0),
        'Hydrophobe': (0.0, 20.0),
        'PosIonizable': (0.0, 10.0),
        'NegIonizable': (0.0, 10.0),
        'max_dist_3d': (0.0, 100.0),
        'max_angle_3d': (0.0, 180.0),
        'max_plane_angle_3d': (0.0, 90.0)
    }

    def enforce_chemical_bounds(data_dict):
        """对字典中的化学特征进行强制截断，防止极端异常值毁灭全局方差"""
        for prop, bounds in FEATURE_BOUNDS.items():
            if prop in data_dict and data_dict[prop] is not None:
                val = data_dict[prop]
                if isinstance(val, (int, float)) and np.isfinite(val):
                    data_dict[prop] = float(np.clip(val, bounds[0], bounds[1]))
        return data_dict

    has_3d_conformer = original_mol_with_conf.GetNumConformers() > 0
    original_conformer = original_mol_with_conf.GetConformer(0) if has_3d_conformer else None
    
    # 丢弃计数器初始化
    dropped_fragment_count = 0

    def get_pharmacophore_features(mol_obj):
        feats = factory.GetFeaturesForMol(mol_obj)
        feature_counts = {'HBA': 0, 'HBD': 0, 'Aromatic': 0, 'Hydrophobe': 0, 'PosIonizable': 0, 'NegIonizable': 0}
        for f in feats:
            family = f.GetFamily()
            if family == 'Acceptor': feature_counts['HBA'] += 1
            elif family == 'Donor': feature_counts['HBD'] += 1
            elif family == 'Aromatic': feature_counts['Aromatic'] += 1
            elif family == 'Hydrophobe': feature_counts['Hydrophobe'] += 1
            elif family == 'PosIonizable': feature_counts['PosIonizable'] += 1
            elif family == 'NegIonizable': feature_counts['NegIonizable'] += 1
        return feature_counts

    def _is_healthy(*values):
        for value in values:
            if not np.isfinite(value):
                return False
            if abs(value) > 1e6:
                return False
        return True

    def calculate_local_properties(mol_obj):
        """基于当前片段结构重新计算全部局部理化属性。"""
        frag_logp = rdMolDescriptors.CalcCrippenDescriptors(mol_obj)[0]
        frag_tpsa_rdkit = rdMolDescriptors.CalcTPSA(mol_obj)

        AllChem.ComputeGasteigerCharges(mol_obj)
        charges_raw = [
            atom.GetProp('_GasteigerCharge')
            for atom in mol_obj.GetAtoms()
            if atom.HasProp('_GasteigerCharge')
        ]
        charges_float = [
            float(charge)
            for charge in charges_raw
            if np.isfinite(float(charge))
        ]
        frag_avg_charge = (
            np.nanmean(charges_float)
            if charges_float else 0.0
        )

        dist_mat = GetDistanceMatrix(mol_obj)
        eccentricities = dist_mat.max(axis=1)
        if len(eccentricities) == 0:
            raise ValueError("拓扑畸形，无法计算偏心率")
        frag_avg_ecc = np.nanmean(eccentricities)
        frag_ecc_range = np.ptp(eccentricities)

        if not _is_healthy(
            frag_avg_ecc,
            frag_ecc_range,
            frag_logp,
            frag_tpsa_rdkit,
            frag_avg_charge,
        ):
            raise ValueError(
                "局部属性计算结果包含不合法数值 "
                "(如 NaN/Inf 或 RDKit 孤岛惩罚值)"
            )

        return {
            "avg_ecc": frag_avg_ecc,
            "ecc_range": frag_ecc_range,
            "avg_logp": frag_logp,
            "tpsa": frag_tpsa_rdkit,
            "avg_charge": frag_avg_charge,
        }

    # --- 处理共价片段 (重构版：队列+二次切割) ---
    current_mol_results_data = []
    
    # 1. 初始化队列
    processing_queue = list(final_covalent_fragments)
    max_iterations = 1000 
    iterations = 0

    while processing_queue and iterations < max_iterations:
        fragment = processing_queue.pop(0)
        iterations += 1
        
        # === A. 深度清洗 (Dummy -> H -> Sanitize -> RemoveHs) ===
        rw_mol = Chem.RWMol(fragment)
        dummies = [a for a in rw_mol.GetAtoms() if a.GetAtomicNum() == 0]
        
        for atom in dummies:
            # 重置键类型
            for bond in atom.GetBonds():
                if bond.GetBondType() != Chem.BondType.SINGLE:
                    bond.SetBondType(Chem.BondType.SINGLE)
                    bond.SetIsAromatic(False)
            
            # [关键] 将虚原子变为 H
            atom.SetAtomicNum(1)        
            atom.SetIsotope(0)          
            atom.SetAtomMapNum(0)
            if atom.HasProp("_original_index"): atom.ClearProp("_original_index")
            if atom.HasProp("_cut_id"): atom.ClearProp("_cut_id")
            # [核心修复 3]: 同步清理内部索引，防止虚原子干扰特征提取
            if atom.HasProp("_internal_index"): atom.ClearProp("_internal_index")

            # 激活邻居原子的隐式氢重算
            for neighbor in atom.GetNeighbors():
                neighbor.SetNoImplicit(False) 
                neighbor.SetNumExplicitHs(0) 
                neighbor.UpdatePropertyCache(strict=False)

        clean_temp = rw_mol.GetMol()
        
        # 3. 结构修复 (Sanitize) - 分级尝试与 3D 距离抢救
        lg = RDLogger.logger()
        lg.setLevel(RDLogger.CRITICAL)
        rescued_by_cutting = False
        
        try:
            Chem.SanitizeMol(clean_temp)
        except Exception:
            try:
                Chem.SanitizeMol(clean_temp, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES)
            except Exception as secondary_exc:
                if origin_type == "ligand":
                    raise LigandSecondaryCleaningError(
                        f"ligand 二次清洗失败: {secondary_exc}"
                    ) from secondary_exc
                # pocket 无需端口重建，保留原有 3D 键长抢救机制。
                clean_temp = advanced_rescue_valence_errors(clean_temp)
                rescued_by_cutting = True
                try:
                    Chem.SanitizeMol(
                        clean_temp,
                        sanitizeOps=(
                            Chem.SanitizeFlags.SANITIZE_ALL
                            ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES
                        ),
                    )
                except Exception:
                    pass
        finally:
            lg.setLevel(RDLogger.CRITICAL)

        # 4. 去氢 (RemoveHs) - 保护有机立体氢，定点清除金属/类金属上的畸形假氢
        try:
            # 第一步：先执行 RDKit 默认去氢（安全，保留有机片段的立体氢）
            clean_fragment = Chem.RemoveHs(clean_temp)
            
            # 第二步：针对遗留的金属/类金属氢进行定点暴力清除
            rw_mol = Chem.RWMol(clean_fragment)
            h_to_remove = []
            
            for atom in rw_mol.GetAtoms():
                if atom.GetAtomicNum() == 1: # 找到残留的氢原子
                    neighbor = atom.GetNeighbors()[0]
                    # 常见有机重原子白名单: B(5), C(6), N(7), O(8), F(9), Si(14), P(15), S(16), Cl(17), Se(34), Br(35), I(53)
                    organic_heavy_nums = {5, 6, 7, 8, 9, 14, 15, 16, 17, 34, 35, 53}
                    
                    if neighbor.GetAtomicNum() not in organic_heavy_nums:
                        h_to_remove.append(atom.GetIdx())
            
            # 从大到小删除原子，防止 RWMol 内部索引错乱
            for idx in sorted(h_to_remove, reverse=True):
                rw_mol.RemoveAtom(idx)
                
            clean_fragment = rw_mol.GetMol()
        except:
            clean_fragment = clean_temp

        if rescued_by_cutting:
            split_frags = list(Chem.GetMolFrags(clean_fragment, asMols=True))
            if len(split_frags) > 1:
                # 如果断裂成了多个独立碎片，将它们放回队列头部，重新进行标准化清洗
                processing_queue[0:0] = split_frags
                continue 
        
        # =========================================================================
        # B. 二次精修流水线 (Rule 3 -> Rule 8 连续执行)
        # =========================================================================
        pipeline_frags = [clean_fragment]
        
        # --- 步骤 1: 准备拓扑信息 (为 Rule 3) ---
        try:
            temp_dist_mat = GetDistanceMatrix(clean_fragment)
            temp_eccs = temp_dist_mat.max(axis=1)
            temp_bond_info = get_ring_bond_info(clean_fragment, temp_eccs)
        except Exception:
            temp_bond_info = {} 

        # --- 步骤 2: 执行 Rule 3 (大环切割) ---
        try:
            r3_results = rule3_cut_large_rings(
                pipeline_frags,
                temp_bond_info,
                track_ports=(origin_type == "ligand"),
            )
        except PortValidationError:
            raise
        except Exception:
            r3_results = pipeline_frags
        
        pipeline_frags = r3_results
        
        # --- 步骤 3: 执行 Rule 8 (无环链切割) ---
        try:
            r8_results = rule8_cut_acyclic_chains(
                pipeline_frags,
                track_ports=(origin_type == "ligand"),
            )
        except PortValidationError:
            raise
        except Exception:
            r8_results = pipeline_frags
            
        # =========================================================================
        # [核心修订] C. 判定结果与递归
        # =========================================================================
        # 只要结果中出现了虚原子 (AtomicNum == 0)，说明发生了切割或开环
        has_new_dummy = False
        for res_mol in r8_results:
            for atom in res_mol.GetAtoms():
                if atom.GetAtomicNum() == 0:
                    has_new_dummy = True
                    break
            if has_new_dummy: break
            
        if has_new_dummy:
            # 如果产生了新的切割痕迹，放回队列重新清洗 (将虚原子转H)
            processing_queue[0:0] = r8_results
            continue 
            
        final_processed_mol = r8_results[0]
        neutralized_local_properties = None
        neutralized_feature_counts = None
        neutralized_candidate, neutralization_changed = (
            neutralize_pocket_fragment_if_possible(
                final_processed_mol,
                origin_type,
            )
        )
        if neutralization_changed:
            try:
                candidate_local_properties = calculate_local_properties(
                    neutralized_candidate
                )
                candidate_feature_counts = get_pharmacophore_features(
                    neutralized_candidate
                )
            except Exception:
                # 中和后的结构只要有任一局部特征无法重算，就完整回退原片段。
                neutralization_changed = False
            else:
                final_processed_mol = neutralized_candidate
                neutralized_local_properties = candidate_local_properties
                neutralized_feature_counts = candidate_feature_counts

        # =========================================================================
        # D. 最终阶段过滤检查
        # =========================================================================
        if not is_fragment_valid(final_processed_mol):
            dropped_fragment_count += 1
            continue 

        # === E. 生成最终 SMILES 和属性  ===
        standardized_metal_smi = get_standardized_metal_smiles(final_processed_mol)
        if standardized_metal_smi:
            final_smiles = standardized_metal_smi
        else:
            try:
                final_smiles = Chem.MolToSmiles(final_processed_mol, canonical=True)
            except:
                try:
                    final_smiles = Chem.MolToSmiles(clean_temp, canonical=True)
                except:
                    final_smiles = "<ERROR>"

        if final_smiles == "<ERROR>":
            dropped_fragment_count += 1
            continue

        match_indices = []
        for atom in final_processed_mol.GetAtoms():
            if atom.HasProp("_internal_index"):
                match_indices.append(atom.GetIntProp("_internal_index"))
        
        feature_counts = (
            neutralized_feature_counts
            if neutralized_feature_counts is not None
            else get_pharmacophore_features(final_processed_mol)
        )
        r = {
            "mol": final_processed_mol, 
            "clean_mol": final_processed_mol, 
            "smiles": final_smiles,
            "avg_ecc": -1.0, "ecc_range": 0.0, "avg_logp": -1.0, "tpsa": -1.0, "avg_charge": -1.0,
            "HBA": float(feature_counts['HBA']), "HBD": float(feature_counts['HBD']),
            "Aromatic": float(feature_counts['Aromatic']), "Hydrophobe": float(feature_counts['Hydrophobe']),
            "PosIonizable": float(feature_counts['PosIonizable']), "NegIonizable": float(feature_counts['NegIonizable']),
            "hac": calculate_heavy_atom_count(final_processed_mol),
            "ring_count": calculate_ring_count(final_processed_mol),
            "is_metal": False,
        }

        # ==========================================================
        # 属性赋值：Plan A (全局映射) -> Plan B (局部计算) -> Plan C (处决)
        # ==========================================================
        properties_calculated = False

        if neutralized_local_properties is not None:
            r.update(neutralized_local_properties)
            properties_calculated = True
        
        # === Plan A: 优先尝试使用高质量的预计算全局属性映射 ===
        if (
            not properties_calculated
            and match_indices
            and original_eccentricities is not None
            and len(original_eccentricities) > 0
        ):
            try:
                frag_eccentricities = [original_eccentricities[i] for i in match_indices if i < len(original_eccentricities)]
                if frag_eccentricities:
                    avg_ecc = np.nanmean(frag_eccentricities)
                    ecc_range = np.ptp(frag_eccentricities)
                    avg_logp = np.nanmean([original_logp_contribs[i] for i in match_indices])
                    tpsa_rdkit = np.nansum([original_tpsa_contribs[i] for i in match_indices])
                    avg_charge = np.nanmean([original_charges[i] for i in match_indices])

                    # 【核心修订】：只有所有数值都健康，才认可 Plan A 成功！
                    if _is_healthy(avg_ecc, ecc_range, avg_logp, tpsa_rdkit, avg_charge):
                        r.update({
                            "avg_ecc": avg_ecc, "ecc_range": ecc_range, "avg_logp": avg_logp,
                            "tpsa": tpsa_rdkit, "avg_charge": avg_charge
                        })
                        properties_calculated = True
                    else:
                        # 如果不健康，静默跳过，让程序自然进入 Plan B
                        pass
            except Exception:
                pass # Plan A 代码崩溃，静默放行，交给 Plan B 兜底

        # === Plan B: 全局预计算如果失败，立刻对切好的碎片进行局部实时计算 ===
        if not properties_calculated:
            try:
                r.update(calculate_local_properties(final_processed_mol))
                properties_calculated = True
                
            except Exception as e:
                # === Plan C: 畸形到连局部属性也算不出，坚决放弃该复合物！ ===
                # 抛出异常被外层捕获
                raise ValueError(f"碎片极度畸形，无法计算化学属性，触发复合物丢弃: {e}")

        # --- 兜底判断：即使 _internal_index 丢失，只要有 _original_index 就能恢复坐标 ---
        has_original_index = any(a.HasProp("_original_index") for a in final_processed_mol.GetAtoms())

        geometry_restored = False
        if has_3d_conformer and (match_indices or has_original_index):
            new_conf = Chem.Conformer(final_processed_mol.GetNumAtoms())
            try:
                for atom in final_processed_mol.GetAtoms():
                    if atom.HasProp("_original_index"):
                        orig_idx = atom.GetIntProp("_original_index")
                        if orig_idx < original_conformer.GetNumAtoms():
                            orig_pos = original_conformer.GetAtomPosition(orig_idx)
                            new_conf.SetAtomPosition(atom.GetIdx(), orig_pos)
                        else:
                            new_conf.SetAtomPosition(atom.GetIdx(), (0.0, 0.0, 0.0))
                    else:
                        new_conf.SetAtomPosition(atom.GetIdx(), (0.0, 0.0, 0.0))

                final_processed_mol.AddConformer(new_conf, assignId=True)
                geometry_restored = True
            except Exception:
                geometry_restored = False

        # 最终统一到化学规范 SMILES 的原子顺序，并同步端口锚点编号。
        canonical_mol, canonical_smiles, ports = canonicalize_fragment_with_ports(
            final_processed_mol
        )
        if standardized_metal_smi:
            canonical_smiles = standardized_metal_smi
        r["mol"] = canonical_mol
        r["clean_mol"] = canonical_mol
        r["smiles"] = canonical_smiles
        r["ports"] = ports

        # 必须在规范原子顺序和端口映射确定后再生成空间描述符。
        # 这样 ref_coord_3d 的三行与 decoder 使用完全相同的规范锚点。
        if geometry_restored and canonical_mol.GetNumConformers() > 0:
            try:
                r['centroid_3d'] = calculate_centroid(canonical_mol)
                r['ref_coord_3d'] = calculate_reference_coordinate_system(
                    canonical_mol
                )
                if r['ref_coord_3d'] is None:
                    raise ValueError("规范片段无法生成参考坐标")
                r['max_dist_3d'] = calculate_max_distance(canonical_mol)
                r['max_angle_3d'] = calculate_max_angle(canonical_mol)
                r['max_plane_angle_3d'] = calculate_max_plane_angle(
                    canonical_mol
                )
            except Exception:
                r['max_dist_3d'] = -1.0
                r['max_angle_3d'] = -1.0
                r['max_plane_angle_3d'] = -1.0
                r['centroid_3d'] = None
                r['ref_coord_3d'] = None
        else:
            r['max_dist_3d'] = -1.0
            r['max_angle_3d'] = -1.0
            r['max_plane_angle_3d'] = -1.0
            r['centroid_3d'] = None
            r['ref_coord_3d'] = None

        r = enforce_chemical_bounds(r)
        current_mol_results_data.append(r)

    storage.append({
        'id': mol_data['IDs'], 'smiles': mol_data['Smiles'],
        'results': current_mol_results_data
    })
    
    return dropped_fragment_count

def update_online_stats(existing_stats, new_data_point):
    """
    [新移入] 使用Welford在线算法，根据单个新数据点来精确更新统计量。
    """
    count = existing_stats.get('count', 0)
    mean = existing_stats.get('mean', 0.0)
    M2 = existing_stats.get('M2', 0.0)
    
    count += 1
    delta = new_data_point - mean
    mean += delta / count
    M2 += delta * (new_data_point - mean)
    
    return {'count': count, 'mean': mean, 'M2': M2}

# ----------------------------------------
# 主切割流程
# ----------------------------------------

def execute_fragmentation_pipeline(main_mol_for_matching, bond_info, track_ports):
    """
    碎片化流水线。
    Rule 1/2 已在预处理阶段完成。本函数执行 Rule 3、Rule 5 至 Rule 9；
    Rule 4 的实现予以保留，但不再进入 2D/3D 切割流程。
    ligand 使用稳定端口切割，pocket 只切割而不记录端口。
    """

    fragments_in_pipeline = [main_mol_for_matching]
    fragments_in_pipeline = rule3_cut_large_rings(
        fragments_in_pipeline, bond_info, track_ports
    )
    fragments_in_pipeline = rule5_cut_multi_ring_fragments(
        fragments_in_pipeline, main_mol_for_matching, bond_info, track_ports
    )
    fragments_in_pipeline = rule6_cut_ring_substituents(
        fragments_in_pipeline, track_ports
    )
    fragments_in_pipeline = rule7_cut_acid_like_groups(
        fragments_in_pipeline, track_ports
    )
    fragments_in_pipeline = rule8_cut_acyclic_chains(
        fragments_in_pipeline, track_ports
    )
    final_covalent_fragments = rule9_cut_large_carbon_fragments(
        fragments_in_pipeline, main_mol_for_matching, track_ports
    )

    return final_covalent_fragments

def rule1_filter_ions_and_complexing_agents(mol, origin_type):
    """
    1. 获取所有不连通片段。
    2. 检查每个片段的 SMILES 是否在预定义的离子/溶剂列表中。
    已知普通离子和中性络合剂直接删除。pocket 中独立金属始终保留；
    ligand 中独立金属视为真实离子并删除。连接态金属交给 Rule 2。
    """
    
    # --- 1. 定义白名单 (保持原有列表，可根据需要扩充) ---
    NEUTRAL_COMPLEXING_AGENTS = {
        'O': 'H2O', 'Cl': 'HCl', 'Br': 'HBr', 'I': 'HI', 'F': 'HF',
        'OS(=O)(=O)O': 'H2SO4', 'O=[N+]([O-])O': 'HNO3', 'OP(=O)(O)O': 'H3PO4',
        'CC(=O)O': 'Acetic Acid', 'C(=O)O': 'Formic Acid',
        'N': 'Ammonia',
        # 结构编辑误操作可能产生的独立单碳组分。
        'C': 'Accidental isolated carbon fragment',
    }

    POTENTIAL_IONIC_SPECIES = {
        # 单原子离子 (符号)
        'FE', 'CU', 'MN', 'CO', 'NI', 'MO', 'V', 'CR', 'ZN', 'MG', 'CA', 'CD', 'HG', 'PB', 'SN',
        'NA', 'K', 'LI', 'RB', 'CS', 'AG',
        'AL', 'GA', 'BI', 'IN', 'TL',
        'EU', 'GD', 'LA', 'YB', 'SM', 'CE', 'SR', 'BA',
        'CL', 'BR', 'I', 'F',
        
        # 多原子离子 (SMILES)
        'O', 'N',
        'O=S(=O)=O', 'O=[N+]=O', 'O=P(O)(O)O', 'O=P(O)O', 
        'O=C=O', 'CC(=O)=O', 'C(#N)[S]', 'O=C(C=O)=O', 
        'O=Cl(=O)=O', 'O=C(CC=O)=O', 'O=C(CCC=O)=O', 
        'O=C(O)C(O)(C=O)C=O', 'O=C([C@H](O)[C@@H](O)C=O)=O', 
        'O=P(O)(O)OCC(O)CO', 'O=P(O)(O)OP(=O)(O)O', 'O=C(C(F)(F)F)=O'
    }

    # --- 2. 辅助函数：判断是否为离子 ---
    def is_known_ion(fragment):
        # 移除氢以匹配标准 SMILES
        temp_mol = Chem.RemoveHs(fragment)
        try:
            # 1. 尝试标准 SMILES 匹配
            smi = Chem.MolToSmiles(temp_mol, canonical=True)
            if smi in POTENTIAL_IONIC_SPECIES or smi in NEUTRAL_COMPLEXING_AGENTS:
                return True, smi
            
            # 2. 尝试单原子符号匹配 (针对带电荷的金属离子，如 [Fe+3])
            if temp_mol.GetNumAtoms() == 1:
                symbol = temp_mol.GetAtomWithIdx(0).GetSymbol().upper()
                if symbol in POTENTIAL_IONIC_SPECIES:
                    return True, symbol
            
            return False, smi
        except:
            return False, ""

    # --- 3. 主流程 ---
    fragments = list(Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False))
    
    if not fragments:
        return [], 0

    retained_parts = []
    removed_count = 0

    for frag in fragments:
        if is_metal_only_fragment(frag):
            if origin_type == "pocket":
                retained_parts.append(frag)
            else:
                removed_count += 1
            continue

        is_ion, _ = is_known_ion(frag)
        if is_ion:
            removed_count += 1
        else:
            retained_parts.append(frag)

    return retained_parts, removed_count


def rule2_separate_metals(fragments):
    """规则2: 无端口断开所有涉及金属的键，并保留单原子金属节点。"""
    organic_output = []
    metal_output = []
    for frag in fragments:
        # 只要任一端是金属原子就断开，包括金属-非金属和金属-金属键；
        # 不限制 SINGLE、配位键、UNSPECIFIED（SMILES 中常显示为 ``~``）
        # 等具体键型。
        bonds_to_cut = [
            bond.GetIdx()
            for bond in frag.GetBonds()
            if (
                is_metal_atom(bond.GetBeginAtom())
                or is_metal_atom(bond.GetEndAtom())
            )
        ]
        if bonds_to_cut:
            rw_mol = Chem.RWMol(frag)
            endpoints = [
                (
                    frag.GetBondWithIdx(idx).GetBeginAtomIdx(),
                    frag.GetBondWithIdx(idx).GetEndAtomIdx(),
                )
                for idx in sorted(set(bonds_to_cut))
            ]
            for begin_idx, end_idx in endpoints:
                rw_mol.RemoveBond(begin_idx, end_idx)
            new_frags = Chem.GetMolFrags(
                rw_mol.GetMol(), asMols=True, sanitizeFrags=False
            )
        else:
            new_frags = (frag,)

        for new_frag in new_frags:
            if get_num_non_dummy_atoms(new_frag) == 0:
                continue
            if is_metal_only_fragment(new_frag):
                real_atom_count = sum(
                    atom.GetAtomicNum() != 0
                    for atom in new_frag.GetAtoms()
                )
                if real_atom_count != 1:
                    raise ValueError(
                        "Rule 2 金属原子化失败，仍得到多原子金属组分"
                    )
                metal_output.append(new_frag)
            else:
                organic_output.append(new_frag)
    return organic_output, metal_output

def rule3_cut_large_rings(fragments, bond_info, track_ports):
    """
    规则3 (增强版): 切割大环。
    逻辑更新：
    1. 优先处理显式的 8 元及以上大环（保持原有逻辑）。
    2. 如果存在“隐式大环”（即键在环上，但不属于任何 < 8 的小环），也将其视为大环的一部分进行切割。
       这解决了如环糊精（Cyclodextrin）等由小环串联成的笼状/大环结构无法被切割的问题。
    """
    output = []
    for frag in fragments:
        # --- 步骤 1: 收集小环信息 ---
        # 获取所有环的键列表
        all_ring_bonds = frag.GetRingInfo().BondRings()
        
        # 记录所有属于 "小环" (size < 8) 的键的索引
        # 如果一个键属于 6元环，它就被保护起来，除非它是大环唯一的切点
        bonds_in_small_rings = set()
        for ring in all_ring_bonds:
            if len(ring) < 8:
                bonds_in_small_rings.update(ring)

        # --- 步骤 2: 寻找显式大环 (原有逻辑) ---
        large_rings_bonds = [ring for ring in all_ring_bonds if len(ring) >= 8]
        
        candidate_to_cut = None # 存储最佳切割键 {'idx': int, 'ecc': float}

        if large_rings_bonds:
            # 策略 A: 存在显式大环 (如 12-crown-4)
            largest_ring_bonds = max(large_rings_bonds, key=len)
            
            # 优先找 d 类键 (连接桥头但不共用)
            candidates = [{'idx': idx, 'ecc': bond_info.get(idx, {}).get('ecc', 999)} 
                          for idx in largest_ring_bonds 
                              if (info := bond_info.get(idx)) and info['is_d']
                              and frag.GetBondWithIdx(idx).GetBondType() == Chem.BondType.SINGLE]
            
            # 其次找普通单环键 (非桥头)
            if not candidates:
                candidates = [{'idx': idx, 'ecc': bond_info.get(idx, {}).get('ecc', 999)} 
                              for idx in largest_ring_bonds 
                              if (info := bond_info.get(idx)) and info['is_a'] and not info['is_b']
                              and frag.GetBondWithIdx(idx).GetBondType() == Chem.BondType.SINGLE]
            
            if candidates:
                candidate_to_cut = min(candidates, key=lambda x: x['ecc'])

        # --- 步骤 3: 寻找隐式大环 (新增逻辑) ---
        # 如果策略 A 没有找到切点 (或者根本没有显式大环)，尝试策略 B
        if not candidate_to_cut:
            implicit_candidates = []
            for bond in frag.GetBonds():
                idx = bond.GetIdx()
                # 核心逻辑：键在环内，但不在任何小环内，且是单键
                if (bond.IsInRing() and 
                    idx not in bonds_in_small_rings and 
                    bond.GetBondType() == Chem.BondType.SINGLE):
                    
                    # 获取该键的偏心率
                    ecc = bond_info.get(idx, {}).get('ecc', 999)
                    implicit_candidates.append({'idx': idx, 'ecc': ecc})
            
            if implicit_candidates:
                # 同样选择最中心的键进行切割
                candidate_to_cut = min(implicit_candidates, key=lambda x: x['ecc'])

        # --- 步骤 4: 执行切割 ---
        if candidate_to_cut:
            try:
                # 执行切割
                newly_fragmented = fragment_on_bonds_by_origin(
                    frag, [candidate_to_cut['idx']], track_ports
                )
                new_frags = Chem.GetMolFrags(newly_fragmented, asMols=True)
                # 过滤掉只含虚原子的碎片
                valid_frags = [f for f in new_frags if get_num_non_dummy_atoms(f) > 0]
                if valid_frags:
                    output.extend(valid_frags)
                else:
                    output.append(frag)
            except PortValidationError:
                raise
            except Exception:
                output.append(frag)
        else:
            # 既没有显式大环，也没有隐式大环结构，保持原样
            output.append(frag)
            
    return output

def rule6_cut_ring_substituents(fragments, track_ports):
    """
    规则6 (修复版): 切割环上取代基。
    
    修复内容：
    在遍历环原子的邻居时，显式排除原子序数为 0 的虚原子 (Dummy Atoms)。
    防止将上一轮切割产生的 '*' 标记误判为新的取代基从而导致无限切割或核心丢失。
    """
    final_fragments = []
    to_process = list(fragments)
    iteration_limit = 100 
    
    while to_process and iteration_limit > 0:
        frag = to_process.pop(0)
        
        if frag.GetRingInfo().NumRings() == 0:
            final_fragments.append(frag)
            continue
            
        bonds_to_cut = set()
        ring_atom_indices = {a.GetIdx() for a in frag.GetAtoms() if a.IsInRing()}
        
        for r_idx in ring_atom_indices:
            r_atom = frag.GetAtomWithIdx(r_idx)
            
            for bond in r_atom.GetBonds():
                neighbor = bond.GetOtherAtom(r_atom)
                
                # [关键修复]：如果邻居是虚原子，直接跳过！
                # 这阻止了对切割位点的二次切割
                if neighbor.GetAtomicNum() == 0:
                    continue
                    
                n_idx = neighbor.GetIdx()
                
                # 情况 A: 内部环键 -> 跳过
                if n_idx in ring_atom_indices and bond.IsInRing():
                    continue
                
                # 情况 B: 环间连接子
                if n_idx in ring_atom_indices and not bond.IsInRing():
                    if bond.GetBondType() == Chem.BondType.SINGLE:
                        bonds_to_cut.add(bond.GetIdx())
                    continue
                
                # 情况 C: 环-取代基 连接
                if n_idx not in ring_atom_indices:
                    # 再次确保取代基不是虚原子 (双重保险)
                    if neighbor.GetAtomicNum() == 0:
                        continue

                    if bond.GetBondType() == Chem.BondType.SINGLE:
                        bonds_to_cut.add(bond.GetIdx())
                    else:
                        target_bond_idx = _bfs_find_nearest_single_bond(frag, n_idx, r_idx)
                        if target_bond_idx is not None:
                            bonds_to_cut.add(target_bond_idx)

        if bonds_to_cut:
            try:
                newly_fragmented = fragment_on_bonds_by_origin(
                    frag, list(bonds_to_cut), track_ports
                )
                new_frags = Chem.GetMolFrags(newly_fragmented, asMols=True)
                valid_frags = [f for f in new_frags if get_num_non_dummy_atoms(f) > 0]
                
                if valid_frags:
                    to_process.extend(valid_frags)
                    iteration_limit -= 1
                else:
                    final_fragments.append(frag)
            except PortValidationError:
                raise
            except Exception:
                final_fragments.append(frag)
        else:
            final_fragments.append(frag)
            
    return final_fragments

def rule8_cut_acyclic_chains(fragments, track_ports):
    """
    规则8 (安全修复版): 切割无环长链。
    
    修复内容：
    在 Check 1 (P2逻辑) 和 Check 2 (杂原子检查) 中，显式排除连接虚原子(AtomicNum=0)的键。
    防止程序试图切断 "C-*" 这种连接标记，避免死循环或错误碎片。
    """
    final_fragments = []
    fragments_to_process = list(fragments)
    processed_smiles = set()

    while fragments_to_process:
        current_fragment = fragments_to_process.pop(0)        
        try:
            # 1. 更新属性缓存，允许 RDKit 重新计算价态
            current_fragment.UpdatePropertyCache(strict=False)            
            # 2. 尝试快速修复 (Fast Sanitize)
            Chem.SanitizeMol(current_fragment, 
                             sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_CLEANUP)
        except Exception:
            pass
        try:
            current_smi = Chem.MolToSmiles(current_fragment, canonical=True)
            if current_smi in processed_smiles:
                final_fragments.append(current_fragment)
                continue
            processed_smiles.add(current_smi)
        except Exception:
            final_fragments.append(current_fragment)
            continue
        
        if current_fragment.GetRingInfo().NumRings() > 0 or get_num_non_dummy_atoms(current_fragment) < 3:
            final_fragments.append(current_fragment)
            continue
            
        # ==================================================================
        # Check 1: 无环链基础逻辑 (基于 Carbon 'a' 的切割)
        # ==================================================================
        bonds_to_cut_this_iteration = set()
        carbon_a_indices = identify_carbon_a_atoms(current_fragment)

        # P1 Bonds: 连接两个 Carbon 'a' 的单键 (优先切断官能团之间的连接)
        p1_bonds = {b.GetIdx() for b in current_fragment.GetBonds() 
                    if b.GetBondType() == Chem.BondType.SINGLE 
                    and b.GetBeginAtomIdx() in carbon_a_indices 
                    and b.GetEndAtomIdx() in carbon_a_indices}
        
        if p1_bonds:
            bonds_to_cut_this_iteration.update(p1_bonds)
        else:
            # P2 Bonds: Carbon 'a' 周围的其他特定键
            p2_bonds = set()
            for a_idx in carbon_a_indices:
                atom_a = current_fragment.GetAtomWithIdx(a_idx)
                
                # --- [逻辑重构] ---
                # 不使用模糊的 is_unsaturated，而是显式检查是否存在刚性双/三键
                has_rigid_unsaturation = False
                for bond in atom_a.GetBonds():
                    if bond.GetBondType() in [Chem.BondType.DOUBLE, Chem.BondType.TRIPLE]:
                        has_rigid_unsaturation = True
                        break
                
                if has_rigid_unsaturation:
                    # === 分支 A: 真正的官能团中心 (如 C=O, C=N, C=C) ===
                    # 这里的逻辑是为了保护官能团不被拆散
                    
                    unsaturated_partner_is_heteroatom = False
                    for bond in atom_a.GetBonds():
                        if bond.GetBondType() in [Chem.BondType.DOUBLE, Chem.BondType.TRIPLE]:
                            partner_atom = bond.GetOtherAtom(atom_a)
                            if partner_atom.GetAtomicNum() not in [6, 1, 0]:
                                unsaturated_partner_is_heteroatom = True
                                break
                    
                    if unsaturated_partner_is_heteroatom:
                        # 场景: 保护 C=O, C=N 等杂原子双键
                        # 策略: 切断连接该碳原子的"碳-碳单键"，保留 C-Het 连接
                        for bond in atom_a.GetBonds():
                            other = bond.GetOtherAtom(atom_a)
                            if other.GetAtomicNum() == 0: continue # 避开虚原子
                            
                            # 只切断连接 Carbon 的单键 (隔离整个官能团)
                            if bond.GetBondType() == Chem.BondType.SINGLE and other.GetAtomicNum() == 6:
                                p2_bonds.add(bond.GetIdx())
                    else:
                        # 场景: 纯碳双键 (C=C)
                        # 策略: 切断该碳原子与非氢真实原子之间的所有单键
                        for bond in atom_a.GetBonds():
                            other = bond.GetOtherAtom(atom_a)
                            if other.GetAtomicNum() in (0, 1):
                                continue
                            
                            if bond.GetBondType() == Chem.BondType.SINGLE:
                                p2_bonds.add(bond.GetIdx())

                else:
                    # === 分支 B: 表面饱和 (或仅含 Aromatic 伪影) 的 Carbon 'a' ===
                    # 用户需求: 只要没有双键/三键，就强制切断与杂原子的连接
                    # 这能有效处理 CC(C)CCNC=O 中的 C-N 键，无论它是 Single 还是 Aromatic
                    
                    for bond in atom_a.GetBonds():
                        other_atom = bond.GetOtherAtom(atom_a)
                        
                        # 1. 安全检查: 忽略虚原子
                        if other_atom.GetAtomicNum() == 0: continue

                        # 2. 目标检查: 对方必须是杂原子 (非C, 非H)
                        if other_atom.GetAtomicNum() not in [6, 1, 0]:
                            
                            # 3. 动作: 切断连接
                            # 端口重连统一只学习单键。
                            if bond.GetBondType() == Chem.BondType.SINGLE:
                                p2_bonds.add(bond.GetIdx())
            
            bonds_to_cut_this_iteration.update(p2_bonds)

        if bonds_to_cut_this_iteration:
            newly_fragmented_mol = fragment_on_bonds_by_origin(
                current_fragment, list(bonds_to_cut_this_iteration), track_ports
            )
            new_frags = Chem.GetMolFrags(
                newly_fragmented_mol,
                asMols=True,
                sanitizeFrags=False,
            )
            fragments_to_process.extend([frag for frag in new_frags if get_num_non_dummy_atoms(frag) > 0])
            continue

        # ==================================================================
        # Check 2: 新增逻辑 (杂原子长链检查)
        # ==================================================================
        
        heteroatom_count = sum(1 for a in current_fragment.GetAtoms() if a.GetAtomicNum() not in (6, 1, 0))
        
        if heteroatom_count >= 5:
            try:
                d_mat = GetDistanceMatrix(current_fragment)
                atom_eccentricities = d_mat.max(axis=1)
                
                best_cut_bond_idx = None
                min_bond_ecc = float('inf')
                
                for bond in current_fragment.GetBonds():
                    # 端口重建只切非氢真实原子之间的键。
                    begin_atomic_number = bond.GetBeginAtom().GetAtomicNum()
                    end_atomic_number = bond.GetEndAtom().GetAtomicNum()
                    if (
                        begin_atomic_number in (0, 1)
                        or end_atomic_number in (0, 1)
                    ):
                        continue

                    if bond.GetBondType() == Chem.BondType.SINGLE:
                        b_idx = bond.GetBeginAtomIdx()
                        e_idx = bond.GetEndAtomIdx()
                        bond_ecc = (atom_eccentricities[b_idx] + atom_eccentricities[e_idx]) / 2.0
                        
                        if bond_ecc < min_bond_ecc:
                            min_bond_ecc = bond_ecc
                            best_cut_bond_idx = bond.GetIdx()
                
                if best_cut_bond_idx is not None:
                    newly_fragmented = fragment_on_bonds_by_origin(
                        current_fragment, [best_cut_bond_idx], track_ports
                    )
                    new_frags = Chem.GetMolFrags(
                        newly_fragmented,
                        asMols=True,
                        sanitizeFrags=False,
                    )
                    fragments_to_process.extend([frag for frag in new_frags if get_num_non_dummy_atoms(frag) > 0])
                    continue

            except PortValidationError:
                raise
            except Exception:
                pass

        final_fragments.append(current_fragment)
            
    return final_fragments

def rule4_cut_spiro_centers(fragments, main_mol_for_matching, bond_info, track_ports):
    """规则4: 切割螺环中心。将通过单个螺原子连接的两个环分离开。"""
    to_process = list(fragments)
    while True:
        cut_made, next_process, processed = False, [], []
        for frag in to_process:
            best_cut = {'bonds': [], 'priority_score': float('inf')}
            clean = Chem.RWMol(frag); [clean.RemoveAtom(i) for i in sorted([a.GetIdx() for a in clean.GetAtoms() if a.GetAtomicNum() == 0], reverse=True)]
            match = main_mol_for_matching.GetSubstructMatch(clean)
            if not match: processed.append(frag); continue
            
            b_head_atoms = {idx for b_idx, info in bond_info.items() if info['is_b'] for idx in (main_mol_for_matching.GetBondWithIdx(b_idx).GetBeginAtomIdx(), main_mol_for_matching.GetBondWithIdx(b_idx).GetEndAtomIdx()) if idx in match}
            spiro_atoms = [a for a in frag.GetAtoms() if a.GetDegree() >= 4 and frag.GetRingInfo().NumAtomRings(a.GetIdx()) >= 2 and (match[a.GetIdx()] if a.GetIdx() < len(match) else -1) not in b_head_atoms]
            if not spiro_atoms: processed.append(frag); continue
            
            for atom in spiro_atoms:
                single_bonds = [
                    bond for bond in atom.GetBonds()
                    if bond.GetBondType() == Chem.BondType.SINGLE
                ]
                for pair in combinations(single_bonds, 2):
                    indices = [b.GetIdx() for b in pair]
                    temp_frag_mol = Chem.FragmentOnBonds(frag, indices, addDummies=True)
                    subs = list(Chem.GetMolFrags(temp_frag_mol, asMols=True))
                    if len(subs) == 2:
                        score = abs(rdMolDescriptors.CalcExactMolWt(subs[0]) - rdMolDescriptors.CalcExactMolWt(subs[1]))
                        if score < best_cut['priority_score']:
                            best_cut = {'bonds': indices, 'priority_score': score}
            
            if best_cut['bonds']:
                newly_fragmented = fragment_on_bonds_by_origin(
                    frag, best_cut['bonds'], track_ports
                )
                new_frags = Chem.GetMolFrags(newly_fragmented, asMols=True)
                next_process.extend([f for f in new_frags if get_num_non_dummy_atoms(f) > 0])
                cut_made = True
            else: processed.append(frag)
        
        to_process = processed + next_process
        if not cut_made: break
    return to_process

def rule5_cut_multi_ring_fragments(fragments, main_mol_for_matching, bond_info, track_ports):
    """规则5: 切割多环片段。处理复杂的稠环或桥环体系。"""
    to_process = list(fragments)
    while True:
        cut_made, next_process, processed = False, [], []
        for frag in to_process:
            if frag.GetRingInfo().NumRings() < 3:
                processed.append(frag); continue
            
            clean = Chem.RWMol(frag); [clean.RemoveAtom(i) for i in sorted([a.GetIdx() for a in clean.GetAtoms() if a.GetAtomicNum() == 0], reverse=True)]
            match = main_mol_for_matching.GetSubstructMatch(clean)
            if not match: processed.append(frag); continue
            
            cand_rings = find_type_a_rings(frag, main_mol_for_matching, match, bond_info)
            if not cand_rings: processed.append(frag); continue
            
            ecc_frag = GetDistanceMatrix(frag).max(axis=1)
            best_ring = min(cand_rings, key=lambda r: np.mean([ecc_frag[idx] for idx in r['atom_indices']]))
            
            if best_ring:
                map_o2f = {orig_idx: frag_idx for frag_idx, orig_idx in enumerate(match)}
                frag_indices = [b.GetIdx() for o_idx in best_ring['orig_d_bond_indices'] if (o_bond := main_mol_for_matching.GetBondWithIdx(o_idx)) and (f_a1 := map_o2f.get(o_bond.GetBeginAtomIdx())) is not None and (f_a2 := map_o2f.get(o_bond.GetEndAtomIdx())) is not None and (b := frag.GetBondBetweenAtoms(f_a1, f_a2)) and b.GetBondType() == Chem.BondType.SINGLE]
                if frag_indices:
                    newly_fragmented = fragment_on_bonds_by_origin(
                        frag, frag_indices, track_ports
                    )
                    new_frags = Chem.GetMolFrags(newly_fragmented, asMols=True)
                    next_process.extend([f for f in new_frags if get_num_non_dummy_atoms(f) > 0])
                    cut_made = True
                else: processed.append(frag)
            else: processed.append(frag)
        to_process = processed + next_process
        if not cut_made: break
    return to_process

def rule9_cut_large_carbon_fragments(fragments, main_mol_for_matching, track_ports):
    """规则9: 切割大的纯碳片段。处理残留的、大于6个碳的脂肪片段。"""
    final_fragments = []
    to_process = list(fragments)
    while to_process:
        frag = to_process.pop(0)
        
        is_all_carbon = all(a.GetAtomicNum() == 6 for a in frag.GetAtoms() if a.GetAtomicNum() != 0)
        num_carbons = sum(1 for a in frag.GetAtoms() if a.GetAtomicNum() == 6)
        
        if not is_all_carbon or num_carbons < 4:
            final_fragments.append(frag); continue
            
        best_cut = {'bond_idx': -1, 'mass_diff': float('inf'), 'avg_idx': float('inf')}
        clean_frag = Chem.RWMol(frag); [clean_frag.RemoveAtom(i) for i in sorted([a.GetIdx() for a in clean_frag.GetAtoms() if a.GetAtomicNum() == 0], reverse=True)]
        match_indices = main_mol_for_matching.GetSubstructMatch(clean_frag.GetMol())
        if not match_indices:
            final_fragments.append(frag); continue

        for bond in frag.GetBonds():
            if bond.IsInRing() or bond.GetBondType() != Chem.BondType.SINGLE: continue
            
            try:
                temp_frag_mol = Chem.FragmentOnBonds(frag, [bond.GetIdx()], addDummies=True)
                subs = list(Chem.GetMolFrags(temp_frag_mol, asMols=True))
                if len(subs) == 2:
                    mass_diff = abs(rdMolDescriptors.CalcExactMolWt(subs[0]) - rdMolDescriptors.CalcExactMolWt(subs[1]))
                    avg_idx = (match_indices[bond.GetBeginAtomIdx()] + match_indices[bond.GetEndAtomIdx()]) / 2.0
                    
                    if mass_diff < best_cut['mass_diff'] or (mass_diff == best_cut['mass_diff'] and avg_idx < best_cut['avg_idx']):
                        best_cut = {'bond_idx': bond.GetIdx(), 'mass_diff': mass_diff, 'avg_idx': avg_idx}
            except Exception:
                continue

        if best_cut['bond_idx'] != -1:
            newly_fragmented = fragment_on_bonds_by_origin(
                frag, [best_cut['bond_idx']], track_ports
            )
            new_frags = Chem.GetMolFrags(newly_fragmented, asMols=True)
            to_process.extend([f for f in new_frags if get_num_non_dummy_atoms(f) > 0])
        else:
            final_fragments.append(frag)
    return final_fragments

def rule7_cut_acid_like_groups(fragments, track_ports):
    """
    规则7: 切割含多个类酸官能团的片段。
    """
    final_fragments = []
    to_process = list(fragments)

    while to_process:
        frag = to_process.pop(0)

        def is_target_heteroatom(atom):
            if atom.GetAtomicNum() == 0: return False # 虚原子不是目标
            if atom.GetSymbol() not in ['P', 'S', 'N', 'B', 'C', 'Si', 'Se', 'Mn', 'As', 'Cr', 'Cl', 'Br', 'I']:
                return False
            # 检查双键氧和单键氧 (忽略连接到虚原子的键)
            has_double_bond_O = False
            has_single_bond_O = False
            for b in atom.GetBonds():
                other = b.GetOtherAtom(atom)
                if other.GetAtomicNum() == 0: continue # 忽略虚原子
                if other.GetAtomicNum() == 8:
                    if b.GetBondType() == Chem.BondType.DOUBLE: has_double_bond_O = True
                    if b.GetBondType() == Chem.BondType.SINGLE: has_single_bond_O = True
            
            return has_double_bond_O and has_single_bond_O

        current_frag_to_process = frag
        while True:
            target_atoms = [a for a in current_frag_to_process.GetAtoms() if is_target_heteroatom(a)]
            if len(target_atoms) < 2:
                final_fragments.append(current_frag_to_process)
                break

            candidate_bonds = []
            try:
                ecc_frag = GetDistanceMatrix(current_frag_to_process).max(axis=1)
            except:
                final_fragments.append(current_frag_to_process)
                break

            target_atom_indices = {a.GetIdx() for a in target_atoms}

            for atom in target_atoms:
                for bond in atom.GetBonds():
                    if not (bond.GetBondType() == Chem.BondType.SINGLE and not bond.IsInRing()): continue
                    
                    bridge_atom = bond.GetOtherAtom(atom)
                    
                    # [修复点 1] 忽略虚原子桥
                    if bridge_atom.GetAtomicNum() == 0: continue

                    is_bridge = False
                    for neighbor_of_bridge in bridge_atom.GetNeighbors():
                        if neighbor_of_bridge.GetAtomicNum() == 0: continue # 忽略虚原子邻居
                        
                        if neighbor_of_bridge.GetIdx() in target_atom_indices and neighbor_of_bridge.GetIdx() != atom.GetIdx():
                            is_bridge = True
                            break
                    if is_bridge:
                        bond_ecc = (ecc_frag[bond.GetBeginAtomIdx()] + ecc_frag[bond.GetEndAtomIdx()]) / 2.0
                        candidate_bonds.append({'idx': bond.GetIdx(), 'ecc': bond_ecc})
            
            # 如果没找到 P-O-P 这种桥，找 P-O-C
            if not candidate_bonds:
                for atom in target_atoms:
                    for bond in atom.GetBonds():
                        other_atom = bond.GetOtherAtom(atom)
                        # [修复点 2] 忽略虚原子
                        if other_atom.GetAtomicNum() == 0: continue
                        
                        if bond.GetBondType() == Chem.BondType.SINGLE and not bond.IsInRing():
                            if other_atom.GetAtomicNum() == 6:
                                bond_ecc = (ecc_frag[bond.GetBeginAtomIdx()] + ecc_frag[bond.GetEndAtomIdx()]) / 2.0
                                candidate_bonds.append({'idx': bond.GetIdx(), 'ecc': bond_ecc})

            if not candidate_bonds:
                final_fragments.append(current_frag_to_process)
                break

            bond_to_cut = min(candidate_bonds, key=lambda x: x['ecc'])
            
            try:
                newly_fragmented = fragment_on_bonds_by_origin(
                    current_frag_to_process, [bond_to_cut['idx']], track_ports
                )
                new_frags_list = Chem.GetMolFrags(newly_fragmented, asMols=True)
                filtered_frags = [f for f in new_frags_list if get_num_non_dummy_atoms(f) > 0]
                
                if filtered_frags:
                    filtered_frags.sort(key=get_num_non_dummy_atoms, reverse=True)
                    current_frag_to_process = filtered_frags.pop(0)
                    to_process.extend(filtered_frags)
                else:
                    final_fragments.append(current_frag_to_process)
                    break
            except PortValidationError:
                raise
            except Exception:
                final_fragments.append(current_frag_to_process)
                break

    return final_fragments


# ----------------------------------------
# PDB 解析与口袋截取
# ----------------------------------------

# 已知溶剂/缓冲液/离子残基名称（从配体候选中排除）
KNOWN_SOLVENT_RESIDUES = {
    'HOH', 'WAT', 'H2O', 'DOD', 'D2O',
    'SO4', 'PO4', 'NO3', 'CO3', 'CL', 'BR', 'IOD', 'FLC',
    'GOL', 'EDO', 'PEG', 'PGE', 'MPD', 'DMS', 'ACE', 'NH2',
    'BME', 'MES', 'TRS', 'EPE', 'FMT', 'ACT', 'IMD', 'IPA',
    'CA', 'ZN', 'MG', 'NA', 'K', 'MN', 'FE', 'NI', 'CU', 'CO',
    'CD', 'HG', 'PB', 'SR', 'BA', 'CS', 'RB', 'LI'
}


def parse_hetatm_residues(pdb_path, exclude_residues=None):
    """
    解析 PDB 文件，识别所有非溶剂的 HETATM 残基作为配体候选。
    同时收集所有 ATOM（蛋白）记录以备后用。

    Args:
        pdb_path: PDB 文件路径
        exclude_residues: 要排除的残基名称集合，默认使用 KNOWN_SOLVENT_RESIDUES

    Returns:
        ligand_candidates: list[dict]，每个元素包含:
            - 'resName', 'chainID', 'resSeq', 'label'
            - 'atom_lines': 原始 PDB 行列表
            - 'coords': np.ndarray (N, 3)
        protein_lines: list[str]，所有 ATOM 记录行
    """
    if exclude_residues is None:
        exclude_residues = KNOWN_SOLVENT_RESIDUES

    hetatm_groups = {}  # key=(chainID, resName, resSeq) -> {'lines':[], 'coords':[]}
    protein_lines = []

    with open(pdb_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            record = line[:6].strip()
            if record == 'ATOM':
                protein_lines.append(line)
            elif record == 'HETATM':
                try:
                    resName = line[17:20].strip()
                    chainID = line[21].strip() if len(line) > 21 else ''
                    resSeq = line[22:26].strip()
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                except (ValueError, IndexError):
                    continue

                key = (chainID, resName, resSeq)
                if key not in hetatm_groups:
                    hetatm_groups[key] = {'lines': [], 'coords': []}
                hetatm_groups[key]['lines'].append(line)
                hetatm_groups[key]['coords'].append([x, y, z])

    # 过滤：排除溶剂/离子
    ligand_candidates = []
    for (chainID, resName, resSeq), data in hetatm_groups.items():
        if resName.upper() in exclude_residues:
            continue
        # 至少2个原子才可能是配体（排除单原子金属残基）
        if len(data['coords']) < 2:
            continue
        ligand_candidates.append({
            'resName': resName,
            'chainID': chainID,
            'resSeq': resSeq,
            'label': f"{resName}:{chainID}:{resSeq} ({len(data['coords'])} atoms)",
            'atom_lines': data['lines'],
            'coords': np.array(data['coords'])
        })

    return ligand_candidates, protein_lines


def extract_pocket_around_ligand(pdb_path, ligand_atom_coords, protein_lines=None, radius=10.0):
    """
    根据配体原子坐标，从蛋白 ATOM 记录中截取 radius 范围内的完整残基。

    Args:
        pdb_path: PDB 文件路径（当 protein_lines 为 None 时用于读取）
        ligand_atom_coords: np.ndarray (M, 3)，配体原子坐标
        protein_lines: list[str]，预解析的 ATOM 行（可选）
        radius: 截取半径（Å），默认 10.0

    Returns:
        pocket_mol: RDKit Mol（口袋分子，已做容错处理）
        pocket_pdb_lines: list[str]，用于保存的 PDB 行
    """
    import scipy.spatial.distance

    # 如果没有预传入蛋白行，从文件读取
    if protein_lines is None:
        protein_lines = []
        with open(pdb_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if line[:6].strip() == 'ATOM':
                    protein_lines.append(line)

    # 按残基分组 (chainID, resSeq, iCode)
    residue_groups = {}
    for line in protein_lines:
        try:
            chainID = line[21] if len(line) > 21 else ''
            resSeq = line[22:26].strip()
            iCode = line[26] if len(line) > 26 else ''
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
        except (ValueError, IndexError):
            continue

        key = (chainID, resSeq, iCode)
        if key not in residue_groups:
            residue_groups[key] = {'lines': [], 'coords': []}
        residue_groups[key]['lines'].append(line)
        residue_groups[key]['coords'].append([x, y, z])

    # 判断每个残基是否在 radius 范围内
    pocket_lines = []
    for key, data in residue_groups.items():
        res_coords = np.array(data['coords'])
        dists = scipy.spatial.distance.cdist(res_coords, ligand_atom_coords)
        if dists.min() < radius:
            pocket_lines.extend(data['lines'])

    if not pocket_lines:
        return None, []

    # 用 PDB block 构建 RDKit Mol
    pdb_block = ''.join(pocket_lines) + 'END\n'
    pocket_mol = Chem.MolFromPDBBlock(pdb_block, removeHs=True, sanitize=False)

    if pocket_mol is not None:
        pocket_mol = advanced_rescue_valence_errors(pocket_mol)
        if pocket_mol is not None:
            try:
                pocket_mol.UpdatePropertyCache(strict=False)
                Chem.SanitizeMol(pocket_mol,
                                 sanitizeOps=Chem.SANITIZE_SYMMRINGS |
                                             Chem.SANITIZE_SETCONJUGATION |
                                             Chem.SANITIZE_SETHYBRIDIZATION)
            except:
                pass

    return pocket_mol, pocket_lines


def build_ligand_mol_from_pdb_lines(ligand_lines):
    """
    从 HETATM 行构建配体的 RDKit Mol 对象。
    CONECT 只用于确定连接关系；键级统一由 OpenBabel 感知并映射回
    RDKit 分子，以免只有连接信息的 PDB 将芳香环误写成全单键环。
    """
    pdb_block = ''.join(ligand_lines) + 'END\n'

    # 有 CONECT 时禁止按距离补键，但不能把“存在 CONECT”等同于
    # “CONECT 已完整编码键级”；许多 PDB 的 CONECT 只记录连接关系。
    has_conect = any(line[:6].strip() == "CONECT" for line in ligand_lines)
    mol = Chem.MolFromPDBBlock(
        pdb_block,
        removeHs=True,
        sanitize=False,
        proximityBonding=not has_conect,
    )
    if mol is None:
        return None

    # 用 OpenBabel 感知键级，但保留 RDKit/PDB 已确定的连接关系、坐标和
    # 原子顺序。映射失败时必须中止，不能静默保留全单键拓扑。
    try:
        from openbabel import openbabel as ob
        ob_conv = ob.OBConversion()
        ob_conv.SetInFormat("pdb")
        ob_conv.SetOutFormat("sdf")
        ob_mol = ob.OBMol()
        ob_conv.ReadString(ob_mol, pdb_block)
        ob_mol.PerceiveBondOrders()
        sdf_block = ob_conv.WriteString(ob_mol)

        ref_mol = Chem.MolFromMolBlock(sdf_block, removeHs=True, sanitize=False)
        if ref_mol is None:
            raise ValueError("OpenBabel 未返回可解析的 SDF 分子")
        if ref_mol.GetNumAtoms() != mol.GetNumAtoms():
            raise ValueError(
                "OpenBabel 感知键级后改变了配体原子数量: "
                f"{mol.GetNumAtoms()} -> {ref_mol.GetNumAtoms()}"
            )

        for atom_index in range(mol.GetNumAtoms()):
            source_atomic_number = mol.GetAtomWithIdx(
                atom_index
            ).GetAtomicNum()
            inferred_atomic_number = ref_mol.GetAtomWithIdx(
                atom_index
            ).GetAtomicNum()
            if source_atomic_number != inferred_atomic_number:
                raise ValueError(
                    "OpenBabel 感知键级后改变了配体原子顺序: "
                    f"index={atom_index}, "
                    f"RDKit={source_atomic_number}, "
                    f"OpenBabel={inferred_atomic_number}"
                )

        rw = Chem.RWMol(mol)
        for bond in rw.GetBonds():
            idx1 = bond.GetBeginAtomIdx()
            idx2 = bond.GetEndAtomIdx()
            ref_bond = ref_mol.GetBondBetweenAtoms(idx1, idx2)
            if ref_bond is None:
                raise ValueError(
                    "OpenBabel 感知结果缺少 PDB 已记录的连接: "
                    f"{idx1}-{idx2}"
                )
            bond.SetBondType(ref_bond.GetBondType())
            bond.SetIsAromatic(ref_bond.GetIsAromatic())

        for atom_index in range(rw.GetNumAtoms()):
            rw.GetAtomWithIdx(atom_index).SetIsAromatic(
                ref_mol.GetAtomWithIdx(atom_index).GetIsAromatic()
            )
        mol = rw.GetMol()
    except Exception as exc:
        raise ValueError(
            "无法为 PDB 参考配体恢复可靠键级，已停止处理以避免将"
            "芳香环静默保存为全单键环: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    # 正常的抢救和 sanitize 流程
    mol = advanced_rescue_valence_errors(mol)
    if mol is not None:
        try:
            mol.UpdatePropertyCache(strict=False)
            Chem.SanitizeMol(mol,
                            sanitizeOps=Chem.SANITIZE_SYMMRINGS |
                                        Chem.SANITIZE_SETCONJUGATION |
                                        Chem.SANITIZE_SETHYBRIDIZATION |
                                        Chem.SANITIZE_SETAROMATICITY)
        except Exception as exc:
            raise ValueError(
                "PDB 参考配体恢复键级后无法通过 RDKit 净化: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    return mol

# ----------------------------------------
# Kabsch 反映射与 PyG 节点操作
# ----------------------------------------

def kabsch_align_fragment_to_reference(smiles, target_centroid, target_ref_coords):
    """
    Kabsch 反映射：从 SMILES 生成 3D conformer，对齐到存储的参考坐标系。

    原理：
      正向编码: mol -> centroid + ref_coords[A,B,C] + SMILES
      反向解码: SMILES + centroid + ref_coords -> aligned 3D mol

    步骤：
      1. SMILES -> 3D conformer (EmbedMolecule)
      2. 计算 conformer 的 canonical 参考系和质心
      3. Kabsch SVD 求旋转矩阵 R
      4. 所有原子: new_pos = R @ (pos - centroid_gen) + target_centroid

    Args:
        smiles: 碎片的 canonical SMILES
        target_centroid: (3,) 目标质心坐标
        target_ref_coords: (3, 3) 目标框架坐标 [A, B, C]

    Returns:
        RDKit Mol（带对齐后的 3D conformer）或 None
    """
    from rdkit.Geometry import Point3D

    target_centroid = np.asarray(target_centroid, dtype=np.float64)
    target_ref_coords = np.asarray(target_ref_coords, dtype=np.float64)

    # 清理 SMILES（去除离子后缀标记）
    clean_smiles = smiles.rstrip('*')
    if clean_smiles in ('<UNK>', '<ERROR>', '') or not clean_smiles:
        return None

    try:
        mol = Chem.MolFromSmiles(clean_smiles)
        if not mol:
            return None
        mol = Chem.AddHs(mol)

        # 生成 3D conformer
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        result = AllChem.EmbedMolecule(mol, params)
        if result != 0:
            result = AllChem.EmbedMolecule(mol, randomSeed=42)
            if result != 0:
                return None

        try:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
        except:
            pass

        mol = Chem.RemoveHs(mol)

        if mol.GetNumConformers() == 0:
            return None

        num_heavy = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() > 1)

        # --- 单原子：直接放在目标质心 ---
        if num_heavy <= 1:
            conf = mol.GetConformer(0)
            for i in range(mol.GetNumAtoms()):
                conf.SetAtomPosition(i, Point3D(
                    float(target_centroid[0]),
                    float(target_centroid[1]),
                    float(target_centroid[2])
                ))
            return mol

        # --- 多原子：Kabsch 对齐 ---
        gen_centroid = np.array(calculate_centroid(mol))
        gen_ref = calculate_reference_coordinate_system(mol)  # (3, 3)

        if gen_ref is None:
            return None

        # 中心化
        gen_ref_centered = gen_ref - gen_centroid
        target_ref_centered = target_ref_coords - target_centroid

        # Kabsch SVD
        H = gen_ref_centered.T @ target_ref_centered
        U, S, Vt = np.linalg.svd(H)

        # 处理反射
        d = np.linalg.det(Vt.T @ U.T)
        sign_matrix = np.diag([1.0, 1.0, d])
        R = Vt.T @ sign_matrix @ U.T

        # 应用变换
        conf = mol.GetConformer(0)
        for i in range(mol.GetNumAtoms()):
            pos = np.array(list(conf.GetAtomPosition(i)))
            new_pos = R @ (pos - gen_centroid) + target_centroid
            conf.SetAtomPosition(i, Point3D(
                float(new_pos[0]), float(new_pos[1]), float(new_pos[2])
            ))

        return mol

    except Exception:
        return None


def remove_nodes_from_pyg_data(data, nodes_to_delete):
    """
    从 PyG Data 中移除指定节点及其关联的边，并重新映射边索引。

    Args:
        data: torch_geometric.data.Data
        nodes_to_delete: set[int]，要删除的节点索引集合

    Returns:
        新的 Data 对象
    """
    import torch
    from torch_geometric.data import Data

    total_nodes = data.x.size(0)
    kept_indices = sorted(set(range(total_nodes)) - set(nodes_to_delete))

    if not kept_indices:
        return None

    old_to_new = {old: new for new, old in enumerate(kept_indices)}

    kept_tensor = torch.tensor(kept_indices, dtype=torch.long)
    new_x = data.x[kept_tensor]
    new_pos = data.pos[kept_tensor] if hasattr(data, 'pos') and data.pos is not None else None
    new_frag_embeds = data.frag_embeds[kept_tensor] if hasattr(data, 'frag_embeds') and data.frag_embeds is not None else None
    new_ref_coords = data.ref_coords[kept_tensor] if hasattr(data, 'ref_coords') and data.ref_coords is not None else None
    new_node_type = data.node_type[kept_tensor] if hasattr(data, 'node_type') and data.node_type is not None else None
    new_metal_mask = data.metal_mask[kept_tensor] if hasattr(data, 'metal_mask') and data.metal_mask is not None else None
    new_hac = data.hac[kept_tensor] if hasattr(data, 'hac') and data.hac is not None else None
    new_ring_count = data.ring_count[kept_tensor] if hasattr(data, 'ring_count') and data.ring_count is not None else None
    connection_edge_index = (
        data.connection_edge_index
        if hasattr(data, 'connection_edge_index')
        and data.connection_edge_index is not None
        else None
    )
    connection_atom_ids = (
        data.connection_atom_ids
        if hasattr(data, 'connection_atom_ids')
        and data.connection_atom_ids is not None
        else None
    )
    connection_distance = (
        data.connection_distance
        if hasattr(data, 'connection_distance')
        and data.connection_distance is not None
        else None
    )
    connection_fields = (
        connection_edge_index,
        connection_atom_ids,
        connection_distance,
    )
    if any(field is not None for field in connection_fields) and not all(
        field is not None for field in connection_fields
    ):
        raise ValueError('连接节点、原子编号和距离字段必须同时存在')

    new_connection_edge_index = None
    new_connection_atom_ids = None
    new_connection_distance = None
    if connection_edge_index is not None:
        if connection_edge_index.dim() != 2 or connection_edge_index.size(0) != 2:
            raise ValueError('connection_edge_index 形状必须为 [2, E]')
        num_connections = connection_edge_index.size(1)
        if tuple(connection_atom_ids.shape) != (num_connections, 2):
            raise ValueError('connection_atom_ids 形状必须为 [E, 2]')
        if tuple(connection_distance.shape) != (num_connections,):
            raise ValueError('connection_distance 形状必须为 [E]')

        connection_mask = torch.tensor(
            [
                int(first_node) in old_to_new and int(second_node) in old_to_new
                for first_node, second_node
                in connection_edge_index.t().tolist()
            ],
            dtype=torch.bool,
            device=connection_edge_index.device,
        )
        new_connection_edge_index = connection_edge_index[
            :, connection_mask
        ].clone()
        for connection_idx in range(new_connection_edge_index.size(1)):
            first_node = int(new_connection_edge_index[0, connection_idx])
            second_node = int(new_connection_edge_index[1, connection_idx])
            new_connection_edge_index[0, connection_idx] = old_to_new[first_node]
            new_connection_edge_index[1, connection_idx] = old_to_new[second_node]
        new_connection_atom_ids = connection_atom_ids[connection_mask].clone()
        new_connection_distance = connection_distance[connection_mask].clone()

    if data.edge_index is not None and data.edge_index.size(1) > 0:
        edge_mask = torch.zeros(data.edge_index.size(1), dtype=torch.bool)
        for i in range(data.edge_index.size(1)):
            u = data.edge_index[0, i].item()
            v = data.edge_index[1, i].item()
            if u in old_to_new and v in old_to_new:
                edge_mask[i] = True

        new_edge_index = data.edge_index[:, edge_mask].clone()
        for i in range(new_edge_index.size(1)):
            new_edge_index[0, i] = old_to_new[new_edge_index[0, i].item()]
            new_edge_index[1, i] = old_to_new[new_edge_index[1, i].item()]

        new_edge_attr = data.edge_attr[edge_mask] if hasattr(data, 'edge_attr') and data.edge_attr is not None else None
    else:
        new_edge_index = torch.empty((2, 0), dtype=torch.long)
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            edge_feature_dim = data.edge_attr.size(1) if data.edge_attr.dim() == 2 else 3
            new_edge_attr = torch.empty(
                (0, edge_feature_dim), dtype=data.edge_attr.dtype
            )
        else:
            new_edge_attr = None

    new_data = Data(
        x=new_x,
        edge_index=new_edge_index,
        edge_attr=new_edge_attr,
        pdb_id=data.pdb_id if hasattr(data, 'pdb_id') else None,
        sample_id=data.sample_id if hasattr(data, 'sample_id') else None,
        source=data.source if hasattr(data, 'source') else None,
        meb_center=data.meb_center if hasattr(data, 'meb_center') else None,
        n_pocket_nodes=data.n_pocket_nodes if hasattr(data, 'n_pocket_nodes') else None, # <-- 新增这一行
    )
    if new_pos is not None:
        new_data.pos = new_pos
    if new_frag_embeds is not None:
        new_data.frag_embeds = new_frag_embeds
    if new_ref_coords is not None:
        new_data.ref_coords = new_ref_coords
    if new_node_type is not None:
        new_data.node_type = new_node_type
    if new_metal_mask is not None:
        new_data.metal_mask = new_metal_mask
    if new_hac is not None:
        new_data.hac = new_hac
    if new_ring_count is not None:
        new_data.ring_count = new_ring_count
    if new_connection_edge_index is not None:
        new_data.connection_edge_index = new_connection_edge_index
        new_data.connection_atom_ids = new_connection_atom_ids
        new_data.connection_distance = new_connection_distance

    return new_data
