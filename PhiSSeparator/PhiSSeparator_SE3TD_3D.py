# =============================================================================
# 单元格 1: 初始化、导入模块、参数设定与模型加载
# =============================================================================

import os
import gc
import json
import math
import pickle
import multiprocessing
import tempfile
from itertools import combinations

# --- 防止多进程与底层多线程冲突 (超算必备) ---
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import pandas as pd
import numpy as np
import torch

from rdkit import Chem, RDLogger
from torch_geometric.data import Data
from tqdm import tqdm

import utils

# --- 限制 PyTorch 内部线程数 ---
torch.set_num_threads(1)

# --- 固定随机种子 ---
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
print(f"全局随机种子已固定为: {RANDOM_SEED}")

# --- 获取超算可用的 CPU 核心数 ---
# 如果使用 Slurm 调度系统，自动读取分配的核心数
# 获取物理核心数，但强行限制最高并发数（根据你的节点总内存大小调整）
max_allowed_workers = 16
detected_cores = int(os.environ.get("SLURM_CPUS_PER_TASK", multiprocessing.cpu_count()))
NUM_WORKERS = min(detected_cores, max_allowed_workers)


# =============================================================================
#                            超参数与文件路径
# =============================================================================

NUM_COMPLEXES_TO_PROCESS = None
LL_PP_CENTROID_CUTOFF = 5.0
LP_CENTROID_CUTOFF = 7.0
PROPERTIES_TO_EXCLUDE_FOR_NORM = {'centroid_3d', 'ref_coord_3d', 'Shard ID', 'hac', 'ring_count'}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if not os.path.isdir(os.path.join(PROJECT_ROOT, 'Datas')):
    raise RuntimeError(
        f'无法定位项目数据目录：{os.path.join(PROJECT_ROOT, "Datas")}'
    )

PROCESSED_DATA_ROOT = os.path.join(PROJECT_ROOT, 'Datas', 'Processed_datas')
CLEAN_DATA_ROOT = os.path.join(PROCESSED_DATA_ROOT, 'HiQBind_PDBbind_clean')
ACCEPTED_MANIFEST = os.path.join(CLEAN_DATA_ROOT, 'accepted_manifest.csv')
COMPLEXES_DIR = os.path.join(CLEAN_DATA_ROOT, 'complexes')

output_dir = os.path.join(PROCESSED_DATA_ROOT, 'SE3TD_HiQBind_PDBbind_128_Pygdata')
PYG_DATA_DIR = os.path.join(output_dir, 'pyg_pending')
LOG_DIR = os.path.join(output_dir, 'logs')
SHARED_DATA_DIR = os.path.join(PROCESSED_DATA_ROOT, 'SE3TD_128_Shared_data')
os.makedirs(PYG_DATA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(SHARED_DATA_DIR, exist_ok=True)

RAW_CORPUS_FILE = os.path.join(SHARED_DATA_DIR, 'fragment_raw_corpus.npz')
RAW_CORPUS_METADATA_FILE = os.path.join(SHARED_DATA_DIR, 'fragment_raw_corpus_metadata.json')
RAW_ATOM_CORPUS_FILE = os.path.join(
    SHARED_DATA_DIR, 'fragment_atom_raw_corpus.npz'
)
RAW_ATOM_CORPUS_METADATA_FILE = os.path.join(
    SHARED_DATA_DIR, 'fragment_atom_raw_corpus_metadata.json'
)
NORM_STATS_FILE = os.path.join(SHARED_DATA_DIR, 'normalization_stats.json')
SUMMARY_FILE = os.path.join(output_dir, 'processing_summary.json')
POISON_BLACKLIST_FILE = os.path.join(LOG_DIR, 'poison_molecules_blacklist.txt')

print(f"当前工作目录: {os.getcwd()}")
print(f"数据输入根目录: {os.path.abspath(CLEAN_DATA_ROOT)}")
print(f"数据输出目录: {os.path.abspath(output_dir)}")
print(f"共享数据目录: {os.path.abspath(SHARED_DATA_DIR)}")

FEATURE_COLUMNS = [
    'avg_ecc', 'ecc_range',
    'avg_logp', 'tpsa', 'avg_charge', 
    'HBA', 'HBD', 'Aromatic', 'Hydrophobe', 'PosIonizable', 'NegIonizable',
    'max_dist_3d', 'max_angle_3d', 'max_plane_angle_3d'
]

# =============================================================================
# 特征归一化策略分类字典
# =============================================================================
ANGLE_FEATURES = {'max_angle_3d': 180.0, 'max_plane_angle_3d': 180.0}
COUNT_FEATURES = {'HBA', 'HBD', 'Aromatic', 'Hydrophobe', 'PosIonizable', 'NegIonizable'}

# =============================================================================
#                      粗语料配置与状态初始化
# =============================================================================

# =============================================================================
#                              状态加载
# =============================================================================

vocabulary = set()
print("粗语料词汇集合已初始化；当前流程只生成固定粗特征，不生成词向量。")

# 归一化统计量只在目标目录中不存在统计文件时计算。
if os.path.isfile(NORM_STATS_FILE):
    try:
        with open(NORM_STATS_FILE, 'r', encoding='utf-8') as f:
            norm_stats = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"归一化统计文件无法读取: {NORM_STATS_FILE}: {exc}") from exc

    if not isinstance(norm_stats, dict) or not norm_stats:
        raise RuntimeError(f"归一化统计文件为空或格式非法: {NORM_STATS_FILE}")
    missing_norm_features = set(FEATURE_COLUMNS) - set(norm_stats)
    if missing_norm_features:
        raise RuntimeError(
            f"归一化统计文件缺少特征: {sorted(missing_norm_features)}"
        )
    for feature_name in FEATURE_COLUMNS:
        feature_stats = norm_stats[feature_name]
        if not isinstance(feature_stats, dict) or not {'count', 'mean', 'M2'} <= set(feature_stats):
            raise RuntimeError(f"特征 {feature_name} 的归一化统计格式非法")
        count = feature_stats['count']
        mean = feature_stats['mean']
        m2 = feature_stats['M2']
        if (
            not isinstance(count, (int, float)) or count <= 0
            or not isinstance(mean, (int, float)) or not np.isfinite(mean)
            or not isinstance(m2, (int, float)) or not np.isfinite(m2) or m2 < 0
        ):
            raise RuntimeError(f"特征 {feature_name} 的归一化统计数值非法")
    normalization_stats_loaded = True
    print(f"已加载并复用归一化统计量: {NORM_STATS_FILE}")
else:
    norm_stats = {}
    normalization_stats_loaded = False
    print(f"未找到归一化统计文件，将使用本次全部有效数据计算: {NORM_STATS_FILE}")

print("-" * 50)
print("初始化全部完成。")


# =============================================================================
# 单元格 2: 辅助函数定义 
# =============================================================================

def load_prepared_complexes(manifest_path, complexes_dir):
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"未找到预处理清单: {manifest_path}")
    manifest = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    required = {'output_id', 'pdbid', 'source', 'materialized'}
    missing = required - set(manifest.columns)
    if missing:
        raise RuntimeError(f"accepted_manifest.csv 缺少字段: {sorted(missing)}")

    materialized = manifest['materialized'].str.strip().str.lower().isin({'true', '1', 'yes'})
    manifest = manifest.loc[materialized].copy()
    records = []
    for row in manifest.to_dict('records'):
        output_id = row['output_id'].strip()
        complex_dir = os.path.join(complexes_dir, output_id)
        records.append({
            'sample_id': output_id,
            'pdb_id': row['pdbid'].strip().lower(),
            'source': row['source'].strip(),
            'ligand_sdf_path': os.path.join(complex_dir, 'ligand.sdf'),
            'pocket_path': os.path.join(complex_dir, 'receptor.pdb'),
        })
    return records

def process_single_fragment(fragment, origin_type, sample_id, source_fragment_id, id_start_offset):
    fragment_stats = {
        'removed_ion_fragments': 0,
        'reconstruction_failed': 0,
        'retained_metal_nodes': 0,
    }
    try:
        pre_calc_data = utils.prepare_molecule_for_cutting(
            fragment, source_fragment_id, origin_type
        )
        fragment_stats['removed_ion_fragments'] = pre_calc_data['removed_fragment_count']
        fragment_stats['reconstruction_failed'] = pre_calc_data['reconstruction_failed_count']
        fragment_stats['retained_metal_nodes'] = len(pre_calc_data['metal_fragments'])

        try:
            original_smiles = Chem.MolToSmiles(fragment)
        except Exception:
            original_smiles = '<ERROR>'

        dropped_count = 0
        temp_results = []
        for component in pre_calc_data['organic_components']:
            main_mol = component['main_mol_for_matching']
            final_covalent_shards = utils.execute_fragmentation_pipeline(
                main_mol, component['bond_info'], track_ports=(origin_type == 'ligand')
            )
            results_storage = []
            dropped_count += utils.calculate_descriptors_and_store(
                storage=results_storage,
                mol_data={'IDs': sample_id, 'Smiles': original_smiles},
                original_mol_with_conf=fragment,
                main_mol_for_matching=main_mol,
                final_covalent_fragments=final_covalent_shards,
                original_eccentricities=component['original_eccentricities'],
                original_logp_contribs=component['original_logp_contribs'],
                original_tpsa_contribs=component['original_tpsa_contribs'],
                original_charges=component['original_charges'],
                origin_type=origin_type,
            )
            if results_storage:
                temp_results.extend(results_storage[0]['results'])

        for metal_fragment in pre_calc_data['metal_fragments']:
            metal_record = utils.build_metal_shard_record(metal_fragment)
            if metal_record is None:
                dropped_count += 1
            else:
                temp_results.append(metal_record)

        shards_list = []
        for i, shard_info in enumerate(temp_results):
            shard_info['origin'] = origin_type
            shard_info['Shard ID'] = id_start_offset + i
            shards_list.append(shard_info)
        if shards_list:
            return shards_list, None, dropped_count, fragment_stats
        return [], '未生成结果', dropped_count, fragment_stats
    except (utils.PortValidationError, utils.LigandSecondaryCleaningError):
        raise
    except Exception as e:
        return [], str(e), 1, fragment_stats
    
def build_final_normalization_params(stats):
    params = {}
    for prop in FEATURE_COLUMNS:
        st = stats[prop]
        count, mean, M2 = st['count'], st['mean'], st['M2']
        std = np.sqrt(M2 / count) if count >= 2 else 0.0
        params[prop] = {'mean': float(mean), 'std': float(std)}
    return params

def save_state_and_report(
    current_vocab_set, stats, stats_file, count, save_stats=True,
):
    fragment_corpus, fragment_metadata, merge_stats = (
        utils.merge_raw_fragment_corpus(
            current_vocab_set,
            RAW_CORPUS_FILE,
            RAW_CORPUS_METADATA_FILE,
        )
    )
    if fragment_corpus is None or fragment_metadata is None:
        raise RuntimeError('片段粗语料中没有可保存的有效片段 SMILES')

    atom_metadata = utils.save_raw_fragment_atom_corpus(
        fragment_corpus['smiles'].tolist(),
        RAW_ATOM_CORPUS_FILE,
        RAW_ATOM_CORPUS_METADATA_FILE,
        source_fragment_corpus_sha256=fragment_metadata['corpus_sha256'],
        preserve_resolver_entries=True,
    )

    if save_stats:
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)

    stats_message = '统计量已保存' if save_stats else '已复用现有统计量'
    print(
        f"--- 进度报告 (已处理 {count} 个复合物): "
        f"有效唯一SMILES={fragment_metadata['num_smiles']}, "
        f"失败SMILES={fragment_metadata['invalid_smiles_count']}, "
        f"{stats_message} ---"
    )
    print(
        "    共享片段粗语料合并: "
        f"原有={merge_stats['existing_vocab_size']}, "
        f"本次新增={merge_stats['new_vocab_added']}, "
        f"最终={merge_stats['shared_vocab_size_after']}"
    )
    print(f"    粗语料NPZ: {RAW_CORPUS_FILE}")
    print(f"    粗语料元数据: {RAW_CORPUS_METADATA_FILE}")
    print(
        "    原子粗语料: "
        f"片段={atom_metadata['num_smiles']}, "
        f"原子={atom_metadata['num_atoms']}, "
        f"键={atom_metadata['num_bonds']}, "
        f"动态新增={atom_metadata['resolver_added_smiles_count']}"
    )
    print(f"    原子粗语料NPZ: {RAW_ATOM_CORPUS_FILE}")
    print(f"    原子粗语料元数据: {RAW_ATOM_CORPUS_METADATA_FILE}")
    return {
        'fragment_metadata': fragment_metadata,
        'atom_metadata': atom_metadata,
        'merge_stats': merge_stats,
    }


def update_vocab_and_normalization_stats(
    processed_groups,
    current_vocab_set,
    stats,
    calculate_stats,
):
    """
    按原有 shard 顺序更新词汇集合和 Welford 全局统计量。

    该函数在主进程收到每个 worker 结果时立即执行，从而无需把全部结果
    保留到阶段 D；遍历顺序与原先对 batch_final_data 的顺序遍历一致。
    """
    all_shards = [
        shard
        for group in processed_groups
        for shard in group['shards']
    ]
    for shard in all_shards:
        fragment_smi = shard.get('smiles')
        if not fragment_smi:
            raise ValueError('节点缺少化学规范SMILES')
        current_vocab_set.add(fragment_smi)

        if not calculate_stats:
            continue
        for prop, value in shard.items():
            if prop in PROPERTIES_TO_EXCLUDE_FOR_NORM:
                continue
            if (
                isinstance(value, (int, float, np.integer, np.floating))
                and not isinstance(value, (bool, np.bool_))
                and math.isfinite(float(value))
            ):
                existing = stats.get(
                    prop,
                    {'count': 0, 'mean': 0.0, 'M2': 0.0},
                )
                stats[prop] = utils.update_online_stats(existing, value)


def save_intermediate_result(result, spool_dir, sequence_index):
    """原子写入一个 worker 成功结果，供阶段 E 逐个回读。"""
    complex_final_data = {
        'complex_info': result['complex_info'],
        'processed_groups': result['processed_groups'],
        'connection_edge_index': result['connection_edge_index'],
        'connection_atom_ids': result['connection_atom_ids'],
        'connection_distance': result['connection_distance'],
    }
    complex_edge_data = {
        'complex_info': result['complex_info'],
        'edges': result['edges'],
    }
    payload = (complex_final_data, complex_edge_data)

    final_path = os.path.join(
        spool_dir,
        f"{sequence_index:08d}.pickle",
    )
    temp_path = f"{final_path}.tmp"
    try:
        with open(temp_path, 'wb') as file_obj:
            pickle.dump(payload, file_obj, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temp_path, final_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
    return final_path


def load_intermediate_result(path):
    """读取由 save_intermediate_result 写入的内部暂存结果。"""
    with open(path, 'rb') as file_obj:
        payload = pickle.load(file_obj)
    if not isinstance(payload, tuple) or len(payload) != 2:
        raise ValueError(f"中间结果格式非法: {path}")
    return payload


# =============================================================================
# 新增: 多进程 Worker 核心函数 (运行于独立进程中)
# =============================================================================
def worker_process_complex(complex_data):
    """处理一个标准化复合物：严格加载、切割、连接配对和三类空间建边。"""
    RDLogger.logger().setLevel(RDLogger.CRITICAL)

    result = {
        'success': False,
        'complex_info': complex_data,
        'processed_groups': None,
        'edges': None,
        'connection_edge_index': None,
        'connection_atom_ids': None,
        'connection_distance': None,
        'error': '',
        'stats': {
            'success': 0, 'failed': 0, 'dropped_complex': 0, 'drop_exception': 0,
            'reconstruction_failed_ligand': 0, 'reconstruction_failed_pocket': 0,
            'no_valid_ligand': 0, 'no_valid_pocket': 0,
            'ligand_secondary_clean_failed': 0,
            'connection_constraint_filtered_complexes': 0,
            'node_pairs_with_two_atom_pairs': 0,
            'removed_ligand_ion_fragments': 0, 'removed_pocket_ion_fragments': 0,
            'retained_ligand_metal_nodes': 0, 'retained_pocket_metal_nodes': 0,
        },
    }

    # === 阶段 A: 加载上游已验证的标准结构 ===
    try:
        ligand_mol = Chem.MolFromMolFile(
            complex_data['ligand_sdf_path'], removeHs=True, sanitize=True, strictParsing=True
        )
    except Exception as exc:
        ligand_mol = None
        result['error'] = f"ligand_load_failed: {exc}"
    if ligand_mol is None or ligand_mol.GetNumAtoms() == 0 or ligand_mol.GetNumConformers() == 0:
        result['stats']['failed'] += 1
        return result

    try:
        pocket_mol = Chem.MolFromPDBFile(
            complex_data['pocket_path'], removeHs=True, sanitize=False
        )
    except Exception as exc:
        pocket_mol = None
        result['error'] = f"pocket_load_failed: {exc}"
    if pocket_mol is None or pocket_mol.GetNumAtoms() == 0:
        result['stats']['failed'] += 1
        return result
    pocket_mol = utils.advanced_rescue_valence_errors(pocket_mol)

    ligand_fragments = list(Chem.GetMolFrags(ligand_mol, asMols=True, sanitizeFrags=False))
    pocket_fragments = list(Chem.GetMolFrags(pocket_mol, asMols=True, sanitizeFrags=False))
    if not ligand_fragments or not pocket_fragments:
        result['stats']['failed'] += 1
        result['error'] = 'empty_ligand_or_pocket_fragments'
        return result

    # === 阶段 B: 切割、描述符与连接原子配对 ===
    valid_processed_groups = []
    local_shard_id_counter = 0
    complex_total_dropped_fragments = 0
    fragment_jobs = (
        [('ligand', fragment) for fragment in ligand_fragments]
        + [('pocket', fragment) for fragment in pocket_fragments]
    )
    try:
        for fragment_index, (origin_type, original_fragment) in enumerate(fragment_jobs):
            source_fragment_id = f"{complex_data['sample_id']}:{origin_type}:{fragment_index}"
            shards, error, dropped_cnt, fragment_stats = process_single_fragment(
                original_fragment, origin_type, complex_data['sample_id'],
                source_fragment_id, local_shard_id_counter
            )
            complex_total_dropped_fragments += dropped_cnt
            result['stats'][f'reconstruction_failed_{origin_type}'] += fragment_stats['reconstruction_failed']
            result['stats'][f'removed_{origin_type}_ion_fragments'] += fragment_stats['removed_ion_fragments']
            result['stats'][f'retained_{origin_type}_metal_nodes'] += fragment_stats['retained_metal_nodes']
            if shards:
                valid_processed_groups.append({
                    'original_fragment': original_fragment, 'shards': shards
                })
                local_shard_id_counter += len(shards)

        if complex_total_dropped_fragments > 10:
            raise utils.PortValidationError('切割质量控制未通过')
        has_valid_ligand = any(
            shard['origin'] == 'ligand'
            for group in valid_processed_groups for shard in group['shards']
        )
        if not has_valid_ligand:
            result['stats']['no_valid_ligand'] += 1
            raise utils.PortValidationError('复合物没有有效 ligand 片段')
        has_valid_pocket = any(
            shard['origin'] == 'pocket'
            for group in valid_processed_groups for shard in group['shards']
        )
        if not has_valid_pocket:
            result['stats']['no_valid_pocket'] += 1
            raise utils.PortValidationError('复合物没有有效 pocket 片段')

        all_shards = [
            shard for group in valid_processed_groups for shard in group['shards']
        ]
        (
            connection_edge_index,
            connection_atom_ids,
            connection_distance,
        ) = utils.build_fragment_connections(
            all_shards
        )
    except utils.ConnectionTopologyError as exc:
        result['stats']['connection_constraint_filtered_complexes'] += 1
        result['stats']['dropped_complex'] += 1
        result['error'] = f"connection_constraint_failed: {exc}"
        return result
    except utils.LigandSecondaryCleaningError as exc:
        result['stats']['ligand_secondary_clean_failed'] += 1
        result['stats']['dropped_complex'] += 1
        result['error'] = f"ligand_secondary_clean_failed: {exc}"
        return result
    except Exception as exc:
        result['stats']['dropped_complex'] += 1
        result['error'] = f"fragment_or_connection_validation_failed: {exc}"
        return result

    # === 阶段 C: 只按距离构建 LL / LP / PP 三类空间边 ===
    edge_attributes = {}
    for first, second in combinations(all_shards, 2):
        if first.get('centroid_3d') is None or second.get('centroid_3d') is None:
            continue
        distance = float(np.linalg.norm(
            np.asarray(first['centroid_3d']) - np.asarray(second['centroid_3d'])
        ))
        same_origin = first['origin'] == second['origin']
        cutoff = LL_PP_CENTROID_CUTOFF if same_origin else LP_CENTROID_CUTOFF
        if distance >= cutoff:
            continue
        edge = tuple(sorted((first['Shard ID'], second['Shard ID'])))
        origins = sorted((first['origin'], second['origin']))
        edge_attributes[edge] = (
            0 if origins == ['ligand', 'ligand']
            else 1 if origins == ['ligand', 'pocket']
            else 2
        )

    # 旧版共价边能力保留在 utils.build_legacy_covalent_edges，当前图不写入该结果。
    for group in valid_processed_groups:
        group.pop('original_fragment', None)
        for shard in group['shards']:
            shard.pop('mol', None)
            shard.pop('clean_mol', None)
            shard.pop('ports', None)

    result['processed_groups'] = valid_processed_groups
    result['edges'] = edge_attributes
    result['connection_edge_index'] = connection_edge_index
    result['connection_atom_ids'] = connection_atom_ids
    result['connection_distance'] = connection_distance
    result['success'] = True
    result['stats']['success'] = 1
    return result


# =============================================================================
# 主程序入口 (包含数据准备与多进程调度)
# =============================================================================
if __name__ == '__main__':

    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')

    print("正在读取标准化复合物清单...")
    complexes_to_process_list = load_prepared_complexes(
        ACCEPTED_MANIFEST, COMPLEXES_DIR
    )
    if NUM_COMPLEXES_TO_PROCESS and isinstance(NUM_COMPLEXES_TO_PROCESS, int):
        complexes_to_process_list = complexes_to_process_list[:NUM_COMPLEXES_TO_PROCESS]
    if not complexes_to_process_list:
        raise RuntimeError("accepted_manifest.csv 中没有可处理的实体复合物。")

    total_complexes = len(complexes_to_process_list)
    # 方案 A：一次处理全部数据，确保均值和标准差来自全局统计。
    num_batches = 1

    print(f"清单加载完成，本次计划处理 {total_complexes} 个复合物。")
    if normalization_stats_loaded:
        print("统计策略: 复用目标目录中已有的全局归一化统计量。")
    else:
        print("统计策略: 全部复合物作为一个批次，统一计算全局归一化参数。")

    final_norm_params = (
        build_final_normalization_params(norm_stats)
        if normalization_stats_loaded else {}
    )
    total_saved_pyg = 0
    
    # 仅在目标目录没有有效统计文件时，使用本次全部有效数据计算。
    needs_calc_stats = not normalization_stats_loaded

    # --- 开始批处理大循环 (主进程协调) ---
    for batch_idx in range(num_batches):
        
        start_idx = 0
        end_idx = total_complexes
        current_batch_list = complexes_to_process_list
        
        print(f"\n{'='*60}")
        print(f">>> 开始处理批次 {batch_idx + 1} / {num_batches} (使用 {NUM_WORKERS} 进程)")
        print(f"{'='*60}")

        spool_dir = tempfile.mkdtemp(
            prefix=f"se3td_batch_{batch_idx + 1}_",
            dir=LOG_DIR,
        )
        spool_paths = []
        successful_result_count = 0
        # pool.map 按输入顺序返回结果；在这里在线更新即可保持原 Welford 顺序。
        calculate_stats = needs_calc_stats
        print(f"  [中间结果目录] {spool_dir}")
        stat_keys = [
            'success', 'failed', 'dropped_complex', 'drop_exception',
            'reconstruction_failed_ligand', 'reconstruction_failed_pocket',
            'no_valid_ligand', 'no_valid_pocket',
            'ligand_secondary_clean_failed', 'pyg_save_failed',
            'connection_constraint_filtered_complexes',
            'node_pairs_with_two_atom_pairs',
            'removed_ligand_ion_fragments', 'removed_pocket_ion_fragments',
            'retained_ligand_metal_nodes', 'retained_pocket_metal_nodes',
        ]
        agg_stats = {key: 0 for key in stat_keys}

        # ---------------------------------------------------------------------
        # 阶段 A/B/C: 并行执行重度计算任务 (Pebble 外部强杀与精准追踪版)
        # ---------------------------------------------------------------------
        print(f"  [批次 {batch_idx+1}] 多进程执行阶段 A/B/C (加载/切割/建边)...")
        
        from pebble import ProcessPool
        from concurrent.futures import TimeoutError
        
        # 使用 pebble 的 ProcessPool，它能在 C++ 死锁时直接从 OS 层面杀掉 worker
        with ProcessPool(max_workers=NUM_WORKERS) as pool:
            # 发起映射任务，并设置物理超时为 120 秒
            future = pool.map(worker_process_complex, current_batch_list, timeout=120)
            
            # 获取结果迭代器
            iterator = future.result()
            
            # 手动控制 tqdm 进度条
            pbar = tqdm(total=len(current_batch_list), desc=f"  B{batch_idx+1} 进度")
            
            # [新增] 任务索引计数器，用于精准追踪报错分子
            task_idx = 0 
            
            try:
                while True:
                    try:
                        # 尝试获取下一个结果
                        res = next(iterator)

                    except StopIteration:
                        # 本批次所有任务遍历完成
                        break

                    except TimeoutError:
                        # [拦截成功！] 揪出导致超时的分子信息
                        poison_item = current_batch_list[task_idx]
                        sample_id = poison_item['sample_id']

                        agg_stats['dropped_complex'] += 1
                        agg_stats['drop_exception'] += 1

                        # 1. 在终端实时打印警告（不破坏进度条）
                        tqdm.write(f"  [系统击杀] C++ 死锁拦截！样本: {sample_id} (耗时 > 120s)")

                        # 2. 写入本地黑名单文件
                        with open(POISON_BLACKLIST_FILE, "a", encoding="utf-8") as f:
                            f.write(f"Timeout: SAMPLE={sample_id}\n")

                    except Exception as e:
                        # 捕获其他未知崩溃（比如返回的数据结构不对等）
                        poison_item = current_batch_list[task_idx]
                        sample_id = poison_item['sample_id']

                        agg_stats['dropped_complex'] += 1
                        agg_stats['drop_exception'] += 1
                        tqdm.write(f"  [程序崩溃] 样本: {sample_id} | 错误信息: {str(e)}")

                    else:
                        # pool.map 保证结果顺序与输入顺序一致。
                        for k in agg_stats.keys():
                            agg_stats[k] += res['stats'].get(k, 0)

                        if res['success']:
                            successful_result_count += 1
                            update_vocab_and_normalization_stats(
                                res['processed_groups'],
                                vocabulary,
                                norm_stats,
                                calculate_stats,
                            )
                            try:
                                spool_path = save_intermediate_result(
                                    res,
                                    spool_dir,
                                    task_idx,
                                )
                            except Exception as exc:
                                sample_id = res['complex_info']['sample_id']
                                raise RuntimeError(
                                    "中间结果写入失败，已停止以避免生成不完整数据: "
                                    f"{sample_id}: {exc}"
                                ) from exc
                            spool_paths.append(spool_path)

                    # 仅在确实消费了一个任务结果（成功或异常）后推进索引。
                    res = None
                    pbar.update(1)
                    task_idx += 1
            finally:
                pbar.close()

        # 释放 Pebble 的结果迭代器与最后一个 worker 返回对象，再进入阶段 D。
        del iterator
        del future
        gc.collect()

        print(f"    -> 阶段 A/B/C 统计: 成功 {agg_stats['success']} 个, "
              f"加载失败 {agg_stats['failed']}, 质控剔除 {agg_stats['dropped_complex']}, "
              f"进程异常 {agg_stats['drop_exception']}")
        print(
            "    -> 连接约束过滤: "
            f"{agg_stats['connection_constraint_filtered_complexes']} 个复合物 "
            "（节点自环、重复原子对或同一节点对超过两个成键原子对）"
        )

        # ---------------------------------------------------------------------
        # 阶段 D: 粗语料生成与归一化 (主进程单线程执行)
        # ---------------------------------------------------------------------
        print(f"  [批次 {batch_idx+1}] 阶段 D: 保存粗语料与全局统计量...")

        if successful_result_count <= 0 or not spool_paths:
            raise RuntimeError("阶段 A/B/C 没有可供阶段 D/E 使用的成功结果")
        if successful_result_count != len(spool_paths):
            raise RuntimeError(
                "成功结果数与中间文件数不一致: "
                f"{successful_result_count} != {len(spool_paths)}"
            )
        if calculate_stats:
            needs_calc_stats = False
                            
        corpus_report = save_state_and_report(
            vocabulary, norm_stats, NORM_STATS_FILE,
            end_idx, save_stats=calculate_stats,
        )

        if calculate_stats and norm_stats:
            print(f"    -> [系统] 正在生成全局归一化参数 (Mean/Std)...")
            final_norm_params = build_final_normalization_params(norm_stats)
            print(f"       已准备好 {len(final_norm_params)} 个属性的归一化参数。")
        
        if not final_norm_params and batch_idx > 0:
            print("    -> [警告] 非首批次且无归一化参数，可能是首批次没有有效数据或统计文件为空！")

        # ---------------------------------------------------------------------
        # 阶段 E: 待回填 PyG 组装与保存
        # ---------------------------------------------------------------------
        print(f"  [批次 {batch_idx+1}] 阶段 E: 生成并保存待回填 PyG 对象...")
        
        batch_saved_count = 0
        stage_e_progress = tqdm(
            spool_paths,
            desc=f"  B{batch_idx+1} PyG写入",
        )

        for spool_path in stage_e_progress:
            sample_id = os.path.basename(spool_path)
            complex_final_data = None
            complex_edge_data = None
            data = None
            try:
                complex_final_data, complex_edge_data = (
                    load_intermediate_result(spool_path)
                )
                sample_id = complex_final_data['complex_info']['sample_id']
                pdb_id = complex_final_data['complex_info']['pdb_id']
                source = complex_final_data['complex_info']['source']
                all_shards = [shard for group in complex_final_data['processed_groups'] for shard in group['shards']]
                all_shards.sort(key=lambda s: s['Shard ID'])

                if not all_shards or any(s.get('centroid_3d') is None or s.get('ref_coord_3d') is None for s in all_shards):
                    raise ValueError('存在缺少 3D 质心或参考坐标的节点')

                ligand_centroids = np.array([s['centroid_3d'] for s in all_shards if s['origin'] == 'ligand'])
                if len(ligand_centroids) == 0:
                    raise ValueError('图中没有可用于中心化的 ligand 节点')
                ligand_meb_center, _ = utils.calculate_minimum_enclosing_ball(ligand_centroids)
                
                node_features_list, node_type_list, metal_mask_list = [], [], []
                node_pos_list, ref_coords_list, fragment_smiles_list = [], [], []
                hac_list, ring_count_list, ligand_node_indices = [], [], []

                for shard in all_shards:
                    if shard['origin'] == 'ligand':
                        node_type = [1.0, 0.0, 0.0]
                        ligand_node_indices.append(len(node_features_list))
                    else:
                        node_type = [0.0, 1.0, 0.0]
                    other_features = []
                    
                    for prop in FEATURE_COLUMNS:
                        value = shard.get(prop, 0.0)
                        
                        if prop in final_norm_params:
                            p = final_norm_params[prop]
                            mean, std = p['mean'], p['std']
                            
                            # === 策略 A: 角度特征 (物理极限映射 [0, 1]) ===
                            if prop in ANGLE_FEATURES:
                                norm_val = float(np.clip(value / ANGLE_FEATURES[prop], 0.0, 1.0))
                                
                            # === 策略 B: 计数特征 (限制最小方差，防止零膨胀爆表) ===
                            elif prop in COUNT_FEATURES:
                                safe_std = max(std, 1.0) # 强制最小 std 为 1.0
                                norm_val = (value - mean) / safe_std
                                norm_val = float(np.clip(norm_val, -6.0, 6.0)) # 保底钳制
                                
                            # === 策略 C: 连续型理化特征 (标准 Z-score + 物理钳制) ===
                            else:
                                safe_std = std if std > 1e-5 else 1.0
                                norm_val = (value - mean) / safe_std
                                norm_val = float(np.clip(norm_val, -6.0, 6.0)) # 强制钳制，扼杀梯度爆炸
                                
                            other_features.append(norm_val)
                        else:
                            other_features.append(value)
                            
                    node_features_list.append(other_features)
                    node_type_list.append(node_type)
                    metal_mask_list.append(bool(shard.get('is_metal', False)))
                    hac_list.append(int(shard['hac']))
                    ring_count_list.append(int(shard['ring_count']))
                    node_pos_list.append((np.array(shard['centroid_3d']) - ligand_meb_center).tolist())
                    ref_coords_list.append((np.array(shard['ref_coord_3d']) - ligand_meb_center).tolist())
                    
                    fragment_smi = shard.get('smiles')
                    if not fragment_smi:
                        raise ValueError('节点缺少规范SMILES，无法供离线128维编码器回填')
                    fragment_smiles_list.append(fragment_smi)
                    vocabulary.add(fragment_smi)

                # 追加位于配体最小包围球球心的全局虚拟节点。
                virtual_node_index = len(node_features_list)
                node_features_list.append([0.0] * len(FEATURE_COLUMNS))
                node_type_list.append([0.0, 0.0, 1.0])
                metal_mask_list.append(False)
                hac_list.append(0)
                ring_count_list.append(0)
                node_pos_list.append([0.0, 0.0, 0.0])
                fragment_smiles_list.append("<GLOBAL>")
                ref_coords_list.append(np.zeros((3, 3), dtype=float).tolist())

                x = torch.tensor(node_features_list, dtype=torch.float)
                node_type = torch.tensor(node_type_list, dtype=torch.float)
                metal_mask = torch.tensor(metal_mask_list, dtype=torch.bool)
                hac = torch.tensor(hac_list, dtype=torch.long)
                ring_count = torch.tensor(ring_count_list, dtype=torch.long)
                pos = torch.tensor(node_pos_list, dtype=torch.float)
                ref_coords = torch.tensor(np.array(ref_coords_list), dtype=torch.float)
                connection_edge_index = torch.tensor(
                    complex_final_data['connection_edge_index'],
                    dtype=torch.long,
                ).reshape(2, -1)
                connection_atom_ids = torch.tensor(
                    complex_final_data['connection_atom_ids'],
                    dtype=torch.long,
                ).reshape(-1, 2)
                connection_distance = torch.tensor(
                    complex_final_data['connection_distance'],
                    dtype=torch.float,
                ).reshape(-1)

                num_connections = connection_edge_index.size(1)
                if tuple(connection_atom_ids.shape) != (num_connections, 2):
                    raise ValueError(
                        'connection_atom_ids 与 connection_edge_index 数量不一致'
                    )
                if tuple(connection_distance.shape) != (num_connections,):
                    raise ValueError(
                        'connection_distance 与 connection_edge_index 数量不一致'
                    )
                if not torch.isfinite(connection_distance).all():
                    raise ValueError('connection_distance 包含 NaN 或 Inf')
                if bool((connection_distance < 0).any()):
                    raise ValueError('connection_distance 包含负数')

                node_pair_counts = {}
                if num_connections > 0:
                    if (
                        int(connection_edge_index.min()) < 0
                        or int(connection_edge_index.max()) >= len(all_shards)
                    ):
                        raise ValueError(
                            'connection_edge_index 含越界节点编号'
                        )
                    if bool(
                        (
                            connection_edge_index[0]
                            == connection_edge_index[1]
                        ).any()
                    ):
                        raise ValueError(
                            'connection_edge_index 包含节点自身连接'
                        )
                    if bool((connection_atom_ids < 0).any()):
                        raise ValueError(
                            'connection_atom_ids 包含负规范原子编号'
                        )

                    for first_node, second_node in (
                        connection_edge_index.t().tolist()
                    ):
                        node_pair = tuple(sorted((
                            int(first_node), int(second_node)
                        )))
                        node_pair_counts[node_pair] = (
                            node_pair_counts.get(node_pair, 0) + 1
                        )
                        if node_pair_counts[node_pair] > 2:
                            raise ValueError(
                                f'节点对 {node_pair} 包含超过两个成键原子对'
                            )
                two_atom_pair_node_pair_count = sum(
                    count == 2 for count in node_pair_counts.values()
                )
                
                edges_dict = complex_edge_data['edges']
                edge_list, edge_attr_list = [], []
                for (id1, id2), et in edges_dict.items():
                    edge_list.append([id1, id2])
                    if et not in (0, 1, 2):
                        raise ValueError(f'未知空间边类型: {et}')
                    attr = [0.0] * 4
                    attr[et] = 1.0
                    edge_attr_list.append(attr)

                # DL 边无视距离，将每个 ligand 节点连接到全局虚拟节点。
                for ligand_node_index in ligand_node_indices:
                    edge_list.append([virtual_node_index, ligand_node_index])
                    edge_attr_list.append([0.0, 0.0, 0.0, 1.0])
                
                if edge_list:
                    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
                    edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)
                    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
                    edge_attr = torch.cat([edge_attr, edge_attr], dim=0)
                else:
                    edge_index = torch.empty((2, 0), dtype=torch.long)
                    edge_attr = torch.empty((0, 4), dtype=torch.float)

                data = Data(
                    x=x, node_type=node_type, metal_mask=metal_mask,
                    hac=hac, ring_count=ring_count,
                    pos=pos, edge_index=edge_index, edge_attr=edge_attr, 
                    pdb_id=pdb_id, # 统一使用 pdb_id
                    fragment_smiles=fragment_smiles_list, ref_coords=ref_coords,
                    sample_id=sample_id, source=source,
                    connection_edge_index=connection_edge_index,
                    connection_atom_ids=connection_atom_ids,
                    connection_distance=connection_distance,
                )
                
                # [核心修改 1]：根据 pdb_id 的前两个字符作为子文件夹名
                sub_folder_name = sample_id[:2] if len(sample_id) >= 2 else "misc"
                target_dir = os.path.join(PYG_DATA_DIR, sub_folder_name)
                
                # 确保子文件夹存在
                os.makedirs(target_dir, exist_ok=True)
                
                # 将文件存入对应的子文件夹中，以 pdb_id 命名
                torch.save(data, os.path.join(target_dir, f"{sample_id}.pt"))

                batch_saved_count += 1
                total_saved_pyg += 1
                agg_stats['node_pairs_with_two_atom_pairs'] += (
                    two_atom_pair_node_pair_count
                )
                try:
                    os.remove(spool_path)
                except OSError as cleanup_exc:
                    tqdm.write(
                        "  [中间文件清理警告] "
                        f"{spool_path} | {cleanup_exc}"
                    )
                 
            except Exception as e:
                agg_stats['pyg_save_failed'] += 1
                tqdm.write(f"  [PyG 保存失败] 样本: {sample_id} | {e}")
                continue
            finally:
                # 防止循环变量继续持有上一个复合物的大型嵌套结构与张量。
                complex_final_data = None
                complex_edge_data = None
                data = None

        stage_e_progress.close()
        print(f"    -> PyG保存完成: 本批次保存 {batch_saved_count} 个。")
        print(
            "    -> 最终数据中拥有两个成键原子对的节点对: "
            f"{agg_stats['node_pairs_with_two_atom_pairs']} 对"
        )
        remaining_intermediates = [
            path for path in spool_paths if os.path.exists(path)
        ]
        if remaining_intermediates:
            print(
                "    -> [警告] 保留 "
                f"{len(remaining_intermediates)} 个未清理中间文件用于排查: "
                f"{spool_dir}"
            )
        else:
            try:
                os.rmdir(spool_dir)
            except OSError as cleanup_exc:
                print(
                    f"    -> [中间目录清理警告] {spool_dir}: "
                    f"{cleanup_exc}"
                )

        agg_stats['planned_complexes'] = total_complexes
        agg_stats['saved_pyg'] = total_saved_pyg
        agg_stats['raw_vocab_size'] = len(vocabulary)
        agg_stats['shared_vocab_size_before'] = (
            corpus_report['merge_stats']['existing_vocab_size']
        )
        agg_stats['new_vocab_candidates'] = (
            corpus_report['merge_stats']['new_vocab_candidates']
        )
        agg_stats['new_vocab_added'] = (
            corpus_report['merge_stats']['new_vocab_added']
        )
        agg_stats['shared_vocab_size_after'] = (
            corpus_report['merge_stats']['shared_vocab_size_after']
        )
        agg_stats['existing_vocab_normalized'] = (
            corpus_report['merge_stats']['existing_vocab_normalized']
        )
        agg_stats['corpus_created'] = (
            corpus_report['merge_stats']['corpus_created']
        )
        agg_stats['corpus_rewritten'] = (
            corpus_report['merge_stats']['corpus_rewritten']
        )
        agg_stats['atom_corpus_num_smiles'] = (
            corpus_report['atom_metadata']['num_smiles']
        )
        agg_stats['atom_corpus_num_atoms'] = (
            corpus_report['atom_metadata']['num_atoms']
        )
        agg_stats['atom_corpus_num_bonds'] = (
            corpus_report['atom_metadata']['num_bonds']
        )
        agg_stats['atom_corpus_resolver_added_smiles'] = (
            corpus_report['atom_metadata']['resolver_added_smiles_count']
        )
        agg_stats['embedding_status'] = 'pending_offline_128d'
        with open(SUMMARY_FILE, 'w', encoding='utf-8') as summary_file:
            json.dump(agg_stats, summary_file, ensure_ascii=False, indent=2)
        del remaining_intermediates
        del spool_paths
        gc.collect() 
        
        # [批次收尾]：强制操作系统将内存缓存同步到硬盘 (这部分还在 for 循环内)
        print(f"  [批次 {batch_idx+1}] 正在强制将内存数据落盘，请稍候...")
        if hasattr(os, 'sync'):
            os.sync()
        print(f"  [批次 {batch_idx+1}] 完成。内存已清理且已安全落盘。")

        # =========================================================================
        # 单批次方案的全局收尾。
        # =========================================================================
        print(f"\n{'='*60}")
        print(f"全部处理完毕！共成功生成并保存 {total_saved_pyg} 个 HiQBind/PDBbind 复合物的 PyG 图数据。")
        print(f"全局统计结果详见 {NORM_STATS_FILE}")
        print(f"粗语料详见 {RAW_CORPUS_FILE} 与 {RAW_CORPUS_METADATA_FILE}")
        print(
            "原子粗语料详见 "
            f"{RAW_ATOM_CORPUS_FILE} 与 {RAW_ATOM_CORPUS_METADATA_FILE}"
        )
        print(f"待回填PyG详见 {PYG_DATA_DIR}；离线128维回填完成前不可用于现有e3nn训练。")
        print(f"{'='*60}")
