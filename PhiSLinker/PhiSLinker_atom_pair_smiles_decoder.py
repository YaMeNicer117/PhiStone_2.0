#!/usr/bin/env python3
"""Decode aligned fragment conformers and finalize receptor complexes.

The standalone CLI supports either a direct SMILES input or a 128-dimensional
fragment embedding.  Importable helpers additionally write multi-fragment SDF
collections, translate disconnected fragments against a fixed nonmetal protein
pocket, and relax a connected ligand with MMFF94 or UFF.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence
from uuid import uuid4

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit.Geometry import Point3D


# =============================================================================
# User-configurable parameters
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SHARED_DATA_DIR = (
    PROJECT_ROOT / "Datas" / "Processed_datas" / "SE3TD_128_Shared_data"
)

DEFAULT_INPUT_JSON_PATH = SCRIPT_DIR / "single_fragment_request.json"
DEFAULT_OUTPUT_SDF_PATH = SCRIPT_DIR / "single_fragment_3d.sdf"
DEFAULT_VOCAB_NPZ_PATH = SHARED_DATA_DIR / "fragment_embeddings_128d.npz"
DEFAULT_VOCAB_JSON_PATH = SHARED_DATA_DIR / "fragment_embeddings_128d.json"
DEFAULT_CUSTOM_VOCAB_PATH = (
    SHARED_DATA_DIR
    / "fragment_custom_embedding_128d"
    / "fragment_embeddings_128d.npz"
)

EMBEDDING_DIM = 128
ETKDG_RANDOM_SEED = 42
# 宽松手性回退是否忽略距离几何平滑失败，以提高困难片段的嵌入成功率。
RELAXED_CHIRALITY_IGNORE_SMOOTHING_FAILURES = True
OPENBABEL_EXECUTABLE = "obabel"
OPENBABEL_GEN3D_MODE = "fast"
# 单次 Open Babel 构象生成允许的最长时间，单位为秒。
OPENBABEL_TIMEOUT_SECONDS = 60.0
OPENBABEL_RANDOM_SEED = ETKDG_RANDOM_SEED
OPENBABEL_MAX_TOPOLOGY_MATCHES = 4_096

# 单片段构象执行 MMFF 或 UFF 优化时允许的最大迭代次数。
FORCE_FIELD_MAX_ITERS = 200
MMFF_VARIANT = "MMFF94"
# MMFF 不支持或失败时，是否允许回退到 UFF 优化。
ENABLE_UFF_FALLBACK = True
# 是否仅在 UFF 参数检查和优化期间屏蔽重复的 UFFTYPER 日志。
SUPPRESS_UFF_TYPER_LOGS = True
# 输出片段是否保留显式氢；关闭时重建模板不保留任何显式氢或立体标记。
OUTPUT_EXPLICIT_HYDROGENS = False

# 未连接片段在原子匹配推理前执行刚体平移梯度下降的参数。
TGD_MAX_STEPS = 60
TGD_LEARNING_RATE = 1.0e-7
TGD_MAX_STEP_DIST = 2.0e-2
TGD_FORCE_TOL = 0.2
# TGD 纳入完整氨基酸残基的蛋白质口袋接触距离，单位为埃。
TGD_POCKET_CUTOFF_ANGSTROM = 6.0
# TGD 口袋力场不纳入金属元素。
TGD_EXCLUDED_METAL_ATOMIC_NUMBERS = frozenset(
    {
        3,
        4,
        11,
        12,
        13,
        19,
        20,
        31,
        37,
        38,
        49,
        50,
        55,
        56,
        81,
        82,
        83,
        84,
        87,
        88,
    }
    | set(range(21, 31))
    | set(range(39, 49))
    | set(range(57, 81))
    | set(range(89, 117))
)

# 是否要求余弦候选同时满足预测 HAC 和 Ring Count 约束。
ENABLE_FRAGMENT_MATCH_HAC_RING_CONSTRAINTS = False
# 允许接受的最低余弦相似度；None 表示不设置余弦硬阈值。
FRAGMENT_MATCH_MIN_COSINE_SIMILARITY: float | None = None
# 诊断信息记录的余弦候选数量；启用 UFF 准入时也作为候选搜索上限。
FRAGMENT_MATCH_DIAGNOSTIC_TOP_K = 4
# 非 overflow 情况下，候选 HAC 与预测 HAC 允许的最大绝对误差。
FRAGMENT_MATCH_HAC_TOLERANCE = 2
# 非 overflow 情况下，候选 Ring Count 与预测值允许的最大绝对误差。
FRAGMENT_MATCH_RING_TOLERANCE = 1
# 开启 HAC/RING 约束时，按余弦降序实际检查的最大候选数量。
FRAGMENT_MATCH_MAX_CONSTRAINT_CANDIDATES = 4

# 完整连接配体执行 MMFF94/UFF 内部优化时的参数。
# 最大优化迭代次数；达到上限但坐标仍合法时保留当前构象。
LIGAND_RELAX_MAX_ITERATIONS = 120
# 相邻优化状态的能量变化低于该阈值时允许停止。
LIGAND_RELAX_ENERGY_TOLERANCE = 1.0e-4

MAX_FRAGMENT_AUTOMORPHISMS = 4_096
MAX_SYMMETRY_COMPONENT_COMBINATIONS = 10_240
NUMERIC_EPSILON = 1.0e-12
FRAME_RANK_TOLERANCE = 1.0e-8
CANONICAL_ATOM_ID_PROP = "_PhiStoneCanonicalAtomId"
RETAINED_NON_TETRAHEDRAL_H_PROP = (
    "_PhiStoneRetainedNonTetrahedralHydrogen"
)
NON_TETRAHEDRAL_CHIRAL_TAGS = frozenset(
    tag
    for tag in (
        getattr(Chem.ChiralType, "CHI_SQUAREPLANAR", None),
        getattr(Chem.ChiralType, "CHI_TRIGONALBIPYRAMIDAL", None),
        getattr(Chem.ChiralType, "CHI_OCTAHEDRAL", None),
    )
    if tag is not None
)


@dataclass(frozen=True)
class FragmentVocabulary:
    """Validated and normalized fragment vocabulary."""

    vocab_ids: np.ndarray
    smiles: tuple[str, ...]
    normalized_embeddings: np.ndarray
    hac_values: np.ndarray
    ring_counts: np.ndarray
    source_path: Path


class FragmentMatchConstraintError(ValueError):
    """No inspected cosine candidate satisfied inference constraints."""

    def __init__(self, message: str, diagnostics: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.diagnostics = dict(diagnostics)


@dataclass(frozen=True)
class FragmentConformerTemplate:
    """Reusable unaligned conformer with stable heavy-atom identifiers."""

    canonical_smiles: str
    molecule: Chem.Mol
    canonical_heavy_atom_ids: tuple[int, ...]
    generation_metadata: Mapping[str, Any]


@dataclass(frozen=True)
class _EmbeddingAttemptSpec:
    """One deterministic RDKit embedding configuration."""

    profile: str
    random_seed: int
    use_random_coords: bool
    enforce_chirality: bool
    remove_stereochemistry: bool = False
    ignore_smoothing_failures: bool = False


def _validate_runtime_configuration() -> None:
    if EMBEDDING_DIM <= 0:
        raise ValueError("EMBEDDING_DIM must be positive")
    if ETKDG_RANDOM_SEED < 0:
        raise ValueError("ETKDG_RANDOM_SEED cannot be negative")
    if not isinstance(OPENBABEL_EXECUTABLE, str) or not (
        OPENBABEL_EXECUTABLE.strip()
    ):
        raise ValueError("OPENBABEL_EXECUTABLE must be a non-empty string")
    if OPENBABEL_GEN3D_MODE != "fast":
        raise ValueError("OPENBABEL_GEN3D_MODE must remain 'fast'")
    if OPENBABEL_TIMEOUT_SECONDS <= 0:
        raise ValueError("OPENBABEL_TIMEOUT_SECONDS must be positive")
    if OPENBABEL_RANDOM_SEED < 0:
        raise ValueError("OPENBABEL_RANDOM_SEED cannot be negative")
    if OPENBABEL_MAX_TOPOLOGY_MATCHES <= 0:
        raise ValueError(
            "OPENBABEL_MAX_TOPOLOGY_MATCHES must be positive"
        )
    if FORCE_FIELD_MAX_ITERS < 0:
        raise ValueError("FORCE_FIELD_MAX_ITERS cannot be negative")
    if TGD_MAX_STEPS <= 0:
        raise ValueError("TGD_MAX_STEPS must be positive")
    if TGD_LEARNING_RATE <= 0.0:
        raise ValueError("TGD_LEARNING_RATE must be positive")
    if TGD_MAX_STEP_DIST <= 0.0:
        raise ValueError("TGD_MAX_STEP_DIST must be positive")
    if TGD_FORCE_TOL < 0.0:
        raise ValueError("TGD_FORCE_TOL cannot be negative")
    if TGD_POCKET_CUTOFF_ANGSTROM <= 0.0:
        raise ValueError("TGD_POCKET_CUTOFF_ANGSTROM must be positive")
    if not isinstance(ENABLE_FRAGMENT_MATCH_HAC_RING_CONSTRAINTS, bool):
        raise TypeError(
            "ENABLE_FRAGMENT_MATCH_HAC_RING_CONSTRAINTS must be a bool"
        )
    if FRAGMENT_MATCH_DIAGNOSTIC_TOP_K <= 0:
        raise ValueError(
            "FRAGMENT_MATCH_DIAGNOSTIC_TOP_K must be positive"
        )
    if FRAGMENT_MATCH_MIN_COSINE_SIMILARITY is not None and not (
        -1.0 <= float(FRAGMENT_MATCH_MIN_COSINE_SIMILARITY) <= 1.0
    ):
        raise ValueError(
            "FRAGMENT_MATCH_MIN_COSINE_SIMILARITY must be within [-1, 1]"
        )
    if FRAGMENT_MATCH_HAC_TOLERANCE < 0:
        raise ValueError("FRAGMENT_MATCH_HAC_TOLERANCE cannot be negative")
    if FRAGMENT_MATCH_RING_TOLERANCE < 0:
        raise ValueError("FRAGMENT_MATCH_RING_TOLERANCE cannot be negative")
    if FRAGMENT_MATCH_MAX_CONSTRAINT_CANDIDATES <= 0:
        raise ValueError(
            "FRAGMENT_MATCH_MAX_CONSTRAINT_CANDIDATES must be positive"
        )
    if MAX_FRAGMENT_AUTOMORPHISMS <= 0:
        raise ValueError("MAX_FRAGMENT_AUTOMORPHISMS must be positive")
    if MAX_SYMMETRY_COMPONENT_COMBINATIONS <= 0:
        raise ValueError(
            "MAX_SYMMETRY_COMPONENT_COMBINATIONS must be positive"
        )
    if LIGAND_RELAX_MAX_ITERATIONS <= 0:
        raise ValueError(
            "LIGAND_RELAX_MAX_ITERATIONS must be positive"
        )
    if LIGAND_RELAX_ENERGY_TOLERANCE <= 0.0:
        raise ValueError(
            "LIGAND_RELAX_ENERGY_TOLERANCE must be positive"
        )


def _as_finite_array(
    value: Any,
    *,
    name: str,
    expected_shape: tuple[int, ...],
    dtype: np.dtype[Any] = np.float64,
) -> np.ndarray:
    if value is None:
        raise ValueError(f"{name} is required")
    try:
        array = np.asarray(value, dtype=dtype)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values") from exc

    if array.shape != expected_shape:
        raise ValueError(
            f"{name} must have shape {expected_shape}, got {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or Inf")
    return array


def _resolve_vocab_path(vocab_path: str | Path | None) -> Path:
    if vocab_path is not None:
        resolved = Path(vocab_path).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Vocabulary file not found: {resolved}")
        return resolved

    if DEFAULT_VOCAB_NPZ_PATH.is_file():
        return DEFAULT_VOCAB_NPZ_PATH
    if DEFAULT_VOCAB_JSON_PATH.is_file():
        return DEFAULT_VOCAB_JSON_PATH

    raise FileNotFoundError(
        "No default vocabulary file was found. Checked: "
        f"{DEFAULT_VOCAB_NPZ_PATH} and {DEFAULT_VOCAB_JSON_PATH}"
    )


def _load_npz_vocabulary(path: Path) -> tuple[np.ndarray, list[str], np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        required_fields = {"smiles", "fragment_embeddings"}
        missing_fields = required_fields - set(loaded.files)
        if missing_fields:
            raise ValueError(
                f"NPZ vocabulary is missing fields: {sorted(missing_fields)}"
            )
        smiles_array = np.asarray(loaded["smiles"])
        embedding_array = np.asarray(loaded["fragment_embeddings"])

    if smiles_array.ndim != 1:
        raise ValueError(
            f"NPZ smiles must be one-dimensional, got {smiles_array.shape}"
        )
    if not np.issubdtype(embedding_array.dtype, np.number):
        raise ValueError("NPZ fragment_embeddings must be numeric")

    smiles = smiles_array.astype(str).tolist()
    vocab_ids = np.arange(len(smiles), dtype=np.int64)
    embeddings = embedding_array.astype(np.float32, copy=False)
    return vocab_ids, smiles, embeddings


def _load_json_vocabulary(
    path: Path,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    with path.open("r", encoding="utf-8") as file_obj:
        records = json.load(file_obj)

    if not isinstance(records, list) or not records:
        raise ValueError("JSON vocabulary must be a non-empty list")

    smiles: list[str] = []
    embeddings: list[np.ndarray] = []
    vocab_ids: list[int] = []
    required_fields = {"vocab_id", "smiles", "embedding"}

    for expected_id, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(
                f"JSON vocabulary record {expected_id} must be an object"
            )
        missing_fields = required_fields - set(record)
        if missing_fields:
            raise ValueError(
                f"JSON vocabulary record {expected_id} is missing fields: "
                f"{sorted(missing_fields)}"
            )

        vocab_id = record["vocab_id"]
        if (
            not isinstance(vocab_id, int)
            or isinstance(vocab_id, bool)
            or vocab_id != expected_id
        ):
            raise ValueError(
                f"JSON vocabulary record {expected_id} has invalid vocab_id "
                f"{vocab_id!r}; expected {expected_id}"
            )

        record_smiles = record["smiles"]
        if not isinstance(record_smiles, str) or not record_smiles.strip():
            raise ValueError(
                f"JSON vocabulary record {expected_id} has invalid smiles"
            )

        embedding = _as_finite_array(
            record["embedding"],
            name=f"JSON vocabulary embedding {expected_id}",
            expected_shape=(EMBEDDING_DIM,),
            dtype=np.float32,
        )
        vocab_ids.append(vocab_id)
        smiles.append(record_smiles)
        embeddings.append(embedding)

    return (
        np.asarray(vocab_ids, dtype=np.int64),
        smiles,
        np.stack(embeddings, axis=0).astype(np.float32, copy=False),
    )


def mol_from_smiles_quietly(smiles: str) -> Chem.Mol | None:
    """Parse one SMILES while containing RDKit messages to this call only."""

    log_blocker = rdBase.BlockLogs()
    try:
        return Chem.MolFromSmiles(smiles)
    finally:
        del log_blocker


def _validate_and_normalize_vocabulary(
    *,
    vocab_ids: np.ndarray,
    smiles: Sequence[str],
    embeddings: np.ndarray,
    source_path: Path,
) -> FragmentVocabulary:
    if not smiles:
        raise ValueError("Vocabulary is empty")
    invalid_smiles_indices = [
        index
        for index, value in enumerate(smiles)
        if not isinstance(value, str) or not value.strip()
    ]
    if invalid_smiles_indices:
        raise ValueError(
            "Vocabulary contains empty or invalid SMILES at rows "
            f"{invalid_smiles_indices[:10]}"
        )
    if vocab_ids.shape != (len(smiles),):
        raise ValueError("Vocabulary ID count does not match the SMILES count")
    if embeddings.shape != (len(smiles), EMBEDDING_DIM):
        raise ValueError(
            "Vocabulary embedding matrix must have shape "
            f"({len(smiles)}, {EMBEDDING_DIM}), got {embeddings.shape}"
        )
    if not np.all(np.isfinite(embeddings)):
        raise ValueError("Vocabulary embeddings contain NaN or Inf")
    if len(set(smiles)) != len(smiles):
        raise ValueError("Vocabulary contains duplicate SMILES")

    hac_values: list[int] = []
    ring_counts: list[int] = []
    for row_index, value in enumerate(smiles):
        molecule = mol_from_smiles_quietly(value)
        if molecule is None:
            raise ValueError(
                f"Vocabulary SMILES at row {row_index} cannot be parsed: "
                f"{value!r}"
            )
        hac = sum(
            atom.GetAtomicNum() > 1 for atom in molecule.GetAtoms()
        )
        if hac <= 0:
            raise ValueError(
                f"Vocabulary SMILES at row {row_index} has no heavy atom: "
                f"{value!r}"
            )
        hac_values.append(int(hac))
        ring_counts.append(
            int(rdMolDescriptors.CalcNumRings(molecule))
        )

    embedding_values = embeddings.astype(np.float32, copy=False)
    norms = np.linalg.norm(embedding_values, axis=1)
    zero_indices = np.flatnonzero(norms <= NUMERIC_EPSILON)
    if zero_indices.size:
        preview = zero_indices[:10].tolist()
        raise ValueError(
            "Vocabulary contains zero-norm embeddings at rows "
            f"{preview}"
        )

    normalized = embedding_values / norms.astype(np.float32)[:, None]
    return FragmentVocabulary(
        vocab_ids=np.asarray(vocab_ids, dtype=np.int64),
        smiles=tuple(smiles),
        normalized_embeddings=normalized,
        hac_values=np.asarray(hac_values, dtype=np.int64),
        ring_counts=np.asarray(ring_counts, dtype=np.int64),
        source_path=source_path,
    )


def load_fragment_vocabulary(
    vocab_path: str | Path | None = None,
) -> FragmentVocabulary:
    """Load and validate the JSON or NPZ fragment vocabulary."""

    _validate_runtime_configuration()
    path = _resolve_vocab_path(vocab_path)
    suffix = path.suffix.lower()

    if suffix == ".npz":
        vocab_ids, smiles, embeddings = _load_npz_vocabulary(path)
    elif suffix == ".json":
        vocab_ids, smiles, embeddings = _load_json_vocabulary(path)
    else:
        raise ValueError(
            f"Unsupported vocabulary format {path.suffix!r}; "
            "expected .npz or .json"
        )

    return _validate_and_normalize_vocabulary(
        vocab_ids=vocab_ids,
        smiles=smiles,
        embeddings=embeddings,
        source_path=path,
    )


def match_fragment_embedding(
    frame_embedding: Sequence[float] | np.ndarray,
    *,
    vocab_path: str | Path | None = None,
    vocabulary: FragmentVocabulary | None = None,
    min_cosine_similarity: float | None = (
        FRAGMENT_MATCH_MIN_COSINE_SIMILARITY
    ),
    diagnostic_top_k: int = FRAGMENT_MATCH_DIAGNOSTIC_TOP_K,
    require_uff_parameters: bool = False,
    enable_hac_ring_constraints: bool = (
        ENABLE_FRAGMENT_MATCH_HAC_RING_CONSTRAINTS
    ),
    predicted_hac: int | None = None,
    predicted_ring_count: int | None = None,
    hac_overflow: bool = False,
    ring_overflow: bool = False,
    hac_tolerance: int = FRAGMENT_MATCH_HAC_TOLERANCE,
    ring_tolerance: int = FRAGMENT_MATCH_RING_TOLERANCE,
    max_constraint_candidates: int = (
        FRAGMENT_MATCH_MAX_CONSTRAINT_CANDIDATES
    ),
) -> tuple[str, dict[str, Any]]:
    """Resolve an embedding by cosine rank and reconstruction constraints."""

    query = _as_finite_array(
        frame_embedding,
        name="frame_embedding",
        expected_shape=(EMBEDDING_DIM,),
        dtype=np.float32,
    )
    query_norm = float(np.linalg.norm(query))
    if query_norm <= NUMERIC_EPSILON:
        raise ValueError("frame_embedding must not be a zero vector")

    if vocabulary is not None and vocab_path is not None:
        raise ValueError("Provide vocabulary or vocab_path, not both")
    if vocabulary is None:
        vocabulary = load_fragment_vocabulary(vocab_path)
    if not isinstance(vocabulary, FragmentVocabulary):
        raise TypeError("vocabulary must be a FragmentVocabulary")
    if diagnostic_top_k <= 0:
        raise ValueError("diagnostic_top_k must be positive")
    if not isinstance(require_uff_parameters, bool):
        raise TypeError("require_uff_parameters must be a bool")
    if max_constraint_candidates <= 0:
        raise ValueError("max_constraint_candidates must be positive")
    if hac_tolerance < 0 or ring_tolerance < 0:
        raise ValueError("HAC and ring tolerances cannot be negative")
    if not isinstance(enable_hac_ring_constraints, bool):
        raise TypeError("enable_hac_ring_constraints must be a bool")
    has_hac_prediction = predicted_hac is not None
    has_ring_prediction = predicted_ring_count is not None
    if enable_hac_ring_constraints and (
        has_hac_prediction != has_ring_prediction
    ):
        raise ValueError(
            "predicted_hac and predicted_ring_count must be provided together"
        )
    constraints_available = has_hac_prediction and has_ring_prediction
    constrained = bool(
        enable_hac_ring_constraints and constraints_available
    )
    if enable_hac_ring_constraints and not constraints_available and (
        hac_overflow or ring_overflow
    ):
        raise ValueError(
            "overflow flags require predicted HAC and ring values"
        )
    if constrained:
        predicted_hac = int(predicted_hac)
        predicted_ring_count = int(predicted_ring_count)
        if predicted_hac < 0 or predicted_ring_count < 0:
            raise ValueError("predicted HAC and ring values cannot be negative")
    else:
        predicted_hac = None
        predicted_ring_count = None
        hac_overflow = False
        ring_overflow = False
    if (
        min_cosine_similarity is not None
        and not -1.0 <= float(min_cosine_similarity) <= 1.0
    ):
        raise ValueError("min_cosine_similarity must be in [-1,1]")
    normalized_query = query / query_norm
    similarities = vocabulary.normalized_embeddings @ normalized_query
    similarities = np.clip(similarities, -1.0, 1.0)

    ranked_indices = np.argsort(-similarities, kind="stable")
    selected_index: int | None = None
    candidate_checks: list[dict[str, Any]] = []
    inspect_count = 1
    if constrained:
        inspect_count = min(
            int(max_constraint_candidates), len(ranked_indices)
        )
    if require_uff_parameters:
        inspect_count = min(int(diagnostic_top_k), len(ranked_indices))
        if constrained:
            inspect_count = min(
                inspect_count,
                int(max_constraint_candidates),
            )

    for rank, raw_index in enumerate(ranked_indices[:inspect_count], start=1):
        index = int(raw_index)
        similarity = float(similarities[index])
        actual_hac = int(vocabulary.hac_values[index])
        actual_ring_count = int(vocabulary.ring_counts[index])
        cosine_pass = bool(
            min_cosine_similarity is None
            or similarity >= float(min_cosine_similarity)
        )
        if constrained:
            hac_delta = actual_hac - int(predicted_hac)
            ring_delta = actual_ring_count - int(predicted_ring_count)
            hac_pass = bool(
                actual_hac >= int(predicted_hac)
                if hac_overflow
                else abs(hac_delta) <= int(hac_tolerance)
            )
            ring_pass = bool(
                actual_ring_count >= int(predicted_ring_count)
                if ring_overflow
                else abs(ring_delta) <= int(ring_tolerance)
            )
        else:
            hac_delta = None
            ring_delta = None
            hac_pass = True
            ring_pass = True

        uff_parameters_evaluated = False
        uff_parameters_supported: bool | None = None
        uff_parameter_error: str | None = None
        if (
            require_uff_parameters
            and cosine_pass
            and hac_pass
            and ring_pass
        ):
            uff_parameters_evaluated = True
            candidate_smiles = vocabulary.smiles[index]
            try:
                candidate_molecule = mol_from_smiles_quietly(
                    candidate_smiles
                )
                if candidate_molecule is None:
                    uff_parameter_error = (
                        "rdkit_parse_failed_during_uff_parameter_check"
                    )
                else:
                    with _suppress_uff_typer_logs():
                        uff_parameters_supported = bool(
                            AllChem.UFFHasAllMoleculeParams(
                                candidate_molecule
                            )
                        )
                    if not uff_parameters_supported:
                        uff_parameter_error = "uff_parameters_unavailable"
            except Exception as exc:
                uff_parameters_supported = False
                uff_parameter_error = (
                    "uff_parameter_check_exception: "
                    f"{type(exc).__name__}: {exc}"
                )
        uff_pass = bool(
            not require_uff_parameters
            or uff_parameters_supported is True
        )

        rejection_reasons: list[str] = []
        if not cosine_pass:
            rejection_reasons.append("below_min_cosine_similarity")
        if not hac_pass:
            rejection_reasons.append("hac_constraint_failed")
        if not ring_pass:
            rejection_reasons.append("ring_constraint_failed")
        if (
            require_uff_parameters
            and cosine_pass
            and hac_pass
            and ring_pass
            and not uff_pass
        ):
            rejection_reasons.append(
                uff_parameter_error or "uff_parameter_check_failed"
            )
        accepted = bool(
            cosine_pass and hac_pass and ring_pass and uff_pass
        )
        record = {
            "rank": rank,
            "vocab_id": int(vocabulary.vocab_ids[index]),
            "smiles": vocabulary.smiles[index],
            "cosine_similarity": similarity,
            "actual_hac": actual_hac,
            "actual_ring_count": actual_ring_count,
            "hac_delta": hac_delta,
            "ring_delta": ring_delta,
            "hac_pass": hac_pass,
            "ring_pass": ring_pass,
            "cosine_pass": cosine_pass,
            "uff_parameters_required": bool(require_uff_parameters),
            "uff_parameters_evaluated": uff_parameters_evaluated,
            "uff_parameters_supported": uff_parameters_supported,
            "uff_parameter_error": uff_parameter_error,
            "accepted": accepted,
            "rejection_reasons": rejection_reasons,
        }
        candidate_checks.append(record)
        if accepted:
            selected_index = index
            break
        if not cosine_pass:
            break

    diagnostics = {
        "vocabulary_path": str(vocabulary.source_path),
        "hac_ring_constraints_enabled": bool(
            enable_hac_ring_constraints
        ),
        "predicted_hac": predicted_hac,
        "predicted_ring_count": predicted_ring_count,
        "hac_overflow": bool(hac_overflow),
        "ring_overflow": bool(ring_overflow),
        "hac_tolerance": int(hac_tolerance),
        "ring_tolerance": int(ring_tolerance),
        "max_constraint_candidates": int(max_constraint_candidates),
        "minimum_cosine_similarity": min_cosine_similarity,
        "uff_parameters_required": bool(require_uff_parameters),
        "candidate_search_top_k": int(inspect_count),
        "candidate_checks": candidate_checks,
    }
    if selected_index is None:
        if constrained or require_uff_parameters:
            rejected_summary = "; ".join(
                "rank={rank}, smiles={smiles!r}, reasons={reasons}".format(
                    rank=int(record["rank"]),
                    smiles=str(record["smiles"]),
                    reasons=list(record["rejection_reasons"]),
                )
                for record in candidate_checks
            )
            raise FragmentMatchConstraintError(
                "No inspected cosine candidate satisfied reconstruction "
                f"constraints; rejected candidates: {rejected_summary}",
                diagnostics,
            )
        best_similarity = float(similarities[int(ranked_indices[0])])
        raise ValueError(
            "Best vocabulary match is below min_cosine_similarity: "
            f"{best_similarity:.8f} < {float(min_cosine_similarity):.8f}"
        )

    selected_similarity = float(similarities[selected_index])
    top_count = min(
        max(int(diagnostic_top_k), len(candidate_checks)),
        len(ranked_indices),
    )
    checked_by_rank = {
        int(record["rank"]): record for record in candidate_checks
    }
    top_matches: list[dict[str, Any]] = []
    for rank, raw_index in enumerate(ranked_indices[:top_count], start=1):
        index = int(raw_index)
        record = checked_by_rank.get(rank)
        if record is None:
            record = {
                "rank": rank,
                "vocab_id": int(vocabulary.vocab_ids[index]),
                "smiles": vocabulary.smiles[index],
                "cosine_similarity": float(similarities[index]),
                "actual_hac": int(vocabulary.hac_values[index]),
                "actual_ring_count": int(vocabulary.ring_counts[index]),
                "uff_parameters_required": bool(
                    require_uff_parameters
                ),
                "uff_parameters_evaluated": False,
                "uff_parameters_supported": None,
                "uff_parameter_error": None,
                "accepted": index == selected_index,
                "evaluated": False,
            }
        else:
            record = {**record, "evaluated": True}
        top_matches.append(record)

    metadata = {
        "vocab_id": int(vocabulary.vocab_ids[selected_index]),
        "cosine_similarity": selected_similarity,
        "selected_rank": next(
            int(record["rank"])
            for record in candidate_checks
            if bool(record["accepted"])
        ),
        "vocabulary_path": str(vocabulary.source_path),
        "top_matches": top_matches,
        "candidate_checks": candidate_checks,
        "match_constraints": diagnostics,
    }
    return vocabulary.smiles[selected_index], metadata


def _canonical_atom_id(atom: Chem.Atom) -> int:
    if atom.HasProp(CANONICAL_ATOM_ID_PROP):
        return int(atom.GetIntProp(CANONICAL_ATOM_ID_PROP))
    return int(atom.GetIdx())


def select_reference_anchor_atom_indices(
    molecule: Chem.Mol,
) -> tuple[int, int, int]:
    """Select frame anchors from canonical identity and graph topology only.

    ``molecule`` must already use the canonical-SMILES atom order.  The
    selection deliberately avoids conformer distances, so the original SDF
    and a newly generated conformer cannot choose different symmetric atoms
    merely because their bond lengths differ slightly.
    """

    if molecule is None:
        raise TypeError("molecule cannot be None")
    heavy_atom_indices = [
        atom.GetIdx()
        for atom in molecule.GetAtoms()
        if atom.GetAtomicNum() > 1
    ]
    if not heavy_atom_indices:
        raise ValueError("Cannot build a reference frame without heavy atoms")

    heavy_atom_indices.sort(
        key=lambda index: _canonical_atom_id(
            molecule.GetAtomWithIdx(index)
        )
    )
    if len(heavy_atom_indices) == 1:
        anchor = int(heavy_atom_indices[0])
        return anchor, anchor, anchor
    if len(heavy_atom_indices) == 2:
        anchor_a, anchor_b = map(int, heavy_atom_indices)
        return anchor_a, anchor_b, anchor_b

    distance_matrix = np.asarray(
        Chem.GetDistanceMatrix(molecule),
        dtype=np.float64,
    )
    anchor_a = int(heavy_atom_indices[0])
    anchor_b = min(
        (index for index in heavy_atom_indices if index != anchor_a),
        key=lambda index: (
            -float(distance_matrix[anchor_a, index]),
            _canonical_atom_id(molecule.GetAtomWithIdx(index)),
        ),
    )
    distance_ab = float(distance_matrix[anchor_a, anchor_b])

    def anchor_c_key(index: int) -> tuple[float, float, float, int]:
        distance_ac = float(distance_matrix[anchor_a, index])
        distance_bc = float(distance_matrix[anchor_b, index])
        path_excess = distance_ac + distance_bc - distance_ab
        return (
            -path_excess,
            -min(distance_ac, distance_bc),
            -(distance_ac + distance_bc),
            _canonical_atom_id(molecule.GetAtomWithIdx(index)),
        )

    anchor_c = min(
        (
            index
            for index in heavy_atom_indices
            if index not in (anchor_a, anchor_b)
        ),
        key=anchor_c_key,
    )
    return int(anchor_a), int(anchor_b), int(anchor_c)


def calculate_reference_coordinate_system(
    mol_with_conf: Chem.Mol,
) -> np.ndarray | None:
    """Return coordinates of three identity-stable canonical frame anchors."""

    if mol_with_conf is None or mol_with_conf.GetNumConformers() == 0:
        return None
    try:
        anchor_indices = select_reference_anchor_atom_indices(mol_with_conf)
    except (TypeError, ValueError):
        return None

    conformer = mol_with_conf.GetConformer(0)
    return np.asarray(
        [
            list(conformer.GetAtomPosition(atom_index))
            for atom_index in anchor_indices
        ],
        dtype=np.float64,
    )


def _optimization_metadata() -> dict[str, Any]:
    return {
        "force_field": "none",
        "optimization_status": "not_optimized",
        "optimization_return_code": None,
        "fallback_reason": None,
        "force_field_max_iters": int(FORCE_FIELD_MAX_ITERS),
        "force_field_attempts": [],
    }


@contextmanager
def _suppress_uff_typer_logs() -> Iterator[None]:
    """Temporarily block RDKit logs only around UFF typer calls."""

    if not SUPPRESS_UFF_TYPER_LOGS:
        yield
        return
    blocker_factory = getattr(rdBase, "BlockLogs", None)
    if blocker_factory is None:
        # Preserve compatibility with older RDKit builds.  UFF calculations
        # still run normally, but their messages cannot be scoped safely.
        yield
        return
    blocker = blocker_factory()
    try:
        yield
    finally:
        del blocker


def _optimize_conformer(mol: Chem.Mol) -> dict[str, Any]:
    metadata = _optimization_metadata()
    attempts: list[dict[str, Any]] = metadata["force_field_attempts"]
    mmff_failure_reason: str | None = None

    try:
        mmff_supported = bool(AllChem.MMFFHasAllMoleculeParams(mol))
    except Exception as exc:
        mmff_supported = False
        mmff_failure_reason = (
            f"mmff_parameter_check_exception: {type(exc).__name__}: {exc}"
        )

    if not mmff_supported:
        if mmff_failure_reason is None:
            mmff_failure_reason = "mmff_parameters_unavailable"
        attempts.append(
            {
                "method": MMFF_VARIANT,
                "supported": False,
                "return_code": None,
                "error": mmff_failure_reason,
            }
        )
    else:
        try:
            return_code = int(
                AllChem.MMFFOptimizeMolecule(
                    mol,
                    mmffVariant=MMFF_VARIANT,
                    maxIters=int(FORCE_FIELD_MAX_ITERS),
                )
            )
            attempts.append(
                {
                    "method": MMFF_VARIANT,
                    "supported": True,
                    "return_code": return_code,
                    "error": None,
                }
            )
            metadata["force_field"] = MMFF_VARIANT
            metadata["optimization_return_code"] = return_code
            if return_code == 0:
                metadata["optimization_status"] = "converged"
                return metadata
            metadata["optimization_status"] = "max_iterations_reached"
            mmff_failure_reason = (
                f"mmff_not_converged_return_code_{return_code}"
            )
        except Exception as exc:
            mmff_failure_reason = (
                f"mmff_optimization_exception: {type(exc).__name__}: {exc}"
            )
            attempts.append(
                {
                    "method": MMFF_VARIANT,
                    "supported": True,
                    "return_code": None,
                    "error": mmff_failure_reason,
                }
            )
            metadata["force_field"] = MMFF_VARIANT
            metadata["optimization_status"] = "failed"

    metadata["fallback_reason"] = mmff_failure_reason
    if not ENABLE_UFF_FALLBACK:
        return metadata

    try:
        with _suppress_uff_typer_logs():
            uff_supported = bool(AllChem.UFFHasAllMoleculeParams(mol))
    except Exception as exc:
        uff_supported = False
        uff_parameter_error = (
            f"uff_parameter_check_exception: {type(exc).__name__}: {exc}"
        )
    else:
        uff_parameter_error = (
            None if uff_supported else "uff_parameters_unavailable"
        )

    if not uff_supported:
        attempts.append(
            {
                "method": "UFF",
                "supported": False,
                "return_code": None,
                "error": uff_parameter_error,
            }
        )
        return metadata

    try:
        with _suppress_uff_typer_logs():
            return_code = int(
                AllChem.UFFOptimizeMolecule(
                    mol,
                    maxIters=int(FORCE_FIELD_MAX_ITERS),
                )
            )
        attempts.append(
            {
                "method": "UFF",
                "supported": True,
                "return_code": return_code,
                "error": None,
            }
        )
        metadata["force_field"] = "UFF"
        metadata["optimization_return_code"] = return_code
        metadata["optimization_status"] = (
            "converged" if return_code == 0 else "max_iterations_reached"
        )
    except Exception as exc:
        error = (
            f"uff_optimization_exception: {type(exc).__name__}: {exc}"
        )
        attempts.append(
            {
                "method": "UFF",
                "supported": True,
                "return_code": None,
                "error": error,
            }
        )
        metadata["force_field"] = "UFF"
        metadata["optimization_return_code"] = None
        metadata["optimization_status"] = "failed"

    return metadata


def _embedding_attempt_specs() -> tuple[_EmbeddingAttemptSpec, ...]:
    """Return exactly three ordered ETKDGv3 attempts with one fixed seed."""

    return (
        _EmbeddingAttemptSpec(
            profile="etkdg_v3_strict_stereochemistry",
            random_seed=int(ETKDG_RANDOM_SEED),
            use_random_coords=False,
            enforce_chirality=True,
            remove_stereochemistry=False,
        ),
        _EmbeddingAttemptSpec(
            profile="etkdg_v3_without_stereochemistry",
            random_seed=int(ETKDG_RANDOM_SEED),
            use_random_coords=False,
            enforce_chirality=False,
            remove_stereochemistry=True,
        ),
        _EmbeddingAttemptSpec(
            profile="etkdg_v3_without_stereochemistry_random_coords",
            random_seed=int(ETKDG_RANDOM_SEED),
            use_random_coords=True,
            enforce_chirality=False,
            remove_stereochemistry=True,
            ignore_smoothing_failures=bool(
                RELAXED_CHIRALITY_IGNORE_SMOOTHING_FAILURES
            ),
        ),
    )


def _single_atom_embedding_metadata() -> dict[str, Any]:
    return {
        "embedding_method": "not_required_single_atom",
        "embedding_profile": "not_required_single_atom",
        "embedding_random_seed": None,
        "embedding_chirality_relaxed": False,
        "embedding_attempt_count": 0,
        "embedding_attempts": [],
    }


def _replace_conformer_coordinates(
    target_mol: Chem.Mol,
    source_mol: Chem.Mol,
    source_atom_indices_by_target: Sequence[int],
) -> None:
    """Copy coordinates without changing the target molecule's atom order."""

    mapping = tuple(int(index) for index in source_atom_indices_by_target)
    if len(mapping) != target_mol.GetNumAtoms():
        raise RuntimeError(
            "Coordinate mapping length does not match the target atom count"
        )
    if len(set(mapping)) != len(mapping):
        raise RuntimeError("Coordinate mapping contains duplicate source atoms")
    if any(index < 0 or index >= source_mol.GetNumAtoms() for index in mapping):
        raise RuntimeError("Coordinate mapping contains an invalid atom index")
    if source_mol.GetNumConformers() == 0:
        raise RuntimeError("Coordinate source has no conformer")

    source_conformer = source_mol.GetConformer(0)
    target_conformer = Chem.Conformer(target_mol.GetNumAtoms())
    target_conformer.Set3D(True)
    for target_index, source_index in enumerate(mapping):
        target_atom = target_mol.GetAtomWithIdx(target_index)
        source_atom = source_mol.GetAtomWithIdx(source_index)
        if (
            target_atom.GetAtomicNum() != source_atom.GetAtomicNum()
            or target_atom.GetFormalCharge() != source_atom.GetFormalCharge()
            or target_atom.GetIsotope() != source_atom.GetIsotope()
        ):
            raise RuntimeError(
                "Coordinate mapping changes atom identity at target index "
                f"{target_index}"
            )
        position = source_conformer.GetAtomPosition(source_index)
        coordinates = np.asarray(
            [position.x, position.y, position.z],
            dtype=np.float64,
        )
        if not np.isfinite(coordinates).all():
            raise RuntimeError(
                "Coordinate source contains a non-finite position at atom "
                f"index {source_index}"
            )
        target_conformer.SetAtomPosition(
            target_index,
            Point3D(*(float(value) for value in coordinates)),
        )

    target_mol.RemoveAllConformers()
    target_mol.AddConformer(target_conformer, assignId=True)


def _embed_conformer_with_fallbacks(
    mol: Chem.Mol,
) -> dict[str, Any]:
    """Try strict ETKDGv3, then two stereo-free deterministic attempts."""

    attempt_records: list[dict[str, Any]] = []
    for spec in _embedding_attempt_specs():
        attempt_mol = Chem.Mol(mol)
        attempt_mol.RemoveAllConformers()
        if spec.remove_stereochemistry:
            Chem.RemoveStereochemistry(attempt_mol)
        return_code: int | None = None
        error: str | None = None

        try:
            parameters = AllChem.ETKDGv3()
            parameters.randomSeed = int(spec.random_seed)
            parameters.useRandomCoords = bool(spec.use_random_coords)
            parameters.enforceChirality = bool(spec.enforce_chirality)
            if spec.ignore_smoothing_failures:
                parameters.ignoreSmoothingFailures = True
            return_code = int(
                AllChem.EmbedMolecule(attempt_mol, parameters)
            )
        except Exception as exc:
            error = (
                f"{type(exc).__name__}: {exc}"
            )

        conformer_created = bool(attempt_mol.GetNumConformers() > 0)
        success = bool(
            return_code is not None
            and return_code >= 0
            and conformer_created
        )
        attempt_records.append(
            {
                "profile": spec.profile,
                "random_seed": int(spec.random_seed),
                "use_random_coords": bool(spec.use_random_coords),
                "enforce_chirality": bool(spec.enforce_chirality),
                "stereochemistry_removed": bool(
                    spec.remove_stereochemistry
                ),
                "ignore_smoothing_failures": bool(
                    spec.ignore_smoothing_failures
                ),
                "return_code": return_code,
                "conformer_created": conformer_created,
                "error": error,
            }
        )
        if success:
            _replace_conformer_coordinates(
                mol,
                attempt_mol,
                range(mol.GetNumAtoms()),
            )
            return {
                "embedding_method": "ETKDGv3",
                "embedding_profile": spec.profile,
                "embedding_random_seed": int(spec.random_seed),
                "embedding_chirality_relaxed": bool(
                    spec.remove_stereochemistry
                ),
                "embedding_attempt_count": len(attempt_records),
                "embedding_attempts": attempt_records,
            }

    return {
        "embedding_method": None,
        "embedding_profile": "all_etkdg_v3_attempts_failed",
        "embedding_random_seed": int(ETKDG_RANDOM_SEED),
        "embedding_chirality_relaxed": True,
        "embedding_attempt_count": len(attempt_records),
        "embedding_attempts": attempt_records,
    }


def is_non_tetrahedral_stereo_atom(atom: Chem.Atom) -> bool:
    """Return whether RDKit marks an atom as SP, TBP, or octahedral."""

    return atom.GetChiralTag() in NON_TETRAHEDRAL_CHIRAL_TAGS


def _heavy_atom_only_copy(mol: Chem.Mol) -> Chem.Mol:
    parameters = Chem.RemoveHsParameters()
    parameters.removeNontetrahedralNeighbors = True
    try:
        heavy_mol = Chem.RemoveHs(
            Chem.Mol(mol),
            parameters,
            sanitize=True,
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to construct a heavy-atom-only molecule: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if any(atom.GetAtomicNum() <= 1 for atom in heavy_mol.GetAtoms()):
        raise RuntimeError("Heavy-atom-only molecule still contains hydrogen")
    return heavy_mol


def _reconstruction_atom_copy(mol: Chem.Mol) -> Chem.Mol:
    """Return a hydrogen-free, non-stereochemical reconstruction template."""

    parameters = Chem.RemoveHsParameters()
    parameters.removeNontetrahedralNeighbors = True
    log_blocker = rdBase.BlockLogs()
    try:
        reconstruction_mol = Chem.RemoveHs(
            Chem.Mol(mol),
            parameters,
            sanitize=True,
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to remove hydrogens from reconstruction template: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    finally:
        del log_blocker

    if any(atom.GetAtomicNum() == 1 for atom in reconstruction_mol.GetAtoms()):
        raise RuntimeError(
            "Hydrogen-free reconstruction template still contains hydrogen"
        )
    Chem.RemoveStereochemistry(reconstruction_mol)
    return reconstruction_mol


def _openbabel_input_smiles(mol: Chem.Mol) -> str:
    """Tag every canonical heavy atom so Open Babel can preserve identity."""

    tagged_mol = Chem.Mol(mol)
    tagged_mol.RemoveAllConformers()
    for atom in tagged_mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)
    mapped_smiles = Chem.MolToSmiles(
        tagged_mol,
        canonical=False,
        isomericSmiles=True,
    )
    if not mapped_smiles:
        raise RuntimeError("Failed to serialize the Open Babel input SMILES")
    return mapped_smiles


def _atom_topology_signature(atom: Chem.Atom) -> tuple[int, int, int, bool, int]:
    return (
        int(atom.GetAtomicNum()),
        int(atom.GetFormalCharge()),
        int(atom.GetIsotope()),
        bool(atom.GetIsAromatic()),
        int(atom.GetNumRadicalElectrons()),
    )


def _bond_topology_signature(bond: Chem.Bond) -> tuple[str, bool]:
    return (str(bond.GetBondType()), bool(bond.GetIsAromatic()))


def _validate_canonical_heavy_atom_order(
    reference_mol: Chem.Mol,
    output_mol: Chem.Mol,
) -> None:
    """Require the returned molecule to retain canonical heavy-atom indices."""

    reference_heavy = _heavy_atom_only_copy(reference_mol)
    output_heavy = _heavy_atom_only_copy(output_mol)
    atom_count = reference_heavy.GetNumAtoms()
    if output_heavy.GetNumAtoms() != atom_count:
        raise RuntimeError(
            "Canonical heavy-atom count changed during conformer generation"
        )
    for atom_index in range(atom_count):
        reference_atom = reference_heavy.GetAtomWithIdx(atom_index)
        output_atom = output_heavy.GetAtomWithIdx(atom_index)
        if _atom_topology_signature(reference_atom) != (
            _atom_topology_signature(output_atom)
        ):
            raise RuntimeError(
                "Canonical heavy-atom order changed at index "
                f"{atom_index}"
            )

    if reference_heavy.GetNumBonds() != output_heavy.GetNumBonds():
        raise RuntimeError(
            "Canonical heavy-atom bond count changed during generation"
        )
    for reference_bond in reference_heavy.GetBonds():
        begin = reference_bond.GetBeginAtomIdx()
        end = reference_bond.GetEndAtomIdx()
        output_bond = output_heavy.GetBondBetweenAtoms(begin, end)
        if output_bond is None or _bond_topology_signature(
            reference_bond
        ) != _bond_topology_signature(output_bond):
            raise RuntimeError(
                "Canonical heavy-atom indexed connectivity changed during "
                "conformer generation"
            )


def _validate_openbabel_atom_mapping(
    target_mol: Chem.Mol,
    source_mol: Chem.Mol,
    source_atom_indices_by_target: Sequence[int],
) -> tuple[int, ...]:
    mapping = tuple(int(index) for index in source_atom_indices_by_target)
    atom_count = target_mol.GetNumAtoms()
    if source_mol.GetNumAtoms() != atom_count or len(mapping) != atom_count:
        raise RuntimeError("Open Babel changed the heavy-atom count")
    if len(set(mapping)) != atom_count:
        raise RuntimeError("Open Babel atom mapping is not one-to-one")
    if any(index < 0 or index >= atom_count for index in mapping):
        raise RuntimeError("Open Babel atom mapping contains an invalid index")

    for target_index, source_index in enumerate(mapping):
        target_atom = target_mol.GetAtomWithIdx(target_index)
        source_atom = source_mol.GetAtomWithIdx(source_index)
        if _atom_topology_signature(target_atom) != (
            _atom_topology_signature(source_atom)
        ):
            raise RuntimeError(
                "Open Babel changed atom identity at canonical heavy-atom "
                f"index {target_index}"
            )

    if target_mol.GetNumBonds() != source_mol.GetNumBonds():
        raise RuntimeError("Open Babel changed the heavy-atom bond count")
    for target_bond in target_mol.GetBonds():
        begin = mapping[target_bond.GetBeginAtomIdx()]
        end = mapping[target_bond.GetEndAtomIdx()]
        source_bond = source_mol.GetBondBetweenAtoms(begin, end)
        if source_bond is None or _bond_topology_signature(target_bond) != (
            _bond_topology_signature(source_bond)
        ):
            raise RuntimeError(
                "Open Babel changed heavy-atom connectivity or bond type"
            )
    return mapping


def _resolve_openbabel_atom_mapping(
    target_mol: Chem.Mol,
    source_mol: Chem.Mol,
) -> tuple[tuple[int, ...], str]:
    """Resolve source coordinates back to canonical RDKit heavy-atom IDs."""

    atom_map_numbers = [
        int(atom.GetAtomMapNum()) for atom in source_mol.GetAtoms()
    ]
    if any(atom_map_numbers):
        expected = set(range(1, target_mol.GetNumAtoms() + 1))
        if set(atom_map_numbers) != expected or len(atom_map_numbers) != len(
            set(atom_map_numbers)
        ):
            raise RuntimeError(
                "Open Babel returned incomplete or duplicate atom-map IDs"
            )
        source_by_map_number = {
            map_number: source_index
            for source_index, map_number in enumerate(atom_map_numbers)
        }
        mapping = tuple(
            source_by_map_number[target_index + 1]
            for target_index in range(target_mol.GetNumAtoms())
        )
        return (
            _validate_openbabel_atom_mapping(
                target_mol,
                source_mol,
                mapping,
            ),
            "atom_map_numbers",
        )

    target_query = Chem.Mol(target_mol)
    source_candidate = Chem.Mol(source_mol)
    Chem.RemoveStereochemistry(target_query)
    Chem.RemoveStereochemistry(source_candidate)
    for molecule in (target_query, source_candidate):
        for atom in molecule.GetAtoms():
            atom.SetAtomMapNum(0)

    raw_matches = source_candidate.GetSubstructMatches(
        target_query,
        uniquify=False,
        useChirality=False,
        maxMatches=int(OPENBABEL_MAX_TOPOLOGY_MATCHES),
    )
    valid_matches: set[tuple[int, ...]] = set()
    for raw_match in raw_matches:
        try:
            valid_match = _validate_openbabel_atom_mapping(
                target_mol,
                source_mol,
                raw_match,
            )
        except RuntimeError:
            continue
        valid_matches.add(valid_match)

    if len(valid_matches) == 1:
        return next(iter(valid_matches)), "unique_topology_match"
    if not valid_matches:
        raise RuntimeError(
            "Open Babel output has no valid full heavy-atom topology match"
        )
    raise RuntimeError(
        "Open Babel discarded atom-map IDs and the heavy-atom mapping is "
        "topologically ambiguous"
    )


def _read_openbabel_sdf(output_path: Path) -> Chem.Mol:
    supplier = Chem.SDMolSupplier(
        str(output_path),
        removeHs=False,
        sanitize=True,
        strictParsing=True,
    )
    molecules = [molecule for molecule in supplier if molecule is not None]
    if len(molecules) != 1:
        raise RuntimeError(
            "Open Babel output must contain exactly one valid SDF molecule; "
            f"found {len(molecules)}"
        )
    if molecules[0].GetNumConformers() == 0:
        raise RuntimeError("Open Babel output molecule has no conformer")
    return molecules[0]


def _compact_process_message(value: str | None, limit: int = 1_000) -> str:
    compact = " ".join((value or "").split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def _generate_openbabel_conformer(
    original_mol: Chem.Mol,
) -> tuple[Chem.Mol, dict[str, Any]]:
    """Run optional Open Babel Gen3D and restore canonical RDKit atom order."""

    executable = shutil.which(OPENBABEL_EXECUTABLE)
    if executable is None:
        raise RuntimeError(
            f"Open Babel executable not found: {OPENBABEL_EXECUTABLE!r}"
        )

    target_mol = _heavy_atom_only_copy(original_mol)
    mapped_smiles = _openbabel_input_smiles(target_mol)
    with tempfile.TemporaryDirectory(prefix="phislinker_obabel_") as temp_dir:
        temporary_directory = Path(temp_dir)
        input_path = temporary_directory / "fragment.smi"
        output_path = temporary_directory / "fragment.sdf"
        input_path.write_text(
            f"{mapped_smiles}\tPhiSLinker\n",
            encoding="utf-8",
        )
        environment = os.environ.copy()
        environment["OB_RANDOM_SEED"] = str(int(OPENBABEL_RANDOM_SEED))
        command = [
            executable,
            "-ismi",
            str(input_path),
            "-osdf",
            "-O",
            str(output_path),
            "--gen3d",
            OPENBABEL_GEN3D_MODE,
        ]
        try:
            completed = subprocess.run(
                command,
                cwd=str(temporary_directory),
                env=environment,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=float(OPENBABEL_TIMEOUT_SECONDS),
                check=False,
                shell=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                "Open Babel Gen3D timed out after "
                f"{OPENBABEL_TIMEOUT_SECONDS:g} seconds"
            ) from exc
        except OSError as exc:
            raise RuntimeError(
                "Failed to start Open Babel Gen3D: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        if completed.returncode != 0:
            detail = _compact_process_message(
                completed.stderr or completed.stdout
            )
            suffix = f"; output: {detail}" if detail else ""
            raise RuntimeError(
                "Open Babel Gen3D returned code "
                f"{completed.returncode}{suffix}"
            )
        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise RuntimeError("Open Babel Gen3D produced no SDF output")

        source_mol = _heavy_atom_only_copy(
            _read_openbabel_sdf(output_path)
        )
        mapping, mapping_source = _resolve_openbabel_atom_mapping(
            target_mol,
            source_mol,
        )
        _replace_conformer_coordinates(
            target_mol,
            source_mol,
            mapping,
        )

    return target_mol, {
        "backend": "OpenBabel",
        "profile": "openbabel_gen3d_fast",
        "random_seed": int(OPENBABEL_RANDOM_SEED),
        "gen3d_mode": OPENBABEL_GEN3D_MODE,
        "timeout_seconds": float(OPENBABEL_TIMEOUT_SECONDS),
        "return_code": 0,
        "conformer_created": True,
        "atom_mapping_source": mapping_source,
        "error": None,
    }


def _generate_3d_conformer(
    smiles: str,
) -> tuple[Chem.Mol, str, dict[str, Any]]:
    _validate_runtime_configuration()
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError("smiles must be a non-empty string")

    mol = mol_from_smiles_quietly(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")

    canonical_smiles = Chem.MolToSmiles(
        mol,
        canonical=True,
        isomericSmiles=True,
    )
    heavy_atom_count = sum(
        atom.GetAtomicNum() > 1 for atom in mol.GetAtoms()
    )
    if heavy_atom_count == 0:
        raise ValueError(
            "Fragment must contain at least one heavy atom "
            "(atomic number greater than 1)"
        )

    if mol.GetNumAtoms() == 1 and not OUTPUT_EXPLICIT_HYDROGENS:
        single_atom_mol = Chem.Mol(mol)
        Chem.RemoveStereochemistry(single_atom_mol)
        conformer = Chem.Conformer(single_atom_mol.GetNumAtoms())
        conformer.SetAtomPosition(0, Point3D(0.0, 0.0, 0.0))
        single_atom_mol.AddConformer(conformer, assignId=True)
        metadata = {
            **_single_atom_embedding_metadata(),
            **_optimization_metadata(),
        }
        metadata["optimization_status"] = "not_required_single_atom"
        metadata["canonical_heavy_atom_order_preserved"] = True
        metadata["retained_nontetrahedral_hydrogen_count"] = 0
        return single_atom_mol, canonical_smiles, metadata

    try:
        working_mol = Chem.AddHs(mol)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to add temporary hydrogens: {type(exc).__name__}: {exc}"
        ) from exc

    embedding_metadata = _embed_conformer_with_fallbacks(working_mol)
    if embedding_metadata["embedding_method"] == "ETKDGv3":
        optimization_metadata = _optimize_conformer(working_mol)
        if OUTPUT_EXPLICIT_HYDROGENS:
            output_mol = working_mol
        else:
            output_mol = _reconstruction_atom_copy(working_mol)
    else:
        try:
            openbabel_mol, openbabel_attempt = (
                _generate_openbabel_conformer(mol)
            )
        except Exception as exc:
            raise RuntimeError(
                "RDKit failed all three deterministic ETKDGv3 attempts; "
                "Open Babel Gen3D fast fallback also failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        etkdg_attempts = list(embedding_metadata["embedding_attempts"])
        embedding_metadata = {
            "embedding_method": "OpenBabel Gen3D",
            "embedding_profile": "openbabel_gen3d_fast",
            "embedding_random_seed": int(OPENBABEL_RANDOM_SEED),
            "embedding_chirality_relaxed": False,
            "embedding_attempt_count": len(etkdg_attempts) + 1,
            "embedding_attempts": [
                *etkdg_attempts,
                openbabel_attempt,
            ],
        }
        optimization_metadata = _optimization_metadata()
        optimization_metadata.update(
            {
                "force_field": "OpenBabel Gen3D fast",
                "optimization_status": "completed_by_openbabel",
                "optimization_return_code": 0,
                "fallback_reason": "all_etkdg_v3_attempts_failed",
                "force_field_attempts": [
                    {
                        "method": "OpenBabel Gen3D fast",
                        "supported": True,
                        "return_code": 0,
                        "error": None,
                    }
                ],
            }
        )
        if OUTPUT_EXPLICIT_HYDROGENS:
            try:
                output_mol = Chem.AddHs(openbabel_mol, addCoords=True)
            except Exception as exc:
                raise RuntimeError(
                    "Failed to add explicit hydrogens to the Open Babel "
                    f"conformer: {type(exc).__name__}: {exc}"
                ) from exc
        else:
            try:
                hydrated_openbabel_mol = Chem.AddHs(
                    openbabel_mol,
                    addCoords=True,
                )
                output_mol = _reconstruction_atom_copy(
                    hydrated_openbabel_mol
                )
            except Exception as exc:
                raise RuntimeError(
                    "Failed to construct a hydrogen-free Open Babel "
                    "reconstruction conformer: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc

    if output_mol.GetNumConformers() == 0:
        raise RuntimeError("The generated molecule has no conformer")
    _validate_canonical_heavy_atom_order(mol, output_mol)
    generation_metadata = {
        **embedding_metadata,
        **optimization_metadata,
        "canonical_heavy_atom_order_preserved": True,
        "retained_nontetrahedral_hydrogen_count": 0,
    }
    return output_mol, canonical_smiles, generation_metadata


def _validate_spatial_inputs(
    center: Sequence[float] | np.ndarray,
    reference_frame: Sequence[Sequence[float]] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    center_array = _as_finite_array(
        center,
        name="center",
        expected_shape=(3,),
        dtype=np.float64,
    )
    frame_array = _as_finite_array(
        reference_frame,
        name="reference_frame",
        expected_shape=(3, 3),
        dtype=np.float64,
    )

    return center_array, frame_array


def _align_conformer(
    mol: Chem.Mol,
    *,
    center: np.ndarray,
    reference_frame: np.ndarray,
) -> dict[str, Any]:
    if mol.GetNumConformers() == 0:
        raise ValueError("Cannot align a molecule without a conformer")

    conformer = mol.GetConformer(0)
    atom_positions = np.asarray(conformer.GetPositions(), dtype=np.float64)
    heavy_mask = np.asarray(
        [atom.GetAtomicNum() > 1 for atom in mol.GetAtoms()],
        dtype=bool,
    )
    heavy_atom_count = int(np.count_nonzero(heavy_mask))
    if heavy_atom_count == 0:
        raise ValueError("Cannot align a fragment without heavy atoms")

    molecule_center = atom_positions[heavy_mask].mean(axis=0)
    anchor_indices = select_reference_anchor_atom_indices(mol)
    anchor_atom_ids = [
        _canonical_atom_id(mol.GetAtomWithIdx(index))
        for index in anchor_indices
    ]
    target_frame_rank = int(
        np.linalg.matrix_rank(
            reference_frame,
            tol=FRAME_RANK_TOLERANCE,
        )
    )

    if heavy_atom_count == 1:
        aligned_positions = atom_positions - molecule_center + center
        for atom_index, position in enumerate(aligned_positions):
            conformer.SetAtomPosition(
                atom_index,
                Point3D(*map(float, position)),
            )
        return {
            "alignment_status": "translation_only_single_heavy_atom",
            "reference_alignment_rmsd": None,
            "rotation_matrix": np.eye(3, dtype=np.float64).tolist(),
            "rotation_determinant": 1.0,
            "reference_anchor_atom_ids": anchor_atom_ids,
            "source_frame_rank": 0,
            "target_frame_rank": target_frame_rank,
            "alignment_singular_values": [0.0, 0.0, 0.0],
        }

    source_reference = calculate_reference_coordinate_system(mol)
    if source_reference is None:
        raise RuntimeError(
            "Failed to calculate the fragment reference coordinate system"
        )

    source_vectors = source_reference - molecule_center
    source_frame_rank = int(
        np.linalg.matrix_rank(
            source_vectors,
            tol=FRAME_RANK_TOLERANCE,
        )
    )
    if source_frame_rank < 1:
        raise RuntimeError(
            "multi-atom fragment produced a rank-zero source frame"
        )
    if target_frame_rank < 1:
        raise ValueError(
            "multi-atom reference_frame must contain at least one direction"
        )

    covariance = source_vectors.T @ reference_frame
    left_vectors, singular_values, right_vectors_transposed = np.linalg.svd(
        covariance
    )
    right_vectors = right_vectors_transposed.T
    rotation = right_vectors @ left_vectors.T

    if np.linalg.det(rotation) < 0:
        right_vectors[:, -1] *= -1
        rotation = right_vectors @ left_vectors.T

    rotation_determinant = float(np.linalg.det(rotation))
    if not np.isfinite(rotation_determinant) or rotation_determinant <= 0.0:
        raise RuntimeError(
            "reference alignment did not produce a proper rotation"
        )

    centered_positions = atom_positions - molecule_center
    aligned_positions = centered_positions @ rotation.T + center
    for atom_index, position in enumerate(aligned_positions):
        conformer.SetAtomPosition(
            atom_index,
            Point3D(*map(float, position)),
        )

    aligned_source_vectors = source_vectors @ rotation.T
    alignment_rmsd = float(
        np.sqrt(np.mean((aligned_source_vectors - reference_frame) ** 2))
    )
    if min(source_frame_rank, target_frame_rank) == 1:
        alignment_status = "axis_aligned_and_translated"
    elif min(source_frame_rank, target_frame_rank) == 2:
        alignment_status = "planar_frame_rotated_and_translated"
    else:
        alignment_status = "rotated_and_translated"
    return {
        "alignment_status": alignment_status,
        "reference_alignment_rmsd": alignment_rmsd,
        "rotation_matrix": rotation.tolist(),
        "rotation_determinant": rotation_determinant,
        "reference_anchor_atom_ids": anchor_atom_ids,
        "source_frame_rank": source_frame_rank,
        "target_frame_rank": target_frame_rank,
        "alignment_singular_values": singular_values.tolist(),
    }


def build_fragment_conformer_template(
    smiles: str,
) -> FragmentConformerTemplate:
    """Generate one reusable conformer and attach canonical heavy-atom IDs.

    The canonical IDs are the RDKit atom indices obtained by parsing the
    canonical isomeric SMILES.  Explicit hydrogen atoms may therefore create
    gaps in the heavy-atom ID sequence; those gaps are intentionally retained.
    """

    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError("smiles must be a non-empty string")

    parsed_molecule = mol_from_smiles_quietly(smiles)
    if parsed_molecule is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")
    canonical_smiles = Chem.MolToSmiles(
        parsed_molecule,
        canonical=True,
        isomericSmiles=True,
    )
    canonical_molecule = mol_from_smiles_quietly(canonical_smiles)
    if canonical_molecule is None:
        raise RuntimeError(
            "RDKit failed to parse its own canonical SMILES: "
            f"{canonical_smiles!r}"
        )

    canonical_heavy_atoms = [
        (atom.GetIdx(), atom.GetAtomicNum())
        for atom in canonical_molecule.GetAtoms()
        if atom.GetAtomicNum() > 1
    ]
    molecule, generated_canonical_smiles, generation_metadata = (
        _generate_3d_conformer(canonical_smiles)
    )
    generated_heavy_atoms = [
        atom for atom in molecule.GetAtoms() if atom.GetAtomicNum() > 1
    ]

    if generated_canonical_smiles != canonical_smiles:
        raise RuntimeError(
            "Canonical SMILES changed during conformer generation: "
            f"{canonical_smiles!r} -> {generated_canonical_smiles!r}"
        )
    if len(generated_heavy_atoms) != len(canonical_heavy_atoms):
        raise RuntimeError(
            "Heavy-atom count changed during conformer generation: "
            f"{len(canonical_heavy_atoms)} -> {len(generated_heavy_atoms)}"
        )

    canonical_heavy_atom_ids: list[int] = []
    for generated_atom, (canonical_atom_id, atomic_number) in zip(
        generated_heavy_atoms,
        canonical_heavy_atoms,
    ):
        if generated_atom.GetAtomicNum() != atomic_number:
            raise RuntimeError(
                "Heavy-atom order changed during conformer generation at "
                f"canonical atom ID {canonical_atom_id}"
            )
        generated_atom.SetIntProp(
            CANONICAL_ATOM_ID_PROP,
            int(canonical_atom_id),
        )
        canonical_heavy_atom_ids.append(int(canonical_atom_id))

    return FragmentConformerTemplate(
        canonical_smiles=canonical_smiles,
        molecule=Chem.Mol(molecule),
        canonical_heavy_atom_ids=tuple(canonical_heavy_atom_ids),
        generation_metadata=dict(generation_metadata),
    )


def align_fragment_conformer_molecule(
    template: FragmentConformerTemplate,
    *,
    center: Sequence[float] | np.ndarray,
    reference_frame: Sequence[Sequence[float]] | np.ndarray,
) -> tuple[Chem.Mol, dict[str, Any]]:
    """Align and return a full template, including permitted explicit H atoms.

    ``reference_frame`` follows the decoder convention: its three rows are
    vectors relative to ``center``, not absolute points.
    """

    if not isinstance(template, FragmentConformerTemplate):
        raise TypeError("template must be a FragmentConformerTemplate")

    center_array, frame_array = _validate_spatial_inputs(
        center,
        reference_frame,
    )
    molecule = Chem.Mol(template.molecule)
    alignment_metadata = _align_conformer(
        molecule,
        center=center_array,
        reference_frame=frame_array,
    )

    atom_ids: list[int] = []
    for atom in molecule.GetAtoms():
        if atom.GetAtomicNum() <= 1:
            continue
        if not atom.HasProp(CANONICAL_ATOM_ID_PROP):
            raise RuntimeError(
                "Conformer template is missing a canonical heavy-atom ID"
            )
        atom_ids.append(atom.GetIntProp(CANONICAL_ATOM_ID_PROP))

    expected_ids = tuple(int(atom_id) for atom_id in atom_ids)
    if expected_ids != template.canonical_heavy_atom_ids:
        raise RuntimeError(
            "Canonical heavy-atom IDs changed while copying the template"
        )

    metadata = {
        "canonical_smiles": template.canonical_smiles,
        "center": center_array.tolist(),
        "reference_frame": frame_array.tolist(),
        **dict(template.generation_metadata),
        **alignment_metadata,
    }
    return molecule, metadata


def align_fragment_conformer_template(
    template: FragmentConformerTemplate,
    *,
    center: Sequence[float] | np.ndarray,
    reference_frame: Sequence[Sequence[float]] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Align a cached template and return heavy-atom IDs and coordinates."""

    molecule, metadata = align_fragment_conformer_molecule(
        template,
        center=center,
        reference_frame=reference_frame,
    )
    conformer = molecule.GetConformer(0)
    atom_ids: list[int] = []
    coordinates: list[list[float]] = []
    for atom in molecule.GetAtoms():
        if atom.GetAtomicNum() <= 1:
            continue
        position = conformer.GetAtomPosition(atom.GetIdx())
        atom_ids.append(atom.GetIntProp(CANONICAL_ATOM_ID_PROP))
        coordinates.append(
            [float(position.x), float(position.y), float(position.z)]
        )

    return (
        np.asarray(atom_ids, dtype=np.int64),
        np.asarray(coordinates, dtype=np.float64),
        metadata,
    )


def decode_single_fragment_3d(
    *,
    smiles: str | None = None,
    frame_embedding: Sequence[float] | np.ndarray | None = None,
    center: Sequence[float] | np.ndarray,
    reference_frame: Sequence[Sequence[float]] | np.ndarray,
    vocab_path: str | Path | None = None,
    vocabulary: FragmentVocabulary | None = None,
    min_cosine_similarity: float | None = (
        FRAGMENT_MATCH_MIN_COSINE_SIMILARITY
    ),
    diagnostic_top_k: int = FRAGMENT_MATCH_DIAGNOSTIC_TOP_K,
) -> tuple[Chem.Mol, dict[str, Any]]:
    """Resolve, generate, minimize, and align one fragment.

    Exactly one of ``smiles`` and ``frame_embedding`` must be supplied.
    ``reference_frame`` is a 3x3 matrix whose rows are frame vectors relative
    to ``center``.
    """

    _validate_runtime_configuration()
    has_smiles = smiles is not None
    has_embedding = frame_embedding is not None
    if has_smiles == has_embedding:
        raise ValueError(
            "Exactly one of smiles and frame_embedding must be provided"
        )

    center_array, frame_array = _validate_spatial_inputs(
        center,
        reference_frame,
    )

    if has_smiles:
        if not isinstance(smiles, str) or not smiles.strip():
            raise ValueError("smiles must be a non-empty string")
        resolved_smiles = smiles
        identity_metadata: dict[str, Any] = {
            "input_mode": "smiles",
            "vocab_id": None,
            "cosine_similarity": None,
            "vocabulary_path": None,
            "top_matches": [],
        }
    else:
        if frame_embedding is None:
            raise AssertionError("frame_embedding validation invariant failed")
        resolved_smiles, match_metadata = match_fragment_embedding(
            frame_embedding,
            vocab_path=vocab_path,
            vocabulary=vocabulary,
            min_cosine_similarity=min_cosine_similarity,
            diagnostic_top_k=diagnostic_top_k,
        )
        identity_metadata = {
            "input_mode": "frame_embedding",
            **match_metadata,
        }

    molecule, canonical_smiles, optimization_metadata = (
        _generate_3d_conformer(resolved_smiles)
    )
    alignment_metadata = _align_conformer(
        molecule,
        center=center_array,
        reference_frame=frame_array,
    )

    metadata = {
        **identity_metadata,
        "resolved_smiles": resolved_smiles,
        "canonical_smiles": canonical_smiles,
        "center": center_array.tolist(),
        "reference_frame": frame_array.tolist(),
        "output_explicit_hydrogens": bool(OUTPUT_EXPLICIT_HYDROGENS),
        **optimization_metadata,
        **alignment_metadata,
    }
    return molecule, metadata


def _set_sdf_properties(
    mol: Chem.Mol,
    metadata: Mapping[str, Any],
) -> None:
    mol.SetProp("_Name", str(metadata["canonical_smiles"]))
    scalar_properties = {
        "InputMode": metadata["input_mode"],
        "ResolvedSMILES": metadata["resolved_smiles"],
        "CanonicalSMILES": metadata["canonical_smiles"],
        "VocabID": metadata["vocab_id"],
        "CosineSimilarity": metadata["cosine_similarity"],
        "VocabularyPath": metadata["vocabulary_path"],
        "EmbeddingMethod": metadata["embedding_method"],
        "EmbeddingProfile": metadata["embedding_profile"],
        "EmbeddingRandomSeed": metadata["embedding_random_seed"],
        "EmbeddingChiralityRelaxed": metadata[
            "embedding_chirality_relaxed"
        ],
        "EmbeddingAttemptCount": metadata["embedding_attempt_count"],
        "CanonicalHeavyAtomOrderPreserved": metadata[
            "canonical_heavy_atom_order_preserved"
        ],
        "ForceField": metadata["force_field"],
        "OptimizationStatus": metadata["optimization_status"],
        "OptimizationReturnCode": metadata["optimization_return_code"],
        "FallbackReason": metadata["fallback_reason"],
        "AlignmentStatus": metadata["alignment_status"],
        "ReferenceAlignmentRMSD": metadata["reference_alignment_rmsd"],
        "OutputExplicitHydrogens": metadata[
            "output_explicit_hydrogens"
        ],
        "RetainedNonTetrahedralHydrogenCount": metadata[
            "retained_nontetrahedral_hydrogen_count"
        ],
    }
    for property_name, value in scalar_properties.items():
        mol.SetProp(
            property_name,
            "" if value is None else str(value),
        )

    json_properties = {
        "Center": metadata["center"],
        "ReferenceFrame": metadata["reference_frame"],
        "RotationMatrix": metadata["rotation_matrix"],
        "TopMatches": metadata["top_matches"],
        "EmbeddingAttempts": metadata["embedding_attempts"],
        "ForceFieldAttempts": metadata["force_field_attempts"],
    }
    for property_name, value in json_properties.items():
        mol.SetProp(
            property_name,
            json.dumps(value, ensure_ascii=False, separators=(",", ":")),
        )


def remove_nonisotopic_hydrogens(
    molecule: Chem.Mol,
    *,
    sanitize: bool = True,
    context: str = "molecule",
) -> tuple[Chem.Mol, int]:
    """Remove every ordinary explicit H while retaining isotope-labelled H."""

    if molecule is None:
        raise ValueError(f"{context} is None")
    ordinary_hydrogen_count = sum(
        atom.GetAtomicNum() == 1 and atom.GetIsotope() == 0
        for atom in molecule.GetAtoms()
    )
    parameters = Chem.RemoveHsParameters()
    parameters.removeAndTrackIsotopes = False
    parameters.removeIsotopes = False
    parameters.removeDefiningBondStereo = True
    parameters.removeDegreeZero = True
    parameters.removeDummyNeighbors = True
    parameters.removeHigherDegrees = True
    parameters.removeHydrides = True
    parameters.removeInSGroups = True
    parameters.removeMapped = True
    parameters.removeNontetrahedralNeighbors = True
    parameters.removeOnlyHNeighbors = True
    parameters.removeWithQuery = True
    parameters.removeWithWedgedBond = True
    parameters.showWarnings = False
    try:
        output_molecule = Chem.RemoveHs(
            Chem.Mol(molecule),
            parameters,
            sanitize=sanitize,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to remove non-isotopic hydrogens from {context}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    remaining_indices = [
        atom.GetIdx()
        for atom in output_molecule.GetAtoms()
        if atom.GetAtomicNum() == 1 and atom.GetIsotope() == 0
    ]
    if remaining_indices:
        raise RuntimeError(
            f"{context} still contains non-isotopic hydrogen atoms at "
            f"indices {remaining_indices}"
        )
    return output_molecule, ordinary_hydrogen_count


def write_fragment_sdf(
    mol: Chem.Mol,
    metadata: Mapping[str, Any],
    output_path: str | Path,
) -> Path:
    """Atomically write one molecule and its decode metadata to SDF."""

    path = Path(output_path).expanduser().resolve()
    if path.suffix.lower() != ".sdf":
        raise ValueError(f"Output path must end with .sdf: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)

    molecule_to_write, _ = remove_nonisotopic_hydrogens(
        mol,
        context="fragment SDF output",
    )
    _set_sdf_properties(molecule_to_write, metadata)

    temporary_path = path.with_name(
        f".{path.name}.{uuid4().hex}.tmp"
    )
    writer: Chem.SDWriter | None = None
    try:
        writer = Chem.SDWriter(str(temporary_path))
        if writer is None:
            raise OSError(
                f"RDKit could not open temporary SDF: {temporary_path}"
            )
        writer.write(molecule_to_write)
        writer.close()
        writer = None
        temporary_path.replace(path)
    except Exception:
        if writer is not None:
            writer.close()
        temporary_path.unlink(missing_ok=True)
        raise

    return path


def write_fragment_collection_sdf(
    molecules: Sequence[Chem.Mol],
    metadata_records: Sequence[Mapping[str, Any]],
    output_path: str | Path,
) -> Path:
    """Atomically write one aligned fragment per SDF record."""

    if len(molecules) != len(metadata_records):
        raise ValueError("molecules and metadata_records must have equal length")
    if not molecules:
        raise ValueError("fragment collection cannot be empty")
    path = Path(output_path).expanduser().resolve()
    if path.suffix.lower() != ".sdf":
        raise ValueError(f"Output path must end with .sdf: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    writer: Chem.SDWriter | None = None
    try:
        writer = Chem.SDWriter(str(temporary_path))
        if writer is None:
            raise OSError(f"RDKit could not open temporary SDF: {temporary_path}")
        for record_index, (molecule, metadata) in enumerate(
            zip(molecules, metadata_records)
        ):
            if molecule is None or molecule.GetNumConformers() == 0:
                raise ValueError(
                    f"fragment record {record_index} has no 3D conformer"
                )
            output_molecule, _ = remove_nonisotopic_hydrogens(
                molecule,
                context=f"fragment collection record {record_index}",
            )
            name = metadata.get(
                "canonical_smiles",
                metadata.get("resolved_smiles", f"fragment_{record_index}"),
            )
            output_molecule.SetProp("_Name", str(name))
            for property_name, metadata_key in (
                ("NodeIndex", "node_index"),
                ("GraphIndex", "graph_index"),
                ("IsFixed", "is_fixed"),
                ("InputSMILES", "input_smiles"),
                ("ResolvedSMILES", "resolved_smiles"),
                ("CanonicalSMILES", "canonical_smiles"),
                ("CosineSimilarity", "cosine_similarity"),
                ("AlignmentStatus", "alignment_status"),
                ("ReferenceAlignmentRMSD", "reference_alignment_rmsd"),
            ):
                value = metadata.get(metadata_key)
                output_molecule.SetProp(
                    property_name,
                    "" if value is None else str(value),
                )
            output_molecule.SetProp(
                "ReconstructionMetadataJSON",
                json.dumps(
                    dict(metadata),
                    ensure_ascii=False,
                    separators=(",", ":"),
                    default=str,
                ),
            )
            writer.write(output_molecule)
        writer.close()
        writer = None
        temporary_path.replace(path)
    except Exception:
        if writer is not None:
            writer.close()
        temporary_path.unlink(missing_ok=True)
        raise
    return path


def _write_single_molecule_sdf(
    molecule: Chem.Mol,
    output_path: str | Path,
    *,
    properties: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically write one 3D molecule with optional scalar properties."""

    if molecule is None or molecule.GetNumConformers() == 0:
        raise ValueError("molecule must contain one 3D conformer")
    path = Path(output_path).expanduser().resolve()
    if path.suffix.lower() != ".sdf":
        raise ValueError(f"Output path must end with .sdf: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    output_molecule, _ = remove_nonisotopic_hydrogens(
        molecule,
        context="final ligand SDF output",
    )
    for name, value in dict(properties or {}).items():
        output_molecule.SetProp(str(name), "" if value is None else str(value))
    temporary_path = path.with_name(
        f".{path.stem}.{uuid4().hex}.tmp.sdf"
    )
    writer: Chem.SDWriter | None = None
    try:
        writer = Chem.SDWriter(str(temporary_path))
        if writer is None:
            raise OSError(f"RDKit could not open temporary SDF: {temporary_path}")
        writer.write(output_molecule)
        writer.close()
        writer = None
        if not temporary_path.is_file() or temporary_path.stat().st_size <= 0:
            raise OSError(f"RDKit produced an empty SDF: {temporary_path}")
        temporary_path.replace(path)
    except Exception:
        if writer is not None:
            writer.close()
        temporary_path.unlink(missing_ok=True)
        raise
    return path


def _protein_only_copy(
    receptor: Chem.Mol,
    *,
    exclude_metals: bool = False,
) -> Chem.Mol:
    """Keep amino-acid PDB ATOM records, optionally excluding metals."""

    editable = Chem.RWMol(Chem.Mol(receptor))
    remove_indices: list[int] = []
    for atom in editable.GetAtoms():
        residue = atom.GetPDBResidueInfo()
        if (
            residue is None
            or residue.GetIsHeteroAtom()
            or (
                exclude_metals
                and int(atom.GetAtomicNum())
                in TGD_EXCLUDED_METAL_ATOMIC_NUMBERS
            )
        ):
            remove_indices.append(atom.GetIdx())
    for atom_index in reversed(remove_indices):
        editable.RemoveAtom(int(atom_index))
    protein = editable.GetMol()
    if protein.GetNumAtoms() == 0:
        raise ValueError("receptor PDB contains no protein ATOM records")
    Chem.SanitizeMol(protein)
    return protein


def _coordinates(molecule: Chem.Mol) -> np.ndarray:
    if molecule.GetNumConformers() == 0:
        raise ValueError("molecule has no conformer")
    return np.asarray(
        molecule.GetConformer(0).GetPositions(),
        dtype=np.float64,
    )


def translate_disconnected_fragments_tgd(
    molecule: Chem.Mol,
    fragment_atom_indices: Mapping[int, Sequence[int]],
    *,
    receptor_pdb_path: str | Path,
    anchor_fragment_indices: Sequence[int] = (),
) -> tuple[Chem.Mol, dict[int, np.ndarray], dict[str, Any]]:
    """Rigidly translate fragments against a fixed nonmetal protein pocket.

    RDKit exposes the UFF energy gradient, so the descent direction is the
    negative gradient.  The force field includes fragment-fragment and
    fragment-pocket interactions, while protein coordinates never move.  All
    atoms belonging to one fragment receive the same displacement; fragment
    orientation and internal geometry therefore remain unchanged.  Failure is
    diagnostic-only and returns the original molecule.
    """

    _validate_runtime_configuration()
    if molecule is None:
        raise ValueError("molecule must not be None")
    if molecule.GetNumAtoms() == 0:
        raise ValueError("molecule must contain at least one atom")
    if molecule.GetNumConformers() != 1:
        raise ValueError("molecule must contain exactly one conformer")
    if not fragment_atom_indices:
        raise ValueError("fragment_atom_indices must not be empty")
    receptor_path = Path(receptor_pdb_path).expanduser().resolve()

    atom_count = int(molecule.GetNumAtoms())
    normalized_groups: dict[int, tuple[int, ...]] = {}
    covered_atoms: list[int] = []
    for raw_fragment_index, raw_atom_indices in fragment_atom_indices.items():
        if isinstance(raw_fragment_index, bool):
            raise TypeError("fragment indices must be integers")
        fragment_index = int(raw_fragment_index)
        atom_indices = tuple(sorted(int(value) for value in raw_atom_indices))
        if not atom_indices:
            raise ValueError(
                f"fragment {fragment_index} does not contain any atoms"
            )
        if len(set(atom_indices)) != len(atom_indices):
            raise ValueError(
                f"fragment {fragment_index} contains duplicate atom indices"
            )
        if any(index < 0 or index >= atom_count for index in atom_indices):
            raise IndexError(
                f"fragment {fragment_index} contains an out-of-range atom index"
            )
        if fragment_index in normalized_groups:
            raise ValueError(f"duplicate fragment index: {fragment_index}")
        normalized_groups[fragment_index] = atom_indices
        covered_atoms.extend(atom_indices)
    if sorted(covered_atoms) != list(range(atom_count)):
        raise ValueError(
            "fragment_atom_indices must cover every molecule atom exactly once"
        )

    requested_anchors = {int(value) for value in anchor_fragment_indices}
    unknown_anchors = requested_anchors - set(normalized_groups)
    if unknown_anchors:
        raise ValueError(
            "anchor fragments are absent from fragment_atom_indices: "
            f"{sorted(unknown_anchors)}"
        )
    if requested_anchors:
        anchors = requested_anchors
        anchor_source = "caller_fixed_fragments"
    else:
        anchors = {min(normalized_groups)}
        anchor_source = "lowest_fragment_index_fallback"
    movable_fragments = sorted(set(normalized_groups) - anchors)
    zero_translations = {
        fragment_index: np.zeros(3, dtype=np.float64)
        for fragment_index in normalized_groups
    }
    base_metadata: dict[str, Any] = {
        "backend": "RDKit_UFF",
        "method": "translation_gradient_descent",
        "gradient_convention": "negative_rdkit_energy_gradient",
        "fragment_count": len(normalized_groups),
        "anchor_fragment_indices": sorted(anchors),
        "anchor_source": anchor_source,
        "movable_fragment_indices": movable_fragments,
        "max_steps": int(TGD_MAX_STEPS),
        "learning_rate": float(TGD_LEARNING_RATE),
        "max_step_distance_angstrom": float(TGD_MAX_STEP_DIST),
        "force_tolerance": float(TGD_FORCE_TOL),
        "ignore_interfragment_interactions": False,
        "receptor_included": True,
        "receptor_coordinates_fixed": True,
        "receptor_pdb": str(receptor_path),
        "pocket_cutoff_angstrom": float(TGD_POCKET_CUTOFF_ANGSTROM),
        "pocket_contact_heavy_atom_count": 0,
        "pocket_residue_count": 0,
        "pocket_atom_count": 0,
        "metal_elements_excluded": True,
        "excluded_receptor_metal_atom_count": 0,
        "excluded_receptor_metal_symbols": [],
    }

    def fallback(
        *,
        status: str,
        error: str | None,
        supported: bool | None,
    ) -> tuple[Chem.Mol, dict[int, np.ndarray], dict[str, Any]]:
        metadata = {
            **base_metadata,
            "status": status,
            "accepted": False,
            "supported": supported,
            "iterations_completed": 0,
            "stop_reason": status,
            "initial_energy": None,
            "final_energy": None,
            "maximum_internal_distance_change_angstrom": 0.0,
            "fragment_translations_angstrom": {
                str(index): [0.0, 0.0, 0.0]
                for index in sorted(normalized_groups)
            },
            "steps": [],
            "error": error,
        }
        return Chem.Mol(molecule), dict(zero_translations), metadata

    if len(normalized_groups) == 1:
        output = Chem.Mol(molecule)
        metadata = {
            **base_metadata,
            "status": "not_required_single_fragment",
            "accepted": True,
            "supported": None,
            "iterations_completed": 0,
            "stop_reason": "single_fragment",
            "initial_energy": None,
            "final_energy": None,
            "maximum_internal_distance_change_angstrom": 0.0,
            "fragment_translations_angstrom": {
                str(index): [0.0, 0.0, 0.0]
                for index in sorted(normalized_groups)
            },
            "steps": [],
            "error": None,
        }
        return output, dict(zero_translations), metadata
    if not movable_fragments:
        output = Chem.Mol(molecule)
        metadata = {
            **base_metadata,
            "status": "not_required_all_fragments_anchored",
            "accepted": True,
            "supported": None,
            "iterations_completed": 0,
            "stop_reason": "all_fragments_anchored",
            "initial_energy": None,
            "final_energy": None,
            "maximum_internal_distance_change_angstrom": 0.0,
            "fragment_translations_angstrom": {
                str(index): [0.0, 0.0, 0.0]
                for index in sorted(normalized_groups)
            },
            "steps": [],
            "error": None,
        }
        return output, dict(zero_translations), metadata

    original = Chem.Mol(molecule)
    working = Chem.Mol(molecule)
    try:
        Chem.SanitizeMol(working)
        if not receptor_path.is_file():
            raise FileNotFoundError(f"receptor PDB not found: {receptor_path}")
        receptor = Chem.MolFromPDBFile(
            str(receptor_path),
            removeHs=False,
            sanitize=False,
            proximityBonding=True,
        )
        if receptor is None or receptor.GetNumConformers() == 0:
            raise ValueError(
                f"RDKit could not read receptor PDB: {receptor_path}"
            )
        excluded_metal_symbols = tuple(
            str(atom.GetSymbol())
            for atom in receptor.GetAtoms()
            if int(atom.GetAtomicNum())
            in TGD_EXCLUDED_METAL_ATOMIC_NUMBERS
        )
        base_metadata.update(
            {
                "excluded_receptor_metal_atom_count": len(
                    excluded_metal_symbols
                ),
                "excluded_receptor_metal_symbols": sorted(
                    excluded_metal_symbols
                ),
            }
        )
        nonmetal_protein = _protein_only_copy(
            receptor,
            exclude_metals=True,
        )
        contact_indices = _pocket_atom_indices(
            nonmetal_protein,
            working,
            float(TGD_POCKET_CUTOFF_ANGSTROM),
        )
        if not contact_indices:
            raise ValueError(
                "no nonmetal protein heavy atom lies within "
                f"{TGD_POCKET_CUTOFF_ANGSTROM:.3f} A of the fragments"
            )
        pocket = _protein_residue_subset(nonmetal_protein, contact_indices)
        pocket_atom_count = int(pocket.GetNumAtoms())
        base_metadata.update(
            {
                "pocket_contact_heavy_atom_count": len(contact_indices),
                "pocket_residue_count": _pocket_residue_count(
                    nonmetal_protein,
                    contact_indices,
                ),
                "pocket_atom_count": pocket_atom_count,
            }
        )

        combined = Chem.CombineMols(pocket, working)
        Chem.GetSymmSSSR(combined)
        with _suppress_uff_typer_logs():
            supported = bool(AllChem.UFFHasAllMoleculeParams(combined))
        if not supported:
            return fallback(
                status="parameters_unavailable",
                error="uff_parameters_unavailable_for_tgd_pocket_system",
                supported=False,
            )
        vdw_threshold = max(10.0, float(TGD_POCKET_CUTOFF_ANGSTROM))
        with _suppress_uff_typer_logs():
            force_field = AllChem.UFFGetMoleculeForceField(
                combined,
                vdwThresh=vdw_threshold,
                confId=0,
                ignoreInterfragInteractions=False,
            )
        if force_field is None:
            return fallback(
                status="force_field_unavailable",
                error="rdkit_did_not_create_tgd_pocket_force_field",
                supported=True,
            )
        if int(force_field.NumPoints()) != int(combined.GetNumAtoms()):
            raise RuntimeError("TGD pocket UFF point count is inconsistent")
        force_field.Initialize()

        pocket_positions = _coordinates(pocket).copy()
        original_positions = _coordinates(original).copy()
        positions = original_positions.copy()
        initial_flat_positions = np.concatenate(
            (pocket_positions, positions),
            axis=0,
        ).reshape(-1).tolist()
        initial_energy = float(force_field.CalcEnergy(initial_flat_positions))
        if not np.isfinite(initial_energy):
            raise RuntimeError("initial TGD UFF energy is not finite")

        step_logs: list[dict[str, Any]] = []
        stop_reason = "max_steps_reached"
        for step_index in range(int(TGD_MAX_STEPS)):
            flat_positions = np.concatenate(
                (pocket_positions, positions),
                axis=0,
            ).reshape(-1).tolist()
            gradient = np.asarray(
                force_field.CalcGrad(flat_positions),
                dtype=np.float64,
            )
            if gradient.size != combined.GetNumAtoms() * 3:
                raise RuntimeError(
                    "TGD pocket UFF gradient size differs from coordinate size"
                )
            gradient = gradient.reshape(combined.GetNumAtoms(), 3)
            if not np.all(np.isfinite(gradient)):
                raise RuntimeError("TGD UFF gradient contains NaN or Inf")
            force_vectors = -gradient[pocket_atom_count:]
            centroids = {
                fragment_index: np.mean(positions[list(atom_indices)], axis=0)
                for fragment_index, atom_indices in normalized_groups.items()
            }
            displacements: dict[int, np.ndarray] = {}
            fragment_step_records: list[dict[str, Any]] = []
            maximum_net_force = 0.0
            has_movement = False

            for fragment_index in sorted(normalized_groups):
                atom_indices = normalized_groups[fragment_index]
                net_force = np.sum(force_vectors[list(atom_indices)], axis=0)
                net_force_magnitude = float(np.linalg.norm(net_force))
                maximum_net_force = max(maximum_net_force, net_force_magnitude)
                displacement = np.zeros(3, dtype=np.float64)
                reversed_by_centroid_rule = False
                is_anchor = fragment_index in anchors
                if (
                    not is_anchor
                    and net_force_magnitude >= float(TGD_FORCE_TOL)
                ):
                    displacement = net_force * float(TGD_LEARNING_RATE)
                    displacement_magnitude = float(np.linalg.norm(displacement))
                    if displacement_magnitude > float(TGD_MAX_STEP_DIST):
                        displacement *= float(TGD_MAX_STEP_DIST) / (
                            displacement_magnitude
                        )

                    current_centroid = centroids[fragment_index]
                    old_distance_sum = sum(
                        float(np.linalg.norm(current_centroid - other_centroid))
                        for other_index, other_centroid in centroids.items()
                        if other_index != fragment_index
                    )
                    if old_distance_sum > 0.0:
                        proposed_centroid = current_centroid + displacement
                        new_distance_sum = sum(
                            float(
                                np.linalg.norm(
                                    proposed_centroid - other_centroid
                                )
                            )
                            for other_index, other_centroid in centroids.items()
                            if other_index != fragment_index
                        )
                        if new_distance_sum < old_distance_sum:
                            displacement *= -1.0
                            reversed_by_centroid_rule = True
                    if float(np.linalg.norm(displacement)) > NUMERIC_EPSILON:
                        has_movement = True
                displacements[fragment_index] = displacement
                fragment_step_records.append(
                    {
                        "fragment_index": fragment_index,
                        "is_anchor": is_anchor,
                        "net_force_magnitude": net_force_magnitude,
                        "displacement_magnitude_angstrom": float(
                            np.linalg.norm(displacement)
                        ),
                        "reversed_by_centroid_rule": (
                            reversed_by_centroid_rule
                        ),
                    }
                )

            for fragment_index, displacement in displacements.items():
                positions[list(normalized_groups[fragment_index])] += displacement
            if not np.all(np.isfinite(positions)):
                raise RuntimeError("TGD produced non-finite coordinates")
            step_logs.append(
                {
                    "step": step_index + 1,
                    "maximum_net_force_magnitude": maximum_net_force,
                    "fragments": fragment_step_records,
                }
            )
            if not has_movement:
                stop_reason = "no_movable_fragment_above_force_tolerance"
                break
            if maximum_net_force < float(TGD_FORCE_TOL):
                stop_reason = "force_tolerance_reached"
                break

        working_conformer = working.GetConformer(0)
        for atom_index, position in enumerate(positions):
            working_conformer.SetAtomPosition(
                int(atom_index),
                Point3D(*map(float, position)),
            )
        Chem.SanitizeMol(working)

        translations: dict[int, np.ndarray] = {}
        maximum_internal_change = 0.0
        for fragment_index, atom_indices in normalized_groups.items():
            index_array = np.asarray(atom_indices, dtype=np.int64)
            initial_fragment_positions = original_positions[index_array]
            final_fragment_positions = positions[index_array]
            translation = np.mean(final_fragment_positions, axis=0) - np.mean(
                initial_fragment_positions,
                axis=0,
            )
            translations[fragment_index] = translation.astype(
                np.float64,
                copy=True,
            )
            initial_distances = np.linalg.norm(
                initial_fragment_positions[:, None, :]
                - initial_fragment_positions[None, :, :],
                axis=2,
            )
            final_distances = np.linalg.norm(
                final_fragment_positions[:, None, :]
                - final_fragment_positions[None, :, :],
                axis=2,
            )
            maximum_internal_change = max(
                maximum_internal_change,
                float(np.max(np.abs(final_distances - initial_distances))),
            )
        if maximum_internal_change > 1.0e-6:
            raise RuntimeError(
                "TGD changed fragment internal geometry by "
                f"{maximum_internal_change:.6g} A"
            )

        final_flat_positions = np.concatenate(
            (pocket_positions, positions),
            axis=0,
        ).reshape(-1).tolist()
        final_energy = float(force_field.CalcEnergy(final_flat_positions))
        if not np.isfinite(final_energy):
            raise RuntimeError("final TGD UFF energy is not finite")
        metadata = {
            **base_metadata,
            "status": "completed",
            "accepted": True,
            "supported": True,
            "iterations_completed": len(step_logs),
            "stop_reason": stop_reason,
            "initial_energy": initial_energy,
            "final_energy": final_energy,
            "energy_change": final_energy - initial_energy,
            "vdw_threshold_angstrom": float(vdw_threshold),
            "maximum_internal_distance_change_angstrom": (
                maximum_internal_change
            ),
            "fragment_translations_angstrom": {
                str(index): list(map(float, translations[index]))
                for index in sorted(translations)
            },
            "steps": step_logs,
            "error": None,
        }
        return working, translations, metadata
    except Exception as exc:
        return fallback(
            status="failed_fallback_original_coordinates",
            error=f"{type(exc).__name__}: {exc}",
            supported=None,
        )


def _pocket_atom_indices(
    protein: Chem.Mol,
    ligand: Chem.Mol,
    cutoff_angstrom: float,
) -> tuple[int, ...]:
    protein_positions = _coordinates(protein)
    ligand_positions = _coordinates(ligand)
    ligand_heavy = np.asarray(
        [
            atom.GetIdx()
            for atom in ligand.GetAtoms()
            if atom.GetAtomicNum() > 1
        ],
        dtype=np.int64,
    )
    if ligand_heavy.size == 0:
        raise ValueError("ligand contains no heavy atoms")
    selected: list[int] = []
    cutoff_squared = float(cutoff_angstrom) ** 2
    ligand_heavy_positions = ligand_positions[ligand_heavy]
    for atom in protein.GetAtoms():
        if atom.GetAtomicNum() <= 1:
            continue
        delta = ligand_heavy_positions - protein_positions[atom.GetIdx()]
        if bool(np.any(np.sum(delta * delta, axis=1) <= cutoff_squared)):
            selected.append(int(atom.GetIdx()))
    return tuple(selected)


def _pocket_residue_count(
    protein: Chem.Mol,
    atom_indices: Sequence[int],
) -> int:
    residues: set[tuple[str, int, str, str]] = set()
    for atom_index in atom_indices:
        residue = protein.GetAtomWithIdx(int(atom_index)).GetPDBResidueInfo()
        if residue is None:
            continue
        residues.add(
            (
                residue.GetChainId().strip(),
                int(residue.GetResidueNumber()),
                residue.GetInsertionCode().strip(),
                residue.GetResidueName().strip(),
            )
        )
    return len(residues)


def _protein_residue_subset(
    protein: Chem.Mol,
    contact_atom_indices: Sequence[int],
) -> Chem.Mol:
    """Retain complete residues containing at least one selected atom."""

    selected_residues: set[tuple[str, int, str, str]] = set()
    for atom_index in contact_atom_indices:
        residue = protein.GetAtomWithIdx(int(atom_index)).GetPDBResidueInfo()
        if residue is None:
            continue
        selected_residues.add(
            (
                residue.GetChainId().strip(),
                int(residue.GetResidueNumber()),
                residue.GetInsertionCode().strip(),
                residue.GetResidueName().strip(),
            )
        )
    if not selected_residues:
        raise ValueError(
            "no complete protein residue was selected for optimization"
        )

    editable = Chem.RWMol(Chem.Mol(protein))
    remove_indices: list[int] = []
    for atom in editable.GetAtoms():
        residue = atom.GetPDBResidueInfo()
        if residue is None:
            remove_indices.append(atom.GetIdx())
            continue
        key = (
            residue.GetChainId().strip(),
            int(residue.GetResidueNumber()),
            residue.GetInsertionCode().strip(),
            residue.GetResidueName().strip(),
        )
        if key not in selected_residues:
            remove_indices.append(atom.GetIdx())
    for atom_index in reversed(remove_indices):
        editable.RemoveAtom(int(atom_index))
    pocket = editable.GetMol()
    if pocket.GetNumAtoms() == 0:
        raise ValueError("selected protein pocket is empty")
    Chem.SanitizeMol(pocket)
    return pocket


def _ligand_with_pdb_residue_info(ligand: Chem.Mol) -> Chem.Mol:
    output = Chem.Mol(ligand)
    element_counts: dict[str, int] = {}
    for atom in output.GetAtoms():
        symbol = atom.GetSymbol().upper()
        element_counts[symbol] = element_counts.get(symbol, 0) + 1
        atom_name = f"{symbol}{element_counts[symbol]}"[:4].rjust(4)
        residue = Chem.AtomPDBResidueInfo()
        residue.SetName(atom_name)
        residue.SetResidueName("LIG")
        residue.SetResidueNumber(1)
        residue.SetChainId("Z")
        residue.SetIsHeteroAtom(True)
        atom.SetMonomerInfo(residue)
    return output


def _write_complex_pdb(
    receptor: Chem.Mol,
    ligand: Chem.Mol,
    output_path: str | Path,
) -> Path:
    path = Path(output_path).expanduser().resolve()
    if path.suffix.lower() != ".pdb":
        raise ValueError(f"Output path must end with .pdb: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    clean_receptor, _ = remove_nonisotopic_hydrogens(
        receptor,
        sanitize=False,
        context="final complex receptor",
    )
    clean_ligand, _ = remove_nonisotopic_hydrogens(
        ligand,
        context="final complex ligand",
    )
    complex_molecule = Chem.CombineMols(
        clean_receptor,
        _ligand_with_pdb_residue_info(clean_ligand),
    )
    temporary_path = path.with_name(
        f".{path.stem}.{uuid4().hex}.tmp.pdb"
    )
    try:
        Chem.MolToPDBFile(complex_molecule, str(temporary_path), confId=0)
        if not temporary_path.is_file() or temporary_path.stat().st_size == 0:
            raise OSError("RDKit produced an empty complex PDB")
        temporary_path.replace(path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return path


def _run_fixed_receptor_force_field_attempt(
    protein: Chem.Mol,
    ligand: Chem.Mol,
    *,
    method: str,
    cutoff_angstrom: float,
    max_iterations: int,
) -> tuple[Chem.Mol | None, dict[str, Any]]:
    """Run one isolated force-field attempt from the original coordinates."""

    attempt: dict[str, Any] = {
        "method": method,
        "supported": None,
        "status": "failed",
        "return_code": None,
        "initial_energy": None,
        "final_energy": None,
        "error": None,
    }
    try:
        ligand_input = Chem.Mol(ligand)
        protein_atom_count = protein.GetNumAtoms()
        ligand_atom_count = ligand_input.GetNumAtoms()
        combined = Chem.CombineMols(protein, ligand_input)
        Chem.SanitizeMol(combined)
        system = Chem.AddHs(combined, addCoords=True)

        if method == MMFF_VARIANT:
            supported = bool(AllChem.MMFFHasAllMoleculeParams(system))
            attempt["supported"] = supported
            if not supported:
                attempt["status"] = "parameters_unavailable"
                attempt["error"] = "mmff94_parameters_unavailable"
                return None, attempt
            properties = AllChem.MMFFGetMoleculeProperties(
                system, mmffVariant=MMFF_VARIANT
            )
            if properties is None:
                raise RuntimeError(
                    "RDKit did not create MMFF94 molecule properties"
                )
            force_field = AllChem.MMFFGetMoleculeForceField(
                system,
                properties,
                nonBondedThresh=float(cutoff_angstrom),
                confId=0,
                ignoreInterfragInteractions=False,
            )
        elif method == "UFF":
            with _suppress_uff_typer_logs():
                supported = bool(AllChem.UFFHasAllMoleculeParams(system))
            attempt["supported"] = supported
            if not supported:
                attempt["status"] = "parameters_unavailable"
                attempt["error"] = "uff_parameters_unavailable"
                return None, attempt
            with _suppress_uff_typer_logs():
                force_field = AllChem.UFFGetMoleculeForceField(
                    system,
                    vdwThresh=float(cutoff_angstrom),
                    confId=0,
                    ignoreInterfragInteractions=False,
                )
        else:
            raise ValueError(f"unsupported force-field method: {method}")
        if force_field is None:
            raise RuntimeError(f"RDKit did not create a {method} force field")

        original_atom_count = protein_atom_count + ligand_atom_count
        fixed_indices = set(range(protein_atom_count))
        for atom_index in range(original_atom_count, system.GetNumAtoms()):
            atom = system.GetAtomWithIdx(atom_index)
            if any(
                neighbour.GetIdx() < protein_atom_count
                for neighbour in atom.GetNeighbors()
            ):
                fixed_indices.add(atom_index)
        for atom_index in sorted(fixed_indices):
            force_field.AddFixedPoint(int(atom_index))

        force_field.Initialize()
        initial_energy = float(force_field.CalcEnergy())
        return_code = int(
            force_field.Minimize(maxIts=int(max_iterations))
        )
        final_energy = float(force_field.CalcEnergy())
        attempt.update(
            {
                "return_code": return_code,
                "initial_energy": initial_energy,
                "final_energy": final_energy,
            }
        )
        if not np.isfinite(initial_energy) or not np.isfinite(final_energy):
            raise RuntimeError(f"{method} produced a non-finite energy")
        if return_code != 0:
            attempt["status"] = "max_iterations_reached"
            attempt["error"] = f"nonconverged_return_code_{return_code}"
            return None, attempt

        optimized_ligand = Chem.Mol(ligand_input)
        source_conformer = system.GetConformer(0)
        target_conformer = optimized_ligand.GetConformer(0)
        for local_index in range(ligand_atom_count):
            source_index = protein_atom_count + local_index
            if (
                system.GetAtomWithIdx(source_index).GetAtomicNum()
                != optimized_ligand.GetAtomWithIdx(
                    local_index
                ).GetAtomicNum()
            ):
                raise RuntimeError(
                    "RDKit changed original ligand atom ordering while adding H"
                )
            target_conformer.SetAtomPosition(
                local_index,
                source_conformer.GetAtomPosition(source_index),
            )
        attempt["status"] = "converged"
        return optimized_ligand, attempt
    except Exception as exc:
        attempt["status"] = "failed"
        attempt["error"] = f"{type(exc).__name__}: {exc}"
        return None, attempt


def _legacy_optimize_ligand_with_fixed_receptor(
    ligand: Chem.Mol,
    *,
    receptor_pdb_path: str | Path,
    output_pdb_path: str | Path,
    cutoff_angstrom: float = 6.0,
    max_iterations: int = 200,
) -> tuple[Chem.Mol, dict[str, Any]]:
    """Inactive atom-wise protein/ligand force-field implementation."""

    if ligand is None or ligand.GetNumConformers() == 0:
        raise ValueError("ligand must contain one 3D conformer")
    if cutoff_angstrom <= 0.0:
        raise ValueError("cutoff_angstrom must be positive")
    if int(max_iterations) <= 0:
        raise ValueError("max_iterations must be positive")
    receptor_path = Path(receptor_pdb_path).expanduser().resolve()
    if not receptor_path.is_file():
        raise FileNotFoundError(f"receptor PDB not found: {receptor_path}")

    receptor = Chem.MolFromPDBFile(
        str(receptor_path),
        removeHs=False,
        sanitize=False,
        proximityBonding=True,
    )
    if receptor is None or receptor.GetNumConformers() == 0:
        raise ValueError(f"RDKit could not read receptor PDB: {receptor_path}")

    original_ligand = Chem.Mol(ligand)
    Chem.SanitizeMol(original_ligand)
    initial_positions = _coordinates(original_ligand).copy()
    selected_ligand = Chem.Mol(original_ligand)
    selected_method: str | None = None
    status = "not_optimized"
    warning: str | None = None
    attempts: list[dict[str, Any]] = []
    full_protein: Chem.Mol | None = None
    pocket_indices: tuple[int, ...] = ()
    protein: Chem.Mol | None = None

    try:
        full_protein = _protein_only_copy(receptor)
        pocket_indices = _pocket_atom_indices(
            full_protein,
            original_ligand,
            float(cutoff_angstrom),
        )
        if not pocket_indices:
            raise ValueError(
                "no protein heavy atom lies within "
                f"{float(cutoff_angstrom):.3f} A of the ligand"
            )
        protein = _protein_residue_subset(full_protein, pocket_indices)
    except Exception as exc:
        preparation_error = f"{type(exc).__name__}: {exc}"
        warning = (
            "复合物优化系统准备失败；已使用连通后的原始配体坐标输出"
            f"未优化复合物。{preparation_error}"
        )
        attempts.append(
            {
                "method": "system_preparation",
                "supported": None,
                "status": "failed",
                "return_code": None,
                "initial_energy": None,
                "final_energy": None,
                "error": preparation_error,
            }
        )
        for method in (MMFF_VARIANT, "UFF"):
            attempts.append(
                {
                    "method": method,
                    "supported": None,
                    "status": "skipped_system_preparation_failed",
                    "return_code": None,
                    "initial_energy": None,
                    "final_energy": None,
                    "error": preparation_error,
                }
            )

    if protein is not None:
        for method, success_status in (
            (MMFF_VARIANT, "mmff94_converged"),
            ("UFF", "uff_converged"),
        ):
            optimized, attempt = _run_fixed_receptor_force_field_attempt(
                protein,
                original_ligand,
                method=method,
                cutoff_angstrom=float(cutoff_angstrom),
                max_iterations=int(max_iterations),
            )
            attempts.append(attempt)
            if optimized is not None:
                selected_ligand = optimized
                selected_method = method
                status = success_status
                break
        if selected_method is None:
            reasons = "; ".join(
                f"{attempt['method']}: {attempt.get('error')}"
                for attempt in attempts
            )
            warning = (
                "MMFF94 和 UFF 优化均失败；已使用连通后的原始配体"
                f"坐标输出未优化复合物。{reasons}"
            )

    output_path = _write_complex_pdb(
        receptor,
        selected_ligand,
        output_pdb_path,
    )
    final_positions = _coordinates(selected_ligand)
    heavy_indices = np.asarray(
        [
            atom.GetIdx()
            for atom in selected_ligand.GetAtoms()
            if atom.GetAtomicNum() > 1
        ],
        dtype=np.int64,
    )
    displacement_norms = np.linalg.norm(
        final_positions[heavy_indices] - initial_positions[heavy_indices],
        axis=1,
    )
    final_pocket_indices: tuple[int, ...] = ()
    if full_protein is not None:
        try:
            final_pocket_indices = _pocket_atom_indices(
                full_protein,
                selected_ligand,
                float(cutoff_angstrom),
            )
        except Exception:
            final_pocket_indices = ()
    selected_attempt = next(
        (
            attempt
            for attempt in attempts
            if attempt.get("method") == selected_method
        ),
        None,
    )
    metadata = {
        "status": status,
        "selected_method": selected_method,
        "max_iterations": int(max_iterations),
        "attempts": attempts,
        "initial_energy": (
            selected_attempt.get("initial_energy")
            if selected_attempt is not None
            else None
        ),
        "final_energy": (
            selected_attempt.get("final_energy")
            if selected_attempt is not None
            else None
        ),
        "fallback_to_unoptimized": selected_method is None,
        "warning": warning,
        "nonbonded_cutoff_angstrom": float(cutoff_angstrom),
        "ignore_interfragment_interactions": False,
        "protein_coordinates_fixed": True,
        "optimization_pocket_atom_count": (
            int(protein.GetNumAtoms()) if protein is not None else 0
        ),
        "ligand_atom_count": int(selected_ligand.GetNumAtoms()),
        "initial_pocket_heavy_atom_count": len(pocket_indices),
        "initial_pocket_residue_count": (
            _pocket_residue_count(full_protein, pocket_indices)
            if full_protein is not None
            else 0
        ),
        "final_pocket_heavy_atom_count": len(final_pocket_indices),
        "final_pocket_residue_count": (
            _pocket_residue_count(full_protein, final_pocket_indices)
            if full_protein is not None
            else 0
        ),
        "ligand_heavy_atom_rms_displacement_angstrom": float(
            np.sqrt(np.mean(displacement_norms ** 2))
        ),
        "ligand_heavy_atom_max_displacement_angstrom": float(
            np.max(displacement_norms)
        ),
        "receptor_pdb": str(receptor_path),
        "complex_pdb": str(output_path),
    }
    return selected_ligand, metadata


def _run_ligand_force_field_attempt(
    ligand: Chem.Mol,
    *,
    method: str,
) -> tuple[Chem.Mol | None, dict[str, Any]]:
    """Relax one ligand without position constraints."""

    attempt: dict[str, Any] = {
        "stage": "ligand_internal_relaxation",
        "method": method,
        "supported": None,
        "status": "failed",
        "accepted": False,
        "return_code": None,
        "initial_energy": None,
        "final_energy": None,
        "max_iterations": int(LIGAND_RELAX_MAX_ITERATIONS),
        "energy_tolerance": float(LIGAND_RELAX_ENERGY_TOLERANCE),
        "force_tolerance": float(NUMERIC_EPSILON),
        "position_constraints_applied": False,
        "error": None,
    }
    try:
        ligand_input = Chem.Mol(ligand)
        Chem.SanitizeMol(ligand_input)
        original_atom_count = int(ligand_input.GetNumAtoms())
        system = Chem.AddHs(ligand_input, addCoords=True)

        if method == MMFF_VARIANT:
            supported = bool(AllChem.MMFFHasAllMoleculeParams(system))
            attempt["supported"] = supported
            if not supported:
                attempt["status"] = "parameters_unavailable"
                attempt["error"] = "mmff94_parameters_unavailable"
                return None, attempt
            properties = AllChem.MMFFGetMoleculeProperties(
                system,
                mmffVariant=MMFF_VARIANT,
            )
            if properties is None:
                raise RuntimeError(
                    "RDKit did not create MMFF94 molecule properties"
                )
            force_field = AllChem.MMFFGetMoleculeForceField(
                system,
                properties,
                confId=0,
                ignoreInterfragInteractions=False,
            )
        elif method == "UFF":
            with _suppress_uff_typer_logs():
                supported = bool(AllChem.UFFHasAllMoleculeParams(system))
            attempt["supported"] = supported
            if not supported:
                attempt["status"] = "parameters_unavailable"
                attempt["error"] = "uff_parameters_unavailable"
                return None, attempt
            with _suppress_uff_typer_logs():
                force_field = AllChem.UFFGetMoleculeForceField(
                    system,
                    confId=0,
                    ignoreInterfragInteractions=False,
                )
        else:
            raise ValueError(f"unsupported force-field method: {method}")
        if force_field is None:
            raise RuntimeError(f"RDKit did not create a {method} force field")

        heavy_atom_count = sum(
            1
            for atom_index in range(original_atom_count)
            if system.GetAtomWithIdx(atom_index).GetAtomicNum() > 1
        )
        if heavy_atom_count == 0:
            raise ValueError("ligand contains no heavy atoms")

        force_field.Initialize()
        initial_energy = float(force_field.CalcEnergy())
        return_code = int(
            force_field.Minimize(
                maxIts=int(LIGAND_RELAX_MAX_ITERATIONS),
                forceTol=float(NUMERIC_EPSILON),
                energyTol=float(LIGAND_RELAX_ENERGY_TOLERANCE),
            )
        )
        final_energy = float(force_field.CalcEnergy())
        if not np.isfinite(initial_energy) or not np.isfinite(final_energy):
            raise RuntimeError(f"{method} produced a non-finite energy")

        optimized_ligand = Chem.Mol(ligand_input)
        source_conformer = system.GetConformer(0)
        target_conformer = optimized_ligand.GetConformer(0)
        for atom_index in range(original_atom_count):
            if (
                system.GetAtomWithIdx(atom_index).GetAtomicNum()
                != optimized_ligand.GetAtomWithIdx(atom_index).GetAtomicNum()
            ):
                raise RuntimeError(
                    "RDKit changed original ligand atom ordering while adding H"
                )
            target_conformer.SetAtomPosition(
                atom_index,
                source_conformer.GetAtomPosition(atom_index),
            )
        Chem.SanitizeMol(optimized_ligand)
        if not np.all(np.isfinite(_coordinates(optimized_ligand))):
            raise RuntimeError(f"{method} produced non-finite coordinates")

        attempt.update(
            {
                "status": (
                    "converged"
                    if return_code == 0
                    else "max_iterations_reached"
                ),
                "accepted": True,
                "return_code": return_code,
                "initial_energy": initial_energy,
                "final_energy": final_energy,
                "energy_change": final_energy - initial_energy,
                "heavy_atom_count": heavy_atom_count,
            }
        )
        return optimized_ligand, attempt
    except Exception as exc:
        attempt["status"] = "failed"
        attempt["error"] = f"{type(exc).__name__}: {exc}"
        return None, attempt


def optimize_ligand_with_fixed_receptor(
    ligand: Chem.Mol,
    *,
    receptor_pdb_path: str | Path,
    output_pdb_path: str | Path,
    relaxed_ligand_sdf_path: str | Path | None = None,
) -> tuple[Chem.Mol, dict[str, Any]]:
    """Relax ligand internals and write it with an unchanged receptor."""

    _validate_runtime_configuration()
    if ligand is None or ligand.GetNumConformers() == 0:
        raise ValueError("ligand must contain one 3D conformer")
    receptor_path = Path(receptor_pdb_path).expanduser().resolve()
    if not receptor_path.is_file():
        raise FileNotFoundError(f"receptor PDB not found: {receptor_path}")
    receptor = Chem.MolFromPDBFile(
        str(receptor_path),
        removeHs=False,
        sanitize=False,
        proximityBonding=True,
    )
    if receptor is None or receptor.GetNumConformers() == 0:
        raise ValueError(f"RDKit could not read receptor PDB: {receptor_path}")

    original_ligand = Chem.Mol(ligand)
    Chem.SanitizeMol(original_ligand)
    original_positions = _coordinates(original_ligand).copy()
    relaxed_ligand: Chem.Mol | None = None
    relaxation_method: str | None = None
    relaxation_attempts: list[dict[str, Any]] = []
    methods = [MMFF_VARIANT]
    if ENABLE_UFF_FALLBACK:
        methods.append("UFF")
    for method in methods:
        candidate, attempt = _run_ligand_force_field_attempt(
            original_ligand,
            method=method,
        )
        relaxation_attempts.append(attempt)
        if candidate is not None:
            relaxed_ligand = candidate
            relaxation_method = method
            break

    warnings: list[str] = []
    if relaxed_ligand is None:
        relaxed_ligand = Chem.Mol(original_ligand)
        reasons = "; ".join(
            f"{attempt['method']}: {attempt.get('error')}"
            for attempt in relaxation_attempts
        )
        warnings.append(
            "MMFF94/UFF 配体内部优化失败，保留原始连接构象。"
            f"{reasons}"
        )

    accepted_relaxation_attempt = next(
        (
            attempt
            for attempt in relaxation_attempts
            if bool(attempt.get("accepted"))
        ),
        None,
    )
    relaxed_output_path: Path | None = None
    if relaxed_ligand_sdf_path is not None:
        relaxed_output_path = _write_single_molecule_sdf(
            relaxed_ligand,
            relaxed_ligand_sdf_path,
            properties={
                "OptimizationStage": "ligand_internal_relaxation",
                "SelectedForceField": relaxation_method,
                "MaxIterations": LIGAND_RELAX_MAX_ITERATIONS,
                "EnergyTolerance": LIGAND_RELAX_ENERGY_TOLERANCE,
            },
        )

    selected_ligand = Chem.Mol(relaxed_ligand)
    selected_method = relaxation_method
    status = (
        "ligand_relaxed"
        if relaxation_method is not None
        else "unoptimized"
    )
    complex_output_path = _write_complex_pdb(
        receptor,
        selected_ligand,
        output_pdb_path,
    )
    final_positions = _coordinates(selected_ligand)
    heavy_indices = np.asarray(
        [
            atom.GetIdx()
            for atom in selected_ligand.GetAtoms()
            if atom.GetAtomicNum() > 1
        ],
        dtype=np.int64,
    )
    displacement_norms = np.linalg.norm(
        final_positions[heavy_indices] - original_positions[heavy_indices],
        axis=1,
    )
    warning = " ".join(warnings) if warnings else None
    metadata = {
        "status": status,
        "selected_method": selected_method,
        "warning": warning,
        "fallback_to_unoptimized": relaxation_method is None,
        "ligand_relaxation": {
            "selected_method": relaxation_method,
            "max_iterations": int(LIGAND_RELAX_MAX_ITERATIONS),
            "energy_tolerance": float(LIGAND_RELAX_ENERGY_TOLERANCE),
            "force_tolerance": float(NUMERIC_EPSILON),
            "position_constraints_applied": False,
            "attempts": relaxation_attempts,
            "initial_energy": (
                accepted_relaxation_attempt.get("initial_energy")
                if accepted_relaxation_attempt is not None
                else None
            ),
            "final_energy": (
                accepted_relaxation_attempt.get("final_energy")
                if accepted_relaxation_attempt is not None
                else None
            ),
            "energy_change": (
                accepted_relaxation_attempt.get("energy_change")
                if accepted_relaxation_attempt is not None
                else None
            ),
        },
        "ligand_atom_count": int(selected_ligand.GetNumAtoms()),
        "ligand_heavy_atom_rms_displacement_angstrom": float(
            np.sqrt(np.mean(displacement_norms ** 2))
        ),
        "ligand_heavy_atom_max_displacement_angstrom": float(
            np.max(displacement_norms)
        ),
        "receptor_pdb": str(receptor_path),
        "complex_pdb": str(complex_output_path),
        "relaxed_ligand_sdf": (
            str(relaxed_output_path) if relaxed_output_path is not None else None
        ),
    }
    return selected_ligand, metadata


def optimize_ligand_with_fixed_receptor_mmff94(
    ligand: Chem.Mol,
    **kwargs: Any,
) -> tuple[Chem.Mol, dict[str, Any]]:
    """Compatibility wrapper for unconstrained ligand relaxation."""

    return optimize_ligand_with_fixed_receptor(ligand, **kwargs)


def _load_request_json(input_path: str | Path) -> Mapping[str, Any]:
    path = Path(input_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Input JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as file_obj:
        request = json.load(file_obj)
    if not isinstance(request, dict):
        raise ValueError("Input JSON root must be an object")
    return request


def _parse_arguments(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Decode one fragment from SMILES or a 128D embedding, then "
            "generate and align its 3D conformation."
        )
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT_JSON_PATH),
        help=(
            "Single-request JSON path "
            f"(default: {DEFAULT_INPUT_JSON_PATH})"
        ),
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_SDF_PATH),
        help=f"Output SDF path (default: {DEFAULT_OUTPUT_SDF_PATH})",
    )
    parser.add_argument(
        "--vocab",
        default=None,
        help=(
            "Fragment vocabulary (.npz or .json). Required only for "
            "frame_embedding mode; defaults to NPZ then JSON under the "
            "shared data directory."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_arguments(argv)
    try:
        request = _load_request_json(arguments.input)
        molecule, metadata = decode_single_fragment_3d(
            smiles=request.get("smiles"),
            frame_embedding=request.get("frame_embedding"),
            center=request.get("center"),
            reference_frame=request.get("reference_frame"),
            vocab_path=arguments.vocab,
        )
        output_path = write_fragment_sdf(
            molecule,
            metadata,
            arguments.output,
        )
        result = {
            **metadata,
            "output_path": str(output_path),
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(
            f"[Error] {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
