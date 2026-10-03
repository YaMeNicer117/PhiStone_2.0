#!/usr/bin/env python3
"""Reconstruct a stage-one SE3TD result with atom-pair inference.

Generated ligand-node embeddings are resolved against a configured fragment
vocabulary, while fixed ligand nodes retain the SMILES stored in the original
condition ``.pt``.  The atom-pair checkpoint receives no atom-pair labels.
Its query graph uses stage-one covalent predictions except that fixed-fixed
node pairs are replaced by the covalent pairs implied by the original atom-pair
labels in the condition ``.pt``.  During reconstruction, fixed-fixed atom-pair
predictions are likewise replaced by those original labels; all other
connections remain model predictions.  The aligned fragments are then
    connected, sanitized, best-effort net-charge-neutralized, and optionally
    optimized against the normalized receptor.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations, permutations, product
from math import prod
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import numpy as np
import torch
from rdkit import Chem, rdBase
from rdkit.Geometry import Point3D

try:
    from PhiSLinker_atom_pair_config import (
        ATOM_PAIR_SUPERVISION_POLICY,
        ATOM_VOCAB_PATH,
        CONFORMER_TEMPLATE_CACHE_SIZE,
        FRAGMENT_VOCAB_PATH,
        OUTPUT_ROOT,
        ModelConfig,
    )
    from PhiSLinker_atom_pair_dataset import (
        ConnectionFeatureBuilder,
        discover_pyg_files,
        group_connections,
        validate_graph_contract,
    )
    from PhiSLinker_atom_pair_model import PhiSLinkerAtomPairModel
    from PhiSLinker_atom_pair_predict import predict_data_object
    from PhiSLinker_atom_pair_smiles_decoder import (
        CANONICAL_ATOM_ID_PROP,
        DEFAULT_CUSTOM_VOCAB_PATH,
        ENABLE_FRAGMENT_MATCH_HAC_RING_CONSTRAINTS,
        FRAGMENT_MATCH_DIAGNOSTIC_TOP_K,
        FRAGMENT_MATCH_HAC_TOLERANCE,
        FRAGMENT_MATCH_MAX_CONSTRAINT_CANDIDATES,
        FRAGMENT_MATCH_MIN_COSINE_SIMILARITY,
        FRAGMENT_MATCH_RING_TOLERANCE,
        MAX_FRAGMENT_AUTOMORPHISMS,
        MAX_SYMMETRY_COMPONENT_COMBINATIONS,
        RETAINED_NON_TETRAHEDRAL_H_PROP,
        FragmentConformerTemplate,
        FragmentMatchConstraintError,
        FragmentVocabulary,
        align_fragment_conformer_molecule,
        build_fragment_conformer_template,
        is_non_tetrahedral_stereo_atom,
        load_fragment_vocabulary,
        match_fragment_embedding,
        optimize_ligand_with_fixed_receptor,
        remove_nonisotopic_hydrogens,
        translate_disconnected_fragments_tgd,
    )
    from PhiSLinker_atom_pair_utils import (
        AtomVocabularyLookup,
        ConformerGeometryCache,
        FragmentCandidateCache,
        FragmentVocabularyLookup,
        consume_connection_hydrogen,
        neutralize_reconstructed_molecule,
        normalize_smiles_value,
        seed_everything,
        torch_load_compat,
    )
except ImportError:  # pragma: no cover - package-style import fallback
    from .PhiSLinker_atom_pair_config import (
        ATOM_PAIR_SUPERVISION_POLICY,
        ATOM_VOCAB_PATH,
        CONFORMER_TEMPLATE_CACHE_SIZE,
        FRAGMENT_VOCAB_PATH,
        OUTPUT_ROOT,
        ModelConfig,
    )
    from .PhiSLinker_atom_pair_dataset import (
        ConnectionFeatureBuilder,
        discover_pyg_files,
        group_connections,
        validate_graph_contract,
    )
    from .PhiSLinker_atom_pair_model import PhiSLinkerAtomPairModel
    from .PhiSLinker_atom_pair_predict import predict_data_object
    from .PhiSLinker_atom_pair_smiles_decoder import (
        CANONICAL_ATOM_ID_PROP,
        DEFAULT_CUSTOM_VOCAB_PATH,
        ENABLE_FRAGMENT_MATCH_HAC_RING_CONSTRAINTS,
        FRAGMENT_MATCH_DIAGNOSTIC_TOP_K,
        FRAGMENT_MATCH_HAC_TOLERANCE,
        FRAGMENT_MATCH_MAX_CONSTRAINT_CANDIDATES,
        FRAGMENT_MATCH_MIN_COSINE_SIMILARITY,
        FRAGMENT_MATCH_RING_TOLERANCE,
        MAX_FRAGMENT_AUTOMORPHISMS,
        MAX_SYMMETRY_COMPONENT_COMBINATIONS,
        RETAINED_NON_TETRAHEDRAL_H_PROP,
        FragmentConformerTemplate,
        FragmentMatchConstraintError,
        FragmentVocabulary,
        align_fragment_conformer_molecule,
        build_fragment_conformer_template,
        is_non_tetrahedral_stereo_atom,
        load_fragment_vocabulary,
        match_fragment_embedding,
        optimize_ligand_with_fixed_receptor,
        remove_nonisotopic_hydrogens,
        translate_disconnected_fragments_tgd,
    )
    from .PhiSLinker_atom_pair_utils import (
        AtomVocabularyLookup,
        ConformerGeometryCache,
        FragmentCandidateCache,
        FragmentVocabularyLookup,
        consume_connection_hydrogen,
        neutralize_reconstructed_molecule,
        normalize_smiles_value,
        seed_everything,
        torch_load_compat,
    )

DEFAULT_CHECKPOINT_ROOT = OUTPUT_ROOT / "checkpoints"
DEFAULT_OUTPUT_PARENT = OUTPUT_ROOT / "random5_comparison"


class RetryableReconstructionError(RuntimeError):
    """A newly sampled stage-one result may resolve this failure."""

    def __init__(
        self,
        message: str,
        *,
        stage: str,
        diagnostics: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.diagnostics = dict(diagnostics or {})


class FatalConditionError(RuntimeError):
    """The condition itself cannot be reconstructed by another sample."""

    def __init__(
        self,
        message: str,
        *,
        stage: str,
        diagnostics: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.diagnostics = dict(diagnostics or {})


@dataclass(frozen=True)
class _FragmentSymmetry:
    """Whole-fragment automorphisms expressed in canonical atom IDs."""

    canonical_atom_ids: tuple[int, ...]
    automorphism_targets: tuple[tuple[int, ...], ...]
    status: str


class _FragmentTemplateCache:
    """Small process-local cache for reusable decoder templates."""

    def __init__(self) -> None:
        self._templates: dict[str, FragmentConformerTemplate] = {}

    def get(self, canonical_smiles: str) -> FragmentConformerTemplate:
        template = self._templates.get(canonical_smiles)
        if template is None:
            template = build_fragment_conformer_template(canonical_smiles)
            self._templates[canonical_smiles] = template
        return template


@dataclass
class AtomPairInferenceRuntime:
    """Reusable atom-pair model state for stage-two reconstruction."""

    checkpoint_path: Path
    checkpoint_epoch: int
    model_config: ModelConfig
    device: torch.device
    model: PhiSLinkerAtomPairModel
    builder: ConnectionFeatureBuilder
    candidate_cache: FragmentCandidateCache
    template_cache: _FragmentTemplateCache
    vocabulary_fingerprints: Mapping[str, str]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct one saved SE3TD stage-one result with the atom-pair "
            "checkpoint and write fragment, ligand, and complex structures"
        )
    )
    parser.add_argument("--stage1-result", type=Path, required=True)
    parser.add_argument("--condition", type=Path, required=True)
    parser.add_argument("--receptor", type=Path, default=None)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=(
            "checkpoint to evaluate; when omitted, use the most recently "
            "modified checkpoints/*/best_model.pt"
        ),
    )
    parser.add_argument("--fragment-vocab", type=Path, default=FRAGMENT_VOCAB_PATH)
    parser.add_argument("--atom-vocab", type=Path, default=ATOM_VOCAB_PATH)
    parser.add_argument(
        "--custom-fragment-vocab",
        type=Path,
        default=DEFAULT_CUSTOM_VOCAB_PATH,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--min-cosine-similarity",
        type=float,
        default=FRAGMENT_MATCH_MIN_COSINE_SIMILARITY,
    )
    parser.add_argument(
        "--diagnostic-top-k",
        type=int,
        default=FRAGMENT_MATCH_DIAGNOSTIC_TOP_K,
    )
    args = parser.parse_args()
    if args.diagnostic_top_k <= 0:
        parser.error("--diagnostic-top-k must be positive")
    if (
        args.min_cosine_similarity is not None
        and not -1.0 <= args.min_cosine_similarity <= 1.0
    ):
        parser.error("--min-cosine-similarity must be in [-1,1]")
    return args


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _resolve_checkpoint_path(requested: Path | None) -> Path:
    if requested is not None:
        return requested.expanduser().resolve()

    checkpoint_root = DEFAULT_CHECKPOINT_ROOT.expanduser().resolve()
    if not checkpoint_root.is_dir():
        raise FileNotFoundError(
            "automatic checkpoint directory not found: "
            f"{checkpoint_root}; pass --checkpoint explicitly"
        )

    candidates: list[tuple[int, str, Path]] = []
    for candidate in checkpoint_root.glob("*/best_model.pt"):
        try:
            modification_time = candidate.stat().st_mtime_ns
        except FileNotFoundError:
            # A concurrently running trainer may atomically replace this file.
            continue
        resolved = candidate.resolve()
        candidates.append(
            (modification_time, str(resolved), resolved)
        )
    if not candidates:
        raise FileNotFoundError(
            "no best_model.pt was found below "
            f"{checkpoint_root}; pass --checkpoint explicitly"
        )
    return max(candidates, key=lambda item: (item[0], item[1]))[2]


def _load_checkpoint(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"checkpoint not found: {resolved}")
    payload = torch_load_compat(resolved, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError("checkpoint payload must be a dictionary")
    if (
        str(payload.get("atom_pair_supervision_policy", ""))
        != ATOM_PAIR_SUPERVISION_POLICY
    ):
        raise ValueError("checkpoint uses an incompatible supervision policy")
    if str(payload.get("training_stage", "")) != "global":
        raise ValueError(
            "stage-two reconstruction requires a global-stage checkpoint"
        )
    if "model_state_dict" not in payload:
        raise KeyError("checkpoint lacks model_state_dict")
    return payload


def _model_config_from_checkpoint(payload: Mapping[str, Any]) -> ModelConfig:
    try:
        configuration = payload["configuration"]
        if not isinstance(configuration, Mapping):
            raise TypeError("configuration must be a mapping")
        model_values = configuration["model"]
        if not isinstance(model_values, Mapping):
            raise TypeError("configuration.model must be a mapping")
        config = ModelConfig(**dict(model_values))
    except Exception as exc:
        raise ValueError("checkpoint contains an invalid model configuration") from exc
    config.validate()
    return config


def _validate_trusted_fragment_vocabulary_prefix(
    trusted_profile: Mapping[str, Any],
    fragment_vocab: FragmentVocabularyLookup,
) -> tuple[int, int]:
    raw_num_smiles = trusted_profile.get("num_smiles")
    if isinstance(raw_num_smiles, bool) or not isinstance(raw_num_smiles, int):
        raise ValueError(
            "trusted fragment vocabulary profile has an invalid num_smiles"
        )
    trusted_num_smiles = int(raw_num_smiles)
    current_num_smiles = len(fragment_vocab.smiles)
    if current_num_smiles < trusted_num_smiles:
        raise ValueError(
            "current fragment vocabulary is shorter than the trusted "
            "checkpoint vocabulary: "
            f"current={current_num_smiles}, trusted={trusted_num_smiles}"
        )
    if trusted_profile.get("embedding_dim") != fragment_vocab.expected_dim:
        raise ValueError(
            "current fragment vocabulary embedding dimension differs from "
            "the trusted checkpoint profile"
        )

    prefix_profile = fragment_vocab.profile_for_prefix(trusted_num_smiles)
    for key in ("array_sha256", "table_sha256"):
        if key not in trusted_profile:
            raise ValueError(
                f"trusted fragment vocabulary profile lacks {key!r}"
            )
        if trusted_profile[key] != prefix_profile[key]:
            raise ValueError(
                "current fragment vocabulary is not an append-only extension "
                "of the trusted checkpoint vocabulary: "
                f"first {trusted_num_smiles} rows differ in {key}"
            )
    return trusted_num_smiles, current_num_smiles


def _atom_vocabulary_compatibility_anchors_path(
    atom_vocab: AtomVocabularyLookup,
) -> Path:
    return atom_vocab.path.with_name(
        f"{atom_vocab.path.stem}_compatibility_anchors.json"
    )


def _validate_anchored_atom_vocabulary_prefix(
    saved_archive_sha256: str,
    atom_vocab: AtomVocabularyLookup,
) -> tuple[int, int, Path]:
    anchor_path = _atom_vocabulary_compatibility_anchors_path(
        atom_vocab
    )
    if not anchor_path.is_file():
        raise ValueError(
            "atom vocabulary archive differs from checkpoint and the "
            f"compatibility anchor file is missing: {anchor_path}"
        )
    try:
        with anchor_path.open("r", encoding="utf-8") as file_obj:
            payload = json.load(file_obj)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"failed to read atom vocabulary compatibility anchors: "
            f"{anchor_path}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise ValueError("atom vocabulary compatibility anchors are invalid")
    if payload.get("table_file_name") != atom_vocab.path.name:
        raise ValueError(
            "atom vocabulary compatibility anchors refer to another table"
        )
    anchors = payload.get("anchors")
    if not isinstance(anchors, Mapping):
        raise ValueError("atom vocabulary compatibility anchors are invalid")
    anchor = anchors.get(saved_archive_sha256)
    if not isinstance(anchor, Mapping):
        raise ValueError(
            "atom vocabulary archive differs from checkpoint and no "
            "compatibility anchor exists for the checkpoint SHA256: "
            f"{saved_archive_sha256}"
        )
    if anchor.get("archive_sha256") != saved_archive_sha256:
        raise ValueError(
            "atom vocabulary compatibility anchor archive SHA256 is invalid"
        )
    encoder_sha256 = anchor.get("atom_encoder_checkpoint_sha256")
    if (
        not isinstance(encoder_sha256, str)
        or len(encoder_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in encoder_sha256.lower()
        )
    ):
        raise ValueError(
            "atom vocabulary compatibility anchor has an invalid encoder "
            "checkpoint SHA256"
        )
    raw_num_smiles = anchor.get("num_smiles")
    if (
        isinstance(raw_num_smiles, bool)
        or not isinstance(raw_num_smiles, int)
        or raw_num_smiles < 1
    ):
        raise ValueError(
            "atom vocabulary compatibility anchor has an invalid "
            "num_smiles"
        )
    anchored_num_smiles = int(raw_num_smiles)
    current_num_smiles = len(atom_vocab.smiles)
    if current_num_smiles < anchored_num_smiles:
        raise ValueError(
            "current atom vocabulary is shorter than the checkpoint "
            "compatibility anchor: "
            f"current={current_num_smiles}, anchored={anchored_num_smiles}"
        )
    prefix_profile = atom_vocab.profile_for_prefix(
        anchored_num_smiles
    )
    for key in (
        "embedding_dim",
        "num_smiles",
        "num_atoms",
        "array_sha256",
        "table_sha256",
    ):
        if anchor.get(key) != prefix_profile[key]:
            raise ValueError(
                "current atom vocabulary is not an append-only extension "
                "of the checkpoint vocabulary: "
                f"first {anchored_num_smiles} fragment rows differ in {key}"
            )
    return anchored_num_smiles, current_num_smiles, anchor_path


def _verify_vocabulary_fingerprints(
    checkpoint: Mapping[str, Any],
    fragment_vocab: FragmentVocabularyLookup,
    atom_vocab: AtomVocabularyLookup,
    *,
    trusted_fragment_vocab_profile: Mapping[str, Any] | None = None,
    trusted_fragment_vocab_path: str | Path | None = None,
) -> dict[str, str]:
    saved = checkpoint.get("vocabulary_fingerprints")
    if not isinstance(saved, Mapping):
        raise ValueError("checkpoint lacks vocabulary fingerprints")
    current = {
        "fragment_vocab_sha256": fragment_vocab.sha256,
        "atom_vocab_sha256": atom_vocab.sha256,
    }
    mismatches = [
        key for key, value in current.items() if str(saved.get(key)) != value
    ]
    if not mismatches:
        return current

    fragment_fingerprint_key = "fragment_vocab_sha256"
    atom_fingerprint_key = "atom_vocab_sha256"
    if fragment_fingerprint_key in mismatches:
        if (
            trusted_fragment_vocab_profile is None
            or trusted_fragment_vocab_path is None
        ):
            raise ValueError(
                "current vocabularies differ from checkpoint: "
                + ", ".join(mismatches)
            )

        trusted_path = (
            Path(trusted_fragment_vocab_path).expanduser().resolve()
        )
        if fragment_vocab.path != trusted_path:
            raise ValueError(
                "atom-pair fragment vocabulary is not the same file that "
                "passed the trusted main-checkpoint validation: "
                f"atom_pair={fragment_vocab.path}, trusted={trusted_path}"
            )
        _validate_trusted_fragment_vocabulary_prefix(
            trusted_fragment_vocab_profile,
            fragment_vocab,
        )

    if atom_fingerprint_key in mismatches:
        saved_atom_sha256 = saved.get(atom_fingerprint_key)
        if not isinstance(saved_atom_sha256, str):
            raise ValueError(
                "checkpoint has an invalid atom vocabulary SHA256"
            )
        anchored_count, current_count, anchor_path = (
            _validate_anchored_atom_vocabulary_prefix(
                saved_atom_sha256,
                atom_vocab,
            )
        )
        warnings.warn(
            "atom vocabulary archive differs from the atom-pair checkpoint, "
            "but its anchored semantic prefix is unchanged; allowing "
            "append-only compatibility: "
            f"anchored_prefix={anchored_count}, current={current_count}, "
            f"anchors={anchor_path}",
            RuntimeWarning,
            stacklevel=2,
        )
    return current


def resolve_custom_fragment_vocabulary_path(
    requested: str | Path,
) -> Path:
    """Resolve either an explicit vocabulary file or its containing folder."""

    path = Path(requested).expanduser().resolve()
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(
            f"custom fragment vocabulary was not found: {path}"
        )
    preferred = path / "fragment_embeddings_128d.npz"
    if preferred.is_file():
        return preferred.resolve()
    npz_candidates = sorted(
        candidate.resolve() for candidate in path.glob("*.npz")
        if candidate.is_file()
    )
    if len(npz_candidates) == 1:
        return npz_candidates[0]
    json_candidates = sorted(
        candidate.resolve() for candidate in path.glob("*.json")
        if candidate.is_file()
    )
    if not npz_candidates and len(json_candidates) == 1:
        return json_candidates[0]
    raise ValueError(
        "custom fragment vocabulary directory must contain "
        "fragment_embeddings_128d.npz or one unambiguous .npz/.json file: "
        f"{path}"
    )


def load_reconstruction_vocabulary(
    requested: str | Path,
) -> FragmentVocabulary:
    return load_fragment_vocabulary(
        resolve_custom_fragment_vocabulary_path(requested)
    )


def load_atom_pair_inference_runtime(
    *,
    checkpoint_path: str | Path | None,
    fragment_vocab_path: str | Path,
    atom_vocab_path: str | Path,
    device: str | torch.device = "auto",
    trusted_fragment_vocab_profile: Mapping[str, Any] | None = None,
    trusted_fragment_vocab_path: str | Path | None = None,
) -> AtomPairInferenceRuntime:
    """Load and compatibility-check atom-pair inference dependencies once."""

    resolved_device = (
        _resolve_device(device)
        if isinstance(device, str)
        else torch.device(device)
    )
    resolved_checkpoint = _resolve_checkpoint_path(
        None if checkpoint_path is None else Path(checkpoint_path)
    )
    checkpoint = _load_checkpoint(resolved_checkpoint)
    model_config = _model_config_from_checkpoint(checkpoint)
    fragment_vocab = FragmentVocabularyLookup(
        fragment_vocab_path,
        expected_dim=model_config.fragment_embedding_dim,
    )
    atom_vocab = AtomVocabularyLookup(
        atom_vocab_path,
        expected_dim=model_config.atom_embedding_dim,
    )
    fingerprints = _verify_vocabulary_fingerprints(
        checkpoint,
        fragment_vocab,
        atom_vocab,
        trusted_fragment_vocab_profile=trusted_fragment_vocab_profile,
        trusted_fragment_vocab_path=trusted_fragment_vocab_path,
    )
    candidate_cache = FragmentCandidateCache(atom_vocab)
    builder = ConnectionFeatureBuilder(
        fragment_vocab,
        candidate_cache,
        ConformerGeometryCache(CONFORMER_TEMPLATE_CACHE_SIZE),
        model_config,
    )
    model = PhiSLinkerAtomPairModel(model_config).to(resolved_device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return AtomPairInferenceRuntime(
        checkpoint_path=resolved_checkpoint,
        checkpoint_epoch=int(checkpoint.get("epoch", -1)),
        model_config=model_config,
        device=resolved_device,
        model=model,
        builder=builder,
        candidate_cache=candidate_cache,
        template_cache=_FragmentTemplateCache(),
        vocabulary_fingerprints=fingerprints,
    )


def _create_output_directory(
    requested: Path | None,
    *,
    seed: int,
) -> Path:
    if requested is not None:
        target = requested.expanduser().resolve()
        if target.exists():
            if not target.is_dir():
                raise NotADirectoryError(f"output path is not a directory: {target}")
            if any(target.iterdir()):
                raise FileExistsError(
                    f"output directory must be empty to avoid overwrites: {target}"
                )
        else:
            target.mkdir(parents=True, exist_ok=False)
        return target

    parent = DEFAULT_OUTPUT_PARENT.expanduser().resolve()
    parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{timestamp}_seed{seed}"
    target = parent / base_name
    counter = 1
    while target.exists():
        target = parent / f"{base_name}_{counter:02d}"
        counter += 1
    target.mkdir(parents=False, exist_ok=False)
    return target


def _safe_name(value: str, fallback: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    if not cleaned:
        cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", fallback).strip("._")
    return (cleaned or "sample")[:96]


def _prediction_view_without_atom_labels(
    data: Any,
    *,
    source_path: str,
) -> SimpleNamespace:
    """Copy only inference fields and deliberately omit connection_atom_ids."""

    validated = validate_graph_contract(data, require_targets=False)
    if validated["connection_edge_index"].shape[1] == 0:
        raise ValueError("graph contains no inter-fragment connection queries")
    if isinstance(data, Mapping):
        sample_id = data.get("sample_id", Path(source_path).stem)
    else:
        sample_id = getattr(data, "sample_id", Path(source_path).stem)
    return SimpleNamespace(
        node_type=validated["node_type"],
        pos=validated["pos"],
        ref_coords=validated["ref_coords"],
        fragment_smiles=validated["smiles"],
        edge_index=validated["edge_index"],
        edge_attr=validated["edge_attr"],
        connection_edge_index=validated["connection_edge_index"],
        sample_id=str(sample_id),
    )


def _write_single_sdf(
    molecule: Chem.Mol,
    output_path: Path,
    *,
    overwrite: bool = False,
) -> Path:
    target = output_path.expanduser().resolve()
    if target.suffix.lower() != ".sdf":
        raise ValueError(f"SDF output must end with .sdf: {target}")
    if target.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing SDF: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    output_molecule, _ = remove_nonisotopic_hydrogens(
        molecule,
        context=f"reconstruction SDF output {target.name}",
    )
    temporary = target.with_name(
        f".{target.stem}.{uuid4().hex}.tmp.sdf"
    )
    writer: Chem.SDWriter | None = None
    try:
        writer = Chem.SDWriter(str(temporary))
        if writer is None:
            raise OSError(f"RDKit could not open temporary SDF: {temporary}")
        writer.write(output_molecule)
        writer.close()
        writer = None
        if not temporary.is_file() or temporary.stat().st_size <= 0:
            raise OSError(f"RDKit produced an empty SDF: {temporary}")
        temporary.replace(target)
    except Exception:
        if writer is not None:
            writer.close()
        temporary.unlink(missing_ok=True)
        raise
    return target


def _enumerate_fragment_symmetry(
    fragment: Chem.Mol,
    canonical_atom_ids: Sequence[int],
) -> _FragmentSymmetry:
    """Enumerate complete graph automorphisms without breaking stereochemistry."""

    canonical_ids = tuple(int(value) for value in canonical_atom_ids)
    remove_hydrogen_parameters = Chem.RemoveHsParameters()
    remove_hydrogen_parameters.removeNontetrahedralNeighbors = True
    try:
        target = Chem.RemoveHs(
            Chem.Mol(fragment),
            remove_hydrogen_parameters,
            sanitize=True,
        )
    except Exception as exc:
        raise RuntimeError(
            "failed to construct heavy-atom topology for symmetry analysis: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    target.RemoveAllConformers()
    if any(atom.GetAtomicNum() <= 1 for atom in target.GetAtoms()):
        raise RuntimeError(
            "symmetry-analysis molecule still contains explicit hydrogen"
        )
    if len(canonical_ids) != target.GetNumAtoms():
        raise RuntimeError(
            "canonical atom ID count does not match fragment heavy-atom count"
        )
    if len(set(canonical_ids)) != len(canonical_ids):
        raise RuntimeError("fragment canonical atom IDs are not unique")
    identity = canonical_ids
    if len(canonical_ids) <= 1:
        return _FragmentSymmetry(
            canonical_atom_ids=canonical_ids,
            automorphism_targets=(identity,),
            status="identity_only",
        )

    for atom in target.GetAtoms():
        # Input atom-map labels identify atoms but are not chemical symmetry
        # constraints.  Charge, isotope and stereochemistry remain intact.
        atom.SetAtomMapNum(0)
    query = Chem.Mol(target)
    raw_matches = target.GetSubstructMatches(
        query,
        uniquify=False,
        useChirality=True,
        maxMatches=int(MAX_FRAGMENT_AUTOMORPHISMS + 1),
    )
    if len(raw_matches) > MAX_FRAGMENT_AUTOMORPHISMS:
        return _FragmentSymmetry(
            canonical_atom_ids=canonical_ids,
            automorphism_targets=(identity,),
            status="skipped_automorphism_limit",
        )

    automorphisms: set[tuple[int, ...]] = {identity}
    expected_local_indices = set(range(target.GetNumAtoms()))
    for match in raw_matches:
        local_indices = tuple(int(value) for value in match)
        if set(local_indices) != expected_local_indices:
            raise RuntimeError(
                "RDKit returned an incomplete fragment automorphism"
            )
        automorphisms.add(
            tuple(canonical_ids[index] for index in local_indices)
        )

    ordered = (
        identity,
        *sorted(value for value in automorphisms if value != identity),
    )
    return _FragmentSymmetry(
        canonical_atom_ids=canonical_ids,
        automorphism_targets=ordered,
        status=("enumerated" if len(ordered) > 1 else "identity_only"),
    )


def _decode_disconnected_ligand(
    validated: Mapping[str, Any],
    *,
    candidate_cache: FragmentCandidateCache,
    template_cache: _FragmentTemplateCache,
) -> tuple[
    Chem.Mol,
    dict[int, dict[int, int]],
    dict[int, _FragmentSymmetry],
    list[dict[str, Any]],
    list[Chem.Mol],
]:
    """Decode all ligand nodes into one disconnected, positioned molecule."""

    node_type: torch.Tensor = validated["node_type"]
    node_pos: torch.Tensor = validated["pos"]
    ref_coords: torch.Tensor = validated["ref_coords"]
    smiles: Sequence[str] = validated["smiles"]

    combined: Chem.Mol | None = None
    combined_positions: list[np.ndarray] = []
    node_atom_map: dict[int, dict[int, int]] = {}
    node_symmetry: dict[int, _FragmentSymmetry] = {}
    fragment_records: list[dict[str, Any]] = []
    decoded_fragments: list[Chem.Mol] = []
    atom_offset = 0

    for node_index in range(int(node_type.shape[0])):
        if not bool(node_type[node_index, 0]):
            continue

        input_smiles = str(smiles[node_index])
        chemistry = candidate_cache.get(input_smiles)
        template = template_cache.get(chemistry.canonical_smiles)
        center = node_pos[node_index].numpy().astype(np.float64, copy=False)
        reference_frame = (
            ref_coords[node_index] - node_pos[node_index][None, :]
        ).numpy().astype(np.float64, copy=False)
        fragment, alignment = align_fragment_conformer_molecule(
            template,
            center=center,
            reference_frame=reference_frame,
        )
        if fragment.GetNumConformers() == 0:
            raise RuntimeError(
                f"decoded fragment has no conformer for node {node_index}"
            )
        aligned_conformer = fragment.GetConformer(0)
        local_positions = np.asarray(
            aligned_conformer.GetPositions(),
            dtype=np.float64,
        )
        heavy_indices = [
            atom.GetIdx()
            for atom in fragment.GetAtoms()
            if atom.GetAtomicNum() > 1
        ]
        atom_ids = np.asarray(
            [
                fragment.GetAtomWithIdx(atom_index).GetIntProp(
                    CANONICAL_ATOM_ID_PROP
                )
                for atom_index in heavy_indices
            ],
            dtype=np.int64,
        )
        if len(heavy_indices) != len(atom_ids):
            raise RuntimeError(
                f"decoded heavy-atom ID count mismatch for node {node_index}"
            )
        if len(set(map(int, atom_ids.tolist()))) != len(atom_ids):
            raise RuntimeError(
                f"duplicate canonical atom IDs for ligand node {node_index}"
            )
        for atom in fragment.GetAtoms():
            atom.SetIntProp("_PhiStoneSourceNodeIndex", int(node_index))
            if atom.GetAtomicNum() == 1:
                raise RuntimeError(
                    "comparison reconstruction requires a hydrogen-free "
                    f"fragment for node {node_index}"
                )
        symmetry = _enumerate_fragment_symmetry(
            fragment,
            atom_ids.tolist(),
        )

        canonical_to_combined: dict[int, int] = {}
        for row, local_atom_index in enumerate(heavy_indices):
            canonical_atom_id = int(atom_ids[row])
            canonical_to_combined[canonical_atom_id] = (
                atom_offset + local_atom_index
            )
            atom = fragment.GetAtomWithIdx(local_atom_index)
            atom.SetIntProp(
                CANONICAL_ATOM_ID_PROP, canonical_atom_id
            )

        decoded_fragments.append(Chem.Mol(fragment))
        fragment.RemoveAllConformers()

        if combined is None:
            combined = Chem.Mol(fragment)
        else:
            combined = Chem.CombineMols(combined, fragment)
        combined_positions.extend(local_positions)
        node_atom_map[node_index] = canonical_to_combined
        node_symmetry[node_index] = symmetry
        fragment_records.append(
            {
                "node_index": node_index,
                "input_smiles": input_smiles,
                "canonical_smiles": chemistry.canonical_smiles,
                "center": center.tolist(),
                "reference_frame_relative": reference_frame.tolist(),
                "atom_count": fragment.GetNumAtoms(),
                "heavy_atom_count": len(heavy_indices),
                "retained_nontetrahedral_hydrogen_count": 0,
                "symmetry_status": symmetry.status,
                "symmetry_automorphism_count": len(
                    symmetry.automorphism_targets
                ),
                "alignment_status": alignment.get("alignment_status"),
                "reference_alignment_rmsd": alignment.get(
                    "reference_alignment_rmsd"
                ),
            }
        )
        atom_offset += fragment.GetNumAtoms()

    if combined is None or not node_atom_map:
        raise ValueError("PyGData does not contain any ligand fragment nodes")
    if combined.GetNumAtoms() != len(combined_positions):
        raise RuntimeError("combined ligand atom/coordinate counts differ")

    combined.RemoveAllConformers()
    conformer = Chem.Conformer(combined.GetNumAtoms())
    conformer.Set3D(True)
    for atom_index, position in enumerate(combined_positions):
        conformer.SetAtomPosition(
            atom_index,
            Point3D(*map(float, position)),
        )
    combined.AddConformer(conformer, assignId=True)
    return (
        combined,
        node_atom_map,
        node_symmetry,
        fragment_records,
        decoded_fragments,
    )


def _apply_preconnection_fragment_tgd(
    atom_pair_data: Any,
    base_molecule: Chem.Mol,
    fragment_records: Sequence[Mapping[str, Any]],
    decoded_fragments: Sequence[Chem.Mol],
    fixed_node_indices: set[int],
    receptor_pdb_path: str | Path,
) -> tuple[Chem.Mol, list[dict[str, Any]], list[Chem.Mol], dict[str, Any]]:
    """Run pocket-aware fragment TGD and synchronize downstream coordinates."""

    if len(fragment_records) != len(decoded_fragments):
        raise ValueError("fragment records and decoded molecules differ in length")
    fragment_atom_indices: dict[int, list[int]] = {}
    for atom in base_molecule.GetAtoms():
        if not atom.HasProp("_PhiStoneSourceNodeIndex"):
            raise RuntimeError(
                "decoded fragment atom is missing its source node index"
            )
        node_index = int(atom.GetIntProp("_PhiStoneSourceNodeIndex"))
        fragment_atom_indices.setdefault(node_index, []).append(int(atom.GetIdx()))

    record_node_indices = [int(record["node_index"]) for record in fragment_records]
    if len(set(record_node_indices)) != len(record_node_indices):
        raise ValueError("fragment records contain duplicate node indices")
    if set(record_node_indices) != set(fragment_atom_indices):
        raise ValueError(
            "fragment records do not cover every decoded source node"
        )
    anchor_node_indices = sorted(
        set(fragment_atom_indices).intersection(fixed_node_indices)
    )
    (
        translated_base_molecule,
        translations,
        tgd_metadata,
    ) = translate_disconnected_fragments_tgd(
        base_molecule,
        fragment_atom_indices,
        receptor_pdb_path=receptor_pdb_path,
        anchor_fragment_indices=anchor_node_indices,
    )
    if set(translations) != set(fragment_atom_indices):
        raise RuntimeError("TGD did not return one translation per fragment")

    try:
        original_pos = getattr(atom_pair_data, "pos")
        original_ref_coords = getattr(atom_pair_data, "ref_coords")
    except AttributeError as exc:
        raise AttributeError(
            "atom-pair data is missing pos/ref_coords for TGD synchronization"
        ) from exc
    if not torch.is_tensor(original_pos) or not torch.is_tensor(
        original_ref_coords
    ):
        raise TypeError("atom-pair pos/ref_coords must be tensors")
    if original_pos.ndim != 2 or original_pos.shape[1] != 3:
        raise ValueError("atom-pair pos must have shape [N,3]")
    if original_ref_coords.shape != (original_pos.shape[0], 3, 3):
        raise ValueError("atom-pair ref_coords must have shape [N,3,3]")
    updated_pos = original_pos.clone()
    updated_ref_coords = original_ref_coords.clone()
    original_relative_frames = (
        original_ref_coords - original_pos[:, None, :]
    )
    for node_index, translation in translations.items():
        if node_index < 0 or node_index >= int(updated_pos.shape[0]):
            raise IndexError(f"TGD node index is out of range: {node_index}")
        translation_array = np.asarray(translation, dtype=np.float64)
        if translation_array.shape != (3,) or not np.all(
            np.isfinite(translation_array)
        ):
            raise ValueError(
                f"TGD returned an invalid translation for node {node_index}"
            )
        translation_tensor = torch.as_tensor(
            translation_array,
            dtype=updated_pos.dtype,
            device=updated_pos.device,
        )
        updated_pos[node_index] += translation_tensor
        updated_ref_coords[node_index] += translation_tensor[None, :]
    updated_relative_frames = updated_ref_coords - updated_pos[:, None, :]
    for node_index in translations:
        if not torch.allclose(
            updated_relative_frames[node_index],
            original_relative_frames[node_index],
            rtol=0.0,
            atol=1.0e-5,
        ):
            raise RuntimeError(
                "TGD changed a fragment reference frame during translation: "
                f"node={node_index}"
            )
    setattr(atom_pair_data, "pos", updated_pos)
    setattr(atom_pair_data, "ref_coords", updated_ref_coords)

    translated_fragments: list[Chem.Mol] = []
    translated_records: list[dict[str, Any]] = []
    anchor_set = {
        int(value) for value in tgd_metadata.get("anchor_fragment_indices", [])
    }
    for raw_record, raw_fragment in zip(fragment_records, decoded_fragments):
        record = dict(raw_record)
        node_index = int(record["node_index"])
        translation = np.asarray(translations[node_index], dtype=np.float64)
        fragment = Chem.Mol(raw_fragment)
        if fragment.GetNumConformers() != 1:
            raise ValueError(
                f"decoded fragment {node_index} must contain one conformer"
            )
        conformer = fragment.GetConformer(0)
        for atom_index in range(fragment.GetNumAtoms()):
            position = conformer.GetAtomPosition(atom_index)
            translated_position = np.asarray(
                [float(position.x), float(position.y), float(position.z)],
                dtype=np.float64,
            ) + translation
            conformer.SetAtomPosition(
                atom_index,
                Point3D(*map(float, translated_position)),
            )
        center_before = np.asarray(record["center"], dtype=np.float64)
        if center_before.shape != (3,):
            raise ValueError(
                f"fragment {node_index} center must contain three coordinates"
            )
        record["center_before_tgd"] = center_before.tolist()
        record["center"] = (center_before + translation).tolist()
        record["tgd_translation_angstrom"] = translation.tolist()
        record["tgd_is_anchor"] = node_index in anchor_set
        translated_fragments.append(fragment)
        translated_records.append(record)

    applied_node_indices = sorted(
        node_index
        for node_index, translation in translations.items()
        if float(np.linalg.norm(translation)) > 1.0e-12
    )
    synchronized_metadata = dict(tgd_metadata)
    synchronized_metadata["coordinate_synchronization"] = {
        "status": "complete",
        "model_inference_order": "tgd_then_atom_pair_prediction",
        "updated_fields": [
            "decoded_disconnected_molecule",
            "decoded_fragment_molecules",
            "atom_pair_data.pos",
            "atom_pair_data.ref_coords",
            "fragment_record.center",
        ],
        "translated_node_indices": applied_node_indices,
        "relative_reference_frames_preserved": True,
    }
    return (
        translated_base_molecule,
        translated_records,
        translated_fragments,
        synchronized_metadata,
    )


def _warn_preconnection_tgd_fallback(
    metadata: Mapping[str, Any],
) -> None:
    """Print the complete reason why TGD fell back to original coordinates."""

    status = str(metadata.get("status") or "unknown")
    stop_reason = str(metadata.get("stop_reason") or status)
    error_value = metadata.get("error")
    error_text = (
        str(error_value).strip()
        if error_value is not None and str(error_value).strip()
        else "未提供异常信息"
    )
    supported_value = metadata.get("supported")
    if supported_value is True:
        supported_text = "是"
    elif supported_value is False:
        supported_text = "否"
    else:
        supported_text = "未知"
    print(
        "      警告: 片段 TGD 未被采用，保留滑移前坐标并继续原子匹配。\n"
        f"        TGD 状态: {status}\n"
        f"        停止原因: {stop_reason}\n"
        f"        UFF 参数支持: {supported_text}\n"
        f"        具体原因: {error_text}\n"
        "        TGD 参数: "
        f"max_steps={metadata.get('max_steps')}, "
        f"learning_rate={metadata.get('learning_rate')}, "
        "max_step_distance_angstrom="
        f"{metadata.get('max_step_distance_angstrom')}, "
        f"force_tolerance={metadata.get('force_tolerance')}"
    )


def _prepare_comparison_records(
    predictions: Sequence[Mapping[str, Any]],
    truth_groups: Sequence[Any],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    truth_by_nodes = {
        (int(group.node_a), int(group.node_b)): group for group in truth_groups
    }
    if len(truth_by_nodes) != len(truth_groups):
        raise RuntimeError("ground-truth query node pairs are not unique")

    query_records: list[dict[str, Any]] = []
    predicted_connections: list[dict[str, Any]] = []
    seen_nodes: set[tuple[int, int]] = set()
    for prediction in predictions:
        node_pair = (
            int(prediction["node_a"]),
            int(prediction["node_b"]),
        )
        if node_pair in seen_nodes:
            raise RuntimeError(f"duplicate predicted query: {node_pair}")
        seen_nodes.add(node_pair)
        try:
            truth_group = truth_by_nodes[node_pair]
        except KeyError as exc:
            raise RuntimeError(
                f"prediction contains an unknown query: {node_pair}"
            ) from exc

        raw_selected = prediction.get("selected_pairs")
        if not isinstance(raw_selected, Sequence):
            raise TypeError("prediction selected_pairs must be a sequence")
        selected_pairs: list[dict[str, Any]] = []
        predicted_pair_set: set[tuple[int, int]] = set()
        for selected in raw_selected:
            if not isinstance(selected, Mapping):
                raise TypeError("selected pair records must be mappings")
            atom_pair = (
                int(selected["atom_a_id"]),
                int(selected["atom_b_id"]),
            )
            if atom_pair in predicted_pair_set:
                raise RuntimeError(
                    f"duplicate predicted atom pair for query {node_pair}: {atom_pair}"
                )
            predicted_pair_set.add(atom_pair)
            selected_record = {
                "atom_a_id": atom_pair[0],
                "atom_b_id": atom_pair[1],
                "pair_probability": float(selected["pair_probability"]),
            }
            selected_pairs.append(selected_record)
            predicted_connections.append(
                {
                    "node_a": node_pair[0],
                    "node_b": node_pair[1],
                    **selected_record,
                }
            )

        predicted_count = int(prediction["predicted_count"])
        if predicted_count != len(selected_pairs):
            raise RuntimeError(
                f"predicted_count disagrees with selected_pairs for {node_pair}"
            )
        truth_pairs = [
            {
                "atom_a_id": int(atom_a),
                "atom_b_id": int(atom_b),
            }
            for atom_a, atom_b in truth_group.true_atom_pairs
        ]
        truth_pair_set = {
            (item["atom_a_id"], item["atom_b_id"]) for item in truth_pairs
        }
        query_record: dict[str, Any] = {
            "node_a": node_pair[0],
            "node_b": node_pair[1],
            "smiles_a": str(prediction["smiles_a"]),
            "smiles_b": str(prediction["smiles_b"]),
            "candidate_count": int(prediction["candidate_count"]),
            "geometry_valid": bool(prediction["geometry_valid"]),
            "prediction_status": str(prediction["status"]),
            "predicted_count": predicted_count,
            "ground_truth_count": len(truth_pairs),
            "count_correct": predicted_count == len(truth_pairs),
            "predicted_pairs": selected_pairs,
            "ground_truth_pairs": truth_pairs,
            "exact_atom_pair_set": predicted_pair_set == truth_pair_set,
        }
        if prediction.get("geometry_error"):
            query_record["geometry_error"] = str(
                prediction["geometry_error"]
            )
        query_records.append(query_record)

    expected_nodes = set(truth_by_nodes)
    if seen_nodes != expected_nodes:
        missing = sorted(expected_nodes - seen_nodes)
        raise RuntimeError(f"predictions are missing ground-truth queries: {missing}")

    truth_connections = [
        {
            "node_a": int(group.node_a),
            "node_b": int(group.node_b),
            "atom_a_id": int(atom_a),
            "atom_b_id": int(atom_b),
        }
        for group in truth_groups
        for atom_a, atom_b in group.true_atom_pairs
    ]
    return query_records, predicted_connections, truth_connections


def _relevant_automorphism_choices(
    symmetry: _FragmentSymmetry,
    referenced_atom_ids: Sequence[int],
) -> tuple[dict[int, int], ...]:
    referenced = tuple(sorted(set(map(int, referenced_atom_ids))))
    positions = {
        atom_id: index
        for index, atom_id in enumerate(symmetry.canonical_atom_ids)
    }
    missing = [atom_id for atom_id in referenced if atom_id not in positions]
    if missing:
        raise KeyError(
            "connection atom IDs are absent from fragment symmetry data: "
            f"{missing}"
        )

    choices: list[dict[int, int]] = []
    seen_actions: set[tuple[int, ...]] = set()
    for targets in symmetry.automorphism_targets:
        action = tuple(targets[positions[atom_id]] for atom_id in referenced)
        if action in seen_actions:
            continue
        seen_actions.add(action)
        choices.append(dict(zip(referenced, action)))
    if not choices:
        raise RuntimeError("fragment symmetry produced no usable action")
    identity = {atom_id: atom_id for atom_id in referenced}
    if choices[0] != identity:
        choices = [identity, *[choice for choice in choices if choice != identity]]
    return tuple(choices)


def _connection_components(
    connections: Sequence[Mapping[str, Any]],
) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    adjacency: dict[int, set[int]] = {}
    for connection in connections:
        node_a = int(connection["node_a"])
        node_b = int(connection["node_b"])
        if node_a == node_b:
            raise ValueError("inter-fragment connections cannot be self-loops")
        adjacency.setdefault(node_a, set()).add(node_b)
        adjacency.setdefault(node_b, set()).add(node_a)

    components: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    remaining = set(adjacency)
    while remaining:
        start = min(remaining)
        stack = [start]
        component_nodes: set[int] = set()
        while stack:
            node = stack.pop()
            if node in component_nodes:
                continue
            component_nodes.add(node)
            stack.extend(sorted(adjacency[node] - component_nodes, reverse=True))
        remaining.difference_update(component_nodes)
        connection_indices = tuple(
            index
            for index, connection in enumerate(connections)
            if int(connection["node_a"]) in component_nodes
        )
        components.append(
            (tuple(sorted(component_nodes)), connection_indices)
        )
    return components


def _combined_atom_index(
    node_atom_map: Mapping[int, Mapping[int, int]],
    node_index: int,
    canonical_atom_id: int,
) -> int:
    try:
        return int(node_atom_map[node_index][canonical_atom_id])
    except KeyError as exc:
        raise KeyError(
            "connection references an unavailable canonical atom ID: "
            f"node={node_index}, atom={canonical_atom_id}"
        ) from exc


def _inter_fragment_distance(
    conformer: Chem.Conformer,
    combined_atom_a: int,
    combined_atom_b: int,
) -> float:
    position_a = conformer.GetAtomPosition(combined_atom_a)
    position_b = conformer.GetAtomPosition(combined_atom_b)
    delta = np.asarray(
        [
            float(position_a.x - position_b.x),
            float(position_a.y - position_b.y),
            float(position_a.z - position_b.z),
        ],
        dtype=np.float64,
    )
    distance = float(np.linalg.norm(delta))
    if not np.isfinite(distance):
        raise RuntimeError("inter-fragment bond distance is not finite")
    return distance


def _resolve_symmetry_equivalent_connections(
    base_molecule: Chem.Mol,
    node_atom_map: Mapping[int, Mapping[int, int]],
    node_symmetry: Mapping[int, _FragmentSymmetry],
    connections: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Choose one whole-fragment automorphism per node, jointly by distance."""

    if base_molecule.GetNumConformers() == 0:
        raise RuntimeError("disconnected ligand has no conformer")
    conformer = base_molecule.GetConformer(0)
    referenced_by_node: dict[int, set[int]] = {}
    original_distances: dict[int, float] = {}
    for index, connection in enumerate(connections):
        node_a = int(connection["node_a"])
        node_b = int(connection["node_b"])
        atom_a = int(connection["atom_a_id"])
        atom_b = int(connection["atom_b_id"])
        referenced_by_node.setdefault(node_a, set()).add(atom_a)
        referenced_by_node.setdefault(node_b, set()).add(atom_b)
        combined_a = _combined_atom_index(node_atom_map, node_a, atom_a)
        combined_b = _combined_atom_index(node_atom_map, node_b, atom_b)
        original_distances[index] = _inter_fragment_distance(
            conformer,
            combined_a,
            combined_b,
        )

    choices_by_node: dict[int, tuple[dict[int, int], ...]] = {}
    for node_index, referenced_ids in referenced_by_node.items():
        try:
            symmetry = node_symmetry[node_index]
        except KeyError as exc:
            raise KeyError(
                f"missing fragment symmetry data for node {node_index}"
            ) from exc
        choices_by_node[node_index] = _relevant_automorphism_choices(
            symmetry,
            sorted(referenced_ids),
        )

    resolved_atom_ids: dict[int, tuple[int, int]] = {}
    resolved_distances: dict[int, float] = {}
    component_records: list[dict[str, Any]] = []
    for component_nodes, connection_indices in _connection_components(
        connections
    ):
        choice_counts = {
            node: len(choices_by_node[node]) for node in component_nodes
        }
        combination_count = int(prod(choice_counts.values()))
        best_result: tuple[
            tuple[float, int, tuple[int, ...]],
            dict[int, tuple[int, int]],
            dict[int, float],
        ] | None = None

        if combination_count <= MAX_SYMMETRY_COMPONENT_COMBINATIONS:
            node_choice_lists = [
                choices_by_node[node] for node in component_nodes
            ]
            for selected_choices in product(*node_choice_lists):
                selected_by_node = dict(zip(component_nodes, selected_choices))
                candidate_ids: dict[int, tuple[int, int]] = {}
                candidate_distances: dict[int, float] = {}
                seen_combined_bonds: set[tuple[int, int]] = set()
                changed_endpoint_count = 0
                tie_break_ids: list[int] = []
                valid = True
                for connection_index in connection_indices:
                    connection = connections[connection_index]
                    node_a = int(connection["node_a"])
                    node_b = int(connection["node_b"])
                    original_a = int(connection["atom_a_id"])
                    original_b = int(connection["atom_b_id"])
                    resolved_a = selected_by_node[node_a][original_a]
                    resolved_b = selected_by_node[node_b][original_b]
                    combined_a = _combined_atom_index(
                        node_atom_map, node_a, resolved_a
                    )
                    combined_b = _combined_atom_index(
                        node_atom_map, node_b, resolved_b
                    )
                    combined_bond = tuple(sorted((combined_a, combined_b)))
                    if combined_bond in seen_combined_bonds:
                        valid = False
                        break
                    seen_combined_bonds.add(combined_bond)
                    candidate_ids[connection_index] = (
                        resolved_a,
                        resolved_b,
                    )
                    candidate_distances[connection_index] = (
                        _inter_fragment_distance(
                            conformer,
                            combined_a,
                            combined_b,
                        )
                    )
                    changed_endpoint_count += int(resolved_a != original_a)
                    changed_endpoint_count += int(resolved_b != original_b)
                    tie_break_ids.extend((resolved_a, resolved_b))
                if not valid:
                    continue
                total_distance = float(sum(candidate_distances.values()))
                score = (
                    total_distance,
                    changed_endpoint_count,
                    tuple(tie_break_ids),
                )
                if best_result is None or score < best_result[0]:
                    best_result = (score, candidate_ids, candidate_distances)

        if best_result is None:
            chosen_ids = {
                index: (
                    int(connections[index]["atom_a_id"]),
                    int(connections[index]["atom_b_id"]),
                )
                for index in connection_indices
            }
            chosen_distances = {
                index: original_distances[index]
                for index in connection_indices
            }
            status = (
                "skipped_combination_limit"
                if combination_count > MAX_SYMMETRY_COMPONENT_COMBINATIONS
                else "identity_fallback"
            )
        else:
            _, chosen_ids, chosen_distances = best_result
            adjusted = any(
                chosen_ids[index]
                != (
                    int(connections[index]["atom_a_id"]),
                    int(connections[index]["atom_b_id"]),
                )
                for index in connection_indices
            )
            if all(count == 1 for count in choice_counts.values()):
                status = "no_nontrivial_symmetry"
            else:
                status = "optimized" if adjusted else "identity_optimal"

        resolved_atom_ids.update(chosen_ids)
        resolved_distances.update(chosen_distances)
        original_total = float(
            sum(original_distances[index] for index in connection_indices)
        )
        resolved_total = float(
            sum(chosen_distances[index] for index in connection_indices)
        )
        component_records.append(
            {
                "nodes": list(component_nodes),
                "connection_count": len(connection_indices),
                "choice_counts": {
                    str(node): count for node, count in choice_counts.items()
                },
                "combination_count": combination_count,
                "status": status,
                "original_total_bond_length_angstrom": original_total,
                "resolved_total_bond_length_angstrom": resolved_total,
                "bond_length_reduction_angstrom": (
                    original_total - resolved_total
                ),
            }
        )

    resolved_connections: list[dict[str, Any]] = []
    for index, connection in enumerate(connections):
        original_a = int(connection["atom_a_id"])
        original_b = int(connection["atom_b_id"])
        resolved_a, resolved_b = resolved_atom_ids[index]
        resolved_record = dict(connection)
        resolved_record.update(
            {
                "original_atom_a_id": original_a,
                "original_atom_b_id": original_b,
                "resolved_atom_a_id": resolved_a,
                "resolved_atom_b_id": resolved_b,
                "atom_a_id": resolved_a,
                "atom_b_id": resolved_b,
                "symmetry_adjusted": (
                    resolved_a != original_a or resolved_b != original_b
                ),
                "original_bond_length_angstrom": original_distances[index],
                "resolved_bond_length_angstrom": resolved_distances[index],
            }
        )
        resolved_connections.append(resolved_record)

    original_total = float(sum(original_distances.values()))
    resolved_total = float(sum(resolved_distances.values()))
    summary = {
        "method": "whole_fragment_automorphism_minimum_total_bond_length",
        "component_combination_limit": int(
            MAX_SYMMETRY_COMPONENT_COMBINATIONS
        ),
        "connection_count": len(resolved_connections),
        "adjusted_connection_count": sum(
            int(record["symmetry_adjusted"])
            for record in resolved_connections
        ),
        "original_total_bond_length_angstrom": original_total,
        "resolved_total_bond_length_angstrom": resolved_total,
        "bond_length_reduction_angstrom": original_total - resolved_total,
        "components": component_records,
    }
    return resolved_connections, summary


def _select_explicit_hydrogen_replacements(
    molecule: Chem.Mol,
    endpoints: Sequence[Mapping[str, Any]],
) -> dict[tuple[int, str], dict[str, Any]]:
    """Match retained H atoms to new bonds by minimum opposite-atom distance."""

    if molecule.GetNumConformers() == 0:
        raise ValueError("hydrogen selection requires an aligned conformer")
    conformer = molecule.GetConformer(0)
    endpoints_by_center: dict[int, list[Mapping[str, Any]]] = {}
    for endpoint in endpoints:
        center_atom_index = int(endpoint["center_atom_index"])
        endpoints_by_center.setdefault(center_atom_index, []).append(endpoint)

    selected: dict[tuple[int, str], dict[str, Any]] = {}
    for center_atom_index, center_endpoints in sorted(
        endpoints_by_center.items()
    ):
        center_atom = molecule.GetAtomWithIdx(center_atom_index)
        hydrogen_atom_indices: list[int] = []
        for neighbour in center_atom.GetNeighbors():
            if neighbour.GetAtomicNum() != 1:
                continue
            if not neighbour.HasProp(RETAINED_NON_TETRAHEDRAL_H_PROP):
                raise RuntimeError(
                    "connection center has an unclassified explicit hydrogen"
                )
            hydrogen_atom_indices.append(int(neighbour.GetIdx()))
        hydrogen_atom_indices.sort()
        if not hydrogen_atom_indices:
            continue
        if not is_non_tetrahedral_stereo_atom(center_atom):
            raise RuntimeError(
                "retained explicit hydrogen is not attached to a supported "
                "non-tetrahedral stereocenter"
            )

        ordered_endpoints = sorted(
            center_endpoints,
            key=lambda item: (
                int(item["connection_index"]),
                str(item["side"]),
            ),
        )
        endpoint_count = len(ordered_endpoints)
        hydrogen_count = len(hydrogen_atom_indices)
        assignment_candidates: list[tuple[tuple[int, int], ...]] = []
        if hydrogen_count >= endpoint_count:
            for ordered_hydrogens in permutations(
                hydrogen_atom_indices,
                endpoint_count,
            ):
                assignment_candidates.append(
                    tuple(enumerate(ordered_hydrogens))
                )
        else:
            for endpoint_positions in combinations(
                range(endpoint_count),
                hydrogen_count,
            ):
                for ordered_hydrogens in permutations(
                    hydrogen_atom_indices,
                ):
                    assignment_candidates.append(
                        tuple(zip(endpoint_positions, ordered_hydrogens))
                    )

        best: tuple[
            tuple[float, tuple[tuple[int, int], ...]],
            tuple[tuple[int, int, float], ...],
        ] | None = None
        for assignment in assignment_candidates:
            evaluated: list[tuple[int, int, float]] = []
            total_distance = 0.0
            for endpoint_position, hydrogen_atom_index in assignment:
                opposite_atom_index = int(
                    ordered_endpoints[endpoint_position]["opposite_atom_index"]
                )
                distance = _inter_fragment_distance(
                    conformer,
                    hydrogen_atom_index,
                    opposite_atom_index,
                )
                total_distance += distance
                evaluated.append(
                    (endpoint_position, hydrogen_atom_index, distance)
                )
            signature = tuple(
                (endpoint_position, hydrogen_atom_index)
                for endpoint_position, hydrogen_atom_index, _ in evaluated
            )
            score = (total_distance, signature)
            evaluated_tuple = tuple(evaluated)
            if best is None or score < best[0]:
                best = (score, evaluated_tuple)
        if best is None:
            raise RuntimeError("failed to assign retained explicit hydrogens")

        for endpoint_position, hydrogen_atom_index, distance in best[1]:
            endpoint = ordered_endpoints[endpoint_position]
            endpoint_key = (
                int(endpoint["connection_index"]),
                str(endpoint["side"]),
            )
            selected[endpoint_key] = {
                "selection_method": (
                    "minimum_distance_to_opposite_matched_atom"
                ),
                "hydrogen_atom_index_before_removal": hydrogen_atom_index,
                "center_atom_index_before_removal": center_atom_index,
                "opposite_atom_index_before_removal": int(
                    endpoint["opposite_atom_index"]
                ),
                "distance_to_opposite_atom_angstrom": float(distance),
            }
    return selected


def _remap_atom_index_after_removals(
    atom_index: int,
    removed_atom_indices: Sequence[int],
) -> int:
    return int(atom_index) - sum(
        removed_index < int(atom_index)
        for removed_index in removed_atom_indices
    )


def _reassign_affected_nontetrahedral_stereo(
    molecule: Chem.Mol,
    affected_center_indices: Sequence[int],
) -> list[dict[str, Any]]:
    """Re-derive edited non-tetrahedral tags from final 3D coordinates."""

    unique_centers = tuple(sorted(set(map(int, affected_center_indices))))
    if not unique_centers:
        return []
    if (
        hasattr(Chem, "GetAllowNontetrahedralChirality")
        and not Chem.GetAllowNontetrahedralChirality()
    ):
        raise RuntimeError(
            "RDKit non-tetrahedral chirality perception is disabled"
        )

    original_tags = tuple(
        atom.GetChiralTag() for atom in molecule.GetAtoms()
    )
    original_tag_names: dict[int, str] = {}
    permutation_property = "_chiralPermutation"
    for center_atom_index in unique_centers:
        center_atom = molecule.GetAtomWithIdx(center_atom_index)
        if not is_non_tetrahedral_stereo_atom(center_atom):
            raise RuntimeError(
                "explicit hydrogen replacement center lost its original "
                f"non-tetrahedral tag at atom {center_atom_index}"
            )
        original_tag_names[center_atom_index] = str(
            center_atom.GetChiralTag()
        )
        center_atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
        if center_atom.HasProp(permutation_property):
            center_atom.ClearProp(permutation_property)

    try:
        Chem.AssignAtomChiralTagsFromStructure(
            molecule,
            confId=0,
            replaceExistingTags=False,
        )
    except Exception as exc:
        raise RuntimeError(
            "failed to reassign non-tetrahedral stereochemistry from 3D: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    affected_set = set(unique_centers)
    for atom_index, original_tag in enumerate(original_tags):
        if atom_index in affected_set:
            continue
        atom = molecule.GetAtomWithIdx(atom_index)
        if atom.GetChiralTag() == original_tag:
            continue
        atom.SetChiralTag(original_tag)
        if (
            original_tag == Chem.ChiralType.CHI_UNSPECIFIED
            and atom.HasProp(permutation_property)
        ):
            atom.ClearProp(permutation_property)

    records: list[dict[str, Any]] = []
    for center_atom_index in unique_centers:
        center_atom = molecule.GetAtomWithIdx(center_atom_index)
        if not is_non_tetrahedral_stereo_atom(center_atom):
            raise RuntimeError(
                "3D reassignment did not preserve a non-tetrahedral tag at "
                f"atom {center_atom_index}"
            )
        permutation: int | None = None
        if center_atom.HasProp(permutation_property):
            permutation_value = center_atom.GetPropsAsDict(
                includePrivate=True,
                includeComputed=False,
            ).get(permutation_property)
            if permutation_value is not None:
                permutation = int(permutation_value)
        records.append(
            {
                "atom_index_after_removal": center_atom_index,
                "original_chiral_tag": original_tag_names[
                    center_atom_index
                ],
                "reassigned_chiral_tag": str(center_atom.GetChiralTag()),
                "reassigned_chiral_permutation": permutation,
                "method": "AssignAtomChiralTagsFromStructure_3d",
            }
        )
    return records


def _identity_resolved_connections(
    base_molecule: Chem.Mol,
    node_atom_map: Mapping[int, Mapping[int, int]],
    connections: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Resolve no symmetries while retaining normal connection diagnostics."""

    if base_molecule.GetNumConformers() == 0:
        raise RuntimeError("disconnected ligand has no conformer")
    conformer = base_molecule.GetConformer(0)
    resolved_connections: list[dict[str, Any]] = []
    total_distance = 0.0
    for connection in connections:
        node_a = int(connection["node_a"])
        node_b = int(connection["node_b"])
        atom_a = int(connection["atom_a_id"])
        atom_b = int(connection["atom_b_id"])
        combined_a = _combined_atom_index(node_atom_map, node_a, atom_a)
        combined_b = _combined_atom_index(node_atom_map, node_b, atom_b)
        distance = _inter_fragment_distance(
            conformer,
            combined_a,
            combined_b,
        )
        total_distance += distance
        record = dict(connection)
        record.update(
            {
                "original_atom_a_id": atom_a,
                "original_atom_b_id": atom_b,
                "resolved_atom_a_id": atom_a,
                "resolved_atom_b_id": atom_b,
                "atom_a_id": atom_a,
                "atom_b_id": atom_b,
                "symmetry_adjusted": False,
                "original_bond_length_angstrom": distance,
                "resolved_bond_length_angstrom": distance,
            }
        )
        resolved_connections.append(record)
    summary = {
        "method": "identity_mapping_for_chemistry_probe",
        "component_combination_limit": 0,
        "connection_count": len(resolved_connections),
        "adjusted_connection_count": 0,
        "original_total_bond_length_angstrom": total_distance,
        "resolved_total_bond_length_angstrom": total_distance,
        "bond_length_reduction_angstrom": 0.0,
        "components": [],
    }
    return resolved_connections, summary


def _connect_fragments(
    base_molecule: Chem.Mol,
    node_atom_map: Mapping[int, Mapping[int, int]],
    node_symmetry: Mapping[int, _FragmentSymmetry],
    connections: Sequence[Mapping[str, Any]],
    *,
    mode: str,
    source_file: str,
    sample_id: str,
    checkpoint_path: str,
    resolve_symmetry: bool = True,
    neutralize_output: bool = True,
) -> tuple[Chem.Mol, dict[str, Any]]:
    if not isinstance(resolve_symmetry, bool):
        raise TypeError("resolve_symmetry must be a bool")
    if not isinstance(neutralize_output, bool):
        raise TypeError("neutralize_output must be a bool")
    if resolve_symmetry:
        resolved_connections, symmetry_summary = (
            _resolve_symmetry_equivalent_connections(
                base_molecule,
                node_atom_map,
                node_symmetry,
                connections,
            )
        )
    else:
        resolved_connections, symmetry_summary = (
            _identity_resolved_connections(
                base_molecule,
                node_atom_map,
                connections,
            )
        )
    connection_work: list[dict[str, Any]] = []
    seen_atom_pairs: set[tuple[int, int]] = set()
    for connection_index, connection in enumerate(resolved_connections):
        node_a = int(connection["node_a"])
        node_b = int(connection["node_b"])
        atom_a_id = int(connection["atom_a_id"])
        atom_b_id = int(connection["atom_b_id"])
        if node_a == node_b:
            raise ValueError("inter-fragment connections cannot be self-loops")
        try:
            combined_atom_a = int(node_atom_map[node_a][atom_a_id])
            combined_atom_b = int(node_atom_map[node_b][atom_b_id])
        except KeyError as exc:
            raise KeyError(
                "connection references an unavailable canonical atom ID: "
                f"nodes={(node_a, node_b)}, atoms={(atom_a_id, atom_b_id)}"
            ) from exc
        if (
            base_molecule.GetBondBetweenAtoms(
                combined_atom_a,
                combined_atom_b,
            )
            is not None
        ):
            raise ValueError(
                "duplicate inter-fragment bond: "
                f"nodes={(node_a, node_b)}, atoms={(atom_a_id, atom_b_id)}"
            )
        atom_pair_key = tuple(sorted((combined_atom_a, combined_atom_b)))
        if atom_pair_key in seen_atom_pairs:
            raise ValueError(
                "duplicate requested inter-fragment bond: "
                f"nodes={(node_a, node_b)}, atoms={(atom_a_id, atom_b_id)}"
            )
        seen_atom_pairs.add(atom_pair_key)
        connection_work.append(
            {
                "connection_index": connection_index,
                "connection": connection,
                "node_a": node_a,
                "node_b": node_b,
                "atom_a_id": atom_a_id,
                "atom_b_id": atom_b_id,
                "combined_atom_a_before_removal": combined_atom_a,
                "combined_atom_b_before_removal": combined_atom_b,
            }
        )
    removed_hydrogen_indices: list[int] = []
    editable = Chem.RWMol(Chem.Mol(base_molecule))
    editable.UpdatePropertyCache(strict=False)

    normalized_connections: list[dict[str, Any]] = []
    for work_item in connection_work:
        connection_index = int(work_item["connection_index"])
        connection = work_item["connection"]
        node_a = int(work_item["node_a"])
        node_b = int(work_item["node_b"])
        atom_a_id = int(work_item["atom_a_id"])
        atom_b_id = int(work_item["atom_b_id"])
        combined_atom_a = int(
            work_item["combined_atom_a_before_removal"]
        )
        combined_atom_b = int(
            work_item["combined_atom_b_before_removal"]
        )

        atom_a_hydrogen_detail = None
        atom_b_hydrogen_detail = None
        atom_a_hydrogen = consume_connection_hydrogen(
            editable.GetAtomWithIdx(combined_atom_a)
        )
        atom_b_hydrogen = consume_connection_hydrogen(
            editable.GetAtomWithIdx(combined_atom_b)
        )

        editable.AddBond(
            combined_atom_a,
            combined_atom_b,
            Chem.BondType.SINGLE,
        )
        editable.UpdatePropertyCache(strict=False)
        record: dict[str, Any] = {
            "node_a": node_a,
            "node_b": node_b,
            "original_atom_a_id": int(connection["original_atom_a_id"]),
            "original_atom_b_id": int(connection["original_atom_b_id"]),
            "resolved_atom_a_id": atom_a_id,
            "resolved_atom_b_id": atom_b_id,
            "atom_a_id": atom_a_id,
            "atom_b_id": atom_b_id,
            "symmetry_adjusted": bool(connection["symmetry_adjusted"]),
            "original_bond_length_angstrom": float(
                connection["original_bond_length_angstrom"]
            ),
            "resolved_bond_length_angstrom": float(
                connection["resolved_bond_length_angstrom"]
            ),
            "bond_type": "SINGLE",
            "atom_a_consumed_hydrogen": atom_a_hydrogen,
            "atom_b_consumed_hydrogen": atom_b_hydrogen,
            "atom_a_hydrogen_selection": atom_a_hydrogen_detail,
            "atom_b_hydrogen_selection": atom_b_hydrogen_detail,
        }
        if "pair_probability" in connection:
            record["pair_probability"] = float(
                connection["pair_probability"]
            )
        if "connection_source" in connection:
            record["connection_source"] = str(
                connection["connection_source"]
            )
        if "candidate_rank" in connection:
            record["candidate_rank"] = int(connection["candidate_rank"])
        if "selection_adjusted" in connection:
            record["selection_adjusted"] = bool(
                connection["selection_adjusted"]
            )
        normalized_connections.append(record)

    raw_molecule = editable.GetMol()
    Chem.RemoveStereochemistry(raw_molecule)
    stereo_reassignments: list[dict[str, Any]] = []
    try:
        output_molecule = Chem.Mol(raw_molecule)
        Chem.SanitizeMol(output_molecule)
    except Exception as exc:
        raise ValueError(
            f"{mode} reconstruction failed RDKit sanitization: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if neutralize_output:
        try:
            output_molecule, charge_neutralization = (
                neutralize_reconstructed_molecule(output_molecule)
            )
        except (TypeError, ValueError, RuntimeError) as exc:
            total_formal_charge = sum(
                int(atom.GetFormalCharge())
                for atom in output_molecule.GetAtoms()
            )
            remaining_charged_atoms = [
                {
                    "atom_index": int(atom.GetIdx()),
                    "symbol": atom.GetSymbol(),
                    "formal_charge": int(atom.GetFormalCharge()),
                }
                for atom in output_molecule.GetAtoms()
                if int(atom.GetFormalCharge()) != 0
            ]
            charge_neutralization = {
                "status": "fallback_unneutralized",
                "method": "net_charge_proton_transfer",
                "fallback_molecule_source": (
                    "sanitized_pre_neutralization_reconstruction"
                ),
                "initial_total_formal_charge": total_formal_charge,
                "final_total_formal_charge": total_formal_charge,
                "added_hydrogen_count": 0,
                "removed_hydrogen_count": 0,
                "remaining_charge_separated_atom_count": len(
                    remaining_charged_atoms
                ),
                "remaining_charge_separated_atoms": remaining_charged_atoms,
                "failure": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
                "steps": [],
            }
    else:
        total_formal_charge = sum(
            int(atom.GetFormalCharge()) for atom in output_molecule.GetAtoms()
        )
        remaining_charged_atoms = [
            {
                "atom_index": int(atom.GetIdx()),
                "symbol": atom.GetSymbol(),
                "formal_charge": int(atom.GetFormalCharge()),
            }
            for atom in output_molecule.GetAtoms()
            if int(atom.GetFormalCharge()) != 0
        ]
        charge_neutralization = {
            "status": "skipped_intermediate_constraint_validation",
            "method": "net_charge_proton_transfer",
            "initial_total_formal_charge": total_formal_charge,
            "final_total_formal_charge": total_formal_charge,
            "added_hydrogen_count": 0,
            "removed_hydrogen_count": 0,
            "remaining_charge_separated_atom_count": len(
                remaining_charged_atoms
            ),
            "remaining_charge_separated_atoms": remaining_charged_atoms,
            "steps": [],
        }
    sanitization_status = "ok"
    sanitization_error: str | None = None
    hydrogen_replacement_summary = {
        "method": "ordinary_atom_level_hydrogen_consumption",
        "removed_explicit_hydrogen_count": len(removed_hydrogen_indices),
        "stereo_reassignments": stereo_reassignments,
    }

    output_molecule.SetProp("_Name", f"{sample_id}_{mode}")
    output_molecule.SetProp("ReconstructionMode", mode)
    output_molecule.SetProp("SourceFile", source_file)
    output_molecule.SetProp("SampleID", sample_id)
    output_molecule.SetProp("Checkpoint", checkpoint_path)
    output_molecule.SetProp("InterFragmentBondType", "SINGLE")
    output_molecule.SetProp(
        "InterFragmentBondCount", str(len(normalized_connections))
    )
    output_molecule.SetProp("SanitizationStatus", sanitization_status)
    output_molecule.SetProp(
        "SanitizationError", sanitization_error or ""
    )
    output_molecule.SetProp(
        "TotalFormalCharge",
        str(charge_neutralization["final_total_formal_charge"]),
    )
    output_molecule.SetProp(
        "ChargeNeutralizationStatus",
        str(charge_neutralization["status"]),
    )
    output_molecule.SetProp(
        "ChargeNeutralizationJSON",
        json.dumps(
            charge_neutralization,
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    )
    output_molecule.SetProp(
        "ConnectionsJSON",
        json.dumps(
            normalized_connections,
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    )
    output_molecule.SetProp(
        "SymmetryResolutionJSON",
        json.dumps(
            symmetry_summary,
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    )
    output_molecule.SetProp(
        "NonTetrahedralHydrogenReplacementJSON",
        json.dumps(
            hydrogen_replacement_summary,
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    )
    return output_molecule, {
        "sanitization_status": sanitization_status,
        "sanitization_error": sanitization_error,
        "inter_fragment_bond_count": len(normalized_connections),
        "connections": normalized_connections,
        "symmetry_resolution": symmetry_summary,
        "non_tetrahedral_hydrogen_replacement": (
            hydrogen_replacement_summary
        ),
        "charge_neutralization": charge_neutralization,
    }


def _annotate_symmetry_resolved_queries(
    query_records: Sequence[dict[str, Any]],
    predicted_connections: Sequence[Mapping[str, Any]],
    truth_connections: Sequence[Mapping[str, Any]],
) -> None:
    def group_pairs(
        records: Sequence[Mapping[str, Any]],
    ) -> dict[tuple[int, int], list[dict[str, int]]]:
        grouped: dict[tuple[int, int], list[dict[str, int]]] = {}
        for record in records:
            nodes = (int(record["node_a"]), int(record["node_b"]))
            grouped.setdefault(nodes, []).append(
                {
                    "atom_a_id": int(record["atom_a_id"]),
                    "atom_b_id": int(record["atom_b_id"]),
                }
            )
        for pairs in grouped.values():
            pairs.sort(key=lambda item: (item["atom_a_id"], item["atom_b_id"]))
        return grouped

    predicted_by_nodes = group_pairs(predicted_connections)
    truth_by_nodes = group_pairs(truth_connections)
    for query_record in query_records:
        nodes = (
            int(query_record["node_a"]),
            int(query_record["node_b"]),
        )
        predicted_pairs = predicted_by_nodes.get(nodes, [])
        truth_pairs = truth_by_nodes.get(nodes, [])
        predicted_set = {
            (item["atom_a_id"], item["atom_b_id"])
            for item in predicted_pairs
        }
        truth_set = {
            (item["atom_a_id"], item["atom_b_id"])
            for item in truth_pairs
        }
        query_record["symmetry_resolved_predicted_pairs"] = predicted_pairs
        query_record["symmetry_resolved_ground_truth_pairs"] = truth_pairs
        query_record["symmetry_aware_exact_atom_pair_set"] = (
            predicted_set == truth_set
        )


def _data_field(data: Any, name: str) -> Any:
    if isinstance(data, Mapping):
        if name not in data:
            raise KeyError(f"data is missing field {name!r}")
        return data[name]
    if not hasattr(data, name):
        raise AttributeError(f"data is missing field {name!r}")
    return getattr(data, name)


def _stage1_mapping(
    payload: Mapping[str, Any],
    name: str,
) -> Mapping[str, Any]:
    value = payload.get(name)
    if not isinstance(value, Mapping):
        raise TypeError(f"stage-one result {name!r} must be a mapping")
    return value


def _cpu_tensor(value: Any, *, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    result = value.detach().cpu()
    if result.is_floating_point() and not bool(torch.isfinite(result).all()):
        raise ValueError(f"{name} contains NaN/Inf")
    return result


def _scalar_int(value: Any, *, name: str) -> int:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"{name} must be scalar")
        value = value.item()
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be an integer") from exc
    return result


def _require_condition_coordinate_contract(
    condition_data: Any,
    *,
    source_path: str,
) -> None:
    try:
        mode = _data_field(condition_data, "ref_coords_mode")
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise FatalConditionError(
            "Condition .pt is missing the relative ref_coords mode; "
            "rerun PhiSSeparator_work_data_process.ipynb",
            stage="condition_coordinate_contract",
            diagnostics={
                "condition_pt": source_path,
                "error": f"{type(exc).__name__}: {exc}",
            },
        ) from exc
    if mode != "relative_to_node_pos":
        raise FatalConditionError(
            "Condition .pt does not use the required relative ref_coords "
            "contract; rerun PhiSSeparator_work_data_process.ipynb",
            stage="condition_coordinate_contract",
            diagnostics={
                "condition_pt": source_path,
                "ref_coords_mode": mode,
            },
        )


def preflight_fixed_condition_fragments(
    condition_pt_path: str | Path,
    *,
    atom_pair_runtime: AtomPairInferenceRuntime,
) -> dict[str, Any]:
    """Run the complete decoder chain for every fixed ligand node once."""

    condition_path = Path(condition_pt_path).expanduser().resolve()
    if not condition_path.is_file():
        raise FileNotFoundError(f"condition .pt not found: {condition_path}")
    condition_data = torch_load_compat(condition_path, map_location="cpu")
    _require_condition_coordinate_contract(
        condition_data,
        source_path=str(condition_path),
    )

    node_type = _cpu_tensor(
        _data_field(condition_data, "node_type"),
        name="condition.node_type",
    ).float()
    pos = _cpu_tensor(
        _data_field(condition_data, "pos"), name="condition.pos"
    ).float()
    ref_coords = _cpu_tensor(
        _data_field(condition_data, "ref_coords"),
        name="condition.ref_coords",
    ).float()
    raw_smiles = _data_field(condition_data, "fragment_smiles")
    if isinstance(raw_smiles, str):
        raise TypeError("condition fragment_smiles must be a sequence")
    smiles_values = [normalize_smiles_value(value) for value in raw_smiles]
    num_nodes = int(node_type.shape[0])
    if node_type.shape != (num_nodes, 3):
        raise ValueError("condition.node_type must have shape [N,3]")
    if pos.shape != (num_nodes, 3):
        raise ValueError("condition.pos must have shape [N,3]")
    if ref_coords.shape != (num_nodes, 3, 3):
        raise ValueError("condition.ref_coords must have shape [N,3,3]")
    if len(smiles_values) != num_nodes:
        raise ValueError("condition fragment_smiles length differs from N")

    fixed_indices = torch.nonzero(
        node_type[:, 0].bool(), as_tuple=False
    ).flatten().tolist()
    records: list[dict[str, Any]] = []
    for node_index in fixed_indices:
        smiles = smiles_values[node_index]
        try:
            chemistry = atom_pair_runtime.candidate_cache.get(smiles)
            template = atom_pair_runtime.template_cache.get(
                chemistry.canonical_smiles
            )
            absolute_reference_frame = (
                pos[node_index][None, :] + ref_coords[node_index]
            )
            fragment, alignment = align_fragment_conformer_molecule(
                template,
                center=pos[node_index].numpy(),
                reference_frame=(
                    absolute_reference_frame - pos[node_index][None, :]
                ).numpy(),
            )
            if fragment.GetNumConformers() == 0:
                raise RuntimeError("decoder returned no conformer")
            coordinates = np.asarray(
                fragment.GetConformer(0).GetPositions(), dtype=np.float64
            )
            if not np.all(np.isfinite(coordinates)):
                raise RuntimeError("decoded conformer contains NaN or Inf")
        except Exception as exc:
            diagnostics = {
                "condition_pt": str(condition_path),
                "node_index": int(node_index),
                "smiles": smiles,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            raise FatalConditionError(
                "固定片段经过 ETKDG、Open Babel 回退及完整 decoder "
                "流程后仍无法生成有效构象，后续推理已中止: "
                f"node={node_index}, SMILES={smiles!r}; "
                f"{type(exc).__name__}: {exc}",
                stage="fixed_fragment_decoder_preflight",
                diagnostics=diagnostics,
            ) from exc

        records.append(
            {
                "node_index": int(node_index),
                "input_smiles": smiles,
                "canonical_smiles": template.canonical_smiles,
                "embedding_method": template.generation_metadata.get(
                    "embedding_method"
                ),
                "optimization_status": template.generation_metadata.get(
                    "optimization_status"
                ),
                "alignment_status": alignment.get("alignment_status"),
            }
        )

    return {
        "condition_pt": str(condition_path),
        "status": "passed",
        "fixed_fragment_count": len(records),
        "fragments": records,
    }


def _edge_index_tensor(value: Any, *, name: str) -> torch.Tensor:
    edge_index = _cpu_tensor(value, name=name).long()
    if edge_index.numel() == 0:
        return edge_index.reshape(2, 0)
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(f"{name} must have shape [2,E]")
    return edge_index.contiguous()


def _condition_fixed_connections(
    condition_data: Any,
    *,
    source_path: str,
) -> tuple[
    set[int],
    tuple[tuple[int, int], ...],
    list[dict[str, Any]],
]:
    """Read fixed-fixed atom-pair labels from the original condition graph."""

    try:
        node_type = _cpu_tensor(
            _data_field(condition_data, "node_type"),
            name="condition.node_type",
        ).float()
        if node_type.ndim != 2 or node_type.shape[1] != 3:
            raise ValueError("condition.node_type must have shape [N,3]")
        fixed_node_indices = set(
            torch.nonzero(
                node_type[:, 0].bool(), as_tuple=False
            ).flatten().tolist()
        )
        if len(fixed_node_indices) < 2:
            return fixed_node_indices, (), []

        validated = validate_graph_contract(
            condition_data,
            require_targets=True,
        )
        truth_groups = group_connections(validated, require_targets=True)

        node_pairs: set[tuple[int, int]] = set()
        connections: list[dict[str, Any]] = []
        seen_connections: set[tuple[int, int, int, int]] = set()
        for group in truth_groups:
            raw_node_a = int(group.node_a)
            raw_node_b = int(group.node_b)
            if raw_node_a == raw_node_b:
                raise ValueError(
                    "condition connection labels contain a self-loop"
                )
            if (
                raw_node_a not in fixed_node_indices
                or raw_node_b not in fixed_node_indices
            ):
                raise ValueError(
                    "condition connection labels contain a non-fixed-ligand "
                    f"endpoint: {(raw_node_a, raw_node_b)}"
                )

            swap_endpoints = raw_node_a > raw_node_b
            node_a, node_b = (
                (raw_node_b, raw_node_a)
                if swap_endpoints
                else (raw_node_a, raw_node_b)
            )
            raw_atom_pairs = tuple(group.true_atom_pairs)
            if not raw_atom_pairs:
                raise ValueError(
                    "condition connection query has no atom-pair label: "
                    f"{(node_a, node_b)}"
                )
            node_pairs.add((node_a, node_b))
            for raw_atom_a, raw_atom_b in raw_atom_pairs:
                atom_a, atom_b = (
                    (int(raw_atom_b), int(raw_atom_a))
                    if swap_endpoints
                    else (int(raw_atom_a), int(raw_atom_b))
                )
                connection_key = (node_a, node_b, atom_a, atom_b)
                if connection_key in seen_connections:
                    raise ValueError(
                        "condition contains a duplicate fixed-fixed atom "
                        f"connection: {connection_key}"
                    )
                seen_connections.add(connection_key)
                connections.append(
                    {
                        "node_a": node_a,
                        "node_b": node_b,
                        "atom_a_id": atom_a,
                        "atom_b_id": atom_b,
                        "connection_source": (
                            "condition_pt_fixed_fixed_ground_truth"
                        ),
                    }
                )
    except FatalConditionError:
        raise
    except (
        AttributeError,
        IndexError,
        KeyError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        raise FatalConditionError(
            "Condition .pt fixed-fixed atom connections are unavailable or "
            f"invalid: {type(exc).__name__}: {exc}",
            stage="fixed_connection_contract",
            diagnostics={
                "condition_pt": source_path,
                "error": f"{type(exc).__name__}: {exc}",
            },
        ) from exc

    connections.sort(
        key=lambda item: (
            item["node_a"],
            item["node_b"],
            item["atom_a_id"],
            item["atom_b_id"],
        )
    )
    return fixed_node_indices, tuple(sorted(node_pairs)), connections


def _ligand_connected_components(
    ligand_nodes: set[int],
    connections: Sequence[tuple[int, int]],
) -> list[tuple[int, ...]]:
    """Return deterministic connected components, including isolated nodes."""

    adjacency = {node: set() for node in ligand_nodes}
    for first, second in connections:
        if first == second:
            raise ValueError("ligand connectivity graph contains a self-loop")
        if first not in adjacency or second not in adjacency:
            raise ValueError(
                "ligand connectivity graph contains a non-ligand endpoint"
            )
        adjacency[first].add(second)
        adjacency[second].add(first)

    components: list[tuple[int, ...]] = []
    remaining = set(ligand_nodes)
    while remaining:
        root = min(remaining)
        visited: set[int] = set()
        stack = [root]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            stack.extend(sorted(adjacency[node] - visited, reverse=True))
        remaining.difference_update(visited)
        components.append(tuple(sorted(visited)))

    return sorted(components, key=lambda item: (-len(item), item))


def _filter_stage1_graph_edges(
    stage1_result: Mapping[str, Any],
    section_name: str,
    *,
    ligand_nodes: set[int],
    selected_nodes: set[int],
    graph_index_remap: Mapping[int, int],
    condition_node_count: int,
    expected_ligand_endpoints: int,
) -> torch.Tensor:
    section = _stage1_mapping(stage1_result, section_name)
    edge_index = _edge_index_tensor(
        section.get("edge_index_graph"),
        name=f"{section_name}.edge_index_graph",
    )
    filtered_pairs: list[tuple[int, int]] = []
    for raw_first, raw_second in edge_index.t().tolist():
        first = int(raw_first)
        second = int(raw_second)
        ligand_endpoints = {
            endpoint
            for endpoint in (first, second)
            if endpoint in ligand_nodes
        }
        if len(ligand_endpoints) != expected_ligand_endpoints:
            raise ValueError(
                f"{section_name} contains an unexpected endpoint type"
            )
        if not ligand_endpoints.issubset(selected_nodes):
            continue

        remapped: list[int] = []
        for endpoint in (first, second):
            if endpoint in ligand_nodes:
                remapped.append(graph_index_remap[endpoint])
            elif 0 <= endpoint < condition_node_count:
                remapped.append(endpoint)
            else:
                raise IndexError(
                    f"{section_name} contains an out-of-range endpoint"
                )
        filtered_pairs.append((remapped[0], remapped[1]))

    if not filtered_pairs:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(filtered_pairs, dtype=torch.long).t().contiguous()


def _select_stage1_reconstruction_component(
    stage1_result: Mapping[str, Any],
    condition_data: Any,
    fixed_node_indices: set[int],
    condition_fixed_node_pairs: Sequence[tuple[int, int]],
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    """Select and compact the largest acceptable ligand component."""

    nodes = _stage1_mapping(stage1_result, "nodes")
    metadata = _stage1_mapping(stage1_result, "metadata")
    graph_indices = _cpu_tensor(
        nodes.get("graph_index"), name="nodes.graph_index"
    ).long()
    is_fixed = _cpu_tensor(
        nodes.get("is_fixed"), name="nodes.is_fixed"
    ).bool()
    is_generated = _cpu_tensor(
        nodes.get("is_generated"), name="nodes.is_generated"
    ).bool()
    ligand_count = int(graph_indices.numel())
    if ligand_count < 1:
        raise ValueError("stage-one result contains no ligand nodes")
    if graph_indices.shape != (ligand_count,):
        raise ValueError("nodes.graph_index must be one-dimensional")
    if is_fixed.shape != (ligand_count,) or is_generated.shape != (ligand_count,):
        raise ValueError("stage-one fixed/generated masks have invalid shape")
    if not bool(torch.all(is_fixed ^ is_generated)):
        raise ValueError("each ligand node must be exactly fixed or generated")

    graph_index_values = tuple(map(int, graph_indices.tolist()))
    ligand_node_set = set(graph_index_values)
    if len(ligand_node_set) != ligand_count:
        raise ValueError("nodes.graph_index contains duplicates")
    stage_fixed_node_indices = {
        graph_index_values[index]
        for index in torch.nonzero(is_fixed, as_tuple=False).flatten().tolist()
    }
    if stage_fixed_node_indices != fixed_node_indices:
        raise ValueError(
            "stage-one fixed ligand indices differ from the condition graph"
        )

    condition_node_type = _cpu_tensor(
        _data_field(condition_data, "node_type"),
        name="condition.node_type",
    )
    if condition_node_type.ndim != 2 or condition_node_type.shape[1] != 3:
        raise ValueError("condition.node_type must have shape [N,3]")
    condition_node_count = int(condition_node_type.shape[0])
    if any(
        index < 0 or index >= condition_node_count
        for index in fixed_node_indices
    ):
        raise ValueError("fixed ligand index is outside the condition graph")
    stage_generated_node_indices = ligand_node_set - fixed_node_indices
    if any(index < condition_node_count for index in stage_generated_node_indices):
        raise ValueError("generated ligand index overlaps the condition graph")

    covalent = _stage1_mapping(stage1_result, "covalent_edges")
    covalent_edges = _edge_index_tensor(
        covalent.get("edge_index_graph"),
        name="covalent_edges.edge_index_graph",
    )
    stage1_predicted_pairs: set[tuple[int, int]] = set()
    for raw_first, raw_second in covalent_edges.t().tolist():
        first = int(raw_first)
        second = int(raw_second)
        if first == second:
            raise ValueError("covalent fragment graph contains a self-loop")
        if first not in ligand_node_set or second not in ligand_node_set:
            raise ValueError(
                "covalent fragment graph contains a non-ligand endpoint"
            )
        stage1_predicted_pairs.add(tuple(sorted((first, second))))

    hybrid_pairs = {
        pair
        for pair in stage1_predicted_pairs
        if not (
            pair[0] in fixed_node_indices
            and pair[1] in fixed_node_indices
        )
    }
    for raw_first, raw_second in condition_fixed_node_pairs:
        first = int(raw_first)
        second = int(raw_second)
        if first == second:
            raise ValueError("condition fixed connection contains a self-loop")
        if first not in fixed_node_indices or second not in fixed_node_indices:
            raise ValueError(
                "condition fixed connection contains a non-fixed endpoint"
            )
        hybrid_pairs.add(tuple(sorted((first, second))))

    components = _ligand_connected_components(
        ligand_node_set,
        tuple(sorted(hybrid_pairs)),
    )
    raw_minimum = metadata.get(
        "minimum_connected_component_ligand_nodes"
    )
    legacy_threshold_fallback = raw_minimum is None
    minimum_nodes = (
        ligand_count
        if legacy_threshold_fallback
        else _scalar_int(
            raw_minimum,
            name="metadata.minimum_connected_component_ligand_nodes",
        )
    )
    if minimum_nodes < 1 or minimum_nodes > ligand_count:
        raise ValueError(
            "metadata.minimum_connected_component_ligand_nodes must be "
            f"within [1, {ligand_count}]"
        )

    component_records = [
        {
            "rank": rank,
            "node_count": len(component),
            "original_graph_indices": list(component),
            "contains_all_fixed_nodes": fixed_node_indices.issubset(component),
        }
        for rank, component in enumerate(components, start=1)
    ]
    maximum_size = len(components[0])
    largest_components = [
        component for component in components if len(component) == maximum_size
    ]
    eligible_components = [
        component
        for component in largest_components
        if fixed_node_indices.issubset(component)
    ]
    diagnostics: dict[str, Any] = {
        "selection_policy": (
            "largest_hybrid_covalent_component_meeting_minimum_and_"
            "containing_all_fixed_nodes"
        ),
        "threshold_source": (
            "legacy_all_ligand_nodes"
            if legacy_threshold_fallback
            else "stage1_metadata"
        ),
        "minimum_connected_component_ligand_nodes": minimum_nodes,
        "original_ligand_node_count": ligand_count,
        "fixed_original_graph_indices": sorted(fixed_node_indices),
        "component_count": len(components),
        "components": component_records,
    }
    if not eligible_components:
        diagnostics.update(
            {
                "status": "failed",
                "failure_reason": (
                    "no maximum-size component contains every fixed node"
                ),
            }
        )
        raise RetryableReconstructionError(
            "The largest ligand component does not contain every fixed node",
            stage="stage1_ligand_component_selection",
            diagnostics=diagnostics,
        )

    selected_component = eligible_components[0]
    selected_node_set = set(selected_component)
    if len(selected_component) < minimum_nodes:
        diagnostics.update(
            {
                "status": "failed",
                "failure_reason": (
                    "largest fixed-containing component is below the "
                    "required ligand-node count"
                ),
                "selected_component_original_graph_indices": list(
                    selected_component
                ),
                "selected_component_node_count": len(selected_component),
            }
        )
        raise RetryableReconstructionError(
            "The largest ligand component is smaller than the required "
            f"{minimum_nodes} nodes",
            stage="stage1_ligand_component_selection",
            diagnostics=diagnostics,
        )

    selected_generated_indices = sorted(
        selected_node_set - fixed_node_indices
    )
    graph_index_remap = {index: index for index in fixed_node_indices}
    graph_index_remap.update(
        {
            original_index: condition_node_count + offset
            for offset, original_index in enumerate(selected_generated_indices)
        }
    )
    discarded_indices = sorted(ligand_node_set - selected_node_set)
    diagnostics.update(
        {
            "status": "success",
            "selected_component_rank": components.index(selected_component) + 1,
            "selected_component_node_count": len(selected_component),
            "selected_component_original_graph_indices": list(
                selected_component
            ),
            "discarded_original_graph_indices": discarded_indices,
            "discarded_node_count": len(discarded_indices),
            "filtered_for_reconstruction": bool(discarded_indices),
            "graph_index_mapping": [
                {
                    "original_graph_index": original_index,
                    "reconstruction_graph_index": graph_index_remap[
                        original_index
                    ],
                }
                for original_index in sorted(selected_node_set)
            ],
        }
    )
    if not discarded_indices:
        return stage1_result, diagnostics

    keep_local_indices = [
        local_index
        for local_index, graph_index in enumerate(graph_index_values)
        if graph_index in selected_node_set
    ]
    selected_graph_indices_in_local_order = [
        graph_index_values[local_index]
        for local_index in keep_local_indices
    ]
    keep_index = torch.tensor(keep_local_indices, dtype=torch.long)
    filtered_nodes: dict[str, torch.Tensor] = {}
    for field_name in (
        "graph_index",
        "is_fixed",
        "is_generated",
        "pos",
        "ref_coords_relative",
        "ref_coords_absolute",
        "feature_128",
        "feature_128_normalized",
        "hac_value",
        "hac_overflow",
        "ring_value",
        "ring_overflow",
    ):
        value = _cpu_tensor(nodes.get(field_name), name=f"nodes.{field_name}")
        if value.ndim < 1 or int(value.shape[0]) != ligand_count:
            raise ValueError(
                f"nodes.{field_name} does not align with ligand nodes"
            )
        filtered_nodes[field_name] = value.index_select(0, keep_index).clone()
    filtered_nodes["graph_index"] = torch.tensor(
        [
            graph_index_remap[index]
            for index in selected_graph_indices_in_local_order
        ],
        dtype=torch.long,
    )

    filtered_metadata = dict(metadata)
    filtered_metadata.update(
        {
            "num_fixed_ligand_nodes": len(fixed_node_indices),
            "num_generated_ligand_nodes": len(selected_generated_indices),
            "num_ligand_nodes": len(selected_component),
            "minimum_connected_component_ligand_nodes": minimum_nodes,
        }
    )
    filtered_result: dict[str, Any] = {
        "metadata": filtered_metadata,
        "nodes": filtered_nodes,
        "ll_edges": {
            "edge_index_graph": _filter_stage1_graph_edges(
                stage1_result,
                "ll_edges",
                ligand_nodes=ligand_node_set,
                selected_nodes=selected_node_set,
                graph_index_remap=graph_index_remap,
                condition_node_count=condition_node_count,
                expected_ligand_endpoints=2,
            )
        },
        "lp_edges": {
            "edge_index_graph": _filter_stage1_graph_edges(
                stage1_result,
                "lp_edges",
                ligand_nodes=ligand_node_set,
                selected_nodes=selected_node_set,
                graph_index_remap=graph_index_remap,
                condition_node_count=condition_node_count,
                expected_ligand_endpoints=1,
            )
        },
        "covalent_edges": {
            "edge_index_graph": _filter_stage1_graph_edges(
                stage1_result,
                "covalent_edges",
                ligand_nodes=ligand_node_set,
                selected_nodes=selected_node_set,
                graph_index_remap=graph_index_remap,
                condition_node_count=condition_node_count,
                expected_ligand_endpoints=2,
            )
        },
    }
    return filtered_result, diagnostics


def _resolve_stage1_ligand_smiles(
    stage1_result: Mapping[str, Any],
    condition_data: Any,
    custom_vocabulary: FragmentVocabulary,
    *,
    min_cosine_similarity: float | None,
    diagnostic_top_k: int,
    enable_hac_ring_constraints: bool,
    hac_tolerance: int,
    ring_tolerance: int,
    max_constraint_candidates: int,
) -> tuple[dict[int, str], list[dict[str, Any]]]:
    nodes = _stage1_mapping(stage1_result, "nodes")
    graph_indices = _cpu_tensor(
        nodes.get("graph_index"), name="nodes.graph_index"
    ).long()
    is_fixed = _cpu_tensor(nodes.get("is_fixed"), name="nodes.is_fixed").bool()
    is_generated = _cpu_tensor(
        nodes.get("is_generated"), name="nodes.is_generated"
    ).bool()
    features = _cpu_tensor(
        nodes.get("feature_128"), name="nodes.feature_128"
    ).float()
    hac_values = _cpu_tensor(
        nodes.get("hac_value"), name="nodes.hac_value"
    ).long()
    ring_values = _cpu_tensor(
        nodes.get("ring_value"), name="nodes.ring_value"
    ).long()
    hac_overflow = _cpu_tensor(
        nodes.get("hac_overflow"), name="nodes.hac_overflow"
    ).bool()
    ring_overflow = _cpu_tensor(
        nodes.get("ring_overflow"), name="nodes.ring_overflow"
    ).bool()
    ligand_count = int(graph_indices.numel())
    if graph_indices.shape != (ligand_count,):
        raise ValueError("nodes.graph_index must be one-dimensional")
    if is_fixed.shape != (ligand_count,) or is_generated.shape != (ligand_count,):
        raise ValueError("stage-one fixed/generated masks have invalid shape")
    if features.shape != (ligand_count, 128):
        raise ValueError(
            "nodes.feature_128 must have shape "
            f"{(ligand_count, 128)}, got {tuple(features.shape)}"
        )
    for name, value in (
        ("hac_value", hac_values),
        ("ring_value", ring_values),
        ("hac_overflow", hac_overflow),
        ("ring_overflow", ring_overflow),
    ):
        if value.shape != (ligand_count,):
            raise ValueError(f"nodes.{name} must have shape [{ligand_count}]")
    if not bool(torch.all(is_fixed ^ is_generated)):
        raise ValueError("each ligand node must be exactly fixed or generated")
    if len(set(map(int, graph_indices.tolist()))) != ligand_count:
        raise ValueError("nodes.graph_index contains duplicates")

    condition_smiles_raw = _data_field(condition_data, "fragment_smiles")
    if isinstance(condition_smiles_raw, str):
        raise TypeError("condition fragment_smiles must be a sequence")
    condition_smiles = tuple(
        normalize_smiles_value(value) for value in condition_smiles_raw
    )
    condition_node_type = _cpu_tensor(
        _data_field(condition_data, "node_type"),
        name="condition.node_type",
    ).float()
    if condition_node_type.ndim != 2 or condition_node_type.shape[1] != 3:
        raise ValueError("condition.node_type must have shape [N,3]")
    if len(condition_smiles) != int(condition_node_type.shape[0]):
        raise ValueError("condition fragment_smiles length differs from node count")

    resolved: dict[int, str] = {}
    records: list[dict[str, Any]] = []
    for ligand_index, graph_index_tensor in enumerate(graph_indices):
        graph_index = int(graph_index_tensor.item())
        fixed = bool(is_fixed[ligand_index].item())
        if fixed:
            if graph_index < 0 or graph_index >= len(condition_smiles):
                raise IndexError(
                    f"fixed ligand graph index is outside condition graph: {graph_index}"
                )
            if not bool(condition_node_type[graph_index, 0]):
                raise ValueError(
                    f"fixed node {graph_index} is not a ligand in the condition graph"
                )
            smiles = condition_smiles[graph_index]
            match_metadata: dict[str, Any] = {
                "vocab_id": None,
                "cosine_similarity": None,
                "selected_rank": None,
                "vocabulary_path": None,
                "top_matches": [],
                "candidate_checks": [],
                "match_constraints": None,
            }
            source = "condition_fragment_smiles"
        else:
            if graph_index < len(condition_smiles):
                raise ValueError(
                    f"generated node {graph_index} overlaps the condition graph"
                )
            try:
                smiles, match_metadata = match_fragment_embedding(
                    features[ligand_index].numpy(),
                    vocabulary=custom_vocabulary,
                    min_cosine_similarity=min_cosine_similarity,
                    diagnostic_top_k=diagnostic_top_k,
                    require_uff_parameters=True,
                    enable_hac_ring_constraints=(
                        enable_hac_ring_constraints
                    ),
                    predicted_hac=int(hac_values[ligand_index].item()),
                    predicted_ring_count=int(
                        ring_values[ligand_index].item()
                    ),
                    hac_overflow=bool(
                        hac_overflow[ligand_index].item()
                    ),
                    ring_overflow=bool(
                        ring_overflow[ligand_index].item()
                    ),
                    hac_tolerance=hac_tolerance,
                    ring_tolerance=ring_tolerance,
                    max_constraint_candidates=max_constraint_candidates,
                )
            except (FragmentMatchConstraintError, ValueError) as exc:
                diagnostics = dict(getattr(exc, "diagnostics", {}))
                diagnostics.update(
                    {
                        "ligand_index": ligand_index,
                        "graph_index": graph_index,
                        "hac_ring_constraints_enabled": bool(
                            enable_hac_ring_constraints
                        ),
                        "predicted_hac": (
                            int(hac_values[ligand_index].item())
                            if enable_hac_ring_constraints
                            else None
                        ),
                        "predicted_ring_count": (
                            int(ring_values[ligand_index].item())
                            if enable_hac_ring_constraints
                            else None
                        ),
                    }
                )
                raise RetryableReconstructionError(
                    "Generated fragment vocabulary matching failed for "
                    f"graph node {graph_index}: {exc}",
                    stage="fragment_vocabulary_match",
                    diagnostics=diagnostics,
                ) from exc
            source = "custom_fragment_embedding_cosine_match"
        resolved[graph_index] = smiles
        records.append(
            {
                "ligand_index": ligand_index,
                "graph_index": graph_index,
                "is_fixed": fixed,
                "is_generated": not fixed,
                "hac_ring_constraints_enabled": (
                    bool(enable_hac_ring_constraints) if not fixed else None
                ),
                "smiles_source": source,
                "resolved_smiles": smiles,
                **match_metadata,
            }
        )
    return resolved, records


def _build_stage1_atom_pair_data(
    stage1_result: Mapping[str, Any],
    condition_data: Any,
    resolved_ligand_smiles: Mapping[int, str],
    condition_fixed_node_pairs: Sequence[tuple[int, int]],
    *,
    source_path: str,
) -> tuple[
    SimpleNamespace,
    tuple[tuple[int, int], ...],
    tuple[tuple[int, int], ...],
]:
    nodes = _stage1_mapping(stage1_result, "nodes")
    metadata = _stage1_mapping(stage1_result, "metadata")
    _require_condition_coordinate_contract(
        condition_data,
        source_path=source_path,
    )
    graph_indices = _cpu_tensor(
        nodes.get("graph_index"), name="nodes.graph_index"
    ).long()
    ligand_pos = _cpu_tensor(nodes.get("pos"), name="nodes.pos").float()
    ligand_ref_relative = _cpu_tensor(
        nodes.get("ref_coords_relative"),
        name="nodes.ref_coords_relative",
    ).float()
    ligand_count = int(graph_indices.numel())
    if ligand_pos.shape != (ligand_count, 3):
        raise ValueError("nodes.pos has invalid shape")
    if ligand_ref_relative.shape != (ligand_count, 3, 3):
        raise ValueError("nodes.ref_coords_relative has invalid shape")

    condition_node_type = _cpu_tensor(
        _data_field(condition_data, "node_type"),
        name="condition.node_type",
    ).float()
    condition_pos = _cpu_tensor(
        _data_field(condition_data, "pos"), name="condition.pos"
    ).float()
    condition_ref_relative = _cpu_tensor(
        _data_field(condition_data, "ref_coords"),
        name="condition.ref_coords",
    ).float()
    condition_edge_index = _edge_index_tensor(
        _data_field(condition_data, "edge_index"),
        name="condition.edge_index",
    )
    condition_edge_attr = _cpu_tensor(
        _data_field(condition_data, "edge_attr"),
        name="condition.edge_attr",
    ).float()
    raw_smiles = _data_field(condition_data, "fragment_smiles")
    if isinstance(raw_smiles, str):
        raise TypeError("condition fragment_smiles must be a sequence")
    condition_smiles = [
        normalize_smiles_value(value) for value in raw_smiles
    ]
    condition_count = int(condition_node_type.shape[0])
    if condition_node_type.shape != (condition_count, 3):
        raise ValueError("condition.node_type must have shape [N,3]")
    if condition_pos.shape != (condition_count, 3):
        raise ValueError("condition.pos must have shape [N,3]")
    if condition_ref_relative.shape != (condition_count, 3, 3):
        raise ValueError("condition.ref_coords must have shape [N,3,3]")
    if condition_edge_attr.shape != (condition_edge_index.shape[1], 4):
        raise ValueError("condition.edge_attr must have shape [E,4]")
    if len(condition_smiles) != condition_count:
        raise ValueError("condition fragment_smiles length differs from N")

    graph_index_values = tuple(map(int, graph_indices.tolist()))
    generated_indices = sorted(
        index for index in graph_index_values if index >= condition_count
    )
    expected_generated = list(
        range(condition_count, condition_count + len(generated_indices))
    )
    if generated_indices != expected_generated:
        raise ValueError(
            "generated ligand graph indices must be contiguous and appended "
            f"after condition nodes: {generated_indices}"
        )
    final_count = condition_count + len(generated_indices)
    if any(index < 0 or index >= final_count for index in graph_index_values):
        raise IndexError("stage-one ligand graph index is out of range")
    original_ligand_indices = set(
        torch.nonzero(
            condition_node_type[:, 0].bool(), as_tuple=False
        ).flatten().tolist()
    )
    stage_fixed_indices = {
        index for index in graph_index_values if index < condition_count
    }
    if stage_fixed_indices != original_ligand_indices:
        raise ValueError(
            "stage-one fixed ligand indices differ from the condition graph"
        )
    if set(resolved_ligand_smiles) != set(graph_index_values):
        raise ValueError("resolved ligand SMILES do not cover every ligand node")

    generated_node_type = torch.zeros(
        (len(generated_indices), 3), dtype=torch.float32
    )
    if len(generated_indices):
        generated_node_type[:, 0] = 1.0
    node_type = torch.cat(
        (condition_node_type, generated_node_type), dim=0
    )
    pos = torch.cat(
        (condition_pos, torch.zeros((len(generated_indices), 3))), dim=0
    ).float()
    ref_coords_absolute = torch.cat(
        (
            condition_pos[:, None, :] + condition_ref_relative,
            torch.zeros((len(generated_indices), 3, 3)),
        ),
        dim=0,
    ).float()
    fragment_smiles = condition_smiles + [""] * len(generated_indices)
    for ligand_index, graph_index in enumerate(graph_index_values):
        if graph_index < condition_count:
            if not torch.allclose(
                ligand_pos[ligand_index],
                condition_pos[graph_index],
                rtol=0.0,
                atol=1.0e-6,
            ) or not torch.allclose(
                ligand_ref_relative[ligand_index],
                condition_ref_relative[graph_index],
                rtol=0.0,
                atol=1.0e-6,
            ):
                raise FatalConditionError(
                    "Stage-one fixed-node geometry differs from the "
                    f"condition .pt at graph node {graph_index}",
                    stage="fixed_geometry_consistency",
                    diagnostics={
                        "condition_pt": source_path,
                        "graph_index": graph_index,
                    },
                )
            expected_relative = condition_ref_relative[graph_index]
        else:
            pos[graph_index] = ligand_pos[ligand_index]
            ref_coords_absolute[graph_index] = (
                ligand_pos[ligand_index][None, :]
                + ligand_ref_relative[ligand_index]
            )
            expected_relative = ligand_ref_relative[ligand_index]
        fragment_smiles[graph_index] = resolved_ligand_smiles[graph_index]
        restored_relative = (
            ref_coords_absolute[graph_index] - pos[graph_index][None, :]
        )
        if not torch.allclose(
            restored_relative,
            expected_relative,
            rtol=0.0,
            atol=1.0e-6,
        ):
            raise RuntimeError("relative reference-frame restoration failed")

    directed_edges: dict[tuple[int, int], int] = {}

    def add_directed_edge(source: int, target: int, edge_type: int) -> None:
        if source == target:
            raise ValueError("atom-pair graph cannot contain self-loops")
        if source < 0 or target < 0 or source >= final_count or target >= final_count:
            raise IndexError("atom-pair graph edge index is out of range")
        key = (source, target)
        previous = directed_edges.get(key)
        if previous is not None and previous != edge_type:
            raise ValueError(f"edge {key} has conflicting types")
        directed_edges[key] = edge_type

    pp_mask = condition_edge_attr[:, 2].bool()
    for source, target in condition_edge_index[:, pp_mask].t().tolist():
        add_directed_edge(int(source), int(target), 2)

    for section_name, edge_type in (("ll_edges", 0), ("lp_edges", 1)):
        section = _stage1_mapping(stage1_result, section_name)
        selected = _edge_index_tensor(
            section.get("edge_index_graph"),
            name=f"{section_name}.edge_index_graph",
        )
        for first, second in selected.t().tolist():
            first = int(first)
            second = int(second)
            if (
                first < 0
                or second < 0
                or first >= final_count
                or second >= final_count
            ):
                raise IndexError(f"{section_name} contains an out-of-range node")
            first_is_ligand = bool(node_type[first, 0])
            second_is_ligand = bool(node_type[second, 0])
            first_is_protein = bool(node_type[first, 1])
            second_is_protein = bool(node_type[second, 1])
            if edge_type == 0 and not (first_is_ligand and second_is_ligand):
                raise ValueError("ll_edges contains a non-ligand endpoint")
            if edge_type == 1 and not (
                (first_is_ligand and second_is_protein)
                or (first_is_protein and second_is_ligand)
            ):
                raise ValueError("lp_edges must contain one L and one P endpoint")
            add_directed_edge(first, second, edge_type)
            add_directed_edge(second, first, edge_type)

    ordered_edges = sorted(directed_edges)
    if ordered_edges:
        edge_index = torch.tensor(ordered_edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.zeros((len(ordered_edges), 4), dtype=torch.float32)
        for row, key in enumerate(ordered_edges):
            edge_attr[row, directed_edges[key]] = 1.0
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 4), dtype=torch.float32)

    covalent = _stage1_mapping(stage1_result, "covalent_edges")
    covalent_edges = _edge_index_tensor(
        covalent.get("edge_index_graph"),
        name="covalent_edges.edge_index_graph",
    )
    query_pairs_set: set[tuple[int, int]] = set()
    for first, second in covalent_edges.t().tolist():
        first = int(first)
        second = int(second)
        if (
            first < 0
            or second < 0
            or first >= final_count
            or second >= final_count
        ):
            raise IndexError("covalent fragment graph contains an out-of-range node")
        if first == second:
            raise ValueError("covalent fragment graph contains a self-loop")
        if not bool(node_type[first, 0]) or not bool(node_type[second, 0]):
            raise ValueError("covalent fragment graph contains a non-L endpoint")
        query_pairs_set.add(tuple(sorted((first, second))))
    stage1_predicted_pairs = tuple(sorted(query_pairs_set))

    ligand_node_set = set(graph_index_values)
    fixed_pair_set: set[tuple[int, int]] = set()
    for raw_first, raw_second in condition_fixed_node_pairs:
        first = int(raw_first)
        second = int(raw_second)
        if first == second:
            raise ValueError("condition fixed connection contains a self-loop")
        pair = (min(first, second), max(first, second))
        if (
            pair[0] not in original_ligand_indices
            or pair[1] not in original_ligand_indices
        ):
            raise ValueError(
                "condition fixed connection contains a non-fixed endpoint: "
                f"{pair}"
            )
        if pair in fixed_pair_set:
            raise ValueError(
                f"duplicate condition fixed connection node pair: {pair}"
            )
        fixed_pair_set.add(pair)

    # Keep the raw stage-one pairs for diagnostics.  The atom-pair model and
    # reconstruction instead use condition-derived covalent pairs between two
    # fixed nodes and stage-one predictions for every other node combination.
    atom_pair_query_pair_set = {
        pair
        for pair in stage1_predicted_pairs
        if not (
            pair[0] in original_ligand_indices
            and pair[1] in original_ligand_indices
        )
    }
    atom_pair_query_pair_set.update(fixed_pair_set)
    atom_pair_query_pairs = tuple(sorted(atom_pair_query_pair_set))
    if len(ligand_node_set) > 1:
        if not atom_pair_query_pairs:
            raise ValueError(
                "hybrid covalent graph has no edge for a multi-fragment ligand"
            )
        adjacency = {node: set() for node in ligand_node_set}
        for first, second in atom_pair_query_pairs:
            adjacency[first].add(second)
            adjacency[second].add(first)
        visited: set[int] = set()
        stack = [next(iter(ligand_node_set))]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            stack.extend(adjacency[node] - visited)
        if visited != ligand_node_set:
            missing = sorted(ligand_node_set - visited)
            raise ValueError(
                "hybrid covalent graph is disconnected; unreachable ligand "
                f"nodes: {missing}"
            )

    connection_edge_index = (
        torch.tensor(atom_pair_query_pairs, dtype=torch.long).t().contiguous()
        if atom_pair_query_pairs
        else torch.empty((2, 0), dtype=torch.long)
    )
    data = SimpleNamespace(
        node_type=node_type,
        pos=pos,
        ref_coords=ref_coords_absolute,
        fragment_smiles=fragment_smiles,
        edge_index=edge_index,
        edge_attr=edge_attr,
        connection_edge_index=connection_edge_index,
        sample_id=str(metadata.get("sample_id", Path(source_path).stem)),
    )
    validate_graph_contract(data, require_targets=False)
    return data, stage1_predicted_pairs, atom_pair_query_pairs


def _prediction_connections(
    predictions: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    query_records: list[dict[str, Any]] = []
    connections: list[dict[str, Any]] = []
    seen_queries: set[tuple[int, int]] = set()
    for prediction in predictions:
        node_pair = (
            int(prediction["node_a"]),
            int(prediction["node_b"]),
        )
        if node_pair in seen_queries:
            raise RuntimeError(f"duplicate atom-pair query: {node_pair}")
        seen_queries.add(node_pair)
        selected = prediction.get("selected_pairs")
        if not isinstance(selected, Sequence):
            raise TypeError("prediction selected_pairs must be a sequence")
        selected_records: list[dict[str, Any]] = []
        for selected_index, pair in enumerate(selected, start=1):
            if not isinstance(pair, Mapping):
                raise TypeError("selected atom-pair record must be a mapping")
            record = {
                "candidate_rank": int(
                    pair.get("candidate_rank", selected_index)
                ),
                "atom_a_id": int(pair["atom_a_id"]),
                "atom_b_id": int(pair["atom_b_id"]),
                "pair_probability": float(pair["pair_probability"]),
            }
            if not 0.0 <= record["pair_probability"] <= 1.0:
                raise ValueError(
                    f"invalid selected pair probability for query {node_pair}"
                )
            selected_records.append(record)
            connections.append(
                {
                    "node_a": node_pair[0],
                    "node_b": node_pair[1],
                    **record,
                }
            )
        if int(prediction["predicted_count"]) != len(selected_records):
            raise RuntimeError(
                f"predicted_count mismatch for query {node_pair}"
            )
        raw_ranked = prediction.get("ranked_pairs")
        if not isinstance(raw_ranked, Sequence):
            raise TypeError(
                f"prediction ranked_pairs must be a sequence for {node_pair}"
            )
        ranked_records: list[dict[str, Any]] = []
        seen_atom_pairs: set[tuple[int, int]] = set()
        previous_sort_key: tuple[float, int, int] | None = None
        for expected_rank, pair in enumerate(raw_ranked, start=1):
            if not isinstance(pair, Mapping):
                raise TypeError("ranked atom-pair record must be a mapping")
            atom_a_id = int(pair["atom_a_id"])
            atom_b_id = int(pair["atom_b_id"])
            probability = float(pair["pair_probability"])
            candidate_rank = int(pair.get("candidate_rank", expected_rank))
            if candidate_rank != expected_rank:
                raise ValueError(
                    f"non-contiguous candidate ranks for query {node_pair}"
                )
            if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise ValueError(
                    f"invalid ranked pair probability for query {node_pair}"
                )
            atom_pair = (atom_a_id, atom_b_id)
            if atom_pair in seen_atom_pairs:
                raise ValueError(
                    f"duplicate ranked atom pair for query {node_pair}: "
                    f"{atom_pair}"
                )
            seen_atom_pairs.add(atom_pair)
            sort_key = (-probability, atom_a_id, atom_b_id)
            if previous_sort_key is not None and sort_key < previous_sort_key:
                raise ValueError(
                    f"ranked atom pairs are not deterministically sorted for "
                    f"query {node_pair}"
                )
            previous_sort_key = sort_key
            ranked_records.append(
                {
                    "candidate_rank": candidate_rank,
                    "atom_a_id": atom_a_id,
                    "atom_b_id": atom_b_id,
                    "pair_probability": probability,
                }
            )
        if int(prediction["candidate_count"]) != len(ranked_records):
            raise RuntimeError(
                f"candidate_count mismatch for query {node_pair}"
            )
        for selected_record in selected_records:
            candidate_rank = int(selected_record["candidate_rank"])
            if not 1 <= candidate_rank <= len(ranked_records):
                raise RuntimeError(
                    f"selected candidate rank is invalid for query {node_pair}"
                )
            ranked_record = ranked_records[candidate_rank - 1]
            selected_identity = (
                int(selected_record["atom_a_id"]),
                int(selected_record["atom_b_id"]),
            )
            ranked_identity = (
                int(ranked_record["atom_a_id"]),
                int(ranked_record["atom_b_id"]),
            )
            if selected_identity != ranked_identity:
                raise RuntimeError(
                    f"selected_pairs do not match ranked_pairs for query {node_pair}"
                )
        query_record = dict(prediction)
        query_record["selected_pairs"] = selected_records
        query_record["ranked_pairs"] = ranked_records
        query_records.append(query_record)
    return query_records, connections


def _connection_signature(
    connection: Mapping[str, Any],
) -> tuple[int, int, int, int]:
    node_a = int(connection["node_a"])
    node_b = int(connection["node_b"])
    atom_a_id = int(connection["atom_a_id"])
    atom_b_id = int(connection["atom_b_id"])
    if node_a > node_b:
        node_a, node_b = node_b, node_a
        atom_a_id, atom_b_id = atom_b_id, atom_a_id
    return node_a, node_b, atom_a_id, atom_b_id


def _validate_fixed_connection_constraints(
    base_molecule: Chem.Mol,
    node_atom_map: Mapping[int, Mapping[int, int]],
    node_symmetry: Mapping[int, _FragmentSymmetry],
    fixed_connections: Sequence[Mapping[str, Any]],
    *,
    source_file: str,
    sample_id: str,
    checkpoint_path: str,
) -> None:
    """Run the formal chemistry pipeline on locked condition connections."""

    if not fixed_connections:
        return
    log_blocker = rdBase.BlockLogs()
    try:
        _connect_fragments(
            base_molecule,
            node_atom_map,
            node_symmetry,
            fixed_connections,
            mode="fixed_constraint_validation",
            source_file=source_file,
            sample_id=sample_id,
            checkpoint_path=checkpoint_path,
            resolve_symmetry=False,
            neutralize_output=False,
        )
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        diagnostics = {
            "strategy": "locked_fixed_connection_validation",
            "status": "fixed_constraints_invalid",
            "fixed_connection_count": len(fixed_connections),
            "error": f"{type(exc).__name__}: {exc}",
        }
        raise FatalConditionError(
            "Fixed-fixed atom connections failed molecule-level chemistry "
            f"validation: {exc}",
            stage="fixed_connection_valence",
            diagnostics=diagnostics,
        ) from exc
    finally:
        del log_blocker


def _select_global_reconstruction_combination(
    base_molecule: Chem.Mol,
    node_atom_map: Mapping[int, Mapping[int, int]],
    node_symmetry: Mapping[int, _FragmentSymmetry],
    global_combinations: Sequence[Mapping[str, Any]],
    query_records: Sequence[dict[str, Any]],
    fixed_node_indices: set[int],
    condition_fixed_connections: Sequence[Mapping[str, Any]],
    *,
    source_file: str,
    sample_id: str,
    checkpoint_path: str,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    Chem.Mol,
    dict[str, Any],
    dict[str, Any],
]:
    """Formally validate global combinations in score order."""

    fixed_connections = [dict(record) for record in condition_fixed_connections]
    _validate_fixed_connection_constraints(
        base_molecule,
        node_atom_map,
        node_symmetry,
        fixed_connections,
        source_file=source_file,
        sample_id=sample_id,
        checkpoint_path=checkpoint_path,
    )
    expected_fixed = sorted(
        _connection_signature(record) for record in fixed_connections
    )
    failures: list[dict[str, Any]] = []

    for expected_rank, raw_combination in enumerate(global_combinations, start=1):
        if not isinstance(raw_combination, Mapping):
            raise TypeError("global combination records must be mappings")
        rank = int(raw_combination.get("global_combination_rank", expected_rank))
        if rank != expected_rank:
            raise ValueError("global combination ranks must be contiguous")
        raw_connections = raw_combination.get("connections")
        raw_predicted = raw_combination.get("predicted_connections")
        if not isinstance(raw_connections, Sequence):
            raise TypeError("global combination connections must be a sequence")
        if not isinstance(raw_predicted, Sequence):
            raise TypeError(
                "global combination predicted_connections must be a sequence"
            )
        connections = [dict(record) for record in raw_connections]
        predicted_connections = [dict(record) for record in raw_predicted]
        expected_connections = sorted(
            [
                *expected_fixed,
                *(
                    _connection_signature(record)
                    for record in predicted_connections
                ),
            ]
        )
        actual_connections = sorted(
            _connection_signature(record) for record in connections
        )
        if actual_connections != expected_connections:
            raise ValueError(
                "global combination does not equal fixed truth union predictions"
            )

        log_blocker = rdBase.BlockLogs()
        try:
            molecule, connection_status = _connect_fragments(
                base_molecule,
                node_atom_map,
                node_symmetry,
                connections,
                mode=f"global_combination_{rank}",
                source_file=source_file,
                sample_id=sample_id,
                checkpoint_path=checkpoint_path,
            )
            components = Chem.GetMolFrags(
                molecule,
                asMols=False,
                sanitizeFrags=False,
            )
            if len(components) != 1:
                raise ValueError(
                    "global atom-pair combination did not produce one connected "
                    f"ligand; component_count={len(components)}"
                )
        except (KeyError, RuntimeError, TypeError, ValueError) as exc:
            failures.append(
                {
                    "global_combination_rank": rank,
                    "global_score": raw_combination.get("global_score"),
                    "capacity_usage": raw_combination.get("capacity_usage", []),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        finally:
            del log_blocker

        ring_pruning = dict(raw_combination.get("four_membered_ring_pruning", {
            "removed_connection_count": 0,
            "removed_connections": [],
        }))
        removed_by_query: dict[tuple[int, int], list[dict[str, Any]]] = {}
        for connection in ring_pruning["removed_connections"]:
            node_pair = (int(connection["node_a"]), int(connection["node_b"]))
            removed_by_query.setdefault(node_pair, []).append(dict(connection))
        selected_by_query: dict[tuple[int, int], list[dict[str, Any]]] = {}
        for connection in connections:
            node_pair = (
                int(connection["node_a"]),
                int(connection["node_b"]),
            )
            selected_by_query.setdefault(node_pair, []).append(dict(connection))
        for query_record in query_records:
            node_pair = (
                int(query_record["node_a"]),
                int(query_record["node_b"]),
            )
            is_fixed_fixed = (
                node_pair[0] in fixed_node_indices
                and node_pair[1] in fixed_node_indices
            )
            selected = selected_by_query.get(node_pair, [])
            removed = removed_by_query.get(node_pair, [])
            query_record["reconstruction_selected_pairs"] = selected
            query_record["selected_global_combination_rank"] = rank
            query_record["four_membered_ring_pruning"] = {
                "removed_connection_count": len(removed),
                "removed_connections": removed,
            }
            query_record["selection_adjusted"] = (
                False
                if is_fixed_fixed
                else bool(removed) or any(
                    int(record.get("candidate_rank", 1)) != 1
                    for record in selected
                )
            )
            if not is_fixed_fixed:
                # A pruned covalent query is valid and must remain an empty selection.
                selected_pairs = [
                    {
                        "candidate_rank": int(record.get("candidate_rank", 1)),
                        "atom_a_id": int(record["atom_a_id"]),
                        "atom_b_id": int(record["atom_b_id"]),
                        "pair_probability": float(record["pair_probability"]),
                    }
                    for record in selected
                ]
                query_record["predicted_count"] = len(selected_pairs)
                for key in ("selected_pairs", "predicted_pairs"):
                    if key in query_record:
                        query_record[key] = [dict(record) for record in selected_pairs]
                status = (
                    "pruned_new_four_membered_ring" if removed
                    else "ok" if query_record["geometry_valid"] else "ok_geometry_fallback"
                )
                for key in ("status", "prediction_status"):
                    if key in query_record:
                        query_record[key] = status
                if "global_combination_rank" in query_record:
                    query_record["global_combination_rank"] = rank
                    query_record["global_combination_score"] = float(
                        raw_combination.get("global_score", 0.0)
                    )
                if "ground_truth_pairs" in query_record:
                    truth_pairs = query_record["ground_truth_pairs"]
                    query_record["count_correct"] = len(selected_pairs) == len(truth_pairs)
                    query_record["exact_atom_pair_set"] = {
                        (record["atom_a_id"], record["atom_b_id"])
                        for record in selected_pairs
                    } == {
                        (record["atom_a_id"], record["atom_b_id"])
                        for record in truth_pairs
                    }

        connection_status["four_membered_ring_pruning"] = ring_pruning
        summary = {
            "strategy": "global_top_k_beam_then_formal_chemistry_validation",
            "status": "success",
            "selected_global_combination_rank": rank,
            "selected_global_score": float(
                raw_combination.get("global_score", 0.0)
            ),
            "global_score_basis": raw_combination.get("global_score_basis"),
            "fixed_connection_count": len(fixed_connections),
            "predicted_connection_count": len(predicted_connections),
            "capacity_usage": raw_combination.get("capacity_usage", []),
            "failed_higher_ranked_combinations": failures,
            "validated_combination_count": rank,
            "four_membered_ring_pruning": ring_pruning,
        }
        return (
            connections,
            predicted_connections,
            molecule,
            connection_status,
            summary,
        )

    summary = {
        "strategy": "global_top_k_beam_then_formal_chemistry_validation",
        "status": "failed",
        "fixed_connection_count": len(fixed_connections),
        "offered_combination_count": len(global_combinations),
        "failed_combinations": failures,
        "failure_reason": "all global combinations failed formal reconstruction",
    }
    raise RetryableReconstructionError(
        "All capacity-feasible global atom-pair combinations failed formal "
        "hydrogen removal, symmetry, sanitization, or connectivity checks",
        stage="atom_pair_global_combination_validation",
        diagnostics=summary,
    )


def reconstruct_stage1_result(
    stage1_result: Mapping[str, Any],
    *,
    condition_pt_path: str | Path,
    receptor_pdb_path: str | Path,
    output_dir: str | Path,
    custom_vocabulary: FragmentVocabulary,
    atom_pair_runtime: AtomPairInferenceRuntime,
    fixed_preflight_completed: bool = False,
    min_cosine_similarity: float | None = (
        FRAGMENT_MATCH_MIN_COSINE_SIMILARITY
    ),
    diagnostic_top_k: int = FRAGMENT_MATCH_DIAGNOSTIC_TOP_K,
    enable_hac_ring_constraints: bool = (
        ENABLE_FRAGMENT_MATCH_HAC_RING_CONSTRAINTS
    ),
    hac_tolerance: int = FRAGMENT_MATCH_HAC_TOLERANCE,
    ring_tolerance: int = FRAGMENT_MATCH_RING_TOLERANCE,
    max_constraint_candidates: int = (
        FRAGMENT_MATCH_MAX_CONSTRAINT_CANDIDATES
    ),
) -> dict[str, Any]:
    """Reconstruct with condition fixed-fixed and predicted remaining bonds."""

    if not isinstance(stage1_result, Mapping):
        raise TypeError("stage1_result must be a mapping")
    if not isinstance(enable_hac_ring_constraints, bool):
        raise TypeError("enable_hac_ring_constraints must be a bool")
    if diagnostic_top_k <= 0:
        raise ValueError("diagnostic_top_k must be positive")
    if hac_tolerance < 0 or ring_tolerance < 0:
        raise ValueError("HAC and ring tolerances cannot be negative")
    if max_constraint_candidates <= 0:
        raise ValueError("max_constraint_candidates must be positive")
    condition_path = Path(condition_pt_path).expanduser().resolve()
    if not condition_path.is_file():
        raise FileNotFoundError(f"condition .pt not found: {condition_path}")
    receptor_path = Path(receptor_pdb_path).expanduser().resolve()
    if not receptor_path.is_file():
        raise FileNotFoundError(f"normalized receptor not found: {receptor_path}")
    target_dir = Path(output_dir).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    if not fixed_preflight_completed:
        preflight_fixed_condition_fragments(
            condition_path,
            atom_pair_runtime=atom_pair_runtime,
        )
    condition_data = torch_load_compat(condition_path, map_location="cpu")
    (
        fixed_node_indices,
        condition_fixed_node_pairs,
        condition_fixed_connections,
    ) = _condition_fixed_connections(
        condition_data,
        source_path=str(condition_path),
    )

    try:
        (
            reconstruction_stage1_result,
            ligand_component_selection,
        ) = _select_stage1_reconstruction_component(
            stage1_result,
            condition_data,
            fixed_node_indices,
            condition_fixed_node_pairs,
        )
    except RetryableReconstructionError:
        raise
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise RetryableReconstructionError(
            f"Stage-one ligand component selection failed: {exc}",
            stage="stage1_ligand_component_selection",
            diagnostics={"error": f"{type(exc).__name__}: {exc}"},
        ) from exc

    resolved_smiles, resolution_records = _resolve_stage1_ligand_smiles(
        reconstruction_stage1_result,
        condition_data,
        custom_vocabulary,
        min_cosine_similarity=min_cosine_similarity,
        diagnostic_top_k=diagnostic_top_k,
        enable_hac_ring_constraints=enable_hac_ring_constraints,
        hac_tolerance=hac_tolerance,
        ring_tolerance=ring_tolerance,
        max_constraint_candidates=max_constraint_candidates,
    )
    try:
        (
            atom_pair_data,
            stage1_predicted_pairs,
            atom_pair_query_pairs,
        ) = _build_stage1_atom_pair_data(
            reconstruction_stage1_result,
            condition_data,
            resolved_smiles,
            condition_fixed_node_pairs,
            source_path=str(condition_path),
        )
    except FatalConditionError:
        raise
    except (ValueError, RuntimeError) as exc:
        raise RetryableReconstructionError(
            f"Stage-one covalent graph could not be reconstructed: {exc}",
            stage="atom_pair_graph_build",
        ) from exc
    metadata = _stage1_mapping(reconstruction_stage1_result, "metadata")
    sample_id = str(metadata.get("sample_id", condition_path.stem))
    ignored_stage1_fixed_pairs = tuple(
        pair
        for pair in stage1_predicted_pairs
        if pair[0] in fixed_node_indices and pair[1] in fixed_node_indices
    )
    try:
        validated = validate_graph_contract(
            atom_pair_data, require_targets=False
        )
        (
            base_molecule,
            node_atom_map,
            node_symmetry,
            fragment_records,
            decoded_fragments,
        ) = _decode_disconnected_ligand(
            validated,
            candidate_cache=atom_pair_runtime.candidate_cache,
            template_cache=atom_pair_runtime.template_cache,
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        raise RetryableReconstructionError(
            f"Generated fragment decoding failed: {exc}",
            stage="generated_fragment_decoder",
        ) from exc
    try:
        (
            base_molecule,
            fragment_records,
            decoded_fragments,
            pre_connection_tgd,
        ) = _apply_preconnection_fragment_tgd(
            atom_pair_data,
            base_molecule,
            fragment_records,
            decoded_fragments,
            fixed_node_indices,
            receptor_path,
        )
    except (AttributeError, IndexError, TypeError, ValueError, RuntimeError) as exc:
        raise RetryableReconstructionError(
            f"Pre-connection fragment TGD synchronization failed: {exc}",
            stage="pre_connection_fragment_tgd",
        ) from exc
    if not bool(pre_connection_tgd.get("accepted")):
        _warn_preconnection_tgd_fallback(pre_connection_tgd)

    if atom_pair_query_pairs:
        try:
            prediction_payload = predict_data_object(
                _prediction_view_without_atom_labels(
                    atom_pair_data,
                    source_path=str(condition_path),
                ),
                source_path=str(condition_path),
                builder=atom_pair_runtime.builder,
                model=atom_pair_runtime.model,
                device=atom_pair_runtime.device,
                include_ranked_pairs=True,
                fixed_connections=condition_fixed_connections,
            )
            raw_predictions = prediction_payload.get("predictions")
            if not isinstance(raw_predictions, Sequence):
                raise TypeError("atom-pair prediction payload is invalid")
            query_records, local_selected_connections = _prediction_connections(
                raw_predictions
            )
            raw_global_combinations = prediction_payload.get(
                "global_combinations"
            )
            if not isinstance(raw_global_combinations, Sequence):
                raise TypeError("global atom-pair combinations are unavailable")
            global_combinations = list(raw_global_combinations)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise RetryableReconstructionError(
                "Atom-pair prediction failed.",
                stage="atom_pair_prediction",
                diagnostics={
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            ) from exc
    else:
        prediction_payload = {
            "source_file": str(condition_path),
            "sample_id": sample_id,
            "query_count": 0,
            "predictions": [],
            "global_search": {
                "status": "success",
                "legal_combination_count": 1,
            },
            "global_combinations": [
                {
                    "global_combination_rank": 1,
                    "global_score": 0.0,
                    "predicted_connections": [],
                    "fixed_connections": [],
                    "connections": [],
                    "capacity_usage": [],
                    "overflow_count": 0,
                }
            ],
        }
        query_records = []
        local_selected_connections = []
        global_combinations = list(prediction_payload["global_combinations"])

    ignored_fixed_predictions = [
        dict(connection)
        for connection in local_selected_connections
        if int(connection["node_a"]) in fixed_node_indices
        and int(connection["node_b"]) in fixed_node_indices
    ]

    for query_record in query_records:
        node_a = int(query_record["node_a"])
        node_b = int(query_record["node_b"])
        is_fixed_fixed = (
            node_a in fixed_node_indices and node_b in fixed_node_indices
        )
        query_record["is_fixed_fixed"] = is_fixed_fixed
        query_record["reconstruction_policy"] = (
            "condition_pt_fixed_fixed_ground_truth"
            if is_fixed_fixed
            else "atom_pair_model_global_combination"
        )
        query_record["covalent_query_source"] = (
            "condition_pt_atom_matching_relation"
            if is_fixed_fixed
            else "stage1_covalent_prediction"
        )

    try:
        (
            reconstruction_connections,
            predicted_connections,
            connected_molecule,
            connection_status,
            atom_pair_selection,
        ) = _select_global_reconstruction_combination(
            base_molecule,
            node_atom_map,
            node_symmetry,
            global_combinations,
            query_records,
            fixed_node_indices,
            condition_fixed_connections,
            source_file=str(condition_path),
            sample_id=sample_id,
            checkpoint_path=str(atom_pair_runtime.checkpoint_path),
        )
    except (FatalConditionError, RetryableReconstructionError):
        raise
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise RetryableReconstructionError(
            f"Atom-pair constrained selection failed: {exc}",
            stage="atom_pair_connection_selection",
        ) from exc

    applied_connection_node_pairs = tuple(
        sorted(
            {
                tuple(
                    sorted(
                        (
                            int(connection["node_a"]),
                            int(connection["node_b"]),
                        )
                    )
                )
                for connection in reconstruction_connections
            }
        )
    )
    resolution_by_graph_index = {
        int(record["graph_index"]): record for record in resolution_records
    }
    enriched_fragment_records: list[dict[str, Any]] = []
    for record in fragment_records:
        graph_index = int(record["node_index"])
        resolution = resolution_by_graph_index[graph_index]
        enriched_fragment_records.append(
            {
                **record,
                "graph_index": graph_index,
                "is_fixed": bool(resolution["is_fixed"]),
                "is_generated": bool(resolution["is_generated"]),
                "smiles_source": resolution["smiles_source"],
                "resolved_smiles": resolution["resolved_smiles"],
                "cosine_similarity": resolution["cosine_similarity"],
                "vocab_id": resolution["vocab_id"],
                "selected_rank": resolution["selected_rank"],
                "top_matches": resolution["top_matches"],
                "candidate_checks": resolution["candidate_checks"],
                "match_constraints": resolution["match_constraints"],
            }
        )

    unconnected_molecule = Chem.Mol(base_molecule)
    unconnected_molecule.SetProp(
        "_Name",
        f"{sample_id}_decoded_unconnected",
    )
    unconnected_molecule.SetProp(
        "FragmentCount",
        str(len(enriched_fragment_records)),
    )
    unconnected_molecule.SetProp("InterFragmentBondCount", "0")
    unconnected_molecule.SetProp(
        "FragmentMetadataJSON",
        json.dumps(
            enriched_fragment_records,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        ),
    )
    unconnected_path = _write_single_sdf(
        unconnected_molecule,
        target_dir / "fragments_unconnected.sdf",
        overwrite=True,
    )
    complex_path = target_dir / "complex_final.pdb"
    _, optimization = optimize_ligand_with_fixed_receptor(
        connected_molecule,
        receptor_pdb_path=receptor_path,
        output_pdb_path=complex_path,
        relaxed_ligand_sdf_path=(
            target_dir / "ligand_relaxed.sdf"
        ),
    )
    if optimization.get("warning"):
        print(f"      警告: {optimization['warning']}")

    report = {
        "status": "complete",
        "sample_id": sample_id,
        "condition_pt": str(condition_path),
        "receptor_pdb": str(receptor_path),
        "custom_fragment_vocabulary": str(custom_vocabulary.source_path),
        "atom_pair_checkpoint": str(atom_pair_runtime.checkpoint_path),
        "atom_pair_checkpoint_epoch": atom_pair_runtime.checkpoint_epoch,
        "atom_pair_vocabulary_fingerprints": dict(
            atom_pair_runtime.vocabulary_fingerprints
        ),
        "fragment_match_policy": {
            "minimum_cosine_similarity": min_cosine_similarity,
            "diagnostic_top_k": int(diagnostic_top_k),
            "hac_ring_constraints_enabled": bool(
                enable_hac_ring_constraints
            ),
            "hac_tolerance": int(hac_tolerance),
            "ring_tolerance": int(ring_tolerance),
            "max_constraint_candidates": int(max_constraint_candidates),
        },
        "pre_connection_tgd": pre_connection_tgd,
        "coordinate_contract": {
            "condition_ref_coords": "relative_to_node_pos",
            "stage1_ref_coords": "relative_to_ligand_node_pos",
            "atom_pair_ref_coords": "pos_plus_relative_frame",
            "decoder_reference_frame": "relative_to_ligand_node_pos",
            "atom_pair_inference_order": (
                "decode_disconnected_then_tgd_then_atom_pair_prediction"
            ),
            "fixed_ligand_geometry_source": "condition_pt",
            "generated_ligand_geometry_source": "stage1_prediction",
        },
        "ligand_fragment_count": len(enriched_fragment_records),
        "ligand_component_selection": ligand_component_selection,
        "stage1_predicted_covalent_edge_count": len(
            stage1_predicted_pairs
        ),
        "covalent_fragment_query_count": len(atom_pair_query_pairs),
        "connection_policy": {
            "fixed_fixed_source": (
                "condition_pt.connection_edge_index+connection_atom_ids"
            ),
            "fixed_fixed_covalent_edge_derivation": (
                "condition atom-matching relation; independent of graph "
                "edge_index LL edges"
            ),
            "other_connection_source": "atom_pair_model_global_combination",
            "new_four_membered_ring_policy": (
                "keep_fixed_then_descending_pair_probability"
            ),
            "removed_four_membered_ring_connection_count": atom_pair_selection[
                "four_membered_ring_pruning"
            ]["removed_connection_count"],
            "inter_fragment_bond_type": "SINGLE",
            "symmetry_resolution": (
                "whole_fragment_automorphism_minimum_total_bond_length"
            ),
            "fixed_node_indices": sorted(fixed_node_indices),
            "stage1_predicted_covalent_node_pairs": [
                list(pair) for pair in stage1_predicted_pairs
            ],
            "ignored_stage1_predicted_fixed_fixed_node_pairs": [
                list(pair) for pair in ignored_stage1_fixed_pairs
            ],
            "condition_derived_fixed_fixed_covalent_node_pairs": [
                list(pair) for pair in condition_fixed_node_pairs
            ],
            "atom_pair_query_node_pairs": [
                list(pair) for pair in atom_pair_query_pairs
            ],
            "applied_atom_connection_node_pairs": [
                list(pair) for pair in applied_connection_node_pairs
            ],
            "ignored_stage1_predicted_fixed_fixed_edge_count": len(
                ignored_stage1_fixed_pairs
            ),
            "condition_fixed_fixed_covalent_edge_count": len(
                condition_fixed_node_pairs
            ),
            "model_predicted_connection_count": len(
                predicted_connections
            ),
            "ignored_model_fixed_fixed_connection_count": len(
                ignored_fixed_predictions
            ),
            "condition_fixed_fixed_connection_count": len(
                condition_fixed_connections
            ),
            "applied_connection_count": len(reconstruction_connections),
            "ignored_model_fixed_fixed_connections": (
                ignored_fixed_predictions
            ),
            "condition_fixed_fixed_connections": (
                condition_fixed_connections
            ),
        },
        "fragment_smiles_resolution": resolution_records,
        "atom_pair_prediction": prediction_payload,
        "atom_pair_queries": query_records,
        "atom_pair_selection": atom_pair_selection,
        "connection_status": connection_status,
        "fragments": enriched_fragment_records,
        "complex_optimization": optimization,
        "outputs": {
            "unconnected_fragments_sdf": str(unconnected_path),
            "relaxed_ligand_sdf": optimization.get(
                "relaxed_ligand_sdf"
            ),
            "complex_pdb": str(complex_path.resolve()),
        },
    }
    return report


def _process_graph(
    path: Path,
    *,
    sample_number: int,
    output_dir: Path,
    receptor_pdb_path: str | Path,
    builder: ConnectionFeatureBuilder,
    candidate_cache: FragmentCandidateCache,
    template_cache: _FragmentTemplateCache,
    model: PhiSLinkerAtomPairModel,
    device: torch.device,
    checkpoint_path: str,
) -> dict[str, Any]:
    source_path = str(path.expanduser().resolve())
    data = torch_load_compat(path, map_location="cpu")

    validated_geometry = validate_graph_contract(data, require_targets=False)
    (
        base_molecule,
        node_atom_map,
        node_symmetry,
        fragment_records,
        decoded_fragments,
    ) = _decode_disconnected_ligand(
        validated_geometry,
        candidate_cache=candidate_cache,
        template_cache=template_cache,
    )
    (
        base_molecule,
        fragment_records,
        decoded_fragments,
        pre_connection_tgd,
    ) = _apply_preconnection_fragment_tgd(
        data,
        base_molecule,
        fragment_records,
        decoded_fragments,
        set(),
        receptor_pdb_path,
    )
    if not bool(pre_connection_tgd.get("accepted")):
        _warn_preconnection_tgd_fallback(pre_connection_tgd)

    # The model receives a separate object that has no connection_atom_ids.
    prediction_view = _prediction_view_without_atom_labels(
        data,
        source_path=source_path,
    )
    prediction_payload = predict_data_object(
        prediction_view,
        source_path=source_path,
        builder=builder,
        model=model,
        device=device,
    )
    predictions = prediction_payload["predictions"]
    if not isinstance(predictions, Sequence):
        raise TypeError("prediction payload predictions must be a sequence")
    global_combinations = prediction_payload.get("global_combinations")
    if not isinstance(global_combinations, Sequence):
        raise TypeError("prediction payload global_combinations must be a sequence")

    # Labels are read only after inference, for the comparison reconstruction.
    validated = validate_graph_contract(data, require_targets=True)
    truth_groups = group_connections(validated, require_targets=True)
    if not truth_groups:
        raise ValueError("graph contains no ground-truth connection queries")
    query_records, _local_predicted_connections, truth_connections = (
        _prepare_comparison_records(predictions, truth_groups)
    )
    sample_id = str(prediction_payload["sample_id"])
    (
        _reconstruction_connections,
        predicted_connections,
        predicted_molecule,
        predicted_status,
        atom_pair_selection,
    ) = _select_global_reconstruction_combination(
        base_molecule,
        node_atom_map,
        node_symmetry,
        global_combinations,
        query_records,
        set(),
        (),
        source_file=source_path,
        sample_id=sample_id,
        checkpoint_path=checkpoint_path,
    )
    truth_molecule, truth_status = _connect_fragments(
        base_molecule,
        node_atom_map,
        node_symmetry,
        truth_connections,
        mode="ground_truth",
        source_file=source_path,
        sample_id=sample_id,
        checkpoint_path=checkpoint_path,
    )
    _annotate_symmetry_resolved_queries(
        query_records,
        predicted_status["connections"],
        truth_status["connections"],
    )

    safe_sample_id = _safe_name(sample_id, path.stem)
    prefix = f"{sample_number:02d}_{safe_sample_id}"
    predicted_path = _write_single_sdf(
        predicted_molecule,
        output_dir / f"{prefix}_predicted.sdf",
    )
    truth_path = _write_single_sdf(
        truth_molecule,
        output_dir / f"{prefix}_ground_truth.sdf",
    )

    exact_queries = sum(
        int(record["exact_atom_pair_set"]) for record in query_records
    )
    symmetry_exact_queries = sum(
        int(record["symmetry_aware_exact_atom_pair_set"])
        for record in query_records
    )
    return {
        "source_file": source_path,
        "sample_id": sample_id,
        "ligand_fragment_count": len(fragment_records),
        "query_count": len(query_records),
        "exact_query_count": exact_queries,
        "end_to_end_exact": exact_queries / len(query_records),
        "all_queries_exact": exact_queries == len(query_records),
        "symmetry_aware_exact_query_count": symmetry_exact_queries,
        "symmetry_aware_end_to_end_exact": (
            symmetry_exact_queries / len(query_records)
        ),
        "all_queries_symmetry_aware_exact": (
            symmetry_exact_queries == len(query_records)
        ),
        "predicted_sdf": str(predicted_path),
        "ground_truth_sdf": str(truth_path),
        "predicted_reconstruction": predicted_status,
        "atom_pair_selection": atom_pair_selection,
        "pre_connection_tgd": pre_connection_tgd,
        "ground_truth_reconstruction": truth_status,
        "fragments": fragment_records,
        "queries": query_records,
    }


def main() -> None:
    args = _parse_args()
    stage1_path = args.stage1_result.expanduser().resolve()
    stage1_result = torch_load_compat(stage1_path, map_location="cpu")
    if not isinstance(stage1_result, Mapping):
        raise TypeError(f"stage-one result is not a mapping: {stage1_path}")
    condition_path = args.condition.expanduser().resolve()
    receptor_path = (
        args.receptor.expanduser().resolve()
        if args.receptor is not None
        else condition_path.parent / "full_receptor_normalized.pdb"
    )
    custom_vocabulary = load_reconstruction_vocabulary(
        args.custom_fragment_vocab
    )
    runtime = load_atom_pair_inference_runtime(
        checkpoint_path=args.checkpoint,
        fragment_vocab_path=args.fragment_vocab,
        atom_vocab_path=args.atom_vocab,
        device=args.device,
    )
    report = reconstruct_stage1_result(
        stage1_result,
        condition_pt_path=condition_path,
        receptor_pdb_path=receptor_path,
        output_dir=args.output_dir,
        custom_vocabulary=custom_vocabulary,
        atom_pair_runtime=runtime,
        min_cosine_similarity=args.min_cosine_similarity,
        diagnostic_top_k=args.diagnostic_top_k,
    )
    print(f"3D reconstruction complete: {report['outputs']}")


if __name__ == "__main__":
    main()
