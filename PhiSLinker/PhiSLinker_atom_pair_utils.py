"""Shared utilities for atom-pair training and inference.

This module is the single production implementation of vocabulary lookup,
chemical candidate eligibility, reconstruction charge neutralization, and
conformer caching.  Candidate eligibility treats hydrogen substitution
separately and probes N/S atoms by adding one temporary single bond without
changing their formal charge.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import sys
import tempfile
from collections import OrderedDict
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence, TypeVar

import numpy as np
import torch
from rdkit import Chem, rdBase

try:
    from tqdm.auto import tqdm as _tqdm
except ImportError:  # Keep progress reporting available without tqdm.
    _tqdm = None

try:
    from PhiSLinker_atom_pair_smiles_decoder import (
        FragmentConformerTemplate,
        align_fragment_conformer_template,
        build_fragment_conformer_template,
        mol_from_smiles_quietly,
    )
except ImportError:  # pragma: no cover - package-style import fallback
    from .PhiSLinker_atom_pair_smiles_decoder import (
        FragmentConformerTemplate,
        align_fragment_conformer_template,
        build_fragment_conformer_template,
        mol_from_smiles_quietly,
    )


GLOBAL_SMILES_TOKEN = "<GLOBAL>"
VALENCE_PROBED_SPECIAL_ELEMENTS = frozenset({"N", "S"})
NEGATIVE_HALOGEN_ELEMENTS = frozenset({"F", "Cl", "Br", "I"})
KNOWN_NEUTRALIZED_NITROGEN_TARGET_REASON = (
    "neutral_n_valence3_without_hydrogen"
)
CHEMICALLY_INELIGIBLE_TARGET_REASON = "chemically_ineligible"

_ProgressItem = TypeVar("_ProgressItem")


def progress_iterable(
    iterable: Iterable[_ProgressItem],
    *,
    description: str,
    total: int | None = None,
    unit: str = "item",
) -> Iterator[_ProgressItem]:
    """Iterate with tqdm or a dependency-free percentage fallback."""

    if total is None:
        try:
            total = len(iterable)  # type: ignore[arg-type]
        except (TypeError, AttributeError):
            total = None
    if total is not None and total < 0:
        raise ValueError("progress total cannot be negative")

    if _tqdm is not None:
        progress = _tqdm(
            iterable,
            total=total,
            desc=description,
            unit=unit,
            dynamic_ncols=True,
            file=sys.stdout,
        )
        try:
            yield from progress
        finally:
            progress.close()
        return

    completed = 0
    last_percentage = -1

    def render() -> None:
        nonlocal last_percentage
        if total is None:
            print(
                f"\r{description}: {completed} {unit}",
                end="",
                flush=True,
            )
            return
        percentage = 100 if total == 0 else int(100 * completed / total)
        if percentage == last_percentage and completed != total:
            return
        last_percentage = percentage
        print(
            f"\r{description}: {completed}/{total} {unit} "
            f"({percentage:3d}%)",
            end="",
            flush=True,
        )

    try:
        render()
        for item in iterable:
            yield item
            completed += 1
            render()
    finally:
        print()


def normalize_smiles_value(value: Any) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    result = str(value)
    if not result:
        raise ValueError("SMILES values must be non-empty")
    return result


def file_sha256(path: str | Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file_obj:
        while True:
            block = file_obj.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def torch_load_compat(path: str | Path, map_location: str | torch.device = "cpu") -> Any:
    """Load trusted local PyG data across the PyTorch 2.6 weights-only change."""

    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # pragma: no cover - older PyTorch
        return torch.load(path, map_location=map_location)


def _json_ready(value: Any) -> Any:
    if is_dataclass(value):
        return _json_ready(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    return value


def atomic_write_json(path: str | Path, payload: Any) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as file_obj:
            json.dump(_json_ready(payload), file_obj, ensure_ascii=False, indent=2)
            file_obj.flush()
            os.fsync(file_obj.fileno())
        temporary.replace(target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return target


def atomic_torch_save(path: str | Path, payload: Any) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return target


def gaussian_rbf(
    distances: torch.Tensor,
    *,
    num_centers: int,
    max_distance: float,
) -> torch.Tensor:
    """Encode distances with fixed Gaussian radial basis functions."""

    if num_centers <= 0 or max_distance <= 0.0:
        raise ValueError("RBF configuration must be positive")
    centers = torch.linspace(
        0.0,
        float(max_distance),
        int(num_centers),
        device=distances.device,
        dtype=distances.dtype,
    )
    spacing = (
        float(max_distance) / max(int(num_centers) - 1, 1)
    )
    gamma = 1.0 / max(spacing * spacing, 1.0e-12)
    return torch.exp(-gamma * (distances.unsqueeze(-1) - centers) ** 2)


class FragmentVocabularyLookup:
    """Exact SMILES -> fixed 128-dimensional fragment vector lookup."""

    REQUIRED_FIELDS = ("smiles", "fragment_embeddings")

    @staticmethod
    def _array_sha256(values: np.ndarray, *, is_string: bool = False) -> str:
        """Return the stable semantic digest used by the standard vocabulary."""

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

    @classmethod
    def _table_sha256(cls, arrays: Mapping[str, np.ndarray]) -> str:
        digest = hashlib.sha256()
        for key in cls.REQUIRED_FIELDS:
            values = np.asarray(arrays[key])
            digest.update(key.encode("utf-8"))
            digest.update(json.dumps(list(values.shape)).encode("ascii"))
            digest.update(
                cls._array_sha256(
                    values, is_string=(key == "smiles")
                ).encode("ascii")
            )
        return digest.hexdigest()

    def __init__(self, path: str | Path, expected_dim: int = 128) -> None:
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"fragment vocabulary not found: {self.path}")
        with np.load(self.path, allow_pickle=False) as archive:
            missing = set(self.REQUIRED_FIELDS) - set(archive.files)
            if missing:
                raise KeyError(f"fragment vocabulary is missing fields: {sorted(missing)}")
            raw_smiles = np.asarray(archive["smiles"])
            embeddings = np.asarray(archive["fragment_embeddings"], dtype=np.float32)
        if raw_smiles.ndim != 1:
            raise ValueError("fragment vocabulary smiles must be one-dimensional")
        smiles = tuple(normalize_smiles_value(value) for value in raw_smiles)
        if embeddings.shape != (len(smiles), int(expected_dim)):
            raise ValueError(
                "fragment embedding shape mismatch: "
                f"expected {(len(smiles), int(expected_dim))}, got {embeddings.shape}"
            )
        if not np.all(np.isfinite(embeddings)):
            raise ValueError("fragment vocabulary contains NaN or Inf")
        if len(set(smiles)) != len(smiles):
            raise ValueError("fragment vocabulary contains duplicate SMILES")
        if GLOBAL_SMILES_TOKEN in smiles:
            raise ValueError(f"{GLOBAL_SMILES_TOKEN} must not be present in the vocabulary")
        self.expected_dim = int(expected_dim)
        self.smiles = smiles
        self.embeddings = np.ascontiguousarray(embeddings)
        self._index = {value: index for index, value in enumerate(smiles)}
        self.sha256 = file_sha256(self.path)

    def profile_for_prefix(self, num_smiles: int) -> dict[str, Any]:
        """Build a stable content profile for the first vocabulary rows."""

        if isinstance(num_smiles, bool) or not isinstance(num_smiles, int):
            raise TypeError("fragment vocabulary prefix size must be an integer")
        current_num_smiles = len(self.smiles)
        if not 1 <= num_smiles <= current_num_smiles:
            raise ValueError(
                "fragment vocabulary prefix size must be within "
                f"[1, {current_num_smiles}], got {num_smiles}"
            )
        arrays = {
            "smiles": np.asarray(self.smiles[:num_smiles], dtype=np.str_),
            "fragment_embeddings": self.embeddings[:num_smiles],
        }
        return {
            "embedding_dim": self.expected_dim,
            "num_smiles": num_smiles,
            "array_sha256": {
                key: self._array_sha256(
                    values, is_string=(key == "smiles")
                )
                for key, values in arrays.items()
            },
            "table_sha256": self._table_sha256(arrays),
        }

    def __contains__(self, smiles: str) -> bool:
        return smiles in self._index

    def vector_numpy(self, smiles: str) -> np.ndarray:
        try:
            row = self._index[smiles]
        except KeyError as exc:
            raise KeyError(f"SMILES is absent from fragment vocabulary: {smiles!r}") from exc
        return self.embeddings[row]


@dataclass(frozen=True)
class AtomVocabularyEntry:
    smiles: str
    canonical_atom_ids: np.ndarray
    embeddings: np.ndarray


class AtomVocabularyLookup:
    """Exact SMILES -> canonical-atom-ID/64D-vector ragged lookup."""

    REQUIRED_FIELDS = (
        "smiles",
        "atom_offsets",
        "canonical_atom_id",
        "atom_embeddings",
    )

    @staticmethod
    def _array_sha256(
        values: np.ndarray, *, is_string: bool = False
    ) -> str:
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

    @classmethod
    def _table_sha256(cls, arrays: Mapping[str, np.ndarray]) -> str:
        """Match ChemCut's authoritative atom-table checksum exactly."""

        digest = hashlib.sha256()
        for key in cls.REQUIRED_FIELDS:
            values = np.asarray(arrays[key])
            digest.update(key.encode("utf-8"))
            digest.update(json.dumps(list(values.shape)).encode("ascii"))
            if key == "smiles":
                for value in values.astype(str).tolist():
                    encoded = value.encode("utf-8")
                    digest.update(len(encoded).to_bytes(8, "little"))
                    digest.update(encoded)
            else:
                contiguous = np.ascontiguousarray(values)
                digest.update(contiguous.dtype.str.encode("ascii"))
                digest.update(contiguous.tobytes())
        return digest.hexdigest()

    def __init__(self, path: str | Path, expected_dim: int = 64) -> None:
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"atom vocabulary not found: {self.path}")
        with np.load(self.path, allow_pickle=False) as archive:
            missing = set(self.REQUIRED_FIELDS) - set(archive.files)
            if missing:
                raise KeyError(f"atom vocabulary is missing fields: {sorted(missing)}")
            raw_smiles = np.asarray(archive["smiles"])
            raw_offsets = np.asarray(archive["atom_offsets"])
            raw_atom_ids = np.asarray(archive["canonical_atom_id"])
            embeddings = np.asarray(archive["atom_embeddings"], dtype=np.float32)
        if not np.issubdtype(raw_offsets.dtype, np.integer):
            raise TypeError("atom_offsets must use an integer dtype")
        if not np.issubdtype(raw_atom_ids.dtype, np.integer):
            raise TypeError("canonical_atom_id must use an integer dtype")
        offsets = raw_offsets.astype(np.int64, copy=False)
        atom_ids = raw_atom_ids.astype(np.int64, copy=False)
        if raw_smiles.ndim != 1:
            raise ValueError("atom vocabulary smiles must be one-dimensional")
        smiles = tuple(normalize_smiles_value(value) for value in raw_smiles)
        if len(set(smiles)) != len(smiles):
            raise ValueError("atom vocabulary contains duplicate SMILES")
        if offsets.shape != (len(smiles) + 1,):
            raise ValueError("atom_offsets must have len(smiles) + 1 entries")
        if offsets[0] != 0 or np.any(np.diff(offsets) < 0):
            raise ValueError("atom_offsets must start at zero and be non-decreasing")
        if int(offsets[-1]) != len(atom_ids):
            raise ValueError("last atom offset does not match canonical_atom_id length")
        if embeddings.shape != (len(atom_ids), int(expected_dim)):
            raise ValueError(
                "atom embedding shape mismatch: "
                f"expected {(len(atom_ids), int(expected_dim))}, got {embeddings.shape}"
            )
        if np.any(atom_ids < 0):
            raise ValueError("canonical atom IDs cannot be negative")
        if not np.all(np.isfinite(embeddings)):
            raise ValueError("atom vocabulary contains NaN or Inf")
        self.expected_dim = int(expected_dim)
        self.smiles = smiles
        self.atom_offsets = offsets
        self.canonical_atom_ids = atom_ids
        self.embeddings = np.ascontiguousarray(embeddings)
        self._index = {value: index for index, value in enumerate(smiles)}
        self.sha256 = file_sha256(self.path)

    def profile_for_prefix(self, num_smiles: int) -> dict[str, Any]:
        """构建前 N 个片段及其全部 ragged 原子行的稳定画像。"""
        if isinstance(num_smiles, bool) or not isinstance(num_smiles, int):
            raise TypeError("atom vocabulary prefix size must be an integer")
        current_num_smiles = len(self.smiles)
        if not 1 <= num_smiles <= current_num_smiles:
            raise ValueError(
                "atom vocabulary prefix size must be within "
                f"[1, {current_num_smiles}], got {num_smiles}"
            )
        num_atoms = int(self.atom_offsets[num_smiles])
        arrays = {
            "smiles": np.asarray(
                self.smiles[:num_smiles], dtype=np.str_
            ),
            "atom_offsets": np.asarray(
                self.atom_offsets[:num_smiles + 1], dtype=np.int64
            ),
            "canonical_atom_id": np.asarray(
                self.canonical_atom_ids[:num_atoms], dtype=np.int32
            ),
            "atom_embeddings": np.asarray(
                self.embeddings[:num_atoms], dtype=np.float32
            ),
        }
        return {
            "embedding_dim": self.expected_dim,
            "num_smiles": num_smiles,
            "num_atoms": num_atoms,
            "array_sha256": {
                key: self._array_sha256(
                    values, is_string=(key == "smiles")
                )
                for key, values in arrays.items()
            },
            "table_sha256": self._table_sha256(arrays),
        }

    def lookup(self, smiles: str) -> AtomVocabularyEntry:
        try:
            index = self._index[smiles]
        except KeyError as exc:
            raise KeyError(f"SMILES is absent from atom vocabulary: {smiles!r}") from exc
        start = int(self.atom_offsets[index])
        end = int(self.atom_offsets[index + 1])
        return AtomVocabularyEntry(
            smiles=smiles,
            canonical_atom_ids=self.canonical_atom_ids[start:end],
            embeddings=self.embeddings[start:end],
        )


@dataclass(frozen=True)
class FragmentCandidateEntry:
    input_smiles: str
    canonical_smiles: str
    eligible_atom_ids: tuple[int, ...]
    atom_capacities: Mapping[int, int]
    atom_embeddings: Mapping[int, np.ndarray]
    ignored_vocab_atom_ids: tuple[int, ...]
    ineligible_atom_reasons: Mapping[int, str]

    def embedding(self, canonical_atom_id: int) -> np.ndarray:
        try:
            return self.atom_embeddings[int(canonical_atom_id)]
        except KeyError as exc:
            raise KeyError(
                f"canonical atom ID {canonical_atom_id} has no embedding for "
                f"{self.input_smiles!r}"
            ) from exc

    def ineligible_reason(self, canonical_atom_id: int) -> str | None:
        return self.ineligible_atom_reasons.get(int(canonical_atom_id))

    def capacity(self, canonical_atom_id: int) -> int:
        try:
            return int(self.atom_capacities[int(canonical_atom_id)])
        except KeyError as exc:
            raise KeyError(
                f"canonical atom ID {canonical_atom_id} has no connection "
                f"capacity for {self.input_smiles!r}"
            ) from exc


def atom_has_consumable_h(atom: Chem.Atom) -> bool:
    """Return the exact replaceable-H predicate agreed for this project."""

    return (
        int(atom.GetNumExplicitHs()) > 0
        or int(atom.GetNumImplicitHs()) > 0
        or any(neighbour.GetAtomicNum() == 1 for neighbour in atom.GetNeighbors())
    )


def atom_is_known_neutralized_nitrogen_target(atom: Chem.Atom) -> bool:
    """Identify the known cut-data error that must not become a target.

    These atoms are neutral, hydrogen-free nitrogens whose explicit valence is
    already three.  Reattaching the cut single bond would require restoring a
    positive charge, which the source fragment no longer contains.
    """

    return (
        int(atom.GetAtomicNum()) == 7
        and int(atom.GetFormalCharge()) == 0
        and not atom_has_consumable_h(atom)
        and int(
            atom.GetValence(which=Chem.ValenceType.EXPLICIT)
        )
        == 3
    )


def atom_accepts_single_bond_without_charge_change(atom: Chem.Atom) -> bool:
    """Return whether RDKit accepts one more single bond at this atom.

    The probe deliberately preserves the atom's formal charge.  Consequently,
    a neutral trivalent nitrogen is not treated as a latent quaternary
    ammonium centre, while a three-coordinate positively charged nitrogen can
    accept the fourth bond.
    """

    molecule = atom.GetOwningMol()
    atom_index = int(atom.GetIdx())
    editable = Chem.RWMol(Chem.Mol(molecule))
    dummy_atom = Chem.Atom(0)
    dummy_atom.SetNoImplicit(True)
    dummy_index = int(editable.AddAtom(dummy_atom))
    editable.AddBond(atom_index, dummy_index, Chem.BondType.SINGLE)
    log_blocker = rdBase.BlockLogs()
    try:
        sanitize_result = Chem.SanitizeMol(editable, catchErrors=True)
    except Exception:
        return False
    finally:
        del log_blocker
    return sanitize_result == Chem.SanitizeFlags.SANITIZE_NONE


def consume_connection_hydrogen(atom: Chem.Atom) -> str | None:
    """Consume one atom-level hydrogen before forming a replacement bond.

    Returned values are diagnostic labels.  Reconstruction molecules must not
    contain hydrogen atoms; explicit and implicit values here are atom-level
    hydrogen counts on the selected heavy atom.
    """

    explicit_hydrogens = int(atom.GetNumExplicitHs())
    if explicit_hydrogens > 0:
        atom.SetNumExplicitHs(explicit_hydrogens - 1)
        return "explicit_h_count"

    implicit_hydrogens = int(atom.GetNumImplicitHs())
    if implicit_hydrogens > 0:
        # Freeze the remaining hydrogen count so the new bond replaces exactly
        # one implicit hydrogen instead of relying on a stale property cache.
        atom.SetNoImplicit(True)
        atom.SetNumExplicitHs(implicit_hydrogens - 1)
        return "implicit_h_count"

    if any(neighbour.GetAtomicNum() == 1 for neighbour in atom.GetNeighbors()):
        raise RuntimeError(
            "connection atom has an explicit hydrogen neighbour; expected a "
            "heavy-atom-only reconstruction molecule"
        )
    return None


def _molecule_total_formal_charge(molecule: Chem.Mol) -> int:
    return sum(int(atom.GetFormalCharge()) for atom in molecule.GetAtoms())


def neutralize_reconstructed_molecule(
    molecule: Chem.Mol,
) -> tuple[Chem.Mol, dict[str, Any]]:
    """Return a sanitized copy whose net formal charge is zero.

    Only the excess sign of the molecule's *net* charge is changed.  A molecule
    whose total formal charge is already zero is returned without changing any
    atom-level charges, so charge-separated groups remain intact.  When a net
    charge must be removed, sites farthest from an opposite formal charge are
    tried first; this preserves local charge-balanced motifs without relying on
    a functional-group whitelist.

    Net-charged inputs are converted to a hydrogen-collapsed non-stereo form
    before neutralization.  Net-neutral charge-separated groups are returned
    unchanged.  Each accepted step represents one proton transfer: a
    negatively charged atom gains one atom-level hydrogen while its formal
    charge increases by one, or a positively charged atom loses one atom-level
    hydrogen while its formal charge decreases by one.  The heavy-atom graph
    and coordinates are not changed.
    """

    if molecule is None:
        raise ValueError("molecule must not be None")
    if molecule.GetNumAtoms() == 0:
        raise ValueError("molecule must contain at least one atom")

    output = Chem.Mol(molecule)
    initial_total_charge = _molecule_total_formal_charge(output)
    if initial_total_charge != 0:
        Chem.RemoveStereochemistry(output)
        remove_hydrogen_parameters = Chem.RemoveHsParameters()
        remove_hydrogen_parameters.removeNontetrahedralNeighbors = True
        log_blocker = rdBase.BlockLogs()
        try:
            output = Chem.RemoveHs(
                output,
                remove_hydrogen_parameters,
                sanitize=True,
            )
        except Exception as exc:
            raise ValueError(
                "failed to prepare a non-stereo molecule for charge "
                f"neutralization: {type(exc).__name__}: {exc}"
            ) from exc
        finally:
            del log_blocker

    log_blocker = rdBase.BlockLogs()
    try:
        initial_sanitize = Chem.SanitizeMol(output, catchErrors=True)
    finally:
        del log_blocker
    if initial_sanitize != Chem.SanitizeFlags.SANITIZE_NONE:
        raise ValueError(
            "charge neutralization requires a sanitized input molecule; "
            f"sanitize_status={initial_sanitize}, code={int(initial_sanitize)}"
        )

    current_total_charge = initial_total_charge
    steps: list[dict[str, Any]] = []

    for step_index in range(abs(initial_total_charge)):
        if current_total_charge == 0:
            break
        neutralizing_sign = 1 if current_total_charge > 0 else -1
        opposite_indices = [
            int(atom.GetIdx())
            for atom in output.GetAtoms()
            if int(atom.GetFormalCharge()) * neutralizing_sign < 0
        ]
        distance_matrix = Chem.GetDistanceMatrix(output)
        candidates: list[dict[str, Any]] = []
        unavailable_positive_atoms: list[dict[str, Any]] = []
        for atom in output.GetAtoms():
            formal_charge = int(atom.GetFormalCharge())
            if formal_charge * neutralizing_sign <= 0:
                continue
            atom_index = int(atom.GetIdx())
            atom_level_hydrogens = (
                int(atom.GetNumExplicitHs()) + int(atom.GetNumImplicitHs())
            )
            explicit_hydrogen_neighbours = sum(
                neighbour.GetAtomicNum() == 1 for neighbour in atom.GetNeighbors()
            )
            if neutralizing_sign > 0 and atom_level_hydrogens <= 0:
                unavailable_positive_atoms.append(
                    {
                        "atom_index": atom_index,
                        "symbol": atom.GetSymbol(),
                        "formal_charge": formal_charge,
                        "explicit_hydrogen_neighbour_count": int(
                            explicit_hydrogen_neighbours
                        ),
                    }
                )
                continue
            nearest_opposite_distance = (
                min(
                    float(distance_matrix[atom_index, opposite_index])
                    for opposite_index in opposite_indices
                )
                if opposite_indices
                else None
            )
            candidates.append(
                {
                    "atom_index": atom_index,
                    "symbol": atom.GetSymbol(),
                    "formal_charge": formal_charge,
                    "atom_level_hydrogen_count": atom_level_hydrogens,
                    "explicit_hydrogen_neighbour_count": int(
                        explicit_hydrogen_neighbours
                    ),
                    "nearest_opposite_charge_distance_bonds": (
                        nearest_opposite_distance
                    ),
                }
            )

        candidates.sort(
            key=lambda item: (
                0
                if item["nearest_opposite_charge_distance_bonds"] is None
                else 1,
                -float(item["nearest_opposite_charge_distance_bonds"] or 0.0),
                int(item["atom_index"]),
            )
        )
        rejected_candidates: list[dict[str, Any]] = []
        accepted = False
        for candidate in candidates:
            atom_index = int(candidate["atom_index"])
            trial = Chem.Mol(output)
            trial_atom = trial.GetAtomWithIdx(atom_index)
            formal_charge_before = int(trial_atom.GetFormalCharge())
            atom_level_hydrogens_before = (
                int(trial_atom.GetNumExplicitHs())
                + int(trial_atom.GetNumImplicitHs())
            )
            if neutralizing_sign > 0:
                formal_charge_after = formal_charge_before - 1
                atom_level_hydrogens_after = atom_level_hydrogens_before - 1
                action = "remove_proton_from_positive_atom"
            else:
                formal_charge_after = formal_charge_before + 1
                atom_level_hydrogens_after = atom_level_hydrogens_before + 1
                action = "add_proton_to_negative_atom"

            trial_atom.SetFormalCharge(formal_charge_after)
            trial_atom.SetNoImplicit(True)
            trial_atom.SetNumExplicitHs(atom_level_hydrogens_after)
            trial_atom.UpdatePropertyCache(strict=False)
            log_blocker = rdBase.BlockLogs()
            try:
                try:
                    sanitize_result = Chem.SanitizeMol(
                        trial,
                        catchErrors=True,
                    )
                    sanitize_error = None
                except Exception as exc:
                    sanitize_result = None
                    sanitize_error = f"{type(exc).__name__}: {exc}"
            finally:
                del log_blocker
            if sanitize_result != Chem.SanitizeFlags.SANITIZE_NONE:
                rejected_candidates.append(
                    {
                        **candidate,
                        "sanitize_status": (
                            str(sanitize_result)
                            if sanitize_result is not None
                            else "exception"
                        ),
                        "sanitize_code": (
                            int(sanitize_result)
                            if sanitize_result is not None
                            else None
                        ),
                        "error": sanitize_error,
                    }
                )
                continue

            total_charge_after = _molecule_total_formal_charge(trial)
            expected_total_charge = (
                current_total_charge - neutralizing_sign
            )
            if total_charge_after != expected_total_charge:
                raise RuntimeError(
                    "one proton-transfer step changed the total formal charge "
                    f"unexpectedly: expected {expected_total_charge}, got "
                    f"{total_charge_after}"
                )
            output = trial
            steps.append(
                {
                    "step": step_index + 1,
                    "action": action,
                    "atom_index": atom_index,
                    "symbol": candidate["symbol"],
                    "formal_charge_before": formal_charge_before,
                    "formal_charge_after": formal_charge_after,
                    "atom_level_hydrogen_count_before": (
                        atom_level_hydrogens_before
                    ),
                    "atom_level_hydrogen_count_after": (
                        atom_level_hydrogens_after
                    ),
                    "nearest_opposite_charge_distance_bonds": candidate[
                        "nearest_opposite_charge_distance_bonds"
                    ],
                    "total_formal_charge_before": current_total_charge,
                    "total_formal_charge_after": total_charge_after,
                    "rejected_higher_priority_candidates": rejected_candidates,
                }
            )
            current_total_charge = total_charge_after
            accepted = True
            break

        if not accepted:
            direction = (
                "remove a proton from a positively charged atom"
                if neutralizing_sign > 0
                else "add a proton to a negatively charged atom"
            )
            unavailable_summary = (
                f"; positive_atoms_without_atom_level_hydrogen="
                f"{unavailable_positive_atoms}"
                if unavailable_positive_atoms
                else ""
            )
            raise ValueError(
                "could not neutralize reconstructed molecule: unable to "
                f"{direction} while preserving a sanitized heavy-atom graph; "
                f"current_total_formal_charge={current_total_charge}; "
                f"rejected_candidates={rejected_candidates}"
                f"{unavailable_summary}"
            )

    final_total_charge = _molecule_total_formal_charge(output)
    if final_total_charge != 0:
        raise RuntimeError(
            "charge neutralization ended with a nonzero total formal charge: "
            f"{final_total_charge}"
        )
    remaining_charged_atoms = [
        {
            "atom_index": int(atom.GetIdx()),
            "symbol": atom.GetSymbol(),
            "formal_charge": int(atom.GetFormalCharge()),
        }
        for atom in output.GetAtoms()
        if int(atom.GetFormalCharge()) != 0
    ]
    metadata = {
        "status": "neutralized" if steps else "not_required_already_neutral",
        "method": "net_charge_proton_transfer",
        "site_priority": (
            "farthest_from_opposite_formal_charge_then_atom_index"
        ),
        "initial_total_formal_charge": initial_total_charge,
        "final_total_formal_charge": final_total_charge,
        "added_hydrogen_count": sum(
            step["action"] == "add_proton_to_negative_atom" for step in steps
        ),
        "removed_hydrogen_count": sum(
            step["action"] == "remove_proton_from_positive_atom"
            for step in steps
        ),
        "remaining_charge_separated_atom_count": len(remaining_charged_atoms),
        "remaining_charge_separated_atoms": remaining_charged_atoms,
        "steps": steps,
    }
    return output, metadata


def atom_chemistry_diagnostics(atom: Chem.Atom) -> dict[str, Any]:
    """Return JSON-safe chemistry state for one RDKit atom.

    This is intentionally diagnostic-only: it does not infer capacity or
    modify the molecule.  Keeping the raw RDKit charge, valence and hydrogen
    views together makes disagreements in the capacity probe inspectable.
    """

    def valence(which: Chem.ValenceType) -> int | str:
        try:
            return int(atom.GetValence(which=which))
        except Exception as exc:  # Invalid tentative molecules are diagnostic.
            return f"{type(exc).__name__}: {exc}"

    neighbours: list[dict[str, Any]] = []
    for neighbour in atom.GetNeighbors():
        bond = atom.GetOwningMol().GetBondBetweenAtoms(
            int(atom.GetIdx()), int(neighbour.GetIdx())
        )
        neighbours.append(
            {
                "atom_index": int(neighbour.GetIdx()),
                "symbol": neighbour.GetSymbol(),
                "atomic_number": int(neighbour.GetAtomicNum()),
                "formal_charge": int(neighbour.GetFormalCharge()),
                "bond_type": str(bond.GetBondType()) if bond is not None else None,
                "bond_order": (
                    float(bond.GetBondTypeAsDouble()) if bond is not None else None
                ),
                "bond_is_aromatic": bool(bond.GetIsAromatic()) if bond else False,
            }
        )
    neighbours.sort(key=lambda item: int(item["atom_index"]))
    explicit_h_neighbours = sum(
        int(item["atomic_number"]) == 1 for item in neighbours
    )
    atomic_number = int(atom.GetAtomicNum())
    periodic_table = Chem.GetPeriodicTable()
    try:
        allowed_valences = [
            int(value) for value in periodic_table.GetValenceList(atomic_number)
        ]
    except Exception as exc:
        allowed_valences = [f"{type(exc).__name__}: {exc}"]

    explicit_h_count = int(atom.GetNumExplicitHs())
    implicit_h_count = int(atom.GetNumImplicitHs())
    return {
        "atom_index": int(atom.GetIdx()),
        "symbol": atom.GetSymbol(),
        "atomic_number": atomic_number,
        "formal_charge": int(atom.GetFormalCharge()),
        "is_aromatic": bool(atom.GetIsAromatic()),
        "hybridization": str(atom.GetHybridization()),
        "chiral_tag": str(atom.GetChiralTag()),
        "degree": int(atom.GetDegree()),
        "heavy_atom_degree": sum(
            int(item["atomic_number"]) > 1 for item in neighbours
        ),
        "explicit_valence": valence(Chem.ValenceType.EXPLICIT),
        "implicit_valence": valence(Chem.ValenceType.IMPLICIT),
        "explicit_h_count": explicit_h_count,
        "implicit_h_count": implicit_h_count,
        "explicit_h_neighbour_count": explicit_h_neighbours,
        "total_visible_h_count": (
            explicit_h_count + implicit_h_count + explicit_h_neighbours
        ),
        "no_implicit": bool(atom.GetNoImplicit()),
        "radical_electron_count": int(atom.GetNumRadicalElectrons()),
        "periodic_table_allowed_valences": allowed_valences,
        "neighbours": neighbours,
    }


def _chemistry_problem_diagnostics(molecule: Chem.Mol) -> list[dict[str, Any]]:
    """Best-effort RDKit chemistry problem details for a failed probe."""

    log_blocker = rdBase.BlockLogs()
    try:
        problems = Chem.DetectChemistryProblems(molecule)
    except Exception as exc:
        return [
            {
                "type": "diagnostic_error",
                "message": f"{type(exc).__name__}: {exc}",
            }
        ]
    finally:
        del log_blocker

    result: list[dict[str, Any]] = []
    for problem in problems:
        record: dict[str, Any] = {
            "type": str(problem.GetType()),
            "message": str(problem.Message()),
        }
        for attribute, key in (
            ("GetAtomIdx", "atom_index"),
            ("GetAtomIndices", "atom_indices"),
        ):
            method = getattr(problem, attribute, None)
            if method is None:
                continue
            try:
                value = method()
                record[key] = (
                    [int(item) for item in value]
                    if key == "atom_indices"
                    else int(value)
                )
            except Exception:
                pass
        result.append(record)
    return result


def probe_atom_single_bond_capacity(
    molecule: Chem.Mol,
    atom_index: int,
    *,
    maximum_capacity: int = 8,
    diagnostic_steps: list[dict[str, Any]] | None = None,
) -> int:
    """Return the number of sequential extra single bonds RDKit accepts.

    Each successful step mirrors reconstruction: one replaceable hydrogen is
    consumed when present, a charge-neutral dummy endpoint is attached by a
    single bond, and the complete temporary molecule is sanitized.  Formal
    charges are never altered.  The bounded loop protects preflight from
    malformed chemistry while comfortably exceeding ordinary organic valence.
    """

    if maximum_capacity <= 0:
        raise ValueError("maximum_capacity must be positive")
    if atom_index < 0 or atom_index >= molecule.GetNumAtoms():
        raise IndexError(f"atom index is out of range: {atom_index}")

    editable = Chem.RWMol(Chem.Mol(molecule))
    current_index = int(atom_index)
    capacity = 0
    for step_index in range(int(maximum_capacity)):
        atom = editable.GetAtomWithIdx(current_index)
        step_record: dict[str, Any] | None = None
        if diagnostic_steps is not None:
            step_record = {
                "step": step_index + 1,
                "atom_before": atom_chemistry_diagnostics(atom),
            }
        explicit_hydrogen_neighbours = sorted(
            int(neighbour.GetIdx())
            for neighbour in atom.GetNeighbors()
            if neighbour.GetAtomicNum() == 1
        )
        if explicit_hydrogen_neighbours:
            hydrogen_index = explicit_hydrogen_neighbours[0]
            editable.RemoveAtom(hydrogen_index)
            if hydrogen_index < current_index:
                current_index -= 1
            atom = editable.GetAtomWithIdx(current_index)
            hydrogen_action = {
                "method": "explicit_h_atom",
                "removed_atom_index": hydrogen_index,
            }
        else:
            hydrogen_action = {
                "method": consume_connection_hydrogen(atom),
                "removed_atom_index": None,
            }
        if step_record is not None:
            step_record["hydrogen_action"] = hydrogen_action

        dummy_atom = Chem.Atom(0)
        dummy_atom.SetNoImplicit(True)
        dummy_index = int(editable.AddAtom(dummy_atom))
        editable.AddBond(current_index, dummy_index, Chem.BondType.SINGLE)
        log_blocker = rdBase.BlockLogs()
        try:
            sanitize_result = Chem.SanitizeMol(editable, catchErrors=True)
        except Exception as exc:
            if step_record is not None:
                step_record["accepted"] = False
                step_record["sanitize"] = {
                    "status": "exception",
                    "error": f"{type(exc).__name__}: {exc}",
                }
                step_record["chemistry_problems"] = (
                    _chemistry_problem_diagnostics(editable)
                )
                diagnostic_steps.append(step_record)
            break
        finally:
            del log_blocker
        if step_record is not None:
            step_record["sanitize"] = {
                "status": str(sanitize_result),
                "code": int(sanitize_result),
            }
        if sanitize_result != Chem.SanitizeFlags.SANITIZE_NONE:
            if step_record is not None:
                step_record["accepted"] = False
                step_record["chemistry_problems"] = (
                    _chemistry_problem_diagnostics(editable)
                )
                diagnostic_steps.append(step_record)
            break
        capacity += 1
        if step_record is not None:
            step_record["accepted"] = True
            step_record["atom_after"] = atom_chemistry_diagnostics(
                editable.GetAtomWithIdx(current_index)
            )
            diagnostic_steps.append(step_record)
    return capacity


def atom_is_connection_candidate(atom: Chem.Atom) -> bool:
    symbol = atom.GetSymbol()
    formal_charge = int(atom.GetFormalCharge())
    if atom_has_consumable_h(atom):
        return True
    if symbol in VALENCE_PROBED_SPECIAL_ELEMENTS:
        return atom_accepts_single_bond_without_charge_change(atom)
    return symbol in NEGATIVE_HALOGEN_ELEMENTS and formal_charge < 0


def diagnose_atom_single_bond_capacity(
    smiles: str,
    atom_index: int,
    *,
    maximum_capacity: int = 8,
) -> dict[str, Any]:
    """Explain every step of the production capacity probe for one atom."""

    parsed = mol_from_smiles_quietly(smiles)
    if parsed is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")
    canonical_smiles = Chem.MolToSmiles(
        parsed, canonical=True, isomericSmiles=True
    )
    molecule = mol_from_smiles_quietly(canonical_smiles)
    if molecule is None:
        raise RuntimeError(
            f"RDKit could not reparse canonical SMILES {canonical_smiles!r}"
        )
    if atom_index < 0 or atom_index >= molecule.GetNumAtoms():
        raise IndexError(f"atom index is out of range: {atom_index}")

    atom = molecule.GetAtomWithIdx(int(atom_index))
    diagnostic_steps: list[dict[str, Any]] = []
    capacity = probe_atom_single_bond_capacity(
        molecule,
        int(atom_index),
        maximum_capacity=maximum_capacity,
        diagnostic_steps=diagnostic_steps,
    )
    symbol = atom.GetSymbol()
    formal_charge = int(atom.GetFormalCharge())
    return {
        "input_smiles": smiles,
        "canonical_smiles": canonical_smiles,
        "atom_id": int(atom_index),
        "initial_atom": atom_chemistry_diagnostics(atom),
        "candidate_rule": {
            "has_consumable_hydrogen": atom_has_consumable_h(atom),
            "is_valence_probed_special_element": (
                symbol in VALENCE_PROBED_SPECIAL_ELEMENTS
            ),
            "is_negative_halogen": (
                symbol in NEGATIVE_HALOGEN_ELEMENTS and formal_charge < 0
            ),
            "accepts_one_bond_without_charge_change": (
                atom_accepts_single_bond_without_charge_change(atom)
            ),
            "eligible": atom_is_connection_candidate(atom),
        },
        "computed_capacity": int(capacity),
        "maximum_capacity_tested": int(maximum_capacity),
        "probe_steps": diagnostic_steps,
    }


def diagnose_fragment_connection_combination(
    node_smiles: Mapping[int, str],
    connections: Sequence[Mapping[str, int]],
) -> dict[str, Any]:
    """Try all labelled single bonds together and report sanitization details.

    This deliberately omits geometry and symmetry resolution.  Its only role
    is to answer whether the canonical fragment chemistry accepts the complete
    labelled combination under the same hydrogen-consumption rules used by
    formal reconstruction.
    """

    diagnostics: dict[str, Any] = {
        "status": "not_run",
        "connection_count": len(connections),
        "fragments": [],
        "connections": [],
    }
    stage = "fragment_parsing"
    editable: Chem.RWMol | None = None
    try:
        referenced_nodes = sorted(
            {
                int(connection[key])
                for connection in connections
                for key in ("node_a", "node_b")
            }
        )
        combined: Chem.Mol | None = None
        node_atom_offsets: dict[int, int] = {}
        atom_offset = 0
        for node_index in referenced_nodes:
            try:
                input_smiles = str(node_smiles[node_index])
            except KeyError as exc:
                raise KeyError(
                    f"connection references node without SMILES: {node_index}"
                ) from exc
            parsed = mol_from_smiles_quietly(input_smiles)
            if parsed is None:
                raise ValueError(f"RDKit could not parse SMILES: {input_smiles!r}")
            canonical_smiles = Chem.MolToSmiles(
                parsed, canonical=True, isomericSmiles=True
            )
            fragment = mol_from_smiles_quietly(canonical_smiles)
            if fragment is None:
                raise RuntimeError(
                    "RDKit could not reparse canonical SMILES "
                    f"{canonical_smiles!r}"
                )
            node_atom_offsets[node_index] = atom_offset
            diagnostics["fragments"].append(
                {
                    "node": node_index,
                    "input_smiles": input_smiles,
                    "canonical_smiles": canonical_smiles,
                    "atom_offset": atom_offset,
                    "atom_count": int(fragment.GetNumAtoms()),
                }
            )
            combined = (
                Chem.Mol(fragment)
                if combined is None
                else Chem.CombineMols(combined, fragment)
            )
            atom_offset += int(fragment.GetNumAtoms())

        if combined is None:
            diagnostics.update(
                {
                    "status": "passed",
                    "stage": "no_connections",
                    "sanitize": {
                        "status": str(Chem.SanitizeFlags.SANITIZE_NONE),
                        "code": int(Chem.SanitizeFlags.SANITIZE_NONE),
                    },
                }
            )
            return diagnostics

        stage = "endpoint_validation"
        normalized: list[dict[str, int]] = []
        endpoint_rows: list[dict[str, Any]] = []
        for connection_index, raw_connection in enumerate(connections):
            connection = {
                "connection_index": int(connection_index),
                "node_a": int(raw_connection["node_a"]),
                "node_b": int(raw_connection["node_b"]),
                "atom_a_id": int(raw_connection["atom_a_id"]),
                "atom_b_id": int(raw_connection["atom_b_id"]),
            }
            if connection["node_a"] == connection["node_b"]:
                raise ValueError("inter-fragment connection cannot be a self-loop")
            for side in ("a", "b"):
                node_index = connection[f"node_{side}"]
                atom_id = connection[f"atom_{side}_id"]
                fragment_record = next(
                    item
                    for item in diagnostics["fragments"]
                    if int(item["node"]) == node_index
                )
                if atom_id < 0 or atom_id >= int(fragment_record["atom_count"]):
                    raise IndexError(
                        "connection atom ID is out of range: "
                        f"node={node_index}, atom_id={atom_id}"
                    )
                combined_index = node_atom_offsets[node_index] + atom_id
                atom = combined.GetAtomWithIdx(combined_index)
                if atom.GetAtomicNum() <= 1:
                    raise ValueError(
                        "connection endpoint must be a heavy atom: "
                        f"node={node_index}, atom_id={atom_id}"
                    )
                endpoint_rows.append(
                    {
                        "connection_index": int(connection_index),
                        "side": side,
                        "node": node_index,
                        "atom_id": atom_id,
                        "combined_atom_index_before_removal": combined_index,
                        "initial_atom": atom_chemistry_diagnostics(atom),
                    }
                )
            normalized.append(connection)

        diagnostics["endpoint_initial_states"] = endpoint_rows

        stage = "explicit_hydrogen_assignment"
        endpoints_by_center: dict[int, list[dict[str, Any]]] = {}
        for endpoint in endpoint_rows:
            center_index = int(endpoint["combined_atom_index_before_removal"])
            endpoints_by_center.setdefault(center_index, []).append(endpoint)
        explicit_hydrogen_by_endpoint: dict[tuple[int, str], int] = {}
        for center_index, center_endpoints in sorted(endpoints_by_center.items()):
            center_atom = combined.GetAtomWithIdx(center_index)
            hydrogen_indices = sorted(
                int(neighbour.GetIdx())
                for neighbour in center_atom.GetNeighbors()
                if neighbour.GetAtomicNum() == 1
            )
            ordered_endpoints = sorted(
                center_endpoints,
                key=lambda item: (
                    int(item["connection_index"]), str(item["side"])
                ),
            )
            for endpoint, hydrogen_index in zip(
                ordered_endpoints, hydrogen_indices
            ):
                explicit_hydrogen_by_endpoint[
                    (int(endpoint["connection_index"]), str(endpoint["side"]))
                ] = hydrogen_index

        removed_hydrogen_indices = sorted(
            set(explicit_hydrogen_by_endpoint.values())
        )
        if len(removed_hydrogen_indices) != len(
            explicit_hydrogen_by_endpoint
        ):
            raise RuntimeError(
                "one explicit hydrogen was assigned to multiple connections"
            )
        editable = Chem.RWMol(Chem.Mol(combined))
        for hydrogen_index in reversed(removed_hydrogen_indices):
            if editable.GetAtomWithIdx(hydrogen_index).GetAtomicNum() != 1:
                raise RuntimeError("selected replacement atom is not hydrogen")
            editable.RemoveAtom(hydrogen_index)
        editable.UpdatePropertyCache(strict=False)

        def remap(atom_index: int) -> int:
            return int(atom_index) - sum(
                removed < int(atom_index)
                for removed in removed_hydrogen_indices
            )

        stage = "bond_addition"
        endpoint_lookup = {
            (int(item["connection_index"]), str(item["side"])): item
            for item in endpoint_rows
        }
        for connection in normalized:
            connection_index = int(connection["connection_index"])
            atom_indices: dict[str, int] = {}
            hydrogen_actions: dict[str, dict[str, Any]] = {}
            for side in ("a", "b"):
                endpoint_key = (connection_index, side)
                endpoint = endpoint_lookup[endpoint_key]
                current_index = remap(
                    int(endpoint["combined_atom_index_before_removal"])
                )
                atom_indices[side] = current_index
                explicit_hydrogen_index = explicit_hydrogen_by_endpoint.get(
                    endpoint_key
                )
                if explicit_hydrogen_index is not None:
                    hydrogen_actions[side] = {
                        "method": "explicit_h_atom",
                        "removed_atom_index_before_removal": (
                            explicit_hydrogen_index
                        ),
                    }
                else:
                    hydrogen_actions[side] = {
                        "method": consume_connection_hydrogen(
                            editable.GetAtomWithIdx(current_index)
                        ),
                        "removed_atom_index_before_removal": None,
                    }
            if editable.GetBondBetweenAtoms(
                atom_indices["a"], atom_indices["b"]
            ) is not None:
                raise ValueError(
                    "duplicate requested inter-fragment bond: "
                    f"{connection}"
                )
            editable.AddBond(
                atom_indices["a"], atom_indices["b"], Chem.BondType.SINGLE
            )
            editable.UpdatePropertyCache(strict=False)
            diagnostics["connections"].append(
                {
                    **connection,
                    "combined_atom_a_after_removal": atom_indices["a"],
                    "combined_atom_b_after_removal": atom_indices["b"],
                    "hydrogen_actions": hydrogen_actions,
                }
            )

        stage = "final_sanitization"
        log_blocker = rdBase.BlockLogs()
        try:
            sanitize_result = Chem.SanitizeMol(editable, catchErrors=True)
        finally:
            del log_blocker
        diagnostics["sanitize"] = {
            "status": str(sanitize_result),
            "code": int(sanitize_result),
        }
        if sanitize_result != Chem.SanitizeFlags.SANITIZE_NONE:
            diagnostics.update(
                {
                    "status": "failed",
                    "stage": stage,
                    "chemistry_problems": _chemistry_problem_diagnostics(
                        editable
                    ),
                }
            )
            return diagnostics

        diagnostics.update(
            {
                "status": "passed",
                "stage": stage,
                "removed_explicit_hydrogen_count": len(
                    removed_hydrogen_indices
                ),
                "endpoint_final_states": [
                    {
                        "node": int(endpoint["node"]),
                        "atom_id": int(endpoint["atom_id"]),
                        "atom": atom_chemistry_diagnostics(
                            editable.GetAtomWithIdx(
                                remap(
                                    int(
                                        endpoint[
                                            "combined_atom_index_before_removal"
                                        ]
                                    )
                                )
                            )
                        ),
                    }
                    for endpoint in endpoint_rows
                ],
            }
        )
        return diagnostics
    except Exception as exc:
        diagnostics.update(
            {
                "status": "failed",
                "stage": stage,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        if editable is not None:
            diagnostics["chemistry_problems"] = (
                _chemistry_problem_diagnostics(editable)
            )
        return diagnostics


class FragmentCandidateCache:
    """Cache only the final eligible IDs and their fixed atom vectors."""

    def __init__(self, atom_vocabulary: AtomVocabularyLookup) -> None:
        self.atom_vocabulary = atom_vocabulary
        self._cache: dict[str, FragmentCandidateEntry] = {}
        self._errors: dict[str, Exception] = {}

    def get(self, smiles: str) -> FragmentCandidateEntry:
        if smiles in self._cache:
            return self._cache[smiles]
        if smiles in self._errors:
            raise self._errors[smiles]
        try:
            entry = self._build(smiles)
        except Exception as exc:
            self._errors[smiles] = exc
            raise
        self._cache[smiles] = entry
        return entry

    def _build(self, smiles: str) -> FragmentCandidateEntry:
        vocabulary_entry = self.atom_vocabulary.lookup(smiles)
        parsed = mol_from_smiles_quietly(smiles)
        if parsed is None:
            raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")
        canonical_smiles = Chem.MolToSmiles(
            parsed, canonical=True, isomericSmiles=True
        )
        molecule = mol_from_smiles_quietly(canonical_smiles)
        if molecule is None:
            raise RuntimeError(f"RDKit could not reparse canonical SMILES {canonical_smiles!r}")

        ids = vocabulary_entry.canonical_atom_ids.astype(np.int64, copy=False)
        if len(set(map(int, ids.tolist()))) != len(ids):
            raise ValueError(f"duplicate canonical atom IDs for {smiles!r}")

        embedding_map: dict[int, np.ndarray] = {}
        eligible: list[int] = []
        capacities: dict[int, int] = {}
        ignored: list[int] = []
        ineligible_reasons: dict[int, str] = {}
        for row, raw_atom_id in enumerate(ids.tolist()):
            atom_id = int(raw_atom_id)
            if atom_id >= molecule.GetNumAtoms():
                ignored.append(atom_id)
                continue
            atom = molecule.GetAtomWithIdx(atom_id)
            if atom.GetAtomicNum() <= 1:
                ignored.append(atom_id)
                continue
            embedding_map[atom_id] = vocabulary_entry.embeddings[row]
            if atom_is_connection_candidate(atom):
                capacity = probe_atom_single_bond_capacity(molecule, atom_id)
                if capacity <= 0:
                    raise RuntimeError(
                        "candidate eligibility/capacity disagreement for "
                        f"{smiles!r} atom {atom_id}"
                    )
                eligible.append(atom_id)
                capacities[atom_id] = capacity
            elif atom_is_known_neutralized_nitrogen_target(atom):
                ineligible_reasons[atom_id] = (
                    KNOWN_NEUTRALIZED_NITROGEN_TARGET_REASON
                )
            else:
                ineligible_reasons[atom_id] = CHEMICALLY_INELIGIBLE_TARGET_REASON

        return FragmentCandidateEntry(
            input_smiles=smiles,
            canonical_smiles=canonical_smiles,
            eligible_atom_ids=tuple(sorted(eligible)),
            atom_capacities=dict(sorted(capacities.items())),
            atom_embeddings=embedding_map,
            ignored_vocab_atom_ids=tuple(sorted(ignored)),
            ineligible_atom_reasons=dict(sorted(ineligible_reasons.items())),
        )


@dataclass(frozen=True)
class GlobalAtomPairChoice:
    """One query-local candidate used by capacity-aware global search."""

    query_index: int
    candidate_index: int
    node_a: int
    node_b: int
    atom_a_id: int
    atom_b_id: int
    score: float

    @property
    def endpoints(self) -> tuple[tuple[int, int], tuple[int, int]]:
        return (
            (int(self.node_a), int(self.atom_a_id)),
            (int(self.node_b), int(self.atom_b_id)),
        )


@dataclass(frozen=True)
class GlobalAtomPairCombination:
    """A complete one-choice-per-query result from global beam search."""

    choices: tuple[GlobalAtomPairChoice, ...]
    score: float
    capacity_usage: tuple[tuple[int, int, int], ...]
    overflow_count: int

    @property
    def candidate_indices(self) -> tuple[int, ...]:
        return tuple(choice.candidate_index for choice in self.choices)


@dataclass
class _GlobalBeamState:
    candidate_indices: tuple[int, ...]
    score: float
    usage: dict[tuple[int, int], int]
    overflow_count: int


def beam_search_atom_pair_combinations(
    query_choices: Sequence[Sequence[GlobalAtomPairChoice]],
    atom_capacities: Mapping[tuple[int, int], int],
    *,
    initial_usage: Mapping[tuple[int, int], int] | None = None,
    prefer_distinct_node_atoms: bool = False,
    beam_width: int = 64,
    max_legal_results: int = 8,
    max_conflict_results: int = 0,
) -> tuple[list[GlobalAtomPairCombination], list[GlobalAtomPairCombination]]:
    """Search legal combinations and, optionally, high-score conflicts.

    Legal and already-conflicting partial beams are pruned independently so a
    high-scoring overflow cannot crowd every chemically feasible path out of
    the legal beam.  When ``prefer_distinct_node_atoms`` is enabled, fewer
    repeated uses of the same ``(node, atom)`` endpoint take priority over the
    model score.  Scores are ordinary Python floats; callers may search on
    detached values and gather the chosen live logits afterwards.
    """

    if beam_width <= 0:
        raise ValueError("beam_width must be positive")
    if max_legal_results < 0 or max_conflict_results < 0:
        raise ValueError("result limits cannot be negative")
    capacities = {
        (int(key[0]), int(key[1])): int(value)
        for key, value in atom_capacities.items()
    }
    if any(value < 0 for value in capacities.values()):
        raise ValueError("atom capacities cannot be negative")
    starting_usage = {
        (int(key[0]), int(key[1])): int(value)
        for key, value in (initial_usage or {}).items()
    }
    for key, used in starting_usage.items():
        if key not in capacities:
            raise KeyError(f"initial usage references unknown atom {key}")
        if used < 0:
            raise ValueError("initial atom usage cannot be negative")
        if used > capacities[key]:
            raise ValueError(
                f"fixed atom usage exceeds capacity for {key}: "
                f"usage={used}, capacity={capacities[key]}"
            )

    legal_beam = [
        _GlobalBeamState((), 0.0, dict(starting_usage), 0)
    ]
    conflict_beam: list[_GlobalBeamState] = []
    choice_lookup: list[dict[int, GlobalAtomPairChoice]] = []

    def repeated_node_atom_use_count(state: _GlobalBeamState) -> int:
        return sum(max(0, int(used) - 1) for used in state.usage.values())

    def sort_key(state: _GlobalBeamState) -> tuple[Any, ...]:
        if prefer_distinct_node_atoms:
            return (
                repeated_node_atom_use_count(state),
                -float(state.score),
                state.candidate_indices,
            )
        return (
            -float(state.score),
            state.candidate_indices,
        )

    for expected_query, raw_choices in enumerate(query_choices):
        choices = tuple(raw_choices)
        if not choices:
            return [], []
        for choice_position, choice in enumerate(choices):
            if not isinstance(choice, GlobalAtomPairChoice):
                actual_type = (
                    f"{type(choice).__module__}.{type(choice).__qualname__}"
                )
                raise TypeError(
                    "query_choices must contain GlobalAtomPairChoice instances; "
                    f"query={expected_query}, position={choice_position}, "
                    f"actual_type={actual_type}, value={choice!r}"
                )
        if any(int(choice.query_index) != expected_query for choice in choices):
            raise ValueError(
                "query_choices must be ordered and use zero-based local "
                "query indices"
            )
        choices_by_index: dict[int, GlobalAtomPairChoice] = {}
        for choice in choices:
            candidate_index = int(choice.candidate_index)
            if candidate_index in choices_by_index:
                raise ValueError(
                    "candidate indices must be unique within each query; "
                    f"query={expected_query}, candidate_index={candidate_index}"
                )
            choices_by_index[candidate_index] = choice
        choice_lookup.append(choices_by_index)

        next_legal: list[_GlobalBeamState] = []
        next_conflict: list[_GlobalBeamState] = []
        for state in (*legal_beam, *conflict_beam):
            for choice in choices:
                usage = dict(state.usage)
                overflow = int(state.overflow_count)
                for endpoint in choice.endpoints:
                    try:
                        capacity = capacities[endpoint]
                    except KeyError as exc:
                        raise KeyError(
                            f"candidate references atom without capacity: {endpoint}"
                        ) from exc
                    previous = usage.get(endpoint, 0)
                    current = previous + 1
                    usage[endpoint] = current
                    overflow += max(0, current - capacity) - max(
                        0, previous - capacity
                    )
                candidate_index = int(choice.candidate_index)
                expanded = _GlobalBeamState(
                    candidate_indices=state.candidate_indices + (candidate_index,),
                    score=float(state.score) + float(choice.score),
                    usage=usage,
                    overflow_count=overflow,
                )
                if overflow:
                    if max_conflict_results:
                        next_conflict.append(expanded)
                else:
                    next_legal.append(expanded)
        next_legal.sort(key=sort_key)
        legal_beam = next_legal[:beam_width]
        if max_conflict_results:
            next_conflict.sort(key=sort_key)
            conflict_beam = next_conflict[:beam_width]
        else:
            conflict_beam = []
        if not legal_beam and not conflict_beam:
            break

    def finalize(
        states: Sequence[_GlobalBeamState], limit: int
    ) -> list[GlobalAtomPairCombination]:
        result: list[GlobalAtomPairCombination] = []
        for state in sorted(states, key=sort_key)[:limit]:
            selected_count = len(state.candidate_indices)
            query_count = len(choice_lookup)
            if selected_count != query_count:
                candidate_indices_type = (
                    f"{type(state.candidate_indices).__module__}."
                    f"{type(state.candidate_indices).__qualname__}"
                )
                choice_lookup_type = (
                    f"{type(choice_lookup).__module__}."
                    f"{type(choice_lookup).__qualname__}"
                )
                raise RuntimeError(
                    "beam state is incomplete: "
                    f"selected={selected_count}, queries={query_count}, "
                    f"candidate_indices_type={candidate_indices_type}, "
                    f"choice_lookup_type={choice_lookup_type}, "
                    f"candidate_indices={state.candidate_indices!r}"
                )
            rebuilt_choices: list[GlobalAtomPairChoice] = []
            for query_index, candidate_index in enumerate(
                state.candidate_indices
            ):
                try:
                    rebuilt_choices.append(
                        choice_lookup[query_index][candidate_index]
                    )
                except KeyError as exc:
                    raise RuntimeError(
                        "beam state references an unknown candidate: "
                        f"query={query_index}, "
                        f"candidate_index={candidate_index}"
                    ) from exc
            result.append(
                GlobalAtomPairCombination(
                    choices=tuple(rebuilt_choices),
                    score=float(state.score),
                    capacity_usage=tuple(
                        (node, atom_id, used)
                        for (node, atom_id), used in sorted(state.usage.items())
                        if used
                    ),
                    overflow_count=int(state.overflow_count),
                )
            )
        return result

    return (
        finalize(legal_beam, max_legal_results),
        finalize(conflict_beam, max_conflict_results),
    )


@dataclass(frozen=True)
class AlignedGeometryResult:
    valid: bool
    coordinates: Mapping[int, np.ndarray]
    error: str | None


class _ConformerGenerationFailure(RuntimeError):
    """A reusable conformer template could not be generated."""


class ConformerGeometryCache:
    """LRU template cache with process-local and persistent failure caching."""

    def __init__(
        self,
        max_templates: int = 4096,
        *,
        failure_registry_dir: str | Path | None = None,
    ) -> None:
        if max_templates < 0:
            raise ValueError("max_templates cannot be negative")
        self.max_templates = int(max_templates)
        self._templates: OrderedDict[str, FragmentConformerTemplate] = OrderedDict()
        self._failures: dict[str, str] = {}
        self.template_hits = 0
        self.template_misses = 0
        self.failure_registry_dir = (
            Path(failure_registry_dir).expanduser().resolve()
            if failure_registry_dir is not None
            else None
        )
        self._failure_marker_dir = (
            self.failure_registry_dir / "failed_smiles"
            if self.failure_registry_dir is not None
            else None
        )
        self._failure_occurrence_dir = (
            self.failure_registry_dir / "source_occurrences"
            if self.failure_registry_dir is not None
            else None
        )
        if self._failure_marker_dir is not None:
            self._failure_marker_dir.mkdir(parents=True, exist_ok=True)
        if self._failure_occurrence_dir is not None:
            self._failure_occurrence_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _smiles_sha256(smiles: str) -> str:
        return hashlib.sha256(smiles.encode("utf-8")).hexdigest()

    def _failure_marker_path(self, smiles: str) -> Path | None:
        if self._failure_marker_dir is None:
            return None
        return self._failure_marker_dir / f"{self._smiles_sha256(smiles)}.json"

    def _persistent_failure_message(self, smiles: str) -> str | None:
        marker_path = self._failure_marker_path(smiles)
        if marker_path is None or not marker_path.is_file():
            return None
        try:
            payload = json.loads(marker_path.read_text(encoding="utf-8"))
            if not isinstance(payload, Mapping):
                raise TypeError("failure marker must contain a JSON object")
            registered_smiles = str(payload.get("canonical_smiles", ""))
            if registered_smiles != smiles:
                raise ValueError(
                    "failure marker canonical SMILES does not match its hash"
                )
            message = str(payload.get("failure_error", "")).strip()
            if not message:
                raise ValueError("failure marker has no failure_error")
            return message
        except Exception as exc:
            return (
                "Persistent conformer failure marker is unreadable: "
                f"{marker_path}: {type(exc).__name__}: {exc}"
            )

    def _write_failure_marker(self, smiles: str, message: str) -> None:
        marker_path = self._failure_marker_path(smiles)
        if marker_path is None:
            return
        atomic_write_json(
            marker_path,
            {
                "canonical_smiles": smiles,
                "smiles_sha256": self._smiles_sha256(smiles),
                "failure_error": message,
            },
        )

    def _write_failure_occurrence(
        self,
        smiles: str,
        message: str,
        *,
        source_path: str | None,
        sample_id: str | None,
        node_index: int | None,
        dataset_split: str | None,
    ) -> None:
        if self._failure_occurrence_dir is None:
            return
        resolved_source = str(source_path or "<unknown>")
        resolved_sample = str(sample_id or Path(resolved_source).stem)
        resolved_split = str(dataset_split or "unspecified")
        resolved_node = int(node_index) if node_index is not None else -1
        identity = json.dumps(
            [
                smiles,
                resolved_source,
                resolved_sample,
                resolved_node,
                resolved_split,
            ],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        occurrence_sha256 = hashlib.sha256(
            identity.encode("utf-8")
        ).hexdigest()
        smiles_sha256 = self._smiles_sha256(smiles)
        occurrence_dir = self._failure_occurrence_dir / smiles_sha256
        occurrence_dir.mkdir(parents=True, exist_ok=True)
        occurrence_path = occurrence_dir / f"{occurrence_sha256}.json"
        if occurrence_path.is_file():
            return
        atomic_write_json(
            occurrence_path,
            {
                "canonical_smiles": smiles,
                "smiles_sha256": smiles_sha256,
                "failure_error": message,
                "dataset_split": resolved_split,
                "source_path": resolved_source,
                "pyg_file": Path(resolved_source).name,
                "sample_id": resolved_sample,
                "node_index": resolved_node,
            },
        )

    def _template(self, smiles: str) -> FragmentConformerTemplate:
        if smiles in self._templates:
            self.template_hits += 1
            template = self._templates.pop(smiles)
            self._templates[smiles] = template
            return template
        if smiles in self._failures:
            self.template_hits += 1
            raise _ConformerGenerationFailure(self._failures[smiles])

        persistent_message = self._persistent_failure_message(smiles)
        if persistent_message is not None:
            self.template_hits += 1
            self._failures[smiles] = persistent_message
            raise _ConformerGenerationFailure(persistent_message)

        self.template_misses += 1
        try:
            template = build_fragment_conformer_template(smiles)
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            self._failures[smiles] = message
            try:
                self._write_failure_marker(smiles, message)
            except Exception as registry_exc:
                message = (
                    f"{message} | persistent failure registry write failed: "
                    f"{type(registry_exc).__name__}: {registry_exc}"
                )
                self._failures[smiles] = message
            raise _ConformerGenerationFailure(message) from exc

        self._templates[smiles] = template
        if self.max_templates > 0:
            while len(self._templates) > self.max_templates:
                self._templates.popitem(last=False)
        return template

    def align(
        self,
        smiles: str,
        *,
        center: Sequence[float] | np.ndarray,
        reference_frame: Sequence[Sequence[float]] | np.ndarray,
        source_path: str | None = None,
        sample_id: str | None = None,
        node_index: int | None = None,
        dataset_split: str | None = None,
    ) -> AlignedGeometryResult:
        try:
            template = self._template(smiles)
        except _ConformerGenerationFailure as exc:
            message = str(exc)
            try:
                self._write_failure_occurrence(
                    smiles,
                    message,
                    source_path=source_path,
                    sample_id=sample_id,
                    node_index=node_index,
                    dataset_split=dataset_split,
                )
            except Exception as registry_exc:
                message = (
                    f"{message} | failure source record write failed: "
                    f"{type(registry_exc).__name__}: {registry_exc}"
                )
            return AlignedGeometryResult(
                False,
                {},
                f"{type(exc).__name__}: {message}",
            )

        try:
            atom_ids, coordinates, _ = align_fragment_conformer_template(
                template,
                center=center,
                reference_frame=reference_frame,
            )
            atom_ids = np.asarray(atom_ids, dtype=np.int64).reshape(-1)
            coordinates = np.asarray(coordinates, dtype=np.float64).reshape(-1, 3)
            if len(atom_ids) != len(coordinates):
                raise RuntimeError("aligned atom ID and coordinate counts differ")
            if not np.all(np.isfinite(coordinates)):
                raise RuntimeError("aligned coordinates contain NaN or Inf")
            coordinate_map = {
                int(atom_id): coordinates[row].astype(np.float32, copy=False)
                for row, atom_id in enumerate(atom_ids.tolist())
            }
            return AlignedGeometryResult(True, coordinate_map, None)
        except Exception as exc:
            return AlignedGeometryResult(
                False,
                {},
                f"{type(exc).__name__}: {exc}",
            )

    def write_failure_report(
        self,
        report_path: str | Path,
    ) -> dict[str, Any]:
        """Consolidate immutable worker records into one readable JSON file."""

        read_errors: list[str] = []
        records_by_smiles: dict[str, dict[str, Any]] = {}
        if self._failure_marker_dir is not None:
            for marker_path in sorted(self._failure_marker_dir.glob("*.json")):
                try:
                    payload = json.loads(
                        marker_path.read_text(encoding="utf-8")
                    )
                    smiles = str(payload["canonical_smiles"])
                    records_by_smiles[smiles] = {
                        "canonical_smiles": smiles,
                        "smiles_sha256": str(
                            payload.get("smiles_sha256", marker_path.stem)
                        ),
                        "failure_error": str(payload["failure_error"]),
                        "sources": {},
                    }
                except Exception as exc:
                    read_errors.append(
                        f"{marker_path}: {type(exc).__name__}: {exc}"
                    )

        occurrence_count = 0
        if self._failure_occurrence_dir is not None:
            occurrence_paths = sorted(
                self._failure_occurrence_dir.glob("*/*.json")
            )
            for occurrence_path in occurrence_paths:
                try:
                    payload = json.loads(
                        occurrence_path.read_text(encoding="utf-8")
                    )
                    smiles = str(payload["canonical_smiles"])
                    record = records_by_smiles.setdefault(
                        smiles,
                        {
                            "canonical_smiles": smiles,
                            "smiles_sha256": str(
                                payload.get(
                                    "smiles_sha256",
                                    self._smiles_sha256(smiles),
                                )
                            ),
                            "failure_error": str(
                                payload.get("failure_error", "unknown")
                            ),
                            "sources": {},
                        },
                    )
                    source_key = json.dumps(
                        [
                            payload.get("dataset_split", "unspecified"),
                            payload.get("source_path", "<unknown>"),
                            payload.get("sample_id", "<unknown>"),
                        ],
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    sources: dict[str, dict[str, Any]] = record["sources"]
                    source = sources.setdefault(
                        source_key,
                        {
                            "dataset_split": str(
                                payload.get("dataset_split", "unspecified")
                            ),
                            "source_path": str(
                                payload.get("source_path", "<unknown>")
                            ),
                            "pyg_file": str(payload.get("pyg_file", "")),
                            "sample_id": str(
                                payload.get("sample_id", "<unknown>")
                            ),
                            "node_indices": set(),
                        },
                    )
                    source["node_indices"].add(
                        int(payload.get("node_index", -1))
                    )
                    occurrence_count += 1
                except Exception as exc:
                    read_errors.append(
                        f"{occurrence_path}: {type(exc).__name__}: {exc}"
                    )

        records: list[dict[str, Any]] = []
        for smiles in sorted(records_by_smiles):
            raw_record = records_by_smiles[smiles]
            raw_sources: dict[str, dict[str, Any]] = raw_record["sources"]
            sources: list[dict[str, Any]] = []
            for source_key in sorted(raw_sources):
                source = dict(raw_sources[source_key])
                source["node_indices"] = sorted(source["node_indices"])
                sources.append(source)
            records.append(
                {
                    "canonical_smiles": raw_record["canonical_smiles"],
                    "smiles_sha256": raw_record["smiles_sha256"],
                    "failure_error": raw_record["failure_error"],
                    "sources": sources,
                }
            )

        report = {
            "registry_directory": (
                str(self.failure_registry_dir)
                if self.failure_registry_dir is not None
                else None
            ),
            "failed_smiles_count": len(records),
            "source_occurrence_count": occurrence_count,
            "records": records,
            "registry_read_errors": read_errors,
        }
        atomic_write_json(report_path, report)
        return report


def finite_float(value: Any, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result
