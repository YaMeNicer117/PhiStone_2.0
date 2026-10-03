#!/usr/bin/env python3
"""Build a neutral, element- and ring-filtered vocabulary of 3D-decodable SMILES.

Source SMILES whose total formal charge is nonzero or whose atoms fall outside
the allowed drug-relevant element set are excluded before 3D decoding.
Fragments with five or more rings, any three- or four-membered ring, or a
bridged ring system are also excluded using RDKit ring perception. Each
remaining SMILES is passed to ``build_fragment_conformer_template`` from
``PhiSLinker_atom_pair_smiles_decoder.py``.  That entry point generates an
unaligned conformer, so this filtering step does not translate the centroid or
align the molecule to a reference frame.  Generated coordinates are used only
to decide whether decoding succeeded; the output vocabulary contains the
original SMILES and their unchanged embedding rows.  Three additional
vocabularies contain the retained fragments with at most five, six, and seven
heavy atoms for survival linker, activity modify, and survival modify use.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Sequence
from uuid import uuid4

import numpy as np
from rdkit.Chem import rdMolDescriptors

if __package__:
    from .PhiSLinker_atom_pair_smiles_decoder import (
        DEFAULT_VOCAB_NPZ_PATH,
        EMBEDDING_DIM,
        build_fragment_conformer_template,
        mol_from_smiles_quietly,
    )
else:
    from PhiSLinker_atom_pair_smiles_decoder import (
        DEFAULT_VOCAB_NPZ_PATH,
        EMBEDDING_DIM,
        build_fragment_conformer_template,
        mol_from_smiles_quietly,
    )


DEFAULT_OUTPUT_DIR = (
    DEFAULT_VOCAB_NPZ_PATH.parent / "fragment_custom_embedding_128d"
)
OUTPUT_VOCAB_FILENAME = "fragment_embeddings_128d.npz"
OUTPUT_SURVIVAL_LINKER_VOCAB_FILENAME = (
    "fragment_survival_linker_embeddings_128d.npz"
)
OUTPUT_ACTIVITY_MODIFY_VOCAB_FILENAME = (
    "fragment_activity_modify_embeddings_128d.npz"
)
OUTPUT_SURVIVAL_MODIFY_VOCAB_FILENAME = (
    "fragment_survival_modify_embeddings_128d.npz"
)
OUTPUT_REPORT_FILENAME = "fragment_decode_report.json"
MAX_SURVIVAL_LINKER_HEAVY_ATOMS = 5
MAX_ACTIVITY_MODIFY_HEAVY_ATOMS = 6
MAX_SURVIVAL_MODIFY_HEAVY_ATOMS = 7
MAX_FRAGMENT_RINGS = 4
DISALLOWED_RING_SIZES = (3, 4)
ALLOWED_FRAGMENT_ELEMENTS = (
    ("H", 1),
    ("C", 6),
    ("N", 7),
    ("O", 8),
    ("F", 9),
    ("P", 15),
    ("S", 16),
    ("Cl", 17),
    ("Br", 35),
    ("I", 53),
)
ALLOWED_FRAGMENT_ATOMIC_NUMBERS = frozenset(
    atomic_number for _, atomic_number in ALLOWED_FRAGMENT_ELEMENTS
)
REQUIRED_NPZ_FIELDS = frozenset({"smiles", "fragment_embeddings"})
ERROR_MESSAGE_LIMIT = 2_000


def _load_source_vocabulary(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load and validate the source arrays without changing their values."""

    if not path.is_file():
        raise FileNotFoundError(f"Source vocabulary not found: {path}")
    if path.suffix.lower() != ".npz":
        raise ValueError(f"Source vocabulary must be an NPZ file: {path}")

    with np.load(path, allow_pickle=False) as loaded:
        missing_fields = REQUIRED_NPZ_FIELDS - set(loaded.files)
        if missing_fields:
            raise ValueError(
                "Source vocabulary is missing NPZ fields: "
                f"{sorted(missing_fields)}"
            )
        smiles_array = np.array(loaded["smiles"], copy=True)
        embedding_array = np.array(
            loaded["fragment_embeddings"],
            copy=True,
        )

    if smiles_array.ndim != 1:
        raise ValueError(
            "Source smiles must be one-dimensional, got "
            f"{smiles_array.shape}"
        )
    if smiles_array.dtype.kind not in {"U", "S"}:
        raise ValueError(
            "Source smiles must use a NumPy Unicode or bytes dtype, got "
            f"{smiles_array.dtype}"
        )
    if embedding_array.shape != (len(smiles_array), EMBEDDING_DIM):
        raise ValueError(
            "Source fragment_embeddings must have shape "
            f"({len(smiles_array)}, {EMBEDDING_DIM}), got "
            f"{embedding_array.shape}"
        )
    if not np.issubdtype(embedding_array.dtype, np.number):
        raise ValueError("Source fragment_embeddings must be numeric")
    if not np.all(np.isfinite(embedding_array)):
        raise ValueError("Source fragment_embeddings contain NaN or Inf")

    smiles_text: list[str] = []
    for index, value in enumerate(smiles_array):
        if isinstance(value, (bytes, np.bytes_)):
            try:
                text = bytes(value).decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError(
                    f"Source SMILES row {index} is not valid UTF-8"
                ) from exc
        else:
            text = str(value)
        if not text.strip():
            raise ValueError(f"Source SMILES row {index} is empty")
        smiles_text.append(text)

    if not smiles_text:
        raise ValueError("Source vocabulary is empty")
    return smiles_array, embedding_array, smiles_text


def _validate_decoded_template(template: Any) -> None:
    """Confirm that the decoder returned one finite 3D conformer."""

    molecule = template.molecule
    if molecule.GetNumConformers() == 0:
        raise RuntimeError("Decoder returned a molecule without a conformer")

    conformer = molecule.GetConformer(0)
    coordinates = np.asarray(
        [
            list(conformer.GetAtomPosition(atom_index))
            for atom_index in range(molecule.GetNumAtoms())
        ],
        dtype=np.float64,
    )
    expected_shape = (molecule.GetNumAtoms(), 3)
    if coordinates.shape != expected_shape:
        raise RuntimeError(
            "Decoder returned coordinates with shape "
            f"{coordinates.shape}; expected {expected_shape}"
        )
    if not np.all(np.isfinite(coordinates)):
        raise RuntimeError("Decoder returned NaN or Inf coordinates")


def _compact_exception_message(exc: Exception) -> str:
    message = " ".join(str(exc).split())
    if len(message) <= ERROR_MESSAGE_LIMIT:
        return message
    return message[: ERROR_MESSAGE_LIMIT - 3] + "..."


def _inspect_fragment_smiles(
    smiles: str,
) -> tuple[int, list[dict[str, int | str]], int, list[int], int] | None:
    """Return charge, element/ring checks, and bridgehead atom count."""

    molecule = mol_from_smiles_quietly(smiles)
    if molecule is None:
        return None

    total_formal_charge = sum(
        int(atom.GetFormalCharge()) for atom in molecule.GetAtoms()
    )
    disallowed_element_pairs = sorted(
        {
            (int(atom.GetAtomicNum()), str(atom.GetSymbol()))
            for atom in molecule.GetAtoms()
            if int(atom.GetAtomicNum())
            not in ALLOWED_FRAGMENT_ATOMIC_NUMBERS
        }
    )
    disallowed_elements: list[dict[str, int | str]] = [
        {"symbol": symbol, "atomic_number": atomic_number}
        for atomic_number, symbol in disallowed_element_pairs
    ]
    ring_info = molecule.GetRingInfo()
    ring_count = int(ring_info.NumRings())
    disallowed_ring_sizes = sorted(
        {
            len(ring)
            for ring in ring_info.AtomRings()
            if len(ring) in DISALLOWED_RING_SIZES
        }
    )
    bridgehead_atom_count = int(
        rdMolDescriptors.CalcNumBridgeheadAtoms(molecule)
    )
    return (
        total_formal_charge,
        disallowed_elements,
        ring_count,
        disallowed_ring_sizes,
        bridgehead_atom_count,
    )


def _filter_decodable_smiles(
    smiles: Sequence[str],
    *,
    progress_interval: int,
) -> tuple[list[int], list[int], list[int], list[int], list[dict[str, Any]], float]:
    """Return retained/subset indices and filtering failures."""

    retained_indices: list[int] = []
    survival_linker_source_indices: list[int] = []
    activity_modify_source_indices: list[int] = []
    survival_modify_source_indices: list[int] = []
    failures: list[dict[str, Any]] = []
    total_count = len(smiles)
    start_time = time.perf_counter()

    for index, value in enumerate(smiles):
        inspection = _inspect_fragment_smiles(value)
        if inspection is None:
            error_message = "RDKit could not parse the source SMILES"
            failures.append(
                {
                    "source_index": int(index),
                    "smiles": value,
                    "error_type": "InvalidSmiles",
                    "error": error_message,
                }
            )
            print(
                f"[{index + 1}/{total_count}] excluded {value!r}: "
                f"InvalidSmiles: {error_message}",
                file=sys.stderr,
                flush=True,
            )
        elif inspection[0] != 0:
            total_formal_charge = inspection[0]
            error_message = (
                "Total formal charge must be zero; got "
                f"{total_formal_charge}"
            )
            failures.append(
                {
                    "source_index": int(index),
                    "smiles": value,
                    "error_type": "NonzeroTotalFormalCharge",
                    "error": error_message,
                    "total_formal_charge": int(total_formal_charge),
                }
            )
            print(
                f"[{index + 1}/{total_count}] excluded {value!r}: "
                f"NonzeroTotalFormalCharge: {error_message}",
                file=sys.stderr,
                flush=True,
            )
        elif inspection[1]:
            disallowed_elements = inspection[1]
            disallowed_text = ", ".join(
                f"{item['symbol']} (Z={item['atomic_number']})"
                for item in disallowed_elements
            )
            error_message = (
                "Fragment contains elements outside the allowed set: "
                f"{disallowed_text}"
            )
            failures.append(
                {
                    "source_index": int(index),
                    "smiles": value,
                    "error_type": "DisallowedElement",
                    "error": error_message,
                    "disallowed_elements": disallowed_elements,
                }
            )
            print(
                f"[{index + 1}/{total_count}] excluded {value!r}: "
                f"DisallowedElement: {error_message}",
                file=sys.stderr,
                flush=True,
            )
        elif inspection[2] > MAX_FRAGMENT_RINGS:
            ring_count = inspection[2]
            error_message = (
                f"Fragment must have at most {MAX_FRAGMENT_RINGS} rings; "
                f"got {ring_count}"
            )
            failures.append(
                {
                    "source_index": int(index),
                    "smiles": value,
                    "error_type": "TooManyRings",
                    "error": error_message,
                    "ring_count": ring_count,
                }
            )
            print(
                f"[{index + 1}/{total_count}] excluded {value!r}: "
                f"TooManyRings: {error_message}",
                file=sys.stderr,
                flush=True,
            )
        elif inspection[3]:
            disallowed_ring_sizes = inspection[3]
            error_message = (
                "Fragment contains disallowed ring sizes: "
                + ", ".join(str(size) for size in disallowed_ring_sizes)
            )
            failures.append(
                {
                    "source_index": int(index),
                    "smiles": value,
                    "error_type": "DisallowedRingSize",
                    "error": error_message,
                    "disallowed_ring_sizes": disallowed_ring_sizes,
                }
            )
            print(
                f"[{index + 1}/{total_count}] excluded {value!r}: "
                f"DisallowedRingSize: {error_message}",
                file=sys.stderr,
                flush=True,
            )
        elif inspection[4] > 0:
            bridgehead_atom_count = inspection[4]
            error_message = (
                "Fragment contains a bridged ring system with "
                f"{bridgehead_atom_count} bridgehead atom(s)"
            )
            failures.append(
                {
                    "source_index": int(index),
                    "smiles": value,
                    "error_type": "BridgedRing",
                    "error": error_message,
                    "bridgehead_atom_count": bridgehead_atom_count,
                }
            )
            print(
                f"[{index + 1}/{total_count}] excluded {value!r}: "
                f"BridgedRing: {error_message}",
                file=sys.stderr,
                flush=True,
            )
        else:
            try:
                template = build_fragment_conformer_template(value)
                _validate_decoded_template(template)
                heavy_atom_count = sum(
                    1
                    for atom in template.molecule.GetAtoms()
                    if atom.GetAtomicNum() > 1
                )
            except Exception as exc:
                error_message = _compact_exception_message(exc)
                failures.append(
                    {
                        "source_index": int(index),
                        "smiles": value,
                        "error_type": type(exc).__name__,
                        "error": error_message,
                    }
                )
                print(
                    f"[{index + 1}/{total_count}] excluded {value!r}: "
                    f"{type(exc).__name__}: {error_message}",
                    file=sys.stderr,
                    flush=True,
                )
            else:
                retained_indices.append(index)
                if heavy_atom_count <= MAX_SURVIVAL_LINKER_HEAVY_ATOMS:
                    survival_linker_source_indices.append(index)
                if heavy_atom_count <= MAX_ACTIVITY_MODIFY_HEAVY_ATOMS:
                    activity_modify_source_indices.append(index)
                if heavy_atom_count <= MAX_SURVIVAL_MODIFY_HEAVY_ATOMS:
                    survival_modify_source_indices.append(index)

        processed_count = index + 1
        if progress_interval and (
            processed_count % progress_interval == 0
            or processed_count == total_count
        ):
            print(
                f"[{processed_count}/{total_count}] retained="
                f"{len(retained_indices)}, survival_linker="
                f"{len(survival_linker_source_indices)}, activity_modify="
                f"{len(activity_modify_source_indices)}, survival_modify="
                f"{len(survival_modify_source_indices)}, excluded={len(failures)}",
                flush=True,
            )

    elapsed_seconds = time.perf_counter() - start_time
    return (
        retained_indices,
        survival_linker_source_indices,
        activity_modify_source_indices,
        survival_modify_source_indices,
        failures,
        elapsed_seconds,
    )


def _temporary_path(target_path: Path) -> Path:
    return target_path.with_name(
        f".{target_path.stem}.{uuid4().hex}.tmp{target_path.suffix}"
    )


def _atomic_save_npz(
    target_path: Path,
    *,
    smiles: np.ndarray,
    fragment_embeddings: np.ndarray,
) -> None:
    temporary_path = _temporary_path(target_path)
    try:
        np.savez_compressed(
            temporary_path,
            smiles=smiles,
            fragment_embeddings=fragment_embeddings,
        )
        os.replace(temporary_path, target_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _atomic_save_json(target_path: Path, payload: dict[str, Any]) -> None:
    temporary_path = _temporary_path(target_path)
    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as file_obj:
            json.dump(payload, file_obj, ensure_ascii=False, indent=2)
            file_obj.write("\n")
        os.replace(temporary_path, target_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _parse_arguments(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter a fragment NPZ vocabulary by zero total formal charge "
            "and allowed elements, "
            f"allow at most {MAX_FRAGMENT_RINGS} rings, "
            "exclude three- and four-membered rings, "
            "exclude bridged ring systems, "
            "require unaligned 3D decoding success, "
            "then build survival linker, activity modify, and survival modify subsets."
        )
    )
    parser.add_argument(
        "--input-vocab",
        type=Path,
        default=DEFAULT_VOCAB_NPZ_PATH,
        help=f"Source NPZ vocabulary (default: {DEFAULT_VOCAB_NPZ_PATH})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=25,
        help="Print a progress summary every N rows; use 0 to disable.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing output vocabulary and report files.",
    )
    arguments = parser.parse_args(argv)
    if arguments.progress_interval < 0:
        parser.error("--progress-interval must be non-negative")
    return arguments


def run(arguments: argparse.Namespace) -> tuple[Path, Path]:
    source_path = arguments.input_vocab.expanduser().resolve()
    output_dir = arguments.output_dir.expanduser().resolve()
    output_vocab_path = output_dir / OUTPUT_VOCAB_FILENAME
    output_survival_linker_vocab_path = (
        output_dir / OUTPUT_SURVIVAL_LINKER_VOCAB_FILENAME
    )
    output_activity_modify_vocab_path = (
        output_dir / OUTPUT_ACTIVITY_MODIFY_VOCAB_FILENAME
    )
    output_survival_modify_vocab_path = (
        output_dir / OUTPUT_SURVIVAL_MODIFY_VOCAB_FILENAME
    )
    output_report_path = output_dir / OUTPUT_REPORT_FILENAME

    if source_path in {
        output_vocab_path,
        output_survival_linker_vocab_path,
        output_activity_modify_vocab_path,
        output_survival_modify_vocab_path,
    }:
        raise ValueError("Input and output vocabulary paths must differ")

    existing_outputs = [
        path
        for path in (
            output_vocab_path,
            output_survival_linker_vocab_path,
            output_activity_modify_vocab_path,
            output_survival_modify_vocab_path,
            output_report_path,
        )
        if path.exists()
    ]
    if existing_outputs and not arguments.overwrite:
        existing_text = ", ".join(str(path) for path in existing_outputs)
        raise FileExistsError(
            "Output already exists; pass --overwrite to replace it: "
            f"{existing_text}"
        )

    smiles_array, embedding_array, smiles_text = _load_source_vocabulary(
        source_path
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    (
        retained_indices,
        survival_linker_source_indices,
        activity_modify_source_indices,
        survival_modify_source_indices,
        failures,
        elapsed_seconds,
    ) = _filter_decodable_smiles(
        smiles_text,
        progress_interval=int(arguments.progress_interval),
    )
    if not retained_indices:
        raise RuntimeError(
            "No source SMILES could be decoded; no vocabulary was written"
        )

    retained_index_array = np.asarray(retained_indices, dtype=np.int64)
    filtered_smiles = smiles_array[retained_index_array]
    filtered_embeddings = embedding_array[retained_index_array]
    survival_linker_index_array = np.asarray(
        survival_linker_source_indices, dtype=np.int64
    )
    survival_linker_smiles = smiles_array[survival_linker_index_array]
    survival_linker_embeddings = embedding_array[survival_linker_index_array]
    activity_modify_index_array = np.asarray(
        activity_modify_source_indices, dtype=np.int64
    )
    activity_modify_smiles = smiles_array[activity_modify_index_array]
    activity_modify_embeddings = embedding_array[activity_modify_index_array]
    survival_modify_index_array = np.asarray(
        survival_modify_source_indices, dtype=np.int64
    )
    survival_modify_smiles = smiles_array[survival_modify_index_array]
    survival_modify_embeddings = embedding_array[survival_modify_index_array]
    nonzero_charge_excluded_count = sum(
        failure.get("error_type") == "NonzeroTotalFormalCharge"
        for failure in failures
    )
    disallowed_element_excluded_count = sum(
        failure.get("error_type") == "DisallowedElement"
        for failure in failures
    )
    invalid_smiles_excluded_count = sum(
        failure.get("error_type") == "InvalidSmiles"
        for failure in failures
    )
    too_many_rings_excluded_count = sum(
        failure.get("error_type") == "TooManyRings"
        for failure in failures
    )
    disallowed_ring_size_excluded_count = sum(
        failure.get("error_type") == "DisallowedRingSize"
        for failure in failures
    )
    bridged_ring_excluded_count = sum(
        failure.get("error_type") == "BridgedRing"
        for failure in failures
    )

    report: dict[str, Any] = {
        "format_version": 1,
        "source_vocabulary": str(source_path),
        "output_vocabulary": str(output_vocab_path),
        "output_survival_linker_vocabulary": str(
            output_survival_linker_vocab_path
        ),
        "output_activity_modify_vocabulary": str(
            output_activity_modify_vocab_path
        ),
        "output_survival_modify_vocabulary": str(
            output_survival_modify_vocab_path
        ),
        "decoder_entrypoint": (
            "PhiSLinker_atom_pair_smiles_decoder."
            "build_fragment_conformer_template"
        ),
        "centroid_translation_applied": False,
        "reference_frame_alignment_applied": False,
        "required_total_formal_charge": 0,
        "max_fragment_rings": MAX_FRAGMENT_RINGS,
        "disallowed_ring_sizes": list(DISALLOWED_RING_SIZES),
        "bridged_rings_allowed": False,
        "allowed_element_symbols": [
            symbol for symbol, _ in ALLOWED_FRAGMENT_ELEMENTS
        ],
        "source_count": len(smiles_text),
        "retained_count": len(retained_indices),
        "survival_linker_count": len(survival_linker_source_indices),
        "survival_linker_max_heavy_atoms": MAX_SURVIVAL_LINKER_HEAVY_ATOMS,
        "activity_modify_count": len(activity_modify_source_indices),
        "activity_modify_max_heavy_atoms": MAX_ACTIVITY_MODIFY_HEAVY_ATOMS,
        "survival_modify_count": len(survival_modify_source_indices),
        "survival_modify_max_heavy_atoms": MAX_SURVIVAL_MODIFY_HEAVY_ATOMS,
        "excluded_count": len(failures),
        "nonzero_total_formal_charge_excluded_count": (
            nonzero_charge_excluded_count
        ),
        "disallowed_element_excluded_count": (
            disallowed_element_excluded_count
        ),
        "invalid_smiles_excluded_count": invalid_smiles_excluded_count,
        "too_many_rings_excluded_count": too_many_rings_excluded_count,
        "disallowed_ring_size_excluded_count": (
            disallowed_ring_size_excluded_count
        ),
        "bridged_ring_excluded_count": bridged_ring_excluded_count,
        "embedding_dimension": int(embedding_array.shape[1]),
        "embedding_dtype": str(embedding_array.dtype),
        "elapsed_seconds": round(float(elapsed_seconds), 6),
        "retained_source_indices": retained_indices,
        "survival_linker_source_indices": survival_linker_source_indices,
        "activity_modify_source_indices": activity_modify_source_indices,
        "survival_modify_source_indices": survival_modify_source_indices,
        "failures": failures,
    }

    _atomic_save_npz(
        output_vocab_path,
        smiles=filtered_smiles,
        fragment_embeddings=filtered_embeddings,
    )
    _atomic_save_npz(
        output_survival_linker_vocab_path,
        smiles=survival_linker_smiles,
        fragment_embeddings=survival_linker_embeddings,
    )
    _atomic_save_npz(
        output_activity_modify_vocab_path,
        smiles=activity_modify_smiles,
        fragment_embeddings=activity_modify_embeddings,
    )
    _atomic_save_npz(
        output_survival_modify_vocab_path,
        smiles=survival_modify_smiles,
        fragment_embeddings=survival_modify_embeddings,
    )
    _atomic_save_json(output_report_path, report)

    print(
        "Completed: "
        f"retained={len(retained_indices)}, "
        f"survival_linker={len(survival_linker_source_indices)}, "
        f"activity_modify={len(activity_modify_source_indices)}, "
        f"survival_modify={len(survival_modify_source_indices)}, "
        f"excluded={len(failures)}, "
        f"nonzero_charge_excluded={nonzero_charge_excluded_count}, "
        f"disallowed_element_excluded="
        f"{disallowed_element_excluded_count}, "
        f"invalid_smiles_excluded={invalid_smiles_excluded_count}, "
        f"too_many_rings_excluded={too_many_rings_excluded_count}, "
        f"disallowed_ring_size_excluded={disallowed_ring_size_excluded_count}, "
        f"bridged_ring_excluded={bridged_ring_excluded_count}, "
        f"elapsed={elapsed_seconds:.2f}s"
    )
    print(f"Vocabulary: {output_vocab_path}")
    print(f"Survival linker vocabulary: {output_survival_linker_vocab_path}")
    print(f"Activity modify vocabulary: {output_activity_modify_vocab_path}")
    print(f"Survival modify vocabulary: {output_survival_modify_vocab_path}")
    print(f"Report: {output_report_path}")
    return output_vocab_path, output_report_path


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_arguments(argv)
    try:
        run(arguments)
    except Exception as exc:
        print(
            f"ERROR: {type(exc).__name__}: {_compact_exception_message(exc)}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
