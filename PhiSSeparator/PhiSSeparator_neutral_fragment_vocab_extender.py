"""Append neutral counterparts of charged fragments to standard vocabularies.

The original charged entries are retained.  This script only discovers
neutral counterparts; all vocabulary mutations are delegated to the existing
fragment and atom embedding resolvers so their append-only metadata, locks,
transactions, JSON mirrors, and compatibility anchors remain authoritative.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Iterator, Sequence
from typing import TypeVar

import numpy as np
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

try:
    from PhiSSeparator_atom_encoder_resolver import (
        DEFAULT_EMBEDDING_TABLE as ATOM_EMBEDDING_TABLE_PATH,
        AtomEmbeddingResolver,
    )
    from PhiSSeparator_fragment_encoder_resolver import (
        EMBEDDING_TABLE_PATH as FRAGMENT_EMBEDDING_TABLE_PATH,
        FragmentEmbeddingResolver,
    )
except ImportError:  # 支持作为 ChemCut 子模块导入
    from .PhiSSeparator_atom_encoder_resolver import (
        DEFAULT_EMBEDDING_TABLE as ATOM_EMBEDDING_TABLE_PATH,
        AtomEmbeddingResolver,
    )
    from .PhiSSeparator_fragment_encoder_resolver import (
        EMBEDDING_TABLE_PATH as FRAGMENT_EMBEDDING_TABLE_PATH,
        FragmentEmbeddingResolver,
    )


_ProgressValue = TypeVar("_ProgressValue")


def _render_progress(
    description: str,
    completed: int,
    total: int,
) -> None:
    width = 30
    ratio = 1.0 if total == 0 else min(max(completed / total, 0.0), 1.0)
    filled = int(width * ratio)
    bar = "#" * filled + "-" * (width - filled)
    percentage = ratio * 100.0
    print(
        f"\r{description} |{bar}| {completed}/{total} ({percentage:5.1f}%)",
        end="\n" if completed >= total else "",
        file=sys.stderr,
        flush=True,
    )


def _progress_iterable(
    values: Sequence[_ProgressValue],
    *,
    description: str,
) -> Iterator[_ProgressValue]:
    total = len(values)
    if total == 0:
        _render_progress(description, 0, 0)
        return

    last_percentage = -1
    for completed, value in enumerate(values, start=1):
        yield value
        percentage = int(completed * 100 / total)
        if percentage != last_percentage or completed == total:
            _render_progress(description, completed, total)
            last_percentage = percentage


def _text_value(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _load_vocabulary_smiles(path: str) -> list[str]:
    resolved = os.path.abspath(os.fspath(path))
    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"标准词表不存在: {resolved}")
    with np.load(resolved, allow_pickle=False) as payload:
        if "smiles" not in payload.files:
            raise KeyError(f"标准词表缺少 smiles 数组: {resolved}")
        return [_text_value(value) for value in payload["smiles"].tolist()]


def _mol_from_smiles(smiles: str) -> Chem.Mol | None:
    return Chem.MolFromSmiles(smiles)


def _total_formal_charge(molecule: Chem.Mol) -> int:
    return sum(int(atom.GetFormalCharge()) for atom in molecule.GetAtoms())


def _neutral_counterpart(
    smiles: str,
    uncharger: rdMolStandardize.Uncharger,
) -> tuple[int | None, str | None, str | None]:
    """Return source charge, neutral canonical SMILES, and failure reason."""

    molecule = _mol_from_smiles(smiles)
    if molecule is None:
        return None, None, "rdkit_parse_failed"

    source_charge = _total_formal_charge(molecule)
    if source_charge == 0:
        return 0, None, None

    neutralization_input = Chem.Mol(molecule)
    Chem.RemoveStereochemistry(neutralization_input)
    remove_hydrogen_parameters = Chem.RemoveHsParameters()
    remove_hydrogen_parameters.removeNontetrahedralNeighbors = True

    try:
        neutralization_input = Chem.RemoveHs(
            neutralization_input,
            remove_hydrogen_parameters,
            sanitize=True,
        )
        neutral = uncharger.uncharge(neutralization_input)
        Chem.SanitizeMol(neutral)
    except Exception as exc:
        return (
            source_charge,
            None,
            f"neutralization_failed: {type(exc).__name__}: {exc}",
        )

    neutral_charge = _total_formal_charge(neutral)
    if neutral_charge != 0:
        return (
            source_charge,
            None,
            f"remaining_formal_charge={neutral_charge}",
        )
    if neutral.GetNumHeavyAtoms() != molecule.GetNumHeavyAtoms():
        return source_charge, None, "heavy_atom_count_changed"

    try:
        neutral_smiles = Chem.MolToSmiles(
            neutral,
            canonical=True,
            isomericSmiles=False,
        )
    except Exception as exc:
        return (
            source_charge,
            None,
            f"canonicalization_failed: {type(exc).__name__}: {exc}",
        )
    reparsed = _mol_from_smiles(neutral_smiles)
    if reparsed is None or _total_formal_charge(reparsed) != 0:
        return source_charge, None, "neutral_canonical_smiles_validation_failed"
    return source_charge, neutral_smiles, None


def _resolution_summary(value: dict[str, object] | None) -> dict[str, object] | None:
    if value is None:
        return None
    keys = (
        "requested_count",
        "canonical_smiles",
        "source",
        "newly_added_to_standard_table",
        "newly_added_to_raw_corpus",
    )
    return {key: value.get(key) for key in keys}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "扫描标准片段/原子词表，并通过两个 resolver 追加带电片段的"
            "零总形式电荷对应物"
        )
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="两个冻结编码器使用的设备，默认 auto",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只报告中和映射，不实例化 resolver 或写入词表",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fragment_smiles = _load_vocabulary_smiles(
        FRAGMENT_EMBEDDING_TABLE_PATH
    )
    atom_smiles = _load_vocabulary_smiles(ATOM_EMBEDDING_TABLE_PATH)
    fragment_set = set(fragment_smiles)
    atom_set = set(atom_smiles)
    source_smiles = list(
        dict.fromkeys([*fragment_smiles, *atom_smiles])
    )

    uncharger = rdMolStandardize.Uncharger()
    neutral_mappings: list[dict[str, object]] = []
    unresolved_charged: list[dict[str, object]] = []
    unparsed_smiles: list[str] = []
    charged_fragment_count = 0

    for smiles in _progress_iterable(
        source_smiles,
        description="扫描并中和词表",
    ):
        source_charge, neutral_smiles, failure_reason = (
            _neutral_counterpart(smiles, uncharger)
        )
        if source_charge is None:
            unparsed_smiles.append(smiles)
            continue
        if source_charge == 0:
            continue

        charged_fragment_count += 1
        if neutral_smiles is None:
            unresolved_charged.append(
                {
                    "source_smiles": smiles,
                    "source_formal_charge": source_charge,
                    "reason": failure_reason,
                }
            )
            continue
        neutral_mappings.append(
            {
                "source_smiles": smiles,
                "source_formal_charge": source_charge,
                "neutral_smiles": neutral_smiles,
                "neutral_formal_charge": 0,
                "neutral_present_in_fragment_vocabulary": (
                    neutral_smiles in fragment_set
                ),
                "neutral_present_in_atom_vocabulary": (
                    neutral_smiles in atom_set
                ),
            }
        )

    neutral_smiles_values = list(
        dict.fromkeys(
            str(record["neutral_smiles"])
            for record in neutral_mappings
        )
    )
    atom_resolution = None
    fragment_resolution = None
    if neutral_smiles_values and not args.dry_run:
        # 两个 resolver 都先完成初始化和一致性检查，再开始追加。
        resolver_stage_total = 4
        _render_progress("追加中和词表", 0, resolver_stage_total)
        atom_resolver = AtomEmbeddingResolver(device=args.device)
        _render_progress("追加中和词表", 1, resolver_stage_total)
        fragment_resolver = FragmentEmbeddingResolver(
            device_spec=args.device
        )
        _render_progress("追加中和词表", 2, resolver_stage_total)

        # 原子编码器的元素准入更严格，优先执行可减少无效的片段追加。
        atom_resolver.resolve_many(neutral_smiles_values)
        _render_progress("追加中和词表", 3, resolver_stage_total)
        atom_resolution = _resolution_summary(
            atom_resolver.last_resolution
        )
        fragment_resolver.resolve_many(neutral_smiles_values)
        _render_progress("追加中和词表", 4, resolver_stage_total)
        fragment_resolution = _resolution_summary(
            fragment_resolver.last_resolution
        )

    report = {
        "status": "dry_run" if args.dry_run else "applied",
        "fragment_vocabulary_path": os.path.abspath(
            os.fspath(FRAGMENT_EMBEDDING_TABLE_PATH)
        ),
        "atom_vocabulary_path": os.path.abspath(
            os.fspath(ATOM_EMBEDDING_TABLE_PATH)
        ),
        "fragment_vocabulary_smiles_count": len(fragment_smiles),
        "atom_vocabulary_smiles_count": len(atom_smiles),
        "scanned_unique_smiles_count": len(source_smiles),
        "charged_fragment_count": charged_fragment_count,
        "neutralizable_mapping_count": len(neutral_mappings),
        "unique_neutral_smiles_count": len(neutral_smiles_values),
        "unresolved_charged_fragment_count": len(unresolved_charged),
        "unparsed_smiles_count": len(unparsed_smiles),
        "neutral_mappings": neutral_mappings,
        "unresolved_charged_fragments": unresolved_charged,
        "unparsed_smiles": unparsed_smiles,
        "atom_resolver": atom_resolution,
        "fragment_resolver": fragment_resolution,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
