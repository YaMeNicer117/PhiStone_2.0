#!/usr/bin/env python3
"""Filter, deduplicate, crop, and merge HiQBind and PDBbind complexes.

The script never modifies source data.  Its default is ``--mode copy`` so it
can be run directly from ``Datas/Pre_datas``.  Use ``--mode dry-run`` to write
validation reports without materializing complex files.

Deduplication key
-----------------
``(lower-case PDB ID, charge/protonation-normalized canonical isomeric SMILES)``

SMILES are always generated from the selected ligand SDF.  Stereochemistry and
isotopes are retained.  Tautomers are not normalized.  HiQBind wins a duplicate
group when its refined ligand and receptor both pass validation.

Pocket definition
-----------------
If any atom of a receptor residue is within the cutoff of any ligand atom, the
entire residue is retained.  Water is excluded by default.  Non-water HETATM
records, including metals and cofactors, are retained with the same rule.
Relevant LINK, SSBOND, ANISOU, and CONECT records are preserved.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import re
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Sequence

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem.MolStandardize import rdMolStandardize
except ImportError:  # A clear message is emitted from main().
    Chem = None  # type: ignore[assignment]
    RDLogger = None  # type: ignore[assignment]
    rdMolStandardize = None  # type: ignore[assignment]


LOGGER = logging.getLogger("prepare_hiqbind_pdbbind")
PDB_ID_RE = re.compile(r"^[A-Za-z0-9]{4}$")

SOURCE_PRIORITY = {
    "hiqbind": 0,
    "pdbbind_pl": 1,
    "pdbbind_nl": 2,
}

DEFAULT_WATER_RESNAMES = {
    "D2O",
    "DOD",
    "H2O",
    "HOH",
    "OH2",
    "SOL",
    "TIP",
    "TIP3",
    "TIP4",
    "WAT",
}

# These are standalone small molecules, not functional groups inside a larger
# ligand.  The comparison is made against the complete largest fragment.
DEFAULT_EXCLUDED_SMALL_MOLECULE_SMILES = (
    "O=C([O-])[O-]",          # carbonate
    "[C-]#N",                 # cyanide
    "N=C=O",                  # cyanate/isocyanate parent
    "N=C=S",                  # thiocyanate/isothiocyanate parent
    "O=C=O",                  # carbon dioxide
)

# Charge, protonation, isotope, and stereochemical labels are ignored only for
# matching these complete standalone acid structures.  They remain significant
# in the normal ligand deduplication key where applicable.
DEFAULT_EXCLUDED_ORGANIC_ACID_SMILES = (
    "OC=O",                         # formic acid / formate
    "CC(=O)O",                      # acetic acid / acetate
    "CCC(=O)O",                     # propionic acid / propionate
    "CC(O)C(=O)O",                  # lactic acid / lactate
    "O=C(O)C(=O)O",                 # oxalic acid / oxalate
    "O=C(O)CC(=O)O",                # malonic acid / malonate
    "O=C(O)CCC(=O)O",               # succinic acid / succinate
    "O=C(O)C=CC(=O)O",              # maleic/fumaric acid
    "O=C(O)CC(O)C(=O)O",            # malic acid / malate
    "O=C(O)C(O)C(O)C(=O)O",         # tartaric acid / tartrate
    "O=C(O)CC(O)(CC(=O)O)C(=O)O",   # citric acid / citrate
    "OCC(=O)O",                     # glycolic acid / glycolate
    "CC(=O)C(=O)O",                 # pyruvic acid / pyruvate
    "O=C(O)CCC(=O)C(=O)O",          # alpha-ketoglutaric acid
)

# Metals only.  Metalloids such as B, Si, As, and Te are intentionally absent.
METAL_ATOMIC_NUMBERS = frozenset(
    {
        3, 4, 11, 12, 13, 19, 20, 31, 37, 38, 49, 50, 55, 56,
        81, 82, 83, 84, 87, 88, 113, 114, 115, 116,
    }
    | set(range(21, 31))
    | set(range(39, 49))
    | set(range(57, 81))
    | set(range(89, 113))
)


class ValidationError(RuntimeError):
    """Expected per-complex validation failure."""

    def __init__(self, reason: str, detail: str):
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


@dataclass(frozen=True)
class Candidate:
    source: str
    pdb_id: str
    complex_id: str
    ligand_path: Path
    receptor_path: Path
    additive_path: Path | None = None


@dataclass(frozen=True)
class LigandInfo:
    candidate: Candidate
    canonical_smiles: str
    dedup_smiles: str
    atom_count: int
    heavy_atom_count: int
    fragment_count: int
    formal_charge: int


@dataclass(frozen=True)
class ExcludedFragmentKeys:
    small_molecules: frozenset[str]
    organic_acids: frozenset[str]


@dataclass(frozen=True)
class Rejection:
    source: str
    pdb_id: str
    complex_id: str
    ligand_path: Path | None
    receptor_path: Path | None
    stage: str
    reason: str
    detail: str


@dataclass(frozen=True)
class ResidueKey:
    chain_id: str
    residue_number: str
    insertion_code: str
    residue_name: str


@dataclass(frozen=True)
class AtomEntry:
    line_index: int
    source_index: int
    record_name: str
    serial: str
    residue: ResidueKey
    xyz: tuple[float, float, float]


@dataclass(frozen=True)
class PocketResult:
    lines: tuple[str, ...]
    atom_count: int
    polymer_residue_count: int
    hetatm_residue_count: int
    used_first_model_only: bool


def normalize_pdb_id(value: str) -> str:
    pdb_id = value.strip().lower()
    if not PDB_ID_RE.fullmatch(pdb_id):
        raise ValueError(f"invalid PDB ID: {value!r}")
    return pdb_id


def path_for_report(path: Path | None, repository_root: Path) -> str:
    if path is None:
        return ""
    try:
        return str(path.resolve().relative_to(repository_root.resolve()))
    except ValueError:
        return str(path.resolve())


def require_path(path: Path, expected: str) -> None:
    if expected == "file" and not path.is_file():
        raise FileNotFoundError(f"required file not found: {path}")
    if expected == "directory" and not path.is_dir():
        raise FileNotFoundError(f"required directory not found: {path}")


def read_hiqbind_index_pdb_ids(csv_path: Path) -> set[str]:
    """Read only the PDBID field; ligand SMILES in the CSV are never used."""
    pdb_ids: set[str] = set()
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "PDBID" not in reader.fieldnames:
            raise ValueError(f"HiQBind index has no PDBID column: {csv_path}")
        for line_number, row in enumerate(reader, start=2):
            raw = (row.get("PDBID") or "").strip()
            if not raw:
                continue
            try:
                pdb_ids.add(normalize_pdb_id(raw))
            except ValueError as exc:
                raise ValueError(
                    f"invalid PDBID at {csv_path}:{line_number}: {raw!r}"
                ) from exc
    return pdb_ids


def read_pdbbind_index_pdb_ids(index_path: Path) -> set[str]:
    pdb_ids: set[str] = set()
    with index_path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            first_field = stripped.split(maxsplit=1)[0]
            if PDB_ID_RE.fullmatch(first_field):
                pdb_ids.add(first_field.lower())
    if not pdb_ids:
        raise ValueError(f"no PDB IDs found in index: {index_path}")
    return pdb_ids


def choose_expected_file(
    directory: Path,
    exact_name: str,
    suffix: str,
) -> tuple[Path | None, str]:
    exact = directory / exact_name
    if exact.is_file():
        return exact, ""
    matches = sorted(
        (p for p in directory.iterdir() if p.is_file() and p.name.lower().endswith(suffix)),
        key=lambda p: p.name.lower(),
    )
    if len(matches) == 1:
        return matches[0], ""
    if not matches:
        return None, f"no file ending with {suffix!r}"
    return None, f"multiple files ending with {suffix!r}: {[p.name for p in matches]}"


def discover_hiqbind(
    index_pdb_ids: set[str],
    root: Path,
) -> tuple[list[Candidate], list[Rejection]]:
    candidates: list[Candidate] = []
    issues: list[Rejection] = []
    seen_pdb_ids: set[str] = set()

    for pdb_directory in sorted(
        (p for p in root.iterdir() if p.is_dir()), key=lambda p: p.name.lower()
    ):
        try:
            pdb_id = normalize_pdb_id(pdb_directory.name)
        except ValueError:
            issues.append(
                Rejection(
                    "hiqbind", "", pdb_directory.name, None, None,
                    "discovery", "unexpected_pdb_directory",
                    f"directory name is not a four-character PDB ID: {pdb_directory.name}",
                )
            )
            continue

        seen_pdb_ids.add(pdb_id)
        if pdb_id not in index_pdb_ids:
            issues.append(
                Rejection(
                    "hiqbind", pdb_id, pdb_directory.name, None, None,
                    "discovery", "pdbid_not_in_hiqbind_index",
                    "PDB directory is not listed in the HiQBind index",
                )
            )
            continue

        complex_directories = sorted(
            (p for p in pdb_directory.iterdir() if p.is_dir()),
            key=lambda p: p.name.lower(),
        )
        if not complex_directories:
            issues.append(
                Rejection(
                    "hiqbind", pdb_id, pdb_directory.name, None, None,
                    "discovery", "no_complex_directory",
                    "PDB directory contains no complex subdirectory",
                )
            )
            continue

        for complex_directory in complex_directories:
            complex_id = complex_directory.name
            ligand_path, ligand_error = choose_expected_file(
                complex_directory,
                f"{complex_id}_ligand_refined.sdf",
                "_ligand_refined.sdf",
            )
            receptor_path, receptor_error = choose_expected_file(
                complex_directory,
                f"{complex_id}_protein_refined.pdb",
                "_protein_refined.pdb",
            )
            additive_path = complex_directory / f"{complex_id}_hetatm.pdb"
            additive_error = "" if additive_path.is_file() else (
                f"missing additive file {additive_path.name!r}"
            )
            if ligand_path is None or receptor_path is None or additive_error:
                detail = "; ".join(
                    x for x in (ligand_error, receptor_error, additive_error) if x
                )
                issues.append(
                    Rejection(
                        "hiqbind", pdb_id, complex_id, ligand_path, receptor_path,
                        "discovery", "missing_or_ambiguous_structure_file", detail,
                    )
                )
                continue
            candidates.append(
                Candidate(
                    "hiqbind", pdb_id, complex_id, ligand_path, receptor_path,
                    additive_path,
                )
            )

    for missing_id in sorted(index_pdb_ids - seen_pdb_ids):
        issues.append(
            Rejection(
                "hiqbind", missing_id, missing_id, None, None,
                "discovery", "indexed_pdbid_directory_not_found",
                "PDB ID occurs in the HiQBind index but has no data directory",
            )
        )
    return candidates, issues


def build_pdbbind_directory_map(roots: Sequence[Path]) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = defaultdict(list)
    for root in roots:
        for directory in root.iterdir():
            if directory.is_dir() and PDB_ID_RE.fullmatch(directory.name):
                result[directory.name.lower()].append(directory)
    return result


def discover_pdbbind(
    source: str,
    index_pdb_ids: set[str],
    roots: Sequence[Path],
    receptor_suffix: str,
) -> tuple[list[Candidate], list[Rejection]]:
    candidates: list[Candidate] = []
    issues: list[Rejection] = []
    directories = build_pdbbind_directory_map(roots)

    for pdb_id in sorted(index_pdb_ids):
        matched_directories = sorted(directories.get(pdb_id, []), key=lambda p: str(p).lower())
        if not matched_directories:
            issues.append(
                Rejection(
                    source, pdb_id, pdb_id, None, None,
                    "discovery", "indexed_pdbid_directory_not_found",
                    "PDB ID occurs in the PDBbind index but has no data directory",
                )
            )
            continue

        for directory in matched_directories:
            ligand_path, ligand_error = choose_expected_file(
                directory, f"{pdb_id}_ligand.sdf", "_ligand.sdf"
            )
            receptor_path, receptor_error = choose_expected_file(
                directory, f"{pdb_id}{receptor_suffix}", receptor_suffix
            )
            if ligand_path is None or receptor_path is None:
                detail = "; ".join(x for x in (ligand_error, receptor_error) if x)
                issues.append(
                    Rejection(
                        source, pdb_id, directory.name, ligand_path, receptor_path,
                        "discovery", "missing_or_ambiguous_structure_file", detail,
                    )
                )
                continue
            candidates.append(
                Candidate(source, pdb_id, directory.name, ligand_path, receptor_path)
            )
    return candidates, issues


def remove_nonisotopic_hydrogens(mol):
    params = Chem.RemoveHsParameters()
    params.removeIsotopes = False
    return Chem.RemoveHs(mol, params, sanitize=True)


def canonical_isomeric_smiles(mol, neutralize: bool) -> str:
    work = Chem.Mol(mol)
    if neutralize:
        # ChargeParent standardizes common protonation states more completely
        # than Uncharger alone (notably for zwitterions).  Process each
        # disconnected SDF fragment separately so counterions/components are
        # not silently discarded from the deduplication identity.
        component_smiles: list[str] = []
        fragments = Chem.GetMolFrags(work, asMols=True, sanitizeFrags=True)
        for fragment in fragments:
            normalized = rdMolStandardize.ChargeParent(fragment)
            normalized = remove_nonisotopic_hydrogens(normalized)
            Chem.AssignStereochemistry(normalized, cleanIt=True, force=True)
            component = Chem.MolToSmiles(
                normalized, canonical=True, isomericSmiles=True
            )
            if not component:
                raise ValidationError(
                    "smiles_generation_failed",
                    "RDKit returned an empty normalized component SMILES",
                )
            component_smiles.append(component)
        smiles = ".".join(sorted(component_smiles))
    else:
        work = remove_nonisotopic_hydrogens(work)
        Chem.AssignStereochemistry(work, cleanIt=True, force=True)
        smiles = Chem.MolToSmiles(work, canonical=True, isomericSmiles=True)
    if not smiles:
        raise ValidationError("smiles_generation_failed", "RDKit returned an empty SMILES")
    return smiles


def exclusion_parent_smiles(mol) -> str:
    """Canonical parent key used only for whole-fragment exclusion matching."""
    normalized = rdMolStandardize.ChargeParent(Chem.Mol(mol))
    normalized = remove_nonisotopic_hydrogens(normalized)
    smiles = Chem.MolToSmiles(
        normalized,
        canonical=True,
        isomericSmiles=False,
        doRandom=False,
    )
    if not smiles:
        raise ValidationError(
            "exclusion_smiles_generation_failed",
            "RDKit returned an empty exclusion parent SMILES",
        )
    return smiles


def read_single_sdf_molecule(path: Path):
    try:
        supplier = Chem.SDMolSupplier(
            str(path), removeHs=False, sanitize=False, strictParsing=True
        )
        record_count = len(supplier)
    except Exception as exc:
        raise ValidationError("sdf_read_failed", str(exc)) from exc

    if record_count != 1:
        raise ValidationError(
            "sdf_record_count_not_one",
            f"expected exactly one SDF record, found {record_count}",
        )
    try:
        raw_mol = supplier[0]
    except Exception as exc:
        raise ValidationError("sdf_parse_failed", str(exc)) from exc
    if raw_mol is None:
        raise ValidationError("sdf_parse_failed", "RDKit could not parse the SDF record")

    mol = Chem.Mol(raw_mol)
    try:
        Chem.SanitizeMol(mol)
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    except Exception as exc:
        raise ValidationError("sdf_sanitize_failed", str(exc)) from exc
    return mol


def molecule_coordinates(mol) -> tuple[tuple[float, float, float], ...]:
    if mol.GetNumConformers() != 1:
        raise ValidationError(
            "sdf_conformer_count_not_one",
            f"expected one conformer, found {mol.GetNumConformers()}",
        )
    conformer = mol.GetConformer(0)
    coordinates: list[tuple[float, float, float]] = []
    for atom_index in range(mol.GetNumAtoms()):
        point = conformer.GetAtomPosition(atom_index)
        xyz = (float(point.x), float(point.y), float(point.z))
        if not all(math.isfinite(value) for value in xyz):
            raise ValidationError(
                "nonfinite_ligand_coordinate",
                f"atom {atom_index} has non-finite coordinates: {xyz}",
            )
        coordinates.append(xyz)

    for bond in mol.GetBonds():
        if bond.GetBondType() == Chem.BondType.UNSPECIFIED:
            raise ValidationError(
                "unspecified_bond_type",
                f"bond {bond.GetIdx()} has an unspecified bond type",
            )
        first = coordinates[bond.GetBeginAtomIdx()]
        second = coordinates[bond.GetEndAtomIdx()]
        distance_sq = sum((a - b) ** 2 for a, b in zip(first, second))
        if distance_sq < 0.01 or distance_sq > 25.0:
            raise ValidationError(
                "implausible_bond_length",
                f"bond {bond.GetIdx()} length is {math.sqrt(distance_sq):.3f} A",
            )
    return tuple(coordinates)


def inspect_ligand(
    candidate: Candidate,
    excluded_fragment_keys: ExcludedFragmentKeys,
) -> tuple[LigandInfo, object]:
    mol = read_single_sdf_molecule(candidate.ligand_path)
    atoms = tuple(mol.GetAtoms())
    if any(atom.GetAtomicNum() == 0 for atom in atoms):
        raise ValidationError("dummy_atom", "ligand contains a dummy/unknown atom")

    heavy_atom_count = sum(atom.GetAtomicNum() > 1 for atom in atoms)
    if heavy_atom_count < 2:
        raise ValidationError(
            "fewer_than_two_non_hydrogen_atoms",
            f"ligand has {heavy_atom_count} non-hydrogen atom(s)",
        )

    fragment_mols = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if not fragment_mols:
        raise ValidationError("no_molecular_fragment", "ligand has no molecular fragment")
    main_fragment = max(
        fragment_mols,
        key=lambda fragment: (
            sum(atom.GetAtomicNum() > 1 for atom in fragment.GetAtoms()),
            fragment.GetNumAtoms(),
        ),
    )
    main_heavy_atoms = sum(
        atom.GetAtomicNum() > 1 for atom in main_fragment.GetAtoms()
    )
    if main_heavy_atoms < 2:
        raise ValidationError(
            "no_connected_fragment_with_two_heavy_atoms",
            "no connected ligand fragment contains at least two non-hydrogen atoms",
        )
    main_fragment_metals = sorted(
        {
            atom.GetSymbol()
            for atom in main_fragment.GetAtoms()
            if atom.GetAtomicNum() in METAL_ATOMIC_NUMBERS
        }
    )
    if main_fragment_metals:
        raise ValidationError(
            "metal_containing_ligand",
            "largest ligand fragment contains metal atom(s): "
            + ",".join(main_fragment_metals),
        )
    if not any(atom.GetAtomicNum() == 6 for atom in main_fragment.GetAtoms()):
        raise ValidationError(
            "inorganic_or_metal_ligand",
            "largest ligand fragment contains no carbon atom",
        )

    # Coordinate and bond checks are applied before a SMILES key is accepted.
    molecule_coordinates(mol)

    canonical_smiles = canonical_isomeric_smiles(mol, neutralize=False)
    dedup_smiles = canonical_isomeric_smiles(mol, neutralize=True)
    main_fragment_key = exclusion_parent_smiles(main_fragment)
    if main_fragment_key in excluded_fragment_keys.organic_acids:
        raise ValidationError(
            "standalone_small_organic_acid",
            f"excluded complete organic-acid fragment: {main_fragment_key}",
        )
    if main_fragment_key in excluded_fragment_keys.small_molecules:
        raise ValidationError(
            "standalone_small_ion_or_acid_residue",
            f"excluded largest fragment: {main_fragment_key}",
        )

    info = LigandInfo(
        candidate=candidate,
        canonical_smiles=canonical_smiles,
        dedup_smiles=dedup_smiles,
        atom_count=mol.GetNumAtoms(),
        heavy_atom_count=heavy_atom_count,
        fragment_count=len(fragment_mols),
        formal_charge=Chem.GetFormalCharge(mol),
    )
    return info, mol


def exclusion_keys_from_smiles(smiles_values: Iterable[str]) -> frozenset[str]:
    keys: set[str] = set()
    for value in smiles_values:
        mol = Chem.MolFromSmiles(value)
        if mol is None:
            raise ValueError(f"invalid excluded SMILES: {value!r}")
        keys.add(exclusion_parent_smiles(mol))
    return frozenset(keys)


def load_excluded_fragment_keys(
    extra_smiles_file: Path | None,
) -> ExcludedFragmentKeys:
    small_molecule_values = list(DEFAULT_EXCLUDED_SMALL_MOLECULE_SMILES)
    if extra_smiles_file is not None:
        with extra_smiles_file.open("r", encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, start=1):
                value = line.split("#", 1)[0].strip()
                if value:
                    small_molecule_values.append(value)

    return ExcludedFragmentKeys(
        small_molecules=exclusion_keys_from_smiles(small_molecule_values),
        organic_acids=exclusion_keys_from_smiles(
            DEFAULT_EXCLUDED_ORGANIC_ACID_SMILES
        ),
    )


def first_model_bounds(lines: Sequence[str]) -> tuple[int, int, bool]:
    model_starts = [i for i, line in enumerate(lines) if line.startswith("MODEL ")]
    if not model_starts:
        return 0, len(lines), False
    start = model_starts[0] + 1
    end = next(
        (i for i in range(start, len(lines)) if lines[i].startswith("ENDMDL")),
        len(lines),
    )
    return start, end, len(model_starts) > 1


def atom_residue_key(line: str) -> ResidueKey:
    padded = line.ljust(80)
    return ResidueKey(
        chain_id=padded[21:22].strip(),
        residue_number=padded[22:26].strip(),
        insertion_code=padded[26:27].strip(),
        residue_name=padded[17:20].strip().upper(),
    )


def parse_atom_entry(line_index: int, source_index: int, line: str) -> AtomEntry:
    padded = line.ljust(80)
    try:
        xyz = (
            float(padded[30:38]),
            float(padded[38:46]),
            float(padded[46:54]),
        )
    except ValueError as exc:
        raise ValidationError(
            "invalid_receptor_coordinate",
            f"cannot parse receptor coordinates on line {line_index + 1}",
        ) from exc
    if not all(math.isfinite(value) for value in xyz):
        raise ValidationError(
            "nonfinite_receptor_coordinate",
            f"line {line_index + 1} has non-finite coordinates: {xyz}",
        )
    return AtomEntry(
        line_index=line_index,
        source_index=source_index,
        record_name=padded[0:6].strip(),
        serial=padded[6:11].strip(),
        residue=atom_residue_key(padded),
        xyz=xyz,
    )


def build_ligand_grid(
    ligand_coordinates: Sequence[tuple[float, float, float]],
    cell_size: float,
) -> dict[tuple[int, int, int], list[tuple[float, float, float]]]:
    grid: dict[tuple[int, int, int], list[tuple[float, float, float]]] = defaultdict(list)
    for xyz in ligand_coordinates:
        cell = tuple(math.floor(value / cell_size) for value in xyz)
        grid[cell].append(xyz)
    return grid


def coordinate_is_within_cutoff(
    xyz: tuple[float, float, float],
    ligand_grid: dict[tuple[int, int, int], list[tuple[float, float, float]]],
    cutoff: float,
) -> bool:
    cutoff_sq = cutoff * cutoff
    cell = tuple(math.floor(value / cutoff) for value in xyz)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                for ligand_xyz in ligand_grid.get(
                    (cell[0] + dx, cell[1] + dy, cell[2] + dz), ()
                ):
                    distance_sq = sum(
                        (a - b) ** 2 for a, b in zip(xyz, ligand_xyz)
                    )
                    if distance_sq <= cutoff_sq:
                        return True
    return False


def link_residue_keys(line: str) -> tuple[ResidueKey, ResidueKey]:
    padded = line.ljust(80)
    return (
        ResidueKey(
            padded[21:22].strip(), padded[22:26].strip(),
            padded[26:27].strip(), padded[17:20].strip().upper(),
        ),
        ResidueKey(
            padded[51:52].strip(), padded[52:56].strip(),
            padded[56:57].strip(), padded[47:50].strip().upper(),
        ),
    )


def ssbond_residue_keys(line: str) -> tuple[ResidueKey, ResidueKey]:
    padded = line.ljust(80)
    return (
        ResidueKey(
            padded[15:16].strip(), padded[17:21].strip(),
            padded[21:22].strip(), padded[11:14].strip().upper(),
        ),
        ResidueKey(
            padded[29:30].strip(), padded[31:35].strip(),
            padded[35:36].strip(), padded[25:28].strip().upper(),
        ),
    )


def replace_pdb_atom_serial(line: str, new_serial: int) -> str:
    if new_serial > 99999:
        raise ValidationError(
            "pocket_atom_serial_overflow",
            "pocket has more atoms than the five-column PDB serial field supports",
        )
    padded = line if len(line) >= 11 else line.ljust(11)
    return padded[:6] + f"{new_serial:5d}" + padded[11:]


def filtered_conect_line(
    source_index: int,
    line: str,
    serial_map: dict[tuple[int, str], int],
) -> str | None:
    fields = [line[index:index + 5].strip() for index in range(6, len(line), 5)]
    fields = [field for field in fields if field]
    if len(fields) < 2:
        return None
    source_key = (source_index, fields[0])
    if source_key not in serial_map:
        return None
    retained_targets = [
        serial_map[(source_index, field)]
        for field in fields[1:]
        if (source_index, field) in serial_map
    ]
    if not retained_targets:
        return None
    retained_serials = [serial_map[source_key], *retained_targets]
    return "CONECT" + "".join(f"{field:5d}" for field in retained_serials)


def build_pocket(
    receptor_path: Path,
    additive_path: Path | None,
    ligand_coordinates: Sequence[tuple[float, float, float]],
    cutoff: float,
    water_resnames: set[str],
    keep_water: bool,
) -> PocketResult:
    source_specs = [(receptor_path, False)]
    if additive_path is not None:
        source_specs.append((additive_path, True))

    combined_lines: list[tuple[int, str]] = []
    all_source_lines: list[tuple[int, str]] = []
    used_first_model_only = False
    for source_index, (source_path, hetatm_only) in enumerate(source_specs):
        try:
            source_text = source_path.read_text(encoding="latin-1")
        except OSError as exc:
            raise ValidationError("receptor_read_failed", str(exc)) from exc
        source_lines = source_text.splitlines()
        model_start, model_end, source_used_first_model = first_model_bounds(source_lines)
        used_first_model_only = used_first_model_only or source_used_first_model
        all_source_lines.extend((source_index, line) for line in source_lines)
        for line in source_lines[model_start:model_end]:
            if hetatm_only and not line.startswith(("HETATM", "ANISOU", "TER")):
                continue
            combined_lines.append((source_index, line))

    residue_atoms: dict[ResidueKey, list[AtomEntry]] = defaultdict(list)
    entries_by_line: dict[int, AtomEntry] = {}
    for line_index, (source_index, line) in enumerate(combined_lines):
        if not (line.startswith("ATOM  ") or line.startswith("HETATM")):
            continue
        residue = atom_residue_key(line)
        if not keep_water and residue.residue_name in water_resnames:
            continue
        atom = parse_atom_entry(line_index, source_index, line)
        residue_atoms[atom.residue].append(atom)
        entries_by_line[line_index] = atom

    if not residue_atoms:
        raise ValidationError(
            "receptor_has_no_nonwater_atoms",
            "receptor contains no selectable ATOM/HETATM records",
        )

    ligand_grid = build_ligand_grid(ligand_coordinates, cutoff)
    selected_residues = {
        residue
        for residue, atoms in residue_atoms.items()
        if any(
            coordinate_is_within_cutoff(atom.xyz, ligand_grid, cutoff)
            for atom in atoms
        )
    }
    if not selected_residues:
        raise ValidationError(
            "empty_pocket",
            "no receptor residue is within the requested cutoff; coordinate frames may differ",
        )

    selected_entries = [
        atom
        for residue in selected_residues
        for atom in residue_atoms[residue]
    ]
    selected_line_indices = {atom.line_index for atom in selected_entries}
    serial_map: dict[tuple[int, str], int] = {}
    for new_serial, atom in enumerate(
        sorted(selected_entries, key=lambda item: item.line_index), start=1
    ):
        if not atom.serial:
            raise ValidationError(
                "missing_atom_serial",
                f"selected atom on combined line {atom.line_index + 1} has no serial",
            )
        key = (atom.source_index, atom.serial)
        if key in serial_map:
            raise ValidationError(
                "duplicate_atom_serial",
                f"source {atom.source_index} repeats atom serial {atom.serial}",
            )
        serial_map[key] = new_serial

    polymer_residues = {
        residue
        for residue in selected_residues
        if any(atom.record_name == "ATOM" for atom in residue_atoms[residue])
    }
    hetatm_residues = selected_residues - polymer_residues
    if not polymer_residues:
        raise ValidationError(
            "pocket_has_no_polymer_residue",
            "only non-polymer HETATM residues are within the cutoff",
        )

    output_lines = [
        "REMARK 900 GENERATED BY prepare_hiqbind_pdbbind.py",
        f"REMARK 900 SOURCE {receptor_path.name}",
        f"REMARK 900 RESIDUE-BASED POCKET CUTOFF {cutoff:.3f} ANGSTROM",
        "REMARK 900 WATER EXCLUDED" if not keep_water else "REMARK 900 WATER RETAINED",
    ]
    if additive_path is not None:
        output_lines.insert(2, f"REMARK 900 ADDITIVES {additive_path.name}")

    # Preserve explicit residue-level connections only when both endpoints remain.
    seen_connection_records: set[str] = set()
    for _, line in all_source_lines:
        if line.startswith("LINK  "):
            first, second = link_residue_keys(line)
            retained = line.rstrip()
            if (
                first in selected_residues
                and second in selected_residues
                and retained not in seen_connection_records
            ):
                output_lines.append(retained)
                seen_connection_records.add(retained)
        elif line.startswith("SSBOND"):
            first, second = ssbond_residue_keys(line)
            retained = line.rstrip()
            if (
                first in selected_residues
                and second in selected_residues
                and retained not in seen_connection_records
            ):
                output_lines.append(retained)
                seen_connection_records.add(retained)

    polymer_open = False
    gap_after_polymer = False
    last_polymer_residue: ResidueKey | None = None
    for line_index, (source_index, line) in enumerate(combined_lines):
        if line.startswith("ATOM  "):
            residue = atom_residue_key(line)
            if line_index in selected_line_indices:
                if residue != last_polymer_residue:
                    chain_changed = (
                        last_polymer_residue is not None
                        and residue.chain_id != last_polymer_residue.chain_id
                    )
                    if polymer_open and (gap_after_polymer or chain_changed):
                        output_lines.append("TER")
                        polymer_open = False
                    last_polymer_residue = residue
                    gap_after_polymer = False
                atom = entries_by_line[line_index]
                output_lines.append(
                    replace_pdb_atom_serial(
                        line.rstrip(), serial_map[(atom.source_index, atom.serial)]
                    )
                )
                polymer_open = True
            elif polymer_open:
                gap_after_polymer = True
        elif line.startswith("HETATM"):
            if line_index in selected_line_indices:
                atom = entries_by_line[line_index]
                output_lines.append(
                    replace_pdb_atom_serial(
                        line.rstrip(), serial_map[(atom.source_index, atom.serial)]
                    )
                )
        elif line.startswith("ANISOU"):
            old_serial = line.ljust(11)[6:11].strip()
            serial_key = (source_index, old_serial)
            if serial_key in serial_map:
                output_lines.append(
                    replace_pdb_atom_serial(line.rstrip(), serial_map[serial_key])
                )
        elif line.startswith("TER"):
            if polymer_open:
                output_lines.append("TER")
            polymer_open = False
            gap_after_polymer = False
            last_polymer_residue = None

    if polymer_open:
        output_lines.append("TER")

    seen_conect_records: set[str] = set()
    for source_index, line in all_source_lines:
        if line.startswith("CONECT"):
            retained = filtered_conect_line(source_index, line, serial_map)
            if retained is not None and retained not in seen_conect_records:
                output_lines.append(retained)
                seen_conect_records.add(retained)
    output_lines.append("END")

    return PocketResult(
        lines=tuple(output_lines),
        atom_count=len(selected_entries),
        polymer_residue_count=len(polymer_residues),
        hetatm_residue_count=len(hetatm_residues),
        used_first_model_only=used_first_model_only,
    )


def output_id_for(pdb_id: str, dedup_smiles: str) -> str:
    digest = hashlib.sha256(
        f"{pdb_id}\0{dedup_smiles}".encode("utf-8")
    ).hexdigest()[:12]
    return f"{pdb_id}_{digest}"


def candidate_sort_key(info: LigandInfo) -> tuple[int, str, str]:
    candidate = info.candidate
    return (
        SOURCE_PRIORITY[candidate.source],
        candidate.complex_id.lower(),
        str(candidate.ligand_path).lower(),
    )


def ensure_empty_output_directory(output: Path) -> None:
    if output.exists():
        if not output.is_dir():
            raise FileExistsError(f"output path exists and is not a directory: {output}")
        if any(output.iterdir()):
            raise FileExistsError(
                f"output directory is not empty; choose a new path: {output}"
            )
    else:
        output.mkdir(parents=True, exist_ok=False)


def write_csv_report(path: Path, fieldnames: Sequence[str], rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def rejection_to_row(rejection: Rejection, repository_root: Path) -> dict[str, str]:
    return {
        "source": rejection.source,
        "pdbid": rejection.pdb_id,
        "complex_id": rejection.complex_id,
        "ligand_input": path_for_report(rejection.ligand_path, repository_root),
        "receptor_input": path_for_report(rejection.receptor_path, repository_root),
        "stage": rejection.stage,
        "reason": rejection.reason,
        "detail": rejection.detail,
    }


def build_argument_parser(
    pre_data_root: Path,
    repository_root: Path,
) -> argparse.ArgumentParser:
    data_root = pre_data_root
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hiqbind-index", type=Path,
        default=data_root / "Hiqbind" / "hiqbind_sm_metadata.csv",
    )
    parser.add_argument(
        "--hiqbind-root", type=Path,
        default=data_root / "Hiqbind" / "raw_data_hiq_sm",
    )
    parser.add_argument(
        "--pdbbind-pl-index", type=Path,
        default=data_root / "PDBbind" / "index" / "INDEX_general_PL.2025",
    )
    parser.add_argument(
        "--pdbbind-nl-index", type=Path,
        default=data_root / "PDBbind" / "index" / "INDEX_general_NL.2025",
    )
    parser.add_argument(
        "--pdbbind-pl-root", action="append", type=Path, default=None,
        help="Repeat for each PDBbind protein-ligand year directory.",
    )
    parser.add_argument(
        "--pdbbind-nl-root", type=Path,
        default=data_root / "PDBbind" / "NA-L",
    )
    parser.add_argument(
        "--output", type=Path,
        default=repository_root / "Datas" / "Processed_datas" / "HiQBind_PDBbind_clean",
    )
    parser.add_argument(
        "--mode", choices=("dry-run", "copy"), default="copy",
        help="copy materializes ligand.sdf/receptor.pdb; dry-run writes reports only.",
    )
    parser.add_argument("--cutoff", type=float, default=10.0)
    parser.add_argument(
        "--keep-water", action="store_true",
        help="Retain water within the cutoff. Default: exclude water.",
    )
    parser.add_argument(
        "--water-resname", action="append", default=None,
        help="Additional water residue name; may be repeated.",
    )
    parser.add_argument(
        "--extra-excluded-smiles", type=Path, default=None,
        help="Optional text file with one additional standalone excluded SMILES per line.",
    )
    parser.add_argument(
        "--max-candidates", type=int, default=None,
        help="Diagnostic-only limit applied after deterministic candidate sorting.",
    )
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument(
        "--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO",
    )
    return parser


def run(
    args: argparse.Namespace,
    repository_root: Path,
    pre_data_root: Path,
) -> int:
    if args.cutoff <= 0:
        raise ValueError("--cutoff must be greater than zero")
    if args.max_candidates is not None and args.max_candidates <= 0:
        raise ValueError("--max-candidates must be greater than zero")
    if args.progress_every <= 0:
        raise ValueError("--progress-every must be greater than zero")

    default_pl_roots = [
        pre_data_root / "PDBbind" / name
        for name in ("1981-2000", "2001-2010", "2011-2020", "2021-2024")
    ]
    pdbbind_pl_roots = args.pdbbind_pl_root or default_pl_roots

    for path, expected in (
        (args.hiqbind_index, "file"),
        (args.hiqbind_root, "directory"),
        (args.pdbbind_pl_index, "file"),
        (args.pdbbind_nl_index, "file"),
        (args.pdbbind_nl_root, "directory"),
    ):
        require_path(path, expected)
    for root in pdbbind_pl_roots:
        require_path(root, "directory")
    if args.extra_excluded_smiles is not None:
        require_path(args.extra_excluded_smiles, "file")

    ensure_empty_output_directory(args.output)
    excluded_fragment_keys = load_excluded_fragment_keys(args.extra_excluded_smiles)
    water_resnames = set(DEFAULT_WATER_RESNAMES)
    if args.water_resname:
        water_resnames.update(name.strip().upper() for name in args.water_resname if name.strip())

    LOGGER.info("Reading index identifiers (SMILES columns are not used)")
    hiqbind_ids = read_hiqbind_index_pdb_ids(args.hiqbind_index)
    pdbbind_pl_ids = read_pdbbind_index_pdb_ids(args.pdbbind_pl_index)
    pdbbind_nl_ids = read_pdbbind_index_pdb_ids(args.pdbbind_nl_index)

    LOGGER.info("Discovering structure pairs")
    hiq_candidates, hiq_issues = discover_hiqbind(hiqbind_ids, args.hiqbind_root)
    pl_candidates, pl_issues = discover_pdbbind(
        "pdbbind_pl", pdbbind_pl_ids, pdbbind_pl_roots, "_protein.pdb"
    )
    nl_candidates, nl_issues = discover_pdbbind(
        "pdbbind_nl", pdbbind_nl_ids, [args.pdbbind_nl_root], "_nucleic_acid.pdb"
    )
    candidates = sorted(
        [*hiq_candidates, *pl_candidates, *nl_candidates],
        key=lambda item: (
            item.pdb_id, SOURCE_PRIORITY[item.source], item.complex_id.lower(),
            str(item.ligand_path).lower(),
        ),
    )
    discovered_before_limit = len(candidates)
    if args.max_candidates is not None:
        candidates = candidates[:args.max_candidates]

    rejections: list[Rejection] = [*hiq_issues, *pl_issues, *nl_issues]
    validated_by_key: dict[tuple[str, str], list[LigandInfo]] = defaultdict(list)
    LOGGER.info("Validating %d ligand SDF files", len(candidates))
    for position, candidate in enumerate(candidates, start=1):
        try:
            info, _ = inspect_ligand(candidate, excluded_fragment_keys)
        except ValidationError as exc:
            rejections.append(
                Rejection(
                    candidate.source, candidate.pdb_id, candidate.complex_id,
                    candidate.ligand_path, candidate.receptor_path,
                    "ligand_validation", exc.reason, exc.detail,
                )
            )
        except Exception as exc:
            rejections.append(
                Rejection(
                    candidate.source, candidate.pdb_id, candidate.complex_id,
                    candidate.ligand_path, candidate.receptor_path,
                    "ligand_validation", "unexpected_ligand_error",
                    f"{type(exc).__name__}: {exc}",
                )
            )
        else:
            validated_by_key[(candidate.pdb_id, info.dedup_smiles)].append(info)
        if position % args.progress_every == 0:
            LOGGER.info("Validated %d/%d ligand files", position, len(candidates))

    accepted_rows: list[dict] = []
    duplicate_rows: list[dict] = []
    complexes_root = args.output / "complexes"
    if args.mode == "copy":
        complexes_root.mkdir(parents=True, exist_ok=False)

    LOGGER.info("Selecting representatives and generating 10 A pockets")
    for group_number, ((pdb_id, dedup_smiles), group) in enumerate(
        sorted(validated_by_key.items(), key=lambda item: item[0]), start=1
    ):
        ranked = sorted(group, key=candidate_sort_key)
        chosen_index: int | None = None
        chosen_info: LigandInfo | None = None
        chosen_pocket: PocketResult | None = None

        for rank_index, info in enumerate(ranked):
            candidate = info.candidate
            try:
                rechecked_info, mol = inspect_ligand(candidate, excluded_fragment_keys)
                if rechecked_info.dedup_smiles != dedup_smiles:
                    raise ValidationError(
                        "non_deterministic_smiles",
                        "SDF produced a different deduplication SMILES on re-read",
                    )
                ligand_xyz = molecule_coordinates(mol)
                pocket = build_pocket(
                    candidate.receptor_path,
                    candidate.additive_path,
                    ligand_xyz,
                    args.cutoff,
                    water_resnames,
                    args.keep_water,
                )
            except ValidationError as exc:
                rejections.append(
                    Rejection(
                        candidate.source, candidate.pdb_id, candidate.complex_id,
                        candidate.ligand_path, candidate.receptor_path,
                        "pocket_generation", exc.reason, exc.detail,
                    )
                )
                continue
            except Exception as exc:
                rejections.append(
                    Rejection(
                        candidate.source, candidate.pdb_id, candidate.complex_id,
                        candidate.ligand_path, candidate.receptor_path,
                        "pocket_generation", "unexpected_pocket_error",
                        f"{type(exc).__name__}: {exc}",
                    )
                )
                continue

            chosen_index = rank_index
            chosen_info = info
            chosen_pocket = pocket
            break

        if chosen_info is None or chosen_pocket is None or chosen_index is None:
            continue

        candidate = chosen_info.candidate
        output_id = output_id_for(pdb_id, dedup_smiles)
        complex_output = complexes_root / output_id
        ligand_output = complex_output / "ligand.sdf"
        receptor_output = complex_output / "receptor.pdb"
        if args.mode == "copy":
            complex_output.mkdir(parents=False, exist_ok=False)
            shutil.copy2(candidate.ligand_path, ligand_output)
            receptor_output.write_text(
                "\n".join(chosen_pocket.lines) + "\n",
                encoding="ascii",
                errors="strict",
                newline="\n",
            )

        accepted_rows.append(
            {
                "output_id": output_id,
                "pdbid": pdb_id,
                "dedup_smiles": dedup_smiles,
                "canonical_sdf_smiles": chosen_info.canonical_smiles,
                "source": candidate.source,
                "complex_id": candidate.complex_id,
                "ligand_input": path_for_report(candidate.ligand_path, repository_root),
                "receptor_input": path_for_report(candidate.receptor_path, repository_root),
                "additive_input": path_for_report(candidate.additive_path, repository_root),
                "ligand_output": path_for_report(ligand_output, repository_root),
                "receptor_output": path_for_report(receptor_output, repository_root),
                "materialized": args.mode == "copy",
                "ligand_atoms": chosen_info.atom_count,
                "ligand_heavy_atoms": chosen_info.heavy_atom_count,
                "ligand_fragments": chosen_info.fragment_count,
                "ligand_formal_charge": chosen_info.formal_charge,
                "pocket_atoms": chosen_pocket.atom_count,
                "pocket_polymer_residues": chosen_pocket.polymer_residue_count,
                "pocket_hetatm_residues": chosen_pocket.hetatm_residue_count,
                "first_model_only": chosen_pocket.used_first_model_only,
            }
        )

        # Entries ranked before the winner failed pocket generation and are in
        # rejected_manifest.csv.  Entries after the winner are true duplicates.
        for duplicate_info in ranked[chosen_index + 1:]:
            duplicate = duplicate_info.candidate
            duplicate_rows.append(
                {
                    "pdbid": pdb_id,
                    "dedup_smiles": dedup_smiles,
                    "kept_source": candidate.source,
                    "kept_complex_id": candidate.complex_id,
                    "kept_ligand_input": path_for_report(
                        candidate.ligand_path, repository_root
                    ),
                    "discarded_source": duplicate.source,
                    "discarded_complex_id": duplicate.complex_id,
                    "discarded_ligand_input": path_for_report(
                        duplicate.ligand_path, repository_root
                    ),
                    "reason": "same_pdbid_and_normalized_isomeric_smiles",
                }
            )

        if group_number % args.progress_every == 0:
            LOGGER.info(
                "Processed %d/%d deduplication groups",
                group_number,
                len(validated_by_key),
            )

    accepted_fields = (
        "output_id", "pdbid", "dedup_smiles", "canonical_sdf_smiles", "source",
        "complex_id", "ligand_input", "receptor_input", "ligand_output",
        "additive_input", "receptor_output", "materialized", "ligand_atoms", "ligand_heavy_atoms",
        "ligand_fragments", "ligand_formal_charge", "pocket_atoms",
        "pocket_polymer_residues", "pocket_hetatm_residues", "first_model_only",
    )
    rejected_fields = (
        "source", "pdbid", "complex_id", "ligand_input", "receptor_input",
        "stage", "reason", "detail",
    )
    duplicate_fields = (
        "pdbid", "dedup_smiles", "kept_source", "kept_complex_id",
        "kept_ligand_input", "discarded_source", "discarded_complex_id",
        "discarded_ligand_input", "reason",
    )
    write_csv_report(args.output / "accepted_manifest.csv", accepted_fields, accepted_rows)
    write_csv_report(
        args.output / "rejected_manifest.csv",
        rejected_fields,
        (rejection_to_row(item, repository_root) for item in rejections),
    )
    write_csv_report(args.output / "duplicates_manifest.csv", duplicate_fields, duplicate_rows)

    discovered_by_source = Counter(candidate.source for candidate in candidates)
    accepted_by_source = Counter(row["source"] for row in accepted_rows)
    rejection_by_stage = Counter(item.stage for item in rejections)
    rejection_by_reason = Counter(item.reason for item in rejections)
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "cutoff_angstrom": args.cutoff,
        "water_excluded": not args.keep_water,
        "smiles_source": "ligand_sdf_only",
        "deduplication": {
            "key": ["lowercase_pdbid", "neutralized_canonical_isomeric_smiles"],
            "stereochemistry_retained": True,
            "isotopes_retained": True,
            "tautomer_normalization": False,
            "source_priority": SOURCE_PRIORITY,
        },
        "index_pdbid_counts": {
            "hiqbind": len(hiqbind_ids),
            "pdbbind_pl": len(pdbbind_pl_ids),
            "pdbbind_nl": len(pdbbind_nl_ids),
        },
        "discovered_before_limit": discovered_before_limit,
        "candidates_processed": len(candidates),
        "max_candidates": args.max_candidates,
        "discovered_by_source": dict(sorted(discovered_by_source.items())),
        "valid_ligand_dedup_groups": len(validated_by_key),
        "accepted_complexes": len(accepted_rows),
        "accepted_by_source": dict(sorted(accepted_by_source.items())),
        "duplicate_complexes": len(duplicate_rows),
        "rejected_records": len(rejections),
        "rejected_by_stage": dict(sorted(rejection_by_stage.items())),
        "rejected_by_reason": dict(sorted(rejection_by_reason.items())),
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    LOGGER.info(
        "Done: accepted=%d duplicates=%d rejected=%d reports=%s",
        len(accepted_rows), len(duplicate_rows), len(rejections), args.output,
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    pre_data_root = Path(__file__).resolve().parent
    repository_root = pre_data_root.parents[1]
    parser = build_argument_parser(pre_data_root, repository_root)
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    if Chem is None or rdMolStandardize is None:
        parser.error(
            "RDKit is required but is not installed in this Python environment. "
            "The script does not install dependencies automatically."
        )
    RDLogger.DisableLog("rdApp.warning")
    try:
        return run(args, repository_root, pre_data_root)
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
