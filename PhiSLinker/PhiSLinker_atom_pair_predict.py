"""Capacity-aware global inference for PhiSLinker atom pairs."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from rdkit import Chem

try:
    from PhiSLinker_atom_pair_config import (
        ATOM_PAIR_MIN_PROBABILITY,
        ATOM_PAIR_SUPERVISION_POLICY,
        ATOM_VOCAB_PATH,
        CONFORMER_TEMPLATE_CACHE_SIZE,
        FRAGMENT_VOCAB_PATH,
        GLOBAL_BEAM_WIDTH,
        GLOBAL_INFERENCE_COMBINATIONS,
        GLOBAL_PREFER_DISTINCT_NODE_ATOMS,
        GLOBAL_TOP_K,
        ModelConfig,
    )
    from PhiSLinker_atom_pair_dataset import (
        AtomPairBatch,
        ConnectionFeatureBuilder,
        collate_prepared_graphs,
    )
    from PhiSLinker_atom_pair_model import AtomPairModelOutput, PhiSLinkerAtomPairModel
    from PhiSLinker_atom_pair_utils import (
        AtomVocabularyLookup,
        ConformerGeometryCache,
        FragmentCandidateCache,
        FragmentVocabularyLookup,
        GlobalAtomPairChoice,
        atomic_write_json,
        beam_search_atom_pair_combinations,
        torch_load_compat,
    )
except ImportError:  # pragma: no cover - package-style import fallback
    from .PhiSLinker_atom_pair_config import (
        ATOM_PAIR_MIN_PROBABILITY,
        ATOM_PAIR_SUPERVISION_POLICY,
        ATOM_VOCAB_PATH,
        CONFORMER_TEMPLATE_CACHE_SIZE,
        FRAGMENT_VOCAB_PATH,
        GLOBAL_BEAM_WIDTH,
        GLOBAL_INFERENCE_COMBINATIONS,
        GLOBAL_PREFER_DISTINCT_NODE_ATOMS,
        GLOBAL_TOP_K,
        ModelConfig,
    )
    from .PhiSLinker_atom_pair_dataset import (
        AtomPairBatch,
        ConnectionFeatureBuilder,
        collate_prepared_graphs,
    )
    from .PhiSLinker_atom_pair_model import AtomPairModelOutput, PhiSLinkerAtomPairModel
    from .PhiSLinker_atom_pair_utils import (
        AtomVocabularyLookup,
        ConformerGeometryCache,
        FragmentCandidateCache,
        FragmentVocabularyLookup,
        GlobalAtomPairChoice,
        atomic_write_json,
        beam_search_atom_pair_combinations,
        torch_load_compat,
    )

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict globally consistent atom pairs for .pt graph files"
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fragment-vocab", type=Path, default=FRAGMENT_VOCAB_PATH)
    parser.add_argument("--atom-vocab", type=Path, default=ATOM_VOCAB_PATH)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--pattern", type=str, default="*.pt")
    return parser.parse_args()


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


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
        raise ValueError("global inference requires a global-stage checkpoint")
    if "model_state_dict" not in payload:
        raise KeyError("checkpoint lacks model_state_dict")
    return payload


def _model_config_from_checkpoint(payload: Mapping[str, Any]) -> ModelConfig:
    try:
        config = ModelConfig(**dict(payload["configuration"]["model"]))
    except Exception as exc:
        raise ValueError("checkpoint contains an invalid model configuration") from exc
    config.validate()
    return config


def _verify_vocabulary_fingerprints(
    payload: Mapping[str, Any],
    fragment_vocab: FragmentVocabularyLookup,
    atom_vocab: AtomVocabularyLookup,
) -> None:
    saved = payload.get("vocabulary_fingerprints")
    if not isinstance(saved, Mapping):
        raise ValueError("checkpoint lacks vocabulary fingerprints")
    expected = {
        "fragment_vocab_sha256": fragment_vocab.sha256,
        "atom_vocab_sha256": atom_vocab.sha256,
    }
    mismatches = [key for key, value in expected.items() if saved.get(key) != value]
    if mismatches:
        raise ValueError(
            "inference vocabularies differ from checkpoint: " + ", ".join(mismatches)
        )


def _candidate_bounds(batch: AtomPairBatch, query_index: int) -> tuple[int, int]:
    start = int(batch.query_ptr[query_index].item())
    end = int(batch.query_ptr[query_index + 1].item())
    if end <= start:
        raise ValueError(f"query {query_index} has no candidates")
    return start, end


def _ranked_rows(
    output: AtomPairModelOutput,
    batch: AtomPairBatch,
    query_index: int,
) -> tuple[list[int], torch.Tensor, list[bool]]:
    start, end = _candidate_bounds(batch, query_index)
    probabilities = torch.softmax(output.pair_logits[start:end].float(), dim=0)
    # Any pair drawn from N, O, F, Cl, Br, I, At, Ts is forbidden for new bonds.
    restricted_elements = {7, 8, 9, 17, 35, 53, 85, 117}
    atomic_numbers: list[list[int]] = []
    for side in ("a", "b"):
        smiles = str(batch.query_metadata[query_index][f"smiles_{side}"])
        parsed = Chem.MolFromSmiles(smiles)
        if parsed is None:
            raise ValueError(f"RDKit could not parse fragment SMILES: {smiles!r}")
        # Match the canonical atom IDs used by FragmentCandidateCache.
        canonical_smiles = Chem.MolToSmiles(
            parsed, canonical=True, isomericSmiles=True
        )
        molecule = Chem.MolFromSmiles(canonical_smiles)
        if molecule is None:
            raise ValueError(f"RDKit could not reparse SMILES: {canonical_smiles!r}")
        atomic_numbers.append([atom.GetAtomicNum() for atom in molecule.GetAtoms()])
    allowed = [
        not (
            atomic_numbers[0][atom_a_id] in restricted_elements
            and atomic_numbers[1][atom_b_id] in restricted_elements
        )
        for atom_a_id, atom_b_id in (
            batch.candidate_atom_ids[start:end].detach().cpu().tolist()
        )
    ]
    # Mask after softmax: keep all other probabilities unchanged, without renormalizing.
    probabilities = probabilities.masked_fill(
        ~torch.tensor(allowed, dtype=torch.bool, device=probabilities.device), 0.0
    )
    rows = sorted(
        range(start, end),
        key=lambda row: (
            -float(probabilities[row - start].item()),
            int(batch.candidate_atom_ids[row, 0].item()),
            int(batch.candidate_atom_ids[row, 1].item()),
            row,
        ),
    )
    return rows, probabilities, allowed


@torch.no_grad()
def format_batch_predictions(
    output: AtomPairModelOutput,
    batch: AtomPairBatch,
    *,
    include_ranked_pairs: bool = False,
    selected_candidate_rows: Mapping[int, int | None] | None = None,
    fixed_query_indices: set[int] | None = None,
    selected_global_score: float | None = None,
) -> list[dict[str, Any]]:
    """Format query records; an explicit None row means the bond was pruned."""

    if output.pair_logits.shape != (batch.num_candidates,):
        raise ValueError("pair_logits must have shape [C]")
    if len(batch.query_metadata) != batch.num_queries:
        raise ValueError("query_metadata length must equal Q")
    selected_rows = dict(selected_candidate_rows or {})
    fixed_queries = set(fixed_query_indices or set())
    results: list[dict[str, Any]] = []
    for query_index in range(batch.num_queries):
        start, end = _candidate_bounds(batch, query_index)
        ranked, probabilities, allowed = _ranked_rows(output, batch, query_index)
        rank_by_row = {row: rank for rank, row in enumerate(ranked, start=1)}
        selectable_rows = (
            ranked if query_index in fixed_queries
            else [row for row in ranked if allowed[row - start]]
        )
        if not selectable_rows:
            raise ValueError(f"query {query_index} has no allowed element pair")
        selected_row = selected_rows.get(query_index, selectable_rows[0])
        if selected_row is not None and not start <= selected_row < end:
            raise ValueError(
                f"selected candidate row does not belong to query {query_index}"
            )
        if (
            selected_row is not None and query_index not in fixed_queries
            and not allowed[selected_row - start]
        ):
            raise ValueError(f"query {query_index} selected a forbidden element pair")
        ranked_pairs: list[dict[str, Any]] = []
        for rank, row in enumerate(ranked, start=1):
            ranked_pairs.append(
                {
                    "candidate_rank": rank,
                    "atom_a_id": int(batch.candidate_atom_ids[row, 0].item()),
                    "atom_b_id": int(batch.candidate_atom_ids[row, 1].item()),
                    "atom_a_capacity": int(
                        batch.candidate_atom_capacities[row, 0].item()
                    ),
                    "atom_b_capacity": int(
                        batch.candidate_atom_capacities[row, 1].item()
                    ),
                    "pair_probability": float(probabilities[row - start].item()),
                    "element_pair_allowed": allowed[row - start],
                }
            )
        selected_pairs = (
            [dict(ranked_pairs[rank_by_row[selected_row] - 1])]
            if selected_row is not None else []
        )
        metadata = dict(batch.query_metadata[query_index])
        geometry_valid = bool(output.geometry_used[query_index].item())
        result: dict[str, Any] = {
            "node_a": int(metadata["node_a"]),
            "node_b": int(metadata["node_b"]),
            "smiles_a": str(metadata["smiles_a"]),
            "smiles_b": str(metadata["smiles_b"]),
            "predicted_count": len(selected_pairs),
            "candidate_count": end - start,
            "geometry_valid": geometry_valid,
            "status": (
                "pruned_new_four_membered_ring" if selected_row is None
                else "ok" if geometry_valid else "ok_geometry_fallback"
            ),
            "excluded_from_global_search": query_index in fixed_queries,
            "selected_pairs": selected_pairs,
        }
        if query_index not in fixed_queries and selected_global_score is not None:
            result["global_combination_rank"] = 1
            result["global_combination_score"] = float(selected_global_score)
        if include_ranked_pairs:
            result["ranked_pairs"] = ranked_pairs
        if metadata.get("geometry_error"):
            result["geometry_error"] = str(metadata["geometry_error"])
        results.append(result)
    return results


def _normalize_fixed_connection(record: Mapping[str, Any]) -> dict[str, Any]:
    node_a = int(record["node_a"])
    node_b = int(record["node_b"])
    atom_a_id = int(record["atom_a_id"])
    atom_b_id = int(record["atom_b_id"])
    if node_a == node_b:
        raise ValueError("fixed connection cannot be a node self-loop")
    if node_a > node_b:
        node_a, node_b = node_b, node_a
        atom_a_id, atom_b_id = atom_b_id, atom_a_id
    return {
        **dict(record),
        "node_a": node_a,
        "node_b": node_b,
        "atom_a_id": atom_a_id,
        "atom_b_id": atom_b_id,
        "connection_source": "condition_pt_fixed_fixed_ground_truth",
    }


def _batch_capacity_map(batch: AtomPairBatch) -> dict[tuple[int, int], int]:
    capacities: dict[tuple[int, int], int] = {}
    for query_index in range(batch.num_queries):
        start, end = _candidate_bounds(batch, query_index)
        nodes = (
            int(batch.query_node_index[0, query_index].item()),
            int(batch.query_node_index[1, query_index].item()),
        )
        for row in range(start, end):
            for column, node in enumerate(nodes):
                key = (node, int(batch.candidate_atom_ids[row, column].item()))
                capacity = int(batch.candidate_atom_capacities[row, column].item())
                previous = capacities.setdefault(key, capacity)
                if previous != capacity:
                    raise ValueError(f"inconsistent capacity for atom {key}")
    return capacities


def _build_fixed_atom_adjacency(
    batch: AtomPairBatch,
    fixed_connections: Sequence[Mapping[str, Any]],
) -> dict[tuple[int, int], set[tuple[int, int]]]:
    """Use canonical atom IDs for fragment-internal and fixed inter-fragment bonds."""

    adjacency: dict[tuple[int, int], set[tuple[int, int]]] = {}
    seen_nodes: set[int] = set()
    for query_index, metadata in enumerate(batch.query_metadata):
        for column, side in enumerate(("a", "b")):
            node = int(batch.query_node_index[column, query_index].item())
            if node in seen_nodes:
                continue
            seen_nodes.add(node)
            smiles = str(metadata[f"smiles_{side}"])
            parsed = Chem.MolFromSmiles(smiles)
            if parsed is None:
                raise ValueError(f"RDKit could not parse fragment SMILES: {smiles!r}")
            canonical_smiles = Chem.MolToSmiles(
                parsed, canonical=True, isomericSmiles=True
            )
            molecule = Chem.MolFromSmiles(canonical_smiles)
            if molecule is None:
                raise ValueError(f"RDKit could not reparse SMILES: {canonical_smiles!r}")
            for atom in molecule.GetAtoms():
                if atom.GetAtomicNum() > 1:
                    adjacency[(node, atom.GetIdx())] = set()
            for bond in molecule.GetBonds():
                a = (node, bond.GetBeginAtomIdx())
                b = (node, bond.GetEndAtomIdx())
                if a in adjacency and b in adjacency:
                    adjacency[a].add(b)
                    adjacency[b].add(a)
    for connection in fixed_connections:
        a = (int(connection["node_a"]), int(connection["atom_a_id"]))
        b = (int(connection["node_b"]), int(connection["atom_b_id"]))
        adjacency[a].add(b)
        adjacency[b].add(a)
    return adjacency


def _prune_new_four_membered_rings(
    base_adjacency: Mapping[tuple[int, int], set[tuple[int, int]]],
    predicted_connections: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep higher-probability new bonds, skipping those closing a four-atom cycle."""

    adjacency = {atom: set(neighbors) for atom, neighbors in base_adjacency.items()}
    ordered_indices = sorted(
        range(len(predicted_connections)),
        key=lambda index: (
            -float(predicted_connections[index]["pair_probability"]),
            *(int(predicted_connections[index][key]) for key in (
                "node_a", "node_b", "atom_a_id", "atom_b_id"
            )),
        ),
    )
    removed_indices: set[int] = set()
    removed: list[dict[str, Any]] = []
    for index in ordered_indices:
        connection = predicted_connections[index]
        a = (int(connection["node_a"]), int(connection["atom_a_id"]))
        b = (int(connection["node_b"]), int(connection["atom_b_id"]))
        # A simple three-bond path a-x-y-b becomes a four-membered ring.
        # Check all such paths, including cycles that also have shorter paths.
        ring_path = next(
            (
                (a, x, y, b)
                for x in sorted(adjacency[a]) if x != b
                for y in sorted(adjacency[b]) if y != a and y != x
                if y in adjacency[x]
            ),
            None,
        )
        if ring_path is not None:
            removed_indices.add(index)
            removed.append({
                **dict(connection),
                "removal_reason": "new_four_membered_ring",
                "ring_atoms": [
                    {"node": node, "atom_id": atom_id}
                    for node, atom_id in ring_path
                ],
            })
            continue
        adjacency[a].add(b)
        adjacency[b].add(a)
    kept = [
        dict(connection)
        for index, connection in enumerate(predicted_connections)
        if index not in removed_indices
    ]
    return kept, removed


@torch.no_grad()
def _global_decode(
    output: AtomPairModelOutput,
    batch: AtomPairBatch,
    *,
    fixed_connections: Sequence[Mapping[str, Any]],
    min_probability: float,
    prefer_distinct_node_atoms: bool,
    top_k: int,
    beam_width: int,
    max_combinations: int,
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
    dict[int, int | None],
    set[int],
]:
    if top_k <= 0 or beam_width <= 0 or max_combinations <= 0:
        raise ValueError("global search limits must be positive")
    min_probability = float(min_probability)
    if not math.isfinite(min_probability) or not 0.0 <= min_probability <= 1.0:
        raise ValueError("min_probability must be finite and in [0, 1]")
    capacities = _batch_capacity_map(batch)
    query_by_nodes = {
        (
            int(batch.query_node_index[0, query_index].item()),
            int(batch.query_node_index[1, query_index].item()),
        ): query_index
        for query_index in range(batch.num_queries)
    }
    normalized_fixed = [
        _normalize_fixed_connection(record) for record in fixed_connections
    ]
    initial_usage: dict[tuple[int, int], int] = {}
    fixed_query_indices: set[int] = set()
    for record in normalized_fixed:
        node_pair = (int(record["node_a"]), int(record["node_b"]))
        try:
            fixed_query_indices.add(query_by_nodes[node_pair])
        except KeyError as exc:
            raise ValueError(
                f"fixed connection does not correspond to a model query: {node_pair}"
            ) from exc
        for key in (
            (node_pair[0], int(record["atom_a_id"])),
            (node_pair[1], int(record["atom_b_id"])),
        ):
            if key not in capacities:
                raise ValueError(f"fixed connection atom has no capacity: {key}")
            initial_usage[key] = initial_usage.get(key, 0) + 1

    unknown_queries = [
        query_index
        for query_index in range(batch.num_queries)
        if query_index not in fixed_query_indices
    ]
    query_choices: list[list[GlobalAtomPairChoice]] = []
    probability_by_row: dict[int, float] = {}
    rank_by_row: dict[int, int] = {}
    blocked_queries: list[dict[str, Any]] = []
    probability_rejected_candidate_count = 0
    forbidden_element_pair_candidate_count = 0
    for local_query_index, query_index in enumerate(unknown_queries):
        ranked, probabilities, allowed = _ranked_rows(output, batch, query_index)
        start, _ = _candidate_bounds(batch, query_index)
        for rank, row in enumerate(ranked, start=1):
            rank_by_row[row] = rank
            probability_by_row[row] = float(probabilities[row - start].item())
        eligible_rows = [
            row
            for row in ranked
            if allowed[row - start] and probability_by_row[row] >= min_probability
        ]
        allowed_candidate_count = sum(allowed)
        forbidden_candidate_count = len(ranked) - allowed_candidate_count
        forbidden_element_pair_candidate_count += forbidden_candidate_count
        probability_rejected_candidate_count += allowed_candidate_count - len(eligible_rows)
        if not eligible_rows:
            blocked_queries.append(
                {
                    "query_index": int(query_index),
                    "node_a": int(batch.query_node_index[0, query_index].item()),
                    "node_b": int(batch.query_node_index[1, query_index].item()),
                    "candidate_count": len(ranked),
                    "allowed_candidate_count": allowed_candidate_count,
                    "forbidden_element_pair_candidate_count": forbidden_candidate_count,
                    "highest_pair_probability": probability_by_row[ranked[0]],
                    "failure_reason": (
                        "all_candidates_forbidden_element_pairs"
                        if allowed_candidate_count == 0
                        else "no_allowed_candidate_above_minimum_probability"
                    ),
                }
            )
            continue
        choices: list[GlobalAtomPairChoice] = []
        for row in eligible_rows[:top_k]:
            probability = probability_by_row[row]
            choices.append(
                GlobalAtomPairChoice(
                    query_index=local_query_index,
                    candidate_index=row,
                    node_a=int(batch.query_node_index[0, query_index].item()),
                    node_b=int(batch.query_node_index[1, query_index].item()),
                    atom_a_id=int(batch.candidate_atom_ids[row, 0].item()),
                    atom_b_id=int(batch.candidate_atom_ids[row, 1].item()),
                    score=math.log(max(probability, torch.finfo(torch.float32).tiny)),
                )
            )
        query_choices.append(choices)

    if blocked_queries:
        diagnostics = {
            "strategy": "capacity_aware_global_top_k_beam",
            "status": (
                "no_allowed_element_pair"
                if any(record["allowed_candidate_count"] == 0 for record in blocked_queries)
                else "no_candidate_above_minimum_probability"
            ),
            "minimum_pair_probability": min_probability,
            "prefer_distinct_node_atoms": bool(prefer_distinct_node_atoms),
            "top_k": int(top_k),
            "beam_width": int(beam_width),
            "requested_combination_count": int(max_combinations),
            "legal_combination_count": 0,
            "query_count": batch.num_queries,
            "fixed_query_count": len(fixed_query_indices),
            "predicted_query_count": len(unknown_queries),
            "fixed_connection_count": len(normalized_fixed),
            "probability_rejected_candidate_count": (
                probability_rejected_candidate_count
            ),
            "forbidden_element_pair_candidate_count": forbidden_element_pair_candidate_count,
            "blocked_query_count": len(blocked_queries),
            "blocked_queries": blocked_queries,
        }
        return [], diagnostics, {}, fixed_query_indices

    legal, _ = beam_search_atom_pair_combinations(
        query_choices,
        capacities,
        initial_usage=initial_usage,
        prefer_distinct_node_atoms=prefer_distinct_node_atoms,
        beam_width=beam_width,
        max_legal_results=max_combinations,
        max_conflict_results=0,
    )
    base_adjacency = _build_fixed_atom_adjacency(batch, normalized_fixed) if legal else {}
    combinations: list[dict[str, Any]] = []
    for combination_rank, combination in enumerate(legal, start=1):
        predicted: list[dict[str, Any]] = []
        for choice in combination.choices:
            row = choice.candidate_index
            predicted.append(
                {
                    "node_a": choice.node_a,
                    "node_b": choice.node_b,
                    "atom_a_id": choice.atom_a_id,
                    "atom_b_id": choice.atom_b_id,
                    "candidate_rank": rank_by_row[row],
                    "pair_probability": probability_by_row[row],
                    "connection_source": "atom_pair_model_global_prediction",
                }
            )
        predicted, removed_connections = _prune_new_four_membered_rings(
            base_adjacency, predicted
        )
        usage = dict(initial_usage)
        for connection in predicted:
            for side in ("a", "b"):
                endpoint = (
                    int(connection[f"node_{side}"]),
                    int(connection[f"atom_{side}_id"]),
                )
                usage[endpoint] = usage.get(endpoint, 0) + 1
        usage_records = [
            {
                "node": node,
                "atom_id": atom_id,
                "used": used,
                "capacity": capacities[(node, atom_id)],
                "remaining": capacities[(node, atom_id)] - used,
            }
            for (node, atom_id), used in sorted(usage.items())
            if used
        ]
        full_connections = [
            *[dict(record) for record in normalized_fixed],
            *predicted,
        ]
        combinations.append(
            {
                "global_combination_rank": combination_rank,
                "global_score": float(combination.score),
                "global_score_basis": "sum_log_pair_probability_before_ring_pruning",
                "predicted_connections": predicted,
                "fixed_connections": [dict(record) for record in normalized_fixed],
                "connections": full_connections,
                "capacity_usage": usage_records,
                "overflow_count": 0,
                "four_membered_ring_pruning": {
                    "removed_connection_count": len(removed_connections),
                    "removed_connections": removed_connections,
                },
            }
        )

    best_rows: dict[int, int | None] = {}
    if legal:
        kept_node_pairs = {
            (int(record["node_a"]), int(record["node_b"]))
            for record in combinations[0]["predicted_connections"]
        }
        for query_index, choice in zip(unknown_queries, legal[0].choices):
            best_rows[query_index] = (
                choice.candidate_index
                if (choice.node_a, choice.node_b) in kept_node_pairs else None
            )
    diagnostics = {
        "strategy": "capacity_aware_global_top_k_beam",
        "status": "success" if legal else "no_legal_combination",
        "minimum_pair_probability": min_probability,
        "prefer_distinct_node_atoms": bool(prefer_distinct_node_atoms),
        "top_k": int(top_k),
        "beam_width": int(beam_width),
        "requested_combination_count": int(max_combinations),
        "legal_combination_count": len(combinations),
        "query_count": batch.num_queries,
        "fixed_query_count": len(fixed_query_indices),
        "predicted_query_count": len(unknown_queries),
        "fixed_connection_count": len(normalized_fixed),
        "probability_rejected_candidate_count": (
            probability_rejected_candidate_count
        ),
        "forbidden_element_pair_candidate_count": forbidden_element_pair_candidate_count,
        "blocked_query_count": 0,
        "new_four_membered_ring_policy": "keep_fixed_then_descending_pair_probability",
        "global_score_basis": "sum_log_pair_probability_before_ring_pruning",
    }
    return combinations, diagnostics, best_rows, fixed_query_indices


@torch.no_grad()
def predict_data_object(
    data: Any,
    *,
    source_path: str,
    builder: ConnectionFeatureBuilder,
    model: PhiSLinkerAtomPairModel,
    device: torch.device,
    include_ranked_pairs: bool = False,
    fixed_connections: Sequence[Mapping[str, Any]] = (),
    min_probability: float = ATOM_PAIR_MIN_PROBABILITY,
    prefer_distinct_node_atoms: bool = GLOBAL_PREFER_DISTINCT_NODE_ATOMS,
    top_k: int = GLOBAL_TOP_K,
    beam_width: int = GLOBAL_BEAM_WIDTH,
    max_combinations: int = GLOBAL_INFERENCE_COMBINATIONS,
) -> dict[str, Any]:
    prepared = builder.prepare(
        data, include_targets=False, source_path=source_path
    )
    batch = collate_prepared_graphs([prepared])
    device_batch = batch.to(device, non_blocking=device.type == "cuda")
    output = model(device_batch, apply_geometry_dropout=False)
    (
        global_combinations,
        global_search,
        best_rows,
        fixed_query_indices,
    ) = _global_decode(
        output,
        device_batch,
        fixed_connections=fixed_connections,
        min_probability=min_probability,
        prefer_distinct_node_atoms=prefer_distinct_node_atoms,
        top_k=top_k,
        beam_width=beam_width,
        max_combinations=max_combinations,
    )
    if not global_combinations:
        raise RuntimeError(
            "capacity-aware global search found no legal complete "
            f"combination: {global_search}"
        )
    best_score = (
        float(global_combinations[0]["global_score"])
        if global_combinations
        else None
    )
    predictions = format_batch_predictions(
        output,
        device_batch,
        include_ranked_pairs=include_ranked_pairs,
        selected_candidate_rows=best_rows,
        fixed_query_indices=fixed_query_indices,
        selected_global_score=best_score,
    )
    return {
        "source_file": source_path,
        "sample_id": prepared.sample_id,
        "query_count": prepared.num_queries,
        "predictions": predictions,
        "global_search": global_search,
        "global_combinations": global_combinations,
    }


def _discover_inputs(path: Path, pattern: str) -> list[Path]:
    resolved = path.expanduser().resolve()
    if resolved.is_file():
        return [resolved]
    if not resolved.is_dir():
        raise FileNotFoundError(f"input does not exist: {resolved}")
    files = sorted(item.resolve() for item in resolved.rglob(pattern) if item.is_file())
    if not files:
        raise FileNotFoundError(f"no {pattern!r} files found below {resolved}")
    return files


def _default_output(input_path: Path) -> Path:
    resolved = input_path.expanduser().resolve()
    if resolved.is_file():
        return resolved.with_name(f"{resolved.stem}_atom_pair_predictions.json")
    return resolved / "atom_pair_predictions.json"


def main() -> None:
    args = _parse_args()
    device = _resolve_device(args.device)
    checkpoint = _load_checkpoint(args.checkpoint)
    model_config = _model_config_from_checkpoint(checkpoint)
    fragment_vocab = FragmentVocabularyLookup(
        args.fragment_vocab, expected_dim=model_config.fragment_embedding_dim
    )
    atom_vocab = AtomVocabularyLookup(
        args.atom_vocab, expected_dim=model_config.atom_embedding_dim
    )
    _verify_vocabulary_fingerprints(checkpoint, fragment_vocab, atom_vocab)
    builder = ConnectionFeatureBuilder(
        fragment_vocab,
        FragmentCandidateCache(atom_vocab),
        ConformerGeometryCache(CONFORMER_TEMPLATE_CACHE_SIZE),
        model_config,
    )
    model = PhiSLinkerAtomPairModel(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    input_files = _discover_inputs(args.input, args.pattern)
    graphs: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for path in input_files:
        try:
            data = torch_load_compat(path, map_location="cpu")
            graphs.append(
                predict_data_object(
                    data,
                    source_path=str(path),
                    builder=builder,
                    model=model,
                    device=device,
                )
            )
        except Exception as exc:
            failures.append(
                {"source_file": str(path), "error": f"{type(exc).__name__}: {exc}"}
            )

    payload = {
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "training_stage": "global",
        "global_search": {
            "minimum_pair_probability": ATOM_PAIR_MIN_PROBABILITY,
            "prefer_distinct_node_atoms": GLOBAL_PREFER_DISTINCT_NODE_ATOMS,
            "top_k": GLOBAL_TOP_K,
            "beam_width": GLOBAL_BEAM_WIDTH,
            "max_combinations": GLOBAL_INFERENCE_COMBINATIONS,
        },
        "files_requested": len(input_files),
        "files_succeeded": len(graphs),
        "files_failed": len(failures),
        "graphs": graphs,
        "failures": failures,
    }
    output_path = (args.output or _default_output(args.input)).expanduser().resolve()
    atomic_write_json(output_path, payload)
    print(
        f"Prediction complete: {len(graphs)}/{len(input_files)} files succeeded; "
        f"output={output_path}"
    )
    if failures:
        raise RuntimeError(
            f"prediction failed for {len(failures)} files; details are in {output_path}"
        )


if __name__ == "__main__":
    main()


__all__ = ["format_batch_predictions", "predict_data_object"]
