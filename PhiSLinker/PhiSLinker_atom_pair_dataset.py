"""Dataset and batching for inter-fragment atom-pair prediction."""

from __future__ import annotations

import math
import random
from collections import Counter, OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

try:
    from PhiSLinker_atom_pair_config import (
        ATOM_PAIR_SUPERVISION_POLICY,
        ModelConfig,
    )
    from PhiSLinker_atom_pair_utils import (
        GLOBAL_SMILES_TOKEN,
        KNOWN_NEUTRALIZED_NITROGEN_TARGET_REASON,
        AlignedGeometryResult,
        ConformerGeometryCache,
        FragmentCandidateEntry,
        FragmentCandidateCache,
        FragmentVocabularyLookup,
        atomic_write_json,
        diagnose_atom_single_bond_capacity,
        diagnose_fragment_connection_combination,
        gaussian_rbf,
        normalize_smiles_value,
        progress_iterable,
        torch_load_compat,
    )
except ImportError:  # pragma: no cover - package-style import fallback
    from .PhiSLinker_atom_pair_config import (
        ATOM_PAIR_SUPERVISION_POLICY,
        ModelConfig,
    )
    from .PhiSLinker_atom_pair_utils import (
        GLOBAL_SMILES_TOKEN,
        KNOWN_NEUTRALIZED_NITROGEN_TARGET_REASON,
        AlignedGeometryResult,
        ConformerGeometryCache,
        FragmentCandidateEntry,
        FragmentCandidateCache,
        FragmentVocabularyLookup,
        atomic_write_json,
        diagnose_atom_single_bond_capacity,
        diagnose_fragment_connection_combination,
        gaussian_rbf,
        normalize_smiles_value,
        progress_iterable,
        torch_load_compat,
    )


@dataclass(frozen=True)
class ConnectionGroup:
    node_a: int
    node_b: int
    true_atom_pairs: tuple[tuple[int, int], ...]

    @property
    def true_count(self) -> int:
        return len(self.true_atom_pairs)


class TrueCombinationCapacityError(ValueError):
    """Structured preflight failure for a labelled capacity disagreement."""

    def __init__(self, message: str, diagnostics: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.diagnostics = dict(diagnostics)


@dataclass(frozen=True)
class ExcludedTargetQuery:
    """One corrupt supervised query deliberately omitted from training."""

    node_a: int
    node_b: int
    reason: str
    true_atom_pairs: tuple[tuple[int, int], ...]
    missing_true_atom_pairs: tuple[tuple[int, int], ...]
    affected_atoms: tuple[tuple[str, int, int], ...]


@dataclass(frozen=True)
class GraphFileStats:
    path: str
    num_nodes: int
    num_queries: int
    num_candidates: int
    num_filtered_double_queries: int
    excluded_target_queries: tuple[ExcludedTargetQuery, ...] = ()

    @property
    def num_excluded_queries(self) -> int:
        return len(self.excluded_target_queries)

    @property
    def num_original_queries(self) -> int:
        return (
            self.num_queries
            + self.num_excluded_queries
            + self.num_filtered_double_queries
        )


@dataclass
class PreparedConnectionGraph:
    node_fragment_embeddings: torch.Tensor
    node_type: torch.Tensor
    node_pos: torch.Tensor
    edge_index: torch.Tensor
    edge_features: torch.Tensor
    query_node_index: torch.Tensor
    candidate_query_index: torch.Tensor
    candidate_atom_embeddings_a: torch.Tensor
    candidate_atom_embeddings_b: torch.Tensor
    candidate_atom_ids: torch.Tensor
    candidate_atom_capacities: torch.Tensor
    candidate_distances: torch.Tensor
    query_geometry_valid: torch.Tensor
    positive_candidate_mask: torch.Tensor | None
    query_metadata: list[dict[str, Any]]
    sample_id: str

    @property
    def num_nodes(self) -> int:
        return int(self.node_type.shape[0])

    @property
    def num_queries(self) -> int:
        return int(self.query_node_index.shape[1])

    @property
    def num_candidates(self) -> int:
        return int(self.candidate_query_index.shape[0])


@dataclass
class AtomPairBatch:
    """Flattened graph/query/candidate tensors consumed by the model."""

    node_fragment_embeddings: torch.Tensor
    node_type: torch.Tensor
    node_pos: torch.Tensor
    edge_index: torch.Tensor
    edge_features: torch.Tensor
    query_node_index: torch.Tensor
    query_ptr: torch.Tensor
    graph_query_ptr: torch.Tensor
    candidate_query_index: torch.Tensor
    candidate_atom_embeddings_a: torch.Tensor
    candidate_atom_embeddings_b: torch.Tensor
    candidate_atom_ids: torch.Tensor
    candidate_atom_capacities: torch.Tensor
    candidate_distances: torch.Tensor
    query_geometry_valid: torch.Tensor
    positive_candidate_mask: torch.Tensor | None
    query_metadata: list[dict[str, Any]]
    sample_ids: list[str]

    @property
    def num_nodes(self) -> int:
        return int(self.node_type.shape[0])

    @property
    def num_queries(self) -> int:
        return int(self.query_node_index.shape[1])

    @property
    def num_candidates(self) -> int:
        return int(self.candidate_query_index.shape[0])

    def to(self, device: torch.device | str, non_blocking: bool = False) -> "AtomPairBatch":
        values: dict[str, Any] = {}
        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)
            if isinstance(value, torch.Tensor):
                values[field_name] = value.to(device, non_blocking=non_blocking)
            else:
                values[field_name] = value
        return AtomPairBatch(**values)


def _field(data: Any, name: str) -> Any:
    if isinstance(data, Mapping) and name in data:
        return data[name]
    if hasattr(data, name):
        return getattr(data, name)
    try:
        return data[name]
    except Exception as exc:
        raise KeyError(f"PyGData is missing required field {name!r}") from exc


def _tensor(data: Any, name: str) -> torch.Tensor:
    value = _field(data, name)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value).__name__}")
    return value


def _normalize_connection_tensor(value: torch.Tensor, *, name: str) -> torch.Tensor:
    value = value.detach().cpu().long()
    if value.numel() == 0:
        return value.reshape(2, 0)
    if value.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional")
    if value.shape[0] == 2:
        return value.contiguous()
    if value.shape[1] == 2:
        return value.t().contiguous()
    raise ValueError(f"{name} must have shape [2,C] or [C,2], got {tuple(value.shape)}")


def validate_graph_contract(data: Any, *, require_targets: bool) -> dict[str, Any]:
    node_type = _tensor(data, "node_type").detach().cpu().float()
    pos = _tensor(data, "pos").detach().cpu().float()
    ref_coords = _tensor(data, "ref_coords").detach().cpu().float()
    edge_index = _tensor(data, "edge_index").detach().cpu().long()
    edge_attr = _tensor(data, "edge_attr").detach().cpu().float()
    connection_edge_index = _normalize_connection_tensor(
        _tensor(data, "connection_edge_index"), name="connection_edge_index"
    )

    if node_type.ndim != 2 or node_type.shape[1] != 3:
        raise ValueError(f"node_type must have shape [N,3], got {tuple(node_type.shape)}")
    num_nodes = int(node_type.shape[0])
    if pos.shape != (num_nodes, 3):
        raise ValueError(f"pos must have shape {(num_nodes, 3)}, got {tuple(pos.shape)}")
    if ref_coords.shape != (num_nodes, 3, 3):
        raise ValueError(
            f"ref_coords must have shape {(num_nodes, 3, 3)}, got {tuple(ref_coords.shape)}"
        )
    if not torch.isfinite(node_type).all() or not torch.isfinite(pos).all():
        raise ValueError("node_type or pos contains NaN/Inf")
    if not torch.isfinite(ref_coords).all():
        raise ValueError("ref_coords contains NaN/Inf")
    if not torch.all((node_type == 0.0) | (node_type == 1.0)):
        raise ValueError("node_type must be strict one-hot")
    if not torch.all(node_type.sum(dim=1) == 1.0):
        raise ValueError("each node_type row must contain exactly one active class")

    raw_smiles = _field(data, "fragment_smiles")
    if isinstance(raw_smiles, str):
        raise TypeError("fragment_smiles must be a sequence")
    smiles = tuple(normalize_smiles_value(value) for value in list(raw_smiles))
    if len(smiles) != num_nodes:
        raise ValueError(
            f"fragment_smiles length must be {num_nodes}, got {len(smiles)}"
        )

    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2,E]")
    if edge_attr.shape != (edge_index.shape[1], 4):
        raise ValueError(
            f"edge_attr must have shape {(edge_index.shape[1], 4)}, got {tuple(edge_attr.shape)}"
        )
    if edge_index.numel() and (
        int(edge_index.min()) < 0 or int(edge_index.max()) >= num_nodes
    ):
        raise IndexError("edge_index contains an out-of-range node index")
    if not torch.isfinite(edge_attr).all():
        raise ValueError("edge_attr contains NaN/Inf")

    if connection_edge_index.numel() and (
        int(connection_edge_index.min()) < 0
        or int(connection_edge_index.max()) >= num_nodes
    ):
        raise IndexError("connection_edge_index contains an out-of-range node index")

    connection_atom_ids: torch.Tensor | None = None
    if require_targets:
        connection_atom_ids = _tensor(data, "connection_atom_ids").detach().cpu().long()
        connection_count = int(connection_edge_index.shape[1])
        if connection_atom_ids.numel() == 0:
            connection_atom_ids = connection_atom_ids.reshape(0, 2)
        if connection_atom_ids.shape == (2, connection_count) and connection_count != 2:
            connection_atom_ids = connection_atom_ids.t().contiguous()
        if connection_atom_ids.shape != (connection_count, 2):
            raise ValueError(
                "connection_atom_ids must have shape [C,2], got "
                f"{tuple(connection_atom_ids.shape)} for C={connection_count}"
            )
        if connection_atom_ids.numel() and int(connection_atom_ids.min()) < 0:
            raise ValueError("connection_atom_ids cannot contain negative IDs")

    return {
        "num_nodes": num_nodes,
        "node_type": node_type,
        "pos": pos,
        "ref_coords": ref_coords,
        "smiles": smiles,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "connection_edge_index": connection_edge_index,
        "connection_atom_ids": connection_atom_ids,
    }


def group_connections(validated: Mapping[str, Any], *, require_targets: bool) -> tuple[ConnectionGroup, ...]:
    edge_index: torch.Tensor = validated["connection_edge_index"]
    atom_ids: torch.Tensor | None = validated["connection_atom_ids"]
    node_type: torch.Tensor = validated["node_type"]
    groups: "OrderedDict[tuple[int, int], list[tuple[int, int]]]" = OrderedDict()

    for record_index in range(edge_index.shape[1]):
        first_node = int(edge_index[0, record_index])
        second_node = int(edge_index[1, record_index])
        if first_node == second_node:
            raise ValueError("connection_edge_index cannot contain self-loops")
        if not bool(node_type[first_node, 0]) or not bool(node_type[second_node, 0]):
            raise ValueError("connection endpoints must both be ligand nodes")

        first_atom = second_atom = -1
        if require_targets:
            assert atom_ids is not None
            first_atom = int(atom_ids[record_index, 0])
            second_atom = int(atom_ids[record_index, 1])
        if first_node > second_node:
            first_node, second_node = second_node, first_node
            first_atom, second_atom = second_atom, first_atom
        node_pair = (first_node, second_node)
        records = groups.setdefault(node_pair, [])
        if require_targets:
            atom_pair = (first_atom, second_atom)
            if atom_pair in records:
                raise ValueError(f"duplicate connection atom pair {node_pair + atom_pair}")
            records.append(atom_pair)

    result: list[ConnectionGroup] = []
    for (node_a, node_b), true_pairs in sorted(groups.items()):
        if require_targets and len(true_pairs) not in (1, 2):
            raise ValueError(
                f"node pair {(node_a, node_b)} must have one or two true atom pairs, "
                f"got {len(true_pairs)}"
            )
        result.append(
            ConnectionGroup(
                node_a=node_a,
                node_b=node_b,
                true_atom_pairs=tuple(true_pairs),
            )
        )
    return tuple(result)


class ConnectionFeatureBuilder:
    """Build the exact same model inputs for training and inference."""

    def __init__(
        self,
        fragment_vocabulary: FragmentVocabularyLookup,
        candidate_cache: FragmentCandidateCache,
        geometry_cache: ConformerGeometryCache,
        model_config: ModelConfig,
    ) -> None:
        self.fragment_vocabulary = fragment_vocabulary
        self.candidate_cache = candidate_cache
        self.geometry_cache = geometry_cache
        self.model_config = model_config

    def _node_embeddings(
        self, smiles: Sequence[str], node_type: torch.Tensor
    ) -> torch.Tensor:
        rows: list[np.ndarray] = []
        zero = np.zeros(
            (self.model_config.fragment_embedding_dim,), dtype=np.float32
        )
        for node_index, fragment_smiles in enumerate(smiles):
            is_global = bool(node_type[node_index, 2])
            if is_global:
                if fragment_smiles != GLOBAL_SMILES_TOKEN:
                    raise ValueError(
                        f"global node {node_index} must use {GLOBAL_SMILES_TOKEN!r}"
                    )
                rows.append(zero)
            else:
                rows.append(self.fragment_vocabulary.vector_numpy(fragment_smiles))
        return torch.from_numpy(np.stack(rows, axis=0).astype(np.float32, copy=False))

    def _model_edges(
        self,
        validated: Mapping[str, Any],
        groups: Sequence[ConnectionGroup],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        edge_index: torch.Tensor = validated["edge_index"]
        edge_attr: torch.Tensor = validated["edge_attr"]
        pos: torch.Tensor = validated["pos"]
        node_type: torch.Tensor = validated["node_type"]
        edge_map: dict[tuple[int, int], torch.Tensor] = {}

        # Drop every DL edge in this temporary view.  Source PyGData is untouched.
        for edge_number in range(edge_index.shape[1]):
            if bool(edge_attr[edge_number, 3]):
                continue
            source = int(edge_index[0, edge_number])
            target = int(edge_index[1, edge_number])
            # Keep the global node in the temporary node tensor, but isolate it
            # even if a malformed input happens to label a global edge as a
            # non-DL type.
            if bool(node_type[source, 2]) or bool(node_type[target, 2]):
                continue
            types = torch.zeros(4, dtype=torch.float32)
            types[:3] = edge_attr[edge_number, :3]
            key = (source, target)
            if key in edge_map:
                edge_map[key] = torch.maximum(edge_map[key], types)
            else:
                edge_map[key] = types

        for group in groups:
            for source, target in (
                (group.node_a, group.node_b),
                (group.node_b, group.node_a),
            ):
                types = edge_map.setdefault(
                    (source, target), torch.zeros(4, dtype=torch.float32)
                )
                types[3] = 1.0

        if not edge_map:
            return (
                torch.empty((2, 0), dtype=torch.long),
                torch.empty(
                    (0, self.model_config.edge_input_dim), dtype=torch.float32
                ),
            )

        ordered_keys = sorted(edge_map)
        model_edge_index = torch.tensor(ordered_keys, dtype=torch.long).t().contiguous()
        type_features = torch.stack([edge_map[key] for key in ordered_keys], dim=0)
        source, target = model_edge_index
        distances = torch.linalg.vector_norm(pos[target] - pos[source], dim=1)
        distance_rbf = gaussian_rbf(
            distances,
            num_centers=self.model_config.rbf_bins,
            max_distance=self.model_config.rbf_max_distance,
        )
        model_edge_features = torch.cat([type_features, distance_rbf], dim=1)
        return model_edge_index, model_edge_features.float()

    def _aligned_node(
        self,
        node_index: int,
        validated: Mapping[str, Any],
        local_cache: dict[int, AlignedGeometryResult],
        *,
        source_path: str,
        sample_id: str,
        dataset_split: str,
    ) -> AlignedGeometryResult:
        if node_index in local_cache:
            return local_cache[node_index]
        smiles = validated["smiles"][node_index]
        chemistry = self.candidate_cache.get(smiles)
        center = validated["pos"][node_index].numpy().astype(np.float64, copy=False)
        reference_frame = (
            validated["ref_coords"][node_index] - validated["pos"][node_index][None, :]
        ).numpy().astype(np.float64, copy=False)
        result = self.geometry_cache.align(
            chemistry.canonical_smiles,
            center=center,
            reference_frame=reference_frame,
            source_path=source_path,
            sample_id=sample_id,
            node_index=node_index,
            dataset_split=dataset_split,
        )
        local_cache[node_index] = result
        return result

    @staticmethod
    def _known_target_query_exclusion(
        group: ConnectionGroup,
        left: FragmentCandidateEntry,
        right: FragmentCandidateEntry,
        missing_pairs: Sequence[tuple[int, int]],
    ) -> ExcludedTargetQuery | None:
        """Return the narrow, approved exclusion for charge-lost N targets."""

        affected_atoms: set[tuple[str, int, int]] = set()
        for atom_a_id, atom_b_id in missing_pairs:
            pair_has_known_error = False
            if (
                left.ineligible_reason(atom_a_id)
                == KNOWN_NEUTRALIZED_NITROGEN_TARGET_REASON
            ):
                affected_atoms.add(("a", group.node_a, int(atom_a_id)))
                pair_has_known_error = True
            if (
                right.ineligible_reason(atom_b_id)
                == KNOWN_NEUTRALIZED_NITROGEN_TARGET_REASON
            ):
                affected_atoms.add(("b", group.node_b, int(atom_b_id)))
                pair_has_known_error = True
            if not pair_has_known_error:
                return None

        return ExcludedTargetQuery(
            node_a=group.node_a,
            node_b=group.node_b,
            reason=KNOWN_NEUTRALIZED_NITROGEN_TARGET_REASON,
            true_atom_pairs=group.true_atom_pairs,
            missing_true_atom_pairs=tuple(missing_pairs),
            affected_atoms=tuple(sorted(affected_atoms)),
        )

    def _filter_target_groups(
        self,
        validated: Mapping[str, Any],
        groups: Sequence[ConnectionGroup],
        *,
        source_path: str,
    ) -> tuple[
        tuple[ConnectionGroup, ...],
        tuple[ExcludedTargetQuery, ...],
        int,
    ]:
        """Drop double-pair queries and the approved source-data defect."""

        retained: list[ConnectionGroup] = []
        excluded: list[ExcludedTargetQuery] = []
        filtered_double_queries = 0
        for group in groups:
            if group.true_count == 2:
                filtered_double_queries += 1
                continue
            left = self.candidate_cache.get(validated["smiles"][group.node_a])
            right = self.candidate_cache.get(validated["smiles"][group.node_b])
            candidate_set = {
                (left_id, right_id)
                for left_id in left.eligible_atom_ids
                for right_id in right.eligible_atom_ids
            }
            missing = tuple(
                pair for pair in group.true_atom_pairs if pair not in candidate_set
            )
            if missing:
                known_exclusion = self._known_target_query_exclusion(
                    group,
                    left,
                    right,
                    missing,
                )
                if known_exclusion is not None:
                    excluded.append(known_exclusion)
                    continue
                raise ValueError(
                    f"{source_path}: true atom pairs are absent from chemical "
                    f"candidates: nodes={(group.node_a, group.node_b)}, "
                    f"missing={list(missing)}"
                )
            if not candidate_set:
                raise ValueError(
                    f"{source_path}: zero chemical candidates for node pair "
                    f"{(group.node_a, group.node_b)}"
                )
            retained.append(group)
        return tuple(retained), tuple(excluded), filtered_double_queries

    def _validate_true_capacity(
        self,
        validated: Mapping[str, Any],
        groups: Sequence[ConnectionGroup],
        *,
        source_path: str,
    ) -> None:
        """Fail preflight when the retained true combination exceeds capacity."""

        usage: Counter[tuple[int, int]] = Counter()
        sources: dict[tuple[int, int], list[tuple[int, int]]] = {}
        capacities: dict[tuple[int, int], int] = {}
        for group in groups:
            if group.true_count != 1:
                raise ValueError(
                    "retained supervision must contain exactly one atom pair "
                    f"for nodes {(group.node_a, group.node_b)}"
                )
            atom_a_id, atom_b_id = group.true_atom_pairs[0]
            left = self.candidate_cache.get(validated["smiles"][group.node_a])
            right = self.candidate_cache.get(validated["smiles"][group.node_b])
            for key, capacity in (
                ((group.node_a, atom_a_id), left.capacity(atom_a_id)),
                ((group.node_b, atom_b_id), right.capacity(atom_b_id)),
            ):
                usage[key] += 1
                capacities[key] = int(capacity)
                sources.setdefault(key, []).append((group.node_a, group.node_b))

        violations: list[dict[str, Any]] = [
            {
                "node": node,
                "atom_id": atom_id,
                "usage": int(used),
                "capacity": int(capacities[(node, atom_id)]),
                "queries": sources[(node, atom_id)],
            }
            for (node, atom_id), used in sorted(usage.items())
            if used > capacities[(node, atom_id)]
        ]
        if violations:
            diagnostic_violations: list[dict[str, Any]] = []
            for violation in violations:
                node_index = int(violation["node"])
                atom_id = int(violation["atom_id"])
                try:
                    atom_probe = diagnose_atom_single_bond_capacity(
                        validated["smiles"][node_index],
                        atom_id,
                    )
                except Exception as exc:
                    atom_probe = {
                        "status": "diagnostic_failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                diagnostic_violations.append(
                    {
                        **violation,
                        "node_smiles": validated["smiles"][node_index],
                        "atom_probe": atom_probe,
                    }
                )

            truth_connections = [
                {
                    "node_a": int(group.node_a),
                    "node_b": int(group.node_b),
                    "atom_a_id": int(group.true_atom_pairs[0][0]),
                    "atom_b_id": int(group.true_atom_pairs[0][1]),
                }
                for group in groups
            ]
            referenced_nodes = {
                int(connection[key])
                for connection in truth_connections
                for key in ("node_a", "node_b")
            }
            full_combination = diagnose_fragment_connection_combination(
                {
                    node_index: validated["smiles"][node_index]
                    for node_index in sorted(referenced_nodes)
                },
                truth_connections,
            )
            if full_combination.get("status") == "passed":
                interpretation = "capacity_probe_underestimation_candidate"
            elif full_combination.get("stage") == "final_sanitization":
                interpretation = (
                    "true_combination_rejected_by_canonical_fragment_chemistry"
                )
            else:
                interpretation = "full_combination_diagnostic_failed"
            diagnostics = {
                "constraint": (
                    "reuse_across_distinct_node_pairs_is_allowed_up_to_the_"
                    "atom_single_bond_capacity"
                ),
                "interpretation": interpretation,
                "violations": diagnostic_violations,
                "full_true_combination": full_combination,
            }
            raise TrueCombinationCapacityError(
                f"{source_path}: retained true global combination exceeds "
                f"atom capacity: {violations}",
                diagnostics,
            )

    def inspect(self, data: Any, *, path: str = "<memory>") -> GraphFileStats:
        """Validate labels/candidate coverage without running conformer generation."""

        validated = validate_graph_contract(data, require_targets=True)
        # Fail before training if any ordinary node cannot use the fixed
        # fragment lookup (and verify the isolated global-node convention).
        self._node_embeddings(validated["smiles"], validated["node_type"])
        original_groups = group_connections(validated, require_targets=True)
        (
            groups,
            excluded_target_queries,
            filtered_double_queries,
        ) = self._filter_target_groups(
            validated,
            original_groups,
            source_path=str(path),
        )
        self._validate_true_capacity(
            validated,
            groups,
            source_path=str(path),
        )
        total_candidates = 0
        for group in groups:
            left = self.candidate_cache.get(validated["smiles"][group.node_a])
            right = self.candidate_cache.get(validated["smiles"][group.node_b])
            candidate_set = {
                (left_id, right_id)
                for left_id in left.eligible_atom_ids
                for right_id in right.eligible_atom_ids
            }
            total_candidates += len(candidate_set)
        return GraphFileStats(
            path=str(path),
            num_nodes=int(validated["num_nodes"]),
            num_queries=len(groups),
            num_candidates=total_candidates,
            num_filtered_double_queries=filtered_double_queries,
            excluded_target_queries=excluded_target_queries,
        )

    def prepare(
        self,
        data: Any,
        *,
        include_targets: bool,
        source_path: str = "<memory>",
        dataset_split: str = "inference",
    ) -> PreparedConnectionGraph:
        resolved_source_path = str(source_path)
        resolved_dataset_split = str(dataset_split).strip()
        if not resolved_dataset_split:
            raise ValueError("dataset_split must be a non-empty string")
        sample_id = str(
            getattr(data, "sample_id", Path(resolved_source_path).stem)
        )
        validated = validate_graph_contract(data, require_targets=include_targets)
        original_groups = group_connections(
            validated,
            require_targets=include_targets,
        )
        excluded_target_queries: tuple[ExcludedTargetQuery, ...] = ()
        filtered_double_queries = 0
        if include_targets:
            (
                groups,
                excluded_target_queries,
                filtered_double_queries,
            ) = self._filter_target_groups(
                validated,
                original_groups,
                source_path=resolved_source_path,
            )
            self._validate_true_capacity(
                validated,
                groups,
                source_path=resolved_source_path,
            )
        else:
            groups = original_groups
        node_fragment_embeddings = self._node_embeddings(
            validated["smiles"], validated["node_type"]
        )
        model_edge_index, model_edge_features = self._model_edges(validated, groups)

        query_nodes: list[tuple[int, int]] = []
        candidate_query: list[int] = []
        candidate_embeddings_a: list[np.ndarray] = []
        candidate_embeddings_b: list[np.ndarray] = []
        candidate_atom_ids: list[tuple[int, int]] = []
        candidate_atom_capacities: list[tuple[int, int]] = []
        candidate_distances: list[float] = []
        query_geometry_valid: list[bool] = []
        positive_mask: list[bool] = []
        query_metadata: list[dict[str, Any]] = []
        aligned_nodes: dict[int, AlignedGeometryResult] = {}

        for query_index, group in enumerate(groups):
            left_smiles = validated["smiles"][group.node_a]
            right_smiles = validated["smiles"][group.node_b]
            left = self.candidate_cache.get(left_smiles)
            right = self.candidate_cache.get(right_smiles)
            candidate_pairs = [
                (left_id, right_id)
                for left_id in left.eligible_atom_ids
                for right_id in right.eligible_atom_ids
            ]
            if not candidate_pairs:
                raise ValueError(
                    f"{source_path}: zero candidates for nodes "
                    f"{(group.node_a, group.node_b)}"
                )

            left_geometry = self._aligned_node(
                group.node_a,
                validated,
                aligned_nodes,
                source_path=resolved_source_path,
                sample_id=sample_id,
                dataset_split=resolved_dataset_split,
            )
            right_geometry = self._aligned_node(
                group.node_b,
                validated,
                aligned_nodes,
                source_path=resolved_source_path,
                sample_id=sample_id,
                dataset_split=resolved_dataset_split,
            )
            geometry_valid = left_geometry.valid and right_geometry.valid
            if geometry_valid:
                geometry_valid = all(
                    left_id in left_geometry.coordinates
                    and right_id in right_geometry.coordinates
                    for left_id, right_id in candidate_pairs
                )

            query_nodes.append((group.node_a, group.node_b))
            query_geometry_valid.append(bool(geometry_valid))
            true_pair_set = set(group.true_atom_pairs)
            for left_id, right_id in candidate_pairs:
                candidate_query.append(query_index)
                candidate_atom_ids.append((left_id, right_id))
                candidate_atom_capacities.append(
                    (left.capacity(left_id), right.capacity(right_id))
                )
                candidate_embeddings_a.append(left.embedding(left_id))
                candidate_embeddings_b.append(right.embedding(right_id))
                positive_mask.append((left_id, right_id) in true_pair_set)
                if geometry_valid:
                    distance = float(
                        np.linalg.norm(
                            left_geometry.coordinates[left_id]
                            - right_geometry.coordinates[right_id]
                        )
                    )
                    if not math.isfinite(distance):
                        raise ValueError("candidate distance is not finite")
                    candidate_distances.append(distance)
                else:
                    candidate_distances.append(0.0)

            geometry_errors = [
                value
                for value in (left_geometry.error, right_geometry.error)
                if value is not None
            ]
            query_metadata.append(
                {
                    "source_path": resolved_source_path,
                    "sample_id": sample_id,
                    "dataset_split": resolved_dataset_split,
                    "node_a": group.node_a,
                    "node_b": group.node_b,
                    "smiles_a": left_smiles,
                    "smiles_b": right_smiles,
                    "candidate_count": len(candidate_pairs),
                    "actual_geometry_valid": bool(geometry_valid),
                    "geometry_error": " | ".join(geometry_errors) or None,
                    "true_atom_pairs": list(group.true_atom_pairs),
                    "filtered_double_query_count": filtered_double_queries,
                    "source_excluded_target_query_count": len(
                        excluded_target_queries
                    ),
                }
            )

        num_candidates = len(candidate_query)
        atom_dim = self.model_config.atom_embedding_dim
        if num_candidates:
            embeddings_a_tensor = torch.from_numpy(
                np.stack(candidate_embeddings_a).astype(np.float32, copy=False)
            )
            embeddings_b_tensor = torch.from_numpy(
                np.stack(candidate_embeddings_b).astype(np.float32, copy=False)
            )
            atom_ids_tensor = torch.tensor(candidate_atom_ids, dtype=torch.long)
            atom_capacities_tensor = torch.tensor(
                candidate_atom_capacities, dtype=torch.long
            )
        else:
            embeddings_a_tensor = torch.empty((0, atom_dim), dtype=torch.float32)
            embeddings_b_tensor = torch.empty((0, atom_dim), dtype=torch.float32)
            atom_ids_tensor = torch.empty((0, 2), dtype=torch.long)
            atom_capacities_tensor = torch.empty((0, 2), dtype=torch.long)

        return PreparedConnectionGraph(
            node_fragment_embeddings=node_fragment_embeddings.float(),
            node_type=validated["node_type"].float(),
            node_pos=validated["pos"].float(),
            edge_index=model_edge_index,
            edge_features=model_edge_features,
            query_node_index=(
                torch.tensor(query_nodes, dtype=torch.long).t().contiguous()
                if query_nodes
                else torch.empty((2, 0), dtype=torch.long)
            ),
            candidate_query_index=torch.tensor(candidate_query, dtype=torch.long),
            candidate_atom_embeddings_a=embeddings_a_tensor,
            candidate_atom_embeddings_b=embeddings_b_tensor,
            candidate_atom_ids=atom_ids_tensor,
            candidate_atom_capacities=atom_capacities_tensor,
            candidate_distances=torch.tensor(candidate_distances, dtype=torch.float32),
            query_geometry_valid=torch.tensor(query_geometry_valid, dtype=torch.bool),
            positive_candidate_mask=(
                torch.tensor(positive_mask, dtype=torch.bool)
                if include_targets
                else None
            ),
            query_metadata=query_metadata,
            sample_id=sample_id,
        )


class AtomPairGraphDataset(Dataset[PreparedConnectionGraph]):
    def __init__(
        self,
        file_paths: Sequence[str | Path],
        builder: ConnectionFeatureBuilder,
        *,
        include_targets: bool,
        dataset_split: str = "unspecified",
    ) -> None:
        self.file_paths = tuple(Path(path).expanduser().resolve() for path in file_paths)
        self.builder = builder
        self.include_targets = bool(include_targets)
        self.dataset_split = str(dataset_split).strip()
        if not self.dataset_split:
            raise ValueError("dataset_split must be a non-empty string")

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, index: int) -> PreparedConnectionGraph:
        path = self.file_paths[index]
        data = torch_load_compat(path, map_location="cpu")
        return self.builder.prepare(
            data,
            include_targets=self.include_targets,
            source_path=str(path),
            dataset_split=self.dataset_split,
        )


def collate_prepared_graphs(graphs: Sequence[PreparedConnectionGraph]) -> AtomPairBatch:
    if not graphs:
        raise ValueError("cannot collate an empty graph list")

    node_embeddings: list[torch.Tensor] = []
    node_types: list[torch.Tensor] = []
    node_positions: list[torch.Tensor] = []
    edge_indices: list[torch.Tensor] = []
    edge_features: list[torch.Tensor] = []
    query_indices: list[torch.Tensor] = []
    candidate_query_indices: list[torch.Tensor] = []
    candidate_embeddings_a: list[torch.Tensor] = []
    candidate_embeddings_b: list[torch.Tensor] = []
    candidate_atom_ids: list[torch.Tensor] = []
    candidate_atom_capacities: list[torch.Tensor] = []
    candidate_distances: list[torch.Tensor] = []
    query_geometry_valid: list[torch.Tensor] = []
    positive_masks: list[torch.Tensor] = []
    query_ptr = [0]
    graph_query_ptr = [0]
    metadata: list[dict[str, Any]] = []
    sample_ids: list[str] = []

    node_offset = 0
    query_offset = 0
    positive_presence = [
        graph.positive_candidate_mask is not None for graph in graphs
    ]
    if any(positive_presence) and not all(positive_presence):
        raise ValueError("cannot mix supervised and inference graphs in one batch")
    targets_present = all(positive_presence)

    for graph in graphs:
        node_embeddings.append(graph.node_fragment_embeddings)
        node_types.append(graph.node_type)
        node_positions.append(graph.node_pos)
        edge_indices.append(graph.edge_index + node_offset)
        edge_features.append(graph.edge_features)
        query_indices.append(graph.query_node_index + node_offset)
        candidate_query_indices.append(graph.candidate_query_index + query_offset)
        candidate_embeddings_a.append(graph.candidate_atom_embeddings_a)
        candidate_embeddings_b.append(graph.candidate_atom_embeddings_b)
        candidate_atom_ids.append(graph.candidate_atom_ids)
        candidate_atom_capacities.append(graph.candidate_atom_capacities)
        candidate_distances.append(graph.candidate_distances)
        query_geometry_valid.append(graph.query_geometry_valid)
        if targets_present:
            assert graph.positive_candidate_mask is not None
            positive_masks.append(graph.positive_candidate_mask)

        if graph.num_queries:
            counts = torch.bincount(
                graph.candidate_query_index, minlength=graph.num_queries
            ).tolist()
            for count in counts:
                query_ptr.append(query_ptr[-1] + int(count))
        metadata.extend(graph.query_metadata)
        sample_ids.append(graph.sample_id)
        graph_query_ptr.append(graph_query_ptr[-1] + graph.num_queries)
        node_offset += graph.num_nodes
        query_offset += graph.num_queries

    return AtomPairBatch(
        node_fragment_embeddings=torch.cat(node_embeddings, dim=0),
        node_type=torch.cat(node_types, dim=0),
        node_pos=torch.cat(node_positions, dim=0),
        edge_index=torch.cat(edge_indices, dim=1),
        edge_features=torch.cat(edge_features, dim=0),
        query_node_index=torch.cat(query_indices, dim=1),
        query_ptr=torch.tensor(query_ptr, dtype=torch.long),
        graph_query_ptr=torch.tensor(graph_query_ptr, dtype=torch.long),
        candidate_query_index=torch.cat(candidate_query_indices, dim=0),
        candidate_atom_embeddings_a=torch.cat(candidate_embeddings_a, dim=0),
        candidate_atom_embeddings_b=torch.cat(candidate_embeddings_b, dim=0),
        candidate_atom_ids=torch.cat(candidate_atom_ids, dim=0),
        candidate_atom_capacities=torch.cat(
            candidate_atom_capacities, dim=0
        ),
        candidate_distances=torch.cat(candidate_distances, dim=0),
        query_geometry_valid=torch.cat(query_geometry_valid, dim=0),
        positive_candidate_mask=(
            torch.cat(positive_masks, dim=0) if targets_present else None
        ),
        query_metadata=metadata,
        sample_ids=sample_ids,
    )


class CandidateBudgetBatchSampler(Sampler[list[int]]):
    """Pack whole graphs without truncating any candidate pairs."""

    def __init__(
        self,
        candidate_counts: Sequence[int],
        *,
        max_graphs: int,
        max_candidates: int,
        shuffle: bool,
        seed: int,
    ) -> None:
        if max_graphs <= 0 or max_candidates <= 0:
            raise ValueError("batch budgets must be positive")
        self.candidate_counts = tuple(max(0, int(value)) for value in candidate_counts)
        self.max_graphs = int(max_graphs)
        self.max_candidates = int(max_candidates)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _indices(self) -> list[int]:
        indices = list(range(len(self.candidate_counts)))
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(indices)
        return indices

    def _pack(self, indices: Iterable[int]) -> Iterator[list[int]]:
        batch: list[int] = []
        candidate_total = 0
        for index in indices:
            graph_candidates = self.candidate_counts[index]
            exceeds = batch and (
                len(batch) >= self.max_graphs
                or candidate_total + graph_candidates > self.max_candidates
            )
            if exceeds:
                yield batch
                batch = []
                candidate_total = 0
            batch.append(index)
            candidate_total += graph_candidates
            if len(batch) >= self.max_graphs:
                yield batch
                batch = []
                candidate_total = 0
        if batch:
            yield batch

    def __iter__(self) -> Iterator[list[int]]:
        yield from self._pack(self._indices())

    def __len__(self) -> int:
        return sum(1 for _ in self._pack(self._indices()))


def build_dataloader(
    dataset: AtomPairGraphDataset,
    stats: Sequence[GraphFileStats],
    *,
    max_graphs: int,
    max_candidates: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
    pin_memory: bool,
) -> tuple[DataLoader, CandidateBudgetBatchSampler]:
    if len(dataset) != len(stats):
        raise ValueError("dataset and stats lengths differ")
    sampler = CandidateBudgetBatchSampler(
        [stat.num_candidates for stat in stats],
        max_graphs=max_graphs,
        max_candidates=max_candidates,
        shuffle=shuffle,
        seed=seed,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_prepared_graphs,
        persistent_workers=(num_workers > 0),
    )
    return loader, sampler


def discover_pyg_files(data_dir: str | Path, pattern: str = "*.pt") -> list[Path]:
    root = Path(data_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"PyG data directory not found: {root}")
    files = sorted(path.resolve() for path in root.rglob(pattern) if path.is_file())
    if not files:
        raise FileNotFoundError(f"no {pattern!r} files found below {root}")
    return files


def run_preflight(
    file_paths: Sequence[str | Path],
    builder: ConnectionFeatureBuilder,
    *,
    report_path: str | Path | None = None,
) -> list[GraphFileStats]:
    stats: list[GraphFileStats] = []
    excluded_files: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for path_value in progress_iterable(
        file_paths,
        description="Preflight",
        total=len(file_paths),
        unit="file",
    ):
        path = Path(path_value).expanduser().resolve()
        try:
            data = torch_load_compat(path, map_location="cpu")
            stats.append(builder.inspect(data, path=str(path)))
        except Exception as exc:
            failure: dict[str, Any] = {
                "path": str(path),
                "error": f"{type(exc).__name__}: {exc}",
            }
            if isinstance(exc, TrueCombinationCapacityError):
                failure["failure_type"] = "true_combination_capacity"
                failure["diagnostics"] = dict(exc.diagnostics)
                if (
                    exc.diagnostics.get("interpretation")
                    == "true_combination_rejected_by_canonical_fragment_chemistry"
                ):
                    failure["exclusion_reason"] = (
                        "invalid_true_combination_chemistry"
                    )
                    excluded_files.append(failure)
                    continue
            failures.append(failure)
    capacity_records = [
        *excluded_files,
        *[
            failure
            for failure in failures
            if failure.get("failure_type") == "true_combination_capacity"
        ],
    ]
    report = {
        "supervision_policy": ATOM_PAIR_SUPERVISION_POLICY,
        "files_scanned": len(file_paths),
        "files_valid": len(stats),
        "files_excluded": len(excluded_files),
        "files_eligible_for_split": sum(item.num_queries > 0 for item in stats),
        "files_without_trainable_queries": sum(
            item.num_queries == 0 for item in stats
        ),
        "files_failed": len(failures),
        "capacity_excluded_file_count": len(excluded_files),
        "capacity_failure_count": sum(
            failure.get("failure_type") == "true_combination_capacity"
            for failure in failures
        ),
        "capacity_full_combination_status_counts": dict(
            Counter(
                failure.get("diagnostics", {})
                .get("full_true_combination", {})
                .get("status", "unknown")
                for failure in capacity_records
            )
        ),
        "excluded_file_reason_counts": dict(
            Counter(
                str(item["exclusion_reason"])
                for item in excluded_files
            )
        ),
        "original_total_queries": sum(
            item.num_original_queries for item in stats
        ),
        "total_queries": sum(item.num_queries for item in stats),
        "excluded_target_queries": sum(
            item.num_excluded_queries for item in stats
        ),
        "excluded_target_query_reason_counts": dict(
            Counter(
                exclusion.reason
                for item in stats
                for exclusion in item.excluded_target_queries
            )
        ),
        "total_candidates": sum(item.num_candidates for item in stats),
        "filtered_double_queries": sum(
            item.num_filtered_double_queries for item in stats
        ),
        "excluded_files": excluded_files,
        "failures": failures,
        "excluded_target_query_details": [
            {
                "path": item.path,
                **asdict(exclusion),
            }
            for item in stats
            for exclusion in item.excluded_target_queries
        ],
        "files": [asdict(item) for item in stats],
    }
    if report_path is not None:
        atomic_write_json(report_path, report)
    if excluded_files:
        print(
            "Preflight exclusions: "
            f"invalid_true_combination_files={len(excluded_files)}"
        )
    if failures:
        first = failures[0]
        report_hint = (
            f"; inspect failures[].diagnostics in "
            f"{Path(report_path).expanduser().resolve()}"
            if report_path is not None
            else ""
        )
        raise RuntimeError(
            f"preflight failed for {len(failures)} files; first failure: "
            f"{first['path']}: {first['error']}{report_hint}"
        )
    return stats


def _split_one_stratum(
    indices: list[int], train_fraction: float, rng: random.Random
) -> tuple[list[int], list[int]]:
    rng.shuffle(indices)
    if len(indices) <= 1:
        return indices, []
    train_count = int(round(len(indices) * train_fraction))
    train_count = min(max(train_count, 1), len(indices) - 1)
    return indices[:train_count], indices[train_count:]


def create_or_load_split(
    stats: Sequence[GraphFileStats],
    *,
    data_dir: str | Path,
    manifest_path: str | Path,
    train_fraction: float,
    seed: int,
    rebuild: bool = False,
) -> tuple[list[GraphFileStats], list[GraphFileStats]]:
    root = Path(data_dir).expanduser().resolve()
    manifest = Path(manifest_path).expanduser().resolve()
    by_relative: dict[str, GraphFileStats] = {}
    for item in stats:
        relative = str(Path(item.path).resolve().relative_to(root))
        by_relative[relative] = item

    if manifest.is_file() and not rebuild:
        import json

        with manifest.open("r", encoding="utf-8") as file_obj:
            payload = json.load(file_obj)
        if (
            str(payload.get("supervision_policy", ""))
            != ATOM_PAIR_SUPERVISION_POLICY
        ):
            raise ValueError(
                "split manifest uses an incompatible supervision policy; "
                "rebuild it with --rebuild-split"
            )
        train_names = list(payload.get("train", []))
        validation_names = list(payload.get("validation", []))
        if not train_names or not validation_names:
            raise ValueError("split manifest has an empty training or validation set")
        if len(set(train_names)) != len(train_names) or len(
            set(validation_names)
        ) != len(validation_names):
            raise ValueError("split manifest contains duplicate file entries")
        overlap = set(train_names) & set(validation_names)
        if overlap:
            raise ValueError(
                f"split manifest has train/validation overlap: {sorted(overlap)[:10]}"
            )
        missing = [name for name in train_names + validation_names if name not in by_relative]
        if missing:
            raise ValueError(f"split manifest references missing files: {missing[:10]}")
        eligible_names = {
            name for name, item in by_relative.items() if item.num_queries > 0
        }
        listed_names = set(train_names) | set(validation_names)
        if listed_names != eligible_names:
            omitted = sorted(eligible_names - listed_names)
            unexpected = sorted(listed_names - eligible_names)
            raise ValueError(
                "split manifest does not exactly match the current eligible graph set; "
                f"omitted={omitted[:10]}, unexpected={unexpected[:10]}"
            )
        return (
            [by_relative[name] for name in train_names],
            [by_relative[name] for name in validation_names],
        )

    eligible = [index for index, item in enumerate(stats) if item.num_queries > 0]
    if len(eligible) < 2:
        raise ValueError(
            "at least two graph files with retained single-pair queries are "
            "required for a train/validation split"
        )
    rng = random.Random(seed)
    train_indices, validation_indices = _split_one_stratum(
        eligible, train_fraction, rng
    )
    if not train_indices or not validation_indices:
        raise ValueError("random split produced an empty training or validation set")

    train_stats = [stats[index] for index in train_indices]
    validation_stats = [stats[index] for index in validation_indices]
    payload = {
        "supervision_policy": ATOM_PAIR_SUPERVISION_POLICY,
        "seed": int(seed),
        "train_fraction": float(train_fraction),
        "stratified_by_double_connection": False,
        "train": [str(Path(item.path).resolve().relative_to(root)) for item in train_stats],
        "validation": [
            str(Path(item.path).resolve().relative_to(root))
            for item in validation_stats
        ],
    }
    atomic_write_json(manifest, payload)
    return train_stats, validation_stats
