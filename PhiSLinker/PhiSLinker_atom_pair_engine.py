"""Local ranking and capacity-aware global-combination training engine."""

from __future__ import annotations

import hashlib
import math
import random
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from torch import Tensor, nn

try:
    from PhiSLinker_atom_pair_config import TrainConfig
    from PhiSLinker_atom_pair_dataset import AtomPairBatch
    from PhiSLinker_atom_pair_model import AtomPairModelOutput
    from PhiSLinker_atom_pair_utils import (
        GlobalAtomPairChoice,
        beam_search_atom_pair_combinations,
        progress_iterable,
    )
except ImportError:  # pragma: no cover - package-style import fallback
    from .PhiSLinker_atom_pair_config import TrainConfig
    from .PhiSLinker_atom_pair_dataset import AtomPairBatch
    from .PhiSLinker_atom_pair_model import AtomPairModelOutput
    from .PhiSLinker_atom_pair_utils import (
        GlobalAtomPairChoice,
        beam_search_atom_pair_combinations,
        progress_iterable,
    )


TRAINING_STAGES = frozenset({"local", "global"})


@dataclass(frozen=True)
class ObjectiveResult:
    """Differentiable objective plus additive epoch statistics."""

    total_loss: Tensor
    local_loss: Tensor
    valid_combination_loss: Tensor
    conflict_combination_loss: Tensor
    local_loss_sum: Tensor
    valid_combination_loss_sum: Tensor
    conflict_combination_loss_sum: Tensor
    local_query_count: int
    valid_combination_count: int
    conflict_combination_count: int
    query_count: int
    graph_count: int
    fixed_query_count: int
    local_top1_correct: int
    unknown_query_count: int
    truth_in_top_k_count: int
    global_exact_count: int


@dataclass(frozen=True)
class EpochResult:
    total_loss: float
    local_loss: float
    valid_combination_loss: float
    conflict_combination_loss: float
    query_count: int
    graph_count: int
    local_query_count: int
    valid_combination_count: int
    conflict_combination_count: int
    fixed_query_count: int
    local_top1_accuracy: float | None = None
    top_k_truth_recall: float | None = None
    global_exact_accuracy: float | None = None

    def to_dict(self) -> dict[str, float | int | None]:
        return {
            "total_loss": self.total_loss,
            "local_loss": self.local_loss,
            "valid_combination_loss": self.valid_combination_loss,
            "conflict_combination_loss": self.conflict_combination_loss,
            "query_count": self.query_count,
            "graph_count": self.graph_count,
            "local_query_count": self.local_query_count,
            "valid_combination_count": self.valid_combination_count,
            "conflict_combination_count": self.conflict_combination_count,
            "fixed_query_count": self.fixed_query_count,
            "local_top1_accuracy": self.local_top1_accuracy,
            "top_k_truth_recall": self.top_k_truth_recall,
            "global_exact_accuracy": self.global_exact_accuracy,
        }


def _validate_stage(training_stage: str) -> str:
    stage = str(training_stage).strip().lower()
    if stage not in TRAINING_STAGES:
        raise ValueError(f"training_stage must be one of {sorted(TRAINING_STAGES)}")
    return stage


def _require_supervision(batch: AtomPairBatch) -> Tensor:
    positive_mask = batch.positive_candidate_mask
    if positive_mask is None:
        raise ValueError("training/validation batch does not contain supervision")
    if positive_mask.shape != (batch.num_candidates,):
        raise ValueError("positive_candidate_mask must have shape [C]")
    return positive_mask.bool()


def _candidate_bounds(batch: AtomPairBatch, query_index: int) -> tuple[int, int]:
    if batch.query_ptr.shape != (batch.num_queries + 1,):
        raise ValueError("query_ptr must have shape [Q+1]")
    start = int(batch.query_ptr[query_index].item())
    end = int(batch.query_ptr[query_index + 1].item())
    if end <= start:
        raise ValueError(f"query {query_index} has no candidates")
    expected = batch.candidate_query_index[start:end]
    if not torch.all(expected == query_index):
        raise ValueError(
            "candidate rows must be contiguous and agree with query_ptr; "
            f"query {query_index} is inconsistent"
        )
    return start, end


def _segment_logsumexp(values: Tensor, index: Tensor, num_segments: int) -> Tensor:
    values = values.float()
    maxima = values.new_full((num_segments,), -torch.inf)
    if hasattr(maxima, "scatter_reduce_"):
        maxima.scatter_reduce_(0, index, values, reduce="amax", include_self=True)
    else:  # pragma: no cover - only for obsolete PyTorch versions
        for segment in range(num_segments):
            maxima[segment] = values[index == segment].max()
    shifted = torch.exp(values - maxima.index_select(0, index))
    sums = values.new_zeros((num_segments,))
    sums.index_add_(0, index, shifted)
    return maxima + torch.log(sums)


def grouped_log_probabilities(pair_logits: Tensor, batch: AtomPairBatch) -> Tensor:
    """Return query-local log-softmax values for all flat candidates."""

    if pair_logits.shape != (batch.num_candidates,):
        raise ValueError("pair_logits must have shape [C]")
    candidate_counts = torch.bincount(
        batch.candidate_query_index, minlength=batch.num_queries
    )
    if torch.any(candidate_counts == 0):
        raise ValueError("every query must have at least one candidate")
    logits = pair_logits.float()
    normalizers = _segment_logsumexp(
        logits, batch.candidate_query_index, batch.num_queries
    )
    return logits - normalizers.index_select(0, batch.candidate_query_index)


def _stable_random(*parts: Any) -> random.Random:
    digest = hashlib.sha256(
        "\x1f".join(str(part) for part in parts).encode("utf-8")
    ).digest()
    return random.Random(int.from_bytes(digest[:8], "big", signed=False))


def sample_fixed_query_mask(
    batch: AtomPairBatch,
    train_config: TrainConfig,
    *,
    training_stage: str,
    random_seed: int,
    epoch: int,
    validation: bool,
) -> Tensor:
    """Sample whole fixed query pairs independently for each graph."""

    stage = _validate_stage(training_stage)
    mask = torch.zeros(
        (batch.num_queries,), dtype=torch.bool, device=batch.query_node_index.device
    )
    if stage == "local":
        return mask
    if batch.graph_query_ptr.shape != (len(batch.sample_ids) + 1,):
        raise ValueError("graph_query_ptr must have shape [G+1]")

    for graph_index, sample_id in enumerate(batch.sample_ids):
        start = int(batch.graph_query_ptr[graph_index].item())
        end = int(batch.graph_query_ptr[graph_index + 1].item())
        query_count = end - start
        if query_count < 2:
            continue
        source_path = str(batch.query_metadata[start].get("source_path", ""))
        epoch_key: int | str = "validation" if validation else int(epoch)
        rng = _stable_random(
            int(random_seed), epoch_key, sample_id, source_path
        )
        if rng.random() >= train_config.fixed_query_scenario_probability:
            continue
        fraction = rng.uniform(
            train_config.fixed_query_min_fraction,
            train_config.fixed_query_max_fraction,
        )
        fixed_count = int(round(fraction * query_count))
        fixed_count = min(max(fixed_count, 1), query_count - 1)
        for query_index in rng.sample(range(start, end), fixed_count):
            mask[query_index] = True
    return mask


def _true_candidate_rows(
    batch: AtomPairBatch, positive_mask: Tensor
) -> tuple[Tensor, Tensor]:
    positive_counts = torch.bincount(
        batch.candidate_query_index[positive_mask], minlength=batch.num_queries
    )
    if not torch.all(positive_counts == 1):
        raise ValueError(
            "single-pair supervision requires exactly one positive candidate "
            "for every retained query"
        )
    rows = torch.nonzero(positive_mask, as_tuple=False).flatten()
    ordered = rows.new_empty((batch.num_queries,))
    ordered[batch.candidate_query_index.index_select(0, rows)] = rows
    return ordered, positive_counts


def _capacity_map_for_graph(
    batch: AtomPairBatch, query_start: int, query_end: int
) -> dict[tuple[int, int], int]:
    if batch.candidate_atom_capacities.shape != (batch.num_candidates, 2):
        raise ValueError("candidate_atom_capacities must have shape [C,2]")
    capacities: dict[tuple[int, int], int] = {}
    for query_index in range(query_start, query_end):
        start, end = _candidate_bounds(batch, query_index)
        node_a = int(batch.query_node_index[0, query_index].item())
        node_b = int(batch.query_node_index[1, query_index].item())
        for row in range(start, end):
            for endpoint, atom_column in ((node_a, 0), (node_b, 1)):
                atom_id = int(batch.candidate_atom_ids[row, atom_column].item())
                capacity = int(
                    batch.candidate_atom_capacities[row, atom_column].item()
                )
                key = (endpoint, atom_id)
                previous = capacities.setdefault(key, capacity)
                if previous != capacity:
                    raise ValueError(
                        f"inconsistent capacity for atom {key}: "
                        f"{previous} versus {capacity}"
                    )
    return capacities


def _choice_for_row(
    batch: AtomPairBatch,
    log_probabilities: Tensor,
    *,
    local_query_index: int,
    global_query_index: int,
    candidate_row: int,
) -> GlobalAtomPairChoice:
    return GlobalAtomPairChoice(
        query_index=int(local_query_index),
        candidate_index=int(candidate_row),
        node_a=int(batch.query_node_index[0, global_query_index].item()),
        node_b=int(batch.query_node_index[1, global_query_index].item()),
        atom_a_id=int(batch.candidate_atom_ids[candidate_row, 0].item()),
        atom_b_id=int(batch.candidate_atom_ids[candidate_row, 1].item()),
        score=float(log_probabilities[candidate_row].detach().item()),
    )


def _ranked_candidate_rows(
    batch: AtomPairBatch,
    log_probabilities: Tensor,
    query_index: int,
) -> list[int]:
    start, end = _candidate_bounds(batch, query_index)
    return sorted(
        range(start, end),
        key=lambda row: (
            -float(log_probabilities[row].detach().item()),
            int(batch.candidate_atom_ids[row, 0].item()),
            int(batch.candidate_atom_ids[row, 1].item()),
            row,
        ),
    )


def _global_graph_terms(
    batch: AtomPairBatch,
    log_probabilities: Tensor,
    true_rows: Tensor,
    fixed_mask: Tensor,
    train_config: TrainConfig,
    *,
    graph_index: int,
    inject_truth: bool,
) -> tuple[list[Tensor], list[Tensor], int, int, int]:
    query_start = int(batch.graph_query_ptr[graph_index].item())
    query_end = int(batch.graph_query_ptr[graph_index + 1].item())
    capacities = _capacity_map_for_graph(batch, query_start, query_end)
    initial_usage: dict[tuple[int, int], int] = {}
    unknown_queries: list[int] = []
    for query_index in range(query_start, query_end):
        if bool(fixed_mask[query_index].item()):
            row = int(true_rows[query_index].item())
            node_a = int(batch.query_node_index[0, query_index].item())
            node_b = int(batch.query_node_index[1, query_index].item())
            endpoints = (
                (node_a, int(batch.candidate_atom_ids[row, 0].item())),
                (node_b, int(batch.candidate_atom_ids[row, 1].item())),
            )
            for endpoint in endpoints:
                initial_usage[endpoint] = initial_usage.get(endpoint, 0) + 1
        else:
            unknown_queries.append(query_index)

    query_choices: list[list[GlobalAtomPairChoice]] = []
    truth_indices: list[int] = []
    truth_in_top_k = 0
    for local_query_index, query_index in enumerate(unknown_queries):
        ranked = _ranked_candidate_rows(batch, log_probabilities, query_index)
        true_row = int(true_rows[query_index].item())
        top_rows = ranked[: train_config.global_top_k]
        truth_in_top_k += int(true_row in top_rows)
        if inject_truth and true_row not in top_rows:
            if len(top_rows) >= train_config.global_top_k:
                top_rows[-1] = true_row
            else:
                top_rows.append(true_row)
            top_rows = sorted(
                set(top_rows),
                key=lambda row: (
                    -float(log_probabilities[row].detach().item()),
                    int(batch.candidate_atom_ids[row, 0].item()),
                    int(batch.candidate_atom_ids[row, 1].item()),
                    row,
                ),
            )
        query_choices.append(
            [
                _choice_for_row(
                    batch,
                    log_probabilities,
                    local_query_index=local_query_index,
                    global_query_index=query_index,
                    candidate_row=row,
                )
                for row in top_rows
            ]
        )
        truth_indices.append(true_row)

    sample_id = str(batch.sample_ids[graph_index])
    source_path = ""
    if query_start < query_end and query_start < len(batch.query_metadata):
        source_path = str(
            batch.query_metadata[query_start].get("source_path", "")
        )
    try:
        legal, conflicts = beam_search_atom_pair_combinations(
            query_choices,
            capacities,
            initial_usage=initial_usage,
            beam_width=train_config.global_beam_width,
            max_legal_results=train_config.global_max_legal_negatives + 1,
            max_conflict_results=train_config.global_max_conflict_negatives,
        )
    except Exception as exc:
        choice_counts = tuple(len(choices) for choices in query_choices)
        raise RuntimeError(
            "global beam search failed: "
            f"graph_index={graph_index}, sample_id={sample_id!r}, "
            f"source_path={source_path!r}, "
            f"query_range=[{query_start}, {query_end}), "
            f"graph_queries={query_end - query_start}, "
            f"unknown_queries={len(unknown_queries)}, "
            f"fixed_queries={query_end - query_start - len(unknown_queries)}, "
            f"choice_counts={choice_counts}, "
            f"capacity_atoms={len(capacities)}, "
            f"initial_usage_atoms={len(initial_usage)}, "
            f"inject_truth={inject_truth}"
        ) from exc
    truth_tuple = tuple(truth_indices)
    truth_index_tensor = torch.tensor(
        truth_indices, dtype=torch.long, device=log_probabilities.device
    )
    true_score = log_probabilities.index_select(0, truth_index_tensor).sum()

    def checked_candidate_indices(
        combination: Any,
        *,
        result_kind: str,
        result_index: int,
    ) -> tuple[int, ...]:
        combination_type = (
            f"{type(combination).__module__}."
            f"{type(combination).__qualname__}"
        )
        try:
            choices = combination.choices
            choice_types = tuple(
                f"{type(choice).__module__}.{type(choice).__qualname__}"
                for choice in choices
            )
        except Exception as exc:
            raise RuntimeError(
                "global beam result choices cannot be inspected: "
                f"graph_index={graph_index}, sample_id={sample_id!r}, "
                f"source_path={source_path!r}, result_kind={result_kind!r}, "
                f"result_index={result_index}, "
                f"combination_type={combination_type}"
            ) from exc

        context = (
            f"graph_index={graph_index}, sample_id={sample_id!r}, "
            f"source_path={source_path!r}, result_kind={result_kind!r}, "
            f"result_index={result_index}, "
            f"combination_type={combination_type}, "
            f"choices_type={type(choices).__module__}."
            f"{type(choices).__qualname__}, choice_types={choice_types}"
        )
        invalid_choice_positions = tuple(
            position
            for position, choice in enumerate(choices)
            if not isinstance(choice, GlobalAtomPairChoice)
        )
        if invalid_choice_positions:
            raise RuntimeError(
                "global beam result contains invalid choice objects: "
                f"{context}, invalid_positions={invalid_choice_positions}"
            )

        try:
            candidate_indices = combination.candidate_indices
        except Exception as exc:
            raise RuntimeError(
                "global beam result candidate indices cannot be read: "
                f"{context}"
            ) from exc
        if (
            not isinstance(candidate_indices, tuple)
            or len(candidate_indices) != len(choices)
            or len(candidate_indices) != len(query_choices)
            or any(type(index) is not int for index in candidate_indices)
        ):
            raise RuntimeError(
                "global beam result candidate indices are inconsistent: "
                f"{context}, candidate_indices={candidate_indices!r}, "
                f"expected_count={len(query_choices)}"
            )
        return candidate_indices

    valid_terms: list[Tensor] = []
    first_legal_indices: tuple[int, ...] | None = None
    for result_index, combination in enumerate(legal):
        candidate_indices = checked_candidate_indices(
            combination,
            result_kind="legal",
            result_index=result_index,
        )
        if result_index == 0:
            first_legal_indices = candidate_indices
        if candidate_indices == truth_tuple:
            continue
        index_tensor = torch.tensor(
            candidate_indices,
            dtype=torch.long,
            device=log_probabilities.device,
        )
        negative_score = log_probabilities.index_select(0, index_tensor).sum()
        valid_terms.append(
            F.softplus(
                negative_score
                - true_score
                + float(train_config.valid_combination_margin)
            )
        )
        if len(valid_terms) >= train_config.global_max_legal_negatives:
            break

    conflict_terms: list[Tensor] = []
    for result_index, combination in enumerate(
        conflicts[: train_config.global_max_conflict_negatives]
    ):
        candidate_indices = checked_candidate_indices(
            combination,
            result_kind="conflict",
            result_index=result_index,
        )
        index_tensor = torch.tensor(
            candidate_indices,
            dtype=torch.long,
            device=log_probabilities.device,
        )
        negative_score = log_probabilities.index_select(0, index_tensor).sum()
        margin = (
            float(train_config.valid_combination_margin)
            + int(combination.overflow_count)
            * float(train_config.conflict_overflow_margin)
        )
        conflict_terms.append(F.softplus(negative_score - true_score + margin))

    global_exact = int(first_legal_indices == truth_tuple)
    return (
        valid_terms,
        conflict_terms,
        truth_in_top_k,
        len(unknown_queries),
        global_exact,
    )


def compute_objective(
    output: AtomPairModelOutput,
    batch: AtomPairBatch,
    *,
    training_stage: str,
    train_config: TrainConfig,
    epoch: int,
    validation: bool,
) -> ObjectiveResult:
    """Compute local NLL and optional capacity-aware combination margins."""

    stage = _validate_stage(training_stage)
    if output.pair_logits.shape != (batch.num_candidates,):
        raise ValueError("pair_logits must have shape [C]")
    positive_mask = _require_supervision(batch)
    true_rows, _ = _true_candidate_rows(batch, positive_mask)
    log_probabilities = grouped_log_probabilities(output.pair_logits, batch)
    fixed_mask = sample_fixed_query_mask(
        batch,
        train_config,
        training_stage=stage,
        random_seed=train_config.random_seed,
        epoch=epoch,
        validation=validation,
    )

    candidate_counts = torch.bincount(
        batch.candidate_query_index, minlength=batch.num_queries
    )
    effective_local = (~fixed_mask) & (candidate_counts > 1)
    local_terms = -log_probabilities.index_select(0, true_rows)
    local_query_count = int(effective_local.sum().item())
    if local_query_count:
        local_loss_sum = local_terms[effective_local].sum()
        local_loss = local_loss_sum / float(local_query_count)
    else:
        local_loss_sum = output.pair_logits.float().sum() * 0.0
        local_loss = local_loss_sum

    unknown_mask = ~fixed_mask
    local_top1_correct = 0
    for query_index in torch.nonzero(unknown_mask, as_tuple=False).flatten().tolist():
        ranked = _ranked_candidate_rows(batch, log_probabilities, query_index)
        local_top1_correct += int(ranked[0] == int(true_rows[query_index].item()))

    valid_terms: list[Tensor] = []
    conflict_terms: list[Tensor] = []
    truth_in_top_k = 0
    unknown_query_count = int(unknown_mask.sum().item())
    global_exact_count = 0
    if stage == "global":
        if batch.graph_query_ptr.shape != (len(batch.sample_ids) + 1,):
            raise ValueError("graph_query_ptr must have shape [G+1]")
        truth_in_top_k = 0
        unknown_query_count = 0
        for graph_index in range(len(batch.sample_ids)):
            (
                graph_valid_terms,
                graph_conflict_terms,
                graph_top_k,
                graph_unknown_count,
                graph_exact,
            ) = _global_graph_terms(
                batch,
                log_probabilities,
                true_rows,
                fixed_mask,
                train_config,
                graph_index=graph_index,
                inject_truth=not validation,
            )
            valid_terms.extend(graph_valid_terms)
            conflict_terms.extend(graph_conflict_terms)
            truth_in_top_k += graph_top_k
            unknown_query_count += graph_unknown_count
            global_exact_count += graph_exact

    zero = output.pair_logits.float().sum() * 0.0
    valid_loss_sum = torch.stack(valid_terms).sum() if valid_terms else zero
    conflict_loss_sum = (
        torch.stack(conflict_terms).sum() if conflict_terms else zero
    )
    valid_loss = (
        valid_loss_sum / float(len(valid_terms)) if valid_terms else zero
    )
    conflict_loss = (
        conflict_loss_sum / float(len(conflict_terms)) if conflict_terms else zero
    )
    total_loss = local_loss * float(train_config.local_loss_weight)
    if stage == "global":
        total_loss = (
            total_loss
            + valid_loss * float(train_config.valid_combination_loss_weight)
            + conflict_loss
            * float(train_config.conflict_combination_loss_weight)
        )

    return ObjectiveResult(
        total_loss=total_loss,
        local_loss=local_loss,
        valid_combination_loss=valid_loss,
        conflict_combination_loss=conflict_loss,
        local_loss_sum=local_loss_sum,
        valid_combination_loss_sum=valid_loss_sum,
        conflict_combination_loss_sum=conflict_loss_sum,
        local_query_count=local_query_count,
        valid_combination_count=len(valid_terms),
        conflict_combination_count=len(conflict_terms),
        query_count=batch.num_queries,
        graph_count=len(batch.sample_ids),
        fixed_query_count=int(fixed_mask.sum().item()),
        local_top1_correct=local_top1_correct,
        unknown_query_count=unknown_query_count,
        truth_in_top_k_count=truth_in_top_k,
        global_exact_count=global_exact_count,
    )


class _EpochAccumulator:
    def __init__(self) -> None:
        self.local_loss_sum = 0.0
        self.valid_loss_sum = 0.0
        self.conflict_loss_sum = 0.0
        self.local_query_count = 0
        self.valid_count = 0
        self.conflict_count = 0
        self.query_count = 0
        self.graph_count = 0
        self.fixed_query_count = 0
        self.local_top1_correct = 0
        self.unknown_query_count = 0
        self.truth_in_top_k_count = 0
        self.global_exact_count = 0

    def add(self, objective: ObjectiveResult) -> None:
        self.local_loss_sum += float(objective.local_loss_sum.detach().item())
        self.valid_loss_sum += float(
            objective.valid_combination_loss_sum.detach().item()
        )
        self.conflict_loss_sum += float(
            objective.conflict_combination_loss_sum.detach().item()
        )
        self.local_query_count += objective.local_query_count
        self.valid_count += objective.valid_combination_count
        self.conflict_count += objective.conflict_combination_count
        self.query_count += objective.query_count
        self.graph_count += objective.graph_count
        self.fixed_query_count += objective.fixed_query_count
        self.local_top1_correct += objective.local_top1_correct
        self.unknown_query_count += objective.unknown_query_count
        self.truth_in_top_k_count += objective.truth_in_top_k_count
        self.global_exact_count += objective.global_exact_count

    def finish(
        self,
        *,
        training_stage: str,
        train_config: TrainConfig,
        include_metrics: bool,
    ) -> EpochResult:
        stage = _validate_stage(training_stage)
        if self.query_count <= 0 or self.graph_count <= 0:
            raise RuntimeError("epoch contains no connection queries")
        local_loss = (
            self.local_loss_sum / self.local_query_count
            if self.local_query_count
            else 0.0
        )
        valid_loss = (
            self.valid_loss_sum / self.valid_count if self.valid_count else 0.0
        )
        conflict_loss = (
            self.conflict_loss_sum / self.conflict_count
            if self.conflict_count
            else 0.0
        )
        total_loss = local_loss * train_config.local_loss_weight
        if stage == "global":
            total_loss += (
                valid_loss * train_config.valid_combination_loss_weight
                + conflict_loss * train_config.conflict_combination_loss_weight
            )
        if not all(
            math.isfinite(value)
            for value in (local_loss, valid_loss, conflict_loss, total_loss)
        ):
            raise FloatingPointError("epoch loss contains NaN/Inf")
        metrics = {
            "local_top1_accuracy": None,
            "top_k_truth_recall": None,
            "global_exact_accuracy": None,
        }
        if include_metrics:
            metrics["local_top1_accuracy"] = (
                self.local_top1_correct / self.unknown_query_count
                if self.unknown_query_count
                else None
            )
            if stage == "global":
                metrics["top_k_truth_recall"] = (
                    self.truth_in_top_k_count / self.unknown_query_count
                    if self.unknown_query_count
                    else None
                )
                metrics["global_exact_accuracy"] = (
                    self.global_exact_count / self.graph_count
                )
        return EpochResult(
            total_loss=total_loss,
            local_loss=local_loss,
            valid_combination_loss=valid_loss,
            conflict_combination_loss=conflict_loss,
            query_count=self.query_count,
            graph_count=self.graph_count,
            local_query_count=self.local_query_count,
            valid_combination_count=self.valid_count,
            conflict_combination_count=self.conflict_count,
            fixed_query_count=self.fixed_query_count,
            **metrics,
        )


def _autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def train_one_epoch(
    model: nn.Module,
    loader: Iterable[AtomPairBatch],
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    training_stage: str,
    train_config: TrainConfig,
    epoch: int,
    gradient_clip_norm: float,
    use_amp: bool,
    scaler: Any | None = None,
) -> EpochResult:
    model.train()
    accumulator = _EpochAccumulator()
    for cpu_batch in progress_iterable(loader, description="Training", unit="batch"):
        batch = cpu_batch.to(device, non_blocking=device.type == "cuda")
        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(device, use_amp):
            output = model(batch, apply_geometry_dropout=True)
            objective = compute_objective(
                output,
                batch,
                training_stage=training_stage,
                train_config=train_config,
                epoch=epoch,
                validation=False,
            )
        if not bool(torch.isfinite(objective.total_loss).item()):
            raise FloatingPointError("training batch total loss is NaN/Inf")
        scaler_enabled = scaler is not None and bool(scaler.is_enabled())
        if scaler_enabled:
            scaler.scale(objective.total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            objective.total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()
        accumulator.add(objective)
    return accumulator.finish(
        training_stage=training_stage,
        train_config=train_config,
        include_metrics=False,
    )


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loader: Iterable[AtomPairBatch],
    *,
    device: torch.device,
    training_stage: str,
    train_config: TrainConfig,
    use_amp: bool,
) -> EpochResult:
    model.eval()
    accumulator = _EpochAccumulator()
    for cpu_batch in progress_iterable(
        loader, description="Validation", unit="batch"
    ):
        batch = cpu_batch.to(device, non_blocking=device.type == "cuda")
        with _autocast_context(device, use_amp):
            output = model(batch, apply_geometry_dropout=False)
            objective = compute_objective(
                output,
                batch,
                training_stage=training_stage,
                train_config=train_config,
                epoch=0,
                validation=True,
            )
        accumulator.add(objective)
    return accumulator.finish(
        training_stage=training_stage,
        train_config=train_config,
        include_metrics=True,
    )


@torch.no_grad()
def validate_one_epoch_detailed(
    model: nn.Module,
    loader: Iterable[AtomPairBatch],
    *,
    device: torch.device,
    training_stage: str,
    train_config: TrainConfig,
    use_amp: bool,
) -> tuple[EpochResult, dict[str, Any]]:
    """Run validation once and add compact single-pair ranking diagnostics."""

    model.eval()
    accumulator = _EpochAccumulator()
    query_count = 0
    recall_at_3 = 0
    recall_at_5 = 0
    reciprocal_rank_sum = 0.0
    for cpu_batch in progress_iterable(
        loader, description="Detailed validation", unit="batch"
    ):
        batch = cpu_batch.to(device, non_blocking=device.type == "cuda")
        with _autocast_context(device, use_amp):
            output = model(batch, apply_geometry_dropout=False)
            objective = compute_objective(
                output,
                batch,
                training_stage=training_stage,
                train_config=train_config,
                epoch=0,
                validation=True,
            )
        accumulator.add(objective)
        positive_mask = _require_supervision(batch)
        true_rows, _ = _true_candidate_rows(batch, positive_mask)
        log_probabilities = grouped_log_probabilities(output.pair_logits, batch)
        for query_index in range(batch.num_queries):
            ranking = _ranked_candidate_rows(batch, log_probabilities, query_index)
            true_row = int(true_rows[query_index].item())
            rank = ranking.index(true_row) + 1
            recall_at_3 += int(rank <= 3)
            recall_at_5 += int(rank <= 5)
            reciprocal_rank_sum += 1.0 / rank
            query_count += 1
    result = accumulator.finish(
        training_stage=training_stage,
        train_config=train_config,
        include_metrics=True,
    )
    if query_count == 0:
        raise RuntimeError("detailed evaluation contains no queries")
    details = {
        "ranking": {
            "recall_at_3": recall_at_3 / query_count,
            "recall_at_5": recall_at_5 / query_count,
            "mean_reciprocal_rank": reciprocal_rank_sum / query_count,
        },
        "global": {
            "top_k": train_config.global_top_k,
            "beam_width": train_config.global_beam_width,
            "top_k_truth_recall": result.top_k_truth_recall,
            "exact_combination_accuracy": result.global_exact_accuracy,
        },
    }
    return result, details


__all__ = [
    "EpochResult",
    "ObjectiveResult",
    "TRAINING_STAGES",
    "compute_objective",
    "grouped_log_probabilities",
    "sample_fixed_query_mask",
    "train_one_epoch",
    "validate_one_epoch",
    "validate_one_epoch_detailed",
]
