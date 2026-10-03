"""Neural network for inter-fragment single atom-pair ranking.

The model consumes a *flat* batch.  ``batch`` may be a mapping or any object
whose attributes have the following names (``N`` nodes, ``E`` directed graph
edges, ``Q`` unique undirected fragment-pair queries and ``C`` candidate atom
pairs):

``node_fragment_embeddings`` (float, ``[N, 128]``)
    Frozen fragment-vocabulary embeddings.  The data layer must put zeros in
    the row of the disconnected ``<GLOBAL>`` node.
``node_type`` (float, ``[N, 3]``)
    Ligand/pocket/global one-hot node type.
``node_pos`` (float, ``[N, 3]``)
    Fragment centroids.  They are used only to build the rotation/translation
    invariant graph edge geometry.
``edge_index`` (long, ``[2, E]``)
    Directed source-to-destination indices: row 0 is source ``j`` and row 1
    is destination ``i``.
``edge_features`` (float, ``[E, 20]``)
    ``LL/LP/PP/KNOWN_CONNECTION`` followed by 16 centroid-distance RBFs.
    DL edges must already have been removed by the data layer.
``query_node_index`` (long, ``[2, Q]``)
    The two node indices of every unique connection query.
``candidate_query_index`` (long, ``[C]``)
    Query index for each flat candidate row.  Candidate rows belonging to a
    query need not be contiguous.
``candidate_atom_embeddings_a`` / ``candidate_atom_embeddings_b``
    Frozen 64-dimensional atom-vocabulary embeddings, each ``[C, 64]``.
``candidate_distances`` (float, ``[C]``)
    Reconstructed atom-pair distance in angstrom.  Use zero for queries whose
    conformer/alignment failed; distance is never used as a hard filter.
``query_geometry_valid`` (bool or 0/1, ``[Q]``)
    Whether reconstructed atom geometry is available for the whole query.

``forward(batch, apply_geometry_dropout=False, force_disable_geometry=False,
geometry_generator=None)`` returns :class:`AtomPairModelOutput`:

``pair_logits`` (``[C]``)
    Flat candidate logits.  Grouped softmax/loss is deliberately handled by
    the training engine, using ``candidate_query_index``.
``node_states`` (``[N, 192]``)
    Contextual fragment states, useful for diagnostics.
``geometry_used`` (bool, ``[Q]``)
    Geometry actually visible in this call, after optional query-level
    dropout/forced disabling.  If a query is dropped, all of its 18 pair
    geometry features (16 RBF + capped distance + valid flag) are exactly 0.

The module never reads labels, ``connection_distance``, PyG ``x``, HAC, ring
counts, metal masks, sample identity, or DL edges.  Candidate chemistry and
conformer construction belong to the dataset/utilities modules.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

try:  # Package import (``import PhiSLinker...``).
    from .PhiSLinker_atom_pair_config import DEFAULT_MODEL_CONFIG, ModelConfig
except ImportError:  # Script-directory import on the training server.
    from PhiSLinker_atom_pair_config import DEFAULT_MODEL_CONFIG, ModelConfig

try:
    from torch_geometric.utils import softmax as _pyg_segment_softmax
except ImportError:  # Keep the model importable for CPU-only unit tests.
    _pyg_segment_softmax = None


@dataclass(frozen=True)
class AtomPairModelOutput(Mapping[str, Tensor]):
    """Structured model output with both attribute and mapping access."""

    pair_logits: Tensor
    node_states: Tensor
    geometry_used: Tensor

    def __getitem__(self, key: str) -> Tensor:
        if key not in self.keys():
            raise KeyError(key)
        return getattr(self, key)

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys())

    def __len__(self) -> int:
        return 3

    @staticmethod
    def keys() -> tuple[str, ...]:
        return (
            "pair_logits",
            "node_states",
            "geometry_used",
        )


def _batch_field(batch: Any, name: str) -> Tensor:
    """Read a required tensor from a mapping or attribute-style batch."""

    if isinstance(batch, Mapping):
        if name not in batch:
            raise KeyError(f"model batch is missing required field {name!r}")
        value = batch[name]
    else:
        try:
            value = getattr(batch, name)
        except AttributeError as exc:
            raise AttributeError(
                f"model batch is missing required attribute {name!r}"
            ) from exc
    if not isinstance(value, Tensor):
        raise TypeError(
            f"batch.{name} must be a torch.Tensor, got {type(value).__name__}"
        )
    return value


def _segment_softmax(logits: Tensor, index: Tensor, num_segments: int) -> Tensor:
    """Softmax over edges sharing a destination, without torch_scatter."""

    if logits.ndim != 1 or index.ndim != 1 or logits.shape != index.shape:
        raise ValueError("segment softmax expects equally shaped 1-D tensors")
    if logits.numel() == 0:
        return logits
    if _pyg_segment_softmax is not None:
        return _pyg_segment_softmax(logits, index, num_nodes=num_segments)

    # Fallback for minimal test environments without torch_geometric.  Modern
    # PyTorch supplies scatter_reduce_; the short loop supports older builds.
    maxima = logits.new_full((num_segments,), -torch.inf)
    if hasattr(maxima, "scatter_reduce_"):
        maxima.scatter_reduce_(
            0, index, logits, reduce="amax", include_self=True
        )
    else:  # pragma: no cover - only for obsolete PyTorch versions.
        for segment in torch.unique(index).tolist():
            mask = index == segment
            maxima[segment] = logits[mask].max()
    exponentials = torch.exp(logits - maxima.index_select(0, index))
    denominators = logits.new_zeros((num_segments,))
    denominators.index_add_(0, index, exponentials)
    tiny = torch.finfo(exponentials.dtype).tiny
    return exponentials / denominators.index_select(0, index).clamp_min(tiny)


def apply_query_geometry_mask(
    candidate_geometry: Tensor,
    candidate_query_index: Tensor,
    query_geometry_valid: Tensor,
    dropout_probability: float,
    *,
    apply_dropout: bool = False,
    force_disable: bool = False,
    generator: torch.Generator | None = None,
    random_values: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Mask complete query groups and return ``(geometry, geometry_used)``.

    ``random_values`` is an optional ``[Q]`` tensor intended for deterministic
    unit tests.  Supplying both it and ``generator`` is an error.  Geometry is
    never dropped candidate by candidate: one Boolean decision is expanded
    through ``candidate_query_index`` to every row belonging to that query.
    """

    if candidate_geometry.ndim != 2:
        raise ValueError("candidate_geometry must have shape [C, D]")
    if candidate_query_index.ndim != 1:
        raise ValueError("candidate_query_index must have shape [C]")
    if query_geometry_valid.ndim != 1:
        raise ValueError("query_geometry_valid must have shape [Q]")
    if candidate_geometry.shape[0] != candidate_query_index.shape[0]:
        raise ValueError(
            "candidate_geometry and candidate_query_index must have equal rows"
        )
    if not 0.0 <= float(dropout_probability) < 1.0:
        raise ValueError("dropout_probability must be in [0, 1)")
    if generator is not None and random_values is not None:
        raise ValueError("provide generator or random_values, not both")
    if candidate_query_index.numel() and (
        int(candidate_query_index.min()) < 0
        or int(candidate_query_index.max()) >= query_geometry_valid.shape[0]
    ):
        raise IndexError("candidate_query_index contains an invalid query")

    valid = query_geometry_valid.to(dtype=torch.bool)
    geometry_used = valid.clone()
    if force_disable:
        geometry_used.zero_()
    elif apply_dropout and dropout_probability > 0.0:
        if random_values is not None:
            if random_values.shape != valid.shape:
                raise ValueError(
                    "random_values must have the same shape as "
                    "query_geometry_valid"
                )
            draws = random_values.to(device=valid.device, dtype=torch.float32)
        elif generator is None:
            draws = torch.rand(
                valid.shape, device=valid.device, dtype=torch.float32
            )
        else:
            generator_device = getattr(
                generator, "device", torch.device("cpu")
            )
            draws = torch.rand(
                valid.shape,
                device=generator_device,
                dtype=torch.float32,
                generator=generator,
            ).to(device=valid.device)
        geometry_used &= draws >= float(dropout_probability)

    candidate_used = geometry_used.index_select(0, candidate_query_index)
    mask = candidate_used.to(dtype=candidate_geometry.dtype).unsqueeze(-1)
    return candidate_geometry * mask, geometry_used


class GaussianRBF(nn.Module):
    """Fixed Gaussian radial basis with centers spanning ``[0, max_distance]``."""

    def __init__(self, bins: int = 16, max_distance: float = 12.0) -> None:
        super().__init__()
        if bins <= 0:
            raise ValueError("bins must be positive")
        if max_distance <= 0.0:
            raise ValueError("max_distance must be positive")
        centers = torch.linspace(0.0, float(max_distance), int(bins))
        if bins == 1:
            width = float(max_distance)
        else:
            width = float(max_distance) / float(bins - 1)
        self.bins = int(bins)
        self.max_distance = float(max_distance)
        self.register_buffer("centers", centers, persistent=True)
        self.register_buffer(
            "inverse_width_squared",
            torch.tensor(1.0 / (width * width), dtype=torch.float32),
            persistent=True,
        )

    def forward(self, distances: Tensor) -> Tensor:
        if distances.ndim != 1:
            raise ValueError(
                f"RBF distances must be 1-D, got shape {tuple(distances.shape)}"
            )
        # Do not clamp before the RBF: a distance beyond the configured range
        # should decay away from the final center.  This is identical to the
        # edge-RBF implementation in the dataset module.  Only the separate
        # scalar pair-distance feature is capped below.
        centers = self.centers.to(dtype=distances.dtype)
        scale = self.inverse_width_squared.to(dtype=distances.dtype)
        delta = distances.unsqueeze(-1) - centers
        return torch.exp(-scale * delta.square())


class EdgeAwareAttentionLayer(nn.Module):
    """One residual edge-aware attention/message-passing layer."""

    def __init__(
        self,
        node_dim: int,
        edge_input_dim: int,
        edge_hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.node_dim = int(node_dim)
        self.edge_input_dim = int(edge_input_dim)

        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input_dim, edge_hidden_dim),
            nn.LayerNorm(edge_hidden_dim),
            nn.SiLU(),
            nn.Linear(edge_hidden_dim, edge_hidden_dim),
            nn.SiLU(),
        )
        self.message_mlp = nn.Sequential(
            nn.Linear(node_dim + edge_hidden_dim, node_dim),
            nn.LayerNorm(node_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(node_dim, node_dim),
        )
        self.attention_mlp = nn.Sequential(
            nn.Linear(2 * node_dim + edge_hidden_dim, edge_hidden_dim),
            nn.SiLU(),
            nn.Linear(edge_hidden_dim, 1),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(2 * node_dim, node_dim),
            nn.LayerNorm(node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, node_dim),
        )
        self.residual_dropout = nn.Dropout(dropout)
        self.output_norm = nn.LayerNorm(node_dim)

    def forward(
        self, node_states: Tensor, edge_index: Tensor, edge_features: Tensor
    ) -> Tensor:
        num_nodes = node_states.shape[0]
        source, destination = edge_index[0], edge_index[1]

        if edge_features.shape[0] == 0:
            aggregated = node_states.new_zeros((num_nodes, self.node_dim))
        else:
            edge_states = self.edge_mlp(edge_features)
            source_states = node_states.index_select(0, source)
            destination_states = node_states.index_select(0, destination)
            messages = self.message_mlp(
                torch.cat((source_states, edge_states), dim=-1)
            )
            attention_logits = self.attention_mlp(
                torch.cat(
                    (destination_states, source_states, edge_states), dim=-1
                )
            ).squeeze(-1)
            attention = _segment_softmax(
                attention_logits, destination, num_segments=num_nodes
            )
            aggregated = node_states.new_zeros((num_nodes, self.node_dim))
            aggregated.index_add_(
                0, destination, messages * attention.unsqueeze(-1)
            )

        update = self.update_mlp(torch.cat((node_states, aggregated), dim=-1))
        return self.output_norm(
            node_states + self.residual_dropout(update)
        )


class AtomFusionMLP(nn.Module):
    """Fuse a fixed atom embedding with its contextual fragment state."""

    def __init__(
        self,
        atom_embedding_dim: int,
        node_hidden_dim: int,
        atom_hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.atom_embedding_dim = int(atom_embedding_dim)
        self.node_hidden_dim = int(node_hidden_dim)
        self.atom_hidden_dim = int(atom_hidden_dim)
        fusion_input_dim = atom_embedding_dim + node_hidden_dim
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_input_dim, atom_hidden_dim),
            nn.LayerNorm(atom_hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(atom_hidden_dim, atom_hidden_dim),
        )
        self.atom_projection = nn.Linear(
            atom_embedding_dim, atom_hidden_dim, bias=False
        )
        self.fragment_projection = nn.Linear(
            node_hidden_dim, atom_hidden_dim, bias=False
        )
        self.output_norm = nn.LayerNorm(atom_hidden_dim)

    def forward(self, atom_embeddings: Tensor, fragment_states: Tensor) -> Tensor:
        fused = self.fusion_mlp(
            torch.cat((atom_embeddings, fragment_states), dim=-1)
        )
        residual = self.atom_projection(atom_embeddings)
        residual = residual + self.fragment_projection(fragment_states)
        return self.output_norm(fused + residual)


class PhiSLinkerAtomPairModel(nn.Module):
    """Query-local candidate ranker shared by local and global training."""

    def __init__(self, config: ModelConfig | Mapping[str, Any] | None = None):
        super().__init__()
        if config is None:
            config = DEFAULT_MODEL_CONFIG
        elif isinstance(config, Mapping):
            config = ModelConfig(**dict(config))
        if not isinstance(config, ModelConfig):
            raise TypeError("config must be ModelConfig, a mapping, or None")
        config.validate()
        self.config = config

        self.node_mlp = nn.Sequential(
            nn.Linear(config.node_input_dim, config.node_hidden_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.node_hidden_dim, config.node_hidden_dim),
        )
        self.node_norm = nn.LayerNorm(config.node_hidden_dim)
        self.gnn_layers = nn.ModuleList(
            EdgeAwareAttentionLayer(
                node_dim=config.node_hidden_dim,
                edge_input_dim=config.edge_input_dim,
                edge_hidden_dim=config.edge_hidden_dim,
                dropout=config.dropout,
            )
            for _ in range(config.num_gnn_layers)
        )

        self.atom_fusion = AtomFusionMLP(
            atom_embedding_dim=config.atom_embedding_dim,
            node_hidden_dim=config.node_hidden_dim,
            atom_hidden_dim=config.atom_hidden_dim,
            dropout=config.dropout,
        )
        self.pair_mlp = nn.Sequential(
            nn.Linear(config.pair_input_dim, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(64, 1),
        )
        self.distance_rbf = GaussianRBF(
            bins=config.rbf_bins,
            max_distance=config.rbf_max_distance,
        )

    def encode_nodes(
        self,
        node_fragment_embeddings: Tensor,
        node_type: Tensor,
        edge_index: Tensor,
        edge_features: Tensor,
    ) -> Tensor:
        """Encode fragment nodes using three edge-aware attention layers."""

        node_input = torch.cat((node_fragment_embeddings, node_type), dim=-1)
        node_states = self.node_norm(self.node_mlp(node_input))
        for layer in self.gnn_layers:
            node_states = layer(node_states, edge_index, edge_features)
        return node_states

    def build_pair_geometry(
        self,
        candidate_distances: Tensor,
        candidate_query_index: Tensor,
        query_geometry_valid: Tensor,
        *,
        apply_geometry_dropout: bool = False,
        force_disable_geometry: bool = False,
        geometry_generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Build 18-D candidate geometry and the query-level used mask.

        Dropout is sampled once per query, never independently per candidate.
        ``geometry_generator`` may be a CPU or same-device generator; it exists
        so deterministic tests/training can control only this random stream.
        """

        dtype = candidate_distances.dtype
        rbf = self.distance_rbf(candidate_distances)
        scaled = (
            candidate_distances.clamp(
                min=0.0, max=self.config.rbf_max_distance
            )
            / self.config.rbf_max_distance
        ).unsqueeze(-1)
        raw_geometry = torch.cat(
            (rbf, scaled, torch.ones_like(scaled, dtype=dtype)), dim=-1
        )
        return apply_query_geometry_mask(
            raw_geometry,
            candidate_query_index,
            query_geometry_valid,
            self.config.geometry_dropout,
            apply_dropout=apply_geometry_dropout,
            force_disable=force_disable_geometry,
            generator=geometry_generator,
        )

    def _validate_batch_shapes(self, tensors: Mapping[str, Tensor]) -> None:
        """Fail early on interface drift while avoiding expensive value scans."""

        cfg = self.config
        expected_2d = {
            "node_fragment_embeddings": (None, cfg.fragment_embedding_dim),
            "node_type": (None, cfg.node_type_dim),
            "node_pos": (None, 3),
            "edge_index": (2, None),
            "edge_features": (None, cfg.edge_input_dim),
            "query_node_index": (2, None),
            "candidate_atom_embeddings_a": (None, cfg.atom_embedding_dim),
            "candidate_atom_embeddings_b": (None, cfg.atom_embedding_dim),
        }
        for name, expected in expected_2d.items():
            value = tensors[name]
            if value.ndim != 2:
                raise ValueError(
                    f"batch.{name} must be 2-D, got {tuple(value.shape)}"
                )
            for axis, size in enumerate(expected):
                if size is not None and value.shape[axis] != size:
                    raise ValueError(
                        f"batch.{name} axis {axis} must have size {size}, "
                        f"got shape {tuple(value.shape)}"
                    )
        for name in (
            "candidate_query_index",
            "candidate_distances",
            "query_geometry_valid",
        ):
            if tensors[name].ndim != 1:
                raise ValueError(
                    f"batch.{name} must be 1-D, got {tuple(tensors[name].shape)}"
                )

        num_nodes = tensors["node_fragment_embeddings"].shape[0]
        num_edges = tensors["edge_index"].shape[1]
        num_queries = tensors["query_node_index"].shape[1]
        num_candidates = tensors["candidate_query_index"].shape[0]
        equalities = {
            "node_type rows": (tensors["node_type"].shape[0], num_nodes),
            "node_pos rows": (tensors["node_pos"].shape[0], num_nodes),
            "edge_features rows": (
                tensors["edge_features"].shape[0],
                num_edges,
            ),
            "candidate_atom_embeddings_a rows": (
                tensors["candidate_atom_embeddings_a"].shape[0],
                num_candidates,
            ),
            "candidate_atom_embeddings_b rows": (
                tensors["candidate_atom_embeddings_b"].shape[0],
                num_candidates,
            ),
            "candidate_distances rows": (
                tensors["candidate_distances"].shape[0],
                num_candidates,
            ),
            "query_geometry_valid rows": (
                tensors["query_geometry_valid"].shape[0],
                num_queries,
            ),
        }
        for label, (actual, expected) in equalities.items():
            if actual != expected:
                raise ValueError(f"{label} must be {expected}, got {actual}")

        for name in (
            "edge_index",
            "query_node_index",
            "candidate_query_index",
        ):
            if tensors[name].dtype != torch.long:
                raise TypeError(f"batch.{name} must have dtype torch.long")

        device = tensors["node_fragment_embeddings"].device
        for name, value in tensors.items():
            if value.device != device:
                raise ValueError(
                    f"all model batch tensors must share device {device}; "
                    f"batch.{name} is on {value.device}"
                )

        # Range checks synchronize only tiny index tensors and make corrupted
        # batch offsets fail here instead of deep inside index_select.
        edge_index = tensors["edge_index"]
        query_nodes = tensors["query_node_index"]
        candidate_queries = tensors["candidate_query_index"]
        if edge_index.numel() and (
            int(edge_index.min()) < 0 or int(edge_index.max()) >= num_nodes
        ):
            raise IndexError("batch.edge_index contains an out-of-range node")
        if query_nodes.numel() and (
            int(query_nodes.min()) < 0 or int(query_nodes.max()) >= num_nodes
        ):
            raise IndexError(
                "batch.query_node_index contains an out-of-range node"
            )
        if candidate_queries.numel() and (
            int(candidate_queries.min()) < 0
            or int(candidate_queries.max()) >= num_queries
        ):
            raise IndexError(
                "batch.candidate_query_index contains an out-of-range query"
            )

    def forward(
        self,
        batch: Any,
        *,
        apply_geometry_dropout: bool = False,
        force_disable_geometry: bool = False,
        geometry_generator: torch.Generator | None = None,
    ) -> AtomPairModelOutput:
        fields = (
            "node_fragment_embeddings",
            "node_type",
            "node_pos",
            "edge_index",
            "edge_features",
            "query_node_index",
            "candidate_query_index",
            "candidate_atom_embeddings_a",
            "candidate_atom_embeddings_b",
            "candidate_distances",
            "query_geometry_valid",
        )
        tensors = {name: _batch_field(batch, name) for name in fields}
        self._validate_batch_shapes(tensors)

        node_states = self.encode_nodes(
            tensors["node_fragment_embeddings"],
            tensors["node_type"],
            tensors["edge_index"],
            tensors["edge_features"],
        )
        query_nodes = tensors["query_node_index"]
        candidate_queries = tensors["candidate_query_index"]
        query_a_states = node_states.index_select(0, query_nodes[0])
        query_b_states = node_states.index_select(0, query_nodes[1])

        candidate_a_fragment_states = query_a_states.index_select(
            0, candidate_queries
        )
        candidate_b_fragment_states = query_b_states.index_select(
            0, candidate_queries
        )
        candidate_a_states = self.atom_fusion(
            tensors["candidate_atom_embeddings_a"],
            candidate_a_fragment_states,
        )
        candidate_b_states = self.atom_fusion(
            tensors["candidate_atom_embeddings_b"],
            candidate_b_fragment_states,
        )
        pair_geometry, geometry_used = self.build_pair_geometry(
            tensors["candidate_distances"],
            candidate_queries,
            tensors["query_geometry_valid"],
            apply_geometry_dropout=apply_geometry_dropout,
            force_disable_geometry=force_disable_geometry,
            geometry_generator=geometry_generator,
        )
        pair_geometry = pair_geometry.to(dtype=candidate_a_states.dtype)
        pair_features = torch.cat(
            (
                candidate_a_states + candidate_b_states,
                torch.abs(candidate_a_states - candidate_b_states),
                candidate_a_states * candidate_b_states,
                pair_geometry,
            ),
            dim=-1,
        )
        pair_logits = self.pair_mlp(pair_features).squeeze(-1)

        return AtomPairModelOutput(
            pair_logits=pair_logits,
            node_states=node_states,
            geometry_used=geometry_used,
        )


# Short aliases keep downstream scripts readable and checkpoint construction
# explicit while retaining one canonical implementation.
AtomPairModel = PhiSLinkerAtomPairModel
AtomPairPredictor = PhiSLinkerAtomPairModel


__all__ = [
    "AtomPairModel",
    "AtomPairModelOutput",
    "AtomPairPredictor",
    "AtomFusionMLP",
    "EdgeAwareAttentionLayer",
    "GaussianRBF",
    "PhiSLinkerAtomPairModel",
    "apply_query_geometry_mask",
]
