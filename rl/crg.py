"""Checkpoint-relation graph encoder for the frozen Paper-A Sec. 5.1 path."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from fjsp_sfr.rl.graph import (
    CHECKPOINT,
    JOB,
    MACHINE,
    PAPER_GRAPH,
    PAPER_RELATION_SPECS,
    HeteroGraphData,
)


@dataclass(frozen=True)
class CRGEncoding:
    """Public state and intermediate representations required by later heads."""

    state: torch.Tensor
    node_embeddings: dict[str, torch.Tensor]
    attentions: tuple[dict[str, torch.Tensor], ...]


def _relation_softmax(
    scores: torch.Tensor,
    targets: torch.Tensor,
    num_targets: int,
) -> torch.Tensor:
    """Softmax independently for every relation, destination, and head."""

    weights = torch.zeros_like(scores)
    for destination in range(num_targets):
        mask = targets == destination
        if bool(mask.any()):
            weights[mask] = torch.softmax(scores[mask], dim=0)
    return weights


class RelationGATv2Layer(nn.Module):
    """One relation-specific, edge-aware multi-head GATv2 layer."""

    def __init__(self, hidden_dim: int = 64, num_heads: int = 4) -> None:
        super().__init__()
        if hidden_dim <= 0 or hidden_dim % num_heads:
            raise ValueError("hidden_dim must be positive and divisible by num_heads")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.attention_projection = nn.ModuleDict()
        self.edge_attention = nn.ModuleDict()
        self.value_projection = nn.ModuleDict()
        self.edge_value = nn.ModuleDict()
        self.attention_vector = nn.ParameterDict()
        for relation, (_source, _target, edge_dim) in PAPER_RELATION_SPECS.items():
            self.edge_attention[relation] = nn.Linear(edge_dim, hidden_dim, bias=False)
            self.attention_projection[relation] = nn.Linear(
                3 * hidden_dim, hidden_dim, bias=False
            )
            self.value_projection[relation] = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.edge_value[relation] = nn.Linear(edge_dim, hidden_dim, bias=False)
            parameter = nn.Parameter(torch.empty(num_heads, self.head_dim))
            nn.init.xavier_uniform_(parameter)
            self.attention_vector[relation] = parameter
        self.output_projection = nn.ModuleDict(
            {
                node_type: nn.Linear(hidden_dim, hidden_dim, bias=False)
                for node_type in (JOB, MACHINE, CHECKPOINT)
            }
        )
        self.normalization = nn.ModuleDict(
            {node_type: nn.LayerNorm(hidden_dim) for node_type in (JOB, MACHINE, CHECKPOINT)}
        )
        self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(
        self,
        nodes: dict[str, torch.Tensor],
        edge_index: dict[str, torch.Tensor],
        edge_attr: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        aggregated = {node_type: torch.zeros_like(values) for node_type, values in nodes.items()}
        attentions: dict[str, torch.Tensor] = {}
        for relation, (source_type, target_type, edge_dim) in PAPER_RELATION_SPECS.items():
            index = edge_index[relation]
            attributes = edge_attr[relation]
            if index.shape != (2, attributes.shape[0]) or attributes.shape[1:] != (edge_dim,):
                raise ValueError(f"invalid edge tensors for relation {relation}")
            if not index.shape[1]:
                attentions[relation] = attributes.new_empty((0, self.num_heads))
                continue
            source = nodes[source_type][index[0]]
            target = nodes[target_type][index[1]]
            edge_for_attention = self.edge_attention[relation](attributes)
            projected = self.activation(
                self.attention_projection[relation](
                    torch.cat((source, target, edge_for_attention), dim=-1)
                )
            ).view(-1, self.num_heads, self.head_dim)
            scores = (projected * self.attention_vector[relation]).sum(dim=-1)
            alpha = _relation_softmax(scores, index[1], nodes[target_type].shape[0])
            values = (
                self.value_projection[relation](source)
                + self.edge_value[relation](attributes)
            ).view(-1, self.num_heads, self.head_dim)
            messages = (alpha.unsqueeze(-1) * values).reshape(-1, self.hidden_dim)
            relation_sum = torch.zeros_like(nodes[target_type])
            relation_sum.index_add_(0, index[1], messages)
            aggregated[target_type] = aggregated[target_type] + relation_sum
            attentions[relation] = alpha
        output = {
            node_type: self.normalization[node_type](
                values + self.output_projection[node_type](aggregated[node_type])
            )
            for node_type, values in nodes.items()
        }
        return output, attentions


def _mean_pool(
    values: torch.Tensor,
    batch: torch.Tensor | None,
    num_graphs: int,
) -> torch.Tensor:
    if batch is None:
        return values.mean(dim=0) if values.shape[0] else values.new_zeros(values.shape[1])
    result = values.new_zeros((num_graphs, values.shape[1]))
    if values.shape[0]:
        result.index_add_(0, batch, values)
        counts = torch.bincount(batch, minlength=num_graphs).to(values.dtype).clamp_min(1.0)
        result = result / counts.unsqueeze(-1)
    return result


class CRGEncoder(nn.Module):
    """Two-layer public CRG encoder with job/checkpoint mean pooling."""

    def __init__(
        self,
        job_input_dim: int = 14,
        machine_input_dim: int = 2,
        checkpoint_input_dim: int = 12,
        global_input_dim: int = 7,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        output_dim: int | None = None,
    ) -> None:
        super().__init__()
        if (job_input_dim, machine_input_dim, checkpoint_input_dim, global_input_dim) != (14, 2, 12, 7):
            raise ValueError("CRG requires the frozen 14/2/12 node and 7 global dimensions")
        if num_layers != 2 or num_heads != 4 or hidden_dim != 64:
            raise ValueError("formal CRG requires 2 layers, 4 heads, and hidden_dim=64")
        self.hidden_dim = hidden_dim
        self.input_projection = nn.ModuleDict(
            {
                JOB: nn.Linear(job_input_dim, hidden_dim),
                MACHINE: nn.Linear(machine_input_dim, hidden_dim),
                CHECKPOINT: nn.Linear(checkpoint_input_dim, hidden_dim),
            }
        )
        self.layers = nn.ModuleList(
            [RelationGATv2Layer(hidden_dim, num_heads) for _ in range(num_layers)]
        )
        resolved_output = hidden_dim if output_dim is None else output_dim
        self.state_projection = nn.Sequential(
            nn.Linear(2 * hidden_dim + global_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, resolved_output),
        )
        self.output_dim = resolved_output

    @staticmethod
    def _validate(data: HeteroGraphData) -> None:
        if data.graph_config != PAPER_GRAPH:
            raise ValueError("CRGEncoder accepts paper_graph only; no legacy fallback")
        if data.job_x.shape[1:] != (14,) or data.machine_x.shape[1:] != (2,):
            raise ValueError("paper graph job/machine dimensions must be 14/2")
        if data.checkpoint_x.shape[1:] != (12,):
            raise ValueError("paper graph checkpoint dimension must be 12")
        if data.global_x.shape[-1:] != (7,):
            raise ValueError("paper graph global dimension must be 7")
        expected = set(PAPER_RELATION_SPECS)
        if set(data.relation_edge_index) != expected or set(data.relation_edge_attr) != expected:
            raise ValueError("paper graph must contain exactly the six frozen relations")

    def encode(self, data: HeteroGraphData) -> CRGEncoding:
        self._validate(data)
        nodes = {
            JOB: self.input_projection[JOB](data.job_x),
            MACHINE: self.input_projection[MACHINE](data.machine_x),
            CHECKPOINT: self.input_projection[CHECKPOINT](data.checkpoint_x),
        }
        layer_attentions: list[dict[str, torch.Tensor]] = []
        for layer in self.layers:
            nodes, attention = layer(
                nodes, data.relation_edge_index, data.relation_edge_attr
            )
            layer_attentions.append(attention)
        job_pool = _mean_pool(nodes[JOB], data.job_batch, data.num_graphs)
        checkpoint_pool = _mean_pool(
            nodes[CHECKPOINT], data.checkpoint_batch, data.num_graphs
        )
        global_x = data.global_x
        batched = data.job_batch is not None or global_x.ndim == 2
        if batched:
            if job_pool.ndim == 1:
                job_pool = job_pool.unsqueeze(0)
            if checkpoint_pool.ndim == 1:
                checkpoint_pool = checkpoint_pool.unsqueeze(0)
            if global_x.ndim == 1:
                global_x = global_x.unsqueeze(0)
        state = self.state_projection(torch.cat((job_pool, checkpoint_pool, global_x), dim=-1))
        return CRGEncoding(state, nodes, tuple(layer_attentions))

    def forward(self, data: HeteroGraphData) -> torch.Tensor:
        return self.encode(data).state


__all__ = ["CRGEncoder", "CRGEncoding", "RelationGATv2Layer"]
