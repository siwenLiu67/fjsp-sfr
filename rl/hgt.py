"""修改版 Heterogeneous Graph Transformer。

与 Hu 2020 的区别：
1. message 计算拼接数值边属性。
2. attention 使用 ReLU 而不是 softmax。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from fjsp_sfr.rl.graph import HeteroGraphData


_BASE_EDGE_TYPES = {
    "job_machine": ("job", "machine"),
    "job_checkpoint": ("job", "checkpoint"),
}
_NODE_TYPES = ("job", "machine", "checkpoint")


class HGTLayer(nn.Module):
    """支持数值边属性和双向消息的单层 HGT。"""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        edge_dims: dict[str, int],
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if num_heads <= 0 or hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        unknown = set(edge_dims) - set(_BASE_EDGE_TYPES)
        if unknown:
            raise ValueError(f"unknown edge types: {sorted(unknown)}")

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.edge_dims = dict(edge_dims)

        message_layers: dict[str, nn.Module] = {}
        attention_layers: dict[str, nn.Module] = {}
        for edge_type, edge_dim in edge_dims.items():
            if edge_dim < 0:
                raise ValueError("edge dimensions must be non-negative")
            source_type, target_type = _BASE_EDGE_TYPES[edge_type]
            for direction in (
                f"{source_type}_to_{target_type}",
                f"{target_type}_to_{source_type}",
            ):
                message_layers[direction] = nn.Linear(
                    hidden_dim + edge_dim, hidden_dim
                )
                attention_layers[direction] = nn.Linear(
                    hidden_dim + edge_dim, num_heads
                )
        self.message_layers = nn.ModuleDict(message_layers)
        self.attention_layers = nn.ModuleDict(attention_layers)
        self.aggregate_layers = nn.ModuleDict(
            {
                node_type: nn.Linear(hidden_dim, hidden_dim)
                for node_type in _NODE_TYPES
            }
        )
        self.norms = nn.ModuleDict(
            {node_type: nn.LayerNorm(hidden_dim) for node_type in _NODE_TYPES}
        )

    def _accumulate_direction(
        self,
        *,
        direction: str,
        source_x: torch.Tensor,
        target_indices: torch.Tensor,
        source_indices: torch.Tensor,
        edge_attr: torch.Tensor,
        aggregated: torch.Tensor,
        degrees: torch.Tensor,
    ) -> None:
        if source_indices.numel() == 0:
            return
        source_hidden = source_x[source_indices]
        message_input = torch.cat((source_hidden, edge_attr), dim=-1)
        messages = self.message_layers[direction](message_input)
        attention = torch.relu(self.attention_layers[direction](message_input))
        messages = messages.reshape(-1, self.num_heads, self.head_dim)
        messages = (messages * attention.unsqueeze(-1)).reshape(
            -1, self.hidden_dim
        )
        aggregated.index_add_(0, target_indices, messages)
        degrees.index_add_(
            0,
            target_indices,
            torch.ones(
                target_indices.shape[0],
                device=degrees.device,
                dtype=degrees.dtype,
            ),
        )

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[str, torch.Tensor],
        edge_attr_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """执行带 ReLU attention 的双向异构消息传递。"""

        missing_nodes = set(_NODE_TYPES) - set(x_dict)
        if missing_nodes:
            raise ValueError(f"missing node types: {sorted(missing_nodes)}")

        aggregated = {
            node_type: x_dict[node_type].new_zeros(
                (x_dict[node_type].shape[0], self.hidden_dim)
            )
            for node_type in _NODE_TYPES
        }
        degrees = {
            node_type: x_dict[node_type].new_zeros(
                (x_dict[node_type].shape[0],)
            )
            for node_type in _NODE_TYPES
        }

        for edge_type, edge_dim in self.edge_dims.items():
            if edge_type not in edge_index_dict or edge_type not in edge_attr_dict:
                raise ValueError(f"missing edge data for {edge_type}")
            edge_index = edge_index_dict[edge_type]
            edge_attr = edge_attr_dict[edge_type]
            if edge_index.ndim != 2 or edge_index.shape[0] != 2:
                raise ValueError(f"{edge_type} edge_index must have shape (2, E)")
            if edge_attr.ndim != 2 or edge_attr.shape != (
                edge_index.shape[1],
                edge_dim,
            ):
                raise ValueError(
                    f"{edge_type} edge_attr must have shape (E, {edge_dim})"
                )

            source_type, target_type = _BASE_EDGE_TYPES[edge_type]
            source_indices = edge_index[0]
            target_indices = edge_index[1]
            self._accumulate_direction(
                direction=f"{source_type}_to_{target_type}",
                source_x=x_dict[source_type],
                source_indices=source_indices,
                target_indices=target_indices,
                edge_attr=edge_attr,
                aggregated=aggregated[target_type],
                degrees=degrees[target_type],
            )
            self._accumulate_direction(
                direction=f"{target_type}_to_{source_type}",
                source_x=x_dict[target_type],
                source_indices=target_indices,
                target_indices=source_indices,
                edge_attr=edge_attr,
                aggregated=aggregated[source_type],
                degrees=degrees[source_type],
            )

        output: dict[str, torch.Tensor] = {}
        for node_type in _NODE_TYPES:
            normalized = aggregated[node_type] / degrees[node_type].clamp_min(
                1.0
            ).unsqueeze(-1)
            output[node_type] = self.norms[node_type](
                x_dict[node_type] + self.aggregate_layers[node_type](normalized)
            )
        return output


class HGTEncoder(nn.Module):
    """多层 HGT 与节点池化组成的全局状态编码器。"""

    def __init__(
        self,
        job_input_dim: int = 11,
        machine_input_dim: int = 2,
        checkpoint_input_dim: int = 8,
        global_input_dim: int = 7,
        hidden_dim: int = 64,
        num_layers: int = 3,
        num_heads: int = 4,
        output_dim: int | None = None,
    ) -> None:
        super().__init__()
        if num_layers < 0:
            raise ValueError("num_layers must be non-negative")
        if hidden_dim <= 0 or hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be positive and divisible by num_heads")
        resolved_output_dim = hidden_dim if output_dim is None else output_dim
        if resolved_output_dim <= 0:
            raise ValueError("output_dim must be positive")

        self.hidden_dim = hidden_dim
        self.output_dim = resolved_output_dim
        self.proj_job = nn.Linear(job_input_dim, hidden_dim)
        self.proj_machine = nn.Linear(machine_input_dim, hidden_dim)
        self.proj_checkpoint = nn.Linear(checkpoint_input_dim, hidden_dim)
        edge_dims = {"job_machine": 1, "job_checkpoint": 2}
        self.layers = nn.ModuleList(
            HGTLayer(hidden_dim, num_heads, edge_dims)
            for _ in range(num_layers)
        )
        self.output_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + global_input_dim, resolved_output_dim),
            nn.ReLU(),
            nn.Linear(resolved_output_dim, resolved_output_dim),
        )

    def forward(self, data: HeteroGraphData) -> torch.Tensor:
        x_dict = {
            "job": self.proj_job(data.job_x),
            "machine": self.proj_machine(data.machine_x),
            "checkpoint": self.proj_checkpoint(data.checkpoint_x),
        }
        edge_index_dict = {
            "job_machine": data.job_machine_edge_index,
            "job_checkpoint": data.job_checkpoint_edge_index,
        }
        edge_attr_dict = {
            "job_machine": data.job_machine_edge_attr,
            "job_checkpoint": data.job_checkpoint_edge_attr,
        }
        for layer in self.layers:
            x_dict = layer(x_dict, edge_index_dict, edge_attr_dict)

        if x_dict["job"].shape[0] == 0:
            job_pool = data.global_x.new_zeros((self.hidden_dim,))
        else:
            job_pool = x_dict["job"].mean(dim=0)
        if x_dict["checkpoint"].shape[0] == 0:
            checkpoint_pool = data.global_x.new_zeros((self.hidden_dim,))
        else:
            checkpoint_pool = x_dict["checkpoint"].mean(dim=0)
        return self.output_mlp(
            torch.cat((job_pool, checkpoint_pool, data.global_x), dim=-1)
        )
