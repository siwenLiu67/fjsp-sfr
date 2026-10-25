"""PPO actor-critic using pooled node features instead of graph message passing."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical

from fjsp_sfr.rl.graph import HeteroGraphData


def _mean_or_zeros(x: torch.Tensor, width: int) -> torch.Tensor:
    return x.mean(dim=0) if x.shape[0] else x.new_zeros((width,))


def _max_or_zeros(x: torch.Tensor, width: int) -> torch.Tensor:
    return x.max(dim=0).values if x.shape[0] else x.new_zeros((width,))


class MLPPPOAgent(nn.Module):
    """Mean/max pooled job features + mean checkpoint features + globals."""

    def __init__(
        self,
        action_size: int,
        *,
        job_input_dim: int = 14,
        checkpoint_input_dim: int = 8,
        global_input_dim: int = 7,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        if action_size <= 0:
            raise ValueError("action_size must be positive")
        self.action_size = action_size
        self.job_input_dim = job_input_dim
        self.checkpoint_input_dim = checkpoint_input_dim
        input_dim = 2 * job_input_dim + checkpoint_input_dim + global_input_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_dim, action_size)
        self.value_head = nn.Linear(hidden_dim, 1)

    def encode(self, data: HeteroGraphData) -> torch.Tensor:
        pooled = torch.cat(
            (
                _mean_or_zeros(data.job_x, self.job_input_dim),
                _max_or_zeros(data.job_x, self.job_input_dim),
                _mean_or_zeros(data.checkpoint_x, self.checkpoint_input_dim),
                data.global_x,
            )
        )
        return self.encoder(pooled)

    def forward(self, data, action_mask=None):
        hidden = self.encode(data)
        logits = self.policy_head(hidden)
        if action_mask is not None:
            mask = action_mask.to(device=logits.device, dtype=torch.bool)
            if mask.shape != (self.action_size,) or not bool(mask.any()):
                raise ValueError("invalid action_mask")
            logits = logits.masked_fill(~mask, float("-inf"))
        return torch.log_softmax(logits, dim=-1), self.value_head(hidden).squeeze(-1)

    @torch.no_grad()
    def select_action(self, data, action_mask=None):
        log_probs, value = self(data, action_mask)
        action = Categorical(logits=log_probs).sample()
        return int(action.item()), float(log_probs[action].item()), float(value.item())

    @torch.no_grad()
    def greedy_action(self, data, action_mask=None) -> int:
        log_probs, _value = self(data, action_mask)
        return int(log_probs.argmax().item())


__all__ = ["MLPPPOAgent"]
