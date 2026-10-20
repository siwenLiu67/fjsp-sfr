"""HGT 编码器上的 PPO 策略与价值网络。"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical

from fjsp_sfr.rl.graph import HeteroGraphData
from fjsp_sfr.rl.hgt import HGTEncoder


class HGTPPOAgent(nn.Module):
    """HGT 编码器 + 策略头 + 价值头。"""

    def __init__(
        self,
        action_size: int,
        hidden_dim: int = 64,
        **encoder_kwargs,
    ) -> None:
        super().__init__()
        if action_size <= 0:
            raise ValueError("action_size must be positive")
        self.encoder = HGTEncoder(hidden_dim=hidden_dim, **encoder_kwargs)
        output_dim = encoder_kwargs.get("output_dim") or hidden_dim
        self.policy_head = nn.Linear(output_dim, action_size)
        self.value_head = nn.Linear(output_dim, 1)
        self.action_size = action_size

    def forward(
        self,
        data: HeteroGraphData,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(data)
        logits = self.policy_head(z)
        if action_mask is not None:
            if action_mask.shape != (self.action_size,):
                raise ValueError("action_mask must have shape (action_size,)")
            mask = action_mask.to(device=logits.device, dtype=torch.bool)
            if not bool(mask.any()):
                raise ValueError("action_mask must allow at least one action")
            logits = logits.masked_fill(~mask, float("-inf"))
        log_probs = torch.log_softmax(logits, dim=-1)
        value = self.value_head(z).squeeze(-1)
        return log_probs, value

    @torch.no_grad()
    def select_action(
        self,
        data: HeteroGraphData,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[int, float, float]:
        log_probs, value = self.forward(data, action_mask)
        distribution = Categorical(logits=log_probs)
        action = distribution.sample()
        return int(action.item()), float(log_probs[action].item()), float(value.item())

    @torch.no_grad()
    def greedy_action(
        self,
        data: HeteroGraphData,
        action_mask: torch.Tensor | None = None,
    ) -> int:
        log_probs, _value = self.forward(data, action_mask)
        return int(log_probs.argmax().item())
