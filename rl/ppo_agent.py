"""Explicit legacy-HGT and formal-CRG PPO policy/value networks."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical

from fjsp_sfr.rl.crg import CRGEncoder, CRGEncoding
from fjsp_sfr.rl.certificate_critic import (
    CRITIC_VARIANTS,
    CertificateCritic,
    CriticOutput,
    CriticStepContext,
)
from fjsp_sfr.rl.graph import PAPER_GRAPH, HeteroGraphData
from fjsp_sfr.rl.hgt import HGTEncoder


class HGTPPOAgent(nn.Module):
    """HGT 编码器 + 策略头 + 价值头。"""

    def __init__(
        self,
        action_size: int,
        hidden_dim: int = 64,
        encoder_type: str = "legacy_hgt",
        critic_variant: str = "legacy",
        **encoder_kwargs,
    ) -> None:
        super().__init__()
        if action_size <= 0:
            raise ValueError("action_size must be positive")
        if encoder_type == "legacy_hgt":
            self.encoder = HGTEncoder(hidden_dim=hidden_dim, **encoder_kwargs)
        elif encoder_type == "crg_gatv2":
            self.encoder = CRGEncoder(hidden_dim=hidden_dim, **encoder_kwargs)
        else:
            raise ValueError(f"unknown encoder_type: {encoder_type}")
        self.encoder_type = encoder_type
        output_dim = encoder_kwargs.get("output_dim") or hidden_dim
        self.policy_head = nn.Linear(output_dim, action_size)
        if critic_variant == "legacy":
            self.value_head: nn.Linear | None = nn.Linear(output_dim, 1)
            self.certificate_critic: CertificateCritic | None = None
        elif critic_variant in CRITIC_VARIANTS:
            if encoder_type != "crg_gatv2" or output_dim != hidden_dim:
                raise ValueError("C1-C4 require the public CRG state representation")
            self.value_head = None
            self.certificate_critic = CertificateCritic(
                critic_variant, hidden_dim=hidden_dim
            )
        else:
            raise ValueError(f"unknown critic_variant: {critic_variant}")
        self.critic_variant = critic_variant
        self.action_size = action_size

    def _masked_actor_log_probs(
        self,
        z: torch.Tensor,
        action_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        logits = self.policy_head(z)
        if action_mask is not None:
            if action_mask.shape != (self.action_size,):
                raise ValueError("action_mask must have shape (action_size,)")
            mask = action_mask.to(device=logits.device, dtype=torch.bool)
            if not bool(mask.any()):
                raise ValueError("action_mask must allow at least one action")
            logits = logits.masked_fill(~mask, float("-inf"))
        return torch.log_softmax(logits, dim=-1)

    def actor_log_probs(
        self,
        data: HeteroGraphData,
        action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Public Actor path; it has no argument for privileged information."""

        if self.encoder_type == "crg_gatv2":
            encoding = self.encoder.encode(data)
            z = encoding.state
        else:
            z = self.encoder(data)
        return self._masked_actor_log_probs(z, action_mask)

    def forward(
        self,
        data: HeteroGraphData,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.encoder_type == "crg_gatv2" and data.graph_config != PAPER_GRAPH:
            raise ValueError("crg_gatv2 requires paper_graph; no legacy fallback")
        if self.encoder_type == "legacy_hgt" and data.graph_config == PAPER_GRAPH:
            raise ValueError("paper_graph requires crg_gatv2; HGT fallback is forbidden")
        if self.critic_variant != "legacy":
            raise RuntimeError(
                "C1-C4 forward requires explicit critic context and reward scale"
            )
        z = self.encoder(data)
        log_probs = self._masked_actor_log_probs(z, action_mask)
        assert self.value_head is not None
        value = self.value_head(z).squeeze(-1)
        return log_probs, value

    def forward_with_critic(
        self,
        data: HeteroGraphData,
        action_mask: torch.Tensor | None,
        critic_context: CriticStepContext | None,
        *,
        f_ref: float,
        reward_scale: float,
    ) -> tuple[torch.Tensor, CriticOutput, CRGEncoding]:
        """Share only the public encoder; private fields enter the Critic after it."""

        if self.critic_variant not in CRITIC_VARIANTS:
            raise RuntimeError("forward_with_critic requires C1-C4")
        encoding = self.encoder.encode(data)
        log_probs = self._masked_actor_log_probs(encoding.state, action_mask)
        assert self.certificate_critic is not None
        output = self.certificate_critic(
            encoding.state,
            encoding.node_embeddings["checkpoint"],
            critic_context,
            f_ref=f_ref,
            reward_scale=reward_scale,
        )
        return log_probs, output, encoding

    def encode_public(self, data: HeteroGraphData) -> CRGEncoding | torch.Tensor:
        """Expose the public state/node interface for later actor/critic heads."""

        if self.encoder_type == "crg_gatv2":
            return self.encoder.encode(data)
        return self.encoder(data)

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
