"""Scalar C1--C4 critics frozen by Paper A Sec. 5.4 and 6.4."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


CRITIC_VARIANTS = ("c1", "c2", "c3", "c4")
PRIVILEGED_FEATURE_NAMES = (
    "privileged_lower_over_q",
    "privileged_upper_over_q",
    "timely_future_supply_over_q",
    "timely_future_job_count_compressed",
)


@dataclass(frozen=True)
class CriticStepContext:
    """Per-decision checkpoint data, aligned with graph checkpoint node order."""

    checkpoint_keys: tuple[tuple[int, int], ...]
    weights: torch.Tensor
    required_quantities: torch.Tensor
    online_endpoints: torch.Tensor
    privileged_features: torch.Tensor | None = None
    privileged_lower: torch.Tensor | None = None
    privileged_upper: torch.Tensor | None = None

    def validate(self, *, privileged: bool) -> None:
        count = len(self.checkpoint_keys)
        if self.weights.shape != (count,) or self.required_quantities.shape != (count,):
            raise ValueError("checkpoint weights and quantities must align with keys")
        if self.online_endpoints.shape != (count, 2):
            raise ValueError("online endpoints must have shape [checkpoints, 2]")
        private = (
            self.privileged_features,
            self.privileged_lower,
            self.privileged_upper,
        )
        if privileged:
            if any(item is None for item in private):
                raise ValueError("privileged critic requires all private fields")
            assert self.privileged_features is not None
            assert self.privileged_lower is not None
            assert self.privileged_upper is not None
            if self.privileged_features.shape != (
                count,
                len(PRIVILEGED_FEATURE_NAMES),
            ):
                raise ValueError("privileged feature layout does not match its contract")
            if self.privileged_lower.shape != (count,) or self.privileged_upper.shape != (count,):
                raise ValueError("privileged bounds must align with checkpoint keys")
        elif any(item is not None for item in private):
            raise ValueError("public critic must not receive privileged fields")


@dataclass(frozen=True)
class CriticTargets:
    """Undiscounted Monte-Carlo labels from one completed episode."""

    remaining_total_cost: float
    remaining_tardiness: float
    checkpoint_shortfalls: torch.Tensor


@dataclass(frozen=True)
class CriticOutput:
    """Raw-cost predictions plus the reward-scaled scalar used by GAE."""

    total_value: torch.Tensor
    total_cost: torch.Tensor
    remaining_tardiness: torch.Tensor | None
    checkpoint_shortfalls: torch.Tensor


class CertificateCritic(nn.Module):
    """C1 scalar or C2--C4 checkpoint-decomposed value estimator."""

    def __init__(self, variant: str, hidden_dim: int = 64) -> None:
        super().__init__()
        if variant not in CRITIC_VARIANTS:
            raise ValueError(f"unknown critic variant: {variant}")
        self.variant = variant
        self.hidden_dim = hidden_dim
        if variant == "c1":
            self.total_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
            self.tt_head = None
            self.shortfall_head = None
            return
        self.total_head = None
        self.tt_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        private_width = len(PRIVILEGED_FEATURE_NAMES) if variant in {"c3", "c4"} else 0
        checkpoint_input = 2 * hidden_dim + 2 + private_width
        self.shortfall_head = nn.Sequential(
            nn.Linear(checkpoint_input, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    @property
    def uses_privileged_inputs(self) -> bool:
        return self.variant in {"c3", "c4"}

    def forward(
        self,
        state: torch.Tensor,
        checkpoint_embeddings: torch.Tensor,
        context: CriticStepContext | None,
        *,
        f_ref: float,
        reward_scale: float,
    ) -> CriticOutput:
        if state.ndim != 1 or state.shape[0] != self.hidden_dim:
            raise ValueError("critic expects one unbatched public state vector")
        if not torch.isfinite(torch.tensor(float(f_ref))) or f_ref <= 0.0:
            raise ValueError("f_ref must be finite and positive")
        if self.variant == "c1":
            assert self.total_head is not None
            total_cost = F.softplus(self.total_head(state).squeeze(-1))
            total_value = -float(reward_scale) / float(f_ref) * total_cost
            return CriticOutput(
                total_value=total_value,
                total_cost=total_cost,
                remaining_tardiness=None,
                checkpoint_shortfalls=state.new_empty((0,)),
            )

        if context is None:
            raise ValueError(f"{self.variant} requires checkpoint context")
        context.validate(privileged=self.uses_privileged_inputs)
        count = len(context.checkpoint_keys)
        if checkpoint_embeddings.shape != (count, self.hidden_dim):
            raise ValueError("checkpoint embeddings must align with critic context")
        assert self.tt_head is not None and self.shortfall_head is not None
        tardiness = F.softplus(self.tt_head(state).squeeze(-1))
        if count:
            rows = [
                state.expand(count, -1),
                checkpoint_embeddings,
                context.online_endpoints,
            ]
            if self.uses_privileged_inputs:
                assert context.privileged_features is not None
                rows.append(context.privileged_features)
            raw = self.shortfall_head(torch.cat(rows, dim=-1)).squeeze(-1)
            if self.variant == "c4":
                assert context.privileged_lower is not None
                assert context.privileged_upper is not None
                width = context.privileged_upper - context.privileged_lower
                if bool((width < -1e-9).any()):
                    raise ValueError("privileged certificate interval is empty")
                predictions = torch.where(
                    width <= 1e-12,
                    context.privileged_lower,
                    context.privileged_lower + width * torch.sigmoid(raw),
                )
            else:
                predictions = F.softplus(raw)
        else:
            predictions = state.new_empty((0,))
        total_cost = tardiness + (context.weights * predictions).sum()
        total_value = -float(reward_scale) / float(f_ref) * total_cost
        return CriticOutput(total_value, total_cost, tardiness, predictions)

    def loss(
        self,
        output: CriticOutput,
        targets: CriticTargets,
        context: CriticStepContext | None,
        *,
        f_ref: float,
    ) -> torch.Tensor:
        """C1 MC scalar MSE or C2--C4 direct-sum component MSE."""

        if self.variant == "c1":
            target = output.total_cost.new_tensor(targets.remaining_total_cost)
            return ((output.total_cost - target) / float(f_ref)).square()
        if context is None:
            raise ValueError("decomposed critic loss requires checkpoint context")
        context.validate(privileged=self.uses_privileged_inputs)
        expected = targets.checkpoint_shortfalls.to(output.total_cost.device)
        if expected.shape != output.checkpoint_shortfalls.shape:
            raise ValueError("checkpoint labels must align with predictions")
        if self.variant == "c4":
            assert context.privileged_lower is not None
            assert context.privileged_upper is not None
            if bool(
                (
                    (expected < context.privileged_lower - 1e-8)
                    | (expected > context.privileged_upper + 1e-8)
                ).any()
            ):
                raise ValueError("true checkpoint shortfall violates privileged certificate")
        assert output.remaining_tardiness is not None
        tt_target = output.total_cost.new_tensor(targets.remaining_tardiness)
        tt_loss = ((output.remaining_tardiness - tt_target) / float(f_ref)).square()
        checkpoint_loss = (
            (
                context.weights
                * (output.checkpoint_shortfalls - expected)
                / float(f_ref)
            ).square().sum()
        )
        return tt_loss + checkpoint_loss


__all__ = [
    "CRITIC_VARIANTS",
    "PRIVILEGED_FEATURE_NAMES",
    "CertificateCritic",
    "CriticOutput",
    "CriticStepContext",
    "CriticTargets",
]
