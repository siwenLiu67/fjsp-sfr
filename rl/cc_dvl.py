"""Certificate-constrained distributional value learning (Paper A Sec. 5.4)."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from fjsp_sfr.rl.certificate_critic import (
    PRIVILEGED_FEATURE_NAMES,
    CriticStepContext,
    CriticTargets,
)


DISTRIBUTIONAL_CRITIC_VARIANTS = ("d1", "d2")


@dataclass(frozen=True)
class QuantileCriticOutput:
    """Raw component predictions and the reward-scaled risk-neutral value."""

    total_value: torch.Tensor
    total_cost: torch.Tensor
    remaining_tardiness: torch.Tensor
    checkpoint_quantiles: torch.Tensor

    @property
    def mean_checkpoint_shortfalls(self) -> torch.Tensor:
        if not self.checkpoint_quantiles.shape[0]:
            return self.checkpoint_quantiles.new_empty((0,))
        return self.checkpoint_quantiles.mean(dim=-1)


def quantile_levels(
    count: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return the frozen midpoint grid tau_i=(i-1/2)/K."""

    if count <= 0:
        raise ValueError("num_quantiles must be positive")
    return (torch.arange(count, device=device, dtype=dtype) + 0.5) / count


def pinball_loss(residual: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
    """Elementwise rho_tau(residual) on the original shortfall scale."""

    return residual * (levels - (residual < 0.0).to(residual.dtype))


def ordered_interval_positions(logits: torch.Tensor) -> torch.Tensor:
    """K ordered positions from K+1 stable positive softmax increments.

    Exact strict inequalities hold mathematically.  Floating-point underflow may
    make adjacent positions equal for extreme logits, which does not violate
    monotonicity or the closed certificate bounds.
    """

    if logits.ndim != 2 or logits.shape[1] < 2:
        raise ValueError("ordered positions require shape [heads, K+1]")
    increments = torch.softmax(logits, dim=-1)
    return increments.cumsum(dim=-1)[:, :-1]


def ordered_unbounded_quantiles(raw: torch.Tensor) -> torch.Tensor:
    """Nonnegative ordered D1 quantiles without a certificate upper bound."""

    if raw.ndim != 2 or not raw.shape[1]:
        raise ValueError("unbounded quantiles require shape [heads, K]")
    return F.softplus(raw).cumsum(dim=-1)


def width_dispersion_rows(
    quantiles: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> tuple[tuple[float, float, float], ...]:
    """Return (certificate width, quantile range, normalized range) rows."""

    if quantiles.ndim != 2 or lower.ndim != 1 or upper.ndim != 1:
        raise ValueError("quantiles must be 2D and endpoints must be 1D")
    if lower.shape != upper.shape or quantiles.shape[0] != lower.shape[0]:
        raise ValueError("quantiles and certificate endpoints must align")
    rows = []
    for index in range(quantiles.shape[0]):
        width = max(0.0, float((upper[index] - lower[index]).detach().item()))
        spread = float(
            (quantiles[index].max() - quantiles[index].min()).detach().item()
        )
        rows.append((width, spread, spread / width if width > 0.0 else 0.0))
    return tuple(rows)


def summarize_width_dispersion(
    rows: list[tuple[float, float, float]] | tuple[tuple[float, float, float], ...],
) -> dict[str, dict[str, float | int]]:
    """Stratify D2 dispersion by certificate width for diagnostics only."""

    bins = (
        ("degenerate", lambda width: width <= 1e-12),
        ("narrow_(0,1]", lambda width: 1e-12 < width <= 1.0),
        ("medium_(1,5]", lambda width: 1.0 < width <= 5.0),
        ("wide_(5,inf)", lambda width: width > 5.0),
    )
    output: dict[str, dict[str, float | int]] = {}
    for label, predicate in bins:
        selected = [row for row in rows if predicate(row[0])]
        output[label] = {
            "count": len(selected),
            "mean_width": (
                sum(row[0] for row in selected) / len(selected) if selected else 0.0
            ),
            "mean_quantile_range": (
                sum(row[1] for row in selected) / len(selected) if selected else 0.0
            ),
            "mean_normalized_range": (
                sum(row[2] for row in selected) / len(selected) if selected else 0.0
            ),
        }
    return output


class CCDVLCritic(nn.Module):
    """D1 unbounded ordered or D2 certificate-bounded ordered quantile Critic."""

    def __init__(
        self,
        variant: str,
        *,
        hidden_dim: int = 64,
        num_quantiles: int,
        quantile_loss_coef: float,
    ) -> None:
        super().__init__()
        if variant not in DISTRIBUTIONAL_CRITIC_VARIANTS:
            raise ValueError(f"unknown distributional critic variant: {variant}")
        if num_quantiles <= 0:
            raise ValueError("num_quantiles must be positive")
        if not math.isfinite(quantile_loss_coef) or quantile_loss_coef < 0.0:
            raise ValueError("quantile_loss_coef must be finite and non-negative")
        self.variant = variant
        self.hidden_dim = hidden_dim
        self.num_quantiles = int(num_quantiles)
        self.quantile_loss_coef = float(quantile_loss_coef)
        self.tt_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        checkpoint_input = 2 * hidden_dim + 2 + len(PRIVILEGED_FEATURE_NAMES)
        output_width = self.num_quantiles + int(variant == "d2")
        self.quantile_head = nn.Sequential(
            nn.Linear(checkpoint_input, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_width),
        )

    @property
    def uses_privileged_inputs(self) -> bool:
        return True

    def forward(
        self,
        state: torch.Tensor,
        checkpoint_embeddings: torch.Tensor,
        context: CriticStepContext | None,
        *,
        f_ref: float,
        reward_scale: float,
    ) -> QuantileCriticOutput:
        if state.ndim != 1 or state.shape[0] != self.hidden_dim:
            raise ValueError("critic expects one unbatched public state vector")
        if not math.isfinite(float(f_ref)) or f_ref <= 0.0:
            raise ValueError("f_ref must be finite and positive")
        if context is None:
            raise ValueError(f"{self.variant} requires privileged checkpoint context")
        context.validate(privileged=True)
        count = len(context.checkpoint_keys)
        if checkpoint_embeddings.shape != (count, self.hidden_dim):
            raise ValueError("checkpoint embeddings must align with critic context")
        tardiness = F.softplus(self.tt_head(state).squeeze(-1))
        if count:
            assert context.privileged_features is not None
            rows = torch.cat(
                (
                    state.expand(count, -1),
                    checkpoint_embeddings,
                    context.online_endpoints,
                    context.privileged_features,
                ),
                dim=-1,
            )
            raw = self.quantile_head(rows)
            if self.variant == "d1":
                quantiles = ordered_unbounded_quantiles(raw)
            else:
                assert context.privileged_lower is not None
                assert context.privileged_upper is not None
                width = context.privileged_upper - context.privileged_lower
                if bool((width < -1e-9).any()):
                    raise ValueError("privileged certificate interval is empty")
                positions = ordered_interval_positions(raw)
                bounded = context.privileged_lower[:, None] + width[:, None] * positions
                quantiles = torch.where(
                    width[:, None] <= 1e-12,
                    context.privileged_lower[:, None].expand_as(bounded),
                    bounded,
                )
        else:
            quantiles = state.new_empty((0, self.num_quantiles))
        means = (
            quantiles.mean(dim=-1)
            if count
            else state.new_empty((0,))
        )
        total_cost = tardiness + (context.weights * means).sum()
        total_value = -float(reward_scale) / float(f_ref) * total_cost
        return QuantileCriticOutput(total_value, total_cost, tardiness, quantiles)

    def loss(
        self,
        output: QuantileCriticOutput,
        targets: CriticTargets,
        context: CriticStepContext | None,
        *,
        f_ref: float,
    ) -> torch.Tensor:
        if context is None:
            raise ValueError("distributional critic loss requires checkpoint context")
        context.validate(privileged=True)
        expected = targets.checkpoint_shortfalls.to(output.total_cost.device)
        if expected.shape != (len(context.checkpoint_keys),):
            raise ValueError("checkpoint labels must align with predictions")
        if output.checkpoint_quantiles.shape != (
            expected.shape[0],
            self.num_quantiles,
        ):
            raise ValueError("quantile predictions have the wrong shape")
        if self.variant == "d2":
            assert context.privileged_lower is not None
            assert context.privileged_upper is not None
            if bool(
                (
                    (expected < context.privileged_lower - 1e-8)
                    | (expected > context.privileged_upper + 1e-8)
                ).any()
            ):
                raise ValueError("true checkpoint shortfall violates privileged certificate")

        tt_target = output.total_cost.new_tensor(targets.remaining_tardiness)
        tt_loss = (
            (output.remaining_tardiness - tt_target) / float(f_ref)
        ).square()
        levels = quantile_levels(
            self.num_quantiles,
            device=output.total_cost.device,
            dtype=output.total_cost.dtype,
        )
        residual = expected[:, None] - output.checkpoint_quantiles
        per_head = pinball_loss(residual, levels[None, :])
        if self.variant == "d2":
            assert context.privileged_lower is not None
            assert context.privileged_upper is not None
            nondegenerate = (
                context.privileged_upper - context.privileged_lower
            ) > 1e-12
            per_head = torch.where(
                nondegenerate[:, None], per_head, torch.zeros_like(per_head)
            )
        quantile_loss = (
            context.weights / float(f_ref) * per_head.mean(dim=-1)
        ).sum()
        return tt_loss + self.quantile_loss_coef * quantile_loss


__all__ = [
    "CCDVLCritic",
    "DISTRIBUTIONAL_CRITIC_VARIANTS",
    "QuantileCriticOutput",
    "ordered_interval_positions",
    "ordered_unbounded_quantiles",
    "pinball_loss",
    "quantile_levels",
    "summarize_width_dispersion",
    "width_dispersion_rows",
]
