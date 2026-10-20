"""Lightweight theory-guided factorized candidate-priority actor."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class FactorizedPriorityOutput:
    priority_logits: torch.Tensor
    factor_values: torch.Tensor
    phi_raw_values: torch.Tensor
    phi_centered_values: torch.Tensor
    factor_contributions: torch.Tensor
    interaction_raw_values: torch.Tensor
    interaction_centered_values: torch.Tensor
    interaction_values: torch.Tensor
    context_gates: torch.Tensor


class LearnablePiecewiseLinear1D(nn.Module):
    """Eight-knot learnable scalar function on ``[0, 1]``."""

    def __init__(self, num_knots: int = 8, *, monotonic: bool = False) -> None:
        super().__init__()
        if num_knots < 2:
            raise ValueError("piecewise interpolation requires at least two knots")
        knots = torch.linspace(0.0, 1.0, num_knots)
        self.register_buffer("knot_positions", knots)
        centered_identity = knots - knots.mean()
        self.knot_values = nn.Parameter(centered_identity)
        self.monotonic = monotonic

    def raw_knot_values(self) -> torch.Tensor:
        values = self.knot_values
        if self.monotonic:
            values = torch.cummax(values, dim=0).values
        return values

    def centered_knot_values(self) -> torch.Tensor:
        values = self.raw_knot_values()
        return values - values.mean()

    def effective_knot_values(self) -> torch.Tensor:
        """Backward-compatible name for the centered curve used by priority."""

        return self.centered_knot_values()

    def _interpolate(
        self, values: torch.Tensor, knot_values: torch.Tensor
    ) -> torch.Tensor:
        scaled = values.clamp(0.0, 1.0) * (len(self.knot_positions) - 1)
        lower = scaled.floor().long().clamp(max=len(self.knot_positions) - 2)
        fraction = scaled - lower.to(scaled.dtype)
        lower_values = knot_values[lower]
        upper_values = knot_values[lower + 1]
        return lower_values + fraction * (upper_values - lower_values)

    def raw_values(self, values: torch.Tensor) -> torch.Tensor:
        return self._interpolate(values, self.raw_knot_values())

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self._interpolate(values, self.centered_knot_values())


class LearnableBilinearSurface2D(nn.Module):
    """Learnable scalar surface on a fixed 4-by-4 unit-square grid."""

    def __init__(self, grid_size: int = 4) -> None:
        super().__init__()
        if grid_size < 2:
            raise ValueError("bilinear interpolation requires at least a 2x2 grid")
        self.grid_values = nn.Parameter(torch.zeros((grid_size, grid_size)))

    def centered_grid_values(self) -> torch.Tensor:
        row_mean = self.grid_values.mean(dim=1, keepdim=True)
        column_mean = self.grid_values.mean(dim=0, keepdim=True)
        global_mean = self.grid_values.mean()
        return self.grid_values - row_mean - column_mean + global_mean

    def _interpolate(
        self,
        first: torch.Tensor,
        second: torch.Tensor,
        grid_values: torch.Tensor,
    ) -> torch.Tensor:
        if first.shape != second.shape:
            raise ValueError("bilinear surface inputs must have the same shape")
        grid_size = grid_values.shape[0]
        x = first.clamp(0.0, 1.0) * (grid_size - 1)
        y = second.clamp(0.0, 1.0) * (grid_size - 1)
        x0 = x.floor().long().clamp(max=grid_size - 2)
        y0 = y.floor().long().clamp(max=grid_size - 2)
        dx = x - x0.to(x.dtype)
        dy = y - y0.to(y.dtype)
        lower = grid_values[y0, x0] * (1.0 - dx) + grid_values[
            y0, x0 + 1
        ] * dx
        upper = grid_values[y0 + 1, x0] * (1.0 - dx) + grid_values[
            y0 + 1, x0 + 1
        ] * dx
        return torch.tanh(lower * (1.0 - dy) + upper * dy)

    def raw_values(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        return self._interpolate(first, second, self.grid_values)

    def forward(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        return self._interpolate(
            first, second, self.centered_grid_values()
        )


class TheoryGuidedFactorizedPriorityActor(nn.Module):
    """Combine five theory factors using shared state gates and two interactions."""

    def __init__(
        self,
        *,
        use_context_gate: bool = True,
        use_interactions: bool = True,
        monotonic: bool = False,
    ) -> None:
        super().__init__()
        self.use_context_gate = use_context_gate
        self.use_interactions = use_interactions
        self.factor_functions = nn.ModuleList(
            LearnablePiecewiseLinear1D(monotonic=monotonic) for _ in range(5)
        )
        self.interaction_gu = LearnableBilinearSurface2D()
        self.interaction_fu = LearnableBilinearSurface2D()
        self.gate = nn.Linear(128, 7)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    @property
    def phi_g(self) -> LearnablePiecewiseLinear1D:
        return self.factor_functions[0]

    @property
    def phi_f(self) -> LearnablePiecewiseLinear1D:
        return self.factor_functions[1]

    @property
    def phi_u(self) -> LearnablePiecewiseLinear1D:
        return self.factor_functions[2]

    @property
    def phi_m(self) -> LearnablePiecewiseLinear1D:
        return self.factor_functions[3]

    @property
    def phi_t(self) -> LearnablePiecewiseLinear1D:
        return self.factor_functions[4]

    def forward(
        self,
        factor_values: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        global_context: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> FactorizedPriorityOutput:
        if factor_values.ndim != 3 or factor_values.shape[-1] != 5:
            raise ValueError("factor_values must have shape [batch, candidates, 5]")
        if candidate_embeddings.shape[:2] != factor_values.shape[:2]:
            raise ValueError("candidate embeddings and factors must align")
        if candidate_embeddings.shape[-1] != 64 or global_context.shape[-1] != 64:
            raise ValueError("factorized actor requires 64-dimensional encoder outputs")
        if candidate_mask.shape != factor_values.shape[:2]:
            raise ValueError("candidate mask must have shape [batch, candidates]")

        valid = candidate_mask.unsqueeze(-1).to(candidate_embeddings.dtype)
        pooled = (candidate_embeddings * valid).sum(dim=1) / candidate_mask.sum(
            dim=1, keepdim=True
        ).clamp_min(1).to(candidate_embeddings.dtype)
        context = torch.cat((global_context, pooled), dim=-1)
        learned_gates = 2.0 * torch.sigmoid(self.gate(context))
        context_gates = (
            learned_gates if self.use_context_gate else torch.ones_like(learned_gates)
        )

        phi_raw_values = torch.stack(
            tuple(
                function.raw_values(factor_values[:, :, index])
                for index, function in enumerate(self.factor_functions)
            ),
            dim=-1,
        )
        phi_centered_values = torch.stack(
            tuple(
                function(factor_values[:, :, index])
                for index, function in enumerate(self.factor_functions)
            ),
            dim=-1,
        )
        factor_contributions = phi_centered_values * context_gates[:, None, :5]
        interaction_raw_values = torch.stack(
            (
                self.interaction_gu.raw_values(
                    factor_values[:, :, 0], factor_values[:, :, 2]
                ),
                self.interaction_fu.raw_values(
                    factor_values[:, :, 1], factor_values[:, :, 2]
                ),
            ),
            dim=-1,
        )
        interaction_centered_values = torch.stack(
            (
                self.interaction_gu(factor_values[:, :, 0], factor_values[:, :, 2]),
                self.interaction_fu(factor_values[:, :, 1], factor_values[:, :, 2]),
            ),
            dim=-1,
        )
        priority = factor_contributions.sum(dim=-1)
        if self.use_interactions:
            priority = priority + (
                interaction_centered_values * context_gates[:, None, 5:]
            ).sum(dim=-1)
        priority = priority.masked_fill(~candidate_mask, float("-inf"))
        return FactorizedPriorityOutput(
            priority_logits=priority,
            factor_values=factor_values * valid,
            phi_raw_values=phi_raw_values * valid,
            phi_centered_values=phi_centered_values * valid,
            factor_contributions=factor_contributions * valid,
            interaction_raw_values=interaction_raw_values * valid,
            interaction_centered_values=interaction_centered_values * valid,
            interaction_values=interaction_centered_values * valid,
            context_gates=context_gates,
        )
