"""Dynamic equivalence classes over concrete dispatch outcomes."""

from __future__ import annotations

from collections.abc import Sequence

import torch


ConcreteDispatch = tuple[int, int, int]
ActionGroups = tuple[tuple[int, ...], ...]


def build_action_groups(
    concrete_dispatches: Sequence[ConcreteDispatch],
    action_mask: Sequence[bool] | torch.Tensor,
    *,
    merge_equivalent: bool,
) -> ActionGroups:
    """Partition applicable rules by identical job/operation/machine dispatch."""

    mask = tuple(bool(value) for value in action_mask)
    if len(concrete_dispatches) != len(mask):
        raise ValueError("every rule must have one concrete dispatch result")
    applicable = tuple(index for index, valid in enumerate(mask) if valid)
    if not applicable:
        raise ValueError("every decision must have at least one applicable rule")
    if not merge_equivalent:
        return tuple((index,) for index in applicable)
    grouped: dict[ConcreteDispatch, list[int]] = {}
    for index in applicable:
        grouped.setdefault(concrete_dispatches[index], []).append(index)
    return tuple(tuple(indices) for indices in grouped.values())


def validate_action_groups(
    action_groups: ActionGroups,
    action_mask: Sequence[bool] | torch.Tensor,
) -> None:
    """Fail closed unless groups exactly partition applicable rules."""

    mask = tuple(bool(value) for value in action_mask)
    applicable = tuple(index for index, valid in enumerate(mask) if valid)
    if not action_groups or any(not group for group in action_groups):
        raise ValueError("each decision requires non-empty action groups")
    flattened = tuple(index for group in action_groups for index in group)
    if tuple(sorted(flattened)) != applicable or len(set(flattened)) != len(flattened):
        raise ValueError("action groups must partition applicable rules exactly")
    if any(tuple(sorted(group)) != group for group in action_groups):
        raise ValueError("rule ids within an action group must be sorted")


def aggregate_rule_log_probs(
    rule_log_probs: torch.Tensor,
    action_groups: ActionGroups,
    action_mask: Sequence[bool] | torch.Tensor | None = None,
) -> torch.Tensor:
    """Return log probabilities of groups as exact sums of rule probabilities."""

    if rule_log_probs.ndim != 1:
        raise ValueError("rule_log_probs must be one-dimensional")
    if action_mask is None:
        flattened = tuple(index for group in action_groups for index in group)
        if (
            not flattened
            or len(set(flattened)) != len(flattened)
            or min(flattened) < 0
            or max(flattened) >= rule_log_probs.shape[0]
        ):
            raise ValueError("action groups contain invalid rule indices")
    else:
        validate_action_groups(action_groups, action_mask)
    return torch.stack(
        [
            torch.logsumexp(
                rule_log_probs[
                    torch.tensor(group, dtype=torch.long, device=rule_log_probs.device)
                ],
                dim=0,
            )
            for group in action_groups
        ]
    )


__all__ = [
    "ActionGroups",
    "ConcreteDispatch",
    "aggregate_rule_log_probs",
    "build_action_groups",
    "validate_action_groups",
]
