"""Stable eight-rule action and decision types."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class RuleAction(IntEnum):
    PROJECTED_TARDINESS = 0
    SHORTEST_REMAINING_PROCESSING = 1
    FCFS = 2
    MULTI_CHECKPOINT_SERVICE_VALUE = 3
    SERVICE_VALUE_DENSITY = 4
    CRITICAL_CHECKPOINT_RESCUE = 5
    FORECAST_SHORTAGE_PRIORITY = 6
    FORECAST_LATE_ARRIVAL_PRIORITY = 7

    @property
    def paper_number(self) -> int:
        return int(self) + 1


@dataclass(frozen=True)
class RuleDecision:
    action: RuleAction
    job_id: int
    operation_id: int
    machine_id: int
    decision_time: int
    score_components: tuple[tuple[str, float | int | str], ...]

    @property
    def paper_rule_number(self) -> int:
        return self.action.paper_number


@dataclass(frozen=True)
class DirectCandidateDecision:
    """Concrete candidate-index action produced without dispatch-rule mediation."""

    candidate_index: int
    job_id: int
    operation_id: int
    machine_id: int
    decision_time: int
    score_components: tuple[tuple[str, float | int | str], ...] = ()
