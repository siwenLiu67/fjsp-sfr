"""Classic dispatching rules evaluated from raw candidate and job quantities."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from fjsp_sfr.core.online import SchedulingProblem
from fjsp_sfr.core.state import ScheduleState
from fjsp_sfr.dispatch.ca_metrics import CADecisionMetrics


class ClassicRuleKind(str, Enum):
    SPT = "SPT"
    EDD = "EDD"
    ATC = "ATC"
    MIN_SLACK = "MIN_SLACK"


@dataclass(frozen=True)
class ClassicRule:
    """Duck-compatible rule whose scores are computed from unnormalised data."""

    name: str
    kind: ClassicRuleKind
    atc_k: float = 2.0
    expression: str = "raw classic dispatch rule"

    @property
    def rule_type(self):
        return None

    def is_applicable(self, _summary_row) -> bool:
        return True

    def evaluate_candidates(
        self,
        problem: SchedulingProblem,
        state: ScheduleState,
        metrics: CADecisionMetrics,
    ) -> list[float]:
        candidates = tuple(sorted(metrics.candidates, key=lambda item: item.job_id))
        remaining_by_job: dict[int, float] = {}
        for candidate in candidates:
            job = problem.get_job(candidate.job_id)
            next_index = state.next_op_index_for_job(job.job_id)
            remaining_by_job[job.job_id] = float(
                candidate.processing_time
                + sum(
                    operation.min_processing_time
                    for operation in job.operations[next_index + 1 :]
                )
            )
        mean_processing = sum(remaining_by_job.values()) / max(
            1, len(remaining_by_job)
        )
        scores: list[float] = []
        for candidate in candidates:
            job = problem.get_job(candidate.job_id)
            remaining = remaining_by_job[job.job_id]
            if self.kind == ClassicRuleKind.SPT:
                score = -float(candidate.processing_time)
            elif self.kind == ClassicRuleKind.EDD:
                score = -float(job.due_date)
            elif self.kind == ClassicRuleKind.MIN_SLACK:
                score = -float(job.due_date - state.current_time - remaining)
            elif self.kind == ClassicRuleKind.ATC:
                allowance = max(
                    0.0,
                    float(job.due_date - state.current_time) - remaining,
                )
                score = math.exp(
                    -allowance / (self.atc_k * max(mean_processing, 1e-12))
                ) / max(remaining, 1e-12)
            else:  # pragma: no cover - Enum construction prevents this branch.
                raise ValueError(f"unsupported classic rule kind: {self.kind}")
            scores.append(score)
        return scores

    def to_dict(self) -> dict:
        return {
            "kind": "classic",
            "name": self.name,
            "classic_kind": self.kind.value,
            "atc_k": self.atc_k,
            "expression": self.expression,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ClassicRule":
        return cls(
            name=data["name"],
            kind=ClassicRuleKind(data["classic_kind"]),
            atc_k=float(data.get("atc_k", 2.0)),
            expression=data.get("expression", "raw classic dispatch rule"),
        )


def build_classic_rule_library():
    from fjsp_sfr.dispatch.rule_library import RuleLibrary

    return RuleLibrary(
        rules=(
            ClassicRule("SPT", ClassicRuleKind.SPT),
            ClassicRule("EDD", ClassicRuleKind.EDD),
            ClassicRule("ATC_K2", ClassicRuleKind.ATC, atc_k=2.0),
            ClassicRule("MIN_SLACK", ClassicRuleKind.MIN_SLACK),
        )
    )


__all__ = [
    "ClassicRule",
    "ClassicRuleKind",
    "build_classic_rule_library",
]
