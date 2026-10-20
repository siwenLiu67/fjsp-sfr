"""Eight lightweight job-selection rules for one fixed trigger machine."""

from __future__ import annotations

from collections.abc import Callable

from fjsp_sfr.core.online import SchedulingProblem

from .ca_metrics import CADecisionMetrics, CandidateMetrics, CheckpointMetrics
from .types import RuleAction, RuleDecision


_Selector = Callable[
    [SchedulingProblem, CADecisionMetrics],
    tuple[CandidateMetrics, tuple[tuple[str, float | int | str], ...]],
]


def _reachable_checkpoints(
    metrics: CADecisionMetrics,
    candidate: CandidateMetrics,
) -> tuple[CheckpointMetrics, ...]:
    return tuple(
        metrics.checkpoint(candidate.entity_id, checkpoint_index)
        for checkpoint_index in candidate.reachable_unmet_checkpoints
    )


def _late_arrival_eligible(checkpoint: CheckpointMetrics) -> bool:
    return (
        checkpoint.future_required > 0.0
        and checkpoint.conservative_forecast >= checkpoint.future_required
        and checkpoint.release_slack is not None
    )


def rule_applicability_mask(
    problem: SchedulingProblem,
    metrics: CADecisionMetrics,
) -> tuple[bool, ...]:
    """Return direct applicability only; no rule aliases or fallbacks."""

    del problem
    if not metrics.is_active:
        return (False,) * len(RuleAction)

    service_positive = any(
        candidate.service_value > 0.0 for candidate in metrics.candidates
    )
    reachable = any(
        candidate.reachable_unmet_checkpoints for candidate in metrics.candidates
    )
    forecast_shortage = any(
        checkpoint.arrival_insufficiency > 0.0
        for candidate in metrics.candidates
        for checkpoint in _reachable_checkpoints(metrics, candidate)
    )
    forecast_late = any(
        _late_arrival_eligible(checkpoint)
        for candidate in metrics.candidates
        for checkpoint in _reachable_checkpoints(metrics, candidate)
    )
    return (
        True,
        True,
        True,
        service_positive,
        service_positive,
        reachable,
        forecast_shortage,
        forecast_late,
    )


def select_job(
    problem: SchedulingProblem,
    metrics: CADecisionMetrics,
    action: RuleAction | int,
) -> RuleDecision:
    if not metrics.is_active or metrics.machine_id is None:
        raise ValueError("a dispatch rule requires an active trigger machine")
    selected_action = RuleAction(action)
    if not rule_applicability_mask(problem, metrics)[int(selected_action)]:
        raise ValueError(
            f"R{selected_action.paper_number} primary condition is not applicable"
        )
    selector: _Selector = {
        RuleAction.PROJECTED_TARDINESS: _r1,
        RuleAction.SHORTEST_REMAINING_PROCESSING: _r2,
        RuleAction.FCFS: _r3,
        RuleAction.MULTI_CHECKPOINT_SERVICE_VALUE: _r4,
        RuleAction.SERVICE_VALUE_DENSITY: _r5,
        RuleAction.CRITICAL_CHECKPOINT_RESCUE: _r6,
        RuleAction.FORECAST_SHORTAGE_PRIORITY: _r7,
        RuleAction.FORECAST_LATE_ARRIVAL_PRIORITY: _r8,
    }[selected_action]
    candidate, scores = selector(problem, metrics)
    return RuleDecision(
        action=selected_action,
        job_id=candidate.job_id,
        operation_id=candidate.operation_id,
        machine_id=metrics.machine_id,
        decision_time=metrics.decision_time,
        score_components=scores,
    )


def _select_max(
    scored: tuple[tuple[float, CandidateMetrics], ...],
) -> tuple[float, CandidateMetrics]:
    return min(
        scored,
        key=lambda item: (
            -item[0],
            item[1].optimistic_completion,
            item[1].job_id,
        ),
    )


def _select_min(
    scored: tuple[tuple[float, CandidateMetrics], ...],
) -> tuple[float, CandidateMetrics]:
    return min(
        scored,
        key=lambda item: (
            item[0],
            item[1].optimistic_completion,
            item[1].job_id,
        ),
    )


def _r1(problem: SchedulingProblem, metrics: CADecisionMetrics):
    scored = tuple(
        (
            candidate.optimistic_completion
            - problem.get_job(candidate.job_id).due_date,
            candidate,
        )
        for candidate in metrics.candidates
    )
    score, selected = _select_max(scored)
    return selected, (("projected_tardiness", score),)


def _r2(problem: SchedulingProblem, metrics: CADecisionMetrics):
    del problem
    scored = tuple(
        (float(metrics.remaining_time(candidate)), candidate)
        for candidate in metrics.candidates
    )
    score, selected = _select_min(scored)
    return selected, (("remaining_processing", score),)


def _r3(problem: SchedulingProblem, metrics: CADecisionMetrics):
    scored = tuple(
        (float(problem.get_job(candidate.job_id).release_time), candidate)
        for candidate in metrics.candidates
    )
    score, selected = _select_min(scored)
    return selected, (("release_time", score),)


def _r4(problem: SchedulingProblem, metrics: CADecisionMetrics):
    del problem
    scored = tuple(
        (candidate.service_value, candidate) for candidate in metrics.candidates
    )
    score, selected = _select_max(scored)
    return selected, (("service_value", score),)


def _r5(problem: SchedulingProblem, metrics: CADecisionMetrics):
    epsilon = problem.r5_epsilon
    scored = tuple(
        (
            candidate.service_value
            / (metrics.remaining_time(candidate) + epsilon),
            candidate,
        )
        for candidate in metrics.candidates
    )
    score, selected = _select_max(scored)
    return selected, (
        ("service_value_density", score),
        ("service_value", selected.service_value),
    )


def _r6(problem: SchedulingProblem, metrics: CADecisionMetrics):
    epsilon = problem.r5_epsilon
    scored = tuple(
        (
            min(
                (
                    checkpoint.checkpoint_time
                    - candidate.optimistic_completion
                )
                /
                (
                    checkpoint.checkpoint_time
                    - metrics.decision_time
                    + epsilon
                )
                for checkpoint in _reachable_checkpoints(metrics, candidate)
            ),
            candidate,
        )
        for candidate in metrics.candidates
        if candidate.reachable_unmet_checkpoints
    )
    score, selected = _select_min(scored)
    return selected, (("critical_checkpoint_margin", score),)


def _r7(problem: SchedulingProblem, metrics: CADecisionMetrics):
    epsilon = problem.r5_epsilon
    scored: list[tuple[float, CandidateMetrics, CheckpointMetrics]] = []
    for candidate in metrics.candidates:
        quantity = float(problem.get_job(candidate.job_id).quantity)
        values = tuple(
            (
                checkpoint.arrival_insufficiency
                / (checkpoint.residual_gap + epsilon)
                * checkpoint.checkpoint_weight
                * min(quantity, checkpoint.residual_gap),
                checkpoint,
            )
            for checkpoint in _reachable_checkpoints(metrics, candidate)
            if checkpoint.arrival_insufficiency > 0.0
        )
        if values:
            value, checkpoint = min(
                values,
                key=lambda item: (
                    -item[0],
                    item[1].checkpoint_time,
                    item[1].checkpoint_index,
                ),
            )
            scored.append((value, candidate, checkpoint))
    score, selected, checkpoint = min(
        scored,
        key=lambda item: (
            -item[0],
            item[1].optimistic_completion,
            item[1].job_id,
        ),
    )
    return selected, (
        ("forecast_shortage_priority", score),
        ("entity_id", checkpoint.entity_id),
        ("checkpoint_index", checkpoint.checkpoint_index),
    )


def _r8(problem: SchedulingProblem, metrics: CADecisionMetrics):
    epsilon = problem.r5_epsilon
    scored: list[tuple[float, CandidateMetrics, CheckpointMetrics]] = []
    for candidate in metrics.candidates:
        quantity = float(problem.get_job(candidate.job_id).quantity)
        values = tuple(
            (
                checkpoint.future_required
                / (checkpoint.residual_gap + epsilon)
                * (
                    1.0
                    - checkpoint.release_slack
                    / (
                        checkpoint.checkpoint_time
                        - metrics.decision_time
                        + epsilon
                    )
                )
                * checkpoint.checkpoint_weight
                * min(quantity, checkpoint.residual_gap),
                checkpoint,
            )
            for checkpoint in _reachable_checkpoints(metrics, candidate)
            if _late_arrival_eligible(checkpoint)
        )
        if values:
            value, checkpoint = min(
                values,
                key=lambda item: (
                    -item[0],
                    item[1].checkpoint_time,
                    item[1].checkpoint_index,
                ),
            )
            scored.append((value, candidate, checkpoint))
    score, selected, checkpoint = min(
        scored,
        key=lambda item: (
            -item[0],
            item[1].optimistic_completion,
            item[1].job_id,
        ),
    )
    return selected, (
        ("forecast_late_arrival_priority", score),
        ("entity_id", checkpoint.entity_id),
        ("checkpoint_index", checkpoint.checkpoint_index),
    )
