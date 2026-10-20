"""Lightweight tokens describing each rule's current concrete decision."""

from __future__ import annotations

import math

from fjsp_sfr.core.online import SchedulingProblem
from fjsp_sfr.core.state import ScheduleState
from fjsp_sfr.dispatch.ca_metrics import (
    CADecisionMetrics,
    CandidateMetrics,
    CheckpointMetrics,
)
from fjsp_sfr.dispatch.ca_rules import rule_applicability_mask, select_job
from fjsp_sfr.dispatch.types import RuleAction

from .ca_observation import resolve_observation_scales


RULE_TOKEN_FEATURES = (
    "applicable",
    "primary_score",
    "processing_ratio",
    "optimistic_remaining_time",
    "projected_due_slack",
    "quantity_ratio",
    "service_value_ratio",
    "reachable_checkpoint_ratio",
    "arrival_insufficiency_ratio",
    "forecast_release_slack",
)

RuleToken = tuple[
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
]


def _score_map(components) -> dict[str, float | int | str]:
    return {str(name): value for name, value in components}


def _candidate(metrics: CADecisionMetrics, job_id: int) -> CandidateMetrics:
    return next(item for item in metrics.candidates if item.job_id == job_id)


def _primary_score(
    problem: SchedulingProblem,
    action: RuleAction,
    scores: dict[str, float | int | str],
    *,
    time_scale: float,
    processing_scale: float,
    service_scale: float,
) -> float:
    if action == RuleAction.PROJECTED_TARDINESS:
        raw = float(scores.get("projected_tardiness", 0.0)) / time_scale
    elif action == RuleAction.SHORTEST_REMAINING_PROCESSING:
        raw = float(scores.get("remaining_processing", 0.0)) / time_scale
    elif action == RuleAction.FCFS:
        raw = float(scores.get("release_time", 0.0)) / time_scale
    elif action == RuleAction.MULTI_CHECKPOINT_SERVICE_VALUE:
        raw = float(scores.get("service_value", 0.0)) / service_scale
    elif action == RuleAction.SERVICE_VALUE_DENSITY:
        raw = (
            float(scores.get("service_value_density", 0.0))
            * processing_scale
            / service_scale
        )
    elif action == RuleAction.CRITICAL_CHECKPOINT_RESCUE:
        raw = float(scores.get("critical_checkpoint_margin", 0.0))
    elif action == RuleAction.FORECAST_SHORTAGE_PRIORITY:
        raw = float(scores.get("forecast_shortage_priority", 0.0)) / service_scale
    else:
        raw = float(scores.get("forecast_late_arrival_priority", 0.0)) / service_scale
    return math.tanh(raw)


def _r7_context(
    problem: SchedulingProblem,
    metrics: CADecisionMetrics,
    candidate: CandidateMetrics,
) -> CheckpointMetrics | None:
    quantity = float(problem.get_job(candidate.job_id).quantity)
    choices = []
    for checkpoint_index in candidate.reachable_unmet_checkpoints:
        checkpoint = metrics.checkpoint(candidate.entity_id, checkpoint_index)
        if checkpoint.arrival_insufficiency <= 0.0:
            continue
        contribution = (
            checkpoint.checkpoint_weight
            * checkpoint.arrival_insufficiency
            / (checkpoint.residual_gap + problem.r5_epsilon)
            * min(quantity, checkpoint.residual_gap)
        )
        choices.append((contribution, checkpoint))
    if not choices:
        return None
    return min(
        choices,
        key=lambda item: (
            -item[0],
            item[1].checkpoint_time,
            item[1].checkpoint_index,
        ),
    )[1]


def _r8_context(
    problem: SchedulingProblem,
    metrics: CADecisionMetrics,
    candidate: CandidateMetrics,
) -> CheckpointMetrics | None:
    quantity = float(problem.get_job(candidate.job_id).quantity)
    choices = []
    for checkpoint_index in candidate.reachable_unmet_checkpoints:
        checkpoint = metrics.checkpoint(candidate.entity_id, checkpoint_index)
        if not (
            checkpoint.checkpoint_time >= metrics.decision_time
            and checkpoint.future_required > 0.0
            and checkpoint.conservative_forecast >= checkpoint.future_required
            and checkpoint.release_slack is not None
        ):
            continue
        dependency = checkpoint.future_required / (
            checkpoint.residual_gap + problem.r5_epsilon
        )
        lateness = 1.0 - checkpoint.release_slack / (
            checkpoint.checkpoint_time
            - metrics.decision_time
            + problem.r5_epsilon
        )
        contribution = (
            checkpoint.checkpoint_weight
            * dependency
            * lateness
            * min(quantity, checkpoint.residual_gap)
        )
        choices.append((contribution, checkpoint))
    if not choices:
        return None
    return min(
        choices,
        key=lambda item: (
            -item[0],
            item[1].checkpoint_time,
            item[1].checkpoint_index,
        ),
    )[1]


def build_rule_tokens(
    problem: SchedulingProblem,
    state: ScheduleState,
    metrics: CADecisionMetrics,
    *,
    time_scale: float | None = None,
    processing_scale: float | None = None,
) -> tuple[tuple[RuleToken, ...], tuple[tuple[int, int], ...]]:
    """Encode the actual selected candidate and primary score for R1--R8."""

    if not metrics.is_active:
        return (), ()
    t_scale, p_scale = resolve_observation_scales(
        problem,
        state,
        time_scale=time_scale,
        processing_scale=processing_scale,
    )
    quantity_scale = max(
        checkpoint.required_quantity
        for entity in problem.entities
        for checkpoint in entity.checkpoints
    )
    service_scale = max(
        checkpoint.weight * checkpoint.required_quantity
        for entity in problem.entities
        for checkpoint in entity.checkpoints
    )
    quantity_scale = max(float(quantity_scale), 1e-12)
    service_scale = max(float(service_scale), 1e-12)
    unmet_by_entity = {
        entity.entity_id: sum(
            checkpoint.entity_id == entity.entity_id
            and checkpoint.checkpoint_time >= state.current_time
            and checkpoint.residual_gap > 0.0
            for checkpoint in metrics.checkpoints.values()
        )
        for entity in problem.entities
    }
    applicable = rule_applicability_mask(problem, metrics)
    tokens: list[RuleToken] = []
    candidate_keys = []
    for action in RuleAction:
        if not applicable[int(action)]:
            tokens.append((0.0,) * len(RULE_TOKEN_FEATURES))
            candidate_keys.append((-1, -1))
            continue
        decision = select_job(problem, metrics, action)
        candidate = _candidate(metrics, decision.job_id)
        job = problem.get_job(candidate.job_id)
        if job.release_time > state.current_time:
            raise ValueError("rule tokens may only describe released jobs")
        scores = _score_map(decision.score_components)
        checkpoint = None
        if action == RuleAction.FORECAST_SHORTAGE_PRIORITY:
            checkpoint = _r7_context(problem, metrics, candidate)
        elif action == RuleAction.FORECAST_LATE_ARRIVAL_PRIORITY:
            checkpoint = _r8_context(problem, metrics, candidate)
        if checkpoint is None:
            insufficiency_ratio = 0.0
            release_slack = -1.0
        else:
            insufficiency_ratio = (
                checkpoint.arrival_insufficiency
                / max(checkpoint.required_quantity, 1e-12)
            )
            release_slack = (
                checkpoint.release_slack / t_scale
                if checkpoint.release_slack is not None
                else -1.0
            )
        reachable_denominator = max(1, unmet_by_entity[candidate.entity_id])
        tokens.append(
            (
                float(applicable[int(action)]),
                _primary_score(
                    problem,
                    action,
                    scores,
                    time_scale=t_scale,
                    processing_scale=p_scale,
                    service_scale=service_scale,
                ),
                candidate.processing_time / p_scale,
                (candidate.optimistic_completion - state.current_time) / t_scale,
                (job.due_date - candidate.optimistic_completion) / t_scale,
                float(job.quantity) / quantity_scale,
                candidate.service_value / service_scale,
                len(candidate.reachable_unmet_checkpoints) / reachable_denominator,
                insufficiency_ratio,
                release_slack,
            )
        )
        candidate_keys.append((decision.job_id, decision.operation_id))
    return tuple(tokens), tuple(candidate_keys)
