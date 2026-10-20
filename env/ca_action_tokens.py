"""Rule-conditioned action tokens for the eight dispatch actions."""

from __future__ import annotations

import math

from fjsp_sfr.core.online import SchedulingProblem
from fjsp_sfr.core.state import ScheduleState
from fjsp_sfr.dispatch.ca_metrics import CADecisionMetrics, CandidateMetrics, CheckpointMetrics
from fjsp_sfr.dispatch.ca_rules import rule_applicability_mask, select_job
from fjsp_sfr.dispatch.types import RuleAction

from .ca_observation import resolve_observation_scales


ACTION_TOKEN_FEATURE_NAMES = (
    "primary_applicable",
    "fallback_to_r6",
    "rule_mode_code",
    "has_checkpoint_context",
    "candidate_quantity_ratio",
    "operation_processing_ratio",
    "downstream_processing_ratio",
    "optimistic_completion_horizon",
    "candidate_due_slack",
    "candidate_service_value_ratio",
    "normalized_primary_score",
    "future_required_ratio",
    "arrival_insufficiency_ratio",
    "forecast_release_slack",
)

ActionToken = tuple[
    float, float, float, float, float, float, float,
    float, float, float, float, float, float, float,
]


def _score_map(components) -> dict[str, float | int | str]:
    return {str(name): value for name, value in components}


def _candidate(metrics: CADecisionMetrics, job_id: int) -> CandidateMetrics:
    return next(item for item in metrics.candidates if item.job_id == job_id)


def _context_checkpoint(
    metrics: CADecisionMetrics,
    candidate: CandidateMetrics,
    scores: dict[str, float | int | str],
) -> CheckpointMetrics | None:
    entity_id = scores.get("entity_id")
    checkpoint_index = scores.get("checkpoint_index")
    if isinstance(entity_id, int) and isinstance(checkpoint_index, int):
        return metrics.checkpoint(entity_id, checkpoint_index)
    reachable = tuple(
        metrics.checkpoint(candidate.entity_id, index)
        for index in candidate.reachable_unmet_checkpoints
    )
    if not reachable:
        return None
    return min(
        reachable,
        key=lambda item: (
            item.checkpoint_time,
            -item.arrival_insufficiency,
            item.checkpoint_index,
        ),
    )


def _normalized_rule_score(
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


def _mode_code(
    action: RuleAction, scores: dict[str, float | int | str]
) -> float:
    del action, scores
    return 0.0


def build_rule_action_tokens(
    problem: SchedulingProblem,
    state: ScheduleState,
    metrics: CADecisionMetrics,
    *,
    time_scale: float | None = None,
    processing_scale: float | None = None,
) -> tuple[tuple[ActionToken, ...], tuple[tuple[int, int], ...]]:
    """Encode each applicable rule's selected candidate without fallbacks."""

    if not metrics.is_active:
        return (), ()
    t_scale, p_scale = resolve_observation_scales(
        problem,
        state,
        time_scale=time_scale,
        processing_scale=processing_scale,
    )
    max_required = max(
        checkpoint.required_quantity
        for entity in problem.entities
        for checkpoint in entity.checkpoints
    )
    service_scale = max(
        checkpoint.weight * checkpoint.required_quantity
        for entity in problem.entities
        for checkpoint in entity.checkpoints
    )
    max_required = max(max_required, 1e-12)
    service_scale = max(service_scale, 1e-12)
    applicable = rule_applicability_mask(problem, metrics)
    tokens: list[ActionToken] = []
    candidate_keys: list[tuple[int, int]] = []
    for action in RuleAction:
        if not applicable[int(action)]:
            tokens.append((0.0,) * len(ACTION_TOKEN_FEATURE_NAMES))
            candidate_keys.append((-1, -1))
            continue
        decision = select_job(problem, metrics, action)
        candidate = _candidate(metrics, decision.job_id)
        job = problem.get_job(candidate.job_id)
        scores = _score_map(decision.score_components)
        checkpoint = _context_checkpoint(metrics, candidate, scores)
        if checkpoint is None:
            has_context = 0.0
            future_required = arrival_insufficiency = release_slack = 0.0
        else:
            has_context = 1.0
            denominator = max(checkpoint.required_quantity, 1e-12)
            future_required = checkpoint.future_required / denominator
            arrival_insufficiency = checkpoint.arrival_insufficiency / denominator
            release_slack = (
                checkpoint.release_slack / t_scale
                if checkpoint.release_slack is not None
                else 0.0
            )
        tokens.append(
            (
                float(applicable[int(action)]),
                0.0,
                _mode_code(action, scores),
                has_context,
                float(job.quantity) / max_required,
                candidate.processing_time / p_scale,
                candidate.downstream_min_processing / p_scale,
                (candidate.optimistic_completion - metrics.decision_time) / t_scale,
                (job.due_date - candidate.optimistic_completion) / t_scale,
                candidate.service_value / service_scale,
                _normalized_rule_score(
                    action,
                    scores,
                    time_scale=t_scale,
                    processing_scale=p_scale,
                    service_scale=service_scale,
                ),
                future_required,
                arrival_insufficiency,
                release_slack,
            )
        )
        candidate_keys.append((decision.job_id, decision.operation_id))
    return tuple(tokens), tuple(candidate_keys)
