"""Current fixed-rule policies plus isolated historical replay support."""

from __future__ import annotations

from fjsp_sfr.core.online import SchedulingProblem
from fjsp_sfr.dispatch.ca_metrics import CADecisionMetrics, CandidateMetrics
from fjsp_sfr.dispatch.ca_rules import rule_applicability_mask, select_job
from fjsp_sfr.dispatch.types import RuleAction, RuleDecision
from fjsp_sfr.env.ca_rule_selection import CAStepResult, OnlineRuleSelectionEnv


CURRENT_FIXED_RULE_FALLBACK = RuleAction.PROJECTED_TARDINESS


def current_fixed_rule_action(
    problem: SchedulingProblem,
    metrics: CADecisionMetrics,
    target_rule: RuleAction,
) -> RuleAction:
    """Choose a current legal rule, using the common R1 fallback when masked."""

    mask = rule_applicability_mask(problem, metrics)
    return target_rule if mask[int(target_rule)] else CURRENT_FIXED_RULE_FALLBACK


def step_current_fixed_rule(
    env: OnlineRuleSelectionEnv,
    target_rule: RuleAction,
) -> CAStepResult:
    """Advance only through the current RuleAction/action-mask/env.step path."""

    problem = env._problem_view()
    action = current_fixed_rule_action(problem, env.current_metrics, target_rule)
    return env.step(action)


# Historical replay support below is deliberately separate from current fixed
# benchmarks. External checkpoint calendars of the original 12 held-out instances
# let that archived policy be reproduced, but must never be used as a current
# R1--R8 comparator.
#
# These
# constants let the historical benchmark be replayed after the normal generator
# was made causal; they are inputs to generation, never inferred from job data.
HISTORICAL_HELDOUT_CHECKPOINT_BASES = (
    50,
    47,
    50,
    55,
    58,
    62,
    56,
    70,
    58,
    48,
    59,
    62,
)


def _service_candidate(metrics: CADecisionMetrics) -> CandidateMetrics:
    return min(
        metrics.candidates,
        key=lambda candidate: (
            -candidate.service_value,
            candidate.optimistic_completion,
            candidate.job_id,
        ),
    )


def _projected_slack_candidate(
    problem: SchedulingProblem, metrics: CADecisionMetrics
) -> CandidateMetrics:
    return min(
        metrics.candidates,
        key=lambda candidate: (
            problem.get_job(candidate.job_id).due_date
            - candidate.optimistic_completion,
            candidate.optimistic_completion,
            candidate.job_id,
        ),
    )


def _service_loss(
    problem: SchedulingProblem,
    metrics: CADecisionMetrics,
    candidate: CandidateMetrics,
    completion: int,
) -> float:
    job = problem.get_job(candidate.job_id)
    loss = 0.0
    entity = problem.get_entity(candidate.entity_id)
    for index, checkpoint in enumerate(entity.checkpoints):
        checkpoint_metrics = metrics.checkpoint(candidate.entity_id, index)
        if checkpoint.time < metrics.decision_time:
            continue
        if completion > checkpoint.time:
            loss += checkpoint.weight * min(
                float(job.quantity), checkpoint_metrics.residual_gap
            )
    return loss


def _compare_historical_local_order(
    problem: SchedulingProblem,
    metrics: CADecisionMetrics,
    job_a: CandidateMetrics,
    job_b: CandidateMetrics,
) -> tuple[float, float]:
    """Preserve the historical benchmark comparison outside the live rules."""

    tau = metrics.decision_time
    c_a_ab = tau + job_a.processing_time + job_a.downstream_min_processing
    c_b_ab = (
        tau
        + job_a.processing_time
        + job_b.processing_time
        + job_b.downstream_min_processing
    )
    c_b_ba = tau + job_b.processing_time + job_b.downstream_min_processing
    c_a_ba = (
        tau
        + job_b.processing_time
        + job_a.processing_time
        + job_a.downstream_min_processing
    )

    def objective(
        first: CandidateMetrics,
        c_first: int,
        second: CandidateMetrics,
        c_second: int,
    ) -> float:
        tardiness = max(
            0, c_first - problem.get_job(first.job_id).due_date
        ) + max(0, c_second - problem.get_job(second.job_id).due_date)
        service_loss = _service_loss(
            problem, metrics, first, c_first
        ) + _service_loss(problem, metrics, second, c_second)
        return problem.alpha * tardiness + problem.beta * service_loss

    return (
        objective(job_a, c_a_ab, job_b, c_b_ab),
        objective(job_b, c_b_ba, job_a, c_a_ba),
    )


def frozen_r4_fallback_decision(
    problem: SchedulingProblem, metrics: CADecisionMetrics
) -> RuleDecision:
    """Frozen pre-tournament P6 fallback used by the historical R4 benchmark."""

    service = _service_candidate(metrics)
    tardiness = _projected_slack_candidate(problem, metrics)
    if service.job_id == tardiness.job_id:
        selected = service
        scores = (
            ("frozen_fallback", "same_service_and_tardiness_candidate"),
        )
    else:
        h_service_first, h_tardiness_first = _compare_historical_local_order(
            problem, metrics, service, tardiness
        )
        if h_service_first < h_tardiness_first:
            selected = service
        elif h_tardiness_first < h_service_first:
            selected = tardiness
        else:
            selected = min((service, tardiness), key=lambda candidate: candidate.job_id)
        scores = (
            ("frozen_fallback", "pre_tournament_P6"),
            ("H_service_first", h_service_first),
            ("H_tardiness_first", h_tardiness_first),
        )
    return RuleDecision(
        action=RuleAction.PROJECTED_TARDINESS,
        job_id=selected.job_id,
        operation_id=selected.operation_id,
        machine_id=metrics.machine_id,
        decision_time=metrics.decision_time,
        score_components=scores,
    )


def frozen_fixed_r4_decision(
    problem: SchedulingProblem, metrics: CADecisionMetrics
) -> RuleDecision:
    """Select R4 when applicable, otherwise use its immutable fallback."""

    mask = rule_applicability_mask(problem, metrics)
    if mask[int(RuleAction.SERVICE_VALUE_DENSITY)]:
        return select_job(problem, metrics, RuleAction.SERVICE_VALUE_DENSITY)
    return frozen_r4_fallback_decision(problem, metrics)


def step_frozen_fixed_r4(env: OnlineRuleSelectionEnv) -> CAStepResult:
    """Advance the archived pre-eight-rule policy for historical replay only."""

    problem = env._problem_view()
    decision = frozen_fixed_r4_decision(problem, env.current_metrics)
    if decision.action == RuleAction.SERVICE_VALUE_DENSITY:
        return env.step(RuleAction.SERVICE_VALUE_DENSITY)
    return env.step_explicit_decision(decision)
