"""Rule features, including the formal 14-dimensional Table 5-2 vector.

``theory11`` and ``theory11_plus3`` are retained as explicitly historical
interfaces.  The manuscript-aligned interface is ``paper14``; keeping a
separate identifier prevents old serialized rules from silently acquiring new
mathematical semantics.
"""

from __future__ import annotations

import math

from fjsp_sfr.core.online import SchedulingProblem
from fjsp_sfr.core.state import ScheduleState
from fjsp_sfr.dispatch.ca_metrics import (
    CADecisionMetrics,
    CandidateMetrics,
    analyze_checkpoint_capacity,
    optimistic_job_completion_lower_bound,
    select_target_checkpoint,
    timely_visible_jobs,
)
from fjsp_sfr.env.ca_observation import (
    build_theory_factor_values,
    resolve_observation_scales,
)


THEORY_FEATURE_NAMES: tuple[str, ...] = (
    "gap_pressure",
    "forecast_deficit",
    "checkpoint_slack_urgency",
    "multi_checkpoint_value",
    "tardiness_exposure",
    "due_date_slack",
    "capacity_pressure",
    "forecast_flag",
    "capacity_flag",
    "quantity_contribution",
    "processing_ratio",
)

# Sec. 5, Table 5-2.  This order is a public contract shared by formal rules
# and job-node construction.  Do not reuse THEORY_FEATURE_NAMES here: that
# tuple names the pre-contract 11-dimensional feature interface.
PAPER14_FEATURE_NAMES: tuple[str, ...] = (
    "gap_pressure",
    "timely_eligibility",
    "checkpoint_slack_urgency",
    "quantity_contribution",
    "multi_checkpoint_value",
    "forecast_deficit",
    "capacity_pressure",
    "capacity_flag",
    "due_date_slack",
    "tardiness_exposure",
    "proc_time",
    "processing_ratio",
    "remaining_work",
    "critical_ratio",
)

GENERAL_DISPATCH_FEATURE_NAMES: tuple[str, ...] = (
    "proc_time",
    "remaining_work",
    "critical_ratio",
)

FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "theory11": THEORY_FEATURE_NAMES,
    "theory11_plus3": THEORY_FEATURE_NAMES + GENERAL_DISPATCH_FEATURE_NAMES,
    "paper14": PAPER14_FEATURE_NAMES,
}

BATCH_CONTEXT_NAMES: dict[str, tuple[str, ...]] = {
    "theory11": (),
    "theory11_plus3": ("mean_remaining_work",),
    "paper14": (),
}


# Latest Sec. 6 generator distribution: p ~ discrete U[1, 50] and
# L ~ discrete U[2, 5].  Thus p_bar=25.5 and W_bar=3.5*25.5=89.25.
PAPER_MEAN_PROCESSING_TIME = 25.5
PAPER_MEAN_JOB_WORK = 3.5 * PAPER_MEAN_PROCESSING_TIME


def feature_names_for(feature_set: str = "theory11") -> tuple[str, ...]:
    """返回特征集的稳定列顺序。"""

    try:
        return FEATURE_SETS[feature_set]
    except KeyError as error:
        raise ValueError(f"unknown feature_set: {feature_set}") from error


def feature_set_for_names(feature_names: tuple[str, ...]) -> str:
    """Resolve a canonical feature-set name from its ordered names."""

    for name, names in FEATURE_SETS.items():
        if tuple(feature_names) == names:
            return name
    raise ValueError("feature names do not match a supported feature set")


def expression_names_for(feature_set: str = "theory11") -> tuple[str, ...]:
    """Names available to expressions, including non-node batch context."""

    return feature_names_for(feature_set) + BATCH_CONTEXT_NAMES[feature_set]


def batch_context_for_rows(
    feature_set: str, rows: tuple[tuple[float, ...], ...]
) -> dict[str, float]:
    """Derive batch aggregates without adding dimensions to job nodes."""

    if feature_set != "theory11_plus3" or not rows:
        return {}
    names = feature_names_for(feature_set)
    index = names.index("remaining_work")
    raw_normalized = [
        row[index] / max(1e-12, 1.0 - row[index]) for row in rows
    ]
    return {"mean_remaining_work": sum(raw_normalized) / len(raw_normalized)}


def _clamp_unit(value: float) -> float:
    return max(0.0, min(float(value), 1.0))


def _monotone_compress(nonnegative_ratio: float) -> float:
    """Map ``[0, +inf)`` to ``[0, 1)`` with an invertible monotone map.

    ``x / (1 + x)`` preserves every strict ordering, avoids hard clipping of
    large ratios, and has materially more resolution around the reference
    value ``x=1`` than a saturating clamp.
    """

    ratio = max(0.0, float(nonnegative_ratio))
    return ratio / (1.0 + ratio)


def _safe_ratio(numerator: float, denominator: float) -> float:
    """Paper-wide zero/non-finite convention for ratio features."""

    numerator = float(numerator)
    denominator = float(denominator)
    if (
        denominator == 0.0
        or not math.isfinite(numerator)
        or not math.isfinite(denominator)
    ):
        return 0.0
    value = numerator / denominator
    return value if math.isfinite(value) else 0.0


def _signed_compress(value: float) -> float:
    """Return sigma_pm(x)=x/(1+abs(x)), with a finite-value guard."""

    value = float(value)
    if not math.isfinite(value):
        return 0.0
    return value / (1.0 + abs(value))


def _instance_processing_scales(problem: SchedulingProblem) -> tuple[float, float]:
    """Historical fallback scales for legacy/ad-hoc fixtures.

    Formal ``paper14`` scenario runs never use this path: they use the fixed
    generation-distribution scales from Sec. 6.  The fallback exists only so
    small hand-built fixtures can state their scale without changing the old
    feature interfaces.
    """

    alternative_times = [
        float(alternative.processing_time)
        for job in problem.jobs
        for operation in job.operations
        for alternative in operation.alternatives
    ]
    job_minimum_totals = [
        float(sum(operation.min_processing_time for operation in job.operations))
        for job in problem.jobs
    ]
    mean_operation_time = sum(alternative_times) / max(1, len(alternative_times))
    mean_job_work = sum(job_minimum_totals) / max(1, len(job_minimum_totals))
    return max(mean_operation_time, 1e-12), max(mean_job_work, 1e-12)


def resolve_paper_feature_scales(
    problem: SchedulingProblem,
    *,
    mean_processing_time: float | None = None,
    mean_job_work: float | None = None,
) -> tuple[float, float]:
    """Resolve ``(p_bar, W_bar)`` without using a realized future sample.

    Explicit arguments and ``metadata['feature_normalization_scales']`` are
    intended for deterministic hand fixtures.  Formal center/L9 instances are
    identified by their public generator tag and use the fixed Sec. 6
    distribution values.  Untagged inputs must declare the scales explicitly;
    deriving them from a realized instance would leak future-job statistics.
    """

    if (mean_processing_time is None) != (mean_job_work is None):
        raise ValueError("p_bar and W_bar must be supplied together")
    if mean_processing_time is not None and mean_job_work is not None:
        p_bar, w_bar = float(mean_processing_time), float(mean_job_work)
    else:
        metadata = getattr(problem, "metadata", {}) or {}
        declared = metadata.get("feature_normalization_scales")
        if declared is not None:
            p_bar = float(declared["bar_p"])
            w_bar = float(declared["bar_W"])
        elif metadata.get("generator") in {"scenario_generator_v1", "scenario_generator_v2"}:
            p_bar = PAPER_MEAN_PROCESSING_TIME
            w_bar = PAPER_MEAN_JOB_WORK
        else:
            raise ValueError(
                "paper14 requires public generation-distribution scales; "
                "set feature_normalization_scales or pass p_bar and W_bar"
            )
    if (
        not math.isfinite(p_bar)
        or not math.isfinite(w_bar)
        or p_bar <= 0.0
        or w_bar <= 0.0
    ):
        raise ValueError("p_bar and W_bar must be finite and positive")
    return p_bar, w_bar


def _remaining_unstarted_minimum_work(
    problem: SchedulingProblem,
    state: ScheduleState,
    candidate: CandidateMetrics,
) -> float:
    """Return Table 5-2 ``W_j`` for a dispatch candidate.

    A candidate has no running operation.  ``W_j`` therefore includes the
    minimum eligible-machine duration of its current operation and every
    downstream unstarted operation; it does not use the trigger-machine
    duration for the current operation.
    """

    job = problem.get_job(candidate.job_id)
    next_index = state.next_op_index_for_job(job.job_id)
    return float(
        sum(operation.min_processing_time for operation in job.operations[next_index:])
    )


def _active_positive_gap_checkpoints(
    metrics: CADecisionMetrics,
    *,
    entity_id: int | None = None,
):
    return tuple(
        checkpoint
        for (_key, checkpoint) in sorted(
            metrics.checkpoints.items(),
            key=lambda item: (
                item[1].checkpoint_time,
                item[1].entity_id,
                item[1].checkpoint_index,
            ),
        )
        if checkpoint.checkpoint_time > metrics.decision_time
        and checkpoint.residual_gap > 0.0
        and (entity_id is None or checkpoint.entity_id == entity_id)
    )


def _paper14_row(
    problem: SchedulingProblem,
    state: ScheduleState,
    metrics: CADecisionMetrics,
    candidate: CandidateMetrics,
    *,
    bar_p: float,
    bar_w: float,
) -> tuple[float, ...]:
    """Build one manuscript-aligned Table 5-2 feature row."""

    job = problem.get_job(candidate.job_id)
    completion_lb = float(optimistic_job_completion_lower_bound(job, state))
    due_slack = float(job.due_date) - completion_lb
    remaining_work = _remaining_unstarted_minimum_work(
        problem, state, candidate
    )

    # Features 9--14 do not depend on a target checkpoint.
    due_date_slack = _signed_compress(_safe_ratio(due_slack, bar_w))
    tardiness_exposure = _monotone_compress(
        _safe_ratio(max(0.0, -due_slack), bar_w)
    )
    proc_time = _monotone_compress(
        _safe_ratio(float(candidate.processing_time), bar_p)
    )
    processing_ratio = _safe_ratio(
        float(candidate.processing_time),
        float(candidate.processing_time + candidate.downstream_min_processing),
    )
    remaining_work_feature = _monotone_compress(
        _safe_ratio(remaining_work, bar_w)
    )
    critical_ratio = (
        _monotone_compress(
            _safe_ratio(float(job.due_date - state.current_time), remaining_work)
        )
        if job.due_date > state.current_time
        else 0.0
    )

    target = select_target_checkpoint(metrics, candidate)
    if target is None:
        checkpoint_features = (0.0,) * 8
    else:
        entity_active = _active_positive_gap_checkpoints(
            metrics, entity_id=candidate.entity_id
        )
        all_active = _active_positive_gap_checkpoints(metrics)
        max_weight = max(
            (float(checkpoint.checkpoint_weight) for checkpoint in all_active),
            default=0.0,
        )
        gap = float(target.residual_gap)
        target_omega = timely_visible_jobs(
            problem,
            state,
            candidate.entity_id,
            target.checkpoint_index,
        )
        timely_job_ids = {visible.job_id for visible in target_omega}

        numerator = 0.0
        denominator = 0.0
        for checkpoint in entity_active:
            weight = float(checkpoint.checkpoint_weight)
            checkpoint_gap = float(checkpoint.residual_gap)
            denominator += weight * checkpoint_gap
            timely_ids = {
                visible.job_id
                for visible in timely_visible_jobs(
                    problem,
                    state,
                    candidate.entity_id,
                    checkpoint.checkpoint_index,
                )
            }
            if job.job_id in timely_ids:
                numerator += weight * min(float(job.quantity), checkpoint_gap)

        capacity = analyze_checkpoint_capacity(
            problem,
            state,
            candidate.entity_id,
            target.checkpoint_index,
            residual_gap=gap,
        )
        checkpoint_features = (
            _safe_ratio(float(target.checkpoint_weight), max_weight)
            * _safe_ratio(gap, float(target.required_quantity)),
            float(job.job_id in timely_job_ids),
            _monotone_compress(
                _safe_ratio(
                    completion_lb - float(state.current_time),
                    float(target.checkpoint_time - state.current_time),
                )
            ),
            _safe_ratio(min(float(job.quantity), gap), gap),
            _safe_ratio(numerator, denominator),
            _safe_ratio(float(target.arrival_insufficiency), gap),
            (
                _safe_ratio(
                    capacity.capacity_shortfall,
                    capacity.additional_gap,
                )
                if capacity.applicable
                else 0.0
            ),
            float(capacity.applicable),
        )

    row = checkpoint_features + (
        due_date_slack,
        tardiness_exposure,
        proc_time,
        processing_ratio,
        remaining_work_feature,
        critical_ratio,
    )
    return tuple(0.0 if not math.isfinite(value) else float(value) for value in row)


def _reachable_checkpoints(
    metrics: CADecisionMetrics,
    candidate: CandidateMetrics,
):
    return tuple(
        metrics.checkpoint(candidate.entity_id, checkpoint_index)
        for checkpoint_index in candidate.reachable_unmet_checkpoints
    )


def build_extended_theory_features(
    problem: SchedulingProblem,
    state: ScheduleState,
    metrics: CADecisionMetrics,
    *,
    time_scale: float | None = None,
    feature_set: str = "theory11",
    mean_processing_time: float | None = None,
    mean_job_work: float | None = None,
) -> tuple[tuple[float, ...], ...]:
    """构建完整理论特征矩阵。

    Args:
        problem, state, metrics: 当前决策点信息。
        time_scale: 时间归一化因子，None 则自动计算。

    Returns:
        shape = (num_candidates, 11 或 14)。``paper14`` 的第9维可为负。
    """

    feature_names = feature_names_for(feature_set)

    candidates = tuple(sorted(metrics.candidates, key=lambda item: item.job_id))
    if feature_set == "paper14":
        bar_p, bar_w = resolve_paper_feature_scales(
            problem,
            mean_processing_time=mean_processing_time,
            mean_job_work=mean_job_work,
        )
        rows = tuple(
            _paper14_row(
                problem,
                state,
                metrics,
                candidate,
                bar_p=bar_p,
                bar_w=bar_w,
            )
            for candidate in candidates
        )
        if any(len(row) != len(feature_names) for row in rows):
            raise RuntimeError("paper14 row width does not match feature names")
        return rows

    resolved_time_scale, _processing_scale = resolve_observation_scales(
        problem,
        state,
        time_scale=time_scale,
    )
    epsilon = float(problem.r5_epsilon)
    legacy_rows = build_theory_factor_values(
        problem,
        state,
        metrics,
        time_scale=resolved_time_scale,
    )
    if len(legacy_rows) != len(candidates):
        raise ValueError("legacy theory features must align with current candidates")

    mean_operation_time, mean_job_work = _instance_processing_scales(problem)

    rows: list[tuple[float, ...]] = []
    for candidate, legacy in zip(candidates, legacy_rows, strict=True):
        job = problem.get_job(candidate.job_id)
        checkpoints = _reachable_checkpoints(metrics, candidate)

        due_date_slack = _clamp_unit(
            (job.due_date - candidate.optimistic_completion) / resolved_time_scale
        )
        capacity_pressure = _clamp_unit(
            max(
                (
                    checkpoint.optimistic_required_workload
                    / (
                        checkpoint.remaining_machine_processing_capacity
                        + epsilon
                    )
                    for checkpoint in checkpoints
                ),
                default=0.0,
            )
        )
        forecast_flag = float(
            any(checkpoint.arrival_insufficiency > 0.0 for checkpoint in checkpoints)
        )
        capacity_flag = float(
            any(
                checkpoint.optimistic_required_workload
                > checkpoint.remaining_machine_processing_capacity
                for checkpoint in checkpoints
            )
        )
        quantity_contribution = _clamp_unit(
            max(
                (
                    min(float(job.quantity), checkpoint.residual_gap)
                    / (checkpoint.residual_gap + epsilon)
                    for checkpoint in checkpoints
                ),
                default=0.0,
            )
        )
        total_processing = (
            candidate.processing_time + candidate.downstream_min_processing
        )
        processing_ratio = _clamp_unit(
            candidate.processing_time / (total_processing + epsilon)
        )

        base_row = tuple(
            _clamp_unit(value)
            for value in (
                *legacy,
                due_date_slack,
                capacity_pressure,
                forecast_flag,
                capacity_flag,
                quantity_contribution,
                processing_ratio,
            )
        )
        if feature_set == "theory11":
            row = base_row
        else:
            remaining_minimum_work = float(
                candidate.processing_time + candidate.downstream_min_processing
            )
            due_horizon = float(job.due_date - state.current_time)
            critical_ratio_raw = (
                due_horizon / (remaining_minimum_work + epsilon)
                if due_horizon > 0.0
                else 0.0
            )
            row = base_row + (
                _monotone_compress(
                    float(candidate.processing_time) / mean_operation_time
                ),
                _monotone_compress(remaining_minimum_work / mean_job_work),
                _monotone_compress(critical_ratio_raw),
            )
        if len(row) != len(feature_names):  # pragma: no cover - invariant guard.
            raise RuntimeError("feature row width does not match feature_set")
        rows.append(row)

    return tuple(rows)


__all__ = [
    "FEATURE_SETS",
    "GENERAL_DISPATCH_FEATURE_NAMES",
    "PAPER14_FEATURE_NAMES",
    "PAPER_MEAN_JOB_WORK",
    "PAPER_MEAN_PROCESSING_TIME",
    "THEORY_FEATURE_NAMES",
    "build_extended_theory_features",
    "batch_context_for_rows",
    "expression_names_for",
    "feature_names_for",
    "feature_set_for_names",
    "resolve_paper_feature_scales",
]
