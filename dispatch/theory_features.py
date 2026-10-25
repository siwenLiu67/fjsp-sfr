"""论文 Table 5-2 理论特征与可选的通用调度特征。"""

from __future__ import annotations

from fjsp_sfr.core.online import SchedulingProblem
from fjsp_sfr.core.state import ScheduleState
from fjsp_sfr.dispatch.ca_metrics import CADecisionMetrics, CandidateMetrics
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

GENERAL_DISPATCH_FEATURE_NAMES: tuple[str, ...] = (
    "proc_time",
    "remaining_work",
    "critical_ratio",
)

FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "theory11": THEORY_FEATURE_NAMES,
    "theory11_plus3": THEORY_FEATURE_NAMES + GENERAL_DISPATCH_FEATURE_NAMES,
}

BATCH_CONTEXT_NAMES: dict[str, tuple[str, ...]] = {
    "theory11": (),
    "theory11_plus3": ("mean_remaining_work",),
}


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

    if feature_set == "theory11" or not rows:
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


def _instance_processing_scales(problem: SchedulingProblem) -> tuple[float, float]:
    """Return visible-instance mean alternative time and mean job minimum work."""

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
) -> tuple[tuple[float, ...], ...]:
    """构建完整理论特征矩阵。

    Args:
        problem, state, metrics: 当前决策点信息。
        time_scale: 时间归一化因子，None 则自动计算。

    Returns:
        shape = (num_candidates, 11 或 14)，每个值在 [0, 1]。
    """

    feature_names = feature_names_for(feature_set)

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
    candidates = tuple(sorted(metrics.candidates, key=lambda item: item.job_id))
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
    "THEORY_FEATURE_NAMES",
    "build_extended_theory_features",
    "batch_context_for_rows",
    "expression_names_for",
    "feature_names_for",
    "feature_set_for_names",
]
