"""论文 Table 5-2 完整理论特征。所有值归一化到 [0, 1]。"""

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


def _clamp_unit(value: float) -> float:
    return max(0.0, min(float(value), 1.0))


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
) -> tuple[tuple[float, ...], ...]:
    """构建完整理论特征矩阵。

    Args:
        problem, state, metrics: 当前决策点信息。
        time_scale: 时间归一化因子，None 则自动计算。

    Returns:
        shape = (num_candidates, 11)，每个值在 [0, 1]。
    """

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

        row = tuple(
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
        rows.append(row)

    return tuple(rows)
