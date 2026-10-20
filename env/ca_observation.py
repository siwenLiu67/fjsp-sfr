"""Forecast-aware checkpoint and candidate observations for CA-PPO."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import fmean, pstdev

from fjsp_sfr.core.models import Job
from fjsp_sfr.core.online import SchedulingProblem
from fjsp_sfr.core.state import ScheduleState
from fjsp_sfr.dispatch.ca_metrics import CADecisionMetrics, build_ca_decision_metrics


SHOP_FEATURE_NAMES = (
    "machine_busy_ratio",
    "mean_machine_ready_delay",
    "ready_time_dispersion",
    "dispatchable_job_ratio",
    "mean_remaining_workload",
    "mean_projected_job_slack",
    "job_tardiness_risk_ratio",
)

CHECKPOINT_FEATURE_NAMES = (
    "residual_gap_ratio",
    "time_to_checkpoint",
    "visible_coverage_ratio",
    "future_required_ratio",
    "conservative_forecast_ratio",
    "arrival_insufficiency_ratio",
    "forecast_release_slack",
    "checkpoint_importance",
)

CANDIDATE_FEATURE_NAMES = (
    "operation_processing_ratio",
    "optimistic_remaining_time",
    "projected_due_slack",
    "quantity_ratio",
    "service_value_ratio",
    "reachable_checkpoint_ratio",
)

RELATIONAL_CANDIDATE_FEATURE_NAMES = (
    "current_processing_time",
    "remaining_workload",
    "time_to_due_date",
    "job_quantity",
)

RELATIONAL_CHECKPOINT_FEATURE_NAMES = (
    "residual_gap",
    "time_to_checkpoint",
    "forecast_demand_before_checkpoint",
    "forecast_uncertainty",
    "checkpoint_importance",
)

JOB_CHECKPOINT_RELATION_NAMES = (
    "completion_margin",
    "quantity_contribution",
)

THEORY_FACTOR_NAMES = (
    "gap_pressure",
    "forecast_deficit",
    "checkpoint_slack_urgency",
    "multi_checkpoint_value",
    "tardiness_exposure",
)


ShopVector = tuple[float, float, float, float, float, float, float]
CheckpointToken = tuple[float, float, float, float, float, float, float, float]
CandidateToken = tuple[float, float, float, float, float, float]
RelationalCandidateToken = tuple[float, float, float, float]
RelationalCheckpointToken = tuple[float, float, float, float, float]
JobCheckpointRelation = tuple[float, float]
TheoryFactorToken = tuple[float, float, float, float, float]


def resolve_observation_scales(
    problem: SchedulingProblem,
    state: ScheduleState,
    *,
    time_scale: float | None = None,
    processing_scale: float | None = None,
) -> tuple[float, float]:
    """Return causal scales shared by checkpoint and rule-action tokens."""

    max_time = max(
        checkpoint.time
        for entity in problem.entities
        for checkpoint in entity.checkpoints
    )
    t_scale = float(time_scale or max(1, max_time))
    if t_scale <= 0.0:
        raise ValueError("time_scale must be positive")
    if processing_scale is None:
        released_jobs = tuple(
            job for job in problem.jobs if job.release_time <= state.current_time
        )
        visible_total = sum(
            operation.min_processing_time
            for job in released_jobs
            for operation in job.operations
        )
        p_scale = float(max(1, visible_total / max(1, len(released_jobs))))
    else:
        p_scale = float(processing_scale)
    if p_scale <= 0.0:
        raise ValueError("processing_scale must be positive")
    return t_scale, p_scale


@dataclass(frozen=True)
class CAObservation:
    shop: ShopVector
    checkpoint_tokens: tuple[CheckpointToken, ...]
    checkpoint_keys: tuple[tuple[int, int], ...]
    candidate_tokens: tuple[CandidateToken, ...]
    candidate_job_ids: tuple[int, ...]
    relational_candidate_tokens: tuple[RelationalCandidateToken, ...]
    relational_checkpoint_tokens: tuple[RelationalCheckpointToken, ...]
    relational_checkpoint_keys: tuple[tuple[int, int], ...]
    candidate_checkpoint_relations: tuple[tuple[JobCheckpointRelation, ...], ...]
    candidate_checkpoint_mask: tuple[tuple[bool, ...], ...]
    theory_factor_values: tuple[TheoryFactorToken, ...]

    def __post_init__(self) -> None:
        if len(self.shop) != len(SHOP_FEATURE_NAMES):
            raise ValueError("shop stream must have seven features")
        if len(self.checkpoint_tokens) != len(self.checkpoint_keys):
            raise ValueError("checkpoint tokens and keys must align")
        if any(len(token) != len(CHECKPOINT_FEATURE_NAMES) for token in self.checkpoint_tokens):
            raise ValueError("every checkpoint token must have eight features")
        if len(self.candidate_tokens) != len(self.candidate_job_ids):
            raise ValueError("candidate tokens and job ids must align")
        if any(len(token) != len(CANDIDATE_FEATURE_NAMES) for token in self.candidate_tokens):
            raise ValueError("every candidate token must have six features")
        if self.candidate_job_ids != tuple(sorted(self.candidate_job_ids)):
            raise ValueError("candidate jobs must be sorted by job_id")
        if len(self.relational_candidate_tokens) != len(self.candidate_job_ids):
            raise ValueError("relational candidates and candidate job ids must align")
        if any(
            len(token) != len(RELATIONAL_CANDIDATE_FEATURE_NAMES)
            for token in self.relational_candidate_tokens
        ):
            raise ValueError("every relational candidate token must have four features")
        if len(self.relational_checkpoint_tokens) != len(
            self.relational_checkpoint_keys
        ):
            raise ValueError("relational checkpoint tokens and keys must align")
        if any(
            len(token) != len(RELATIONAL_CHECKPOINT_FEATURE_NAMES)
            for token in self.relational_checkpoint_tokens
        ):
            raise ValueError("every relational checkpoint token must have five features")
        checkpoint_count = len(self.relational_checkpoint_tokens)
        if len(self.candidate_checkpoint_relations) != len(self.candidate_job_ids):
            raise ValueError("one relation row is required per candidate")
        if len(self.candidate_checkpoint_mask) != len(self.candidate_job_ids):
            raise ValueError("one checkpoint mask row is required per candidate")
        if any(
            len(row) != checkpoint_count
            or any(
                len(relation) != len(JOB_CHECKPOINT_RELATION_NAMES)
                for relation in row
            )
            for row in self.candidate_checkpoint_relations
        ):
            raise ValueError(
                "candidate-checkpoint relations must have shape [jobs, checkpoints, 2]"
            )
        if any(len(row) != checkpoint_count for row in self.candidate_checkpoint_mask):
            raise ValueError("candidate-checkpoint mask must align with relation rows")
        if len(self.theory_factor_values) != len(self.candidate_job_ids):
            raise ValueError("one theory-factor row is required per candidate")
        if any(
            len(row) != len(THEORY_FACTOR_NAMES)
            or any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in row)
            for row in self.theory_factor_values
        ):
            raise ValueError("theory factors must be finite values in [0, 1]")


def remaining_workload(job: Job, state: ScheduleState) -> float:
    ongoing = next(
        (
            scheduled
            for (job_id, _operation_id), scheduled in state.ongoing_operations.items()
            if job_id == job.job_id
        ),
        None,
    )
    if ongoing is None:
        start_index = state.next_op_index_for_job(job.job_id)
        residual = 0.0
    else:
        operation = next(item for item in job.operations if item.op_id == ongoing.op_id)
        start_index = operation.sequence_index + 1
        residual = float(max(0, ongoing.end_time - state.current_time))
    return residual + sum(
        operation.min_processing_time for operation in job.operations[start_index:]
    )


def build_candidate_tokens(
    problem: SchedulingProblem,
    state: ScheduleState,
    metrics: CADecisionMetrics,
    *,
    time_scale: float | None = None,
    processing_scale: float | None = None,
) -> tuple[tuple[CandidateToken, ...], tuple[int, ...]]:
    """Encode released jobs executable on the current trigger machine."""

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

    tokens: list[CandidateToken] = []
    job_ids: list[int] = []
    for candidate in sorted(metrics.candidates, key=lambda item: item.job_id):
        job = problem.get_job(candidate.job_id)
        if job.release_time > state.current_time:
            raise ValueError("candidate tokens may only contain released jobs")
        reachable_denominator = max(1, unmet_by_entity[job.entity_id])
        tokens.append(
            (
                candidate.processing_time / p_scale,
                (candidate.optimistic_completion - state.current_time) / t_scale,
                (job.due_date - candidate.optimistic_completion) / t_scale,
                float(job.quantity) / quantity_scale,
                candidate.service_value / service_scale,
                len(candidate.reachable_unmet_checkpoints) / reachable_denominator,
            )
        )
        job_ids.append(job.job_id)
    return tuple(tokens), tuple(job_ids)


def build_relational_candidate_checkpoint_tokens(
    problem: SchedulingProblem,
    state: ScheduleState,
    metrics: CADecisionMetrics,
    *,
    time_scale: float | None = None,
    processing_scale: float | None = None,
) -> tuple[
    tuple[RelationalCandidateToken, ...],
    tuple[RelationalCheckpointToken, ...],
    tuple[tuple[int, int], ...],
    tuple[tuple[JobCheckpointRelation, ...], ...],
    tuple[tuple[bool, ...], ...],
]:
    """Build the minimal candidate/checkpoint/relation streams for the encoder."""

    t_scale, p_scale = resolve_observation_scales(
        problem,
        state,
        time_scale=time_scale,
        processing_scale=processing_scale,
    )
    quantity_scale = max(
        float(checkpoint.required_quantity)
        for entity in problem.entities
        for checkpoint in entity.checkpoints
    )
    max_weight = max(
        float(checkpoint.weight)
        for entity in problem.entities
        for checkpoint in entity.checkpoints
    )
    quantity_scale = max(quantity_scale, 1e-12)
    max_weight = max(max_weight, 1e-12)

    checkpoints = tuple(sorted(metrics.checkpoints.items()))
    checkpoint_tokens: list[RelationalCheckpointToken] = []
    checkpoint_keys: list[tuple[int, int]] = []
    for key, checkpoint in checkpoints:
        required = max(float(checkpoint.required_quantity), 1e-12)
        checkpoint_tokens.append(
            (
                checkpoint.residual_gap / required,
                (checkpoint.checkpoint_time - state.current_time) / t_scale,
                checkpoint.forecast_demand / required,
                checkpoint.forecast_uncertainty / required,
                checkpoint.checkpoint_weight / max_weight,
            )
        )
        checkpoint_keys.append(key)

    candidate_tokens: list[RelationalCandidateToken] = []
    relation_rows: list[tuple[JobCheckpointRelation, ...]] = []
    mask_rows: list[tuple[bool, ...]] = []
    for candidate in sorted(metrics.candidates, key=lambda item: item.job_id):
        job = problem.get_job(candidate.job_id)
        if job.release_time > state.current_time:
            raise ValueError("relational candidates may only contain released jobs")
        candidate_tokens.append(
            (
                candidate.processing_time / p_scale,
                (candidate.optimistic_completion - state.current_time) / p_scale,
                (job.due_date - state.current_time) / t_scale,
                float(job.quantity) / quantity_scale,
            )
        )
        relations: list[JobCheckpointRelation] = []
        masks: list[bool] = []
        for (_key, checkpoint) in checkpoints:
            relations.append(
                (
                    math.tanh(
                        (checkpoint.checkpoint_time - candidate.optimistic_completion)
                        / t_scale
                    ),
                    float(job.quantity)
                    / (float(checkpoint.required_quantity) + 1e-12),
                )
            )
            masks.append(
                checkpoint.entity_id == candidate.entity_id
                and checkpoint.checkpoint_time > state.current_time
            )
        relation_rows.append(tuple(relations))
        mask_rows.append(tuple(masks))

    return (
        tuple(candidate_tokens),
        tuple(checkpoint_tokens),
        tuple(checkpoint_keys),
        tuple(relation_rows),
        tuple(mask_rows),
    )


def build_theory_factor_values(
    problem: SchedulingProblem,
    state: ScheduleState,
    metrics: CADecisionMetrics,
    *,
    time_scale: float | None = None,
) -> tuple[TheoryFactorToken, ...]:
    """Build ``[f_G, f_F, f_U, f_M, f_T]`` for every current candidate."""

    horizon, _processing_scale = resolve_observation_scales(
        problem, state, time_scale=time_scale
    )
    epsilon = max(float(problem.r5_epsilon), 1e-12)
    visible_quantity = {
        entity.entity_id: float(
            sum(
                job.quantity
                for job in problem.jobs_of_entity(entity.entity_id)
                if job.release_time <= state.current_time
                and not state.is_job_completed(job.job_id)
            )
        )
        for entity in problem.entities
    }

    factor_rows: list[TheoryFactorToken] = []
    for candidate in sorted(metrics.candidates, key=lambda item: item.job_id):
        job = problem.get_job(candidate.job_id)
        future = tuple(
            checkpoint
            for (_key, checkpoint) in sorted(metrics.checkpoints.items())
            if checkpoint.entity_id == candidate.entity_id
            and checkpoint.checkpoint_time > state.current_time
        )
        if future:
            next_checkpoint = future[0]
            gap = next_checkpoint.residual_gap
            required_workload = next_checkpoint.optimistic_required_workload
            machine_capacity = (
                next_checkpoint.remaining_machine_processing_capacity
            )
            gap_pressure = required_workload / (
                required_workload + machine_capacity + epsilon
            )
            forecast_deficit = (
                max(
                    0.0,
                    gap
                    - visible_quantity[candidate.entity_id]
                    - next_checkpoint.forecast_demand,
                )
                / (gap + epsilon)
                if gap > 0.0
                else 0.0
            )
            reachable = tuple(
                checkpoint
                for checkpoint in future
                if candidate.optimistic_completion <= checkpoint.checkpoint_time
            )
            if reachable:
                earliest = reachable[0]
                urgency = (
                    (candidate.optimistic_completion - state.current_time)
                    / (earliest.checkpoint_time - state.current_time + epsilon)
                )
            else:
                urgency = 0.0
            total_weight = sum(checkpoint.checkpoint_weight for checkpoint in future)
            multi_checkpoint_value = sum(
                checkpoint.checkpoint_weight
                * min(float(job.quantity), checkpoint.residual_gap)
                / (checkpoint.residual_gap + epsilon)
                for checkpoint in reachable
            ) / (total_weight + epsilon)
        else:
            gap_pressure = 0.0
            forecast_deficit = 0.0
            urgency = 0.0
            multi_checkpoint_value = 0.0

        projected_tardiness = max(
            0.0, float(candidate.optimistic_completion - job.due_date)
        )
        tardiness_exposure = math.tanh(projected_tardiness / horizon)
        factor_rows.append(
            tuple(
                min(1.0, max(0.0, value))
                for value in (
                    gap_pressure,
                    forecast_deficit,
                    urgency,
                    multi_checkpoint_value,
                    tardiness_exposure,
                )
            )
        )
    return tuple(factor_rows)


def build_ca_observation(
    problem: SchedulingProblem,
    state: ScheduleState,
    *,
    metrics: CADecisionMetrics | None = None,
    time_scale: float | None = None,
    processing_scale: float | None = None,
    coverage_cap: float = 2.0,
) -> CAObservation:
    if coverage_cap <= 0.0:
        raise ValueError("coverage_cap must be positive")
    current_metrics = metrics or build_ca_decision_metrics(problem, state)
    # Checkpoint times are exogenous and known from the start.  Job due dates and
    # processing data may only contribute after their job has been released.
    # In particular, never let a hidden future job alter an observation scale.
    t_scale, p_scale = resolve_observation_scales(
        problem,
        state,
        time_scale=time_scale,
        processing_scale=processing_scale,
    )

    machines = tuple(problem.machines)
    delays = tuple(
        max(0.0, state.machine_available_times.get(machine.machine_id, 0) - state.current_time)
        / t_scale
        for machine in machines
    )
    busy_ratio = sum(delay > 0.0 for delay in delays) / len(machines)
    unfinished = tuple(
        job
        for job in problem.jobs
        if job.release_time <= state.current_time and not state.is_job_completed(job.job_id)
    )
    dispatchable = 0
    workloads: list[float] = []
    slacks: list[float] = []
    for job in unfinished:
        next_index = state.next_op_index_for_job(job.job_id)
        if next_index < job.num_operations:
            operation = job.operation_at(next_index)
            if not state.is_operation_ongoing(job.job_id, operation.op_id) and any(
                state.is_machine_idle(machine_id) for machine_id in operation.eligible_machines
            ):
                dispatchable += 1
        workload = remaining_workload(job, state)
        workloads.append(workload)
        slacks.append(job.due_date - state.current_time - workload)

    shop: ShopVector = (
        busy_ratio,
        fmean(delays) if delays else 0.0,
        pstdev(delays) if len(delays) > 1 else 0.0,
        dispatchable / max(1, len(unfinished)),
        fmean(workloads) / p_scale if workloads else 0.0,
        fmean(math.tanh(slack / t_scale) for slack in slacks) if slacks else 0.0,
        sum(slack < 0.0 for slack in slacks) / len(slacks) if slacks else 0.0,
    )

    max_weight = max(
        checkpoint.weight for entity in problem.entities for checkpoint in entity.checkpoints
    )
    tokens: list[CheckpointToken] = []
    keys: list[tuple[int, int]] = []
    for key, checkpoint in sorted(current_metrics.checkpoints.items()):
        if checkpoint.residual_gap <= 0.0 or checkpoint.checkpoint_time < state.current_time:
            continue
        required = checkpoint.required_quantity
        gap_denominator = max(checkpoint.residual_gap, 1e-12)
        tokens.append(
            (
                checkpoint.residual_gap / required,
                (checkpoint.checkpoint_time - state.current_time) / t_scale,
                min(checkpoint.visible_coverage / gap_denominator, coverage_cap),
                checkpoint.future_required / required,
                checkpoint.conservative_forecast / required,
                checkpoint.arrival_insufficiency / required,
                (
                    checkpoint.release_slack / t_scale
                    if checkpoint.release_slack is not None
                    else -1.0
                ),
                checkpoint.checkpoint_weight / max_weight,
            )
        )
        keys.append(key)
    candidate_tokens, candidate_job_ids = build_candidate_tokens(
        problem,
        state,
        current_metrics,
        time_scale=t_scale,
        processing_scale=p_scale,
    )
    (
        relational_candidate_tokens,
        relational_checkpoint_tokens,
        relational_checkpoint_keys,
        candidate_checkpoint_relations,
        candidate_checkpoint_mask,
    ) = build_relational_candidate_checkpoint_tokens(
        problem,
        state,
        current_metrics,
        time_scale=t_scale,
        processing_scale=p_scale,
    )
    theory_factor_values = build_theory_factor_values(
        problem,
        state,
        current_metrics,
        time_scale=t_scale,
    )
    return CAObservation(
        shop,
        tuple(tokens),
        tuple(keys),
        candidate_tokens,
        candidate_job_ids,
        relational_candidate_tokens,
        relational_checkpoint_tokens,
        relational_checkpoint_keys,
        candidate_checkpoint_relations,
        candidate_checkpoint_mask,
        theory_factor_values,
    )
