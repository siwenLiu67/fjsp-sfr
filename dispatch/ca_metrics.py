"""Paper-A scheduling quantities for a machine-triggered decision."""

from __future__ import annotations

from dataclasses import dataclass

from fjsp_sfr.core.forecast import ForecastSnapshot, empty_forecast_snapshot
from fjsp_sfr.core.models import FJSPInstance, Job
from fjsp_sfr.core.online import OnlineProblemView, SchedulingProblem
from fjsp_sfr.core.state import ScheduleState


@dataclass(frozen=True)
class CandidateMetrics:
    job_id: int
    entity_id: int
    operation_id: int
    machine_id: int
    processing_time: int
    downstream_min_processing: int
    optimistic_completion: int
    service_value: float
    reachable_unmet_checkpoints: tuple[int, ...]


@dataclass(frozen=True)
class CheckpointMetrics:
    entity_id: int
    checkpoint_index: int
    checkpoint_time: int
    checkpoint_weight: float
    required_quantity: float
    residual_gap: float
    visible_coverage: float
    optimistic_required_workload: float
    remaining_machine_processing_capacity: float
    future_required: float
    forecast_demand: float
    forecast_uncertainty: float
    conservative_forecast: float
    arrival_insufficiency: float
    release_threshold_time: float | None
    release_slack: float | None

@dataclass(frozen=True)
class CADecisionMetrics:
    decision_time: int
    machine_id: int | None
    candidates: tuple[CandidateMetrics, ...]
    checkpoints: dict[tuple[int, int], CheckpointMetrics]
    forecast_update_epoch: int

    @property
    def is_active(self) -> bool:
        return self.machine_id is not None and bool(self.candidates)

    def checkpoint(self, entity_id: int, checkpoint_index: int) -> CheckpointMetrics:
        return self.checkpoints[(entity_id, checkpoint_index)]

    def gap(self, entity_id: int, checkpoint_index: int) -> float:
        return self.checkpoint(entity_id, checkpoint_index).residual_gap

    def visible_coverage(self, entity_id: int, checkpoint_index: int) -> float:
        return self.checkpoint(entity_id, checkpoint_index).visible_coverage

    def remaining_time(self, candidate: CandidateMetrics) -> int:
        return candidate.optimistic_completion - self.decision_time


def residual_checkpoint_gap(
    problem: SchedulingProblem,
    state: ScheduleState,
    entity_id: int,
    checkpoint_index: int,
) -> float:
    entity = problem.get_entity(entity_id)
    completed = sum(
        job.quantity
        for job in problem.jobs_of_entity(entity_id)
        if state.completed_jobs.get(job.job_id, state.current_time + 1)
        <= state.current_time
    )
    return max(0.0, entity.required_quantity(checkpoint_index) - completed)


def candidate_jobs(
    problem: SchedulingProblem,
    state: ScheduleState,
    machine_id: int,
) -> tuple[Job, ...]:
    """Return J_u(tau): released jobs immediately executable on machine u."""

    if not state.is_machine_idle(machine_id):
        return ()
    candidates: list[Job] = []
    for job in sorted(problem.jobs, key=lambda item: item.job_id):
        if job.release_time > state.current_time or state.is_job_completed(job.job_id):
            continue
        next_index = state.next_op_index_for_job(job.job_id)
        if next_index >= job.num_operations:
            continue
        operation = job.operation_at(next_index)
        if state.is_operation_ongoing(job.job_id, operation.op_id):
            continue
        if machine_id in operation.eligible_machines:
            candidates.append(job)
    return tuple(candidates)


def next_trigger_machine(
    problem: SchedulingProblem,
    state: ScheduleState,
) -> int | None:
    """Choose the next idle machine in a fixed, deterministic id order."""

    for machine in sorted(problem.machines, key=lambda item: item.machine_id):
        if candidate_jobs(problem, state, machine.machine_id):
            return machine.machine_id
    return None


def optimistic_candidate_completion(job: Job, state: ScheduleState, machine_id: int) -> int:
    """C_lower_ju(tau), fixing the next operation on the trigger machine."""

    next_index = state.next_op_index_for_job(job.job_id)
    operation = job.operation_at(next_index)
    return (
        state.current_time
        + operation.processing_time_on(machine_id)
        + sum(item.min_processing_time for item in job.operations[next_index + 1 :])
    )


def _ongoing_end(state: ScheduleState, job: Job) -> tuple[int, int] | None:
    for (job_id, _operation_id), scheduled in state.ongoing_operations.items():
        if job_id == job.job_id:
            operation = next(
                item for item in job.operations if item.op_id == scheduled.op_id
            )
            return scheduled.end_time, operation.sequence_index + 1
    return None


def compute_optimistic_checkpoint_workload_capacity(
    problem: SchedulingProblem,
    state: ScheduleState,
    entity_id: int,
    checkpoint_index: int,
    *,
    residual_gap: float | None = None,
) -> tuple[float, float]:
    """Return optimistic required workload and available related-machine time.

    The quantity gap is converted with the smallest visible unfinished
    workload-per-quantity ratio for the entity.  This is an optimistic lower
    bound on required processing.  Capacity is the sum of free machine-time
    remaining before the checkpoint over machines eligible for those jobs'
    unscheduled remaining operations; it is an optimistic upper bound because
    it ignores contention with other entities and precedence interactions.
    """

    checkpoint = problem.get_entity(entity_id).checkpoints[checkpoint_index]
    visible_jobs = tuple(
        job
        for job in problem.jobs_of_entity(entity_id)
        if job.release_time <= state.current_time
        and not state.is_job_completed(job.job_id)
    )
    workload_ratios: list[float] = []
    relevant_machines: set[int] = set()
    for job in visible_jobs:
        ongoing = _ongoing_end(state, job)
        start_index = (
            ongoing[1]
            if ongoing is not None
            else state.next_op_index_for_job(job.job_id)
        )
        remaining_operations = job.operations[start_index:]
        workload = sum(
            operation.min_processing_time for operation in remaining_operations
        )
        if workload > 0:
            workload_ratios.append(float(workload) / float(job.quantity))
        relevant_machines.update(
            machine_id
            for operation in remaining_operations
            for machine_id in operation.eligible_machines
        )

    gap = (
        residual_checkpoint_gap(
            problem, state, entity_id, checkpoint_index
        )
        if residual_gap is None
        else max(0.0, float(residual_gap))
    )
    required_workload = (
        gap * min(workload_ratios) if workload_ratios else 0.0
    )
    available_capacity = float(
        sum(
            max(
                0,
                checkpoint.time
                - max(
                    state.current_time,
                    state.machine_available_times.get(machine_id, 0),
                ),
            )
            for machine_id in relevant_machines
        )
    )
    return required_workload, available_capacity


def visible_optimistic_completion(job: Job, state: ScheduleState) -> int:
    """Best optimistic completion C*_j(tau) for an already released job.

    Keeping this precondition here prevents future-job release times and
    processing attributes from entering any caller that computes visible
    coverage or agent-facing features.
    """

    if job.release_time > state.current_time:
        raise ValueError("visible optimistic completion requires a released job")

    ongoing = _ongoing_end(state, job)
    if ongoing is None:
        ready = max(state.current_time, job.release_time)
        start_index = state.next_op_index_for_job(job.job_id)
    else:
        ready, start_index = ongoing
    for operation in job.operations[start_index:]:
        ready = min(
            max(ready, state.machine_available_times.get(alt.machine_id, 0))
            + alt.processing_time
            for alt in operation.alternatives
        )
    return ready


def multi_checkpoint_service_value(
    problem: SchedulingProblem,
    state: ScheduleState,
    job: Job,
    optimistic_completion: int,
    gaps: dict[tuple[int, int], float] | None = None,
) -> tuple[float, tuple[int, ...]]:
    entity = problem.get_entity(job.entity_id)
    gap_lookup = gaps or {
        (entity.entity_id, index): residual_checkpoint_gap(
            problem, state, entity.entity_id, index
        )
        for index in range(len(entity.checkpoints))
    }
    value = 0.0
    reachable: list[int] = []
    for index, checkpoint in enumerate(entity.checkpoints):
        gap = gap_lookup[(entity.entity_id, index)]
        if checkpoint.time < state.current_time or gap <= 0.0:
            continue
        if optimistic_completion <= checkpoint.time:
            reachable.append(index)
            value += checkpoint.weight * min(float(job.quantity), gap)
    return value, tuple(reachable)


def compute_optimistic_visible_coverage(
    problem: SchedulingProblem,
    state: ScheduleState,
    entity_id: int,
    checkpoint_index: int,
) -> float:
    checkpoint = problem.get_entity(entity_id).checkpoints[checkpoint_index]
    return float(
        sum(
            job.quantity
            for job in problem.jobs_of_entity(entity_id)
            if job.release_time <= state.current_time
            and not state.is_job_completed(job.job_id)
            and visible_optimistic_completion(job, state) <= checkpoint.time
        )
    )


def compute_future_required_quantity(residual_gap: float, visible_coverage: float) -> float:
    return max(0.0, residual_gap - visible_coverage)


def _overlap_fraction(bucket_start: int, bucket_end: int, tau: int, checkpoint: int) -> float:
    overlap = max(0, min(bucket_end, checkpoint) - max(bucket_start, tau))
    return overlap / float(bucket_end - bucket_start)


def compute_conservative_arrival_forecast(
    snapshot: ForecastSnapshot,
    entity_id: int,
    checkpoint_time: int,
    tau: int,
    kappa: float,
) -> float:
    """Return F^- over ``(tau, checkpoint_time]``.

    Each bucket stores forecast *quantity for the full calendar bucket* in the
    same unit as ``Job.quantity`` and checkpoint targets.  Partial overlap is
    converted to a remaining quantity by a uniform-within-bucket interpolation.
    Providers that already return a remaining-bucket quantity must not use this
    representation, because applying the overlap fraction again would undercount.
    """

    return compute_arrival_forecast_summary(
        snapshot,
        entity_id,
        checkpoint_time,
        tau,
        kappa,
    )[2]


def compute_arrival_forecast_summary(
    snapshot: ForecastSnapshot,
    entity_id: int,
    checkpoint_time: int,
    tau: int,
    kappa: float,
) -> tuple[float, float, float]:
    """Return forecast mean, uncertainty, and conservative demand before a checkpoint.

    Mean and uncertainty use the same partial-bucket interpolation as the
    existing conservative forecast.  The uncertainty is the sum of the
    overlap-weighted bucket sigmas, retaining the physical quantity unit used
    by checkpoint demand and residual gaps.
    """

    if kappa < 0.0:
        raise ValueError("forecast risk parameter kappa must be non-negative")
    demand = 0.0
    uncertainty = 0.0
    conservative = 0.0
    for bucket in snapshot.for_entity(entity_id):
        fraction = _overlap_fraction(
            bucket.start, bucket.end, tau, checkpoint_time
        )
        demand += fraction * bucket.d_hat
        uncertainty += fraction * bucket.sigma
        conservative += fraction * bucket.conservative_quantity(kappa)
    return demand, uncertainty, conservative


def compute_arrival_insufficiency(
    future_required: float,
    conservative_forecast: float,
) -> float:
    return max(0.0, future_required - conservative_forecast)


def compute_release_threshold_time(
    snapshot: ForecastSnapshot,
    entity_id: int,
    checkpoint_time: int,
    tau: int,
    kappa: float,
    future_required: float,
) -> float | None:
    if future_required <= 0.0:
        return None
    cumulative = 0.0
    for bucket in snapshot.for_entity(entity_id):
        fraction = _overlap_fraction(bucket.start, bucket.end, tau, checkpoint_time)
        if fraction <= 0.0:
            continue
        cumulative += fraction * bucket.conservative_quantity(kappa)
        if cumulative + 1e-12 >= future_required:
            return float(min(bucket.end, checkpoint_time))
    return None


def compute_release_slack(
    checkpoint_time: int,
    release_threshold_time: float | None,
) -> float | None:
    if release_threshold_time is None:
        return None
    return max(0.0, float(checkpoint_time) - release_threshold_time)


def _snapshot(problem: SchedulingProblem, tau: int) -> ForecastSnapshot:
    if isinstance(problem, OnlineProblemView):
        return problem.forecast_snapshot
    if isinstance(problem, FJSPInstance) and problem.forecast_provider is not None:
        return problem.forecast_provider.snapshot_at(tau)
    return empty_forecast_snapshot(
        (entity.entity_id for entity in problem.entities), update_epoch=0
    )


def build_ca_decision_metrics(
    problem: SchedulingProblem,
    state: ScheduleState,
    *,
    machine_id: int | None = None,
) -> CADecisionMetrics:
    trigger = next_trigger_machine(problem, state) if machine_id is None else machine_id
    if trigger is not None and not candidate_jobs(problem, state, trigger):
        raise ValueError("trigger machine must be idle with at least one candidate")

    gaps = {
        (entity.entity_id, index): residual_checkpoint_gap(
            problem, state, entity.entity_id, index
        )
        for entity in problem.entities
        for index in range(len(entity.checkpoints))
    }
    snapshot = _snapshot(problem, state.current_time)
    kappa = problem.forecast_risk_kappa
    checkpoints: dict[tuple[int, int], CheckpointMetrics] = {}
    for entity in problem.entities:
        for index, checkpoint in enumerate(entity.checkpoints):
            gap = gaps[(entity.entity_id, index)]
            required_workload, machine_capacity = (
                compute_optimistic_checkpoint_workload_capacity(
                    problem,
                    state,
                    entity.entity_id,
                    index,
                    residual_gap=gap,
                )
            )
            visible = compute_optimistic_visible_coverage(
                problem, state, entity.entity_id, index
            )
            future_required = compute_future_required_quantity(gap, visible)
            forecast_demand, forecast_uncertainty, forecast = (
                compute_arrival_forecast_summary(
                    snapshot,
                    entity.entity_id,
                    checkpoint.time,
                    state.current_time,
                    kappa,
                )
            )
            insufficiency = compute_arrival_insufficiency(future_required, forecast)
            theta = compute_release_threshold_time(
                snapshot,
                entity.entity_id,
                checkpoint.time,
                state.current_time,
                kappa,
                future_required,
            )
            checkpoints[(entity.entity_id, index)] = CheckpointMetrics(
                entity_id=entity.entity_id,
                checkpoint_index=index,
                checkpoint_time=checkpoint.time,
                checkpoint_weight=checkpoint.weight,
                required_quantity=checkpoint.required_quantity,
                residual_gap=gap,
                visible_coverage=visible,
                optimistic_required_workload=required_workload,
                remaining_machine_processing_capacity=machine_capacity,
                future_required=future_required,
                forecast_demand=forecast_demand,
                forecast_uncertainty=forecast_uncertainty,
                conservative_forecast=forecast,
                arrival_insufficiency=insufficiency,
                release_threshold_time=theta,
                release_slack=compute_release_slack(checkpoint.time, theta),
            )

    metrics: list[CandidateMetrics] = []
    if trigger is not None:
        for job in candidate_jobs(problem, state, trigger):
            next_index = state.next_op_index_for_job(job.job_id)
            operation = job.operation_at(next_index)
            completion = optimistic_candidate_completion(job, state, trigger)
            value, reachable = multi_checkpoint_service_value(
                problem, state, job, completion, gaps
            )
            metrics.append(
                CandidateMetrics(
                    job_id=job.job_id,
                    entity_id=job.entity_id,
                    operation_id=operation.op_id,
                    machine_id=trigger,
                    processing_time=operation.processing_time_on(trigger),
                    downstream_min_processing=sum(
                        item.min_processing_time for item in job.operations[next_index + 1 :]
                    ),
                    optimistic_completion=completion,
                    service_value=value,
                    reachable_unmet_checkpoints=reachable,
                )
            )
    return CADecisionMetrics(
        decision_time=state.current_time,
        machine_id=trigger,
        candidates=tuple(metrics),
        checkpoints=checkpoints,
        forecast_update_epoch=snapshot.update_epoch,
    )
