"""Paper-A scheduling quantities for a machine-triggered decision."""

from __future__ import annotations

import math
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
    confirmed_supply: float = 0.0
    additional_gap: float = 0.0
    timely_visible_quantity: float = 0.0
    additional_candidate_quantity: float = 0.0
    minimum_work_per_quantity: float | None = None
    capacity_applicable: bool = False
    capacity_shortfall: float = 0.0
    timely_job_ids: tuple[int, ...] = ()
    additional_job_ids: tuple[int, ...] = ()
    eligible_machine_ids: tuple[int, ...] = ()
    # Public Sec. 4.6 trajectory-intersected certificate endpoints.  They are
    # populated by the environment after same-time events and intentionally do
    # not contain any privileged future summary.
    online_shortfall_lower: float | None = None
    online_shortfall_upper: float | None = None


@dataclass(frozen=True)
class CheckpointCapacityAnalysis:
    """Exact Sec. 4.3 quantities for one visible checkpoint state."""

    entity_id: int
    checkpoint_index: int
    residual_gap: float
    timely_jobs: tuple[Job, ...]
    confirmed_jobs: tuple[Job, ...]
    additional_jobs: tuple[Job, ...]
    confirmed_supply: float
    additional_gap: float
    timely_quantity: float
    additional_quantity: float
    minimum_work_per_quantity: float | None
    eligible_machine_ids: tuple[int, ...]
    available_capacity: float
    required_workload: float
    applicable: bool
    capacity_shortfall: float


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


def select_target_checkpoint(
    metrics: CADecisionMetrics,
    candidate: CandidateMetrics,
) -> CheckpointMetrics | None:
    """Select the single target checkpoint defined in Sec. 4.5.

    The earliest positive-gap unsettled checkpoint reachable by the job is
    preferred.  If none is reachable, the earliest positive-gap unsettled
    checkpoint of the same entity is returned.  ``None`` denotes no target.
    """

    eligible = tuple(
        checkpoint
        for (_key, checkpoint) in sorted(
            metrics.checkpoints.items(),
            key=lambda item: (
                item[1].checkpoint_time,
                item[1].checkpoint_index,
            ),
        )
        if checkpoint.entity_id == candidate.entity_id
        and checkpoint.checkpoint_time > metrics.decision_time
        and checkpoint.residual_gap > 0.0
    )
    if not eligible:
        return None
    reachable = set(candidate.reachable_unmet_checkpoints)
    return next(
        (
            checkpoint
            for checkpoint in eligible
            if checkpoint.checkpoint_index in reachable
        ),
        eligible[0],
    )


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


def _ongoing_operation(state: ScheduleState, job: Job):
    matches = tuple(
        scheduled
        for (job_id, _operation_id), scheduled in state.ongoing_operations.items()
        if job_id == job.job_id
    )
    if len(matches) > 1:
        raise RuntimeError(f"job {job.job_id} has multiple ongoing operations")
    if not matches:
        return None
    scheduled = matches[0]
    operation = next(
        item for item in job.operations if item.op_id == scheduled.op_id
    )
    return scheduled, operation


def optimistic_job_completion_lower_bound(job: Job, state: ScheduleState) -> int:
    """Return Eq. (4.2) using only the legal visible state.

    ``C_LB = t + p_run + sum(p_min for every unstarted operation)``.
    Machine queues and a hypothetical trigger-machine assignment never enter
    this value.
    """

    if job.release_time > state.current_time:
        raise ValueError("completion lower bound requires a released job")
    if state.is_job_completed(job.job_id):
        return int(state.completed_jobs[job.job_id])
    ongoing = _ongoing_operation(state, job)
    if ongoing is None:
        first_unstarted = state.next_op_index_for_job(job.job_id)
        running_remaining = 0
    else:
        scheduled, operation = ongoing
        first_unstarted = operation.sequence_index + 1
        running_remaining = max(0, scheduled.end_time - state.current_time)
    return int(
        state.current_time
        + running_remaining
        + sum(
            operation.min_processing_time
            for operation in job.operations[first_unstarted:]
        )
    )


def optimistic_candidate_completion(
    job: Job, state: ScheduleState, machine_id: int | None = None
) -> int:
    """Compatibility alias for the trigger-machine-invariant Eq. (4.2)."""

    del machine_id
    return optimistic_job_completion_lower_bound(job, state)


def timely_visible_jobs(
    problem: SchedulingProblem,
    state: ScheduleState,
    entity_id: int,
    checkpoint_index: int,
) -> tuple[Job, ...]:
    """Return Omega_rkt from released, unfinished jobs only."""

    checkpoint = problem.get_entity(entity_id).checkpoints[checkpoint_index]
    if state.current_time >= checkpoint.time:
        return ()
    return tuple(
        job
        for job in sorted(problem.jobs_of_entity(entity_id), key=lambda item: item.job_id)
        if job.release_time <= state.current_time
        and not state.is_job_completed(job.job_id)
        and optimistic_job_completion_lower_bound(job, state) <= checkpoint.time
    )


def _unstarted_operations(job: Job, state: ScheduleState):
    ongoing = _ongoing_operation(state, job)
    first_unstarted = (
        state.next_op_index_for_job(job.job_id)
        if ongoing is None
        else ongoing[1].sequence_index + 1
    )
    return job.operations[first_unstarted:]


def analyze_checkpoint_capacity(
    problem: SchedulingProblem,
    state: ScheduleState,
    entity_id: int,
    checkpoint_index: int,
    *,
    residual_gap: float | None = None,
) -> CheckpointCapacityAnalysis:
    """Compute A, G+, Omega+, nu, Cap and Delta_cap from Sec. 4.3."""

    checkpoint = problem.get_entity(entity_id).checkpoints[checkpoint_index]
    gap = (
        residual_checkpoint_gap(problem, state, entity_id, checkpoint_index)
        if residual_gap is None
        else max(0.0, float(residual_gap))
    )
    omega = timely_visible_jobs(
        problem, state, entity_id, checkpoint_index
    ) if gap > 0.0 else ()
    confirmed: list[Job] = []
    for job in omega:
        ongoing = _ongoing_operation(state, job)
        if ongoing is None:
            continue
        scheduled, operation = ongoing
        if (
            operation.sequence_index == job.num_operations - 1
            and scheduled.end_time <= checkpoint.time
        ):
            confirmed.append(job)
    confirmed_ids = {job.job_id for job in confirmed}
    additional = tuple(job for job in omega if job.job_id not in confirmed_ids)
    confirmed_supply = float(sum(job.quantity for job in confirmed))
    additional_gap = max(0.0, gap - confirmed_supply)
    timely_quantity = float(sum(job.quantity for job in omega))
    additional_quantity = float(sum(job.quantity for job in additional))

    work_ratios: list[float] = []
    relevant_machines: set[int] = set()
    for job in additional:
        unstarted = _unstarted_operations(job, state)
        workload = float(sum(op.min_processing_time for op in unstarted))
        if workload > 0.0:
            work_ratios.append(workload / float(job.quantity))
        relevant_machines.update(
            machine_id
            for operation in unstarted
            for machine_id in operation.eligible_machines
        )

    applicable = additional_gap > 0.0 and additional_quantity >= additional_gap
    if applicable and not work_ratios:
        raise RuntimeError("applicable capacity analysis requires positive additional work")
    nu = min(work_ratios) if work_ratios else None
    available_capacity = float(
        sum(
            max(
                0,
                checkpoint.time
                - state.current_time
                - max(
                    0,
                    state.machine_available_times.get(machine_id, state.current_time)
                    - state.current_time,
                ),
            )
            for machine_id in relevant_machines
        )
    )
    required_workload = (
        additional_gap * float(nu) if applicable and nu is not None else 0.0
    )
    capacity_shortfall = (
        max(0.0, additional_gap - available_capacity / float(nu))
        if applicable and nu is not None
        else 0.0
    )
    return CheckpointCapacityAnalysis(
        entity_id=entity_id,
        checkpoint_index=checkpoint_index,
        residual_gap=gap,
        timely_jobs=omega,
        confirmed_jobs=tuple(confirmed),
        additional_jobs=additional,
        confirmed_supply=confirmed_supply,
        additional_gap=additional_gap,
        timely_quantity=timely_quantity,
        additional_quantity=additional_quantity,
        minimum_work_per_quantity=nu,
        eligible_machine_ids=tuple(sorted(relevant_machines)),
        available_capacity=available_capacity,
        required_workload=required_workload,
        applicable=applicable,
        capacity_shortfall=capacity_shortfall,
    )


def compute_optimistic_checkpoint_workload_capacity(
    problem: SchedulingProblem,
    state: ScheduleState,
    entity_id: int,
    checkpoint_index: int,
    *,
    residual_gap: float | None = None,
) -> tuple[float, float]:
    """Compatibility tuple ``(G_plus * nu, CapR)`` from Sec. 4.3."""

    analysis = analyze_checkpoint_capacity(
        problem,
        state,
        entity_id,
        checkpoint_index,
        residual_gap=residual_gap,
    )
    return analysis.required_workload, analysis.available_capacity


def visible_optimistic_completion(job: Job, state: ScheduleState) -> int:
    """Best optimistic completion C*_j(tau) for an already released job.

    Keeping this precondition here prevents future-job release times and
    processing attributes from entering any caller that computes visible
    coverage or agent-facing features.
    """

    if job.release_time > state.current_time:
        raise ValueError("visible optimistic completion requires a released job")

    return optimistic_job_completion_lower_bound(job, state)


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
        if checkpoint.time <= state.current_time or gap <= 0.0:
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
    return float(
        sum(
            job.quantity
            for job in timely_visible_jobs(
                problem, state, entity_id, checkpoint_index
            )
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
    """Return cumulative mean, standard deviation and conservative quantity.

    Bucket variances are accumulated over ``(tau, checkpoint_time]`` before a
    *single* uncertainty correction is applied.  The conservative result is
    the non-negative prefix maximum required by Eq. (4-*) of the manuscript,
    not a sum of separately truncated bucket forecasts.  For a partial bucket,
    homogeneous-within-bucket interpolation scales mean and variance linearly
    with elapsed time.
    """

    if kappa < 0.0:
        raise ValueError("forecast risk parameter kappa must be non-negative")
    demand = 0.0
    variance = 0.0
    conservative = 0.0
    for bucket in snapshot.for_entity(entity_id):
        fraction = _overlap_fraction(
            bucket.start, bucket.end, tau, checkpoint_time
        )
        if fraction <= 0.0:
            continue
        demand += fraction * bucket.d_hat
        variance += fraction * bucket.sigma * bucket.sigma
        conservative = max(
            conservative,
            demand - kappa * math.sqrt(max(0.0, variance)),
        )
    return demand, math.sqrt(max(0.0, variance)), max(0.0, conservative)


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
    cumulative_mean = 0.0
    cumulative_variance = 0.0
    prefix_conservative = 0.0
    for bucket in snapshot.for_entity(entity_id):
        segment_start = max(bucket.start, tau)
        segment_end = min(bucket.end, checkpoint_time)
        fraction = _overlap_fraction(bucket.start, bucket.end, tau, checkpoint_time)
        if fraction <= 0.0:
            continue
        segment_mean = fraction * bucket.d_hat
        segment_variance = fraction * bucket.sigma * bucket.sigma
        end_mean = cumulative_mean + segment_mean
        end_variance = cumulative_variance + segment_variance
        end_value = max(
            prefix_conservative,
            end_mean - kappa * math.sqrt(max(0.0, end_variance)),
        )
        if end_value + 1e-12 >= future_required:
            duration = float(segment_end - segment_start)

            def conservative_at(elapsed: float) -> float:
                share = 0.0 if duration <= 0.0 else elapsed / duration
                mean = cumulative_mean + share * segment_mean
                variance = cumulative_variance + share * segment_variance
                return max(
                    prefix_conservative,
                    mean - kappa * math.sqrt(max(0.0, variance)),
                )

            lower, upper = 0.0, duration
            for _ in range(64):
                middle = (lower + upper) / 2.0
                if conservative_at(middle) + 1e-12 >= future_required:
                    upper = middle
                else:
                    lower = middle
            return float(segment_start) + upper
        cumulative_mean = end_mean
        cumulative_variance = end_variance
        prefix_conservative = end_value
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
            capacity = analyze_checkpoint_capacity(
                problem,
                state,
                entity.entity_id,
                index,
                residual_gap=gap,
            )
            required_workload = capacity.required_workload
            machine_capacity = capacity.available_capacity
            visible = capacity.timely_quantity
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
                confirmed_supply=capacity.confirmed_supply,
                additional_gap=capacity.additional_gap,
                timely_visible_quantity=capacity.timely_quantity,
                additional_candidate_quantity=capacity.additional_quantity,
                minimum_work_per_quantity=capacity.minimum_work_per_quantity,
                capacity_applicable=capacity.applicable,
                capacity_shortfall=capacity.capacity_shortfall,
                timely_job_ids=tuple(job.job_id for job in capacity.timely_jobs),
                additional_job_ids=tuple(
                    job.job_id for job in capacity.additional_jobs
                ),
                eligible_machine_ids=capacity.eligible_machine_ids,
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
