"""Schedule feasibility checks for simulated and solver-produced schedules."""

from __future__ import annotations

from dataclasses import dataclass
from collections import Counter, defaultdict

from .models import FJSPInstance, Job, Operation, ScheduledOperation
from .state import ScheduleState


@dataclass(frozen=True)
class ScheduleIssue:
    code: str
    message: str


@dataclass(frozen=True)
class ScheduleValidationReport:
    issues: tuple[ScheduleIssue, ...]

    @property
    def feasible(self) -> bool:
        return not self.issues

    @property
    def codes(self) -> tuple[str, ...]:
        return tuple(issue.code for issue in self.issues)

    def summary(self, *, max_issues: int = 3) -> str:
        if not self.issues:
            return "schedule feasible"
        shown = "; ".join(issue.message for issue in self.issues[:max_issues])
        remaining = len(self.issues) - max_issues
        if remaining > 0:
            return f"{shown}; ... ({remaining} more)"
        return shown


def validate_schedule(instance: FJSPInstance, state: ScheduleState) -> ScheduleValidationReport:
    issues: list[ScheduleIssue] = []
    operations_by_key = _operations_by_key(instance)
    scheduled_by_key: defaultdict[tuple[int, int], list[ScheduledOperation]] = defaultdict(list)
    for scheduled in state.scheduled_operations:
        scheduled_by_key[(scheduled.job_id, scheduled.op_id)].append(scheduled)

    _check_operation_coverage(instance, scheduled_by_key, issues)
    _check_scheduled_operations(instance, operations_by_key, state, issues)
    _check_job_precedence_and_completion(instance, scheduled_by_key, state, issues)
    _check_machine_overlap(state, issues)

    return ScheduleValidationReport(tuple(issues))


def _operations_by_key(instance: FJSPInstance) -> dict[tuple[int, int], Operation]:
    return {
        (job.job_id, op.op_id): op
        for job in instance.jobs
        for op in job.operations
    }


def _check_operation_coverage(
    instance: FJSPInstance,
    scheduled_by_key: dict[tuple[int, int], list[ScheduledOperation]],
    issues: list[ScheduleIssue],
) -> None:
    expected = {
        (job.job_id, op.op_id)
        for job in instance.jobs
        for op in job.operations
    }
    seen = set(scheduled_by_key)
    for key in sorted(expected - seen):
        issues.append(ScheduleIssue("missing_operation", f"operation {key} was not scheduled"))
    for key in sorted(seen - expected):
        issues.append(ScheduleIssue("unknown_operation", f"operation {key} is not in the instance"))
    for key, scheduled_ops in sorted(scheduled_by_key.items()):
        if len(scheduled_ops) > 1:
            issues.append(ScheduleIssue("duplicate_operation", f"operation {key} was scheduled {len(scheduled_ops)} times"))


def _check_scheduled_operations(
    instance: FJSPInstance,
    operations_by_key: dict[tuple[int, int], Operation],
    state: ScheduleState,
    issues: list[ScheduleIssue],
) -> None:
    machine_ids = {machine.machine_id for machine in instance.machines}
    completed_operations = set(state.completed_operations)
    counts = Counter((scheduled.job_id, scheduled.op_id) for scheduled in state.scheduled_operations)
    for scheduled in state.scheduled_operations:
        key = (scheduled.job_id, scheduled.op_id)
        if key not in operations_by_key:
            continue
        if counts[key] > 1:
            continue
        job = instance.get_job(scheduled.job_id)
        op = operations_by_key[key]
        if scheduled.machine_id not in machine_ids:
            issues.append(ScheduleIssue("unknown_machine", f"machine {scheduled.machine_id} is not in the instance"))
            continue
        try:
            duration = op.processing_time_on(scheduled.machine_id)
        except KeyError:
            issues.append(ScheduleIssue("ineligible_machine", f"operation {key} cannot run on machine {scheduled.machine_id}"))
            continue
        if scheduled.end_time - scheduled.start_time != duration:
            issues.append(ScheduleIssue("duration_mismatch", f"operation {key} has duration {scheduled.end_time - scheduled.start_time}, expected {duration}"))
        if scheduled.start_time < job.release_time:
            issues.append(ScheduleIssue("release_violation", f"operation {key} starts before release time {job.release_time}"))
        if key not in completed_operations:
            issues.append(ScheduleIssue("completion_state_mismatch", f"operation {key} is scheduled but not marked completed"))


def _check_job_precedence_and_completion(
    instance: FJSPInstance,
    scheduled_by_key: dict[tuple[int, int], list[ScheduledOperation]],
    state: ScheduleState,
    issues: list[ScheduleIssue],
) -> None:
    for job in instance.jobs:
        previous: ScheduledOperation | None = None
        for op in job.operations:
            scheduled_ops = scheduled_by_key.get((job.job_id, op.op_id), [])
            if len(scheduled_ops) != 1:
                previous = None
                continue
            scheduled = scheduled_ops[0]
            if previous is not None and scheduled.start_time < previous.end_time:
                issues.append(
                    ScheduleIssue(
                        "precedence_violation",
                        f"job {job.job_id} operation {op.op_id} starts before previous operation completes",
                    )
                )
            previous = scheduled
        _check_job_completion(job, scheduled_by_key, state, issues)


def _check_job_completion(
    job: Job,
    scheduled_by_key: dict[tuple[int, int], list[ScheduledOperation]],
    state: ScheduleState,
    issues: list[ScheduleIssue],
) -> None:
    last_op = job.operations[-1]
    scheduled_ops = scheduled_by_key.get((job.job_id, last_op.op_id), [])
    completion_time = state.completed_jobs.get(job.job_id)
    if len(scheduled_ops) != 1:
        if completion_time is None:
            issues.append(ScheduleIssue("missing_job_completion", f"job {job.job_id} has no completion time"))
        return
    expected_completion = scheduled_ops[0].end_time
    if completion_time is None:
        issues.append(ScheduleIssue("missing_job_completion", f"job {job.job_id} has no completion time"))
    elif completion_time != expected_completion:
        issues.append(ScheduleIssue("job_completion_mismatch", f"job {job.job_id} completion is {completion_time}, expected {expected_completion}"))


def _check_machine_overlap(state: ScheduleState, issues: list[ScheduleIssue]) -> None:
    by_machine: defaultdict[int, list[ScheduledOperation]] = defaultdict(list)
    for scheduled in state.scheduled_operations:
        by_machine[scheduled.machine_id].append(scheduled)
    for machine_id, scheduled_ops in by_machine.items():
        ordered = sorted(scheduled_ops, key=lambda item: (item.start_time, item.end_time, item.job_id, item.op_id))
        for previous, current in zip(ordered, ordered[1:]):
            if current.start_time < previous.end_time:
                issues.append(
                    ScheduleIssue(
                        "machine_overlap",
                        f"machine {machine_id} overlaps operation {(previous.job_id, previous.op_id)} and {(current.job_id, current.op_id)}",
                    )
                )
