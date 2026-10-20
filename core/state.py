"""Mutable state for event-driven FJSP-SFR simulation."""

from __future__ import annotations

from dataclasses import dataclass, field

from .events import EventQueue
from .models import ScheduledOperation


@dataclass
class ScheduleState:
    current_time: int = 0
    machine_available_times: dict[int, int] = field(default_factory=dict)
    completed_operations: set[tuple[int, int]] = field(default_factory=set)
    ongoing_operations: dict[tuple[int, int], ScheduledOperation] = field(default_factory=dict)
    scheduled_operations: list[ScheduledOperation] = field(default_factory=list)
    completed_jobs: dict[int, int] = field(default_factory=dict)
    event_queue: EventQueue = field(default_factory=EventQueue)
    last_completed_op_index: dict[int, int] = field(default_factory=dict)

    def is_machine_idle(self, machine_id: int) -> bool:
        return self.current_time >= self.machine_available_times.get(machine_id, 0)

    def is_job_completed(self, job_id: int) -> bool:
        return job_id in self.completed_jobs

    def is_operation_completed(self, job_id: int, op_id: int) -> bool:
        return (job_id, op_id) in self.completed_operations

    def is_operation_ongoing(self, job_id: int, op_id: int) -> bool:
        return (job_id, op_id) in self.ongoing_operations

    def next_op_index_for_job(self, job_id: int) -> int:
        return self.last_completed_op_index.get(job_id, -1) + 1

    def mark_operation_started(
        self,
        job_id: int,
        op_id: int,
        machine_id: int,
        start_time: int,
        end_time: int,
    ) -> None:
        scheduled = ScheduledOperation(job_id, op_id, machine_id, start_time, end_time)
        self.ongoing_operations[(job_id, op_id)] = scheduled
        self.scheduled_operations.append(scheduled)
        self.machine_available_times[machine_id] = end_time

    def mark_operation_completed(self, job_id: int, op_id: int, sequence_index: int) -> None:
        self.completed_operations.add((job_id, op_id))
        self.ongoing_operations.pop((job_id, op_id), None)
        self.last_completed_op_index[job_id] = sequence_index

    def mark_job_completed(self, job_id: int, completion_time: int) -> None:
        self.completed_jobs[job_id] = completion_time

    def copy_for_algorithm(self) -> "ScheduleState":
        return ScheduleState(
            current_time=self.current_time,
            machine_available_times=dict(self.machine_available_times),
            completed_operations=set(self.completed_operations),
            ongoing_operations=dict(self.ongoing_operations),
            scheduled_operations=list(self.scheduled_operations),
            completed_jobs=dict(self.completed_jobs),
            event_queue=EventQueue(),
            last_completed_op_index=dict(self.last_completed_op_index),
        )

