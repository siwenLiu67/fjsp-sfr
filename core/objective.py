"""Objective computation for FJSP-SFR schedules."""

from __future__ import annotations

from dataclasses import dataclass

from .models import FJSPInstance
from .state import ScheduleState


@dataclass(frozen=True)
class ObjectiveResult:
    total_tardiness: int
    weighted_checkpoint_shortfall: float
    Z: float
    zero_shortfall_checkpoint_rate: float
    per_checkpoint_shortfall: dict[tuple[int, int], float]
    per_job_tardiness: dict[int, int]
    per_checkpoint_completed_quantity: dict[tuple[int, int], int]


def compute_objective(instance: FJSPInstance, state: ScheduleState) -> ObjectiveResult:
    per_job_tardiness: dict[int, int] = {}
    for job in instance.jobs:
        completion = state.completed_jobs.get(job.job_id)
        if completion is None:
            raise ValueError(f"job {job.job_id} has no completion time")
        per_job_tardiness[job.job_id] = max(0, completion - job.due_date)

    total_tardiness = sum(per_job_tardiness.values())
    per_shortfall: dict[tuple[int, int], float] = {}
    per_completed_quantity: dict[tuple[int, int], int] = {}
    weighted_shortfall = 0.0
    zero_count = 0
    checkpoint_count = 0

    for entity in instance.entities:
        entity_jobs = instance.jobs_of_entity(entity.entity_id)
        for checkpoint_index, checkpoint in enumerate(entity.checkpoints):
            completed_quantity = sum(
                job.quantity
                for job in entity_jobs
                if state.completed_jobs.get(job.job_id, checkpoint.time + 1) <= checkpoint.time
            )
            required = entity.required_quantity(checkpoint_index)
            shortfall = max(0.0, float(required - completed_quantity))
            key = (entity.entity_id, checkpoint_index)
            per_completed_quantity[key] = completed_quantity
            per_shortfall[key] = shortfall
            weighted_shortfall += checkpoint.weight * shortfall
            zero_count += int(shortfall == 0.0)
            checkpoint_count += 1

    zero_rate = zero_count / checkpoint_count if checkpoint_count else 0.0
    z_value = instance.alpha * total_tardiness + instance.beta * weighted_shortfall
    return ObjectiveResult(
        total_tardiness=total_tardiness,
        weighted_checkpoint_shortfall=weighted_shortfall,
        Z=z_value,
        zero_shortfall_checkpoint_rate=zero_rate,
        per_checkpoint_shortfall=per_shortfall,
        per_job_tardiness=per_job_tardiness,
        per_checkpoint_completed_quantity=per_completed_quantity,
    )
