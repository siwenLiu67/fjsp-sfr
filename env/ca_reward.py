"""Decision-to-decision incremental objective reward for CA-PPO."""

from __future__ import annotations

from dataclasses import dataclass

from fjsp_sfr.core.online import SchedulingProblem
from fjsp_sfr.core.state import ScheduleState


@dataclass(frozen=True)
class CARewardBreakdown:
    total: float
    delta_tardiness: float
    delta_weighted_shortfall: float


def assessed_tardiness(problem: SchedulingProblem, state: ScheduleState) -> float:
    """Return tardiness already incurred by the current physical time."""

    total = 0
    for job in problem.jobs:
        completion = state.completed_jobs.get(job.job_id)
        if completion is not None and completion <= state.current_time:
            total += max(0, completion - job.due_date)
        elif job.release_time <= state.current_time:
            total += max(0, state.current_time - job.due_date)
    return float(total)


def assessed_weighted_shortfall(
    problem: SchedulingProblem,
    state: ScheduleState,
) -> float:
    """Count only checkpoints whose physical assessment time has arrived."""

    total = 0.0
    for entity in problem.entities:
        jobs = problem.jobs_of_entity(entity.entity_id)
        for k, checkpoint in enumerate(entity.checkpoints):
            if checkpoint.time > state.current_time:
                continue
            completed = sum(
                job.quantity
                for job in jobs
                if state.completed_jobs.get(job.job_id, checkpoint.time + 1)
                <= checkpoint.time
            )
            total += checkpoint.weight * max(
                0.0, float(entity.required_quantity(k) - completed)
            )
    return total


def ca_transition_reward(
    problem: SchedulingProblem,
    previous_state: ScheduleState,
    next_state: ScheduleState,
) -> CARewardBreakdown:
    if next_state.current_time < previous_state.current_time:
        raise ValueError("next state cannot precede the previous state")
    delta_tt = assessed_tardiness(problem, next_state) - assessed_tardiness(
        problem, previous_state
    )
    delta_wsf = assessed_weighted_shortfall(
        problem, next_state
    ) - assessed_weighted_shortfall(problem, previous_state)
    return CARewardBreakdown(
        total=-(problem.alpha * delta_tt + problem.beta * delta_wsf),
        delta_tardiness=delta_tt,
        delta_weighted_shortfall=delta_wsf,
    )
