"""Leakage-safe online problem view."""

from __future__ import annotations

from dataclasses import dataclass, field

from .forecast import ForecastSnapshot
from .models import FJSPInstance, Job, Machine, ServiceEntity


@dataclass
class OnlineProblemView:
    jobs: tuple[Job, ...]
    entities: tuple[ServiceEntity, ...]
    machines: tuple[Machine, ...]
    alpha: float
    beta: float
    forecast_snapshot: ForecastSnapshot
    forecast_risk_kappa: float
    r5_epsilon: float
    current_time: int
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.current_time < 0:
            raise ValueError("online-view current time must be non-negative")
        leaked = tuple(
            job.job_id for job in self.jobs if job.release_time > self.current_time
        )
        if leaked:
            raise ValueError(
                f"online view cannot contain unreleased jobs at t={self.current_time}: {leaked}"
            )

    @classmethod
    def from_instance(
        cls,
        instance: FJSPInstance,
        *,
        visible_job_ids: set[int],
        forecast_snapshot: ForecastSnapshot,
        current_time: int,
    ) -> "OnlineProblemView":
        visible = tuple(
            job
            for job in instance.jobs
            if job.job_id in visible_job_ids and job.release_time <= current_time
        )
        safe_ids = tuple(sorted(job.job_id for job in visible))
        metadata = {
            "visible_job_ids": safe_ids,
            "forecast_update_epoch": forecast_snapshot.update_epoch,
        }
        # Public generator/scaling metadata is legal online information.  It
        # lets Table 5-2 use fixed distribution scales without consulting the
        # realized (possibly unreleased) job sample.
        for key in ("generator", "feature_normalization_scales"):
            if key in instance.metadata:
                metadata[key] = instance.metadata[key]
        return cls(
            jobs=visible,
            entities=instance.entities,
            machines=instance.machines,
            alpha=instance.alpha,
            beta=instance.beta,
            forecast_snapshot=forecast_snapshot,
            forecast_risk_kappa=instance.forecast_risk_kappa,
            r5_epsilon=instance.r5_epsilon,
            current_time=current_time,
            metadata=metadata,
        )

    @property
    def num_jobs(self) -> int:
        return len(self.jobs)

    @property
    def num_machines(self) -> int:
        return len(self.machines)

    @property
    def num_entities(self) -> int:
        return len(self.entities)

    def get_job(self, job_id: int) -> Job:
        for job in self.jobs:
            if job.job_id == job_id:
                return job
        raise KeyError(f"visible job {job_id} not found")

    def get_entity(self, entity_id: int) -> ServiceEntity:
        for entity in self.entities:
            if entity.entity_id == entity_id:
                return entity
        raise KeyError(f"entity {entity_id} not found")

    def jobs_of_entity(self, entity_id: int) -> tuple[Job, ...]:
        return tuple(job for job in self.jobs if job.entity_id == entity_id)


SchedulingProblem = FJSPInstance | OnlineProblemView
