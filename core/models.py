"""Problem data classes for forecast-aware dynamic FJSP-SFR."""

from __future__ import annotations

from dataclasses import dataclass, field

from .forecast import ForecastProvider


@dataclass(frozen=True)
class OperationAlternative:
    machine_id: int
    processing_time: int

    def __post_init__(self) -> None:
        if self.machine_id < 0:
            raise ValueError("machine_id must be non-negative")
        if self.processing_time <= 0:
            raise ValueError("processing_time must be positive")


@dataclass(frozen=True)
class Operation:
    op_id: int
    job_id: int
    sequence_index: int
    alternatives: tuple[OperationAlternative, ...]

    def __post_init__(self) -> None:
        if self.op_id < 0:
            raise ValueError("op_id must be non-negative")
        if self.job_id < 0:
            raise ValueError("operation job_id must be non-negative")
        if self.sequence_index < 0:
            raise ValueError("sequence_index must be non-negative")
        if not self.alternatives:
            raise ValueError("operation must have at least one eligible machine")
        machine_ids = [alternative.machine_id for alternative in self.alternatives]
        if len(machine_ids) != len(set(machine_ids)):
            raise ValueError("operation has duplicate machine alternatives")

    @property
    def eligible_machines(self) -> tuple[int, ...]:
        return tuple(alternative.machine_id for alternative in self.alternatives)

    @property
    def min_processing_time(self) -> int:
        return min(alternative.processing_time for alternative in self.alternatives)

    def processing_time_on(self, machine_id: int) -> int:
        for alternative in self.alternatives:
            if alternative.machine_id == machine_id:
                return alternative.processing_time
        raise KeyError(f"machine {machine_id} is not eligible for operation {self.op_id}")


@dataclass(frozen=True)
class Job:
    job_id: int
    entity_id: int
    release_time: int
    due_date: int
    quantity: int
    operations: tuple[Operation, ...]

    def __post_init__(self) -> None:
        if self.job_id < 0:
            raise ValueError("job_id must be non-negative")
        if self.entity_id < 0:
            raise ValueError("entity_id must be non-negative")
        if self.release_time < 0:
            raise ValueError("release_time must be non-negative")
        if self.due_date <= 0:
            raise ValueError("job due_date must be positive")
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")
        if not self.operations:
            raise ValueError("job must have at least one operation")
        for index, operation in enumerate(self.operations):
            if operation.job_id != self.job_id:
                raise ValueError("operation job_id must match its job")
            if operation.sequence_index != index:
                raise ValueError("operation sequence indices must be consecutive from zero")

    @property
    def num_operations(self) -> int:
        return len(self.operations)

    def operation_at(self, sequence_index: int) -> Operation:
        return self.operations[sequence_index]


@dataclass(frozen=True)
class Checkpoint:
    time: int
    required_quantity: float
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.time <= 0:
            raise ValueError("checkpoint time must be positive")
        if self.required_quantity <= 0.0:
            raise ValueError("checkpoint required_quantity must be positive")
        if self.weight <= 0.0:
            raise ValueError("checkpoint weight must be positive")


@dataclass(frozen=True)
class ServiceEntity:
    entity_id: int
    checkpoints: tuple[Checkpoint, ...]
    checkpoint_profile_id: str = ""

    def __post_init__(self) -> None:
        if self.entity_id < 0:
            raise ValueError("entity_id must be non-negative")
        if not self.checkpoints:
            raise ValueError("service entity must have at least one checkpoint")
        times = [checkpoint.time for checkpoint in self.checkpoints]
        targets = [checkpoint.required_quantity for checkpoint in self.checkpoints]
        if times != sorted(times) or len(times) != len(set(times)):
            raise ValueError("checkpoint times must be strictly increasing")
        if any(left > right for left, right in zip(targets, targets[1:])):
            raise ValueError("checkpoint targets must be nondecreasing")

    def required_quantity(self, checkpoint_index: int) -> float:
        return self.checkpoints[checkpoint_index].required_quantity


@dataclass(frozen=True)
class Machine:
    machine_id: int

    def __post_init__(self) -> None:
        if self.machine_id < 0:
            raise ValueError("machine_id must be non-negative")


@dataclass
class FJSPInstance:
    jobs: tuple[Job, ...]
    entities: tuple[ServiceEntity, ...]
    machines: tuple[Machine, ...]
    alpha: float = 1.0
    beta: float = 1.0
    forecast_provider: ForecastProvider | None = None
    forecast_risk_kappa: float = 1.0
    r5_epsilon: float = 1e-9
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.jobs or not self.entities or not self.machines:
            raise ValueError("instance must contain jobs, entities, and machines")
        job_ids = [job.job_id for job in self.jobs]
        entity_ids = [entity.entity_id for entity in self.entities]
        machine_ids = [machine.machine_id for machine in self.machines]
        if len(job_ids) != len(set(job_ids)):
            raise ValueError("duplicate job ids")
        if len(entity_ids) != len(set(entity_ids)):
            raise ValueError("duplicate entity ids")
        if len(machine_ids) != len(set(machine_ids)):
            raise ValueError("duplicate machine ids")
        if self.alpha <= 0.0 or self.beta <= 0.0:
            raise ValueError("alpha and beta must be positive")
        if self.forecast_risk_kappa < 0.0 or self.r5_epsilon <= 0.0:
            raise ValueError("kappa must be non-negative and R5 epsilon positive")

        entity_id_set = set(entity_ids)
        machine_id_set = set(machine_ids)
        for job in self.jobs:
            if job.entity_id not in entity_id_set:
                raise ValueError(f"job {job.job_id} references unknown entity")
            for operation in job.operations:
                for alternative in operation.alternatives:
                    if alternative.machine_id not in machine_id_set:
                        raise ValueError(
                            f"operation {operation.op_id} references unknown machine"
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
        raise KeyError(f"job {job_id} not found")

    def get_entity(self, entity_id: int) -> ServiceEntity:
        for entity in self.entities:
            if entity.entity_id == entity_id:
                return entity
        raise KeyError(f"entity {entity_id} not found")

    def jobs_of_entity(self, entity_id: int) -> tuple[Job, ...]:
        return tuple(job for job in self.jobs if job.entity_id == entity_id)


@dataclass(frozen=True)
class ScheduledOperation:
    job_id: int
    op_id: int
    machine_id: int
    start_time: int
    end_time: int

    def __post_init__(self) -> None:
        if self.job_id < 0 or self.op_id < 0 or self.machine_id < 0:
            raise ValueError("scheduled operation ids must be non-negative")
        if self.start_time < 0:
            raise ValueError("start_time must be non-negative")
        if self.end_time <= self.start_time:
            raise ValueError("end_time must be greater than start_time")
