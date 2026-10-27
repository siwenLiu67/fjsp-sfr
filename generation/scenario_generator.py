"""Dynamic FJSP-SFR scenarios used by the center/L9 formal experiments."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean

from fjsp_sfr.core.models import (
    Checkpoint, FJSPInstance, Job, Machine, Operation, OperationAlternative,
    ServiceEntity,
)

CONFIG_PATH = Path(__file__).with_name("scenario_config.json")


@dataclass(frozen=True)
class ScenarioSpec:
    machines: int
    new_jobs: int
    load: float
    ddt: float
    service_weight_scale: float = 1.0


def load_scenario_config(path: Path = CONFIG_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def scenario_names(path: Path = CONFIG_PATH) -> tuple[str, ...]:
    return tuple(load_scenario_config(path)["scenarios"])


def scenario_spec(name: str, path: Path = CONFIG_PATH) -> ScenarioSpec:
    payload = load_scenario_config(path)
    try:
        raw = payload["scenarios"][name]
    except KeyError as error:
        raise ValueError(f"unknown scenario: {name}") from error
    # ``new_jobs`` remains part of ScenarioSpec and make_instance's positional
    # interface for backward compatibility, but is now fixed across scenarios.
    new_jobs = int(raw.get("new_jobs", payload["fixed"]["new_jobs"]))
    return ScenarioSpec(
        int(raw["machines"]),
        new_jobs,
        float(raw["load"]),
        float(raw["ddt"]),
        float(raw.get("service_weight_scale", 1.0)),
    )


def seed_values(name: str, path: Path = CONFIG_PATH) -> tuple[int, ...]:
    try:
        lower, upper = load_scenario_config(path)["seeds"][name]
    except KeyError as error:
        raise ValueError(f"unknown seed partition: {name}") from error
    return tuple(range(int(lower), int(upper) + 1))


def _arrival_times(initial_jobs: int, new_jobs: int, mean_interval: float, rng: random.Random) -> list[int]:
    arrivals = [0] * initial_jobs
    clock = 0.0
    for _ in range(new_jobs):
        clock += rng.expovariate(1.0 / mean_interval)
        arrivals.append(max(1, int(round(clock))))
    return arrivals


def make_instance(
    machines: int, new_jobs: int, load: float, ddt: float, seed: int,
    *, config_path: Path = CONFIG_PATH, service_weight_scale: float = 1.0,
) -> FJSPInstance:
    """Generate an instance from machines, new jobs, target load and due tightness."""

    if (
        machines <= 0
        or new_jobs < 0
        or not 0.0 < load <= 1.0
        or ddt <= 0.0
        or service_weight_scale <= 0.0
    ):
        raise ValueError("invalid scenario factors")
    payload = load_scenario_config(config_path)
    fixed, checkpoint_cfg = payload["fixed"], payload["checkpoint"]
    initial_jobs = int(fixed["initial_jobs"])
    op_low, op_high = map(int, fixed["operations_per_job"])
    proc_low, proc_high = map(int, fixed["processing_time"])
    quantity_low, quantity_high = map(int, fixed["quantity"])
    expected_work = fmean((op_low, op_high)) * fmean((proc_low, proc_high))
    mean_interval = expected_work / (load * machines)
    rng = random.Random(int(seed))
    total_jobs = initial_jobs + int(new_jobs)
    releases = _arrival_times(initial_jobs, new_jobs, mean_interval, rng)
    entity_count = max(1, round(total_jobs / 8))
    assignments = list(range(entity_count))
    assignments.extend(rng.randrange(entity_count) for _ in range(total_jobs - entity_count))
    rng.shuffle(assignments)

    raw_jobs: list[dict] = []
    operation_id = 0
    for job_id in range(total_jobs):
        operations: list[Operation] = []
        mean_route_work = 0.0
        for sequence_index in range(rng.randint(op_low, op_high)):
            eligible = rng.sample(range(machines), rng.randint(1, machines))
            alternatives = tuple(
                OperationAlternative(machine_id, rng.randint(proc_low, proc_high))
                for machine_id in eligible
            )
            operations.append(Operation(operation_id, job_id, sequence_index, alternatives))
            operation_id += 1
            mean_route_work += fmean(item.processing_time for item in alternatives)
        release = releases[job_id]
        raw_jobs.append({
            "job_id": job_id,
            "entity_id": assignments[job_id],
            "release_time": release,
            "due_date": max(2, release + 1, round(release + ddt * mean_route_work)),
            "quantity": rng.randint(quantity_low, quantity_high),
            "operations": tuple(operations),
            "mean_route_work": mean_route_work,
        })

    entities: list[ServiceEntity] = []
    first_ratio = float(checkpoint_cfg["first_time_ratio"])
    tightness = float(checkpoint_cfg["tightness"])
    commitment_ratio = float(checkpoint_cfg["commitment_ratio"])
    first_demand_ratio = float(checkpoint_cfg["first_demand_ratio"])
    weight_scale = float(checkpoint_cfg.get("weight_scale", 1.0))
    weights = tuple(
        float(value) * weight_scale * float(service_weight_scale)
        for value in checkpoint_cfg["weights"]
    )
    for entity_id in range(entity_count):
        group = [item for item in raw_jobs if item["entity_id"] == entity_id]
        final_deadline = max(item["due_date"] for item in group)
        origin = min(item["release_time"] for item in group)
        first_time = origin + round((final_deadline - origin) * first_ratio * tightness)
        final_deadline = max(2, final_deadline)
        first_time = max(1, min(first_time, final_deadline - 1))
        total_quantity = sum(item["quantity"] for item in group)
        commitment = max(1.0, total_quantity * commitment_ratio)
        entities.append(ServiceEntity(entity_id, (
            Checkpoint(first_time, max(1.0, commitment * first_demand_ratio), weights[0]),
            Checkpoint(final_deadline, commitment, weights[1]),
        ), "scenario_shared_service_heavy"))

    jobs = tuple(Job(
        item["job_id"], item["entity_id"], item["release_time"], item["due_date"],
        item["quantity"], item["operations"],
    ) for item in raw_jobs)
    arrival_window = max(releases) if new_jobs else 0
    observation_window = arrival_window + initial_jobs * expected_work / (load * machines)
    measured_load = sum(item["mean_route_work"] for item in raw_jobs) / max(1.0, machines * observation_window)
    return FJSPInstance(
        jobs=jobs,
        entities=tuple(entities),
        machines=tuple(Machine(index) for index in range(machines)),
        metadata={
            "generator": "scenario_generator_v1", "seed": int(seed),
            "machines": machines, "initial_jobs": initial_jobs, "new_jobs": new_jobs,
            "target_load": load, "measured_load": measured_load, "ddt": ddt,
            "mean_arrival_interval": mean_interval, "load_time_window": observation_window,
            "checkpoint": dict(checkpoint_cfg),
            "service_weight_scale": float(service_weight_scale),
        },
    )


def make_scenario_instance(scenario: str = "center", seed: int = 42, *, config_path: Path = CONFIG_PATH) -> FJSPInstance:
    spec = scenario_spec(scenario, config_path)
    return make_instance(
        spec.machines,
        spec.new_jobs,
        spec.load,
        spec.ddt,
        seed,
        config_path=config_path,
        service_weight_scale=spec.service_weight_scale,
    )


def all_seed_partitions(path: Path = CONFIG_PATH) -> dict[str, tuple[int, ...]]:
    return {name: tuple(range(int(bounds[0]), int(bounds[1]) + 1)) for name, bounds in load_scenario_config(path)["seeds"].items()}


__all__ = ["CONFIG_PATH", "ScenarioSpec", "all_seed_partitions", "load_scenario_config", "make_instance", "make_scenario_instance", "scenario_names", "scenario_spec", "seed_values"]
