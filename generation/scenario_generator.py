"""Dynamic FJSP-SFR scenarios used by the center/L9 formal experiments."""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean

from fjsp_sfr.core.forecast import build_compound_poisson_forecast_provider
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


def build_exogenous_service_entities(
    *,
    entity_count: int,
    expected_total_quantity_per_entity: float | None = None,
    final_checkpoint_time: int | None = None,
    entity_expected_quantities: tuple[float, ...] | None = None,
    entity_final_times: tuple[int, ...] | None = None,
    first_time_ratio: float,
    first_demand_ratio: float,
    commitment_ratio: float,
    weights: tuple[float, float],
    profile_id: str = "scenario_exogenous_commitment",
) -> tuple[ServiceEntity, ...]:
    """Construct published commitments without accepting a job realization.

    The inputs are all scenario-level public quantities.  In particular, this
    function has no route, release, due-date, entity-assignment, or realized
    service-quantity argument.  The still-TBD formula that maps scenario
    parameters to ``final_checkpoint_time`` is intentionally kept outside this
    interface so an author-confirmed calendar can be supplied without changing
    commitment semantics.
    """

    if entity_count <= 0:
        raise ValueError("entity_count must be positive")
    if entity_expected_quantities is None:
        if expected_total_quantity_per_entity is None:
            raise ValueError("expected quantity must be specified")
        entity_expected_quantities = (float(expected_total_quantity_per_entity),) * entity_count
    if entity_final_times is None:
        if final_checkpoint_time is None:
            raise ValueError("final checkpoint time must be specified")
        entity_final_times = (int(final_checkpoint_time),) * entity_count
    if len(entity_expected_quantities) != entity_count or len(entity_final_times) != entity_count:
        raise ValueError("entity commitments and deadlines must align with entity count")
    if any(not math.isfinite(value) or value <= 0 for value in entity_expected_quantities):
        raise ValueError("all expected entity quantities must be positive and finite")
    if any(value < 2 for value in entity_final_times):
        raise ValueError("final checkpoint times must be at least two")
    if not 0.0 < first_time_ratio < 1.0:
        raise ValueError("first checkpoint time ratio must lie in (0, 1)")
    if not 0.0 < first_demand_ratio < 1.0:
        raise ValueError("first checkpoint demand ratio must lie in (0, 1)")
    if commitment_ratio <= 0.0 or any(weight <= 0.0 for weight in weights):
        raise ValueError("commitment ratio and checkpoint weights must be positive")

    result = []
    for entity_id, (expected, deadline) in enumerate(
        zip(entity_expected_quantities, entity_final_times, strict=True)
    ):
        first_time = max(1, min(deadline - 1, math.floor(deadline * first_time_ratio + 0.5)))
        final_quantity = expected * commitment_ratio
        result.append(
            ServiceEntity(
                entity_id,
                (
                    Checkpoint(first_time, final_quantity * first_demand_ratio, weights[0]),
                    Checkpoint(deadline, final_quantity, weights[1]),
                ),
                profile_id,
            )
        )
    return tuple(result)


def _expected_checkpoint_horizon(
    *, new_jobs: int, mean_interval: float, expected_route_work: float
) -> float:
    """Published service window: N_new / lambda + nominal expected job work."""
    return new_jobs * mean_interval + expected_route_work


def _independent_rng(seed: int, namespace: str) -> random.Random:
    """Independent deterministic streams unaffected by parameter-grid changes."""
    digest = hashlib.sha256(f"paperA-scenario-v2:{seed}:{namespace}".encode()).digest()
    return random.Random(int.from_bytes(digest[:16], "big"))


def scenario_config_hash(config: dict) -> str:
    """Digest the full published scenario configuration, not the job realization."""
    data = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


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
    total_jobs = initial_jobs + int(new_jobs)
    entity_count = max(1, round(total_jobs / 8))

    # Public commitments and forecasts are constructed before, and without,
    # the RNG used for the realized job stream below.
    first_ratio = float(checkpoint_cfg["first_time_ratio"])
    commitment_ratio = float(checkpoint_cfg["commitment_ratio"])
    first_demand_ratio = float(checkpoint_cfg["first_demand_ratio"])
    weight_scale = float(checkpoint_cfg.get("weight_scale", 1.0))
    weights = tuple(
        float(value) * weight_scale * float(service_weight_scale)
        for value in checkpoint_cfg["weights"]
    )
    expected_quantity = fmean(range(quantity_low, quantity_high + 1))
    expected_quantity_squared = fmean(
        value * value for value in range(quantity_low, quantity_high + 1)
    )
    share_rng = _independent_rng(seed, "public_entity_shares")
    raw_shares = [share_rng.uniform(0.8, 1.2) for _ in range(entity_count)]
    total_share = sum(raw_shares)
    shares = tuple(value / total_share for value in raw_shares)
    deadlines_rng = _independent_rng(seed, "public_deadline_perturbations")
    perturbations = tuple(deadlines_rng.uniform(-0.10, 0.10) for _ in range(entity_count))
    base_horizon = _expected_checkpoint_horizon(
        new_jobs=new_jobs, mean_interval=mean_interval, expected_route_work=expected_work
    )
    final_times = tuple(max(2, math.floor(base_horizon * (1 + xi) + 0.5)) for xi in perturbations)
    expected_entity_quantities = tuple(total_jobs * expected_quantity * pi for pi in shares)
    entities = build_exogenous_service_entities(
        entity_count=entity_count,
        entity_expected_quantities=expected_entity_quantities,
        entity_final_times=final_times,
        first_time_ratio=first_ratio,
        first_demand_ratio=first_demand_ratio,
        commitment_ratio=commitment_ratio,
        weights=weights,
    )
    total_arrival_rate = 1.0 / mean_interval
    forecast_provider = build_compound_poisson_forecast_provider(
        entity_ids=range(entity_count),
        total_arrival_rate=total_arrival_rate,
        quantity_mean=expected_quantity,
        quantity_second_moment=expected_quantity_squared,
        max_time=max(final_times),
        entity_shares={entity_id: shares[entity_id] for entity_id in range(entity_count)},
    )

    releases = _arrival_times(initial_jobs, new_jobs, mean_interval, _independent_rng(seed, "arrivals"))
    assigned_rng = _independent_rng(seed, "entity_assignments")
    assignments = assigned_rng.choices(range(entity_count), weights=shares, k=total_jobs)
    rng = _independent_rng(seed, "job_attributes")

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

    jobs = tuple(Job(
        item["job_id"], item["entity_id"], item["release_time"], item["due_date"],
        item["quantity"], item["operations"],
    ) for item in raw_jobs)
    arrival_window = max(releases) if new_jobs else 0
    observation_window = arrival_window + initial_jobs * expected_work / (load * machines)
    measured_load = sum(item["mean_route_work"] for item in raw_jobs) / max(1.0, machines * observation_window)
    return FJSPInstance(
        jobs=jobs,
        entities=entities,
        machines=tuple(Machine(index) for index in range(machines)),
        forecast_provider=forecast_provider,
        metadata={
            "generator": "scenario_generator_v2", "scenario_config_hash": scenario_config_hash(payload),
            "seed": int(seed),
            "machines": machines, "initial_jobs": initial_jobs, "new_jobs": new_jobs,
            "target_load": load, "measured_load": measured_load, "ddt": ddt,
            "mean_arrival_interval": mean_interval, "load_time_window": observation_window,
            "checkpoint": dict(checkpoint_cfg),
            "service_weight_scale": float(service_weight_scale),
            "service_commitment_source": "scenario_parameter_expectation",
            "realization_used_to_build_commitment": False,
            "checkpoint_calendar_source": "published_service_window_with_entity_perturbation",
            "expected_checkpoint_horizon": base_horizon,
            "entity_shares": shares,
            "entity_deadline_perturbations": perturbations,
            "entity_final_times": final_times,
            "expected_quantity": expected_quantity,
            "expected_quantity_second_moment": expected_quantity_squared,
            "expected_commitment_quantity_per_entity": (
                tuple(value * commitment_ratio for value in expected_entity_quantities)
            ),
            "forecast_source": "homogeneous_compound_poisson_public_statistics",
            "realization_used_to_build_forecast": False,
            "total_arrival_rate": total_arrival_rate,
            "entity_arrival_rates": tuple(total_arrival_rate * pi for pi in shares),
            "forecast_horizon": max(final_times),
            "forecast_bucket_length_status": "not_required_for_homogeneous_moments",
            "forecast_risk_kappa_status": "model_default_author_confirmation_required",
            "certificate_public_bounds": {
                "planned_total_jobs": total_jobs,
                "planned_initial_jobs": initial_jobs,
                "planned_new_jobs": int(new_jobs),
                "quantity_upper_bound": quantity_high,
                "minimum_job_work": op_low * proc_low,
                "source": "published_scenario_generation_bounds",
            },
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


__all__ = [
    "CONFIG_PATH",
    "ScenarioSpec",
    "all_seed_partitions",
    "build_exogenous_service_entities",
    "load_scenario_config",
    "make_instance",
    "make_scenario_instance",
    "scenario_names",
    "scenario_config_hash",
    "scenario_spec",
    "seed_values",
]
