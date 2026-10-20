"""Reproducible latent-demand scenarios, arrivals, and service commitments."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from statistics import fmean
from fjsp_sfr.core.forecast import build_forecast_provider_from_latent_demand
from fjsp_sfr.core.models import (
    Checkpoint,
    FJSPInstance,
    Job,
    Machine,
    Operation,
    OperationAlternative,
    ServiceEntity,
)


def _validate_strict_sequence(name: str, values: tuple[float, ...]) -> None:
    if any(value <= 0.0 or value > 1.0 for value in values):
        raise ValueError(f"{name} values must lie in (0, 1]")
    if any(left >= right for left, right in zip(values, values[1:])):
        raise ValueError(f"{name} must be strictly increasing")


@dataclass(frozen=True)
class CheckpointProfile:
    profile_id: str
    time_ratios: tuple[float, ...]
    target_ratios: tuple[float, ...]
    weights: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.profile_id:
            raise ValueError("checkpoint profile_id must not be empty")
        size = len(self.time_ratios)
        if size not in (2, 3):
            raise ValueError("checkpoint profiles must contain two or three checkpoints")
        if len(self.target_ratios) != size or len(self.weights) != size:
            raise ValueError("checkpoint profile components must have equal lengths")
        _validate_strict_sequence("time_ratios", self.time_ratios)
        _validate_strict_sequence("target_ratios", self.target_ratios)
        if any(weight <= 0.0 for weight in self.weights):
            raise ValueError("checkpoint weights must be positive")
        if self.time_ratios[-1] != 1.0 or self.target_ratios[-1] != 1.0:
            raise ValueError("checkpoint profiles must end at ratio 1.0")


_PROFILES = (
    CheckpointProfile("balanced_2", (0.50, 1.00), (0.45, 1.00), (1.0, 2.0)),
    CheckpointProfile("early_pressure_2", (0.40, 1.00), (0.60, 1.00), (1.0, 3.0)),
    CheckpointProfile("balanced_3", (0.33, 0.67, 1.00), (0.30, 0.65, 1.00), (1.0, 2.0, 3.0)),
    CheckpointProfile("front_loaded_3", (0.25, 0.55, 1.00), (0.45, 0.78, 1.00), (1.0, 2.5, 4.0)),
)
DEFAULT_CHECKPOINT_PROFILES = {profile.profile_id: profile for profile in _PROFILES}

ARRIVAL_PROFILES = ("uniform", "front_loaded", "back_loaded", "bursty")


@dataclass(frozen=True)
class LatentDemandScenario:
    """Scenario-level demand means shared by commitment, arrivals, and forecast."""

    bucket_starts: tuple[int, ...]
    expected_quantity_by_entity: tuple[float, ...]
    bucket_means_by_entity: tuple[tuple[float, ...], ...]
    arrival_profile_by_entity: tuple[str, ...]

    def bucket_records(self) -> tuple[tuple[int, int, float], ...]:
        return tuple(
            (entity_id, bucket_start, means[bucket_index])
            for entity_id, means in enumerate(self.bucket_means_by_entity)
            for bucket_index, bucket_start in enumerate(self.bucket_starts)
        )


@dataclass(frozen=True)
class InstanceConfig:
    num_instances: int
    num_jobs: int
    num_machines: int
    num_entities: int
    ops_per_job: tuple[int, int]
    checkpoints_per_entity: tuple[int, int]
    checkpoint_tightness: float
    checkpoint_profile_pool: tuple[str, ...] = tuple(DEFAULT_CHECKPOINT_PROFILES)
    quantity_range: tuple[int, int] = (1, 5)
    proc_time_range: tuple[int, int] = (1, 20)
    eligible_machines_range: tuple[int, int] = (1, 3)
    service_commitment_scale_range: tuple[float, float] = (0.8, 1.05)
    job_due_date_factor_range: tuple[float, float] = (0.85, 1.15)
    arrival_intensity: str = "medium"
    arrival_profile_pool: tuple[str, ...] = ARRIVAL_PROFILES
    latent_entity_concentration: float = 4.0
    bucket_size: int = 10
    forecast_horizon_buckets: int = 6
    forecast_update_interval: int = 10
    forecast_error_scale: float = 1.0
    forecast_horizon_uncertainty_growth: float = 0.15
    forecast_risk_kappa: float = 1.0
    r5_epsilon: float = 1e-9
    alpha: float = 1.0
    beta: float = 1.0
    group_name: str = "generated"

    def __post_init__(self) -> None:
        if min(self.num_instances, self.num_jobs, self.num_machines, self.num_entities) <= 0:
            raise ValueError("instance counts must be positive")
        if self.num_entities > self.num_jobs:
            raise ValueError("num_entities cannot exceed num_jobs")
        for name, bounds in (
            ("ops_per_job", self.ops_per_job),
            ("quantity_range", self.quantity_range),
            ("proc_time_range", self.proc_time_range),
            ("eligible_machines_range", self.eligible_machines_range),
        ):
            _validate_int_range(name, bounds)
        if self.eligible_machines_range[0] > self.num_machines:
            raise ValueError("eligible machine lower bound cannot exceed num_machines")
        if not 1 <= self.checkpoints_per_entity[0] <= self.checkpoints_per_entity[1] <= 3:
            raise ValueError("checkpoints_per_entity bounds must be between 1 and 3")
        if self.checkpoint_tightness <= 0.0:
            raise ValueError("checkpoint_tightness must be positive")
        _validate_float_range(
            "service_commitment_scale_range", self.service_commitment_scale_range
        )
        _validate_float_range("job_due_date_factor_range", self.job_due_date_factor_range)
        if not self.checkpoint_profile_pool:
            raise ValueError("checkpoint_profile_pool must not be empty")
        if any(profile not in DEFAULT_CHECKPOINT_PROFILES for profile in self.checkpoint_profile_pool):
            raise ValueError("checkpoint_profile_pool contains an unknown profile")
        if self.arrival_intensity not in ARRIVAL_WINDOW_FACTORS:
            raise ValueError("arrival_intensity must be one of: static, low, medium, high")
        if not self.arrival_profile_pool or any(
            profile not in ARRIVAL_PROFILES for profile in self.arrival_profile_pool
        ):
            raise ValueError("arrival_profile_pool contains an unknown profile")
        if self.latent_entity_concentration <= 0.0:
            raise ValueError("latent_entity_concentration must be positive")
        if min(self.bucket_size, self.forecast_horizon_buckets, self.forecast_update_interval) <= 0:
            raise ValueError("forecast bucket, horizon, and update interval must be positive")
        if min(self.forecast_error_scale, self.forecast_horizon_uncertainty_growth, self.forecast_risk_kappa) < 0.0:
            raise ValueError("forecast error, growth, and kappa must be non-negative")
        if self.r5_epsilon <= 0.0 or self.alpha <= 0.0 or self.beta <= 0.0:
            raise ValueError("epsilon and objective weights must be positive")


def generate_instance(
    config: InstanceConfig,
    *,
    seed: int,
    instance_index: int = 0,
    structure_seed: int | None = None,
    latent_seed: int | None = None,
    actual_seed: int | None = None,
    commitment_seed: int | None = None,
    due_seed: int | None = None,
    forecast_seed: int | None = None,
    checkpoint_time_base: int | None = None,
) -> FJSPInstance:
    master_seed = seed + instance_index * 10_007
    stream_offset = instance_index * 10_007

    def resolved(override: int | None, default_offset: int) -> int:
        return (
            master_seed + default_offset
            if override is None
            else int(override) + stream_offset
        )

    resolved_seeds = {
        "structure": resolved(structure_seed, 1_009),
        "latent": resolved(latent_seed, 2_003),
        "actual": resolved(actual_seed, 3_001),
        "commitment": resolved(commitment_seed, 4_001),
        "due": resolved(due_seed, 5_003),
        "forecast": resolved(forecast_seed, 91_003),
    }
    structure_rng = random.Random(resolved_seeds["structure"])
    latent_rng = random.Random(resolved_seeds["latent"])
    actual_rng = random.Random(resolved_seeds["actual"])
    commitment_rng = random.Random(resolved_seeds["commitment"])
    due_rng = random.Random(resolved_seeds["due"])
    machines = tuple(Machine(machine_id=index) for index in range(config.num_machines))
    raw_jobs: list[dict] = []
    for job_id in range(config.num_jobs):
        operations: list[Operation] = []
        for sequence_index in range(structure_rng.randint(*config.ops_per_job)):
            eligible_count = min(
                config.num_machines,
                structure_rng.randint(*config.eligible_machines_range),
            )
            machine_ids = structure_rng.sample(
                range(config.num_machines), eligible_count
            )
            operations.append(
                Operation(
                    op_id=job_id * 100 + sequence_index,
                    job_id=job_id,
                    sequence_index=sequence_index,
                    alternatives=tuple(
                        OperationAlternative(
                            machine_id,
                            structure_rng.randint(*config.proc_time_range),
                        )
                        for machine_id in machine_ids
                    ),
                )
            )
        raw_jobs.append(
            {
                "job_id": job_id,
                "operations": tuple(operations),
            }
        )

    # The checkpoint calendar is an exogenous service commitment.  It must not
    # inspect the realized job routes, alternatives, or processing times, since
    # those attributes include jobs that have not yet been released.  The
    # default base therefore uses only distributional parameters known before
    # the episode is realized.  An explicit base is supported for replaying a
    # frozen external commitment calendar (for example, a historical benchmark).
    checkpoint_base = (
        _external_checkpoint_time_base(config)
        if checkpoint_time_base is None
        else int(checkpoint_time_base)
    )
    if checkpoint_base <= 0:
        raise ValueError("checkpoint_time_base must be positive")
    latent = _build_latent_demand_scenario(
        config, checkpoint_base=checkpoint_base, rng=latent_rng
    )
    entity_assignments = list(range(config.num_entities))
    entity_assignments.extend(
        _sample_index(latent.expected_quantity_by_entity, actual_rng)
        for _ in range(config.num_jobs - config.num_entities)
    )
    actual_rng.shuffle(entity_assignments)
    latest_release = _latest_release_time(config.arrival_intensity, checkpoint_base)
    for raw, entity_id in zip(raw_jobs, entity_assignments, strict=True):
        bucket_index = _sample_index(
            latent.bucket_means_by_entity[entity_id], actual_rng
        )
        bucket_start = latent.bucket_starts[bucket_index]
        bucket_end = min(
            bucket_start + config.bucket_size - 1,
            latest_release,
        )
        raw["entity_id"] = entity_id
        raw["quantity"] = actual_rng.randint(*config.quantity_range)
        raw["release_time"] = actual_rng.randint(bucket_start, bucket_end)
    minimum_checkpoints, maximum_checkpoints = config.checkpoints_per_entity
    profile_pool = [
        DEFAULT_CHECKPOINT_PROFILES[profile_id]
        for profile_id in config.checkpoint_profile_pool
        if minimum_checkpoints
        <= len(DEFAULT_CHECKPOINT_PROFILES[profile_id].time_ratios)
        <= maximum_checkpoints
    ]
    if not profile_pool:
        raise ValueError(
            "checkpoint_profile_pool has no profile within checkpoints_per_entity"
        )
    commitment_rng.shuffle(profile_pool)
    entities: list[ServiceEntity] = []
    commitment_scales: list[float] = []
    for entity_id in range(config.num_entities):
        profile = profile_pool[entity_id % len(profile_pool)]
        final_time = max(
            len(profile.time_ratios),
            round(checkpoint_base * (0.9 + 0.2 * entity_id / max(1, config.num_entities - 1))),
        )
        times = _checkpoint_times(final_time, profile.time_ratios)
        commitment_scale = commitment_rng.uniform(
            *config.service_commitment_scale_range
        )
        commitment_scales.append(commitment_scale)
        final_target = max(
            1.0,
            latent.expected_quantity_by_entity[entity_id] * commitment_scale,
        )
        checkpoints = tuple(
            Checkpoint(time, final_target * target_ratio, weight)
            for time, target_ratio, weight in zip(
                times, profile.target_ratios, profile.weights, strict=True
            )
        )
        entities.append(ServiceEntity(entity_id, checkpoints, profile.profile_id))

    jobs: list[Job] = []
    for raw in raw_jobs:
        entity = entities[raw["entity_id"]]
        minimum_work = sum(operation.min_processing_time for operation in raw["operations"])
        due_date = max(
            raw["release_time"] + minimum_work,
            round(
                entity.checkpoints[-1].time
                * due_rng.uniform(*config.job_due_date_factor_range)
            ),
        )
        jobs.append(
            Job(
                job_id=raw["job_id"],
                entity_id=raw["entity_id"],
                release_time=raw["release_time"],
                due_date=due_date,
                quantity=raw["quantity"],
                operations=raw["operations"],
            )
        )

    max_time = max(
        max(checkpoint.time for entity in entities for checkpoint in entity.checkpoints),
        max(job.release_time for job in jobs),
    )
    forecast_provider = build_forecast_provider_from_latent_demand(
        latent.bucket_records(),
        entity_ids=range(config.num_entities),
        bucket_size=config.bucket_size,
        horizon_buckets=config.forecast_horizon_buckets,
        update_interval=config.forecast_update_interval,
        max_time=max_time,
        error_scale=config.forecast_error_scale,
        horizon_uncertainty_growth=config.forecast_horizon_uncertainty_growth,
        seed=resolved_seeds["forecast"],
    )
    return FJSPInstance(
        jobs=tuple(jobs),
        entities=tuple(entities),
        machines=machines,
        alpha=config.alpha,
        beta=config.beta,
        forecast_provider=forecast_provider,
        forecast_risk_kappa=config.forecast_risk_kappa,
        r5_epsilon=config.r5_epsilon,
        metadata={
            "group_name": config.group_name,
            "seed": seed,
            "instance_index": instance_index,
            "random_stream_seeds": resolved_seeds,
            "service_commitment_source": "latent_expected_demand",
            "checkpoint_timeline_source": (
                "exogenous_config_expectation"
                if checkpoint_time_base is None
                else "external_fixed_calendar"
            ),
            "checkpoint_time_base": checkpoint_base,
            "future_job_structure_used_to_build_checkpoint_timeline": False,
            "actual_arrival_source": "independent_latent_realization",
            "forecast_source": "independent_latent_observation",
            "realization_used_to_build_commitment": False,
            "realization_used_to_build_forecast": False,
            "latent_expected_quantity_by_entity": (
                latent.expected_quantity_by_entity
            ),
            "service_commitment_scales": tuple(commitment_scales),
            "arrival_profile_by_entity": latent.arrival_profile_by_entity,
            "bucket_size": config.bucket_size,
            "forecast_horizon_buckets": config.forecast_horizon_buckets,
            "forecast_update_interval": config.forecast_update_interval,
        },
    )


def _expected_discrete_minimum(
    bounds: tuple[int, int], sample_count: int
) -> float:
    """Expected minimum of iid discrete uniforms, using config data only."""

    lower, upper = bounds
    support = upper - lower + 1
    return float(
        lower
        + sum(
            ((upper - threshold + 1) / support) ** sample_count
            for threshold in range(lower + 1, upper + 1)
        )
    )


def _external_checkpoint_time_base(config: InstanceConfig) -> int:
    """Build a causal checkpoint horizon without consulting realized jobs."""

    eligible_draws = range(
        config.eligible_machines_range[0],
        config.eligible_machines_range[1] + 1,
    )
    expected_min_processing = fmean(
        _expected_discrete_minimum(
            config.proc_time_range,
            min(config.num_machines, eligible_count),
        )
        for eligible_count in eligible_draws
    )
    expected_operations_per_job = fmean(config.ops_per_job)
    expected_total_min_work = (
        config.num_jobs * expected_operations_per_job * expected_min_processing
    )
    reference_makespan = max(
        1, round(expected_total_min_work / config.num_machines)
    )
    return max(3, round(reference_makespan * config.checkpoint_tightness))


def _validate_int_range(name: str, bounds: tuple[int, int]) -> None:
    if len(bounds) != 2 or bounds[0] <= 0 or bounds[1] < bounds[0]:
        raise ValueError(f"{name} must be a positive ordered pair")


def _validate_float_range(name: str, bounds: tuple[float, float]) -> None:
    if len(bounds) != 2 or bounds[0] <= 0.0 or bounds[1] < bounds[0]:
        raise ValueError(f"{name} must be a positive ordered pair")


def _checkpoint_times(final_time: int, ratios: tuple[float, ...]) -> tuple[int, ...]:
    times = [max(1, round(final_time * ratio)) for ratio in ratios]
    times[-1] = final_time
    for index in range(1, len(times)):
        times[index] = max(times[index], times[index - 1] + 1)
    if times[-1] != final_time:
        raise ValueError("checkpoint horizon is too short for a strict sequence")
    return tuple(times)


ARRIVAL_WINDOW_FACTORS = {
    "static": 0.0,
    "low": 0.80,
    "medium": 0.50,
    "high": 0.25,
}


def _latest_release_time(intensity: str, checkpoint_base: int) -> int:
    return max(0, round(checkpoint_base * ARRIVAL_WINDOW_FACTORS[intensity]))


def _sample_index(weights: tuple[float, ...] | list[float], rng: random.Random) -> int:
    total = sum(weights)
    if total <= 0.0:
        raise ValueError("sampling weights must have a positive sum")
    threshold = rng.random() * total
    cumulative = 0.0
    for index, weight in enumerate(weights):
        cumulative += weight
        if threshold <= cumulative:
            return index
    return len(weights) - 1


def _temporal_weights(
    profile: str,
    bucket_starts: tuple[int, ...],
    *,
    latest_release: int,
    bucket_size: int,
    rng: random.Random,
) -> tuple[float, ...]:
    horizon = max(1, latest_release + 1)
    burst_center = rng.random() if profile == "bursty" else 0.5
    weights = []
    for start in bucket_starts:
        stop = min(start + bucket_size, horizon)
        width = max(1, stop - start)
        midpoint = (start + stop) / (2.0 * horizon)
        if profile == "uniform":
            shape = 1.0
        elif profile == "front_loaded":
            shape = 1.25 - midpoint
        elif profile == "back_loaded":
            shape = 0.25 + midpoint
        elif profile == "bursty":
            shape = 0.20 + 3.0 * math.exp(
                -((midpoint - burst_center) / 0.16) ** 2
            )
        else:
            raise ValueError(f"unknown arrival profile {profile}")
        weights.append(width * shape)
    return tuple(weights)


def _build_latent_demand_scenario(
    config: InstanceConfig,
    *,
    checkpoint_base: int,
    rng: random.Random,
) -> LatentDemandScenario:
    """Draw latent demand without consulting the eventual job realization."""

    latest_release = _latest_release_time(config.arrival_intensity, checkpoint_base)
    bucket_starts = tuple(range(0, latest_release + 1, config.bucket_size)) or (0,)
    entity_weights = tuple(
        rng.gammavariate(config.latent_entity_concentration, 1.0)
        for _ in range(config.num_entities)
    )
    nominal_total = config.num_jobs * fmean(config.quantity_range)
    weight_total = sum(entity_weights)
    expected_by_entity = tuple(
        nominal_total * weight / weight_total for weight in entity_weights
    )
    profiles = tuple(
        rng.choice(config.arrival_profile_pool) for _ in range(config.num_entities)
    )
    means = []
    for entity_id, profile in enumerate(profiles):
        weights = _temporal_weights(
            profile,
            bucket_starts,
            latest_release=latest_release,
            bucket_size=config.bucket_size,
            rng=rng,
        )
        total = sum(weights)
        means.append(
            tuple(expected_by_entity[entity_id] * weight / total for weight in weights)
        )
    return LatentDemandScenario(
        bucket_starts,
        expected_by_entity,
        tuple(means),
        profiles,
    )
