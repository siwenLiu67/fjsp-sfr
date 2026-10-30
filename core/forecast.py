"""Entity-level rolling arrival forecasts with fixed calendar buckets.

Synthetic forecast construction deliberately accepts only latent bucket means
``(entity_id, bucket_start, mean_quantity)``.  The realized job-arrival stream,
job routes, processing times, eligible machines, and unreleased job objects
never enter forecast generation.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable, Mapping, Protocol


@dataclass(frozen=True)
class ForecastBucket:
    """Forecast quantity and uncertainty for one complete calendar bucket.

    ``d_hat`` and ``sigma`` use the same physical quantity unit as
    ``Job.quantity``, ``Checkpoint.required_quantity``, and residual gaps.  They
    are not job counts and are not pre-truncated to the portion after ``tau``.
    """

    start: int
    end: int
    d_hat: float
    sigma: float

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError("forecast bucket must satisfy 0 <= start < end")
        if self.d_hat < 0.0 or self.sigma < 0.0:
            raise ValueError("forecast mean and uncertainty must be non-negative")

    def conservative_quantity(self, kappa: float) -> float:
        if kappa < 0.0:
            raise ValueError("forecast risk parameter kappa must be non-negative")
        return max(0.0, self.d_hat - kappa * self.sigma)


@dataclass(frozen=True)
class ForecastSnapshot:
    update_epoch: int
    buckets_by_entity: Mapping[int, tuple[ForecastBucket, ...]]

    def __post_init__(self) -> None:
        if self.update_epoch < 0:
            raise ValueError("forecast update epoch must be non-negative")
        reference: tuple[tuple[int, int], ...] | None = None
        for entity_id, buckets in self.buckets_by_entity.items():
            if entity_id < 0:
                raise ValueError("forecast entity id must be non-negative")
            bounds = tuple((bucket.start, bucket.end) for bucket in buckets)
            if bounds != tuple(sorted(bounds)):
                raise ValueError("forecast buckets must be time ordered")
            if len(bounds) != len(set(bounds)):
                raise ValueError("forecast buckets must be unique")
            if reference is None:
                reference = bounds
            elif bounds != reference:
                raise ValueError("all entities must share the same bucket boundaries")

    def for_entity(self, entity_id: int) -> tuple[ForecastBucket, ...]:
        return tuple(self.buckets_by_entity.get(entity_id, ()))


class ForecastProvider(Protocol):
    """Read-only rolling forecast interface used by the online environment."""

    @property
    def update_interval(self) -> int: ...

    def snapshot_at(self, time: int) -> ForecastSnapshot: ...


class FixedForecastProvider:
    """Deterministic sequence of forecasts held constant between update epochs."""

    def __init__(
        self,
        snapshots: Iterable[ForecastSnapshot],
        *,
        update_interval: int,
    ) -> None:
        if update_interval <= 0:
            raise ValueError("forecast update interval must be positive")
        ordered = tuple(sorted(snapshots, key=lambda item: item.update_epoch))
        if not ordered or ordered[0].update_epoch != 0:
            raise ValueError("forecast snapshots must start at epoch zero")
        epochs = tuple(item.update_epoch for item in ordered)
        if len(epochs) != len(set(epochs)):
            raise ValueError("forecast update epochs must be unique")
        self._snapshots = ordered
        self._update_interval = update_interval

    @property
    def update_interval(self) -> int:
        return self._update_interval

    def snapshot_at(self, time: int) -> ForecastSnapshot:
        if time < 0:
            raise ValueError("forecast query time must be non-negative")
        selected = self._snapshots[0]
        for snapshot in self._snapshots[1:]:
            if snapshot.update_epoch > time:
                break
            selected = snapshot
        return selected


def empty_forecast_snapshot(
    entity_ids: Iterable[int], *, update_epoch: int = 0
) -> ForecastSnapshot:
    return ForecastSnapshot(
        update_epoch=update_epoch,
        buckets_by_entity={int(entity_id): () for entity_id in entity_ids},
    )


def build_compound_poisson_forecast_provider(
    *,
    entity_ids: Iterable[int],
    total_arrival_rate: float,
    quantity_mean: float,
    quantity_second_moment: float,
    max_time: int,
) -> FixedForecastProvider:
    """Build the paper's leakage-safe homogeneous compound-Poisson forecast.

    Arrivals are split uniformly across the supplied service entities, so the
    rate for each entity is ``total_arrival_rate / len(entity_ids)``.  Over a
    horizon of length ``s`` the quantity mean and variance are respectively

    ``lambda_r * s * E[q]`` and ``lambda_r * s * E[q**2]``.

    A single calendar bucket covers the published checkpoint horizon.  This is
    an exact representation of the homogeneous process for the consumers in
    :mod:`dispatch.ca_metrics`; it deliberately avoids choosing the manuscript's
    still-undetermined display bucket length ``Delta_b``.  No realized job or
    unreleased-job attribute is accepted by this interface.
    """

    ids = tuple(sorted(set(int(entity_id) for entity_id in entity_ids)))
    if not ids:
        raise ValueError("compound-Poisson forecast requires at least one entity")
    if any(entity_id < 0 for entity_id in ids):
        raise ValueError("forecast entity ids must be non-negative")
    if total_arrival_rate < 0.0:
        raise ValueError("total arrival rate must be non-negative")
    if quantity_mean < 0.0 or quantity_second_moment < 0.0:
        raise ValueError("quantity moments must be non-negative")
    if quantity_second_moment + 1e-12 < quantity_mean * quantity_mean:
        raise ValueError("E[q**2] cannot be smaller than E[q]**2")
    if max_time <= 0:
        raise ValueError("forecast horizon must be positive")

    entity_rate = float(total_arrival_rate) / len(ids)
    horizon_mean = entity_rate * max_time * float(quantity_mean)
    horizon_variance = entity_rate * max_time * float(quantity_second_moment)
    buckets = {
        entity_id: (
            ForecastBucket(
                0,
                int(max_time),
                horizon_mean,
                math.sqrt(horizon_variance),
            ),
        )
        for entity_id in ids
    }
    # There is no stochastic forecast update in the homogeneous model.  The
    # online consumer recomputes the remaining interval from its current time.
    # ``max_time + 1`` is therefore only an interface cadence, not Delta_b.
    return FixedForecastProvider(
        (ForecastSnapshot(update_epoch=0, buckets_by_entity=buckets),),
        update_interval=int(max_time) + 1,
    )


def build_forecast_provider_from_latent_demand(
    latent_bucket_means: Iterable[tuple[int, int, float]],
    *,
    entity_ids: Iterable[int],
    bucket_size: int,
    horizon_buckets: int,
    update_interval: int,
    max_time: int,
    error_scale: float,
    horizon_uncertainty_growth: float = 0.15,
    seed: int = 0,
) -> FixedForecastProvider:
    """Build forecasts as an independent noisy observation of latent demand.

    Actual realized arrivals are not an input.  Both the actual arrival stream
    and this forecast are generated separately from the same latent scenario.
    The third field is expected physical demand quantity for a complete calendar
    bucket, not a realized quantity or arrival count.
    """

    if min(bucket_size, horizon_buckets, update_interval) <= 0:
        raise ValueError("bucket size, horizon, and update interval must be positive")
    if max_time < 0 or error_scale < 0.0 or horizon_uncertainty_growth < 0.0:
        raise ValueError("forecast ranges and uncertainty parameters must be non-negative")

    ids = tuple(sorted(set(int(entity_id) for entity_id in entity_ids)))
    if not ids:
        raise ValueError("forecast requires at least one entity")
    latent: dict[tuple[int, int], float] = {}
    for entity_id, bucket_start, mean_quantity in latent_bucket_means:
        if entity_id not in ids:
            raise ValueError(f"latent demand references unknown entity {entity_id}")
        if bucket_start < 0 or mean_quantity < 0.0:
            raise ValueError("latent bucket start and mean must be non-negative")
        if bucket_start % bucket_size:
            raise ValueError("latent bucket starts must align with bucket_size")
        key = (entity_id, bucket_start)
        if key in latent:
            raise ValueError("latent entity/bucket means must be unique")
        latent[key] = float(mean_quantity)

    rng = random.Random(seed)
    # One latent standardized error is attached to each calendar bucket for the
    # whole episode.  Rolling snapshots revise the uncertainty scale as the
    # bucket approaches, but do not independently redraw its direction/magnitude.
    # This yields a coherent forecast path instead of unrelated successive views.
    bucket_errors = {
        key: rng.gauss(0.0, 1.0)
        for key in sorted(latent)
    }
    snapshots: list[ForecastSnapshot] = []
    final_epoch = max_time + update_interval
    for update_epoch in range(0, final_epoch + 1, update_interval):
        first_start = (update_epoch // bucket_size) * bucket_size
        by_entity: dict[int, tuple[ForecastBucket, ...]] = {}
        for entity_id in ids:
            buckets: list[ForecastBucket] = []
            for horizon_index in range(horizon_buckets):
                start = first_start + horizon_index * bucket_size
                latent_mean = latent.get((entity_id, start), 0.0)
                sigma = error_scale * math.sqrt(latent_mean) * (
                    1.0 + horizon_uncertainty_growth * horizon_index
                )
                standardized_error = bucket_errors.get((entity_id, start), 0.0)
                estimate = max(0.0, latent_mean + standardized_error * sigma)
                buckets.append(ForecastBucket(start, start + bucket_size, estimate, sigma))
            by_entity[entity_id] = tuple(buckets)
        snapshots.append(ForecastSnapshot(update_epoch, by_entity))
    return FixedForecastProvider(snapshots, update_interval=update_interval)
