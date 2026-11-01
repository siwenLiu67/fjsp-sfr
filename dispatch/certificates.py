"""Deterministic checkpoint-shortfall certificates from Paper A Sec. 4.6.

The online construction accepts :class:`OnlineProblemView` deliberately: this
prevents a caller from accidentally using sampled, unreleased jobs.  The
privileged construction is separate and its immutable record is intended only
for a training Critic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Mapping

from fjsp_sfr.core.models import FJSPInstance
from fjsp_sfr.core.online import OnlineProblemView
from fjsp_sfr.core.state import ScheduleState
from fjsp_sfr.dispatch.ca_metrics import analyze_checkpoint_capacity


CheckpointKey = tuple[int, int]
_TOLERANCE = 1e-9


class CertificateError(RuntimeError):
    """Base class for a deterministic-certificate contract violation."""


class CertificateContradictionError(CertificateError):
    """Raised when historical intersection produces an empty interval."""


@dataclass(frozen=True)
class ShortfallInterval:
    """Closed deterministic interval for a final checkpoint shortfall."""

    lower: float
    upper: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.lower) or not math.isfinite(self.upper):
            raise CertificateError("certificate endpoints must be finite")
        if self.lower < -_TOLERANCE:
            raise CertificateError("certificate lower endpoint must be non-negative")
        if self.lower > self.upper + _TOLERANCE:
            raise CertificateContradictionError(
                f"empty certificate interval [{self.lower}, {self.upper}]"
            )

    @property
    def width(self) -> float:
        return max(0.0, self.upper - self.lower)


def intersect_intervals(
    previous: ShortfallInterval | None,
    current: ShortfallInterval,
    *,
    checkpoint_key: CheckpointKey | None = None,
) -> ShortfallInterval:
    """Intersect one current interval with its trajectory history."""

    if previous is None:
        return current
    lower = max(previous.lower, current.lower)
    upper = min(previous.upper, current.upper)
    if lower > upper + _TOLERANCE:
        where = "" if checkpoint_key is None else f" for checkpoint {checkpoint_key}"
        raise CertificateContradictionError(
            "certificate history is contradictory"
            f"{where}: previous=[{previous.lower}, {previous.upper}], "
            f"current=[{current.lower}, {current.upper}]"
        )
    # Inputs are integer-valued in the formal generator.  This branch only
    # removes an insignificant floating-point inversion without widening.
    if lower > upper:
        lower = upper
    return ShortfallInterval(lower, upper)


@dataclass(frozen=True)
class OnlineCheckpointCertificate:
    """Public certificate and its auditable Sec. 4.6 components."""

    checkpoint_key: CheckpointKey
    checkpoint_time: int
    required_quantity: float
    completed_supply: float
    confirmed_supply: float
    visible_supply_bound: float
    future_supply_bound: float
    instantaneous: ShortfallInterval
    historical: ShortfallInterval


@dataclass(frozen=True)
class PrivilegedCheckpointCertificate:
    """Critic-only certificate using the sampled future exogenous arrivals."""

    checkpoint_key: CheckpointKey
    future_supply_bound: float
    future_job_count: int
    instantaneous: ShortfallInterval
    historical: ShortfallInterval


@dataclass(frozen=True)
class PrivilegedCertificateRecord:
    """Immutable training-only record; never embedded in public observations."""

    decision_time: int
    certificates: tuple[PrivilegedCheckpointCertificate, ...] = ()

    def checkpoint(
        self, entity_id: int, checkpoint_index: int
    ) -> PrivilegedCheckpointCertificate:
        key = (entity_id, checkpoint_index)
        for certificate in self.certificates:
            if certificate.checkpoint_key == key:
                return certificate
        raise KeyError(f"no active privileged certificate for {key}")


@dataclass
class CertificateHistory:
    """Per-episode historical intersections for online and private bounds."""

    online: dict[CheckpointKey, ShortfallInterval] = field(default_factory=dict)
    privileged: dict[CheckpointKey, ShortfallInterval] = field(default_factory=dict)

    def update_online(
        self, key: CheckpointKey, interval: ShortfallInterval
    ) -> ShortfallInterval:
        result = intersect_intervals(self.online.get(key), interval, checkpoint_key=key)
        self.online[key] = result
        return result

    def update_privileged(
        self, key: CheckpointKey, interval: ShortfallInterval
    ) -> ShortfallInterval:
        result = intersect_intervals(
            self.privileged.get(key), interval, checkpoint_key=key
        )
        self.privileged[key] = result
        return result


def _public_future_bounds(problem: OnlineProblemView) -> tuple[int | None, float | None, float]:
    """Return remaining count, q_max and W_min using public metadata only.

    Missing industrial bounds deliberately produce a rigorous loose bound:
    unknown count/quantity is represented by ``None`` and later maps to an
    infinite future-supply upper bound; unknown work maps to zero.
    """

    raw = problem.metadata.get("certificate_public_bounds", {})
    if not isinstance(raw, Mapping):
        raise CertificateError("certificate_public_bounds must be a mapping")
    total = raw.get("planned_total_jobs")
    if total is None:
        remaining = None
    else:
        total = int(total)
        remaining = total - len(problem.jobs)
        if remaining < 0:
            raise CertificateError(
                "visible job count exceeds the published planned job count"
            )
    q_max = raw.get("quantity_upper_bound")
    if q_max is not None:
        q_max = float(q_max)
        if not math.isfinite(q_max) or q_max <= 0.0:
            raise CertificateError("published quantity upper bound must be positive")
    work_min = float(raw.get("minimum_job_work", 0.0))
    if not math.isfinite(work_min) or work_min < 0.0:
        raise CertificateError("published minimum job work must be non-negative")
    return remaining, q_max, work_min


def _online_future_supply_bound(
    problem: OnlineProblemView, checkpoint_time: int
) -> float:
    remaining, q_max, work_min = _public_future_bounds(problem)
    if remaining == 0 or problem.current_time + work_min > checkpoint_time:
        return 0.0
    if remaining is None or q_max is None:
        return math.inf
    return float(remaining) * q_max


def _visible_supply_components(
    problem: OnlineProblemView,
    state: ScheduleState,
    key: CheckpointKey,
) -> tuple[float, float, float, float]:
    entity_id, checkpoint_index = key
    checkpoint = problem.get_entity(entity_id).checkpoints[checkpoint_index]
    analysis = analyze_checkpoint_capacity(
        problem, state, entity_id, checkpoint_index
    )
    gap = analysis.residual_gap
    completed_supply = max(0.0, float(checkpoint.required_quantity) - gap)
    if not analysis.additional_jobs:
        visible_bound = 0.0
    else:
        nu = analysis.minimum_work_per_quantity
        if nu is None or not math.isfinite(nu) or nu <= 0.0:
            raise CertificateError(
                f"checkpoint {key} has visible contributors without positive nu"
            )
        visible_bound = min(
            analysis.additional_quantity,
            analysis.available_capacity / nu,
        )
    return completed_supply, analysis.confirmed_supply, visible_bound, gap


def compute_online_certificate(
    problem: OnlineProblemView,
    state: ScheduleState,
    entity_id: int,
    checkpoint_index: int,
    *,
    history: ShortfallInterval | None = None,
) -> OnlineCheckpointCertificate | None:
    """Compute the leakage-safe online certificate for one active checkpoint."""

    if not isinstance(problem, OnlineProblemView):
        raise TypeError("online certificates require an OnlineProblemView")
    key = (entity_id, checkpoint_index)
    checkpoint = problem.get_entity(entity_id).checkpoints[checkpoint_index]
    completed, confirmed, visible_bound, gap = _visible_supply_components(
        problem, state, key
    )
    if state.current_time >= checkpoint.time or gap <= 0.0:
        return None
    future_bound = _online_future_supply_bound(problem, checkpoint.time)
    lower = max(
        0.0,
        float(checkpoint.required_quantity)
        - completed
        - confirmed
        - visible_bound
        - future_bound,
    )
    upper = max(
        0.0,
        float(checkpoint.required_quantity) - completed - confirmed,
    )
    instantaneous = ShortfallInterval(lower, upper)
    historical = intersect_intervals(history, instantaneous, checkpoint_key=key)
    return OnlineCheckpointCertificate(
        checkpoint_key=key,
        checkpoint_time=checkpoint.time,
        required_quantity=float(checkpoint.required_quantity),
        completed_supply=completed,
        confirmed_supply=confirmed,
        visible_supply_bound=visible_bound,
        future_supply_bound=future_bound,
        instantaneous=instantaneous,
        historical=historical,
    )


def compute_privileged_certificate(
    instance: FJSPInstance,
    state: ScheduleState,
    online: OnlineCheckpointCertificate,
    *,
    history: ShortfallInterval | None = None,
) -> PrivilegedCheckpointCertificate:
    """Compute the training-only interval from sampled unreleased jobs."""

    entity_id, _checkpoint_index = online.checkpoint_key
    future_jobs = tuple(
        job
        for job in instance.jobs
        if job.release_time > state.current_time
        and job.entity_id == entity_id
        and job.release_time
        + sum(operation.min_processing_time for operation in job.operations)
        <= online.checkpoint_time
    )
    future_bound = float(sum(job.quantity for job in future_jobs))
    if (
        math.isfinite(online.future_supply_bound)
        and future_bound > online.future_supply_bound + _TOLERANCE
    ):
        raise CertificateError(
            f"privileged future bound exceeds published online bound for "
            f"{online.checkpoint_key}"
        )
    lower = max(
        0.0,
        online.required_quantity
        - online.completed_supply
        - online.confirmed_supply
        - online.visible_supply_bound
        - future_bound,
    )
    instantaneous = ShortfallInterval(lower, online.instantaneous.upper)
    historical = intersect_intervals(
        history, instantaneous, checkpoint_key=online.checkpoint_key
    )
    return PrivilegedCheckpointCertificate(
        checkpoint_key=online.checkpoint_key,
        future_supply_bound=future_bound,
        future_job_count=len(future_jobs),
        instantaneous=instantaneous,
        historical=historical,
    )


def update_active_certificates(
    problem: OnlineProblemView,
    instance: FJSPInstance,
    state: ScheduleState,
    history: CertificateHistory,
    *,
    include_privileged: bool = False,
) -> tuple[
    dict[CheckpointKey, OnlineCheckpointCertificate],
    PrivilegedCertificateRecord,
]:
    """Update all active heads after events and before observation building."""

    online_records: dict[CheckpointKey, OnlineCheckpointCertificate] = {}
    privileged_records: list[PrivilegedCheckpointCertificate] = []
    for entity in problem.entities:
        for checkpoint_index, _checkpoint in enumerate(entity.checkpoints):
            key = (entity.entity_id, checkpoint_index)
            online = compute_online_certificate(
                problem,
                state,
                *key,
                history=history.online.get(key),
            )
            if online is None:
                continue
            online_historical = history.update_online(key, online.instantaneous)
            online = replace(online, historical=online_historical)
            online_records[key] = online
            if include_privileged:
                privileged = compute_privileged_certificate(
                    instance,
                    state,
                    online,
                    history=history.privileged.get(key),
                )
                privileged_historical = history.update_privileged(
                    key, privileged.instantaneous
                )
                privileged = replace(
                    privileged, historical=privileged_historical
                )
                if (
                    online.historical.lower
                    > privileged.historical.lower + _TOLERANCE
                ):
                    raise CertificateError(
                        "online lower bound exceeds privileged lower bound"
                    )
                if (
                    privileged.historical.upper
                    > online.historical.upper + _TOLERANCE
                ):
                    raise CertificateError(
                        "privileged upper bound exceeds online upper bound"
                    )
                privileged_records.append(privileged)
    return online_records, PrivilegedCertificateRecord(
        decision_time=state.current_time,
        certificates=tuple(privileged_records),
    )


__all__ = [
    "CertificateContradictionError",
    "CertificateError",
    "CertificateHistory",
    "CheckpointKey",
    "OnlineCheckpointCertificate",
    "PrivilegedCertificateRecord",
    "PrivilegedCheckpointCertificate",
    "ShortfallInterval",
    "compute_online_certificate",
    "compute_privileged_certificate",
    "intersect_intervals",
    "update_active_certificates",
]
