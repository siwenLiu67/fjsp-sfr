"""Global Paper-A reward-reference calibration on independent instances."""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

from fjsp_sfr.core.models import FJSPInstance
from fjsp_sfr.core.objective import compute_objective
from fjsp_sfr.dispatch.classic_rules import ClassicRule, ClassicRuleKind
from fjsp_sfr.env.ca_rule_selection import CAPPORuleSelectionEnv


GLOBAL_REWARD_REFERENCE = "global_calibration_spt_atc_mean"


@dataclass(frozen=True)
class CalibrationInstanceResult:
    seed: int
    instance_sha256: str
    f_spt: float
    f_atc: float


@dataclass(frozen=True)
class GlobalRewardScale:
    """Auditable fixed scalar shared by every formal learning run."""

    f_ref: float
    calibration_seeds: tuple[int, ...]
    instances: tuple[CalibrationInstanceResult, ...]
    payload_sha256: str
    calibration_partition: str = "calibration_center"
    scenario: str = "center"


def _canonical_json(payload: object) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _instance_fingerprint(instance: FJSPInstance) -> str:
    """Hash objective-relevant realized data without serializing providers."""

    payload = {
        "jobs": [
            {
                "job_id": job.job_id,
                "entity_id": job.entity_id,
                "release_time": job.release_time,
                "due_date": job.due_date,
                "quantity": job.quantity,
                "operations": [
                    {
                        "op_id": operation.op_id,
                        "sequence_index": operation.sequence_index,
                        "alternatives": [
                            [alternative.machine_id, alternative.processing_time]
                            for alternative in operation.alternatives
                        ],
                    }
                    for operation in job.operations
                ],
            }
            for job in instance.jobs
        ],
        "entities": [
            {
                "entity_id": entity.entity_id,
                "checkpoints": [
                    [checkpoint.time, checkpoint.required_quantity, checkpoint.weight]
                    for checkpoint in entity.checkpoints
                ],
            }
            for entity in instance.entities
        ],
        "machines": [machine.machine_id for machine in instance.machines],
        "alpha": instance.alpha,
        "beta": instance.beta,
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _evaluate_fixed_rule(instance: FJSPInstance, rule: ClassicRule) -> float:
    rollout = deepcopy(instance)
    env = CAPPORuleSelectionEnv(rollout)
    env.reset()
    while env.current_metrics and env.current_metrics.is_active:
        result = env.step_principle(0, rule)
        if result.terminated:
            break
    return float(compute_objective(rollout, env._state).Z)


def compute_global_reward_scale(
    instances: Sequence[FJSPInstance] | Iterable[FJSPInstance],
    calibration_seeds: Sequence[int] | Iterable[int],
    *,
    calibration_partition: str = "calibration_center",
    scenario: str = "center",
) -> GlobalRewardScale:
    """Compute Eq. (global-reward-scale) from SPT and ATC(K=2)."""

    instance_rows = tuple(instances)
    seeds = tuple(int(seed) for seed in calibration_seeds)
    if not instance_rows:
        raise ValueError("global F_ref calibration requires at least one instance")
    if len(instance_rows) != len(seeds):
        raise ValueError("calibration seed count must match instance count")
    if len(set(seeds)) != len(seeds):
        raise ValueError("calibration seeds must be unique")
    spt = ClassicRule("SPT", ClassicRuleKind.SPT)
    atc = ClassicRule("ATC_K2", ClassicRuleKind.ATC, atc_k=2.0)
    results = tuple(
        CalibrationInstanceResult(
            seed=seed,
            instance_sha256=_instance_fingerprint(instance),
            f_spt=_evaluate_fixed_rule(instance, spt),
            f_atc=_evaluate_fixed_rule(instance, atc),
        )
        for seed, instance in zip(seeds, instance_rows, strict=True)
    )
    f_ref = max(
        1.0,
        sum(row.f_spt + row.f_atc for row in results) / (2.0 * len(results)),
    )
    body = {
        "schema": "paperA_global_reward_scale_v1",
        "reference": GLOBAL_REWARD_REFERENCE,
        "formula": "max(1, mean_I[(F_SPT(I)+F_ATC_K2(I))/2])",
        "f_ref": f_ref,
        "calibration_seeds": list(seeds),
        "instances": [asdict(row) for row in results],
        "calibration_partition": calibration_partition,
        "scenario": scenario,
    }
    digest = hashlib.sha256(_canonical_json(body).encode("utf-8")).hexdigest()
    return GlobalRewardScale(
        f_ref,
        seeds,
        results,
        digest,
        calibration_partition,
        scenario,
    )


def write_global_reward_scale(
    path: str | Path, scale: GlobalRewardScale
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "paperA_global_reward_scale_v1",
        "reference": GLOBAL_REWARD_REFERENCE,
        "formula": "max(1, mean_I[(F_SPT(I)+F_ATC_K2(I))/2])",
        "f_ref": scale.f_ref,
        "calibration_seeds": list(scale.calibration_seeds),
        "instances": [asdict(row) for row in scale.instances],
        "calibration_partition": scale.calibration_partition,
        "scenario": scale.scenario,
        "payload_sha256": scale.payload_sha256,
    }
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output


def load_global_reward_scale(path: str | Path) -> GlobalRewardScale:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    digest = payload.pop("payload_sha256", None)
    if not isinstance(digest, str):
        raise ValueError("reward-scale manifest is missing payload_sha256")
    expected = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
    if digest != expected:
        raise ValueError("reward-scale manifest hash mismatch")
    if payload.get("reference") != GLOBAL_REWARD_REFERENCE:
        raise ValueError("reward-scale manifest uses an incompatible reference")
    f_ref = float(payload["f_ref"])
    if not math.isfinite(f_ref) or f_ref < 1.0:
        raise ValueError("reward-scale manifest F_ref must be finite and at least one")
    rows = tuple(
        CalibrationInstanceResult(
            seed=int(row["seed"]),
            instance_sha256=str(row["instance_sha256"]),
            f_spt=float(row["f_spt"]),
            f_atc=float(row["f_atc"]),
        )
        for row in payload["instances"]
    )
    if not rows:
        raise ValueError("reward-scale manifest has no calibration rows")
    recomputed = max(
        1.0,
        sum(row.f_spt + row.f_atc for row in rows) / (2.0 * len(rows)),
    )
    if not math.isclose(f_ref, recomputed, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("reward-scale manifest F_ref does not match its rows")
    seeds = tuple(int(seed) for seed in payload["calibration_seeds"])
    if seeds != tuple(row.seed for row in rows):
        raise ValueError("reward-scale manifest seed rows are inconsistent")
    return GlobalRewardScale(
        f_ref,
        seeds,
        rows,
        digest,
        str(payload.get("calibration_partition", "")),
        str(payload.get("scenario", "")),
    )


__all__ = [
    "CalibrationInstanceResult",
    "GLOBAL_REWARD_REFERENCE",
    "GlobalRewardScale",
    "compute_global_reward_scale",
    "load_global_reward_scale",
    "write_global_reward_scale",
]
