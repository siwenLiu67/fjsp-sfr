"""Editable scale and difficulty profiles for formal experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fjsp_sfr.generation.instance_generator import InstanceConfig, generate_instance


CONFIG_PATH = Path(__file__).with_name("scale_config.json")


def load_scale_config(path: Path = CONFIG_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def scaled_instance_config(
    scale: str = "M", difficulty: str = "balanced"
) -> InstanceConfig:
    payload = load_scale_config()
    try:
        size = payload["scales"][scale]
        tuning = payload["difficulties"][difficulty]
    except KeyError as error:
        raise ValueError(f"unknown scale or difficulty: {error.args[0]}") from error
    return InstanceConfig(
        num_instances=1,
        num_jobs=int(size["jobs"]),
        num_machines=int(size["machines"]),
        num_entities=int(size["entities"]),
        ops_per_job=(2, 4),
        checkpoints_per_entity=(2, 2),
        checkpoint_tightness=float(tuning["checkpoint_tightness"]),
        checkpoint_profile_pool=("balanced_2", "early_pressure_2"),
        eligible_machines_range=(1, min(3, int(size["machines"]))),
        service_commitment_scale_range=tuple(
            float(value) for value in tuning["service_commitment_scale_range"]
        ),
        job_due_date_factor_range=tuple(
            float(value) for value in tuning["job_due_date_factor_range"]
        ),
        checkpoint_weight_scale=float(
            tuning.get("checkpoint_weight_scale", 1.0)
        ),
        group_name=f"formal_{scale}_{difficulty}",
    )


def make_scaled_instance(
    seed: int, *, scale: str = "M", difficulty: str = "balanced"
):
    return generate_instance(
        scaled_instance_config(scale, difficulty), seed=int(seed)
    )


def add_scale_arguments(parser: argparse.ArgumentParser) -> None:
    payload = load_scale_config()
    parser.add_argument(
        "--scale", choices=tuple(payload["scales"]), default="M"
    )
    parser.add_argument(
        "--difficulty",
        choices=tuple(payload["difficulties"]),
        default="balanced",
    )


__all__ = [
    "CONFIG_PATH",
    "add_scale_arguments",
    "load_scale_config",
    "make_scaled_instance",
    "scaled_instance_config",
]
