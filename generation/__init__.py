"""Benchmark instance and rolling-forecast generation."""

from .instance_generator import (
    ARRIVAL_WINDOW_FACTORS,
    ARRIVAL_PROFILES,
    DEFAULT_CHECKPOINT_PROFILES,
    CheckpointProfile,
    InstanceConfig,
    LatentDemandScenario,
    generate_instance,
)
from .scale_config import make_scaled_instance, scaled_instance_config

__all__ = [
    "ARRIVAL_WINDOW_FACTORS",
    "ARRIVAL_PROFILES",
    "DEFAULT_CHECKPOINT_PROFILES",
    "CheckpointProfile",
    "InstanceConfig",
    "LatentDemandScenario",
    "generate_instance",
    "make_scaled_instance",
    "scaled_instance_config",
]
