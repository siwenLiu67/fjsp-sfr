"""Precompute the fixed Paper-A F_ref on the independent calibration split."""

from __future__ import annotations

import argparse
from pathlib import Path

from fjsp_sfr.generation.scenario_generator import make_scenario_instance, seed_values
from fjsp_sfr.rl.reward_scale import (
    compute_global_reward_scale,
    write_global_reward_scale,
)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="center")
    parser.add_argument(
        "--output",
        default="experiments/paperA_global_f_ref.json",
    )
    args = parser.parse_args(argv)
    seeds = seed_values("calibration_center")
    instances = tuple(
        make_scenario_instance(args.scenario, seed) for seed in seeds
    )
    scale = compute_global_reward_scale(
        instances,
        seeds,
        calibration_partition="calibration_center",
        scenario=args.scenario,
    )
    output = write_global_reward_scale(Path(args.output), scale)
    print(f"F_ref={scale.f_ref:.12g}")
    print(f"calibration_instances={len(scale.instances)}")
    print(f"payload_sha256={scale.payload_sha256}")
    print(f"manifest={output}")


if __name__ == "__main__":
    main()
