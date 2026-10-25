"""Evaluate the four classic dispatching rules and export F/TT/WSF by seed."""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from pathlib import Path

from fjsp_sfr.core.objective import compute_objective
from fjsp_sfr.dispatch.classic_rules import build_classic_rule_library
from fjsp_sfr.env.ca_rule_selection import CAPPORuleSelectionEnv
from fjsp_sfr.generation.scale_config import add_scale_arguments, make_scaled_instance


def evaluate_rule(rule, instance):
    rollout = deepcopy(instance)
    env = CAPPORuleSelectionEnv(rollout)
    env.reset()
    while env.current_metrics and env.current_metrics.is_active:
        result = env.step_principle(0, rule)
        if result.terminated:
            break
    return compute_objective(rollout, env._state)


def evaluate_classic(
    *, scale: str, difficulty: str, seeds: tuple[int, ...], output: Path
) -> Path:
    library = build_classic_rule_library()
    rows = []
    for seed in seeds:
        instance = make_scaled_instance(seed, scale=scale, difficulty=difficulty)
        for rule in library.rules:
            objective = evaluate_rule(rule, instance)
            rows.append(
                (
                    seed,
                    rule.name,
                    objective.Z,
                    objective.total_tardiness,
                    objective.weighted_checkpoint_shortfall,
                )
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("seed", "rule", "F", "TT", "WSF"))
        writer.writerows(rows)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_scale_arguments(parser)
    parser.add_argument("--seed_start", type=int, default=100)
    parser.add_argument("--seed_end", type=int, default=149)
    parser.add_argument("--output", default="experiments/classic4_results.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = evaluate_classic(
        scale=args.scale,
        difficulty=args.difficulty,
        seeds=tuple(range(args.seed_start, args.seed_end + 1)),
        output=Path(args.output),
    )
    print(path.resolve())


if __name__ == "__main__":
    main()
