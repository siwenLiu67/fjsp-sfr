"""Calibrate formal scale profiles with classic and default dispatch rules."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean

from fjsp_sfr.dispatch.classic_rules import build_classic_rule_library
from fjsp_sfr.experiments.build_gated_library import default_rule
from fjsp_sfr.experiments.eval_classic import evaluate_rule
from fjsp_sfr.generation.scale_config import load_scale_config, make_scaled_instance


VALIDATION_SEEDS = tuple(range(100, 150))


def calibrate(
    *, difficulty: str = "balanced", seeds: tuple[int, ...] = VALIDATION_SEEDS
) -> list[dict]:
    rules = [*build_classic_rule_library().rules, default_rule()]
    rows: list[dict] = []
    scale_payload = load_scale_config()["scales"]
    for scale, size in scale_payload.items():
        per_rule = {rule.name: [] for rule in rules}
        checkpoint_counts = []
        for seed in seeds:
            instance = make_scaled_instance(
                seed, scale=scale, difficulty=difficulty
            )
            checkpoint_counts.append(
                sum(len(entity.checkpoints) for entity in instance.entities)
            )
            for rule in rules:
                objective = evaluate_rule(rule, instance)
                per_rule[rule.name].append(objective)
        mean_f = {
            name: fmean(item.Z for item in values)
            for name, values in per_rule.items()
        }
        all_objectives = [item for values in per_rule.values() for item in values]
        mean_tt = fmean(item.total_tardiness for item in all_objectives)
        mean_wsf = fmean(
            item.weighted_checkpoint_shortfall for item in all_objectives
        )
        mean_total = mean_tt + mean_wsf
        best_classic_objectives = []
        for index in range(len(seeds)):
            spt = per_rule["SPT"][index]
            atc = per_rule["ATC_K2"][index]
            best_classic_objectives.append(
                spt if spt.Z <= atc.Z else atc
            )
        best_classic_f = fmean(item.Z for item in best_classic_objectives)
        best_classic_tt = fmean(
            item.total_tardiness for item in best_classic_objectives
        )
        best_classic_wsf = fmean(
            item.weighted_checkpoint_shortfall
            for item in best_classic_objectives
        )
        minimum = min(mean_f.values())
        maximum = max(mean_f.values())
        rows.append(
            {
                "scale": scale,
                "jobs": size["jobs"],
                "machines": size["machines"],
                "entities": size["entities"],
                "checkpoints": fmean(checkpoint_counts),
                "mean_F_by_rule": mean_f,
                "TT_fraction": mean_tt / mean_total if mean_total else 0.0,
                "WSF_fraction": mean_wsf / mean_total if mean_total else 0.0,
                "best_classic_mean_F": best_classic_f,
                "best_classic_TT_fraction": (
                    best_classic_tt / best_classic_f
                    if best_classic_f
                    else 0.0
                ),
                "best_classic_WSF_fraction": (
                    best_classic_wsf / best_classic_f
                    if best_classic_f
                    else 0.0
                ),
                "max_relative_rule_gap": (
                    (maximum - minimum) / minimum if minimum > 0.0 else 0.0
                ),
            }
        )
    return rows


def write_outputs(rows: list[dict], json_path: Path, markdown_path: Path) -> None:
    json_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    rule_names = list(rows[0]["mean_F_by_rule"])
    header = [
        "Scale", "Jobs", "Machines", "Entities", "Checkpoints", *rule_names,
        "TT/F (all)", "WSF/F (all)", "Best(SPT,ATC) F",
        "Best(SPT,ATC) TT/F", "Best(SPT,ATC) WSF/F", "Max rule gap",
    ]
    lines = [
        "# Difficulty calibration",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    for row in rows:
        values = [
            row["scale"], row["jobs"], row["machines"], row["entities"],
            f"{row['checkpoints']:.1f}",
            *(f"{row['mean_F_by_rule'][name]:.2f}" for name in rule_names),
            f"{row['TT_fraction']:.2%}",
            f"{row['WSF_fraction']:.2%}",
            f"{row['best_classic_mean_F']:.2f}",
            f"{row['best_classic_TT_fraction']:.2%}",
            f"{row['best_classic_WSF_fraction']:.2%}",
            f"{row['max_relative_rule_gap']:.2%}",
        ]
        lines.append("| " + " | ".join(map(str, values)) + " |")
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--difficulty", default="balanced")
    parser.add_argument("--seed_start", type=int, default=100)
    parser.add_argument("--seed_end", type=int, default=149)
    parser.add_argument(
        "--json", default="experiments/difficulty_calibration.json"
    )
    parser.add_argument(
        "--markdown", default="experiments/difficulty_calibration.md"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = calibrate(
        difficulty=args.difficulty,
        seeds=tuple(range(args.seed_start, args.seed_end + 1)),
    )
    write_outputs(rows, Path(args.json), Path(args.markdown))
    print(Path(args.markdown).read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
