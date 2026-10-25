"""Calibrate the shared checkpoint parameters on center validation instances."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from statistics import fmean

from fjsp_sfr.experiments.scenario_tools import classic_rules_with_default, evaluate_rule, summarize
from fjsp_sfr.generation.scenario_generator import CONFIG_PATH, load_scenario_config, make_scenario_instance, scenario_names, seed_values


TIGHTNESS_GRID = (0.55, 0.65, 0.75, 0.85, 0.95)
COMMITMENT_GRID = (0.75, 0.85, 0.95, 1.05, 1.15)


def _instances(config_path: Path, seeds: tuple[int, ...], scenario: str = "center"):
    return [make_scenario_instance(scenario, seed, config_path=config_path) for seed in seeds]


def _evaluate(instances) -> dict[str, dict[str, float]]:
    return {
        rule.name: summarize([evaluate_rule(rule, instance) for instance in instances])
        for rule in classic_rules_with_default()
    }


def _better_spt_atc(table: dict[str, dict[str, float]]) -> tuple[str, float]:
    name = min(("SPT", "ATC_K2"), key=lambda item: table[item]["F"])
    return name, table[name]["WSF"] / max(table[name]["F"], 1e-12)


def calibrate(config_path: Path = CONFIG_PATH) -> tuple[dict, list[dict]]:
    original = load_scenario_config(config_path)
    candidates = []
    for tightness in TIGHTNESS_GRID:
        for commitment in COMMITMENT_GRID:
            trial = deepcopy(original)
            trial["checkpoint"]["tightness"] = tightness
            trial["checkpoint"]["commitment_ratio"] = commitment
            temporary = config_path.with_name(f".{config_path.stem}_calibration.json")
            temporary.write_text(json.dumps(trial, ensure_ascii=False, indent=2), encoding="utf-8")
            table = _evaluate(_instances(temporary, seed_values("calibration_center", config_path)))
            baseline, share = _better_spt_atc(table)
            candidates.append((0.0 if 0.35 <= share <= 0.50 else min(abs(share - 0.35), abs(share - 0.50)), tightness, commitment, baseline, share, table))
            temporary.unlink(missing_ok=True)
    _, tightness, commitment, baseline, share, center_table = min(candidates, key=lambda item: (item[0], abs(item[4] - 0.425)))
    original["checkpoint"]["tightness"] = tightness
    original["checkpoint"]["commitment_ratio"] = commitment
    config_path.write_text(json.dumps(original, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    overview = []
    for scenario in scenario_names(config_path):
        instances = _instances(
            config_path,
            seed_values("calibration_overview", config_path),
            scenario,
        )
        table = _evaluate(instances)
        best_name = min(table, key=lambda name: table[name]["F"])
        best = table[best_name]
        overview.append({
            "scenario": scenario, "best_rule": best_name, **best,
            "measured_load": fmean(instance.metadata["measured_load"] for instance in instances),
            "TT_share": best["TT"] / max(best["F"], 1e-12),
            "WSF_share": best["WSF"] / max(best["F"], 1e-12),
        })
    return {"tightness": tightness, "commitment_ratio": commitment, "baseline": baseline, "WSF_share": share, "center": center_table}, overview


def write_report(calibration: dict, overview: list[dict], output: Path) -> None:
    lines = ["# Scenario calibration", "", f"- tightness: {calibration['tightness']}", f"- commitment ratio: {calibration['commitment_ratio']}", f"- fixed classic baseline: {calibration['baseline']}", f"- WSF/F: {calibration['WSF_share']:.2%}", "", "| Scenario | Best classic | F | utilization | TT/F | WSF/F |", "|---|---|---:|---:|---:|---:|"]
    lines.extend(f"| {row['scenario']} | {row['best_rule']} | {row['F']:.3f} | {row['measured_load']:.3f} | {row['TT_share']:.2%} | {row['WSF_share']:.2%} |" for row in overview)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--out", type=Path, default=Path("experiments/scenario_calibration.md"))
    args = parser.parse_args()
    calibration, overview = calibrate(args.config)
    write_report(calibration, overview, args.out)


if __name__ == "__main__":
    main()
