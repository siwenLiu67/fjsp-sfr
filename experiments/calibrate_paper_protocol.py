"""Paper A P4.2 center-scenario calibration; no automatic relaxation of thresholds.

Usage: python -m fjsp_sfr.experiments.calibrate_paper_protocol --output-dir /tmp/p42
Only independent center calibration seeds are used. All 60 combinations are evaluated.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import tempfile
from copy import deepcopy
from pathlib import Path
from statistics import fmean

from fjsp_sfr.dispatch.classic_rules import ClassicRule, ClassicRuleKind
from fjsp_sfr.env.ca_rule_selection import CAPPORuleSelectionEnv
from fjsp_sfr.generation.scenario_generator import (
    CONFIG_PATH, load_scenario_config, make_scenario_instance, seed_values,
)

RHO = (0.70, 0.85, 1.00, 1.15)
THETA = (0.45, 0.55, 0.65)
OMEGA = (0.5, 1.0, 2.0, 4.0, 8.0)
KAPPA = (0.25, 0.50, 0.75, 1.00)
RULES = (
    ClassicRule("SPT", ClassicRuleKind.SPT),
    ClassicRule("ATC", ClassicRuleKind.ATC, atc_k=2.0),
)


def simulate_classic_timeline(rule, instance) -> dict[int, int]:
    """Compute one true, full scheduling rollout; never use future jobs in dispatch."""
    env = CAPPORuleSelectionEnv(deepcopy(instance))
    env.reset()
    while env.current_metrics and env.current_metrics.is_active:
        result = env.step_principle(0, rule)
        if result.terminated:
            break
    completed = dict(env._state.completed_jobs)
    if len(completed) != len(instance.jobs):
        raise ValueError("calibration rule did not complete all jobs")
    return completed


def evaluate_fixed_timeline(instance, completed: dict[int, int]) -> dict:
    """Reprice a precomputed SPT/ATC schedule against public candidate contracts."""
    tt = sum(max(0, completed[job.job_id] - job.due_date) for job in instance.jobs)
    wsf = 0.0
    early = 0
    for entity in instance.entities:
        jobs = instance.jobs_of_entity(entity.entity_id)
        for index, checkpoint in enumerate(entity.checkpoints):
            delivered = sum(
                job.quantity for job in jobs if completed[job.job_id] <= checkpoint.time
            )
            shortage = max(0.0, checkpoint.required_quantity - delivered)
            wsf += checkpoint.weight * shortage
            if index == 0 and shortage > 0.0:
                early += 1
    return {
        "F": tt + wsf, "TT": tt, "WSF": wsf,
        "early_shortfall_count": early, "early_checkpoint_count": len(instance.entities),
    }


def unavoidable_supply_shortfall(instance) -> float:
    """Full-realization lower bound for OFFLINE diagnostics only."""
    return sum(
        entity.checkpoints[-1].weight * max(
            0.0, entity.checkpoints[-1].required_quantity
            - sum(job.quantity for job in instance.jobs if job.entity_id == entity.entity_id)
        )
        for entity in instance.entities
    )


def score_candidate(runs: dict[str, list[dict]], rho: float, theta: float, omega: float) -> dict:
    """Choose a single reference rule by mean F; report ratios of sums."""
    if any(not runs[rule.name] for rule in RULES):
        raise ValueError("a candidate needs nonempty SPT and ATC results")
    selected = min(RULES, key=lambda rule: (fmean(row["F"] for row in runs[rule.name]), RULES.index(rule)))
    items = runs[selected.name]
    sum_f = sum(row["F"] for row in items)
    sum_wsf = sum(row["WSF"] for row in items)
    sum_l = sum(row["L"] for row in items)
    sum_cp = sum(row["early_checkpoint_count"] for row in items)
    r_wsf = sum_wsf / sum_f if sum_f > 0 else math.nan
    r_irr = sum_l / sum_wsf if sum_wsf > 0 else math.nan
    p_early = sum(row["early_shortfall_count"] for row in items) / sum_cp if sum_cp > 0 else math.nan
    passed = (
        0.35 <= r_wsf <= 0.50
        and 0.20 <= p_early <= 0.80
        and r_irr <= 0.40
    )
    return {
        "rho": rho, "theta": theta, "omega": omega, "reference_rule": selected.name,
        "mean_F": fmean(row["F"] for row in items),
        "mean_TT": fmean(row["TT"] for row in items),
        "mean_WSF": fmean(row["WSF"] for row in items),
        "mean_L": fmean(row["L"] for row in items),
        "R_WSF": r_wsf, "P_early": p_early, "R_irr": r_irr,
        "pass_wsf": 0.35 <= r_wsf <= 0.50,
        "pass_early": 0.20 <= p_early <= 0.80,
        "pass_irr": r_irr <= 0.40,
        "eligible": passed,
    }


def calibrate(output_dir: Path, *, config_path: Path = CONFIG_PATH, seeds: tuple[int, ...] | None = None) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    base = load_scenario_config(config_path)
    if seeds is None:
        seeds = seed_values("calibration_center", config_path)
    if not seeds:
        raise ValueError("no calibration seeds")
    # Explicitly reject testing/training/selection seed overlap.
    reserved = set()
    for name, bounds in base["seeds"].items():
        if name != "calibration_center":
            reserved.update(range(int(bounds[0]), int(bounds[1]) + 1))
    if set(seeds) & reserved:
        raise ValueError("calibration seeds overlap other experiment partitions")

    # Classical dispatch scores depend on processing time/due date, not public
    # service targets. Compute each seed/rule schedule once. Reprice the frozen
    # completed-job timeline against all 60 public contract combinations.
    timelines = {}
    for seed in seeds:
        instance = make_scenario_instance("center", seed, config_path=config_path)
        for rule in RULES:
            timelines[(seed, rule.name)] = simulate_classic_timeline(rule, instance)

    # Validate schedule invariance on predeclared boundary cases before reuse.
    guard_cfg = json.loads(json.dumps(base))
    guard_cfg["checkpoint"]["commitment_ratio"] = RHO[-1]
    guard_cfg["checkpoint"]["first_time_ratio"] = THETA[0]
    guard_cfg["checkpoint"]["weights"] = [OMEGA[-1], 3 * OMEGA[-1]]
    with tempfile.TemporaryDirectory(prefix="paperA-p42-guard-") as folder:
        path = Path(folder) / "scenario.json"
        path.write_text(json.dumps(guard_cfg, sort_keys=True), encoding="utf-8")
        for seed in seeds[:min(2, len(seeds))]:
            altered = make_scenario_instance("center", seed, config_path=path)
            for rule in RULES:
                if simulate_classic_timeline(rule, altered) != timelines[(seed, rule.name)]:
                    raise RuntimeError("classical rollout changed under service contract; disable timeline reuse")

    rows = []
    for rho, theta in itertools.product(RHO, THETA):
        cfg = json.loads(json.dumps(base))
        cp = cfg["checkpoint"]
        cp["commitment_ratio"] = rho
        cp["first_time_ratio"] = theta
        cp["first_demand_ratio"] = 0.55
        cp["weights"] = [1.0, 3.0]
        cp["weight_scale"] = 1.0
        with tempfile.TemporaryDirectory(prefix="paperA-p42-") as folder:
            path = Path(folder) / "scenario.json"
            path.write_text(json.dumps(cfg, sort_keys=True), encoding="utf-8")
            baseline: dict[str, list[dict]] = {rule.name: [] for rule in RULES}
            for seed in seeds:
                instance = make_scenario_instance("center", seed, config_path=path)
                lower = unavoidable_supply_shortfall(instance)
                for rule in RULES:
                    metrics = evaluate_fixed_timeline(instance, timelines[(seed, rule.name)])
                    metrics["L"] = lower
                    baseline[rule.name].append(metrics)
            for omega in OMEGA:
                adjusted: dict[str, list[dict]] = {}
                for name, results in baseline.items():
                    adjusted[name] = [
                        {
                            **row,
                            "WSF": omega * row["WSF"],
                            "L": omega * row["L"],
                            "F": row["TT"] + omega * row["WSF"],
                        }
                        for row in results
                    ]
                rows.append(score_candidate(adjusted, rho, theta, omega))
    if len(rows) != 60:
        raise AssertionError("expected exactly 60 calibration rows")
    destination = output_dir / "paperA_service_calibration_60.csv"
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    valid = [row for row in rows if row["eligible"]]
    winner = min(
        valid,
        key=lambda row: (
            abs(row["R_WSF"] - 0.425), row["R_irr"],
            row["rho"], row["theta"], row["omega"],
        ),
        default=None,
    )
    result = {
        "status": "PASS" if winner is not None else "FAIL",
        "generator": "scenario_generator_v2",
        "seed_count": len(seeds),
        "seed_start": seeds[0], "seed_end": seeds[-1],
        "total_candidates": len(rows), "qualified_candidates": len(valid),
        "pass_wsf": sum(row["pass_wsf"] for row in rows),
        "pass_early": sum(row["pass_early"] for row in rows),
        "pass_irr": sum(row["pass_irr"] for row in rows),
        "winner": winner,
        "downstream_gates": "BLOCKED: do not calibrate kappa, F_ref or train until author confirms",
    }
    (output_dir / "paperA_p42_calibration_result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(calibrate(args.output_dir), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
