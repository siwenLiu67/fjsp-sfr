"""Aggregate completed formal runs into paper tables and figures 6-3--6-5."""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_PROJECT_PARENT = _PROJECT_ROOT.parent
for _import_path in (str(_PROJECT_ROOT), str(_PROJECT_PARENT)):
    if _import_path not in sys.path:
        sys.path.insert(0, _import_path)

import argparse
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


plt.rcParams["font.sans-serif"] = [
    "Noto Sans CJK SC",
    "Hiragino Sans GB",
    "Songti SC",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


def load_runs(root: Path) -> list[dict]:
    runs = []
    for path in sorted(root.glob("*/summary.json")):
        item = json.loads(path.read_text(encoding="utf-8"))
        item["summary_path"] = str(path)
        csv_path = Path(item["per_instance_csv"])
        with csv_path.open(encoding="utf-8", newline="") as handle:
            item["rows"] = [
                {"seed": int(row["seed"]), **{key: float(row[key]) for key in ("F", "TT", "WSF")}}
                for row in csv.DictReader(handle)
            ]
        runs.append(item)
    return runs


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row)) if rows else ["status"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows or [{"status": "no matching completed runs"}])


def group_runs(runs, predicate=lambda _run: True):
    groups = defaultdict(list)
    for run in runs:
        if predicate(run):
            key = (
                run["difficulty"], run["eval_scale"], run["library"], run["selector"],
                run["mask_checkpoint"], run["mask_forecast"],
            )
            groups[key].append(run)
    return groups


def aggregate_instances(group: list[dict], metric="F") -> dict[int, float]:
    by_seed = defaultdict(list)
    for run in group:
        for row in run["rows"]:
            by_seed[row["seed"]].append(row[metric])
    return {seed: statistics.fmean(values) for seed, values in by_seed.items()}


def bootstrap(differences, seed=20261023, repeats=5000):
    if not differences:
        return math.nan, math.nan, math.nan
    rng = random.Random(seed)
    samples = sorted(
        statistics.fmean(rng.choice(differences) for _ in differences)
        for _ in range(repeats)
    )
    return (
        statistics.fmean(differences),
        samples[int(0.025 * repeats)],
        samples[min(repeats - 1, int(0.975 * repeats))],
    )


def main_table(runs):
    groups = group_runs(
        runs,
        lambda run: run["difficulty"] == "service_heavy"
        and run["scale"] == run["eval_scale"]
        and not run["mask_checkpoint"]
        and not run["mask_forecast"],
    )
    rows = []
    references = {}
    for key, group in groups.items():
        difficulty, scale, library, selector, _mc, _mf = key
        values = [run["means"]["F"] for run in group]
        tt = [run["means"]["TT"] for run in group]
        wsf = [run["means"]["WSF"] for run in group]
        row = {
            "scale": scale,
            "library": library,
            "selector": selector,
            "F_mean": statistics.fmean(values),
            "F_seed_std": statistics.pstdev(values),
            "TT_M": statistics.fmean(tt) if scale == "M" else "",
            "WSF_M": statistics.fmean(wsf) if scale == "M" else "",
        }
        rows.append(row)
        if selector == "hgt_ppo" and library in {"classic4", "handwritten6"}:
            references[(scale, library)] = row["F_mean"]
    for row in rows:
        classic = references.get((row["scale"], "classic4"))
        manual = references.get((row["scale"], "handwritten6"))
        row["improvement_vs_classic_percent"] = (
            (classic - row["F_mean"]) / classic * 100 if classic else ""
        )
        row["improvement_vs_handwritten_percent"] = (
            (manual - row["F_mean"]) / manual * 100 if manual else ""
        )
    return rows, groups


def paired_table(groups):
    rows = []
    for scale in ("S", "M", "L"):
        method_key = ("service_heavy", scale, "pool_mix", "hgt_ppo", False, False)
        if method_key not in groups:
            continue
        method = aggregate_instances(groups[method_key])
        for key, group in groups.items():
            if key[1] != scale or key == method_key:
                continue
            comparison = aggregate_instances(group)
            shared = sorted(set(method) & set(comparison))
            diffs = [method[seed] - comparison[seed] for seed in shared]
            mean, low, high = bootstrap(diffs)
            rows.append(
                {
                    "scale": scale,
                    "comparison_library": key[2],
                    "comparison_selector": key[3],
                    "paired_difference": mean,
                    "ci_low": low,
                    "ci_high": high,
                    "win_rate": sum(value < 0 for value in diffs) / len(diffs),
                }
            )
    return rows


def source_and_selector_tables(main_rows):
    sources = [row for row in main_rows if row["selector"] == "hgt_ppo"]
    selectors = [row for row in main_rows if row["library"] == "pool_mix"]
    return sources, selectors


def ablation_table(runs):
    return [
        {
            "scale": run["eval_scale"],
            "seed": run["seed"],
            "mask_checkpoint": run["mask_checkpoint"],
            "mask_forecast": run["mask_forecast"],
            **run["means"],
        }
        for run in runs
        if run["mask_checkpoint"] or run["mask_forecast"]
    ]


def robustness_table(runs):
    rows = []
    for run in runs:
        if run["library"] not in {"p3b1_mix", "p3b2_mix", "p3b3_mix", "pool_mix"}:
            continue
        blocks = [run["rows"][index : index + 30] for index in range(0, len(run["rows"]), 30)]
        block_means = [statistics.fmean(row["F"] for row in block) for block in blocks if block]
        rows.append(
            {
                "scale": run["eval_scale"],
                "library": run["library"],
                "seed": run["seed"],
                "F": run["means"]["F"],
                "block_mean": statistics.fmean(block_means),
                "block_std": statistics.pstdev(block_means) if len(block_means) > 1 else 0.0,
                "block_range": max(block_means) - min(block_means) if block_means else 0.0,
                "generalization_X": run["scale"] == "M" and run["eval_scale"] == "X",
            }
        )
    return rows


def computation_table(runs, experiments_dir: Path):
    rows = [
        {
            "kind": "training_evaluation",
            "library": run["library"],
            "scale": run["scale"],
            "selector": run["selector"],
            "seed": run["seed"],
            "training_seconds": run.get("training_seconds", 0.0),
            "decision_seconds_mean": run.get("decision_seconds_mean", 0.0),
        }
        for run in runs
    ]
    for path in sorted(experiments_dir.glob("generate_*_manifest.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("dry_run"):
            continue
        usage = manifest.get("llm_token_usage", {})
        rows.append(
            {
                "kind": "llm_generation",
                "library": manifest.get("tag", ""),
                "scale": manifest.get("scale", ""),
                "selector": "",
                "seed": "",
                "training_seconds": "",
                "decision_seconds_mean": "",
                "llm_calls": manifest.get("llm_api_request_count", ""),
                "prompt_tokens": usage.get("prompt_tokens", ""),
                "completion_tokens": usage.get("completion_tokens", ""),
                "total_tokens": usage.get("total_tokens", ""),
                "generation_seconds": manifest.get(
                    "generation_seconds_this_invocation", ""
                ),
            }
        )
    return rows


def make_figures(output: Path, paired, runs, robustness):
    fig, ax = plt.subplots(figsize=(8, max(3, len(paired) * 0.25)))
    if paired:
        y = list(range(len(paired)))
        means = [row["paired_difference"] for row in paired]
        low = [means[i] - paired[i]["ci_low"] for i in y]
        high = [paired[i]["ci_high"] - means[i] for i in y]
        ax.errorbar(means, y, xerr=[low, high], fmt="o")
        ax.set_yticks(y, [f"{row['scale']}:{row['comparison_library']}/{row['comparison_selector']}" for row in paired])
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_title("图6-3 本文方法的配对差与95%置信区间")
    fig.tight_layout(); fig.savefig(output / "fig6_3.png", dpi=200); plt.close(fig)

    pool = [run for run in runs if run["library"] == "pool_mix" and run["selector"] == "hgt_ppo"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    labels = [f"{run['eval_scale']}-{run['seed']}" for run in pool]
    axes[0].bar(labels, [run.get("total_variation", 0.0) for run in pool])
    axes[1].bar(labels, [run.get("switch_rate", 0.0) for run in pool])
    usage = defaultdict(int)
    for run in pool:
        for name, count in run.get("rule_usage", {}).items(): usage[name] += count
    axes[2].bar(list(usage), list(usage.values()))
    for axis in axes: axis.tick_params(axis="x", rotation=90)
    axes[0].set_title("TV"); axes[1].set_title("切换率"); axes[2].set_title("规则选择分布")
    fig.suptitle("图6-4 规则选择与状态敏感性")
    fig.tight_layout(); fig.savefig(output / "fig6_4.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    if robustness:
        labels = [f"{row['scale']}:{row['library']}:s{row['seed']}" for row in robustness]
        ax.bar(labels, [row["block_std"] for row in robustness])
        ax.tick_params(axis="x", rotation=90)
    ax.set_title("图6-5 30实例分块波动与X规模泛化")
    fig.tight_layout(); fig.savefig(output / "fig6_5.png", dpi=200); plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="experiments/formal")
    parser.add_argument("--output", default="experiments/formal_aggregate")
    args = parser.parse_args(argv)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    runs = load_runs(Path(args.input))
    main_rows, groups = main_table(runs)
    paired = paired_table(groups)
    sources, selectors = source_and_selector_tables(main_rows)
    ablations = ablation_table(runs)
    robustness = robustness_table(runs)
    write_csv(output / "table6_5_main.csv", main_rows)
    write_csv(output / "table6_6_paired.csv", paired)
    write_csv(output / "table6_7_rule_sources.csv", sources)
    write_csv(output / "table6_7_selectors.csv", selectors)
    write_csv(output / "table6_8_information_ablation.csv", ablations)
    write_csv(output / "table6_9_robustness_generalization.csv", robustness)
    write_csv(
        output / "table6_computation.csv",
        computation_table(runs, Path(args.input).resolve().parent),
    )
    make_figures(output, paired, runs, robustness)
    print(f"aggregated {len(runs)} runs into {output}")


if __name__ == "__main__":
    main()
