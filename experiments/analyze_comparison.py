"""Statistical analysis and publication artifacts for formal comparisons."""

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
import numpy as np

from fjsp_sfr.experiments.build_comparison import build_comparison, load_groups
from fjsp_sfr.generation.scenario_generator import load_scenario_config

try:
    from scipy import stats
except ImportError:  # Results explicitly record unavailable optional analyses.
    stats = None


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict], columns: tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader(); writer.writerows(rows)


def bootstrap_ci(values: list[float], *, samples: int = 5000, seed: int = 20261025) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    rng, size = random.Random(seed), len(values)
    means = sorted(statistics.fmean(rng.choice(values) for _ in range(size)) for _ in range(samples))
    return means[int(samples * 0.025)], means[min(samples - 1, int(samples * 0.975))]


def instance_means(rows: list[dict[str, str]]) -> dict[str, dict[tuple[str, str], float]]:
    buckets: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        buckets[(row["method"], row["scenario_id"], row["instance_id"])].append(float(row["F"]))
    result: dict[str, dict[tuple[str, str], float]] = defaultdict(dict)
    for (method, scenario, instance), values in buckets.items():
        result[method][(scenario, instance)] = statistics.fmean(values)
    return result


def main_result(summary: list[dict[str, str]]) -> list[dict]:
    values: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in summary:
        values[row["method"]].append(row)
    output = []
    for method, rows in values.items():
        record = {"method": method, "label": rows[0]["label"], "seeds": len(rows)}
        for metric in ("F", "TT", "WSF"):
            samples = [float(row[metric]) for row in rows]
            record[f"{metric}_mean"] = statistics.fmean(samples)
            record[f"{metric}_std"] = statistics.stdev(samples) if len(samples) > 1 else 0.0
        output.append(record)
    return sorted(output, key=lambda row: row["F_mean"])


def paired_table(rows: list[dict[str, str]], proposed: str) -> list[dict]:
    means = instance_means(rows)
    if proposed not in means:
        return []
    output = []
    for method, values in means.items():
        if method == proposed:
            continue
        keys = sorted(set(means[proposed]) & set(values))
        differences = [means[proposed][key] - values[key] for key in keys]
        low, high = bootstrap_ci(differences)
        output.append({
            "proposed": proposed, "comparison": method, "n": len(keys),
            "paired_difference": statistics.fmean(differences) if differences else float("nan"),
            "ci_low": low, "ci_high": high,
            "win_rate": sum(value < 0 for value in differences) / len(differences) if differences else float("nan"),
            "tie_rate": sum(value == 0 for value in differences) / len(differences) if differences else float("nan"),
        })
    return output


def rank_tests(rows: list[dict[str, str]]) -> tuple[list[dict], list[dict], dict]:
    means = instance_means(rows)
    methods = sorted(means)
    common = sorted(set.intersection(*(set(means[method]) for method in methods))) if methods else []
    matrix = np.array([[means[method][key] for method in methods] for key in common], dtype=float)
    average_ranks = np.zeros(len(methods))
    for row in matrix:
        order = np.argsort(row)
        ranks = np.empty(len(methods)); ranks[order] = np.arange(1, len(methods) + 1)
        average_ranks += ranks
    if len(common):
        average_ranks /= len(common)
    rank_rows = [{"method": method, "average_rank": float(rank)} for method, rank in zip(methods, average_ranks, strict=True)]
    friedman = {"statistic": float("nan"), "p_value": float("nan"), "n": len(common), "methods": len(methods)}
    wilcoxon_rows = []
    raw = []
    if stats is not None and len(common) >= 2 and len(methods) >= 3:
        result = stats.friedmanchisquare(*(matrix[:, index] for index in range(len(methods))))
        friedman.update(statistic=float(result.statistic), p_value=float(result.pvalue))
        for left in range(len(methods)):
            for right in range(left + 1, len(methods)):
                result = stats.wilcoxon(matrix[:, left], matrix[:, right], zero_method="zsplit")
                raw.append((left, right, float(result.statistic), float(result.pvalue)))
        ordered = sorted(range(len(raw)), key=lambda index: raw[index][3])
        adjusted = [1.0] * len(raw)
        running = 0.0
        for rank, index in enumerate(ordered):
            corrected = min(1.0, raw[index][3] * (len(raw) - rank))
            running = max(running, corrected); adjusted[index] = running
        for index, (left, right, statistic, p_value) in enumerate(raw):
            wilcoxon_rows.append({"left": methods[left], "right": methods[right], "statistic": statistic, "p_value": p_value, "holm_p": adjusted[index]})
    q_alpha = 2.569
    friedman["nemenyi_cd"] = q_alpha * math.sqrt(len(methods) * (len(methods) + 1) / max(1, 6 * len(common)))
    return rank_rows, wilcoxon_rows, friedman


def anova_and_tukey(rows: list[dict[str, str]]) -> tuple[list[dict], list[dict]]:
    means = instance_means(rows)
    methods = sorted(means)
    common = sorted(set.intersection(*(set(means[method]) for method in methods))) if methods else []
    records = [{"method": method, "instance": f"{key[0]}:{key[1]}", "F": means[method][key]} for method in methods for key in common]
    anova_rows, tukey_rows = [], []
    try:
        import pandas as pd
        import statsmodels.api as sm
        from statsmodels.formula.api import ols
        from statsmodels.stats.multicomp import pairwise_tukeyhsd
        frame = pd.DataFrame(records)
        table = sm.stats.anova_lm(ols("F ~ C(method) + C(instance)", data=frame).fit(), typ=2)
        for term, values in table.iterrows():
            anova_rows.append({"term": term, "sum_sq": values.get("sum_sq"), "df": values.get("df"), "F": values.get("F"), "p_value": values.get("PR(>F)")})
        tukey = pairwise_tukeyhsd(frame["F"], frame["method"])
        for row in tukey.summary().data[1:]:
            tukey_rows.append(dict(zip(("group1", "group2", "mean_diff", "p_adj", "lower", "upper", "reject"), row, strict=True)))
    except ImportError:
        anova_rows.append({"term": "unavailable", "sum_sq": "", "df": "", "F": "", "p_value": "statsmodels/pandas missing"})
    return anova_rows, tukey_rows


def confidence_rows(rows: list[dict[str, str]]) -> list[dict]:
    means = instance_means(rows)
    output = []
    for method, values in means.items():
        samples = list(values.values()); mean = statistics.fmean(samples)
        se = statistics.stdev(samples) / math.sqrt(len(samples)) if len(samples) > 1 else 0.0
        output.append({"method": method, "mean": mean, "ci_low": mean - 1.96 * se, "ci_high": mean + 1.96 * se, "lsd_low": mean - 1.645 * se, "lsd_high": mean + 1.645 * se})
    return sorted(output, key=lambda row: row["mean"])


def plot_intervals(rows: list[dict], output: Path, low: str, high: str, title: str) -> None:
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    figure, axis = plt.subplots(figsize=(9, max(4, 0.45 * len(rows))))
    positions = np.arange(len(rows)); means = np.array([row["mean"] for row in rows])
    errors = np.array([[row["mean"] - row[low] for row in rows], [row[high] - row["mean"] for row in rows]])
    axis.errorbar(means, positions, xerr=errors, fmt="o", capsize=3)
    axis.set_yticks(positions, [row["method"] for row in rows]); axis.invert_yaxis()
    axis.set_xlabel("F"); axis.set_title(title); axis.grid(axis="x", alpha=0.25)
    figure.tight_layout(); figure.savefig(output, dpi=220); plt.close(figure)


def plot_cd(rank_rows: list[dict], cd: float, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(9, 2.8))
    for index, row in enumerate(sorted(rank_rows, key=lambda item: item["average_rank"])):
        axis.scatter(row["average_rank"], 0.2 + index % 2 * 0.2)
        axis.text(row["average_rank"], 0.25 + index % 2 * 0.2, row["method"], rotation=30, ha="center")
    axis.plot([1, 1 + cd], [0.8, 0.8], color="black", linewidth=3); axis.text(1 + cd / 2, 0.84, f"CD={cd:.2f}", ha="center")
    axis.set_xlim(0.8, max((row["average_rank"] for row in rank_rows), default=1) + 0.8); axis.set_ylim(0, 1); axis.set_yticks([]); axis.set_xlabel("平均序值")
    figure.tight_layout(); figure.savefig(output, dpi=220); plt.close(figure)


def runtime_table(summary: list[dict[str, str]]) -> list[dict]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in summary: grouped[row["method"]].append(row)
    return [{"method": method, "decision_seconds": statistics.fmean(float(row["decision_seconds_mean"]) for row in items), "training_or_search_hours": statistics.fmean(float(row["training_seconds"]) for row in items) / 3600.0} for method, items in grouped.items()]


def plot_convergence(result_root: Path, evaluation_set: str, methods: list[dict], output: Path) -> None:
    figure, axis = plt.subplots(figsize=(9, 5.5))
    for method in methods:
        curves = []
        for path in sorted((result_root / evaluation_set / method["directory"]).glob("seed_*/train_curve.csv")):
            rows = read_rows(path)
            if rows:
                curves.append(([int(row["update"]) for row in rows], [float(row["val_F_greedy"]) for row in rows]))
        if not curves:
            continue
        for x_values, values in curves:
            axis.plot(x_values, values, color="tab:blue", alpha=0.15, linewidth=0.8)
        common = min(len(values) for _x, values in curves)
        x_values = curves[0][0][:common]
        matrix = np.array([values[:common] for _x, values in curves])
        mean = matrix.mean(axis=0)
        smooth = np.convolve(mean, np.ones(5) / 5, mode="same") if common >= 5 else mean
        ci = 1.96 * matrix.std(axis=0) / math.sqrt(matrix.shape[0])
        axis.plot(x_values, smooth, label=method["label"], linewidth=2)
        axis.fill_between(x_values, mean - ci, mean + ci, alpha=0.12)
    axis.set_xlabel("Update"); axis.set_ylabel("Validation F"); axis.set_title("收敛曲线")
    axis.grid(alpha=0.25); axis.legend(); figure.tight_layout(); figure.savefig(output, dpi=220); plt.close(figure)


def plot_factor_effects(rows: list[dict], output: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(10, 7))
    for axis, factor in zip(axes.flat, ("machines", "new_jobs", "load", "ddt"), strict=True):
        selected = [row for row in rows if row["factor"] == factor]
        axis.plot([row["level"] for row in selected], [row["mean_F"] for row in selected], marker="o")
        axis.set_title(factor); axis.set_xlabel("level"); axis.set_ylabel("mean F"); axis.grid(alpha=0.25)
    figure.tight_layout(); figure.savefig(output, dpi=220); plt.close(figure)


def l9_tables(rows: list[dict[str, str]], proposed: str) -> tuple[list[dict], list[dict], list[dict]]:
    means = instance_means(rows); baseline_methods = [name for name in ("spt", "atc") if name in means]
    scenarios = sorted({row["scenario_id"] for row in rows})
    improvement, wins = [], []
    for scenario in scenarios:
        keys = sorted(key for key in means.get(proposed, {}) if key[0] == scenario)
        if not keys or not baseline_methods: continue
        diffs, relative = [], []
        for key in keys:
            baseline = min(means[name][key] for name in baseline_methods)
            diff = means[proposed][key] - baseline; diffs.append(diff); relative.append(-diff / max(baseline, 1e-12) * 100)
        low, high = bootstrap_ci(diffs)
        improvement.append({"scenario": scenario, "improvement_percent": statistics.fmean(relative), "difference": statistics.fmean(diffs), "ci_low": low, "ci_high": high})
        for method, values in means.items():
            if method == proposed: continue
            common = [key for key in keys if key in values]
            wins.append({"scenario": scenario, "comparison": method, "win_rate": sum(means[proposed][key] < values[key] for key in common) / len(common) if common else float("nan")})
    config = load_scenario_config()["scenarios"]
    factor_rows = []
    for factor in ("machines", "new_jobs", "load", "ddt"):
        levels: dict[float, list[float]] = defaultdict(list)
        for scenario in scenarios:
            for key, value in means.get(proposed, {}).items():
                if key[0] == scenario: levels[float(config[scenario][factor])].append(value)
        factor_rows.extend({"factor": factor, "level": level, "mean_F": statistics.fmean(values)} for level, values in sorted(levels.items()))
    return improvement, factor_rows, wins


def write_markdown(path: Path, title: str, tables: list[tuple[str, list[dict]]]) -> None:
    lines = [f"# {title}", ""]
    for heading, rows in tables:
        lines.extend([f"## {heading}", ""])
        if not rows: lines.extend(["No completed data.", ""]); continue
        columns = tuple(rows[0]); lines.append("| " + " | ".join(columns) + " |"); lines.append("|" + "---|" * len(columns))
        lines.extend("| " + " | ".join(str(row[column]) for column in columns) + " |" for row in rows); lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_tex(path: Path, rows: list[dict]) -> None:
    line_break = chr(92) * 2
    lines = ["\\begin{tabular}{lrrr}", "\\toprule", f"Method & F & TT & WSF {line_break}", "\\midrule"]
    lines.extend(
        f"{row['label']} & {row['F_mean']:.3f} $\\pm$ {row['F_std']:.3f} "
        f"& {row['TT_mean']:.3f} & {row['WSF_mean']:.3f} " + line_break
        for row in rows
    )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", dest="evaluation_set", choices=("center", "l9"), required=True)
    parser.add_argument("--group", choices=tuple(load_groups()), required=True)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--comparison", type=Path, default=Path("comparison"))
    args = parser.parse_args()
    summary_path, long_path = build_comparison(args.results, args.evaluation_set, args.group, args.comparison)
    destination = summary_path.parent; summary, long_rows = read_rows(summary_path), read_rows(long_path)
    methods = load_groups()[args.group]; proposed = next((item["id"] for item in methods if item.get("proposed")), methods[-1]["id"])
    main_rows = main_result(summary); paired = paired_table(long_rows, proposed)
    rank_rows, wilcoxon, friedman = rank_tests(long_rows); anova, tukey = anova_and_tukey(long_rows)
    intervals, runtimes = confidence_rows(long_rows), runtime_table(summary)
    write_rows(destination / "main_results.csv", main_rows, tuple(main_rows[0]) if main_rows else ("method",))
    write_rows(destination / "paired_comparisons.csv", paired, tuple(paired[0]) if paired else ("proposed", "comparison"))
    write_rows(destination / "anova.csv", anova, tuple(anova[0]) if anova else ("term",))
    write_rows(destination / "tukey_hsd.csv", tukey, tuple(tukey[0]) if tukey else ("group1", "group2"))
    write_rows(destination / "average_ranks.csv", rank_rows, ("method", "average_rank"))
    write_rows(destination / "wilcoxon_holm.csv", wilcoxon, tuple(wilcoxon[0]) if wilcoxon else ("left", "right"))
    write_rows(destination / "runtime.csv", runtimes, ("method", "decision_seconds", "training_or_search_hours"))
    (destination / "friedman.json").write_text(json.dumps(friedman, indent=2) + "\n", encoding="utf-8")
    plot_intervals(intervals, destination / "mean_f_ci.png", "ci_low", "ci_high", "方法 F 均值及 95% 置信区间")
    plot_intervals(intervals, destination / "mean_f_lsd.png", "lsd_low", "lsd_high", "方法 F 均值及 LSD 区间")
    plot_cd(rank_rows, friedman["nemenyi_cd"], destination / "critical_difference.png")
    plot_convergence(args.results, args.evaluation_set, methods, destination / "convergence.png")
    extra_tables = []
    if args.evaluation_set == "l9":
        improvement, factors, wins = l9_tables(long_rows, proposed)
        write_rows(destination / "l9_improvement.csv", improvement, tuple(improvement[0]) if improvement else ("scenario",))
        write_rows(destination / "l9_factor_effects.csv", factors, tuple(factors[0]) if factors else ("factor",))
        write_rows(destination / "l9_win_rates.csv", wins, tuple(wins[0]) if wins else ("scenario",))
        plot_factor_effects(factors, destination / "l9_factor_effects.png")
        extra_tables = [("L9 improvement", improvement), ("L9 factor effects", factors), ("L9 win rates", wins)]
    tables = [("Main results", main_rows), ("Paired comparisons", paired), ("Runtime", runtimes), ("Average ranks", rank_rows), *extra_tables]
    write_markdown(destination / "analysis.md", f"{args.evaluation_set} / {args.group}", tables)
    write_tex(destination / "main_results.tex", main_rows)


if __name__ == "__main__":
    main()
