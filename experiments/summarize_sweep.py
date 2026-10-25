"""Summarize the validation-only learning-rate sweep."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fjsp_sfr.experiments.hparam_sweep import ALGORITHMS, LEARNING_RATES, aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("experiments/sweep"))
    parser.add_argument("--output", type=Path, default=Path("experiments/sweep_summary"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows, _best = aggregate(args.input)
    table = args.output / "learning_rate_summary.csv"
    with table.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("algo", "learning_rate", "validation_F_mean", "seed_std", "completed_seeds"))
        writer.writeheader(); writer.writerows(rows)

    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    figure, axis = plt.subplots(figsize=(8, 5))
    for algo in ALGORITHMS:
        selected = [row for row in rows if row["algo"] == algo]
        axis.errorbar(
            [row["learning_rate"] for row in selected],
            [row["validation_F_mean"] for row in selected],
            yerr=[row["seed_std"] for row in selected], marker="o", label=algo,
        )
    axis.set_xscale("log")
    axis.set_xlabel("学习率")
    axis.set_ylabel("验证集平均 F")
    axis.set_title("四类 HGT 算法的学习率敏感性")
    axis.grid(alpha=0.25); axis.legend(); figure.tight_layout()
    figure.savefig(args.output / "learning_rate_sensitivity.png", dpi=220)
    plt.close(figure)


if __name__ == "__main__":
    main()
