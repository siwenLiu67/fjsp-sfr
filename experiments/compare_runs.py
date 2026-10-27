"""Paired comparisons for per-instance streaming PPO evaluation CSV files."""

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
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path


BLOCK_COUNT = 10
BLOCK_SIZE = 30


@dataclass(frozen=True)
class MethodSeries:
    label: str
    path: Path
    column: str
    values_by_seed: dict[int, float]


def read_method_series(label: str, path: str | Path, column: str) -> MethodSeries:
    """Read one numeric method column indexed by unique integer seed."""

    csv_path = Path(path)
    with csv_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "seed" not in reader.fieldnames:
            raise ValueError(f"{csv_path} must contain a 'seed' column")
        if column not in reader.fieldnames:
            raise ValueError(
                f"column {column!r} not found in {csv_path}; "
                f"available={reader.fieldnames}"
            )
        values: dict[int, float] = {}
        for row_number, row in enumerate(reader, start=2):
            seed = int(row["seed"])
            if seed in values:
                raise ValueError(f"duplicate seed {seed} in {csv_path}")
            value = float(row[column])
            if not math.isfinite(value):
                raise ValueError(
                    f"non-finite value at {csv_path}:{row_number}, "
                    f"column {column!r}"
                )
            values[seed] = value
    if not values:
        raise ValueError(f"{csv_path} contains no data rows")
    return MethodSeries(label, csv_path, column, values)


def bootstrap_mean_ci(
    differences: list[float],
    *,
    seed: int = 20_261_021,
    replicates: int = 10_000,
) -> tuple[float, float, float]:
    rng = random.Random(seed)
    size = len(differences)
    means = sorted(
        sum(differences[rng.randrange(size)] for _ in range(size)) / size
        for _ in range(replicates)
    )
    return (
        statistics.fmean(differences),
        means[int(0.025 * replicates)],
        means[min(replicates - 1, int(0.975 * replicates))],
    )


def aligned_values(
    left: MethodSeries,
    right: MethodSeries,
) -> tuple[list[int], list[float], list[float]]:
    left_seeds = set(left.values_by_seed)
    right_seeds = set(right.values_by_seed)
    if left_seeds != right_seeds:
        missing_left = sorted(right_seeds - left_seeds)
        missing_right = sorted(left_seeds - right_seeds)
        raise ValueError(
            "CSV seed sets differ: "
            f"missing from left={missing_left}, missing from right={missing_right}"
        )
    seeds = sorted(left_seeds)
    return (
        seeds,
        [left.values_by_seed[item] for item in seeds],
        [right.values_by_seed[item] for item in seeds],
    )


def block_means(values: list[float]) -> list[float]:
    expected = BLOCK_COUNT * BLOCK_SIZE
    if len(values) != expected:
        raise ValueError(
            f"block analysis requires exactly {expected} instances; "
            f"received {len(values)}"
        )
    return [
        statistics.fmean(values[start : start + BLOCK_SIZE])
        for start in range(0, expected, BLOCK_SIZE)
    ]


def print_block_summary(label: str, values: list[float]) -> None:
    blocks = block_means(values)
    print(f"  {label}:")
    print("    10个块的平均F: " + ", ".join(f"{value:.2f}" for value in blocks))
    print(
        f"    块均值={statistics.fmean(blocks):.2f}, "
        f"块间标准差={statistics.stdev(blocks):.2f}, "
        f"最小={min(blocks):.2f}, 最大={max(blocks):.2f}, "
        f"最大-最小={max(blocks) - min(blocks):.2f}"
    )


def print_paired_comparison(
    label: str,
    left: MethodSeries,
    right: MethodSeries,
    *,
    bootstrap_seed: int,
) -> None:
    seeds, left_values, right_values = aligned_values(left, right)
    left_mean = statistics.fmean(left_values)
    right_mean = statistics.fmean(right_values)
    relative = (left_mean - right_mean) / right_mean * 100.0
    differences = [
        left_value - right_value
        for left_value, right_value in zip(left_values, right_values, strict=True)
    ]
    wins = sum(value < 0.0 for value in differences)
    difference, low, high = bootstrap_mean_ci(
        differences,
        seed=bootstrap_seed,
    )

    print("\n" + "=" * 100)
    print(label)
    print("=" * 100)
    print(
        f"左侧: {left.label} | {left.path} | 列={left.column} | "
        f"平均 F={left_mean:.2f}"
    )
    print(
        f"右侧: {right.label} | {right.path} | 列={right.column} | "
        f"平均 F={right_mean:.2f}"
    )
    print(f"左侧相对右侧: {relative:+.2f}%")
    print(
        f"左侧胜率: {wins}/{len(seeds)}="
        f"{wins / len(seeds) * 100:.1f}%"
    )
    print(
        f"配对差值（左-右）均值={difference:+.2f}, "
        f"bootstrap 95% CI [{low:+.2f}, {high:+.2f}]"
    )
    print("\n30实例分块分析（按 seed 升序）:")
    print_block_summary(left.label, left_values)
    print_block_summary(right.label, right_values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pair",
        action="append",
        nargs=5,
        metavar=("LABEL", "LEFT_CSV", "LEFT_COLUMN", "RIGHT_CSV", "RIGHT_COLUMN"),
        required=True,
        help="repeat for each paired comparison",
    )
    parser.add_argument("--bootstrap-seed", type=int, default=20_261_021)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for index, (label, left_path, left_column, right_path, right_column) in enumerate(
        args.pair
    ):
        left = read_method_series(
            f"{Path(left_path).stem}:{left_column}", left_path, left_column
        )
        right = read_method_series(
            f"{Path(right_path).stem}:{right_column}", right_path, right_column
        )
        print_paired_comparison(
            label,
            left,
            right,
            bootstrap_seed=args.bootstrap_seed + index,
        )


if __name__ == "__main__":
    main()
