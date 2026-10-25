"""Collect completed formal runs into comparison-ready long tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


GROUPS_PATH = Path(__file__).with_name("groups.json")


def load_groups(path: Path = GROUPS_PATH) -> dict[str, list[dict]]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict], columns: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def build_comparison(
    result_root: Path,
    evaluation_set: str,
    group_name: str,
    output_root: Path,
    groups_path: Path = GROUPS_PATH,
) -> tuple[Path, Path]:
    groups = load_groups(groups_path)
    if group_name not in groups:
        raise ValueError(f"unknown comparison group: {group_name}")
    summary_rows: list[dict] = []
    long_rows: list[dict] = []
    for method in groups[group_name]:
        method_root = result_root / evaluation_set / method["directory"]
        if not method_root.exists():
            continue
        for seed_dir in sorted(method_root.glob("seed_*")):
            if not (seed_dir / "COMPLETED").exists():
                continue
            summary_path, instances_path = seed_dir / "summary.json", seed_dir / "per_instance.csv"
            if not summary_path.exists() or not instances_path.exists():
                continue
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            seed = int(seed_dir.name.removeprefix("seed_"))
            summary_rows.append({
                "set": evaluation_set, "group": group_name,
                "method": method["id"], "label": method["label"], "seed": seed,
                "F": summary["means"]["F"], "TT": summary["means"]["TT"],
                "WSF": summary["means"]["WSF"],
                "training_seconds": summary.get("training_seconds", 0.0),
                "solve_seconds_per_instance": summary.get("solve_seconds_per_instance", 0.0),
                "decision_seconds_mean": summary.get("decision_seconds_mean", 0.0),
            })
            for row in _read_csv(instances_path):
                long_rows.append({
                    "set": evaluation_set, "group": group_name,
                    "method": method["id"], "label": method["label"], "seed": seed,
                    "scenario_id": row["scenario_id"], "instance_id": row["instance_id"],
                    "F": row["F"], "TT": row["TT"], "WSF": row["WSF"],
                    "solve_seconds": row.get("solve_seconds", ""),
                })
    destination = output_root / evaluation_set / group_name
    summary_path = destination / "comparison_table.csv"
    long_path = destination / "per_instance_long.csv"
    _write_csv(summary_path, summary_rows, ("set", "group", "method", "label", "seed", "F", "TT", "WSF", "training_seconds", "solve_seconds_per_instance", "decision_seconds_mean"))
    _write_csv(long_path, long_rows, ("set", "group", "method", "label", "seed", "scenario_id", "instance_id", "F", "TT", "WSF", "solve_seconds"))
    return summary_path, long_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", dest="evaluation_set", choices=("center", "l9"), required=True)
    parser.add_argument("--group", choices=tuple(load_groups()), required=True)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=Path("comparison"))
    args = parser.parse_args()
    build_comparison(args.results, args.evaluation_set, args.group, args.out)


if __name__ == "__main__":
    main()
