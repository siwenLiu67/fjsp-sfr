"""Learning-rate sweep for the four HGT learning algorithms on validation only."""

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
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, pstdev


ALGORITHMS = ("hgt_ppo", "hgt_ddqn", "hgt_a2c", "hgt_rainbow")
LEARNING_RATES = (1e-4, 3e-4, 1e-3)
NETWORK_SEEDS = (1, 2, 3)
EPISODES = 2000
DEFAULT_LIBRARY = "pool_center_mix"


@dataclass(frozen=True)
class SweepTask:
    algo: str
    learning_rate: float
    seed: int

    @property
    def name(self) -> str:
        return f"{self.algo}_lr_{self.learning_rate:.0e}_seed_{self.seed}"


def task_directory(root: Path, task: SweepTask) -> Path:
    return root / task.name


def command_for(task: SweepTask, destination: Path, library: str) -> list[str]:
    return [
        sys.executable, "experiments/run_algo.py", "--scenario", "center",
        "--eval_set", "center", "--validation_only", "--algo", task.algo,
        "--seeds", str(task.seed), "--library", library,
        "--episodes", str(EPISODES), "--episodes_per_update", "32",
        "--learning_rate", str(task.learning_rate), "--out_root", str(destination / "run"),
    ]


def _find_summary(destination: Path) -> Path:
    matches = tuple((destination / "run" / "validation").glob("*/seed_*/summary.json"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one validation summary under {destination}, found {len(matches)}")
    return matches[0]


def run_task(task: SweepTask, root: Path, library: str) -> tuple[SweepTask, str]:
    destination = task_directory(root, task)
    completed = destination / "COMPLETED"
    if completed.exists():
        return task, "skipped"
    destination.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    with (destination / "run.log").open("w", encoding="utf-8") as log:
        subprocess.run(command_for(task, destination, library), check=True, env=env, stdout=log, stderr=subprocess.STDOUT)
    summary = json.loads(_find_summary(destination).read_text(encoding="utf-8"))
    summary.update({"algo": task.algo, "learning_rate": task.learning_rate, "network_seed": task.seed, "episodes": EPISODES, "evaluation": "validation_center_only"})
    (destination / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    completed.write_text("ok\n", encoding="utf-8")
    return task, "completed"


def aggregate(root: Path) -> tuple[list[dict], dict[str, float]]:
    rows, best = [], {}
    for algo in ALGORITHMS:
        algo_rows = []
        for learning_rate in LEARNING_RATES:
            values = []
            for seed in NETWORK_SEEDS:
                path = task_directory(root, SweepTask(algo, learning_rate, seed)) / "summary.json"
                marker = path.parent / "COMPLETED"
                if marker.exists() and path.exists():
                    values.append(float(json.loads(path.read_text(encoding="utf-8"))["means"]["F"]))
            row = {
                "algo": algo, "learning_rate": learning_rate,
                "validation_F_mean": fmean(values) if values else float("nan"),
                "seed_std": pstdev(values) if len(values) > 1 else 0.0,
                "completed_seeds": len(values),
            }
            rows.append(row); algo_rows.append(row)
        completed_rows = [row for row in algo_rows if row["completed_seeds"] == len(NETWORK_SEEDS)]
        if completed_rows:
            best[algo] = min(completed_rows, key=lambda row: row["validation_F_mean"])["learning_rate"]
    return rows, best


def write_aggregate(root: Path) -> None:
    rows, best = aggregate(root)
    with (root / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("algo", "learning_rate", "validation_F_mean", "seed_std", "completed_seeds"))
        writer.writeheader(); writer.writerows(rows)
    Path("experiments/best_lr.json").write_text(json.dumps(best, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algo", choices=ALGORITHMS)
    parser.add_argument("--learning_rate", type=float, choices=LEARNING_RATES)
    parser.add_argument("--seed", type=int, choices=NETWORK_SEEDS)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--library", default=DEFAULT_LIBRARY)
    parser.add_argument("--output-root", type=Path, default=Path("experiments/sweep"))
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    algorithms = (args.algo,) if args.algo else ALGORITHMS
    rates = (args.learning_rate,) if args.learning_rate else LEARNING_RATES
    seeds = (args.seed,) if args.seed else NETWORK_SEEDS
    tasks = [SweepTask(algo, rate, seed) for algo in algorithms for rate in rates for seed in seeds]
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = [executor.submit(run_task, task, args.output_root, args.library) for task in tasks]
        for future in as_completed(futures):
            future.result()
    write_aggregate(args.output_root)


if __name__ == "__main__":
    main()
