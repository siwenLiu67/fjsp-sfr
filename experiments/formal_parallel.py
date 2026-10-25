"""Parallel, resumable launcher for the formal experiment matrix."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from experiments.formal_matrix import RunSpec, build_matrix, command_for


def _run_one(
    run: RunSpec, output_root: Path, smoke: bool, task_env: dict[str, str]
) -> tuple[str, str]:
    run_dir = output_root / run.name
    completed = run_dir / "COMPLETED"
    if completed.exists():
        return run.name, "skipped"
    if not run.library_file.exists():
        raise FileNotFoundError(f"missing library for {run.name}: {run.library_file}")

    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"
    with log_path.open("w", encoding="utf-8") as log_handle:
        subprocess.run(
            command_for(run, output_root, smoke),
            check=True,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=task_env,
        )
    completed.write_text("ok\n", encoding="utf-8")
    return run.name, "completed"


def execute(runs: list[RunSpec], output_root: Path, smoke: bool, jobs: int) -> None:
    # A stable name-keyed map prevents two workers from writing the same run.
    unique = list({run.name: run for run in runs}.values())
    independent = [run for run in unique if run.checkpoint_run is None]
    dependent = [run for run in unique if run.checkpoint_run is not None]
    task_env = os.environ.copy()
    task_env.update(
        {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
    )

    def run_phase(phase: list[RunSpec]) -> None:
        with ThreadPoolExecutor(max_workers=jobs) as executor:
            futures = {
                executor.submit(_run_one, run, output_root, smoke, task_env): run
                for run in phase
            }
            for future in as_completed(futures):
                name, status = future.result()
                print(f"[{status}] {name}")

    # Generalization evaluations consume checkpoints produced by M-scale runs.
    run_phase(independent)
    run_phase(dependent)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--output-root", type=Path, default=Path("experiments/formal"))
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be at least 1")

    runs = build_matrix()
    if args.list:
        for run in runs:
            print(shlex.join(command_for(run, args.output_root, args.smoke)))
        return
    execute(runs, args.output_root, args.smoke, args.jobs)


if __name__ == "__main__":
    main()
