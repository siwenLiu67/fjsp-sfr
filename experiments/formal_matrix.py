"""Build, list, smoke-test, or execute the complete formal experiment matrix."""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_PROJECT_PARENT = _PROJECT_ROOT.parent
for _import_path in (str(_PROJECT_ROOT), str(_PROJECT_PARENT)):
    if _import_path not in sys.path:
        sys.path.insert(0, _import_path)

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"


@dataclass(frozen=True)
class RunSpec:
    name: str
    library_id: str
    library_file: str
    scale: str
    seed: int
    selector: str = "hgt_ppo"
    difficulty: str = "service_heavy"
    mask_checkpoint: bool = False
    mask_forecast: bool = False
    eval_scale: str | None = None
    checkpoint_run: str | None = None


def library_path(library_id: str, scale: str) -> str:
    mapping = {
        "classic4": "experiments/classic4_library.json",
        "handwritten6": "handwritten6",
        "random_nogate_mix": f"experiments/llm_library_random_pool_{scale}_nogate_mix.json",
        "pool_g": f"experiments/llm_library_pool_{scale}_g.json",
        "pool_mix": f"experiments/llm_library_pool_{scale}_mix.json",
        "p3b1_mix": f"experiments/llm_library_p3b1_{scale}_mix.json",
        "p3b2_mix": f"experiments/llm_library_p3b2_{scale}_mix.json",
        "p3b3_mix": f"experiments/llm_library_p3b3_{scale}_mix.json",
    }
    return mapping[library_id]


def build_matrix() -> list[RunSpec]:
    runs: list[RunSpec] = []
    libraries = (
        "classic4",
        "handwritten6",
        "random_nogate_mix",
        "pool_g",
        "pool_mix",
        "p3b1_mix",
        "p3b2_mix",
        "p3b3_mix",
    )
    for scale in ("S", "M", "L"):
        for library_id in libraries:
            for seed in (1, 2, 3):
                name = f"service_heavy_{scale}_{library_id}_hgt_ppo_seed{seed}"
                runs.append(RunSpec(name, library_id, library_path(library_id, scale), scale, seed))

    for scale in ("S", "M", "L"):
        for selector in ("mlp_ppo", "dqn", "random", "fixed"):
            for seed in (1, 2, 3):
                name = f"service_heavy_{scale}_pool_mix_{selector}_seed{seed}"
                runs.append(
                    RunSpec(name, "pool_mix", library_path("pool_mix", scale), scale, seed, selector)
                )

    for mask_name in ("checkpoint", "forecast"):
        for seed in (1, 2, 3):
            name = f"service_heavy_M_pool_mix_hgt_ppo_mask_{mask_name}_seed{seed}"
            runs.append(
                RunSpec(
                    name,
                    "pool_mix",
                    library_path("pool_mix", "M"),
                    "M",
                    seed,
                    mask_checkpoint=mask_name == "checkpoint",
                    mask_forecast=mask_name == "forecast",
                )
            )

    for library_id in ("pool_mix", "classic4"):
        for seed in (1, 2, 3):
            name = f"balanced_M_{library_id}_hgt_ppo_seed{seed}"
            runs.append(
                RunSpec(
                    name,
                    library_id,
                    library_path(library_id, "M"),
                    "M",
                    seed,
                    difficulty="balanced",
                )
            )

    for seed in (1, 2, 3):
        source = f"service_heavy_M_pool_mix_hgt_ppo_seed{seed}"
        runs.append(
            RunSpec(
                f"service_heavy_M_to_X_pool_mix_hgt_ppo_seed{seed}",
                "pool_mix",
                library_path("pool_mix", "M"),
                "M",
                seed,
                eval_scale="X",
                checkpoint_run=source,
            )
        )
    return runs


def command_for(spec: RunSpec, output_root: Path, *, smoke: bool) -> list[str]:
    output_dir = output_root / spec.name
    command = [
        sys.executable,
        "experiments/formal_experiment.py",
        "--library-id",
        spec.library_id,
        "--library-file",
        spec.library_file,
        "--scale",
        spec.scale,
        "--difficulty",
        spec.difficulty,
        "--seed",
        str(spec.seed),
        "--selector",
        spec.selector,
        "--episodes",
        "5000",
        "--episodes-per-update",
        "32",
        "--output-dir",
        str(output_dir),
    ]
    if spec.mask_checkpoint:
        command.append("--mask-checkpoint")
    if spec.mask_forecast:
        command.append("--mask-forecast")
    if spec.eval_scale:
        command += ["--eval-scale", spec.eval_scale]
    if spec.checkpoint_run:
        command += ["--checkpoint", str(output_root / spec.checkpoint_run / "best.pt")]
    if smoke:
        command.append("--smoke")
    return command


def plan_lines(runs: list[RunSpec], output_root: Path, *, smoke: bool) -> list[str]:
    trained = sum(
        run.selector not in {"random", "fixed"} and run.checkpoint_run is None
        for run in runs
    )
    lines = [
        f"runs={len(runs)}, trained={trained}, evaluation_only={len(runs) - trained}",
        "estimated serial formal runtime: 48.78 h training + about 1.2 h evaluation = 50.0 h",
    ]
    for index, run in enumerate(runs, 1):
        lines.append(f"{index:03d} " + shlex.join(command_for(run, output_root, smoke=smoke)))
    return lines


def execute(runs: list[RunSpec], output_root: Path, *, smoke: bool) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    for index, run in enumerate(runs, 1):
        command = command_for(run, output_root, smoke=smoke)
        summary = output_root / run.name / "summary.json"
        if summary.exists():
            print(f"[{index}/{len(runs)}] SKIP {run.name}", flush=True)
            continue
        library_path_value = command[command.index("--library-file") + 1]
        if library_path_value != "handwritten6" and not (ROOT / library_path_value).exists():
            raise FileNotFoundError(f"missing library for {run.name}: {library_path_value}")
        log_path = output_root / run.name / "run.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[{index}/{len(runs)}] RUN {run.name}", flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"{run.name} failed; see {log_path}")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--output-root")
    args = parser.parse_args(argv)
    runs = build_matrix()
    output_root = Path(
        args.output_root
        or ("experiments/formal_smoke" if args.smoke else "experiments/formal")
    )
    if args.list:
        lines = plan_lines(runs, output_root, smoke=args.smoke)
        print("\n".join(lines))
        plan_path = EXPERIMENTS / (
            "formal_smoke_run_list.txt" if args.smoke else "formal_run_list.txt"
        )
        plan_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"plan written: {plan_path}")
    else:
        execute(runs, output_root, smoke=args.smoke)


if __name__ == "__main__":
    main()
