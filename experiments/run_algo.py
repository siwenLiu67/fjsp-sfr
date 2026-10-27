"""Unified center-training and center/L9 evaluation entry point."""

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
import os
import random
import statistics
import time
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch

torch.set_num_threads(1)

from fjsp_sfr.dispatch.classic_rules import build_classic_rule_library
from fjsp_sfr.baselines.rolling_ga import RollingGAConfig, run_rolling_ga, time_probe
from fjsp_sfr.dispatch.rule_library import RuleLibrary
from fjsp_sfr.experiments.formal_experiment import best_fixed_action, evaluate_episode, make_agent
from fjsp_sfr.experiments.manifest import file_sha256, write_manifest
from fjsp_sfr.generation.scenario_generator import make_scenario_instance, scenario_names, seed_values
from fjsp_sfr.rl.ppo_trainer import PPOTrainer
from fjsp_sfr.rl.ddqn_agent import DDQNTrainer, HGTDDQNAgent
from fjsp_sfr.rl.a2c_agent import A2CTrainer, HGTA2CAgent


FEATURE_SET = "theory11_plus3"
TRAIN_BASE_SEED = seed_values("training")[0]
ALGO_CODES = {
    "spt": "01_spt", "edd": "02_edd", "atc": "03_atc",
    "minslack": "04_minslack", "best_single": "05_best_single",
    "random_select": "06_random_select", "hgt_ppo": "10_hgt_ppo",
    "rolling_ga": "07_rolling_ga",
    "mlp_ppo": "11_mlp_ppo", "hgt_ddqn": "12_hgt_ddqn",
    "hgt_a2c": "13_hgt_a2c", "hgt_rainbow": "14_hgt_rainbow",
}
CLASSIC_ACTIONS = {"spt": 0, "edd": 1, "atc": 2, "minslack": 3}
LIBRARY_ALIASES = {
    "pool_center_mix": "experiments/llm_library_pool_center_mix.json",
    "pool_center_g": "experiments/llm_library_pool_center_g.json",
    "handwritten_center": "experiments/handwritten_center.json",
    "random_center_nogate": "experiments/llm_library_random_center_nogate.json",
    "random_center_nogate_mix": "experiments/llm_library_random_center_nogate_mix.json",
    "classic4": "experiments/classic4_library.json",
}


def resolve_library(value: str) -> Path:
    return Path(LIBRARY_ALIASES.get(value, value))


def resolve_runtime_library(value: str) -> Path:
    path = resolve_library(value)
    if path.exists() or value == "classic4":
        return path
    fallback = resolve_library("classic4")
    return fallback if fallback.exists() else path


def variant_name(args) -> str:
    base = ALGO_CODES[args.algo]
    if args.algo in {"hgt_ppo", "mlp_ppo", "hgt_ddqn", "hgt_a2c", "hgt_rainbow", "best_single", "random_select"}:
        base += f"_{args.library}"
    if args.mask != "none":
        base += f"_mask_{args.mask}"
    if args.algo == "rolling_ga" and args.ga_time_probe:
        base += "_time_probe"
    return base


def evaluation_plan(args) -> list[tuple[str, int]]:
    if args.validation_only:
        return [("center", seed) for seed in seed_values("validation_center")]
    if args.eval_set == "center":
        return [("center", seed) for seed in seed_values("test_center")]
    selected = tuple(name for name in scenario_names() if name.startswith("l9_"))
    if args.scenario.startswith("l9_"):
        selected = (args.scenario,)
    return [
        (scenario, seed)
        for scenario in selected
        for seed in seed_values(f"test_{scenario}")
    ]


def _best_lr(algo: str) -> float:
    path = Path("experiments/best_lr.json")
    if not path.exists():
        return 3e-4
    payload = json.loads(path.read_text(encoding="utf-8"))
    value = payload.get(algo, payload.get("algorithms", {}).get(algo, 3e-4))
    return float(value["learning_rate"] if isinstance(value, dict) else value)


def make_learning_agent(algo: str, action_size: int):
    if algo == "hgt_ddqn":
        return HGTDDQNAgent(action_size, hidden_dim=64, num_layers=2, num_heads=4, job_input_dim=14)
    if algo == "hgt_a2c":
        return HGTA2CAgent(action_size=action_size, hidden_dim=64, num_layers=2, num_heads=4, job_input_dim=14)
    if algo == "hgt_rainbow":
        from fjsp_sfr.rl.rainbow_agent import HGTRainbowAgent
        return HGTRainbowAgent(action_size=action_size, hidden_dim=64, num_layers=2, num_heads=4, job_input_dim=14)
    return make_agent(algo, action_size)


def _rolling_task(payload):
    scenario_id, instance_seed, network_seed, config = payload
    started = time.perf_counter()
    result = run_rolling_ga(make_scenario_instance(scenario_id, instance_seed), config=config, seed=network_seed * 100000 + instance_seed)
    return {
        "scenario_id": scenario_id, "instance_id": instance_seed,
        "F": result.F, "TT": result.TT, "WSF": result.WSF,
        "solve_seconds": time.perf_counter() - started,
        "reschedule_count": result.reschedule_count,
        "mean_reschedule_seconds": result.mean_reschedule_seconds,
        "mean_generations": result.mean_generations,
        "decision_seconds": result.decision_seconds,
    }


def run_rolling_seed(args, network_seed: int, run_dir: Path) -> None:
    if args.eval_set != "center" or args.validation_only:
        raise ValueError("rolling_ga is evaluated on the center scenario only")
    config = RollingGAConfig(
        population_size=args.ga_pop, crossover_probability=args.ga_pc,
        mutation_probability=args.ga_pm, time_limit=args.ga_time_limit,
        tournament_size=args.ga_tournament, elite=args.ga_elite,
        max_generations=args.ga_max_gen,
    )
    plan = evaluation_plan(args)[: args.n_instances]
    if args.ga_time_probe:
        probes = [make_scenario_instance(scenario, seed) for scenario, seed in plan[:3]]
        payload = time_probe(probes, config=config, seed=network_seed)
        (run_dir / "ga_time_probe.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        (run_dir / "COMPLETED").write_text("ok\n", encoding="utf-8")
        return
    payloads = [(scenario, seed, network_seed, config) for scenario, seed in plan]
    if args.workers == 1:
        rows = [_rolling_task(payload) for payload in payloads]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            rows = list(executor.map(_rolling_task, payloads))
    with (run_dir / "per_instance.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0])); writer.writeheader(); writer.writerows(rows)
    with (run_dir / "train_curve.csv").open("w", encoding="utf-8", newline="") as handle:
        csv.DictWriter(handle, fieldnames=("update", "episodes", "mean_train_F", "val_F_greedy", "wall_seconds")).writeheader()
    summary = {
        "algo": "rolling_ga", "network_seed": network_seed,
        "means": {key: statistics.fmean(row[key] for row in rows) for key in ("F", "TT", "WSF")},
        "std": {key: statistics.pstdev(row[key] for row in rows) for key in ("F", "TT", "WSF")},
        "training_seconds": 0.0,
        "solve_seconds_per_instance": statistics.fmean(row["solve_seconds"] for row in rows),
        "decision_seconds_mean": sum(row["decision_seconds"] for row in rows) / max(1, sum(row["reschedule_count"] for row in rows)),
        "mean_reschedule_count": statistics.fmean(row["reschedule_count"] for row in rows),
        "mean_generations": statistics.fmean(row["mean_generations"] for row in rows),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_manifest(run_dir / "manifest.json", {"kind": "rolling_ga", "config": vars(args), "network_seed": network_seed})
    (run_dir / "run.log").write_text(json.dumps(summary, ensure_ascii=False) + "\n", encoding="utf-8")
    (run_dir / "COMPLETED").write_text("ok\n", encoding="utf-8")


def train_ppo(
    args,
    library: RuleLibrary,
    library_path: Path,
    network_seed: int,
    run_dir: Path,
):
    random.seed(network_seed)
    torch.manual_seed(network_seed)
    agent = make_learning_agent(args.algo, library.action_size())
    lr = args.learning_rate if args.learning_rate is not None else _best_lr(args.algo)
    factory = lambda seed: make_scenario_instance("center", seed)
    trainer = PPOTrainer(
        agent=agent, library=library, instances=[], instance_factory=factory,
        base_seed=TRAIN_BASE_SEED, lr=lr, gamma=0.99, gae_lambda=0.95,
        epochs_per_update=4, reward_scale=10.0, entropy_coef=0.02,
        feature_set=FEATURE_SET,
        mask_checkpoint=args.mask == "checkpoint",
        mask_forecast=args.mask == "forecast",
    )
    validation_seeds = seed_values("validation_center")[: args.n_validation_instances]
    validation_instances = [factory(seed) for seed in validation_seeds]
    updates = math.ceil(args.episodes / args.episodes_per_update)
    curve, started = [], time.perf_counter()
    for update in range(updates):
        fraction = update / max(1, updates - 1)
        trainer.entropy_coef = 0.02 + fraction * (0.002 - 0.02)
        for group in trainer.optimizer.param_groups:
            group["lr"] = lr + fraction * (1e-4 - lr)
        count = min(args.episodes_per_update, args.episodes - update * args.episodes_per_update)
        stats = trainer.train(1, count)[0]
        if (update + 1) % max(1, updates // 10) == 0 or update + 1 == updates:
            values = [
                evaluate_episode(
                    library, instance, selector=args.algo, agent=agent,
                    rng=random.Random(seed),
                    mask_checkpoint=args.mask == "checkpoint",
                    mask_forecast=args.mask == "forecast",
                )["F"]
                for seed, instance in zip(validation_seeds, validation_instances, strict=True)
            ]
            curve.append({
                "update": update + 1,
                "episodes": min((update + 1) * args.episodes_per_update, args.episodes),
                "mean_train_F": -float(stats.total_reward),
                "val_F_greedy": statistics.fmean(values),
                "wall_seconds": time.perf_counter() - started,
            })
    checkpoint = {
        "agent_state_dict": agent.state_dict(), "algo": args.algo,
        "library_sha256": file_sha256(library_path),
        "network_seed": network_seed, "episodes": args.episodes,
    }
    torch.save(checkpoint, run_dir / "model_last.pt")
    return agent, curve, time.perf_counter() - started


def train_value_or_a2c(args, library, network_seed):
    random.seed(network_seed); torch.manual_seed(network_seed)
    agent = make_learning_agent(args.algo, library.action_size())
    factory = lambda seed: make_scenario_instance("center", seed)
    lr = args.learning_rate if args.learning_rate is not None else _best_lr(args.algo)
    started = time.perf_counter(); curve = []
    if args.algo == "hgt_ddqn":
        trainer = DDQNTrainer(agent, library, factory, base_seed=TRAIN_BASE_SEED, feature_set=FEATURE_SET, lr=lr, gamma=0.99, reward_scale=10.0, mask_checkpoint=args.mask == "checkpoint", mask_forecast=args.mask == "forecast")
        stats = trainer.train(args.episodes)
        curve.append({"update": max(1, math.ceil(args.episodes / args.episodes_per_update)), "episodes": args.episodes, "mean_train_F": -statistics.fmean(stats.episode_rewards), "val_F_greedy": float("nan"), "wall_seconds": time.perf_counter() - started})
    elif args.algo == "hgt_a2c":
        trainer = A2CTrainer(agent, library, factory, base_seed=TRAIN_BASE_SEED, lr=lr, gamma=0.99, gae_lambda=0.95, reward_scale=10.0, feature_set=FEATURE_SET, mask_checkpoint=args.mask == "checkpoint", mask_forecast=args.mask == "forecast")
        stats = trainer.train(args.episodes, episodes_per_update=args.episodes_per_update)
        curve = [{"update": index, "episodes": min(index * args.episodes_per_update, args.episodes), "mean_train_F": -item.mean_reward, "val_F_greedy": float("nan"), "wall_seconds": time.perf_counter() - started} for index, item in enumerate(stats, 1)]
    else:
        from fjsp_sfr.rl.rainbow_agent import RainbowTrainer
        trainer = RainbowTrainer(agent, library, factory, base_seed=TRAIN_BASE_SEED, lr=lr, gamma=0.99, reward_scale=10.0, feature_set=FEATURE_SET, mask_checkpoint=args.mask == "checkpoint", mask_forecast=args.mask == "forecast")
        stats = trainer.train(args.episodes)
        curve.append({"update": max(1, math.ceil(args.episodes / args.episodes_per_update)), "episodes": args.episodes, "mean_train_F": -statistics.fmean(stats.episode_rewards), "val_F_greedy": float("nan"), "wall_seconds": time.perf_counter() - started})
    validation_selector = "dqn" if args.algo in {"hgt_ddqn", "hgt_rainbow"} else args.algo
    validation_values = [
        evaluate_episode(
            library, factory(seed), selector=validation_selector, agent=agent,
            rng=random.Random(network_seed * 100000 + seed),
            mask_checkpoint=args.mask == "checkpoint",
            mask_forecast=args.mask == "forecast",
        )["F"]
        for seed in seed_values("validation_center")
    ]
    if curve:
        curve[-1]["val_F_greedy"] = statistics.fmean(validation_values)
    return agent, curve, time.perf_counter() - started


def load_center_model(args, library: RuleLibrary, network_seed: int):
    center_dir = Path(args.out_root) / "center" / variant_name(args) / f"seed_{network_seed}"
    checkpoint_path = center_dir / "model_last.pt"
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload["library_sha256"] != file_sha256(resolve_runtime_library(args.library)):
        raise ValueError("checkpoint rule-library hash mismatch")
    if payload["algo"] != args.algo:
        raise ValueError("checkpoint algorithm mismatch")
    agent = make_learning_agent(args.algo, library.action_size())
    agent.load_state_dict(payload["agent_state_dict"])
    agent.eval()
    return agent


def run_seed(args, network_seed: int) -> None:
    output_set = "validation" if args.validation_only else args.eval_set
    run_dir = Path(args.out_root) / output_set / variant_name(args) / f"seed_{network_seed}"
    completed = run_dir / "COMPLETED"
    if completed.exists():
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.algo == "rolling_ga":
        run_rolling_seed(args, network_seed, run_dir)
        return
    library_path = resolve_runtime_library(args.library)
    if args.algo in CLASSIC_ACTIONS:
        library = build_classic_rule_library()
    else:
        library = RuleLibrary.from_json(str(library_path))
    selector, fixed_action, agent = "fixed", None, None
    curve, training_seconds = [], 0.0
    if args.algo in CLASSIC_ACTIONS:
        fixed_action = CLASSIC_ACTIONS[args.algo]
    elif args.algo == "best_single":
        selection_seeds = seed_values("selection_center")[: max(2, args.n_instances)]
        selection = [make_scenario_instance("center", seed) for seed in selection_seeds]
        fixed_action = best_fixed_action(library, selection)
    elif args.algo == "random_select":
        selector = "random"
    else:
        selector = "dqn" if args.algo in {"hgt_ddqn", "hgt_rainbow"} else args.algo
        if args.eval_set == "center" or args.validation_only:
            if args.algo in {"hgt_ppo", "mlp_ppo"}:
                agent, curve, training_seconds = train_ppo(
                    args, library, library_path, network_seed, run_dir
                )
            else:
                agent, curve, training_seconds = train_value_or_a2c(args, library, network_seed)
                torch.save({
                    "agent_state_dict": agent.state_dict(), "algo": args.algo,
                    "library_sha256": file_sha256(library_path),
                    "network_seed": network_seed, "episodes": args.episodes,
                }, run_dir / "model_last.pt")
        else:
            agent = load_center_model(args, library, network_seed)

    rows, total_decision_seconds, total_decisions = [], 0.0, 0
    solve_started = time.perf_counter()
    plan = evaluation_plan(args)[: args.n_instances]
    for scenario_id, instance_seed in plan:
        instance = make_scenario_instance(scenario_id, instance_seed)
        started = time.perf_counter()
        result = evaluate_episode(
            library, instance, selector=selector, agent=agent,
            fixed_action=fixed_action, rng=random.Random(network_seed * 100000 + instance_seed),
            mask_checkpoint=args.mask == "checkpoint",
            mask_forecast=args.mask == "forecast",
        )
        solve_seconds = time.perf_counter() - started
        decisions = len(result["actions"])
        total_decisions += decisions
        total_decision_seconds += result["decision_seconds"] or solve_seconds
        rows.append({
            "scenario_id": scenario_id, "instance_id": instance_seed,
            "F": result["F"], "TT": result["TT"], "WSF": result["WSF"],
            "solve_seconds": solve_seconds,
        })
    with (run_dir / "per_instance.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("scenario_id", "instance_id", "F", "TT", "WSF", "solve_seconds"))
        writer.writeheader(); writer.writerows(rows)
    with (run_dir / "train_curve.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("update", "episodes", "mean_train_F", "val_F_greedy", "wall_seconds"))
        writer.writeheader(); writer.writerows(curve)
    summary = {
        "algo": args.algo, "library": args.library, "mask": args.mask,
        "eval_set": args.eval_set, "network_seed": network_seed,
        "means": {key: statistics.fmean(row[key] for row in rows) for key in ("F", "TT", "WSF")},
        "std": {key: statistics.pstdev(row[key] for row in rows) for key in ("F", "TT", "WSF")},
        "training_seconds": training_seconds,
        "solve_seconds_per_instance": statistics.fmean(row["solve_seconds"] for row in rows),
        "decision_seconds_mean": total_decision_seconds / max(1, total_decisions),
        "evaluation_seconds": time.perf_counter() - solve_started,
        "fixed_rule": library.get_rule(fixed_action).name if fixed_action is not None else None,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_manifest(run_dir / "manifest.json", {
        "kind": "run_algo", "config": vars(args), "network_seed": network_seed,
        "library_sha256": file_sha256(library_path) if args.algo not in CLASSIC_ACTIONS else file_sha256("experiments/classic4_library.json"),
        "evaluation_instances": plan,
    })
    (run_dir / "run.log").write_text(json.dumps(summary, ensure_ascii=False) + "\n", encoding="utf-8")
    completed.write_text("ok\n", encoding="utf-8")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=scenario_names(), default="center")
    parser.add_argument("--algo", choices=tuple(ALGO_CODES), required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--library", default="pool_center_mix")
    parser.add_argument("--mask", choices=("none", "checkpoint", "forecast"), default="none")
    parser.add_argument("--out_root", default="results")
    parser.add_argument("--eval_set", choices=("center", "l9"), default="center")
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--episodes_per_update", type=int, default=32)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--validation_only", action="store_true")
    parser.add_argument("--ga_pop", type=int, default=50)
    parser.add_argument("--ga_pc", type=float, default=0.8)
    parser.add_argument("--ga_pm", type=float, default=0.2)
    parser.add_argument("--ga_time_limit", type=float, default=3.0)
    parser.add_argument("--ga_tournament", type=int, default=3)
    parser.add_argument("--ga_elite", type=int, default=1)
    parser.add_argument("--ga_max_gen", type=int)
    parser.add_argument("--ga_time_probe", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--n_instances", type=int, default=100)
    parser.add_argument("--n_validation_instances", type=int, default=50)
    args = parser.parse_args(argv)
    if args.eval_set == "l9" and args.algo in {"hgt_ppo", "mlp_ppo"}:
        pass
    return args


def main() -> None:
    args = parse_args()
    for network_seed in args.seeds:
        run_seed(args, network_seed)


if __name__ == "__main__":
    main()
