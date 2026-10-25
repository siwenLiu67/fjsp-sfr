"""Streaming PPO training with validation selection and held-out testing."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

PACKAGE_PARENT = Path(__file__).resolve().parents[2]
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from fjsp_sfr.core.objective import compute_objective
from fjsp_sfr.dispatch.rule_library import RuleLibrary
from fjsp_sfr.dispatch.theory_features import (
    build_extended_theory_features,
    feature_names_for,
)
from fjsp_sfr.env.ca_observation import resolve_observation_scales
from fjsp_sfr.env.ca_rule_selection import CAPPORuleSelectionEnv
from fjsp_sfr.experiments.validate_framework import (
    build_rule_library,
    evaluate_random_policy,
    evaluate_single_rule,
    make_fjsp_instance,
)
from fjsp_sfr.experiments.manifest import file_sha256, write_manifest
from fjsp_sfr.rl.graph import build_hetero_graph
from fjsp_sfr.rl.ppo_agent import HGTPPOAgent
from fjsp_sfr.rl.ppo_trainer import PPOTrainer


TRAIN_BASE_SEED = 1000
VALIDATION_SEEDS = tuple(range(100, 150))
TEST_SEEDS = tuple(range(200, 500))
LR_START, LR_END = 3e-4, 1e-4
ENTROPY_START, ENTROPY_END = 0.02, 0.002
TOP3_RULE_NAMES = {
    "service_value",
    "tardiness_avoidance",
    "capacity_aware",
}


@dataclass(frozen=True)
class EvaluationPoint:
    update: int
    episodes: int
    ppo_mean_f: float
    sampled_mean_f: float
    fixed_mean_f: float
    gap_percent: float
    entropy: float
    recent_reward: float
    rule_usage: dict[str, int]
    total_variation: float
    switch_rate: float


@dataclass(frozen=True)
class PolicyEvaluation:
    values: list[float]
    usage: dict[str, int]
    tt_values: list[float] | None = None
    wsf_values: list[float] | None = None
    total_variation: float | None = None
    switch_rate: float | None = None


class Tee:
    """Mirror text to the terminal and a persistent log file."""

    def __init__(self, *streams) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
        self.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def make_stream_instance(seed: int):
    return make_fjsp_instance(
        seed, n_jobs=15, n_machines=5, n_entities=2
    )


def linear_schedule(
    start: float,
    end: float,
    update_index: int,
    total_updates: int,
) -> float:
    """Interpolate using a zero-based global update index."""

    if total_updates <= 1:
        return end
    fraction = update_index / (total_updates - 1)
    return start + fraction * (end - start)


def select_rule_library(
    name: str | None = None,
    library_file: str | None = None,
) -> tuple[RuleLibrary, RuleLibrary]:
    """Return the selected library and the complete six-rule reference library."""

    all_six = build_rule_library()
    if library_file is not None:
        selected = RuleLibrary.from_json(library_file)
        if selected.action_size() <= 0:
            raise ValueError("external rule library must contain at least one rule")
        return selected, all_six
    name = name or "all6"
    if name == "all6":
        return all_six, all_six
    selected = tuple(
        rule for rule in all_six.rules if rule.name in TOP3_RULE_NAMES
    )
    if {rule.name for rule in selected} != TOP3_RULE_NAMES:
        raise RuntimeError("top3 rule names do not match the six-rule library")
    return RuleLibrary(rules=selected), all_six


def library_source_metadata(
    name: str | None,
    library_file: str | None,
) -> dict[str, str]:
    """Return stable checkpoint metadata for a built-in or JSON library."""

    if library_file is None:
        return {"kind": "builtin", "name": name or "all6"}
    path = Path(library_file).expanduser().resolve()
    payload = path.read_bytes()
    return {
        "kind": "json",
        "path": str(path),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _policy_rollout(
    agent,
    library,
    instance,
    *,
    greedy: bool,
    seed: int,
    feature_set: str = "theory11",
) -> tuple[float, float, float, Counter[str], list[list[float]], list[int]]:
    """Run one policy episode and retain decision probabilities and actions."""

    rollout_instance = deepcopy(instance)
    env = CAPPORuleSelectionEnv(rollout_instance)
    env.reset()
    usage: Counter[str] = Counter()
    probabilities: list[list[float]] = []
    actions: list[int] = []
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    agent.eval()

    while env.current_metrics and env.current_metrics.is_active:
        problem = env._problem_view()
        state = env._state
        metrics = env.current_metrics
        time_scale, processing_scale = resolve_observation_scales(problem, state)
        features = build_extended_theory_features(
            problem,
            state,
            metrics,
            time_scale=time_scale,
            feature_set=feature_set,
        )
        graph = build_hetero_graph(
            problem,
            state,
            metrics,
            features,
            time_scale=time_scale,
            processing_scale=processing_scale,
            feature_set=feature_set,
        )
        mask = torch.ones(library.action_size(), dtype=torch.bool)
        with torch.no_grad():
            log_probs, _value = agent(graph, mask)
            probs = log_probs.exp().detach().cpu()
        if greedy:
            action = int(probs.argmax().item())
        else:
            action = int(torch.multinomial(probs, 1, generator=generator).item())
        probabilities.append(probs.tolist())
        actions.append(action)
        rule = library.get_rule(action)
        usage[rule.name] += 1
        result = env.step_principle(action, rule)
        if result.terminated:
            break

    objective = compute_objective(rollout_instance, env._state)
    return (
        float(objective.Z),
        float(objective.total_tardiness),
        float(objective.weighted_checkpoint_shortfall),
        usage,
        probabilities,
        actions,
    )


def evaluate_policy(
    agent,
    library,
    instances,
    *,
    greedy: bool,
    repeats: int = 1,
    base_seed: int = 0,
    feature_set: str = "theory11",
) -> PolicyEvaluation:
    """Evaluate a greedy or sampled policy, optionally with sensitivity metrics."""

    values: list[float] = []
    tt_values: list[float] = []
    wsf_values: list[float] = []
    usage: Counter[str] = Counter()
    all_probabilities: list[list[float]] = []
    switch_count = 0
    decision_count = 0
    for instance_index, instance in enumerate(instances):
        repeated_values: list[float] = []
        repeated_tt: list[float] = []
        repeated_wsf: list[float] = []
        for repeat in range(repeats):
            objective, tt, wsf, instance_usage, probabilities, actions = _policy_rollout(
                agent,
                library,
                instance,
                greedy=greedy,
                seed=base_seed + instance_index * repeats + repeat,
                feature_set=feature_set,
            )
            repeated_values.append(objective)
            repeated_tt.append(tt)
            repeated_wsf.append(wsf)
            usage.update(instance_usage)
            if greedy:
                all_probabilities.extend(probabilities)
                if actions:
                    counts = Counter(actions)
                    modal_action = min(
                        counts,
                        key=lambda action: (-counts[action], action),
                    )
                    switch_count += sum(
                        action != modal_action for action in actions
                    )
                    decision_count += len(actions)
        values.append(sum(repeated_values) / len(repeated_values))
        tt_values.append(sum(repeated_tt) / len(repeated_tt))
        wsf_values.append(sum(repeated_wsf) / len(repeated_wsf))

    if not greedy:
        return PolicyEvaluation(
            values=values,
            usage=dict(usage),
            tt_values=tt_values,
            wsf_values=wsf_values,
        )
    if not all_probabilities:
        return PolicyEvaluation(
            values=values,
            usage=dict(usage),
            tt_values=tt_values,
            wsf_values=wsf_values,
            total_variation=0.0,
            switch_rate=0.0,
        )
    probability_tensor = torch.tensor(all_probabilities, dtype=torch.float64)
    global_mean = probability_tensor.mean(dim=0)
    total_variation = float(
        (0.5 * (probability_tensor - global_mean).abs().sum(dim=1)).mean()
    )
    switch_rate = switch_count / decision_count if decision_count else 0.0
    return PolicyEvaluation(
        values=values,
        usage=dict(usage),
        tt_values=tt_values,
        wsf_values=wsf_values,
        total_variation=total_variation,
        switch_rate=switch_rate,
    )


def fixed_rule_results(library, instances) -> dict[str, list[float]]:
    return {
        rule.name: [evaluate_single_rule(rule, instance) for instance in instances]
        for rule in library.rules
    }


def means_by_rule(results: dict[str, list[float]]) -> dict[str, float]:
    return {
        name: sum(values) / len(values) for name, values in results.items()
    }


def bootstrap_mean_ci(
    differences: list[float],
    *,
    seed: int,
    replicates: int = 10_000,
) -> tuple[float, float, float]:
    """Return mean and percentile 95% bootstrap confidence interval."""

    rng = random.Random(seed)
    size = len(differences)
    sample_means = sorted(
        sum(differences[rng.randrange(size)] for _ in range(size)) / size
        for _ in range(replicates)
    )
    lower = sample_means[int(0.025 * replicates)]
    upper = sample_means[min(replicates - 1, int(0.975 * replicates))]
    return sum(differences) / size, lower, upper


def new_agent(
    action_size: int, feature_set: str = "theory11"
) -> HGTPPOAgent:
    return HGTPPOAgent(
        action_size=action_size,
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        job_input_dim=len(feature_names_for(feature_set)),
    )


def checkpoint_payload(
    *,
    agent,
    trainer,
    completed_updates: int,
    total_updates: int,
    target_episodes: int,
    episodes_per_update: int,
    stats_history: list[dict[str, float]],
    curve: list[EvaluationPoint],
    best_ppo_f: float,
    library_name: str,
    library_source: dict[str, str],
    seed: int,
    base_seed: int,
    validation_metrics: dict | None = None,
) -> dict:
    schedule_index = max(0, completed_updates - 1)
    return {
        "agent_state_dict": agent.state_dict(),
        "optimizer_state_dict": trainer.optimizer.state_dict(),
        "completed_updates": completed_updates,
        "total_updates": total_updates,
        "target_episodes": target_episodes,
        "episodes_per_update": episodes_per_update,
        "global_episode_count": trainer.global_episode_count,
        "schedule_position": (
            schedule_index / max(total_updates - 1, 1)
            if completed_updates
            else 0.0
        ),
        "entropy_coef": trainer.entropy_coef,
        "learning_rate": trainer.optimizer.param_groups[0]["lr"],
        "stats_history": stats_history,
        "evaluation_curve": [asdict(point) for point in curve],
        "best_ppo_f": best_ppo_f,
        "library_name": library_name,
        "library_source": library_source,
        "seed": seed,
        "base_seed": base_seed,
        "feature_set": trainer.feature_set,
        "validation_metrics": validation_metrics,
        "python_random_state": random.getstate(),
        "torch_random_state": torch.random.get_rng_state(),
    }


def save_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"resume checkpoint does not exist: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def validate_checkpoint_metadata(
    state: dict,
    *,
    library_name: str,
    library_source: dict[str, str],
    seed: int,
    base_seed: int,
    checkpoint_label: str,
) -> None:
    """Validate that a checkpoint belongs to the requested evaluation run."""

    saved_source = state.get("library_source")
    if saved_source is not None:
        if saved_source != library_source:
            raise ValueError(
                f"{checkpoint_label} library source mismatch: "
                f"checkpoint={saved_source}, requested={library_source}"
            )
    else:
        saved_library = state.get("library_name")
        if saved_library is not None and saved_library != library_name:
            raise ValueError(
                f"{checkpoint_label} library mismatch: "
                f"checkpoint={saved_library}, requested={library_name}"
            )

    saved_seed = state.get("seed")
    if saved_seed is not None and int(saved_seed) != seed:
        raise ValueError(
            f"{checkpoint_label} seed mismatch: "
            f"checkpoint={saved_seed}, requested={seed}"
        )
    saved_base_seed = state.get("base_seed")
    if saved_base_seed is not None and int(saved_base_seed) != base_seed:
        raise ValueError(
            f"{checkpoint_label} base_seed mismatch: "
            f"checkpoint={saved_base_seed}, requested={base_seed}"
        )


def write_test_results_csv(
    path: Path,
    *,
    seeds,
    fixed_results: dict[str, list[float]],
    random_values: list[float],
    best_greedy: PolicyEvaluation,
    best_sampled: PolicyEvaluation,
    last_greedy: PolicyEvaluation,
    last_sampled: PolicyEvaluation,
) -> None:
    """Export aligned per-instance fixed, random, and PPO objectives."""

    seeds = tuple(seeds)
    columns: dict[str, list[float]] = {
        **fixed_results,
        "random_mean_10": random_values,
        "best_greedy": best_greedy.values,
        "best_sampled_mean_5": best_sampled.values,
        "last_greedy": last_greedy.values,
        "last_sampled_mean_5": last_sampled.values,
    }
    if best_greedy.tt_values is not None and best_greedy.wsf_values is not None:
        columns["best_greedy_TT"] = best_greedy.tt_values
        columns["best_greedy_WSF"] = best_greedy.wsf_values
    expected = len(seeds)
    for name, values in columns.items():
        if len(values) != expected:
            raise ValueError(
                f"CSV column {name!r} has {len(values)} values; "
                f"expected {expected}"
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("seed", *columns))
        for index, seed in enumerate(seeds):
            writer.writerow(
                (seed, *(columns[name][index] for name in columns))
            )


def _policy_comparison(
    values: list[float],
    fixed_values: list[float],
    *,
    bootstrap_seed: int,
) -> tuple[float, float, int, float, float, float]:
    mean_f = sum(values) / len(values)
    fixed_mean = sum(fixed_values) / len(fixed_values)
    gap = (mean_f - fixed_mean) / fixed_mean * 100.0
    wins = sum(
        ppo < fixed for ppo, fixed in zip(values, fixed_values, strict=True)
    )
    differences = [
        ppo - fixed for ppo, fixed in zip(values, fixed_values, strict=True)
    ]
    difference, low, high = bootstrap_mean_ci(
        differences,
        seed=bootstrap_seed,
    )
    return mean_f, gap, wins, difference, low, high


def _print_policy_result(
    label: str,
    result: PolicyEvaluation,
    fixed_values: list[float],
    *,
    bootstrap_seed: int,
) -> None:
    mean_f, gap, wins, difference, low, high = _policy_comparison(
        result.values,
        fixed_values,
        bootstrap_seed=bootstrap_seed,
    )
    count = len(result.values)
    print(
        f"  {label:<24} F={mean_f:.2f}, vs固定={gap:+.1f}%, "
        f"胜率={wins}/{count}={wins / count * 100:.1f}%"
    )
    print(
        f"    配对差值均值={difference:+.2f}, "
        f"95% CI [{low:+.2f}, {high:+.2f}]"
    )
    print(f"    规则使用分布={result.usage}")


def _print_policy_subset_result(
    label: str,
    values: list[float],
    fixed_values: list[float],
    *,
    bootstrap_seed: int,
) -> None:
    mean_f, gap, wins, difference, low, high = _policy_comparison(
        values,
        fixed_values,
        bootstrap_seed=bootstrap_seed,
    )
    count = len(values)
    print(
        f"  {label:<24} F={mean_f:.2f}, vs固定={gap:+.1f}%, "
        f"胜率={wins}/{count}={wins / count * 100:.1f}%"
    )
    print(
        f"    配对差值均值={difference:+.2f}, "
        f"95% CI [{low:+.2f}, {high:+.2f}]"
    )


def print_test_report(
    *,
    library,
    all_six_library,
    best_agent,
    last_agent,
    test_instances,
    output_csv: Path | None = None,
    output_summary: Path | None = None,
    feature_set: str = "theory11",
) -> None:
    print("\n" + "=" * 118)
    print(
        f"独立测试集评估 (seeds {TEST_SEEDS[0]}-{TEST_SEEDS[-1]})"
    )
    print("=" * 118, flush=True)

    fixed_results = fixed_rule_results(library, test_instances)
    fixed_means = means_by_rule(fixed_results)
    selected_names = set(fixed_results)
    reference_results = {
        rule.name: [evaluate_single_rule(rule, instance) for instance in test_instances]
        for rule in all_six_library.rules
        if rule.name not in selected_names
    }
    all_fixed_results = {**fixed_results, **reference_results}
    all_six_means = {
        name: sum(all_fixed_results[name]) / len(all_fixed_results[name])
        for name in (rule.name for rule in all_six_library.rules)
    }
    best_fixed_name = min(fixed_means, key=fixed_means.get)
    best_fixed_mean = fixed_means[best_fixed_name]
    best_fixed_per_instance = fixed_results[best_fixed_name]
    random_per_instance = [
        sum(
            evaluate_random_policy(
                library,
                instance,
                seed=test_seed * 10_000 + trial,
            )
            for trial in range(10)
        )
        / 10.0
        for test_seed, instance in zip(TEST_SEEDS, test_instances, strict=True)
    ]
    best_greedy = evaluate_policy(
        best_agent, library, test_instances, greedy=True, feature_set=feature_set
    )
    best_sampled = evaluate_policy(
        best_agent,
        library,
        test_instances,
        greedy=False,
        repeats=5,
        base_seed=30_000_000,
        feature_set=feature_set,
    )
    last_greedy = evaluate_policy(
        last_agent, library, test_instances, greedy=True, feature_set=feature_set
    )
    last_sampled = evaluate_policy(
        last_agent,
        library,
        test_instances,
        greedy=False,
        repeats=5,
        base_seed=40_000_000,
        feature_set=feature_set,
    )
    random_mean = sum(random_per_instance) / len(random_per_instance)

    if output_csv is not None:
        write_test_results_csv(
            output_csv,
            seeds=TEST_SEEDS,
            fixed_results=all_fixed_results,
            random_values=random_per_instance,
            best_greedy=best_greedy,
            best_sampled=best_sampled,
            last_greedy=last_greedy,
            last_sampled=last_sampled,
        )
    if output_summary is not None:
        output_summary.parent.mkdir(parents=True, exist_ok=True)
        output_summary.write_text(
            json.dumps(
                {
                    "best_greedy": asdict(best_greedy),
                    "best_sampled": asdict(best_sampled),
                    "last_greedy": asdict(last_greedy),
                    "last_sampled": asdict(last_sampled),
                    "metrics": {
                        "best_state_total_variation": best_greedy.total_variation,
                        "best_state_switch_rate": best_greedy.switch_rate,
                        "last_state_total_variation": last_greedy.total_variation,
                        "last_state_switch_rate": last_greedy.switch_rate,
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    print("六条固定规则的参考平均 F:")
    for name, value in all_six_means.items():
        selected_marker = "*" if name in fixed_means else " "
        print(f" {selected_marker}固定 {name:>22}: {value:.2f}")
    print(
        f"\n所选规则库的最优固定规则: {best_fixed_name} "
        f"(平均 F={best_fixed_mean:.2f})"
    )
    service_value_mean = all_six_means.get("service_value")
    if service_value_mean is not None:
        print(
            "  跨库参考—手写 service_value: "
            f"F={service_value_mean:.2f}"
        )
    print(f"  随机策略(10次均值): F={random_mean:.2f}")
    _print_policy_result(
        "best 贪心",
        best_greedy,
        best_fixed_per_instance,
        bootstrap_seed=20_261_020,
    )
    _print_policy_result(
        "best 采样(5次均值)",
        best_sampled,
        best_fixed_per_instance,
        bootstrap_seed=20_261_021,
    )
    _print_policy_result(
        "last 贪心",
        last_greedy,
        best_fixed_per_instance,
        bootstrap_seed=20_261_022,
    )
    _print_policy_result(
        "last 采样(5次均值)",
        last_sampled,
        best_fixed_per_instance,
        bootstrap_seed=20_261_023,
    )

    subset_size = min(30, len(test_instances))
    subset_fixed = best_fixed_per_instance[:subset_size]
    print(
        f"\n测试集前{subset_size}个实例 "
        f"(seeds {TEST_SEEDS[0]}-{TEST_SEEDS[subset_size - 1]}):"
    )
    _print_policy_subset_result(
        "best 贪心",
        best_greedy.values[:subset_size],
        subset_fixed,
        bootstrap_seed=20_261_030,
    )
    _print_policy_subset_result(
        "best 采样(5次均值)",
        best_sampled.values[:subset_size],
        subset_fixed,
        bootstrap_seed=20_261_031,
    )
    _print_policy_subset_result(
        "last 贪心",
        last_greedy.values[:subset_size],
        subset_fixed,
        bootstrap_seed=20_261_032,
    )
    _print_policy_subset_result(
        "last 采样(5次均值)",
        last_sampled.values[:subset_size],
        subset_fixed,
        bootstrap_seed=20_261_033,
    )
    print(
        f"\n最佳检查点状态敏感度(TV): "
        f"{best_greedy.total_variation:.4f}"
    )
    print(f"最佳检查点切换率: {best_greedy.switch_rate:.2%}")
    if output_csv is not None:
        print(f"逐实例结果已导出: {output_csv.resolve()}")


def run_evaluation_only(args: argparse.Namespace) -> None:
    """Load existing best/last checkpoints and evaluate without training."""

    library_file = getattr(args, "library_file", None)
    library_name = args.library if library_file is None else "json"
    library_source = library_source_metadata(args.library, library_file)
    seed = int(getattr(args, "seed", 42))
    base_seed = int(getattr(args, "base_seed", TRAIN_BASE_SEED))
    feature_set = getattr(args, "feature_set", "theory11")
    library, all_six_library = select_rule_library(args.library, library_file)

    best_path = Path(args.best_checkpoint_path)
    last_path = Path(args.last_checkpoint_path)
    best_state = load_checkpoint(best_path)
    last_state = load_checkpoint(last_path)
    for label, state in (("best checkpoint", best_state), ("last checkpoint", last_state)):
        validate_checkpoint_metadata(
            state,
            library_name=library_name,
            library_source=library_source,
            seed=seed,
            base_seed=base_seed,
            checkpoint_label=label,
        )
        if state.get("feature_set", "theory11") != feature_set:
            raise ValueError(
                f"{label} feature_set mismatch: "
                f"checkpoint={state.get('feature_set', 'theory11')}, "
                f"requested={feature_set}"
            )

    best_agent = new_agent(library.action_size(), feature_set)
    last_agent = new_agent(library.action_size(), feature_set)
    best_agent.load_state_dict(best_state["agent_state_dict"])
    last_agent.load_state_dict(last_state["agent_state_dict"])
    test_instances = [make_stream_instance(test_seed) for test_seed in TEST_SEEDS]

    print("=" * 118)
    print("PPO 仅评估模式（不训练、不修改检查点）")
    print("=" * 118)
    print(f"规则库: {library_source} ({library.action_size()} actions)")
    print(f"检查点: best={best_path.resolve()}")
    print(f"          last={last_path.resolve()}")
    print(f"随机种子: network/sampling={seed}, instance_base={base_seed}")
    print_test_report(
        library=library,
        all_six_library=all_six_library,
        best_agent=best_agent,
        last_agent=last_agent,
        test_instances=test_instances,
        output_csv=Path(args.test_csv_path),
        output_summary=Path(
            getattr(
                args,
                "test_summary_path",
                Path(args.test_csv_path).with_suffix(".json"),
            )
        ),
        feature_set=feature_set,
    )


def run_training(args: argparse.Namespace) -> None:
    if args.episodes <= 0 or args.episodes_per_update <= 0:
        raise ValueError("episode counts must be positive")
    if args.eval_interval <= 0:
        raise ValueError("eval_interval must be positive")

    total_updates = math.ceil(args.episodes / args.episodes_per_update)
    library_file = getattr(args, "library_file", None)
    library_name = args.library if library_file is None else "json"
    source_metadata = library_source_metadata(args.library, library_file)
    seed = int(getattr(args, "seed", 42))
    base_seed = int(getattr(args, "base_seed", TRAIN_BASE_SEED))
    feature_set = getattr(args, "feature_set", "theory11")
    library, all_six_library = select_rule_library(
        args.library, library_file
    )
    library_hash = file_sha256(library_file)
    if library_hash is None:
        canonical_library = json.dumps(
            [rule.to_dict() for rule in library.rules],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        library_hash = hashlib.sha256(canonical_library).hexdigest()
    manifest_path = Path(
        getattr(
            args,
            "manifest_path",
            Path(args.last_checkpoint_path).with_name(
                f"{getattr(args, 'file_prefix', '')}train_manifest.json"
            ),
        )
    )
    write_manifest(
        manifest_path,
        {
            "kind": "ppo_training",
            "tag": getattr(args, "tag", ""),
            "library_source": source_metadata,
            "library_sha256": library_hash,
            "seeds": {
                "network": seed,
                "training_base": base_seed,
                "validation": list(VALIDATION_SEEDS),
                "test": list(TEST_SEEDS),
            },
            "config": {
                "episodes": args.episodes,
                "episodes_per_update": args.episodes_per_update,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "epochs_per_update": 4,
                "reward_scale": 10.0,
                "learning_rate": [LR_START, LR_END],
                "entropy_coef": [ENTROPY_START, ENTROPY_END],
                "hidden_dim": 64,
                "num_layers": 2,
                "num_heads": 4,
                "feature_set": feature_set,
            },
            "status": "started",
        },
    )
    validation_instances = [
        make_stream_instance(seed) for seed in VALIDATION_SEEDS
    ]
    validation_fixed = fixed_rule_results(library, validation_instances)
    validation_means = means_by_rule(validation_fixed)
    validation_best_name = min(validation_means, key=validation_means.get)
    validation_best_mean = validation_means[validation_best_name]

    random.seed(seed)
    torch.manual_seed(seed)
    agent = new_agent(library.action_size(), feature_set)
    trainer = PPOTrainer(
        agent=agent,
        library=library,
        instances=[],
        instance_factory=make_stream_instance,
        base_seed=base_seed,
        lr=LR_START,
        gamma=0.99,
        gae_lambda=0.95,
        epochs_per_update=4,
        reward_scale=10.0,
        entropy_coef=ENTROPY_START,
        feature_set=feature_set,
    )

    completed_updates = 0
    stats_history: list[dict[str, float]] = []
    curve: list[EvaluationPoint] = []
    best_ppo_f = float("inf")
    last_path = Path(args.last_checkpoint_path)
    if args.resume:
        state = load_checkpoint(last_path)
        saved_total = int(state["total_updates"])
        if saved_total != total_updates:
            raise ValueError(
                f"resume total_updates mismatch: checkpoint={saved_total}, "
                f"requested={total_updates}"
            )
        saved_episodes = state.get("target_episodes")
        saved_batch_size = state.get("episodes_per_update")
        if saved_episodes is not None and int(saved_episodes) != args.episodes:
            raise ValueError(
                f"resume episodes mismatch: checkpoint={saved_episodes}, "
                f"requested={args.episodes}"
            )
        if (
            saved_batch_size is not None
            and int(saved_batch_size) != args.episodes_per_update
        ):
            raise ValueError(
                "resume episodes_per_update mismatch: "
                f"checkpoint={saved_batch_size}, "
                f"requested={args.episodes_per_update}"
            )
        saved_source = state.get("library_source")
        if saved_source is not None and saved_source != source_metadata:
            raise ValueError(
                f"resume library source mismatch: checkpoint={saved_source}, "
                f"requested={source_metadata}"
            )
        saved_library = state.get("library_name")
        if (
            saved_source is None
            and saved_library is not None
            and saved_library != library_name
        ):
            raise ValueError(
                f"resume library mismatch: checkpoint={saved_library}, "
                f"requested={library_name}"
            )
        saved_seed = state.get("seed")
        if saved_seed is not None and int(saved_seed) != seed:
            raise ValueError(
                f"resume seed mismatch: checkpoint={saved_seed}, requested={seed}"
            )
        saved_base_seed = state.get("base_seed")
        if saved_base_seed is not None and int(saved_base_seed) != base_seed:
            raise ValueError(
                "resume base_seed mismatch: "
                f"checkpoint={saved_base_seed}, requested={base_seed}"
            )
        if state.get("feature_set", "theory11") != feature_set:
            raise ValueError("resume checkpoint feature_set mismatch")
        agent.load_state_dict(state["agent_state_dict"])
        trainer.optimizer.load_state_dict(state["optimizer_state_dict"])
        completed_updates = int(state["completed_updates"])
        trainer.global_episode_count = int(state["global_episode_count"])
        trainer.entropy_coef = float(state["entropy_coef"])
        stats_history = list(state.get("stats_history", []))
        curve = []
        for saved_point in state.get("evaluation_curve", []):
            point = dict(saved_point)
            point.setdefault("sampled_mean_f", point["ppo_mean_f"])
            point.setdefault("total_variation", 0.0)
            point.setdefault("switch_rate", 0.0)
            curve.append(EvaluationPoint(**point))
        best_ppo_f = float(state.get("best_ppo_f", float("inf")))
        random.setstate(state["python_random_state"])
        torch.random.set_rng_state(state["torch_random_state"])

    print("=" * 118)
    print("PPO streaming-instance 训练")
    print("=" * 118)
    print("规模: 15 jobs, 5 machines, 2 entities")
    print(
        f"规则库: {source_metadata} "
        f"({library.action_size()} actions), tag={args.tag or '<empty>'}"
    )
    print(
        f"训练: {args.episodes} episodes, "
        f"{args.episodes_per_update} episodes/update, {total_updates} updates"
    )
    if args.episodes % args.episodes_per_update:
        print(
            f"最后一次 update 使用 "
            f"{args.episodes % args.episodes_per_update} episodes"
        )
    print(f"已完成 updates: {completed_updates}")
    print(f"随机种子: network/sampling={seed}, instance_base={base_seed}")
    print(f"验证 seeds: {VALIDATION_SEEDS[0]}..{VALIDATION_SEEDS[-1]}")
    print(
        f"测试 seeds: {TEST_SEEDS[0]}..{TEST_SEEDS[-1]} "
        "(仅训练完成后使用)"
    )
    print(
        "PPO: epochs=4, gamma=0.99, gae_lambda=0.95, reward_scale=10, "
        "lr=3e-4->1e-4, entropy_coef=0.02->0.002"
    )
    print(
        f"验证集最优固定规则: {validation_best_name} "
        f"(平均 F={validation_best_mean:.2f})"
    )

    run_end = total_updates
    if args.max_updates_this_run is not None:
        run_end = min(total_updates, completed_updates + args.max_updates_this_run)

    for global_index in range(completed_updates, run_end):
        current_lr = linear_schedule(LR_START, LR_END, global_index, total_updates)
        current_entropy = linear_schedule(
            ENTROPY_START, ENTROPY_END, global_index, total_updates
        )
        for parameter_group in trainer.optimizer.param_groups:
            parameter_group["lr"] = current_lr
        trainer.entropy_coef = current_entropy
        episodes_this_update = min(
            args.episodes_per_update,
            args.episodes - global_index * args.episodes_per_update,
        )
        stats = trainer.train(
            total_updates=1,
            episodes_per_update=episodes_this_update,
        )[0]
        completed_updates = global_index + 1
        stats_history.append(
            {
                "total_reward": stats.total_reward,
                "entropy": stats.entropy,
                "learning_rate": current_lr,
                "entropy_coef": current_entropy,
            }
        )

        should_evaluate = (
            completed_updates % args.eval_interval == 0
            or completed_updates == total_updates
        )
        validation_metrics = None
        if should_evaluate:
            greedy_result = evaluate_policy(
                agent,
                library,
                validation_instances,
                greedy=True,
                feature_set=feature_set,
            )
            sampled_result = evaluate_policy(
                agent,
                library,
                validation_instances,
                greedy=False,
                repeats=5,
                base_seed=10_000_000 + completed_updates * 10_000,
                feature_set=feature_set,
            )
            ppo_mean = sum(greedy_result.values) / len(greedy_result.values)
            sampled_mean = (
                sum(sampled_result.values) / len(sampled_result.values)
            )
            recent = stats_history[-args.eval_interval :]
            recent_reward = sum(item["total_reward"] for item in recent) / len(recent)
            gap = (
                (ppo_mean - validation_best_mean) / validation_best_mean * 100.0
            )
            point = EvaluationPoint(
                update=completed_updates,
                episodes=min(
                    completed_updates * args.episodes_per_update,
                    args.episodes,
                ),
                ppo_mean_f=ppo_mean,
                sampled_mean_f=sampled_mean,
                fixed_mean_f=validation_best_mean,
                gap_percent=gap,
                entropy=stats.entropy,
                recent_reward=recent_reward,
                rule_usage=greedy_result.usage,
                total_variation=greedy_result.total_variation or 0.0,
                switch_rate=greedy_result.switch_rate or 0.0,
            )
            curve.append(point)
            validation_metrics = asdict(point)
            if ppo_mean < best_ppo_f:
                best_ppo_f = ppo_mean
                save_checkpoint(
                    Path(args.best_checkpoint_path),
                    checkpoint_payload(
                        agent=agent,
                        trainer=trainer,
                        completed_updates=completed_updates,
                        total_updates=total_updates,
                        target_episodes=args.episodes,
                        episodes_per_update=args.episodes_per_update,
                        stats_history=stats_history,
                        curve=curve,
                        best_ppo_f=best_ppo_f,
                        library_name=library_name,
                        library_source=source_metadata,
                        seed=seed,
                        base_seed=base_seed,
                        validation_metrics=validation_metrics,
                    ),
                )
            print(
                f"EVAL update={completed_updates:>3} "
                f"episodes={point.episodes:>5} | greedy_F={ppo_mean:.2f} "
                f"sampled_F={sampled_mean:.2f} "
                f"fixed_F={validation_best_mean:.2f} gap={gap:+.1f}% | "
                f"entropy={stats.entropy:.4f} reward={recent_reward:+.4f} | "
                f"lr={current_lr:.7f} ent_coef={current_entropy:.5f} | "
                f"TV={point.total_variation:.4f} "
                f"switch={point.switch_rate:.2%} | "
                f"usage={greedy_result.usage}",
                flush=True,
            )

        payload = checkpoint_payload(
            agent=agent,
            trainer=trainer,
            completed_updates=completed_updates,
            total_updates=total_updates,
            target_episodes=args.episodes,
            episodes_per_update=args.episodes_per_update,
            stats_history=stats_history,
            curve=curve,
            best_ppo_f=best_ppo_f,
            library_name=library_name,
            library_source=source_metadata,
            seed=seed,
            base_seed=base_seed,
            validation_metrics=validation_metrics,
        )
        if should_evaluate:
            save_checkpoint(last_path, payload)
        if completed_updates % 100 == 0:
            save_checkpoint(
                last_path.with_name(
                    f"{args.file_prefix}ppo_update_{completed_updates:04d}.pt"
                ),
                payload,
            )

    if completed_updates < total_updates:
        save_checkpoint(
            last_path,
            checkpoint_payload(
                agent=agent,
                trainer=trainer,
                completed_updates=completed_updates,
                total_updates=total_updates,
                target_episodes=args.episodes,
                episodes_per_update=args.episodes_per_update,
                stats_history=stats_history,
                curve=curve,
                best_ppo_f=best_ppo_f,
                library_name=library_name,
                library_source=source_metadata,
                seed=seed,
                base_seed=base_seed,
            ),
        )
        print(
            f"\n本次运行暂停于 update {completed_updates}/{total_updates}; "
            "可使用 --resume 继续。"
        )
        return

    late_window = stats_history[-min(25, len(stats_history)) :]
    late_entropy = sum(item["entropy"] for item in late_window) / len(late_window)
    entropy_threshold = math.log(library.action_size()) * 0.70
    entropy_converged = late_entropy < entropy_threshold
    comparison_window = min(5, max(1, len(curve) // 2))
    first_mean = sum(
        point.ppo_mean_f for point in curve[:comparison_window]
    ) / comparison_window
    last_mean = sum(
        point.ppo_mean_f for point in curve[-comparison_window:]
    ) / comparison_window
    curve_converged = last_mean < first_mean
    quality_converged = best_ppo_f <= validation_best_mean * 1.01
    print("\n" + "=" * 118)
    print("验证集收敛判据")
    print("=" * 118)
    print(
        f"(1) 末期熵 {late_entropy:.4f} < {entropy_threshold:.4f}: "
        f"{'满足' if entropy_converged else '不满足'}"
    )
    print(
        f"(2) 后{comparison_window}次验证 F {last_mean:.2f} < "
        f"前{comparison_window}次 {first_mean:.2f}: "
        f"{'满足' if curve_converged else '不满足'}"
    )
    print(
        f"(3) 最佳验证 F {best_ppo_f:.2f} <= "
        f"1.01×固定最优 {validation_best_mean * 1.01:.2f}: "
        f"{'满足' if quality_converged else '不满足'}"
    )

    print("\n训练完成，开始仅一次的独立测试集评估...", flush=True)
    best_state = load_checkpoint(Path(args.best_checkpoint_path))
    last_state = load_checkpoint(last_path)
    best_agent = new_agent(library.action_size(), feature_set)
    last_agent = new_agent(library.action_size(), feature_set)
    best_agent.load_state_dict(best_state["agent_state_dict"])
    last_agent.load_state_dict(last_state["agent_state_dict"])
    test_instances = [make_stream_instance(seed) for seed in TEST_SEEDS]
    print_test_report(
        library=library,
        all_six_library=all_six_library,
        best_agent=best_agent,
        last_agent=last_agent,
        test_instances=test_instances,
        output_csv=Path(args.test_csv_path),
        output_summary=Path(args.test_summary_path),
        feature_set=feature_set,
    )


def parse_args() -> argparse.Namespace:
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    library_group = parser.add_mutually_exclusive_group()
    library_group.add_argument(
        "--library", choices=("all6", "top3"), default=None
    )
    library_group.add_argument("--library_file", "--library-file", dest="library_file")
    parser.add_argument("--tag", default="")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="network initialisation and policy-sampling seed",
    )
    parser.add_argument(
        "--base_seed",
        "--base-seed",
        dest="base_seed",
        type=int,
        default=TRAIN_BASE_SEED,
        help="first streaming training-instance seed",
    )
    parser.add_argument(
        "--episodes",
        "--total-episodes",
        dest="episodes",
        type=int,
        default=2_000,
    )
    parser.add_argument(
        "--episodes_per_update",
        "--episodes-per-update",
        dest="episodes_per_update",
        type=int,
        default=16,
    )
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--feature_set",
        "--feature-set",
        dest="feature_set",
        choices=("theory11", "theory11_plus3"),
        default="theory11",
    )
    parser.add_argument(
        "--eval_only",
        "--eval-only",
        dest="eval_only",
        action="store_true",
        help="evaluate existing best/last checkpoints without training",
    )
    parser.add_argument("--max-updates-this-run", type=int)
    args = parser.parse_args()
    if args.library is None and args.library_file is None:
        args.library = "all6"
    if any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in args.tag):
        parser.error("--tag may contain only letters, digits, '_' and '-'")
    if args.eval_only and args.resume:
        parser.error("--eval_only and --resume are mutually exclusive")
    args.file_prefix = f"{args.tag}_" if args.tag else ""
    args.log_path = str(base / f"{args.file_prefix}train_streaming.log")
    args.best_checkpoint_path = str(
        base / f"{args.file_prefix}best_ppo.pt"
    )
    args.last_checkpoint_path = str(
        base / f"{args.file_prefix}last_ppo.pt"
    )
    args.test_csv_path = str(
        base / f"{args.file_prefix}test_per_instance.csv"
    )
    args.test_summary_path = str(
        base / f"{args.file_prefix}test_summary.json"
    )
    args.manifest_path = str(base / f"{args.file_prefix}train_manifest.json")
    return args


def main() -> None:
    args = parse_args()
    log_path = Path(args.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    original_stdout, original_stderr = sys.stdout, sys.stderr
    mode = "a" if args.resume or args.eval_only else "w"
    with log_path.open(mode, encoding="utf-8", buffering=1) as log_file:
        sys.stdout = Tee(original_stdout, log_file)
        sys.stderr = Tee(original_stderr, log_file)
        try:
            if args.eval_only:
                run_evaluation_only(args)
            else:
                run_training(args)
        finally:
            sys.stdout, sys.stderr = original_stdout, original_stderr


if __name__ == "__main__":
    main()
