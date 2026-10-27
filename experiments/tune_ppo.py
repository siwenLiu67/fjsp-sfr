"""15-job PPO convergence diagnostics and hyperparameter comparison."""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_PROJECT_PARENT = _PROJECT_ROOT.parent
for _import_path in (str(_PROJECT_ROOT), str(_PROJECT_PARENT)):
    if _import_path not in sys.path:
        sys.path.insert(0, _import_path)

import math
import random
import sys
import time
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import torch

PACKAGE_PARENT = Path(__file__).resolve().parents[2]
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from fjsp_sfr.core.objective import compute_objective
from fjsp_sfr.dispatch.rule_library import RuleLibrary
from fjsp_sfr.dispatch.theory_features import build_extended_theory_features
from fjsp_sfr.env.ca_observation import resolve_observation_scales
from fjsp_sfr.env.ca_rule_selection import CAPPORuleSelectionEnv
from fjsp_sfr.experiments.validate_framework import (
    EVAL_SEEDS,
    TRAIN_SEEDS,
    build_rule_library,
    evaluate_single_rule,
    make_fjsp_instance,
)
from fjsp_sfr.rl.graph import build_hetero_graph
from fjsp_sfr.rl.ppo_agent import HGTPPOAgent
from fjsp_sfr.rl.ppo_trainer import PPOTrainer, TrainStats


TUNE_EVAL_SEEDS = EVAL_SEEDS[:10]


@dataclass(frozen=True)
class TuneConfig:
    name: str
    entropy_coef: float
    reward_scale: float
    lr: float


@dataclass
class TuneResult:
    config: TuneConfig
    final_entropy: float
    early_reward: float
    late_reward: float
    ppo_mean_f: float
    best_fixed_mean_f: float
    gap_percent: float
    rule_usage: dict[str, int]
    entropy_converged: bool
    reward_improved: bool
    evaluation_converged: bool


CONFIGS = (
    TuneConfig("A", entropy_coef=0.01, reward_scale=1.0, lr=3e-4),
    TuneConfig("B", entropy_coef=0.005, reward_scale=10.0, lr=3e-4),
    TuneConfig("C", entropy_coef=0.001, reward_scale=10.0, lr=3e-4),
    TuneConfig("D", entropy_coef=0.01, reward_scale=10.0, lr=1e-4),
)


def build_instances(seeds: tuple[int, ...]) -> list:
    return [
        make_fjsp_instance(
            seed,
            n_jobs=15,
            n_machines=5,
            n_entities=2,
        )
        for seed in seeds
    ]


def compute_instance_f_refs(
    library: RuleLibrary,
    instances: list,
) -> list[float]:
    """Compute one mean-single-rule baseline for each training instance."""

    return [
        max(
            sum(evaluate_single_rule(rule, instance) for rule in library.rules)
            / library.action_size(),
            1.0,
        )
        for instance in instances
    ]


def compute_fixed_rule_averages(
    library: RuleLibrary,
    instances: list,
) -> dict[str, float]:
    return {
        rule.name: sum(evaluate_single_rule(rule, instance) for instance in instances)
        / len(instances)
        for rule in library.rules
    }


def evaluate_ppo_greedy_with_usage(
    agent: HGTPPOAgent,
    library: RuleLibrary,
    instance,
) -> tuple[float, Counter[str]]:
    rollout_instance = deepcopy(instance)
    env = CAPPORuleSelectionEnv(rollout_instance)
    env.reset()
    usage: Counter[str] = Counter()
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
        )
        graph = build_hetero_graph(
            problem,
            state,
            metrics,
            features,
            time_scale=time_scale,
            processing_scale=processing_scale,
        )
        mask = torch.ones(library.action_size(), dtype=torch.bool)
        action = agent.greedy_action(graph, mask)
        rule = library.get_rule(action)
        usage[rule.name] += 1
        result = env.step_principle(action, rule)
        if result.terminated:
            break

    objective = compute_objective(rollout_instance, env._state)
    return float(objective.Z), usage


def log_update(index: int, stats: TrainStats) -> None:
    print(
        f"  Update {index:>3}: reward={stats.total_reward:+.4f}, "
        f"adv_std={stats.advantage_std:.4f}, "
        f"policy_loss={stats.policy_loss:+.4f}, "
        f"value_loss={stats.value_loss:.4f}, "
        f"entropy={stats.entropy:.4f}, "
        f"clip_fraction={stats.clip_fraction:.4f}, "
        f"approx_kl={stats.approx_kl:.6f}",
        flush=True,
    )


def run_config(
    config: TuneConfig,
    library: RuleLibrary,
    training_instances: list,
    evaluation_instances: list,
    f_refs: list[float],
    best_fixed_mean_f: float,
) -> TuneResult:
    # Give all configurations identical initialization and sampling streams.
    random.seed(42)
    torch.manual_seed(42)
    agent = HGTPPOAgent(
        action_size=library.action_size(),
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
    )
    trainer = PPOTrainer(
        agent=agent,
        library=library,
        instances=training_instances,
        lr=config.lr,
        gamma=0.99,
        gae_lambda=0.95,
        entropy_coef=config.entropy_coef,
        epochs_per_update=4,
        f_ref=f_refs,
        reward_scale=config.reward_scale,
    )

    print("\n" + "=" * 104)
    print(
        f"Config {config.name}: entropy_coef={config.entropy_coef}, "
        f"reward_scale={config.reward_scale:g}, lr={config.lr:g}"
    )
    print("=" * 104, flush=True)
    started_at = time.time()
    stats = trainer.train(
        total_updates=150,
        episodes_per_update=16,
        log_interval=10,
        log_fn=log_update,
    )
    elapsed = time.time() - started_at

    early_reward = sum(item.total_reward for item in stats[:20]) / 20
    late_reward = sum(item.total_reward for item in stats[-20:]) / 20
    final_entropy = sum(item.entropy for item in stats[-20:]) / 20

    ppo_values: list[float] = []
    total_usage: Counter[str] = Counter()
    for instance in evaluation_instances:
        objective, usage = evaluate_ppo_greedy_with_usage(
            agent, library, instance
        )
        ppo_values.append(objective)
        total_usage.update(usage)
    ppo_mean_f = sum(ppo_values) / len(ppo_values)
    gap_percent = (
        (ppo_mean_f - best_fixed_mean_f) / best_fixed_mean_f * 100.0
    )
    initial_entropy = math.log(library.action_size())
    result = TuneResult(
        config=config,
        final_entropy=final_entropy,
        early_reward=early_reward,
        late_reward=late_reward,
        ppo_mean_f=ppo_mean_f,
        best_fixed_mean_f=best_fixed_mean_f,
        gap_percent=gap_percent,
        rule_usage=dict(total_usage),
        entropy_converged=final_entropy < initial_entropy * 0.70,
        reward_improved=late_reward > early_reward,
        evaluation_converged=ppo_mean_f <= best_fixed_mean_f * 1.01,
    )

    print(f"\nConfig {config.name} 结果 (训练耗时 {elapsed:.1f}s):")
    print(f"  训练末期熵(后20轮平均): {final_entropy:.4f}")
    print(f"  reward 前20轮均值:       {early_reward:+.4f}")
    print(f"  reward 后20轮均值:       {late_reward:+.4f}")
    print(f"  评估集 PPO 平均 F:       {ppo_mean_f:.2f}")
    print(f"  最优固定单规则平均 F:   {best_fixed_mean_f:.2f}")
    print(f"  PPO vs 最优固定:          {gap_percent:+.1f}%")
    print(f"  PPO 贪心规则使用分布:      {dict(total_usage)}")
    return result


def print_summary(results: list[TuneResult], action_size: int) -> None:
    best = min(results, key=lambda result: result.ppo_mean_f)
    print("\n" + "=" * 112)
    print("配置汇总（← 为最优配置）")
    print("=" * 112)
    print(
        f"{'配置':>4} | {'entropy':>9} | {'reward前20':>11} | "
        f"{'reward后20':>11} | {'PPO F':>9} | {'固定规则 F':>10} | "
        f"{'差距':>8} | {'收敛项':>6}"
    )
    print("-" * 112)
    for result in results:
        checks = sum(
            (
                result.entropy_converged,
                result.reward_improved,
                result.evaluation_converged,
            )
        )
        marker = " ←" if result is best else ""
        print(
            f"{result.config.name:>4} | {result.final_entropy:>9.4f} | "
            f"{result.early_reward:>+11.4f} | {result.late_reward:>+11.4f} | "
            f"{result.ppo_mean_f:>9.2f} | {result.best_fixed_mean_f:>10.2f} | "
            f"{result.gap_percent:>+7.1f}% | {checks}/3{marker}"
        )

    threshold = math.log(action_size) * 0.70
    print("\n" + "=" * 112)
    print(f"最优配置: {best.config.name}")
    print("收敛判据:")
    print(
        f"  (1) 末期熵 {best.final_entropy:.4f} < "
        f"0.7×ln({action_size})={threshold:.4f}: "
        f"{'满足' if best.entropy_converged else '不满足'}"
    )
    print(
        f"  (2) 后20轮 reward {best.late_reward:+.4f} > "
        f"前20轮 {best.early_reward:+.4f}: "
        f"{'满足' if best.reward_improved else '不满足'}"
    )
    print(
        f"  (3) PPO F {best.ppo_mean_f:.2f} <= "
        f"1.01×最优固定 {best.best_fixed_mean_f * 1.01:.2f}: "
        f"{'满足' if best.evaluation_converged else '不满足'}"
    )
    print(
        "总体结论: "
        + ("已收敛" if all((best.entropy_converged, best.reward_improved, best.evaluation_converged)) else "未同时满足三条收敛判据")
    )


def main() -> None:
    library = build_rule_library()
    training_instances = build_instances(TRAIN_SEEDS)
    evaluation_instances = build_instances(TUNE_EVAL_SEEDS)

    print("=" * 104)
    print("PPO 小规模调参实验")
    print("=" * 104)
    print("规模: 15 jobs, 5 machines, 2 entities")
    print(f"训练 seeds: {list(TRAIN_SEEDS)}")
    print(f"评估 seeds: {list(TUNE_EVAL_SEEDS)}")
    print("每配置: 150 updates, 16 episodes/update, 4 epochs/update")

    print("\n计算逐实例 f_ref 与固定规则基线...", flush=True)
    f_refs = compute_instance_f_refs(library, training_instances)
    print("f_ref by train seed:")
    for seed, value in zip(TRAIN_SEEDS, f_refs, strict=True):
        print(f"  seed {seed}: {value:.2f}")

    fixed_averages = compute_fixed_rule_averages(
        library, evaluation_instances
    )
    best_fixed_name = min(fixed_averages, key=fixed_averages.get)
    best_fixed_mean_f = fixed_averages[best_fixed_name]
    print("评估集固定规则平均 F:")
    for name, value in fixed_averages.items():
        print(f"  {name:>22}: {value:.2f}")
    print(
        f"最优固定单规则: {best_fixed_name} "
        f"(平均 F={best_fixed_mean_f:.2f})"
    )

    results = [
        run_config(
            config,
            library,
            training_instances,
            evaluation_instances,
            f_refs,
            best_fixed_mean_f,
        )
        for config in CONFIGS
    ]
    print_summary(results, library.action_size())


if __name__ == "__main__":
    main()
