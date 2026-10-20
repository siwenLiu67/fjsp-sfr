"""验证框架有效性：规则库 + PPO 选择 vs 单规则 vs 随机。"""

from __future__ import annotations

import random
import sys
import time
from copy import deepcopy
from pathlib import Path

import torch

# This checkout is the ``fjsp_sfr`` package directory itself rather than a
# conventional project root.  Keep the documented direct script invocation
# working without requiring an editable installation.
PACKAGE_PARENT = Path(__file__).resolve().parents[2]
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from fjsp_sfr.core.models import FJSPInstance
from fjsp_sfr.core.objective import compute_objective
from fjsp_sfr.dispatch.rule_library import RuleLibrary
from fjsp_sfr.dispatch.symbolic_rule import RuleType, SymbolicRule
from fjsp_sfr.dispatch.theory_features import build_extended_theory_features
from fjsp_sfr.generation import InstanceConfig, generate_instance
from fjsp_sfr.rl.ca_observation import resolve_observation_scales
from fjsp_sfr.rl.ca_rule_selection import CAPPORuleSelectionEnv
from fjsp_sfr.rl.graph import build_hetero_graph
from fjsp_sfr.rl.ppo_agent import HGTPPOAgent
from fjsp_sfr.rl.ppo_trainer import PPOTrainer


def make_fjsp_instance(
    seed: int = 42,
    *,
    n_jobs: int = 30,
    n_machines: int = 8,
    n_entities: int = 3,
) -> FJSPInstance:
    """构造规模可配置的验证实例。"""

    config = InstanceConfig(
        num_instances=1,
        num_jobs=n_jobs,
        num_machines=n_machines,
        num_entities=n_entities,
        ops_per_job=(2, 4),
        checkpoints_per_entity=(2, 2),
        checkpoint_tightness=1.0,
        checkpoint_profile_pool=("balanced_2", "early_pressure_2"),
        eligible_machines_range=(1, 3),
        group_name="framework_validation",
    )
    return generate_instance(config, seed=seed)


RULES = (
    SymbolicRule(
        name="gap_priority",
        expression="Prioritize high gap pressure",
        rule_type=RuleType.TYPE_I,
        priority_expr="gap_pressure",
    ),
    SymbolicRule(
        name="tardiness_avoidance",
        expression="Avoid tardiness",
        rule_type=RuleType.TYPE_I,
        priority_expr="-tardiness_exposure",
    ),
    SymbolicRule(
        name="due_date_first",
        expression="Earliest due date slack",
        rule_type=RuleType.TYPE_I,
        priority_expr="-due_date_slack",
    ),
    SymbolicRule(
        name="service_value",
        expression="Multi-checkpoint service value",
        rule_type=RuleType.TYPE_I,
        priority_expr="multi_checkpoint_value",
    ),
    SymbolicRule(
        name="gap_tard_combo",
        expression="Balanced gap and tardiness",
        rule_type=RuleType.TYPE_III,
        priority_expr=(
            "0.5 * gap_pressure + 0.3 * tardiness_exposure "
            "+ 0.2 * forecast_deficit"
        ),
    ),
    SymbolicRule(
        name="capacity_aware",
        expression="Capacity-aware with filtering",
        rule_type=RuleType.TYPE_IV,
        priority_expr="gap_pressure + 0.5 * capacity_pressure",
        filter_expr="quantity_contribution - 0.05",
    ),
)

TRAIN_SEEDS = tuple(range(42, 52))
EVAL_SEEDS = tuple(range(100, 120))


def build_rule_library() -> RuleLibrary:
    """构建验证实验使用的固定规则库。"""

    return RuleLibrary(rules=RULES)


def evaluate_single_rule(rule: SymbolicRule, instance: FJSPInstance) -> float:
    """用一条规则跑完整 episode，返回 F。"""

    rollout_instance = deepcopy(instance)
    env = CAPPORuleSelectionEnv(rollout_instance)
    env.reset()
    while env.current_metrics and env.current_metrics.is_active:
        result = env.step_principle(0, rule)
        if result.terminated:
            break
    return float(compute_objective(rollout_instance, env._state).Z)


def evaluate_random_policy(
    library: RuleLibrary,
    instance: FJSPInstance,
    seed: int = 0,
) -> float:
    """在每个决策点均匀随机选一条规则。"""

    rng = random.Random(seed)
    rollout_instance = deepcopy(instance)
    env = CAPPORuleSelectionEnv(rollout_instance)
    env.reset()
    while env.current_metrics and env.current_metrics.is_active:
        rule_index = rng.randrange(library.action_size())
        result = env.step_principle(
            rule_index, library.get_rule(rule_index)
        )
        if result.terminated:
            break
    return float(compute_objective(rollout_instance, env._state).Z)


def compute_f_ref(
    library: RuleLibrary,
    instances: list[FJSPInstance],
) -> float:
    """返回所有训练实例、所有单规则的平均 F 值。"""

    if not instances:
        raise ValueError("instances must not be empty")
    values = [
        evaluate_single_rule(rule, instance)
        for instance in instances
        for rule in library.rules
    ]
    return max(sum(values) / len(values), 1.0)


def train_ppo_policy(
    library: RuleLibrary,
    training_instances: list[FJSPInstance],
    f_ref: float,
    *,
    explore_updates: int = 60,
    exploit_updates: int = 140,
    finetune_updates: int = 100,
) -> HGTPPOAgent:
    """用探索、训练和价值收敛三个阶段训练 PPO。"""

    agent = HGTPPOAgent(
        action_size=library.action_size(),
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
    )
    trainer_explore = PPOTrainer(
        agent=agent,
        library=library,
        instances=training_instances,
        lr=1e-4,
        epochs_per_update=3,
        f_ref=f_ref,
        gamma=1.0,
        gae_lambda=0.95,
        entropy_coef=0.2,
    )

    def log_fn(
        index,
        stats,
        *,
        offset: int = 0,
        include_value_loss: bool = False,
    ) -> None:
        absolute_index = offset + index
        message = (
            f"  Update {absolute_index}: reward={stats.total_reward:.4f}, "
            f"policy_loss={stats.policy_loss:.4f}"
        )
        if include_value_loss:
            message += f", value_loss={stats.value_loss:.4f}"
        message += (
            f", entropy={stats.entropy:.4f}, "
            f"adv_std={stats.advantage_std:.4f}, "
            f"clip_fraction={stats.clip_fraction:.4f}, "
            f"approx_kl={stats.approx_kl:.6f}"
        )
        print(message, flush=True)

    started_at = time.time()
    print(
        f"\nPhase 1: 探索 ({explore_updates} updates, "
        "entropy_coef=0.2, lr=1e-4, gamma=1.0)...",
        flush=True,
    )
    trainer_explore.train(
        total_updates=explore_updates,
        episodes_per_update=4,
        log_interval=20,
        log_fn=log_fn,
    )

    if exploit_updates:
        trainer_exploit = PPOTrainer(
            agent=agent,
            library=library,
            instances=training_instances,
            lr=1e-4,
            epochs_per_update=3,
            f_ref=f_ref,
            gamma=0.99,
            gae_lambda=0.95,
            entropy_coef=0.05,
        )
        print(
            f"\nPhase 2: 训练 ({exploit_updates} updates, "
            "entropy_coef=0.05, lr=1e-4, gamma=0.99)...",
            flush=True,
        )
        trainer_exploit.train(
            total_updates=exploit_updates,
            episodes_per_update=4,
            log_interval=20,
            log_fn=lambda index, stats: log_fn(
                index, stats, offset=explore_updates
            ),
        )

    if finetune_updates:
        trainer_finetune = PPOTrainer(
            agent=agent,
            library=library,
            instances=training_instances,
            lr=5e-5,
            epochs_per_update=5,
            f_ref=f_ref,
            gamma=0.99,
            gae_lambda=0.95,
            entropy_coef=0.01,
        )
        print(
            f"\nPhase 3: 收敛 ({finetune_updates} updates, "
            "entropy_coef=0.01, lr=5e-5, gamma=0.99, "
            "epochs_per_update=5, episodes_per_update=8)...",
            flush=True,
        )
        phase_three_offset = explore_updates + exploit_updates
        trainer_finetune.train(
            total_updates=finetune_updates,
            episodes_per_update=8,
            log_interval=20,
            log_fn=lambda index, stats: log_fn(
                index,
                stats,
                offset=phase_three_offset,
                include_value_loss=True,
            ),
        )
    print(f"训练耗时: {time.time() - started_at:.1f}s", flush=True)
    return agent


def evaluate_ppo_greedy(
    agent: HGTPPOAgent,
    library: RuleLibrary,
    instance: FJSPInstance,
) -> float:
    """用已训练 PPO 的贪心动作跑完整 episode。"""

    agent.eval()
    rollout_instance = deepcopy(instance)
    env = CAPPORuleSelectionEnv(rollout_instance)
    env.reset()
    while env.current_metrics and env.current_metrics.is_active:
        problem = env._problem_view()
        state = env._state
        metrics = env.current_metrics
        time_scale, processing_scale = resolve_observation_scales(
            problem, state
        )
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
        result = env.step_principle(action, library.get_rule(action))
        if result.terminated:
            break

    return float(compute_objective(rollout_instance, env._state).Z)


def run_full_evaluation() -> None:
    """在独立的多种子实例上训练并评估三类策略。"""

    random.seed(42)
    torch.manual_seed(42)
    library = build_rule_library()

    training_instances = [make_fjsp_instance(seed) for seed in TRAIN_SEEDS]
    sample = training_instances[0]
    print("=" * 82)
    print("Paper A 多实例框架验证实验")
    print("=" * 82)
    print(
        f"实例规模: {sample.num_jobs} jobs, "
        f"{sample.num_machines} machines, {sample.num_entities} entities"
    )
    print(f"目标: F = {sample.alpha}*TT + {sample.beta}*WSF")
    print(f"规则库: {library.action_size()} 条规则")
    print(f"训练实例 seeds: {list(TRAIN_SEEDS)}")
    print(f"评估实例 seeds: {list(EVAL_SEEDS)}")

    print("\n计算训练集基线...", flush=True)
    f_ref = compute_f_ref(library, training_instances)
    print(f"f_ref = {f_ref:.2f}", flush=True)

    agent = train_ppo_policy(
        library,
        training_instances,
        f_ref,
        explore_updates=60,
        exploit_updates=140,
        finetune_updates=100,
    )

    print("\n" + "=" * 82)
    print("独立评估集结果")
    print("=" * 82)
    print(
        f"{'Seed':>6} | {'最优单规则 F':>12} {'(规则)':>22} | "
        f"{'随机 F':>10} | {'PPO F':>10} | {'PPO赢?':>7}"
    )
    print("-" * 82)

    ppo_wins = 0
    sum_best = 0.0
    sum_random = 0.0
    sum_ppo = 0.0
    rule_totals = {rule.name: 0.0 for rule in library.rules}
    eval_instances = [make_fjsp_instance(seed) for seed in EVAL_SEEDS]

    for eval_seed, instance in zip(EVAL_SEEDS, eval_instances, strict=True):
        single_results = [
            (evaluate_single_rule(rule, instance), rule.name)
            for rule in library.rules
        ]
        for objective, rule_name in single_results:
            rule_totals[rule_name] += objective
        best_f, best_name = min(single_results, key=lambda item: item[0])

        random_values = [
            evaluate_random_policy(
                library,
                instance,
                seed=eval_seed * 10_000 + trial,
            )
            for trial in range(10)
        ]
        random_f = sum(random_values) / len(random_values)
        ppo_f = evaluate_ppo_greedy(agent, library, instance)
        won = ppo_f < best_f
        ppo_wins += int(won)
        sum_best += best_f
        sum_random += random_f
        sum_ppo += ppo_f

        print(
            f"{eval_seed:>6} | {best_f:>12.2f} {best_name:>22} | "
            f"{random_f:>10.2f} | {ppo_f:>10.2f} | "
            f"{'✓' if won else '':>7}",
            flush=True,
        )

    count = len(EVAL_SEEDS)
    avg_best = sum_best / count
    avg_random = sum_random / count
    avg_ppo = sum_ppo / count
    print("-" * 82)
    print(
        f"{'平均':>6} | {avg_best:>12.2f} {'':>22} | "
        f"{avg_random:>10.2f} | {avg_ppo:>10.2f} |"
    )
    print(
        f"\nPPO 赢过最优单规则的比例: "
        f"{ppo_wins}/{count} = {ppo_wins / count * 100:.1f}%"
    )

    # 最优固定单规则分析：每个实例都始终使用同一条规则。
    print("\n" + "=" * 70)
    print("各规则在全部评估实例上的平均 F:")
    print("-" * 70)

    rule_avg = {
        rule.name: rule_totals[rule.name] / count for rule in library.rules
    }
    for rule in library.rules:
        print(f"  {rule.name:>22}: {rule_avg[rule.name]:.2f}")

    best_fixed_name = min(rule_avg, key=rule_avg.get)
    best_fixed_f = rule_avg[best_fixed_name]
    print(
        f"\n最优固定单规则: {best_fixed_name} "
        f"(平均 F={best_fixed_f:.2f})"
    )
    print(f"PPO 平均 F:     {avg_ppo:.2f}")
    print(f"随机 平均 F:    {avg_random:.2f}")
    print(
        f"PPO vs 最优固定: "
        f"{(avg_ppo - best_fixed_f) / best_fixed_f * 100:+.1f}%"
    )
    print(
        f"PPO vs 随机:     "
        f"{(avg_ppo - avg_random) / avg_random * 100:+.1f}%"
    )


if __name__ == "__main__":
    run_full_evaluation()
