"""One reproducible formal-experiment run (training or evaluation baseline)."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import statistics
import time
from collections import Counter
from copy import deepcopy
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import torch

torch.set_num_threads(1)

from fjsp_sfr.core.objective import compute_objective
from fjsp_sfr.dispatch.rule_library import RuleLibrary
from fjsp_sfr.dispatch.theory_features import build_extended_theory_features, feature_names_for
from fjsp_sfr.env.ca_observation import resolve_observation_scales
from fjsp_sfr.env.ca_rule_selection import CAPPORuleSelectionEnv
from fjsp_sfr.experiments.manifest import file_sha256, write_manifest
from fjsp_sfr.experiments.paperA_contract import (
    FORMAL_EXPERIMENT_CONTRACTS,
    FormalExperimentContract,
    require_runtime_ready,
    resolve_formal_contract,
)
from fjsp_sfr.experiments.train_streaming import linear_schedule
from fjsp_sfr.experiments.validate_framework import build_rule_library
from fjsp_sfr.generation.scale_config import make_scaled_instance
from fjsp_sfr.generation.scenario_generator import (
    seed_values, make_scenario_instance, load_scenario_config, scenario_config_hash,
)
from fjsp_sfr.rl.dqn_agent import DQNTrainer, HGTDQNAgent
from fjsp_sfr.rl.graph import apply_information_masks, build_hetero_graph
from fjsp_sfr.rl.mlp_agent import MLPPPOAgent
from fjsp_sfr.rl.ppo_agent import HGTPPOAgent
from fjsp_sfr.rl.action_groups import aggregate_rule_log_probs
from fjsp_sfr.rl.ppo_trainer import PPOTrainer, concrete_rule_action_groups
from fjsp_sfr.rl.reward_scale import load_global_reward_scale


# Historical compatibility constant.  New callers must select a named
# contract with --contract-config; this alias must not choose the formal path.
FEATURE_SET = "theory11_plus3"
VALIDATION_SEEDS = tuple(range(100, 150))
TRAIN_SEEDS = tuple(range(1000, 6000))
SELECTION_SEEDS = tuple(range(600, 620))
HOLDOUT_SEEDS = tuple(range(700, 720))
TEST_SEEDS_BY_SCALE = {
    "S": tuple(range(10_000, 10_100)),
    "M": tuple(range(10_100, 10_200)),
    "L": tuple(range(10_200, 10_300)),
    "X": tuple(range(10_300, 10_400)),
}


def load_library(path: str | None, *, contract: FormalExperimentContract | None = None) -> RuleLibrary:
    library = build_rule_library() if path == "handwritten6" else RuleLibrary.from_json(path)
    if contract is not None and contract.name == "paper_20261030":
        library.validate_protocol(contract.feature_set, formal=True)
    return library


def graph_for(
    env,
    *,
    contract: FormalExperimentContract,
    mask_checkpoint=False,
    mask_forecast=False,
):
    graph, _rows = graph_and_features_for(
        env,
        contract=contract,
        mask_checkpoint=mask_checkpoint,
        mask_forecast=mask_forecast,
    )
    return graph


def graph_and_features_for(
    env,
    *,
    contract: FormalExperimentContract,
    mask_checkpoint=False,
    mask_forecast=False,
):
    problem, state, metrics = env._problem_view(), env._state, env.current_metrics
    time_scale, processing_scale = resolve_observation_scales(problem, state)
    rows = build_extended_theory_features(
        problem,
        state,
        metrics,
        time_scale=time_scale,
        feature_set=contract.feature_set,
    )
    graph = build_hetero_graph(
        problem,
        state,
        metrics,
        rows,
        time_scale=time_scale,
        processing_scale=processing_scale,
        feature_set=contract.feature_set,
        graph_config=contract.graph_config,
    )
    return apply_information_masks(
        graph,
        feature_set=contract.feature_set,
        mask_checkpoint=mask_checkpoint,
        mask_forecast=mask_forecast,
    ), rows


def make_agent(
    selector: str,
    action_size: int,
    *,
    contract: FormalExperimentContract,
    critic_variant: str = "legacy",
    num_quantiles: int | None = None,
    quantile_loss_coef: float | None = None,
):
    common = dict(action_size=action_size, hidden_dim=64)
    if selector == "hgt_ppo":
        return HGTPPOAgent(
            **common,
            encoder_type=(
                "legacy_hgt"
                if contract.encoder == "legacy_hgt_relu"
                else contract.encoder
            ),
            num_layers=2,
            num_heads=4,
            job_input_dim=contract.node_dimensions[0],
            machine_input_dim=contract.node_dimensions[1],
            checkpoint_input_dim=contract.node_dimensions[2],
            critic_variant=critic_variant,
            num_quantiles=num_quantiles,
            quantile_loss_coef=quantile_loss_coef,
        )
    if selector == "mlp_ppo":
        return MLPPPOAgent(**common, job_input_dim=contract.node_dimensions[0])
    if selector == "dqn":
        return HGTDQNAgent(
            **common,
            num_layers=2,
            num_heads=4,
            job_input_dim=contract.node_dimensions[0],
            machine_input_dim=contract.node_dimensions[1],
            checkpoint_input_dim=contract.node_dimensions[2],
        )
    raise ValueError(f"selector {selector} has no trainable agent")


def evaluate_episode(
    library,
    instance,
    *,
    selector: str,
    agent=None,
    fixed_action: int | None = None,
    rng: random.Random,
    contract: FormalExperimentContract,
    mask_checkpoint=False,
    mask_forecast=False,
    action_equivalence_merge: bool | None = None,
):
    rollout = deepcopy(instance)
    env = CAPPORuleSelectionEnv(rollout)
    env.reset()
    usage = Counter()
    probabilities, actions = [], []
    decision_seconds = 0.0
    merge_actions = (
        contract.action_equivalence_merge
        if action_equivalence_merge is None
        else bool(action_equivalence_merge)
    )
    while env.current_metrics and env.current_metrics.is_active:
        if selector == "random":
            action = rng.randrange(library.action_size())
            probs = [1.0 / library.action_size()] * library.action_size()
        elif selector == "fixed":
            action = int(fixed_action)
            probs = [float(index == action) for index in range(library.action_size())]
        else:
            graph, factor_rows = graph_and_features_for(
                env,
                contract=contract,
                mask_checkpoint=mask_checkpoint,
                mask_forecast=mask_forecast,
            )
            started = time.perf_counter()
            with torch.no_grad():
                if selector == "dqn":
                    values = agent(graph)
                    probs_tensor = torch.softmax(values, dim=-1)
                    action = int(values.argmax().item())
                else:
                    mask = torch.ones(library.action_size(), dtype=torch.bool)
                    if hasattr(agent, "actor_log_probs"):
                        log_probs = agent.actor_log_probs(graph, mask)
                    else:
                        log_probs, _value = agent(graph, mask)
                    probs_tensor = log_probs.exp()
                    if merge_actions:
                        groups, candidate_indices, _dispatches = (
                            concrete_rule_action_groups(
                                env,
                                library,
                                factor_rows,
                                mask,
                                merge_equivalent=True,
                            )
                        )
                        group_log_probs = aggregate_rule_log_probs(
                            log_probs, groups, mask
                        )
                        group_index = int(group_log_probs.argmax().item())
                        action = groups[group_index][0]
                        grouped_candidate_index = candidate_indices[action]
                    else:
                        action = int(log_probs.argmax().item())
            decision_seconds += time.perf_counter() - started
            probs = probs_tensor.cpu().tolist()
        probabilities.append(probs)
        actions.append(action)
        rule = library.get_rule(action)
        usage[rule.name] += 1
        if (
            selector not in {"random", "fixed", "dqn"}
            and merge_actions
        ):
            result = env.step_candidate(grouped_candidate_index)
        else:
            result = env.step_principle(action, rule)
        if result.terminated:
            break
    objective = compute_objective(rollout, env._state)
    return {
        "F": float(objective.Z),
        "TT": float(objective.total_tardiness),
        "WSF": float(objective.weighted_checkpoint_shortfall),
        "usage": dict(usage),
        "probabilities": probabilities,
        "actions": actions,
        "decision_seconds": decision_seconds,
    }


def best_fixed_action(library, instances, *, contract: FormalExperimentContract) -> int:
    means = []
    for index in range(library.action_size()):
        values = [
            evaluate_episode(
                library,
                instance,
                selector="fixed",
                fixed_action=index,
                rng=random.Random(0),
                contract=contract,
            )["F"]
            for instance in instances
        ]
        means.append(sum(values) / len(values))
    return min(range(len(means)), key=means.__getitem__)


def evaluate_set(library, instances, seeds, *, contract, **policy):
    rows, usage = [], Counter()
    all_probabilities, switch_count, decisions = [], 0, 0
    decision_seconds = 0.0
    for seed, instance in zip(seeds, instances, strict=True):
        result = evaluate_episode(
            library,
            instance,
            rng=random.Random(seed + 91_000),
            contract=contract,
            **policy,
        )
        rows.append({"seed": seed, "F": result["F"], "TT": result["TT"], "WSF": result["WSF"]})
        usage.update(result["usage"])
        all_probabilities.extend(result["probabilities"])
        decision_seconds += result["decision_seconds"]
        actions = result["actions"]
        if actions:
            modal = Counter(actions).most_common(1)[0][0]
            switch_count += sum(action != modal for action in actions)
            decisions += len(actions)
    if all_probabilities:
        tensor = torch.tensor(all_probabilities, dtype=torch.float64)
        tv = float((0.5 * (tensor - tensor.mean(dim=0)).abs().sum(dim=1)).mean())
    else:
        tv = 0.0
    return rows, {
        "rule_usage": dict(usage),
        "total_variation": tv,
        "switch_rate": switch_count / decisions if decisions else 0.0,
        "decision_seconds_mean": decision_seconds / decisions if decisions else 0.0,
    }


def train_ppo(
    args,
    library,
    instance_factory,
    validation_instances,
    *,
    contract,
    global_f_ref: float | None = None,
):
    critic_variant = (
        getattr(args, "critic_variant", "c4")
        if contract.name == "paper_20261030"
        else "legacy"
    )
    agent = make_agent(
        args.selector,
        library.action_size(),
        contract=contract,
        critic_variant=critic_variant,
        num_quantiles=getattr(args, "num_quantiles", None),
        quantile_loss_coef=getattr(args, "quantile_loss_coef", None),
    )
    trainer = PPOTrainer(
        agent=agent,
        library=library,
        instances=[],
        instance_factory=instance_factory,
        base_seed=(20000 if contract.name == "paper_20261030" else 1000),
        lr=3e-4,
        gamma=contract.gamma,
        gae_lambda=0.95,
        epochs_per_update=4,
        reward_scale=10.0,
        entropy_coef=0.02,
        feature_set=contract.feature_set,
        graph_config=contract.graph_config,
        mask_checkpoint=args.mask_checkpoint,
        mask_forecast=args.mask_forecast,
        f_ref=global_f_ref if global_f_ref is not None else 1.0,
        reward_reference_mode=(
            "global_calibration"
            if global_f_ref is not None
            else "legacy_per_instance"
        ),
        action_equivalence_merge=(
            contract.action_equivalence_merge
            and not args.disable_action_equivalence_merge
        ),
    )
    updates = math.ceil(args.episodes / args.episodes_per_update)
    best_state, best_value = None, float("inf")
    stats_history = []
    started = time.perf_counter()
    for update in range(updates):
        trainer.entropy_coef = linear_schedule(0.02, 0.002, update, updates)
        lr = linear_schedule(3e-4, 1e-4, update, updates)
        for group in trainer.optimizer.param_groups:
            group["lr"] = lr
        count = min(args.episodes_per_update, args.episodes - update * args.episodes_per_update)
        stats_history.extend(trainer.train(1, count))
        if contract.evaluation_checkpoint == "best_validation" and (
            (update + 1) % max(1, updates // 10) == 0 or update + 1 == updates
        ):
            values, _metrics = evaluate_set(
                library,
                validation_instances,
                VALIDATION_SEEDS[: len(validation_instances)],
                contract=contract,
                selector=args.selector,
                agent=agent,
                mask_checkpoint=args.mask_checkpoint,
                mask_forecast=args.mask_forecast,
                action_equivalence_merge=False,
            )
            mean = statistics.fmean(item["F"] for item in values)
            if mean < best_value:
                best_value = mean
                best_state = deepcopy(agent.state_dict())
    if contract.evaluation_checkpoint == "best_validation" and best_state is not None:
        agent.load_state_dict(best_state)
    elapsed = time.perf_counter() - started
    agent.training_diagnostics = {
        "parameter_count": sum(parameter.numel() for parameter in agent.parameters()),
        "environment_interactions": trainer.environment_interactions,
        "optimizer_updates": trainer.optimizer_update_count,
        "wall_seconds": elapsed,
        "mean_critic_loss": (
            statistics.fmean(item.value_loss for item in stats_history)
            if stats_history
            else 0.0
        ),
        "mean_gradient_norm": (
            statistics.fmean(item.gradient_norm for item in stats_history)
            if stats_history
            else 0.0
        ),
        "mean_critic_gradient_norm": (
            statistics.fmean(item.critic_gradient_norm for item in stats_history)
            if stats_history
            else 0.0
        ),
        "mean_critic_total_cost_mae": (
            statistics.fmean(item.critic_total_cost_mae for item in stats_history)
            if stats_history
            else 0.0
        ),
        "mean_critic_tardiness_mae": (
            statistics.fmean(item.critic_tardiness_mae for item in stats_history)
            if stats_history
            else 0.0
        ),
        "mean_critic_shortfall_mae": (
            statistics.fmean(item.critic_shortfall_mae for item in stats_history)
            if stats_history
            else 0.0
        ),
        "num_quantiles": getattr(args, "num_quantiles", None),
        "quantile_loss_coef": getattr(args, "quantile_loss_coef", None),
        "d2_width_dispersion": trainer.quantile_width_diagnostics(),
    }
    return agent, elapsed


def train_dqn(
    args, library, instance_factory, *, contract, global_f_ref: float | None = None
):
    agent = make_agent("dqn", library.action_size(), contract=contract)
    trainer = DQNTrainer(
        agent,
        library,
        instance_factory,
        base_seed=(20000 if contract.name == "paper_20261030" else 1000),
        feature_set=contract.feature_set,
        lr=3e-4,
        gamma=contract.gamma,
        reward_scale=10.0,
        mask_checkpoint=args.mask_checkpoint,
        mask_forecast=args.mask_forecast,
        fixed_f_ref=global_f_ref,
    )
    started = time.perf_counter()
    trainer.train(args.episodes)
    return agent, time.perf_counter() - started


def run(args):
    contract = resolve_formal_contract(args.contract_config)
    require_runtime_ready(contract)
    global_reward_scale = None
    if contract.reward_reference == "global_calibration_spt_atc_mean":
        if not args.f_ref_manifest:
            raise ValueError(
                "paper_20261030 requires --f-ref-manifest from the independent "
                "calibration split"
            )
        global_reward_scale = load_global_reward_scale(args.f_ref_manifest)
        if global_reward_scale.generator_id != "scenario_generator_v2":
            raise ValueError("formal F_ref must use the corrected scenario_generator_v2")
        if global_reward_scale.scenario_config_hash != scenario_config_hash(load_scenario_config()):
            raise ValueError("formal F_ref scenario configuration hash mismatch")
        if global_reward_scale.calibration_partition != "calibration_center":
            raise ValueError("global F_ref must use the calibration_center split")
        if global_reward_scale.scenario != "center":
            raise ValueError("global F_ref must be calibrated on the center scenario")
        if global_reward_scale.calibration_seeds != seed_values(
            "calibration_center"
        ):
            raise ValueError("global F_ref calibration seeds do not match the frozen split")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "summary.json"
    if summary_path.exists() and not args.force:
        print(f"SKIP completed: {summary_path}")
        return
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    library = load_library(args.library_file, contract=contract)
    train_scale = args.scale
    eval_scale = args.eval_scale or args.scale
    if contract.name == "paper_20261030":
        instance_factory = lambda seed: make_scenario_instance("center", seed)
        validation_seeds = seed_values("validation_center")
    else:
        instance_factory = lambda seed: make_scaled_instance(
            seed, scale=train_scale, difficulty=args.difficulty
        )
        validation_seeds = VALIDATION_SEEDS
    validation_count = 2 if args.smoke else len(validation_seeds)
    validation_instances = [instance_factory(seed) for seed in validation_seeds[:validation_count]]
    fixed_action = None
    agent = None
    training_seconds = 0.0
    if args.checkpoint:
        critic_variant = (
            getattr(args, "critic_variant", "c4")
            if contract.name == "paper_20261030"
            else "legacy"
        )
        agent = make_agent(
            args.selector,
            library.action_size(),
            contract=contract,
            critic_variant=critic_variant,
            num_quantiles=getattr(args, "num_quantiles", None),
            quantile_loss_coef=getattr(args, "quantile_loss_coef", None),
        )
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        expected_hash = file_sha256(args.library_file)
        if state.get("selector") != args.selector:
            raise ValueError("checkpoint selector mismatch")
        if state.get("library_sha256") != expected_hash:
            raise ValueError("checkpoint rule-library hash mismatch")
        expected_f_ref_hash = file_sha256(args.f_ref_manifest)
        if state.get("f_ref_manifest_sha256") != expected_f_ref_hash:
            raise ValueError("checkpoint reward-scale manifest hash mismatch")
        expected_f_ref = (
            global_reward_scale.f_ref
            if global_reward_scale is not None
            else None
        )
        if state.get("f_ref") != expected_f_ref:
            raise ValueError("checkpoint global F_ref mismatch")
        expected_merge = (
            contract.action_equivalence_merge
            and not args.disable_action_equivalence_merge
        )
        if state.get("action_equivalence_merge") != expected_merge:
            raise ValueError("checkpoint action-equivalence setting mismatch")
        if state.get("checkpoint_semantics") != contract.evaluation_checkpoint:
            raise ValueError("checkpoint selection semantics mismatch")
        if state.get("critic_variant", "legacy") != critic_variant:
            raise ValueError("checkpoint critic variant mismatch")
        if state.get("num_quantiles") != getattr(args, "num_quantiles", None):
            raise ValueError("checkpoint num_quantiles mismatch")
        if state.get("quantile_loss_coef") != getattr(
            args, "quantile_loss_coef", None
        ):
            raise ValueError("checkpoint quantile_loss_coef mismatch")
        agent.load_state_dict(state["agent_state_dict"])
        agent.training_diagnostics = state.get("training_diagnostics")
    elif args.selector in {"hgt_ppo", "mlp_ppo"}:
        agent, training_seconds = train_ppo(
            args,
            library,
            instance_factory,
            validation_instances,
            contract=contract,
            global_f_ref=(
                global_reward_scale.f_ref
                if global_reward_scale is not None
                else None
            ),
        )
    elif args.selector == "dqn":
        agent, training_seconds = train_dqn(
            args,
            library,
            instance_factory,
            contract=contract,
            global_f_ref=(
                global_reward_scale.f_ref
                if global_reward_scale is not None
                else None
            ),
        )
    elif args.selector == "fixed":
        fixed_action = best_fixed_action(
            library, validation_instances, contract=contract
        )

    if agent is not None and not args.checkpoint:
        torch.save(
            {
                "agent_state_dict": agent.state_dict(),
                "selector": args.selector,
                "library_sha256": file_sha256(args.library_file),
                "scale": args.scale,
                "difficulty": args.difficulty,
                "seed": args.seed,
                "f_ref": (
                    global_reward_scale.f_ref
                    if global_reward_scale is not None
                    else None
                ),
                "f_ref_manifest_sha256": file_sha256(args.f_ref_manifest),
                "action_equivalence_merge": (
                    contract.action_equivalence_merge
                    and not args.disable_action_equivalence_merge
                ),
                "checkpoint_semantics": contract.evaluation_checkpoint,
                "critic_variant": getattr(agent, "critic_variant", "legacy"),
                "num_quantiles": getattr(args, "num_quantiles", None),
                "quantile_loss_coef": getattr(args, "quantile_loss_coef", None),
                "training_diagnostics": getattr(
                    agent, "training_diagnostics", None
                ),
            },
            output
            / (
                "final.pt"
                if contract.evaluation_checkpoint == "final_training"
                else "best.pt"
            ),
        )

    if contract.name == "paper_20261030":
        scenario = "center"
        test_seeds = seed_values("test_center")
        if args.smoke:
            test_seeds = test_seeds[:3]
        test_instances = [make_scenario_instance(scenario, seed) for seed in test_seeds]
    else:
        test_seeds = TEST_SEEDS_BY_SCALE[eval_scale]
        if args.smoke:
            test_seeds = test_seeds[:3]
        test_instances = [
            make_scaled_instance(seed, scale=eval_scale, difficulty=args.difficulty)
            for seed in test_seeds
        ]
    rows, policy_metrics = evaluate_set(
        library,
        test_instances,
        test_seeds,
        contract=contract,
        selector=args.selector,
        agent=agent,
        fixed_action=fixed_action,
        mask_checkpoint=args.mask_checkpoint,
        mask_forecast=args.mask_forecast,
        action_equivalence_merge=(
            contract.action_equivalence_merge
            and not args.disable_action_equivalence_merge
        ),
    )
    csv_path = output / "per_instance.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("seed", "F", "TT", "WSF"))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "library": args.library_id,
        "contract_config": contract.name,
        "library_file": args.library_file,
        "scale": args.scale,
        "eval_scale": eval_scale,
        "difficulty": args.difficulty,
        "seed": args.seed,
        "selector": args.selector,
        "mask_checkpoint": args.mask_checkpoint,
        "mask_forecast": args.mask_forecast,
        "action_equivalence_merge": (
            contract.action_equivalence_merge
            and not args.disable_action_equivalence_merge
        ),
        "critic_variant": (
            getattr(agent, "critic_variant", None) if agent is not None else None
        ),
        "num_quantiles": getattr(args, "num_quantiles", None),
        "quantile_loss_coef": getattr(args, "quantile_loss_coef", None),
        "training_diagnostics": (
            getattr(agent, "training_diagnostics", None)
            if agent is not None
            else None
        ),
        "episodes": args.episodes,
        "episodes_per_update": args.episodes_per_update,
        "f_ref": (
            global_reward_scale.f_ref
            if global_reward_scale is not None
            else None
        ),
        "f_ref_manifest": args.f_ref_manifest,
        "f_ref_manifest_sha256": file_sha256(args.f_ref_manifest),
        "smoke": args.smoke,
        "means": {
            key: statistics.fmean(row[key] for row in rows)
            for key in ("F", "TT", "WSF")
        },
        "std_F": statistics.pstdev(row["F"] for row in rows),
        "fixed_action": fixed_action,
        "fixed_rule": library.get_rule(fixed_action).name if fixed_action is not None else None,
        "training_seconds": training_seconds,
        **policy_metrics,
        "per_instance_csv": str(csv_path.resolve()),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_manifest(
        output / "manifest.json",
        {
            "kind": "formal_experiment",
            "contract_config": contract.name,
            "library_sha256": file_sha256(args.library_file),
            "f_ref_manifest_sha256": file_sha256(args.f_ref_manifest),
            "f_ref": (
                global_reward_scale.f_ref
                if global_reward_scale is not None
                else None
            ),
            "critic_variant": (
                getattr(agent, "critic_variant", None)
                if agent is not None
                else None
            ),
            "num_quantiles": getattr(args, "num_quantiles", None),
            "quantile_loss_coef": getattr(args, "quantile_loss_coef", None),
            "training_diagnostics": (
                getattr(agent, "training_diagnostics", None)
                if agent is not None
                else None
            ),
            "test_seeds": list(test_seeds),
            "config": {key: value for key, value in vars(args).items() if key != "force"},
        },
    )
    print(json.dumps(summary, ensure_ascii=False))


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract-config",
        choices=tuple(FORMAL_EXPERIMENT_CONTRACTS),
        required=True,
        help="explicit implementation contract; no implicit legacy fallback",
    )
    parser.add_argument(
        "--critic-variant",
        choices=("c1", "c2", "c3", "c4", "d1", "d2"),
        default="c4",
        help="paper_20261030 Critic variant; legacy ignores this option",
    )
    parser.add_argument(
        "--num-quantiles",
        type=int,
        help="D1/D2 quantile count K; select on validation only",
    )
    parser.add_argument(
        "--quantile-loss-coef",
        type=float,
        help="D1/D2 alpha_Q; select on validation only",
    )
    parser.add_argument("--library-id", required=True)
    parser.add_argument("--library-file", required=True)
    parser.add_argument("--scale", choices=tuple(TEST_SEEDS_BY_SCALE), required=True)
    parser.add_argument("--eval-scale", choices=tuple(TEST_SEEDS_BY_SCALE))
    parser.add_argument("--difficulty", choices=("service_heavy", "balanced"), default="service_heavy")
    parser.add_argument("--seed", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--selector", choices=("hgt_ppo", "mlp_ppo", "dqn", "random", "fixed"), default="hgt_ppo")
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--episodes-per-update", type=int, default=32)
    parser.add_argument("--mask-checkpoint", action="store_true")
    parser.add_argument("--mask-forecast", action="store_true")
    parser.add_argument(
        "--disable-action-equivalence-merge",
        action="store_true",
        help="ablation: use raw rule-index PPO probabilities",
    )
    parser.add_argument("--checkpoint")
    parser.add_argument(
        "--f-ref-manifest",
        help="fixed global reward-scale manifest required by paper_20261030",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if args.smoke:
        args.episodes = 3
        args.episodes_per_update = 3
    if args.checkpoint and args.selector in {"random", "fixed"}:
        parser.error("--checkpoint requires a trainable selector")
    if args.critic_variant in {"d1", "d2"}:
        if args.selector != "hgt_ppo":
            parser.error("D1/D2 require the public CRG hgt_ppo selector")
        if args.num_quantiles is None or args.quantile_loss_coef is None:
            parser.error(
                "D1/D2 require explicit --num-quantiles and "
                "--quantile-loss-coef chosen on validation"
            )
        if args.num_quantiles <= 0 or args.quantile_loss_coef < 0.0:
            parser.error("D1/D2 quantile settings must be positive/non-negative")
        if not math.isfinite(args.quantile_loss_coef):
            parser.error("--quantile-loss-coef must be finite")
    elif args.num_quantiles is not None or args.quantile_loss_coef is not None:
        parser.error("quantile settings are valid only for D1/D2")
    return args


if __name__ == "__main__":
    run(parse_args())
