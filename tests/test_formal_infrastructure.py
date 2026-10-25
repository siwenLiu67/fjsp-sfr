import random
from copy import deepcopy

import torch

from fjsp_sfr.dispatch.classic_rules import build_classic_rule_library
from fjsp_sfr.dispatch.rule_library import RuleLibrary
from fjsp_sfr.dispatch.theory_features import build_extended_theory_features
from fjsp_sfr.env.ca_observation import resolve_observation_scales
from fjsp_sfr.env.ca_rule_selection import CAPPORuleSelectionEnv
from fjsp_sfr.experiments.eval_classic import evaluate_rule
from fjsp_sfr.experiments.generate_rule_library import (
    HOLDOUT_SEEDS,
    SELECTION_SEEDS,
    TEST_SEEDS,
    TRAIN_BASE_SEED,
    VALIDATION_SEEDS,
)
from fjsp_sfr.experiments.hparam_sweep import ALGORITHMS, LEARNING_RATES, NETWORK_SEEDS
from fjsp_sfr.generation.scale_config import make_scaled_instance
from fjsp_sfr.rl.graph import build_hetero_graph
from fjsp_sfr.rl.ppo_agent import HGTPPOAgent
from fjsp_sfr.experiments.formal_experiment import (
    HOLDOUT_SEEDS as FORMAL_HOLDOUT_SEEDS,
    SELECTION_SEEDS as FORMAL_SELECTION_SEEDS,
    TEST_SEEDS_BY_SCALE,
    TRAIN_SEEDS as FORMAL_TRAIN_SEEDS,
    VALIDATION_SEEDS as FORMAL_VALIDATION_SEEDS,
)
from fjsp_sfr.experiments.formal_matrix import build_matrix


def test_seed_partitions_are_disjoint_for_every_scale():
    partitions = (
        set(SELECTION_SEEDS),
        set(HOLDOUT_SEEDS),
        set(VALIDATION_SEEDS),
        set(TEST_SEEDS),
        set(range(TRAIN_BASE_SEED, TRAIN_BASE_SEED + 5000)),
    )
    for _scale in ("S", "M", "L", "X"):
        for index, left in enumerate(partitions):
            for right in partitions[index + 1 :]:
                assert left.isdisjoint(right)


def test_formal_test_ranges_are_pairwise_disjoint_from_all_partitions():
    partitions = [
        set(FORMAL_TRAIN_SEEDS),
        set(FORMAL_VALIDATION_SEEDS),
        set(FORMAL_SELECTION_SEEDS),
        set(FORMAL_HOLDOUT_SEEDS),
        *(set(TEST_SEEDS_BY_SCALE[scale]) for scale in ("S", "M", "L", "X")),
    ]
    for index, left in enumerate(partitions):
        for right in partitions[index + 1 :]:
            assert left.isdisjoint(right)


def test_formal_matrix_contains_all_123_runs():
    runs = build_matrix()
    assert len(runs) == 123
    assert sum(run.selector == "dqn" for run in runs) == 9
    assert sum(run.mask_checkpoint for run in runs) == 3
    assert sum(run.mask_forecast for run in runs) == 3
    assert sum(run.eval_scale == "X" for run in runs) == 3


def test_sweep_exposes_four_algorithms_three_rates_and_three_seeds():
    assert len(ALGORITHMS) == 4
    assert LEARNING_RATES == (1e-4, 3e-4, 1e-3)
    assert NETWORK_SEEDS == (1, 2, 3)


def test_classic_library_json_loads_and_uses_raw_candidates():
    library = RuleLibrary.from_json("experiments/classic4_library.json")
    assert [rule.name for rule in library.rules] == [
        "SPT", "EDD", "ATC_K2", "MIN_SLACK"
    ]
    instance = make_scaled_instance(100, scale="S", difficulty="balanced")
    assert evaluate_rule(library.rules[0], instance).Z >= 0.0


def test_each_scale_short_rollout_and_random_ppo_forward():
    llm_library = RuleLibrary.from_json("experiments/llm_library_qwen1_6.json")
    classic = build_classic_rule_library()
    for offset, scale in enumerate(("S", "M", "L", "X")):
        instance = make_scaled_instance(800 + offset, scale=scale)
        for rule in classic.rules:
            assert evaluate_rule(rule, instance).Z >= 0.0
        assert evaluate_rule(llm_library.rules[0], instance).Z >= 0.0

        rollout = deepcopy(instance)
        env = CAPPORuleSelectionEnv(rollout)
        env.reset()
        problem = env._problem_view()
        time_scale, processing_scale = resolve_observation_scales(
            problem, env._state
        )
        features = build_extended_theory_features(
            problem,
            env._state,
            env.current_metrics,
            time_scale=time_scale,
        )
        graph = build_hetero_graph(
            problem,
            env._state,
            env.current_metrics,
            features,
            time_scale=time_scale,
            processing_scale=processing_scale,
        )
        agent = HGTPPOAgent(
            action_size=llm_library.action_size(),
            hidden_dim=64,
            num_layers=2,
            num_heads=4,
        )
        with torch.no_grad():
            log_probs, value = agent(
                graph, torch.ones(llm_library.action_size(), dtype=torch.bool)
            )
        assert log_probs.shape == (llm_library.action_size(),)
        assert value.ndim == 0
        action = random.Random(900 + offset).randrange(llm_library.action_size())
        result = env.step_principle(action, llm_library.get_rule(action))
        assert result.decision.machine_id is not None
