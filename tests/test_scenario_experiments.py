from pathlib import Path

from fjsp_sfr.experiments import build_gated_library as gated
from fjsp_sfr.experiments.hparam_sweep import ALGORITHMS, EPISODES, LEARNING_RATES, NETWORK_SEEDS, SweepTask
from fjsp_sfr.experiments.run_algo import evaluation_plan, parse_args, resolve_library
from fjsp_sfr.generation.scenario_generator import seed_values


def test_gated_library_center_scenario_uses_new_generator(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(gated, "make_scenario_instance", lambda scenario, seed: (sentinel, scenario, seed))
    assert gated.make_instance(600, scenario="center") == (sentinel, "center", 600)


def test_hparam_sweep_matrix_is_four_algorithms_by_three_rates_and_seeds():
    tasks = [SweepTask(algo, rate, seed) for algo in ALGORITHMS for rate in LEARNING_RATES for seed in NETWORK_SEEDS]
    assert len(tasks) == 36
    assert EPISODES == 2000


def test_run_algo_validation_only_never_uses_test_partition():
    args = parse_args(["--algo", "hgt_ppo", "--validation_only"])
    plan = evaluation_plan(args)
    assert [seed for _scenario, seed in plan] == list(seed_values("validation_center"))


def test_default_center_library_alias():
    assert resolve_library("pool_center_mix") == Path("experiments/llm_library_pool_center_mix.json")
