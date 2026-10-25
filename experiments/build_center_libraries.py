"""Build the center-scenario rule libraries without invoking an LLM."""

from __future__ import annotations

import argparse
from pathlib import Path

from fjsp_sfr.dispatch.rule_library import RuleLibrary
from fjsp_sfr.experiments.build_gated_library import build_gated_library, parse_args
from fjsp_sfr.experiments.generate_random_rules import write_candidates
from fjsp_sfr.experiments.scenario_tools import evaluate_rule
from fjsp_sfr.experiments.validate_framework import build_rule_library
from fjsp_sfr.generation.scenario_generator import make_scenario_instance, seed_values


POOL_FILES = tuple(Path(f"experiments/llm_candidates_p3b{index}.json") for index in (1, 2, 3))


def _gated(argv: list[str]) -> None:
    build_gated_library(parse_args(argv))


def build_handwritten(output: Path, k: int = 6) -> None:
    instances = [make_scenario_instance("center", seed) for seed in seed_values("selection_center")]
    rules = list(build_rule_library().rules)
    rules.sort(key=lambda rule: sum(evaluate_rule(rule, instance)["F"] for instance in instances) / len(instances))
    RuleLibrary(tuple(rules[:k])).to_json(str(output))


def build_all(random_seed: int = 20261025) -> None:
    common = ["--scenario", "center", "--feature_set", "theory11_plus3", "--gate_baseline", "best_classic", "--K", "6"]
    candidate_args = [str(path) for path in POOL_FILES]
    _gated(["--candidates", *candidate_args, "--tag", "pool_center_g", "--out", "experiments/llm_library_pool_center_g.json", *common])
    _gated(["--candidates", *candidate_args, "--tag", "pool_center_mix", "--out", "experiments/llm_library_pool_center_mix.json", "--extra_rules", "experiments/classic4_library.json", *common])

    random_path = Path("experiments/random_candidates_center.json")
    write_candidates(random_path, count=120, seed=random_seed)
    _gated(["--candidates", str(random_path), "--tag", "random_center_nogate", "--out", "experiments/llm_library_random_center_nogate.json", "--disable_quality_gate", *common])
    _gated(["--candidates", str(random_path), "--tag", "random_center_nogate_mix", "--out", "experiments/llm_library_random_center_nogate_mix.json", "--disable_quality_gate", "--extra_rules", "experiments/classic4_library.json", *common])
    build_handwritten(Path("experiments/handwritten_center.json"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--random-seed", type=int, default=20261025)
    args = parser.parse_args()
    build_all(args.random_seed)


if __name__ == "__main__":
    main()
