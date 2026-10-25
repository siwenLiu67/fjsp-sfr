"""Shared evaluation helpers for center/L9 scenario experiments."""

from __future__ import annotations

import random
from copy import deepcopy
from statistics import fmean

from fjsp_sfr.core.objective import compute_objective
from fjsp_sfr.dispatch.classic_rules import build_classic_rule_library
from fjsp_sfr.dispatch.rule_library import RuleLibrary
from fjsp_sfr.env.ca_rule_selection import CAPPORuleSelectionEnv
from fjsp_sfr.experiments.build_gated_library import default_rule


def evaluate_rule(rule, instance) -> dict[str, float]:
    rollout = deepcopy(instance)
    env = CAPPORuleSelectionEnv(rollout)
    env.reset()
    while env.current_metrics and env.current_metrics.is_active:
        result = env.step_principle(0, rule)
        if result.terminated:
            break
    objective = compute_objective(rollout, env._state)
    return {
        "F": float(objective.Z),
        "TT": float(objective.total_tardiness),
        "WSF": float(objective.weighted_checkpoint_shortfall),
    }


def evaluate_random(library: RuleLibrary, instance, seed: int) -> dict[str, float]:
    rollout = deepcopy(instance)
    env, rng = CAPPORuleSelectionEnv(rollout), random.Random(seed)
    env.reset()
    while env.current_metrics and env.current_metrics.is_active:
        action = rng.randrange(library.action_size())
        result = env.step_principle(action, library.get_rule(action))
        if result.terminated:
            break
    objective = compute_objective(rollout, env._state)
    return {"F": float(objective.Z), "TT": float(objective.total_tardiness), "WSF": float(objective.weighted_checkpoint_shortfall)}


def classic_rules_with_default() -> tuple[object, ...]:
    return (*build_classic_rule_library().rules, default_rule("theory11_plus3"))


def summarize(rows: list[dict[str, float]]) -> dict[str, float]:
    return {key: fmean(row[key] for row in rows) for key in ("F", "TT", "WSF")}


def bootstrap_ci(differences: list[float], *, seed: int = 20261025, samples: int = 4000) -> tuple[float, float]:
    if not differences:
        return float("nan"), float("nan")
    rng, size = random.Random(seed), len(differences)
    means = sorted(fmean(rng.choice(differences) for _ in range(size)) for _ in range(samples))
    return means[int(0.025 * samples)], means[min(samples - 1, int(0.975 * samples))]


__all__ = ["bootstrap_ci", "classic_rules_with_default", "evaluate_random", "evaluate_rule", "summarize"]
