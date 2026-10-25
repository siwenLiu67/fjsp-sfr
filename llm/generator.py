"""Algorithm 1: 迭代 LLM 规则生成。"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from statistics import fmean
from typing import Protocol

from fjsp_sfr.core.models import FJSPInstance
from fjsp_sfr.dispatch.rule_library import RuleLibrary
from fjsp_sfr.dispatch.symbolic_rule import RuleType, SymbolicRule
from fjsp_sfr.dispatch.theory_features import build_extended_theory_features
from fjsp_sfr.env.ca_rule_selection import CAPPORuleSelectionEnv

from .prompt_template import build_generation_prompt
from .rule_parser import parse_llm_output


class LLMClient(Protocol):
    """LLM 调用接口。"""

    def generate(self, prompt: str) -> str:
        """发送 prompt 并返回文本响应。"""

        ...


@dataclass
class GenerationResult:
    """一轮规则生成与评估结果。"""

    iteration: int
    generated_rules: list[SymbolicRule]
    evaluated_rules: list[tuple[SymbolicRule, float]]
    accepted_rules: list[SymbolicRule]


@lru_cache(maxsize=2)
def _default_rule(feature_set: str = "theory11") -> SymbolicRule:
    return SymbolicRule(
        name="default_due_date_slack",
        expression="due-date slack ascending",
        rule_type=RuleType.TYPE_I,
        priority_expr="-due_date_slack",
        feature_set=feature_set,
    )


def _summary_row(
    factor_rows: tuple[tuple[float, ...], ...],
) -> tuple[float, ...]:
    if not factor_rows:
        return ()
    return tuple(max(column) for column in zip(*factor_rows, strict=True))


def _select_candidate(
    rule: SymbolicRule,
    factor_rows: tuple[tuple[float, ...], ...],
) -> int:
    if not factor_rows:
        raise ValueError("cannot select from an empty candidate set")
    selected_rule = (
        rule
        if rule.is_applicable(_summary_row(factor_rows))
        else _default_rule(rule.feature_set)
    )
    scores = selected_rule.evaluate_batch(factor_rows)
    best_index = 0
    best_score = float("-inf")
    for index, score in enumerate(scores):
        if score > best_score:
            best_score = score
            best_index = index
    return best_index


def _rollout_objective(rule: SymbolicRule, instance: FJSPInstance) -> float:
    env = CAPPORuleSelectionEnv(deepcopy(instance))
    env.reset()
    while True:
        factor_rows = build_extended_theory_features(
            env._problem_view(),
            env._state,
            env.current_metrics,
            feature_set=rule.feature_set,
        )
        step = env.step_candidate(_select_candidate(rule, factor_rows))
        if step.terminated:
            if step.objective is None:
                raise RuntimeError("terminated rollout did not produce an objective")
            return float(step.objective.Z)


def evaluate_rule_objective(
    rule: SymbolicRule,
    instances: list[FJSPInstance],
    *,
    num_episodes: int = 5,
    baseline_objective: float | None = None,
) -> float:
    """计算规则在多实例上的平均（可选标准化）目标值。"""

    if not instances:
        raise ValueError("instances must not be empty")
    if num_episodes <= 0:
        raise ValueError("num_episodes must be positive")
    objectives = [
        _rollout_objective(rule, instance)
        for _episode in range(num_episodes)
        for instance in instances
    ]
    mean_objective = float(fmean(objectives))
    if baseline_objective is None:
        return mean_objective
    if baseline_objective <= 0.0:
        if mean_objective == 0.0:
            return 0.0
        raise ValueError("baseline_objective must be positive")
    return mean_objective / baseline_objective


def compute_behavior_similarity(
    rule_a: SymbolicRule,
    rule_b: SymbolicRule,
    instances: list[FJSPInstance],
    *,
    max_decisions: int = 200,
) -> float:
    """在 rule_a 的轨迹上计算两条规则的 candidate 选择一致率。"""

    if max_decisions <= 0:
        raise ValueError("max_decisions must be positive")
    matches = 0
    decisions = 0
    for instance in instances:
        env = CAPPORuleSelectionEnv(deepcopy(instance))
        env.reset()
        for _decision in range(max_decisions):
            factor_rows = build_extended_theory_features(
                env._problem_view(),
                env._state,
                env.current_metrics,
                feature_set=rule_a.feature_set,
            )
            choice_a = _select_candidate(rule_a, factor_rows)
            choice_b = _select_candidate(rule_b, factor_rows)
            matches += int(choice_a == choice_b)
            decisions += 1
            step = env.step_candidate(choice_a)
            if step.terminated:
                break
    return matches / decisions if decisions else 1.0


def greedy_library_selection(
    candidates: list[SymbolicRule],
    budget: int,
    instances: list[FJSPInstance],
    *,
    similarity_threshold: float = 0.8,
) -> RuleLibrary:
    """按目标质量与行为多样性的边际增益贪心选规则。"""

    if budget <= 0:
        return RuleLibrary(rules=())
    selected = [_default_rule()]
    remaining = [rule for rule in candidates if rule != selected[0]]
    objective_cache: dict[SymbolicRule, float] = {}

    while remaining and len(selected) < budget:
        best_rule: SymbolicRule | None = None
        best_gain = float("-inf")
        for candidate in remaining:
            similarities = [
                compute_behavior_similarity(candidate, existing, instances)
                for existing in selected
            ]
            max_similarity = max(similarities, default=0.0)
            if max_similarity >= similarity_threshold:
                continue
            objective = objective_cache.setdefault(
                candidate,
                evaluate_rule_objective(candidate, instances, num_episodes=1),
            )
            gain = (1.0 / max(objective, 1e-12)) * (1.0 - max_similarity)
            if gain > best_gain:
                best_gain = gain
                best_rule = candidate
        if best_rule is None:
            break
        selected.append(best_rule)
        remaining.remove(best_rule)

    return RuleLibrary(rules=tuple(selected))


def iterative_rule_generation(
    *,
    llm_client: LLMClient,
    instances: list[FJSPInstance],
    max_iterations: int = 5,
    rules_per_iteration: int = 10,
    library_budget: int = 20,
    similarity_threshold: float = 0.8,
    feature_set: str = "theory11",
) -> tuple[RuleLibrary, list[GenerationResult]]:
    """Algorithm 1 主循环。"""

    baseline = evaluate_rule_objective(
        _default_rule(feature_set), instances, num_episodes=1
    )
    accepted_pool: list[SymbolicRule] = []
    results: list[GenerationResult] = []
    feedback: str | None = None

    for iteration in range(max_iterations):
        prompt = build_generation_prompt(
            existing_rules=accepted_pool,
            iteration=iteration,
            num_rules=rules_per_iteration,
            feedback=feedback,
            feature_set=feature_set,
        )
        from fjsp_sfr.dispatch.theory_features import feature_names_for

        generated = parse_llm_output(
            llm_client.generate(prompt), feature_names_for(feature_set)
        )
        evaluated = [
            (
                rule,
                evaluate_rule_objective(rule, instances, num_episodes=1),
            )
            for rule in generated
        ]
        accepted = [rule for rule, objective in evaluated if objective < baseline]
        known_definitions = {
            (rule.name, rule.priority_expr, rule.cond_expr, rule.filter_expr)
            for rule in accepted_pool
        }
        for rule in accepted:
            signature = (
                rule.name,
                rule.priority_expr,
                rule.cond_expr,
                rule.filter_expr,
            )
            if signature not in known_definitions:
                accepted_pool.append(rule)
                known_definitions.add(signature)
        results.append(
            GenerationResult(
                iteration=iteration,
                generated_rules=generated,
                evaluated_rules=evaluated,
                accepted_rules=accepted,
            )
        )
        feedback = (
            f"Parsed {len(generated)} rules; accepted {len(accepted)} with "
            f"objective below baseline {baseline:.6g}."
        )

    library = greedy_library_selection(
        accepted_pool,
        library_budget,
        instances,
        similarity_threshold=similarity_threshold,
    )
    return library, results
