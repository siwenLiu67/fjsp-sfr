"""Build a quality-gated, complementary rule library from cached candidates."""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_PROJECT_PARENT = _PROJECT_ROOT.parent
for _import_path in (str(_PROJECT_ROOT), str(_PROJECT_PARENT)):
    if _import_path not in sys.path:
        sys.path.insert(0, _import_path)

import argparse
import json
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import fmean

from fjsp_sfr.core.objective import compute_objective
from fjsp_sfr.dispatch.classic_rules import ClassicRule
from fjsp_sfr.dispatch.rule_library import RuleLibrary
from fjsp_sfr.dispatch.symbolic_rule import RuleType, SymbolicRule
from fjsp_sfr.dispatch.theory_features import (
    THEORY_FEATURE_NAMES,
    build_extended_theory_features,
)
from fjsp_sfr.env.ca_rule_selection import CAPPORuleSelectionEnv
from fjsp_sfr.generation.scale_config import make_scaled_instance
from fjsp_sfr.generation.scenario_generator import make_scenario_instance


SELECTION_SEEDS = tuple(range(600, 620))
HOLDOUT_SEEDS = tuple(range(700, 720))


@dataclass(frozen=True)
class EvaluatedRule:
    rule: object
    values: tuple[float, ...]
    ratios: tuple[float, ...]
    normalized_objective: float
    source: str = "llm"


@dataclass(frozen=True)
class ReferenceState:
    factor_rows: tuple[tuple[float, ...], ...]
    snapshot: object | None = None


def make_instance(
    seed: int,
    *,
    scale: str = "M",
    difficulty: str = "balanced",
    scenario: str | None = None,
):
    if scenario is not None:
        return make_scenario_instance(scenario, seed)
    return make_scaled_instance(seed, scale=scale, difficulty=difficulty)


def default_rule(feature_set: str = "theory11") -> SymbolicRule:
    """A rule whose false condition activates the environment's EDD-slack fallback."""

    return SymbolicRule(
        name="default_due_date_slack",
        expression="default due-date-slack ordering",
        rule_type=RuleType.TYPE_II,
        priority_expr="0.0",
        cond_expr="False",
        feature_set=feature_set,
    )


def _summary_row(rows: tuple[tuple[float, ...], ...]) -> tuple[float, ...]:
    return tuple(max(column) for column in zip(*rows, strict=True))


def choose_candidate(
    rule: SymbolicRule, rows: tuple[tuple[float, ...], ...]
) -> int:
    if not rows:
        raise ValueError("candidate rows must not be empty")
    if rule.is_applicable(_summary_row(rows)):
        scores = rule.evaluate_batch(rows)
    else:
        slack_index = THEORY_FEATURE_NAMES.index("due_date_slack")
        scores = [-row[slack_index] for row in rows]
    return max(range(len(scores)), key=scores.__getitem__)


def rollout_rule(
    rule: object,
    instance,
    *,
    capture_reference_states: bool = False,
    feature_set: str = "theory11",
) -> tuple[float, tuple[ReferenceState, ...]]:
    rollout = deepcopy(instance)
    env = CAPPORuleSelectionEnv(rollout)
    env.reset()
    states: list[ReferenceState] = []
    while env.current_metrics and env.current_metrics.is_active:
        rows = build_extended_theory_features(
            env._problem_view(),
            env._state,
            env.current_metrics,
            feature_set=feature_set,
        )
        if capture_reference_states and len(rows) >= 2:
            states.append(ReferenceState(rows, env.snapshot()))
        result = env.step_principle(0, rule)
        if result.terminated:
            break
    return float(compute_objective(rollout, env._state).Z), tuple(states)


def evaluate_rules(
    rules: list[object],
    instances: list,
    baseline_values: tuple[float, ...],
    *,
    sources: dict[str, str] | None = None,
    feature_set: str = "theory11",
) -> list[EvaluatedRule]:
    evaluated: list[EvaluatedRule] = []
    for rule in rules:
        values = tuple(
            rollout_rule(rule, instance, feature_set=feature_set)[0]
            for instance in instances
        )
        ratios = tuple(
            value / max(1.0, baseline)
            for value, baseline in zip(values, baseline_values, strict=True)
        )
        evaluated.append(
            EvaluatedRule(
                rule,
                values,
                ratios,
                fmean(ratios),
                (sources or {}).get(rule.name, "llm"),
            )
        )
    return evaluated


def symmetric_similarity(
    left: object,
    right: object,
    reference_states: tuple[ReferenceState, ...],
) -> float:
    """Selection agreement on one fixed default-policy state set."""

    if not reference_states:
        return 1.0
    if isinstance(left, SymbolicRule) and isinstance(right, SymbolicRule):
        matches = sum(
            choose_candidate(left, state.factor_rows)
            == choose_candidate(right, state.factor_rows)
            for state in reference_states
        )
        return matches / len(reference_states)
    matches = 0
    for state in reference_states:
        if state.snapshot is not None:
            left_env = CAPPORuleSelectionEnv.from_snapshot(state.snapshot)
            right_env = CAPPORuleSelectionEnv.from_snapshot(state.snapshot)
            left_choice = left_env.step_principle(0, left).decision.candidate_index
            right_choice = right_env.step_principle(0, right).decision.candidate_index
        else:
            left_choice = choose_candidate(left, state.factor_rows)
            right_choice = choose_candidate(right, state.factor_rows)
        matches += int(left_choice == right_choice)
    return matches / len(reference_states)


def behavior_deduplicate(
    rules: list[EvaluatedRule],
    reference_states: tuple[ReferenceState, ...],
    threshold: float,
) -> tuple[list[EvaluatedRule], dict[tuple[str, str], float], int]:
    retained: list[EvaluatedRule] = []
    similarities: dict[tuple[str, str], float] = {}
    signatures: dict[str, tuple[int, ...]] = {}
    rejected = 0
    for candidate in sorted(
        rules, key=lambda item: (item.normalized_objective, item.rule.name)
    ):
        signature = tuple(
            choose_candidate(candidate.rule, state.factor_rows)
            for state in reference_states
        )
        signatures[candidate.rule.name] = signature
        duplicate = False
        for existing in retained:
            key = tuple(sorted((candidate.rule.name, existing.rule.name)))
            existing_signature = signatures[existing.rule.name]
            value = (
                sum(
                    left == right
                    for left, right in zip(signature, existing_signature, strict=True)
                )
                / len(reference_states)
                if reference_states
                else 1.0
            )
            similarities[key] = value
            if value >= threshold:
                duplicate = True
                break
        if duplicate:
            rejected += 1
        else:
            retained.append(candidate)
    return retained, similarities, rejected


def library_oracle_score(rules: list[EvaluatedRule]) -> float:
    if not rules:
        return 1.0
    return fmean(min(rule.ratios[index] for rule in rules) for index in range(len(rules[0].ratios)))


def greedy_complementary_selection(
    rules: list[EvaluatedRule],
    *,
    budget: int,
    gain_threshold: float,
    initial: list[EvaluatedRule] | None = None,
) -> tuple[list[EvaluatedRule], dict[str, float]]:
    if budget <= 0:
        return [], {}
    selected = list(initial or [])[:budget]
    remaining = sorted(
        [
            item
            for item in rules
            if all(item.rule.name != chosen.rule.name for chosen in selected)
        ],
        key=lambda item: (item.normalized_objective, item.rule.name),
    )
    gains = {item.rule.name: 0.0 for item in selected}
    if not selected:
        if not remaining:
            return [], {}
        selected = [remaining.pop(0)]
        gains[selected[0].rule.name] = 1.0 - selected[0].normalized_objective
    while remaining and len(selected) < budget:
        current = library_oracle_score(selected)
        scored = [
            (current - library_oracle_score([*selected, candidate]), candidate)
            for candidate in remaining
        ]
        gain, best = max(
            scored,
            key=lambda item: (
                item[0],
                -item[1].normalized_objective,
                item[1].rule.name,
            ),
        )
        if gain < gain_threshold:
            break
        selected.append(best)
        remaining.remove(best)
        gains[best.rule.name] = gain
    return selected, gains


def _rule_signature(rule: SymbolicRule) -> tuple:
    return (
        rule.feature_set,
        int(rule.rule_type),
        rule.priority_expr,
        rule.cond_expr,
        rule.filter_expr,
    )


def _load_candidate_rules(
    paths: list[Path],
) -> tuple[list[SymbolicRule], dict[str, str], int]:
    """Merge candidate files in order and deduplicate rule definitions."""

    rules: list[SymbolicRule] = []
    sources: dict[str, str] = {}
    signatures: set[tuple] = set()
    used_names: set[str] = set()
    duplicate_count = 0
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        tag = str(payload.get("tag") or path.stem.removeprefix("llm_candidates_"))
        root_source = str(payload.get("source") or f"llm:{tag}")
        for item in payload["candidates"]:
            rule = SymbolicRule.from_dict(item["rule"])
            signature = _rule_signature(rule)
            if signature in signatures:
                duplicate_count += 1
                continue
            signatures.add(signature)
            original_name = rule.name
            if rule.name in used_names:
                suffix = tag.replace("-", "_")
                candidate_name = f"{rule.name}_{suffix}"
                serial = 2
                while candidate_name in used_names:
                    candidate_name = f"{rule.name}_{suffix}_{serial}"
                    serial += 1
                rule = replace(rule, name=candidate_name)
            used_names.add(rule.name)
            rules.append(rule)
            sources[rule.name] = str(item.get("source") or root_source)
            if rule.name != original_name:
                sources[rule.name] += f" (renamed from {original_name})"
    return rules, sources, duplicate_count


def _max_selected_similarity(
    item: EvaluatedRule,
    selected: list[EvaluatedRule],
    reference_states: tuple[ReferenceState, ...],
) -> float:
    peers = [peer for peer in selected if peer.rule.name != item.rule.name]
    return max(
        (
            symmetric_similarity(item.rule, peer.rule, reference_states)
            for peer in peers
        ),
        default=0.0,
    )


def _evaluate_collection(
    rules: list[object],
    instances: list,
    *,
    gate_baseline: str = "default",
    feature_set: str = "theory11",
) -> tuple[tuple[float, ...], list[EvaluatedRule]]:
    baseline_values = _baseline_values(
        instances, gate_baseline=gate_baseline, feature_set=feature_set
    )
    return baseline_values, evaluate_rules(
        rules, instances, baseline_values, feature_set=feature_set
    )


def _baseline_values(
    instances: list,
    *,
    gate_baseline: str,
    feature_set: str,
) -> tuple[float, ...]:
    if gate_baseline == "default":
        baseline = default_rule(feature_set)
        return tuple(
            rollout_rule(baseline, instance, feature_set=feature_set)[0]
            for instance in instances
        )
    _name, values = _best_classic_baseline(
        instances, feature_set=feature_set
    )
    return values


def _classic_values(
    instances: list, *, feature_set: str
) -> dict[str, tuple[float, ...]]:
    classics = RuleLibrary.from_json("experiments/classic4_library.json")
    result: dict[str, tuple[float, ...]] = {}
    for output_name, stored_name in (("SPT", "SPT"), ("ATC", "ATC_K2")):
        rule = next(rule for rule in classics.rules if rule.name == stored_name)
        result[output_name] = tuple(
            rollout_rule(rule, instance, feature_set=feature_set)[0]
            for instance in instances
        )
    return result


def _best_classic_baseline(
    instances: list, *, feature_set: str
) -> tuple[str, tuple[float, ...]]:
    """Choose one classic by selection-set mean F, then keep it fixed."""

    values = _classic_values(instances, feature_set=feature_set)
    name = min(("SPT", "ATC"), key=lambda key: (fmean(values[key]), key))
    return name, values[name]


def _report_section(
    label: str,
    evaluated: list[EvaluatedRule],
    classic_values: dict[str, tuple[float, ...]] | None = None,
) -> list[str]:
    if not evaluated:
        return [f"## {label}", "", "No rule passed the gate.", ""]
    best_fixed = min(evaluated, key=lambda item: item.normalized_objective)
    oracle = library_oracle_score(evaluated)
    complementarity = best_fixed.normalized_objective - oracle
    percent = complementarity / best_fixed.normalized_objective * 100.0
    raw_best_fixed = fmean(best_fixed.values)
    raw_oracle = fmean(
        min(item.values[index] for item in evaluated)
        for index in range(len(evaluated[0].values))
    )
    lines = [
        f"## {label}",
        "",
        "| Rule | f_H |",
        "|---|---:|",
    ]
    lines.extend(
        f"| {item.rule.name} | {item.normalized_objective:.6f} |"
        for item in sorted(evaluated, key=lambda item: item.normalized_objective)
    )
    lines.extend(
        [
            "",
            f"- 库内最优单规则: {best_fixed.rule.name} ({best_fixed.normalized_objective:.6f})",
            f"- 库内逐实例最优标准化目标: {oracle:.6f}",
            f"- 互补性差距: {complementarity:.6f} ({percent:.2f}%)",
            f"- 库内最优单规则原始平均 F: {raw_best_fixed:.6f}",
            f"- 库内逐实例最优原始平均 F: {raw_oracle:.6f}",
            "",
        ]
    )
    if classic_values:
        for name in ("SPT", "ATC"):
            reference = fmean(classic_values[name])
            gap = (raw_best_fixed / reference - 1.0) * 100.0
            lines.append(
                f"- 最优单规则 vs {name}: {gap:+.2f}% "
                f"(参照平均 F={reference:.6f})"
            )
        lines.append("")
    return lines


def build_gated_library(args: argparse.Namespace) -> tuple[Path, Path]:
    candidate_arguments = (
        [args.candidates]
        if isinstance(args.candidates, (str, Path))
        else list(args.candidates)
    )
    candidate_paths = [Path(path) for path in candidate_arguments]
    output_path = Path(args.out)
    report_argument = getattr(args, "report", None)
    report_path = Path(report_argument) if report_argument else (
        output_path.parent / f"gated_report_{args.tag}.md"
    )
    rules, sources, merged_duplicates = _load_candidate_rules(candidate_paths)
    if args.extra_rules:
        extra_library = RuleLibrary.from_json(args.extra_rules)
        for rule in extra_library.rules:
            if rule.name not in {"SPT", "ATC_K2"}:
                continue
            if rule.name not in sources:
                rules.append(rule)
                sources[rule.name] = "classic"
    selection_instances = [
        make_instance(
            seed,
            scale=args.scale,
            difficulty=args.difficulty,
            scenario=getattr(args, "scenario", None),
        )
        for seed in SELECTION_SEEDS
    ]
    fixed_classic_name = None
    if args.gate_baseline == "best_classic":
        fixed_classic_name, baseline_values = _best_classic_baseline(
            selection_instances, feature_set=args.feature_set
        )
    else:
        baseline_values = _baseline_values(
            selection_instances,
            gate_baseline=args.gate_baseline,
            feature_set=args.feature_set,
        )
    reference_states: list[ReferenceState] = []
    fallback = default_rule(args.feature_set)
    for instance in selection_instances:
        _value, states = rollout_rule(
            fallback,
            instance,
            capture_reference_states=True,
            feature_set=args.feature_set,
        )
        reference_states.extend(states)
    evaluated = evaluate_rules(
        rules,
        selection_instances,
        baseline_values,
        sources=sources,
        feature_set=args.feature_set,
    )
    evaluated_llm = [item for item in evaluated if item.source != "classic"]
    evaluated_classics = [item for item in evaluated if item.source == "classic"]
    quality = (
        list(evaluated_llm)
        if getattr(args, "disable_quality_gate", False)
        else [
            item for item in evaluated_llm if item.normalized_objective < 1.0
        ]
    )
    retained, _similarities, similarity_rejected = behavior_deduplicate(
        quality, tuple(reference_states), args.sim_threshold
    )
    mandatory_classics = evaluated_classics[: min(2, args.K)]
    selected, gains = greedy_complementary_selection(
        retained,
        budget=args.K,
        gain_threshold=args.gain_threshold,
        initial=mandatory_classics,
    )
    if not mandatory_classics:
        selected = sorted(
            selected, key=lambda item: (item.normalized_objective, item.rule.name)
        )
    selection_reasons: dict[str, str] = {
        item.rule.name: "mandatory_classic_bypass_quality_gate"
        for item in mandatory_classics
    }
    for index, item in enumerate(selected):
        if item.rule.name in selection_reasons:
            continue
        selection_reasons[item.rule.name] = (
            "best_normalized_objective"
            if not mandatory_classics and index == 0
            else "greedy_marginal_gain"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_payload = {
        "rules": [
            {
                **item.rule.to_dict(),
                "source": item.source,
                "selection_reason": selection_reasons[item.rule.name],
                "f_H": item.normalized_objective,
                "selection_F": list(item.values),
                "max_selected_similarity": _max_selected_similarity(
                    item, selected, tuple(reference_states)
                ),
                "marginal_gain_at_selection": gains[item.rule.name],
            }
            for item in selected
        ]
    }
    output_path.write_text(
        json.dumps(output_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    holdout_instances = [
        make_instance(
            seed,
            scale=args.scale,
            difficulty=args.difficulty,
            scenario=getattr(args, "scenario", None),
        )
        for seed in HOLDOUT_SEEDS
    ]
    selection_classics = _classic_values(
        selection_instances, feature_set=args.feature_set
    )
    holdout_classics = _classic_values(
        holdout_instances, feature_set=args.feature_set
    )
    if fixed_classic_name is None:
        holdout_baseline = tuple(
            rollout_rule(
                default_rule(args.feature_set),
                instance,
                feature_set=args.feature_set,
            )[0]
            for instance in holdout_instances
        )
    else:
        holdout_baseline = holdout_classics[fixed_classic_name]
    holdout_evaluated = evaluate_rules(
        [item.rule for item in selected],
        holdout_instances,
        holdout_baseline,
        feature_set=args.feature_set,
    )
    selection_selected = [
        next(item for item in evaluated if item.rule.name == selected_item.rule.name)
        for selected_item in selected
    ]
    report_lines = [
        f"# Gated library report {args.tag}",
        "",
        f"- 输入候选文件: {', '.join(str(path) for path in candidate_paths)}",
        f"- 合并后 LLM/随机候选: {len(evaluated_llm)}",
        f"- 合并去重数: {merged_duplicates}",
        f"- 经典规则直接入库: {len(mandatory_classics)}",
        f"- 场景: {getattr(args, 'scenario', None) or f'{args.scale}/{args.difficulty}'}",
        f"- 特征集: {args.feature_set}",
        f"- 质量门槛基线: {args.gate_baseline}",
        f"- 质量门槛: {'关闭' if getattr(args, 'disable_quality_gate', False) else 'f_H < 1'}",
        (
            f"- best_classic 口径: 按选择集平均 F 选定单一固定基线 "
            f"{fixed_classic_name}; f_H 的逐实例分母均来自该规则"
            if fixed_classic_name is not None
            else "- default 口径: 缺省排序作为逐实例基线"
        ),
        f"- 质量门槛淘汰 f_H >= 1: {len(evaluated_llm) - len(quality)}",
        f"- 行为相似度淘汰 Sim >= {args.sim_threshold}: {similarity_rejected}",
        f"- 互补性贪心未入选: {len(retained) - len(selected)}",
        f"- 最终规则数: {len(selected)} / K={args.K}",
        f"- 固定参考状态数: {len(reference_states)}",
        "",
        "## 最终库",
        "",
        "| # | Rule | Source | Selection reason | Type | f_H | max Sim | marginal gain |",
        "|---:|---|---|---|---:|---:|---:|---:|",
    ]
    report_lines.extend(
        f"| {index} | {item.rule.name} | {item.source} | "
        f"{selection_reasons[item.rule.name]} | "
        f"{int(item.rule.rule_type) if item.rule.rule_type is not None else 'classic'} | "
        f"{item.normalized_objective:.6f} | "
        f"{_max_selected_similarity(item, selected, tuple(reference_states)):.6f} | "
        f"{gains[item.rule.name]:.6f} |"
        for index, item in enumerate(selected, 1)
    )
    report_lines.extend([""])
    report_lines.extend(
        _report_section(
            "选择集 seeds 600–619", selection_selected, selection_classics
        )
    )
    report_lines.extend(
        _report_section(
            "保留集 seeds 700–719", holdout_evaluated, holdout_classics
        )
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    return output_path, report_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", nargs="+", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--K", type=int, choices=(3, 6), default=6)
    parser.add_argument("--sim_threshold", type=float, default=0.9)
    parser.add_argument("--gain_threshold", type=float, default=0.002)
    parser.add_argument("--out", required=True)
    parser.add_argument("--report")
    parser.add_argument("--scale", choices=("S", "M", "L", "X"), default="M")
    parser.add_argument("--scenario", choices=("center",), default=None)
    parser.add_argument("--difficulty", default="balanced")
    parser.add_argument(
        "--gate_baseline",
        "--gate-baseline",
        dest="gate_baseline",
        choices=("default", "best_classic"),
        default="default",
    )
    parser.add_argument(
        "--extra_rules",
        "--extra-rules",
        dest="extra_rules",
    )
    parser.add_argument(
        "--feature_set",
        "--feature-set",
        dest="feature_set",
        choices=("theory11", "theory11_plus3"),
        default="theory11",
    )
    parser.add_argument(
        "--disable_quality_gate",
        "--disable-quality-gate",
        dest="disable_quality_gate",
        action="store_true",
        help="retain all candidates before behaviour deduplication",
    )
    args = parser.parse_args(argv)
    if not 0.0 <= args.sim_threshold <= 1.0:
        parser.error("--sim_threshold must lie in [0, 1]")
    if args.gain_threshold < 0.0:
        parser.error("--gain_threshold must be non-negative")
    return args


def main() -> None:
    output, report = build_gated_library(parse_args())
    print(f"library: {output.resolve()}")
    print(f"report:  {report.resolve()}")


if __name__ == "__main__":
    main()
