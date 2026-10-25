"""Build a quality-first three-rule library from cached Qwen candidates.

This script is intentionally offline: it reads previously generated JSON
artifacts, runs only local scheduling simulations, and never constructs an LLM
client or performs an API request.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

PACKAGE_PARENT = Path(__file__).resolve().parents[2]
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from fjsp_sfr.dispatch.rule_library import RuleLibrary
from fjsp_sfr.dispatch.symbolic_rule import RuleType, SymbolicRule
from fjsp_sfr.experiments.validate_framework import (
    build_rule_library,
    evaluate_single_rule,
    make_fjsp_instance,
)
from fjsp_sfr.llm.generator import compute_behavior_similarity


EXPERIMENTS_DIR = Path(__file__).resolve().parent
CANDIDATES_PATH = EXPERIMENTS_DIR / "llm_candidates_qwen1.json"
SIX_LIBRARY_PATH = EXPERIMENTS_DIR / "llm_library_qwen1_6.json"
OUTPUT_PATH = EXPERIMENTS_DIR / "llm_library_qwen1_3q.json"
SELECTION_SEEDS = tuple(range(600, 620))
HOLDOUT_SEEDS = tuple(range(700, 720))
SIMILARITY_THRESHOLD = 0.9
HANDWRITTEN_TOP3_NAMES = (
    "service_value",
    "capacity_aware",
    "tardiness_avoidance",
)
TYPE_LABELS = {
    RuleType.TYPE_I: "I",
    RuleType.TYPE_II: "II",
    RuleType.TYPE_III: "III",
    RuleType.TYPE_IV: "IV",
}


@dataclass(frozen=True)
class Candidate:
    rule: SymbolicRule
    objective: float


def make_instance(seed: int):
    return make_fjsp_instance(
        seed,
        n_jobs=15,
        n_machines=5,
        n_entities=2,
    )


def load_candidates(path: Path) -> list[Candidate]:
    data = json.loads(path.read_text(encoding="utf-8"))
    candidates = [
        Candidate(
            rule=SymbolicRule.from_dict(item["rule"]),
            objective=float(item["objective"]),
        )
        for item in data["candidates"]
    ]
    if len(candidates) != 37:
        raise ValueError(
            f"expected 37 usable candidates, found {len(candidates)}"
        )
    return sorted(
        candidates,
        key=lambda item: (item.objective, item.rule.name),
    )


def percentile(sorted_values: list[float], probability: float) -> float:
    """Linearly interpolate the requested empirical percentile."""

    if not sorted_values:
        raise ValueError("cannot calculate a percentile of an empty sequence")
    position = probability * (len(sorted_values) - 1)
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    fraction = position - lower_index
    return (
        sorted_values[lower_index] * (1.0 - fraction)
        + sorted_values[upper_index] * fraction
    )


def print_candidate_table(
    candidates: list[Candidate],
    six_library: RuleLibrary,
) -> None:
    six_rules = set(six_library.rules)
    print("=" * 150)
    print("可用候选池：按选择集平均 F 升序")
    print("=" * 150)
    print(
        f"{'序号':>4} | {'名称':<34} | {'类型':<4} | "
        f"{'选择集 F':>10} | {'六条库':<6} | 表达式"
    )
    print("-" * 150)
    for index, candidate in enumerate(candidates, 1):
        selected = "是" if candidate.rule in six_rules else "否"
        print(
            f"{index:>4} | {candidate.rule.name:<34} | "
            f"{TYPE_LABELS[candidate.rule.rule_type]:<4} | "
            f"{candidate.objective:>10.2f} | {selected:<6} | "
            f"{candidate.rule.priority_expr}"
        )

    print("\n各规则类型可用数量与该类型最佳 F:")
    for rule_type in RuleType:
        typed = [
            candidate
            for candidate in candidates
            if candidate.rule.rule_type == rule_type
        ]
        best = min(item.objective for item in typed)
        print(
            f"  Type {TYPE_LABELS[rule_type]}: "
            f"可用数量={len(typed)}, 最佳 F={best:.2f}"
        )

    values = sorted(candidate.objective for candidate in candidates)
    print("\n全部候选 F 分布（线性插值分位数）:")
    print(f"  最小值: {values[0]:.2f}")
    print(f"  25% 分位: {percentile(values, 0.25):.2f}")
    print(f"  中位数: {percentile(values, 0.50):.2f}")
    print(f"  75% 分位: {percentile(values, 0.75):.2f}")
    print(f"  最大值: {values[-1]:.2f}")


def build_quality_library(
    candidates: list[Candidate],
    selection_instances: list,
) -> tuple[RuleLibrary, list[tuple[Candidate, float]]]:
    selected: list[Candidate] = []
    audit_rows: list[tuple[Candidate, float]] = []
    for candidate in candidates:
        similarities = [
            compute_behavior_similarity(
                candidate.rule,
                existing.rule,
                selection_instances,
            )
            for existing in selected
        ]
        max_similarity = max(similarities, default=0.0)
        if max_similarity >= SIMILARITY_THRESHOLD:
            continue
        selected.append(candidate)
        audit_rows.append((candidate, max_similarity))
        if len(selected) == 3:
            break
    if len(selected) != 3:
        raise RuntimeError(
            f"could select only {len(selected)} rules below similarity threshold"
        )
    return RuleLibrary(tuple(item.rule for item in selected)), audit_rows


def library_summary(label: str, library: RuleLibrary, instances: list) -> dict:
    per_rule = {
        rule.name: [
            evaluate_single_rule(rule, instance) for instance in instances
        ]
        for rule in library.rules
    }
    means = {
        name: sum(values) / len(values) for name, values in per_rule.items()
    }
    best_name = min(means, key=means.get)
    best_fixed = means[best_name]
    oracle = sum(
        min(values[index] for values in per_rule.values())
        for index in range(len(instances))
    ) / len(instances)
    complementarity = (
        (best_fixed - oracle) / best_fixed * 100.0
        if best_fixed > 0.0
        else 0.0
    )
    return {
        "label": label,
        "means": means,
        "best_name": best_name,
        "best_fixed": best_fixed,
        "oracle": oracle,
        "complementarity": complementarity,
    }


def print_comparison(summaries: list[dict]) -> None:
    print("\n" + "=" * 100)
    print("留出集比较 (seeds 700-719; 未参与选择)")
    print("=" * 100)
    for summary in summaries:
        print(f"\n  {summary['label']}")
        for name, value in summary["means"].items():
            print(f"    {name:>34}: {value:.2f}")
        print(
            f"    库内最优固定: {summary['best_name']} "
            f"({summary['best_fixed']:.2f})"
        )
        print(f"    逐实例事后上界: {summary['oracle']:.2f}")
        print(f"    互补性改善: {summary['complementarity']:.2f}%")


def handwritten_top3() -> RuleLibrary:
    by_name = {rule.name: rule for rule in build_rule_library().rules}
    missing = set(HANDWRITTEN_TOP3_NAMES) - set(by_name)
    if missing:
        raise RuntimeError(f"missing handwritten rules: {sorted(missing)}")
    return RuleLibrary(tuple(by_name[name] for name in HANDWRITTEN_TOP3_NAMES))


def print_training_command() -> None:
    print("\n" + "=" * 100)
    print("训练命令（仅输出，未执行）")
    print("=" * 100)
    print(
        "PYTHONPATH=.. python3 experiments/train_streaming.py \\\n"
        "  --library_file experiments/llm_library_qwen1_3q.json \\\n"
        "  --tag qwen1_3q \\\n"
        "  --seed 42 \\\n"
        "  --base_seed 1000 \\\n"
        "  --episodes 5000 \\\n"
        "  --episodes_per_update 32"
    )


def main() -> None:
    candidates = load_candidates(CANDIDATES_PATH)
    six_library = RuleLibrary.from_json(str(SIX_LIBRARY_PATH))
    print_candidate_table(candidates, six_library)

    selection_instances = [make_instance(seed) for seed in SELECTION_SEEDS]
    quality_library, audit_rows = build_quality_library(
        candidates, selection_instances
    )
    quality_library.to_json(str(OUTPUT_PATH))

    print("\n" + "=" * 100)
    print("质量优先三条库（相似度阈值 < 0.9）")
    print("=" * 100)
    for index, (candidate, max_similarity) in enumerate(audit_rows, 1):
        print(
            f"  {index}. {candidate.rule.name} | "
            f"Type {TYPE_LABELS[candidate.rule.rule_type]} | "
            f"选择集 F={candidate.objective:.2f} | "
            f"与已选规则最大相似度={max_similarity:.4f} | "
            f"{candidate.rule.priority_expr}"
        )
    print(f"已导出: {OUTPUT_PATH}")

    holdout_instances = [make_instance(seed) for seed in HOLDOUT_SEEDS]
    print_comparison(
        [
            library_summary(
                "LLM 质量优先三条库", quality_library, holdout_instances
            ),
            library_summary(
                "手写 top3", handwritten_top3(), holdout_instances
            ),
        ]
    )
    print_training_command()


if __name__ == "__main__":
    main()
