"""Generate, validate, select, and audit an LLM symbolic-rule library."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

PACKAGE_PARENT = Path(__file__).resolve().parents[2]
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from fjsp_sfr.dispatch.rule_library import RuleLibrary
from fjsp_sfr.dispatch.classic_rules import build_classic_rule_library
from fjsp_sfr.dispatch.symbolic_rule import RuleType, SymbolicRule
from fjsp_sfr.dispatch.theory_features import (
    build_extended_theory_features,
    feature_names_for,
)
from fjsp_sfr.env.ca_rule_selection import CAPPORuleSelectionEnv
from fjsp_sfr.experiments.validate_framework import (
    build_rule_library,
    evaluate_single_rule,
    make_fjsp_instance,
)
from fjsp_sfr.experiments.manifest import file_sha256, write_manifest
from fjsp_sfr.experiments.build_gated_library import default_rule
from fjsp_sfr.experiments.eval_classic import evaluate_rule
from fjsp_sfr.generation.scale_config import make_scaled_instance
from fjsp_sfr.llm.generator import compute_behavior_similarity
from fjsp_sfr.llm.prompt_template import build_generation_prompt
from fjsp_sfr.llm.qwen_client import QwenClient
from fjsp_sfr.llm.rule_parser import parse_llm_output_detailed


SELECTION_SEEDS = tuple(range(600, 620))
HOLDOUT_SEEDS = tuple(range(700, 720))
VALIDATION_SEEDS = tuple(range(100, 150))
TEST_SEEDS = tuple(range(200, 500))
TRAIN_BASE_SEED = 1000
DEFAULT_TRAIN_EPISODES = 5000


@dataclass(frozen=True)
class CandidateRecord:
    rule: SymbolicRule
    objective: float
    iteration: int
    total_tardiness: float = math.nan
    weighted_service_shortfall: float = math.nan

    def to_dict(self) -> dict:
        return {
            "rule": self.rule.to_dict(),
            "objective": self.objective,
            "iteration": self.iteration,
            "total_tardiness": self.total_tardiness,
            "weighted_service_shortfall": self.weighted_service_shortfall,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CandidateRecord":
        return cls(
            rule=SymbolicRule.from_dict(data["rule"]),
            objective=float(data["objective"]),
            iteration=int(data["iteration"]),
            total_tardiness=float(data.get("total_tardiness", math.nan)),
            weighted_service_shortfall=float(
                data.get("weighted_service_shortfall", math.nan)
            ),
        )


def assert_seed_partitions_disjoint(
    *, training_episodes: int = DEFAULT_TRAIN_EPISODES
) -> None:
    """Assert all selection, validation, test, and streaming seeds are disjoint."""

    partitions = {
        "selection": set(SELECTION_SEEDS),
        "holdout": set(HOLDOUT_SEEDS),
        "validation": set(VALIDATION_SEEDS),
        "test": set(TEST_SEEDS),
        "training": set(
            range(TRAIN_BASE_SEED, TRAIN_BASE_SEED + training_episodes)
        ),
    }
    names = list(partitions)
    for index, left_name in enumerate(names):
        for right_name in names[index + 1 :]:
            overlap = partitions[left_name] & partitions[right_name]
            if overlap:
                raise AssertionError(
                    f"seed partitions {left_name} and {right_name} overlap: "
                    f"{sorted(overlap)[:5]}"
                )


def make_generation_instance(
    seed: int, *, scale: str = "M", difficulty: str = "service_heavy"
):
    return make_scaled_instance(seed, scale=scale, difficulty=difficulty)


def _rule_signature(rule: SymbolicRule) -> tuple:
    return (
        rule.feature_set,
        int(rule.rule_type),
        rule.priority_expr,
        rule.cond_expr,
        rule.filter_expr,
    )


def _summary_row(rows: tuple[tuple[float, ...], ...]) -> tuple[float, ...]:
    return tuple(max(column) for column in zip(*rows, strict=True)) if rows else ()


def _candidate_choice(
    rule: SymbolicRule, rows: tuple[tuple[float, ...], ...]
) -> int:
    active_rule = rule
    if rule.cond_expr is not None and not rule.is_applicable(_summary_row(rows)):
        names = feature_names_for(rule.feature_set)
        scores = [-row[names.index("due_date_slack")] for row in rows]
    else:
        scores = active_rule.evaluate_batch(rows)
    return max(range(len(scores)), key=lambda index: scores[index])


def inspect_priority_behavior(
    rule: SymbolicRule,
    instances: list,
) -> str | None:
    """Return a degeneracy reason, or ``None`` for a usable priority."""

    raw_scores: list[float] = []
    assert rule._compiled_priority is not None
    for instance in instances:
        rollout = deepcopy(instance)
        env = CAPPORuleSelectionEnv(rollout)
        env.reset()
        while env.current_metrics and env.current_metrics.is_active:
            rows = build_extended_theory_features(
                env._problem_view(),
                env._state,
                env.current_metrics,
                feature_set=rule.feature_set,
            )
            try:
                scores = rule.priority_scores(rows)
            except (ArithmeticError, OverflowError, ValueError):
                return "priority_runtime_error"
            if any(not math.isfinite(score) for score in scores):
                return "non_finite_priority"
            raw_scores.extend(scores)
            try:
                choice = _candidate_choice(rule, rows)
            except (ArithmeticError, OverflowError, ValueError):
                return "priority_runtime_error"
            result = env.step_candidate(choice)
            if result.terminated:
                break
    if not raw_scores or max(raw_scores) - min(raw_scores) <= 1e-12:
        return "constant_priority"
    return None


def evaluate_candidate(
    rule: SymbolicRule, instances: list
) -> tuple[float, float, float]:
    values = [evaluate_rule(rule, instance) for instance in instances]
    objective = sum(value.Z for value in values) / len(values)
    tardiness = sum(value.total_tardiness for value in values) / len(values)
    shortfall = sum(
        value.weighted_checkpoint_shortfall for value in values
    ) / len(values)
    if not all(math.isfinite(value) for value in (objective, tardiness, shortfall)):
        raise ValueError("non-finite simulation objective")
    return objective, tardiness, shortfall


def select_diverse_library(
    candidates: list[CandidateRecord],
    budget: int,
    instances: list,
    *,
    similarity_threshold: float = 0.98,
) -> tuple[list[CandidateRecord], dict[tuple[str, str], float]]:
    """Greedy objective-quality and behavior-diversity selection."""

    if budget <= 0 or not candidates:
        return [], {}
    remaining = sorted(candidates, key=lambda item: (item.objective, item.rule.name))
    selected: list[CandidateRecord] = []
    cache: dict[tuple[str, str], float] = {}

    def similarity(left: SymbolicRule, right: SymbolicRule) -> float:
        key = tuple(sorted((left.name, right.name)))
        if key not in cache:
            cache[key] = compute_behavior_similarity(left, right, instances)
        return cache[key]

    while remaining and len(selected) < budget:
        best: CandidateRecord | None = None
        best_gain = float("-inf")
        for candidate in remaining:
            similarities = [
                similarity(candidate.rule, existing.rule) for existing in selected
            ]
            max_similarity = max(similarities, default=0.0)
            if max_similarity > similarity_threshold:
                continue
            gain = (1.0 / max(candidate.objective, 1e-12)) * (
                1.0 - max_similarity
            )
            if gain > best_gain:
                best_gain = gain
                best = candidate
        if best is None:
            break
        selected.append(best)
        remaining.remove(best)
    return selected, cache


class DryRunClient:
    """Deterministic local response containing valid and invalid candidates."""

    def __init__(self, per_round: int) -> None:
        self.per_round = per_round
        self.calls = 0

    def generate(self, _prompt: str) -> str:
        round_index = self.calls
        self.calls += 1
        templates = [
            ("I", "gap_pressure", "NONE", "NONE"),
            ("I", "-due_date_slack", "NONE", "NONE"),
            ("II", "forecast_deficit + gap_pressure", "NONE", "NONE"),
            ("II", "-tardiness_exposure + processing_ratio", "NONE", "NONE"),
            ("III", "0.6 * multi_checkpoint_value + 0.4 * gap_pressure", "NONE", "NONE"),
            ("III", "capacity_pressure - 0.3 * processing_ratio", "NONE", "NONE"),
            ("IV", "quantity_contribution + forecast_deficit", "forecast_flag", "quantity_contribution - 0.05"),
            ("IV", "checkpoint_slack_urgency + capacity_pressure", "capacity_flag", "capacity_pressure - 0.01"),
            ("I", "1.0", "NONE", "NONE"),
            ("I", "__import__('os')", "NONE", "NONE"),
        ]
        blocks: list[str] = []
        for index in range(self.per_round):
            rule_type, priority, condition, filter_expression = templates[
                index % len(templates)
            ]
            blocks.append(
                "\n".join(
                    [
                        f"RULE: dry_r{round_index}_{index}",
                        f"TYPE: {rule_type}",
                        f"COND: {condition}",
                        f"FILTER: {filter_expression}",
                        f"PRIORITY: {priority}",
                    ]
                )
            )
        return "\n\n".join(blocks)


def _load_state(path: Path) -> tuple[int, list[CandidateRecord], Counter[str], int, int]:
    if not path.exists():
        return 0, [], Counter(), 0, 0
    data = json.loads(path.read_text(encoding="utf-8"))
    return (
        int(data.get("completed_rounds", 0)),
        [CandidateRecord.from_dict(item) for item in data.get("candidates", [])],
        Counter(data.get("reason_counts", {})),
        int(data.get("candidate_total", 0)),
        int(data.get("parsed_total", 0)),
    )


def _save_state(
    path: Path,
    *,
    tag: str,
    completed_rounds: int,
    candidates: list[CandidateRecord],
    reason_counts: Counter[str],
    candidate_total: int,
    parsed_total: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "tag": tag,
                "completed_rounds": completed_rounds,
                "candidate_total": candidate_total,
                "parsed_total": parsed_total,
                "reason_counts": dict(reason_counts),
                "candidates": [item.to_dict() for item in candidates],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _classic_reference(instances: list, feature_set: str) -> dict[str, dict[str, float]]:
    classics = build_classic_rule_library()
    wanted = {
        "SPT": next(rule for rule in classics.rules if rule.name == "SPT"),
        "ATC": next(rule for rule in classics.rules if rule.name == "ATC_K2"),
        "DEFAULT": default_rule(feature_set),
    }
    result: dict[str, dict[str, float]] = {}
    for name, rule in wanted.items():
        objectives = [evaluate_rule(rule, instance) for instance in instances]
        result[name] = {
            "F": sum(item.Z for item in objectives) / len(objectives),
            "TT": sum(item.total_tardiness for item in objectives) / len(objectives),
            "WSF": sum(
                item.weighted_checkpoint_shortfall for item in objectives
            )
            / len(objectives),
        }
    return result


def _classic_reference_text(reference: dict[str, dict[str, float]]) -> str:
    return "\n".join(
        f"- {name}: mean F={metrics['F']:.6f}, "
        f"TT={metrics['TT']:.6f}, WSF={metrics['WSF']:.6f}"
        for name, metrics in reference.items()
    )


def _feedback(
    selected: list[CandidateRecord],
    classic_reference: dict[str, dict[str, float]] | None = None,
) -> str:
    if not selected:
        return "No candidate has passed evaluation yet."
    lines = [
        "Previously selected rules on the selection set (lower is better):"
    ]
    for item in selected:
        line = (
            f"- {item.rule.name}: F={item.objective:.6f}, "
            f"TT={item.total_tardiness:.6f}, "
            f"WSF={item.weighted_service_shortfall:.6f}"
        )
        if classic_reference:
            spt_gap = (
                item.objective
                / max(1.0, classic_reference["SPT"]["F"])
                - 1.0
            ) * 100.0
            atc_gap = (
                item.objective
                / max(1.0, classic_reference["ATC"]["F"])
                - 1.0
            ) * 100.0
            line += f", gap vs SPT={spt_gap:+.2f}%, gap vs ATC={atc_gap:+.2f}%"
        lines.append(line)
    return "\n".join(lines)


def similarity_matrix(rules: list[SymbolicRule], instances: list) -> list[list[float]]:
    matrix: list[list[float]] = []
    for left in rules:
        matrix.append(
            [
                1.0
                if left is right
                else compute_behavior_similarity(left, right, instances)
                for right in rules
            ]
        )
    return matrix


def _library_holdout_summary(
    label: str, library: RuleLibrary, instances: list
) -> dict:
    by_rule = {
        rule.name: [evaluate_single_rule(rule, instance) for instance in instances]
        for rule in library.rules
    }
    means = {
        name: sum(values) / len(values) for name, values in by_rule.items()
    }
    best_name = min(means, key=means.get)
    best_fixed = means[best_name]
    oracle = sum(
        min(values[index] for values in by_rule.values())
        for index in range(len(instances))
    ) / len(instances)
    complementarity = (
        (best_fixed - oracle) / best_fixed * 100.0
        if best_fixed > 0.0
        else 0.0
    )
    return {
        "label": label,
        "rule_means": means,
        "best_name": best_name,
        "best_fixed": best_fixed,
        "oracle": oracle,
        "complementarity": complementarity,
    }


def _print_holdout(summaries: list[dict]) -> None:
    print("\n留出集比较 (seeds 700-719; 未参与选择):")
    for summary in summaries:
        print(f"\n  {summary['label']}")
        for name, value in summary["rule_means"].items():
            print(f"    {name:>28}: {value:.2f}")
        print(
            f"    库内最优固定: {summary['best_name']} "
            f"({summary['best_fixed']:.2f})"
        )
        print(f"    逐实例事后上界: {summary['oracle']:.2f}")
        print(f"    互补性改善: {summary['complementarity']:.2f}%")


def run_generation(args: argparse.Namespace) -> tuple[Path, Path]:
    generation_started = time.monotonic()
    assert_seed_partitions_disjoint()
    feature_set = getattr(args, "feature_set", "theory11")
    scale = getattr(args, "scale", None)
    difficulty = getattr(args, "difficulty", None)
    show_classic_reference = getattr(args, "show_classic_reference", True)
    output_dir = Path(args.output_dir)
    state_path = output_dir / f"llm_candidates_{args.tag}.json"
    six_path = output_dir / f"llm_library_{args.tag}_6.json"
    three_path = output_dir / f"llm_library_{args.tag}_3.json"
    print(f"Model: {args.model}")

    def make_instance(seed: int):
        # Preserve compatibility with tests and callers that monkeypatch the
        # historical one-argument factory.
        if scale is None and difficulty is None:
            return make_generation_instance(seed)
        return make_generation_instance(
            seed,
            scale=scale or "M",
            difficulty=difficulty or "service_heavy",
        )

    selection_instances = [make_instance(seed) for seed in SELECTION_SEEDS]
    holdout_instances = [make_instance(seed) for seed in HOLDOUT_SEEDS]
    classic_reference = (
        _classic_reference(selection_instances, feature_set)
        if show_classic_reference
        else None
    )
    if show_classic_reference:
        print("Classic reference enabled:")
        print(_classic_reference_text(classic_reference or {}))
    else:
        print("Classic reference disabled")
    if args.dry_run:
        client = DryRunClient(args.per_round)
    else:
        client = QwenClient(
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            enable_thinking=args.enable_thinking,
            tag=args.tag,
            log_root=args.log_root,
        )

    completed_rounds, candidates, reason_counts, candidate_total, parsed_total = (
        _load_state(state_path)
    )
    known_signatures = {_rule_signature(item.rule) for item in candidates}

    for iteration in range(completed_rounds, args.rounds):
        provisional, _cache = select_diverse_library(
            candidates,
            args.library_size,
            selection_instances,
            similarity_threshold=0.98,
        )
        prompt = build_generation_prompt(
            existing_rules=[item.rule for item in provisional],
            iteration=iteration,
            num_rules=args.per_round,
            feedback=_feedback(provisional, classic_reference),
            require_type_coverage=True,
            feature_set=feature_set,
            classic_reference=(
                _classic_reference_text(classic_reference)
                if classic_reference
                else None
            ),
        )
        if args.dry_run:
            print(f"\n===== DRY-RUN PROMPT ROUND {iteration + 1} =====")
            print(prompt)
        raw_output = client.generate(prompt)
        parsed, failures = parse_llm_output_detailed(
            raw_output, feature_names_for(feature_set)
        )
        candidate_total += len(parsed) + len(failures)
        parsed_total += len(parsed)
        reason_counts.update(failure.category for failure in failures)

        for rule in parsed:
            signature = _rule_signature(rule)
            if signature in known_signatures:
                reason_counts["duplicate_definition"] += 1
                continue
            try:
                degeneracy = inspect_priority_behavior(
                    rule, selection_instances
                )
            except Exception:
                reason_counts["simulation_error"] += 1
                continue
            if degeneracy is not None:
                reason_counts[degeneracy] += 1
                continue
            try:
                objective, tardiness, shortfall = evaluate_candidate(
                    rule, selection_instances
                )
            except Exception:
                reason_counts["simulation_error"] += 1
                continue
            candidates.append(
                CandidateRecord(
                    rule=rule,
                    objective=objective,
                    iteration=iteration,
                    total_tardiness=tardiness,
                    weighted_service_shortfall=shortfall,
                )
            )
            known_signatures.add(signature)

        provisional, _cache = select_diverse_library(
            candidates,
            args.library_size,
            selection_instances,
            similarity_threshold=0.98,
        )
        behavior_rejected = max(0, len(candidates) - len(provisional))
        print(
            f"Round {iteration + 1}/{args.rounds}: parsed={len(parsed)}, "
            f"usable_pool={len(candidates)}, provisional={len(provisional)}, "
            f"diversity_not_selected={behavior_rejected}",
            flush=True,
        )
        _save_state(
            state_path,
            tag=args.tag,
            completed_rounds=iteration + 1,
            candidates=candidates,
            reason_counts=reason_counts,
            candidate_total=candidate_total,
            parsed_total=parsed_total,
        )

    selected, similarity_cache = select_diverse_library(
        candidates,
        args.library_size,
        selection_instances,
        similarity_threshold=0.98,
    )
    if len(selected) < args.library_size:
        raise RuntimeError(
            f"only {len(selected)} diverse rules passed; "
            f"need library_size={args.library_size}"
        )
    selected_signatures = {_rule_signature(item.rule) for item in selected}
    behavior_duplicates = 0
    for candidate in candidates:
        if _rule_signature(candidate.rule) in selected_signatures:
            continue
        is_behavior_duplicate = False
        for existing in selected:
            key = tuple(sorted((candidate.rule.name, existing.rule.name)))
            similarity = similarity_cache.get(key)
            if similarity is None:
                similarity = compute_behavior_similarity(
                    candidate.rule, existing.rule, selection_instances
                )
                similarity_cache[key] = similarity
            if similarity > 0.98:
                is_behavior_duplicate = True
                break
        if is_behavior_duplicate:
            behavior_duplicates += 1
    reason_counts["behavior_duplicate"] = behavior_duplicates
    _save_state(
        state_path,
        tag=args.tag,
        completed_rounds=args.rounds,
        candidates=candidates,
        reason_counts=reason_counts,
        candidate_total=candidate_total,
        parsed_total=parsed_total,
    )
    six_rules = [item.rule for item in selected]
    three_rules = six_rules[:3]
    RuleLibrary(tuple(six_rules)).to_json(str(six_path))
    RuleLibrary(tuple(three_rules)).to_json(str(three_path))
    write_manifest(
        output_dir / f"generate_{args.tag}_manifest.json",
        {
            "kind": "rule_generation",
            "tag": args.tag,
            "model": args.model,
            "temperature": args.temperature,
            "rounds": args.rounds,
            "per_round": args.per_round,
            "library_size": args.library_size,
            "enable_thinking": args.enable_thinking,
            "dry_run": bool(args.dry_run),
            "feature_set": feature_set,
            "scale": scale or "M",
            "difficulty": difficulty or "service_heavy",
            "show_classic_reference": bool(show_classic_reference),
            "seeds": {
                "selection": list(SELECTION_SEEDS),
                "holdout": list(HOLDOUT_SEEDS),
            },
            "library_files": {
                str(six_path): file_sha256(six_path),
                str(three_path): file_sha256(three_path),
            },
            "llm_api_request_count": int(
                getattr(client, "api_request_count", 0)
            ),
            "llm_token_usage": dict(getattr(client, "usage_totals", {})),
            "generation_seconds_this_invocation": (
                time.monotonic() - generation_started
            ),
        },
    )

    print("\n" + "=" * 100)
    print("生成与选择汇总")
    print("=" * 100)
    parse_rate = parsed_total / candidate_total * 100.0 if candidate_total else 0.0
    print(f"候选总数: {candidate_total}")
    print(f"解析成功: {parsed_total} ({parse_rate:.1f}%)")
    print(f"失败与过滤: {dict(sorted(reason_counts.items()))}")
    print("\n入选规则 (按贪心入选顺序):")
    for index, item in enumerate(selected, 1):
        print(
            f"  {index}. {item.rule.name} | Type {int(item.rule.rule_type)} | "
            f"F={item.objective:.2f} | TT={item.total_tardiness:.2f} | "
            f"WSF={item.weighted_service_shortfall:.2f} | "
            f"{item.rule.priority_expr}"
        )
    type_counts = Counter(int(item.rule.rule_type) for item in selected)
    print(f"类型分布: {dict(sorted(type_counts.items()))}")

    matrix = similarity_matrix(six_rules, selection_instances)
    print("\n入选规则行为相似度矩阵:")
    print(" " * 18 + " ".join(f"R{i + 1:>4}" for i in range(len(six_rules))))
    for index, row in enumerate(matrix):
        print(f"  R{index + 1:<3} {six_rules[index].name[:10]:<10} " + " ".join(f"{value:5.2f}" for value in row))

    summaries = [
        _library_holdout_summary(
            "LLM 六条库", RuleLibrary(tuple(six_rules)), holdout_instances
        ),
        _library_holdout_summary(
            "LLM 三条库", RuleLibrary(tuple(three_rules)), holdout_instances
        ),
        _library_holdout_summary(
            "手写六条库", build_rule_library(), holdout_instances
        ),
    ]
    _print_holdout(summaries)
    print(f"\n已导出: {six_path}")
    print(f"已导出: {three_path}")
    return six_path, three_path


def run_ping(args: argparse.Namespace) -> None:
    print(f"Model: {args.model}")
    client = QwenClient(
        model=args.model,
        temperature=args.temperature,
        max_tokens=min(args.max_tokens, 32),
        timeout=args.timeout,
        enable_thinking=args.enable_thinking,
        tag=args.tag,
        log_root=args.log_root,
    )
    started = time.monotonic()
    reply = client.generate("Reply with exactly: pong")
    elapsed = time.monotonic() - started
    print(f"回复前100字符: {reply[:100]}")
    print(f"耗时: {elapsed:.3f}s")
    print(f"token 用量: {client.last_usage or 'unavailable'}")
    print(f"缓存命中: {client.last_cache_hit}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--per_round", "--per-round", dest="per_round", type=int, default=10)
    parser.add_argument("--library_size", "--library-size", dest="library_size", type=int, default=6)
    parser.add_argument("--max_tokens", "--max-tokens", dest="max_tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--enable_thinking",
        "--enable-thinking",
        dest="enable_thinking",
        choices=("auto", "true", "false"),
        default="false",
    )
    parser.add_argument("--dry_run", "--dry-run", dest="dry_run", action="store_true")
    parser.add_argument(
        "--feature_set",
        "--feature-set",
        dest="feature_set",
        choices=("theory11", "theory11_plus3"),
        default="theory11_plus3",
    )
    parser.add_argument("--scale", choices=("S", "M", "L", "X"), default="M")
    parser.add_argument(
        "--difficulty",
        choices=("easy", "balanced", "hard", "service_heavy"),
        default="service_heavy",
    )
    parser.add_argument(
        "--show_classic_reference",
        "--show-classic-reference",
        dest="show_classic_reference",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="include SPT/ATC/default F, TT and WSF in prompts and feedback",
    )
    parser.add_argument("--ping", action="store_true")
    parser.add_argument(
        "--output-dir", default=str(Path(__file__).resolve().parent)
    )
    parser.add_argument("--log-root", default="llm_logs")
    args = parser.parse_args(argv)
    if any(
        character
        not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
        for character in args.tag
    ):
        parser.error("--tag may contain only letters, digits, '_' and '-'")
    if min(args.rounds, args.per_round, args.library_size) <= 0:
        parser.error("rounds, per_round, and library_size must be positive")
    if args.per_round < 8:
        parser.error(
            "--per_round must be at least 8 to request two rules of each type"
        )
    if args.library_size < 3:
        parser.error("--library_size must be at least 3")
    return args


def main() -> None:
    args = parse_args()
    if args.ping:
        if args.dry_run:
            print(f"Model: {args.model}")
            reply = DryRunClient(1).generate("ping")
            print(f"回复前100字符: {reply[:100]}")
            print("耗时: 0.000s")
            print("token 用量: unavailable (dry run)")
            return
        run_ping(args)
        return
    run_generation(args)


if __name__ == "__main__":
    main()
