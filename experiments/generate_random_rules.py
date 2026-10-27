"""Generate a deterministic, balanced random symbolic-rule baseline."""

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
import random
from collections import Counter
from pathlib import Path

from fjsp_sfr.dispatch.symbolic_rule import RuleType, SymbolicRule
from fjsp_sfr.dispatch.theory_features import feature_names_for


FEATURES = feature_names_for("theory11_plus3")
CONTINUOUS_FEATURES = tuple(
    name for name in FEATURES if name not in {"forecast_flag", "capacity_flag"}
)


def _coefficient(rng: random.Random, low: float = 0.05, high: float = 0.45) -> float:
    return round(rng.uniform(low, high), 2)


def _optional_filter(rng: random.Random) -> str | None:
    if rng.random() >= 0.25:
        return None
    feature = rng.choice(CONTINUOUS_FEATURES)
    return f"{feature} - {rng.uniform(0.05, 0.45):.2f}"


def _make_rule(rng: random.Random, rule_type: RuleType, index: int) -> SymbolicRule:
    cond_expr = None
    filter_expr = _optional_filter(rng)
    if rule_type == RuleType.TYPE_I:
        feature = rng.choice(FEATURES)
        sign = "-" if rng.random() < 0.35 else ""
        priority = f"{sign}{feature}"
    elif rule_type == RuleType.TYPE_II:
        count = rng.choice((2, 3))
        chosen = rng.sample(list(FEATURES), count)
        terms = [chosen[0]]
        for feature in chosen[1:]:
            sign = "+" if rng.random() < 0.65 else "-"
            terms.append(f"{sign} {_coefficient(rng):.2f} * {feature}")
        priority = " ".join(terms)
    elif rule_type == RuleType.TYPE_III:
        chosen = rng.sample(list(FEATURES), rng.randint(3, 5))
        raw_weights = [rng.uniform(0.1, 1.0) for _ in chosen]
        total = sum(raw_weights)
        weights = [value / total for value in raw_weights]
        priority = " + ".join(
            f"{weight:.3f} * {feature}"
            for weight, feature in zip(weights, chosen, strict=True)
        )
    else:
        cond_feature = rng.choice(FEATURES)
        threshold = 0.5 if cond_feature.endswith("_flag") else rng.uniform(0.15, 0.85)
        cond_expr = f"{cond_feature} > {threshold:.2f}"
        primary, secondary = rng.sample(list(FEATURES), 2)
        priority = f"{primary} + {_coefficient(rng):.2f} * {secondary}"
    return SymbolicRule(
        name=f"random_t{int(rule_type)}_{index:03d}",
        expression=priority,
        rule_type=rule_type,
        priority_expr=priority,
        cond_expr=cond_expr,
        filter_expr=filter_expr,
        feature_set="theory11_plus3",
    )


def generate_random_candidates(*, count: int = 120, seed: int = 20261023) -> list[SymbolicRule]:
    if count <= 0 or count % 4:
        raise ValueError("count must be a positive multiple of four")
    rng = random.Random(seed)
    per_type = count // 4
    rules: list[SymbolicRule] = []
    signatures: set[tuple] = set()
    for rule_type in RuleType:
        accepted = 0
        attempts = 0
        while accepted < per_type:
            attempts += 1
            if attempts > per_type * 100:
                raise RuntimeError("could not generate enough unique random rules")
            rule = _make_rule(rng, rule_type, accepted)
            signature = (
                int(rule.rule_type),
                rule.priority_expr,
                rule.cond_expr,
                rule.filter_expr,
            )
            if signature in signatures:
                continue
            signatures.add(signature)
            rules.append(rule)
            accepted += 1
    return rules


def write_candidates(path: Path, *, count: int, seed: int) -> Path:
    rules = generate_random_candidates(count=count, seed=seed)
    payload = {
        "schema_version": 1,
        "tag": "random_pool",
        "source": "random",
        "completed_rounds": 1,
        "candidate_total": len(rules),
        "parsed_total": len(rules),
        "reason_counts": {},
        "random_seed": seed,
        "feature_set": "theory11_plus3",
        "candidates": [
            {
                "rule": rule.to_dict(),
                "source": "random",
                "objective": 0.0,
                "iteration": 0,
                "total_tardiness": 0.0,
                "weighted_service_shortfall": 0.0,
            }
            for rule in rules
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    counts = Counter(int(rule.rule_type) for rule in rules)
    print(f"random candidates: {len(rules)}")
    print(f"type counts: {dict(sorted(counts.items()))}")
    print(f"seed: {seed}")
    print(f"output: {path.resolve()}")
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument(
        "--out", default="experiments/random_candidates_pool.json"
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    write_candidates(Path(args.out), count=args.count, seed=args.seed)


if __name__ == "__main__":
    main()
