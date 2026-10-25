"""Library-level validation for pooled, random, and mixed rule libraries."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean

import numpy as np

from fjsp_sfr.dispatch.classic_rules import build_classic_rule_library
from fjsp_sfr.dispatch.rule_library import RuleLibrary
from fjsp_sfr.experiments.eval_classic import evaluate_rule
from fjsp_sfr.generation.scale_config import make_scaled_instance


@dataclass(frozen=True)
class Metrics:
    F: float
    TT: float
    WSF: float


def _key(rule: object) -> str:
    value = json.dumps(rule.to_dict(), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _evaluate(rule: object, instances: list) -> tuple[Metrics, ...]:
    values = []
    for instance in instances:
        objective = evaluate_rule(rule, instance)
        values.append(
            Metrics(
                float(objective.Z),
                float(objective.total_tardiness),
                float(objective.weighted_checkpoint_shortfall),
            )
        )
    return tuple(values)


def _mean(values: tuple[Metrics, ...]) -> Metrics:
    return Metrics(
        fmean(item.F for item in values),
        fmean(item.TT for item in values),
        fmean(item.WSF for item in values),
    )


def _paired(values: list[float], label: str) -> str:
    data = np.asarray(values, dtype=float)
    seed = int.from_bytes(hashlib.sha256(label.encode()).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    samples = rng.choice(data, size=(5000, len(data)), replace=True).mean(axis=1)
    low, high = np.quantile(samples, (0.025, 0.975))
    return f"{data.mean():+.2f} [{low:+.2f}, {high:+.2f}]"


def _library_specs(root: Path) -> list[tuple[str, str, Path]]:
    specs = []
    for scale in ("S", "M", "L"):
        for batch in ("p3b1", "p3b2", "p3b3"):
            name = f"{batch}_{scale}_mix"
            specs.append((scale, name, root / f"llm_library_{name}.json"))
        for family in ("pool", "random_pool"):
            for variant in ("g", "mix"):
                name = f"{family}_{scale}_{variant}"
                specs.append((scale, name, root / f"llm_library_{name}.json"))
    return specs


def build_report(
    *, root: Path, output: Path, seed_start: int = 100, seed_end: int = 149
) -> Path:
    lines = [
        "# Pre-formal check 4: library-level comparison",
        "",
        f"- difficulty: `service_heavy`",
        f"- validation seeds: {seed_start}–{seed_end}",
        "- paired differences are library value minus SPT value; lower is better.",
        "",
        "| Scale | Library | Rules (source; reason) | Best fixed rule | F | TT | WSF | Oracle F | Complementarity | Best vs SPT (95% CI) | Oracle vs SPT (95% CI) |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    seeds = tuple(range(seed_start, seed_end + 1))
    for scale in ("S", "M", "L"):
        instances = [
            make_scaled_instance(seed, scale=scale, difficulty="service_heavy")
            for seed in seeds
        ]
        spt = next(
            rule
            for rule in build_classic_rule_library().rules
            if rule.name == "SPT"
        )
        cache: dict[str, tuple[Metrics, ...]] = {_key(spt): _evaluate(spt, instances)}
        spt_values = cache[_key(spt)]
        for spec_scale, name, path in _library_specs(root):
            if spec_scale != scale:
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            library = RuleLibrary.from_json(str(path))
            rule_descriptions = [
                f"{item['name']} ({item.get('source', 'unknown')}; "
                f"{item.get('selection_reason', 'unspecified')})"
                for item in payload["rules"]
            ]
            if not library.rules:
                lines.append(
                    f"| {scale} | {name} | no rule passed | — | — | — | — | — | — | — | — |"
                )
                continue
            evaluated: list[tuple[object, tuple[Metrics, ...]]] = []
            for rule in library.rules:
                key = _key(rule)
                if key not in cache:
                    cache[key] = _evaluate(rule, instances)
                evaluated.append((rule, cache[key]))
            best_rule, best_values = min(
                evaluated, key=lambda pair: fmean(item.F for item in pair[1])
            )
            best = _mean(best_values)
            oracle_values = [
                min(values[index].F for _rule, values in evaluated)
                for index in range(len(instances))
            ]
            oracle = fmean(oracle_values)
            complementarity = (best.F - oracle) / best.F * 100.0
            best_diff = [
                value.F - reference.F
                for value, reference in zip(best_values, spt_values, strict=True)
            ]
            oracle_diff = [
                value - reference.F
                for value, reference in zip(oracle_values, spt_values, strict=True)
            ]
            lines.append(
                f"| {scale} | {name} | {'<br>'.join(rule_descriptions)} | "
                f"{best_rule.name} | {best.F:.2f} | {best.TT:.2f} | {best.WSF:.2f} | "
                f"{oracle:.2f} | {complementarity:.2f}% | "
                f"{_paired(best_diff, name + ':best')} | "
                f"{_paired(oracle_diff, name + ':oracle')} |"
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="experiments")
    parser.add_argument("--output", default="experiments/pre_formal_check4.md")
    parser.add_argument("--seed_start", type=int, default=100)
    parser.add_argument("--seed_end", type=int, default=149)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    path = build_report(
        root=Path(args.root),
        output=Path(args.output),
        seed_start=args.seed_start,
        seed_end=args.seed_end,
    )
    print(path.resolve())


if __name__ == "__main__":
    main()
