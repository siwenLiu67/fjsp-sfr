"""Phase-2 rule-library validation without PPO training or LLM calls."""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_PROJECT_PARENT = _PROJECT_ROOT.parent
for _import_path in (str(_PROJECT_ROOT), str(_PROJECT_PARENT)):
    if _import_path not in sys.path:
        sys.path.insert(0, _import_path)

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
class Result:
    F: float
    TT: float
    WSF: float


def _evaluate(rule: object, instances: list) -> tuple[Result, ...]:
    rows = []
    for instance in instances:
        objective = evaluate_rule(rule, instance)
        rows.append(
            Result(
                float(objective.Z),
                float(objective.total_tardiness),
                float(objective.weighted_checkpoint_shortfall),
            )
        )
    return tuple(rows)


def _rule_key(rule: object) -> str:
    payload = json.dumps(rule.to_dict(), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _bootstrap_ci(values: list[float], label: str) -> tuple[float, float]:
    seed = int.from_bytes(hashlib.sha256(label.encode()).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    data = np.asarray(values, dtype=float)
    samples = rng.choice(data, size=(5000, len(data)), replace=True).mean(axis=1)
    low, high = np.quantile(samples, (0.025, 0.975))
    return float(low), float(high)


def _mean(rows: tuple[Result, ...]) -> Result:
    return Result(
        fmean(row.F for row in rows),
        fmean(row.TT for row in rows),
        fmean(row.WSF for row in rows),
    )


def build_report(args: argparse.Namespace) -> Path:
    library_dir = Path(args.library_dir)
    output = Path(args.output)
    seeds = tuple(range(args.seed_start, args.seed_end + 1))
    lines = [
        "# Phase-2 pre-formal check",
        "",
        f"- difficulty: `{args.difficulty}`",
        f"- seeds: {args.seed_start}–{args.seed_end}",
        "- paired difference: library best fixed rule minus classic reference",
        "",
        "| Scale | Library | Best rule | F | TT | WSF | SPT F/TT/WSF | ATC F/TT/WSF | vs SPT (95% CI) | vs ATC (95% CI) |",
        "|---|---|---|---:|---:|---:|---|---|---:|---:|",
    ]
    for scale in args.scales:
        instances = [
            make_scaled_instance(seed, scale=scale, difficulty=args.difficulty)
            for seed in seeds
        ]
        classics = build_classic_rule_library()
        spt = next(rule for rule in classics.rules if rule.name == "SPT")
        atc = next(rule for rule in classics.rules if rule.name == "ATC_K2")
        cache: dict[str, tuple[Result, ...]] = {
            _rule_key(spt): _evaluate(spt, instances),
            _rule_key(atc): _evaluate(atc, instances),
        }
        spt_rows = cache[_rule_key(spt)]
        atc_rows = cache[_rule_key(atc)]
        spt_mean, atc_mean = _mean(spt_rows), _mean(atc_rows)
        for tag in args.tags:
            for suffix in ("g", "mix"):
                label = f"{tag}_{scale}_{suffix}"
                path = library_dir / f"llm_library_{label}.json"
                library = RuleLibrary.from_json(str(path))
                if not library.rules:
                    lines.append(
                        f"| {scale} | {label} | no rule passed gate | — | — | — | "
                        f"{spt_mean.F:.2f}/{spt_mean.TT:.2f}/{spt_mean.WSF:.2f} | "
                        f"{atc_mean.F:.2f}/{atc_mean.TT:.2f}/{atc_mean.WSF:.2f} | — | — |"
                    )
                    continue
                evaluated: list[tuple[object, tuple[Result, ...]]] = []
                for rule in library.rules:
                    key = _rule_key(rule)
                    if key not in cache:
                        cache[key] = _evaluate(rule, instances)
                    evaluated.append((rule, cache[key]))
                best_rule, best_rows = min(
                    evaluated, key=lambda item: fmean(row.F for row in item[1])
                )
                best_mean = _mean(best_rows)
                diff_spt = [
                    row.F - reference.F
                    for row, reference in zip(best_rows, spt_rows, strict=True)
                ]
                diff_atc = [
                    row.F - reference.F
                    for row, reference in zip(best_rows, atc_rows, strict=True)
                ]
                spt_low, spt_high = _bootstrap_ci(diff_spt, label + ":SPT")
                atc_low, atc_high = _bootstrap_ci(diff_atc, label + ":ATC")
                lines.append(
                    f"| {scale} | {label} | {best_rule.name} | "
                    f"{best_mean.F:.2f} | {best_mean.TT:.2f} | {best_mean.WSF:.2f} | "
                    f"{spt_mean.F:.2f}/{spt_mean.TT:.2f}/{spt_mean.WSF:.2f} | "
                    f"{atc_mean.F:.2f}/{atc_mean.TT:.2f}/{atc_mean.WSF:.2f} | "
                    f"{fmean(diff_spt):+.2f} [{spt_low:+.2f}, {spt_high:+.2f}] | "
                    f"{fmean(diff_atc):+.2f} [{atc_low:+.2f}, {atc_high:+.2f}] |"
                )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tags", nargs="+", default=("p3b1", "p3b2", "p3b3"))
    parser.add_argument("--scales", nargs="+", choices=("S", "M", "L"), default=("S", "M", "L"))
    parser.add_argument("--difficulty", default="service_heavy")
    parser.add_argument("--seed_start", type=int, default=100)
    parser.add_argument("--seed_end", type=int, default=149)
    parser.add_argument("--library_dir", default="experiments")
    parser.add_argument("--output", default="experiments/pre_formal_check3.md")
    return parser.parse_args(argv)


def main() -> None:
    path = build_report(parse_args())
    print(path.resolve())


if __name__ == "__main__":
    main()
