"""Pre-formal center check for classic, handwritten and cached LLM rules."""

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
import tempfile
from pathlib import Path
from statistics import fmean

from fjsp_sfr.dispatch.symbolic_rule import SymbolicRule
from fjsp_sfr.experiments.scenario_tools import bootstrap_ci, classic_rules_with_default, evaluate_rule
from fjsp_sfr.experiments.validate_framework import build_rule_library
from fjsp_sfr.generation.scenario_generator import make_scenario_instance, seed_values


DEFAULT_CANDIDATES = tuple(Path(f"experiments/llm_candidates_p3b{index}.json") for index in (1, 2, 3))


def load_qualified_candidates(paths: tuple[Path, ...]) -> tuple[SymbolicRule, ...]:
    rules, signatures = [], set()
    for path in paths:
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for item in payload.get("candidates", []):
            if not item.get("rule") or item.get("filtered") or item.get("failure_reason"):
                continue
            rule = SymbolicRule.from_dict(item["rule"])
            signature = (rule.priority_expr, rule.cond_expr, rule.filter_expr, rule.feature_set)
            if signature not in signatures:
                signatures.add(signature)
                rules.append(rule)
    return tuple(rules)


def build_report(
    candidate_paths: tuple[Path, ...],
    output: Path,
    *,
    smoke: bool = False,
) -> None:
    seeds = seed_values("validation_center")[:3] if smoke else seed_values("validation_center")
    instances = [make_scenario_instance("center", seed) for seed in seeds]
    classics = classic_rules_with_default()
    handwritten = build_rule_library().rules
    llm = load_qualified_candidates(candidate_paths)
    if smoke:
        llm = llm[:3]
    all_rules = (*classics, *handwritten, *llm)
    evaluations = {
        rule.name: [evaluate_rule(rule, instance) for instance in instances]
        for rule in all_rules
    }
    spt = [row["F"] for row in evaluations["SPT"]]
    atc = [row["F"] for row in evaluations["ATC_K2"]]
    rows = []
    for rule in all_rules:
        values = evaluations[rule.name]
        f_values = [row["F"] for row in values]
        spt_diff = [value - reference for value, reference in zip(f_values, spt, strict=True)]
        atc_diff = [value - reference for value, reference in zip(f_values, atc, strict=True)]
        rows.append({
            "name": rule.name,
            "source": "classic" if rule in classics else "handwritten" if rule in handwritten else "llm",
            "F": fmean(f_values), "TT": fmean(row["TT"] for row in values), "WSF": fmean(row["WSF"] for row in values),
            "spt_diff": fmean(spt_diff), "spt_ci": bootstrap_ci(spt_diff),
            "atc_diff": fmean(atc_diff), "atc_ci": bootstrap_ci(atc_diff),
        })
    rows.sort(key=lambda row: (row["F"], row["name"]))
    lines = ["# Center pre-check", "", f"Qualified LLM candidates: {len(llm)}", "", "| Rank | Rule | Source | F | TT | WSF | ΔSPT [95% CI] | ΔATC [95% CI] |", "|---:|---|---|---:|---:|---:|---:|---:|"]
    for index, row in enumerate(rows[:10], 1):
        lines.append(f"| {index} | {row['name']} | {row['source']} | {row['F']:.3f} | {row['TT']:.3f} | {row['WSF']:.3f} | {row['spt_diff']:.3f} [{row['spt_ci'][0]:.3f}, {row['spt_ci'][1]:.3f}] | {row['atc_diff']:.3f} [{row['atc_ci'][0]:.3f}, {row['atc_ci'][1]:.3f}] |")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", nargs="*", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--out", type=Path, default=Path("experiments/pre_check_center.md"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        with tempfile.TemporaryDirectory(prefix="fjsp-pre-check-smoke-") as raw_dir:
            output = Path(raw_dir) / "pre_check_center.md"
            build_report(tuple(args.candidates), output, smoke=True)
            print(f"smoke output: {output}")
        return
    build_report(tuple(args.candidates), args.out)


if __name__ == "__main__":
    main()
