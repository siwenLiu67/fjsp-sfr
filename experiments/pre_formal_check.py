"""Pre-formal fixed-rule evaluation on the validation seed split.

This script performs simulation only: it neither trains a policy nor calls an
LLM.  It writes the complete balanced/hard comparison to
``experiments/pre_formal_check.md``.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, fields
from pathlib import Path
from statistics import fmean

import numpy as np

from fjsp_sfr.dispatch.classic_rules import build_classic_rule_library
from fjsp_sfr.dispatch.rule_library import RuleLibrary
from fjsp_sfr.experiments.build_gated_library import default_rule
from fjsp_sfr.experiments.eval_classic import evaluate_rule
from fjsp_sfr.experiments.validate_framework import (
    build_rule_library,
    make_fjsp_instance,
)
from fjsp_sfr.generation.instance_generator import InstanceConfig
from fjsp_sfr.generation.scale_config import (
    make_scaled_instance,
    scaled_instance_config,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments" / "pre_formal_check.md"
SEEDS = tuple(range(100, 150))
SCALES = ("S", "M", "L")
DIFFICULTIES = ("balanced", "hard")


def _rules() -> tuple[tuple[str, str, object], ...]:
    classic = build_classic_rule_library()
    rows: list[tuple[str, str, object]] = [
        ("classic", rule.name, rule) for rule in classic.rules
    ]
    rows.append(("default", "due_date_slack", default_rule()))
    rows.extend(
        ("handwritten", rule.name, rule) for rule in build_rule_library().rules
    )
    for library_name, filename in (
        ("qwen1_g", "llm_library_qwen1_g.json"),
        ("qwen2_g", "llm_library_qwen2_g.json"),
    ):
        library = RuleLibrary.from_json(str(ROOT / "experiments" / filename))
        rows.extend((library_name, rule.name, rule) for rule in library.rules)
    return tuple(rows)


def _bootstrap_mean_ci(
    differences: tuple[float, ...], *, key: str, resamples: int = 20_000
) -> tuple[float, float, float]:
    values = np.asarray(differences, dtype=float)
    mean = float(values.mean())
    if not np.any(values):
        return mean, 0.0, 0.0
    seed = int.from_bytes(
        hashlib.sha256(key.encode("utf-8")).digest()[:8], "big"
    )
    rng = np.random.default_rng(seed)
    sampled = values[rng.integers(0, len(values), (resamples, len(values)))]
    means = sampled.mean(axis=1)
    low, high = np.quantile(means, (0.025, 0.975))
    return mean, float(low), float(high)


def _fmt(value: float) -> str:
    return f"{value:.2f}"


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _config_differences() -> list[tuple[str, object, object]]:
    old = InstanceConfig(
        num_instances=1,
        num_jobs=15,
        num_machines=5,
        num_entities=2,
        ops_per_job=(2, 4),
        checkpoints_per_entity=(2, 2),
        checkpoint_tightness=1.0,
        checkpoint_profile_pool=("balanced_2", "early_pressure_2"),
        eligible_machines_range=(1, 3),
        group_name="framework_validation",
    )
    new = scaled_instance_config("M", "balanced")
    differences = []
    for field in fields(InstanceConfig):
        old_value = getattr(old, field.name)
        new_value = getattr(new, field.name)
        if old_value != new_value:
            differences.append((field.name, old_value, new_value))
    return differences


def _forecast_signature(instance) -> tuple:
    provider = instance.forecast_provider
    if provider is None:
        return ()
    return (
        provider.update_interval,
        tuple(
            (
                snapshot.update_epoch,
                tuple(
                    (
                        entity_id,
                        tuple(
                            (bucket.start, bucket.end, bucket.d_hat, bucket.sigma)
                            for bucket in buckets
                        ),
                    )
                    for entity_id, buckets in sorted(
                        snapshot.buckets_by_entity.items()
                    )
                ),
            )
            for snapshot in provider._snapshots
        ),
    )


def _instance_signature(instance) -> tuple:
    metadata = dict(instance.metadata)
    metadata.pop("group_name", None)
    return (
        tuple(asdict(job) for job in instance.jobs),
        tuple(asdict(entity) for entity in instance.entities),
        tuple(asdict(machine) for machine in instance.machines),
        instance.alpha,
        instance.beta,
        instance.forecast_risk_kappa,
        instance.r5_epsilon,
        metadata,
        _forecast_signature(instance),
    )


def _evaluate() -> dict:
    rules = _rules()
    all_results: dict = {}
    for difficulty in DIFFICULTIES:
        for scale in SCALES:
            print(f"Evaluating difficulty={difficulty}, scale={scale} ...", flush=True)
            by_rule = {
                (library, name): {"F": [], "TT": [], "WSF": []}
                for library, name, _rule in rules
            }
            for position, seed in enumerate(SEEDS, start=1):
                instance = make_scaled_instance(
                    seed, scale=scale, difficulty=difficulty
                )
                for library, name, rule in rules:
                    objective = evaluate_rule(rule, instance)
                    values = by_rule[(library, name)]
                    values["F"].append(float(objective.Z))
                    values["TT"].append(float(objective.total_tardiness))
                    values["WSF"].append(
                        float(objective.weighted_checkpoint_shortfall)
                    )
                if position % 10 == 0:
                    print(
                        f"  {position}/{len(SEEDS)} instances complete",
                        flush=True,
                    )
            all_results[(difficulty, scale)] = by_rule
    return all_results


def _detailed_table(results: dict, difficulty: str) -> list[str]:
    lines = [
        "| 规模 | 规则库 | 规则 | 平均 F | 平均 TT | 平均 WSF | 配对差 ΔF=规则−SPT | 95% CI |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for scale in SCALES:
        block = results[(difficulty, scale)]
        spt = tuple(block[("classic", "SPT")]["F"])
        for rule_index, ((library, name), values) in enumerate(block.items()):
            differences = tuple(
                value - baseline
                for value, baseline in zip(values["F"], spt, strict=True)
            )
            mean, low, high = _bootstrap_mean_ci(
                differences,
                key=f"{difficulty}:{scale}:{library}:{name}:{rule_index}",
            )
            lines.append(
                "| {} | {} | {} | {} | {} | {} | {} | [{}, {}] |".format(
                    scale,
                    library,
                    name,
                    _fmt(fmean(values["F"])),
                    _fmt(fmean(values["TT"])),
                    _fmt(fmean(values["WSF"])),
                    _fmt(mean),
                    _fmt(low),
                    _fmt(high),
                )
            )
    return lines


def _best_library_table(results: dict, difficulty: str) -> list[str]:
    lines = [
        "| 规模 | 规则库 | 库内最优单规则 | 平均 F | 较 SPT 差值 | 较 SPT | 较 ATC 差值 | 较 ATC |",
        "|---|---|---|---:|---:|:---:|---:|:---:|",
    ]
    library_names = ("handwritten", "qwen1_g", "qwen2_g")
    for scale in SCALES:
        block = results[(difficulty, scale)]
        spt = fmean(block[("classic", "SPT")]["F"])
        atc = fmean(block[("classic", "ATC_K2")]["F"])
        for library in library_names:
            candidates = [
                (name, fmean(values["F"]))
                for (candidate_library, name), values in block.items()
                if candidate_library == library
            ]
            best_name, best_value = min(candidates, key=lambda item: item[1])
            delta_spt = best_value - spt
            delta_atc = best_value - atc
            lines.append(
                f"| {scale} | {library} | {best_name} | {_fmt(best_value)} | "
                f"{_fmt(delta_spt)} ({delta_spt / spt * 100:+.2f}%) | "
                f"{'\u662f' if delta_spt < 0 else '\u5426'} | "
                f"{_fmt(delta_atc)} ({delta_atc / atc * 100:+.2f}%) | "
                f"{'\u662f' if delta_atc < 0 else '\u5426'} |"
            )
    return lines


def _hard_default_table(results: dict) -> list[str]:
    lines = [
        "| 规模 | 缺省排序平均 F | 平均 TT | TT/F | 平均 WSF | WSF/F |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for scale in SCALES:
        values = results[("hard", scale)][("default", "due_date_slack")]
        mean_f = fmean(values["F"])
        mean_tt = fmean(values["TT"])
        mean_wsf = fmean(values["WSF"])
        lines.append(
            f"| {scale} | {_fmt(mean_f)} | {_fmt(mean_tt)} | "
            f"{_pct(mean_tt / mean_f)} | {_fmt(mean_wsf)} | "
            f"{_pct(mean_wsf / mean_f)} |"
        )
    return lines


def _compatibility_table() -> tuple[list[str], int]:
    identical = 0
    for seed in SEEDS:
        old = make_fjsp_instance(
            seed, n_jobs=15, n_machines=5, n_entities=2
        )
        new = make_scaled_instance(seed, scale="M", difficulty="balanced")
        if _instance_signature(old) == _instance_signature(new):
            identical += 1
    lines = [
        "| 检查项 | 结果 |",
        "|---|---|",
        f"| 种子 100–149 中完全一致的实例数 | {identical}/50 |",
    ]
    for name, old, new in _config_differences():
        lines.append(f"| `{name}` | 旧默认 `{old}` → M/balanced `{new}` |")
    return lines, identical


def _conclusions(results: dict, identical: int) -> list[str]:
    lines = []
    for difficulty in DIFFICULTIES:
        for scale in SCALES:
            block = results[(difficulty, scale)]
            spt = fmean(block[("classic", "SPT")]["F"])
            atc = fmean(block[("classic", "ATC_K2")]["F"])
            winners = []
            for library in ("handwritten", "qwen1_g", "qwen2_g"):
                name, value = min(
                    (
                        (name, fmean(values["F"]))
                        for (candidate_library, name), values in block.items()
                        if candidate_library == library
                    ),
                    key=lambda item: item[1],
                )
                winners.append(
                    f"{library}/{name}={value:.2f}"
                    f" (vs SPT {value - spt:+.2f}; vs ATC {value - atc:+.2f})"
                )
            lines.append(f"- {difficulty}/{scale}: " + "；".join(winners) + "。")
    lines.append(
        f"- M/balanced 与旧 15-job 默认生成器完全一致的实例为 "
        f"{identical}/50，因此结论为{'\u4e00致' if identical == 50 else '\u4e0d一致'}。"
    )
    return lines


def main() -> None:
    results = _evaluate()
    compatibility, identical = _compatibility_table()
    sections = [
        "# 正式实验前固定规则核查",
        "",
        "## 结论",
        "",
        *_conclusions(results, identical),
        "",
        "## Balanced：全部规则",
        "",
        *_detailed_table(results, "balanced"),
        "",
        "## Balanced：各库最优单规则",
        "",
        *_best_library_table(results, "balanced"),
        "",
        "## Hard：全部规则",
        "",
        *_detailed_table(results, "hard"),
        "",
        "## Hard：各库最优单规则",
        "",
        *_best_library_table(results, "hard"),
        "",
        "## Hard：缺省排序的目标构成",
        "",
        *_hard_default_table(results),
        "",
        "## M/balanced 与旧默认生成器一致性",
        "",
        *compatibility,
        "",
    ]
    OUTPUT.write_text("\n".join(sections), encoding="utf-8")
    print(f"Report written to {OUTPUT}")


if __name__ == "__main__":
    main()
