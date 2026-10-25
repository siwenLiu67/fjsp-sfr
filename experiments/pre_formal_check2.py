"""Service-heavy fixed-rule evaluation for the second pre-formal check.

Simulation only: no PPO training and no LLM/API calls.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from statistics import fmean

import numpy as np

from fjsp_sfr.dispatch.classic_rules import build_classic_rule_library
from fjsp_sfr.dispatch.rule_library import RuleLibrary
from fjsp_sfr.experiments.build_gated_library import default_rule
from fjsp_sfr.experiments.eval_classic import evaluate_rule
from fjsp_sfr.experiments.validate_framework import build_rule_library
from fjsp_sfr.generation.scale_config import make_scaled_instance


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments" / "pre_formal_check2.md"
SEEDS = tuple(range(100, 150))
SCALES = ("S", "M", "L")


def all_rules() -> tuple[tuple[str, str, object], ...]:
    rows: list[tuple[str, str, object]] = [
        ("classic", rule.name, rule)
        for rule in build_classic_rule_library().rules
    ]
    rows.append(("default", "due_date_slack", default_rule()))
    rows.extend(
        ("handwritten", rule.name, rule) for rule in build_rule_library().rules
    )
    for label, filename in (
        ("qwen1_g", "llm_library_qwen1_g.json"),
        ("qwen2_g", "llm_library_qwen2_g.json"),
    ):
        library = RuleLibrary.from_json(str(ROOT / "experiments" / filename))
        rows.extend((label, rule.name, rule) for rule in library.rules)
    return tuple(rows)


def bootstrap_ci(values: list[float], key: str) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    samples = array[
        rng.integers(0, len(array), size=(20_000, len(array)))
    ].mean(axis=1)
    low, high = np.quantile(samples, (0.025, 0.975))
    return float(array.mean()), float(low), float(high)


def evaluate(difficulty: str, scales: tuple[str, ...]) -> dict:
    rules = all_rules()
    results = {}
    for scale in scales:
        print(f"Evaluating {difficulty}/{scale}", flush=True)
        block = {
            (library, name): {"F": [], "TT": [], "WSF": []}
            for library, name, _rule in rules
        }
        for count, seed in enumerate(SEEDS, 1):
            instance = make_scaled_instance(
                seed, scale=scale, difficulty=difficulty
            )
            for library, name, rule in rules:
                objective = evaluate_rule(rule, instance)
                block[(library, name)]["F"].append(float(objective.Z))
                block[(library, name)]["TT"].append(
                    float(objective.total_tardiness)
                )
                block[(library, name)]["WSF"].append(
                    float(objective.weighted_checkpoint_shortfall)
                )
            if count % 10 == 0:
                print(f"  {count}/{len(SEEDS)}", flush=True)
        results[scale] = block
    return results


def detailed_table(results: dict) -> list[str]:
    lines = [
        "| 规模 | 规则库 | 规则 | 平均 F | 平均 TT | 平均 WSF |",
        "|---|---|---|---:|---:|---:|",
    ]
    for scale in SCALES:
        for (library, name), values in results[scale].items():
            lines.append(
                f"| {scale} | {library} | {name} | "
                f"{fmean(values['F']):.2f} | {fmean(values['TT']):.2f} | "
                f"{fmean(values['WSF']):.2f} |"
            )
    return lines


def best_table(results: dict) -> list[str]:
    lines = [
        "| 规模 | 规则库 | 最优单规则 | 平均 F | ΔF=规则−SPT (95% CI) | ΔF=规则−ATC (95% CI) |",
        "|---|---|---|---:|---:|---:|",
    ]
    for scale in SCALES:
        block = results[scale]
        spt = block[("classic", "SPT")]["F"]
        atc = block[("classic", "ATC_K2")]["F"]
        for library in ("handwritten", "qwen1_g", "qwen2_g"):
            candidates = [
                (name, values)
                for (candidate_library, name), values in block.items()
                if candidate_library == library
            ]
            name, values = min(candidates, key=lambda item: fmean(item[1]["F"]))
            vs_spt = [a - b for a, b in zip(values["F"], spt, strict=True)]
            vs_atc = [a - b for a, b in zip(values["F"], atc, strict=True)]
            ds, ls, hs = bootstrap_ci(vs_spt, f"{scale}:{library}:spt")
            da, la, ha = bootstrap_ci(vs_atc, f"{scale}:{library}:atc")
            lines.append(
                f"| {scale} | {library} | {name} | {fmean(values['F']):.2f} | "
                f"{ds:+.2f} [{ls:+.2f}, {hs:+.2f}] | "
                f"{da:+.2f} [{la:+.2f}, {ha:+.2f}] |"
            )
    return lines


def balanced_m_table(results: dict) -> list[str]:
    block = results["M"]
    selected = [("classic", "SPT"), ("classic", "ATC_K2")]
    for library in ("qwen1_g", "qwen2_g"):
        selected.append(
            min(
                (
                    key
                    for key in block
                    if key[0] == library
                ),
                key=lambda key: fmean(block[key]["F"]),
            )
        )
    lines = [
        "| 规则库 | 规则 | 平均 F | 平均 TT | 平均 WSF | TT/F | WSF/F |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for library, name in selected:
        values = block[(library, name)]
        mean_f = fmean(values["F"])
        mean_tt = fmean(values["TT"])
        mean_wsf = fmean(values["WSF"])
        lines.append(
            f"| {library} | {name} | {mean_f:.2f} | {mean_tt:.2f} | "
            f"{mean_wsf:.2f} | {mean_tt / mean_f:.2%} | "
            f"{mean_wsf / mean_f:.2%} |"
        )
    return lines


def gap_comparison_table(service: dict, balanced: dict) -> list[str]:
    lines = [
        "| 规模 | Balanced 最优服务规则 | Balanced 相对 SPT | Service-heavy 最优服务规则 | Service-heavy 相对 SPT | 相对差距是否缩小 |",
        "|---|---|---:|---|---:|:---:|",
    ]
    for scale in SCALES:
        entries = []
        for result in (balanced, service):
            block = result[scale]
            spt = fmean(block[("classic", "SPT")]["F"])
            library, name, value = min(
                (
                    (library, name, fmean(values["F"]))
                    for (library, name), values in block.items()
                    if library in ("handwritten", "qwen1_g", "qwen2_g")
                ),
                key=lambda item: item[2],
            )
            entries.append((library, name, value - spt, (value - spt) / spt))
        old, new = entries
        lines.append(
            f"| {scale} | {old[0]}/{old[1]} | {old[2]:+.2f} "
            f"({old[3]:+.2%}) | {new[0]}/{new[1]} | {new[2]:+.2f} "
            f"({new[3]:+.2%}) | {'是' if new[3] < old[3] else '否'} |"
        )
    return lines


def calibration_table() -> list[str]:
    rows = json.loads(
        (ROOT / "experiments" / "difficulty_calibration.json").read_text(
            encoding="utf-8"
        )
    )
    lines = [
        "| 规模 | Best(SPT,ATC) 平均 F | TT/F | WSF/F | 35%–50% 目标 |",
        "|---|---:|---:|---:|:---:|",
    ]
    for row in rows:
        fraction = row["best_classic_WSF_fraction"]
        lines.append(
            f"| {row['scale']} | {row['best_classic_mean_F']:.2f} | "
            f"{row['best_classic_TT_fraction']:.2%} | {fraction:.2%} | "
            f"{'达标' if 0.35 <= fraction <= 0.50 else '未达标'} |"
        )
    return lines


def conclusions(service: dict, balanced: dict) -> list[str]:
    lines = []
    for scale in SCALES:
        block = service[scale]
        spt = fmean(block[("classic", "SPT")]["F"])
        best_service = min(
            (
                (library, name, fmean(values["F"]))
                for (library, name), values in block.items()
                if library in ("handwritten", "qwen1_g", "qwen2_g")
            ),
            key=lambda item: item[2],
        )
        lines.append(
            f"- service_heavy/{scale}: 三个库中最优服务规则为 "
            f"{best_service[0]}/{best_service[1]} (F={best_service[2]:.2f})，"
            f"相对 SPT 为 {best_service[2] - spt:+.2f}。"
        )
    if all(
        min(
            fmean(values["F"])
            for (library, _name), values in service[scale].items()
            if library in ("handwritten", "qwen1_g", "qwen2_g")
        )
        / fmean(service[scale][("classic", "SPT")]["F"])
        < min(
            fmean(values["F"])
            for (library, _name), values in balanced[scale].items()
            if library in ("handwritten", "qwen1_g", "qwen2_g")
        )
        / fmean(balanced[scale][("classic", "SPT")]["F"])
        for scale in SCALES
    ):
        lines.append("- Service-heavy 下服务规则相对 SPT 的百分比差距在 S、M、L 三个规模均缩小。")
    return lines


def main() -> None:
    service = evaluate("service_heavy", SCALES)
    balanced = evaluate("balanced", SCALES)
    lines = [
        "# 正式实验前核查 2",
        "",
        "## 结论",
        "",
        *conclusions(service, balanced),
        "",
        "## Service-heavy 难度校准",
        "",
        *calibration_table(),
        "",
        "## 服务成本加重前后：最优服务规则相对 SPT 的差距",
        "",
        *gap_comparison_table(service, balanced),
        "",
        "## Service-heavy：全部固定规则",
        "",
        *detailed_table(service),
        "",
        "## Service-heavy：各库最优单规则配对比较",
        "",
        *best_table(service),
        "",
        "## Balanced/M：SPT、ATC 与 Qwen 库最优单规则分解",
        "",
        *balanced_m_table(balanced),
        "",
    ]
    OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report written to {OUTPUT}")


if __name__ == "__main__":
    main()
