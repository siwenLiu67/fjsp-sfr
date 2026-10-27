"""Calibrate scenario checkpoint severity without changing objective weights."""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_PROJECT_PARENT = _PROJECT_ROOT.parent
for _import_path in (str(_PROJECT_ROOT), str(_PROJECT_PARENT)):
    if _import_path not in sys.path:
        sys.path.insert(0, _import_path)

import argparse
import copy
import hashlib
import json
import math
import os
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable

from fjsp_sfr.experiments.scenario_tools import (
    classic_rules_with_default,
    evaluate_rule,
)
from fjsp_sfr.generation.scenario_generator import (
    CONFIG_PATH,
    scenario_names,
    load_scenario_config,
    make_scenario_instance,
    seed_values,
)


TARGET_LOW = 0.35
TARGET_HIGH = 0.50
TARGET_MID = (TARGET_LOW + TARGET_HIGH) / 2.0
MIN_WSF_RATIO = 1.10
TIGHTNESS_RANGE = (0.30, 1.00)
COMMITMENT_RANGE = (1.00, 2.00)
WEIGHT_SCALE_RANGE = (1.00, 20.00)
TIGHTNESS_GRID = (0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00)
COMMITMENT_GRID = (1.00, 1.20, 1.40, 1.60, 1.80, 2.00)
RULE_ORDER = ("SPT", "EDD", "ATC_K2", "MIN_SLACK", "default_due_date_slack")

DEFAULT_REPORT = Path("experiments/scenario_calibration.md")
DEFAULT_FAILED = Path("experiments/scenario_calibration_failed.json")
DEFAULT_CACHE = Path("experiments/scenario_calibration_cache.json")


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _normalise_rule_name(name: str) -> str:
    return {
        "ATC": "ATC_K2",
        "DEFAULT": "default_due_date_slack",
        "default": "default_due_date_slack",
    }.get(name, name)


def _cache_key(
    scenario: str,
    seed: int,
    rule_name: str,
    tightness: float,
    commitment: float,
    weight_scale: float,
) -> str:
    raw = (
        f"v2|{scenario}|{seed}|{_normalise_rule_name(rule_name)}|"
        f"{tightness:.8g}|{commitment:.8g}|{weight_scale:.8g}"
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _trial_config(
    base: dict[str, Any],
    tightness: float,
    commitment: float,
    weight_scale: float,
) -> dict[str, Any]:
    trial = copy.deepcopy(base)
    checkpoint = trial["checkpoint"]
    checkpoint["tightness"] = float(tightness)
    checkpoint["commitment_ratio"] = float(commitment)
    checkpoint["weight_scale"] = float(weight_scale)
    return trial


def _validated_metrics(result: dict[str, float]) -> dict[str, float]:
    metrics = {name: float(result[name]) for name in ("F", "TT", "WSF")}
    if not math.isclose(
        metrics["F"], metrics["TT"] + metrics["WSF"], rel_tol=1e-9, abs_tol=1e-9
    ):
        raise ValueError(
            "objective decomposition mismatch: "
            f"F={metrics['F']}, TT={metrics['TT']}, WSF={metrics['WSF']}"
        )
    return metrics


def _evaluate_one(payload: dict[str, Any]) -> tuple[str, dict[str, float]]:
    """Worker entry point; a private config file avoids shared mutable state."""
    rules = {
        _normalise_rule_name(rule.name): rule for rule in classic_rules_with_default()
    }
    rule = rules[_normalise_rule_name(payload["rule_name"])]
    trial_config = _trial_config(
        payload["base_config"],
        payload["tightness"],
        payload["commitment"],
        payload["weight_scale"],
    )
    fd, raw_path = tempfile.mkstemp(prefix="fjsp-scenario-", suffix=".json")
    config_path = Path(raw_path)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(trial_config, handle, ensure_ascii=False)
        instance = make_scenario_instance(
            payload["scenario"],
            seed=int(payload["seed"]),
            config_path=config_path,
        )
        metrics = _validated_metrics(evaluate_rule(rule, instance))
        return payload["cache_key"], metrics
    finally:
        config_path.unlink(missing_ok=True)


class EvaluationCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        if path.exists():
            loaded = json.loads(path.read_text(encoding="utf-8"))
            self.values: dict[str, dict[str, float]] = dict(loaded.get("values", loaded))
        else:
            self.values = {}

    def save(self) -> None:
        _json_dump(self.path, {"version": 2, "values": self.values})

    def evaluate(
        self,
        *,
        base_config: dict[str, Any],
        scenario: str,
        seeds: Iterable[int],
        tightness: float,
        commitment: float,
        weight_scale: float,
        jobs: int,
    ) -> dict[str, dict[str, float]]:
        rules = list(classic_rules_with_default())
        payloads: list[dict[str, Any]] = []
        keys: dict[tuple[str, int], str] = {}
        seed_list = tuple(int(seed) for seed in seeds)
        for seed in seed_list:
            for rule in rules:
                rule_name = _normalise_rule_name(rule.name)
                key = _cache_key(
                    scenario, seed, rule_name, tightness, commitment, weight_scale
                )
                keys[(rule_name, seed)] = key
                if key not in self.values:
                    payloads.append(
                        {
                            "cache_key": key,
                            "base_config": base_config,
                            "scenario": scenario,
                            "seed": seed,
                            "rule_name": rule_name,
                            "tightness": tightness,
                            "commitment": commitment,
                            "weight_scale": weight_scale,
                        }
                    )

        if jobs <= 1:
            evaluated = map(_evaluate_one, payloads)
        else:
            with ProcessPoolExecutor(max_workers=jobs) as executor:
                evaluated = executor.map(_evaluate_one, payloads)
        for key, metrics in evaluated:
            self.values[key] = metrics
        if payloads:
            self.save()

        table: dict[str, dict[str, float]] = {}
        for rule in rules:
            rule_name = _normalise_rule_name(rule.name)
            rows = [self.values[keys[(rule_name, seed)]] for seed in seed_list]
            table[rule_name] = {
                metric: fmean(float(row[metric]) for row in rows)
                for metric in ("F", "TT", "WSF")
            }
        return table


def _baseline_metrics(table: dict[str, dict[str, float]]) -> tuple[str, float]:
    name = min(("SPT", "ATC_K2"), key=lambda item: table[item]["F"])
    result = table[name]
    share = result["WSF"] / result["F"] if result["F"] > 0.0 else 0.0
    return name, share


def _wsf_ratio(table: dict[str, dict[str, float]]) -> float:
    values = [max(0.0, table[name]["WSF"]) for name in RULE_ORDER]
    minimum, maximum = min(values), max(values)
    if minimum <= 0.0:
        return math.inf if maximum > 0.0 else 1.0
    return maximum / minimum


def _distance_to_target(share: float) -> float:
    if share < TARGET_LOW:
        return TARGET_LOW - share
    if share > TARGET_HIGH:
        return share - TARGET_HIGH
    return 0.0


def _estimated_weight_scale(table: dict[str, dict[str, float]]) -> float:
    """Exploit the near-linear WSF response to a global checkpoint weight."""
    baseline, _ = _baseline_metrics(table)
    tt = max(0.0, table[baseline]["TT"])
    wsf = max(0.0, table[baseline]["WSF"])
    if wsf <= 0.0:
        return WEIGHT_SCALE_RANGE[1]
    estimate = TARGET_MID * tt / ((1.0 - TARGET_MID) * wsf)
    return min(WEIGHT_SCALE_RANGE[1], max(WEIGHT_SCALE_RANGE[0], estimate))


def _ordered_grid(current: float, values: tuple[float, ...], lower_first: bool) -> list[float]:
    candidates = [min(max(current, min(values)), max(values))]
    below = sorted((value for value in values if value < current), reverse=True)
    above = sorted(value for value in values if value > current)
    candidates.extend(below if lower_first else above)
    candidates.extend(above if lower_first else below)
    unique: list[float] = []
    for value in candidates:
        rounded = round(float(value), 8)
        if rounded not in unique:
            unique.append(rounded)
    return unique


def _trace_row(
    tightness: float,
    commitment: float,
    weight_scale: float,
    table: dict[str, dict[str, float]],
) -> dict[str, Any]:
    baseline, share = _baseline_metrics(table)
    return {
        "tightness": tightness,
        "commitment": commitment,
        "weight_scale": weight_scale,
        "baseline": baseline,
        "wsf_share": share,
        "wsf_rule_ratio": _wsf_ratio(table),
        "rules": table,
    }


def _evaluate_combination(
    *,
    cache: EvaluationCache,
    base_config: dict[str, Any],
    scenario: str,
    seeds: tuple[int, ...],
    tightness: float,
    commitment: float,
    weight_scale: float,
    jobs: int,
    started_at: float,
) -> dict[str, Any]:
    table = cache.evaluate(
        base_config=base_config,
        scenario=scenario,
        seeds=seeds,
        tightness=tightness,
        commitment=commitment,
        weight_scale=weight_scale,
        jobs=jobs,
    )
    row = _trace_row(tightness, commitment, weight_scale, table)
    print(
        "calibration "
        f"tightness={tightness:.3f} commitment={commitment:.3f} "
        f"weight_scale={weight_scale:.4f} elapsed={time.monotonic() - started_at:.1f}s "
        f"WSF/F={row['wsf_share'] * 100:.2f}% "
        f"WSF_max/min={row['wsf_rule_ratio']:.3f}",
        flush=True,
    )
    return row


def _is_success(row: dict[str, Any]) -> bool:
    return (
        TARGET_LOW <= row["wsf_share"] <= TARGET_HIGH
        and row["wsf_rule_ratio"] >= MIN_WSF_RATIO
    )


def _selection_key(row: dict[str, Any]) -> tuple[bool, float, float]:
    differentiation_penalty = max(0.0, MIN_WSF_RATIO - row["wsf_rule_ratio"])
    return (
        differentiation_penalty > 0.0,
        _distance_to_target(row["wsf_share"]),
        differentiation_penalty,
    )


def calibrate(
    config_path: Path,
    *,
    jobs: int = 1,
    cache_path: Path = DEFAULT_CACHE,
    failed_path: Path = DEFAULT_FAILED,
    smoke: bool = False,
    write_config: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    base_config = load_scenario_config(config_path)
    checkpoint = base_config["checkpoint"]
    current_tightness = float(checkpoint["tightness"])
    current_commitment = float(checkpoint["commitment_ratio"])
    current_weight = float(checkpoint.get("weight_scale", 1.0))
    calibration_seeds = tuple(seed_values("calibration_center"))
    if smoke:
        calibration_seeds = calibration_seeds[:2]
    cache = EvaluationCache(cache_path)
    started_at = time.monotonic()
    trajectory: list[dict[str, Any]] = []
    success: dict[str, Any] | None = None

    if smoke:
        tightness_values = list(TIGHTNESS_GRID[:2])
        commitment_values = list(COMMITMENT_GRID[:2])
    else:
        tightness_values = _ordered_grid(current_tightness, TIGHTNESS_GRID, lower_first=True)
        commitment_values = _ordered_grid(current_commitment, COMMITMENT_GRID, lower_first=True)
    for tightness in tightness_values:
        for commitment in commitment_values:
            first_weight = (
                WEIGHT_SCALE_RANGE[0]
                if smoke
                else min(WEIGHT_SCALE_RANGE[1], max(WEIGHT_SCALE_RANGE[0], current_weight))
            )
            first = _evaluate_combination(
                cache=cache,
                base_config=base_config,
                scenario="center",
                seeds=calibration_seeds,
                tightness=tightness,
                commitment=commitment,
                weight_scale=first_weight,
                jobs=jobs,
                started_at=started_at,
            )
            trajectory.append(first)
            if _is_success(first):
                success = first
                break

            estimate = _estimated_weight_scale(first["rules"]) * first_weight
            estimate = min(WEIGHT_SCALE_RANGE[1], max(WEIGHT_SCALE_RANGE[0], estimate))
            seen = {round(first_weight, 8)}
            weight_candidates = (
                (WEIGHT_SCALE_RANGE[1],)
                if smoke
                else (estimate, WEIGHT_SCALE_RANGE[1], WEIGHT_SCALE_RANGE[0])
            )
            for weight_scale in weight_candidates:
                weight_scale = round(float(weight_scale), 6)
                if round(weight_scale, 8) in seen:
                    continue
                seen.add(round(weight_scale, 8))
                row = _evaluate_combination(
                    cache=cache,
                    base_config=base_config,
                    scenario="center",
                    seeds=calibration_seeds,
                    tightness=tightness,
                    commitment=commitment,
                    weight_scale=weight_scale,
                    jobs=jobs,
                    started_at=started_at,
                )
                trajectory.append(row)
                if _is_success(row):
                    success = row
                    break
            if success is not None:
                break
            # Values below the current commitment are deliberately tried first:
            # near-identical WSF indicates an unreachable/saturated pledge.
        if success is not None:
            break

    closest = min(trajectory, key=_selection_key)
    selected = success or closest
    if success is not None:
        status = "success"
        if write_config:
            checkpoint["tightness"] = selected["tightness"]
            checkpoint["commitment_ratio"] = selected["commitment"]
            checkpoint["weight_scale"] = selected["weight_scale"]
            _json_dump(config_path, base_config)
            failed_path.unlink(missing_ok=True)
    else:
        status = "failed"
        _json_dump(
            failed_path,
            {
                "status": status,
                "target_wsf_share": [TARGET_LOW, TARGET_HIGH],
                "minimum_wsf_rule_ratio": MIN_WSF_RATIO,
                "search_ranges": {
                    "tightness": list(TIGHTNESS_RANGE),
                    "commitment_ratio": list(COMMITMENT_RANGE),
                    "weight_scale": list(WEIGHT_SCALE_RANGE),
                },
                "closest": closest,
                "attempts": trajectory,
            },
        )

    overview: list[dict[str, Any]] = []
    overview_seeds = tuple(seed_values("calibration_overview"))
    if smoke:
        overview_seeds = overview_seeds[:1]
    effective = selected if success is not None else {
        "tightness": current_tightness,
        "commitment": current_commitment,
        "weight_scale": current_weight,
    }
    for scenario in scenario_names():
        table = cache.evaluate(
            base_config=base_config,
            scenario=scenario,
            seeds=overview_seeds,
            tightness=float(effective["tightness"]),
            commitment=float(effective["commitment"]),
            weight_scale=float(effective["weight_scale"]),
            jobs=jobs,
        )
        baseline, share = _baseline_metrics(table)
        overview.append(
            {
                "scenario": scenario,
                "baseline": baseline,
                "wsf_share": share,
                "wsf_rule_ratio": _wsf_ratio(table),
                "SPT_TT": table["SPT"]["TT"],
                "SPT_WSF": table["SPT"]["WSF"],
                "rules": table,
            }
        )

    result = {
        "status": status,
        "selected": selected,
        "closest": closest,
        "original": {
            "tightness": current_tightness,
            "commitment": current_commitment,
            "weight_scale": current_weight,
        },
        "target_wsf_share": [TARGET_LOW, TARGET_HIGH],
        "minimum_wsf_rule_ratio": MIN_WSF_RATIO,
        "search_ranges": {
            "tightness": list(TIGHTNESS_RANGE),
            "commitment_ratio": list(COMMITMENT_RANGE),
            "weight_scale": list(WEIGHT_SCALE_RANGE),
        },
    }
    return result, overview, trajectory


def _format_ratio(value: float) -> str:
    return "∞" if math.isinf(value) else f"{value:.3f}"


def write_report(
    path: Path,
    result: dict[str, Any],
    overview: list[dict[str, Any]],
    trajectory: list[dict[str, Any]],
) -> None:
    selected = result["selected"]
    lines: list[str] = []
    if result["status"] != "success":
        lines.extend(
            [
                "# ⚠ 未达到目标",
                "",
                "**搜索范围内没有同时满足 WSF/F=35%–50% 与规则间 WSF "
                "最大最小比值≥1.10 的组合；scenario_config.json 未被改写。**",
                "",
                f"最接近结果：tightness={selected['tightness']:.3f}，"
                f"commitment={selected['commitment']:.3f}，"
                f"weight_scale={selected['weight_scale']:.4f}，"
                f"WSF/F={selected['wsf_share'] * 100:.2f}%，"
                f"WSF max/min={_format_ratio(selected['wsf_rule_ratio'])}。",
            ]
        )
    else:
        lines.extend(["# Scenario calibration", "", "校准达到目标并已写回配置。"])

    lines.extend(
        [
            "",
            "## 参数语义与搜索范围",
            "",
            "- tightness 搜索 0.30–1.00；数值越小，首个检查点越早，通常越难满足。",
            "- commitment_ratio 搜索 1.00–2.00；数值越大，承诺需求越高，通常越难满足；"
            "过大时所有规则都会处于不可达的饱和欠交区，从而失去规则区分度。",
            "- weight_scale 搜索 1–20；它只整体放大检查点欠交成本，"
            "不改变检查点需求或物理可达性，因此 WSF 与它近似线性。",
            "- 目标函数仍采用 α=β=1；weight_scale 属于检查点权重，不是目标函数系数。",
            "",
            "## 中心场景最终/最接近组合",
            "",
            f"参数：tightness={selected['tightness']:.3f}，"
            f"commitment={selected['commitment']:.3f}，"
            f"weight_scale={selected['weight_scale']:.4f}。",
            "",
            "| 规则 | F | TT | WSF |",
            "|---|---:|---:|---:|",
        ]
    )
    for name in RULE_ORDER:
        row = selected["rules"][name]
        lines.append(f"| {name} | {row['F']:.4f} | {row['TT']:.4f} | {row['WSF']:.4f} |")
    lines.extend(
        [
            "",
            f"较优经典规则：{selected['baseline']}；WSF/F="
            f"{selected['wsf_share'] * 100:.2f}%；规则间 WSF 最大最小比="
            f"{_format_ratio(selected['wsf_rule_ratio'])}。",
            "",
            "## 搜索轨迹",
            "",
            "| # | tightness | commitment | weight_scale | baseline | WSF/F | WSF max/min | SPT WSF | EDD WSF | ATC WSF | MIN_SLACK WSF | default WSF |",
            "|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for index, attempt in enumerate(trajectory, 1):
        rules = attempt["rules"]
        lines.append(
            f"| {index} | {attempt['tightness']:.3f} | {attempt['commitment']:.3f} | "
            f"{attempt['weight_scale']:.4f} | {attempt['baseline']} | "
            f"{attempt['wsf_share'] * 100:.2f}% | {_format_ratio(attempt['wsf_rule_ratio'])} | "
            f"{rules['SPT']['WSF']:.4f} | {rules['EDD']['WSF']:.4f} | "
            f"{rules['ATC_K2']['WSF']:.4f} | {rules['MIN_SLACK']['WSF']:.4f} | "
            f"{rules['default_due_date_slack']['WSF']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## 十个场景概览",
            "",
            "| 场景 | 较优经典规则 | WSF/F | WSF max/min | SPT TT | SPT WSF |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in overview:
        lines.append(
            f"| {row['scenario']} | {row['baseline']} | {row['wsf_share'] * 100:.2f}% | "
            f"{_format_ratio(row['wsf_rule_ratio'])} | {row['SPT_TT']:.4f} | "
            f"{row['SPT_WSF']:.4f} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--failed-out", type=Path, default=DEFAULT_FAILED)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be at least 1")
    if args.smoke:
        with tempfile.TemporaryDirectory(prefix="fjsp-calibration-smoke-") as raw_dir:
            smoke_dir = Path(raw_dir)
            result, overview, trajectory = calibrate(
                args.config,
                jobs=args.jobs,
                cache_path=smoke_dir / "cache.json",
                failed_path=smoke_dir / "failed.json",
                smoke=True,
                write_config=False,
            )
            write_report(smoke_dir / "scenario_calibration.md", result, overview, trajectory)
            print(f"smoke output: {smoke_dir}")
        return
    result, overview, trajectory = calibrate(
        args.config, jobs=args.jobs, cache_path=args.cache, failed_path=args.failed_out
    )
    write_report(args.out, result, overview, trajectory)
    if result["status"] != "success":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
