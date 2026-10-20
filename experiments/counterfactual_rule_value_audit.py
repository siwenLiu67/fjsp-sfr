"""Diagnostic switching-value audit for the eight-rule action space."""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, median

from fjsp_sfr.core.models import FJSPInstance
from fjsp_sfr.dispatch import RuleAction, rule_applicability_mask
from fjsp_sfr.dispatch.ca_rules import select_job
from fjsp_sfr.env import (
    CADecisionObservation,
    CAEnvironmentSnapshot,
    OnlineRuleSelectionEnv,
)

from .fixed_rule_policy import current_fixed_rule_action, step_current_fixed_rule


RULES = tuple(RuleAction)


@dataclass(frozen=True)
class TerminalOutcome:
    selected_job_id: int
    F: float
    TT: float
    WSF: float


@dataclass(frozen=True)
class OracleInstanceResult:
    instance_id: int
    fixed_r5_F: float
    fixed_r5_TT: float
    fixed_r5_WSF: float
    oracle_F: float
    oracle_TT: float
    oracle_WSF: float
    improvement: float
    improvement_percent: float
    decisions: int
    non_r5_decisions: int


def fixed_r5_action(observation: CADecisionObservation) -> RuleAction:
    """Return current R5 when legal, otherwise the common current R1 fallback."""

    if observation.action_mask[int(RuleAction.SERVICE_VALUE_DENSITY)]:
        return RuleAction.SERVICE_VALUE_DENSITY
    return RuleAction.PROJECTED_TARDINESS


def _finish_with_fixed_r5(
    env: OnlineRuleSelectionEnv,
    observation: CADecisionObservation | None,
    *,
    first_selected_job: int,
    terminal_step=None,
) -> TerminalOutcome:
    step = terminal_step
    current = observation
    while step is None or not step.terminated:
        if current is None:
            raise RuntimeError("nonterminal branch has no decision observation")
        step = step_current_fixed_rule(env, RuleAction.SERVICE_VALUE_DENSITY)
        current = step.observation
    if step.objective is None:
        raise RuntimeError("terminal branch has no objective")
    return TerminalOutcome(
        selected_job_id=first_selected_job,
        F=step.objective.Z,
        TT=float(step.objective.total_tardiness),
        WSF=step.objective.weighted_checkpoint_shortfall,
    )


def rollout_counterfactual(
    snapshot: CAEnvironmentSnapshot,
    action: RuleAction,
) -> TerminalOutcome:
    """Force one valid action, then apply current fixed R5/R1 to terminal."""

    branch = OnlineRuleSelectionEnv.from_snapshot(snapshot)
    step = branch.step(action)
    return _finish_with_fixed_r5(
        branch,
        step.observation,
        first_selected_job=step.decision.job_id,
        terminal_step=step,
    )


def rollout_fixed_r5(snapshot: CAEnvironmentSnapshot) -> TerminalOutcome:
    """Roll out current R5 with the common R1 fallback from a cloned state."""

    branch = OnlineRuleSelectionEnv.from_snapshot(snapshot)
    first = step_current_fixed_rule(branch, RuleAction.SERVICE_VALUE_DENSITY)
    return _finish_with_fixed_r5(
        branch,
        first.observation,
        first_selected_job=first.decision.job_id,
        terminal_step=first,
    )


def run_fixed_r5(instance: FJSPInstance) -> TerminalOutcome:
    env = OnlineRuleSelectionEnv(instance)
    observation = env.reset()
    first = step_current_fixed_rule(env, RuleAction.SERVICE_VALUE_DENSITY)
    return _finish_with_fixed_r5(
        env,
        first.observation,
        first_selected_job=first.decision.job_id,
        terminal_step=first,
    )


def _state_diagnostics(env: OnlineRuleSelectionEnv) -> dict[str, float | None]:
    metrics = env.current_metrics
    problem = env._problem_view()
    checkpoints = tuple(
        item
        for item in metrics.checkpoints.values()
        if item.checkpoint_time >= metrics.decision_time
        and item.residual_gap > 0.0
    )
    max_a_f = max((item.arrival_insufficiency for item in checkpoints), default=0.0)
    max_r_f = max((item.future_required for item in checkpoints), default=0.0)
    slacks = tuple(
        item.release_slack for item in checkpoints if item.release_slack is not None
    )
    max_phi = max((candidate.service_value for candidate in metrics.candidates), default=0.0)
    action_mask = rule_applicability_mask(problem, metrics)

    def selected_score(action: RuleAction, component: str) -> float:
        if not action_mask[int(action)]:
            return 0.0
        return float(dict(select_job(problem, metrics, action).score_components)[component])

    if action_mask[int(RuleAction.CRITICAL_CHECKPOINT_RESCUE)]:
        r6 = select_job(problem, metrics, RuleAction.CRITICAL_CHECKPOINT_RESCUE)
        r6_scores = dict(r6.score_components)
        h_difference = float(r6_scores.get("critical_checkpoint_margin", 0.0))
    else:
        h_difference = 0.0
    return {
        "max_A_F": max_a_f,
        "max_R_F": max_r_f,
        "min_S_F": min(slacks) if slacks else None,
        "max_Phi": max_phi,
        "R5_service_value_density": selected_score(
            RuleAction.SERVICE_VALUE_DENSITY, "service_value_density"
        ),
        "R6_critical_checkpoint_margin": h_difference,
        "R7_forecast_shortage_priority": selected_score(
            RuleAction.FORECAST_SHORTAGE_PRIORITY,
            "forecast_shortage_priority",
        ),
        "R8_forecast_late_arrival_priority": selected_score(
            RuleAction.FORECAST_LATE_ARRIVAL_PRIORITY,
            "forecast_late_arrival_priority",
        ),
    }


def audit_decision_state(
    env: OnlineRuleSelectionEnv,
    observation: CADecisionObservation,
    *,
    instance_id: int,
) -> dict:
    snapshot = env.snapshot()
    baseline_action = fixed_r5_action(observation)
    baseline = rollout_fixed_r5(snapshot)
    outcomes: dict[RuleAction, TerminalOutcome | None] = {}
    for action in RULES:
        if observation.action_mask[int(action)]:
            outcomes[action] = rollout_counterfactual(snapshot, action)
        else:
            outcomes[action] = None
    # F_R5 denotes current R5 plus the common current R1 fallback when R5 is masked.
    outcomes[RuleAction.SERVICE_VALUE_DENSITY] = baseline
    valid_for_choice = {
        action: outcome
        for action, outcome in outcomes.items()
        if observation.action_mask[int(action)] and outcome is not None
    }
    best_f = min(outcome.F for outcome in valid_for_choice.values())
    co_best = tuple(
        action
        for action, outcome in valid_for_choice.items()
        if abs(outcome.F - best_f) <= 1e-9
    )
    strict_best = co_best[0] if len(co_best) == 1 else None
    record = {
        "instance_id": instance_id,
        "decision_id": observation.decision_index,
        "tau": observation.physical_time,
        "action_mask": tuple(observation.action_mask),
        "baseline_effective_action": f"R{baseline_action.paper_number}",
        "best_rule": "|".join(f"R{action.paper_number}" for action in co_best),
        "strict_best_rule": (
            f"R{strict_best.paper_number}" if strict_best is not None else None
        ),
        "gain_vs_R5": baseline.F - best_f,
    }
    for action in RULES:
        label = f"R{action.paper_number}"
        outcome = outcomes[action]
        record[f"job_{label}"] = (
            outcome.selected_job_id if outcome is not None else None
        )
        record[f"F_{label}"] = outcome.F if outcome is not None else None
        record[f"TT_{label}"] = outcome.TT if outcome is not None else None
        record[f"WSF_{label}"] = outcome.WSF if outcome is not None else None
    record.update(_state_diagnostics(env))
    return record


def collect_fixed_r5_states(
    instances: tuple[FJSPInstance, ...],
    *,
    state_stride: int = 1,
    max_states_per_instance: int | None = None,
) -> tuple[dict, ...]:
    if state_stride <= 0:
        raise ValueError("state_stride must be positive")
    records: list[dict] = []
    for instance_id, instance in enumerate(instances):
        env = OnlineRuleSelectionEnv(instance)
        observation = env.reset()
        sampled = 0
        trajectory_decision = 0
        while True:
            if (
                trajectory_decision % state_stride == 0
                and (
                    max_states_per_instance is None
                    or sampled < max_states_per_instance
                )
            ):
                records.append(
                    audit_decision_state(
                        env, observation, instance_id=instance_id
                    )
                )
                sampled += 1
            step = step_current_fixed_rule(env, RuleAction.SERVICE_VALUE_DENSITY)
            if step.terminated:
                break
            if step.observation is None:
                raise RuntimeError("fixed-R5 trajectory lost its observation")
            observation = step.observation
            trajectory_decision += 1
    return tuple(records)


def _rule_summary(records: tuple[dict, ...], action: RuleAction) -> dict:
    label = f"R{action.paper_number}"
    index = int(action)
    applicable = tuple(row for row in records if row["action_mask"][index])
    disagreements = tuple(
        row
        for row in applicable
        if row[f"job_{label}"] != row["job_R5"]
    )
    gains = tuple(row["F_R5"] - row[f"F_{label}"] for row in applicable)
    tt_gains = tuple(row["TT_R5"] - row[f"TT_{label}"] for row in applicable)
    wsf_gains = tuple(row["WSF_R5"] - row[f"WSF_{label}"] for row in applicable)
    positive = tuple(value for value in gains if value > 1e-9)
    return {
        "applicable_rate": len(applicable) / len(records) if records else 0.0,
        "selected_job_disagreement_rate_given_applicable": (
            len(disagreements) / len(applicable) if applicable else 0.0
        ),
        "strict_best_rate": (
            sum(row["strict_best_rule"] == label for row in records) / len(records)
            if records
            else 0.0
        ),
        "strict_best_rate_given_applicable": (
            sum(row["strict_best_rule"] == label for row in applicable)
            / len(applicable)
            if applicable
            else 0.0
        ),
        "conditional_mean_gain_vs_R5": fmean(gains) if gains else 0.0,
        "conditional_median_gain_vs_R5": median(gains) if gains else 0.0,
        "conditional_mean_TT_gain_vs_R5": fmean(tt_gains) if tt_gains else 0.0,
        "conditional_mean_WSF_gain_vs_R5": fmean(wsf_gains) if wsf_gains else 0.0,
        "positive_gain_rate_given_applicable": (
            len(positive) / len(applicable) if applicable else 0.0
        ),
        "mean_positive_gain": fmean(positive) if positive else 0.0,
    }


def _conditional_summary(records: tuple[dict, ...]) -> dict:
    return {
        "states": len(records),
        "state_fraction": 0.0,
        "rules": {
            f"R{action.paper_number}": _rule_summary(records, action)
            for action in RULES
        },
        "non_r5_opportunity_rate": (
            sum(row["gain_vs_R5"] > 1e-9 for row in records) / len(records)
            if records
            else 0.0
        ),
    }


def summarize_counterfactual_records(records: tuple[dict, ...]) -> dict:
    r7_target = tuple(row for row in records if row["max_A_F"] > 1e-9)
    r5_target = tuple(
        row
        for row in records
        if row["action_mask"][int(RuleAction.SERVICE_VALUE_DENSITY)]
    )
    best_distribution = Counter()
    strict_distribution = Counter(
        row["strict_best_rule"]
        for row in records
        if row["strict_best_rule"] is not None
    )
    for row in records:
        for label in row["best_rule"].split("|"):
            best_distribution[label] += 1
    summary = {
        "sampled_states": len(records),
        "non_r5_opportunity_rate": (
            sum(row["gain_vs_R5"] > 1e-9 for row in records) / len(records)
            if records
            else 0.0
        ),
        "mean_oracle_one_step_gain_vs_R5": (
            fmean(row["gain_vs_R5"] for row in records) if records else 0.0
        ),
        "mean_positive_one_step_gain_vs_R5": (
            fmean(
                row["gain_vs_R5"]
                for row in records
                if row["gain_vs_R5"] > 1e-9
            )
            if any(row["gain_vs_R5"] > 1e-9 for row in records)
            else 0.0
        ),
        "best_rule_distribution_co_best_counts": dict(sorted(best_distribution.items())),
        "best_rule_distribution_strict_counts": dict(sorted(strict_distribution.items())),
        "rules": {
            f"R{action.paper_number}": _rule_summary(records, action)
            for action in RULES
        },
        "conditional": {
            "R7_arrival_insufficiency_positive": _conditional_summary(r7_target),
            "R5_service_value_density_applicable": _conditional_summary(r5_target),
        },
    }
    summary["conditional"]["R7_arrival_insufficiency_positive"]["state_fraction"] = (
        len(r7_target) / len(records) if records else 0.0
    )
    summary["conditional"]["R5_service_value_density_applicable"]["state_fraction"] = (
        len(r5_target) / len(records) if records else 0.0
    )
    return summary


def run_diagnostic_one_step_oracle(
    instance: FJSPInstance,
    *,
    instance_id: int,
) -> OracleInstanceResult:
    baseline = run_fixed_r5(instance)
    env = OnlineRuleSelectionEnv(instance)
    observation = env.reset()
    decisions = 0
    non_r5_decisions = 0
    while True:
        snapshot = env.snapshot()
        valid = tuple(
            action for action in RULES if observation.action_mask[int(action)]
        )
        lookahead = {
            action: rollout_counterfactual(snapshot, action) for action in valid
        }
        problem = env._problem_view()
        baseline_action = current_fixed_rule_action(
            problem, env.current_metrics, RuleAction.SERVICE_VALUE_DENSITY
        )
        baseline_job_id = select_job(
            problem, env.current_metrics, baseline_action
        ).job_id
        selected = min(
            valid,
            key=lambda action: (
                lookahead[action].F,
                lookahead[action].selected_job_id != baseline_job_id,
                int(action),
            ),
        )
        non_r5_decisions += lookahead[selected].selected_job_id != baseline_job_id
        step = env.step(selected)
        decisions += 1
        if step.terminated:
            if step.objective is None:
                raise RuntimeError("oracle terminal step has no objective")
            objective = step.objective
            break
        if step.observation is None:
            raise RuntimeError("oracle nonterminal step has no observation")
        observation = step.observation
    improvement = baseline.F - objective.Z
    return OracleInstanceResult(
        instance_id=instance_id,
        fixed_r5_F=baseline.F,
        fixed_r5_TT=baseline.TT,
        fixed_r5_WSF=baseline.WSF,
        oracle_F=objective.Z,
        oracle_TT=float(objective.total_tardiness),
        oracle_WSF=objective.weighted_checkpoint_shortfall,
        improvement=improvement,
        improvement_percent=(100.0 * improvement / baseline.F),
        decisions=decisions,
        non_r5_decisions=non_r5_decisions,
    )


def summarize_oracle(results: tuple[OracleInstanceResult, ...]) -> dict:
    if not results:
        return {"instances": 0}
    return {
        "instances": len(results),
        "fixed_r5_mean_F": fmean(item.fixed_r5_F for item in results),
        "fixed_r5_mean_TT": fmean(item.fixed_r5_TT for item in results),
        "fixed_r5_mean_WSF": fmean(item.fixed_r5_WSF for item in results),
        "oracle_mean_F": fmean(item.oracle_F for item in results),
        "oracle_mean_TT": fmean(item.oracle_TT for item in results),
        "oracle_mean_WSF": fmean(item.oracle_WSF for item in results),
        "mean_instance_improvement": fmean(item.improvement for item in results),
        "aggregate_improvement_percent": 100.0
        * (
            fmean(item.fixed_r5_F for item in results)
            - fmean(item.oracle_F for item in results)
        )
        / fmean(item.fixed_r5_F for item in results),
        "improved_instance_rate": sum(item.improvement > 1e-9 for item in results)
        / len(results),
        "mean_non_r5_decisions": fmean(item.non_r5_decisions for item in results),
    }


def write_audit_artifacts(
    output_dir: Path,
    records: tuple[dict, ...],
    oracle_results: tuple[OracleInstanceResult, ...],
    summary: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for record in records:
        row = dict(record)
        row["action_mask"] = json.dumps(record["action_mask"])
        rows.append(row)
    with (output_dir / "counterfactual_states.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "diagnostic_oracle_instances.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        oracle_rows = [item.__dict__ for item in oracle_results]
        writer = csv.DictWriter(stream, fieldnames=list(oracle_rows[0]))
        writer.writeheader()
        writer.writerows(oracle_rows)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
