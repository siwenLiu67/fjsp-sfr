"""构造 LLM 生成规则的 prompt。"""

from __future__ import annotations

from fjsp_sfr.dispatch.symbolic_rule import SymbolicRule
from fjsp_sfr.dispatch.theory_features import expression_names_for, feature_names_for


FEATURE_DESCRIPTIONS: dict[str, str] = {
    "gap_pressure": "For the entity's next future checkpoint, required_workload / (required_workload + remaining_machine_capacity + epsilon), clipped to [0,1]. It is 0 when there is no future checkpoint or required workload is 0.",
    "forecast_deficit": "For the next future checkpoint, max(0, residual_gap - visible_unfinished_quantity - forecast_demand) / (residual_gap + epsilon), clipped to [0,1]. It is 0 when there is no positive gap or visible plus forecast quantity covers it.",
    "checkpoint_slack_urgency": "For the earliest future checkpoint reachable by the candidate's optimistic completion, (optimistic_completion - current_time) / (checkpoint_time - current_time + epsilon), clipped to [0,1]. It is 0 when no future checkpoint is reachable (or none exists).",
    "multi_checkpoint_value": "Across reachable future checkpoints, sum(weight * min(job_quantity, residual_gap) / (residual_gap + epsilon)) divided by the total weight of all future checkpoints, then clipped to [0,1]. It is 0 when there is no future/reachable checkpoint or no positive contribution.",
    "tardiness_exposure": "tanh(max(0, optimistic_completion - due_date) / time_scale), in [0,1). It is 0 when optimistic completion is no later than the due date.",
    "due_date_slack": "(due_date - optimistic_completion) / time_scale clipped to [0,1]. It is 0 when the optimistic completion is at or after the due date.",
    "capacity_pressure": "The maximum, over the candidate's reachable unmet checkpoints, of optimistic_required_workload / (remaining_machine_capacity + epsilon), clipped to [0,1]. It is 0 when no such checkpoint is reachable or all required workloads are 0.",
    "forecast_flag": "Binary indicator that at least one reachable unmet checkpoint has strictly positive arrival_insufficiency. It is 0 when none does or no unmet checkpoint is reachable.",
    "capacity_flag": "Binary indicator that at least one reachable unmet checkpoint has optimistic_required_workload greater than remaining_machine_capacity. It is 0 otherwise, including when none is reachable.",
    "quantity_contribution": "The maximum, over reachable unmet checkpoints, of min(job_quantity, residual_gap) / (residual_gap + epsilon), clipped to [0,1]. It is 0 when no unmet checkpoint is reachable or the candidate contributes no quantity.",
    "processing_ratio": "current_operation_processing_time / (current_operation_processing_time + downstream_minimum_processing + epsilon), clipped to [0,1]. It is 0 only when current processing time is 0.",
    "proc_time": "Let x be trigger-machine processing time divided by the instance mean alternative processing time; the feature is x/(1+x), in [0,1). It is 0 when processing time is 0 and preserves processing-time ordering.",
    "remaining_work": "Let x be current processing plus downstream minimum processing divided by the instance mean minimum job work; the feature is x/(1+x), in [0,1). It is 0 when no minimum work remains and preserves remaining-work ordering.",
    "critical_ratio": "Let x=max(0, due_date-current_time)/(remaining_minimum_work+epsilon); the feature is x/(1+x), in [0,1). It is 0 when the due date has been reached/passed (or the positive horizon is 0).",
}


def _existing_rule_text(existing_rules: list[SymbolicRule] | None) -> str:
    if not existing_rules:
        return "(none)"
    return "\n".join(
        (
            f"- {rule.name} (Type {int(rule.rule_type)}): "
            f"COND={rule.cond_expr or 'NONE'}; "
            f"FILTER={rule.filter_expr or 'NONE'}; "
            f"PRIORITY={rule.priority_expr}"
        )
        for rule in existing_rules
    )


def build_generation_prompt(
    *,
    existing_rules: list[SymbolicRule] | None = None,
    iteration: int = 0,
    num_rules: int = 10,
    feedback: str | None = None,
    require_type_coverage: bool = False,
    feature_set: str = "theory11",
    classic_reference: str | None = None,
) -> str:
    """构造约束 LLM 生成可执行符号调度规则的完整 prompt。"""

    feature_names = feature_names_for(feature_set)
    feature_lines = "\n".join(
        f"- {name}: {FEATURE_DESCRIPTIONS[name]}" for name in feature_names
    )
    feedback_text = feedback if feedback else "(none)"
    variables = ", ".join(expression_names_for(feature_set))
    batch_context_text = (
        "\nBatch context (available to expressions but not an extra job-node "
        "dimension):\n- mean_remaining_work: arithmetic mean, over current "
        "candidates, of the uncompressed remaining-work ratio "
        "(remaining minimum work / instance mean minimum job work). It is "
        "nonnegative and is 0 only when every candidate has zero remaining work.\n"
        if feature_set == "theory11_plus3"
        else ""
    )
    coverage_text = (
        "\nThis round must cover all four rule types I-IV, with at least "
        "two rules of each type.\n"
        if require_type_coverage
        else ""
    )
    classic_reference_text = (
        "\nClassic-rule reference on the same selection instances "
        "(lower is better):\n"
        f"{classic_reference}\n"
        "Use these as explicit baselines: seek rules that improve on the "
        "classic rules in total F while making a defensible TT/WSF trade-off.\n"
        if classic_reference
        else ""
    )

    return f"""You are designing symbolic dispatching rules for a dynamic flexible job-shop scheduling problem with cumulative service checkpoints.

The objective is F = alpha * total tardiness (TT) + beta * weighted service shortfall (WSF); lower is better.
{classic_reference_text}

Available normalized candidate features:
{feature_lines}
{batch_context_text}

Every rule has the structure H = <Cond, Filter, Priority> and one of four types:
- Type I, simple priority rule: sort by one feature only.
- Type II, composite rule: combine 2-3 features, with one dominant feature and the others acting as adjustments.
- Type III, weighted priority-index rule: use a weighted sum of several normalized features.
- Type IV, conditional rule: provide COND; when it is true, sort by PRIORITY, otherwise fall back to the default ordering.

FILTER is optional for every rule type.
COND is evaluated once per decision on the maximum value of each feature over the current candidates (not per candidate); FILTER and PRIORITY are evaluated per candidate. Higher PRIORITY is chosen first.
The default ordering used when COND is false is ascending due-date slack.

Expressions may use only these variables:
{variables}

Allowed functions are min, max, abs, math.tanh, math.log, math.exp, and math.sqrt. Use ordinary arithmetic, comparisons, boolean expressions, and conditional expressions only. Do not use imports, lambdas, attributes other than the listed math functions, indexing, or comprehensions.

Return exactly {num_rules} new, diverse rules in this repeated plain-text format:
RULE: <unique_name>
TYPE: I | II | III | IV
COND: <expression> | NONE
FILTER: <expression> | NONE
PRIORITY: <expression>
{coverage_text}

Existing rules to avoid duplicating:
{_existing_rule_text(existing_rules)}

Generation iteration: {iteration}
Feedback from the previous iteration:
{feedback_text}

Generate {num_rules} new rules now. Do not wrap the output in Markdown fences.
"""
