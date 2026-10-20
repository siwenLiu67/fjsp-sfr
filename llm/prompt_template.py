"""构造 LLM 生成规则的 prompt。"""

from __future__ import annotations

from fjsp_sfr.dispatch.symbolic_rule import SymbolicRule
from fjsp_sfr.dispatch.theory_features import THEORY_FEATURE_NAMES


FEATURE_DESCRIPTIONS: dict[str, str] = {
    "gap_pressure": "Normalized residual checkpoint gap pressure for the candidate's entity",
    "forecast_deficit": "Normalized forecast insufficiency relative to checkpoint gap",
    "checkpoint_slack_urgency": "How close the candidate's optimistic completion is to the nearest checkpoint deadline",
    "multi_checkpoint_value": "Normalized multi-checkpoint service value W_j",
    "tardiness_exposure": "Tardiness risk: tanh((optimistic_completion - due_date) / time_scale)",
    "due_date_slack": "Normalized slack between due date and optimistic completion",
    "capacity_pressure": "Ratio of required workload to available machine capacity at related checkpoints",
    "forecast_flag": "Binary: 1 if any reachable checkpoint has forecast insufficiency, else 0",
    "capacity_flag": "Binary: 1 if any reachable checkpoint is capacity-constrained, else 0",
    "quantity_contribution": "Candidate's quantity relative to the checkpoint's residual gap",
    "processing_ratio": "Ratio of current operation processing time to total remaining processing",
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
) -> str:
    """构造约束 LLM 生成可执行符号调度规则的完整 prompt。"""

    feature_lines = "\n".join(
        f"- {name}: {FEATURE_DESCRIPTIONS[name]}" for name in THEORY_FEATURE_NAMES
    )
    feedback_text = feedback if feedback else "(none)"
    variables = ", ".join(THEORY_FEATURE_NAMES)

    return f"""You are designing symbolic dispatching rules for a dynamic flexible job-shop scheduling problem with cumulative service checkpoints.

The objective is F = alpha * total tardiness (TT) + beta * weighted service shortfall (WSF); lower is better.

Available normalized candidate features:
{feature_lines}

Every rule has the structure H = <Target, Cond, Filter, Priority> and one of four types:
- Type I: pure priority.
- Type II: conditional priority; use the default rule when Cond is false.
- Type III: weighted combination of theory features.
- Type IV: candidate Filter followed by Priority, optionally gated by Cond.

Expressions may use only these variables:
{variables}

Allowed functions are min, max, abs, math.tanh, math.log, math.exp, and math.sqrt. Use ordinary arithmetic, comparisons, boolean expressions, and conditional expressions only. Do not use imports, lambdas, attributes other than the listed math functions, indexing, or comprehensions.

Return exactly {num_rules} new, diverse rules in this repeated plain-text format:
RULE: <unique_name>
TYPE: I | II | III | IV
COND: <expression> | NONE
FILTER: <expression> | NONE
PRIORITY: <expression>

Existing rules to avoid duplicating:
{_existing_rule_text(existing_rules)}

Generation iteration: {iteration}
Feedback from the previous iteration:
{feedback_text}

Generate {num_rules} new rules now. Do not wrap the output in Markdown fences.
"""
