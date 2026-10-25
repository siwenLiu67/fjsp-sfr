import math

from conftest import make_test_instance

from fjsp_sfr.dispatch.symbolic_rule import RuleType, SymbolicRule
from fjsp_sfr.dispatch.theory_features import THEORY_FEATURE_NAMES, feature_names_for
from fjsp_sfr.llm.generator import (
    compute_behavior_similarity,
    evaluate_rule_objective,
)
from fjsp_sfr.llm.prompt_template import FEATURE_DESCRIPTIONS, build_generation_prompt
from fjsp_sfr.llm.rule_parser import parse_llm_output


class TestRuleParser:
    def test_parse_single_rule(self):
        text = """
Here is a scheduling rule:

RULE: urgency_combo
TYPE: III
COND: NONE
FILTER: NONE
PRIORITY: 0.5 * gap_pressure + 0.3 * tardiness_exposure + 0.2 * forecast_deficit
"""
        rules = parse_llm_output(text)
        assert len(rules) == 1
        assert rules[0].name == "urgency_combo"
        assert rules[0].rule_type == RuleType.TYPE_III
        assert rules[0].cond_expr is None
        assert rules[0].filter_expr is None

    def test_parse_multiple_rules(self):
        text = """
RULE: rule_a
TYPE: I
COND: NONE
FILTER: NONE
PRIORITY: -due_date_slack

RULE: rule_b
TYPE: IV
COND: forecast_flag
FILTER: quantity_contribution - 0.2
PRIORITY: gap_pressure + capacity_pressure
"""
        rules = parse_llm_output(text)
        assert len(rules) == 2
        assert rules[0].name == "rule_a"
        assert rules[1].filter_expr == "quantity_contribution - 0.2"

    def test_parse_skips_invalid_expression(self):
        text = """
RULE: bad_rule
TYPE: I
COND: NONE
FILTER: NONE
PRIORITY: __import__('os').system('bad')

RULE: good_rule
TYPE: I
COND: NONE
FILTER: NONE
PRIORITY: gap_pressure
"""
        rules = parse_llm_output(text)
        assert len(rules) == 1
        assert rules[0].name == "good_rule"

    def test_parse_skips_unknown_variable(self):
        text = """
RULE: unknown_var_rule
TYPE: I
COND: NONE
FILTER: NONE
PRIORITY: nonexistent_var * 2
"""
        assert parse_llm_output(text) == []

    def test_parse_type_ii_with_cond(self):
        text = """
RULE: cond_rule
TYPE: II
COND: forecast_flag
FILTER: NONE
PRIORITY: forecast_deficit * capacity_pressure
"""
        rules = parse_llm_output(text)
        assert len(rules) == 1
        assert rules[0].cond_expr == "forecast_flag"


class TestPromptTemplate:
    def test_prompt_contains_features(self):
        prompt = build_generation_prompt(num_rules=5)
        for name in THEORY_FEATURE_NAMES:
            assert name in prompt

    def test_prompt_contains_format_instructions(self):
        prompt = build_generation_prompt(num_rules=5)
        assert "RULE:" in prompt
        assert "TYPE:" in prompt
        assert "PRIORITY:" in prompt
        assert "H = <Cond, Filter, Priority>" in prompt
        assert "<Target" not in prompt
        assert "FILTER is optional for every rule type" in prompt
        assert (
            "COND is evaluated once per decision on the maximum value of each "
            "feature over the current candidates (not per candidate); FILTER "
            "and PRIORITY are evaluated per candidate. Higher PRIORITY is "
            "chosen first."
        ) in prompt
        assert (
            "The default ordering used when COND is false is ascending "
            "due-date slack."
        ) in prompt

    def test_feature_descriptions_complete(self):
        for name in THEORY_FEATURE_NAMES:
            assert name in FEATURE_DESCRIPTIONS

    def test_prompt_includes_existing_rules(self):
        existing = [
            SymbolicRule(
                name="existing_one",
                expression="test",
                rule_type=RuleType.TYPE_I,
                priority_expr="gap_pressure",
            )
        ]
        prompt = build_generation_prompt(existing_rules=existing, num_rules=5)
        assert "existing_one" in prompt

    def test_prompt_includes_feedback(self):
        prompt = build_generation_prompt(
            num_rules=5,
            feedback="Rule X performed poorly because it ignores tardiness.",
        )
        assert "performed poorly" in prompt

    def test_plus3_prompt_and_parser_use_new_features(self):
        prompt = build_generation_prompt(
            num_rules=5, feature_set="theory11_plus3"
        )
        for name in ("proc_time", "remaining_work", "critical_ratio"):
            assert name in prompt
        text = """
RULE: spt_like
TYPE: I
COND: NONE
FILTER: NONE
PRIORITY: -proc_time
"""
        rules = parse_llm_output(
            text, feature_names_for("theory11_plus3")
        )
        assert len(rules) == 1
        assert rules[0].feature_set == "theory11_plus3"

    def test_plus3_prompt_includes_classic_reference_and_decompositions(self):
        reference = (
            "- SPT: mean F=10.0, TT=4.0, WSF=6.0\n"
            "- ATC: mean F=9.0, TT=5.0, WSF=4.0\n"
            "- DEFAULT: mean F=12.0, TT=7.0, WSF=5.0"
        )
        prompt = build_generation_prompt(
            num_rules=10,
            feature_set="theory11_plus3",
            classic_reference=reference,
        )
        assert reference in prompt
        assert "improve on the classic rules" in prompt
        for name in ("proc_time", "remaining_work", "critical_ratio"):
            assert FEATURE_DESCRIPTIONS[name] in prompt
        assert "mean uncompressed" not in prompt
        assert "arithmetic mean" in prompt


class TestEvaluator:
    def test_evaluate_rule_objective_runs(self):
        rule = SymbolicRule(
            name="test",
            expression="test",
            rule_type=RuleType.TYPE_I,
            priority_expr="-tardiness_exposure",
        )
        objective = evaluate_rule_objective(
            rule, [make_test_instance()], num_episodes=1
        )
        assert isinstance(objective, float)
        assert objective >= 0.0
        assert math.isfinite(objective)

    def test_behavior_similarity_self(self):
        rule = SymbolicRule(
            name="self",
            expression="self",
            rule_type=RuleType.TYPE_I,
            priority_expr="-tardiness_exposure",
        )
        similarity = compute_behavior_similarity(
            rule, rule, [make_test_instance()]
        )
        assert similarity == 1.0
