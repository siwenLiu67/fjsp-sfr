import os
import tempfile

from fjsp_sfr.dispatch.rule_library import RuleLibrary
from fjsp_sfr.dispatch.symbolic_rule import RuleType, SymbolicRule
from fjsp_sfr.dispatch.theory_features import THEORY_FEATURE_NAMES


def _make_row(**overrides) -> tuple[float, ...]:
    """构造一行特征，未指定的特征默认 0.0。"""

    values = {name: 0.0 for name in THEORY_FEATURE_NAMES}
    values.update(overrides)
    return tuple(values[name] for name in THEORY_FEATURE_NAMES)


class TestTypeI:
    def test_priority_selects_max(self):
        rule = SymbolicRule(
            name="edd",
            expression="earliest due date",
            rule_type=RuleType.TYPE_I,
            priority_expr="-due_date_slack",
        )
        rows = (
            _make_row(due_date_slack=0.8),
            _make_row(due_date_slack=0.2),
            _make_row(due_date_slack=0.5),
        )
        scores = rule.evaluate_batch(rows)
        assert scores.index(max(scores)) == 1

    def test_always_applicable(self):
        rule = SymbolicRule(
            name="t",
            expression="t",
            rule_type=RuleType.TYPE_I,
            priority_expr="gap_pressure",
        )
        assert rule.is_applicable(_make_row())


class TestTypeII:
    def test_cond_gates_applicability(self):
        rule = SymbolicRule(
            name="cond_rule",
            expression="conditional",
            rule_type=RuleType.TYPE_II,
            priority_expr="gap_pressure",
            cond_expr="forecast_flag",
        )
        assert not rule.is_applicable(_make_row(forecast_flag=0.0))
        assert rule.is_applicable(_make_row(forecast_flag=1.0))


class TestTypeIII:
    def test_weighted_combination(self):
        rule = SymbolicRule(
            name="combo",
            expression="weighted combo",
            rule_type=RuleType.TYPE_III,
            priority_expr="0.6 * gap_pressure + 0.4 * tardiness_exposure",
        )
        rows = (
            _make_row(gap_pressure=0.9, tardiness_exposure=0.1),
            _make_row(gap_pressure=0.1, tardiness_exposure=0.9),
        )
        scores = rule.evaluate_batch(rows)
        assert scores[0] > scores[1]
        assert abs(scores[0] - 0.58) < 1e-9
        assert abs(scores[1] - 0.42) < 1e-9


class TestTypeIV:
    def test_filter_excludes_candidates(self):
        rule = SymbolicRule(
            name="filtered",
            expression="filter + priority",
            rule_type=RuleType.TYPE_IV,
            priority_expr="multi_checkpoint_value",
            filter_expr="quantity_contribution - 0.3",
        )
        rows = (
            _make_row(multi_checkpoint_value=0.9, quantity_contribution=0.1),
            _make_row(multi_checkpoint_value=0.3, quantity_contribution=0.5),
            _make_row(multi_checkpoint_value=0.7, quantity_contribution=0.8),
        )
        scores = rule.evaluate_batch(rows)
        assert scores[0] == float("-inf")
        assert scores[2] > scores[1]

    def test_all_filtered_returns_neg_inf(self):
        rule = SymbolicRule(
            name="strict",
            expression="strict filter",
            rule_type=RuleType.TYPE_IV,
            priority_expr="gap_pressure",
            filter_expr="forecast_flag - 1",
        )
        rows = (_make_row(), _make_row())
        scores = rule.evaluate_batch(rows)
        assert all(score == float("-inf") for score in scores)


class TestSerialization:
    def test_round_trip_json(self):
        rules = (
            SymbolicRule(
                name="r1",
                expression="desc1",
                rule_type=RuleType.TYPE_I,
                priority_expr="gap_pressure",
            ),
            SymbolicRule(
                name="r2",
                expression="desc2",
                rule_type=RuleType.TYPE_IV,
                priority_expr="tardiness_exposure",
                cond_expr="forecast_flag",
                filter_expr="capacity_flag",
            ),
        )
        library = RuleLibrary(rules=rules)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as stream:
            path = stream.name
        try:
            library.to_json(path)
            loaded = RuleLibrary.from_json(path)
            assert loaded.action_size() == 2
            assert loaded.rules[0].name == "r1"
            assert loaded.rules[1].cond_expr == "forecast_flag"
            assert loaded.rules[1].filter_expr == "capacity_flag"
        finally:
            os.unlink(path)

    def test_to_dict_from_dict(self):
        rule = SymbolicRule(
            name="x",
            expression="x",
            rule_type=RuleType.TYPE_III,
            priority_expr="0.5 * gap_pressure + 0.5 * forecast_deficit",
        )
        restored = SymbolicRule.from_dict(rule.to_dict())
        assert restored.name == rule.name
        assert restored.priority_expr == rule.priority_expr
        row = _make_row(gap_pressure=0.4, forecast_deficit=0.6)
        assert abs(
            restored.evaluate_batch((row,))[0]
            - rule.evaluate_batch((row,))[0]
        ) < 1e-12


class TestRuleLibrary:
    def test_select_and_dispatch(self):
        rules = (
            SymbolicRule(
                name="a",
                expression="a",
                rule_type=RuleType.TYPE_I,
                priority_expr="gap_pressure",
            ),
            SymbolicRule(
                name="b",
                expression="b",
                rule_type=RuleType.TYPE_I,
                priority_expr="-gap_pressure",
            ),
        )
        library = RuleLibrary(rules=rules)
        rows = (
            _make_row(gap_pressure=0.3),
            _make_row(gap_pressure=0.9),
            _make_row(gap_pressure=0.1),
        )
        assert library.select_and_dispatch(0, rows) == 1
        assert library.select_and_dispatch(1, rows) == 2

    def test_all_filtered_falls_back_to_first(self):
        rule = SymbolicRule(
            name="strict",
            expression="strict",
            rule_type=RuleType.TYPE_IV,
            priority_expr="gap_pressure",
            filter_expr="forecast_flag - 1",
        )
        library = RuleLibrary(rules=(rule,))
        rows = (_make_row(), _make_row())
        assert library.select_and_dispatch(0, rows) == 0
