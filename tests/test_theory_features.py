from conftest import make_test_instance

from fjsp_sfr.dispatch.theory_features import (
    GENERAL_DISPATCH_FEATURE_NAMES,
    THEORY_FEATURE_NAMES,
    build_extended_theory_features,
    feature_names_for,
)
from fjsp_sfr.dispatch.classic_rules import build_classic_rule_library
from fjsp_sfr.dispatch.symbolic_rule import RuleType, SymbolicRule
from fjsp_sfr.env.ca_observation import build_theory_factor_values
from fjsp_sfr.env.ca_rule_selection import CAPPORuleSelectionEnv


def _features():
    env = CAPPORuleSelectionEnv(make_test_instance())
    observation = env.reset()
    assert observation.candidate_job_ids
    features = build_extended_theory_features(
        env._problem_view(), env._state, env.current_metrics
    )
    return env, features


def test_feature_count():
    """特征名数量正确。"""

    assert len(THEORY_FEATURE_NAMES) == 11


def test_feature_names_unique():
    """无重复特征名。"""

    assert len(set(THEORY_FEATURE_NAMES)) == 11


def test_first_five_match_legacy():
    """前 5 个特征名与现有 build_theory_factor_values 一致。"""

    legacy = (
        "gap_pressure",
        "forecast_deficit",
        "checkpoint_slack_urgency",
        "multi_checkpoint_value",
        "tardiness_exposure",
    )
    assert THEORY_FEATURE_NAMES[:5] == legacy


def test_first_five_values_match_legacy():
    env, features = _features()
    legacy = build_theory_factor_values(
        env._problem_view(), env._state, env.current_metrics
    )

    assert tuple(row[:5] for row in features) == legacy


def test_all_values_in_unit_interval():
    """所有特征值在 [0, 1] 范围内。"""

    _env, features = _features()
    for row in features:
        assert len(row) == 11
        for value in row:
            assert 0.0 <= value <= 1.0, f"feature value {value} out of [0,1]"


def test_flags_are_binary():
    """forecast_flag 和 capacity_flag 只取 0 或 1。"""

    _env, features = _features()
    forecast_index = THEORY_FEATURE_NAMES.index("forecast_flag")
    capacity_index = THEORY_FEATURE_NAMES.index("capacity_flag")
    for row in features:
        assert row[forecast_index] in (0.0, 1.0)
        assert row[capacity_index] in (0.0, 1.0)


def test_theory11_explicit_mode_is_bitwise_identical_to_default():
    env, default_rows = _features()
    explicit_rows = build_extended_theory_features(
        env._problem_view(),
        env._state,
        env.current_metrics,
        feature_set="theory11",
    )
    assert explicit_rows == default_rows


def test_plus3_values_and_monotonicity():
    env, base_rows = _features()
    rows = build_extended_theory_features(
        env._problem_view(),
        env._state,
        env.current_metrics,
        feature_set="theory11_plus3",
    )
    names = feature_names_for("theory11_plus3")
    assert names[-3:] == GENERAL_DISPATCH_FEATURE_NAMES
    assert all(len(row) == 14 and row[:11] == base for row, base in zip(rows, base_rows))
    proc_index = names.index("proc_time")
    remaining_index = names.index("remaining_work")
    critical_index = names.index("critical_ratio")
    # Fixture trigger-machine times are job 2 < job 0 < job 1.
    assert rows[2][proc_index] < rows[0][proc_index] < rows[1][proc_index]
    # All fixture jobs have equal remaining minimum work at the first decision.
    assert rows[0][remaining_index] == rows[1][remaining_index] == rows[2][remaining_index]
    # Later due dates give a strictly larger positive critical ratio here.
    assert rows[0][critical_index] < rows[1][critical_index] < rows[2][critical_index]
    assert all(0.0 <= value <= 1.0 for row in rows for value in row)


def test_plus3_symbolic_rules_reproduce_spt_and_atc_ordering():
    classic = {rule.name: rule for rule in build_classic_rule_library().rules}
    atc_expression = (
        "math.exp(-max(0.0, "
        "(remaining_work / max(1e-12, 1.0 - remaining_work)) * "
        "((critical_ratio / max(1e-12, 1.0 - critical_ratio)) - 1.0)) "
        "/ (2.0 * mean_remaining_work)) / max(1e-12, remaining_work / "
        "max(1e-12, 1.0 - remaining_work))"
    )
    symbolic = {
        "SPT": SymbolicRule(
            "symbolic_spt", "-proc_time", RuleType.TYPE_I, "-proc_time",
            feature_set="theory11_plus3",
        ),
        "ATC_K2": SymbolicRule(
            "symbolic_atc", atc_expression, RuleType.TYPE_I, atc_expression,
            feature_set="theory11_plus3",
        ),
    }
    for name in ("SPT", "ATC_K2"):
        classic_env = CAPPORuleSelectionEnv(make_test_instance())
        symbolic_env = CAPPORuleSelectionEnv(make_test_instance())
        classic_env.reset()
        symbolic_env.reset()
        classic_sequence = []
        symbolic_sequence = []
        while True:
            classic_step = classic_env.step_principle(0, classic[name])
            symbolic_step = symbolic_env.step_principle(0, symbolic[name])
            classic_sequence.append(
                (classic_step.decision.job_id, classic_step.decision.machine_id)
            )
            symbolic_sequence.append(
                (symbolic_step.decision.job_id, symbolic_step.decision.machine_id)
            )
            if classic_step.terminated:
                break
        assert symbolic_sequence == classic_sequence
