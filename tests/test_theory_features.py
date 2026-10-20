from conftest import make_test_instance

from fjsp_sfr.dispatch.theory_features import (
    THEORY_FEATURE_NAMES,
    build_extended_theory_features,
)
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
