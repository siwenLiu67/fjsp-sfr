import json
from dataclasses import asdict
from pathlib import Path

import pytest

from experiments.calibrate_scenario import (
    MIN_WSF_RATIO,
    TARGET_HIGH,
    TARGET_LOW,
    _baseline_metrics,
    _estimated_weight_scale,
    _is_success,
    _trial_config,
    _validated_metrics,
    _wsf_ratio,
)
from fjsp_sfr.generation.scenario_generator import (
    CONFIG_PATH,
    load_scenario_config,
    make_scenario_instance,
)


def _rule_table(*, spt_wsf: float = 40.0, atc_wsf: float = 30.0):
    return {
        "SPT": {"F": 140.0, "TT": 100.0, "WSF": spt_wsf},
        "EDD": {"F": 145.0, "TT": 100.0, "WSF": 45.0},
        "ATC_K2": {"F": 130.0, "TT": 100.0, "WSF": atc_wsf},
        "MIN_SLACK": {"F": 150.0, "TT": 100.0, "WSF": 50.0},
        "default_due_date_slack": {"F": 142.0, "TT": 100.0, "WSF": 42.0},
    }


def test_weight_scale_is_written_to_trial_without_changing_objective_weights():
    base = {"checkpoint": {"tightness": 0.55, "commitment_ratio": 1.15}}
    trial = _trial_config(base, 0.4, 1.3, 7.5)
    assert trial["checkpoint"] == {
        "tightness": 0.4,
        "commitment_ratio": 1.3,
        "weight_scale": 7.5,
    }
    assert "weight_scale" not in base["checkpoint"]


def test_weight_scale_multiplies_all_generated_checkpoint_weights(tmp_path: Path):
    config = load_scenario_config(CONFIG_PATH)
    config["checkpoint"]["weight_scale"] = 1.0
    unscaled_path = tmp_path / "unscaled.json"
    unscaled_path.write_text(json.dumps(config), encoding="utf-8")
    config["checkpoint"]["weight_scale"] = 3.0
    scaled_path = tmp_path / "scaled.json"
    scaled_path.write_text(json.dumps(config), encoding="utf-8")

    unscaled = make_scenario_instance("center", seed=321, config_path=unscaled_path)
    scaled = make_scenario_instance("center", seed=321, config_path=scaled_path)

    assert len(unscaled.entities) == len(scaled.entities)
    for left_entity, right_entity in zip(unscaled.entities, scaled.entities):
        assert len(left_entity.checkpoints) == len(right_entity.checkpoints)
        for left, right in zip(left_entity.checkpoints, right_entity.checkpoints):
            left_values = asdict(left)
            right_values = asdict(right)
            assert right_values["weight"] == pytest.approx(left_values["weight"] * 3.0)
            right_values["weight"] = left_values["weight"]
            assert right_values == left_values


def test_calibration_uses_better_mean_classic_rule_and_checks_wsf_spread():
    table = _rule_table()
    baseline, share = _baseline_metrics(table)
    assert baseline == "ATC_K2"
    assert share == pytest.approx(30.0 / 130.0)
    assert _wsf_ratio(table) == pytest.approx(50.0 / 30.0)


def test_weight_estimate_targets_middle_of_requested_share():
    table = _rule_table(atc_wsf=10.0)
    weight = _estimated_weight_scale(table)
    target = 0.5 * (TARGET_LOW + TARGET_HIGH)
    achieved = weight * 10.0 / (100.0 + weight * 10.0)
    assert achieved == pytest.approx(target)


def test_success_requires_share_and_rule_differentiation():
    assert _is_success(
        {"wsf_share": TARGET_LOW, "wsf_rule_ratio": MIN_WSF_RATIO}
    )
    assert not _is_success(
        {"wsf_share": TARGET_HIGH, "wsf_rule_ratio": MIN_WSF_RATIO - 0.01}
    )
    assert not _is_success(
        {"wsf_share": TARGET_LOW - 0.01, "wsf_rule_ratio": MIN_WSF_RATIO + 1.0}
    )


def test_calibration_metrics_require_f_to_equal_tt_plus_wsf():
    assert _validated_metrics({"F": 7, "TT": 3, "WSF": 4}) == {
        "F": 7.0,
        "TT": 3.0,
        "WSF": 4.0,
    }
    with pytest.raises(ValueError, match="objective decomposition mismatch"):
        _validated_metrics({"F": 8, "TT": 3, "WSF": 4})
