"""P4.2 paper-to-code contracts; designed to run without PPO training."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from fjsp_sfr.core.forecast import build_compound_poisson_forecast_provider
from fjsp_sfr.dispatch.rule_library import RuleLibrary
from fjsp_sfr.dispatch.symbolic_rule import RuleType, SymbolicRule
from fjsp_sfr.dispatch.theory_features import feature_names_for
from fjsp_sfr.experiments.calibrate_paper_protocol import score_candidate, unavoidable_supply_shortfall
from fjsp_sfr.generation.scenario_generator import (
    load_scenario_config, make_instance, make_scenario_instance, scenario_config_hash,
)


def _paper_rule(feature: str = "proc_time") -> SymbolicRule:
    return SymbolicRule("feature_check", feature, RuleType.TYPE_I, feature, feature_set="paper14")


def test_symbolic_rule_requires_exact_paper14_names_and_strict_rows():
    rule = _paper_rule()
    data = rule.to_dict()
    assert data["feature_names"] == list(feature_names_for("paper14"))
    assert SymbolicRule.from_dict(data).feature_set == "paper14"
    for bad in (
        {**data, "feature_names": list(reversed(data["feature_names"]))},
        {k: v for k, v in data.items() if k != "feature_names"},
    ):
        with pytest.raises(ValueError):
            SymbolicRule.from_dict(bad)
    values = tuple(float(100 + 7 * index) for index in range(14))
    for index, name in enumerate(feature_names_for("paper14")):
        sentinel = _paper_rule(name)
        assert sentinel.priority_scores((values,)) == [values[index]]
    with pytest.raises(ValueError):
        rule.priority_scores((values[:-1],))


def test_formal_rule_library_rejects_legacy_even_when_width_matches():
    legacy = SymbolicRule("legacy", "legacy", RuleType.TYPE_I, "forecast_deficit", feature_set="theory11_plus3")
    with pytest.raises(ValueError):
        RuleLibrary((legacy,), feature_protocol="paper14").validate_protocol("paper14", formal=True)
    with pytest.raises(ValueError):
        RuleLibrary((_paper_rule(),)).validate_protocol("paper14", formal=True)
    RuleLibrary((_paper_rule(),), feature_protocol="paper14").validate_protocol("paper14", formal=True)


def test_public_entity_shares_govern_forecast_rates():
    provider = build_compound_poisson_forecast_provider(
        entity_ids=(0, 1), total_arrival_rate=0.5,
        quantity_mean=3.0, quantity_second_moment=11.0,
        max_time=20, entity_shares={0: 0.3, 1: 0.7},
    )
    snapshot = provider.snapshot_at(0)
    assert snapshot.for_entity(0)[0].d_hat == pytest.approx(9)
    assert snapshot.for_entity(1)[0].d_hat == pytest.approx(21)
    assert snapshot.for_entity(1)[0].sigma == pytest.approx(math.sqrt(0.5 * 0.7 * 20 * 11))
    with pytest.raises(ValueError):
        build_compound_poisson_forecast_provider(
            entity_ids=(0, 1), total_arrival_rate=0.5,
            quantity_mean=3, quantity_second_moment=11,
            max_time=20, entity_shares={0: 0.4, 1: 0.4},
        )


def test_scenario_v2_exogenous_and_independent_streams(tmp_path: Path):
    first = make_scenario_instance("center", 25100)
    second = make_scenario_instance("center", 25100)
    assert first.metadata["generator"] == "scenario_generator_v2"
    assert first.metadata["scenario_config_hash"] == second.metadata["scenario_config_hash"]
    shares = first.metadata["entity_shares"]
    assert len(shares) == len(first.entities)
    assert sum(shares) == pytest.approx(1.0)
    assert len(set(entity.checkpoints[-1].time for entity in first.entities)) > 1
    assert sum(job.quantity for job in first.jobs) == sum(job.quantity for job in second.jobs)
    original = load_scenario_config()
    variant = json.loads(json.dumps(original))
    variant["checkpoint"]["commitment_ratio"] = 0.40
    variant["checkpoint"]["first_time_ratio"] = 0.65
    variant["checkpoint"]["weights"] = [2, 6]
    config = tmp_path / "config.json"
    config.write_text(json.dumps(variant), encoding="utf-8")
    changed = make_scenario_instance("center", 25100, config_path=config)
    assert [job.entity_id for job in first.jobs] == [job.entity_id for job in changed.jobs]
    assert [job.release_time for job in first.jobs] == [job.release_time for job in changed.jobs]
    assert [job.quantity for job in first.jobs] == [job.quantity for job in changed.jobs]
    assert [entity.checkpoints[-1].time for entity in first.entities] == [
        entity.checkpoints[-1].time for entity in changed.entities
    ]
    assert first.metadata["scenario_config_hash"] == scenario_config_hash(original)
    assert changed.metadata["scenario_config_hash"] != first.metadata["scenario_config_hash"]


def test_ddt_changes_due_dates_but_not_service_windows():
    first = make_instance(10, 50, 0.8, 0.5, 25100)
    second = make_instance(10, 50, 0.8, 1.5, 25100)
    assert [entity.checkpoints for entity in first.entities] == [
        entity.checkpoints for entity in second.entities
    ]
    assert [job.due_date for job in first.jobs] != [job.due_date for job in second.jobs]


def test_offline_lower_bound_and_ratio_of_sums():
    instance = make_scenario_instance("center", 25100)
    manual = sum(
        entity.checkpoints[-1].weight *
        max(0, entity.checkpoints[-1].required_quantity -
            sum(job.quantity for job in instance.jobs if job.entity_id == entity.entity_id))
        for entity in instance.entities
    )
    assert unavoidable_supply_shortfall(instance) == pytest.approx(manual)
    runs = {
        "SPT": [
            {"F": 100, "TT": 60, "WSF": 40, "L": 10, "early_shortfall_count": 2, "early_checkpoint_count": 5},
            {"F": 300, "TT": 180, "WSF": 120, "L": 30, "early_shortfall_count": 4, "early_checkpoint_count": 5},
        ],
        "ATC": [
            {"F": 110, "TT": 65, "WSF": 45, "L": 10, "early_shortfall_count": 2, "early_checkpoint_count": 5},
            {"F": 310, "TT": 185, "WSF": 125, "L": 30, "early_shortfall_count": 4, "early_checkpoint_count": 5},
        ],
    }
    result = score_candidate(runs, 0.7, 0.55, 2)
    assert result["reference_rule"] == "SPT"
    assert result["R_WSF"] == pytest.approx(160 / 400)
    assert result["R_irr"] == pytest.approx(40 / 160)
    assert result["P_early"] == pytest.approx(6 / 10)
    assert result["eligible"] is True
