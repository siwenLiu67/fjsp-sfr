import argparse
import json

import pytest

from fjsp_sfr.dispatch.symbolic_rule import RuleType, SymbolicRule
from fjsp_sfr.experiments import build_gated_library as gated
from conftest import make_test_instance


def _rule(name: str) -> SymbolicRule:
    return SymbolicRule(name, name, RuleType.TYPE_I, "gap_pressure")


def _evaluated(name: str, ratios: tuple[float, ...]):
    return gated.EvaluatedRule(_rule(name), ratios, ratios, sum(ratios) / len(ratios))


def test_quality_gate_keeps_only_rules_below_default():
    rules = [_evaluated("good", (0.8, 0.9)), _evaluated("bad", (1.0, 1.1))]
    assert [item.rule.name for item in rules if item.normalized_objective < 1.0] == ["good"]


def test_similarity_is_symmetric_on_fixed_reference_states():
    rows = ((0.0,) * 11, (1.0,) * 11)
    states = (gated.ReferenceState(rows),)
    left = SymbolicRule("left", "left", RuleType.TYPE_I, "gap_pressure")
    right = SymbolicRule("right", "right", RuleType.TYPE_I, "gap_pressure")
    assert gated.symmetric_similarity(left, right, states) == gated.symmetric_similarity(right, left, states)


def test_marginal_gain_threshold_stops_selection():
    rules = [
        _evaluated("a", (0.80, 0.80)),
        _evaluated("b", (0.799, 0.80)),
        _evaluated("c", (0.60, 0.90)),
    ]
    selected, _gains = gated.greedy_complementary_selection(
        rules, budget=3, gain_threshold=0.01
    )
    assert [item.rule.name for item in selected] == ["c", "b"]


def test_mixed_selection_keeps_mandatory_classics_before_llm():
    classic = gated.EvaluatedRule(
        _rule("SPT"), (1.2, 1.2), (1.2, 1.2), 1.2, "classic"
    )
    llm = _evaluated("llm_good", (0.7, 0.8))
    selected, gains = gated.greedy_complementary_selection(
        [llm], budget=2, gain_threshold=0.0, initial=[classic]
    )
    assert [item.rule.name for item in selected] == ["SPT", "llm_good"]
    assert gains["SPT"] == 0.0


def test_multiple_candidate_files_are_merged_and_deduplicated(tmp_path):
    first = _rule("first")
    duplicate = SymbolicRule(
        "duplicate_name",
        "same definition",
        RuleType.TYPE_I,
        "gap_pressure",
    )
    distinct = SymbolicRule(
        "first",
        "different definition",
        RuleType.TYPE_I,
        "forecast_deficit",
    )
    paths = []
    for tag, rules in (("a", [first]), ("b", [duplicate, distinct])):
        path = tmp_path / f"{tag}.json"
        path.write_text(
            json.dumps(
                {
                    "tag": tag,
                    "candidates": [{"rule": rule.to_dict()} for rule in rules],
                }
            ),
            encoding="utf-8",
        )
        paths.append(path)
    rules, sources, duplicate_count = gated._load_candidate_rules(paths)
    assert len(rules) == 2
    assert duplicate_count == 1
    assert rules[0].name == "first"
    assert rules[1].name.startswith("first_b")
    assert sources[rules[0].name] == "llm:a"


def test_selection_and_holdout_seeds_do_not_overlap():
    assert set(gated.SELECTION_SEEDS).isdisjoint(gated.HOLDOUT_SEEDS)


def test_best_classic_gate_uses_one_rule_selected_by_mean(monkeypatch):
    instances = [make_test_instance(), make_test_instance()]
    monkeypatch.setattr(
        gated,
        "_classic_values",
        lambda _instances, *, feature_set: {
            "SPT": (5.0, 100.0),
            "ATC": (10.0, 20.0),
        },
    )
    values = gated._baseline_values(
        instances, gate_baseline="best_classic", feature_set="theory11"
    )
    # ATC has the lower selection-set mean (15 versus 52.5), so its complete
    # per-instance vector is the denominator; there is no elementwise minimum.
    assert values == (10.0, 20.0)


def test_disable_quality_gate_keeps_rules_worse_than_baseline(tmp_path, monkeypatch):
    candidate = _rule("worse")
    candidate_path = tmp_path / "candidates.json"
    candidate_path.write_text(
        json.dumps({"candidates": [{"rule": candidate.to_dict()}]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(gated, "make_instance", lambda seed, **kwargs: seed)
    monkeypatch.setattr(
        gated,
        "_baseline_values",
        lambda instances, **kwargs: tuple(10.0 for _ in instances),
    )
    monkeypatch.setattr(
        gated,
        "_classic_values",
        lambda instances, **kwargs: {
            "SPT": tuple(10.0 for _ in instances),
            "ATC": tuple(11.0 for _ in instances),
        },
    )
    monkeypatch.setattr(
        gated,
        "rollout_rule",
        lambda rule, instance, **kwargs: (
            (12.0 if rule.name == "worse" else 10.0),
            (),
        ),
    )
    output = tmp_path / "library.json"
    args = argparse.Namespace(
        candidates=[str(candidate_path)],
        tag="nogate",
        K=3,
        sim_threshold=0.9,
        gain_threshold=0.0,
        out=str(output),
        report=str(tmp_path / "report.md"),
        scale="S",
        difficulty="service_heavy",
        gate_baseline="default",
        extra_rules=None,
        feature_set="theory11",
        disable_quality_gate=True,
    )

    gated.build_gated_library(args)

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert [rule["name"] for rule in payload["rules"]] == ["worse"]
    assert payload["rules"][0]["f_H"] == pytest.approx(1.2)
