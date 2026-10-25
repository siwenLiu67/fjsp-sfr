import argparse
import json

from conftest import make_test_instance

from fjsp_sfr.dispatch.rule_library import RuleLibrary
from fjsp_sfr.experiments import generate_rule_library as generation
from fjsp_sfr.experiments import train_streaming


def test_seed_partitions_are_pairwise_disjoint():
    generation.assert_seed_partitions_disjoint(training_episodes=5000)


def test_phase2_generation_defaults_and_metric_roundtrip():
    args = generation.parse_args(["--tag", "p3b1", "--model", "stub"])
    assert args.feature_set == "theory11_plus3"
    assert args.scale == "M"
    assert args.difficulty == "service_heavy"
    assert args.show_classic_reference is True

    rule = generation.SymbolicRule(
        name="metric_rule",
        expression="metric test",
        rule_type=generation.RuleType.TYPE_I,
        priority_expr="-proc_time",
        feature_set="theory11_plus3",
    )
    record = generation.CandidateRecord(
        rule=rule,
        objective=12.0,
        iteration=1,
        total_tardiness=5.0,
        weighted_service_shortfall=7.0,
    )
    restored = generation.CandidateRecord.from_dict(record.to_dict())
    assert restored.total_tardiness == 5.0
    assert restored.weighted_service_shortfall == 7.0


def test_feedback_reports_decomposition_and_classic_gaps():
    rule = generation.SymbolicRule(
        name="feedback_rule",
        expression="feedback test",
        rule_type=generation.RuleType.TYPE_I,
        priority_expr="gap_pressure",
    )
    record = generation.CandidateRecord(
        rule=rule,
        objective=90.0,
        iteration=0,
        total_tardiness=30.0,
        weighted_service_shortfall=60.0,
    )
    feedback = generation._feedback(
        [record],
        {
            "SPT": {"F": 100.0, "TT": 40.0, "WSF": 60.0},
            "ATC": {"F": 80.0, "TT": 30.0, "WSF": 50.0},
        },
    )
    assert "TT=30.000000" in feedback
    assert "WSF=60.000000" in feedback
    assert "gap vs SPT=-10.00%" in feedback
    assert "gap vs ATC=+12.50%" in feedback


def test_dry_run_exports_loadable_libraries_and_trains_two_updates(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(generation, "SELECTION_SEEDS", (600,))
    monkeypatch.setattr(generation, "HOLDOUT_SEEDS", (700,))
    monkeypatch.setattr(
        generation,
        "make_generation_instance",
        lambda _seed: make_test_instance(),
    )
    monkeypatch.setattr(
        generation,
        "compute_behavior_similarity",
        lambda left, right, _instances: (
            1.0
            if generation._rule_signature(left)
            == generation._rule_signature(right)
            else 0.25
        ),
    )
    generation_args = argparse.Namespace(
        tag="drytest",
        model="dry",
        temperature=0.8,
        rounds=2,
        per_round=10,
        library_size=6,
        max_tokens=256,
        timeout=5.0,
        enable_thinking="false",
        dry_run=True,
        output_dir=str(tmp_path),
        log_root=str(tmp_path / "llm_logs"),
    )
    six_path, three_path = generation.run_generation(generation_args)

    six = RuleLibrary.from_json(str(six_path))
    three = RuleLibrary.from_json(str(three_path))
    assert six.action_size() == 6
    assert three.action_size() == 3
    assert [rule.name for rule in three.rules] == [
        rule.name for rule in six.rules[:3]
    ]

    state = json.loads(
        (tmp_path / "llm_candidates_drytest.json").read_text(
            encoding="utf-8"
        )
    )
    assert state["reason_counts"]["unsafe_expression"] >= 1
    assert state["reason_counts"]["constant_priority"] >= 1
    assert state["reason_counts"]["duplicate_definition"] >= 1

    monkeypatch.setattr(train_streaming, "VALIDATION_SEEDS", (100,))
    monkeypatch.setattr(
        train_streaming,
        "make_stream_instance",
        lambda _seed: make_test_instance(),
    )
    train_args = argparse.Namespace(
        library=None,
        library_file=str(six_path),
        tag="external_smoke",
        episodes=3,
        episodes_per_update=1,
        eval_interval=10,
        resume=False,
        max_updates_this_run=2,
        file_prefix="external_smoke_",
        best_checkpoint_path=str(tmp_path / "best.pt"),
        last_checkpoint_path=str(tmp_path / "last.pt"),
        seed=123,
        base_seed=1000,
    )
    train_streaming.run_training(train_args)
    checkpoint = train_streaming.load_checkpoint(tmp_path / "last.pt")
    assert checkpoint["completed_updates"] == 2
    assert checkpoint["seed"] == 123
    assert checkpoint["base_seed"] == 1000
    assert checkpoint["library_source"]["kind"] == "json"
