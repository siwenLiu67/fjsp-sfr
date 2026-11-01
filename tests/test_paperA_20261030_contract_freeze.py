"""Authoritative Phase-0 contracts for the 2026-10-30 Paper A manuscript.

Tests for work assigned to later phases intentionally fail on the Phase-0
implementation.  They are executable gates, not xfails and not legacy oracles.
"""

from __future__ import annotations

import argparse
import random

import pytest

from fjsp_sfr.dispatch.classic_rules import build_classic_rule_library
from fjsp_sfr.env.ca_rule_selection import CAPPORuleSelectionEnv
from fjsp_sfr.experiments.formal_experiment import parse_args, run
from fjsp_sfr.experiments.formal_matrix import RunSpec, command_for
from fjsp_sfr.experiments.paperA_contract import (
    ContractNotReadyError,
    LEGACY,
    PAPER_20261030,
    PAPER_20261030_CONTRACT,
)
from fjsp_sfr.generation.scenario_generator import make_scenario_instance
from fjsp_sfr.generation.scale_config import make_scaled_instance
from fjsp_sfr.rl.graph import PAPER_GRAPH, build_hetero_graph
from fjsp_sfr.rl.ppo_agent import HGTPPOAgent
from fjsp_sfr.rl.ppo_trainer import PPOTrainer


def test_paper_contract_freezes_paper14_gamma_one_and_final_checkpoint():
    assert PAPER_20261030.feature_set == "paper14"
    assert PAPER_20261030.graph_config == PAPER_GRAPH
    assert PAPER_20261030.node_dimensions == (14, 2, 12)
    assert PAPER_20261030.encoder == "crg_gatv2"
    assert PAPER_20261030.gamma == 1.0
    assert PAPER_20261030.reward_reference == "global_calibration_spt_atc_mean"
    assert PAPER_20261030.evaluation_checkpoint == "final_training"


def test_contract_declares_action_equivalence_merge():
    assert PAPER_20261030.action_equivalence_merge is True
    assert LEGACY.action_equivalence_merge is False


def test_contract_no_longer_declares_phase_one_blockers():
    assert not any("certificates" in item for item in PAPER_20261030.blockers)
    assert not any("F_ref" in item for item in PAPER_20261030.blockers)


def test_contract_declares_crg_gatv2():
    assert PAPER_20261030.encoder == "crg_gatv2"
    assert not any("CRG" in item for item in PAPER_20261030.blockers)


def test_contract_declares_critic_variant_blocker():
    assert any("C1-C4" in item and "D1-D2" in item for item in PAPER_20261030.blockers)


def test_formal_entry_requires_explicit_contract_selection(tmp_path):
    common = [
        "--library-id",
        "classic4",
        "--library-file",
        "experiments/classic4_library.json",
        "--scale",
        "S",
        "--seed",
        "1",
        "--output-dir",
        str(tmp_path),
    ]
    with pytest.raises(SystemExit):
        parse_args(common)
    parsed = parse_args(["--contract-config", "legacy", *common])
    assert parsed.contract_config == "legacy"


def test_corrected_contract_fails_closed_instead_of_using_legacy():
    with pytest.raises(ContractNotReadyError, match=PAPER_20261030_CONTRACT):
        run(argparse.Namespace(contract_config=PAPER_20261030_CONTRACT))


def test_historical_matrix_explicitly_requests_legacy(tmp_path):
    command = command_for(
        RunSpec(
            name="legacy-smoke",
            library_id="classic4",
            library_file="experiments/classic4_library.json",
            scale="S",
            seed=1,
        ),
        tmp_path,
        smoke=True,
    )
    index = command.index("--contract-config")
    assert command[index + 1] == "legacy"


def test_authoritative_paper_graph_has_14_2_12_inputs():
    instance = make_scenario_instance("center", 25_000)
    env = CAPPORuleSelectionEnv(instance)
    env.reset()
    graph = build_hetero_graph(
        env._problem_view(),
        env._state,
        env.current_metrics,
        feature_set="paper14",
        graph_config=PAPER_GRAPH,
    )
    assert graph.job_x.shape[1] == 14
    assert graph.machine_x.shape[1] == 2
    assert graph.checkpoint_x.shape[1] == 12
    for row, key in enumerate(graph.checkpoint_keys):
        checkpoint = env.current_metrics.checkpoints[key]
        assert checkpoint.online_shortfall_lower is not None
        assert checkpoint.online_shortfall_upper is not None
        assert graph.checkpoint_x[row, 10].item() == pytest.approx(
            checkpoint.online_shortfall_lower / checkpoint.required_quantity
        )
        assert graph.checkpoint_x[row, 11].item() == pytest.approx(
            checkpoint.online_shortfall_upper / checkpoint.required_quantity
        )


def test_authoritative_streaming_ppo_does_not_recompute_f_ref_per_instance(
    monkeypatch,
):
    instance = make_scaled_instance(12_346, scale="S", difficulty="balanced")
    library = build_classic_rule_library()
    agent = HGTPPOAgent(action_size=library.action_size())
    trainer = PPOTrainer(
        agent,
        library,
        instances=[],
        instance_factory=lambda _seed: instance,
        base_seed=50_000,
        f_ref=123.0,
        reward_reference_mode="global_calibration",
    )

    def forbidden_per_instance_reference(_instance):
        pytest.fail("paper_20261030 must use one precomputed global F_ref")

    monkeypatch.setattr(
        trainer, "_compute_instance_f_ref", forbidden_per_instance_reference
    )
    random.seed(0)
    trainer.collect_rollout(1)
