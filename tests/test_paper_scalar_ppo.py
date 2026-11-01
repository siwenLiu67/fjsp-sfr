"""P3-A contracts for grouped Actor probabilities and scalar PPO training."""

from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from conftest import make_test_instance

from fjsp_sfr.dispatch.classic_rules import ClassicRule, ClassicRuleKind
from fjsp_sfr.dispatch.rule_library import RuleLibrary
from fjsp_sfr.env.ca_rule_selection import CAPPORuleSelectionEnv
from fjsp_sfr.rl.action_groups import aggregate_rule_log_probs
from fjsp_sfr.rl.graph import PAPER_GRAPH
from fjsp_sfr.rl.ppo_agent import HGTPPOAgent
from fjsp_sfr.rl.ppo_trainer import PPOTrainer, concrete_rule_action_groups
from fjsp_sfr.experiments.formal_matrix import RunSpec, command_for


def _paper_instance():
    instance = make_test_instance()
    instance.metadata["feature_normalization_scales"] = {
        "bar_p": 3.0,
        "bar_W": 6.0,
    }
    return instance


def _duplicate_spt_library(size: int = 3) -> RuleLibrary:
    return RuleLibrary(
        tuple(
            ClassicRule(f"SPT_{index}", ClassicRuleKind.SPT)
            for index in range(size)
        )
    )


def _formal_trainer(library: RuleLibrary, *, merge: bool = True) -> PPOTrainer:
    agent = HGTPPOAgent(
        action_size=library.action_size(), encoder_type="crg_gatv2"
    )
    return PPOTrainer(
        agent,
        library,
        instances=[_paper_instance()],
        epochs_per_update=1,
        gamma=1.0,
        gae_lambda=0.95,
        f_ref=20.0,
        reward_scale=10.0,
        feature_set="paper14",
        graph_config=PAPER_GRAPH,
        reward_reference_mode="global_calibration",
        action_equivalence_merge=merge,
    )


def test_group_probability_is_exact_sum_of_rule_probabilities():
    probabilities = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)
    grouped = aggregate_rule_log_probs(
        probabilities.log(), ((0, 2), (1, 3)), torch.ones(4, dtype=torch.bool)
    ).exp()
    assert grouped.tolist() == pytest.approx([0.4, 0.6], abs=1e-12)
    assert grouped.sum().item() == pytest.approx(1.0, abs=1e-12)


def test_action_groups_use_identical_real_job_operation_machine_dispatch():
    library = _duplicate_spt_library()
    trainer = _formal_trainer(library)
    env = CAPPORuleSelectionEnv(deepcopy(trainer.instances[0]))
    env.reset()
    _graph, rows = trainer._build_graph_for_env(env)
    mask = trainer._get_action_mask(env)
    groups, candidate_indices, dispatches = concrete_rule_action_groups(
        env, library, rows, mask, merge_equivalent=True
    )
    assert groups == ((0, 1, 2),)
    assert len(set(dispatches)) == 1
    result = env.step_candidate(candidate_indices[groups[0][0]])
    actual = (
        result.decision.job_id,
        result.decision.operation_id,
        result.decision.machine_id,
    )
    assert actual == dispatches[0]


def test_rollout_behavior_log_probability_recomputes_exactly_before_update():
    trainer = _formal_trainer(_duplicate_spt_library())
    torch.manual_seed(23)
    buffer, _reward = trainer.collect_rollout(2)
    recomputed = trainer.recompute_stored_log_probs(buffer)
    assert len(buffer) > 0
    assert [step.log_prob for step in buffer.steps] == pytest.approx(
        recomputed, abs=5e-7
    )
    assert all(step.action_groups for step in buffer.steps)


def test_single_dispatch_class_has_zero_actor_gradient_but_value_updates():
    logits = torch.tensor([0.2, -0.3, 1.1], requires_grad=True)
    grouped = aggregate_rule_log_probs(
        torch.log_softmax(logits, dim=0),
        ((0, 1, 2),),
        torch.ones(3, dtype=torch.bool),
    )
    (-grouped[0]).backward()
    assert logits.grad is not None
    assert torch.allclose(logits.grad, torch.zeros_like(logits.grad), atol=1e-7)

    trainer = _formal_trainer(_duplicate_spt_library())
    before = trainer.agent.value_head.weight.detach().clone()
    buffer, _reward = trainer.collect_rollout(1)
    assert all(len(step.action_groups) == 1 for step in buffer.steps)
    stats = trainer.update(buffer)
    assert torch.isfinite(torch.tensor(stats.value_loss))
    assert not torch.equal(before, trainer.agent.value_head.weight.detach())


def test_formal_scalar_ppo_smoke_and_final_state_reload(tmp_path):
    library = RuleLibrary(
        (
            ClassicRule("SPT", ClassicRuleKind.SPT),
            ClassicRule("EDD", ClassicRuleKind.EDD),
            ClassicRule("ATC", ClassicRuleKind.ATC),
        )
    )
    trainer = _formal_trainer(library)
    stats = trainer.train(total_updates=1, episodes_per_update=3)
    assert len(stats) == 1
    assert all(
        torch.isfinite(torch.tensor(value))
        for value in (
            stats[0].policy_loss,
            stats[0].value_loss,
            stats[0].entropy,
            stats[0].approx_kl,
        )
    )
    assert trainer.gamma == 1.0
    assert trainer.gae_lambda == 0.95
    assert trainer.global_f_ref == 20.0
    assert trainer.reward_scale == 10.0

    checkpoint = tmp_path / "final.pt"
    torch.save(
        {
            "agent_state_dict": trainer.agent.state_dict(),
            "checkpoint_semantics": "final_training",
            "f_ref": trainer.global_f_ref,
            "action_equivalence_merge": True,
        },
        checkpoint,
    )
    restored = HGTPPOAgent(action_size=3, encoder_type="crg_gatv2")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    restored.load_state_dict(payload["agent_state_dict"])
    env = CAPPORuleSelectionEnv(_paper_instance())
    env.reset()
    graph, rows = trainer._build_graph_for_env(env)
    mask = trainer._get_action_mask(env)
    groups, _indices, _dispatches = concrete_rule_action_groups(
        env, library, rows, mask, merge_equivalent=True
    )
    with torch.no_grad():
        original, _ = trainer.agent(graph, mask)
        reloaded, _ = restored(graph, mask)
    assert torch.equal(original, reloaded)
    assert aggregate_rule_log_probs(original, groups, mask).argmax().item() == (
        aggregate_rule_log_probs(reloaded, groups, mask).argmax().item()
    )
    assert payload["checkpoint_semantics"] == "final_training"


def test_action_equivalence_can_be_disabled_for_ablation():
    trainer = _formal_trainer(_duplicate_spt_library(), merge=False)
    buffer, _reward = trainer.collect_rollout(1)
    assert all(step.action_groups == () for step in buffer.steps)


def test_cross_scale_formal_command_loads_final_not_interim_checkpoint(tmp_path):
    command = command_for(
        RunSpec(
            name="generalization",
            library_id="pool_mix",
            library_file="experiments/library.json",
            scale="M",
            seed=1,
            eval_scale="X",
            checkpoint_run="source-run",
        ),
        tmp_path,
        smoke=True,
    )
    checkpoint = command[command.index("--checkpoint") + 1]
    assert checkpoint.endswith("source-run/final.pt")

