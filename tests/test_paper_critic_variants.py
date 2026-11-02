"""P3-B contracts for the C1--C4 scalar Critic variants."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from conftest import make_test_instance

from fjsp_sfr.dispatch.classic_rules import ClassicRule, ClassicRuleKind
from fjsp_sfr.dispatch.rule_library import RuleLibrary
from fjsp_sfr.env.ca_rule_selection import CAPPORuleSelectionEnv
from fjsp_sfr.experiments import formal_experiment
from fjsp_sfr.experiments.paperA_contract import LEGACY, PAPER_20261030
from fjsp_sfr.rl.certificate_critic import (
    PRIVILEGED_FEATURE_NAMES,
    CertificateCritic,
    CriticStepContext,
    CriticTargets,
)
from fjsp_sfr.rl.graph import PAPER_GRAPH
from fjsp_sfr.rl.ppo_agent import HGTPPOAgent
from fjsp_sfr.rl.ppo_trainer import (
    PPOTrainer,
    build_critic_step_context,
    finalize_episode_critic_targets,
)
from fjsp_sfr.rl.rollout_buffer import RolloutStep


def _instance():
    instance = make_test_instance()
    instance.metadata["feature_normalization_scales"] = {
        "bar_p": 3.0,
        "bar_W": 6.0,
    }
    return instance


def _library() -> RuleLibrary:
    return RuleLibrary(
        (
            ClassicRule("SPT", ClassicRuleKind.SPT),
            ClassicRule("EDD", ClassicRuleKind.EDD),
            ClassicRule("ATC", ClassicRuleKind.ATC),
        )
    )


def _trainer(variant: str) -> PPOTrainer:
    library = _library()
    agent = HGTPPOAgent(
        library.action_size(),
        encoder_type="crg_gatv2",
        critic_variant=variant,
    )
    return PPOTrainer(
        agent,
        library,
        instances=[_instance()],
        epochs_per_update=1,
        gamma=1.0,
        gae_lambda=0.95,
        f_ref=20.0,
        reward_scale=10.0,
        feature_set="paper14",
        graph_config=PAPER_GRAPH,
        reward_reference_mode="global_calibration",
        action_equivalence_merge=True,
    )


@pytest.mark.parametrize("variant", ("c1", "c2", "c3", "c4"))
def test_each_scalar_critic_completes_short_update_with_finite_gradients(variant):
    torch.manual_seed(12)
    trainer = _trainer(variant)
    before = {
        name: parameter.detach().clone()
        for name, parameter in trainer.agent.named_parameters()
        if "certificate_critic" in name
    }
    stats = trainer.train(total_updates=1, episodes_per_update=2)
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
    assert any(
        not torch.equal(before[name], parameter.detach())
        for name, parameter in trainer.agent.named_parameters()
        if name in before
    )
    assert all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in trainer.agent.parameters()
    )


def test_c2_rejects_private_data_and_c3_c4_share_exact_private_layout():
    public = CriticStepContext(
        ((0, 0),),
        torch.tensor([1.0]),
        torch.tensor([5.0]),
        torch.tensor([[0.0, 1.0]]),
    )
    public.validate(privileged=False)
    private = CriticStepContext(
        public.checkpoint_keys,
        public.weights,
        public.required_quantities,
        public.online_endpoints,
        torch.zeros(1, len(PRIVILEGED_FEATURE_NAMES)),
        torch.tensor([0.0]),
        torch.tensor([5.0]),
    )
    with pytest.raises(ValueError, match="must not receive privileged"):
        private.validate(privileged=False)

    trainer = _trainer("c3")
    env = CAPPORuleSelectionEnv(
        deepcopy(trainer.instances[0]), enable_privileged_certificates=True
    )
    env.reset()
    graph, _rows = trainer._build_graph_for_env(env)
    c3 = build_critic_step_context(env, graph, "c3")
    c4 = build_critic_step_context(env, graph, "c4")
    assert c3 is not None and c4 is not None
    assert c3.checkpoint_keys == c4.checkpoint_keys
    for name in (
        "weights",
        "required_quantities",
        "online_endpoints",
        "privileged_features",
        "privileged_lower",
        "privileged_upper",
    ):
        assert torch.equal(getattr(c3, name), getattr(c4, name))


def test_c4_prediction_is_strictly_inside_certificate_and_degenerate_is_exact():
    critic = CertificateCritic("c4", hidden_dim=8)
    context = CriticStepContext(
        ((0, 0), (0, 1)),
        torch.tensor([1.0, 3.0]),
        torch.tensor([5.0, 9.0]),
        torch.zeros(2, 2),
        torch.zeros(2, len(PRIVILEGED_FEATURE_NAMES)),
        torch.tensor([2.0, 1.0]),
        torch.tensor([2.0, 6.0]),
    )
    output = critic(
        torch.randn(8), torch.randn(2, 8), context, f_ref=20.0, reward_scale=10.0
    )
    assert output.checkpoint_shortfalls[0].item() == pytest.approx(2.0)
    assert 1.0 <= output.checkpoint_shortfalls[1].item() <= 6.0
    assert output.total_value.item() == pytest.approx(
        -10.0 / 20.0 * output.total_cost.item()
    )
    invalid = CriticTargets(0.0, 0.0, torch.tensor([1.0, 3.0]))
    with pytest.raises(ValueError, match="violates privileged certificate"):
        critic.loss(output, invalid, context, f_ref=20.0)


def test_complete_trajectory_labels_are_exact_and_settled_heads_are_not_reused():
    def context(keys):
        count = len(keys)
        return CriticStepContext(
            keys,
            torch.ones(count),
            torch.ones(count),
            torch.zeros(count, 2),
        )

    graph = _trainer("c2").collect_rollout(1)[0].steps[0].graph
    mask = torch.ones(3, dtype=torch.bool)
    steps = [
        RolloutStep(
            graph, 0, 0.0, 0.0, 0.0, False, mask,
            critic_context=context(((0, 0), (0, 1))),
            immediate_total_cost=5.0,
            immediate_tardiness=2.0,
        ),
        RolloutStep(
            graph, 0, 0.0, 0.0, 0.0, True, mask,
            critic_context=context(((0, 1),)),
            immediate_total_cost=7.0,
            immediate_tardiness=3.0,
        ),
    ]
    objective = SimpleNamespace(
        Z=12.0,
        total_tardiness=5.0,
        per_checkpoint_shortfall={(0, 0): 2.0, (0, 1): 3.0},
    )
    finalize_episode_critic_targets(steps, objective)
    assert steps[0].critic_targets.remaining_total_cost == pytest.approx(12.0)
    assert steps[0].critic_targets.remaining_tardiness == pytest.approx(5.0)
    assert steps[0].critic_targets.checkpoint_shortfalls.tolist() == [2.0, 3.0]
    assert steps[1].critic_targets.remaining_total_cost == pytest.approx(7.0)
    assert steps[1].critic_targets.remaining_tardiness == pytest.approx(3.0)
    assert steps[1].critic_targets.checkpoint_shortfalls.tolist() == [3.0]


def test_actor_is_private_input_invariant_and_actor_loss_has_no_private_head_gradient():
    trainer = _trainer("c4")
    env = CAPPORuleSelectionEnv(
        deepcopy(trainer.instances[0]), enable_privileged_certificates=True
    )
    env.reset()
    graph, _rows = trainer._build_graph_for_env(env)
    mask = trainer._get_action_mask(env)
    context = build_critic_step_context(env, graph, "c4")
    assert context is not None
    first = trainer.agent.actor_log_probs(graph, mask)
    changed = CriticStepContext(
        context.checkpoint_keys,
        context.weights,
        context.required_quantities,
        context.online_endpoints,
        context.privileged_features + 100.0,
        context.privileged_lower,
        context.privileged_upper,
    )
    second = trainer.agent.actor_log_probs(graph, mask)
    assert torch.equal(first, second)
    # The changed private record is accepted by the Critic but has no Actor API path.
    trainer.agent.forward_with_critic(
        graph, mask, changed, f_ref=20.0, reward_scale=10.0
    )
    actor_loss = -first[0]
    assert trainer.agent.certificate_critic is not None
    private_grads = torch.autograd.grad(
        actor_loss,
        tuple(trainer.agent.certificate_critic.parameters()),
        allow_unused=True,
    )
    assert all(gradient is None for gradient in private_grads)


def test_c4_save_reload_preserves_public_policy_and_private_value(tmp_path):
    trainer = _trainer("c4")
    env = CAPPORuleSelectionEnv(
        deepcopy(trainer.instances[0]), enable_privileged_certificates=True
    )
    env.reset()
    graph, _rows = trainer._build_graph_for_env(env)
    mask = trainer._get_action_mask(env)
    context = build_critic_step_context(env, graph, "c4")
    with torch.no_grad():
        original_logs, original_value, _ = trainer.agent.forward_with_critic(
            graph, mask, context, f_ref=20.0, reward_scale=10.0
        )
    path = tmp_path / "final.pt"
    torch.save(trainer.agent.state_dict(), path)
    restored = HGTPPOAgent(3, encoder_type="crg_gatv2", critic_variant="c4")
    restored.load_state_dict(torch.load(path, weights_only=True))
    with torch.no_grad():
        restored_logs, restored_value, _ = restored.forward_with_critic(
            graph, mask, context, f_ref=20.0, reward_scale=10.0
        )
    assert torch.equal(original_logs, restored_logs)
    assert torch.equal(original_value.total_value, restored_value.total_value)
    assert original_logs.argmax().item() == restored_logs.argmax().item()


def test_legacy_selects_validation_best_but_paper_keeps_final(monkeypatch):
    class DummyAgent(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.marker = torch.nn.Parameter(torch.tensor(0.0))
            self.action_size = 1
            self.critic_variant = "legacy"

    class FakeTrainer:
        def __init__(self, agent, *args, **kwargs):
            self.agent = agent
            self.optimizer = torch.optim.SGD(agent.parameters(), lr=0.1)
            self.entropy_coef = 0.0

        def train(self, *_args, **_kwargs):
            with torch.no_grad():
                self.agent.marker.add_(1.0)

    monkeypatch.setattr(formal_experiment, "PPOTrainer", FakeTrainer)
    monkeypatch.setattr(
        formal_experiment,
        "make_agent",
        lambda *_args, **_kwargs: DummyAgent(),
    )
    monkeypatch.setattr(
        formal_experiment,
        "evaluate_set",
        lambda *_args, agent, **_kwargs: (
            [{"F": float(agent.marker.detach())}],
            {},
        ),
    )
    args = SimpleNamespace(
        selector="hgt_ppo",
        episodes=2,
        episodes_per_update=1,
        mask_checkpoint=False,
        mask_forecast=False,
        disable_action_equivalence_merge=False,
        critic_variant="c4",
    )
    legacy, _ = formal_experiment.train_ppo(
        args, _library(), lambda seed: _instance(), [_instance()], contract=LEGACY
    )
    assert legacy.marker.item() == pytest.approx(1.0)
    paper, _ = formal_experiment.train_ppo(
        args,
        _library(),
        lambda seed: _instance(),
        [_instance()],
        contract=PAPER_20261030,
        global_f_ref=20.0,
    )
    assert paper.marker.item() == pytest.approx(2.0)
