"""P4 contracts for D1/D2 quantile Critics and matched PPO training."""

from __future__ import annotations

from copy import deepcopy
import math

import pytest
import torch

from conftest import make_test_instance

from fjsp_sfr.dispatch.classic_rules import ClassicRule, ClassicRuleKind
from fjsp_sfr.dispatch.rule_library import RuleLibrary
from fjsp_sfr.env.ca_rule_selection import CAPPORuleSelectionEnv
from fjsp_sfr.experiments.formal_experiment import parse_args
from fjsp_sfr.rl.cc_dvl import (
    CCDVLCritic,
    QuantileCriticOutput,
    ordered_interval_positions,
    pinball_loss,
    quantile_levels,
)
from fjsp_sfr.rl.certificate_critic import (
    PRIVILEGED_FEATURE_NAMES,
    CriticStepContext,
    CriticTargets,
)
from fjsp_sfr.rl.graph import PAPER_GRAPH
from fjsp_sfr.rl.ppo_agent import HGTPPOAgent
from fjsp_sfr.rl.ppo_trainer import PPOTrainer, build_critic_step_context


def _context(
    lower=(1.0, 2.0),
    upper=(3.0, 8.0),
    weights=(2.0, 1.0),
) -> CriticStepContext:
    count = len(lower)
    online = (
        torch.tensor([[0.0, 1.0]] * count)
        if count
        else torch.empty((0, 2))
    )
    return CriticStepContext(
        tuple((0, index) for index in range(count)),
        torch.tensor(weights, dtype=torch.float32),
        torch.full((count,), 10.0),
        online,
        torch.zeros(count, len(PRIVILEGED_FEATURE_NAMES)),
        torch.tensor(lower, dtype=torch.float32),
        torch.tensor(upper, dtype=torch.float32),
    )


def _critic(variant: str, *, k: int = 3, alpha_q: float = 1.0, hidden=8):
    return CCDVLCritic(
        variant,
        hidden_dim=hidden,
        num_quantiles=k,
        quantile_loss_coef=alpha_q,
    )


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
    kwargs = {}
    if variant in {"d1", "d2"}:
        kwargs = {"num_quantiles": 3, "quantile_loss_coef": 1.0}
    agent = HGTPPOAgent(
        3,
        encoder_type="crg_gatv2",
        critic_variant=variant,
        **kwargs,
    )
    return PPOTrainer(
        agent,
        _library(),
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


def test_d1_d2_quantiles_are_ordered_and_only_d2_is_certificate_bounded():
    torch.manual_seed(4)
    context = _context()
    state = torch.randn(8)
    checkpoints = torch.randn(2, 8)
    d1 = _critic("d1")
    d2 = _critic("d2")
    d1_output = d1(state, checkpoints, context, f_ref=20.0, reward_scale=10.0)
    d2_output = d2(state, checkpoints, context, f_ref=20.0, reward_scale=10.0)
    assert torch.all(d1_output.checkpoint_quantiles[:, 1:] >= d1_output.checkpoint_quantiles[:, :-1])
    assert torch.all(d1_output.checkpoint_quantiles >= 0.0)
    assert torch.all(d2_output.checkpoint_quantiles[:, 1:] >= d2_output.checkpoint_quantiles[:, :-1])
    assert torch.all(d2_output.checkpoint_quantiles >= context.privileged_lower[:, None])
    assert torch.all(d2_output.checkpoint_quantiles <= context.privileged_upper[:, None])

    # D1 receives both endpoints as inputs but they never clip its output.
    with torch.no_grad():
        d1.quantile_head[-1].weight.zero_()
        d1.quantile_head[-1].bias.fill_(10.0)
    unbounded = d1(state, checkpoints, context, f_ref=20.0, reward_scale=10.0)
    assert bool((unbounded.checkpoint_quantiles[:, -1] > context.privileged_upper).all())


def test_d2_degenerate_interval_is_exact_and_quantile_head_gradient_is_zero():
    context = _context(lower=(3.0,), upper=(3.0,), weights=(2.0,))
    critic = _critic("d2")
    output = critic(
        torch.randn(8), torch.randn(1, 8), context, f_ref=20.0, reward_scale=10.0
    )
    assert torch.equal(output.checkpoint_quantiles, torch.full((1, 3), 3.0))
    targets = CriticTargets(6.0, 0.0, torch.tensor([3.0]))
    loss = critic.loss(output, targets, context, f_ref=20.0)
    loss.backward()
    assert all(
        parameter.grad is None or torch.count_nonzero(parameter.grad).item() == 0
        for parameter in critic.quantile_head.parameters()
    )


@pytest.mark.parametrize(
    "context, target",
    (
        (_context(lower=(10.0,), upper=(10.000001,), weights=(5.0,)), 10.0000005),
        (_context(lower=(1_000_000.0,), upper=(1_000_010.0,), weights=(5.0,)), 1_000_005.0),
        (_context(lower=(), upper=(), weights=()), None),
    ),
)
def test_d2_narrow_large_and_empty_checkpoint_cases_are_finite(context, target):
    critic = _critic("d2")
    count = len(context.checkpoint_keys)
    output = critic(
        torch.randn(8),
        torch.randn(count, 8),
        context,
        f_ref=100.0,
        reward_scale=10.0,
    )
    labels = torch.empty(0) if target is None else torch.tensor([target])
    loss = critic.loss(
        output,
        CriticTargets(float(target or 0.0), 7.0, labels),
        context,
        f_ref=100.0,
    )
    loss.backward()
    assert torch.isfinite(output.total_value)
    assert torch.isfinite(loss)
    assert all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in critic.parameters()
    )


def test_hand_computed_pinball_mean_total_value_and_combined_loss():
    context = _context(lower=(1.0,), upper=(3.0,), weights=(3.0,))
    quantiles = torch.tensor([[1.4, 2.0, 2.6]], requires_grad=True)
    # Mean shortfall=2, hence cost=TT 4 + weight 3 * 2 = 10 and V=-10.
    output = QuantileCriticOutput(
        total_value=torch.tensor(-10.0),
        total_cost=torch.tensor(10.0),
        remaining_tardiness=torch.tensor(4.0, requires_grad=True),
        checkpoint_quantiles=quantiles,
    )
    critic = _critic("d2", alpha_q=2.0)
    targets = CriticTargets(0.0, 6.0, torch.tensor([2.5]))
    loss = critic.loss(output, targets, context, f_ref=10.0)
    # rho values: 1.1/6, 0.5/2, 0.1/6; mean=0.15.
    expected_quantile = 3.0 / 10.0 * 0.15
    expected_tt = ((4.0 - 6.0) / 10.0) ** 2
    assert quantiles.mean().item() == pytest.approx(2.0)
    assert output.total_value.item() == pytest.approx(-10.0)
    assert loss.item() == pytest.approx(expected_tt + 2.0 * expected_quantile)


def test_original_scale_and_width_rescaled_normalized_pinball_are_identical():
    width = torch.tensor([2.0, 0.001, 40.0])
    lower = torch.tensor([1.0, 8.0, 100.0])
    positions = torch.tensor([0.2, 0.5, 0.8])
    truth = torch.tensor([2.3, 8.0007, 117.0])
    prediction = lower + width * positions
    levels = quantile_levels(3)
    raw = pinball_loss(truth - prediction, levels)
    normalized = width * pinball_loss(
        (truth - lower) / width - positions, levels
    )
    assert torch.allclose(raw, normalized, atol=1e-6, rtol=1e-6)


def test_d2_rejects_false_certificate_instead_of_clipping_label():
    context = _context(lower=(1.0,), upper=(3.0,), weights=(1.0,))
    critic = _critic("d2")
    output = critic(
        torch.randn(8), torch.randn(1, 8), context, f_ref=20.0, reward_scale=10.0
    )
    with pytest.raises(ValueError, match="violates privileged certificate"):
        critic.loss(
            output,
            CriticTargets(0.0, 0.0, torch.tensor([3.5])),
            context,
            f_ref=20.0,
        )


def test_d1_d2_use_identical_private_fields_context_and_true_labels():
    context = _context()
    targets = CriticTargets(10.0, 4.0, torch.tensor([2.0, 5.0]))
    state, checkpoints = torch.randn(8), torch.randn(2, 8)
    outputs = []
    for variant in ("d1", "d2"):
        critic = _critic(variant)
        assert critic.uses_privileged_inputs
        assert critic.quantile_head[0].in_features == (
            2 * 8 + 2 + len(PRIVILEGED_FEATURE_NAMES)
        )
        output = critic(
            state, checkpoints, context, f_ref=20.0, reward_scale=10.0
        )
        assert torch.isfinite(critic.loss(output, targets, context, f_ref=20.0))
        outputs.append(output)
    assert outputs[0].checkpoint_quantiles.shape == outputs[1].checkpoint_quantiles.shape


def test_d2_actor_is_private_invariant_and_actor_loss_has_no_critic_gradient():
    trainer = _trainer("d2")
    env = CAPPORuleSelectionEnv(
        deepcopy(trainer.instances[0]), enable_privileged_certificates=True
    )
    env.reset()
    graph, _rows = trainer._build_graph_for_env(env)
    mask = trainer._get_action_mask(env)
    context = build_critic_step_context(env, graph, "d2")
    assert context is not None and context.privileged_features is not None
    original = trainer.agent.actor_log_probs(graph, mask)
    changed = CriticStepContext(
        context.checkpoint_keys,
        context.weights,
        context.required_quantities,
        context.online_endpoints,
        context.privileged_features + 99.0,
        context.privileged_lower,
        context.privileged_upper,
    )
    trainer.agent.forward_with_critic(
        graph, mask, changed, f_ref=20.0, reward_scale=10.0
    )
    assert torch.equal(original, trainer.agent.actor_log_probs(graph, mask))
    assert trainer.agent.certificate_critic is not None
    gradients = torch.autograd.grad(
        -original[0],
        tuple(trainer.agent.certificate_critic.parameters()),
        allow_unused=True,
    )
    assert all(gradient is None for gradient in gradients)


@pytest.mark.parametrize("variant", ("c4", "d1", "d2"))
def test_c4_d1_d2_matched_smoke_save_reload_and_public_actor_inference(
    variant, tmp_path
):
    torch.manual_seed(31)
    trainer = _trainer(variant)
    stats = trainer.train(total_updates=1, episodes_per_update=2)
    assert len(stats) == 1
    assert stats[0].environment_interactions == trainer.environment_interactions
    assert stats[0].optimizer_updates == trainer.optimizer_update_count == 1
    assert all(
        math.isfinite(value)
        for value in (
            stats[0].value_loss,
            stats[0].gradient_norm,
            stats[0].critic_gradient_norm,
        )
    )
    path = tmp_path / f"{variant}.pt"
    torch.save(trainer.agent.state_dict(), path)
    kwargs = (
        {"num_quantiles": 3, "quantile_loss_coef": 1.0}
        if variant in {"d1", "d2"}
        else {}
    )
    restored = HGTPPOAgent(
        3, encoder_type="crg_gatv2", critic_variant=variant, **kwargs
    )
    restored.load_state_dict(torch.load(path, weights_only=True))
    env = CAPPORuleSelectionEnv(_instance())
    env.reset()
    graph, _rows = trainer._build_graph_for_env(env)
    mask = trainer._get_action_mask(env)
    with torch.no_grad():
        before = trainer.agent.actor_log_probs(graph, mask)
        after = restored.actor_log_probs(graph, mask)
    assert torch.equal(before, after)
    assert before.argmax().item() == after.argmax().item()


def test_d2_ppo_value_scale_gae_and_action_equivalence_are_unchanged():
    torch.manual_seed(41)
    trainer = _trainer("d2")
    buffer, _reward = trainer.collect_rollout(2)
    assert all(step.action_groups for step in buffer.steps)
    assert [step.log_prob for step in buffer.steps] == pytest.approx(
        trainer.recompute_stored_log_probs(buffer), abs=5e-7
    )
    returns, advantages = buffer.compute_returns_and_advantages(
        gamma=1.0, gae_lambda=0.95
    )
    assert len(returns) == len(advantages) == len(buffer.steps)
    assert all(math.isfinite(value) for value in (*returns, *advantages))
    first = buffer.steps[0]
    with torch.no_grad():
        _logs, output, _encoding = trainer.agent.forward_with_critic(
            first.graph,
            first.action_mask,
            first.critic_context,
            f_ref=first.f_ref,
            reward_scale=trainer.reward_scale,
        )
    assert output.total_value.item() == pytest.approx(
        -10.0 / first.f_ref * output.total_cost.item()
    )
    diagnostics = trainer.quantile_width_diagnostics()
    assert sum(int(row["count"]) for row in diagnostics.values()) > 0


def test_softmax_increment_positions_follow_midpoint_head_count_contract():
    logits = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
    positions = ordered_interval_positions(logits)
    assert positions.tolist()[0] == pytest.approx([0.25, 0.5, 0.75])
    assert quantile_levels(3).tolist() == pytest.approx([1 / 6, 1 / 2, 5 / 6])


def test_matched_c4_d1_d2_budget_diagnostics_record_equal_interactions():
    records = {}
    for variant in ("c4", "d1", "d2"):
        torch.manual_seed(77)
        trainer = _trainer(variant)
        torch.manual_seed(101)
        stats = trainer.train(total_updates=1, episodes_per_update=2)[0]
        records[variant] = {
            "parameters": sum(
                parameter.numel() for parameter in trainer.agent.parameters()
            ),
            "interactions": trainer.environment_interactions,
            "updates": trainer.optimizer_update_count,
            "critic_error": stats.value_loss,
            "critic_total_cost_mae": stats.critic_total_cost_mae,
            "critic_shortfall_mae": stats.critic_shortfall_mae,
            "gradient_norm": stats.critic_gradient_norm,
        }
    assert len({row["interactions"] for row in records.values()}) == 1
    assert len({row["updates"] for row in records.values()}) == 1
    assert all(row["parameters"] > 0 for row in records.values())
    assert all(math.isfinite(row["critic_error"]) for row in records.values())
    assert all(math.isfinite(row["critic_total_cost_mae"]) for row in records.values())
    assert all(math.isfinite(row["critic_shortfall_mae"]) for row in records.values())
    assert all(math.isfinite(row["gradient_norm"]) for row in records.values())


def test_formal_cli_requires_and_records_prespecified_quantile_settings():
    base = [
        "--contract-config", "paper_20261030",
        "--library-id", "test",
        "--library-file", "rules.json",
        "--scale", "M",
        "--seed", "1",
        "--selector", "hgt_ppo",
        "--critic-variant", "d2",
        "--output-dir", "out",
    ]
    with pytest.raises(SystemExit):
        parse_args(base)
    args = parse_args(
        base + ["--num-quantiles", "7", "--quantile-loss-coef", "0.5"]
    )
    assert args.num_quantiles == 7
    assert args.quantile_loss_coef == pytest.approx(0.5)
