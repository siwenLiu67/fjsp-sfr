import random

import pytest
import torch

from conftest import make_test_instance

from fjsp_sfr.dispatch.rule_library import RuleLibrary
from fjsp_sfr.dispatch.symbolic_rule import RuleType, SymbolicRule
from fjsp_sfr.rl.graph import HeteroGraphData
from fjsp_sfr.rl.ppo_agent import HGTPPOAgent
from fjsp_sfr.rl.ppo_trainer import PPOTrainer, standardize_advantages
from fjsp_sfr.rl.rollout_buffer import RolloutBuffer, RolloutStep


def _dummy_graph() -> HeteroGraphData:
    return HeteroGraphData(
        job_x=torch.zeros(1, 11),
        machine_x=torch.zeros(1, 2),
        checkpoint_x=torch.zeros(0, 8),
        job_machine_edge_index=torch.zeros(2, 0, dtype=torch.long),
        job_machine_edge_attr=torch.zeros(0, 1),
        job_checkpoint_edge_index=torch.zeros(2, 0, dtype=torch.long),
        job_checkpoint_edge_attr=torch.zeros(0, 2),
        global_x=torch.zeros(7),
        num_jobs=1,
        num_machines=1,
        num_checkpoints=0,
    )


class TestRolloutBuffer:
    def test_gae_single_episode(self):
        graph = _dummy_graph()
        mask = torch.ones(3, dtype=torch.bool)
        buffer = RolloutBuffer()
        buffer.add(RolloutStep(graph, 0, -1.0, 5.0, 1.0, False, mask))
        buffer.add(RolloutStep(graph, 1, -1.0, 4.0, 2.0, False, mask))
        buffer.add(RolloutStep(graph, 2, -1.0, 2.0, 3.0, True, mask))

        returns, advantages = buffer.compute_returns_and_advantages(
            gamma=1.0, gae_lambda=1.0
        )
        assert returns == pytest.approx([6.0, 5.0, 3.0])
        assert advantages == pytest.approx([1.0, 1.0, 1.0])

    def test_gae_episode_boundary(self):
        graph = _dummy_graph()
        mask = torch.ones(2, dtype=torch.bool)
        buffer = RolloutBuffer()
        buffer.add(RolloutStep(graph, 0, -1.0, 1.0, 2.0, False, mask))
        buffer.add(RolloutStep(graph, 0, -1.0, 1.0, 3.0, True, mask))
        buffer.add(RolloutStep(graph, 0, -1.0, 2.0, 4.0, True, mask))

        returns, advantages = buffer.compute_returns_and_advantages(
            gamma=1.0, gae_lambda=1.0
        )
        assert returns == pytest.approx([5.0, 3.0, 4.0])
        assert advantages == pytest.approx([4.0, 2.0, 2.0])


@pytest.fixture
def trainer() -> PPOTrainer:
    rules = (
        SymbolicRule(
            name="r1",
            expression="gap",
            rule_type=RuleType.TYPE_I,
            priority_expr="gap_pressure",
        ),
        SymbolicRule(
            name="r2",
            expression="tard",
            rule_type=RuleType.TYPE_I,
            priority_expr="-tardiness_exposure",
        ),
        SymbolicRule(
            name="r3",
            expression="slack",
            rule_type=RuleType.TYPE_I,
            priority_expr="-due_date_slack",
        ),
    )
    library = RuleLibrary(rules=rules)
    agent = HGTPPOAgent(
        action_size=3,
        hidden_dim=16,
        num_layers=1,
        num_heads=4,
    )
    return PPOTrainer(
        agent=agent,
        library=library,
        instances=[make_test_instance()],
        lr=1e-3,
        epochs_per_update=2,
        f_ref=1.0,
    )


class TestPPOTrainer:
    def test_advantages_are_standardized_over_batch(self):
        normalized, raw_std = standardize_advantages([1.0, 2.0, 5.0, 8.0])

        assert raw_std == pytest.approx(
            torch.tensor([1.0, 2.0, 5.0, 8.0]).std(unbiased=False).item()
        )
        assert normalized.mean().item() == pytest.approx(0.0, abs=1e-7)
        assert normalized.std(unbiased=False).item() == pytest.approx(1.0)

    def test_f_ref_sequence_is_aligned_with_instances(self, trainer):
        second_instance = make_test_instance()
        aligned = PPOTrainer(
            agent=trainer.agent,
            library=trainer.library,
            instances=[make_test_instance(), second_instance],
            f_ref=[2.0, 5.0],
        )

        assert aligned.f_refs == (2.0, 5.0)
        with pytest.raises(ValueError, match="length must match"):
            PPOTrainer(
                agent=trainer.agent,
                library=trainer.library,
                instances=[make_test_instance(), second_instance],
                f_ref=[2.0],
            )

    def test_reward_scale_multiplies_collected_rewards(self, trainer):
        random_state = random.getstate()
        torch_state = torch.random.get_rng_state()
        try:
            random.seed(17)
            torch.manual_seed(17)
            trainer.reward_scale = 1.0
            base_buffer, base_mean = trainer.collect_rollout(num_episodes=1)

            random.seed(17)
            torch.manual_seed(17)
            trainer.reward_scale = 10.0
            scaled_buffer, scaled_mean = trainer.collect_rollout(num_episodes=1)
        finally:
            random.setstate(random_state)
            torch.random.set_rng_state(torch_state)

        assert [step.action for step in scaled_buffer.steps] == [
            step.action for step in base_buffer.steps
        ]
        assert [step.reward for step in scaled_buffer.steps] == pytest.approx(
            [step.reward * 10.0 for step in base_buffer.steps]
        )
        assert scaled_mean == pytest.approx(base_mean * 10.0)

    def test_instance_factory_uses_distinct_reproducible_seeds(self, trainer):
        first_seeds = []
        second_seeds = []

        def first_factory(seed):
            first_seeds.append(seed)
            return make_test_instance()

        def second_factory(seed):
            second_seeds.append(seed)
            return make_test_instance()

        first = PPOTrainer(
            agent=trainer.agent,
            library=trainer.library,
            instances=[],
            instance_factory=first_factory,
            base_seed=1000,
            epochs_per_update=1,
        )
        second = PPOTrainer(
            agent=trainer.agent,
            library=trainer.library,
            instances=[],
            instance_factory=second_factory,
            base_seed=1000,
            epochs_per_update=1,
        )

        first.collect_rollout(num_episodes=2)
        first.collect_rollout(num_episodes=1)
        second.collect_rollout(num_episodes=3)

        assert first_seeds == [1000, 1001, 1002]
        assert second_seeds == first_seeds

    def test_instance_factory_f_ref_is_computed_from_all_rules(self, trainer):
        instance = make_test_instance()
        expected = []
        for rule_index, rule in enumerate(trainer.library.rules):
            from copy import deepcopy

            from fjsp_sfr.core.objective import compute_objective
            from fjsp_sfr.env.ca_rule_selection import CAPPORuleSelectionEnv

            rollout_instance = deepcopy(instance)
            env = CAPPORuleSelectionEnv(rollout_instance)
            env.reset()
            while env.current_metrics and env.current_metrics.is_active:
                result = env.step_principle(rule_index, rule)
                if result.terminated:
                    break
            expected.append(
                compute_objective(rollout_instance, env._state).Z
            )

        streaming = PPOTrainer(
            agent=trainer.agent,
            library=trainer.library,
            instances=[],
            instance_factory=lambda _seed: instance,
        )

        assert streaming._compute_instance_f_ref(instance) == pytest.approx(
            sum(expected) / len(expected)
        )

    def test_entropy_schedule_has_exact_endpoints(self, trainer):
        scheduled = PPOTrainer(
            agent=trainer.agent,
            library=trainer.library,
            instances=[make_test_instance()],
            entropy_coef_start=0.02,
            entropy_coef_end=0.002,
        )

        assert scheduled.entropy_coef_at(0, 625) == pytest.approx(0.02)
        assert scheduled.entropy_coef_at(624, 625) == pytest.approx(0.002)
        assert scheduled.entropy_coef_at(312, 625) == pytest.approx(0.011)

    def test_collect_rollout(self, trainer):
        buffer, mean_reward = trainer.collect_rollout(num_episodes=1)
        assert len(buffer) > 0
        assert isinstance(mean_reward, float)

    def test_update_runs(self, trainer):
        buffer, _mean_reward = trainer.collect_rollout(num_episodes=1)
        stats = trainer.update(buffer)
        assert not (stats.policy_loss != stats.policy_loss)
        assert not (stats.value_loss != stats.value_loss)

    def test_train_one_update(self, trainer):
        stats = trainer.train(total_updates=1, episodes_per_update=1)
        assert len(stats) >= 1

    def test_agent_parameters_change(self, trainer):
        parameters_before = {
            name: parameter.clone()
            for name, parameter in trainer.agent.named_parameters()
        }
        trainer.train(total_updates=2, episodes_per_update=1)
        assert any(
            not torch.equal(parameter, parameters_before[name])
            for name, parameter in trainer.agent.named_parameters()
        )
