"""端到端冒烟测试：LLM 解析 → 规则库 → PPO 训练 → 评估。"""

import os
import tempfile

import torch

from conftest import make_test_instance

from fjsp_sfr.dispatch.rule_library import RuleLibrary
from fjsp_sfr.dispatch.symbolic_rule import RuleType, SymbolicRule
from fjsp_sfr.dispatch.theory_features import build_extended_theory_features
from fjsp_sfr.llm.generator import evaluate_rule_objective
from fjsp_sfr.llm.rule_parser import parse_llm_output
from fjsp_sfr.rl.ca_observation import resolve_observation_scales
from fjsp_sfr.rl.ca_rule_selection import CAPPORuleSelectionEnv
from fjsp_sfr.rl.graph import build_hetero_graph
from fjsp_sfr.rl.ppo_agent import HGTPPOAgent
from fjsp_sfr.rl.ppo_trainer import PPOTrainer


class TestEndToEnd:
    def test_parse_to_library_to_train(self):
        """解析规则文本 → 构建规则库 → PPO 训练 1 步 → 评估。"""

        llm_text = """
RULE: gap_tard_combo
TYPE: III
COND: NONE
FILTER: NONE
PRIORITY: 0.4 * gap_pressure + 0.3 * tardiness_exposure + 0.3 * forecast_deficit

RULE: urgent_rescue
TYPE: IV
COND: forecast_flag
FILTER: quantity_contribution - 0.1
PRIORITY: capacity_pressure + gap_pressure

RULE: slack_first
TYPE: I
COND: NONE
FILTER: NONE
PRIORITY: -due_date_slack
"""
        rules = parse_llm_output(llm_text)
        assert len(rules) == 3
        library = RuleLibrary(rules=tuple(rules))
        assert library.action_size() == 3

        agent = HGTPPOAgent(
            action_size=library.action_size(),
            hidden_dim=16,
            num_layers=1,
            num_heads=4,
        )
        instances = [make_test_instance()]
        trainer = PPOTrainer(
            agent=agent,
            library=library,
            instances=instances,
            lr=1e-3,
            epochs_per_update=2,
            f_ref=1.0,
        )
        stats = trainer.train(total_updates=2, episodes_per_update=1)
        assert stats
        assert not (stats[-1].policy_loss != stats[-1].policy_loss)

        env = CAPPORuleSelectionEnv(instances[0])
        env.reset()
        steps = 0
        while env.current_metrics and env.current_metrics.is_active:
            problem = env._problem_view()
            state = env._state
            metrics = env.current_metrics
            time_scale, processing_scale = resolve_observation_scales(
                problem, state
            )
            features = build_extended_theory_features(
                problem,
                state,
                metrics,
                time_scale=time_scale,
            )
            graph = build_hetero_graph(
                problem,
                state,
                metrics,
                features,
                time_scale=time_scale,
                processing_scale=processing_scale,
            )
            mask = torch.ones(library.action_size(), dtype=torch.bool)
            action = agent.greedy_action(graph, mask)
            result = env.step_principle(action, library.get_rule(action))
            steps += 1
            if result.terminated:
                break
        assert steps > 0

    def test_rule_serialization_round_trip(self):
        """规则库 JSON 往返后仍可训练。"""

        library = RuleLibrary(
            rules=(
                SymbolicRule(
                    name="a",
                    expression="a",
                    rule_type=RuleType.TYPE_I,
                    priority_expr="gap_pressure",
                ),
                SymbolicRule(
                    name="b",
                    expression="b",
                    rule_type=RuleType.TYPE_III,
                    priority_expr=(
                        "0.5*multi_checkpoint_value + 0.5*tardiness_exposure"
                    ),
                ),
            )
        )
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as stream:
            path = stream.name
        try:
            library.to_json(path)
            loaded = RuleLibrary.from_json(path)
            agent = HGTPPOAgent(
                action_size=loaded.action_size(),
                hidden_dim=16,
                num_layers=1,
                num_heads=4,
            )
            trainer = PPOTrainer(
                agent=agent,
                library=loaded,
                instances=[make_test_instance()],
                lr=1e-3,
                epochs_per_update=1,
                f_ref=1.0,
            )
            assert trainer.train(total_updates=1, episodes_per_update=1)
        finally:
            os.unlink(path)

    def test_evaluate_rule_objective_consistent(self):
        """同一规则与实例的 f_H 应完全确定。"""

        rule = SymbolicRule(
            name="det",
            expression="det",
            rule_type=RuleType.TYPE_I,
            priority_expr="-due_date_slack",
        )
        instances = [make_test_instance()]
        first = evaluate_rule_objective(rule, instances, num_episodes=1)
        second = evaluate_rule_objective(rule, instances, num_episodes=1)
        assert abs(first - second) < 1e-10
