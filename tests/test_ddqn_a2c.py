import math

import torch

from conftest import make_test_instance
from fjsp_sfr.dispatch.classic_rules import build_classic_rule_library
from fjsp_sfr.rl.a2c_agent import A2CTrainer, HGTA2CAgent
from fjsp_sfr.rl.ddqn_agent import DDQNTrainer, HGTDDQNAgent
from fjsp_sfr.rl.dqn_agent import DQNTransition
from fjsp_sfr.rl.graph import HeteroGraphData


def _graph():
    return HeteroGraphData(
        job_x=torch.rand(3, 14), machine_x=torch.rand(2, 2), checkpoint_x=torch.rand(2, 8),
        job_machine_edge_index=torch.empty((2, 0), dtype=torch.long), job_machine_edge_attr=torch.empty((0, 1)),
        job_checkpoint_edge_index=torch.empty((2, 0), dtype=torch.long), job_checkpoint_edge_attr=torch.empty((0, 2)),
        global_x=torch.rand(7), num_jobs=3, num_machines=2, num_checkpoints=2,
    )


def test_ddqn_forward_and_one_update_are_finite():
    agent = HGTDDQNAgent(4, hidden_dim=16, num_layers=1, num_heads=4, job_input_dim=14)
    assert agent(_graph()).shape == (4,)
    trainer = DDQNTrainer(agent, build_classic_rule_library(), lambda _seed: make_test_instance(), batch_size=2)
    graph = _graph()
    trainer.replay.extend((DQNTransition(graph, 0, -1.0, graph, False), DQNTransition(graph, 1, -0.5, None, True)))
    assert math.isfinite(trainer._learn())


def test_a2c_forward_and_one_update_are_finite():
    library = build_classic_rule_library()
    agent = HGTA2CAgent(action_size=library.action_size(), hidden_dim=16, num_layers=1, num_heads=4, job_input_dim=14)
    log_probs, value = agent(_graph(), torch.ones(library.action_size(), dtype=torch.bool))
    assert log_probs.shape == (library.action_size(),)
    assert value.ndim == 0
    trainer = A2CTrainer(agent, library, lambda _seed: make_test_instance(), lr=1e-4)
    stats = trainer.train(1, episodes_per_update=1)[0]
    assert all(math.isfinite(value) for value in (stats.mean_reward, stats.policy_loss, stats.value_loss, stats.entropy))
