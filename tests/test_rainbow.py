import math

import torch

from fjsp_sfr.dispatch.classic_rules import build_classic_rule_library
from fjsp_sfr.rl.graph import HeteroGraphData
from fjsp_sfr.rl.rainbow_agent import HGTRainbowAgent, RainbowTrainer, RainbowTransition


def _graph():
    return HeteroGraphData(
        job_x=torch.rand(3, 14), machine_x=torch.rand(2, 2), checkpoint_x=torch.rand(2, 8),
        job_machine_edge_index=torch.empty((2, 0), dtype=torch.long), job_machine_edge_attr=torch.empty((0, 1)),
        job_checkpoint_edge_index=torch.empty((2, 0), dtype=torch.long), job_checkpoint_edge_attr=torch.empty((0, 2)),
        global_x=torch.rand(7), num_jobs=3, num_machines=2, num_checkpoints=2,
    )


def test_rainbow_forward_shapes_and_distribution():
    agent = HGTRainbowAgent(4, hidden_dim=16, num_layers=1, num_heads=4, job_input_dim=14)
    assert agent(_graph()).shape == (4,)
    distribution = agent.distribution(_graph())
    assert distribution.shape == (4, 51)
    assert torch.allclose(distribution.sum(dim=-1), torch.ones(4), atol=1e-5)


def test_rainbow_one_update_loss_is_finite():
    library = build_classic_rule_library()
    agent = HGTRainbowAgent(library.action_size(), hidden_dim=16, num_layers=1, num_heads=4, job_input_dim=14)
    trainer = RainbowTrainer(agent, library, lambda _seed: None, batch_size=2)
    graph = _graph()
    trainer.replay.add(RainbowTransition(graph, 0, -1.0, graph, False, 3))
    trainer.replay.add(RainbowTransition(graph, 1, -0.5, None, True, 1))
    assert math.isfinite(trainer._learn())


def test_rainbow_contains_all_six_components():
    agent = HGTRainbowAgent(3, hidden_dim=16, num_layers=1, num_heads=4, job_input_dim=14)
    assert hasattr(agent, "value_out") and hasattr(agent, "advantage_out")
    assert agent.atoms == 51
    trainer = RainbowTrainer(agent, build_classic_rule_library(), lambda _seed: None, n_step=3)
    assert trainer.n_step == 3
    assert trainer.target is not trainer.agent
    assert trainer.replay.alpha > 0
