import torch
import pytest

from fjsp_sfr.rl.graph import HeteroGraphData
from fjsp_sfr.rl.dqn_agent import DQNTrainer, HGTDQNAgent
from fjsp_sfr.rl.mlp_agent import MLPPPOAgent


def _graph(job_dim=14):
    return HeteroGraphData(
        job_x=torch.rand(3, job_dim),
        machine_x=torch.rand(2, 2),
        checkpoint_x=torch.rand(2, 8),
        job_machine_edge_index=torch.empty((2, 0), dtype=torch.long),
        job_machine_edge_attr=torch.empty((0, 1)),
        job_checkpoint_edge_index=torch.empty((2, 0), dtype=torch.long),
        job_checkpoint_edge_attr=torch.empty((0, 2)),
        global_x=torch.rand(7),
        num_jobs=3,
        num_machines=2,
        num_checkpoints=2,
    )


def test_mlp_selector_has_ppo_compatible_interface():
    agent = MLPPPOAgent(4)
    log_probs, value = agent(_graph(), torch.ones(4, dtype=torch.bool))
    assert log_probs.shape == (4,)
    assert value.ndim == 0
    assert torch.exp(log_probs).sum().item() == pytest.approx(1.0)


def test_dqn_selector_and_epsilon_endpoints():
    agent = HGTDQNAgent(
        4, hidden_dim=16, num_layers=1, num_heads=4, job_input_dim=14
    )
    assert agent(_graph()).shape == (4,)
    trainer = object.__new__(DQNTrainer)
    trainer.epsilon_start = 1.0
    trainer.epsilon_end = 0.05
    assert trainer.epsilon_at(0, 10) == pytest.approx(1.0)
    assert trainer.epsilon_at(9, 10) == pytest.approx(0.05)
