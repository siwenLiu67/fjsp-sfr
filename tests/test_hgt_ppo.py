import torch

from fjsp_sfr.rl.graph import HeteroGraphData
from fjsp_sfr.rl.hgt import HGTEncoder, HGTLayer
from fjsp_sfr.rl.ppo_agent import HGTPPOAgent


def _data(
    *,
    num_jobs: int = 3,
    num_machines: int = 2,
    num_checkpoints: int = 1,
) -> HeteroGraphData:
    job_machine_sources = torch.arange(num_jobs, dtype=torch.long)
    job_machine_targets = job_machine_sources.remainder(num_machines)
    if num_checkpoints:
        checkpoint_edge_count = min(num_jobs, num_checkpoints)
        checkpoint_sources = torch.arange(checkpoint_edge_count, dtype=torch.long)
        checkpoint_targets = torch.arange(
            checkpoint_edge_count, dtype=torch.long
        ).remainder(num_checkpoints)
    else:
        checkpoint_sources = torch.empty(0, dtype=torch.long)
        checkpoint_targets = torch.empty(0, dtype=torch.long)
    return HeteroGraphData(
        job_x=torch.randn(num_jobs, 11),
        machine_x=torch.randn(num_machines, 2),
        checkpoint_x=torch.randn(num_checkpoints, 8),
        job_machine_edge_index=torch.stack(
            (job_machine_sources, job_machine_targets)
        ),
        job_machine_edge_attr=torch.randn(num_jobs, 1),
        job_checkpoint_edge_index=torch.stack(
            (checkpoint_sources, checkpoint_targets)
        ),
        job_checkpoint_edge_attr=torch.randn(checkpoint_sources.shape[0], 2),
        global_x=torch.randn(7),
        num_jobs=num_jobs,
        num_machines=num_machines,
        num_checkpoints=num_checkpoints,
    )


class TestHGTLayer:
    def test_output_shape(self):
        hidden_dim = 32
        layer = HGTLayer(
            hidden_dim,
            4,
            edge_dims={"job_machine": 1, "job_checkpoint": 2},
        )
        x_dict = {
            "job": torch.randn(5, hidden_dim),
            "machine": torch.randn(3, hidden_dim),
            "checkpoint": torch.randn(2, hidden_dim),
        }
        edge_index_dict = {
            "job_machine": torch.tensor([[0, 1, 2, 3, 4], [0, 1, 2, 0, 1]]),
            "job_checkpoint": torch.tensor([[0, 1, 2], [0, 1, 0]]),
        }
        edge_attr_dict = {
            "job_machine": torch.randn(5, 1),
            "job_checkpoint": torch.randn(3, 2),
        }
        output = layer(x_dict, edge_index_dict, edge_attr_dict)
        assert output["job"].shape == (5, hidden_dim)
        assert output["machine"].shape == (3, hidden_dim)
        assert output["checkpoint"].shape == (2, hidden_dim)

    def test_no_nan_gradients(self):
        layer = HGTLayer(
            16,
            4,
            edge_dims={"job_machine": 1, "job_checkpoint": 2},
        )
        x_dict = {
            "job": torch.randn(3, 16, requires_grad=True),
            "machine": torch.randn(2, 16),
            "checkpoint": torch.randn(1, 16),
        }
        output = layer(
            x_dict,
            {
                "job_machine": torch.tensor([[0, 1, 2], [0, 1, 0]]),
                "job_checkpoint": torch.tensor([[0, 1], [0, 0]]),
            },
            {
                "job_machine": torch.randn(3, 1),
                "job_checkpoint": torch.randn(2, 2),
            },
        )
        output["job"].sum().backward()
        assert not torch.isnan(x_dict["job"].grad).any()


class TestHGTEncoder:
    def test_output_dim(self):
        encoder = HGTEncoder(hidden_dim=32, num_layers=2, num_heads=4)
        representation = encoder(_data(num_jobs=4, num_machines=3, num_checkpoints=2))
        assert representation.shape == (32,)

    def test_empty_checkpoints(self):
        encoder = HGTEncoder(hidden_dim=16, num_layers=1, num_heads=4)
        representation = encoder(
            _data(num_jobs=2, num_machines=2, num_checkpoints=0)
        )
        assert representation.shape == (16,)
        assert not torch.isnan(representation).any()


class TestPPOAgent:
    def test_select_action_respects_mask(self):
        agent = HGTPPOAgent(
            action_size=5, hidden_dim=16, num_layers=1, num_heads=4
        )
        data = _data()
        mask = torch.tensor([True, False, True, False, True])
        for _ in range(50):
            action, _log_probability, _value = agent.select_action(data, mask)
            assert mask[action]

    def test_greedy_deterministic(self):
        agent = HGTPPOAgent(
            action_size=4, hidden_dim=16, num_layers=1, num_heads=4
        )
        data = _data(num_jobs=2)
        actions = [agent.greedy_action(data) for _ in range(20)]
        assert len(set(actions)) == 1

    def test_forward_shapes(self):
        agent = HGTPPOAgent(
            action_size=6, hidden_dim=16, num_layers=1, num_heads=4
        )
        log_probs, value = agent(_data(num_checkpoints=2))
        assert log_probs.shape == (6,)
        assert value.shape == ()
        assert (log_probs.exp().sum() - 1.0).abs() < 1e-5
