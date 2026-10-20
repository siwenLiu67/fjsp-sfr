import torch

from conftest import make_test_instance

from fjsp_sfr.dispatch.theory_features import build_extended_theory_features
from fjsp_sfr.env.ca_rule_selection import CAPPORuleSelectionEnv
from fjsp_sfr.rl.graph import build_hetero_graph


def _graph():
    env = CAPPORuleSelectionEnv(make_test_instance())
    observation = env.reset()
    assert observation.candidate_job_ids
    features = build_extended_theory_features(
        env._problem_view(), env._state, env.current_metrics
    )
    graph = build_hetero_graph(
        env._problem_view(),
        env._state,
        env.current_metrics,
        features,
    )
    return observation, graph


class TestGraphConstruction:
    def test_node_counts(self):
        observation, graph = _graph()

        assert graph.num_jobs == len(observation.candidate_job_ids)
        assert graph.num_machines > 0
        assert graph.job_x.shape == (graph.num_jobs, 11)
        assert graph.machine_x.shape == (graph.num_machines, 2)
        assert graph.checkpoint_x.shape == (graph.num_checkpoints, 8)

    def test_edge_indices_valid(self):
        _observation, graph = _graph()

        if graph.job_machine_edge_index.numel() > 0:
            assert graph.job_machine_edge_index[0].max() < graph.num_jobs
            assert graph.job_machine_edge_index[1].max() < graph.num_machines
        if graph.job_checkpoint_edge_index.numel() > 0:
            assert graph.job_checkpoint_edge_index[0].max() < graph.num_jobs
            assert graph.job_checkpoint_edge_index[1].max() < graph.num_checkpoints

    def test_tensors_no_nan(self):
        _observation, graph = _graph()

        tensors = (
            graph.job_x,
            graph.machine_x,
            graph.checkpoint_x,
            graph.job_machine_edge_attr,
            graph.job_checkpoint_edge_attr,
            graph.global_x,
        )
        assert all(not torch.isnan(tensor).any() for tensor in tensors)

    def test_global_features_dim(self):
        _observation, graph = _graph()

        assert graph.global_x.shape == (7,)

    def test_edge_attribute_shapes(self):
        _observation, graph = _graph()

        assert graph.job_machine_edge_index.shape[0] == 2
        assert graph.job_machine_edge_attr.shape == (
            graph.job_machine_edge_index.shape[1],
            1,
        )
        assert graph.job_checkpoint_edge_index.shape[0] == 2
        assert graph.job_checkpoint_edge_attr.shape == (
            graph.job_checkpoint_edge_index.shape[1],
            2,
        )
