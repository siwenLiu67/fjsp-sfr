"""P2 executable contracts for the frozen Paper-A checkpoint relation graph."""

from __future__ import annotations

import torch

from fjsp_sfr.rl.crg import CRGEncoder
from fjsp_sfr.rl.graph import (
    CHECKPOINT_TO_JOB_NONTIMELY,
    CHECKPOINT_TO_JOB_TIMELY,
    JOB_TO_CHECKPOINT_NONTIMELY,
    JOB_TO_CHECKPOINT_TIMELY,
    JOB_TO_MACHINE,
    MACHINE_TO_JOB,
    PAPER_GRAPH,
    PAPER_RELATION_SPECS,
    HeteroGraphData,
    batch_hetero_graphs,
)
from fjsp_sfr.rl.ppo_agent import HGTPPOAgent
from fjsp_sfr.experiments.formal_experiment import make_agent
from fjsp_sfr.experiments.paperA_contract import PAPER_20261030


def _graph(*, checkpoints: bool = True) -> HeteroGraphData:
    torch.manual_seed(7)
    job_x = torch.randn(2, 14)
    machine_x = torch.randn(2, 2)
    checkpoint_x = torch.randn(2, 12) if checkpoints else torch.empty(0, 12)
    jm = torch.tensor([[0, 1, 1], [0, 0, 1]], dtype=torch.long)
    jm_attr = torch.tensor([[0.2], [0.4], [0.6]])
    if checkpoints:
        timely = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)
        nontimely = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        timely_attr = torch.tensor([[0.5, 0.2], [0.4, 0.3]])
        nontimely_attr = torch.tensor([[0.5, -0.2], [0.4, -0.3]])
    else:
        timely = nontimely = torch.empty((2, 0), dtype=torch.long)
        timely_attr = nontimely_attr = torch.empty((0, 2))
    relation_index = {
        JOB_TO_MACHINE: jm,
        MACHINE_TO_JOB: jm.flip(0),
        JOB_TO_CHECKPOINT_TIMELY: timely,
        CHECKPOINT_TO_JOB_TIMELY: timely.flip(0),
        JOB_TO_CHECKPOINT_NONTIMELY: nontimely,
        CHECKPOINT_TO_JOB_NONTIMELY: nontimely.flip(0),
    }
    relation_attr = {
        JOB_TO_MACHINE: jm_attr,
        MACHINE_TO_JOB: jm_attr,
        JOB_TO_CHECKPOINT_TIMELY: timely_attr,
        CHECKPOINT_TO_JOB_TIMELY: timely_attr,
        JOB_TO_CHECKPOINT_NONTIMELY: nontimely_attr,
        CHECKPOINT_TO_JOB_NONTIMELY: nontimely_attr,
    }
    all_jc = torch.cat((timely, nontimely), dim=1)
    all_jc_attr = torch.cat((timely_attr, nontimely_attr), dim=0)
    return HeteroGraphData(
        job_x=job_x,
        machine_x=machine_x,
        checkpoint_x=checkpoint_x,
        job_machine_edge_index=jm,
        job_machine_edge_attr=jm_attr,
        job_checkpoint_edge_index=all_jc,
        job_checkpoint_edge_attr=all_jc_attr,
        global_x=torch.arange(7, dtype=torch.float32) / 7.0,
        num_jobs=2,
        num_machines=2,
        num_checkpoints=checkpoint_x.shape[0],
        graph_config=PAPER_GRAPH,
        relation_edge_index=relation_index,
        relation_edge_attr=relation_attr,
    )


def test_paper_graph_has_exactly_six_directed_relations_and_reverse_edges():
    graph = _graph()
    assert set(graph.relation_edge_index) == set(PAPER_RELATION_SPECS)
    assert torch.equal(
        graph.relation_edge_index[MACHINE_TO_JOB],
        graph.relation_edge_index[JOB_TO_MACHINE].flip(0),
    )
    assert torch.equal(
        graph.relation_edge_index[CHECKPOINT_TO_JOB_TIMELY],
        graph.relation_edge_index[JOB_TO_CHECKPOINT_TIMELY].flip(0),
    )
    assert torch.equal(
        graph.relation_edge_index[CHECKPOINT_TO_JOB_NONTIMELY],
        graph.relation_edge_index[JOB_TO_CHECKPOINT_NONTIMELY].flip(0),
    )


def test_relation_attention_normalizes_per_destination_and_head():
    encoder = CRGEncoder()
    encoded = encoder.encode(_graph())
    for layer_attention in encoded.attentions:
        for relation, alpha in layer_attention.items():
            targets = _graph().relation_edge_index[relation][1]
            for target in targets.unique():
                assert torch.allclose(
                    alpha[targets == target].sum(dim=0),
                    torch.ones(4),
                    atol=1e-6,
                )


def test_crg_forward_backward_and_actor_are_finite():
    graph = _graph()
    agent = HGTPPOAgent(action_size=3, encoder_type="crg_gatv2")
    log_probs, value = agent(graph)
    loss = -log_probs.mean() + value.square()
    loss.backward()
    assert torch.isfinite(log_probs).all() and torch.isfinite(value)
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in agent.parameters()
    )
    attention_gradients = [
        parameter.grad
        for name, parameter in agent.named_parameters()
        if "attention_vector" in name
    ]
    assert attention_gradients
    assert any(gradient is not None for gradient in attention_gradients)
    assert all(
        gradient is None or torch.isfinite(gradient).all()
        for gradient in attention_gradients
    )


def test_crg_handles_no_checkpoint_and_variable_size_batch():
    empty = _graph(checkpoints=False)
    encoder = CRGEncoder()
    assert torch.isfinite(encoder(empty)).all()
    batched = batch_hetero_graphs((_graph(), empty))
    encoded = encoder(batched)
    assert encoded.shape == (2, 64)
    assert torch.isfinite(encoded).all()


def test_crg_handles_single_job_machine_and_no_neighbor_checkpoint():
    graph = _graph(checkpoints=False)
    graph.job_x = graph.job_x[:1]
    graph.machine_x = graph.machine_x[:1]
    graph.num_jobs = graph.num_machines = 1
    one_edge = torch.tensor([[0], [0]], dtype=torch.long)
    graph.job_machine_edge_index = one_edge
    graph.job_machine_edge_attr = graph.job_machine_edge_attr[:1]
    graph.relation_edge_index[JOB_TO_MACHINE] = one_edge
    graph.relation_edge_index[MACHINE_TO_JOB] = one_edge
    graph.relation_edge_attr[JOB_TO_MACHINE] = graph.job_machine_edge_attr
    graph.relation_edge_attr[MACHINE_TO_JOB] = graph.job_machine_edge_attr
    encoded = CRGEncoder()(graph)
    assert encoded.shape == (64,)
    assert torch.isfinite(encoded).all()


def test_crg_is_job_permutation_equivariant_and_state_invariant():
    graph = _graph()
    permutation = torch.tensor([1, 0])
    old_to_new = torch.tensor([1, 0])
    indices = {}
    for relation, (source_type, target_type, _width) in PAPER_RELATION_SPECS.items():
        index = graph.relation_edge_index[relation].clone()
        if source_type == "job":
            index[0] = old_to_new[index[0]]
        if target_type == "job":
            index[1] = old_to_new[index[1]]
        indices[relation] = index
    permuted = HeteroGraphData(
        **{
            **graph.__dict__,
            "job_x": graph.job_x[permutation],
            "relation_edge_index": indices,
        }
    )
    torch.manual_seed(11)
    encoder = CRGEncoder().eval()
    left = encoder.encode(graph)
    right = encoder.encode(permuted)
    assert torch.allclose(left.state, right.state, atol=1e-6)
    assert torch.allclose(
        left.node_embeddings["job"][permutation],
        right.node_embeddings["job"],
        atol=1e-6,
    )


def test_crg_rejects_legacy_graph_instead_of_falling_back():
    graph = _graph()
    graph.graph_config = "legacy"
    agent = HGTPPOAgent(action_size=2, encoder_type="crg_gatv2")
    try:
        agent(graph)
    except ValueError as exc:
        assert "no legacy fallback" in str(exc)
    else:
        raise AssertionError("CRG silently accepted a legacy graph")


def test_formal_agent_factory_selects_crg_without_changing_legacy_default():
    formal = make_agent("hgt_ppo", 3, contract=PAPER_20261030)
    assert formal.encoder_type == "crg_gatv2"
    legacy_default = HGTPPOAgent(action_size=3)
    assert legacy_default.encoder_type == "legacy_hgt"
