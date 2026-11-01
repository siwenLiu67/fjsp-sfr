"""论文 Sec 5.1: 异构调度图 G_n = (V, E)。"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from fjsp_sfr.core.online import SchedulingProblem
from fjsp_sfr.core.state import ScheduleState
from fjsp_sfr.dispatch.ca_metrics import (
    CADecisionMetrics,
    CheckpointMetrics,
    optimistic_job_completion_lower_bound,
)
from fjsp_sfr.dispatch.theory_features import (
    build_extended_theory_features,
    feature_names_for,
    resolve_paper_feature_scales,
)


LEGACY_GRAPH = "legacy"
PAPER_GRAPH = "paper_graph"
GRAPH_CONFIGS = (LEGACY_GRAPH, PAPER_GRAPH)

JOB = "job"
MACHINE = "machine"
CHECKPOINT = "checkpoint"
JOB_TO_MACHINE = "job_to_machine"
MACHINE_TO_JOB = "machine_to_job"
JOB_TO_CHECKPOINT_TIMELY = "job_to_checkpoint_timely"
CHECKPOINT_TO_JOB_TIMELY = "checkpoint_to_job_timely"
JOB_TO_CHECKPOINT_NONTIMELY = "job_to_checkpoint_nontimely"
CHECKPOINT_TO_JOB_NONTIMELY = "checkpoint_to_job_nontimely"
PAPER_RELATION_SPECS: dict[str, tuple[str, str, int]] = {
    JOB_TO_MACHINE: (JOB, MACHINE, 1),
    MACHINE_TO_JOB: (MACHINE, JOB, 1),
    JOB_TO_CHECKPOINT_TIMELY: (JOB, CHECKPOINT, 2),
    CHECKPOINT_TO_JOB_TIMELY: (CHECKPOINT, JOB, 2),
    JOB_TO_CHECKPOINT_NONTIMELY: (JOB, CHECKPOINT, 2),
    CHECKPOINT_TO_JOB_NONTIMELY: (CHECKPOINT, JOB, 2),
}


@dataclass
class HeteroGraphData:
    """不依赖 PyG 的轻量异构图数据容器。"""

    job_x: torch.Tensor
    machine_x: torch.Tensor
    checkpoint_x: torch.Tensor

    job_machine_edge_index: torch.Tensor
    job_machine_edge_attr: torch.Tensor

    job_checkpoint_edge_index: torch.Tensor
    job_checkpoint_edge_attr: torch.Tensor

    global_x: torch.Tensor

    num_jobs: int
    num_machines: int
    num_checkpoints: int
    job_ids: tuple[int, ...] = ()
    checkpoint_keys: tuple[tuple[int, int], ...] = ()
    graph_config: str = LEGACY_GRAPH
    relation_edge_index: dict[str, torch.Tensor] = field(default_factory=dict)
    relation_edge_attr: dict[str, torch.Tensor] = field(default_factory=dict)
    job_batch: torch.Tensor | None = None
    machine_batch: torch.Tensor | None = None
    checkpoint_batch: torch.Tensor | None = None
    num_graphs: int = 1


def apply_information_masks(
    data: HeteroGraphData,
    *,
    feature_set: str = "theory11",
    graph_config: str | None = None,
    mask_checkpoint: bool = False,
    mask_forecast: bool = False,
) -> HeteroGraphData:
    """Mask selector inputs while leaving dispatch-rule factors untouched."""

    resolved_graph_config = data.graph_config if graph_config is None else graph_config
    if resolved_graph_config != data.graph_config:
        raise ValueError("mask graph_config must match the graph data")
    if resolved_graph_config == PAPER_GRAPH and feature_set != "paper14":
        raise ValueError("paper_graph masks require feature_set='paper14'")

    job_x = data.job_x.clone()
    checkpoint_x = data.checkpoint_x.clone()
    global_x = data.global_x.clone()
    checkpoint_edge_index = data.job_checkpoint_edge_index.clone()
    checkpoint_edge_attr = data.job_checkpoint_edge_attr.clone()
    relation_edge_index = {
        name: value.clone() for name, value in data.relation_edge_index.items()
    }
    relation_edge_attr = {
        name: value.clone() for name, value in data.relation_edge_attr.items()
    }
    num_checkpoints = data.num_checkpoints

    if mask_forecast:
        names = feature_names_for(feature_set)
        for name in ("forecast_deficit", "forecast_flag"):
            if name in names:
                job_x[:, names.index(name)] = 0.0
        if checkpoint_x.shape[0]:
            if resolved_graph_config == PAPER_GRAPH:
                checkpoint_x[:, 5:8] = 0.0
            else:
                checkpoint_x[:, 3] = 0.0
                checkpoint_x[:, 5] = 0.0
        global_x[..., 5] = 0.0

    if mask_checkpoint:
        checkpoint_x = data.checkpoint_x.new_empty((0, data.checkpoint_x.shape[1]))
        checkpoint_edge_index = data.job_checkpoint_edge_index.new_empty((2, 0))
        checkpoint_edge_attr = data.job_checkpoint_edge_attr.new_empty(
            (0, data.job_checkpoint_edge_attr.shape[1])
        )
        num_checkpoints = 0
        for name, (source_type, target_type, width) in PAPER_RELATION_SPECS.items():
            if CHECKPOINT in (source_type, target_type):
                relation_edge_index[name] = checkpoint_edge_index.new_empty((2, 0))
                relation_edge_attr[name] = checkpoint_edge_attr.new_empty((0, width))

    return HeteroGraphData(
        job_x=job_x,
        machine_x=data.machine_x,
        checkpoint_x=checkpoint_x,
        job_machine_edge_index=data.job_machine_edge_index,
        job_machine_edge_attr=data.job_machine_edge_attr,
        job_checkpoint_edge_index=checkpoint_edge_index,
        job_checkpoint_edge_attr=checkpoint_edge_attr,
        global_x=global_x,
        num_jobs=data.num_jobs,
        num_machines=data.num_machines,
        num_checkpoints=num_checkpoints,
        job_ids=data.job_ids,
        checkpoint_keys=(() if mask_checkpoint else data.checkpoint_keys),
        graph_config=data.graph_config,
        relation_edge_index=relation_edge_index,
        relation_edge_attr=relation_edge_attr,
        job_batch=data.job_batch,
        machine_batch=data.machine_batch,
        checkpoint_batch=(
            None
            if data.checkpoint_batch is None
            else data.checkpoint_batch.new_empty((0,))
            if mask_checkpoint
            else data.checkpoint_batch
        ),
        num_graphs=data.num_graphs,
    )


def _clamp_unit(value: float) -> float:
    return max(0.0, min(float(value), 1.0))


def _safe_ratio(numerator: float, denominator: float) -> float:
    numerator = float(numerator)
    denominator = float(denominator)
    if (
        denominator == 0.0
        or not math.isfinite(numerator)
        or not math.isfinite(denominator)
    ):
        return 0.0
    value = numerator / denominator
    return value if math.isfinite(value) else 0.0


def _sigma(value: float) -> float:
    return _safe_ratio(max(0.0, float(value)), 1.0 + max(0.0, float(value)))


def _sigma_signed(value: float) -> float:
    value = float(value)
    return _safe_ratio(value, 1.0 + abs(value))


def _feature_matrix(
    rows: list[tuple[float, ...]] | tuple[tuple[float, ...], ...],
    width: int,
) -> torch.Tensor:
    if not rows:
        return torch.empty((0, width), dtype=torch.float32)
    return torch.tensor(rows, dtype=torch.float32)


def _edge_index(edges: list[tuple[int, int]]) -> torch.Tensor:
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def _legacy_active_checkpoints(
    state: ScheduleState,
    metrics: CADecisionMetrics,
) -> tuple[tuple[tuple[int, int], CheckpointMetrics], ...]:
    return tuple(
        (key, checkpoint)
        for key, checkpoint in sorted(metrics.checkpoints.items())
        if checkpoint.residual_gap > 0.0
        and checkpoint.checkpoint_time >= state.current_time
    )


def _build_legacy_graph(
    problem: SchedulingProblem,
    state: ScheduleState,
    metrics: CADecisionMetrics,
    theory_features: tuple[tuple[float, ...], ...],
    *,
    time_scale: float = 1.0,
    processing_scale: float = 1.0,
    feature_set: str = "theory11",
) -> HeteroGraphData:
    """从当前决策点构建 job-machine-checkpoint 异构图。"""

    if time_scale <= 0.0:
        raise ValueError("time_scale must be positive")
    if processing_scale <= 0.0:
        raise ValueError("processing_scale must be positive")

    candidates = tuple(sorted(metrics.candidates, key=lambda item: item.job_id))
    feature_names = feature_names_for(feature_set)
    feature_width = len(feature_names)
    if len(theory_features) != len(candidates) or any(
        len(row) != feature_width for row in theory_features
    ):
        raise ValueError(
            "theory features must have shape "
            f"(num_candidates, {feature_width})"
        )

    machines = tuple(sorted(problem.machines, key=lambda item: item.machine_id))
    machine_indices = {
        machine.machine_id: index for index, machine in enumerate(machines)
    }
    active_checkpoints = _legacy_active_checkpoints(state, metrics)
    checkpoint_indices = {
        key: index for index, (key, _checkpoint) in enumerate(active_checkpoints)
    }
    epsilon = float(problem.r5_epsilon)

    job_x = _feature_matrix(theory_features, feature_width)

    machine_rows = [
        (
            float(state.is_machine_idle(machine.machine_id)),
            max(
                0.0,
                float(
                    state.machine_available_times.get(machine.machine_id, 0)
                    - state.current_time
                ),
            )
            / time_scale,
        )
        for machine in machines
    ]
    machine_x = _feature_matrix(machine_rows, 2)

    max_checkpoint_weight = max(
        (checkpoint.checkpoint_weight for _key, checkpoint in active_checkpoints),
        default=1.0,
    )
    checkpoint_rows = []
    for _key, checkpoint in active_checkpoints:
        required = max(float(checkpoint.required_quantity), epsilon)
        capacity_ratio = checkpoint.optimistic_required_workload / (
            checkpoint.remaining_machine_processing_capacity + epsilon
        )
        checkpoint_rows.append(
            (
                _clamp_unit(checkpoint.residual_gap / required),
                _clamp_unit(checkpoint.checkpoint_weight / max_checkpoint_weight),
                _clamp_unit(
                    (checkpoint.checkpoint_time - state.current_time) / time_scale
                ),
                _clamp_unit(checkpoint.arrival_insufficiency / required),
                _clamp_unit(capacity_ratio),
                float(checkpoint.arrival_insufficiency > 0.0),
                _clamp_unit(checkpoint.visible_coverage / required),
                _clamp_unit(checkpoint.future_required / required),
            )
        )
    checkpoint_x = _feature_matrix(checkpoint_rows, 8)

    job_machine_edges: list[tuple[int, int]] = []
    job_machine_attributes: list[tuple[float]] = []
    job_checkpoint_edges: list[tuple[int, int]] = []
    job_checkpoint_attributes: list[tuple[float, float]] = []

    for job_index, candidate in enumerate(candidates):
        job = problem.get_job(candidate.job_id)
        operation = next(
            item for item in job.operations if item.op_id == candidate.operation_id
        )
        for alternative in sorted(
            operation.alternatives, key=lambda item: item.machine_id
        ):
            if alternative.machine_id not in machine_indices:
                continue
            job_machine_edges.append(
                (job_index, machine_indices[alternative.machine_id])
            )
            job_machine_attributes.append(
                (float(alternative.processing_time) / processing_scale,)
            )

        for key, checkpoint in active_checkpoints:
            if checkpoint.entity_id != candidate.entity_id:
                continue
            job_checkpoint_edges.append((job_index, checkpoint_indices[key]))
            job_checkpoint_attributes.append(
                (
                    min(float(job.quantity), checkpoint.residual_gap)
                    / (checkpoint.residual_gap + epsilon),
                    (
                        checkpoint.checkpoint_time
                        - candidate.optimistic_completion
                    )
                    / time_scale,
                )
            )

    job_machine_edge_index = _edge_index(job_machine_edges)
    job_machine_edge_attr = _feature_matrix(job_machine_attributes, 1)
    job_checkpoint_edge_index = _edge_index(job_checkpoint_edges)
    job_checkpoint_edge_attr = _feature_matrix(job_checkpoint_attributes, 2)

    gap_ratios = [
        _clamp_unit(
            checkpoint.residual_gap
            / max(float(checkpoint.required_quantity), epsilon)
        )
        for _key, checkpoint in active_checkpoints
    ]
    tardiness_index = feature_names.index("tardiness_exposure")
    forecast_flag_index = feature_names.index("forecast_flag")
    capacity_flag_index = feature_names.index("capacity_flag")
    global_x = torch.tensor(
        (
            float(state.current_time) / time_scale,
            len(candidates) / max(1, len(problem.jobs)),
            sum(state.is_machine_idle(machine.machine_id) for machine in machines)
            / max(1, len(machines)),
            sum(gap_ratios) / len(gap_ratios) if gap_ratios else 0.0,
            max(
                (row[tardiness_index] for row in theory_features),
                default=0.0,
            ),
            float(
                any(row[forecast_flag_index] > 0.0 for row in theory_features)
            ),
            float(
                any(row[capacity_flag_index] > 0.0 for row in theory_features)
            ),
        ),
        dtype=torch.float32,
    )

    return HeteroGraphData(
        job_x=job_x,
        machine_x=machine_x,
        checkpoint_x=checkpoint_x,
        job_machine_edge_index=job_machine_edge_index,
        job_machine_edge_attr=job_machine_edge_attr,
        job_checkpoint_edge_index=job_checkpoint_edge_index,
        job_checkpoint_edge_attr=job_checkpoint_edge_attr,
        global_x=global_x,
        num_jobs=len(candidates),
        num_machines=len(machines),
        num_checkpoints=len(active_checkpoints),
        job_ids=tuple(candidate.job_id for candidate in candidates),
        checkpoint_keys=tuple(key for key, _checkpoint in active_checkpoints),
        graph_config=LEGACY_GRAPH,
    )


def _paper_active_checkpoints(
    state: ScheduleState,
    metrics: CADecisionMetrics,
) -> tuple[tuple[tuple[int, int], CheckpointMetrics], ...]:
    """Return Table 5-1 checkpoint nodes: unsettled and positive-gap only."""

    return tuple(
        (key, checkpoint)
        for key, checkpoint in sorted(
            metrics.checkpoints.items(),
            key=lambda item: (
                item[1].checkpoint_time,
                item[1].entity_id,
                item[1].checkpoint_index,
            ),
        )
        if checkpoint.residual_gap > 0.0
        and checkpoint.checkpoint_time > state.current_time
    )


def _build_paper_graph(
    problem: SchedulingProblem,
    state: ScheduleState,
    metrics: CADecisionMetrics,
    paper_features: tuple[tuple[float, ...], ...],
) -> HeteroGraphData:
    """Build the exact Sec. 5.1/Table 5-1 heterogeneous state graph."""

    candidates = tuple(sorted(metrics.candidates, key=lambda item: item.job_id))
    machines = tuple(sorted(problem.machines, key=lambda item: item.machine_id))
    active_checkpoints = _paper_active_checkpoints(state, metrics)
    bar_p, bar_w = resolve_paper_feature_scales(problem)
    feature_width = len(feature_names_for("paper14"))
    job_x = _feature_matrix(paper_features, feature_width)

    machine_indices = {
        machine.machine_id: index for index, machine in enumerate(machines)
    }
    checkpoint_indices = {
        key: index for index, (key, _checkpoint) in enumerate(active_checkpoints)
    }

    machine_rows = []
    for machine in machines:
        remaining_occupation = max(
            0.0,
            float(
                state.machine_available_times.get(machine.machine_id, 0)
                - state.current_time
            ),
        )
        machine_rows.append(
            (
                float(state.is_machine_idle(machine.machine_id)),
                _sigma(_safe_ratio(remaining_occupation, bar_p)),
            )
        )
    machine_x = _feature_matrix(machine_rows, 2)

    max_checkpoint_weight = max(
        (float(checkpoint.checkpoint_weight) for _key, checkpoint in active_checkpoints),
        default=0.0,
    )
    checkpoint_rows: list[tuple[float, ...]] = []
    for _key, checkpoint in active_checkpoints:
        if (
            checkpoint.online_shortfall_lower is None
            or checkpoint.online_shortfall_upper is None
        ):
            raise ValueError(
                "paper_graph requires P1 public online certificate endpoints"
            )
        gap = float(checkpoint.residual_gap)
        required = float(checkpoint.required_quantity)
        remaining_time = float(checkpoint.checkpoint_time - state.current_time)
        forecast_slack_defined = checkpoint.release_slack is not None
        checkpoint_rows.append(
            (
                _clamp_unit(
                    _safe_ratio(gap, float(checkpoint.required_quantity))
                ),
                _clamp_unit(
                    _safe_ratio(
                        float(checkpoint.checkpoint_weight),
                        max_checkpoint_weight,
                    )
                ),
                _sigma(_safe_ratio(remaining_time, bar_w)),
                _clamp_unit(
                    _safe_ratio(float(checkpoint.visible_coverage), gap)
                ),
                _clamp_unit(
                    _safe_ratio(float(checkpoint.future_required), gap)
                ),
                _clamp_unit(
                    _safe_ratio(float(checkpoint.arrival_insufficiency), gap)
                ),
                (
                    _clamp_unit(
                        _safe_ratio(float(checkpoint.release_slack), remaining_time)
                    )
                    if forecast_slack_defined
                    else 0.0
                ),
                float(forecast_slack_defined),
                (
                    _clamp_unit(
                        _safe_ratio(
                            float(checkpoint.capacity_shortfall),
                            float(checkpoint.additional_gap),
                        )
                    )
                    if checkpoint.capacity_applicable
                    else 0.0
                ),
                float(checkpoint.capacity_applicable),
                _clamp_unit(
                    _safe_ratio(float(checkpoint.online_shortfall_lower), required)
                ),
                _clamp_unit(
                    _safe_ratio(float(checkpoint.online_shortfall_upper), required)
                ),
            )
        )
    checkpoint_x = _feature_matrix(checkpoint_rows, 12)

    job_machine_edges: list[tuple[int, int]] = []
    job_machine_attributes: list[tuple[float]] = []
    job_checkpoint_edges: list[tuple[int, int]] = []
    job_checkpoint_attributes: list[tuple[float, float]] = []
    timely_edges: list[tuple[int, int]] = []
    timely_attributes: list[tuple[float, float]] = []
    nontimely_edges: list[tuple[int, int]] = []
    nontimely_attributes: list[tuple[float, float]] = []
    for job_index, candidate in enumerate(candidates):
        job = problem.get_job(candidate.job_id)
        operation = next(
            item for item in job.operations if item.op_id == candidate.operation_id
        )
        completion_lb = float(
            optimistic_job_completion_lower_bound(job, state)
        )
        for alternative in sorted(
            operation.alternatives, key=lambda item: item.machine_id
        ):
            if alternative.machine_id not in machine_indices:
                continue
            job_machine_edges.append(
                (job_index, machine_indices[alternative.machine_id])
            )
            job_machine_attributes.append(
                (
                    _sigma(
                        _safe_ratio(
                            float(alternative.processing_time), bar_p
                        )
                    ),
                )
            )

        for key, checkpoint in active_checkpoints:
            if checkpoint.entity_id != candidate.entity_id:
                continue
            edge = (job_index, checkpoint_indices[key])
            job_checkpoint_edges.append(edge)
            gap = float(checkpoint.residual_gap)
            checkpoint_slack = float(checkpoint.checkpoint_time) - completion_lb
            attribute = (
                _clamp_unit(_safe_ratio(min(float(job.quantity), gap), gap)),
                _sigma_signed(_safe_ratio(checkpoint_slack, bar_w)),
            )
            job_checkpoint_attributes.append(attribute)
            if completion_lb <= float(checkpoint.checkpoint_time):
                timely_edges.append(edge)
                timely_attributes.append(attribute)
            else:
                nontimely_edges.append(edge)
                nontimely_attributes.append(attribute)

    job_machine_edge_index = _edge_index(job_machine_edges)
    job_machine_edge_attr = _feature_matrix(job_machine_attributes, 1)
    job_checkpoint_edge_index = _edge_index(job_checkpoint_edges)
    job_checkpoint_edge_attr = _feature_matrix(job_checkpoint_attributes, 2)
    job_machine_reverse = [(target, source) for source, target in job_machine_edges]
    timely_reverse = [(target, source) for source, target in timely_edges]
    nontimely_reverse = [(target, source) for source, target in nontimely_edges]
    relation_edge_index = {
        JOB_TO_MACHINE: _edge_index(job_machine_edges),
        MACHINE_TO_JOB: _edge_index(job_machine_reverse),
        JOB_TO_CHECKPOINT_TIMELY: _edge_index(timely_edges),
        CHECKPOINT_TO_JOB_TIMELY: _edge_index(timely_reverse),
        JOB_TO_CHECKPOINT_NONTIMELY: _edge_index(nontimely_edges),
        CHECKPOINT_TO_JOB_NONTIMELY: _edge_index(nontimely_reverse),
    }
    relation_edge_attr = {
        JOB_TO_MACHINE: _feature_matrix(job_machine_attributes, 1),
        MACHINE_TO_JOB: _feature_matrix(job_machine_attributes, 1),
        JOB_TO_CHECKPOINT_TIMELY: _feature_matrix(timely_attributes, 2),
        CHECKPOINT_TO_JOB_TIMELY: _feature_matrix(timely_attributes, 2),
        JOB_TO_CHECKPOINT_NONTIMELY: _feature_matrix(nontimely_attributes, 2),
        CHECKPOINT_TO_JOB_NONTIMELY: _feature_matrix(nontimely_attributes, 2),
    }

    feature_names = feature_names_for("paper14")
    tardiness_index = feature_names.index("tardiness_exposure")
    max_published_checkpoint = max(
        float(checkpoint.time)
        for entity in problem.entities
        for checkpoint in entity.checkpoints
    )
    visible_job_count = sum(
        job.release_time <= state.current_time for job in problem.jobs
    )
    gap_ratios = [
        _clamp_unit(
            _safe_ratio(
                float(checkpoint.residual_gap),
                float(checkpoint.required_quantity),
            )
        )
        for _key, checkpoint in active_checkpoints
    ]
    global_x = torch.tensor(
        (
            _clamp_unit(
                _safe_ratio(float(state.current_time), max_published_checkpoint)
            ),
            _clamp_unit(_safe_ratio(len(candidates), visible_job_count)),
            _safe_ratio(
                sum(state.is_machine_idle(machine.machine_id) for machine in machines),
                len(machines),
            ),
            sum(gap_ratios) / len(gap_ratios) if gap_ratios else 0.0,
            max(
                (row[tardiness_index] for row in paper_features),
                default=0.0,
            ),
            float(
                any(
                    checkpoint.arrival_insufficiency > 0.0
                    for _key, checkpoint in active_checkpoints
                )
            ),
            float(
                any(
                    checkpoint.capacity_applicable
                    and checkpoint.capacity_shortfall > 0.0
                    for _key, checkpoint in active_checkpoints
                )
            ),
        ),
        dtype=torch.float32,
    )

    return HeteroGraphData(
        job_x=job_x,
        machine_x=machine_x,
        checkpoint_x=checkpoint_x,
        job_machine_edge_index=job_machine_edge_index,
        job_machine_edge_attr=job_machine_edge_attr,
        job_checkpoint_edge_index=job_checkpoint_edge_index,
        job_checkpoint_edge_attr=job_checkpoint_edge_attr,
        global_x=global_x,
        num_jobs=len(candidates),
        num_machines=len(machines),
        num_checkpoints=len(active_checkpoints),
        job_ids=tuple(candidate.job_id for candidate in candidates),
        checkpoint_keys=tuple(key for key, _checkpoint in active_checkpoints),
        graph_config=PAPER_GRAPH,
        relation_edge_index=relation_edge_index,
        relation_edge_attr=relation_edge_attr,
    )


def batch_hetero_graphs(graphs: tuple[HeteroGraphData, ...]) -> HeteroGraphData:
    """Batch variable-sized formal graphs without guessing their configuration."""

    if not graphs:
        raise ValueError("at least one graph is required")
    if any(graph.graph_config != PAPER_GRAPH for graph in graphs):
        raise ValueError("batch_hetero_graphs accepts paper_graph inputs only")
    if any(set(graph.relation_edge_index) != set(PAPER_RELATION_SPECS) for graph in graphs):
        raise ValueError("every paper graph must contain exactly the six relations")

    node_parts = {JOB: [], MACHINE: [], CHECKPOINT: []}
    batches = {JOB: [], MACHINE: [], CHECKPOINT: []}
    relation_indices = {name: [] for name in PAPER_RELATION_SPECS}
    relation_attrs = {name: [] for name in PAPER_RELATION_SPECS}
    offsets = {JOB: 0, MACHINE: 0, CHECKPOINT: 0}
    job_ids: list[int] = []
    checkpoint_keys: list[tuple[int, int]] = []
    for graph_index, graph in enumerate(graphs):
        local = {JOB: graph.job_x, MACHINE: graph.machine_x, CHECKPOINT: graph.checkpoint_x}
        for node_type, values in local.items():
            node_parts[node_type].append(values)
            batches[node_type].append(
                torch.full((values.shape[0],), graph_index, dtype=torch.long, device=values.device)
            )
        for name, (source_type, target_type, _width) in PAPER_RELATION_SPECS.items():
            index = graph.relation_edge_index[name].clone()
            if index.numel():
                index[0] += offsets[source_type]
                index[1] += offsets[target_type]
            relation_indices[name].append(index)
            relation_attrs[name].append(graph.relation_edge_attr[name])
        offsets[JOB] += graph.job_x.shape[0]
        offsets[MACHINE] += graph.machine_x.shape[0]
        offsets[CHECKPOINT] += graph.checkpoint_x.shape[0]
        job_ids.extend(graph.job_ids)
        checkpoint_keys.extend(graph.checkpoint_keys)

    def cat_features(parts: list[torch.Tensor], width: int) -> torch.Tensor:
        return torch.cat(parts, dim=0) if parts else torch.empty((0, width))

    def cat_indices(parts: list[torch.Tensor]) -> torch.Tensor:
        return torch.cat(parts, dim=1) if parts else torch.empty((2, 0), dtype=torch.long)

    rel_index = {name: cat_indices(parts) for name, parts in relation_indices.items()}
    rel_attr = {
        name: cat_features(parts, PAPER_RELATION_SPECS[name][2])
        for name, parts in relation_attrs.items()
    }
    job_machine = rel_index[JOB_TO_MACHINE]
    job_machine_attr = rel_attr[JOB_TO_MACHINE]
    job_checkpoint = torch.cat(
        (rel_index[JOB_TO_CHECKPOINT_TIMELY], rel_index[JOB_TO_CHECKPOINT_NONTIMELY]), dim=1
    )
    job_checkpoint_attr = torch.cat(
        (rel_attr[JOB_TO_CHECKPOINT_TIMELY], rel_attr[JOB_TO_CHECKPOINT_NONTIMELY]), dim=0
    )
    return HeteroGraphData(
        job_x=cat_features(node_parts[JOB], 14),
        machine_x=cat_features(node_parts[MACHINE], 2),
        checkpoint_x=cat_features(node_parts[CHECKPOINT], 12),
        job_machine_edge_index=job_machine,
        job_machine_edge_attr=job_machine_attr,
        job_checkpoint_edge_index=job_checkpoint,
        job_checkpoint_edge_attr=job_checkpoint_attr,
        global_x=torch.stack([graph.global_x for graph in graphs]),
        num_jobs=offsets[JOB],
        num_machines=offsets[MACHINE],
        num_checkpoints=offsets[CHECKPOINT],
        job_ids=tuple(job_ids),
        checkpoint_keys=tuple(checkpoint_keys),
        graph_config=PAPER_GRAPH,
        relation_edge_index=rel_index,
        relation_edge_attr=rel_attr,
        job_batch=torch.cat(batches[JOB]),
        machine_batch=torch.cat(batches[MACHINE]),
        checkpoint_batch=torch.cat(batches[CHECKPOINT]),
        num_graphs=len(graphs),
    )


def build_hetero_graph(
    problem: SchedulingProblem,
    state: ScheduleState,
    metrics: CADecisionMetrics,
    theory_features: tuple[tuple[float, ...], ...] | None = None,
    *,
    time_scale: float = 1.0,
    processing_scale: float = 1.0,
    feature_set: str | None = None,
    graph_config: str = LEGACY_GRAPH,
) -> HeteroGraphData:
    """Build an explicitly selected legacy or manuscript-aligned graph.

    ``paper_graph`` always recomputes ``paper14`` through the canonical feature
    builder.  A caller-supplied matrix is accepted only as an equality check;
    it can never replace or downgrade the formal feature path.
    """

    if graph_config not in GRAPH_CONFIGS:
        raise ValueError(f"unknown graph_config: {graph_config}")
    if graph_config == LEGACY_GRAPH:
        resolved_feature_set = feature_set or "theory11"
        if theory_features is None:
            raise ValueError("legacy graph requires an explicit feature matrix")
        return _build_legacy_graph(
            problem,
            state,
            metrics,
            theory_features,
            time_scale=time_scale,
            processing_scale=processing_scale,
            feature_set=resolved_feature_set,
        )

    if feature_set not in (None, "paper14"):
        raise ValueError("paper_graph requires feature_set='paper14'")
    computed = build_extended_theory_features(
        problem,
        state,
        metrics,
        feature_set="paper14",
    )
    if theory_features is not None and theory_features != computed:
        raise ValueError(
            "paper_graph feature matrix must exactly match canonical paper14"
        )
    return _build_paper_graph(problem, state, metrics, computed)


__all__ = [
    "GRAPH_CONFIGS",
    "LEGACY_GRAPH",
    "PAPER_GRAPH",
    "HeteroGraphData",
    "apply_information_masks",
    "build_hetero_graph",
]
