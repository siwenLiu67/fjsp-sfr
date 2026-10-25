"""论文 Sec 5.1: 异构调度图 G_n = (V, E)。"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from fjsp_sfr.core.online import SchedulingProblem
from fjsp_sfr.core.state import ScheduleState
from fjsp_sfr.dispatch.ca_metrics import CADecisionMetrics, CheckpointMetrics
from fjsp_sfr.dispatch.theory_features import feature_names_for


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


def apply_information_masks(
    data: HeteroGraphData,
    *,
    feature_set: str = "theory11",
    mask_checkpoint: bool = False,
    mask_forecast: bool = False,
) -> HeteroGraphData:
    """Mask selector inputs while leaving dispatch-rule factors untouched."""

    job_x = data.job_x.clone()
    checkpoint_x = data.checkpoint_x.clone()
    global_x = data.global_x.clone()
    checkpoint_edge_index = data.job_checkpoint_edge_index.clone()
    checkpoint_edge_attr = data.job_checkpoint_edge_attr.clone()
    num_checkpoints = data.num_checkpoints

    if mask_forecast:
        names = feature_names_for(feature_set)
        for name in ("forecast_deficit", "forecast_flag"):
            job_x[:, names.index(name)] = 0.0
        if checkpoint_x.shape[0]:
            checkpoint_x[:, 3] = 0.0
            checkpoint_x[:, 5] = 0.0
        global_x[5] = 0.0

    if mask_checkpoint:
        checkpoint_x = data.checkpoint_x.new_empty((0, data.checkpoint_x.shape[1]))
        checkpoint_edge_index = data.job_checkpoint_edge_index.new_empty((2, 0))
        checkpoint_edge_attr = data.job_checkpoint_edge_attr.new_empty(
            (0, data.job_checkpoint_edge_attr.shape[1])
        )
        num_checkpoints = 0

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
    )


def _clamp_unit(value: float) -> float:
    return max(0.0, min(float(value), 1.0))


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


def _active_checkpoints(
    state: ScheduleState,
    metrics: CADecisionMetrics,
) -> tuple[tuple[tuple[int, int], CheckpointMetrics], ...]:
    return tuple(
        (key, checkpoint)
        for key, checkpoint in sorted(metrics.checkpoints.items())
        if checkpoint.residual_gap > 0.0
        and checkpoint.checkpoint_time >= state.current_time
    )


def build_hetero_graph(
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
    active_checkpoints = _active_checkpoints(state, metrics)
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
    )
