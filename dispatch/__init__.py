"""Checkpoint-aware composite dispatching primitives."""

from .ca_metrics import (
    CADecisionMetrics,
    CandidateMetrics,
    CheckpointMetrics,
    build_ca_decision_metrics,
    candidate_jobs,
    compute_arrival_insufficiency,
    compute_conservative_arrival_forecast,
    compute_future_required_quantity,
    compute_optimistic_visible_coverage,
    compute_optimistic_checkpoint_workload_capacity,
    compute_release_slack,
    compute_release_threshold_time,
    multi_checkpoint_service_value,
    next_trigger_machine,
    optimistic_candidate_completion,
    residual_checkpoint_gap,
)
from .ca_rules import rule_applicability_mask, select_job
from .types import DirectCandidateDecision, RuleAction, RuleDecision

__all__ = [
    "CADecisionMetrics",
    "CandidateMetrics",
    "CheckpointMetrics",
    "DirectCandidateDecision",
    "RuleAction",
    "RuleDecision",
    "build_ca_decision_metrics",
    "candidate_jobs",
    "compute_arrival_insufficiency",
    "compute_conservative_arrival_forecast",
    "compute_future_required_quantity",
    "compute_optimistic_visible_coverage",
    "compute_optimistic_checkpoint_workload_capacity",
    "compute_release_slack",
    "compute_release_threshold_time",
    "multi_checkpoint_service_value",
    "next_trigger_machine",
    "optimistic_candidate_completion",
    "residual_checkpoint_gap",
    "rule_applicability_mask",
    "select_job",
]
