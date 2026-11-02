"""Frozen experiment contracts for the Paper A implementation paths.

This module records the corrected 2026-10-30 manuscript contract and keeps
the remaining post-P2 runtime gates explicit.
"""

from __future__ import annotations

from dataclasses import dataclass

from fjsp_sfr.rl.graph import LEGACY_GRAPH, PAPER_GRAPH


LEGACY_CONTRACT = "legacy"
PAPER_20261030_CONTRACT = "paper_20261030"


@dataclass(frozen=True)
class FormalExperimentContract:
    """Explicit, non-inferred configuration for one experiment code path."""

    name: str
    feature_set: str
    graph_config: str
    node_dimensions: tuple[int, int, int]
    encoder: str
    gamma: float
    reward_reference: str
    action_equivalence_merge: bool
    evaluation_checkpoint: str
    runtime_ready: bool
    blockers: tuple[str, ...] = ()


LEGACY = FormalExperimentContract(
    name=LEGACY_CONTRACT,
    feature_set="theory11_plus3",
    graph_config=LEGACY_GRAPH,
    node_dimensions=(14, 2, 8),
    encoder="legacy_hgt_relu",
    gamma=0.99,
    reward_reference="per_training_instance_rule_library_mean",
    action_equivalence_merge=False,
    evaluation_checkpoint="best_validation",
    runtime_ready=True,
)


PAPER_20261030 = FormalExperimentContract(
    name=PAPER_20261030_CONTRACT,
    feature_set="paper14",
    graph_config=PAPER_GRAPH,
    node_dimensions=(14, 2, 12),
    encoder="crg_gatv2",
    gamma=1.0,
    reward_reference="global_calibration_spt_atc_mean",
    action_equivalence_merge=True,
    evaluation_checkpoint="final_training",
    runtime_ready=False,
    blockers=(
        "P5 formal experiment protocol has not completed end-to-end acceptance",
    ),
)


FORMAL_EXPERIMENT_CONTRACTS = {
    LEGACY.name: LEGACY,
    PAPER_20261030.name: PAPER_20261030,
}


class ContractNotReadyError(RuntimeError):
    """Raised instead of silently falling back to a legacy implementation."""


def resolve_formal_contract(name: str) -> FormalExperimentContract:
    """Return an explicitly named contract; never infer or downgrade it."""

    try:
        return FORMAL_EXPERIMENT_CONTRACTS[name]
    except KeyError as exc:
        choices = ", ".join(sorted(FORMAL_EXPERIMENT_CONTRACTS))
        raise ValueError(f"unknown formal contract {name!r}; choose one of: {choices}") from exc


def require_runtime_ready(contract: FormalExperimentContract) -> None:
    """Fail closed when the corrected paper path is selected too early."""

    if contract.runtime_ready:
        return
    details = "; ".join(contract.blockers)
    raise ContractNotReadyError(
        f"contract {contract.name!r} is frozen but not runnable: {details}"
    )


__all__ = [
    "ContractNotReadyError",
    "FORMAL_EXPERIMENT_CONTRACTS",
    "FormalExperimentContract",
    "LEGACY",
    "LEGACY_CONTRACT",
    "PAPER_20261030",
    "PAPER_20261030_CONTRACT",
    "require_runtime_ready",
    "resolve_formal_contract",
]
