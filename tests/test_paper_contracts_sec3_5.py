"""Executable Paper-A contracts frozen from sec3--sec5 (2026-10-27).

These tests intentionally describe the manuscript contract rather than the
legacy implementation.  A failure is therefore evidence to classify; it is
not permission to weaken the assertion or silently change the manuscript.
"""

from __future__ import annotations

import json
import math
import random as stdlib_random
from copy import deepcopy
from dataclasses import dataclass, replace

import pytest
import torch

from fjsp_sfr.core.forecast import (
    ForecastBucket,
    ForecastSnapshot,
    build_compound_poisson_forecast_provider,
)
from fjsp_sfr.core.models import (
    Checkpoint,
    FJSPInstance,
    Job,
    Machine,
    Operation,
    OperationAlternative,
    ScheduledOperation,
    ServiceEntity,
)
from fjsp_sfr.core.objective import compute_objective
from fjsp_sfr.core.online import OnlineProblemView
from fjsp_sfr.core.state import ScheduleState
from fjsp_sfr.dispatch.ca_metrics import (
    CADecisionMetrics,
    CandidateMetrics,
    CheckpointMetrics,
    analyze_checkpoint_capacity,
    build_ca_decision_metrics,
    compute_arrival_forecast_summary,
    compute_optimistic_checkpoint_workload_capacity,
    compute_release_threshold_time,
    optimistic_candidate_completion,
    optimistic_job_completion_lower_bound,
    select_target_checkpoint,
    timely_visible_jobs,
)
from fjsp_sfr.dispatch.classic_rules import ClassicRule, ClassicRuleKind
from fjsp_sfr.dispatch.rule_library import RuleLibrary
from fjsp_sfr.dispatch.symbolic_rule import RuleType, SymbolicRule
from fjsp_sfr.dispatch.theory_features import (
    PAPER14_FEATURE_NAMES,
    PAPER_MEAN_JOB_WORK,
    PAPER_MEAN_PROCESSING_TIME,
    build_extended_theory_features,
    feature_names_for,
    resolve_paper_feature_scales,
)
from fjsp_sfr.env.ca_rule_selection import CAPPORuleSelectionEnv
from fjsp_sfr.experiments.run_algo import resolve_runtime_library
from fjsp_sfr.generation import scenario_generator
from fjsp_sfr.generation.scenario_generator import make_scenario_instance
from fjsp_sfr.rl.graph import PAPER_GRAPH, apply_information_masks, build_hetero_graph
from fjsp_sfr.rl.hgt import HGTEncoder
from fjsp_sfr.rl.ppo_trainer import PPOTrainer


def _operation(
    job_id: int,
    op_id: int,
    sequence: int,
    *alternatives: tuple[int, int],
) -> Operation:
    return Operation(
        op_id,
        job_id,
        sequence,
        tuple(OperationAlternative(machine, duration) for machine, duration in alternatives),
    )


def _job(
    job_id: int,
    *,
    release: int,
    due: int,
    quantity: int,
    operations: tuple[Operation, ...],
    entity: int = 0,
) -> Job:
    return Job(job_id, entity, release, due, quantity, operations)


def _checkpoint_metrics(
    index: int,
    *,
    time: int,
    weight: float,
    required: float,
    gap: float,
    visible: float = 0.0,
    workload: float = 0.0,
    capacity: float = 0.0,
    future_required: float = 0.0,
    forecast: float = 0.0,
    uncertainty: float = 0.0,
    conservative: float = 0.0,
    insufficiency: float = 0.0,
    release_slack: float | None = None,
) -> CheckpointMetrics:
    return CheckpointMetrics(
        entity_id=0,
        checkpoint_index=index,
        checkpoint_time=time,
        checkpoint_weight=weight,
        required_quantity=required,
        residual_gap=gap,
        visible_coverage=visible,
        optimistic_required_workload=workload,
        remaining_machine_processing_capacity=capacity,
        future_required=future_required,
        forecast_demand=forecast,
        forecast_uncertainty=uncertainty,
        conservative_forecast=conservative,
        arrival_insufficiency=insufficiency,
        release_threshold_time=(time - release_slack if release_slack is not None else None),
        release_slack=release_slack,
    )


def _run_fixed_rule(instance: FJSPInstance, rule: object) -> float:
    rollout = deepcopy(instance)
    env = CAPPORuleSelectionEnv(rollout)
    env.reset()
    while True:
        result = env.step_principle(0, rule)
        if result.terminated:
            return float(compute_objective(rollout, env._state).Z)


class _ParameterOnlyAgent(torch.nn.Module):
    def __init__(self, action_size: int) -> None:
        super().__init__()
        self.action_size = action_size
        self.anchor = torch.nn.Parameter(torch.zeros(()))


# ---------------------------------------------------------------------------
# A. Exogenous commitments and online-information isolation (sec3:79--83)
# ---------------------------------------------------------------------------


def test_paper_contract_a_commitments_do_not_change_with_future_quantity_sample(
    tmp_path, monkeypatch
):
    """Same scenario prior + same structural sample, different future q_j sample."""

    payload = scenario_generator.load_scenario_config()
    payload["fixed"] = dict(payload["fixed"])
    payload["fixed"].update(
        {
            "initial_jobs": 1,
            "new_jobs": 1,
            "operations_per_job": [2, 3],
            "processing_time": [10, 20],
            # Unique bounds let the test vary q_j without perturbing any other draw.
            "quantity": [101, 102],
        }
    )
    config_path = tmp_path / "scenario.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    real_random = stdlib_random.Random

    class QuantityControlledRandom:
        def __init__(self, seed: int, future_quantity: int) -> None:
            self._delegate = real_random(seed)
            self._future_quantity = future_quantity
            self._quantity_draws = 0

        def randint(self, lower: int, upper: int) -> int:
            if (lower, upper) == (101, 102):
                self._quantity_draws += 1
                # The single initial job is identical.  Only the unreleased
                # job's realized quantity differs between the two instances.
                return 101 if self._quantity_draws == 1 else self._future_quantity
            return self._delegate.randint(lower, upper)

        def __getattr__(self, name):
            return getattr(self._delegate, name)

    monkeypatch.setattr(
        scenario_generator.random,
        "Random",
        lambda seed: QuantityControlledRandom(seed, 101),
    )
    left = scenario_generator.make_instance(
        2, 1, 0.8, 1.0, 7001, config_path=config_path
    )
    monkeypatch.setattr(
        scenario_generator.random,
        "Random",
        lambda seed: QuantityControlledRandom(seed, 102),
    )
    right = scenario_generator.make_instance(
        2, 1, 0.8, 1.0, 7001, config_path=config_path
    )

    assert left.jobs[0].quantity == right.jobs[0].quantity == 101
    assert left.jobs[1].release_time > 0 and right.jobs[1].release_time > 0
    assert left.jobs[1].quantity != right.jobs[1].quantity
    left_public_commitments = tuple(entity.checkpoints for entity in left.entities)
    right_public_commitments = tuple(entity.checkpoints for entity in right.entities)
    assert left_public_commitments == right_public_commitments
    assert left.forecast_provider is not None
    assert right.forecast_provider is not None
    assert (
        left.forecast_provider.snapshot_at(0)
        == right.forecast_provider.snapshot_at(0)
    )


def test_paper_contract_a_hidden_job_details_do_not_change_current_legal_state():
    visible = _job(
        0,
        release=0,
        due=20,
        quantity=2,
        operations=(_operation(0, 0, 0, (0, 3), (1, 5)),),
    )
    hidden_left = _job(
        1,
        release=10,
        due=30,
        quantity=1,
        operations=(_operation(1, 1, 0, (0, 2)),),
    )
    hidden_right = _job(
        1,
        release=10,
        due=300,
        quantity=99,
        operations=(
            _operation(1, 1, 0, (1, 40)),
            _operation(1, 2, 1, (0, 50)),
        ),
    )
    entity = ServiceEntity(0, (Checkpoint(25, 5.0, 1.0),))
    common = dict(entities=(entity,), machines=(Machine(0), Machine(1)))
    left_env = CAPPORuleSelectionEnv(
        FJSPInstance(jobs=(visible, hidden_left), **common)
    )
    right_env = CAPPORuleSelectionEnv(
        FJSPInstance(jobs=(visible, hidden_right), **common)
    )

    left_observation = left_env.reset()
    right_observation = right_env.reset()

    assert tuple(job.job_id for job in left_env._problem_view().jobs) == (0,)
    assert tuple(job.job_id for job in right_env._problem_view().jobs) == (0,)
    assert left_observation == right_observation
    assert left_env.current_metrics == right_env.current_metrics


# ---------------------------------------------------------------------------
# B. Compound-Poisson forecast contract (sec4:68--84)
# ---------------------------------------------------------------------------


def test_paper_contract_b_accumulates_variances_before_one_conservative_correction():
    snapshot = ForecastSnapshot(
        update_epoch=0,
        buckets_by_entity={
            0: (
                ForecastBucket(0, 10, d_hat=10.0, sigma=2.0),
                ForecastBucket(10, 20, d_hat=6.0, sigma=3.0),
            )
        },
    )

    actual = compute_arrival_forecast_summary(snapshot, 0, 20, 0, 1.0)
    accumulated_sigma = math.sqrt(2.0**2 + 3.0**2)
    expected = (16.0, accumulated_sigma, max(0.0, 16.0 - accumulated_sigma))
    assert actual == pytest.approx(expected)


def test_paper_contract_b_short_window_may_have_zero_conservative_quantity():
    snapshot = ForecastSnapshot(
        update_epoch=0,
        buckets_by_entity={
            0: (ForecastBucket(0, 10, d_hat=1.0, sigma=2.0),)
        },
    )

    mean, sigma, conservative = compute_arrival_forecast_summary(
        snapshot, 0, 1, 0, 1.0
    )
    assert mean == pytest.approx(0.1)
    assert sigma == pytest.approx(math.sqrt(0.4))
    assert conservative == 0.0


def test_paper_contract_b_conservative_value_is_prefix_maximum():
    snapshot = ForecastSnapshot(
        update_epoch=0,
        buckets_by_entity={
            0: (
                ForecastBucket(0, 10, d_hat=10.0, sigma=0.0),
                ForecastBucket(10, 20, d_hat=0.0, sigma=100.0),
            )
        },
    )

    mean, sigma, conservative = compute_arrival_forecast_summary(
        snapshot, 0, 20, 0, 1.0
    )
    assert (mean, sigma) == pytest.approx((10.0, 100.0))
    assert conservative == pytest.approx(10.0)


def test_paper_contract_b_compound_poisson_moments_are_hand_calculated():
    provider = build_compound_poisson_forecast_provider(
        entity_ids=(0, 1),
        total_arrival_rate=2.0,
        quantity_mean=3.0,
        quantity_second_moment=11.0,
        max_time=10,
    )
    bucket = provider.snapshot_at(0).for_entity(0)[0]
    # lambda_r = 2 / 2 = 1, hence E[Q(10)] = 1*10*3 and
    # Var[Q(10)] = 1*10*11.  These values are an independent paper oracle.
    assert bucket.d_hat == pytest.approx(30.0)
    assert bucket.sigma**2 == pytest.approx(110.0)


def test_paper_contract_b_first_coverage_time_uses_cumulative_moments():
    # Here mu(s)=s and v(s)=s.  With kappa=1 and required quantity 6,
    # s - sqrt(s) = 6 has its first positive solution at s=9.
    snapshot = ForecastSnapshot(
        update_epoch=0,
        buckets_by_entity={
            0: (ForecastBucket(0, 100, d_hat=100.0, sigma=10.0),)
        },
    )
    threshold = compute_release_threshold_time(
        snapshot, 0, 100, 0, 1.0, future_required=6.0
    )
    assert threshold == pytest.approx(9.0, abs=1e-8)


def test_paper_contract_b_formal_scenario_has_forecast_through_latest_checkpoint():
    instance = make_scenario_instance("center", 25_000)
    assert instance.forecast_provider is not None
    snapshot = instance.forecast_provider.snapshot_at(0)
    for entity in instance.entities:
        buckets = snapshot.for_entity(entity.entity_id)
        assert buckets
        assert max(bucket.end for bucket in buckets) >= entity.checkpoints[-1].time

    config = scenario_generator.load_scenario_config()
    fixed = config["fixed"]
    op_mean = sum(fixed["operations_per_job"]) / 2.0
    proc_mean = sum(fixed["processing_time"]) / 2.0
    mean_interval = op_mean * proc_mean / (
        scenario_generator.scenario_spec("center").load
        * scenario_generator.scenario_spec("center").machines
    )
    entity_rate = (1.0 / mean_interval) / instance.num_entities
    quantity_values = tuple(
        range(fixed["quantity"][0], fixed["quantity"][1] + 1)
    )
    quantity_mean = sum(quantity_values) / len(quantity_values)
    quantity_second_moment = sum(value * value for value in quantity_values) / len(
        quantity_values
    )
    horizon = max(
        checkpoint.time
        for entity in instance.entities
        for checkpoint in entity.checkpoints
    )
    bucket = snapshot.for_entity(instance.entities[0].entity_id)[0]
    assert bucket.d_hat == pytest.approx(entity_rate * horizon * quantity_mean)
    assert bucket.sigma**2 == pytest.approx(
        entity_rate * horizon * quantity_second_moment
    )
    assert instance.metadata["realization_used_to_build_forecast"] is False


def test_paper_contract_b_forecast_never_counts_as_real_checkpoint_supply():
    job = _job(
        0,
        release=0,
        due=30,
        quantity=5,
        operations=(_operation(0, 0, 0, (0, 1)),),
    )
    provider = build_compound_poisson_forecast_provider(
        entity_ids=(0,),
        total_arrival_rate=100.0,
        quantity_mean=100.0,
        quantity_second_moment=10_000.0,
        max_time=10,
    )
    instance = FJSPInstance(
        jobs=(job,),
        entities=(ServiceEntity(0, (Checkpoint(10, 5.0, 2.0),)),),
        machines=(Machine(0),),
        forecast_provider=provider,
    )
    state = ScheduleState(current_time=20, completed_jobs={0: 20})

    objective = compute_objective(instance, state)
    assert objective.per_checkpoint_completed_quantity[(0, 0)] == 0
    assert objective.weighted_checkpoint_shortfall == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# C. Sec4 theoretical quantities
# ---------------------------------------------------------------------------


def test_paper_contract_c_clb_uses_minimum_time_for_every_unstarted_operation():
    job = _job(
        0,
        release=0,
        due=30,
        quantity=1,
        operations=(
            _operation(0, 0, 0, (0, 9), (1, 3)),
            _operation(0, 1, 1, (0, 2)),
        ),
    )
    state = ScheduleState(current_time=5, machine_available_times={0: 5, 1: 5})
    expected_clb = 5 + 3 + 2

    assert optimistic_candidate_completion(job, state, 0) == expected_clb
    assert optimistic_candidate_completion(job, state, 1) == expected_clb


def test_paper_contract_c_clb_running_operations_and_monotonic_transitions():
    job = _job(
        0,
        release=0,
        due=50,
        quantity=1,
        operations=(
            _operation(0, 0, 0, (0, 7), (1, 3)),
            _operation(0, 1, 1, (0, 5), (1, 2)),
            _operation(0, 2, 2, (0, 6), (1, 4)),
        ),
    )
    state = ScheduleState(current_time=5, machine_available_times={0: 5, 1: 5})
    before_start = optimistic_job_completion_lower_bound(job, state)
    assert before_start == 5 + 3 + 2 + 4

    state.mark_operation_started(0, 0, 0, 5, 12)
    after_start = optimistic_job_completion_lower_bound(job, state)
    assert after_start == 12 + 2 + 4
    assert after_start >= before_start

    state.current_time = 8
    assert optimistic_job_completion_lower_bound(job, state) == after_start
    state.current_time = 12
    state.mark_operation_completed(0, 0, 0)
    after_completion = optimistic_job_completion_lower_bound(job, state)
    assert after_completion == after_start

    state.mark_operation_started(0, 1, 0, 12, 17)
    running_middle = optimistic_job_completion_lower_bound(job, state)
    assert running_middle == 17 + 4
    assert running_middle >= after_completion
    state.current_time = 17
    state.mark_operation_completed(0, 1, 1)
    assert optimistic_job_completion_lower_bound(job, state) == running_middle

    state.mark_operation_started(0, 2, 0, 17, 23)
    state.current_time = 20
    assert optimistic_job_completion_lower_bound(job, state) == 23
    state.current_time = 23
    state.mark_operation_completed(0, 2, 2)
    state.mark_job_completed(0, 23)
    assert optimistic_job_completion_lower_bound(job, state) == 23


def test_paper_contract_c_omega_boundaries_and_online_membership():
    early = _job(
        0, release=0, due=30, quantity=1,
        operations=(_operation(0, 0, 0, (0, 4)),),
    )
    equal = _job(
        1, release=0, due=30, quantity=1,
        operations=(_operation(1, 1, 0, (0, 5)),),
    )
    late = _job(
        2, release=0, due=30, quantity=1,
        operations=(_operation(2, 2, 0, (0, 6)),),
    )
    hidden = _job(
        3, release=6, due=30, quantity=1,
        operations=(_operation(3, 3, 0, (0, 1)),),
    )
    other_entity = _job(
        4, release=0, due=30, quantity=1, entity=1,
        operations=(_operation(4, 4, 0, (0, 1)),),
    )
    completed = _job(
        5, release=0, due=30, quantity=1,
        operations=(_operation(5, 5, 0, (0, 1)),),
    )
    instance = FJSPInstance(
        jobs=(early, equal, late, hidden, other_entity, completed),
        entities=(
            ServiceEntity(0, (Checkpoint(10, 5.0, 1.0),)),
            ServiceEntity(1, (Checkpoint(10, 1.0, 1.0),)),
        ),
        machines=(Machine(0),),
    )
    state = ScheduleState(
        current_time=5,
        machine_available_times={0: 5},
        completed_jobs={5: 4},
    )
    omega = timely_visible_jobs(instance, state, 0, 0)
    assert tuple(job.job_id for job in omega) == (0, 1)


def test_paper_contract_c_timely_set_uses_paper_clb_not_trigger_machine_duration():
    job = _job(
        0,
        release=0,
        due=30,
        quantity=2,
        operations=(
            _operation(0, 0, 0, (0, 9), (1, 3)),
            _operation(0, 1, 1, (0, 2)),
        ),
    )
    instance = FJSPInstance(
        jobs=(job,),
        entities=(ServiceEntity(0, (Checkpoint(11, 2.0, 1.0),)),),
        machines=(Machine(0), Machine(1)),
    )
    state = ScheduleState(current_time=5, machine_available_times={0: 5, 1: 5})
    metrics = build_ca_decision_metrics(instance, state, machine_id=0)

    assert metrics.candidates[0].optimistic_completion == 10
    assert metrics.candidates[0].reachable_unmet_checkpoints == (0,)


def test_paper_contract_c_confirmed_final_operation_supply_reduces_capacity_gap():
    confirmed = _job(
        0,
        release=0,
        due=20,
        quantity=4,
        operations=(_operation(0, 0, 0, (0, 4)),),
    )
    additional = _job(
        1,
        release=0,
        due=20,
        quantity=6,
        operations=(_operation(1, 1, 0, (1, 2)),),
    )
    instance = FJSPInstance(
        jobs=(confirmed, additional),
        entities=(ServiceEntity(0, (Checkpoint(10, 10.0, 1.0),)),),
        machines=(Machine(0), Machine(1)),
    )
    state = ScheduleState(current_time=0, machine_available_times={0: 4, 1: 0})
    state.ongoing_operations[(0, 0)] = ScheduledOperation(0, 0, 0, 0, 4)
    analysis = analyze_checkpoint_capacity(
        instance, state, 0, 0, residual_gap=10.0
    )
    workload, capacity = compute_optimistic_checkpoint_workload_capacity(
        instance, state, 0, 0, residual_gap=10.0
    )

    # A=4, G+=6, nu=2/6; hence the necessary workload is G+*nu=2.
    assert tuple(job.job_id for job in analysis.confirmed_jobs) == (0,)
    assert tuple(job.job_id for job in analysis.additional_jobs) == (1,)
    assert analysis.confirmed_supply == pytest.approx(4.0)
    assert analysis.additional_gap == pytest.approx(6.0)
    assert analysis.minimum_work_per_quantity == pytest.approx(2 / 6)
    assert analysis.applicable is True
    assert analysis.capacity_shortfall == pytest.approx(0.0)
    assert workload == pytest.approx(2.0)
    assert capacity == pytest.approx(10.0)


def test_paper_contract_c_capacity_is_inapplicable_when_visible_quantity_is_too_small():
    job = _job(
        0,
        release=0,
        due=20,
        quantity=2,
        operations=(_operation(0, 0, 0, (0, 2)),),
    )
    instance = FJSPInstance(
        jobs=(job,),
        entities=(ServiceEntity(0, (Checkpoint(10, 10.0, 1.0),)),),
        machines=(Machine(0),),
    )
    state = ScheduleState(current_time=0, machine_available_times={0: 0})
    metrics = build_ca_decision_metrics(instance, state, machine_id=0)
    # sum(q in Omega+)=2 < G+=10, so iota_cap=0 and Delta_cap/G+=0.
    checkpoint = metrics.checkpoint(0, 0)
    assert checkpoint.additional_gap == pytest.approx(10.0)
    assert checkpoint.additional_candidate_quantity == pytest.approx(2.0)
    assert checkpoint.capacity_applicable is False
    assert checkpoint.capacity_shortfall == pytest.approx(0.0)


def test_paper_contract_c_two_entities_multicheckpoint_capacity_case():
    confirmed = _job(
        0, release=0, due=30, quantity=4,
        operations=(_operation(0, 0, 0, (0, 2)),),
    )
    additional = _job(
        1, release=0, due=30, quantity=6,
        operations=(_operation(1, 1, 0, (1, 2)),),
    )
    other_entity = _job(
        2, release=0, due=30, quantity=100, entity=1,
        operations=(_operation(2, 2, 0, (1, 1)),),
    )
    instance = FJSPInstance(
        jobs=(confirmed, additional, other_entity),
        entities=(
            ServiceEntity(0, (Checkpoint(4, 10.0, 1.0), Checkpoint(12, 15.0, 2.0))),
            ServiceEntity(1, (Checkpoint(4, 100.0, 1.0), Checkpoint(12, 100.0, 2.0))),
        ),
        machines=(Machine(0), Machine(1)),
    )
    state = ScheduleState(current_time=0, machine_available_times={0: 2, 1: 3})
    state.ongoing_operations[(0, 0)] = ScheduledOperation(0, 0, 0, 0, 2)

    analysis = analyze_checkpoint_capacity(instance, state, 0, 0)
    assert tuple(job.job_id for job in analysis.timely_jobs) == (0, 1)
    assert tuple(job.job_id for job in analysis.confirmed_jobs) == (0,)
    assert tuple(job.job_id for job in analysis.additional_jobs) == (1,)
    assert analysis.confirmed_supply == pytest.approx(4.0)
    assert analysis.additional_gap == pytest.approx(6.0)
    assert analysis.minimum_work_per_quantity == pytest.approx(1 / 3)
    assert analysis.eligible_machine_ids == (1,)
    assert analysis.available_capacity == pytest.approx(1.0)
    assert analysis.applicable is True
    assert analysis.capacity_shortfall == pytest.approx(3.0)


def test_paper_contract_c_target_falls_back_to_earliest_positive_gap_checkpoint():
    job = _job(
        0,
        release=0,
        due=40,
        quantity=3,
        operations=(_operation(0, 0, 0, (0, 30)),),
    )
    instance = FJSPInstance(
        jobs=(job,),
        entities=(
            ServiceEntity(
                0,
                (Checkpoint(10, 5.0, 1.0), Checkpoint(20, 8.0, 2.0)),
            ),
        ),
        machines=(Machine(0),),
    )
    candidate = CandidateMetrics(0, 0, 0, 0, 30, 0, 30, 0.0, ())
    metrics = CADecisionMetrics(
        0,
        0,
        (candidate,),
        {
            (0, 0): _checkpoint_metrics(0, time=10, weight=1, required=5, gap=5),
            (0, 1): _checkpoint_metrics(1, time=20, weight=2, required=8, gap=8),
        },
        0,
    )
    target = select_target_checkpoint(metrics, candidate)
    assert target is not None
    assert target.checkpoint_index == 0
    assert candidate.reachable_unmet_checkpoints == ()
    assert min(float(job.quantity), target.residual_gap) / target.residual_gap == pytest.approx(0.6)


def test_paper_contract_c_target_prefers_earliest_reachable_positive_gap():
    candidate = CandidateMetrics(0, 0, 0, 0, 3, 0, 8, 0.0, (1, 2))
    metrics = CADecisionMetrics(
        5,
        0,
        (candidate,),
        {
            (0, 0): _checkpoint_metrics(0, time=7, weight=1, required=5, gap=5),
            (0, 1): _checkpoint_metrics(1, time=10, weight=1, required=5, gap=5),
            (0, 2): _checkpoint_metrics(2, time=12, weight=1, required=5, gap=5),
        },
        0,
    )
    target = select_target_checkpoint(metrics, candidate)
    assert target is not None
    assert target.checkpoint_index == 1


def test_paper_contract_c_target_is_none_without_unsettled_positive_gap():
    candidate = CandidateMetrics(0, 0, 0, 0, 3, 0, 8, 0.0, ())
    metrics = CADecisionMetrics(
        10,
        0,
        (candidate,),
        {
            (0, 0): _checkpoint_metrics(0, time=10, weight=1, required=5, gap=5),
            (0, 1): _checkpoint_metrics(1, time=12, weight=1, required=5, gap=0),
        },
        0,
    )
    assert select_target_checkpoint(metrics, candidate) is None


# ---------------------------------------------------------------------------
# D. Exact 14-dimensional rule-feature contract (sec5:91--122)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PaperFeatureInputs:
    weight: float
    max_weight: float
    gap: float
    required: float
    eligible: bool
    clb_minus_now: float
    checkpoint_minus_now: float
    quantity: float
    multi_numerator: float
    multi_denominator: float
    forecast_shortfall: float
    capacity_shortfall: float
    gap_after_confirmed: float
    capacity_applicable: bool
    due_slack: float
    processing_time: float
    mean_processing_time: float
    processing_denominator: float
    remaining_work: float
    mean_job_work: float
    due_horizon: float


def _ratio(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0.0 else numerator / denominator


def _sigma(value: float) -> float:
    value = max(0.0, value)
    return value / (1.0 + value)


def _sigma_signed(value: float) -> float:
    return value / (1.0 + abs(value))


def _paper_feature_vector(values: PaperFeatureInputs) -> tuple[float, ...]:
    return (
        _ratio(values.weight, values.max_weight)
        * _ratio(values.gap, values.required),
        float(values.eligible),
        _sigma(_ratio(values.clb_minus_now, values.checkpoint_minus_now)),
        _ratio(min(values.quantity, values.gap), values.gap),
        _ratio(values.multi_numerator, values.multi_denominator),
        _ratio(values.forecast_shortfall, values.gap),
        (
            _ratio(values.capacity_shortfall, values.gap_after_confirmed)
            if values.capacity_applicable
            else 0.0
        ),
        float(values.capacity_applicable),
        _sigma_signed(_ratio(values.due_slack, values.mean_job_work)),
        _sigma(_ratio(max(0.0, -values.due_slack), values.mean_job_work)),
        _sigma(_ratio(values.processing_time, values.mean_processing_time)),
        _ratio(values.processing_time, values.processing_denominator),
        _sigma(_ratio(values.remaining_work, values.mean_job_work)),
        (
            _sigma(_ratio(values.due_horizon, values.remaining_work))
            if values.due_horizon > 0.0
            else 0.0
        ),
    )


def test_paper_contract_d_reference_vector_has_exact_values_and_boundaries():
    values = PaperFeatureInputs(
        weight=2,
        max_weight=4,
        gap=5,
        required=10,
        eligible=True,
        clb_minus_now=6,
        checkpoint_minus_now=10,
        quantity=3,
        multi_numerator=18,
        multi_denominator=26,
        forecast_shortfall=2,
        capacity_shortfall=1,
        gap_after_confirmed=4,
        capacity_applicable=True,
        due_slack=-2,
        processing_time=6,
        mean_processing_time=4,
        processing_denominator=10,
        remaining_work=6,
        mean_job_work=6,
        due_horizon=4,
    )
    expected = (
        0.25,
        1.0,
        0.375,
        0.6,
        18 / 26,
        0.4,
        0.25,
        1.0,
        -0.25,
        0.25,
        0.6,
        0.6,
        0.5,
        0.4,
    )
    assert _paper_feature_vector(values) == pytest.approx(expected)

    zero_denominators = replace(
        values,
        max_weight=0,
        gap=0,
        required=0,
        checkpoint_minus_now=0,
        multi_denominator=0,
        gap_after_confirmed=0,
        mean_job_work=0,
        mean_processing_time=0,
        processing_denominator=0,
        remaining_work=0,
        due_horizon=-1,
    )
    boundary = _paper_feature_vector(zero_denominators)
    assert all(math.isfinite(value) for value in boundary)
    assert boundary[0] == boundary[2] == boundary[3] == boundary[4] == 0.0
    assert boundary[5] == boundary[6] == boundary[8] == boundary[9] == 0.0
    assert boundary[10] == boundary[11] == boundary[12] == boundary[13] == 0.0


def test_paper_contract_d_production_vector_matches_hand_calculation():
    job = _job(
        0,
        release=0,
        due=4,
        quantity=3,
        operations=(
            _operation(0, 0, 0, (0, 6), (1, 2)),
            _operation(0, 1, 1, (0, 4)),
        ),
    )
    confirmed = _job(
        1,
        release=0,
        due=20,
        quantity=1,
        operations=(_operation(1, 10, 0, (0, 4)),),
    )
    additional = _job(
        2,
        release=0,
        due=20,
        quantity=1,
        operations=(_operation(2, 20, 0, (0, 2)),),
    )
    instance = FJSPInstance(
        jobs=(job, confirmed, additional),
        entities=(
            ServiceEntity(
                0,
                (Checkpoint(10, 10.0, 2.0), Checkpoint(20, 12.0, 4.0)),
            ),
        ),
        machines=(Machine(0), Machine(1)),
        metadata={
            "feature_normalization_scales": {"bar_p": 4.0, "bar_W": 6.0}
        },
    )
    candidate = CandidateMetrics(0, 0, 0, 0, 6, 4, 6, 18.0, (0, 1))
    second_candidate = CandidateMetrics(2, 0, 20, 0, 2, 0, 2, 6.0, (0, 1))
    metrics = CADecisionMetrics(
        0,
        0,
        (candidate, second_candidate),
        {
            (0, 0): _checkpoint_metrics(
                0,
                time=10,
                weight=2,
                required=10,
                gap=5,
                workload=7,
                capacity=6,
                future_required=2,
                conservative=0,
                insufficiency=2,
            ),
            (0, 1): _checkpoint_metrics(
                1, time=20, weight=4, required=12, gap=4
            ),
        },
        0,
    )
    state = ScheduleState(
        current_time=0,
        machine_available_times={0: 4, 1: 10},
        ongoing_operations={
            (1, 10): ScheduledOperation(1, 10, 0, 0, 4),
        },
    )
    actual_rows = build_extended_theory_features(
        instance,
        state,
        metrics,
        feature_set="paper14",
    )
    expected = (
        0.25,
        1.0,
        0.375,
        0.6,
        18 / 26,
        0.4,
        0.25,
        1.0,
        -0.25,
        0.25,
        0.6,
        0.6,
        0.5,
        0.4,
    )
    assert actual_rows[0] == pytest.approx(expected)
    assert actual_rows[1] == pytest.approx(
        (
            0.25,
            1.0,
            1 / 6,
            0.2,
            6 / 26,
            0.4,
            0.25,
            1.0,
            0.75,
            0.0,
            1 / 3,
            1.0,
            0.25,
            10 / 11,
        )
    )


def test_paper_contract_d_without_target_checkpoint_first_eight_features_are_zero():
    job = _job(
        0,
        release=0,
        due=30,
        quantity=1,
        operations=(_operation(0, 0, 0, (0, 2)),),
    )
    instance = FJSPInstance(
        jobs=(job,),
        entities=(ServiceEntity(0, (Checkpoint(10, 1.0, 1.0),)),),
        machines=(Machine(0),),
        metadata={
            "feature_normalization_scales": {"bar_p": 2.0, "bar_W": 2.0}
        },
    )
    state = ScheduleState(current_time=11, machine_available_times={0: 11})
    candidate = CandidateMetrics(0, 0, 0, 0, 2, 0, 13, 0.0, ())
    metrics = CADecisionMetrics(
        11,
        0,
        (candidate,),
        {(0, 0): _checkpoint_metrics(0, time=10, weight=1, required=1, gap=1)},
        0,
    )
    row = build_extended_theory_features(
        instance, state, metrics, feature_set="paper14"
    )[0]
    assert row[:8] == pytest.approx((0.0,) * 8)


def test_paper_contract_d_feature_order_is_exactly_table_5_2():
    assert feature_names_for("paper14") == PAPER14_FEATURE_NAMES == (
        "gap_pressure",
        "timely_eligibility",
        "checkpoint_slack_urgency",
        "quantity_contribution",
        "multi_checkpoint_value",
        "forecast_deficit",
        "capacity_pressure",
        "capacity_flag",
        "due_date_slack",
        "tardiness_exposure",
        "proc_time",
        "processing_ratio",
        "remaining_work",
        "critical_ratio",
    )


def test_paper_contract_d_multi_checkpoint_value_only_counts_timely_checkpoints():
    job = _job(
        0,
        release=0,
        due=30,
        quantity=2,
        operations=(_operation(0, 0, 0, (0, 12)),),
    )
    instance = FJSPInstance(
        jobs=(job,),
        entities=(
            ServiceEntity(
                0,
                (Checkpoint(10, 5.0, 1.0), Checkpoint(20, 5.0, 2.0)),
            ),
        ),
        machines=(Machine(0),),
        metadata={
            "feature_normalization_scales": {"bar_p": 4.0, "bar_W": 12.0}
        },
    )
    state = ScheduleState(current_time=0, machine_available_times={0: 0})
    candidate = CandidateMetrics(0, 0, 0, 0, 12, 0, 12, 4.0, (1,))
    metrics = CADecisionMetrics(
        0,
        0,
        (candidate,),
        {
            (0, 0): _checkpoint_metrics(
                0, time=10, weight=1, required=5, gap=5
            ),
            (0, 1): _checkpoint_metrics(
                1, time=20, weight=2, required=5, gap=5
            ),
        },
        0,
    )
    row = build_extended_theory_features(
        instance, state, metrics, feature_set="paper14"
    )[0]
    assert row[1] == 1.0  # Eligible for the selected second checkpoint.
    assert row[4] == pytest.approx(4 / 15)  # No value from the missed first one.


def test_paper_contract_d_zero_checkpoint_weights_use_safe_zero_convention():
    job = _job(
        0,
        release=0,
        due=20,
        quantity=1,
        operations=(_operation(0, 0, 0, (0, 2)),),
    )
    instance = FJSPInstance(
        jobs=(job,),
        entities=(ServiceEntity(0, (Checkpoint(10, 5.0, 1.0),)),),
        machines=(Machine(0),),
        metadata={
            "feature_normalization_scales": {"bar_p": 2.0, "bar_W": 2.0}
        },
    )
    candidate = CandidateMetrics(0, 0, 0, 0, 2, 0, 2, 0.0, (0,))
    checkpoint = replace(
        _checkpoint_metrics(0, time=10, weight=1, required=5, gap=5),
        checkpoint_weight=0.0,
    )
    row = build_extended_theory_features(
        instance,
        ScheduleState(current_time=0, machine_available_times={0: 0}),
        CADecisionMetrics(0, 0, (candidate,), {(0, 0): checkpoint}, 0),
        feature_set="paper14",
    )[0]
    assert row[0] == 0.0
    assert row[4] == 0.0
    assert all(math.isfinite(value) for value in row)


def test_paper_contract_d_overdue_slack_and_trigger_machine_dependence():
    job = _job(
        0,
        release=0,
        due=7,
        quantity=1,
        operations=(
            _operation(0, 0, 0, (0, 2), (1, 8)),
            _operation(0, 1, 1, (0, 3)),
        ),
    )
    instance = FJSPInstance(
        jobs=(job,),
        entities=(ServiceEntity(0, (Checkpoint(5, 1.0, 1.0),)),),
        machines=(Machine(0), Machine(1)),
        metadata={
            "feature_normalization_scales": {"bar_p": 4.0, "bar_W": 10.0}
        },
    )
    state = ScheduleState(current_time=10, machine_available_times={0: 10, 1: 10})
    checkpoint = _checkpoint_metrics(0, time=5, weight=1, required=1, gap=1)
    low = CandidateMetrics(0, 0, 0, 0, 2, 3, 15, 0.0, ())
    high = CandidateMetrics(0, 0, 0, 1, 8, 3, 15, 0.0, ())
    low_row = build_extended_theory_features(
        instance,
        state,
        CADecisionMetrics(10, 0, (low,), {(0, 0): checkpoint}, 0),
        feature_set="paper14",
    )[0]
    high_row = build_extended_theory_features(
        instance,
        state,
        CADecisionMetrics(10, 1, (high,), {(0, 0): checkpoint}, 0),
        feature_set="paper14",
    )[0]

    assert low_row[:10] == pytest.approx(high_row[:10])
    assert low_row[12:] == pytest.approx(high_row[12:])
    assert low_row[8] == pytest.approx(-4 / 9)  # sigma_pm(-8/10)
    assert low_row[9] == pytest.approx(4 / 9)   # sigma(8/10)
    assert low_row[10] < high_row[10]
    assert low_row[11] < high_row[11]
    assert low_row[13] == 0.0  # d_j <= t


def test_paper_contract_d_multiple_overdue_candidates_remain_distinguishable():
    jobs = (
        _job(
            0,
            release=0,
            due=7,
            quantity=1,
            operations=(_operation(0, 0, 0, (0, 2)),),
        ),
        _job(
            1,
            release=0,
            due=3,
            quantity=1,
            operations=(_operation(1, 10, 0, (0, 4)),),
        ),
    )
    instance = FJSPInstance(
        jobs=jobs,
        entities=(ServiceEntity(0, (Checkpoint(5, 2.0, 1.0),)),),
        machines=(Machine(0),),
        metadata={
            "feature_normalization_scales": {"bar_p": 4.0, "bar_W": 10.0}
        },
    )
    state = ScheduleState(current_time=10, machine_available_times={0: 10})
    metrics = CADecisionMetrics(
        10,
        0,
        (
            CandidateMetrics(0, 0, 0, 0, 2, 0, 12, 0.0, ()),
            CandidateMetrics(1, 0, 10, 0, 4, 0, 14, 0.0, ()),
        ),
        {(0, 0): _checkpoint_metrics(0, time=5, weight=1, required=2, gap=2)},
        0,
    )
    rows = build_extended_theory_features(
        instance, state, metrics, feature_set="paper14"
    )
    assert all(row[:8] == pytest.approx((0.0,) * 8) for row in rows)
    assert rows[0][8] < 0.0 and rows[1][8] < rows[0][8]
    assert rows[0][9] > 0.0 and rows[1][9] > rows[0][9]


def test_paper_contract_d_formal_scales_are_distribution_fixed_online():
    instance = make_scenario_instance("center", 25000)
    assert resolve_paper_feature_scales(instance) == pytest.approx(
        (PAPER_MEAN_PROCESSING_TIME, PAPER_MEAN_JOB_WORK)
    )
    visible_ids = {
        job.job_id for job in instance.jobs if job.release_time <= 0
    }
    assert instance.forecast_provider is not None
    online = OnlineProblemView.from_instance(
        instance,
        visible_job_ids=visible_ids,
        forecast_snapshot=instance.forecast_provider.snapshot_at(0),
        current_time=0,
    )
    assert resolve_paper_feature_scales(online) == pytest.approx((25.5, 89.25))
    assert len(online.jobs) < len(instance.jobs)


def test_paper_contract_d_rule_entry_uses_paper14_due_slack_position():
    jobs = (
        _job(
            0,
            release=0,
            due=10,
            quantity=1,
            operations=(_operation(0, 0, 0, (0, 2)),),
        ),
        _job(
            1,
            release=0,
            due=5,
            quantity=1,
            operations=(_operation(1, 10, 0, (0, 2)),),
        ),
    )
    instance = FJSPInstance(
        jobs=jobs,
        entities=(ServiceEntity(0, (Checkpoint(100, 2.0, 1.0),)),),
        machines=(Machine(0),),
        metadata={
            "feature_normalization_scales": {"bar_p": 2.0, "bar_W": 2.0}
        },
    )
    rule = SymbolicRule(
        "paper14_cond_false",
        "default due-slack fallback",
        RuleType.TYPE_IV,
        "gap_pressure",
        cond_expr="False",
        feature_set="paper14",
    )
    env = CAPPORuleSelectionEnv(instance)
    env.reset()
    result = env.step_principle(0, rule)
    assert result.decision.job_id == 1


# ---------------------------------------------------------------------------
# E. Heterogeneous graph contract (sec5:19--75)
# ---------------------------------------------------------------------------


def _graph_fixture(current_time: int = 0):
    job = _job(
        0,
        release=0,
        due=30,
        quantity=3,
        operations=(
            _operation(0, 0, 0, (0, 6), (1, 2)),
            _operation(0, 1, 1, (0, 4)),
        ),
    )
    instance = FJSPInstance(
        jobs=(job,),
        entities=(
            ServiceEntity(
                0,
                (Checkpoint(10, 5.0, 1.0), Checkpoint(20, 8.0, 2.0)),
            ),
        ),
        machines=(Machine(0), Machine(1), Machine(2)),
        metadata={
            "feature_normalization_scales": {"bar_p": 3.0, "bar_W": 4.0}
        },
    )
    candidate = CandidateMetrics(
        0, 0, 0, 0, 6, 4, current_time + 6, 0.0, (0, 1)
    )
    metrics = CADecisionMetrics(
        current_time,
        0,
        (candidate,),
        {
            (0, 0): _checkpoint_metrics(
                0, time=10, weight=1, required=5, gap=5
            ),
            (0, 1): _checkpoint_metrics(
                1, time=20, weight=2, required=8, gap=8
            ),
        },
        0,
    )
    state = ScheduleState(
        current_time=current_time,
        machine_available_times={0: current_time, 1: current_time, 2: current_time},
    )
    return instance, state, metrics


def _contract_paper_graph(instance, state, metrics):
    features = build_extended_theory_features(
        instance, state, metrics, feature_set="paper14"
    )
    return build_hetero_graph(
        instance,
        state,
        metrics,
        features,
        feature_set="paper14",
        graph_config=PAPER_GRAPH,
    )


def test_paper_contract_e_node_dimensions_are_14_2_10():
    instance, state, metrics = _graph_fixture()
    graph = _contract_paper_graph(instance, state, metrics)
    assert graph.job_x.shape == (1, 14)
    assert graph.machine_x.shape == (3, 2)
    assert graph.checkpoint_x.shape == (2, 10)


def test_paper_contract_e_checkpoint_at_current_time_is_already_settled():
    instance, state, metrics = _graph_fixture(current_time=10)
    graph = _contract_paper_graph(instance, state, metrics)
    assert graph.num_checkpoints == 1
    assert graph.job_checkpoint_edge_index.shape[1] == 1


def test_paper_contract_e_edge_attributes_use_paper_compressions():
    instance, state, metrics = _graph_fixture()
    graph = _contract_paper_graph(instance, state, metrics)

    assert graph.job_machine_edge_attr[:, 0].tolist() == pytest.approx(
        [2.0 / 3.0, 0.4]
    )
    first_checkpoint_edge = graph.job_checkpoint_edge_attr[0]
    assert first_checkpoint_edge.tolist() == pytest.approx([0.6, 0.5])


def test_paper_contract_e_no_checkpoint_and_no_neighbor_cases_are_finite():
    instance, state, metrics = _graph_fixture(current_time=21)
    graph = _contract_paper_graph(instance, state, metrics)

    assert graph.checkpoint_x.shape == (0, 10)
    assert graph.job_checkpoint_edge_index.shape == (2, 0)
    assert torch.isfinite(graph.machine_x).all()
    # Machine 2 deliberately has no job-machine neighbors.
    assert graph.job_machine_edge_index[1].tolist() == [0, 1]
    assert graph.global_x[3].item() == 0.0
    assert graph.global_x[5].item() == 0.0
    assert graph.global_x[6].item() == 0.0


def _paper_graph_integration_fixture():
    candidate_zero = _job(
        0,
        release=0,
        due=8,
        quantity=2,
        entity=0,
        operations=(
            _operation(0, 0, 0, (0, 3), (1, 2)),
            _operation(0, 1, 1, (0, 2)),
        ),
    )
    candidate_one = _job(
        1,
        release=0,
        due=6,
        quantity=1,
        entity=1,
        operations=(_operation(1, 10, 0, (0, 5)),),
    )
    confirmed = _job(
        2,
        release=0,
        due=20,
        quantity=1,
        entity=0,
        operations=(_operation(2, 20, 0, (1, 4)),),
    )
    instance = FJSPInstance(
        jobs=(candidate_zero, candidate_one, confirmed),
        entities=(
            ServiceEntity(
                0,
                (Checkpoint(10, 3.0, 1.0), Checkpoint(20, 5.0, 2.0)),
            ),
            ServiceEntity(
                1,
                (Checkpoint(12, 1.0, 1.0), Checkpoint(22, 2.0, 3.0)),
            ),
        ),
        machines=(Machine(0), Machine(1)),
        metadata={
            "feature_normalization_scales": {"bar_p": 4.0, "bar_W": 8.0}
        },
    )
    state = ScheduleState(
        current_time=0,
        machine_available_times={0: 0, 1: 4},
        ongoing_operations={
            (2, 20): ScheduledOperation(2, 20, 1, 0, 4),
        },
    )
    metrics = build_ca_decision_metrics(instance, state, machine_id=0)
    return instance, state, metrics


def test_paper_contract_e_c_d_e_job_features_are_identical_end_to_end():
    instance, state, metrics = _paper_graph_integration_fixture()
    candidates = tuple(sorted(metrics.candidates, key=lambda item: item.job_id))
    assert tuple(candidate.job_id for candidate in candidates) == (0, 1)

    target = select_target_checkpoint(metrics, candidates[0])
    assert target is not None and target.checkpoint_index == 0
    capacity = analyze_checkpoint_capacity(instance, state, 0, 0)
    assert tuple(job.job_id for job in capacity.confirmed_jobs) == (2,)
    assert capacity.confirmed_supply == 1.0

    paper14 = build_extended_theory_features(
        instance, state, metrics, feature_set="paper14"
    )
    graph = build_hetero_graph(
        instance,
        state,
        metrics,
        paper14,
        feature_set="paper14",
        graph_config=PAPER_GRAPH,
    )
    expected_tensor = torch.tensor(paper14, dtype=torch.float32)
    assert graph.job_ids == (0, 1)
    assert torch.equal(graph.job_x, expected_tensor)
    assert graph.checkpoint_keys == ((0, 0), (1, 0), (0, 1), (1, 1))
    assert graph.checkpoint_x.shape == (4, 10)
    assert graph.job_checkpoint_edge_index.shape[1] == 4
    assert torch.all((0.0 <= graph.job_machine_edge_attr) & (graph.job_machine_edge_attr <= 1.0))
    assert torch.all((0.0 <= graph.job_checkpoint_edge_attr[:, 0]) & (graph.job_checkpoint_edge_attr[:, 0] <= 1.0))
    assert torch.all((-1.0 < graph.job_checkpoint_edge_attr[:, 1]) & (graph.job_checkpoint_edge_attr[:, 1] < 1.0))


def test_paper_contract_e_paper_graph_is_explicit_and_never_falls_back():
    instance, state, metrics = _paper_graph_integration_fixture()
    paper14 = build_extended_theory_features(
        instance, state, metrics, feature_set="paper14"
    )
    with pytest.raises(ValueError, match="requires feature_set='paper14'"):
        build_hetero_graph(
            instance,
            state,
            metrics,
            paper14,
            feature_set="theory11_plus3",
            graph_config=PAPER_GRAPH,
        )
    with pytest.raises(ValueError, match="exactly match canonical paper14"):
        build_hetero_graph(
            instance,
            state,
            metrics,
            tuple((0.0,) * 14 for _candidate in metrics.candidates),
            feature_set="paper14",
            graph_config=PAPER_GRAPH,
        )


def test_paper_contract_e_hidden_future_job_changes_do_not_change_graph():
    visible = _job(
        0,
        release=0,
        due=8,
        quantity=1,
        operations=(_operation(0, 0, 0, (0, 2)),),
    )
    hidden_left = _job(
        1,
        release=10,
        due=20,
        quantity=1,
        operations=(_operation(1, 10, 0, (0, 3)),),
    )
    hidden_right = _job(
        1,
        release=50,
        due=100,
        quantity=9,
        operations=(
            _operation(1, 10, 0, (0, 30)),
            _operation(1, 11, 1, (0, 40)),
        ),
    )
    common = {
        "entities": (ServiceEntity(0, (Checkpoint(20, 2.0, 1.0),)),),
        "machines": (Machine(0),),
        "metadata": {
            "feature_normalization_scales": {"bar_p": 4.0, "bar_W": 8.0}
        },
    }
    graphs = []
    for hidden in (hidden_left, hidden_right):
        env = CAPPORuleSelectionEnv(
            FJSPInstance(jobs=(visible, hidden), **common)
        )
        env.reset()
        problem = env._problem_view()
        features = build_extended_theory_features(
            problem,
            env._state,
            env.current_metrics,
            feature_set="paper14",
        )
        graphs.append(
            build_hetero_graph(
                problem,
                env._state,
                env.current_metrics,
                features,
                feature_set="paper14",
                graph_config=PAPER_GRAPH,
            )
        )
    left, right = graphs
    assert left.job_ids == right.job_ids == (0,)
    assert left.checkpoint_keys == right.checkpoint_keys
    for name in (
        "job_x",
        "machine_x",
        "checkpoint_x",
        "job_machine_edge_index",
        "job_machine_edge_attr",
        "job_checkpoint_edge_index",
        "job_checkpoint_edge_attr",
        "global_x",
    ):
        assert torch.equal(getattr(left, name), getattr(right, name)), name


def test_paper_contract_e_checkpoint_columns_match_table_5_1():
    instance, state, metrics = _graph_fixture()
    first = replace(
        metrics.checkpoints[(0, 0)],
        checkpoint_weight=2.0,
        required_quantity=10.0,
        visible_coverage=3.0,
        future_required=2.0,
        arrival_insufficiency=1.0,
        release_threshold_time=6.0,
        release_slack=4.0,
        additional_gap=4.0,
        capacity_applicable=True,
        capacity_shortfall=1.0,
    )
    second = replace(metrics.checkpoints[(0, 1)], checkpoint_weight=4.0)
    metrics = replace(
        metrics,
        checkpoints={(0, 0): first, (0, 1): second},
    )
    graph = _contract_paper_graph(instance, state, metrics)
    assert graph.checkpoint_x[0].tolist() == pytest.approx(
        (
            0.5,
            0.5,
            5 / 7,
            0.6,
            0.4,
            0.2,
            0.4,
            1.0,
            0.25,
            1.0,
        )
    )


def test_paper_contract_e_existing_hgt_accepts_explicit_paper_dimensions():
    instance, state, metrics = _paper_graph_integration_fixture()
    graph = _contract_paper_graph(instance, state, metrics)
    encoder = HGTEncoder(
        job_input_dim=14,
        machine_input_dim=2,
        checkpoint_input_dim=10,
        global_input_dim=7,
        hidden_dim=8,
        num_layers=1,
        num_heads=2,
    )
    encoded = encoder(graph)
    assert encoded.shape == (8,)
    assert torch.isfinite(encoded).all()


def test_paper_contract_e_paper_mask_adapter_uses_formal_column_layout():
    instance, state, metrics = _paper_graph_integration_fixture()
    graph = _contract_paper_graph(instance, state, metrics)
    graph.job_x[:, 5] = 0.5
    graph.checkpoint_x[:, 5:8] = 0.5
    graph.global_x[5] = 1.0
    masked = apply_information_masks(
        graph,
        feature_set="paper14",
        mask_forecast=True,
    )
    assert masked.graph_config == PAPER_GRAPH
    assert torch.all(masked.job_x[:, 5] == 0.0)
    assert torch.all(masked.checkpoint_x[:, 5:8] == 0.0)
    assert masked.global_x[5].item() == 0.0
    assert torch.all(graph.job_x[:, 5] == 0.5)
    with pytest.raises(ValueError, match="feature_set='paper14'"):
        apply_information_masks(graph, mask_forecast=True)


def test_paper_contract_e_checkpoint_edge_margin_matches_timely_eligibility():
    job = _job(
        0,
        release=0,
        due=30,
        quantity=2,
        operations=(_operation(0, 0, 0, (0, 12)),),
    )
    instance = FJSPInstance(
        jobs=(job,),
        entities=(
            ServiceEntity(
                0,
                (Checkpoint(10, 2.0, 1.0), Checkpoint(20, 2.0, 2.0)),
            ),
        ),
        machines=(Machine(0),),
        metadata={
            "feature_normalization_scales": {"bar_p": 4.0, "bar_W": 4.0}
        },
    )
    state = ScheduleState(current_time=0, machine_available_times={0: 0})
    metrics = build_ca_decision_metrics(instance, state, machine_id=0)
    graph = _contract_paper_graph(instance, state, metrics)
    assert timely_visible_jobs(instance, state, 0, 0) == ()
    assert tuple(item.job_id for item in timely_visible_jobs(instance, state, 0, 1)) == (0,)
    assert graph.job_checkpoint_edge_attr[:, 1].tolist() == pytest.approx(
        (-1 / 3, 2 / 3)
    )


# ---------------------------------------------------------------------------
# F. Reward scale and deterministic safe rule execution (sec5:124--224)
# ---------------------------------------------------------------------------


def _f_ref_instance() -> FJSPInstance:
    long_early = _job(
        0,
        release=0,
        due=10,
        quantity=1,
        operations=(_operation(0, 0, 0, (0, 20)),),
    )
    short_late = _job(
        1,
        release=0,
        due=100,
        quantity=1,
        operations=(_operation(1, 1, 0, (0, 1)),),
    )
    return FJSPInstance(
        jobs=(long_early, short_late),
        entities=(ServiceEntity(0, (Checkpoint(200, 2.0, 1.0),)),),
        machines=(Machine(0),),
    )


def test_paper_contract_f_f_ref_is_spt_atc_mean_and_library_invariant():
    instance = _f_ref_instance()
    spt = ClassicRule("SPT", ClassicRuleKind.SPT)
    atc = ClassicRule("ATC", ClassicRuleKind.ATC, atc_k=2.0)
    edd = ClassicRule("EDD", ClassicRuleKind.EDD)
    expected = max((_run_fixed_rule(instance, spt) + _run_fixed_rule(instance, atc)) / 2, 1.0)

    two_rule = PPOTrainer(
        _ParameterOnlyAgent(2),
        RuleLibrary((spt, atc)),
        [instance],
    )
    three_rule = PPOTrainer(
        _ParameterOnlyAgent(3),
        RuleLibrary((spt, atc, edd)),
        [instance],
    )
    assert two_rule._compute_instance_f_ref(instance) == pytest.approx(expected)
    assert three_rule._compute_instance_f_ref(instance) == pytest.approx(expected)


def test_paper_contract_f_incremental_reward_reconstructs_true_objective():
    instance = _f_ref_instance()
    rule = ClassicRule("SPT", ClassicRuleKind.SPT)
    env = CAPPORuleSelectionEnv(deepcopy(instance))
    env.reset()
    total_reward = 0.0
    while True:
        result = env.step_principle(0, rule)
        total_reward += result.reward
        if result.terminated:
            break
    assert -total_reward == pytest.approx(
        compute_objective(env.instance, env._state).Z, rel=1e-9, abs=1e-9
    )


def test_paper_contract_f_zero_division_returns_zero_in_rule_expression():
    rule = SymbolicRule(
        "safe_division",
        "paper safe division",
        RuleType.TYPE_I,
        "1 / gap_pressure",
        feature_set="theory11_plus3",
    )
    assert rule.evaluate_batch(((0.0,) * 14,)) == [0.0]


def test_paper_contract_f_invalid_cond_is_false_and_invalid_filter_excludes():
    invalid_cond = SymbolicRule(
        "invalid_cond",
        "invalid cond",
        RuleType.TYPE_IV,
        "gap_pressure",
        cond_expr="math.log(gap_pressure)",
        feature_set="theory11_plus3",
    )
    invalid_filter = SymbolicRule(
        "invalid_filter",
        "invalid filter",
        RuleType.TYPE_IV,
        "gap_pressure",
        filter_expr="math.log(gap_pressure)",
        feature_set="theory11_plus3",
    )
    row = (0.0,) * 14
    assert invalid_cond.is_applicable(row) is False
    assert invalid_filter.evaluate_batch((row,)) == [float("-inf")]


def test_paper_contract_f_all_filtered_candidates_use_true_due_slack_default():
    less_overdue = _job(
        0,
        release=5,
        due=4,
        quantity=1,
        operations=(_operation(0, 0, 0, (0, 1)),),
    )
    more_overdue = _job(
        1,
        release=5,
        due=1,
        quantity=1,
        operations=(_operation(1, 1, 0, (0, 1)),),
    )
    instance = FJSPInstance(
        jobs=(less_overdue, more_overdue),
        entities=(ServiceEntity(0, (Checkpoint(20, 2.0, 1.0),)),),
        machines=(Machine(0),),
    )
    rule = SymbolicRule(
        "filter_all",
        "filter all candidates",
        RuleType.TYPE_IV,
        "gap_pressure",
        filter_expr="0",
        feature_set="theory11_plus3",
    )
    env = CAPPORuleSelectionEnv(instance)
    env.reset()
    result = env.step_principle(0, rule)
    assert result.decision.job_id == 1


def test_paper_contract_f_missing_requested_library_fails_strictly(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError):
        resolve_runtime_library("missing-formal-library.json")
