"""Phase-1 contracts for Paper A Sec. 4.6 and the global reward scale."""

from __future__ import annotations

import dataclasses
import json
import random
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from fjsp_sfr.core.forecast import (
    ForecastBucket,
    ForecastSnapshot,
    empty_forecast_snapshot,
)
from fjsp_sfr.core.models import (
    Checkpoint,
    FJSPInstance,
    Job,
    Machine,
    Operation,
    OperationAlternative,
    ServiceEntity,
)
from fjsp_sfr.core.objective import compute_objective
from fjsp_sfr.core.online import OnlineProblemView
from fjsp_sfr.core.state import ScheduleState
from fjsp_sfr.dispatch.certificates import (
    CertificateContradictionError,
    CertificateHistory,
    ShortfallInterval,
    compute_online_certificate,
    compute_privileged_certificate,
    intersect_intervals,
)
from fjsp_sfr.dispatch.classic_rules import ClassicRule, ClassicRuleKind
from fjsp_sfr.dispatch.theory_features import build_extended_theory_features
from fjsp_sfr.env.ca_rule_selection import CAPPORuleSelectionEnv
from fjsp_sfr.generation.scale_config import make_scaled_instance
from fjsp_sfr.generation.scenario_generator import all_seed_partitions
from fjsp_sfr.rl.graph import build_hetero_graph
from fjsp_sfr.rl.ppo_agent import HGTPPOAgent
from fjsp_sfr.rl.ppo_trainer import PPOTrainer
from fjsp_sfr.rl.reward_scale import (
    compute_global_reward_scale,
    load_global_reward_scale,
    write_global_reward_scale,
)


def _operation(job_id: int, op_id: int, machine: int, processing: int) -> Operation:
    return Operation(
        op_id=op_id,
        job_id=job_id,
        sequence_index=0,
        alternatives=(OperationAlternative(machine, processing),),
    )


def _certificate_instance(
    *, future_release: int = 5, future_quantity: int = 4
) -> FJSPInstance:
    jobs = (
        Job(0, 0, 0, 20, 2, (_operation(0, 0, 0, 3),)),
        Job(1, 0, 0, 20, 3, (_operation(1, 1, 1, 2),)),
        Job(
            2,
            0,
            future_release,
            20,
            future_quantity,
            (_operation(2, 2, 1, 2),),
        ),
        Job(3, 0, 9, 20, 5, (_operation(3, 3, 1, 2),)),
    )
    return FJSPInstance(
        jobs=jobs,
        entities=(ServiceEntity(0, (Checkpoint(10, 10.0, 2.0),)),),
        machines=(Machine(0), Machine(1)),
        metadata={
            "certificate_public_bounds": {
                "planned_total_jobs": 4,
                "quantity_upper_bound": 5,
                "minimum_job_work": 2,
            }
        },
    )


def _manual_state() -> ScheduleState:
    state = ScheduleState(
        current_time=2,
        machine_available_times={0: 4, 1: 0},
    )
    state.mark_operation_started(0, 0, 0, 1, 4)
    return state


def _online_view(instance: FJSPInstance, state: ScheduleState) -> OnlineProblemView:
    return OnlineProblemView.from_instance(
        instance,
        visible_job_ids={0, 1},
        forecast_snapshot=empty_forecast_snapshot((0,)),
        current_time=state.current_time,
    )


def test_hand_calculated_online_and_privileged_certificates():
    instance = _certificate_instance()
    state = _manual_state()
    online = compute_online_certificate(_online_view(instance, state), state, 0, 0)
    assert online is not None
    assert online.completed_supply == pytest.approx(0.0)
    assert online.confirmed_supply == pytest.approx(2.0)
    assert online.visible_supply_bound == pytest.approx(3.0)
    assert online.future_supply_bound == pytest.approx(10.0)
    assert online.instantaneous == ShortfallInterval(0.0, 8.0)

    privileged = compute_privileged_certificate(instance, state, online)
    assert privileged.future_job_count == 1
    assert privileged.future_supply_bound == pytest.approx(4.0)
    assert privileged.instantaneous == ShortfallInterval(1.0, 8.0)


def test_unknown_industrial_bounds_use_a_rigorous_loose_online_bound():
    instance = _certificate_instance()
    instance.metadata = {}
    state = _manual_state()
    online = compute_online_certificate(_online_view(instance, state), state, 0, 0)
    assert online is not None
    assert online.future_supply_bound == float("inf")
    assert online.instantaneous.lower == 0.0
    assert online.instantaneous.upper == 8.0


def test_soft_forecast_never_changes_hard_certificate_endpoints():
    instance = _certificate_instance()
    state = _manual_state()
    quiet = _online_view(instance, state)
    extreme = dataclasses.replace(
        quiet,
        forecast_snapshot=ForecastSnapshot(
            0,
            {0: (ForecastBucket(0, 10, d_hat=10_000.0, sigma=0.0),)},
        ),
    )
    assert compute_online_certificate(quiet, state, 0, 0) == (
        compute_online_certificate(extreme, state, 0, 0)
    )


def test_settled_or_zero_gap_checkpoints_do_not_create_heads():
    instance = _certificate_instance()
    state = _manual_state()
    state.current_time = 10
    assert compute_online_certificate(
        _online_view(instance, state), state, 0, 0
    ) is None

    completed = _certificate_instance()
    completed.entities = (ServiceEntity(0, (Checkpoint(10, 2.0, 2.0),)),)
    state = ScheduleState(current_time=2, machine_available_times={0: 0, 1: 0})
    state.completed_jobs[0] = 1
    state.completed_operations.add((0, 0))
    state.last_completed_op_index[0] = 0
    assert compute_online_certificate(
        _online_view(completed, state), state, 0, 0
    ) is None


def test_interval_history_only_tightens_and_rejects_contradictions():
    first = ShortfallInterval(1.0, 8.0)
    second = intersect_intervals(first, ShortfallInterval(2.0, 7.0))
    exact = intersect_intervals(second, ShortfallInterval(3.0, 3.0))
    assert second == ShortfallInterval(2.0, 7.0)
    assert exact.width == 0.0
    with pytest.raises(CertificateContradictionError, match="contradictory"):
        intersect_intervals(exact, ShortfallInterval(4.0, 5.0), checkpoint_key=(0, 0))


def test_hidden_future_changes_only_privileged_certificate_and_not_public_graph():
    first = _certificate_instance(future_release=5, future_quantity=4)
    second = _certificate_instance(future_release=9, future_quantity=1)
    env_a = CAPPORuleSelectionEnv(first, enable_privileged_certificates=True)
    env_b = CAPPORuleSelectionEnv(second, enable_privileged_certificates=True)
    observation_a = env_a.reset()
    observation_b = env_b.reset()

    assert observation_a == observation_b
    assert env_a.online_certificates == env_b.online_certificates
    assert env_a.critic_certificate_record != env_b.critic_certificate_record
    public_checkpoint = env_a.current_metrics.checkpoint(0, 0)
    public_certificate = env_a.online_certificates[0]
    assert public_checkpoint.online_shortfall_lower == (
        public_certificate.historical.lower
    )
    assert public_checkpoint.online_shortfall_upper == (
        public_certificate.historical.upper
    )
    assert not any(
        "priv" in field.name for field in dataclasses.fields(observation_a)
    )
    assert not any(
        "priv" in field.name for field in dataclasses.fields(env_a.current_metrics)
    )

    problem_a, problem_b = env_a._problem_view(), env_b._problem_view()
    features_a = build_extended_theory_features(
        problem_a, env_a._state, env_a.current_metrics
    )
    features_b = build_extended_theory_features(
        problem_b, env_b._state, env_b.current_metrics
    )
    graph_a = build_hetero_graph(
        problem_a, env_a._state, env_a.current_metrics, features_a
    )
    graph_b = build_hetero_graph(
        problem_b, env_b._state, env_b.current_metrics, features_b
    )
    for field in (
        "job_x",
        "machine_x",
        "checkpoint_x",
        "job_machine_edge_index",
        "job_machine_edge_attr",
        "job_checkpoint_edge_index",
        "job_checkpoint_edge_attr",
        "global_x",
    ):
        assert torch.equal(getattr(graph_a, field), getattr(graph_b, field))

    torch.manual_seed(19)
    agent = HGTPPOAgent(action_size=2, hidden_dim=16, num_layers=1, num_heads=4)
    agent.eval()
    mask = torch.ones(2, dtype=torch.bool)
    with torch.no_grad():
        logits_a, _ = agent(graph_a, mask)
        logits_b, _ = agent(graph_b, mask)
    assert torch.equal(logits_a, logits_b)


def test_privileged_record_is_opt_in_and_snapshot_restores_histories():
    public_env = CAPPORuleSelectionEnv(_certificate_instance())
    public_env.reset()
    with pytest.raises(RuntimeError, match="not enabled"):
        _ = public_env.critic_certificate_record

    env = CAPPORuleSelectionEnv(
        _certificate_instance(), enable_privileged_certificates=True
    )
    env.reset()
    snapshot = env.snapshot()
    restored = CAPPORuleSelectionEnv.from_snapshot(snapshot)
    assert restored.online_certificates == env.online_certificates
    assert restored.critic_certificate_record == env.critic_certificate_record
    assert restored._certificate_history == env._certificate_history

    result_a = env.step_candidate(0)
    result_b = restored.step_candidate(0)
    assert result_a.observation == result_b.observation
    assert restored.online_certificates == env.online_certificates
    assert restored.critic_certificate_record == env.critic_certificate_record


@pytest.mark.parametrize("seed", (31_001, 31_002, 31_003))
def test_certificates_cover_true_shortfall_on_reachable_random_trajectories(seed):
    instance = make_scaled_instance(seed, scale="S", difficulty="balanced")
    env = CAPPORuleSelectionEnv(instance, enable_privileged_certificates=True)
    env.reset()
    rng = random.Random(seed)
    records = []
    history_by_key: dict[tuple[int, int], ShortfallInterval] = {}
    while env.current_metrics and env.current_metrics.is_active:
        privileged_by_key = {
            item.checkpoint_key: item
            for item in env.critic_certificate_record.certificates
        }
        for online in env.online_certificates:
            private = privileged_by_key[online.checkpoint_key]
            assert 0.0 <= online.historical.lower <= private.historical.lower
            assert private.historical.upper <= online.historical.upper
            assert online.historical.upper <= online.required_quantity
            previous = history_by_key.get(online.checkpoint_key)
            if previous is not None:
                assert online.historical.lower >= previous.lower
                assert online.historical.upper <= previous.upper
            history_by_key[online.checkpoint_key] = online.historical
            records.append((online, private))
        action = rng.randrange(len(env.current_metrics.candidates))
        result = env.step_candidate(action)
        if result.terminated:
            break

    objective = compute_objective(instance, env._state)
    for online, private in records:
        truth = objective.per_checkpoint_shortfall[online.checkpoint_key]
        assert online.historical.lower <= truth <= online.historical.upper
        assert private.historical.lower <= truth <= private.historical.upper


def test_raw_incremental_reward_telescopes_to_unscaled_final_cost():
    instance = make_scaled_instance(31_010, scale="S", difficulty="balanced")
    env = CAPPORuleSelectionEnv(instance)
    env.reset()
    rule = ClassicRule("SPT", ClassicRuleKind.SPT)
    total_raw_reward = 0.0
    while env.current_metrics and env.current_metrics.is_active:
        result = env.step_principle(0, rule)
        total_raw_reward += result.reward
        if result.terminated:
            break
    objective = compute_objective(instance, env._state)
    assert total_raw_reward == pytest.approx(-objective.Z, rel=1e-12, abs=1e-12)


def test_global_reward_scale_manifest_and_trainer_path(tmp_path, monkeypatch):
    instances = (
        make_scaled_instance(31_020, scale="S", difficulty="balanced"),
        make_scaled_instance(31_021, scale="S", difficulty="balanced"),
    )
    scale = compute_global_reward_scale(instances, (31_020, 31_021))
    expected = sum(
        row.f_spt + row.f_atc for row in scale.instances
    ) / (2 * len(scale.instances))
    assert scale.f_ref == pytest.approx(max(1.0, expected))
    path = write_global_reward_scale(tmp_path / "f_ref.json", scale)
    assert load_global_reward_scale(path) == scale
    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["f_ref"] += 1.0
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_global_reward_scale(path)

    from fjsp_sfr.dispatch.classic_rules import build_classic_rule_library

    library = build_classic_rule_library()
    agent = HGTPPOAgent(
        action_size=library.action_size(),
        hidden_dim=16,
        num_layers=1,
        num_heads=4,
    )
    trainer = PPOTrainer(
        agent,
        library,
        instances=[],
        instance_factory=lambda seed: instances[seed % 2],
        base_seed=0,
        f_ref=scale.f_ref,
        reward_reference_mode="global_calibration",
        epochs_per_update=1,
    )

    def forbidden(_instance):
        pytest.fail("global-calibration mode must not recompute per-instance F_ref")

    monkeypatch.setattr(trainer, "_compute_instance_f_ref", forbidden)
    trainer.collect_rollout(2)
    assert trainer.global_f_ref == scale.f_ref


def test_reward_calibration_partition_is_independent():
    partitions = all_seed_partitions()
    calibration = set(partitions["calibration_center"])
    assert calibration
    for name, seeds in partitions.items():
        if name != "calibration_center":
            assert calibration.isdisjoint(seeds), name


def test_committed_global_reward_manifest_uses_frozen_calibration_split():
    path = Path(__file__).parents[1] / "experiments" / "paperA_global_f_ref.json"
    scale = load_global_reward_scale(path)
    assert scale.calibration_partition == "calibration_center"
    assert scale.scenario == "center"
    assert scale.calibration_seeds == all_seed_partitions()["calibration_center"]
    assert scale.f_ref == pytest.approx(1872.4027777777778)


def test_certificate_history_container_keeps_online_and_private_separate():
    history = CertificateHistory()
    history.update_online((0, 0), ShortfallInterval(0.0, 8.0))
    history.update_privileged((0, 0), ShortfallInterval(1.0, 8.0))
    assert history.online[(0, 0)] == ShortfallInterval(0.0, 8.0)
    assert history.privileged[(0, 0)] == ShortfallInterval(1.0, 8.0)
