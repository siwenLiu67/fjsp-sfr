import random
from copy import deepcopy

import pytest

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
from fjsp_sfr.env.ca_rule_selection import CAPPORuleSelectionEnv
from fjsp_sfr.experiments.validate_framework import (
    build_rule_library,
    make_fjsp_instance,
)


def _late_checkpoint_shortfall_instance() -> FJSPInstance:
    """One job finishes at t=1, before an unmet checkpoint at t=10."""

    job = Job(
        job_id=0,
        entity_id=0,
        release_time=0,
        due_date=20,
        quantity=3,
        operations=(
            Operation(
                op_id=0,
                job_id=0,
                sequence_index=0,
                alternatives=(OperationAlternative(0, 1),),
            ),
        ),
    )
    return FJSPInstance(
        jobs=(job,),
        entities=(
            ServiceEntity(
                0,
                (Checkpoint(time=10, required_quantity=10.0, weight=2.5),),
            ),
        ),
        machines=(Machine(0),),
        alpha=1.0,
        beta=1.0,
    )


def _run_policy(instance, library, *, fixed_rule_index=None, random_seed=None):
    rollout_instance = deepcopy(instance)
    env = CAPPORuleSelectionEnv(rollout_instance)
    env.reset()
    rng = random.Random(random_seed)
    cumulative_raw_reward = 0.0
    final_result = None
    while True:
        if fixed_rule_index is None:
            rule_index = rng.randrange(library.action_size())
        else:
            rule_index = fixed_rule_index
        final_result = env.step_principle(
            rule_index,
            library.get_rule(rule_index),
        )
        cumulative_raw_reward += final_result.reward
        if final_result.terminated:
            break
    objective = compute_objective(rollout_instance, env._state)
    return cumulative_raw_reward, objective, env


def test_objective_counts_shortfall_at_checkpoint_after_all_jobs_finish():
    instance = _late_checkpoint_shortfall_instance()
    library = build_rule_library()

    cumulative_reward, objective, env = _run_policy(
        instance,
        library,
        fixed_rule_index=0,
    )

    assert env._state.completed_jobs == {0: 1}
    assert env._state.current_time == 10
    assert objective.per_checkpoint_completed_quantity[(0, 0)] == 3
    assert objective.per_checkpoint_shortfall[(0, 0)] == pytest.approx(7.0)
    assert objective.weighted_checkpoint_shortfall == pytest.approx(17.5)
    assert objective.Z == pytest.approx(17.5)
    assert -cumulative_reward == pytest.approx(objective.Z)


@pytest.mark.parametrize("instance_seed", (31, 32, 33))
def test_normalized_episode_reward_reconstructs_final_objective(instance_seed):
    instance = make_fjsp_instance(
        instance_seed,
        n_jobs=6,
        n_machines=3,
        n_entities=2,
    )
    library = build_rule_library()
    fixed_runs = [
        _run_policy(instance, library, fixed_rule_index=rule_index)
        for rule_index in range(library.action_size())
    ]
    f_ref = max(
        sum(run[1].Z for run in fixed_runs) / len(fixed_runs),
        1.0,
    )

    named_runs = [
        (library.get_rule(index).name, run)
        for index, run in enumerate(fixed_runs)
    ]
    named_runs.append(
        (
            "random_rule_per_decision",
            _run_policy(
                instance,
                library,
                random_seed=100_000 + instance_seed,
            ),
        )
    )

    for policy_name, (raw_reward, objective, env) in named_runs:
        normalized_episode_reward = raw_reward / f_ref
        reconstructed_f = -normalized_episode_reward * f_ref
        difference = reconstructed_f - objective.Z
        assert reconstructed_f == pytest.approx(objective.Z, rel=1e-6), (
            f"seed={instance_seed}, policy={policy_name}, f_ref={f_ref}, "
            f"cumulative_raw_reward={raw_reward}, "
            f"reconstructed_F={reconstructed_f}, objective_F={objective.Z}, "
            f"difference={difference}, terminal_time={env._state.current_time}, "
            f"completed_jobs={env._state.completed_jobs}"
        )
