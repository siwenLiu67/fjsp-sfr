from dataclasses import asdict, is_dataclass

import pytest

from fjsp_sfr.generation.instance_generator import InstanceConfig, generate_instance
from fjsp_sfr.generation.scenario_generator import (
    all_seed_partitions,
    build_exogenous_service_entities,
    load_scenario_config,
    make_scenario_instance,
    scenario_names,
    scenario_spec,
)


@pytest.mark.parametrize("scenario", scenario_names())
def test_all_predefined_scenarios_generate(scenario):
    instance = make_scenario_instance(scenario, 123)
    spec = scenario_spec(scenario)
    assert instance.num_machines == spec.machines
    assert instance.num_jobs == 20 + spec.new_jobs
    assert instance.num_entities == round((20 + spec.new_jobs) / 8)


@pytest.mark.parametrize("scenario", scenario_names())
def test_measured_load_is_near_target(scenario):
    instance = make_scenario_instance(scenario, 321)
    target = scenario_spec(scenario).load
    # A finite Poisson realization fluctuates, especially for the 20-job cells.
    assert instance.metadata["measured_load"] == pytest.approx(target, rel=0.45)


def test_due_date_uses_release_plus_ddt_times_route_mean():
    instance = make_scenario_instance("center", 17)
    ddt = scenario_spec("center").ddt
    for job in instance.jobs:
        mean_work = sum(
            sum(alt.processing_time for alt in operation.alternatives)
            / len(operation.alternatives)
            for operation in job.operations
        )
        assert job.due_date == max(2, job.release_time + 1, round(job.release_time + ddt * mean_work))


def test_each_entity_commitment_uses_public_expectations_not_realized_jobs():
    instance = make_scenario_instance("center", 19)
    config = load_scenario_config()
    expected_q = sum(range(1, 6)) / 5
    shares = instance.metadata["entity_shares"]
    assert sum(shares) == pytest.approx(1.0)
    for entity in instance.entities:
        assert len(entity.checkpoints) == 2
        expected_final_quantity = (
            instance.num_jobs * expected_q * shares[entity.entity_id]
            * config["checkpoint"]["commitment_ratio"]
        )
        assert entity.checkpoints[-1].required_quantity == pytest.approx(
            expected_final_quantity
        )
        assert entity.checkpoints[0].required_quantity == pytest.approx(
            config["checkpoint"]["first_demand_ratio"] * expected_final_quantity
        )
        assert entity.checkpoints[-1].time == instance.metadata["entity_final_times"][entity.entity_id]
    assert instance.metadata["realization_used_to_build_commitment"] is False


def test_parameterized_exogenous_commitment_builder_has_exact_contract():
    entities = build_exogenous_service_entities(
        entity_count=2,
        expected_total_quantity_per_entity=40.0,
        final_checkpoint_time=100,
        first_time_ratio=0.6,
        first_demand_ratio=0.5,
        commitment_ratio=0.8,
        weights=(2.0, 5.0),
    )
    for entity in entities:
        assert tuple(checkpoint.time for checkpoint in entity.checkpoints) == (60, 100)
        assert tuple(
            checkpoint.required_quantity for checkpoint in entity.checkpoints
        ) == pytest.approx((16.0, 32.0))
        assert tuple(checkpoint.weight for checkpoint in entity.checkpoints) == (2.0, 5.0)


def test_scenario_seed_partitions_are_pairwise_disjoint():
    partitions = all_seed_partitions()
    names = tuple(partitions)
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            assert set(partitions[left_name]).isdisjoint(partitions[right_name])


def test_existing_generator_remains_reproducible():
    config = InstanceConfig(
        num_instances=1,
        num_jobs=8,
        num_machines=3,
        num_entities=2,
        ops_per_job=(2, 3),
        checkpoints_per_entity=(2, 2),
        checkpoint_tightness=1.0,
    )
    left = generate_instance(config, seed=99)
    right = generate_instance(config, seed=99)
    left_payload = asdict(left)
    right_payload = asdict(right)
    left_payload.pop("forecast_provider")
    right_payload.pop("forecast_provider")
    assert left_payload == right_payload

    left_provider = left.forecast_provider
    right_provider = right.forecast_provider
    assert type(left_provider) is type(right_provider)

    def comparable_state(provider):
        if provider is None:
            return None
        if is_dataclass(provider):
            return asdict(provider)
        if hasattr(provider, "__dict__"):
            return {
                name: value
                for name, value in vars(provider).items()
                if not callable(value)
            }
        return {
            name: getattr(provider, name)
            for name in getattr(type(provider), "__slots__", ())
        }

    assert comparable_state(left_provider) == comparable_state(right_provider)
