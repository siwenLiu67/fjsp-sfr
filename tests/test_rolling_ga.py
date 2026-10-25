import pytest

from conftest import make_test_instance
from fjsp_sfr.baselines.rolling_ga import Chromosome, RollingGAConfig, decode_schedule, run_rolling_ga


def test_decoder_respects_precedence_release_and_machine_eligibility():
    instance = make_test_instance()
    keys = [(job.job_id, operation.sequence_index) for job in instance.jobs for operation in job.operations]
    chromosome = Chromosome(keys, {key: instance.get_job(key[0]).operations[key[1]].eligible_machines[0] for key in keys})
    decoded = decode_schedule(instance, chromosome)
    machine_end = {}
    job_end = {}
    for item in decoded:
        operation = instance.get_job(item.job_id).operations[item.sequence_index]
        assert item.machine_id in operation.eligible_machines
        assert item.start_time >= instance.get_job(item.job_id).release_time
        assert item.start_time >= machine_end.get(item.machine_id, 0)
        assert item.start_time >= job_end.get(item.job_id, 0)
        machine_end[item.machine_id] = item.end_time
        job_end[item.job_id] = item.end_time


def test_deterministic_generation_limit_is_reproducible():
    instance = make_test_instance()
    config = RollingGAConfig(population_size=8, max_generations=2, time_limit=999)
    left = run_rolling_ga(instance, config=config, seed=7)
    right = run_rolling_ga(instance, config=config, seed=7)
    assert (left.F, left.TT, left.WSF, left.mean_generations) == pytest.approx((right.F, right.TT, right.WSF, right.mean_generations))


def test_final_cost_is_environment_replay_cost():
    result = run_rolling_ga(make_test_instance(), config=RollingGAConfig(population_size=6, max_generations=1), seed=3)
    assert result.F == pytest.approx(result.TT + result.WSF)
