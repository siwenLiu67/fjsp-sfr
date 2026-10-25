"""Rolling-horizon genetic algorithm for dynamic FJSP-SFR."""

from __future__ import annotations

import math
import random
import time
from copy import deepcopy
from dataclasses import dataclass
from statistics import fmean

from fjsp_sfr.core.objective import compute_objective
from fjsp_sfr.dispatch.classic_rules import ClassicRuleKind, build_classic_rule_library
from fjsp_sfr.env.ca_rule_selection import CAPPORuleSelectionEnv


OperationKey = tuple[int, int]


@dataclass(frozen=True)
class RollingGAConfig:
    population_size: int = 50
    crossover_probability: float = 0.8
    mutation_probability: float = 0.2
    time_limit: float = 3.0
    tournament_size: int = 3
    elite: int = 1
    max_generations: int | None = None


@dataclass
class Chromosome:
    order: list[OperationKey]
    machines: dict[OperationKey, int]

    def clone(self) -> "Chromosome":
        return Chromosome(list(self.order), dict(self.machines))


@dataclass(frozen=True)
class DecodedOperation:
    job_id: int
    sequence_index: int
    machine_id: int
    start_time: float
    end_time: float


@dataclass(frozen=True)
class RollingGAResult:
    F: float
    TT: float
    WSF: float
    reschedule_count: int
    mean_reschedule_seconds: float
    mean_generations: float
    total_solve_seconds: float
    decision_seconds: float


def _remaining_keys(instance, arrived_job_ids: set[int], next_indices: dict[int, int]) -> list[OperationKey]:
    return [
        (job.job_id, operation.sequence_index)
        for job in instance.jobs if job.job_id in arrived_job_ids
        for operation in job.operations[next_indices.get(job.job_id, 0) :]
    ]


def decode_schedule(
    instance,
    chromosome: Chromosome,
    *,
    now: float = 0.0,
    machine_available: dict[int, float] | None = None,
    predecessor_completion: dict[int, float] | None = None,
) -> list[DecodedOperation]:
    """Serial schedule-generation decoder satisfying release/precedence/machine constraints."""

    available = {machine.machine_id: float(now) for machine in instance.machines}
    available.update(machine_available or {})
    completion = dict(predecessor_completion or {})
    decoded: list[DecodedOperation] = []
    pending = list(chromosome.order)
    while pending:
        progress = False
        for key in list(pending):
            job_id, sequence_index = key
            if sequence_index > 0 and (job_id, sequence_index - 1) not in {(item.job_id, item.sequence_index) for item in decoded} and job_id not in completion:
                continue
            job = instance.get_job(job_id); operation = job.operations[sequence_index]
            machine_id = chromosome.machines[key]
            if machine_id not in operation.eligible_machines:
                machine_id = min(operation.alternatives, key=lambda item: item.processing_time).machine_id
            predecessor = completion.get(job_id, float(job.release_time))
            start = max(float(now), float(job.release_time), predecessor, available[machine_id])
            end = start + operation.processing_time_on(machine_id)
            decoded.append(DecodedOperation(job_id, sequence_index, machine_id, start, end))
            completion[job_id] = end; available[machine_id] = end; pending.remove(key); progress = True
        if not progress:
            # A malformed order cannot deadlock decoding: schedule the earliest
            # sequence index still pending for its job.
            pending.sort(key=lambda item: (item[1], item[0]))
            key = pending.pop(0); pending.insert(0, key)
            completion.pop(key[0], None)
    return decoded


def schedule_objective(instance, decoded: list[DecodedOperation], arrived_job_ids: set[int]) -> float:
    completion = {job_id: max(item.end_time for item in decoded if item.job_id == job_id) for job_id in arrived_job_ids if any(item.job_id == job_id for item in decoded)}
    tardiness = sum(max(0.0, completion.get(job.job_id, job.release_time) - job.due_date) for job in instance.jobs if job.job_id in arrived_job_ids)
    shortfall = 0.0
    for entity in instance.entities:
        for checkpoint in entity.checkpoints:
            delivered = sum(job.quantity for job in instance.jobs if job.job_id in arrived_job_ids and job.entity_id == entity.entity_id and completion.get(job.job_id, math.inf) <= checkpoint.time)
            shortfall += checkpoint.weight * max(0.0, checkpoint.required_quantity - delivered)
    return instance.alpha * tardiness + instance.beta * shortfall


def _classic_order(instance, keys: list[OperationKey], kind: ClassicRuleKind, now: float) -> list[OperationKey]:
    def value(key):
        job = instance.get_job(key[0]); operation = job.operations[key[1]]
        remaining = sum(item.min_processing_time for item in job.operations[key[1] :])
        if kind == ClassicRuleKind.SPT: return operation.min_processing_time
        if kind == ClassicRuleKind.EDD: return job.due_date
        if kind == ClassicRuleKind.MIN_SLACK: return job.due_date - now - remaining
        allowance = max(0.0, job.due_date - now - remaining)
        return -math.exp(-allowance / max(1.0, 2.0 * remaining)) / max(remaining, 1)
    return sorted(keys, key=lambda key: (value(key), key))


def _random_machines(instance, keys: list[OperationKey], rng: random.Random) -> dict[OperationKey, int]:
    return {key: rng.choice(instance.get_job(key[0]).operations[key[1]].eligible_machines) for key in keys}


def initial_population(instance, keys: list[OperationKey], config: RollingGAConfig, rng: random.Random, now: float) -> list[Chromosome]:
    population = [Chromosome(_classic_order(instance, keys, kind, now), _random_machines(instance, keys, rng)) for kind in ClassicRuleKind]
    while len(population) < config.population_size:
        order = list(keys); rng.shuffle(order)
        population.append(Chromosome(order, _random_machines(instance, keys, rng)))
    return population[: config.population_size]


def _order_crossover(left: list[OperationKey], right: list[OperationKey], rng: random.Random) -> tuple[list[OperationKey], list[OperationKey]]:
    if len(left) < 2: return list(left), list(right)
    start, stop = sorted(rng.sample(range(len(left)), 2)); stop += 1
    def child(primary, secondary):
        segment = primary[start:stop]; remainder = [key for key in secondary if key not in segment]
        return remainder[:start] + segment + remainder[start:]
    return child(left, right), child(right, left)


def _mutate(instance, chromosome: Chromosome, probability: float, rng: random.Random) -> None:
    if len(chromosome.order) >= 2 and rng.random() < probability:
        left, right = rng.sample(range(len(chromosome.order)), 2)
        if rng.random() < 0.5:
            chromosome.order[left], chromosome.order[right] = chromosome.order[right], chromosome.order[left]
        else:
            chromosome.order.insert(right, chromosome.order.pop(left))
    if chromosome.order and rng.random() < probability:
        key = rng.choice(chromosome.order)
        chromosome.machines[key] = rng.choice(instance.get_job(key[0]).operations[key[1]].eligible_machines)


def search(instance, keys: list[OperationKey], arrived: set[int], now: float, config: RollingGAConfig, rng: random.Random) -> tuple[Chromosome, int, float]:
    population = initial_population(instance, keys, config, rng, now)
    started, generations = time.perf_counter(), 0
    def fitness(item): return schedule_objective(instance, decode_schedule(instance, item, now=now), arrived)
    while True:
        if config.max_generations is not None:
            if generations >= config.max_generations: break
        elif time.perf_counter() - started >= config.time_limit:
            break
        scored = sorted(((fitness(item), item) for item in population), key=lambda pair: pair[0])
        next_population = [item.clone() for _score, item in scored[: config.elite]]
        def tournament():
            sample = rng.sample(scored, min(config.tournament_size, len(scored)))
            return min(sample, key=lambda pair: pair[0])[1]
        while len(next_population) < config.population_size:
            left, right = tournament(), tournament(); child_left, child_right = left.clone(), right.clone()
            if rng.random() < config.crossover_probability:
                child_left.order, child_right.order = _order_crossover(left.order, right.order, rng)
                for key in keys:
                    if rng.random() < 0.5: child_left.machines[key] = right.machines[key]
                    if rng.random() < 0.5: child_right.machines[key] = left.machines[key]
            _mutate(instance, child_left, config.mutation_probability, rng); _mutate(instance, child_right, config.mutation_probability, rng)
            next_population.extend((child_left, child_right))
        population = next_population[: config.population_size]; generations += 1
    best = min(population, key=fitness)
    return best, generations, time.perf_counter() - started


class _PlannedOrderRule:
    name = "ROLLING_GA_PLAN"
    rule_type = None
    expression = "rolling GA operation order"
    def __init__(self, ranks: dict[OperationKey, int], machines: dict[OperationKey, int]):
        self.ranks, self.machines = ranks, machines
    def evaluate_candidates(self, _problem, _state, metrics):
        candidates = tuple(sorted(metrics.candidates, key=lambda item: item.job_id))
        keys = [(item.job_id, _state.next_op_index_for_job(item.job_id)) for item in candidates]
        has_assigned = any(self.machines.get(key) == metrics.machine_id for key in keys)
        return [
            -float(self.ranks.get(key, 10**9))
            if not has_assigned or self.machines.get(key) == metrics.machine_id
            else -float(10**12 + self.ranks.get(key, 10**9))
            for key in keys
        ]


def run_rolling_ga(instance, *, config: RollingGAConfig = RollingGAConfig(), seed: int = 1) -> RollingGAResult:
    rollout, rng = deepcopy(instance), random.Random(seed)
    env = CAPPORuleSelectionEnv(rollout); env.reset()
    next_indices = {job.job_id: 0 for job in rollout.jobs}
    seen_arrivals: set[int] = set(); ranks: dict[OperationKey, int] = {}; planned_machines: dict[OperationKey, int] = {}
    durations, generation_counts = [], []
    total_started = time.perf_counter(); decision_seconds = 0.0
    while env.current_metrics and env.current_metrics.is_active:
        now = env.current_metrics.decision_time
        arrived = {job.job_id for job in rollout.jobs if job.release_time <= now}
        if arrived != seen_arrivals or not ranks:
            next_indices = {job.job_id: env._state.next_op_index_for_job(job.job_id) for job in rollout.jobs}
            keys = _remaining_keys(rollout, arrived, next_indices)
            chromosome, generations, duration = search(rollout, keys, arrived, now, config, rng)
            ranks = {key: index for index, key in enumerate(chromosome.order)}
            planned_machines = dict(chromosome.machines)
            durations.append(duration); generation_counts.append(generations); seen_arrivals = arrived
        rule = _PlannedOrderRule(ranks, planned_machines); started = time.perf_counter()
        result = env.step_principle(0, rule); decision_seconds += time.perf_counter() - started
        if result.terminated: break
    objective = compute_objective(rollout, env._state)
    return RollingGAResult(float(objective.Z), float(objective.total_tardiness), float(objective.weighted_checkpoint_shortfall), len(durations), fmean(durations) if durations else 0.0, fmean(generation_counts) if generation_counts else 0.0, time.perf_counter() - total_started, decision_seconds)


def time_probe(instances: list, *, config: RollingGAConfig, seed: int = 1) -> dict[str, float]:
    per_generation = []
    for index, instance in enumerate(instances[:3]):
        arrived = {job.job_id for job in instance.jobs if job.release_time <= 0}
        keys = _remaining_keys(instance, arrived, {job.job_id: 0 for job in instance.jobs})
        probe = RollingGAConfig(**{**config.__dict__, "max_generations": 1})
        _best, generations, duration = search(instance, keys, arrived, 0.0, probe, random.Random(seed + index))
        per_generation.append(duration / max(1, generations))
    mean = fmean(per_generation)
    return {"mean_seconds_per_generation": mean, "generations_in_three_seconds": 3.0 / max(mean, 1e-12)}


__all__ = ["Chromosome", "DecodedOperation", "RollingGAConfig", "RollingGAResult", "decode_schedule", "run_rolling_ga", "schedule_objective", "time_probe"]
