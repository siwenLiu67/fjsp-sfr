"""Machine-triggered event-driven CA-PPO environment."""

from __future__ import annotations

import random
from copy import deepcopy
from dataclasses import dataclass

from fjsp_sfr.core.events import Event, EventType
from fjsp_sfr.core.forecast import empty_forecast_snapshot
from fjsp_sfr.core.models import FJSPInstance, Operation
from fjsp_sfr.core.objective import ObjectiveResult, compute_objective
from fjsp_sfr.core.online import OnlineProblemView
from fjsp_sfr.core.state import ScheduleState
from fjsp_sfr.dispatch.ca_metrics import CADecisionMetrics, build_ca_decision_metrics
from fjsp_sfr.dispatch.ca_rules import rule_applicability_mask, select_job
from fjsp_sfr.dispatch.types import DirectCandidateDecision, RuleAction, RuleDecision

from .ca_observation import build_ca_observation, build_theory_factor_values
from .ca_action_tokens import ActionToken, build_rule_action_tokens
from .ca_reward import CARewardBreakdown, ca_transition_reward
from .ca_rule_tokens import RuleToken, build_rule_tokens


@dataclass(frozen=True)
class CADecisionObservation:
    decision_index: int
    physical_time: int
    trigger_machine_id: int
    forecast_update_epoch: int
    shop: tuple[float, float, float, float, float, float, float]
    checkpoint_tokens: tuple[
        tuple[float, float, float, float, float, float, float, float], ...
    ]
    checkpoint_keys: tuple[tuple[int, int], ...]
    candidate_tokens: tuple[
        tuple[float, float, float, float, float, float], ...
    ]
    candidate_job_ids: tuple[int, ...]
    relational_candidate_tokens: tuple[tuple[float, float, float, float], ...]
    relational_checkpoint_tokens: tuple[
        tuple[float, float, float, float, float], ...
    ]
    relational_checkpoint_keys: tuple[tuple[int, int], ...]
    candidate_checkpoint_relations: tuple[
        tuple[tuple[float, float], ...], ...
    ]
    candidate_checkpoint_mask: tuple[tuple[bool, ...], ...]
    theory_factor_values: tuple[tuple[float, float, float, float, float], ...]
    action_mask: tuple[bool, ...]
    action_tokens: tuple[ActionToken, ...]
    rule_tokens: tuple[RuleToken, ...]
    action_candidate_keys: tuple[tuple[int, int], ...]

    @property
    def features(self):
        return self.shop


@dataclass(frozen=True)
class CAStepResult:
    observation: CADecisionObservation | None
    reward: float
    terminated: bool
    decision: RuleDecision | DirectCandidateDecision
    reward_breakdown: CARewardBreakdown
    objective: ObjectiveResult | None


@dataclass(frozen=True)
class CAEnvironmentSnapshot:
    """Complete branchable simulator state with a fixed future realization."""

    instance: FJSPInstance
    state: ScheduleState
    visible_job_ids: frozenset[int]
    decision_index: int
    current_metrics: CADecisionMetrics | None
    terminated: bool
    has_reset: bool
    time_scale: float | None
    processing_scale: float | None
    coverage_cap: float
    forecast_update_epoch: int
    environment_random_sources: dict[str, random.Random]
    instance_random_sources: dict[str, random.Random]
    forecast_random_sources: dict[str, random.Random]


def _copy_random_sources(owner: object | None) -> dict[str, random.Random]:
    if owner is None or not hasattr(owner, "__dict__"):
        return {}
    return {
        name: deepcopy(value)
        for name, value in vars(owner).items()
        if isinstance(value, random.Random)
    }


def _restore_random_sources(
    owner: object | None, sources: dict[str, random.Random]
) -> None:
    if owner is None:
        return
    for name, source in sources.items():
        setattr(owner, name, deepcopy(source))


class CAPPORuleSelectionEnv:
    """One rule choice per idle-machine trigger, ordered by machine id."""

    def __init__(
        self,
        instance: FJSPInstance,
        *,
        time_scale: float | None = None,
        processing_scale: float | None = None,
        coverage_cap: float = 2.0,
    ) -> None:
        self.instance = instance
        self.time_scale = time_scale
        self.processing_scale = processing_scale
        self.coverage_cap = coverage_cap
        self._state = ScheduleState()
        self._visible_job_ids: set[int] = set()
        self._decision_index = 0
        self._current_metrics: CADecisionMetrics | None = None
        self._terminated = False
        self._has_reset = False

    @property
    def current_metrics(self) -> CADecisionMetrics:
        if self._current_metrics is None:
            raise RuntimeError("the environment has no active decision")
        return self._current_metrics

    @property
    def state_snapshot(self) -> ScheduleState:
        if not self._has_reset:
            raise RuntimeError("reset must be called before reading state")
        return self._state.copy_for_algorithm()

    def snapshot(self) -> CAEnvironmentSnapshot:
        """Capture an isolated counterfactual branch point.

        Synthetic-generation RNGs are normally exhausted before an environment
        is created, so the generated jobs and forecast snapshots already encode
        the complete future realization.  Any live ``random.Random`` sources
        attached to the environment, instance, or forecast provider are copied
        explicitly as well.
        """

        if not self._has_reset:
            raise RuntimeError("reset must be called before taking a snapshot")
        instance = deepcopy(self.instance)
        provider = self.instance.forecast_provider
        return CAEnvironmentSnapshot(
            instance=instance,
            state=deepcopy(self._state),
            visible_job_ids=frozenset(self._visible_job_ids),
            decision_index=self._decision_index,
            current_metrics=deepcopy(self._current_metrics),
            terminated=self._terminated,
            has_reset=self._has_reset,
            time_scale=self.time_scale,
            processing_scale=self.processing_scale,
            coverage_cap=self.coverage_cap,
            forecast_update_epoch=self._forecast_snapshot().update_epoch,
            environment_random_sources=_copy_random_sources(self),
            instance_random_sources=_copy_random_sources(self.instance),
            forecast_random_sources=_copy_random_sources(provider),
        )

    @classmethod
    def from_snapshot(
        cls, snapshot: CAEnvironmentSnapshot
    ) -> "CAPPORuleSelectionEnv":
        instance = deepcopy(snapshot.instance)
        env = cls(
            instance,
            time_scale=snapshot.time_scale,
            processing_scale=snapshot.processing_scale,
            coverage_cap=snapshot.coverage_cap,
        )
        env._state = deepcopy(snapshot.state)
        env._visible_job_ids = set(snapshot.visible_job_ids)
        env._decision_index = snapshot.decision_index
        env._current_metrics = deepcopy(snapshot.current_metrics)
        env._terminated = snapshot.terminated
        env._has_reset = snapshot.has_reset
        _restore_random_sources(env, snapshot.environment_random_sources)
        _restore_random_sources(env.instance, snapshot.instance_random_sources)
        _restore_random_sources(
            env.instance.forecast_provider, snapshot.forecast_random_sources
        )
        if env._forecast_snapshot().update_epoch != snapshot.forecast_update_epoch:
            raise RuntimeError("snapshot forecast epoch was not restored exactly")
        return env

    def clone(self) -> "CAPPORuleSelectionEnv":
        return type(self).from_snapshot(self.snapshot())

    def reset(self) -> CADecisionObservation:
        self._state = ScheduleState(
            machine_available_times={machine.machine_id: 0 for machine in self.instance.machines}
        )
        for job in self.instance.jobs:
            self._state.event_queue.push(Event(job.release_time, EventType.JOB_ARRIVAL, job.job_id))
        for entity in self.instance.entities:
            for checkpoint in entity.checkpoints:
                self._state.event_queue.push(Event(checkpoint.time, EventType.CHECKPOINT))
        self._visible_job_ids = set()
        self._decision_index = 0
        self._current_metrics = None
        self._terminated = False
        self._has_reset = True
        observation = self._advance_to_decision()
        if observation is None:
            raise RuntimeError("instance terminated without an active decision")
        return observation

    def step(self, action: RuleAction | int) -> CAStepResult:
        if not self._has_reset:
            raise RuntimeError("reset must be called before step")
        if self._terminated:
            raise RuntimeError("cannot step a terminated environment")
        problem = self._problem_view()
        selected_action = RuleAction(action)
        action_mask = rule_applicability_mask(problem, self.current_metrics)
        if not action_mask[int(selected_action)]:
            raise ValueError(
                f"R{selected_action.paper_number} primary condition is not applicable"
            )
        decision = select_job(problem, self.current_metrics, selected_action)
        return self._apply_decision(decision)

    def step_candidate(self, candidate_index: int) -> CAStepResult:
        """Apply a current candidate-index action without dispatch-rule mediation."""

        candidates = tuple(
            sorted(self.current_metrics.candidates, key=lambda item: item.job_id)
        )
        if not 0 <= candidate_index < len(candidates):
            raise ValueError("candidate index is out of range for the current decision")
        candidate = candidates[candidate_index]
        assert self.current_metrics.machine_id is not None
        return self.step_explicit_decision(
            DirectCandidateDecision(
                candidate_index=candidate_index,
                job_id=candidate.job_id,
                operation_id=candidate.operation_id,
                machine_id=self.current_metrics.machine_id,
                decision_time=self.current_metrics.decision_time,
                score_components=(("candidate_index", candidate_index),),
            )
        )

    def step_principle(self, principle_index: int, principle) -> CAStepResult:
        """Execute one symbolic principle and dispatch its best candidate.

        Candidate rows are sorted by job id throughout observation construction.
        ``max`` therefore resolves equal scores to the lowest job id, matching
        the deterministic tie-breaking convention used by existing rules.
        """

        if not self._has_reset:
            raise RuntimeError("reset must be called before step")
        if self._terminated:
            raise RuntimeError("cannot step a terminated environment")
        if principle_index < 0:
            raise ValueError("principle index must be non-negative")
        problem = self._problem_view()
        candidates = tuple(
            sorted(self.current_metrics.candidates, key=lambda item: item.job_id)
        )
        if hasattr(principle, "evaluate_candidates"):
            factor_rows = ()
            scores = principle.evaluate_candidates(
                problem, self._state, self.current_metrics
            )
        elif hasattr(principle, "rule_type"):
            from fjsp_sfr.dispatch.theory_features import (
                THEORY_FEATURE_NAMES,
                build_extended_theory_features,
            )

            factor_rows = build_extended_theory_features(
                problem,
                self._state,
                self.current_metrics,
                time_scale=self.time_scale,
                feature_set=getattr(principle, "feature_set", "theory11"),
            )
        else:
            factor_rows = build_theory_factor_values(
                problem,
                self._state,
                self.current_metrics,
                time_scale=self.time_scale,
            )
        if not hasattr(principle, "evaluate_candidates"):
            if len(factor_rows) != len(candidates) or not candidates:
                raise RuntimeError("principle factors must align with current candidates")
            if hasattr(principle, "is_applicable") and not principle.is_applicable(
                tuple(max(column) for column in zip(*factor_rows, strict=True))
            ):
                slack_index = THEORY_FEATURE_NAMES.index("due_date_slack")
                scores = [-row[slack_index] for row in factor_rows]
            else:
                scores = principle.evaluate_batch(factor_rows)
        if len(scores) != len(candidates):
            raise ValueError("principle must return one score per candidate")
        candidate_index = max(range(len(scores)), key=scores.__getitem__)
        candidate = candidates[candidate_index]
        assert self.current_metrics.machine_id is not None
        return self.step_explicit_decision(
            DirectCandidateDecision(
                candidate_index=candidate_index,
                job_id=candidate.job_id,
                operation_id=candidate.operation_id,
                machine_id=self.current_metrics.machine_id,
                decision_time=self.current_metrics.decision_time,
                score_components=(
                    ("principle_index", principle_index),
                    ("principle_name", principle.name),
                    ("principle_expression", principle.expression),
                    ("principle_score", scores[candidate_index]),
                ),
            )
        )

    def step_explicit_decision(
        self, decision: RuleDecision | DirectCandidateDecision
    ) -> CAStepResult:
        """Apply a validated concrete decision without changing scheduling semantics."""

        if not self._has_reset:
            raise RuntimeError("reset must be called before step")
        if self._terminated:
            raise RuntimeError("cannot step a terminated environment")
        if not any(
            candidate.job_id == decision.job_id
            and candidate.operation_id == decision.operation_id
            and candidate.machine_id == decision.machine_id
            for candidate in self.current_metrics.candidates
        ):
            raise ValueError("explicit decision must select a current candidate")
        return self._apply_decision(decision)

    def _apply_decision(
        self, decision: RuleDecision | DirectCandidateDecision
    ) -> CAStepResult:
        previous = self._state.copy_for_algorithm()
        self._start(decision)
        self._decision_index += 1
        observation = self._advance_to_decision()
        following = self._state.copy_for_algorithm()
        breakdown = ca_transition_reward(self.instance, previous, following)
        objective = compute_objective(self.instance, self._state) if self._terminated else None
        return CAStepResult(
            observation=observation,
            reward=breakdown.total,
            terminated=self._terminated,
            decision=decision,
            reward_breakdown=breakdown,
            objective=objective,
        )

    def _forecast_snapshot(self):
        if self.instance.forecast_provider is None:
            return empty_forecast_snapshot(
                (entity.entity_id for entity in self.instance.entities), update_epoch=0
            )
        return self.instance.forecast_provider.snapshot_at(self._state.current_time)

    def _problem_view(self) -> OnlineProblemView:
        return OnlineProblemView.from_instance(
            self.instance,
            visible_job_ids=self._visible_job_ids,
            forecast_snapshot=self._forecast_snapshot(),
            current_time=self._state.current_time,
        )

    def _advance_to_decision(self) -> CADecisionObservation | None:
        while True:
            problem = self._problem_view()
            metrics = build_ca_decision_metrics(problem, self._state)
            if metrics.is_active:
                assert metrics.machine_id is not None
                raw = build_ca_observation(
                    problem,
                    self._state,
                    metrics=metrics,
                    time_scale=self.time_scale,
                    processing_scale=self.processing_scale,
                    coverage_cap=self.coverage_cap,
                )
                action_tokens, action_candidate_keys = build_rule_action_tokens(
                    problem,
                    self._state,
                    metrics,
                    time_scale=self.time_scale,
                    processing_scale=self.processing_scale,
                )
                rule_tokens, rule_candidate_keys = build_rule_tokens(
                    problem,
                    self._state,
                    metrics,
                    time_scale=self.time_scale,
                    processing_scale=self.processing_scale,
                )
                if rule_candidate_keys != action_candidate_keys:
                    raise RuntimeError("rule token candidates must match rule decisions")
                self._current_metrics = metrics
                return CADecisionObservation(
                    decision_index=self._decision_index,
                    physical_time=self._state.current_time,
                    trigger_machine_id=metrics.machine_id,
                    forecast_update_epoch=metrics.forecast_update_epoch,
                    shop=raw.shop,
                    checkpoint_tokens=raw.checkpoint_tokens,
                    checkpoint_keys=raw.checkpoint_keys,
                    candidate_tokens=raw.candidate_tokens,
                    candidate_job_ids=raw.candidate_job_ids,
                    relational_candidate_tokens=raw.relational_candidate_tokens,
                    relational_checkpoint_tokens=raw.relational_checkpoint_tokens,
                    relational_checkpoint_keys=raw.relational_checkpoint_keys,
                    candidate_checkpoint_relations=raw.candidate_checkpoint_relations,
                    candidate_checkpoint_mask=raw.candidate_checkpoint_mask,
                    theory_factor_values=raw.theory_factor_values,
                    action_mask=rule_applicability_mask(problem, metrics),
                    action_tokens=action_tokens,
                    rule_tokens=rule_tokens,
                    action_candidate_keys=action_candidate_keys,
                )
            if not self._state.event_queue:
                self._current_metrics = None
                if len(self._state.completed_jobs) != self.instance.num_jobs:
                    raise RuntimeError("event queue exhausted before all jobs completed")
                self._terminated = True
                return None
            next_time = self._state.event_queue.peek().time
            self._state.current_time = next_time
            while self._state.event_queue and self._state.event_queue.peek().time == next_time:
                self._process_event(self._state.event_queue.pop())

    def _process_event(self, event: Event) -> None:
        if event.event_type == EventType.CHECKPOINT:
            return
        if event.event_type == EventType.JOB_ARRIVAL:
            self._visible_job_ids.add(event.job_id)
            return
        if event.event_type == EventType.OP_COMPLETION:
            job = self.instance.get_job(event.job_id)
            operation = self._find_operation(job.job_id, event.op_id)
            self._state.mark_operation_completed(
                job.job_id, operation.op_id, operation.sequence_index
            )
            if operation.sequence_index == job.num_operations - 1:
                self._state.mark_job_completed(job.job_id, event.time)
            return
        raise ValueError(f"unsupported event type: {event.event_type}")

    def _start(self, decision: RuleDecision | DirectCandidateDecision) -> None:
        if decision.decision_time != self._state.current_time:
            raise ValueError("decision time does not match environment time")
        if decision.machine_id != self.current_metrics.machine_id:
            raise ValueError("decision must use the current trigger machine")
        operation = self._find_operation(decision.job_id, decision.operation_id)
        if not self._state.is_machine_idle(decision.machine_id):
            raise ValueError("decision targets a busy machine")
        if operation.sequence_index != self._state.next_op_index_for_job(decision.job_id):
            raise ValueError("decision does not target the next operation")
        end_time = self._state.current_time + operation.processing_time_on(decision.machine_id)
        self._state.mark_operation_started(
            decision.job_id,
            decision.operation_id,
            decision.machine_id,
            self._state.current_time,
            end_time,
        )
        self._state.event_queue.push(
            Event(
                end_time,
                EventType.OP_COMPLETION,
                decision.job_id,
                decision.operation_id,
                decision.machine_id,
            )
        )

    def _find_operation(self, job_id: int, operation_id: int | None) -> Operation:
        job = self.instance.get_job(job_id)
        for operation in job.operations:
            if operation.op_id == operation_id:
                return operation
        raise KeyError(f"operation {operation_id} not found for job {job_id}")


# Descriptive alias used by counterfactual experiments.
OnlineRuleSelectionEnv = CAPPORuleSelectionEnv
