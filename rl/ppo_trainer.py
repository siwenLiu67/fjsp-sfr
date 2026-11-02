"""论文 Sec 5.3: PPO 训练。"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.distributions import Categorical

from fjsp_sfr.core.models import FJSPInstance
from fjsp_sfr.core.objective import compute_objective
from fjsp_sfr.dispatch.certificates import PrivilegedCheckpointCertificate
from fjsp_sfr.dispatch.classic_rules import ClassicRule
from fjsp_sfr.dispatch.rule_library import RuleLibrary
from fjsp_sfr.dispatch.symbolic_rule import SymbolicRule
from fjsp_sfr.dispatch.theory_features import build_extended_theory_features, feature_names_for
from fjsp_sfr.env.ca_observation import resolve_observation_scales
from fjsp_sfr.env.ca_rule_selection import CAPPORuleSelectionEnv
from fjsp_sfr.rl.graph import (
    GRAPH_CONFIGS,
    LEGACY_GRAPH,
    PAPER_GRAPH,
    HeteroGraphData,
    apply_information_masks,
    build_hetero_graph,
)
from fjsp_sfr.rl.ppo_agent import HGTPPOAgent
from fjsp_sfr.rl.action_groups import (
    ActionGroups,
    ConcreteDispatch,
    aggregate_rule_log_probs,
    build_action_groups,
    validate_action_groups,
)
from fjsp_sfr.rl.certificate_critic import (
    CRITIC_VARIANTS,
    PRIVILEGED_FEATURE_NAMES,
    CriticStepContext,
    CriticTargets,
)
from fjsp_sfr.rl.cc_dvl import (
    DISTRIBUTIONAL_CRITIC_VARIANTS,
    QuantileCriticOutput,
    summarize_width_dispersion,
    width_dispersion_rows,
)
from fjsp_sfr.rl.rollout_buffer import RolloutBuffer, RolloutStep


@dataclass
class TrainStats:
    """一次 PPO update 的聚合统计。"""

    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    advantage_std: float
    clip_fraction: float
    total_reward: float
    gradient_norm: float
    critic_gradient_norm: float
    critic_total_cost_mae: float
    critic_tardiness_mae: float
    critic_shortfall_mae: float
    environment_interactions: int
    optimizer_updates: int


PAPER_CRITIC_VARIANTS = (*CRITIC_VARIANTS, *DISTRIBUTIONAL_CRITIC_VARIANTS)


def standardize_advantages(
    advantages: list[float],
    *,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, float]:
    """在整个 rollout batch 上标准化 advantage。"""

    tensor = torch.tensor(advantages, dtype=torch.float32, device=device)
    if tensor.numel() == 0:
        raise ValueError("advantages must not be empty")
    raw_std = float(tensor.std(unbiased=False).item())
    if tensor.numel() == 1 or raw_std <= 1e-8:
        return tensor - tensor.mean(), raw_std
    return (tensor - tensor.mean()) / (raw_std + 1e-8), raw_std


def _resolve_f_refs(
    f_ref: float | Sequence[float] | Mapping[int, float],
    instance_count: int,
) -> tuple[float, ...]:
    """将标量或按实例给定的基线归一化为 tuple。"""

    if isinstance(f_ref, (int, float)):
        values = (float(f_ref),) * instance_count
    elif isinstance(f_ref, Mapping):
        expected_keys = set(range(instance_count))
        if set(f_ref) != expected_keys:
            raise ValueError(
                "f_ref mapping keys must exactly match instance indices"
            )
        values = tuple(float(f_ref[index]) for index in range(instance_count))
    else:
        values = tuple(float(value) for value in f_ref)
        if len(values) != instance_count:
            raise ValueError("f_ref sequence length must match instances")
    if any(not torch.isfinite(torch.tensor(value)) or value <= 0.0 for value in values):
        raise ValueError("all f_ref values must be finite and positive")
    return values


def _move_graph(data: HeteroGraphData, device: torch.device) -> HeteroGraphData:
    return HeteroGraphData(
        job_x=data.job_x.to(device),
        machine_x=data.machine_x.to(device),
        checkpoint_x=data.checkpoint_x.to(device),
        job_machine_edge_index=data.job_machine_edge_index.to(device),
        job_machine_edge_attr=data.job_machine_edge_attr.to(device),
        job_checkpoint_edge_index=data.job_checkpoint_edge_index.to(device),
        job_checkpoint_edge_attr=data.job_checkpoint_edge_attr.to(device),
        global_x=data.global_x.to(device),
        num_jobs=data.num_jobs,
        num_machines=data.num_machines,
        num_checkpoints=data.num_checkpoints,
        job_ids=data.job_ids,
        checkpoint_keys=data.checkpoint_keys,
        graph_config=data.graph_config,
        relation_edge_index={
            name: value.to(device) for name, value in data.relation_edge_index.items()
        },
        relation_edge_attr={
            name: value.to(device) for name, value in data.relation_edge_attr.items()
        },
        job_batch=None if data.job_batch is None else data.job_batch.to(device),
        machine_batch=(
            None if data.machine_batch is None else data.machine_batch.to(device)
        ),
        checkpoint_batch=(
            None
            if data.checkpoint_batch is None
            else data.checkpoint_batch.to(device)
        ),
        num_graphs=data.num_graphs,
    )


def _summary_row(
    factor_rows: tuple[tuple[float, ...], ...],
) -> tuple[float, ...]:
    return tuple(max(column) for column in zip(*factor_rows, strict=True))


def _rule_candidate_index(
    rule: object,
    factor_rows: tuple[tuple[float, ...], ...],
    *,
    problem=None,
    state=None,
    metrics=None,
) -> int:
    if not factor_rows:
        raise ValueError("cannot dispatch without candidates")
    if isinstance(rule, ClassicRule):
        if problem is None or state is None or metrics is None:
            raise ValueError("classic rules require problem, state, and metrics")
        scores = rule.evaluate_candidates(problem, state, metrics)
    elif rule.is_applicable(_summary_row(factor_rows)):
        scores = rule.evaluate_batch(factor_rows)
    else:
        candidates = tuple(
            sorted(metrics.candidates, key=lambda candidate: candidate.job_id)
        )
        scores = [
            -(
                float(problem.get_job(candidate.job_id).due_date)
                - float(candidate.optimistic_completion)
            )
            for candidate in candidates
        ]

    if scores and not any(math.isfinite(float(score)) for score in scores):
        candidates = tuple(
            sorted(metrics.candidates, key=lambda candidate: candidate.job_id)
        )
        scores = [
            -(
                float(problem.get_job(candidate.job_id).due_date)
                - float(candidate.optimistic_completion)
            )
            for candidate in candidates
        ]

    best_index = 0
    best_score = float("-inf")
    for index, score in enumerate(scores):
        if score > best_score:
            best_score = score
            best_index = index
    return best_index


def concrete_rule_action_groups(
    env: CAPPORuleSelectionEnv,
    library: RuleLibrary,
    factor_rows: tuple[tuple[float, ...], ...],
    action_mask: torch.Tensor,
    *,
    merge_equivalent: bool,
) -> tuple[ActionGroups, tuple[int, ...], tuple[ConcreteDispatch, ...]]:
    """Evaluate deterministic rules once and group their concrete dispatches."""

    candidates = tuple(
        sorted(env.current_metrics.candidates, key=lambda candidate: candidate.job_id)
    )
    candidate_indices: list[int] = []
    dispatches: list[ConcreteDispatch] = []
    per_job_dispatch: dict[int, ConcreteDispatch] = {}
    for rule_index, rule in enumerate(library.rules):
        if not bool(action_mask[rule_index]):
            candidate_indices.append(-1)
            dispatches.append((-1, -1, -1))
            continue
        candidate_index = _rule_candidate_index(
            rule,
            factor_rows,
            problem=env._problem_view(),
            state=env._state,
            metrics=env.current_metrics,
        )
        candidate = candidates[candidate_index]
        dispatch = (
            int(candidate.job_id),
            int(candidate.operation_id),
            int(candidate.machine_id),
        )
        previous = per_job_dispatch.setdefault(candidate.job_id, dispatch)
        if previous != dispatch:
            raise RuntimeError(
                "the same job mapped to different concrete operations or machines"
            )
        candidate_indices.append(candidate_index)
        dispatches.append(dispatch)
    groups = build_action_groups(
        dispatches,
        action_mask,
        merge_equivalent=merge_equivalent,
    )
    validate_action_groups(groups, action_mask)
    for group in groups:
        keys = {dispatches[index] for index in group}
        if len(keys) != 1:
            raise RuntimeError("an action group contains different concrete dispatches")
    return groups, tuple(candidate_indices), tuple(dispatches)


def build_critic_step_context(
    env: CAPPORuleSelectionEnv,
    graph: HeteroGraphData,
    critic_variant: str,
) -> CriticStepContext | None:
    """Build C2 public or C3/C4 private fields in graph checkpoint order."""

    if critic_variant == "c1" or critic_variant == "legacy":
        return None
    if critic_variant not in {"c2", "c3", "c4", "d1", "d2"}:
        raise ValueError(f"unknown critic variant: {critic_variant}")
    device = graph.checkpoint_x.device
    keys = graph.checkpoint_keys
    checkpoints = [env.current_metrics.checkpoints[key] for key in keys]
    weights = torch.tensor(
        [checkpoint.checkpoint_weight for checkpoint in checkpoints],
        dtype=torch.float32,
        device=device,
    )
    quantities = torch.tensor(
        [checkpoint.required_quantity for checkpoint in checkpoints],
        dtype=torch.float32,
        device=device,
    )
    online = (
        graph.checkpoint_x[:, 10:12]
        if graph.checkpoint_x.shape[0]
        else graph.checkpoint_x.new_empty((0, 2))
    )
    if critic_variant == "c2":
        context = CriticStepContext(keys, weights, quantities, online)
        context.validate(privileged=False)
        return context

    private_by_key: dict[tuple[int, int], PrivilegedCheckpointCertificate] = {
        certificate.checkpoint_key: certificate
        for certificate in env.critic_certificate_record.certificates
    }
    missing = [key for key in keys if key not in private_by_key]
    if missing:
        raise RuntimeError(f"privileged certificate record lacks active keys: {missing}")
    lower = torch.tensor(
        [private_by_key[key].historical.lower for key in keys],
        dtype=torch.float32,
        device=device,
    )
    upper = torch.tensor(
        [private_by_key[key].historical.upper for key in keys],
        dtype=torch.float32,
        device=device,
    )
    private_rows = []
    for index, key in enumerate(keys):
        certificate = private_by_key[key]
        quantity = max(float(checkpoints[index].required_quantity), 1e-12)
        count = max(0.0, float(certificate.future_job_count))
        private_rows.append(
            (
                float(certificate.historical.lower) / quantity,
                float(certificate.historical.upper) / quantity,
                float(certificate.future_supply_bound) / quantity,
                count / (1.0 + count),
            )
        )
    private = (
        torch.tensor(private_rows, dtype=torch.float32, device=device)
        if private_rows
        else graph.checkpoint_x.new_empty((0, len(PRIVILEGED_FEATURE_NAMES)))
    )
    context = CriticStepContext(
        keys,
        weights,
        quantities,
        online,
        private,
        lower,
        upper,
    )
    context.validate(privileged=True)
    return context


def finalize_episode_critic_targets(
    steps: Sequence[RolloutStep],
    objective,
) -> None:
    """Attach exact undiscounted remaining-cost labels to one completed episode.

    Checkpoint labels are the final physical shortfalls, aligned only to the
    checkpoints that were active in that decision's public graph.  Therefore a
    settled checkpoint cannot be charged again at later decisions.
    """

    remaining_total = float(objective.Z)
    remaining_tardiness = float(objective.total_tardiness)
    for step in steps:
        context = step.critic_context
        if context is None:
            labels = torch.empty(0, dtype=torch.float32)
        else:
            labels = torch.tensor(
                [
                    float(objective.per_checkpoint_shortfall[key])
                    for key in context.checkpoint_keys
                ],
                dtype=torch.float32,
                device=context.weights.device,
            )
        step.critic_targets = CriticTargets(
            remaining_total_cost=remaining_total,
            remaining_tardiness=remaining_tardiness,
            checkpoint_shortfalls=labels,
        )
        remaining_total -= float(step.immediate_total_cost)
        remaining_tardiness -= float(step.immediate_tardiness)
        if remaining_total < -1e-7 or remaining_tardiness < -1e-7:
            raise RuntimeError("incremental costs exceed the final objective")
        remaining_total = max(0.0, remaining_total)
        remaining_tardiness = max(0.0, remaining_tardiness)

    if abs(remaining_total) > 1e-6 or abs(remaining_tardiness) > 1e-6:
        raise RuntimeError(
            "decision increments do not telescope to the final objective labels"
        )


class PPOTrainer:
    def __init__(
        self,
        agent: HGTPPOAgent,
        library: RuleLibrary,
        instances: list[FJSPInstance] | None = None,
        *,
        lr: float = 3e-4,
        gamma: float = 1.0,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        epochs_per_update: int = 4,
        max_grad_norm: float = 0.5,
        f_ref: float | Sequence[float] | Mapping[int, float] = 1.0,
        reward_scale: float = 1.0,
        instance_factory: Callable[[int], FJSPInstance] | None = None,
        base_seed: int = 0,
        entropy_coef_start: float | None = None,
        entropy_coef_end: float | None = None,
        feature_set: str = "theory11",
        graph_config: str = LEGACY_GRAPH,
        mask_checkpoint: bool = False,
        mask_forecast: bool = False,
        reward_reference_mode: str = "legacy_per_instance",
        action_equivalence_merge: bool = False,
    ) -> None:
        feature_names_for(feature_set)
        if graph_config not in GRAPH_CONFIGS:
            raise ValueError(f"unknown graph_config: {graph_config}")
        resolved_instances = list(instances or [])
        if instance_factory is None and not resolved_instances:
            raise ValueError("instances must not be empty without instance_factory")
        if library.action_size() != agent.action_size:
            raise ValueError("library size must match the agent action size")
        if epochs_per_update <= 0:
            raise ValueError("epochs_per_update must be positive")
        if not torch.isfinite(torch.tensor(reward_scale)) or reward_scale <= 0.0:
            raise ValueError("reward_scale must be finite and positive")
        if entropy_coef < 0.0:
            raise ValueError("entropy_coef must be non-negative")
        if (entropy_coef_start is None) != (entropy_coef_end is None):
            raise ValueError(
                "entropy_coef_start and entropy_coef_end must be provided together"
            )
        if entropy_coef_start is not None and (
            entropy_coef_start < 0.0 or entropy_coef_end < 0.0
        ):
            raise ValueError("scheduled entropy coefficients must be non-negative")
        if reward_reference_mode not in {
            "legacy_per_instance",
            "global_calibration",
        }:
            raise ValueError("unknown reward_reference_mode")
        if reward_reference_mode == "global_calibration":
            if not isinstance(f_ref, (int, float)):
                raise ValueError("global_calibration requires one scalar f_ref")
            if not torch.isfinite(torch.tensor(float(f_ref))) or float(f_ref) < 1.0:
                raise ValueError(
                    "global calibrated f_ref must be finite and at least one"
                )
        if graph_config == PAPER_GRAPH:
            if feature_set != "paper14":
                raise ValueError("paper_graph PPO requires feature_set='paper14'")
            if getattr(agent, "encoder_type", None) != "crg_gatv2":
                raise ValueError("paper_graph PPO requires the CRG actor encoder")
            if gamma != 1.0 or gae_lambda != 0.95:
                raise ValueError("formal PPO requires gamma=1 and GAE lambda=0.95")
            if reward_reference_mode != "global_calibration":
                raise ValueError("formal scalar PPO requires one global calibrated F_ref")
            if reward_scale != 10.0:
                raise ValueError("formal scalar PPO requires reward_scale=10")
            if getattr(agent, "critic_variant", "legacy") in PAPER_CRITIC_VARIANTS:
                if reward_reference_mode != "global_calibration":
                    raise ValueError("paper critics require the global calibrated F_ref")

        self.agent = agent
        self.library = library
        self.instances = resolved_instances
        self.instance_factory = instance_factory
        self.base_seed = int(base_seed)
        self.global_episode_count = 0
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef_start = entropy_coef_start
        self.entropy_coef_end = entropy_coef_end
        self.entropy_coef = (
            float(entropy_coef_start)
            if entropy_coef_start is not None
            else float(entropy_coef)
        )
        self.epochs_per_update = epochs_per_update
        self.max_grad_norm = max_grad_norm
        self.f_ref = f_ref
        self.reward_reference_mode = reward_reference_mode
        self.global_f_ref = (
            float(f_ref) if reward_reference_mode == "global_calibration" else None
        )
        self.f_refs = (
            _resolve_f_refs(f_ref, len(resolved_instances))
            if instance_factory is None
            and reward_reference_mode == "legacy_per_instance"
            else ()
        )
        self.reward_scale = float(reward_scale)
        self.feature_set = feature_set
        self.graph_config = graph_config
        self.mask_checkpoint = bool(mask_checkpoint)
        self.mask_forecast = bool(mask_forecast)
        self.action_equivalence_merge = bool(action_equivalence_merge)
        self.optimizer = torch.optim.Adam(agent.parameters(), lr=lr)
        self.environment_interactions = 0
        self.optimizer_update_count = 0
        self._quantile_width_rows: list[tuple[float, float, float]] = []

    def quantile_width_diagnostics(self) -> dict[str, dict[str, float | int]]:
        """D2 uncertainty diagnostic; never feeds back into optimization."""

        return summarize_width_dispersion(self._quantile_width_rows)

    @property
    def device(self) -> torch.device:
        return next(self.agent.parameters()).device

    def entropy_coef_at(self, update_index: int, total_updates: int) -> float:
        """返回给定 update 的线性衰减熵系数。"""

        if total_updates <= 0:
            raise ValueError("total_updates must be positive")
        if not 0 <= update_index < total_updates:
            raise ValueError("update_index must lie within total_updates")
        if self.entropy_coef_start is None:
            return self.entropy_coef
        if total_updates == 1:
            return float(self.entropy_coef_end)
        fraction = update_index / (total_updates - 1)
        return float(
            self.entropy_coef_start
            + fraction * (self.entropy_coef_end - self.entropy_coef_start)
        )

    def _compute_instance_f_ref(self, instance: FJSPInstance) -> float:
        """用所有固定规则的完整仿真均值计算实例基线。"""

        objectives: list[float] = []
        for rule_index, rule in enumerate(self.library.rules):
            rollout_instance = deepcopy(instance)
            env = CAPPORuleSelectionEnv(rollout_instance)
            env.reset()
            while env.current_metrics and env.current_metrics.is_active:
                result = env.step_principle(rule_index, rule)
                if result.terminated:
                    break
            objectives.append(
                float(compute_objective(rollout_instance, env._state).Z)
            )
        return max(sum(objectives) / len(objectives), 1.0)

    def _build_graph_for_env(
        self, env: CAPPORuleSelectionEnv
    ) -> tuple[HeteroGraphData, tuple[tuple[float, ...], ...]]:
        """从环境当前状态构建图和与 candidate 对齐的特征。"""

        problem = env._problem_view()
        state = env._state
        metrics = env.current_metrics
        time_scale, processing_scale = resolve_observation_scales(problem, state)
        features = build_extended_theory_features(
            problem,
            state,
            metrics,
            time_scale=time_scale,
            feature_set=self.feature_set,
        )
        graph = build_hetero_graph(
            problem,
            state,
            metrics,
            features,
            time_scale=time_scale,
            processing_scale=processing_scale,
            feature_set=self.feature_set,
            graph_config=self.graph_config,
        )
        graph = apply_information_masks(
            graph,
            feature_set=self.feature_set,
            mask_checkpoint=self.mask_checkpoint,
            mask_forecast=self.mask_forecast,
        )
        return _move_graph(graph, self.device), features

    def _get_action_mask(self, env: CAPPORuleSelectionEnv) -> torch.Tensor:
        del env
        return torch.ones(
            self.library.action_size(),
            dtype=torch.bool,
            device=self.device,
        )

    def collect_rollout(
        self, num_episodes: int
    ) -> tuple[RolloutBuffer, float]:
        """随机选实例收集指定数量的完整 episode。"""

        if num_episodes <= 0:
            raise ValueError("num_episodes must be positive")
        buffer = RolloutBuffer()
        episode_rewards: list[float] = []

        for _episode in range(num_episodes):
            if self.instance_factory is not None:
                episode_seed = self.base_seed + self.global_episode_count
                generated = self.instance_factory(episode_seed)
                if not isinstance(generated, FJSPInstance):
                    raise TypeError("instance_factory must return FJSPInstance")
                self.global_episode_count += 1
                instance = deepcopy(generated)
                episode_f_ref = (
                    self.global_f_ref
                    if self.reward_reference_mode == "global_calibration"
                    else self._compute_instance_f_ref(generated)
                )
            else:
                instance_index = random.randrange(len(self.instances))
                instance = deepcopy(self.instances[instance_index])
                episode_f_ref = (
                    self.global_f_ref
                    if self.reward_reference_mode == "global_calibration"
                    else self.f_refs[instance_index]
                )
            assert episode_f_ref is not None
            critic_variant = getattr(self.agent, "critic_variant", "legacy")
            if critic_variant in {"c2", "c3", "c4", "d1", "d2"} and (
                float(instance.alpha) != 1.0 or float(instance.beta) != 1.0
            ):
                raise ValueError("decomposed critics require alpha=beta=1")
            env = CAPPORuleSelectionEnv(
                instance,
                enable_privileged_certificates=critic_variant
                in {"c3", "c4", "d1", "d2"},
            )
            env.reset()
            episode_reward = 0.0
            episode_start = len(buffer.steps)

            while True:
                graph, features = self._build_graph_for_env(env)
                mask = self._get_action_mask(env)
                critic_context = build_critic_step_context(
                    env, graph, critic_variant
                )
                with torch.no_grad():
                    if critic_variant in PAPER_CRITIC_VARIANTS:
                        rule_log_probs, critic_output, _encoding = (
                            self.agent.forward_with_critic(
                                graph,
                                mask,
                                critic_context,
                                f_ref=episode_f_ref,
                                reward_scale=self.reward_scale,
                            )
                        )
                        value_tensor = critic_output.total_value
                        if critic_variant == "d2":
                            if not isinstance(critic_output, QuantileCriticOutput):
                                raise TypeError("D2 must return quantile output")
                            assert critic_context is not None
                            assert critic_context.privileged_lower is not None
                            assert critic_context.privileged_upper is not None
                            self._quantile_width_rows.extend(
                                width_dispersion_rows(
                                    critic_output.checkpoint_quantiles,
                                    critic_context.privileged_lower,
                                    critic_context.privileged_upper,
                                )
                            )
                    else:
                        rule_log_probs, value_tensor = self.agent(graph, mask)
                if self.action_equivalence_merge:
                    groups, candidate_indices, dispatches = concrete_rule_action_groups(
                        env,
                        self.library,
                        features,
                        mask,
                        merge_equivalent=True,
                    )
                    group_log_probs = aggregate_rule_log_probs(
                        rule_log_probs, groups, mask
                    )
                    distribution = Categorical(logits=group_log_probs)
                    group_tensor = distribution.sample()
                    group_index = int(group_tensor.item())
                    action = groups[group_index][0]
                    log_prob = float(distribution.log_prob(group_tensor).item())
                    value = float(value_tensor.item())
                    candidate_index = candidate_indices[action]
                    concrete_dispatch = dispatches[action]
                else:
                    groups = ()
                    group_index = None
                    concrete_dispatch = None
                    distribution = Categorical(logits=rule_log_probs)
                    action_tensor = distribution.sample()
                    action = int(action_tensor.item())
                    log_prob = float(distribution.log_prob(action_tensor).item())
                    value = float(value_tensor.item())
                    rule = self.library.get_rule(action)
                    candidate_index = _rule_candidate_index(
                        rule,
                        features,
                        problem=env._problem_view(),
                        state=env._state,
                        metrics=env.current_metrics,
                    )
                result = env.step_candidate(candidate_index)
                if concrete_dispatch is not None:
                    actual = (
                        int(result.decision.job_id),
                        int(result.decision.operation_id),
                        int(result.decision.machine_id),
                    )
                    if actual != concrete_dispatch:
                        raise RuntimeError(
                            "stored action group does not match the actual dispatch"
                        )
                # result.reward is exactly -(F_cum(t_next) - F_cum(t_now)).
                reward = (
                    float(result.reward)
                    / episode_f_ref
                    * self.reward_scale
                )
                buffer.add(
                    RolloutStep(
                        graph=graph,
                        action=action,
                        log_prob=log_prob,
                        value=value,
                        reward=reward,
                        done=result.terminated,
                        action_mask=mask.clone(),
                        action_groups=groups,
                        action_group_index=group_index,
                        concrete_dispatch=concrete_dispatch,
                        critic_context=critic_context,
                        immediate_total_cost=-float(result.reward),
                        immediate_tardiness=float(
                            result.reward_breakdown.delta_tardiness
                        ),
                        f_ref=float(episode_f_ref),
                    )
                )
                episode_reward += reward
                self.environment_interactions += 1
                if result.terminated:
                    if result.objective is None:
                        raise RuntimeError("terminated episode lacks final objective")
                    if critic_variant in PAPER_CRITIC_VARIANTS:
                        finalize_episode_critic_targets(
                            buffer.steps[episode_start:], result.objective
                        )
                    break

            episode_rewards.append(episode_reward)

        return buffer, float(sum(episode_rewards) / len(episode_rewards))

    def update(self, buffer: RolloutBuffer) -> TrainStats:
        """使用按轨迹计算的 GAE 执行 PPO clipped-objective 更新。"""

        if not buffer.steps:
            raise ValueError("cannot update from an empty rollout buffer")
        returns, advantages = buffer.compute_returns_and_advantages(
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )
        advantage_tensor, advantage_std = standardize_advantages(
            advantages, device=self.device
        )
        return_tensor = torch.tensor(returns, dtype=torch.float32, device=self.device)
        old_log_probs = torch.tensor(
            [step.log_prob for step in buffer.steps],
            dtype=torch.float32,
            device=self.device,
        )

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_kl = 0.0
        total_clip_fraction = 0.0
        total_gradient_norm = 0.0
        total_critic_gradient_norm = 0.0
        total_critic_cost_mae = 0.0
        total_critic_tt_mae = 0.0
        total_critic_shortfall_mae = 0.0
        num_updates = 0

        for _epoch in range(self.epochs_per_update):
            new_log_prob_rows: list[torch.Tensor] = []
            value_rows: list[torch.Tensor] = []
            critic_loss_rows: list[torch.Tensor] = []
            critic_cost_error_rows: list[torch.Tensor] = []
            critic_tt_error_rows: list[torch.Tensor] = []
            critic_shortfall_error_rows: list[torch.Tensor] = []
            entropy_rows: list[torch.Tensor] = []
            for index, step in enumerate(buffer.steps):
                critic_variant = getattr(self.agent, "critic_variant", "legacy")
                if critic_variant in PAPER_CRITIC_VARIANTS:
                    log_probs, critic_output, _encoding = self.agent.forward_with_critic(
                        step.graph,
                        step.action_mask,
                        step.critic_context,
                        f_ref=step.f_ref,
                        reward_scale=self.reward_scale,
                    )
                    value = critic_output.total_value
                    if step.critic_targets is None:
                        raise ValueError("paper critic rollout step lacks MC targets")
                    assert self.agent.certificate_critic is not None
                    critic_loss_rows.append(
                        self.agent.certificate_critic.loss(
                            critic_output,
                            step.critic_targets,
                            step.critic_context,
                            f_ref=step.f_ref,
                        )
                    )
                    target_total = critic_output.total_cost.new_tensor(
                        step.critic_targets.remaining_total_cost
                    )
                    critic_cost_error_rows.append(
                        (critic_output.total_cost - target_total).abs()
                    )
                    if critic_output.remaining_tardiness is not None:
                        target_tt = critic_output.total_cost.new_tensor(
                            step.critic_targets.remaining_tardiness
                        )
                        critic_tt_error_rows.append(
                            (critic_output.remaining_tardiness - target_tt).abs()
                        )
                    expected_shortfalls = step.critic_targets.checkpoint_shortfalls.to(
                        critic_output.total_cost.device
                    )
                    if isinstance(critic_output, QuantileCriticOutput):
                        predicted_shortfalls = (
                            critic_output.mean_checkpoint_shortfalls
                        )
                    else:
                        predicted_shortfalls = critic_output.checkpoint_shortfalls
                    if expected_shortfalls.numel():
                        critic_shortfall_error_rows.append(
                            (predicted_shortfalls - expected_shortfalls).abs().mean()
                        )
                else:
                    log_probs, value = self.agent(step.graph, step.action_mask)
                if step.action_groups:
                    validate_action_groups(step.action_groups, step.action_mask)
                    if step.action_group_index is None:
                        raise ValueError("grouped rollout step lacks group index")
                    group_log_probs = aggregate_rule_log_probs(
                        log_probs, step.action_groups, step.action_mask
                    )
                    new_log_prob_rows.append(
                        group_log_probs[step.action_group_index]
                    )
                    entropy_rows.append(
                        Categorical(logits=group_log_probs).entropy()
                    )
                else:
                    new_log_prob_rows.append(log_probs[step.action])
                    entropy_rows.append(Categorical(logits=log_probs).entropy())
                value_rows.append(value)

            new_log_probs = torch.stack(new_log_prob_rows)
            values = torch.stack(value_rows)
            mean_entropy = torch.stack(entropy_rows).mean()
            log_ratio = new_log_probs - old_log_probs
            ratios = torch.exp(log_ratio)
            surrogate = ratios * advantage_tensor
            clipped_surrogate = torch.clamp(
                ratios,
                1.0 - self.clip_eps,
                1.0 + self.clip_eps,
            ) * advantage_tensor
            policy_loss = -torch.minimum(surrogate, clipped_surrogate).mean()
            if critic_loss_rows:
                value_loss = torch.stack(critic_loss_rows).mean()
            else:
                value_loss = (values - return_tensor).pow(2).mean()
            loss = (
                policy_loss
                + self.value_coef * value_loss
                - self.entropy_coef * mean_entropy
            )

            self.optimizer.zero_grad()
            loss.backward()
            critic_parameters = (
                tuple(self.agent.certificate_critic.parameters())
                if getattr(self.agent, "certificate_critic", None) is not None
                else ()
            )
            critic_gradient_norm = math.sqrt(
                sum(
                    float(parameter.grad.detach().pow(2).sum().item())
                    for parameter in critic_parameters
                    if parameter.grad is not None
                )
            )
            gradient_norm = nn.utils.clip_grad_norm_(
                self.agent.parameters(), self.max_grad_norm
            )
            self.optimizer.step()
            self.optimizer_update_count += 1

            clip_fraction = (
                (ratios - 1.0).abs() > self.clip_eps
            ).float().mean()
            approx_kl = ((ratios - 1.0) - log_ratio).mean()
            total_policy_loss += float(policy_loss.detach().item())
            total_value_loss += float(value_loss.detach().item())
            total_entropy += float(mean_entropy.detach().item())
            total_kl += float(approx_kl.detach().item())
            total_clip_fraction += float(clip_fraction.detach().item())
            total_gradient_norm += float(gradient_norm.detach().item())
            total_critic_gradient_norm += float(critic_gradient_norm)
            if critic_cost_error_rows:
                total_critic_cost_mae += float(
                    torch.stack(critic_cost_error_rows).mean().detach().item()
                )
            if critic_tt_error_rows:
                total_critic_tt_mae += float(
                    torch.stack(critic_tt_error_rows).mean().detach().item()
                )
            if critic_shortfall_error_rows:
                total_critic_shortfall_mae += float(
                    torch.stack(critic_shortfall_error_rows).mean().detach().item()
                )
            num_updates += 1

        denominator = max(num_updates, 1)
        mean_step_reward = sum(step.reward for step in buffer.steps) / len(buffer)
        return TrainStats(
            policy_loss=total_policy_loss / denominator,
            value_loss=total_value_loss / denominator,
            entropy=total_entropy / denominator,
            approx_kl=total_kl / denominator,
            advantage_std=advantage_std,
            clip_fraction=total_clip_fraction / denominator,
            total_reward=float(mean_step_reward),
            gradient_norm=total_gradient_norm / denominator,
            critic_gradient_norm=total_critic_gradient_norm / denominator,
            critic_total_cost_mae=total_critic_cost_mae / denominator,
            critic_tardiness_mae=total_critic_tt_mae / denominator,
            critic_shortfall_mae=total_critic_shortfall_mae / denominator,
            environment_interactions=len(buffer.steps),
            optimizer_updates=num_updates,
        )

    @torch.no_grad()
    def recompute_stored_log_probs(
        self, buffer: RolloutBuffer
    ) -> tuple[float, ...]:
        """Re-evaluate immutable rollout actions under the current policy."""

        values: list[float] = []
        for step in buffer.steps:
            if hasattr(self.agent, "actor_log_probs"):
                rule_log_probs = self.agent.actor_log_probs(
                    step.graph, step.action_mask
                )
            else:
                rule_log_probs, _value = self.agent(
                    step.graph, step.action_mask
                )
            if step.action_groups:
                if step.action_group_index is None:
                    raise ValueError("grouped rollout step lacks group index")
                grouped = aggregate_rule_log_probs(
                    rule_log_probs, step.action_groups, step.action_mask
                )
                values.append(float(grouped[step.action_group_index].item()))
            else:
                values.append(float(rule_log_probs[step.action].item()))
        return tuple(values)

    def train(
        self,
        total_updates: int,
        episodes_per_update: int = 4,
        log_interval: int = 10,
        log_fn=None,
    ) -> list[TrainStats]:
        """收集轨迹并迭代更新 PPO。"""

        all_stats: list[TrainStats] = []
        for update_index in range(total_updates):
            self.entropy_coef = self.entropy_coef_at(
                update_index, total_updates
            )
            buffer, mean_episode_reward = self.collect_rollout(
                episodes_per_update
            )
            if not buffer.steps:
                continue
            stats = self.update(buffer)
            stats.total_reward = mean_episode_reward
            all_stats.append(stats)
            if (
                log_fn is not None
                and log_interval > 0
                and (update_index + 1) % log_interval == 0
            ):
                log_fn(update_index + 1, stats)
        return all_stats
