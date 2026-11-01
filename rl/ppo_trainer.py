"""论文 Sec 5.3: PPO 训练。"""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.distributions import Categorical

from fjsp_sfr.core.models import FJSPInstance
from fjsp_sfr.core.objective import compute_objective
from fjsp_sfr.dispatch.classic_rules import ClassicRule
from fjsp_sfr.dispatch.rule_library import RuleLibrary
from fjsp_sfr.dispatch.symbolic_rule import SymbolicRule
from fjsp_sfr.dispatch.theory_features import (
    THEORY_FEATURE_NAMES,
    build_extended_theory_features,
    feature_names_for,
)
from fjsp_sfr.env.ca_observation import resolve_observation_scales
from fjsp_sfr.env.ca_rule_selection import CAPPORuleSelectionEnv
from fjsp_sfr.rl.graph import (
    HeteroGraphData,
    apply_information_masks,
    build_hetero_graph,
)
from fjsp_sfr.rl.ppo_agent import HGTPPOAgent
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
        slack_index = THEORY_FEATURE_NAMES.index("due_date_slack")
        scores = [-row[slack_index] for row in factor_rows]

    best_index = 0
    best_score = float("-inf")
    for index, score in enumerate(scores):
        if score > best_score:
            best_score = score
            best_index = index
    return best_index


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
        mask_checkpoint: bool = False,
        mask_forecast: bool = False,
        reward_reference_mode: str = "legacy_per_instance",
    ) -> None:
        feature_names_for(feature_set)
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
        self.mask_checkpoint = bool(mask_checkpoint)
        self.mask_forecast = bool(mask_forecast)
        self.optimizer = torch.optim.Adam(agent.parameters(), lr=lr)

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
            env = CAPPORuleSelectionEnv(instance)
            env.reset()
            episode_reward = 0.0

            while True:
                graph, features = self._build_graph_for_env(env)
                mask = self._get_action_mask(env)
                action, log_prob, value = self.agent.select_action(graph, mask)
                rule = self.library.get_rule(action)
                candidate_index = _rule_candidate_index(
                    rule,
                    features,
                    problem=env._problem_view(),
                    state=env._state,
                    metrics=env.current_metrics,
                )
                result = env.step_candidate(candidate_index)
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
                    )
                )
                episode_reward += reward
                if result.terminated:
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
        return_tensor = torch.tensor(
            returns, dtype=torch.float32, device=self.device
        )
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
        num_updates = 0

        for _epoch in range(self.epochs_per_update):
            new_log_prob_rows: list[torch.Tensor] = []
            value_rows: list[torch.Tensor] = []
            entropy_rows: list[torch.Tensor] = []
            for index, step in enumerate(buffer.steps):
                log_probs, value = self.agent(step.graph, step.action_mask)
                new_log_prob_rows.append(log_probs[step.action])
                value_rows.append(value)
                entropy_rows.append(Categorical(logits=log_probs).entropy())

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
            value_loss = (values - return_tensor).pow(2).mean()
            loss = (
                policy_loss
                + self.value_coef * value_loss
                - self.entropy_coef * mean_entropy
            )

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
            self.optimizer.step()

            clip_fraction = (
                (ratios - 1.0).abs() > self.clip_eps
            ).float().mean()
            approx_kl = ((ratios - 1.0) - log_ratio).mean()
            total_policy_loss += float(policy_loss.detach().item())
            total_value_loss += float(value_loss.detach().item())
            total_entropy += float(mean_entropy.detach().item())
            total_kl += float(approx_kl.detach().item())
            total_clip_fraction += float(clip_fraction.detach().item())
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
        )

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
