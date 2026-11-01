"""HGT-DQN selector with replay memory and a target network."""

from __future__ import annotations

import random
from collections import deque
from copy import deepcopy
from dataclasses import dataclass

import torch
import torch.nn as nn

from fjsp_sfr.core.objective import compute_objective
from fjsp_sfr.dispatch.rule_library import RuleLibrary
from fjsp_sfr.dispatch.theory_features import build_extended_theory_features, feature_names_for
from fjsp_sfr.env.ca_observation import resolve_observation_scales
from fjsp_sfr.env.ca_rule_selection import CAPPORuleSelectionEnv
from fjsp_sfr.rl.graph import HeteroGraphData, apply_information_masks, build_hetero_graph
from fjsp_sfr.rl.hgt import HGTEncoder


class HGTDQNAgent(nn.Module):
    def __init__(self, action_size: int, hidden_dim: int = 64, **encoder_kwargs) -> None:
        super().__init__()
        if action_size <= 0:
            raise ValueError("action_size must be positive")
        self.action_size = action_size
        self.encoder = HGTEncoder(hidden_dim=hidden_dim, **encoder_kwargs)
        output_dim = encoder_kwargs.get("output_dim") or hidden_dim
        self.q_head = nn.Linear(output_dim, action_size)

    def forward(self, data: HeteroGraphData) -> torch.Tensor:
        return self.q_head(self.encoder(data))

    @torch.no_grad()
    def greedy_action(self, data: HeteroGraphData) -> int:
        return int(self(data).argmax().item())


@dataclass(frozen=True)
class DQNTransition:
    graph: HeteroGraphData
    action: int
    reward: float
    next_graph: HeteroGraphData | None
    done: bool


@dataclass(frozen=True)
class DQNTrainStats:
    episode_rewards: list[float]
    losses: list[float]
    final_epsilon: float


class DQNTrainer:
    def __init__(
        self,
        agent: HGTDQNAgent,
        library: RuleLibrary,
        instance_factory,
        *,
        base_seed: int = 1000,
        feature_set: str = "theory11_plus3",
        lr: float = 3e-4,
        gamma: float = 0.99,
        reward_scale: float = 10.0,
        replay_capacity: int = 20_000,
        batch_size: int = 64,
        target_update_interval: int = 200,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        mask_checkpoint: bool = False,
        mask_forecast: bool = False,
        fixed_f_ref: float | None = None,
    ) -> None:
        self.agent = agent
        self.target = deepcopy(agent)
        self.target.eval()
        self.library = library
        self.instance_factory = instance_factory
        self.base_seed = int(base_seed)
        self.feature_set = feature_set
        self.gamma = float(gamma)
        self.reward_scale = float(reward_scale)
        self.batch_size = int(batch_size)
        self.target_update_interval = int(target_update_interval)
        self.epsilon_start = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.mask_checkpoint = bool(mask_checkpoint)
        self.mask_forecast = bool(mask_forecast)
        if fixed_f_ref is not None and fixed_f_ref <= 0.0:
            raise ValueError("fixed_f_ref must be positive")
        self.fixed_f_ref = None if fixed_f_ref is None else float(fixed_f_ref)
        self.replay: deque[DQNTransition] = deque(maxlen=replay_capacity)
        self.optimizer = torch.optim.Adam(agent.parameters(), lr=lr)
        self.global_episode_count = 0
        self.gradient_steps = 0

    def _graph(self, env: CAPPORuleSelectionEnv) -> HeteroGraphData:
        problem, state, metrics = env._problem_view(), env._state, env.current_metrics
        time_scale, processing_scale = resolve_observation_scales(problem, state)
        rows = build_extended_theory_features(
            problem, state, metrics, time_scale=time_scale, feature_set=self.feature_set
        )
        graph = build_hetero_graph(
            problem,
            state,
            metrics,
            rows,
            time_scale=time_scale,
            processing_scale=processing_scale,
            feature_set=self.feature_set,
        )
        return apply_information_masks(
            graph,
            feature_set=self.feature_set,
            mask_checkpoint=self.mask_checkpoint,
            mask_forecast=self.mask_forecast,
        )

    def _f_ref(self, instance) -> float:
        if self.fixed_f_ref is not None:
            return self.fixed_f_ref
        values = []
        for index, rule in enumerate(self.library.rules):
            copy = deepcopy(instance)
            env = CAPPORuleSelectionEnv(copy)
            env.reset()
            while env.current_metrics and env.current_metrics.is_active:
                result = env.step_principle(index, rule)
                if result.terminated:
                    break
            values.append(float(compute_objective(copy, env._state).Z))
        return max(sum(values) / len(values), 1.0)

    def epsilon_at(self, episode: int, total_episodes: int) -> float:
        if total_episodes <= 1:
            return self.epsilon_end
        fraction = min(max(episode / (total_episodes - 1), 0.0), 1.0)
        return self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start)

    def _learn(self) -> float | None:
        if len(self.replay) < self.batch_size:
            return None
        batch = random.sample(list(self.replay), self.batch_size)
        predicted = torch.stack([self.agent(item.graph)[item.action] for item in batch])
        targets = []
        with torch.no_grad():
            for item in batch:
                future = 0.0 if item.done else float(self.target(item.next_graph).max().item())
                targets.append(item.reward + self.gamma * future)
        target_tensor = predicted.new_tensor(targets)
        loss = nn.functional.smooth_l1_loss(predicted, target_tensor)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.agent.parameters(), 0.5)
        self.optimizer.step()
        self.gradient_steps += 1
        if self.gradient_steps % self.target_update_interval == 0:
            self.target.load_state_dict(self.agent.state_dict())
        return float(loss.detach().item())

    def train(self, total_episodes: int) -> DQNTrainStats:
        rewards: list[float] = []
        losses: list[float] = []
        for local_episode in range(total_episodes):
            instance = self.instance_factory(self.base_seed + self.global_episode_count)
            self.global_episode_count += 1
            f_ref = self._f_ref(instance)
            env = CAPPORuleSelectionEnv(deepcopy(instance))
            env.reset()
            episode_reward = 0.0
            epsilon = self.epsilon_at(local_episode, total_episodes)
            while env.current_metrics and env.current_metrics.is_active:
                graph = self._graph(env)
                if random.random() < epsilon:
                    action = random.randrange(self.library.action_size())
                else:
                    action = self.agent.greedy_action(graph)
                result = env.step_principle(action, self.library.get_rule(action))
                reward = float(result.reward) / f_ref * self.reward_scale
                next_graph = None if result.terminated else self._graph(env)
                self.replay.append(DQNTransition(graph, action, reward, next_graph, result.terminated))
                learned = self._learn()
                if learned is not None:
                    losses.append(learned)
                episode_reward += reward
                if result.terminated:
                    break
            rewards.append(episode_reward)
        return DQNTrainStats(rewards, losses, self.epsilon_at(total_episodes - 1, total_episodes))


__all__ = ["DQNTrainer", "DQNTrainStats", "HGTDQNAgent"]
