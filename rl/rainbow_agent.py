"""Rainbow selector with a shared HGT encoder and C51 value distribution."""

from __future__ import annotations

import math
import random
from collections import deque
from copy import deepcopy
from dataclasses import dataclass

import torch
import torch.nn as nn

from fjsp_sfr.core.objective import compute_objective
from fjsp_sfr.dispatch.rule_library import RuleLibrary
from fjsp_sfr.dispatch.theory_features import build_extended_theory_features
from fjsp_sfr.env.ca_observation import resolve_observation_scales
from fjsp_sfr.env.ca_rule_selection import CAPPORuleSelectionEnv
from fjsp_sfr.rl.graph import apply_information_masks, build_hetero_graph
from fjsp_sfr.rl.hgt import HGTEncoder


class NoisyLinear(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, sigma: float = 0.5) -> None:
        super().__init__()
        self.input_dim, self.output_dim = input_dim, output_dim
        self.weight_mu = nn.Parameter(torch.empty(output_dim, input_dim))
        self.weight_sigma = nn.Parameter(torch.full((output_dim, input_dim), sigma / math.sqrt(input_dim)))
        self.register_buffer("weight_epsilon", torch.zeros(output_dim, input_dim))
        self.bias_mu = nn.Parameter(torch.empty(output_dim))
        self.bias_sigma = nn.Parameter(torch.full((output_dim,), sigma / math.sqrt(output_dim)))
        self.register_buffer("bias_epsilon", torch.zeros(output_dim))
        bound = 1.0 / math.sqrt(input_dim)
        nn.init.uniform_(self.weight_mu, -bound, bound); nn.init.uniform_(self.bias_mu, -bound, bound)
        self.reset_noise()

    @staticmethod
    def _noise(size: int, device) -> torch.Tensor:
        values = torch.randn(size, device=device)
        return values.sign() * values.abs().sqrt()

    def reset_noise(self) -> None:
        input_noise = self._noise(self.input_dim, self.weight_epsilon.device)
        output_noise = self._noise(self.output_dim, self.weight_epsilon.device)
        self.weight_epsilon.copy_(output_noise.outer(input_noise)); self.bias_epsilon.copy_(output_noise)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight, bias = self.weight_mu, self.bias_mu
        return nn.functional.linear(values, weight, bias)


class HGTRainbowAgent(nn.Module):
    def __init__(
        self,
        action_size: int,
        hidden_dim: int = 64,
        *,
        atoms: int = 51,
        v_min: float = -20.0,
        v_max: float = 0.0,
        **encoder_kwargs,
    ) -> None:
        super().__init__()
        if atoms < 2 or v_max <= v_min:
            raise ValueError("invalid C51 support")
        self.action_size, self.atoms = int(action_size), int(atoms)
        self.v_min, self.v_max = float(v_min), float(v_max)
        self.encoder = HGTEncoder(hidden_dim=hidden_dim, **encoder_kwargs)
        output_dim = encoder_kwargs.get("output_dim") or hidden_dim
        self.value_hidden = NoisyLinear(output_dim, hidden_dim)
        self.value_out = NoisyLinear(hidden_dim, atoms)
        self.advantage_hidden = NoisyLinear(output_dim, hidden_dim)
        self.advantage_out = NoisyLinear(hidden_dim, action_size * atoms)
        self.register_buffer("support", torch.linspace(v_min, v_max, atoms))

    def logits(self, graph) -> torch.Tensor:
        encoded = self.encoder(graph)
        value = self.value_out(torch.relu(self.value_hidden(encoded))).view(1, self.atoms)
        advantage = self.advantage_out(torch.relu(self.advantage_hidden(encoded))).view(self.action_size, self.atoms)
        return value + advantage - advantage.mean(dim=0, keepdim=True)

    def distribution(self, graph) -> torch.Tensor:
        return torch.softmax(self.logits(graph), dim=-1).clamp_min(1e-7)

    def forward(self, graph) -> torch.Tensor:
        return (self.distribution(graph) * self.support).sum(dim=-1)

    @torch.no_grad()
    def greedy_action(self, graph) -> int:
        return int(self(graph).argmax().item())

    def reset_noise(self) -> None:
        for module in self.modules():
            if isinstance(module, NoisyLinear): module.reset_noise()


@dataclass(frozen=True)
class RainbowTransition:
    graph: object
    action: int
    reward: float
    next_graph: object | None
    done: bool
    steps: int


class PrioritizedReplay:
    def __init__(self, capacity: int = 20000, alpha: float = 0.6) -> None:
        self.capacity, self.alpha = int(capacity), float(alpha)
        self.data: list[RainbowTransition] = []; self.priorities: list[float] = []; self.position = 0

    def __len__(self): return len(self.data)

    def add(self, transition: RainbowTransition) -> None:
        priority = max(self.priorities, default=1.0)
        if len(self.data) < self.capacity:
            self.data.append(transition); self.priorities.append(priority)
        else:
            self.data[self.position] = transition; self.priorities[self.position] = priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, size: int, beta: float):
        weights = [priority ** self.alpha for priority in self.priorities]; total = sum(weights)
        probabilities = [weight / total for weight in weights]
        indices = random.choices(range(len(self.data)), weights=probabilities, k=size)
        importance = torch.tensor([(len(self.data) * probabilities[index]) ** (-beta) for index in indices], dtype=torch.float32)
        importance /= importance.max()
        return [self.data[index] for index in indices], indices, importance

    def update(self, indices: list[int], priorities: torch.Tensor) -> None:
        for index, priority in zip(indices, priorities.detach().cpu().tolist(), strict=True):
            self.priorities[index] = max(float(priority), 1e-6)


@dataclass(frozen=True)
class RainbowTrainStats:
    episode_rewards: list[float]
    losses: list[float]


class RainbowTrainer:
    def __init__(
        self,
        agent: HGTRainbowAgent,
        library: RuleLibrary,
        instance_factory,
        *,
        base_seed: int = 1000,
        lr: float = 3e-4,
        gamma: float = 0.99,
        reward_scale: float = 10.0,
        n_step: int = 3,
        batch_size: int = 64,
        replay_capacity: int = 20000,
        target_update_interval: int = 200,
        feature_set: str = "theory11_plus3",
        mask_checkpoint: bool = False,
        mask_forecast: bool = False,
        fixed_f_ref: float | None = None,
    ) -> None:
        self.agent, self.target = agent, deepcopy(agent); self.target.eval()
        self.library, self.instance_factory, self.base_seed = library, instance_factory, int(base_seed)
        self.gamma, self.reward_scale, self.n_step = float(gamma), float(reward_scale), int(n_step)
        self.batch_size, self.target_update_interval = int(batch_size), int(target_update_interval)
        self.feature_set, self.mask_checkpoint, self.mask_forecast = feature_set, mask_checkpoint, mask_forecast
        if fixed_f_ref is not None and fixed_f_ref <= 0.0:
            raise ValueError("fixed_f_ref must be positive")
        self.fixed_f_ref = None if fixed_f_ref is None else float(fixed_f_ref)
        self.replay = PrioritizedReplay(replay_capacity); self.optimizer = torch.optim.Adam(agent.parameters(), lr=lr)
        self.global_episode_count = 0; self.gradient_steps = 0

    def _graph(self, env):
        problem, state, metrics = env._problem_view(), env._state, env.current_metrics
        time_scale, processing_scale = resolve_observation_scales(problem, state)
        rows = build_extended_theory_features(problem, state, metrics, time_scale=time_scale, feature_set=self.feature_set)
        graph = build_hetero_graph(problem, state, metrics, rows, time_scale=time_scale, processing_scale=processing_scale, feature_set=self.feature_set)
        return apply_information_masks(graph, feature_set=self.feature_set, mask_checkpoint=self.mask_checkpoint, mask_forecast=self.mask_forecast)

    def _f_ref(self, instance) -> float:
        if self.fixed_f_ref is not None:
            return self.fixed_f_ref
        values = []
        for index, rule in enumerate(self.library.rules):
            copy = deepcopy(instance); env = CAPPORuleSelectionEnv(copy); env.reset()
            while env.current_metrics and env.current_metrics.is_active:
                result = env.step_principle(index, rule)
                if result.terminated: break
            values.append(float(compute_objective(copy, env._state).Z))
        return max(sum(values) / len(values), 1.0)

    def _flush_n_step(self, queue: deque, *, force: bool = False) -> None:
        while queue and (force or len(queue) >= self.n_step):
            horizon = min(self.n_step, len(queue)); reward = 0.0; final = queue[horizon - 1]
            for index in range(horizon):
                reward += (self.gamma ** index) * queue[index][2]
                if queue[index][4]: final = queue[index]; horizon = index + 1; break
            first = queue[0]
            self.replay.add(RainbowTransition(first[0], first[1], reward, None if final[4] else final[3], final[4], horizon))
            queue.popleft()
            if not force: break

    def _projection(self, transition: RainbowTransition) -> torch.Tensor:
        with torch.no_grad():
            if transition.done:
                target = torch.zeros(self.agent.atoms); value = min(self.agent.v_max, max(self.agent.v_min, transition.reward))
                position = (value - self.agent.v_min) / (self.agent.v_max - self.agent.v_min) * (self.agent.atoms - 1)
                lower, upper = int(math.floor(position)), int(math.ceil(position)); target[lower] += upper - position if upper != lower else 1.0
                if upper != lower: target[upper] += position - lower
                return target
            online_action = int(self.agent(transition.next_graph).argmax().item())
            probability = self.target.distribution(transition.next_graph)[online_action]
            transformed = transition.reward + (self.gamma ** transition.steps) * self.target.support
            transformed = transformed.clamp(self.agent.v_min, self.agent.v_max)
            positions = (transformed - self.agent.v_min) / (self.agent.v_max - self.agent.v_min) * (self.agent.atoms - 1)
            lower, upper = positions.floor().long(), positions.ceil().long(); projected = torch.zeros_like(probability)
            for atom in range(self.agent.atoms):
                if lower[atom] == upper[atom]: projected[lower[atom]] += probability[atom]
                else:
                    projected[lower[atom]] += probability[atom] * (upper[atom] - positions[atom])
                    projected[upper[atom]] += probability[atom] * (positions[atom] - lower[atom])
            return projected

    def _learn(self, beta: float = 0.4) -> float | None:
        if len(self.replay) < self.batch_size: return None
        batch, indices, importance = self.replay.sample(self.batch_size, beta)
        self.agent.reset_noise(); self.target.reset_noise()
        losses = []
        for transition in batch:
            predicted = self.agent.distribution(transition.graph)[transition.action]
            target = self._projection(transition)
            losses.append(-(target * predicted.log()).sum())
        loss_values = torch.stack(losses); loss = (loss_values * importance.to(loss_values.device)).mean()
        self.optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(self.agent.parameters(), 0.5); self.optimizer.step()
        self.replay.update(indices, loss_values); self.gradient_steps += 1
        if self.gradient_steps % self.target_update_interval == 0: self.target.load_state_dict(self.agent.state_dict())
        return float(loss.detach())

    def train(self, total_episodes: int) -> RainbowTrainStats:
        rewards, losses = [], []
        for episode in range(total_episodes):
            instance = self.instance_factory(self.base_seed + self.global_episode_count); self.global_episode_count += 1
            f_ref = self._f_ref(instance); env = CAPPORuleSelectionEnv(deepcopy(instance)); env.reset(); queue = deque(); total = 0.0
            while env.current_metrics and env.current_metrics.is_active:
                graph = self._graph(env); self.agent.reset_noise(); action = self.agent.greedy_action(graph)
                result = env.step_principle(action, self.library.get_rule(action)); reward = float(result.reward) / f_ref * self.reward_scale
                next_graph = None if result.terminated else self._graph(env); queue.append((graph, action, reward, next_graph, result.terminated)); self._flush_n_step(queue)
                learned = self._learn(0.4 + 0.6 * episode / max(1, total_episodes - 1))
                if learned is not None: losses.append(learned)
                total += reward
                if result.terminated: break
            self._flush_n_step(queue, force=True); rewards.append(total)
        return RainbowTrainStats(rewards, losses)


__all__ = ["HGTRainbowAgent", "NoisyLinear", "PrioritizedReplay", "RainbowTrainer", "RainbowTrainStats", "RainbowTransition"]
