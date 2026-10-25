"""Synchronous advantage actor-critic with the common HGT encoder."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.distributions import Categorical

from fjsp_sfr.core.objective import compute_objective
from fjsp_sfr.dispatch.rule_library import RuleLibrary
from fjsp_sfr.dispatch.theory_features import build_extended_theory_features
from fjsp_sfr.env.ca_observation import resolve_observation_scales
from fjsp_sfr.env.ca_rule_selection import CAPPORuleSelectionEnv
from fjsp_sfr.rl.graph import apply_information_masks, build_hetero_graph
from fjsp_sfr.rl.ppo_agent import HGTPPOAgent


class HGTA2CAgent(HGTPPOAgent):
    """Actor/critic heads and shared HGT encoder identical to HGT-PPO."""


@dataclass(frozen=True)
class A2CTrainStats:
    mean_reward: float
    policy_loss: float
    value_loss: float
    entropy: float


class A2CTrainer:
    def __init__(
        self,
        agent: HGTA2CAgent,
        library: RuleLibrary,
        instance_factory,
        *,
        base_seed: int = 1000,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        reward_scale: float = 10.0,
        entropy_start: float = 0.02,
        entropy_end: float = 0.002,
        value_coef: float = 0.5,
        feature_set: str = "theory11_plus3",
        mask_checkpoint: bool = False,
        mask_forecast: bool = False,
    ) -> None:
        self.agent, self.library, self.instance_factory = agent, library, instance_factory
        self.base_seed, self.gamma, self.gae_lambda = int(base_seed), float(gamma), float(gae_lambda)
        self.reward_scale, self.entropy_start, self.entropy_end = float(reward_scale), float(entropy_start), float(entropy_end)
        self.value_coef, self.feature_set = float(value_coef), feature_set
        self.mask_checkpoint, self.mask_forecast = bool(mask_checkpoint), bool(mask_forecast)
        self.optimizer = torch.optim.Adam(agent.parameters(), lr=lr)
        self.global_episode_count = 0

    def _graph(self, env):
        problem, state, metrics = env._problem_view(), env._state, env.current_metrics
        time_scale, processing_scale = resolve_observation_scales(problem, state)
        rows = build_extended_theory_features(problem, state, metrics, time_scale=time_scale, feature_set=self.feature_set)
        graph = build_hetero_graph(problem, state, metrics, rows, time_scale=time_scale, processing_scale=processing_scale, feature_set=self.feature_set)
        return apply_information_masks(graph, feature_set=self.feature_set, mask_checkpoint=self.mask_checkpoint, mask_forecast=self.mask_forecast)

    def _f_ref(self, instance) -> float:
        values = []
        for index, rule in enumerate(self.library.rules):
            copy = deepcopy(instance); env = CAPPORuleSelectionEnv(copy); env.reset()
            while env.current_metrics and env.current_metrics.is_active:
                result = env.step_principle(index, rule)
                if result.terminated: break
            values.append(float(compute_objective(copy, env._state).Z))
        return max(sum(values) / len(values), 1.0)

    def _episode(self, instance):
        env = CAPPORuleSelectionEnv(deepcopy(instance)); env.reset(); f_ref = self._f_ref(instance)
        graphs, actions, rewards, dones = [], [], [], []
        while env.current_metrics and env.current_metrics.is_active:
            graph = self._graph(env)
            with torch.no_grad():
                log_probs, _value = self.agent(graph, torch.ones(self.library.action_size(), dtype=torch.bool))
                action = int(Categorical(logits=log_probs).sample().item())
            result = env.step_principle(action, self.library.get_rule(action))
            graphs.append(graph); actions.append(action); rewards.append(float(result.reward) / f_ref * self.reward_scale); dones.append(result.terminated)
            if result.terminated: break
        return graphs, actions, rewards, dones

    def train(self, total_episodes: int, *, episodes_per_update: int = 32) -> list[A2CTrainStats]:
        updates = (total_episodes + episodes_per_update - 1) // episodes_per_update
        output = []
        for update in range(updates):
            fraction = update / max(1, updates - 1)
            entropy_coef = self.entropy_start + fraction * (self.entropy_end - self.entropy_start)
            trajectories = []
            count = min(episodes_per_update, total_episodes - update * episodes_per_update)
            for _ in range(count):
                instance = self.instance_factory(self.base_seed + self.global_episode_count); self.global_episode_count += 1
                trajectories.append(self._episode(instance))
            policy_terms, value_terms, entropies, total_rewards = [], [], [], []
            for graphs, actions, rewards, dones in trajectories:
                values, log_probabilities, entropy_values = [], [], []
                for graph, action in zip(graphs, actions, strict=True):
                    log_probs, value = self.agent(graph, torch.ones(self.library.action_size(), dtype=torch.bool))
                    distribution = Categorical(logits=log_probs)
                    values.append(value); log_probabilities.append(distribution.log_prob(torch.tensor(action))); entropy_values.append(distribution.entropy())
                advantages = []; gae = values[0].new_zeros(()) if values else torch.tensor(0.0)
                for index in reversed(range(len(rewards))):
                    next_value = values[index + 1].detach() if index + 1 < len(values) and not dones[index] else values[index].new_zeros(())
                    delta = values[index].new_tensor(rewards[index]) + self.gamma * next_value - values[index]
                    gae = delta + self.gamma * self.gae_lambda * (0.0 if dones[index] else 1.0) * gae
                    advantages.insert(0, gae)
                if advantages:
                    advantage_tensor = torch.stack(advantages)
                    returns = advantage_tensor.detach() + torch.stack(values).detach()
                    policy_terms.append(-(torch.stack(log_probabilities) * advantage_tensor.detach()).mean())
                    value_terms.append(nn.functional.mse_loss(torch.stack(values), returns))
                    entropies.append(torch.stack(entropy_values).mean())
                total_rewards.append(sum(rewards))
            policy_loss = torch.stack(policy_terms).mean(); value_loss = torch.stack(value_terms).mean(); entropy = torch.stack(entropies).mean()
            loss = policy_loss + self.value_coef * value_loss - entropy_coef * entropy
            self.optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(self.agent.parameters(), 0.5); self.optimizer.step()
            output.append(A2CTrainStats(sum(total_rewards) / len(total_rewards), float(policy_loss.detach()), float(value_loss.detach()), float(entropy.detach())))
        return output


__all__ = ["A2CTrainer", "A2CTrainStats", "HGTA2CAgent"]
