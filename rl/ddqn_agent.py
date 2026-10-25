"""Double-DQN selector sharing the existing HGT state encoder."""

from __future__ import annotations

import random

import torch
import torch.nn as nn

from fjsp_sfr.rl.dqn_agent import DQNTrainer, HGTDQNAgent


class HGTDDQNAgent(HGTDQNAgent):
    """Semantic name for the HGT Q-network used by Double DQN."""


class DDQNTrainer(DQNTrainer):
    """DQN trainer with online action selection and target evaluation."""

    def _learn(self) -> float | None:
        if len(self.replay) < self.batch_size:
            return None
        batch = random.sample(list(self.replay), self.batch_size)
        predicted = torch.stack([self.agent(item.graph)[item.action] for item in batch])
        targets = []
        with torch.no_grad():
            for item in batch:
                if item.done:
                    future = 0.0
                else:
                    selected = int(self.agent(item.next_graph).argmax().item())
                    future = float(self.target(item.next_graph)[selected].item())
                targets.append(item.reward + self.gamma * future)
        target_tensor = predicted.new_tensor(targets)
        loss = nn.functional.smooth_l1_loss(predicted, target_tensor)
        self.optimizer.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(self.agent.parameters(), 0.5)
        self.optimizer.step(); self.gradient_steps += 1
        if self.gradient_steps % self.target_update_interval == 0:
            self.target.load_state_dict(self.agent.state_dict())
        return float(loss.detach().item())


__all__ = ["DDQNTrainer", "HGTDDQNAgent"]
