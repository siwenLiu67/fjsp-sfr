"""PPO 轨迹缓冲区。"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from fjsp_sfr.rl.graph import HeteroGraphData
from fjsp_sfr.rl.action_groups import ActionGroups, ConcreteDispatch


@dataclass
class RolloutStep:
    """单步决策转移。"""

    graph: HeteroGraphData
    action: int
    log_prob: float
    value: float
    reward: float
    done: bool
    action_mask: torch.Tensor
    action_groups: ActionGroups = ()
    action_group_index: int | None = None
    concrete_dispatch: ConcreteDispatch | None = None


@dataclass
class RolloutBuffer:
    """存储一批完整或带 bootstrap 的轨迹。"""

    steps: list[RolloutStep] = field(default_factory=list)

    def add(self, step: RolloutStep) -> None:
        self.steps.append(step)

    def __len__(self) -> int:
        return len(self.steps)

    def clear(self) -> None:
        self.steps.clear()

    def compute_returns_and_advantages(
        self,
        gamma: float = 1.0,
        gae_lambda: float = 0.95,
        last_value: float = 0.0,
    ) -> tuple[list[float], list[float]]:
        """在 episode 边界处截断递推地计算 GAE 与 returns。"""

        returns = [0.0] * len(self.steps)
        advantages = [0.0] * len(self.steps)
        gae = 0.0

        for index in reversed(range(len(self.steps))):
            step = self.steps[index]
            if index == len(self.steps) - 1:
                next_value = 0.0 if step.done else float(last_value)
            else:
                next_value = 0.0 if step.done else self.steps[index + 1].value

            delta = step.reward + gamma * next_value - step.value
            if step.done:
                gae = delta
            else:
                gae = delta + gamma * gae_lambda * gae
            advantages[index] = float(gae)
            returns[index] = float(gae + step.value)

        return returns, advantages
