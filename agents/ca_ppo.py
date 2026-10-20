"""Checkpoint-attention PPO with a per-token shortfall auxiliary head."""

from __future__ import annotations

import math
import json
from dataclasses import asdict, dataclass
from itertools import chain
from pathlib import Path
from statistics import fmean

import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical

from fjsp_sfr.core.objective import ObjectiveResult
from fjsp_sfr.dispatch.types import RuleAction
from fjsp_sfr.env.ca_action_tokens import ACTION_TOKEN_FEATURE_NAMES
from fjsp_sfr.env.ca_observation import (
    CANDIDATE_FEATURE_NAMES,
    JOB_CHECKPOINT_RELATION_NAMES,
    RELATIONAL_CANDIDATE_FEATURE_NAMES,
    RELATIONAL_CHECKPOINT_FEATURE_NAMES,
    THEORY_FACTOR_NAMES,
)
from fjsp_sfr.env.ca_rule_tokens import RULE_TOKEN_FEATURES
from fjsp_sfr.env.ca_rule_selection import CADecisionObservation, CAPPORuleSelectionEnv

from .relational_encoder import (
    CandidateCheckpointRelationalAttentionEncoder,
    RelationalEncoderOutput,
)
from .factorized_priority_actor import (
    FactorizedPriorityOutput,
    TheoryGuidedFactorizedPriorityActor,
)
from .principle_library import PrincipleLibrary


_NUM_RULES = len(RuleAction)


@dataclass(frozen=True)
class CAPPOConfig:
    shop_hidden: int = 64
    checkpoint_hidden: int = 32
    fused_hidden: int = 64
    encoder_type: str = "checkpoint"
    actor_type: str = "rule_selector"
    principle_library_path: str | None = None
    use_context_gate: bool = True
    use_interactions: bool = True
    action_semantics: str = "rule"
    policy_init: str = "uniform"
    actor_critic_mode: str = "shared"
    learning_rate: float = 3e-4
    actor_learning_rate: float | None = None
    critic_learning_rate: float | None = None
    gamma: float = 1.0
    gae_lambda: float = 1.0
    clip_epsilon: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    auxiliary_coefficient: float = 0.2
    update_epochs: int = 4
    gradient_clip_norm: float = 0.5
    return_scale: float = 1.0
    return_normalization_epsilon: float = 1e-4
    return_normalization_clip: float = 10.0
    seed: int = 0
    device: str = "cpu"

    def __post_init__(self) -> None:
        if min(self.shop_hidden, self.checkpoint_hidden, self.fused_hidden) <= 0:
            raise ValueError("network widths must be positive")
        if self.encoder_type not in {
            "checkpoint",
            "candidate_checkpoint",
            "action_token",
            "rule_conditioned",
        }:
            raise ValueError(
                "encoder_type must be 'checkpoint', 'candidate_checkpoint', "
                "'action_token', or 'rule_conditioned'"
            )
        if self.action_semantics not in {"rule", "unique_job"}:
            raise ValueError("action_semantics must be 'rule' or 'unique_job'")
        if self.actor_type not in {
            "rule_selector",
            "factorized_priority",
            "principle_selector",
        }:
            raise ValueError(
                "actor_type must be 'rule_selector', 'factorized_priority', "
                "or 'principle_selector'"
            )
        if (
            self.actor_type in {"factorized_priority", "principle_selector"}
            and self.encoder_type != "candidate_checkpoint"
        ):
            raise ValueError(
                f"{self.actor_type} requires encoder_type='candidate_checkpoint'"
            )
        if self.actor_type == "principle_selector" and not self.principle_library_path:
            raise ValueError("principle_selector requires principle_library_path")
        if self.actor_critic_mode not in {"shared", "separate"}:
            raise ValueError("actor_critic_mode must be 'shared' or 'separate'")
        if self.actor_critic_mode == "separate" and self.encoder_type != "checkpoint":
            raise ValueError(
                "separate actor-critic requires encoder_type='checkpoint'"
            )
        if self.policy_init not in {
            "uniform",
            "r5_prior",
            "r5_residual_prior",
        }:
            raise ValueError(
                "policy_init must be 'uniform', 'r5_prior', or "
                "'r5_residual_prior'"
            )
        if (
            self.actor_type in {"factorized_priority", "principle_selector"}
            and self.policy_init != "uniform"
        ):
            raise ValueError(f"{self.actor_type} requires policy_init='uniform'")
        if (
            self.policy_init == "r5_residual_prior"
            and self.encoder_type != "checkpoint"
        ):
            raise ValueError(
                "r5_residual_prior requires encoder_type='checkpoint'"
            )
        if self.learning_rate <= 0 or self.update_epochs <= 0:
            raise ValueError("learning rate and update epochs must be positive")
        if self.actor_critic_mode == "separate" and (
            self.actor_learning_rate is None
            or self.critic_learning_rate is None
            or self.actor_learning_rate <= 0
            or self.critic_learning_rate <= 0
        ):
            raise ValueError(
                "separate actor-critic requires positive actor and critic learning rates"
            )
        if not 0.0 <= self.gamma <= 1.0 or not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("gamma and gae_lambda must be in [0, 1]")
        if self.clip_epsilon <= 0:
            raise ValueError("clip epsilon must be positive")
        if self.gradient_clip_norm <= 0.0:
            raise ValueError("gradient clip norm must be positive")
        if min(
            self.return_scale,
            self.return_normalization_epsilon,
            self.return_normalization_clip,
        ) <= 0.0:
            raise ValueError("return scaling and normalization parameters must be positive")


@dataclass
class RunningValueNormalizer:
    """Numerically stable running normalization for return/value targets."""

    epsilon: float = 1e-4
    clip: float = 10.0
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    @property
    def variance(self) -> float:
        if self.count == 0:
            return 1.0
        return max(self.m2 / self.count, self.epsilon**2)

    @property
    def std(self) -> float:
        return self.variance**0.5

    def update(self, values: torch.Tensor) -> None:
        flat = values.detach().to(dtype=torch.float64, device="cpu").flatten()
        batch_count = int(flat.numel())
        if batch_count == 0:
            return
        batch_mean = float(flat.mean().item())
        batch_m2 = float(((flat - batch_mean) ** 2).sum().item())
        if self.count == 0:
            self.count = batch_count
            self.mean = batch_mean
            self.m2 = batch_m2
            return
        total = self.count + batch_count
        delta = batch_mean - self.mean
        self.mean += delta * batch_count / total
        self.m2 += batch_m2 + delta * delta * self.count * batch_count / total
        self.count = total

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        normalized = (values - self.mean) / self.std
        return normalized.clamp(-self.clip, self.clip)

    def denormalize(self, values: torch.Tensor) -> torch.Tensor:
        return values * self.std + self.mean

    def state_dict(self) -> dict[str, float | int]:
        return {"count": self.count, "mean": self.mean, "m2": self.m2}

    def load_state_dict(self, state: dict[str, float | int]) -> None:
        self.count = int(state.get("count", 0))
        self.mean = float(state.get("mean", 0.0))
        self.m2 = float(state.get("m2", 0.0))


class CheckpointAttentionActorCritic(nn.Module):
    """Checkpoint actor-critic with optional candidate or action-token context."""

    def __init__(
        self,
        shop_hidden: int = 64,
        checkpoint_hidden: int = 32,
        fused_hidden: int = 64,
        encoder_type: str = "checkpoint",
        policy_init: str = "uniform",
        actor_type: str = "rule_selector",
        use_context_gate: bool = True,
        use_interactions: bool = True,
        num_principles: int | None = None,
    ) -> None:
        super().__init__()
        if encoder_type not in {
            "checkpoint",
            "candidate_checkpoint",
            "action_token",
            "rule_conditioned",
        }:
            raise ValueError("unknown encoder type")
        if policy_init == "r5_residual_prior" and encoder_type != "checkpoint":
            raise ValueError(
                "r5_residual_prior requires encoder_type='checkpoint'"
            )
        self.shop_hidden = shop_hidden
        self.checkpoint_hidden = checkpoint_hidden
        self.encoder_type = encoder_type
        self.actor_type = actor_type
        self.shop_encoder = nn.Sequential(
            nn.Linear(7, shop_hidden), nn.Tanh(),
            nn.Linear(shop_hidden, shop_hidden), nn.Tanh(),
        )
        self.checkpoint_encoder = nn.Sequential(
            nn.Linear(8, checkpoint_hidden), nn.Tanh(),
            nn.Linear(checkpoint_hidden, checkpoint_hidden), nn.Tanh(),
        )
        self.shop_attention = nn.Linear(shop_hidden, checkpoint_hidden, bias=False)
        self.checkpoint_attention = nn.Linear(
            checkpoint_hidden, checkpoint_hidden, bias=False
        )
        self.attention_score = nn.Linear(checkpoint_hidden, 1, bias=False)
        if encoder_type == "candidate_checkpoint":
            self.relational_encoder = (
                CandidateCheckpointRelationalAttentionEncoder(
                    model_dim=64,
                    num_heads=4,
                )
            )
            fusion_input = 128
        else:
            fusion_input = shop_hidden + checkpoint_hidden
        self.fusion = nn.Sequential(nn.Linear(fusion_input, fused_hidden), nn.Tanh())
        if actor_type == "principle_selector":
            if num_principles is None or num_principles <= 0:
                raise ValueError("principle_selector requires a non-empty library")
            actor_outputs = num_principles
        else:
            actor_outputs = _NUM_RULES
        self.actor = nn.Linear(fused_hidden, actor_outputs)
        if actor_type == "factorized_priority":
            if encoder_type != "candidate_checkpoint":
                raise ValueError(
                    "factorized_priority requires encoder_type='candidate_checkpoint'"
                )
            self.factorized_actor = TheoryGuidedFactorizedPriorityActor(
                use_context_gate=use_context_gate,
                use_interactions=use_interactions,
            )
        prior_logits = torch.zeros(_NUM_RULES)
        if policy_init == "r5_residual_prior":
            nn.init.zeros_(self.actor.weight)
            nn.init.zeros_(self.actor.bias)
            prior_logits[int(RuleAction.SERVICE_VALUE_DENSITY)] = math.log(7.0)
        self.register_buffer("policy_prior_logits", prior_logits)
        self.policy_init = policy_init
        self.action_encoder = nn.Sequential(
            nn.Linear(len(ACTION_TOKEN_FEATURE_NAMES), checkpoint_hidden),
            nn.Tanh(),
            nn.Linear(checkpoint_hidden, checkpoint_hidden),
        )
        self.rule_embedding = nn.Embedding(_NUM_RULES, checkpoint_hidden)
        self.action_context = nn.Linear(fused_hidden, checkpoint_hidden, bias=False)
        self.action_score = nn.Linear(checkpoint_hidden, 1, bias=False)
        if encoder_type == "rule_conditioned":
            self.rule_token_encoder = nn.Sequential(
                nn.Linear(len(RULE_TOKEN_FEATURES), checkpoint_hidden),
                nn.Tanh(),
                nn.Linear(checkpoint_hidden, checkpoint_hidden),
            )
        self.critic = nn.Linear(fused_hidden, 1)
        self.auxiliary_head = nn.Sequential(
            nn.Linear(checkpoint_hidden + shop_hidden, checkpoint_hidden),
            nn.Tanh(),
            nn.Linear(checkpoint_hidden, 1),
        )

    def forward(
        self,
        shop: torch.Tensor,
        checkpoint_tokens: torch.Tensor,
        checkpoint_mask: torch.Tensor,
        action_tokens: torch.Tensor | None = None,
        candidate_tokens: torch.Tensor | None = None,
        candidate_mask: torch.Tensor | None = None,
        rule_tokens: torch.Tensor | None = None,
        relational_candidate_tokens: torch.Tensor | None = None,
        relational_checkpoint_tokens: torch.Tensor | None = None,
        candidate_checkpoint_relations: torch.Tensor | None = None,
        candidate_checkpoint_mask: torch.Tensor | None = None,
        theory_factor_values: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        shop_embedding = self.shop_encoder(shop)
        batch_size, token_count, _ = checkpoint_tokens.shape
        if token_count == 0:
            pooled = torch.zeros(
                (batch_size, self.checkpoint_hidden),
                dtype=shop.dtype,
                device=shop.device,
            )
            attention = torch.zeros((batch_size, 0), device=shop.device)
            predictions = torch.zeros((batch_size, 0), device=shop.device)
        else:
            checkpoint_embedding = self.checkpoint_encoder(checkpoint_tokens)
            scores = self.attention_score(
                torch.tanh(
                    self.checkpoint_attention(checkpoint_embedding)
                    + self.shop_attention(shop_embedding).unsqueeze(1)
                )
            ).squeeze(-1)
            scores = scores.masked_fill(~checkpoint_mask, -1e9)
            attention = torch.softmax(scores, dim=1) * checkpoint_mask.float()
            attention = attention / attention.sum(dim=1, keepdim=True).clamp_min(1e-12)
            pooled = (attention.unsqueeze(-1) * checkpoint_embedding).sum(dim=1)
            expanded_shop = shop_embedding.unsqueeze(1).expand(-1, token_count, -1)
            predictions = self.auxiliary_head(
                torch.cat((checkpoint_embedding, expanded_shop), dim=-1)
            ).squeeze(-1)
        if self.encoder_type == "candidate_checkpoint":
            relational_output = self.encode_candidate_checkpoints(
                shop,
                relational_candidate_tokens,
                relational_checkpoint_tokens,
                candidate_checkpoint_relations,
                candidate_mask,
                candidate_checkpoint_mask,
            )
            fused = self.fusion(relational_output.state_embedding)
            attention = relational_output.attention_weights
        else:
            fused = self.fusion(torch.cat((shop_embedding, pooled), dim=-1))
        if self.actor_type == "factorized_priority":
            if theory_factor_values is None or candidate_mask is None:
                raise ValueError(
                    "factorized_priority requires theory factors and candidate mask"
                )
            factorized_output = self.factorized_actor(
                theory_factor_values,
                relational_output.candidate_embeddings,
                relational_output.global_context,
                candidate_mask,
            )
            logits = factorized_output.priority_logits
        elif self.encoder_type in {"checkpoint", "candidate_checkpoint"}:
            logits = self.actor(fused)
            if self.policy_init == "r5_residual_prior":
                logits = logits + self.policy_prior_logits
        elif self.encoder_type == "action_token":
            if action_tokens is None or action_tokens.shape[1:] != (
                _NUM_RULES,
                len(ACTION_TOKEN_FEATURE_NAMES),
            ):
                raise ValueError(
                    "action-token encoder requires [batch, 8, 14] tokens"
                )
            rule_ids = torch.arange(_NUM_RULES, device=shop.device)
            action_embedding = torch.tanh(
                self.action_encoder(action_tokens)
                + self.rule_embedding(rule_ids).unsqueeze(0)
                + self.action_context(fused).unsqueeze(1)
            )
            logits = self.action_score(action_embedding).squeeze(-1)
        else:
            if rule_tokens is None or rule_tokens.shape[1:] != (
                _NUM_RULES,
                len(RULE_TOKEN_FEATURES),
            ):
                raise ValueError(
                    "rule-conditioned encoder requires [batch, 8, 10] tokens"
                )
            rule_ids = torch.arange(_NUM_RULES, device=shop.device)
            rule_state = torch.tanh(
                self.rule_token_encoder(rule_tokens)
                + self.rule_embedding(rule_ids).unsqueeze(0)
                + self.action_context(fused).unsqueeze(1)
            )
            logits = self.action_score(rule_state).squeeze(-1)
        return logits, self.critic(fused).squeeze(-1), predictions, attention

    def factorized_priority(
        self,
        encoder_output: RelationalEncoderOutput,
        factor_values: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> FactorizedPriorityOutput:
        """Expose all theory-factor contributions for analysis and ablation."""

        if self.actor_type != "factorized_priority":
            raise RuntimeError(
                "factorized priority output requires actor_type='factorized_priority'"
            )
        return self.factorized_actor(
            factor_values,
            encoder_output.candidate_embeddings,
            encoder_output.global_context,
            candidate_mask,
        )

    def encode_candidate_checkpoints(
        self,
        shop: torch.Tensor,
        candidate_tokens: torch.Tensor | None,
        checkpoint_tokens: torch.Tensor | None,
        relations: torch.Tensor | None,
        candidate_mask: torch.Tensor | None,
        candidate_checkpoint_mask: torch.Tensor | None,
    ) -> RelationalEncoderOutput:
        """Expose candidate embeddings for the next actor-development stage."""

        if self.encoder_type != "candidate_checkpoint":
            raise RuntimeError(
                "candidate-checkpoint representations require encoder_type='candidate_checkpoint'"
            )
        if any(
            item is None
            for item in (
                candidate_tokens,
                checkpoint_tokens,
                relations,
                candidate_mask,
                candidate_checkpoint_mask,
            )
        ):
            raise ValueError(
                "candidate-checkpoint encoder requires candidate, checkpoint, relation, and mask tensors"
            )
        assert candidate_tokens is not None
        assert checkpoint_tokens is not None
        assert relations is not None
        assert candidate_mask is not None
        assert candidate_checkpoint_mask is not None
        return self.relational_encoder(
            shop,
            candidate_tokens,
            checkpoint_tokens,
            relations,
            candidate_mask,
            candidate_checkpoint_mask,
        )


class _CheckpointStateEncoder(nn.Module):
    """Independent shop/checkpoint attention stack for one value or policy side."""

    def __init__(
        self,
        shop_hidden: int,
        checkpoint_hidden: int,
        fused_hidden: int,
    ) -> None:
        super().__init__()
        self.checkpoint_hidden = checkpoint_hidden
        self.shop_encoder = nn.Sequential(
            nn.Linear(7, shop_hidden),
            nn.Tanh(),
            nn.Linear(shop_hidden, shop_hidden),
            nn.Tanh(),
        )
        self.checkpoint_encoder = nn.Sequential(
            nn.Linear(8, checkpoint_hidden),
            nn.Tanh(),
            nn.Linear(checkpoint_hidden, checkpoint_hidden),
            nn.Tanh(),
        )
        self.shop_attention = nn.Linear(shop_hidden, checkpoint_hidden, bias=False)
        self.checkpoint_attention = nn.Linear(
            checkpoint_hidden, checkpoint_hidden, bias=False
        )
        self.attention_score = nn.Linear(checkpoint_hidden, 1, bias=False)
        self.fusion = nn.Sequential(
            nn.Linear(shop_hidden + checkpoint_hidden, fused_hidden), nn.Tanh()
        )

    def forward(
        self,
        shop: torch.Tensor,
        checkpoint_tokens: torch.Tensor,
        checkpoint_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        shop_embedding = self.shop_encoder(shop)
        batch_size, token_count, _ = checkpoint_tokens.shape
        if token_count == 0:
            checkpoint_embedding = torch.zeros(
                (batch_size, 0, self.checkpoint_hidden),
                dtype=shop.dtype,
                device=shop.device,
            )
            pooled = torch.zeros(
                (batch_size, self.checkpoint_hidden),
                dtype=shop.dtype,
                device=shop.device,
            )
            attention = torch.zeros((batch_size, 0), device=shop.device)
        else:
            checkpoint_embedding = self.checkpoint_encoder(checkpoint_tokens)
            scores = self.attention_score(
                torch.tanh(
                    self.checkpoint_attention(checkpoint_embedding)
                    + self.shop_attention(shop_embedding).unsqueeze(1)
                )
            ).squeeze(-1)
            scores = scores.masked_fill(~checkpoint_mask, -1e9)
            attention = torch.softmax(scores, dim=1) * checkpoint_mask.float()
            attention = attention / attention.sum(
                dim=1, keepdim=True
            ).clamp_min(1e-12)
            pooled = (attention.unsqueeze(-1) * checkpoint_embedding).sum(dim=1)
        fused = self.fusion(torch.cat((shop_embedding, pooled), dim=-1))
        return fused, shop_embedding, checkpoint_embedding, attention


class SeparateCheckpointActorCritic(nn.Module):
    """Checkpoint policy/value networks with no shared trainable parameters."""

    def __init__(
        self,
        shop_hidden: int = 64,
        checkpoint_hidden: int = 32,
        fused_hidden: int = 64,
        policy_init: str = "uniform",
    ) -> None:
        super().__init__()
        self.actor_encoder = _CheckpointStateEncoder(
            shop_hidden, checkpoint_hidden, fused_hidden
        )
        self.critic_encoder = _CheckpointStateEncoder(
            shop_hidden, checkpoint_hidden, fused_hidden
        )
        self.actor = nn.Linear(fused_hidden, _NUM_RULES)
        self.critic = nn.Linear(fused_hidden, 1)
        self.auxiliary_head = nn.Sequential(
            nn.Linear(checkpoint_hidden + shop_hidden, checkpoint_hidden),
            nn.Tanh(),
            nn.Linear(checkpoint_hidden, 1),
        )
        prior_logits = torch.zeros(_NUM_RULES)
        if policy_init == "r5_residual_prior":
            nn.init.zeros_(self.actor.weight)
            nn.init.zeros_(self.actor.bias)
            prior_logits[int(RuleAction.SERVICE_VALUE_DENSITY)] = math.log(7.0)
        self.register_buffer("policy_prior_logits", prior_logits)
        self.policy_init = policy_init

    def actor_parameters(self):
        return chain(self.actor_encoder.parameters(), self.actor.parameters())

    def critic_parameters(self):
        return chain(
            self.critic_encoder.parameters(),
            self.critic.parameters(),
            self.auxiliary_head.parameters(),
        )

    def forward(
        self,
        shop: torch.Tensor,
        checkpoint_tokens: torch.Tensor,
        checkpoint_mask: torch.Tensor,
        action_tokens: torch.Tensor | None = None,
        candidate_tokens: torch.Tensor | None = None,
        candidate_mask: torch.Tensor | None = None,
        rule_tokens: torch.Tensor | None = None,
        relational_candidate_tokens: torch.Tensor | None = None,
        relational_checkpoint_tokens: torch.Tensor | None = None,
        candidate_checkpoint_relations: torch.Tensor | None = None,
        candidate_checkpoint_mask: torch.Tensor | None = None,
        theory_factor_values: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        del (
            action_tokens,
            candidate_tokens,
            candidate_mask,
            rule_tokens,
            relational_candidate_tokens,
            relational_checkpoint_tokens,
            candidate_checkpoint_relations,
            candidate_checkpoint_mask,
            theory_factor_values,
        )
        actor_fused, _actor_shop, _actor_tokens, actor_attention = (
            self.actor_encoder(shop, checkpoint_tokens, checkpoint_mask)
        )
        critic_fused, critic_shop, critic_tokens, _critic_attention = (
            self.critic_encoder(shop, checkpoint_tokens, checkpoint_mask)
        )
        token_count = checkpoint_tokens.shape[1]
        if token_count == 0:
            predictions = torch.zeros(
                (shop.shape[0], 0), dtype=shop.dtype, device=shop.device
            )
        else:
            expanded_shop = critic_shop.unsqueeze(1).expand(-1, token_count, -1)
            predictions = self.auxiliary_head(
                torch.cat((critic_tokens, expanded_shop), dim=-1)
            ).squeeze(-1)
        logits = self.actor(actor_fused)
        if self.policy_init == "r5_residual_prior":
            logits = logits + self.policy_prior_logits
        return (
            logits,
            self.critic(critic_fused).squeeze(-1),
            predictions,
            actor_attention,
        )


ActionGroup = tuple[int, ...]
ActionGroups = tuple[ActionGroup, ...]


@dataclass(frozen=True)
class _PolicySelection:
    action: RuleAction | int
    action_groups: ActionGroups
    action_group_index: int
    log_probability: float
    value: float
    probabilities: tuple[float, ...]


@dataclass
class _Record:
    observation: CADecisionObservation
    action: int
    action_mask: tuple[bool, ...]
    action_groups: ActionGroups
    action_group_index: int
    old_log_probability: float
    value: float
    reward: float
    terminated: bool
    auxiliary_targets: tuple[float, ...] = ()
    policy_probabilities: tuple[float, ...] = ()
    executed_candidate_index: int | None = None
    risk_regime: str | None = None


@dataclass(frozen=True)
class CAPPOEpisodeResult:
    decision_count: int
    total_reward: float
    objective: ObjectiveResult
    mean_loss: float | None
    update_stats: "PPOUpdateStats | None" = None


@dataclass(frozen=True)
class PPOUpdateStats:
    total_loss: float
    policy_loss: float
    value_loss: float
    auxiliary_loss: float
    entropy: float
    approximate_kl: float
    clip_fraction: float
    explained_variance: float
    global_gradient_norm: float
    mean_max_action_prob: float
    logit_std: float
    ratio_std: float
    mean_advantage: float
    advantage_std: float
    transitions: int


@dataclass(frozen=True)
class PrincipleDiagnostics:
    principle_names: tuple[str, ...]
    selection_counts: tuple[int, ...]
    selection_fractions: tuple[float, ...]
    mean_selection_probabilities: tuple[float, ...]
    regime_selection_fractions: dict[str, tuple[float, ...]]
    candidate_selection_counts: dict[int, int]
    switching_frequency: float


@dataclass(frozen=True)
class CAPPOCollectedEpisode:
    records: tuple[_Record, ...]
    total_reward: float
    objective: ObjectiveResult

    @property
    def decision_count(self) -> int:
        return len(self.records)


def generalized_advantages(
    rewards: torch.Tensor,
    values: torch.Tensor,
    terminated: torch.Tensor,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    advantages = torch.zeros_like(rewards)
    next_advantage = torch.zeros((), dtype=rewards.dtype, device=rewards.device)
    next_value = torch.zeros((), dtype=rewards.dtype, device=rewards.device)
    for index in range(len(rewards) - 1, -1, -1):
        continuing = 1.0 - terminated[index].float()
        delta = rewards[index] + gamma * continuing * next_value - values[index]
        next_advantage = delta + gamma * gae_lambda * continuing * next_advantage
        advantages[index] = next_advantage
        next_value = values[index]
    return advantages, advantages + values


def _batch_observations(observations, device):
    batch = len(observations)
    max_tokens = max((len(obs.checkpoint_tokens) for obs in observations), default=0)
    max_candidates = max((len(obs.candidate_tokens) for obs in observations), default=0)
    shop = torch.tensor([obs.shop for obs in observations], dtype=torch.float32, device=device)
    tokens = torch.zeros((batch, max_tokens, 8), dtype=torch.float32, device=device)
    mask = torch.zeros((batch, max_tokens), dtype=torch.bool, device=device)
    candidate_tokens = torch.zeros(
        (batch, max_candidates, len(CANDIDATE_FEATURE_NAMES)),
        dtype=torch.float32,
        device=device,
    )
    candidate_mask = torch.zeros(
        (batch, max_candidates), dtype=torch.bool, device=device
    )
    action_mask = torch.tensor(
        [observation.action_mask for observation in observations],
        dtype=torch.bool,
        device=device,
    )
    action_tokens = torch.tensor(
        [observation.action_tokens for observation in observations],
        dtype=torch.float32,
        device=device,
    )
    rule_tokens = torch.tensor(
        [observation.rule_tokens for observation in observations],
        dtype=torch.float32,
        device=device,
    )
    for row, observation in enumerate(observations):
        count = len(observation.checkpoint_tokens)
        if count:
            tokens[row, :count] = torch.tensor(
                observation.checkpoint_tokens, dtype=torch.float32, device=device
            )
            mask[row, :count] = True
        candidate_count = len(observation.candidate_tokens)
        if candidate_count:
            candidate_tokens[row, :candidate_count] = torch.tensor(
                observation.candidate_tokens,
                dtype=torch.float32,
                device=device,
            )
            candidate_mask[row, :candidate_count] = True
    return (
        shop,
        tokens,
        mask,
        action_mask,
        action_tokens,
        rule_tokens,
        candidate_tokens,
        candidate_mask,
    )


def _batch_relational_observations(observations, device):
    """Pad the relational streams without altering the legacy PPO batch layout."""

    batch = len(observations)
    max_candidates = max(
        (len(obs.relational_candidate_tokens) for obs in observations), default=0
    )
    max_checkpoints = max(
        (len(obs.relational_checkpoint_tokens) for obs in observations), default=0
    )
    candidate_tokens = torch.zeros(
        (batch, max_candidates, len(RELATIONAL_CANDIDATE_FEATURE_NAMES)),
        dtype=torch.float32,
        device=device,
    )
    checkpoint_tokens = torch.zeros(
        (batch, max_checkpoints, len(RELATIONAL_CHECKPOINT_FEATURE_NAMES)),
        dtype=torch.float32,
        device=device,
    )
    relations = torch.zeros(
        (
            batch,
            max_candidates,
            max_checkpoints,
            len(JOB_CHECKPOINT_RELATION_NAMES),
        ),
        dtype=torch.float32,
        device=device,
    )
    relation_mask = torch.zeros(
        (batch, max_candidates, max_checkpoints),
        dtype=torch.bool,
        device=device,
    )
    for row, observation in enumerate(observations):
        candidate_count = len(observation.relational_candidate_tokens)
        checkpoint_count = len(observation.relational_checkpoint_tokens)
        if candidate_count != len(observation.candidate_tokens):
            raise ValueError("legacy and relational candidate streams must align")
        if candidate_count:
            candidate_tokens[row, :candidate_count] = torch.tensor(
                observation.relational_candidate_tokens,
                dtype=torch.float32,
                device=device,
            )
        if checkpoint_count:
            checkpoint_tokens[row, :checkpoint_count] = torch.tensor(
                observation.relational_checkpoint_tokens,
                dtype=torch.float32,
                device=device,
            )
        if candidate_count and checkpoint_count:
            relations[row, :candidate_count, :checkpoint_count] = torch.tensor(
                observation.candidate_checkpoint_relations,
                dtype=torch.float32,
                device=device,
            )
            relation_mask[row, :candidate_count, :checkpoint_count] = torch.tensor(
                observation.candidate_checkpoint_mask,
                dtype=torch.bool,
                device=device,
            )
    return candidate_tokens, checkpoint_tokens, relations, relation_mask


def _batch_theory_factor_values(observations, device):
    """Pad the five theory factors in current candidate order."""

    batch = len(observations)
    max_candidates = max(
        (len(observation.theory_factor_values) for observation in observations),
        default=0,
    )
    factors = torch.zeros(
        (batch, max_candidates, len(THEORY_FACTOR_NAMES)),
        dtype=torch.float32,
        device=device,
    )
    for row, observation in enumerate(observations):
        candidate_count = len(observation.theory_factor_values)
        if candidate_count != len(observation.candidate_job_ids):
            raise ValueError("theory factors and candidate ids must align")
        if candidate_count:
            factors[row, :candidate_count] = torch.tensor(
                observation.theory_factor_values,
                dtype=torch.float32,
                device=device,
            )
    return factors


def _masked_logits(logits: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
    if logits.shape != action_mask.shape:
        raise ValueError("action mask must have the same shape as actor logits")
    if not action_mask.any(dim=1).all():
        raise ValueError("every decision must have at least one applicable rule")
    return logits.masked_fill(~action_mask, torch.finfo(logits.dtype).min)


def build_action_groups(
    observation: CADecisionObservation,
    action_semantics: str,
) -> ActionGroups:
    """Group applicable rule ids by their concrete scheduling action."""

    applicable = tuple(
        rule_id for rule_id, valid in enumerate(observation.action_mask) if valid
    )
    if not applicable:
        raise ValueError("every decision must have at least one applicable rule")
    if action_semantics == "rule":
        return tuple((rule_id,) for rule_id in applicable)
    if action_semantics != "unique_job":
        raise ValueError("action_semantics must be 'rule' or 'unique_job'")
    if len(observation.action_candidate_keys) != len(observation.action_mask):
        raise ValueError("every rule must have one concrete action candidate key")

    grouped: dict[tuple[int, int], list[int]] = {}
    for rule_id in applicable:
        key = observation.action_candidate_keys[rule_id]
        grouped.setdefault(key, []).append(rule_id)
    return tuple(tuple(rule_ids) for rule_ids in grouped.values())


def _validate_action_groups(
    action_groups: ActionGroups,
    action_mask: tuple[bool, ...],
) -> None:
    if not action_groups or any(not group for group in action_groups):
        raise ValueError("each decision requires non-empty action groups")
    flattened = tuple(rule_id for group in action_groups for rule_id in group)
    applicable = tuple(rule_id for rule_id, valid in enumerate(action_mask) if valid)
    if tuple(sorted(flattened)) != applicable or len(set(flattened)) != len(flattened):
        raise ValueError("action groups must partition the applicable rules exactly")
    if any(tuple(sorted(group)) != group for group in action_groups):
        raise ValueError("rule ids within an action group must be sorted")


def aggregate_action_group_logits(
    rule_logits: torch.Tensor,
    action_groups: tuple[ActionGroups, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Log-sum-exp aggregate rule logits into padded dynamic group logits."""

    if rule_logits.ndim != 2 or rule_logits.shape[1] != _NUM_RULES:
        raise ValueError("rule logits must have shape [batch, 8]")
    if len(action_groups) != rule_logits.shape[0]:
        raise ValueError("one action-group mapping is required per batch row")
    if not action_groups or any(not groups for groups in action_groups):
        raise ValueError("every decision must have at least one action group")

    max_groups = max(len(groups) for groups in action_groups)
    rows: list[torch.Tensor] = []
    masks: list[list[bool]] = []
    for row, groups in enumerate(action_groups):
        values = [
            torch.logsumexp(
                rule_logits[row, torch.tensor(group, device=rule_logits.device)],
                dim=0,
            )
            for group in groups
        ]
        if len(values) < max_groups:
            values.extend(
                rule_logits.new_zeros(()) for _ in range(max_groups - len(values))
            )
        rows.append(torch.stack(values))
        masks.append([True] * len(groups) + [False] * (max_groups - len(groups)))
    return torch.stack(rows), torch.tensor(
        masks, dtype=torch.bool, device=rule_logits.device
    )


def _mean_valid_logit_std(
    logits: torch.Tensor, action_mask: torch.Tensor
) -> torch.Tensor:
    per_state = tuple(
        logits[row][action_mask[row]].std(unbiased=False)
        for row in range(logits.shape[0])
    )
    return torch.stack(per_state).mean()


def _principle_risk_regime(observation: CADecisionObservation) -> str:
    """Coarse, deterministic regime label used only for diagnostics."""

    rows = observation.theory_factor_values
    if not rows:
        return "no_candidates"
    means = tuple(fmean(row[index] for row in rows) for index in range(5))
    labels = ("capacity", "forecast", "urgency", "service_value", "tardiness")
    strongest = max(range(5), key=means.__getitem__)
    return f"high_{labels[strongest]}" if means[strongest] >= 0.5 else "mixed_low"


class CAPPOAgent:
    def __init__(
        self,
        config: CAPPOConfig = CAPPOConfig(),
        *,
        principle_library: PrincipleLibrary | None = None,
    ) -> None:
        self.config = config
        self.device = torch.device(config.device)
        torch.manual_seed(config.seed)
        if config.actor_type == "principle_selector":
            self.principle_library = principle_library or PrincipleLibrary.load(
                config.principle_library_path or ""
            )
        else:
            if principle_library is not None:
                raise ValueError("a principle library is only valid for principle_selector")
            self.principle_library = None
        if config.actor_critic_mode == "separate":
            self.network = SeparateCheckpointActorCritic(
                config.shop_hidden,
                config.checkpoint_hidden,
                config.fused_hidden,
                config.policy_init,
            ).to(self.device)
        else:
            self.network = CheckpointAttentionActorCritic(
                config.shop_hidden,
                config.checkpoint_hidden,
                config.fused_hidden,
                config.encoder_type,
                config.policy_init,
                config.actor_type,
                config.use_context_gate,
                config.use_interactions,
                len(self.principle_library) if self.principle_library is not None else None,
            ).to(self.device)
        if config.policy_init == "r5_prior":
            with torch.no_grad():
                self.network.actor.bias[
                    int(RuleAction.SERVICE_VALUE_DENSITY)
                ].add_(math.log(7.0))
        if config.actor_critic_mode == "separate":
            assert isinstance(self.network, SeparateCheckpointActorCritic)
            self.actor_optimizer = torch.optim.Adam(
                self.network.actor_parameters(), lr=config.actor_learning_rate
            )
            self.critic_optimizer = torch.optim.Adam(
                self.network.critic_parameters(), lr=config.critic_learning_rate
            )
        else:
            self.optimizer = torch.optim.Adam(
                self.network.parameters(), lr=config.learning_rate
            )
        self.value_normalizer = RunningValueNormalizer(
            epsilon=config.return_normalization_epsilon,
            clip=config.return_normalization_clip,
        )

    @property
    def learning_rate(self) -> float:
        if self.config.actor_critic_mode == "separate":
            return float(self.actor_optimizer.param_groups[0]["lr"])
        return float(self.optimizer.param_groups[0]["lr"])

    def evaluate(self, observation: CADecisionObservation):
        (
            shop,
            tokens,
            mask,
            _action_mask,
            action_tokens,
            rule_tokens,
            candidate_tokens,
            candidate_mask,
        ) = _batch_observations((observation,), self.device)
        relational_inputs = (
            _batch_relational_observations((observation,), self.device)
            if self.config.encoder_type == "candidate_checkpoint"
            else (None, None, None, None)
        )
        theory_factors = (
            _batch_theory_factor_values((observation,), self.device)
            if self.config.actor_type == "factorized_priority"
            else None
        )
        return self.network(
            shop,
            tokens,
            mask,
            action_tokens,
            candidate_tokens,
            candidate_mask,
            rule_tokens,
            relational_candidate_tokens=relational_inputs[0],
            relational_checkpoint_tokens=relational_inputs[1],
            candidate_checkpoint_relations=relational_inputs[2],
            candidate_checkpoint_mask=relational_inputs[3],
            theory_factor_values=theory_factors,
        )

    def encode_candidate_checkpoints(
        self, observation: CADecisionObservation
    ) -> RelationalEncoderOutput:
        """Return candidate embeddings, global context, state, and attention."""

        if self.config.encoder_type != "candidate_checkpoint":
            raise RuntimeError(
                "candidate-checkpoint representations require encoder_type='candidate_checkpoint'"
            )
        assert isinstance(self.network, CheckpointAttentionActorCritic)
        shop, *_unused, candidate_mask = _batch_observations(
            (observation,), self.device
        )
        relational_inputs = _batch_relational_observations(
            (observation,), self.device
        )
        return self.network.encode_candidate_checkpoints(
            shop,
            relational_inputs[0],
            relational_inputs[1],
            relational_inputs[2],
            candidate_mask,
            relational_inputs[3],
        )

    def evaluate_factorized_priority(
        self, observation: CADecisionObservation
    ) -> FactorizedPriorityOutput:
        """Return factors, contributions, interactions, gates, and priorities."""

        if self.config.actor_type != "factorized_priority":
            raise RuntimeError(
                "factorized output requires actor_type='factorized_priority'"
            )
        assert isinstance(self.network, CheckpointAttentionActorCritic)
        _shop, *_unused, candidate_mask = _batch_observations(
            (observation,), self.device
        )
        encoder_output = self.encode_candidate_checkpoints(observation)
        factors = _batch_theory_factor_values((observation,), self.device)
        return self.network.factorized_priority(
            encoder_output, factors, candidate_mask
        )

    def select_action(
        self,
        observation: CADecisionObservation,
        *,
        deterministic: bool = False,
    ) -> tuple[RuleAction | int, float, float]:
        selection = self._select_action_details(
            observation, deterministic=deterministic
        )
        return selection.action, selection.log_probability, selection.value

    def _select_action_details(
        self,
        observation: CADecisionObservation,
        *,
        deterministic: bool = False,
    ) -> _PolicySelection:
        with torch.no_grad():
            rule_logits, normalized_value, _aux, _attention = self.evaluate(observation)
            value = self.value_normalizer.denormalize(normalized_value)
            if self.config.actor_type == "factorized_priority":
                candidate_mask = torch.ones_like(rule_logits, dtype=torch.bool)
                masked_logits = _masked_logits(rule_logits, candidate_mask)
                distribution = Categorical(logits=masked_logits)
                candidate_index = (
                    masked_logits.argmax(dim=-1)
                    if deterministic
                    else distribution.sample()
                )
                log_probability = distribution.log_prob(candidate_index)
                selected_index = int(candidate_index.item())
                return _PolicySelection(
                    action=selected_index,
                    action_groups=(),
                    action_group_index=selected_index,
                    log_probability=float(log_probability.item()),
                    value=float(value.item()),
                    probabilities=tuple(
                        float(item) for item in distribution.probs[0].cpu().tolist()
                    ),
                )
            if self.config.actor_type == "principle_selector":
                principle_mask = torch.ones_like(rule_logits, dtype=torch.bool)
                masked_logits = _masked_logits(rule_logits, principle_mask)
                distribution = Categorical(logits=masked_logits)
                principle_index = (
                    masked_logits.argmax(dim=-1)
                    if deterministic
                    else distribution.sample()
                )
                log_probability = distribution.log_prob(principle_index)
                selected_index = int(principle_index.item())
                return _PolicySelection(
                    action=selected_index,
                    action_groups=(),
                    action_group_index=selected_index,
                    log_probability=float(log_probability.item()),
                    value=float(value.item()),
                    probabilities=tuple(
                        float(item) for item in distribution.probs[0].cpu().tolist()
                    ),
                )
            action_groups = build_action_groups(
                observation, self.config.action_semantics
            )
            _validate_action_groups(action_groups, observation.action_mask)
            group_logits, group_mask = aggregate_action_group_logits(
                rule_logits, (action_groups,)
            )
            group_logits = _masked_logits(group_logits, group_mask)
            distribution = Categorical(logits=group_logits)
            group_index = (
                group_logits.argmax(dim=-1)
                if deterministic
                else distribution.sample()
            )
            log_probability = distribution.log_prob(group_index)
            selected_group_index = int(group_index.item())
            representative_rule = action_groups[selected_group_index][0]
        return _PolicySelection(
            action=RuleAction(representative_rule),
            action_groups=action_groups,
            action_group_index=selected_group_index,
            log_probability=float(log_probability.item()),
            value=float(value.item()),
            probabilities=tuple(
                float(item) for item in distribution.probs[0].cpu().tolist()
            ),
        )

    def update(self, records: list[_Record] | tuple[_Record, ...]) -> PPOUpdateStats:
        if not records:
            raise ValueError("PPO update requires at least one transition")
        observations = tuple(record.observation for record in records)
        (
            shop,
            tokens,
            mask,
            _observation_action_mask,
            action_tokens,
            rule_tokens,
            candidate_tokens,
            candidate_mask,
        ) = _batch_observations(observations, self.device)
        relational_inputs = (
            _batch_relational_observations(observations, self.device)
            if self.config.encoder_type == "candidate_checkpoint"
            else (None, None, None, None)
        )
        theory_factors = (
            _batch_theory_factor_values(observations, self.device)
            if self.config.actor_type == "factorized_priority"
            else None
        )
        action_groups = tuple(record.action_groups for record in records)
        if self.config.actor_type in {"factorized_priority", "principle_selector"}:
            for record in records:
                if not 0 <= record.action < len(record.action_mask):
                    raise ValueError("stored direct-policy index is out of range")
                if not record.action_mask[record.action]:
                    raise ValueError("stored direct-policy action must be valid")
        else:
            for record in records:
                _validate_action_groups(record.action_groups, record.action_mask)
                if not 0 <= record.action_group_index < len(record.action_groups):
                    raise ValueError("stored action-group index is out of range")
                if record.action != record.action_groups[record.action_group_index][0]:
                    raise ValueError(
                        "stored action must be its group's representative rule"
                    )
        policy_actions = torch.tensor(
            [
                record.action
                if self.config.actor_type in {"factorized_priority", "principle_selector"}
                else record.action_group_index
                for record in records
            ],
            dtype=torch.long,
            device=self.device,
        )
        old_log_probs = torch.tensor(
            [r.old_log_probability for r in records], dtype=torch.float32, device=self.device
        )
        rewards = torch.tensor(
            [r.reward / self.config.return_scale for r in records],
            dtype=torch.float32,
            device=self.device,
        )
        old_values = torch.tensor([r.value for r in records], dtype=torch.float32, device=self.device)
        terminated = torch.tensor([r.terminated for r in records], dtype=torch.bool, device=self.device)
        advantages, returns = generalized_advantages(
            rewards,
            old_values,
            terminated,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
        )
        self.value_normalizer.update(returns)
        normalized_returns = self.value_normalizer.normalize(returns)
        advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-8)
        auxiliary_targets = torch.zeros_like(mask, dtype=torch.float32)
        for row, record in enumerate(records):
            if record.auxiliary_targets:
                auxiliary_targets[row, : len(record.auxiliary_targets)] = torch.tensor(
                    record.auxiliary_targets, dtype=torch.float32, device=self.device
                )

        diagnostics: list[tuple[float, ...]] = []
        for _ in range(self.config.update_epochs):
            rule_logits, values, predictions, _attention = self.network(
                shop,
                tokens,
                mask,
                action_tokens,
                candidate_tokens,
                candidate_mask,
                rule_tokens,
                relational_candidate_tokens=relational_inputs[0],
                relational_checkpoint_tokens=relational_inputs[1],
                candidate_checkpoint_relations=relational_inputs[2],
                candidate_checkpoint_mask=relational_inputs[3],
                theory_factor_values=theory_factors,
            )
            if self.config.actor_type == "factorized_priority":
                policy_mask = candidate_mask
                policy_logits = _masked_logits(rule_logits, policy_mask)
            elif self.config.actor_type == "principle_selector":
                policy_mask = torch.ones_like(rule_logits, dtype=torch.bool)
                policy_logits = _masked_logits(rule_logits, policy_mask)
            else:
                policy_logits, policy_mask = aggregate_action_group_logits(
                    rule_logits, action_groups
                )
                policy_logits = _masked_logits(policy_logits, policy_mask)
            distribution = Categorical(logits=policy_logits)
            log_probs = distribution.log_prob(policy_actions)
            ratio = torch.exp(log_probs - old_log_probs)
            unclipped = ratio * advantages
            clipped = torch.clamp(
                ratio,
                1.0 - self.config.clip_epsilon,
                1.0 + self.config.clip_epsilon,
            ) * advantages
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            value_loss = F.mse_loss(values, normalized_returns)
            entropy = distribution.entropy().mean()
            auxiliary_loss = (
                F.smooth_l1_loss(predictions[mask], auxiliary_targets[mask])
                if mask.any()
                else torch.zeros((), device=self.device)
            )
            loss = (
                policy_loss
                + self.config.value_coefficient * value_loss
                - self.config.entropy_coefficient * entropy
                + self.config.auxiliary_coefficient * auxiliary_loss
            )
            if self.config.actor_critic_mode == "separate":
                assert isinstance(self.network, SeparateCheckpointActorCritic)
                actor_parameters = tuple(self.network.actor_parameters())
                critic_parameters = tuple(self.network.critic_parameters())
                actor_loss = policy_loss - self.config.entropy_coefficient * entropy
                critic_loss = (
                    self.config.value_coefficient * value_loss
                    + self.config.auxiliary_coefficient * auxiliary_loss
                )
                self.actor_optimizer.zero_grad(set_to_none=True)
                self.critic_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                critic_loss.backward()
                actor_gradient_norm = torch.nn.utils.clip_grad_norm_(
                    actor_parameters, self.config.gradient_clip_norm
                )
                critic_gradient_norm = torch.nn.utils.clip_grad_norm_(
                    critic_parameters, self.config.gradient_clip_norm
                )
                global_gradient_norm = torch.sqrt(
                    actor_gradient_norm.square() + critic_gradient_norm.square()
                )
                self.actor_optimizer.step()
                self.critic_optimizer.step()
            else:
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                global_gradient_norm = torch.nn.utils.clip_grad_norm_(
                    self.network.parameters(), self.config.gradient_clip_norm
                )
                self.optimizer.step()
            with torch.no_grad():
                approximate_kl = (old_log_probs - log_probs).mean()
                clip_fraction = (
                    (ratio - 1.0).abs() > self.config.clip_epsilon
                ).float().mean()
                unnormalized_values = self.value_normalizer.denormalize(values)
                return_variance = returns.var(unbiased=False)
                explained_variance = (
                    1.0
                    - (returns - unnormalized_values).var(unbiased=False)
                    / return_variance
                    if return_variance > 1e-12
                    else torch.zeros((), device=self.device)
                )
                mean_max_action_prob = distribution.probs.max(dim=1).values.mean()
                logit_std = _mean_valid_logit_std(policy_logits, policy_mask)
                ratio_std = ratio.std(unbiased=False)
            diagnostics.append(
                tuple(
                    float(item.detach().cpu().item())
                    for item in (
                        loss,
                        policy_loss,
                        value_loss,
                        auxiliary_loss,
                        entropy,
                        approximate_kl,
                        clip_fraction,
                        explained_variance,
                        global_gradient_norm,
                        mean_max_action_prob,
                        logit_std,
                        ratio_std,
                    )
                )
            )
        means = tuple(fmean(row[index] for row in diagnostics) for index in range(12))
        return PPOUpdateStats(
            *means,
            mean_advantage=float(advantages.mean().detach().cpu().item()),
            advantage_std=float(advantages.std(unbiased=False).detach().cpu().item()),
            transitions=len(records),
        )

    def recompute_stored_log_probabilities(
        self, records: list[_Record] | tuple[_Record, ...]
    ) -> tuple[float, ...]:
        """Re-evaluate rollout actions with their immutable stored masks."""

        if not records:
            return ()
        observations = tuple(record.observation for record in records)
        (
            shop,
            tokens,
            checkpoint_mask,
            _observation_mask,
            action_tokens,
            rule_tokens,
            candidate_tokens,
            candidate_mask,
        ) = _batch_observations(observations, self.device)
        relational_inputs = (
            _batch_relational_observations(observations, self.device)
            if self.config.encoder_type == "candidate_checkpoint"
            else (None, None, None, None)
        )
        theory_factors = (
            _batch_theory_factor_values(observations, self.device)
            if self.config.actor_type == "factorized_priority"
            else None
        )
        action_groups = tuple(record.action_groups for record in records)
        if self.config.actor_type == "rule_selector":
            for record in records:
                _validate_action_groups(record.action_groups, record.action_mask)
        policy_actions = torch.tensor(
            [
                record.action
                if self.config.actor_type in {"factorized_priority", "principle_selector"}
                else record.action_group_index
                for record in records
            ],
            dtype=torch.long,
            device=self.device,
        )
        with torch.no_grad():
            rule_logits, _values, _predictions, _attention = self.network(
                shop,
                tokens,
                checkpoint_mask,
                action_tokens,
                candidate_tokens,
                candidate_mask,
                rule_tokens,
                relational_candidate_tokens=relational_inputs[0],
                relational_checkpoint_tokens=relational_inputs[1],
                candidate_checkpoint_relations=relational_inputs[2],
                candidate_checkpoint_mask=relational_inputs[3],
                theory_factor_values=theory_factors,
            )
            if self.config.actor_type == "factorized_priority":
                policy_logits = _masked_logits(rule_logits, candidate_mask)
            elif self.config.actor_type == "principle_selector":
                policy_logits = _masked_logits(
                    rule_logits, torch.ones_like(rule_logits, dtype=torch.bool)
                )
            else:
                policy_logits, policy_mask = aggregate_action_group_logits(
                    rule_logits, action_groups
                )
                policy_logits = _masked_logits(policy_logits, policy_mask)
            distribution = Categorical(logits=policy_logits)
            log_probabilities = distribution.log_prob(policy_actions)
        return tuple(float(value) for value in log_probabilities.cpu().tolist())

    def collect_episode(self, env: CAPPORuleSelectionEnv) -> CAPPOCollectedEpisode:
        """Collect one complete episode without changing network parameters."""

        observation = env.reset()
        records: list[_Record] = []
        total_reward = 0.0
        while True:
            selection = self._select_action_details(observation)
            if self.config.actor_type == "factorized_priority":
                step = env.step_candidate(int(selection.action))
            elif self.config.actor_type == "principle_selector":
                assert self.principle_library is not None
                principle_index = int(selection.action)
                step = env.step_principle(
                    principle_index, self.principle_library[principle_index]
                )
            else:
                step = env.step(selection.action)
            stored_action_mask = (
                (True,) * len(observation.candidate_job_ids)
                if self.config.actor_type == "factorized_priority"
                else (
                    (True,) * len(self.principle_library)
                    if self.config.actor_type == "principle_selector"
                    and self.principle_library is not None
                    else observation.action_mask
                )
            )
            records.append(
                _Record(
                    observation=observation,
                    action=int(selection.action),
                    action_mask=stored_action_mask,
                    action_groups=selection.action_groups,
                    action_group_index=selection.action_group_index,
                    old_log_probability=selection.log_probability,
                    value=selection.value,
                    reward=step.reward,
                    terminated=step.terminated,
                    policy_probabilities=selection.probabilities,
                    executed_candidate_index=(
                        step.decision.candidate_index
                        if self.config.actor_type == "principle_selector"
                        and hasattr(step.decision, "candidate_index")
                        else None
                    ),
                    risk_regime=(
                        _principle_risk_regime(observation)
                        if self.config.actor_type == "principle_selector"
                        else None
                    ),
                )
            )
            total_reward += step.reward
            if step.terminated:
                if step.objective is None:
                    raise RuntimeError("terminal CA-PPO step has no objective")
                objective = step.objective
                break
            assert step.observation is not None
            observation = step.observation
        for record in records:
            record.auxiliary_targets = tuple(
                objective.per_checkpoint_shortfall[key]
                / env.instance.get_entity(key[0]).required_quantity(key[1])
                for key in record.observation.checkpoint_keys
            )
        return CAPPOCollectedEpisode(tuple(records), total_reward, objective)

    def principle_diagnostics(
        self, records: list[_Record] | tuple[_Record, ...]
    ) -> PrincipleDiagnostics:
        """Summarize probabilities, choices, regimes, candidates, and switching."""

        if self.config.actor_type != "principle_selector" or self.principle_library is None:
            raise RuntimeError("principle diagnostics require actor_type='principle_selector'")
        count = len(self.principle_library)
        selection_counts = [0] * count
        probability_sums = [0.0] * count
        regime_counts: dict[str, list[int]] = {}
        candidate_counts: dict[int, int] = {}
        switches = 0
        comparable = 0
        previous: _Record | None = None
        for record in records:
            selection_counts[record.action] += 1
            for index, probability in enumerate(record.policy_probabilities):
                probability_sums[index] += probability
            regime = record.risk_regime or "unknown"
            regime_counts.setdefault(regime, [0] * count)[record.action] += 1
            if record.executed_candidate_index is not None:
                candidate_counts[record.executed_candidate_index] = (
                    candidate_counts.get(record.executed_candidate_index, 0) + 1
                )
            if previous is not None and record.observation.decision_index != 0:
                comparable += 1
                switches += int(previous.action != record.action)
            previous = record
        total = max(1, len(records))
        return PrincipleDiagnostics(
            principle_names=tuple(item.name for item in self.principle_library.principles),
            selection_counts=tuple(selection_counts),
            selection_fractions=tuple(value / total for value in selection_counts),
            mean_selection_probabilities=tuple(value / total for value in probability_sums),
            regime_selection_fractions={
                regime: tuple(value / max(1, sum(values)) for value in values)
                for regime, values in sorted(regime_counts.items())
            },
            candidate_selection_counts=dict(sorted(candidate_counts.items())),
            switching_frequency=switches / max(1, comparable),
        )

    def save_principle_diagnostics(
        self,
        records: list[_Record] | tuple[_Record, ...],
        path: str | Path,
    ) -> Path:
        """Persist aggregate and transition-level evidence of online switching."""

        summary = self.principle_diagnostics(records)
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(
                {
                    "summary": asdict(summary),
                    "transitions": [
                        {
                            "decision_index": record.observation.decision_index,
                            "physical_time": record.observation.physical_time,
                            "principle_index": record.action,
                            "selection_probabilities": list(record.policy_probabilities),
                            "executed_candidate_index": record.executed_candidate_index,
                            "risk_regime": record.risk_regime,
                        }
                        for record in records
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return target

    def train_episode(self, env: CAPPORuleSelectionEnv) -> CAPPOEpisodeResult:
        episode = self.collect_episode(env)
        stats = self.update(episode.records)
        return CAPPOEpisodeResult(
            episode.decision_count,
            episode.total_reward,
            episode.objective,
            stats.total_loss,
            stats,
        )

    def evaluate_episode(self, env: CAPPORuleSelectionEnv) -> CAPPOEpisodeResult:
        observation = env.reset()
        total_reward = 0.0
        count = 0
        while True:
            action, _log_probability, _value = self.select_action(
                observation, deterministic=True
            )
            if self.config.actor_type == "factorized_priority":
                step = env.step_candidate(int(action))
            elif self.config.actor_type == "principle_selector":
                assert self.principle_library is not None
                step = env.step_principle(int(action), self.principle_library[int(action)])
            else:
                step = env.step(action)
            total_reward += step.reward
            count += 1
            if step.terminated:
                assert step.objective is not None
                return CAPPOEpisodeResult(count, total_reward, step.objective, None)
            assert step.observation is not None
            observation = step.observation

    def save_checkpoint(self, path: str | Path, *, metadata: dict | None = None) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": asdict(self.config),
            "network": self.network.state_dict(),
            "value_normalizer": self.value_normalizer.state_dict(),
            "metadata": dict(metadata or {}),
            "principle_library": (
                self.principle_library.to_dict()
                if self.principle_library is not None
                else None
            ),
        }
        if self.config.actor_critic_mode == "separate":
            payload["actor_optimizer"] = self.actor_optimizer.state_dict()
            payload["critic_optimizer"] = self.critic_optimizer.state_dict()
        else:
            payload["optimizer"] = self.optimizer.state_dict()
        torch.save(payload, target)
        return target

    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
        *,
        device: str | None = None,
    ) -> tuple["CAPPOAgent", dict]:
        payload = torch.load(Path(path), map_location=device or "cpu", weights_only=False)
        raw_config = dict(payload["config"])
        if device is not None:
            raw_config["device"] = device
        legacy_clips = tuple(
            raw_config.pop(name, None)
            for name in (
                "actor_gradient_clip_norm",
                "critic_gradient_clip_norm",
                "auxiliary_gradient_clip_norm",
            )
        )
        if "gradient_clip_norm" not in raw_config:
            raw_config["gradient_clip_norm"] = next(
                (value for value in legacy_clips if value is not None), 0.5
            )
        raw_config.pop("actor_learning_rate_final", None)
        embedded_library = payload.get("principle_library")
        agent = cls(
            CAPPOConfig(**raw_config),
            principle_library=(
                PrincipleLibrary.from_dict(embedded_library)
                if embedded_library is not None
                else None
            ),
        )
        network_state = dict(payload["network"])
        if "policy_prior_logits" not in network_state:
            network_state["policy_prior_logits"] = (
                agent.network.policy_prior_logits.detach().clone()
            )
        agent.network.load_state_dict(network_state)
        if agent.config.actor_critic_mode == "separate":
            actor_optimizer_state = payload.get("actor_optimizer")
            critic_optimizer_state = payload.get("critic_optimizer")
            if actor_optimizer_state is not None:
                agent.actor_optimizer.load_state_dict(actor_optimizer_state)
            if critic_optimizer_state is not None:
                agent.critic_optimizer.load_state_dict(critic_optimizer_state)
        else:
            optimizer_state = payload.get("optimizer") or payload.get(
                "critic_aux_optimizer"
            )
            if optimizer_state is not None:
                agent.optimizer.load_state_dict(optimizer_state)
        agent.value_normalizer.load_state_dict(payload.get("value_normalizer", {}))
        return agent, dict(payload.get("metadata", {}))
