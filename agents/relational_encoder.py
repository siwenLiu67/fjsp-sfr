"""Lightweight candidate-to-checkpoint relational attention state encoder."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class RelationalEncoderOutput:
    """Candidate-level and fixed-dimensional outputs of the relational encoder."""

    candidate_embeddings: torch.Tensor
    global_context: torch.Tensor
    state_embedding: torch.Tensor
    attention_weights: torch.Tensor


class CandidateCheckpointRelationalAttentionEncoder(nn.Module):
    """Global context plus 4-head candidate-to-checkpoint attention.

    Checkpoints do not attend to each other.  Relations affect only attention
    scores; values contain checkpoint information only.
    """

    def __init__(
        self,
        *,
        shop_features: int = 7,
        candidate_features: int = 4,
        checkpoint_features: int = 5,
        relation_features: int = 2,
        model_dim: int = 64,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        if model_dim <= 0 or num_heads <= 0 or model_dim % num_heads:
            raise ValueError("model_dim must be positive and divisible by num_heads")
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.global_encoder = nn.Sequential(
            nn.Linear(shop_features, model_dim),
            nn.Tanh(),
            nn.Linear(model_dim, model_dim),
            nn.Tanh(),
        )
        self.candidate_encoder = nn.Sequential(
            nn.Linear(candidate_features + model_dim, model_dim),
            nn.Tanh(),
            nn.Linear(model_dim, model_dim),
            nn.Tanh(),
        )
        self.checkpoint_encoder = nn.Sequential(
            nn.Linear(checkpoint_features, model_dim),
            nn.Tanh(),
            nn.Linear(model_dim, model_dim),
            nn.Tanh(),
        )
        self.query = nn.Linear(model_dim, model_dim, bias=False)
        self.key = nn.Linear(model_dim, model_dim, bias=False)
        self.value = nn.Linear(model_dim, model_dim, bias=False)
        self.relation_bias = nn.Sequential(
            nn.Linear(relation_features, 16),
            nn.Tanh(),
            nn.Linear(16, num_heads, bias=False),
        )
        self.output_projection = nn.Linear(model_dim, model_dim, bias=False)
        self.final_encoder = nn.Sequential(
            nn.Linear(2 * model_dim, model_dim),
            nn.Tanh(),
            nn.Linear(model_dim, model_dim),
            nn.Tanh(),
        )

    def forward(
        self,
        shop: torch.Tensor,
        candidate_tokens: torch.Tensor,
        checkpoint_tokens: torch.Tensor,
        relations: torch.Tensor,
        candidate_mask: torch.Tensor,
        candidate_checkpoint_mask: torch.Tensor,
    ) -> RelationalEncoderOutput:
        """Encode a padded batch.

        Shapes are ``shop [B,7]``, candidates ``[B,C,4]``, checkpoints
        ``[B,K,5]``, relations ``[B,C,K,2]``, and boolean masks ``[B,C]`` and
        ``[B,C,K]``.
        """

        if shop.ndim != 2:
            raise ValueError("shop must have shape [batch, shop_features]")
        if candidate_tokens.ndim != 3 or checkpoint_tokens.ndim != 3:
            raise ValueError("candidate and checkpoint tokens must be rank three")
        batch_size, candidate_count, _ = candidate_tokens.shape
        checkpoint_count = checkpoint_tokens.shape[1]
        if checkpoint_tokens.shape[0] != batch_size:
            raise ValueError("candidate and checkpoint batches must align")
        if relations.shape[:3] != (batch_size, candidate_count, checkpoint_count):
            raise ValueError("relations must have shape [batch, candidates, checkpoints, 2]")
        if candidate_mask.shape != (batch_size, candidate_count):
            raise ValueError("candidate_mask must have shape [batch, candidates]")
        if candidate_checkpoint_mask.shape != (
            batch_size,
            candidate_count,
            checkpoint_count,
        ):
            raise ValueError(
                "candidate_checkpoint_mask must have shape [batch, candidates, checkpoints]"
            )

        global_context = self.global_encoder(shop)
        expanded_global = global_context.unsqueeze(1).expand(
            -1, candidate_count, -1
        )
        candidate_embedding = self.candidate_encoder(
            torch.cat((candidate_tokens, expanded_global), dim=-1)
        )
        valid_candidates = candidate_mask.unsqueeze(-1).to(candidate_embedding.dtype)

        if checkpoint_count == 0:
            checkpoint_context = candidate_embedding.new_zeros(
                (batch_size, candidate_count, self.model_dim)
            )
            attention_weights = candidate_embedding.new_zeros(
                (batch_size, candidate_count, self.num_heads, 0)
            )
        else:
            checkpoint_embedding = self.checkpoint_encoder(checkpoint_tokens)
            queries = self.query(candidate_embedding).view(
                batch_size, candidate_count, self.num_heads, self.head_dim
            )
            keys = self.key(checkpoint_embedding).view(
                batch_size, checkpoint_count, self.num_heads, self.head_dim
            )
            values = self.value(checkpoint_embedding).view(
                batch_size, checkpoint_count, self.num_heads, self.head_dim
            )
            content_scores = torch.einsum("bchd,bkhd->bhck", queries, keys)
            content_scores = content_scores / math.sqrt(self.head_dim)
            relation_scores = self.relation_bias(relations).permute(0, 3, 1, 2)
            scores = content_scores + relation_scores

            valid_pairs = candidate_checkpoint_mask & candidate_mask.unsqueeze(-1)
            head_mask = valid_pairs.unsqueeze(1)
            scores = scores.masked_fill(
                ~head_mask, torch.finfo(scores.dtype).min
            )
            weights = torch.softmax(scores, dim=-1) * head_mask.to(scores.dtype)
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            context_heads = torch.einsum("bhck,bkhd->bchd", weights, values)
            checkpoint_context = self.output_projection(
                context_heads.reshape(batch_size, candidate_count, self.model_dim)
            )
            attention_weights = weights.permute(0, 2, 1, 3)

        candidate_embeddings = self.final_encoder(
            torch.cat((candidate_embedding, checkpoint_context), dim=-1)
        )
        candidate_embeddings = candidate_embeddings * valid_candidates
        pooled_candidates = candidate_embeddings.sum(dim=1) / candidate_mask.sum(
            dim=1, keepdim=True
        ).clamp_min(1).to(candidate_embeddings.dtype)
        state_embedding = torch.cat((global_context, pooled_candidates), dim=-1)
        return RelationalEncoderOutput(
            candidate_embeddings=candidate_embeddings,
            global_context=global_context,
            state_embedding=state_embedding,
            attention_weights=attention_weights,
        )
