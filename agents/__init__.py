"""Forecast-aware checkpoint-attention PPO agent."""

from .ca_ppo import (
    CAPPOAgent,
    CAPPOCollectedEpisode,
    CAPPOConfig,
    CAPPOEpisodeResult,
    CheckpointAttentionActorCritic,
    PPOUpdateStats,
    PrincipleDiagnostics,
    RunningValueNormalizer,
    generalized_advantages,
)
from .relational_encoder import (
    CandidateCheckpointRelationalAttentionEncoder,
    RelationalEncoderOutput,
)
from .factorized_priority_actor import (
    FactorizedPriorityOutput,
    LearnableBilinearSurface2D,
    LearnablePiecewiseLinear1D,
    TheoryGuidedFactorizedPriorityActor,
)
from .principle_library import (
    FACTOR_SYMBOLS,
    PrincipleLibrary,
    PrincipleSpec,
    PrincipleValidationError,
    canonical_expression,
    parse_expression,
)

__all__ = [
    "CAPPOAgent",
    "CAPPOCollectedEpisode",
    "CAPPOConfig",
    "CAPPOEpisodeResult",
    "CheckpointAttentionActorCritic",
    "CandidateCheckpointRelationalAttentionEncoder",
    "FactorizedPriorityOutput",
    "LearnableBilinearSurface2D",
    "LearnablePiecewiseLinear1D",
    "PPOUpdateStats",
    "PrincipleDiagnostics",
    "PrincipleLibrary",
    "PrincipleSpec",
    "PrincipleValidationError",
    "RunningValueNormalizer",
    "RelationalEncoderOutput",
    "TheoryGuidedFactorizedPriorityActor",
    "FACTOR_SYMBOLS",
    "canonical_expression",
    "generalized_advantages",
    "parse_expression",
]
