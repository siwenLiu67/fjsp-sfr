"""Training and evaluation protocol for forecast-aware CA-PPO."""

from .ca_training import (
    CAEpisodeRecord,
    CAEvaluationSummary,
    CATrainingRecord,
    evaluate_ca_ppo,
    moving_average_objectives,
    train_ca_ppo,
)
from .counterfactual_rule_value_audit import (
    collect_fixed_r5_states,
    run_diagnostic_one_step_oracle,
    summarize_counterfactual_records,
)

__all__ = [
    "CAEpisodeRecord",
    "CAEvaluationSummary",
    "CATrainingRecord",
    "evaluate_ca_ppo",
    "moving_average_objectives",
    "train_ca_ppo",
    "collect_fixed_r5_states",
    "run_diagnostic_one_step_oracle",
    "summarize_counterfactual_records",
]
