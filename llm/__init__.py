"""LLM-assisted symbolic-rule generation utilities."""

from .generator import (
    GenerationResult,
    LLMClient,
    compute_behavior_similarity,
    evaluate_rule_objective,
    greedy_library_selection,
    iterative_rule_generation,
)
from .prompt_template import FEATURE_DESCRIPTIONS, build_generation_prompt
from .rule_parser import parse_llm_output

__all__ = [
    "FEATURE_DESCRIPTIONS",
    "GenerationResult",
    "LLMClient",
    "build_generation_prompt",
    "compute_behavior_similarity",
    "evaluate_rule_objective",
    "greedy_library_selection",
    "iterative_rule_generation",
    "parse_llm_output",
]
