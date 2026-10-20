"""Machine-readable contract supplied to an offline principle generator."""

from __future__ import annotations

import json
from pathlib import Path

from fjsp_sfr.agents.principle_library import (
    ALLOWED_CONSTANTS,
    ALLOWED_FUNCTIONS,
    FACTOR_SYMBOLS,
)


def build_theory_spec() -> dict[str, object]:
    return {
        "schema_version": 1,
        "factor_order": list(FACTOR_SYMBOLS),
        "factors": {
            "G": {
                "name": "Gap Pressure / Capacity Risk",
                "direction": "larger means stronger risk",
                "boundary": "normalized current capacity pressure only; range [0,1]",
            },
            "F": {
                "name": "Forecast Deficit / Future Supply Risk",
                "direction": "larger means stronger risk",
                "boundary": "forecast-aware residual supply deficit only; range [0,1]",
            },
            "U": {
                "name": "Checkpoint Slack Urgency",
                "direction": "larger means stronger urgency",
                "boundary": "candidate completion pressure against reachable checkpoint; range [0,1]",
            },
            "M": {
                "name": "Multi-Checkpoint Value",
                "direction": "larger means greater service value",
                "boundary": "weighted value across reachable future checkpoints; range [0,1]",
            },
            "T": {
                "name": "Tardiness Exposure",
                "direction": "larger means stronger tardiness exposure",
                "boundary": "candidate projected tardiness only; range [0,1]",
            },
        },
        "grammar": {
            "variables": list(FACTOR_SYMBOLS),
            "binary_operators": ["+", "-", "*", "/ (safe divide)"],
            "functions": list(ALLOWED_FUNCTIONS),
            "constants": sorted(ALLOWED_CONSTANTS),
            "expression_node_limit": 80,
        },
        "forbidden": [
            "MLP, embedding, learned residual, or hidden state",
            "candidate embedding or any variable outside G,F,U,M,T",
            "power, comparison, boolean, conditional, attribute, indexing, or arbitrary call",
            "redefinition of the five factor formulas",
        ],
        "diversity_requirements": [
            "vary factor subsets and interactions",
            "include capacity, forecast, urgency, service-value, and tardiness tradeoffs",
            "avoid algebraic and behavioral near-duplicates",
            "provide a theory rationale for every expression",
        ],
        "requested_candidate_count": {"minimum": 30, "maximum": 60},
        "output_contract": {
            "schema_version": 1,
            "principles": [
                {
                    "name": "unique_snake_case_name",
                    "expression": "symbolic expression",
                    "rationale": "theory-grounded explanation",
                    "factors_used": ["G", "U"],
                    "source": "llm",
                }
            ],
        },
    }


def write_theory_spec(path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(build_theory_spec(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return target
