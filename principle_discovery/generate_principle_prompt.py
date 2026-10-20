"""Export the reproducible prompt used for manual/offline LLM synthesis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .theory_spec import build_theory_spec, write_theory_spec


def render_prompt() -> str:
    specification = json.dumps(build_theory_spec(), ensure_ascii=False, indent=2)
    return f"""You are generating a diverse library of interpretable scheduling principles.

The following JSON is the complete and binding theory/grammar contract:

{specification}

Generate 30--60 candidate principles. Every principle must obey the grammar,
use only the declared five factors, accurately list factors_used, and include a
concise theory rationale. Favor complementary decision behavior, not cosmetic
rewrites. Return JSON only, exactly matching output_contract. Do not add
Markdown fences, prose, additional keys, learned components, or state fields.
"""


def export_prompt(output: str | Path, theory_output: str | Path | None = None) -> Path:
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_prompt(), encoding="utf-8")
    if theory_output is not None:
        write_theory_spec(theory_output)
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", default="artifacts/principle_generation_prompt.txt"
    )
    parser.add_argument(
        "--theory-output", default="artifacts/principle_theory_spec.json"
    )
    args = parser.parse_args()
    print(export_prompt(args.output, args.theory_output))


if __name__ == "__main__":
    main()
