"""Validate and normalize JSON returned by an offline LLM."""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from pathlib import Path

from fjsp_sfr.agents.principle_library import (
    PrincipleLibrary,
    PrincipleSpec,
    PrincipleValidationError,
    canonical_expression,
)


@dataclass(frozen=True)
class ImportReport:
    accepted: int
    rejected: tuple[dict[str, object], ...]
    duplicate_names: tuple[str, ...]
    duplicate_expressions: tuple[str, ...]


def _sanity_rows() -> tuple[tuple[float, ...], ...]:
    anchors = (0.0, 0.25, 0.5, 1.0)
    return tuple(itertools.product(anchors, repeat=5))


def import_generated_principles(
    input_path: str | Path,
    output_path: str | Path,
    *,
    report_path: str | Path | None = None,
    minimum_count: int = 1,
) -> tuple[PrincipleLibrary, ImportReport]:
    payload = json.loads(Path(input_path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != {"schema_version", "principles"}:
        raise PrincipleValidationError(
            "LLM result must contain exactly schema_version and principles"
        )
    if payload["schema_version"] != 1 or not isinstance(payload["principles"], list):
        raise PrincipleValidationError("unsupported schema or non-array principles")

    accepted: list[PrincipleSpec] = []
    rejected: list[dict[str, object]] = []
    duplicate_names: list[str] = []
    duplicate_expressions: list[str] = []
    seen_names: set[str] = set()
    seen_expressions: set[str] = set()
    sanity_rows = _sanity_rows()
    for index, raw in enumerate(payload["principles"]):
        try:
            if not isinstance(raw, dict):
                raise PrincipleValidationError("principle must be a JSON object")
            principle = PrincipleSpec.from_dict(raw)
            if principle.source != "llm":
                raise PrincipleValidationError("imported generated principle source must be 'llm'")
            canonical = canonical_expression(principle.expression)
            if principle.name in seen_names:
                duplicate_names.append(principle.name)
                raise PrincipleValidationError("duplicate name")
            if canonical in seen_expressions:
                duplicate_expressions.append(principle.name)
                raise PrincipleValidationError("duplicate expression")
            scores = principle.evaluate_batch(sanity_rows)
            if not scores or any(abs(score) > 1e6 for score in scores):
                raise PrincipleValidationError("numerical sanity test failed")
            seen_names.add(principle.name)
            seen_expressions.add(canonical)
            accepted.append(principle)
        except (PrincipleValidationError, ValueError, TypeError) as error:
            rejected.append({"index": index, "reason": str(error), "candidate": raw})

    if len(accepted) < minimum_count:
        raise PrincipleValidationError(
            f"only {len(accepted)} valid principles; require at least {minimum_count}"
        )
    library = PrincipleLibrary(tuple(accepted))
    library.save(output_path)
    report = ImportReport(
        accepted=len(accepted),
        rejected=tuple(rejected),
        duplicate_names=tuple(duplicate_names),
        duplicate_expressions=tuple(duplicate_expressions),
    )
    if report_path is not None:
        target = Path(report_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(
                {
                    "accepted": report.accepted,
                    "rejected": list(report.rejected),
                    "duplicate_names": list(report.duplicate_names),
                    "duplicate_expressions": list(report.duplicate_expressions),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return library, report
