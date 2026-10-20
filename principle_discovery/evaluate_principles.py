"""Simulator quality, behavioral fingerprints, and complementary selection."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Iterable, Sequence

from fjsp_sfr.agents.principle_library import PrincipleLibrary, PrincipleSpec
from fjsp_sfr.core.models import FJSPInstance
from fjsp_sfr.env import CAPPORuleSelectionEnv
from fjsp_sfr.generation import InstanceConfig, generate_instance


@dataclass(frozen=True)
class FixedPrincipleResult:
    name: str
    objective: float
    total_tardiness: float
    weighted_shortfall: float
    decisions: int


@dataclass(frozen=True)
class PortfolioSelection:
    selected_indices: tuple[int, ...]
    reasons: tuple[str, ...]
    quality_threshold: float


def selected_candidate_index(
    principle: PrincipleSpec, factor_rows: Sequence[Sequence[float]]
) -> int:
    if not factor_rows:
        raise ValueError("a decision state requires at least one candidate")
    scores = principle.evaluate_batch(factor_rows)
    # Rows follow sorted job_id order.  Python max returns the first maximum,
    # reproducing the project's deterministic lowest-job-id tie break.
    return max(range(len(scores)), key=scores.__getitem__)


def behavioral_fingerprints(
    principles: Sequence[PrincipleSpec],
    state_bank: Sequence[Sequence[Sequence[float]]],
) -> tuple[tuple[int, ...], ...]:
    if not state_bank:
        raise ValueError("state bank must not be empty")
    return tuple(
        tuple(selected_candidate_index(principle, state) for state in state_bank)
        for principle in principles
    )


def agreement_matrix(
    fingerprints: Sequence[Sequence[int]],
) -> tuple[tuple[float, ...], ...]:
    if not fingerprints or not fingerprints[0]:
        raise ValueError("fingerprints must be a non-empty rectangular matrix")
    width = len(fingerprints[0])
    if any(len(row) != width for row in fingerprints):
        raise ValueError("all fingerprints must use the same state bank")
    return tuple(
        tuple(
            sum(left == right for left, right in zip(first, second)) / width
            for second in fingerprints
        )
        for first in fingerprints
    )


def behavioral_distance_matrix(
    fingerprints: Sequence[Sequence[int]],
) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple(1.0 - agreement for agreement in row)
        for row in agreement_matrix(fingerprints)
    )


def evaluate_fixed_principle(
    principle: PrincipleSpec,
    instances: Sequence[FJSPInstance],
    *,
    principle_index: int = 0,
) -> FixedPrincipleResult:
    if not instances:
        raise ValueError("discovery instances must not be empty")
    objectives = []
    tardiness = []
    shortfall = []
    decisions = []
    for instance in instances:
        env = CAPPORuleSelectionEnv(instance)
        env.reset()
        count = 0
        while True:
            step = env.step_principle(principle_index, principle)
            count += 1
            if step.terminated:
                assert step.objective is not None
                objectives.append(step.objective.Z)
                tardiness.append(float(step.objective.total_tardiness))
                shortfall.append(step.objective.weighted_checkpoint_shortfall)
                decisions.append(count)
                break
    return FixedPrincipleResult(
        name=principle.name,
        objective=fmean(objectives),
        total_tardiness=fmean(tardiness),
        weighted_shortfall=fmean(shortfall),
        decisions=round(fmean(decisions)),
    )


def build_discovery_state_bank(
    principles: Sequence[PrincipleSpec],
    instances: Sequence[FJSPInstance],
) -> tuple[tuple[tuple[float, ...], ...], ...]:
    """Collect a shared bank from deterministic fixed-principle trajectories."""

    if not principles or not instances:
        raise ValueError("principles and discovery instances must not be empty")
    bank: list[tuple[tuple[float, ...], ...]] = []
    for principle_index, principle in enumerate(principles):
        for instance in instances:
            env = CAPPORuleSelectionEnv(instance)
            observation = env.reset()
            while True:
                bank.append(observation.theory_factor_values)
                step = env.step_principle(principle_index, principle)
                if step.terminated:
                    break
                assert step.observation is not None
                observation = step.observation
    return tuple(bank)


def select_complementary_portfolio(
    principles: Sequence[PrincipleSpec],
    quality: Sequence[float],
    distances: Sequence[Sequence[float]],
    *,
    target_size: int = 8,
    quality_keep_fraction: float = 0.6,
    diversity_weight: float = 0.7,
) -> PortfolioSelection:
    count = len(principles)
    if count == 0 or len(quality) != count or len(distances) != count:
        raise ValueError("principles, quality, and distance matrix must align")
    if any(len(row) != count for row in distances):
        raise ValueError("distance matrix must be square")
    if not 1 <= target_size <= count:
        raise ValueError("target_size must be within the principle count")
    if not 0.0 < quality_keep_fraction <= 1.0:
        raise ValueError("quality_keep_fraction must be in (0,1]")

    ranked = sorted(range(count), key=lambda index: (quality[index], index))
    retained_count = max(target_size, round(count * quality_keep_fraction))
    eligible = ranked[:retained_count]
    threshold = max(quality[index] for index in eligible)
    q_min = min(quality[index] for index in eligible)
    q_span = max(threshold - q_min, 1e-12)
    selected = [eligible[0]]
    reasons = ["best fixed-principle objective"]
    while len(selected) < target_size:
        choices = []
        for index in eligible:
            if index in selected:
                continue
            diversity = min(distances[index][chosen] for chosen in selected)
            normalized_quality = 1.0 - (quality[index] - q_min) / q_span
            factor_sets = [set(principles[chosen].factors_used) for chosen in selected]
            composition_novelty = min(
                1.0
                - len(set(principles[index].factors_used) & factors)
                / max(1, len(set(principles[index].factors_used) | factors))
                for factors in factor_sets
            )
            merit = (
                diversity_weight * diversity
                + 0.2 * normalized_quality
                + 0.1 * composition_novelty
            )
            choices.append((merit, diversity, normalized_quality, -index, index))
        _merit, diversity, normalized_quality, _negative, winner = max(choices)
        selected.append(winner)
        reasons.append(
            f"greedy complement: min behavioral distance={diversity:.3f}, "
            f"normalized quality={normalized_quality:.3f}"
        )
    return PortfolioSelection(tuple(selected), tuple(reasons), threshold)


def write_selection_artifacts(
    library: PrincipleLibrary,
    fixed_results: Sequence[FixedPrincipleResult],
    fingerprints: Sequence[Sequence[int]],
    selection: PortfolioSelection,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if len(fixed_results) != len(library) or len(fingerprints) != len(library):
        raise ValueError("library and evaluation results must align")
    agreements = agreement_matrix(fingerprints)
    distances = behavioral_distance_matrix(fingerprints)
    selected = PrincipleLibrary(
        tuple(library[index] for index in selection.selected_indices)
    )
    selected_path = selected.save(output / "selected_principles.json")

    for filename, matrix in (
        ("pairwise_agreement.csv", agreements),
        ("pairwise_behavioral_distance.csv", distances),
    ):
        with (output / filename).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["principle", *(item.name for item in library.principles)])
            for principle, row in zip(library.principles, matrix):
                writer.writerow([principle.name, *row])

    selected_reasons = dict(zip(selection.selected_indices, selection.reasons))
    report_lines = [
        "# Principle Selection Report",
        "",
        f"Quality filter threshold (lower objective is better): {selection.quality_threshold:.6g}",
        "",
    ]
    for index, (principle, result) in enumerate(
        zip(library.principles, fixed_results)
    ):
        mean_disagreement = fmean(distances[index][other] for other in range(len(library)) if other != index) if len(library) > 1 else 0.0
        kept = index in selected_reasons
        reason = selected_reasons.get(
            index,
            "eliminated by quality filter or lower marginal behavioral diversity",
        )
        report_lines.extend(
            [
                f"## {principle.name}",
                "",
                f"- Expression: `{principle.expression}`",
                f"- Rationale: {principle.rationale}",
                f"- Factors used: {', '.join(principle.factors_used)}",
                f"- Fixed objective / TT / WSF: {result.objective:.6g} / {result.total_tardiness:.6g} / {result.weighted_shortfall:.6g}",
                f"- Mean disagreement: {mean_disagreement:.4f}",
                f"- Decision: {'retained' if kept else 'eliminated'} — {reason}",
                "",
            ]
        )
    report_path = output / "principle_selection_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    (output / "fixed_principle_performance.json").write_text(
        json.dumps([asdict(item) for item in fixed_results], indent=2) + "\n",
        encoding="utf-8",
    )
    return selected_path, report_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select a complementary portfolio from precomputed state/fixed metrics"
    )
    parser.add_argument("--library", required=True)
    parser.add_argument("--state-bank")
    parser.add_argument("--fixed-results")
    parser.add_argument(
        "--discovery-config",
        help="JSON with instance, instance_count, and instance_seed; builds all simulator inputs",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-size", type=int, default=8)
    args = parser.parse_args()
    library = PrincipleLibrary.load(args.library)
    if args.discovery_config:
        if args.state_bank or args.fixed_results:
            parser.error("use discovery-config or precomputed state-bank/fixed-results, not both")
        discovery = json.loads(Path(args.discovery_config).read_text(encoding="utf-8"))
        raw_instance = dict(discovery["instance"])
        tuple_fields = {
            "ops_per_job",
            "checkpoints_per_entity",
            "checkpoint_profile_pool",
            "quantity_range",
            "proc_time_range",
            "eligible_machines_range",
            "service_commitment_scale_range",
            "job_due_date_factor_range",
            "arrival_profile_pool",
        }
        for field in tuple_fields & raw_instance.keys():
            raw_instance[field] = tuple(raw_instance[field])
        instance_config = InstanceConfig(**raw_instance)
        instances = tuple(
            generate_instance(
                instance_config,
                seed=int(discovery["instance_seed"]),
                instance_index=index,
            )
            for index in range(int(discovery["instance_count"]))
        )
        state_bank = build_discovery_state_bank(library.principles, instances)
        fixed_results = tuple(
            evaluate_fixed_principle(principle, instances, principle_index=index)
            for index, principle in enumerate(library.principles)
        )
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / "state_bank.json").write_text(
            json.dumps(state_bank, indent=2) + "\n", encoding="utf-8"
        )
    else:
        if not args.state_bank or not args.fixed_results:
            parser.error("state-bank and fixed-results are required without discovery-config")
        state_bank = json.loads(Path(args.state_bank).read_text(encoding="utf-8"))
        fixed_results = tuple(
            FixedPrincipleResult(**item)
            for item in json.loads(Path(args.fixed_results).read_text(encoding="utf-8"))
        )
    fingerprints = behavioral_fingerprints(library.principles, state_bank)
    distances = behavioral_distance_matrix(fingerprints)
    selection = select_complementary_portfolio(
        library.principles,
        [item.objective for item in fixed_results],
        distances,
        target_size=args.target_size,
    )
    paths = write_selection_artifacts(
        library, fixed_results, fingerprints, selection, args.output_dir
    )
    print(*(str(path) for path in paths), sep="\n")


if __name__ == "__main__":
    main()
