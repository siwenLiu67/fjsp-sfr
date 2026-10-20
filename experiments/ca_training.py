"""Fresh-instance episode-stream training and evaluation for CA-PPO."""

from __future__ import annotations

import csv
import json
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean

from fjsp_sfr.agents.ca_ppo import CAPPOAgent, PPOUpdateStats
from fjsp_sfr.core.models import FJSPInstance
from fjsp_sfr.core.validation import validate_schedule
from fjsp_sfr.dispatch.types import RuleAction
from fjsp_sfr.env.ca_rule_selection import CAPPORuleSelectionEnv

TrainingInstanceFactory = Callable[[int], FJSPInstance]


@dataclass(frozen=True)
class CAEvaluationSummary:
    mean_objective: float
    mean_tardiness: float
    mean_weighted_shortfall: float
    mean_decisions: float


@dataclass(frozen=True)
class CATrainingRecord:
    training_episode: int
    ppo_updates: int
    cumulative_transitions: int
    learning_rate: float
    mean_training_objective: float
    mean_training_reward: float
    validation: CAEvaluationSummary
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
    action_fractions: tuple[float, ...]


@dataclass(frozen=True)
class CAEpisodeRecord:
    training_episode: int
    objective: float
    total_tardiness: float
    weighted_shortfall: float
    reward: float
    decisions: int


def moving_average_objectives(
    objectives: list[float] | tuple[float, ...],
    *,
    window: int = 100,
) -> tuple[float, ...]:
    """Trailing objective averages for reporting only; never used by PPO."""

    if window <= 0:
        raise ValueError("moving-average window must be positive")
    prefix = [0.0]
    for objective in objectives:
        prefix.append(prefix[-1] + float(objective))
    return tuple(
        (prefix[index] - prefix[max(0, index - window)])
        / (index - max(0, index - window))
        for index in range(1, len(prefix))
    )


def evaluate_ca_ppo(
    agent: CAPPOAgent,
    instances: tuple[FJSPInstance, ...],
) -> CAEvaluationSummary:
    if not instances:
        raise ValueError("evaluation instances must not be empty")
    results = []
    for instance in instances:
        env = CAPPORuleSelectionEnv(instance)
        result = agent.evaluate_episode(env)
        report = validate_schedule(instance, env.state_snapshot)
        if not report.feasible:
            raise RuntimeError(report.summary())
        results.append(result)
    return CAEvaluationSummary(
        mean_objective=fmean(result.objective.Z for result in results),
        mean_tardiness=fmean(result.objective.total_tardiness for result in results),
        mean_weighted_shortfall=fmean(
            result.objective.weighted_checkpoint_shortfall for result in results
        ),
        mean_decisions=fmean(result.decision_count for result in results),
    )


def _mean_update_stats(updates: list[PPOUpdateStats]) -> tuple[float, ...]:
    fields = (
        "total_loss",
        "policy_loss",
        "value_loss",
        "auxiliary_loss",
        "entropy",
        "approximate_kl",
        "clip_fraction",
        "explained_variance",
        "global_gradient_norm",
        "mean_max_action_prob",
        "logit_std",
        "ratio_std",
        "mean_advantage",
        "advantage_std",
    )
    return tuple(fmean(getattr(update, field) for update in updates) for field in fields)


def _write_artifacts(
    output_dir: Path,
    agent: CAPPOAgent,
    records: list[CATrainingRecord],
    episode_records: list[CAEpisodeRecord],
    *,
    best_episode: int,
    best_validation: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "agent_config.json"
    if not config_path.exists():
        config_path.write_text(
            json.dumps(asdict(agent.config), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    rows = []
    action_slots = max(len(record.action_fractions) for record in records)
    factorized_actions = agent.config.actor_type == "factorized_priority"
    principle_actions = agent.config.actor_type == "principle_selector"
    for record in records:
        row = {
            key: value
            for key, value in asdict(record).items()
            if key not in {"validation", "action_fractions"}
        }
        row.update(
            {
                "validation_objective": record.validation.mean_objective,
                "validation_tardiness": record.validation.mean_tardiness,
                "validation_weighted_shortfall": record.validation.mean_weighted_shortfall,
                "validation_decisions": record.validation.mean_decisions,
            }
        )
        row.update(
            {
                (
                    f"candidate_index_{index}_fraction"
                    if factorized_actions
                    else (
                        f"principle_{index}_fraction"
                        if principle_actions
                        else f"action_R{index + 1}_fraction"
                    )
                ): (
                    record.action_fractions[index]
                    if index < len(record.action_fractions)
                    else 0.0
                )
                for index in range(action_slots)
            }
        )
        rows.append(row)
    with (output_dir / "training_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "training_episodes.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        episode_rows = [asdict(record) for record in episode_records]
        writer = csv.DictWriter(stream, fieldnames=list(episode_rows[0]))
        writer.writeheader()
        writer.writerows(episode_rows)
    training_history = [
        {
            "training_seed": agent.config.seed,
            "episode": record.training_episode,
            "episode_objective": record.objective,
            "episode_reward": record.reward,
            "total_tardiness": record.total_tardiness,
            "weighted_shortfall": record.weighted_shortfall,
            "decisions": record.decisions,
        }
        for record in episode_records
    ]
    with (output_dir / "training_history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(training_history[0]))
        writer.writeheader()
        writer.writerows(training_history)
    moving_averages = moving_average_objectives(
        tuple(record.objective for record in episode_records), window=100
    )
    convergence_history = [
        {
            "training_seed": agent.config.seed,
            "episode": record.training_episode,
            "episode_objective": record.objective,
            "moving_average_objective": moving_average,
        }
        for record, moving_average in zip(episode_records, moving_averages)
    ]
    with (output_dir / "convergence_history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(convergence_history[0]))
        writer.writeheader()
        writer.writerows(convergence_history)
    validation_history = [
        {
            "training_seed": agent.config.seed,
            "episode": record.training_episode,
            "validation_mean_objective": record.validation.mean_objective,
            "validation_mean_tardiness": record.validation.mean_tardiness,
            "validation_mean_weighted_shortfall": (
                record.validation.mean_weighted_shortfall
            ),
            "validation_mean_decisions": record.validation.mean_decisions,
        }
        for record in records
    ]
    with (output_dir / "validation_history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(validation_history[0]))
        writer.writeheader()
        writer.writerows(validation_history)
    best_checkpoint_path = (output_dir / "best_validation.pt").resolve()
    (output_dir / "best_checkpoint.json").write_text(
        json.dumps(
            {
                "training_seed": agent.config.seed,
                "best_episode": best_episode,
                "best_validation_objective": best_validation,
                "checkpoint_path": str(best_checkpoint_path),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    agent.save_checkpoint(
        output_dir / "last.pt",
        metadata={
            "training_episode": records[-1].training_episode,
            "learning_rate": agent.learning_rate,
            "best_episode": best_episode,
            "best_validation_objective": best_validation,
        },
    )


def train_ca_ppo(
    agent: CAPPOAgent,
    training_instance_factory: TrainingInstanceFactory,
    validation_instances: tuple[FJSPInstance, ...],
    *,
    training_episodes: int,
    rollout_batch_size: int = 8,
    validation_interval: int = 100,
    output_dir: str | Path | None = None,
) -> tuple[CATrainingRecord, ...]:
    """Train on a newly generated, fixed-scale instance in every episode."""

    if not validation_instances:
        raise ValueError("validation set must not be empty")
    if min(training_episodes, rollout_batch_size, validation_interval) <= 0:
        raise ValueError("episode, batch, and validation counts must be positive")

    records: list[CATrainingRecord] = []
    initial_validation = evaluate_ca_ppo(agent, validation_instances)
    best_validation = initial_validation.mean_objective
    best_episode = 0
    artifact_dir = Path(output_dir) if output_dir is not None else None
    if artifact_dir is not None:
        agent.save_checkpoint(
            artifact_dir / "best_validation.pt",
            metadata={
                "training_seed": agent.config.seed,
                "training_episode": 0,
                "learning_rate": agent.learning_rate,
                "validation_objective": best_validation,
            },
        )
    cumulative_transitions = 0
    ppo_updates = 0
    next_validation_episode = validation_interval
    interval_episodes = []
    interval_updates: list[PPOUpdateStats] = []
    interval_actions: Counter[int] = Counter()
    episode_records: list[CAEpisodeRecord] = []

    episode_index = 0
    while episode_index < training_episodes:
        pending = []
        current_batch_size = min(
            rollout_batch_size, training_episodes - episode_index
        )
        for _ in range(current_batch_size):
            instance = training_instance_factory(episode_index)
            env = CAPPORuleSelectionEnv(instance)
            episode = agent.collect_episode(env)
            report = validate_schedule(instance, env.state_snapshot)
            if not report.feasible:
                raise RuntimeError(report.summary())
            interval_episodes.append(episode)
            episode_records.append(
                CAEpisodeRecord(
                    training_episode=episode_index + 1,
                    objective=episode.objective.Z,
                    total_tardiness=float(episode.objective.total_tardiness),
                    weighted_shortfall=episode.objective.weighted_checkpoint_shortfall,
                    reward=episode.total_reward,
                    decisions=episode.decision_count,
                )
            )
            pending.extend(episode.records)
            interval_actions.update(record.action for record in episode.records)
            cumulative_transitions += episode.decision_count
            episode_index += 1
        interval_updates.append(agent.update(pending))
        ppo_updates += 1

        should_validate = (
            episode_index >= next_validation_episode
            or episode_index == training_episodes
        )
        if not should_validate:
            continue
        while next_validation_episode <= episode_index:
            next_validation_episode += validation_interval

        validation = evaluate_ca_ppo(agent, validation_instances)
        diagnostic_means = _mean_update_stats(interval_updates)
        total_actions = sum(interval_actions.values())
        record = CATrainingRecord(
            training_episode=episode_index,
            ppo_updates=ppo_updates,
            cumulative_transitions=cumulative_transitions,
            learning_rate=agent.learning_rate,
            mean_training_objective=fmean(
                episode.objective.Z for episode in interval_episodes
            ),
            mean_training_reward=fmean(
                episode.total_reward for episode in interval_episodes
            ),
            validation=validation,
            total_loss=diagnostic_means[0],
            policy_loss=diagnostic_means[1],
            value_loss=diagnostic_means[2],
            auxiliary_loss=diagnostic_means[3],
            entropy=diagnostic_means[4],
            approximate_kl=diagnostic_means[5],
            clip_fraction=diagnostic_means[6],
            explained_variance=diagnostic_means[7],
            global_gradient_norm=diagnostic_means[8],
            mean_max_action_prob=diagnostic_means[9],
            logit_std=diagnostic_means[10],
            ratio_std=diagnostic_means[11],
            mean_advantage=diagnostic_means[12],
            advantage_std=diagnostic_means[13],
            action_fractions=tuple(
                interval_actions[index] / total_actions
                for index in range(
                    max(interval_actions) + 1
                    if agent.config.actor_type == "factorized_priority"
                    else (
                        len(agent.principle_library)
                        if agent.config.actor_type == "principle_selector"
                        and agent.principle_library is not None
                        else len(RuleAction)
                    )
                )
            ),
        )
        records.append(record)
        if validation.mean_objective < best_validation:
            best_validation = validation.mean_objective
            best_episode = episode_index
            if artifact_dir is not None:
                agent.save_checkpoint(
                    artifact_dir / "best_validation.pt",
                    metadata={
                        "training_seed": agent.config.seed,
                        "training_episode": episode_index,
                        "learning_rate": agent.learning_rate,
                        "validation_objective": best_validation,
                    },
                )
        if artifact_dir is not None:
            _write_artifacts(
                artifact_dir,
                agent,
                records,
                episode_records,
                best_episode=best_episode,
                best_validation=best_validation,
            )
        interval_episodes = []
        interval_updates = []
        interval_actions = Counter()
    return tuple(records)
