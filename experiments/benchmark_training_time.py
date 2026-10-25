"""Time one 20-episode PPO update without saving model results."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch

from fjsp_sfr.dispatch.rule_library import RuleLibrary
from fjsp_sfr.dispatch.theory_features import feature_names_for
from fjsp_sfr.generation.scale_config import make_scaled_instance
from fjsp_sfr.rl.ppo_agent import HGTPPOAgent
from fjsp_sfr.rl.ppo_trainer import PPOTrainer


def benchmark(root: Path, episodes: int = 20) -> list[dict]:
    rows = []
    for scale in ("S", "M", "L"):
        random.seed(42)
        torch.manual_seed(42)
        library_path = root / f"llm_library_pool_{scale}_mix.json"
        library = RuleLibrary.from_json(str(library_path))
        agent = HGTPPOAgent(
            action_size=library.action_size(),
            hidden_dim=64,
            num_layers=2,
            num_heads=4,
            job_input_dim=len(feature_names_for("theory11_plus3")),
        )
        trainer = PPOTrainer(
            agent=agent,
            library=library,
            instances=None,
            instance_factory=lambda seed, current_scale=scale: make_scaled_instance(
                seed, scale=current_scale, difficulty="service_heavy"
            ),
            base_seed=9000,
            lr=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            entropy_coef=0.02,
            epochs_per_update=4,
            reward_scale=10.0,
            feature_set="theory11_plus3",
        )
        started = time.perf_counter()
        stats = trainer.train(
            total_updates=1,
            episodes_per_update=episodes,
            log_interval=0,
        )
        elapsed = time.perf_counter() - started
        if len(stats) != 1:
            raise RuntimeError("timing update did not complete")
        per_episode = elapsed / episodes
        rows.append(
            {
                "scale": scale,
                "episodes": episodes,
                "update_seconds": elapsed,
                "seconds_per_episode": per_episode,
                "estimated_5000_seconds": per_episode * 5000,
                "estimated_5000_hours": per_episode * 5000 / 3600.0,
                "library": str(library_path),
                "action_size": library.action_size(),
            }
        )
        print(
            f"{scale}: {per_episode:.3f}s/episode, "
            f"{elapsed:.3f}s/update, "
            f"5000 episodes ~ {per_episode * 5000 / 3600.0:.2f}h",
            flush=True,
        )
    return rows


def write_reports(rows: list[dict], json_path: Path, markdown_path: Path) -> None:
    json_path.write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# PPO timing estimate",
        "",
        "Each row is one 20-episode update using `theory11_plus3` and the pooled mixed library. No checkpoint is retained.",
        "",
        "| Scale | Actions | Episodes | Seconds/episode | Seconds/update | Estimated 5000 episodes |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['scale']} | {row['action_size']} | {row['episodes']} | "
            f"{row['seconds_per_episode']:.3f} | {row['update_seconds']:.3f} | "
            f"{row['estimated_5000_hours']:.2f} h |"
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="experiments")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--json", default="experiments/training_time_estimate.json")
    parser.add_argument("--markdown", default="experiments/training_time_estimate.md")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    rows = benchmark(Path(args.root), episodes=args.episodes)
    write_reports(rows, Path(args.json), Path(args.markdown))


if __name__ == "__main__":
    main()
