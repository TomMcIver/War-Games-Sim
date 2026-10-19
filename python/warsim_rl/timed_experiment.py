"""Run a wall-clock-bounded real-simulator training experiment.

Training uses the normal curriculum and league. Fixed validation seeds are
never used for promotion or gradient updates. The final held-out suite uses
different seeds and should only be opened after checkpoint selection.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import time
from pathlib import Path

import torch

from .config import load_config
from .env_factory import make_vec_env
from .evaluate import play_match
from .policies import NetPolicy, RandomPolicy
from .trainer import Trainer

VALIDATION_SEEDS = (31001, 42002)
FIELDS = (
    "update", "elapsed_minutes", "stage", "stage_scenario", "train_win_rate",
    "train_draw_rate", "train_score", "train_mean_len", "train_episodes",
    "validation_scenario", "validation_episodes", "validation_wins",
    "validation_draws", "validation_losses", "validation_win_rate",
    "validation_draw_rate", "validation_timeout_rate", "validation_hit_rate",
    "validation_mean_ending_ammo", "validation_mean_len", "anchor_episodes",
    "anchor_wins", "anchor_draws", "anchor_losses", "anchor_win_rate",
    "anchor_draw_rate", "anchor_timeout_rate", "anchor_hit_rate",
    "anchor_mean_ending_ammo", "anchor_mean_len",
)


def evaluate_fixed(trainer: Trainer, stage, seeds: tuple[int, ...], episodes: int) -> dict:
    """Evaluate with fixed policy/opponent RNG and restore training RNG state."""
    saved_rng = torch.get_rng_state()
    was_training = trainer.model.training
    trainer.model.eval()
    results = []
    try:
        for seed in seeds:
            torch.manual_seed(seed)
            env = make_vec_env(
                trainer.cfg.env, seed=seed, n_envs=min(16, episodes),
                team_size=stage.team_size, map_kind=stage.map_kind,
                scenario=stage.scenario, max_ticks=stage.max_ticks,
            )
            try:
                results.append(play_match(
                    NetPolicy(trainer.model, trainer.device), RandomPolicy(seed + 1),
                    env, episodes,
                ))
            finally:
                env.close()
    finally:
        torch.set_rng_state(saved_rng)
        trainer.model.train(was_training)
    n = sum(r.episodes for r in results)
    wins = sum(r.wins for r in results)
    draws = sum(r.draws for r in results)
    shots = sum(r.telemetry.get("learner_shots_fired", 0) for r in results)
    hits = sum(r.telemetry.get("learner_shots_hit", 0) for r in results)
    return {
        "episodes": n, "wins": wins, "draws": draws, "losses": n - wins - draws,
        "win_rate": wins / n, "draw_rate": draws / n,
        "timeout_rate": sum(r.telemetry.get("timeouts", 0) for r in results) / n,
        "hit_rate": hits / max(shots, 1),
        "mean_ending_ammo": sum(r.telemetry.get("learner_mean_ending_ammo", 0) * r.episodes
                                for r in results) / n,
        "mean_len": sum(r.mean_len * r.episodes for r in results) / n,
    }


def run(minutes: float, seed: int, name: str, chunk_updates: int,
        validation_episodes: int, resume: bool = False, rehearsal_frac: float = 0.0) -> Path:
    cfg = load_config("warsim_small")
    cfg.name = name
    cfg.seed = seed
    cfg.curriculum.rehearsal_frac = rehearsal_frac
    cfg.validate()
    cfg.log_every = chunk_updates
    cfg.eval_every = chunk_updates
    cfg.eval_episodes = 64
    cfg.checkpoint_every = chunk_updates
    cfg.league.max_cached_models = max(cfg.league.max_cached_models, cfg.league.pool_size + 2)
    run_dir = Path(cfg.out_dir) / name
    curve_path = run_dir / "learning_curve.csv"
    if not resume and (run_dir / "metrics.csv").exists():
        raise FileExistsError(f"{run_dir} already has training metrics; choose another name")
    start = time.monotonic()
    if resume:
        with curve_path.open(newline="", encoding="utf-8") as file:
            completed = list(csv.DictReader(file))
        last_update = int(completed[-1]["update"])
        origin = curve_path.stat().st_mtime - float(completed[-1]["elapsed_minutes"]) * 60
        start -= time.time() - origin
        metrics_path = run_dir / "metrics.csv"
        with metrics_path.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            header = reader.fieldnames
            kept = [row for row in reader if int(row["update"]) <= last_update]
        tmp = metrics_path.with_suffix(".resume.tmp")
        with tmp.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=header)
            writer.writeheader()
            writer.writerows(kept)
        os.replace(tmp, metrics_path)
    budget = minutes * 60
    trainer = Trainer(cfg, resume="auto" if resume else None)
    history = run_dir / "checkpoints" / "history"
    history.mkdir(parents=True, exist_ok=True)
    anchor = cfg.curriculum.stages[0]
    try:
        with curve_path.open("a" if resume else "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=FIELDS)
            if not resume:
                writer.writeheader()

            def record() -> None:
                stage = trainer.curriculum.current()
                stage_result = evaluate_fixed(trainer, stage, VALIDATION_SEEDS, validation_episodes)
                anchor_result = (stage_result if stage == anchor else
                                 evaluate_fixed(trainer, anchor, VALIDATION_SEEDS,
                                                validation_episodes))
                train = trainer.last_row
                row = {
                    "update": trainer.update,
                    "elapsed_minutes": round((time.monotonic() - start) / 60, 3),
                    "stage": trainer.curriculum.index,
                    "stage_scenario": stage.scenario,
                    "train_win_rate": train.get("win_rate", ""),
                    "train_draw_rate": train.get("draw_rate", ""),
                    "train_score": train.get("score", ""),
                    "train_mean_len": train.get("ep_len", ""),
                    "train_episodes": train.get("episodes", ""),
                    "validation_scenario": stage.scenario,
                    **{f"validation_{key}": value for key, value in stage_result.items()},
                    **{f"anchor_{key}": value for key, value in anchor_result.items()},
                }
                writer.writerow(row)
                file.flush()
                latest = trainer.save()
                shutil.copy2(latest, history / f"u{trainer.update:06d}.pt")
                print(f"validation u{trainer.update}: stage={stage.scenario} "
                      f"W/D/L={stage_result['wins']}/{stage_result['draws']}/{stage_result['losses']} "
                      f"anchor_win={anchor_result['win_rate']:.3f} "
                      f"elapsed={row['elapsed_minutes']:.1f}m", flush=True)

            if not resume:
                record()  # untrained baseline on the same validation setup
            seconds_per_update = None
            while trainer.update < cfg.total_updates and time.monotonic() - start < budget:
                remaining = budget - (time.monotonic() - start)
                size = chunk_updates
                if seconds_per_update is not None:
                    size = min(size, max(1, int((remaining - 20) / seconds_per_update)))
                target = min(trainer.update + size, cfg.total_updates)
                began = time.monotonic()
                trainer.train(until_update=target)
                seconds_per_update = (time.monotonic() - began) / size
                record()
    finally:
        trainer.close()
    (run_dir / "experiment.json").write_text(json.dumps({
        "minutes_budget": minutes, "seed": seed, "rehearsal_frac": rehearsal_frac,
        "validation_seeds": VALIDATION_SEEDS,
        "validation_episodes_per_seed": validation_episodes,
        "updates_completed": trainer.update,
        "elapsed_minutes": round((time.monotonic() - start) / 60, 3),
    }, indent=2) + "\n", encoding="utf-8")
    return run_dir


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Timed simulator learning-curve experiment")
    parser.add_argument("--minutes", type=float, default=30)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--name", default="timed_30m_s1")
    parser.add_argument("--chunk-updates", type=int, default=50)
    parser.add_argument("--validation-episodes-per-seed", type=int, default=32)
    parser.add_argument("--resume", action="store_true", help="resume the last completed chunk")
    parser.add_argument("--rehearsal-frac", type=float, default=0.0,
                        help="share of training envs that replay earlier stages after promotion")
    args = parser.parse_args(argv)
    if args.minutes <= 0 or args.chunk_updates < 1 or args.validation_episodes_per_seed < 2:
        parser.error("minutes, chunk-updates, and validation episodes must be positive")
    run(args.minutes, args.seed, args.name, args.chunk_updates,
        args.validation_episodes_per_seed, args.resume, args.rehearsal_frac)


if __name__ == "__main__":
    main()
