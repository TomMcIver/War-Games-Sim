"""Fixed real-simulator suite for comparing selected policy checkpoints."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch

from .config import EnvConfig
from .env_factory import make_vec_env
from .evaluate import play_match
from .policies import IdlePolicy, NetPolicy, RandomPolicy

DEFAULT_SEEDS = (71011, 82021, 93031, 104041)
SCENARIOS = {
    "duel": (1, "arena", 120),
    "cover": (1, "arena", 200),
    "door": (1, "arena", 250),
    "skirmish": (2, "arena", 300),
    "arena": (2, "arena", 450),
    "building": (4, "building", 600),
    "village": (8, "village", 600),
}


def wilson_interval(successes: int, trials: int) -> list[float]:
    """Approximate 95% interval for the binary win rate."""
    p, z = successes / trials, 1.96
    den = 1 + z * z / trials
    mid = (p + z * z / (2 * trials)) / den
    radius = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / den
    return [max(0.0, mid - radius), min(1.0, mid + radius)]


def _summary(rows: list[dict]) -> dict:
    count = sum(row["episodes"] for row in rows)
    wins = sum(row["wins"] for row in rows)
    draws = sum(row["draws"] for row in rows)
    shots = sum(row.get("shots_fired", 0) for row in rows)
    hits = sum(row.get("shots_hit", 0) for row in rows)
    result = {"episodes": count, "wins": wins, "draws": draws,
              "losses": count - wins - draws, "win_rate": wins / count,
              "win_rate_95pct_interval": wilson_interval(wins, count),
              "score_rate": (wins + 0.5 * draws) / count,
              "mean_len": sum(row["mean_len"] * row["episodes"] for row in rows) / count}
    if any("timeouts" in row for row in rows):
        result.update({"timeouts": sum(row["timeouts"] for row in rows),
                       "timeout_rate": sum(row["timeouts"] for row in rows) / count,
                       "contact_episode_rate": sum(row["contact_episode_rate"] * row["episodes"]
                                                   for row in rows) / count,
                       "shots_fired": shots, "shots_hit": hits,
                       "hit_rate": hits / max(shots, 1),
                       "damage_dealt": sum(row["damage_dealt"] for row in rows),
                       "kills": sum(row["kills"] for row in rows),
                       "mean_ending_ammo": sum(row["mean_ending_ammo"] * row["episodes"]
                                               for row in rows) / count})
    if any("learner_shots_fired" in row for row in rows):
        learner_shots = sum(row["learner_shots_fired"] for row in rows)
        learner_hits = sum(row["learner_shots_hit"] for row in rows)
        result.update({"learner_shots_fired": learner_shots,
                       "learner_shots_hit": learner_hits,
                       "learner_hit_rate": learner_hits / max(learner_shots, 1),
                       "learner_damage_dealt": sum(row["learner_damage_dealt"] for row in rows),
                       "learner_kills": sum(row["learner_kills"] for row in rows),
                       "learner_deaths": sum(row["learner_deaths"] for row in rows),
                       "learner_mean_ending_ammo": sum(row["learner_mean_ending_ammo"] * row["episodes"]
                                                       for row in rows) / count,
                       "learner_mean_survivors": sum(row["learner_mean_survivors"] * row["episodes"]
                                                      for row in rows) / count})
    return result


def run_suite(checkpoints: list[str], scenarios: list[str], seeds: list[int],
              episodes_per_seed: int, n_envs: int, opponents: list[str],
              frozen_checkpoint: str | None = None, deterministic: bool = False,
              torch_threads: int = 4) -> dict:
    if not checkpoints or not scenarios or not seeds or not opponents:
        raise ValueError("suite needs checkpoints, scenarios, seeds, and opponents")
    if episodes_per_seed < 2 or episodes_per_seed % 2 or n_envs < 2:
        raise ValueError("episodes-per-seed must be even and >= 2; n-envs must be >= 2")
    if "frozen" in opponents and frozen_checkpoint is None:
        raise ValueError("frozen opponent needs --frozen-checkpoint")
    torch.set_num_threads(max(1, torch_threads))
    candidates = {path: NetPolicy.from_checkpoint(path, deterministic=deterministic)
                  for path in checkpoints}
    frozen = (NetPolicy.from_checkpoint(frozen_checkpoint, deterministic=deterministic)
              if frozen_checkpoint else None)
    rows = []
    for checkpoint, policy in candidates.items():
        for scenario in scenarios:
            team_size, map_kind, max_ticks = SCENARIOS[scenario]
            cfg = EnvConfig(kind="warsim", team_size=team_size, map_kind=map_kind,
                            scenario=scenario if scenario in ("duel", "cover", "door", "skirmish")
                            else "standard", max_ticks=max_ticks)
            for seed in seeds:
                for opponent_name in opponents:
                    torch.manual_seed(seed)
                    opponent = {"random": lambda: RandomPolicy(seed + 1),
                                "idle": IdlePolicy,
                                "frozen": lambda: frozen}[opponent_name]()
                    parallel = min(n_envs, episodes_per_seed)
                    parallel -= parallel % 2  # equal episode quotas for the two sides
                    env = make_vec_env(cfg, seed=seed, n_envs=parallel)
                    try:
                        result = play_match(policy, opponent, env, episodes_per_seed)
                    finally:
                        env.close()
                    rows.append({"checkpoint": checkpoint, "opponent": opponent_name,
                                 "scenario": scenario, "seed": seed,
                                 "episodes": result.episodes, "wins": result.wins,
                                 "draws": result.draws, "losses": result.losses,
                                 "mean_len": result.mean_len, **result.telemetry})
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["checkpoint"], row["opponent"]].append(row)
    aggregate = {checkpoint: {opponent: _summary(grouped[checkpoint, opponent])
                              for opponent in opponents}
                 for checkpoint in checkpoints}
    by_scenario = {checkpoint: {
        opponent: {scenario: _summary([row for row in grouped[checkpoint, opponent]
                                       if row["scenario"] == scenario])
                   for scenario in scenarios}
        for opponent in opponents} for checkpoint in checkpoints}
    return {"protocol": "heldout-real-sim-v1", "checkpoints": checkpoints,
            "frozen_checkpoint": frozen_checkpoint, "opponents": opponents,
            "deterministic": deterministic, "seeds": seeds,
            "episodes_per_seed": episodes_per_seed, "scenarios": scenarios,
            "aggregate": aggregate, "by_scenario": by_scenario, "results": rows}


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="Compare checkpoints on fixed real-simulator seeds")
    parser.add_argument("--checkpoint", action="append", required=True,
                        help="repeat for each checkpoint to compare")
    parser.add_argument("--opponents", nargs="+", choices=("random", "idle", "frozen"),
                        default=["random", "idle"])
    parser.add_argument("--frozen-checkpoint", help="fixed self-play opponent")
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--episodes-per-seed", type=int, default=32)
    parser.add_argument("--n-envs", type=int, default=16)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--out", type=Path,
                        help="report path (default: heldout.json beside checkpoint folders)")
    args = parser.parse_args(argv)
    if args.episodes_per_seed < 2 or args.episodes_per_seed % 2 or args.n_envs < 2:
        parser.error("episodes-per-seed must be positive and even; n-envs must be >= 2")
    if any(s < 0 for s in args.seeds) or len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be distinct non-negative integers")
    if "frozen" in args.opponents and not args.frozen_checkpoint:
        parser.error("--opponents frozen needs --frozen-checkpoint")
    report = run_suite(args.checkpoint, args.scenarios, args.seeds,
                       args.episodes_per_seed, args.n_envs, args.opponents,
                       args.frozen_checkpoint, args.deterministic, args.torch_threads)
    rendered = json.dumps(report, indent=2)
    out = args.out or Path(args.checkpoint[-1]).parent.parent / "heldout.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return report


if __name__ == "__main__":
    main()
