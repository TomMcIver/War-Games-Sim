"""Repeatable real-simulator baseline without a trained checkpoint.

Example: python -m warsim_rl.baseline --episodes 64 --opponent random
"""

from __future__ import annotations

import argparse
import json

from .config import load_config
from .env_factory import make_vec_env
from .evaluate import play_match
from .policies import IdlePolicy, RandomPolicy


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="Measure a real-simulator random baseline")
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--n-envs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--team-size", type=int, default=1)
    parser.add_argument("--map-kind", choices=("arena", "building", "village"), default="arena")
    parser.add_argument("--scenario", choices=("standard", "duel", "cover", "door", "skirmish"), default="standard")
    parser.add_argument("--max-ticks", type=int, default=600)
    parser.add_argument("--opponent", choices=("random", "idle"), default="random")
    args = parser.parse_args(argv)
    if args.episodes < 1 or args.n_envs < 1 or args.max_ticks < 1:
        parser.error("episodes, n-envs and max-ticks must be positive")

    cfg = load_config("warsim_small").env
    cfg.team_size = args.team_size
    cfg.map_kind = args.map_kind
    cfg.scenario = args.scenario
    cfg.max_ticks = args.max_ticks
    env = make_vec_env(cfg, seed=args.seed, n_envs=min(args.n_envs, args.episodes))
    opponent = RandomPolicy(args.seed + 2) if args.opponent == "random" else IdlePolicy()
    try:
        result = play_match(RandomPolicy(args.seed + 1), opponent, env, args.episodes)
    finally:
        env.close()
    report = {
        "policy": "random",
        "opponent": args.opponent,
        "seed": args.seed,
        "team_size": args.team_size,
        "map_kind": args.map_kind,
        "scenario": args.scenario,
        "max_ticks": args.max_ticks,
        "episodes": result.episodes,
        "wins": result.wins,
        "draws": result.draws,
        "losses": result.losses,
        "mean_len": round(result.mean_len, 1),
        **result.telemetry,
    }
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
