"""Head-to-head evaluation between two policies, plus the evaluation CLI.

    python -m warsim_rl.evaluate --checkpoint runs/x/checkpoints/latest.pt \
        --opponent random --episodes 200

Opponents: ``random`` | ``idle`` | ``checkpoint`` (with --opponent-checkpoint)
| ``latest`` (newest league snapshot of the same run).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .checkpoint import load_file
from .config import EnvConfig, _build, apply_overrides, load_config_dict
from .elo import performance_rating
from .env_factory import make_vec_env
from .policies import NetPolicy, make_builtin
from .rollout import episode_score, gather_agents, team_units


@dataclass(frozen=True)
class MatchResult:
    wins: int
    draws: int
    losses: int
    mean_len: float
    telemetry: dict = field(default_factory=dict)

    @property
    def episodes(self) -> int:
        return self.wins + self.draws + self.losses

    @property
    def score(self) -> float:
        return self.wins + 0.5 * self.draws

    def rates(self) -> dict:
        n = max(self.episodes, 1)
        return {
            "win_rate": self.wins / n,
            "draw_rate": self.draws / n,
            "loss_rate": self.losses / n,
        }


def play_match(policy_a, policy_b, env, episodes: int, swap_sides: bool = True) -> MatchResult:
    """Play ``episodes`` games of A vs B on ``env`` (results from A's view).

    Each env plays a fixed quota of episodes so short episodes are not
    over-represented; with ``swap_sides`` odd envs put A on team 1."""
    E, U = env.n_envs, env.n_units
    n = U // 2
    side_a = (np.arange(E) % 2) if swap_sides else np.zeros(E, dtype=np.int64)
    quota = np.array([episodes // E + (1 if e < episodes % E else 0) for e in range(E)])
    counts = np.zeros(E, dtype=np.int64)
    tally = {1.0: 0, 0.5: 0, 0.0: 0}
    lengths = []
    telemetry_rows = []
    hidden = {
        "a": torch.zeros(E, U, policy_a.hidden_size),
        "b": torch.zeros(E, U, policy_b.hidden_size),
    }
    obs = env.reset()
    reset = np.ones(E, dtype=bool)
    e_idx = np.repeat(np.arange(E), n)
    te = torch.as_tensor(e_idx)
    while (counts < quota).any():
        actions = np.zeros((E, U, 5), dtype=np.uint8)
        for key, policy, side in (("a", policy_a, side_a), ("b", policy_b, 1 - side_a)):
            u_idx = team_units(side, n).reshape(-1)
            tu = torch.as_tensor(u_idx)
            r = torch.as_tensor(reset[e_idx], dtype=torch.float32)
            acts, new_h = policy.act(gather_agents(obs, e_idx, u_idx), hidden[key][te, tu], r)
            hidden[key][te, tu] = new_h
            actions[e_idx, u_idx] = acts
        obs, _, dones, infos = env.step(actions)
        for e in np.nonzero(dones)[0]:
            if counts[e] < quota[e]:
                tally[episode_score(int(infos["winner"][e]), int(side_a[e]))] += 1
                lengths.append(int(infos["episode_len"][e]))
                if "episode_stats" in infos:
                    row = {
                        key: float(values[e]) for key, values in infos["episode_stats"].items()
                    }
                    for name in ("shots_fired", "shots_hit", "damage_dealt", "kills",
                                 "deaths", "ending_ammo", "survivors"):
                        if f"{name}_team0" in row:
                            row[f"learner_{name}"] = row[f"{name}_team{side_a[e]}"]
                            row[f"opponent_{name}"] = row[f"{name}_team{1 - side_a[e]}"]
                    telemetry_rows.append(row)
                counts[e] += 1
        reset = dones.copy()
    return MatchResult(
        tally[1.0], tally[0.5], tally[0.0],
        float(np.mean(lengths)) if lengths else 0.0,
        summarize_telemetry(telemetry_rows, lengths),
    )


def summarize_telemetry(rows: list[dict], lengths: list[int]) -> dict:
    """Aggregate completed real-simulator episodes; absent for the mock env."""
    if not rows:
        return {}
    count = len(rows)
    total = lambda key: sum(row[key] for row in rows)  # noqa: E731
    shots = total("shots_fired")
    report = {
        "team_wipes": sum(row["termination"] == 1 for row in rows),
        "mutual_wipes": sum(row["termination"] == 2 for row in rows),
        "timeouts": sum(row["termination"] == 3 for row in rows),
        "timeout_rate": sum(row["termination"] == 3 for row in rows) / count,
        "contact_episode_rate": sum(row["contact_ticks"] > 0 for row in rows) / count,
        "contact_tick_rate": total("contact_ticks") / max(sum(lengths), 1),
        "shots_fired": int(shots),
        "shots_hit": int(total("shots_hit")),
        "hit_rate": total("shots_hit") / max(shots, 1),
        "damage_dealt": round(total("damage_dealt"), 2),
        "kills": int(total("kills")),
        "deaths": int(total("deaths")),
        "mean_ending_ammo": total("ending_ammo") / count,
    }
    if "learner_shots_fired" in rows[0]:
        learner_shots = total("learner_shots_fired")
        report.update({
            "learner_shots_fired": int(learner_shots),
            "learner_shots_hit": int(total("learner_shots_hit")),
            "learner_hit_rate": total("learner_shots_hit") / max(learner_shots, 1),
            "learner_damage_dealt": round(total("learner_damage_dealt"), 2),
            "learner_kills": int(total("learner_kills")),
            "learner_deaths": int(total("learner_deaths")),
            "learner_mean_ending_ammo": total("learner_ending_ammo") / count,
            "learner_mean_survivors": total("learner_survivors") / count,
            "opponent_mean_survivors": total("opponent_survivors") / count,
        })
    return report


# --------------------------------------------------------------------- CLI
def _run_dir_of(checkpoint: Path) -> Optional[Path]:
    for parent in (checkpoint.parent, checkpoint.parent.parent):
        if (parent / "config.json").is_file():
            return parent
    return None


def _env_config(args: argparse.Namespace, ckpt: dict, run_dir: Optional[Path]) -> EnvConfig:
    if args.config:
        data = load_config_dict(args.config)
    elif "config" in ckpt:
        data = json.loads(ckpt["config"])
    elif run_dir is not None:
        data = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    else:
        data = {}
    data = apply_overrides(data, args.set or [])
    env = _build(EnvConfig, dict(data.get("env", {})))
    stage = ckpt.get("curriculum_stage")
    if isinstance(stage, dict) and not args.config:
        env.team_size = int(stage.get("team_size", env.team_size))
        env.map_kind = str(stage.get("map_kind", env.map_kind))
        env.scenario = str(stage.get("scenario", env.scenario))
        env.max_ticks = int(stage.get("max_ticks") or env.max_ticks)
    for name in ("team_size", "map_kind", "scenario", "max_ticks"):
        value = getattr(args, name)
        if value is not None:
            setattr(env, name, value)
    if args.env_kind:
        env.kind = args.env_kind
    return env


def _resolve_opponent(args: argparse.Namespace, checkpoint: Path, run_dir: Optional[Path]):
    """Return (policy, opponent_id, opponent_path)."""
    if args.opponent in ("random", "idle"):
        return make_builtin(args.opponent, args.seed + 1), args.opponent, None
    if args.opponent == "checkpoint":
        if not args.opponent_checkpoint:
            raise SystemExit("--opponent checkpoint needs --opponent-checkpoint PATH")
        path = Path(args.opponent_checkpoint)
    elif args.opponent == "latest":
        if run_dir is None:
            raise SystemExit("cannot locate the run directory for --opponent latest")
        snaps = sorted(p for p in (run_dir / "pool").glob("u*.pt") if p.resolve() != checkpoint.resolve())
        path = snaps[-1] if snaps else run_dir / "checkpoints" / "latest.pt"
        if not path.is_file():
            raise SystemExit(f"no snapshot found under {run_dir}")
    else:
        raise SystemExit(f"unknown opponent {args.opponent!r}")
    return NetPolicy.from_checkpoint(str(path), deterministic=args.deterministic), path.stem, path


def _opponent_elo(opp_id: str, run_dir: Optional[Path], default: float = 1000.0) -> float:
    if run_dir is not None and (run_dir / "league.json").is_file():
        doc = json.loads((run_dir / "league.json").read_text(encoding="utf-8"))
        for entry in doc.get("pool", []):
            if entry["id"] == opp_id:
                return float(entry["elo"])
    return default


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate a warsim_rl checkpoint")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--opponent", default="random", choices=["random", "idle", "checkpoint", "latest"])
    p.add_argument("--opponent-checkpoint")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--n-envs", type=int, default=16)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--deterministic", action="store_true", help="greedy actions")
    p.add_argument("--no-swap", action="store_true", help="checkpoint always plays team 0")
    p.add_argument("--config", help="config name/path for env settings (default: from checkpoint)")
    p.add_argument("--set", action="append", help="override, e.g. env.max_ticks=200")
    p.add_argument("--env-kind", choices=["mock", "warsim"])
    p.add_argument("--team-size", type=int)
    p.add_argument("--map-kind")
    p.add_argument("--scenario", choices=["standard", "duel", "cover", "door", "skirmish"])
    p.add_argument("--max-ticks", type=int)
    return p


def main(argv: Optional[list] = None) -> dict:
    args = build_parser().parse_args(argv)
    if args.episodes < 1:
        raise SystemExit("--episodes must be >= 1")
    torch.manual_seed(args.seed)
    checkpoint = Path(args.checkpoint)
    ckpt = load_file(checkpoint)
    run_dir = _run_dir_of(checkpoint)
    env_cfg = _env_config(args, ckpt, run_dir)
    me = NetPolicy.from_checkpoint(str(checkpoint), deterministic=args.deterministic)
    opponent, opp_id, _ = _resolve_opponent(args, checkpoint, run_dir)
    env = make_vec_env(env_cfg, seed=args.seed, n_envs=min(args.n_envs, args.episodes))
    try:
        result = play_match(me, opponent, env, args.episodes, swap_sides=not args.no_swap)
    finally:
        env.close()
    opp_elo = _opponent_elo(opp_id, run_dir)
    report = {
        "checkpoint": str(checkpoint),
        "opponent": opp_id,
        "episodes": result.episodes,
        "wins": result.wins,
        "draws": result.draws,
        "losses": result.losses,
        **{k: round(v, 4) for k, v in result.rates().items()},
        "mean_len": round(result.mean_len, 1),
        **result.telemetry,
        "opponent_elo": round(opp_elo, 1),
        "performance_elo": round(performance_rating(opp_elo, result.score, result.episodes), 1),
        "env": {"kind": env_cfg.kind, "team_size": env_cfg.team_size,
                "map_kind": env_cfg.map_kind, "scenario": env_cfg.scenario},
    }
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
