"""Self-play league: opponent pool, PFSP sampling, Elo bookkeeping.

The learner plays one team; the other team is driven by an opponent drawn
per episode: ``latest`` (the learner's current weights, batched into the
learner forward pass) with probability ``p_latest``, otherwise a frozen pool
member (past snapshots + optional random/idle baselines) weighted by
prioritised fictitious self-play on the learner's win-rate against it.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .checkpoint import save_model
from .config import LeagueConfig
from .elo import EloTable
from .models import ActorCritic
from .policies import NetPolicy, make_builtin

LATEST = "latest"
LEARNER = "learner"
BUILTINS = ("random", "idle")


@dataclass
class PoolEntry:
    id: str
    path: Optional[str]
    added_update: int
    wins: float = 0.0
    draws: float = 0.0
    losses: float = 0.0

    @property
    def games(self) -> float:
        return self.wins + self.draws + self.losses

    @property
    def learner_winrate(self) -> float:
        """Learner's smoothed score against this entry (Laplace prior 0.5)."""
        return (self.wins + 0.5 * self.draws + 1.0) / (self.games + 2.0)


def pfsp_weights(winrates: np.ndarray, mode: str, power: float) -> np.ndarray:
    """Higher weight for opponents the learner struggles against."""
    w = np.asarray(winrates, dtype=np.float64)
    if mode == "hard":
        weights = (1.0 - w) ** power
    elif mode == "even":
        weights = w * (1.0 - w)
    elif mode == "uniform":
        weights = np.ones_like(w)
    else:
        raise ValueError(f"unknown pfsp mode {mode!r}")
    weights = weights + 1e-6
    return weights / weights.sum()


class League:
    def __init__(
        self, cfg: LeagueConfig, pool_dir: str | Path, seed: int = 0, device: str = "cpu"
    ) -> None:
        self.cfg = cfg
        self.pool_dir = Path(pool_dir)
        self.device = device
        self.rng = np.random.default_rng(seed)
        self.entries: "OrderedDict[str, PoolEntry]" = OrderedDict()
        anchors = [cfg.anchor] if cfg.anchor else []
        self.elo = EloTable(cfg.initial_elo, cfg.elo_k, anchors)
        self.elo.set(LEARNER, cfg.initial_elo)
        self.latest_results = {"wins": 0.0, "draws": 0.0, "losses": 0.0}
        self._cache: "OrderedDict[str, object]" = OrderedDict()
        if cfg.include_random:
            self._add_entry(PoolEntry("random", None, 0))
        if cfg.include_idle:
            self._add_entry(PoolEntry("idle", None, 0))

    # ----------------------------------------------------------- sampling
    def _add_entry(self, entry: PoolEntry) -> None:
        self.entries[entry.id] = entry
        self.elo.ratings.setdefault(entry.id, self.cfg.initial_elo)

    def sample_opponent(self) -> str:
        if not self.cfg.enabled or not self.entries:
            return LATEST
        if self.rng.random() < self.cfg.p_latest:
            return LATEST
        ids = list(self.entries)
        rates = np.array([self.entries[i].learner_winrate for i in ids])
        probs = pfsp_weights(rates, self.cfg.pfsp_mode, self.cfg.pfsp_power)
        return ids[int(self.rng.choice(len(ids), p=probs))]

    def sample_learner_team(self) -> int:
        return int(self.rng.random() < self.cfg.side_swap_prob)

    # ------------------------------------------------------------ results
    def record(self, opponent: str, score: float) -> None:
        """Record one finished episode from the learner's perspective."""
        key = "wins" if score > 0.75 else "losses" if score < 0.25 else "draws"
        if opponent == LATEST:
            self.latest_results[key] += 1.0
            return
        entry = self.entries.get(opponent)
        if entry is not None:
            d = self.cfg.pfsp_decay
            entry.wins, entry.draws, entry.losses = entry.wins * d, entry.draws * d, entry.losses * d
            setattr(entry, key, getattr(entry, key) + 1.0)
        self.elo.record(LEARNER, opponent, score)

    # ---------------------------------------------------------- snapshots
    def add_snapshot(self, model: ActorCritic, update: int) -> str:
        snap_id = f"u{update:06d}"
        path = self.pool_dir / f"{snap_id}.pt"
        save_model(path, model, {"update": update, "elo": self.elo.get(LEARNER)})
        self._add_entry(PoolEntry(snap_id, str(path), update))
        self.elo.set(snap_id, self.elo.get(LEARNER))
        self._prune()
        return snap_id

    def _prune(self) -> None:
        snaps = [e for e in self.entries.values() if e.path is not None]
        for entry in snaps[: max(0, len(snaps) - self.cfg.pool_size)]:
            del self.entries[entry.id]
            self._cache.pop(entry.id, None)

    def latest_snapshot_path(self) -> Optional[str]:
        snaps = [e for e in self.entries.values() if e.path is not None]
        return snaps[-1].path if snaps else None

    def get_policy(self, opponent: str):
        if opponent in self._cache:
            self._cache.move_to_end(opponent)
            return self._cache[opponent]
        if opponent in BUILTINS:
            policy = make_builtin(opponent, int(self.rng.integers(2**31)))
        else:
            entry = self.entries.get(opponent)
            if entry is None or entry.path is None:
                raise KeyError(f"no pool entry {opponent!r}")
            policy = NetPolicy.from_checkpoint(entry.path, self.device)
        self._cache[opponent] = policy
        while len(self._cache) > self.cfg.max_cached_models:
            self._cache.popitem(last=False)
        return policy

    # ------------------------------------------------------------- state
    def summary(self) -> dict:
        snaps = [e for e in self.entries.values() if e.path is not None]
        return {
            "learner_elo": self.elo.get(LEARNER),
            "pool_size": len(snaps),
            "best_pool_elo": max((self.elo.get(e.id) for e in snaps), default=float("nan")),
        }

    def state_dict(self) -> dict:
        return {
            "entries": [asdict(e) for e in self.entries.values()],
            "elo": self.elo.state_dict(),
            "latest_results": dict(self.latest_results),
            "rng": json.dumps(self.rng.bit_generator.state),
        }

    def load_state_dict(self, state: dict) -> None:
        self.entries = OrderedDict((e["id"], PoolEntry(**e)) for e in state["entries"])
        self.elo.load_state_dict(state["elo"])
        self.latest_results = dict(state["latest_results"])
        self.rng.bit_generator.state = json.loads(state["rng"])
        self._cache.clear()

    def write_json(self, path: str | Path) -> None:
        table = [
            {
                **asdict(e),
                "elo": round(self.elo.get(e.id), 1),
                "learner_winrate": round(e.learner_winrate, 3),
            }
            for e in self.entries.values()
        ]
        doc = {"learner_elo": round(self.elo.get(LEARNER), 1), "anchor": self.cfg.anchor, "pool": table}
        Path(path).write_text(json.dumps(doc, indent=2), encoding="utf-8")
