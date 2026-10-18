"""Build a vectorised env (mock or real ``warsim``) behind one normalised API.

``VecEnvAdapter`` guarantees the trainer always sees:
  obs     dict of float32 numpy arrays (INTERFACE.md shapes)
  rewards float32 [E, U]
  dones   bool    [E]
  infos   {"winner": int8 [E] (-1 when not done / draw), "episode_len": int32 [E]}
whatever exact container types the Rust bindings return.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .config import EnvConfig
from .constants import ACTION_DIMS, OBS_KEYS
from .mock_env import MockVecEnv


class VecEnvAdapter:
    def __init__(self, env: Any, n_envs: int, team_size: int) -> None:
        self.env = env
        self.n_envs = n_envs
        self.team_size = team_size
        self.n_units = int(getattr(env, "n_units", 2 * team_size))
        if self.n_units != 2 * team_size:
            raise ValueError(f"env reports n_units={self.n_units}, expected {2 * team_size}")
        dims = list(getattr(env, "action_dims", ACTION_DIMS))
        if dims != list(ACTION_DIMS):
            raise ValueError(f"env action_dims {dims} != {list(ACTION_DIMS)}")
        self.action_dims = dims

    def reset(self) -> dict:
        return _normalise_obs(self.env.reset())

    def step(self, actions: np.ndarray) -> tuple:
        actions = np.ascontiguousarray(actions, dtype=np.uint8)
        obs, rewards, dones, infos = self.env.step(actions)
        dones_arr = np.asarray(dones).astype(bool).reshape(self.n_envs)
        return (
            _normalise_obs(obs),
            np.asarray(rewards, dtype=np.float32).reshape(self.n_envs, self.n_units),
            dones_arr,
            normalise_infos(infos, self.n_envs, dones_arr),
        )

    def close(self) -> None:
        close = getattr(self.env, "close", None)
        if callable(close):
            close()


class MixedVecEnv:
    """Several ``VecEnvAdapter`` sub-envs presented as one batch of envs.

    Used for curriculum rehearsal: env indices ``[0, n_current)`` belong to the
    first (current-stage) part, the rest to earlier stages. All parts must
    have the same unit count so the model sees one consistent shape."""

    def __init__(self, parts: list) -> None:
        if not parts:
            raise ValueError("MixedVecEnv needs at least one sub-env")
        self.parts = list(parts)
        self.n_units = self.parts[0].n_units
        for part in self.parts:
            if part.n_units != self.n_units:
                raise ValueError(
                    f"sub-envs disagree on n_units: {part.n_units} != {self.n_units}")
        self.team_size = self.parts[0].team_size
        self.action_dims = self.parts[0].action_dims
        self.n_current = self.parts[0].n_envs
        self.n_envs = sum(p.n_envs for p in self.parts)

    def reset(self) -> dict:
        return _concat_dicts([p.reset() for p in self.parts])

    def step(self, actions: np.ndarray) -> tuple:
        results, start = [], 0
        for part in self.parts:
            results.append(part.step(actions[start:start + part.n_envs]))
            start += part.n_envs
        obs, rewards, dones, infos = zip(*results)
        return (
            _concat_dicts(obs),
            np.concatenate(rewards),
            np.concatenate(dones),
            _merge_infos(infos),
        )

    def close(self) -> None:
        for part in self.parts:
            part.close()


def _concat_dicts(dicts) -> dict:
    return {k: np.concatenate([d[k] for d in dicts]) for k in dicts[0]}


def _merge_infos(infos) -> dict:
    merged = {k: np.concatenate([i[k] for i in infos])
              for k in ("winner", "episode_len")}
    if all("episode_stats" in i for i in infos):
        merged["episode_stats"] = _concat_dicts([i["episode_stats"] for i in infos])
    return merged


def _normalise_obs(obs: dict) -> dict:
    missing = [k for k in OBS_KEYS if k not in obs]
    if missing:
        raise KeyError(f"observation missing keys {missing}")
    return {k: np.asarray(obs[k], dtype=np.float32) for k in OBS_KEYS}


def normalise_infos(infos: Any, n_envs: int, dones: np.ndarray) -> dict:
    """Accept a dict of arrays or a list of per-env dicts; return dict of arrays."""
    winner = np.full(n_envs, -1, dtype=np.int8)
    length = np.zeros(n_envs, dtype=np.int32)
    if isinstance(infos, dict):
        w = infos.get("winner", infos.get("winners"))
        if w is not None:
            winner = np.asarray(w).astype(np.int8).reshape(n_envs)
        ln = infos.get("episode_len", infos.get("episode_lens"))
        if ln is not None:
            length = np.asarray(ln).astype(np.int32).reshape(n_envs)
    elif isinstance(infos, (list, tuple)):
        for e, info in enumerate(infos):
            if info:
                winner[e] = int(info.get("winner", -1))
                length[e] = int(info.get("episode_len", 0))
    winner = np.where(dones, winner, -1).astype(np.int8)
    result = {"winner": winner, "episode_len": length}
    if isinstance(infos, dict) and "episode_stats" in infos:
        stats = infos["episode_stats"]
        result["episode_stats"] = {
            key: np.where(dones, np.asarray(value).reshape(n_envs), 0)
            for key, value in stats.items()
        }
    return result


def make_vec_env(
    cfg: EnvConfig,
    seed: int,
    n_envs: Optional[int] = None,
    team_size: Optional[int] = None,
    map_kind: Optional[str] = None,
    scenario: Optional[str] = None,
    max_ticks: Optional[int] = None,
) -> VecEnvAdapter:
    n = int(n_envs if n_envs is not None else cfg.n_envs)
    ts = int(team_size if team_size is not None else cfg.team_size)
    kwargs = dict(
        n_envs=n,
        team_size=ts,
        map_kind=map_kind if map_kind is not None else cfg.map_kind,
        max_ticks=int(max_ticks if max_ticks is not None else cfg.max_ticks),
        seed=int(seed),
        **cfg.extra,
    )
    if cfg.kind == "mock":
        return VecEnvAdapter(MockVecEnv(**kwargs), n, ts)
    if cfg.kind == "warsim":
        kwargs["scenario"] = scenario if scenario is not None else cfg.scenario
        try:
            import warsim  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "env.kind='warsim' needs the Rust bindings (crates/sim-py, built with maturin)"
            ) from exc
        return VecEnvAdapter(warsim.VecEnv(**kwargs), n, ts)
    raise ValueError(f"unknown env kind {cfg.kind!r}")
