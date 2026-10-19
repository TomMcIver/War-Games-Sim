"""Small helpers shared by the binding tests."""

from __future__ import annotations

import numpy as np

import warsim

# Expected obs layout per INTERFACE.md, as functions of (E, U).
OBS_SHAPES = {
    "tokens": lambda e, u: (e, u, u, warsim.TOKEN_FEATS),
    "token_mask": lambda e, u: (e, u, u),
    "local_map": lambda e, u: (e, u, warsim.MAP_CHANNELS, warsim.LOCAL_VIEW, warsim.LOCAL_VIEW),
    "state": lambda e, u: (e, u * warsim.STATE_FEATS),
    "alive": lambda e, u: (e, u),
    "action_availability": lambda e, u: (e, u, 2),
}


def random_actions(rng: np.random.Generator, env: warsim.VecEnv) -> np.ndarray:
    """Uniformly random valid actions, uint8 [n_envs, n_units, 5]."""
    highs = np.asarray(env.action_dims)
    return rng.integers(0, highs, size=(env.n_envs, env.n_units, len(highs)), dtype=np.uint8)


def noop_actions(env: warsim.VecEnv) -> np.ndarray:
    """All-zero (do nothing) actions, uint8 [n_envs, n_units, 5]."""
    return np.zeros((env.n_envs, env.n_units, len(env.action_dims)), dtype=np.uint8)
