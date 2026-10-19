"""Pure-numpy stand-in for ``warsim.VecEnv`` with the same API, shapes, dtypes
and auto-reset semantics (docs/INTERFACE.md).

Toy game: two teams on a small continuous arena with random pillar walls.
Units move/strafe/turn and fire hitscan shots that hit the nearest enemy in a
narrow cone along their facing (range + line of sight + ammo + cooldown).
Vision is team-shared (an enemy is a token while any living ally sees it).
Episode ends when a team is wiped out (+1/-1, team-shared) or at max_ticks
(draw, 0). Small shaping: damage dealt/taken, kills/deaths, per-step cost.

Simplifications vs. the real sim: single floor, no doors/stairs (``interact``
is a no-op), local_map is not fogged, last-seen-age is always 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .constants import (
    ACTION_DIMS,
    LOCAL_VIEW,
    MAP_CHANNELS,
    MAX_TEAM_SIZE,
    STATE_FEATS,
    TOKEN_FEATS,
)

MAP_DENSITY = {"open": 0.0, "arena": 0.06, "cover": 0.14}

VISION = 8.0
RANGE = 7.0
CONE = 0.26  # half-angle (rad) of the shooting cone
TURN = np.pi / 8
FWD_SPEED = 0.4
BACK_SPEED = 0.25
STRAFE_SPEED = 0.3
COOLDOWN = 3
MAX_AMMO = 20
DAMAGE = 0.34
LOS_SAMPLES = 32
SPAWN_DEPTH = 3.5


@dataclass(frozen=True)
class MockRewardConfig:
    damage: float = 0.2
    kill: float = 0.2
    step_cost: float = 0.0
    shaping_scale: float = 1.0


class MockVecEnv:
    """Vectorised toy 2-team shooter with the ``warsim.VecEnv`` Python API."""

    def __init__(
        self,
        n_envs: int = 8,
        team_size: int = 2,
        map_kind: str = "arena",
        max_ticks: int = 120,
        seed: int = 0,
        width: int = 12,
        height: int = 12,
        reward: Optional[dict] = None,
    ) -> None:
        if n_envs < 1:
            raise ValueError("n_envs must be >= 1")
        if not 1 <= team_size <= MAX_TEAM_SIZE:
            raise ValueError(f"team_size must be in [1, {MAX_TEAM_SIZE}]")
        if map_kind not in MAP_DENSITY:
            raise ValueError(f"unknown mock map_kind {map_kind!r}; use one of {sorted(MAP_DENSITY)}")
        if max_ticks < 1:
            raise ValueError("max_ticks must be >= 1")
        if width < 2 * SPAWN_DEPTH + 2 or height < 3:
            raise ValueError("arena too small")
        self.n_envs = n_envs
        self.team_size = team_size
        self.n_units = 2 * team_size
        self.action_dims = list(ACTION_DIMS)
        self.map_kind = map_kind
        self.max_ticks = max_ticks
        self.width = width
        self.height = height
        self.reward_cfg = MockRewardConfig(**(reward or {}))
        self.rng = np.random.default_rng(seed)

        E, U = n_envs, self.n_units
        self.team = np.repeat(np.array([0, 1]), team_size)
        self.pos = np.zeros((E, U, 2))
        self.yaw = np.zeros((E, U))
        self.hp = np.zeros((E, U))
        self.ammo = np.zeros((E, U), dtype=np.int64)
        self.cooldown = np.zeros((E, U), dtype=np.int64)
        self.alive = np.zeros((E, U), dtype=bool)
        self.tick = np.zeros(E, dtype=np.int64)
        self.walls = np.zeros((E, height, width), dtype=bool)
        self._enemy = self.team[:, None] != self.team[None, :]
        self._same = ~self._enemy
        self._eye = np.eye(U, dtype=bool)

    # ------------------------------------------------------------------ API
    def reset(self) -> dict:
        for e in range(self.n_envs):
            self._reset_env(e)
        return self._observe()

    def step(self, actions: np.ndarray) -> tuple:
        act = self._validate(actions)
        cfg = self.reward_cfg
        alive_before = self.alive.copy()
        self._apply_motion(act)
        dealt, taken = self._apply_shots(act)
        died = alive_before & ~self.alive
        self.tick += 1

        team_alive = np.stack([self.alive[:, self.team == t].any(1) for t in (0, 1)], 1)
        wiped = ~team_alive
        done = wiped.any(1) | (self.tick >= self.max_ticks)
        winner = np.full(self.n_envs, -1, dtype=np.int8)
        winner[wiped[:, 1] & ~wiped[:, 0]] = 0
        winner[wiped[:, 0] & ~wiped[:, 1]] = 1

        team_r = np.zeros((self.n_envs, 2))
        for t in (0, 1):
            mine, theirs = self.team == t, self.team != t
            shaping = (
                cfg.damage * (dealt[:, mine].sum(1) - taken[:, mine].sum(1))
                + cfg.kill * (died[:, theirs].sum(1) - died[:, mine].sum(1))
                - cfg.step_cost
            )
            terminal = np.where(winner == t, 1.0, np.where(winner == 1 - t, -1.0, 0.0)) * done
            team_r[:, t] = terminal + cfg.shaping_scale * shaping
        rewards = team_r[:, self.team].astype(np.float32)

        infos = {
            "winner": np.where(done, winner, -1).astype(np.int8),
            "episode_len": np.where(done, self.tick, 0).astype(np.int32),
        }
        for e in np.nonzero(done)[0]:
            self._reset_env(int(e))
        return self._observe(), rewards, done.astype(bool), infos

    def close(self) -> None:
        """No resources to release (API parity with warsim.VecEnv)."""

    # ------------------------------------------------------------ internals
    def _validate(self, actions: np.ndarray) -> np.ndarray:
        act = np.asarray(actions)
        expected = (self.n_envs, self.n_units, len(ACTION_DIMS))
        if act.shape != expected:
            raise ValueError(f"actions must have shape {expected}, got {act.shape}")
        act = act.astype(np.int64)
        dims = np.array(ACTION_DIMS)
        return np.where((act >= 0) & (act < dims), act, 0)  # invalid -> 0

    def _reset_env(self, e: int) -> None:
        rng, W, H, n = self.rng, self.width, self.height, self.team_size
        walls = rng.random((H, W)) < MAP_DENSITY[self.map_kind]
        depth = int(np.ceil(SPAWN_DEPTH))
        walls[:, :depth] = False
        walls[:, W - depth :] = False
        self.walls[e] = walls
        y = rng.uniform(0.5, H - 0.5, size=self.n_units)
        x0 = rng.uniform(0.5, SPAWN_DEPTH, size=n)
        x1 = rng.uniform(W - SPAWN_DEPTH, W - 0.5, size=n)
        self.pos[e] = np.stack([np.concatenate([x0, x1]), y], -1)
        self.yaw[e] = rng.uniform(-np.pi, np.pi, size=self.n_units)
        self.hp[e] = 1.0
        self.ammo[e] = MAX_AMMO
        self.cooldown[e] = 0
        self.alive[e] = True
        self.tick[e] = 0

    def _free(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        inside = (x >= 0.05) & (x <= self.width - 0.05) & (y >= 0.05) & (y <= self.height - 0.05)
        ix = np.clip(np.floor(x).astype(np.int64), 0, self.width - 1)
        iy = np.clip(np.floor(y).astype(np.int64), 0, self.height - 1)
        e_idx = np.arange(self.n_envs)[:, None]
        return inside & ~self.walls[e_idx, iy, ix]

    def _apply_motion(self, act: np.ndarray) -> None:
        alive = self.alive
        turn = np.select([act[..., 2] == 1, act[..., 2] == 2], [TURN, -TURN], 0.0)
        self.yaw = _wrap(self.yaw + turn * alive)
        fwd = np.select([act[..., 0] == 1, act[..., 0] == 2], [FWD_SPEED, -BACK_SPEED], 0.0)
        side = np.select([act[..., 1] == 1, act[..., 1] == 2], [STRAFE_SPEED, -STRAFE_SPEED], 0.0)
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        dx = (fwd * c - side * s) * alive
        dy = (fwd * s + side * c) * alive
        x, y = self.pos[..., 0], self.pos[..., 1]
        nx, ny = x + dx, y + dy
        ok_xy, ok_x, ok_y = self._free(nx, ny), self._free(nx, y), self._free(x, ny)
        self.pos = np.stack(
            [
                np.where(ok_xy | (~ok_xy & ok_x), nx, x),
                np.where(ok_xy | (~ok_xy & ~ok_x & ok_y), ny, y),
            ],
            -1,
        )

    def _pairwise(self) -> tuple:
        rel = self.pos[:, None, :, :] - self.pos[:, :, None, :]  # [E, i, j, 2]: i -> j
        dist = np.linalg.norm(rel, axis=-1)
        return rel, dist

    def _line_of_sight(self, rel: np.ndarray) -> np.ndarray:
        ts = np.linspace(0.0, 1.0, LOS_SAMPLES)[1:-1]
        pts = self.pos[:, :, None, None, :] + rel[:, :, :, None, :] * ts[:, None]
        ix = np.clip(np.floor(pts[..., 0]).astype(np.int64), 0, self.width - 1)
        iy = np.clip(np.floor(pts[..., 1]).astype(np.int64), 0, self.height - 1)
        e_idx = np.arange(self.n_envs)[:, None, None, None]
        return ~self.walls[e_idx, iy, ix].any(-1)

    def _apply_shots(self, act: np.ndarray) -> tuple:
        E, U = self.n_envs, self.n_units
        self.cooldown = np.maximum(self.cooldown - 1, 0)
        fire = (act[..., 3] == 1) & self.alive & (self.cooldown == 0) & (self.ammo > 0)
        self.ammo -= fire
        self.cooldown = np.where(fire, COOLDOWN, self.cooldown)
        dealt = np.zeros((E, U))
        taken = np.zeros((E, U))
        if not fire.any():
            return dealt, taken
        rel, dist = self._pairwise()
        bearing = _wrap(np.arctan2(rel[..., 1], rel[..., 0]) - self.yaw[:, :, None])
        cand = (
            fire[:, :, None]
            & self.alive[:, None, :]
            & self._enemy[None]
            & (dist <= RANGE)
            & (np.abs(bearing) <= CONE)
        )
        cand &= self._line_of_sight(rel)
        masked = np.where(cand, dist, np.inf)
        target = masked.argmin(-1)
        hit = np.isfinite(masked.min(-1))
        e_idx, s_idx = np.nonzero(hit)
        t_idx = target[e_idx, s_idx]
        np.add.at(taken, (e_idx, t_idx), DAMAGE)
        dealt[e_idx, s_idx] = DAMAGE
        self.hp = self.hp - taken
        self.alive &= self.hp > 1e-6
        return dealt, taken

    def _observe(self) -> dict:
        E, U = self.n_envs, self.n_units
        rel, dist = self._pairwise()
        los = self._line_of_sight(rel)
        sees = self.alive[:, :, None] & self.alive[:, None, :] & (dist <= VISION) & los
        team_sees = np.stack([sees[:, self.team == t].any(1) for t in (0, 1)], 1)  # [E, 2, U]
        visible = team_sees[:, self.team, :]  # [E, i, j]

        c = np.cos(self.yaw)[:, :, None]
        s = np.sin(self.yaw)[:, :, None]
        rx = np.clip((c * rel[..., 0] + s * rel[..., 1]) / VISION, -2.0, 2.0)
        ry = np.clip((-s * rel[..., 0] + c * rel[..., 1]) / VISION, -2.0, 2.0)
        ryaw = self.yaw[:, None, :] - self.yaw[:, :, None]
        bcast = np.zeros((E, U, U))

        def per_j(v: np.ndarray) -> np.ndarray:
            return bcast + v[:, None, :]

        is_self = np.broadcast_to(self._eye, (E, U, U))
        is_ally = np.broadcast_to(self._same & ~self._eye, (E, U, U))
        is_enemy = np.broadcast_to(self._enemy, (E, U, U))
        feats = np.stack(
            [
                is_self,
                is_ally,
                is_enemy,
                rx,
                ry,
                bcast,  # same floor
                np.sin(ryaw),
                np.cos(ryaw),
                per_j(self.hp.clip(0.0, 1.0)),
                per_j(self.ammo / MAX_AMMO),
                per_j(self.cooldown / COOLDOWN),
                bcast,  # last-seen-age: only visible enemies are tokens here
            ],
            -1,
        ).astype(np.float32)
        alive_j = self.alive[:, None, :]
        valid = is_self | (is_ally & alive_j) | (is_enemy & visible & alive_j)
        j = np.arange(U)[None, None, :]
        key = np.where(
            is_self, 0.0, np.where(is_ally, 1.0 + j * 1e-3, 2.0 + dist / 100.0)
        )
        key = np.where(valid, key, 9.0)
        order = np.argsort(key, axis=-1, kind="stable")
        tokens = np.take_along_axis(feats, order[..., None], axis=2)
        mask = np.take_along_axis(valid, order, axis=2).astype(np.float32)
        tokens *= mask[..., None]
        assert tokens.shape[-1] == TOKEN_FEATS

        return {
            "tokens": np.ascontiguousarray(tokens),
            "token_mask": mask,
            "local_map": self._local_map(),
            "state": self._state(),
            "alive": self.alive.astype(np.float32),
            "action_availability": np.stack(
                [self.alive & (self.ammo > 0) & (self.cooldown <= 1),
                 np.zeros_like(self.alive, dtype=bool)], -1
            ).astype(np.float32),
        }

    def _local_map(self) -> np.ndarray:
        E, U, V = self.n_envs, self.n_units, LOCAL_VIEW
        half = V // 2
        fwd_off = (half - np.arange(V))[:, None] * np.ones((1, V))  # rows: ahead at top
        right_off = np.ones((V, 1)) * (np.arange(V) - half)[None, :]  # cols: right is +
        c = np.cos(self.yaw)[:, :, None, None]
        s = np.sin(self.yaw)[:, :, None, None]
        wx = self.pos[..., 0][:, :, None, None] + fwd_off * c + right_off * s
        wy = self.pos[..., 1][:, :, None, None] + fwd_off * s - right_off * c
        inside = (wx >= 0) & (wx < self.width) & (wy >= 0) & (wy < self.height)
        ix = np.clip(np.floor(wx).astype(np.int64), 0, self.width - 1)
        iy = np.clip(np.floor(wy).astype(np.int64), 0, self.height - 1)
        e_idx = np.arange(E)[:, None, None, None]
        wall = ~inside | self.walls[e_idx, iy, ix]
        out = np.zeros((E, U, MAP_CHANNELS, V, V), dtype=np.float32)
        out[:, :, 0] = wall
        out[:, :, 2] = ~wall
        return out

    def _state(self) -> np.ndarray:
        E, U = self.n_envs, self.n_units
        team_sign = np.where(self.team == 0, -1.0, 1.0)[None, :] * np.ones((E, 1))
        st = np.stack(
            [
                self.pos[..., 0] / self.width * 2 - 1,
                self.pos[..., 1] / self.height * 2 - 1,
                np.sin(self.yaw),
                np.cos(self.yaw),
                self.hp.clip(0.0, 1.0),
                self.ammo / MAX_AMMO,
                self.cooldown / COOLDOWN,
                self.alive.astype(np.float64),
                team_sign,
                np.zeros((E, U)),  # floor
            ],
            -1,
        )
        assert st.shape[-1] == STATE_FEATS
        return st.reshape(E, U * STATE_FEATS).astype(np.float32)


def _wrap(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2 * np.pi) - np.pi
