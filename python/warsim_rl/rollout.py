"""Experience collection against league opponents.

Per env and per episode the collector picks which side the learner plays
(side swapping) and which opponent drives the other side. Every step does a
single batched forward of the shared network for all learner units plus all
units of envs whose opponent is ``latest`` (self-play against the current
weights); frozen pool opponents get one batched forward per distinct id.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .league import LATEST, League
from .models import ActorCritic, sample_actions
from .policies import to_torch
from .ppo import MAPPO, RolloutBuffer

NET_KEY = "__learner__"


def gather_agents(obs: dict, e_idx: np.ndarray, u_idx: np.ndarray) -> dict:
    """Per-agent observation batch for (env, unit) pairs."""
    return {
        "tokens": obs["tokens"][e_idx, u_idx],
        "token_mask": obs["token_mask"][e_idx, u_idx],
        "local_map": obs["local_map"][e_idx, u_idx],
        "alive": obs["alive"][e_idx, u_idx],
        "action_availability": obs["action_availability"][e_idx, u_idx],
        "state": obs["state"][e_idx],
        "agent_idx": u_idx.astype(np.int64),
    }


def team_units(teams: np.ndarray, team_size: int) -> np.ndarray:
    """[E] team ids -> [E, n] global unit indices of that team."""
    return teams[:, None] * team_size + np.arange(team_size)[None, :]


def episode_score(winner: int, team: int) -> float:
    if winner < 0:
        return 0.5
    return 1.0 if winner == team else 0.0


def terminal_values(env_extra: dict) -> tuple:
    """(win, loss, draw) terminal rewards, honouring warsim reward_* overrides."""
    return (
        float(env_extra.get("reward_win", 1.0)),
        float(env_extra.get("reward_loss", -1.0)),
        float(env_extra.get("reward_draw", 0.0)),
    )


def rescale_shaping(
    rewards: np.ndarray,
    dones: np.ndarray,
    winner: np.ndarray,
    team_size: int,
    scale: float,
    terminal_rewards: tuple = (1.0, -1.0, 0.0),
) -> np.ndarray:
    """Scale the shaping part of rewards, leaving the terminal reward untouched.

    Works for any env obeying INTERFACE.md (terminal reward is +1 win, -1 loss,
    0 draw on the done step, or the given overrides), so it needs no env
    support and can be annealed without rebuilding the env."""
    if scale == 1.0:
        return rewards
    win, loss, draw = terminal_rewards
    E, U = rewards.shape
    unit_team = np.arange(U) // team_size
    w = winner[:, None]
    terminal = np.where(w < 0, draw, np.where(w == unit_team[None], win, loss))
    terminal = terminal * dones[:, None]
    return (terminal + scale * (rewards - terminal)).astype(np.float32)


@dataclass(frozen=True)
class EpisodeResult:
    opponent: str
    learner_team: int
    score: float
    ret: float
    length: int


class Collector:
    def __init__(
        self,
        env,
        model: ActorCritic,
        mappo: MAPPO,
        league: League,
        steps: int,
        device: str = "cpu",
        terminal_rewards: tuple = (1.0, -1.0, 0.0),
    ) -> None:
        self.env, self.model, self.mappo, self.league = env, model, mappo, league
        self.terminal_rewards = terminal_rewards
        self.device = torch.device(device)
        self.E, self.U = env.n_envs, env.n_units
        self.n = self.U // 2
        self.obs = env.reset()
        self.learner_team = np.array([league.sample_learner_team() for _ in range(self.E)])
        self.opponent = [league.sample_opponent() for _ in range(self.E)]
        self.reset_flag = np.ones(self.E, dtype=bool)
        self.ep_return = np.zeros(self.E)
        self.hidden: dict = {}
        shapes = {k: v.shape[2:] for k, v in self.obs.items() if k != "state"}
        shapes["state"] = self.obs["state"].shape[1:]
        self.buffer = RolloutBuffer(steps, self.E, self.n, self.U, model.hidden_size, shapes)

    def _hidden(self, key: str, size: int) -> torch.Tensor:
        h = self.hidden.get(key)
        if h is None or h.shape[-1] != size:
            h = torch.zeros(self.E, self.U, size, device=self.device)
            self.hidden[key] = h
        return h

    def _learner_forward(self, t: int, actions: np.ndarray) -> None:
        E, n = self.E, self.n
        l_units = team_units(self.learner_team, n)
        o_units = team_units(1 - self.learner_team, n)
        latest = np.array([e for e in range(E) if self.opponent[e] == LATEST], dtype=np.int64)
        e_idx = np.concatenate([np.repeat(np.arange(E), n), np.repeat(latest, n)])
        u_idx = np.concatenate([l_units.reshape(-1), o_units[latest].reshape(-1)])
        obs = to_torch(gather_agents(self.obs, e_idx, u_idx), self.device)
        h_all = self._hidden(NET_KEY, self.model.hidden_size)
        te, tu = torch.as_tensor(e_idx), torch.as_tensor(u_idx)
        h = h_all[te, tu]
        reset = torch.as_tensor(self.reset_flag[e_idx], dtype=torch.float32, device=self.device)
        logits, core, new_h = self.model.step(obs, h, reset)
        acts, logp = sample_actions(logits)
        h_all[te, tu] = new_h
        actions[e_idx, u_idx] = acts.cpu().numpy()

        k = E * n
        nl = lambda x: x[:k].reshape(E, n, *x.shape[1:])  # noqa: E731
        value = self.model.value(obs["state"][:k], obs["agent_idx"][:k], core[:k])
        self.buffer.add(
            t,
            tokens=nl(obs["tokens"]),
            token_mask=nl(obs["token_mask"]),
            local_map=nl(obs["local_map"]),
            alive=nl(obs["alive"]),
            action_availability=nl(obs["action_availability"]),
            state=torch.as_tensor(self.obs["state"]),
            agent_idx=nl(obs["agent_idx"]),
            actions=nl(acts),
            logp=nl(logp),
            values=nl(self.mappo.denormalize(value)),
            resets=torch.as_tensor(self.reset_flag, dtype=torch.float32),
            hidden=nl(h),
        )

    def _opponent_forward(self, actions: np.ndarray) -> None:
        o_units = team_units(1 - self.learner_team, self.n)
        for opp in sorted(set(self.opponent) - {LATEST}):
            envs = np.array([e for e in range(self.E) if self.opponent[e] == opp], dtype=np.int64)
            e_idx = np.repeat(envs, self.n)
            u_idx = o_units[envs].reshape(-1)
            policy = self.league.get_policy(opp)
            h_all = self._hidden(opp, policy.hidden_size)
            te, tu = torch.as_tensor(e_idx), torch.as_tensor(u_idx)
            reset = torch.as_tensor(self.reset_flag[e_idx], dtype=torch.float32)
            acts, new_h = policy.act(gather_agents(self.obs, e_idx, u_idx), h_all[te, tu], reset)
            h_all[te, tu] = new_h
            actions[e_idx, u_idx] = acts

    def _finish_episodes(self, dones: np.ndarray, infos: dict, results: list) -> None:
        for e in np.nonzero(dones)[0]:
            team = int(self.learner_team[e])
            score = episode_score(int(infos["winner"][e]), team)
            results.append(
                EpisodeResult(
                    self.opponent[e], team, score, float(self.ep_return[e]), int(infos["episode_len"][e])
                )
            )
            self.league.record(self.opponent[e], score)
            self.ep_return[e] = 0.0
            self.learner_team[e] = self.league.sample_learner_team()
            self.opponent[e] = self.league.sample_opponent()

    @torch.no_grad()
    def collect(self, shaping_scale: float = 1.0) -> list:
        """Fill ``self.buffer`` with one rollout; return finished episodes."""
        self.model.eval()
        results: list = []
        E, n = self.E, self.n
        for t in range(self.buffer.steps):
            actions = np.zeros((E, self.U, 5), dtype=np.uint8)
            self._learner_forward(t, actions)
            self._opponent_forward(actions)
            l_units = team_units(self.learner_team, n)
            obs, rewards, dones, infos = self.env.step(actions)
            rewards = rescale_shaping(
                rewards, dones, infos["winner"], n, shaping_scale, self.terminal_rewards
            )
            r_l = rewards[np.arange(E)[:, None], l_units]
            self.buffer.add(
                t,
                rewards=torch.as_tensor(r_l, dtype=torch.float32),
                dones=torch.as_tensor(dones, dtype=torch.float32),
            )
            self.ep_return += r_l.mean(1)
            self._finish_episodes(dones, infos, results)
            self.reset_flag = dones.copy()
            self.obs = obs
        self._bootstrap()
        live = set(self.opponent) | {NET_KEY}
        self.hidden = {k: v for k, v in self.hidden.items() if k in live}
        return results

    def _bootstrap(self) -> None:
        n = self.n
        e_idx = np.repeat(np.arange(self.E), n)
        u_idx = team_units(self.learner_team, n).reshape(-1)
        obs = to_torch(gather_agents(self.obs, e_idx, u_idx), self.device)
        h = self._hidden(NET_KEY, self.model.hidden_size)[torch.as_tensor(e_idx), torch.as_tensor(u_idx)]
        reset = torch.as_tensor(self.reset_flag[e_idx], dtype=torch.float32)
        _, core, _ = self.model.step(obs, h, reset)
        value = self.model.value(obs["state"], obs["agent_idx"], core)
        self.buffer.last_value = self.mappo.denormalize(value).view(self.E, n)
