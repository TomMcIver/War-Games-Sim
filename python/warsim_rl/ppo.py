"""MAPPO: rollout buffer over [steps, envs, units], GAE, clipped PPO with a
clipped + normalised value loss, annealed entropy bonus and LR, gradient
clipping, and recurrent minibatches of whole truncated-BPTT chunks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

import torch
import torch.nn as nn

from .config import PPOConfig
from .models import ActorCritic, log_prob_entropy


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    last_value: torch.Tensor,
    gamma: float,
    lam: float,
) -> tuple:
    """Generalised advantage estimation.

    ``rewards``/``values``/``dones`` are [T, ...]; ``dones[t] = 1`` if the
    episode ended with the transition taken at step t (so ``values[t+1]``
    belongs to a new episode and must not be bootstrapped). ``last_value`` is
    V(obs_T) for bootstrapping the final step.
    """
    T = rewards.shape[0]
    adv = torch.zeros_like(rewards)
    running = torch.zeros_like(last_value)
    for t in reversed(range(T)):
        next_value = last_value if t == T - 1 else values[t + 1]
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        running = delta + gamma * lam * nonterminal * running
        adv[t] = running
    return adv, adv + values


class ValueNormalizer:
    """Debiased EMA of return mean/variance (as in the MAPPO reference code)."""

    def __init__(self, beta: float = 0.995, eps: float = 1e-5) -> None:
        self.beta = beta
        self.eps = eps
        self.mean = torch.zeros(())
        self.mean_sq = torch.zeros(())
        self.debias = torch.zeros(())

    def update(self, x: torch.Tensor) -> None:
        if x.numel() == 0:
            return
        x = x.detach().double().cpu()
        b = self.beta
        self.mean = b * self.mean.double() + (1 - b) * x.mean()
        self.mean_sq = b * self.mean_sq.double() + (1 - b) * (x**2).mean()
        self.debias = b * self.debias.double() + (1 - b)

    def stats(self) -> tuple:
        if float(self.debias) < self.eps:
            return 0.0, 1.0
        mean = float(self.mean / self.debias)
        var = float(self.mean_sq / self.debias) - mean**2
        return mean, max(var, 1e-2) ** 0.5

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        mean, std = self.stats()
        return (x - mean) / std

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        mean, std = self.stats()
        return x * std + mean

    def state_dict(self) -> dict:
        return {"mean": self.mean, "mean_sq": self.mean_sq, "debias": self.debias}

    def load_state_dict(self, state: dict) -> None:
        self.mean, self.mean_sq, self.debias = state["mean"], state["mean_sq"], state["debias"]


AGENT_OBS = ("tokens", "token_mask", "local_map", "alive")


class RolloutBuffer:
    """Learner-side transitions, laid out [T, E, n, ...] (n = learner units/env)."""

    def __init__(
        self, steps: int, n_envs: int, n_agents: int, n_units: int, hidden: int, obs_shapes: dict
    ) -> None:
        T, E, n = steps, n_envs, n_agents
        self.steps, self.n_envs, self.n_agents, self.n_units = T, E, n, n_units
        z = torch.zeros
        self.obs = {k: z(T, E, n, *obs_shapes[k]) for k in AGENT_OBS}
        self.state = z(T, E, obs_shapes["state"][0])
        self.agent_idx = z(T, E, n, dtype=torch.long)
        self.actions = z(T, E, n, 5, dtype=torch.long)
        self.logp = z(T, E, n)
        self.values = z(T, E, n)
        self.rewards = z(T, E, n)
        self.dones = z(T, E)
        self.resets = z(T, E)
        self.hidden = z(T, E, n, hidden)
        self.last_value = z(E, n)

    def add(self, t: int, **fields: torch.Tensor) -> None:
        for name, value in fields.items():
            if name in self.obs:
                self.obs[name][t] = value
            else:
                getattr(self, name)[t] = value

    def minibatches(
        self, n_minibatches: int, chunk: int, advantages: torch.Tensor, returns: torch.Tensor
    ) -> Iterator[dict]:
        """Yield minibatches of whole (chunk, env) sequences, units flattened:
        every tensor is [L, B*n, ...] except ``h0`` [B*n, H]."""
        T, E, n = self.steps, self.n_envs, self.n_agents
        L = chunk
        n_chunks = T // L
        order = torch.randperm(n_chunks * E)
        for idx in order.chunk(n_minibatches):
            c, e = idx // E, idx % E
            B = len(idx)
            t_idx = c[None, :] * L + torch.arange(L)[:, None]
            e_idx = e[None, :].expand(L, B)

            def per_agent(x: torch.Tensor) -> torch.Tensor:
                y = x[t_idx, e_idx]
                return y.reshape(L, B * n, *y.shape[3:])

            def per_env(x: torch.Tensor) -> torch.Tensor:
                y = x[t_idx, e_idx]
                y = y[:, :, None].expand(L, B, n, *y.shape[2:])
                return y.reshape(L, B * n, *y.shape[3:])

            yield {
                "obs": {k: per_agent(v) for k, v in self.obs.items()},
                "state": per_env(self.state),
                "resets": per_env(self.resets),
                "agent_idx": per_agent(self.agent_idx),
                "actions": per_agent(self.actions),
                "logp": per_agent(self.logp),
                "values": per_agent(self.values),
                "adv": per_agent(advantages),
                "returns": per_agent(returns),
                "h0": self.hidden[c * L, e].reshape(B * n, -1),
            }


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (x * mask).sum() / mask.sum().clamp(min=1.0)


@dataclass
class Schedule:
    start: float
    end: float

    def at(self, progress: float) -> float:
        p = min(max(progress, 0.0), 1.0)
        return self.start + (self.end - self.start) * p


class MAPPO:
    def __init__(self, model: ActorCritic, cfg: PPOConfig) -> None:
        self.model = model
        self.cfg = cfg
        self.opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=1e-5)
        self.value_norm: Optional[ValueNormalizer] = (
            ValueNormalizer(cfg.value_norm_beta) if cfg.value_norm else None
        )
        lr_end = cfg.lr_final if cfg.lr_schedule == "linear" else cfg.lr
        self.lr_sched = Schedule(cfg.lr, lr_end)
        self.ent_sched = Schedule(cfg.ent_coef, cfg.ent_coef_final)

    def denormalize(self, v: torch.Tensor) -> torch.Tensor:
        return v if self.value_norm is None else self.value_norm.denormalize(v)

    def _normalize(self, v: torch.Tensor) -> torch.Tensor:
        return v if self.value_norm is None else self.value_norm.normalize(v)

    def update(self, buf: RolloutBuffer, progress: float) -> dict:
        cfg = self.cfg
        lr, ent_coef = self.lr_sched.at(progress), self.ent_sched.at(progress)
        for group in self.opt.param_groups:
            group["lr"] = lr
        alive = buf.obs["alive"]
        with torch.no_grad():
            dones = buf.dones[..., None].expand_as(buf.rewards)
            adv, ret = compute_gae(
                buf.rewards, buf.values, dones, buf.last_value, cfg.gamma, cfg.gae_lambda
            )
            if self.value_norm is not None:
                self.value_norm.update(ret[alive > 0.5])

        sums: dict = {}
        n_mb = 0
        self.model.train()
        for _ in range(cfg.epochs):
            epoch_kl = []
            for mb in buf.minibatches(cfg.minibatches, cfg.bptt_chunk, adv, ret):
                stats = self._minibatch_step(mb, ent_coef)
                epoch_kl.append(stats["approx_kl"])
                for k, v in stats.items():
                    sums[k] = sums.get(k, 0.0) + v
                n_mb += 1
            if cfg.target_kl is not None and sum(epoch_kl) / len(epoch_kl) > cfg.target_kl:
                break
        out = {k: v / max(n_mb, 1) for k, v in sums.items()}
        mask = alive > 0.5
        y, yhat = ret[mask], buf.values[mask]
        var_y = y.var() if y.numel() > 1 else torch.tensor(0.0)
        out["explained_var"] = float(1 - (y - yhat).var() / var_y) if var_y > 1e-8 else 0.0
        out["lr"] = lr
        out["ent_coef"] = ent_coef
        return out

    def _minibatch_step(self, mb: dict, ent_coef: float) -> dict:
        cfg = self.cfg
        logits, core = self.model.forward_seq(mb["obs"], mb["h0"], mb["resets"])
        new_logp, entropy = log_prob_entropy(logits, mb["actions"])
        value = self.model.value(mb["state"], mb["agent_idx"], core)
        mask = (mb["obs"]["alive"] > 0.5).float()

        adv = mb["adv"]
        a_mean = masked_mean(adv, mask)
        a_std = masked_mean((adv - a_mean) ** 2, mask).sqrt()
        adv = (adv - a_mean) / (a_std + 1e-8)

        log_ratio = new_logp - mb["logp"]
        ratio = log_ratio.exp()
        surr = torch.min(ratio * adv, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * adv)
        pg_loss = -masked_mean(surr, mask)

        ret_n = self._normalize(mb["returns"])
        old_n = self._normalize(mb["values"])
        v_clip = old_n + (value - old_n).clamp(-cfg.value_clip, cfg.value_clip)
        v_loss = 0.5 * masked_mean(torch.max((value - ret_n) ** 2, (v_clip - ret_n) ** 2), mask)
        ent = masked_mean(entropy, mask)

        loss = pg_loss + cfg.vf_coef * v_loss - ent_coef * ent
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
        self.opt.step()

        with torch.no_grad():
            approx_kl = masked_mean((ratio - 1) - log_ratio, mask)
            clipfrac = masked_mean(((ratio - 1).abs() > cfg.clip).float(), mask)
        return {
            "loss": float(loss),
            "pg_loss": float(pg_loss),
            "v_loss": float(v_loss),
            "entropy": float(ent),
            "approx_kl": float(approx_kl),
            "clipfrac": float(clipfrac),
            "grad_norm": float(grad_norm),
        }

    def state_dict(self) -> dict:
        return {
            "optimizer": self.opt.state_dict(),
            "value_norm": None if self.value_norm is None else self.value_norm.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.opt.load_state_dict(state["optimizer"])
        if self.value_norm is not None and state.get("value_norm") is not None:
            self.value_norm.load_state_dict(state["value_norm"])
