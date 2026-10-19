"""Shared per-unit actor (entity attention + local-map CNN + GRU) and a
centralised critic over the full battlefield state (CTDE / MAPPO).

One ``ActorCritic`` instance controls every unit of a team (parameter
sharing). All per-agent inputs are flattened into a batch dimension ``B``.
The actor never sees ``state``; the critic does (training only).
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .constants import ACTION_DIMS, LOCAL_VIEW, MAP_CHANNELS, STATE_FEATS, TOKEN_FEATS

NEG_INF_LOGIT = -1e9


def _ortho(module: nn.Module, gain: float = math.sqrt(2)) -> nn.Module:
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(module.weight, gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    return module


class MaskedSelfAttention(nn.Module):
    """Pre-norm transformer block; padded keys are excluded from attention and
    padded positions are zeroed on output, so padding never leaks."""

    def __init__(self, d: int, n_heads: int, ff_mult: int = 2) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ff_mult * d), nn.ReLU(), nn.Linear(ff_mult * d, d))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, T, d = x.shape
        H = self.n_heads
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, H, d // H).permute(2, 0, 3, 1, 4)
        att = (q @ k.transpose(-1, -2)) / math.sqrt(d // H)
        att = att.masked_fill(~mask[:, None, None, :], float("-inf")).softmax(-1)
        out = (att @ v).transpose(1, 2).reshape(B, T, d)
        x = x + self.proj(out)
        x = x + self.ff(self.ln2(x))
        return x * mask[..., None]


class AttentionPool(nn.Module):
    """Multi-head attention pooling with an external query (the self token)."""

    def __init__(self, d: int, n_heads: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.q = nn.Linear(d, d)
        self.kv = nn.Linear(d, 2 * d)
        self.proj = nn.Linear(d, d)

    def forward(self, query: torch.Tensor, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, T, d = x.shape
        H = self.n_heads
        q = self.q(query).view(B, H, 1, d // H)
        k, v = self.kv(x).view(B, T, 2, H, d // H).permute(2, 0, 3, 1, 4)
        att = (q @ k.transpose(-1, -2)) / math.sqrt(d // H)
        att = att.masked_fill(~mask[:, None, None, :], float("-inf")).softmax(-1)
        return self.proj((att @ v).reshape(B, d))


class ActorEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        d = cfg.d_model
        self.embed = nn.Linear(TOKEN_FEATS, d)
        self.blocks = nn.ModuleList(
            MaskedSelfAttention(d, cfg.n_heads, cfg.ff_mult) for _ in range(cfg.n_layers)
        )
        self.ln = nn.LayerNorm(d)
        self.pool = AttentionPool(d, cfg.n_heads)
        c1, c2 = cfg.cnn_channels
        self.cnn = nn.Sequential(
            nn.Conv2d(MAP_CHANNELS, c1, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(c1, c2, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        side = (LOCAL_VIEW + 1) // 2
        self.cnn_out = nn.Sequential(nn.Linear(c2 * side * side, cfg.cnn_out), nn.ReLU())
        self.mlp = nn.Sequential(
            nn.Linear(2 * d + cfg.cnn_out, cfg.hidden),
            nn.ReLU(),
            nn.Linear(cfg.hidden, cfg.hidden),
            nn.ReLU(),
        )

    def forward(
        self, tokens: torch.Tensor, token_mask: torch.Tensor, local_map: torch.Tensor
    ) -> torch.Tensor:
        mask = token_mask > 0.5
        mask = torch.cat([torch.ones_like(mask[:, :1]), mask[:, 1:]], 1)  # self always real
        x = self.embed(tokens) * mask[..., None]
        for block in self.blocks:
            x = block(x, mask)
        x = self.ln(x) * mask[..., None]
        self_tok = x[:, 0]
        pooled = self.pool(self_tok, x, mask)
        m = self.cnn_out(self.cnn(local_map))
        return self.mlp(torch.cat([self_tok, pooled, m], -1))


class CentralCritic(nn.Module):
    """Value per agent from the full state [B, U*STATE_FEATS], treated as a set
    of U unit entities tagged with is_self / is_teammate (so it is agnostic to
    team size and to which side the agent plays)."""

    def __init__(self, cfg: ModelConfig, core_dim: int) -> None:
        super().__init__()
        d = cfg.critic_d
        self.use_core = cfg.critic_use_actor_feat
        self.embed = nn.Linear(STATE_FEATS + 2, d)
        self.blocks = nn.ModuleList(
            MaskedSelfAttention(d, cfg.n_heads, cfg.ff_mult) for _ in range(cfg.critic_layers)
        )
        self.ln = nn.LayerNorm(d)
        self.pool = AttentionPool(d, cfg.n_heads)
        in_dim = 2 * d + (core_dim if self.use_core else 0)
        self.head = nn.Sequential(
            nn.Linear(in_dim, cfg.critic_hidden), nn.ReLU(), nn.Linear(cfg.critic_hidden, 1)
        )

    def forward(
        self, state: torch.Tensor, agent_idx: torch.Tensor, core: Optional[torch.Tensor]
    ) -> torch.Tensor:
        B, S = state.shape
        U = S // STATE_FEATS
        n_team = U // 2
        ents = state.view(B, U, STATE_FEATS)
        units = torch.arange(U, device=state.device)[None]
        agent_idx = agent_idx.long()
        is_self = (units == agent_idx[:, None]).float()
        is_mate = ((units // n_team) == (agent_idx // n_team)[:, None]).float()
        x = self.embed(torch.cat([ents, is_self[..., None], is_mate[..., None]], -1))
        mask = torch.ones(B, U, dtype=torch.bool, device=state.device)
        for block in self.blocks:
            x = block(x, mask)
        x = self.ln(x)
        self_tok = x[torch.arange(B, device=state.device), agent_idx]
        feats = [self_tok, self.pool(self_tok, x, mask)]
        if self.use_core:
            feats.append(core.detach())
        return self.head(torch.cat(feats, -1)).squeeze(-1)


class ActorCritic(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = ActorEncoder(cfg)
        self.gru = nn.GRUCell(cfg.hidden, cfg.hidden) if cfg.use_gru else None
        self.heads = nn.ModuleList(nn.Linear(cfg.hidden, n) for n in ACTION_DIMS)
        self.critic = CentralCritic(cfg, cfg.hidden)
        self._init_weights()

    def _init_weights(self) -> None:
        self.apply(_ortho)
        for head in self.heads:
            _ortho(head, 0.01)
        _ortho(self.critic.head[-1], 1.0)
        if self.gru is not None:
            for name, p in self.gru.named_parameters():
                if "weight" in name:
                    nn.init.orthogonal_(p)
                else:
                    nn.init.zeros_(p)

    @property
    def hidden_size(self) -> int:
        return self.cfg.hidden

    def initial_state(self, batch: int, device: Optional[torch.device] = None) -> torch.Tensor:
        return torch.zeros(batch, self.cfg.hidden, device=device)

    def _recur(self, feat: torch.Tensor, h: torch.Tensor, reset: torch.Tensor) -> torch.Tensor:
        if self.gru is None:
            return feat
        h = h * (1.0 - reset.float())[:, None]
        return self.gru(feat, h)

    def _logits(self, core: torch.Tensor, alive: Optional[torch.Tensor],
                availability: Optional[torch.Tensor]) -> list:
        logits = [head(core) for head in self.heads]
        if self.cfg.mask_unavailable_actions and availability is not None:
            for head, index in ((3, 0), (4, 1)):
                unavailable = availability[..., index] < 0.5
                mask = torch.stack([torch.zeros_like(unavailable), unavailable], -1)
                logits[head] = logits[head].masked_fill(mask, NEG_INF_LOGIT)
        if alive is not None and self.cfg.mask_dead_actions:
            logits = [mask_dead(lg, alive) for lg in logits]
        return logits

    def step(self, obs: dict, h: torch.Tensor, reset: torch.Tensor) -> tuple:
        """One tick for B agents. Returns (logits list, core [B,H], new hidden)."""
        feat = self.encoder(obs["tokens"], obs["token_mask"], obs["local_map"])
        core = self._recur(feat, h, reset)
        new_h = core if self.gru is not None else h
        return self._logits(core, obs.get("alive"), obs.get("action_availability")), core, new_h

    def forward_seq(self, obs: dict, h0: torch.Tensor, resets: torch.Tensor) -> tuple:
        """Unroll over a chunk: obs fields [L, B, ...], resets [L, B] (1 = new
        episode starts at this step, hidden is zeroed before the step)."""
        L, B = resets.shape
        flat = lambda t: t.reshape(L * B, *t.shape[2:])  # noqa: E731
        feat = self.encoder(flat(obs["tokens"]), flat(obs["token_mask"]), flat(obs["local_map"]))
        feat = feat.view(L, B, -1)
        if self.gru is None:
            core = feat
        else:
            h, outs = h0, []
            for t in range(L):
                h = self._recur(feat[t], h, resets[t])
                outs.append(h)
            core = torch.stack(outs)
        alive = obs.get("alive")
        logits = self._logits(core, alive, obs.get("action_availability"))
        return logits, core

    def value(
        self, state: torch.Tensor, agent_idx: torch.Tensor, core: Optional[torch.Tensor]
    ) -> torch.Tensor:
        lead = agent_idx.shape
        v = self.critic(
            state.reshape(-1, state.shape[-1]),
            agent_idx.reshape(-1),
            None if core is None else core.reshape(-1, core.shape[-1]),
        )
        return v.view(lead)


def mask_dead(logits: torch.Tensor, alive: torch.Tensor) -> torch.Tensor:
    """Dead units may only pick action 0 on every head (their actions are ignored)."""
    dead = (alive < 0.5)[..., None]
    allowed = torch.zeros_like(logits, dtype=torch.bool)
    allowed[..., 0] = True
    return logits.masked_fill(dead & ~allowed, NEG_INF_LOGIT)


def sample_actions(logits: Sequence[torch.Tensor], deterministic: bool = False) -> tuple:
    """Sample every head; returns (actions [.., 5] long, summed log-prob [..])."""
    acts, logp = [], 0.0
    for lg in logits:
        logsm = F.log_softmax(lg, -1)
        if deterministic:
            a = lg.argmax(-1)
        else:
            flat = logsm.reshape(-1, logsm.shape[-1]).exp()
            a = torch.multinomial(flat, 1).view(lg.shape[:-1])
        acts.append(a)
        logp = logp + logsm.gather(-1, a[..., None]).squeeze(-1)
    return torch.stack(acts, -1), logp


def log_prob_entropy(logits: Sequence[torch.Tensor], actions: torch.Tensor) -> tuple:
    """Factored log-prob (sum over heads) and entropy (sum over heads)."""
    logp, ent = 0.0, 0.0
    for i, lg in enumerate(logits):
        logsm = F.log_softmax(lg, -1)
        logp = logp + logsm.gather(-1, actions[..., i : i + 1].long()).squeeze(-1)
        ent = ent - (logsm.exp() * logsm).sum(-1)
    return logp, ent
