"""Policies that can drive any subset of units: the neural policy plus
``RandomPolicy`` / ``IdlePolicy`` baselines.

Common interface::

    policy.hidden_size -> int
    policy.act(batch, hidden, reset) -> (actions uint8 [B, 5], new_hidden [B, H])

``batch`` is a dict of numpy arrays for B agents (see ``rollout.gather_agents``).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from .checkpoint import load_model
from .constants import ACTION_DIMS
from .models import ActorCritic, sample_actions


def to_torch(batch: dict, device: torch.device) -> dict:
    return {k: torch.as_tensor(v, device=device) for k, v in batch.items()}


class RandomPolicy:
    """Uniform random action on every head (sanity baseline / Elo anchor)."""

    hidden_size = 1

    def __init__(self, seed: Optional[int] = None) -> None:
        self.rng = np.random.default_rng(seed)

    def act(self, batch: dict, hidden: torch.Tensor, reset: torch.Tensor) -> tuple:
        B = len(batch["alive"])
        acts = np.stack([self.rng.integers(0, n, size=B) for n in ACTION_DIMS], -1)
        return acts.astype(np.uint8), hidden


class IdlePolicy:
    """Does nothing (all heads 0)."""

    hidden_size = 1

    def act(self, batch: dict, hidden: torch.Tensor, reset: torch.Tensor) -> tuple:
        B = len(batch["alive"])
        return np.zeros((B, len(ACTION_DIMS)), dtype=np.uint8), hidden


class NetPolicy:
    """Wraps an ``ActorCritic`` for acting (no gradients)."""

    def __init__(
        self, model: ActorCritic, device: str | torch.device = "cpu", deterministic: bool = False
    ) -> None:
        self.model = model
        self.device = torch.device(device)
        self.deterministic = deterministic

    @property
    def hidden_size(self) -> int:
        return self.model.hidden_size

    @classmethod
    def from_checkpoint(
        cls, path: str, device: str = "cpu", deterministic: bool = False
    ) -> "NetPolicy":
        model = load_model(path, device)
        model.eval()
        return cls(model, device, deterministic)

    @torch.no_grad()
    def act(self, batch: dict, hidden: torch.Tensor, reset: torch.Tensor) -> tuple:
        obs = to_torch(batch, self.device)
        logits, _, new_h = self.model.step(obs, hidden, reset)
        actions, _ = sample_actions(logits, self.deterministic)
        return actions.cpu().numpy().astype(np.uint8), new_h


def make_builtin(name: str, seed: Optional[int] = None):
    if name == "random":
        return RandomPolicy(seed)
    if name == "idle":
        return IdlePolicy()
    raise ValueError(f"unknown builtin policy {name!r}")
