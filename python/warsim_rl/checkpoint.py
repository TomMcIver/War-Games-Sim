"""Checkpoint I/O. Everything is stored as tensors + plain Python containers so
files load with ``torch.load(weights_only=True)`` (no arbitrary pickles)."""

from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import torch

from .config import ModelConfig, _build
from .models import ActorCritic

MODEL_FORMAT = "warsim_rl.model.v1"
TRAINER_FORMAT = "warsim_rl.trainer.v1"


def _atomic_save(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def load_file(path: str | Path, device: str | torch.device = "cpu") -> dict:
    data = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(data, dict) or data.get("format") not in (MODEL_FORMAT, TRAINER_FORMAT):
        raise ValueError(f"{path} is not a warsim_rl checkpoint")
    return data


def save_model(path: str | Path, model: ActorCritic, meta: Optional[dict] = None) -> None:
    """Lightweight policy snapshot (used for the league pool)."""
    _atomic_save(
        {
            "format": MODEL_FORMAT,
            "model_cfg": asdict(model.cfg),
            "model": model.state_dict(),
            "meta": dict(meta or {}),
        },
        path,
    )


def model_from_payload(data: dict, device: str | torch.device = "cpu") -> ActorCritic:
    cfg = _build(ModelConfig, dict(data["model_cfg"]))
    model = ActorCritic(cfg).to(device)
    model.load_state_dict(data["model"])
    return model


def load_model(path: str | Path, device: str | torch.device = "cpu") -> ActorCritic:
    """Load a model from either a pool snapshot or a full trainer checkpoint."""
    return model_from_payload(load_file(path, device), device)


def save_trainer(path: str | Path, payload: dict) -> None:
    _atomic_save({"format": TRAINER_FORMAT, **payload}, path)
