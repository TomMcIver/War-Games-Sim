"""Curriculum over (team_size, map_kind, max_ticks) stages, promoted when a
win-rate metric reaches the stage threshold after a minimum number of updates."""

from __future__ import annotations

from dataclasses import asdict
from typing import Optional

from .config import CurriculumConfig, EnvConfig, StageConfig


class Curriculum:
    def __init__(self, cfg: CurriculumConfig, env_cfg: EnvConfig) -> None:
        self.metric = cfg.metric
        stages = list(cfg.stages) or [
            StageConfig(env_cfg.team_size, env_cfg.map_kind, env_cfg.max_ticks, 2.0, 0)
        ]
        self.stages = [
            StageConfig(s.team_size, s.map_kind, s.max_ticks or env_cfg.max_ticks, s.promote_winrate, s.min_updates)
            for s in stages
        ]
        self.index = 0
        self.updates_in_stage = 0

    def current(self) -> StageConfig:
        return self.stages[self.index]

    @property
    def is_last(self) -> bool:
        return self.index >= len(self.stages) - 1

    def step(self, metric: Optional[float]) -> bool:
        """Count one update; return True if promoted to the next stage."""
        self.updates_in_stage += 1
        stage = self.current()
        if self.is_last or metric is None or self.updates_in_stage < stage.min_updates:
            return False
        if metric >= stage.promote_winrate:
            self.index += 1
            self.updates_in_stage = 0
            return True
        return False

    def state_dict(self) -> dict:
        return {"index": self.index, "updates_in_stage": self.updates_in_stage}

    def load_state_dict(self, state: dict) -> None:
        self.index = min(int(state["index"]), len(self.stages) - 1)
        self.updates_in_stage = int(state["updates_in_stage"])

    def current_dict(self) -> dict:
        return asdict(self.current())
