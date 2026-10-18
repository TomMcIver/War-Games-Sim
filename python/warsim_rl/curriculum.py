"""Scenario curriculum promoted by win rate and minimum exposure."""

from __future__ import annotations

from dataclasses import asdict
from typing import Optional

from .config import CurriculumConfig, EnvConfig, StageConfig


class Curriculum:
    def __init__(self, cfg: CurriculumConfig, env_cfg: EnvConfig) -> None:
        self.metric = cfg.metric
        stages = list(cfg.stages) or [
            StageConfig(env_cfg.team_size, env_cfg.map_kind, env_cfg.max_ticks, 2.0, 0,
                        env_cfg.scenario)
        ]
        self.stages = [
            StageConfig(s.team_size, s.map_kind, s.max_ticks or env_cfg.max_ticks,
                        s.promote_winrate, s.min_updates, s.scenario, s.min_episodes)
            for s in stages
        ]
        self.index = 0
        self.updates_in_stage = 0
        self.episodes_in_stage = 0

    def current(self) -> StageConfig:
        return self.stages[self.index]

    def rehearsal_stages(self) -> list[StageConfig]:
        """Earlier stages that can share a training batch with the current one.

        The model's unit count is fixed by team size, so only earlier stages
        with the current team size qualify."""
        current = self.current()
        return [s for s in self.stages[: self.index] if s.team_size == current.team_size]

    @property
    def is_last(self) -> bool:
        return self.index >= len(self.stages) - 1

    def step(self, metric: Optional[float], completed_episodes: int = 0) -> bool:
        """Count one update; return True if promoted to the next stage."""
        self.updates_in_stage += 1
        self.episodes_in_stage += completed_episodes
        stage = self.current()
        if (self.is_last or metric is None or metric <= 0
                or self.updates_in_stage < stage.min_updates
                or self.episodes_in_stage < stage.min_episodes):
            return False
        if metric >= stage.promote_winrate:
            self.index += 1
            self.updates_in_stage = 0
            self.episodes_in_stage = 0
            return True
        return False

    def state_dict(self) -> dict:
        return {"index": self.index, "updates_in_stage": self.updates_in_stage,
                "episodes_in_stage": self.episodes_in_stage}

    def load_state_dict(self, state: dict) -> None:
        self.index = min(int(state["index"]), len(self.stages) - 1)
        self.updates_in_stage = int(state["updates_in_stage"])
        self.episodes_in_stage = int(state.get("episodes_in_stage", 0))

    def current_dict(self) -> dict:
        return asdict(self.current())
