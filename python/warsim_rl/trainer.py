"""Training loop: collect -> MAPPO update -> league snapshot -> eval ->
curriculum -> log -> checkpoint. Resumable from ``checkpoints/latest.pt``."""

from __future__ import annotations

import json
import math
import random
import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .checkpoint import load_file, save_trainer
from .config import TrainConfig, config_to_dict
from .curriculum import Curriculum
from .env_factory import make_vec_env
from .evaluate import play_match
from .league import LATEST, League
from .logger import CsvLogger
from .models import ActorCritic
from .policies import NetPolicy, RandomPolicy
from .ppo import MAPPO
from .rollout import Collector, terminal_values

RECENT_EPISODES = 200
NAN = float("nan")


def seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)


def _mean(xs: list) -> float:
    return float(np.mean(xs)) if xs else NAN


class Trainer:
    def __init__(self, cfg: TrainConfig, resume: Optional[str] = None, quiet: bool = False) -> None:
        cfg.validate()
        self.cfg = cfg
        self.quiet = quiet
        seed_everything(cfg.seed, cfg.deterministic)
        torch.set_num_threads(max(1, cfg.torch_threads))
        self.device = cfg.device
        self.run_dir = Path(cfg.out_dir) / cfg.name
        self.ckpt_dir = self.run_dir / "checkpoints"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "config.json").write_text(
            json.dumps(config_to_dict(cfg), indent=2), encoding="utf-8"
        )
        self.model = ActorCritic(cfg.model).to(self.device)
        self.mappo = MAPPO(self.model, cfg.ppo)
        self.league = League(cfg.league, self.run_dir / "pool", cfg.seed + 1, self.device)
        self.curriculum = Curriculum(cfg.curriculum, cfg.env)
        self.update = 0
        self.global_step = 0
        self.recent: deque = deque(maxlen=RECENT_EPISODES)
        self.last_eval = {"win_rate": NAN, "draw_rate": NAN, "loss_rate": NAN}
        self.last_row: dict = {}
        if resume:
            self._load(resume)
        self.logger = CsvLogger(self.run_dir / "metrics.csv")
        self._build_envs()

    # ---------------------------------------------------------------- envs
    def _build_envs(self) -> None:
        stage = self.curriculum.current()
        self.env = make_vec_env(
            self.cfg.env,
            seed=self.cfg.seed * 7919 + self.update,
            team_size=stage.team_size,
            map_kind=stage.map_kind,
            scenario=stage.scenario,
            max_ticks=stage.max_ticks,
        )
        self.collector = Collector(
            self.env,
            self.model,
            self.mappo,
            self.league,
            self.cfg.ppo.rollout_steps,
            self.device,
            terminal_values(self.cfg.env.extra),
        )

    def evaluate_vs_random(self, episodes: Optional[int] = None) -> dict:
        stage = self.curriculum.current()
        episodes = episodes or self.cfg.eval_episodes
        env = make_vec_env(
            self.cfg.env,
            seed=self.cfg.seed + 104729,
            n_envs=min(self.cfg.env.eval_envs, episodes),
            team_size=stage.team_size,
            map_kind=stage.map_kind,
            scenario=stage.scenario,
            max_ticks=stage.max_ticks,
        )
        try:
            res = play_match(
                NetPolicy(self.model, self.device), RandomPolicy(self.cfg.seed + self.update), env, episodes
            )
        finally:
            env.close()
        return {**res.rates(), "mean_len": res.mean_len, **res.telemetry}

    # ---------------------------------------------------------------- loop
    def shaping_scale(self, progress: float) -> float:
        e = self.cfg.env
        return e.shaping_scale + (e.shaping_scale_final - e.shaping_scale) * progress

    def train(self, until_update: Optional[int] = None) -> dict:
        cfg = self.cfg
        limit = cfg.total_updates if until_update is None else min(until_update, cfg.total_updates)
        while self.update < limit:
            t0 = time.perf_counter()
            progress = self.update / max(cfg.total_updates, 1)
            scale = self.shaping_scale(progress)
            results = self.collector.collect(scale)
            t_collect = time.perf_counter() - t0
            stats = self.mappo.update(self.collector.buffer, progress)
            self.update += 1
            steps = self.collector.E * cfg.ppo.rollout_steps
            self.global_step += steps
            self.recent.extend(results)

            if cfg.league.enabled and self.update % cfg.league.snapshot_every == 0:
                self.league.add_snapshot(self.model, self.update)
            evaluated = cfg.eval_every > 0 and self.update % cfg.eval_every == 0
            if evaluated:
                self.last_eval = self.evaluate_vs_random()
            promoted = self._curriculum_step(evaluated, len(results))

            elapsed = time.perf_counter() - t0
            row = self._row(stats, results, scale, steps, elapsed, t_collect)
            self.last_row = row
            self.logger.log(row)
            if not self.quiet and (self.update % cfg.log_every == 0 or promoted):
                self.logger.echo(self._line(row, promoted))
            if self.update % cfg.checkpoint_every == 0 or self.update == cfg.total_updates:
                self.save()
        return self.last_row

    def _curriculum_step(self, evaluated: bool, completed_episodes: int) -> bool:
        if self.curriculum.metric == "eval_random":
            metric = self.last_eval["win_rate"] if evaluated else None
        else:
            pool = [float(r.score == 1.0) for r in self.recent if r.opponent != LATEST]
            metric = _mean(pool) if len(pool) >= 20 else None
        promoted = self.curriculum.step(metric, completed_episodes)
        if promoted:
            self.env.close()
            self._build_envs()
            self.recent.clear()
        return promoted

    def _row(self, stats: dict, results: list, scale: float, steps: int, elapsed: float, t_collect: float) -> dict:
        rec = list(self.recent)
        scores = [r.score for r in rec]
        stage = self.curriculum.current()
        league = self.league.summary()
        return {
            "update": self.update,
            "global_step": self.global_step,
            "stage": self.curriculum.index,
            "team_size": stage.team_size,
            "map_kind": stage.map_kind,
            "scenario": stage.scenario,
            "episodes_in_stage": self.curriculum.episodes_in_stage,
            "episodes": len(results),
            "win_rate": _mean([float(s == 1.0) for s in scores]),
            "draw_rate": _mean([float(s == 0.5) for s in scores]),
            "score": _mean(scores),
            "score_vs_pool": _mean([r.score for r in rec if r.opponent != LATEST]),
            "score_vs_latest": _mean([r.score for r in rec if r.opponent == LATEST]),
            "ep_return": _mean([r.ret for r in rec]),
            "ep_len": _mean([r.length for r in rec]),
            "eval_win_rate_random": self.last_eval["win_rate"],
            "eval_draw_rate_random": self.last_eval["draw_rate"],
            "eval_loss_rate_random": self.last_eval["loss_rate"],
            "eval_timeout_rate_random": self.last_eval.get("timeout_rate", NAN),
            "eval_contact_rate_random": self.last_eval.get("contact_episode_rate", NAN),
            "eval_hit_rate_random": self.last_eval.get("learner_hit_rate", self.last_eval.get("hit_rate", NAN)),
            "eval_mean_ending_ammo_random": self.last_eval.get("learner_mean_ending_ammo", self.last_eval.get("mean_ending_ammo", NAN)),
            "learner_elo": league["learner_elo"],
            "pool_size": league["pool_size"],
            "best_pool_elo": league["best_pool_elo"],
            **{k: stats.get(k, NAN) for k in (
                "loss", "pg_loss", "v_loss", "entropy", "approx_kl", "clipfrac",
                "grad_norm", "explained_var", "lr", "ent_coef",
            )},
            "shaping_scale": scale,
            "sps": steps / max(elapsed, 1e-9),
            "agent_sps": steps * self.collector.U / max(elapsed, 1e-9),
            "collect_frac": t_collect / max(elapsed, 1e-9),
        }

    @staticmethod
    def _line(row: dict, promoted: bool) -> str:
        def f(key: str, spec: str = ".2f") -> str:
            v = row[key]
            return "  -  " if isinstance(v, float) and math.isnan(v) else format(v, spec)

        tag = "  [PROMOTED]" if promoted else ""
        return (
            f"upd {row['update']:5d} step {row['global_step']:9d} st{row['stage']} "
            f"| win {f('win_rate')} score pool {f('score_vs_pool')} latest {f('score_vs_latest')} "
            f"| eval-rand W {f('eval_win_rate_random')} L {f('eval_loss_rate_random')} "
            f"| ret {f('ep_return', '+.3f')} len {f('ep_len', '.1f')} "
            f"| elo {f('learner_elo', '.0f')} "
            f"| pg {f('pg_loss', '+.4f')} v {f('v_loss', '.4f')} ent {f('entropy', '.3f')} "
            f"kl {f('approx_kl', '.4f')} ev {f('explained_var')} | sps {f('sps', '.0f')}{tag}"
        )

    # ----------------------------------------------------------- persist
    def save(self, path: Optional[str | Path] = None) -> Path:
        path = Path(path) if path else self.ckpt_dir / "latest.pt"
        payload = {
            "model_cfg": config_to_dict(self.cfg)["model"],
            "model": self.model.state_dict(),
            "ppo": self.mappo.state_dict(),
            "league": self.league.state_dict(),
            "curriculum": self.curriculum.state_dict(),
            "curriculum_stage": self.curriculum.current_dict(),
            "update": self.update,
            "global_step": self.global_step,
            "config": json.dumps(config_to_dict(self.cfg)),
            "torch_rng": torch.get_rng_state(),
            "last_eval": dict(self.last_eval),
        }
        save_trainer(path, payload)
        self.league.write_json(self.run_dir / "league.json")
        return path

    def _load(self, resume: str) -> None:
        path = self.ckpt_dir / "latest.pt" if resume == "auto" else Path(resume)
        if not path.is_file():
            if resume == "auto":
                return
            raise FileNotFoundError(path)
        data = load_file(path, self.device)
        self.model.load_state_dict(data["model"])
        self.mappo.load_state_dict(data["ppo"])
        self.league.load_state_dict(data["league"])
        self.curriculum.load_state_dict(data["curriculum"])
        self.update = int(data["update"])
        self.global_step = int(data["global_step"])
        self.last_eval = {k: float(v) for k, v in data.get("last_eval", self.last_eval).items()}
        torch.set_rng_state(data["torch_rng"])
        if not self.quiet:
            print(f"resumed from {path} at update {self.update}", flush=True)

    def close(self) -> None:
        self.env.close()
