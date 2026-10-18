"""Plain-dataclass configuration with JSON (or optional YAML) loading.

Configs are nested dataclasses. ``load_config`` accepts a file path or the
name of a bundled config in ``warsim_rl/configs`` (e.g. ``mock_small``), and
``apply_overrides`` handles ``--set section.key=value`` CLI overrides.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Optional, get_args, get_origin, get_type_hints

CONFIG_DIR = Path(__file__).resolve().parent / "configs"


@dataclass
class EnvConfig:
    kind: str = "mock"  # "mock" | "warsim"
    n_envs: int = 32
    team_size: int = 2
    map_kind: str = "arena"
    scenario: str = "standard"
    max_ticks: int = 120
    # Shaping rewards are rescaled on the Python side (terminal +-1 untouched),
    # annealed linearly from shaping_scale to shaping_scale_final.
    shaping_scale: float = 1.0
    shaping_scale_final: float = 1.0
    eval_envs: int = 16
    # Extra keyword arguments forwarded to the env constructor.
    extra: dict = field(default_factory=dict)


@dataclass
class ModelConfig:
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    ff_mult: int = 2
    cnn_channels: tuple = (16, 32)
    cnn_out: int = 64
    hidden: int = 128
    use_gru: bool = True
    critic_d: int = 64
    critic_layers: int = 1
    critic_hidden: int = 128
    critic_use_actor_feat: bool = True
    mask_dead_actions: bool = True
    mask_unavailable_actions: bool = True


@dataclass
class PPOConfig:
    rollout_steps: int = 64
    bptt_chunk: int = 16
    epochs: int = 4
    minibatches: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    value_clip: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    ent_coef_final: float = 0.002
    lr: float = 5e-4
    lr_final: float = 1e-4
    lr_schedule: str = "linear"  # "linear" | "constant"
    max_grad_norm: float = 0.5
    value_norm: bool = True
    value_norm_beta: float = 0.995
    target_kl: Optional[float] = None


@dataclass
class LeagueConfig:
    enabled: bool = True
    p_latest: float = 0.5
    snapshot_every: int = 10
    pool_size: int = 20
    pfsp_mode: str = "hard"  # "hard" | "even" | "uniform"
    pfsp_power: float = 2.0
    pfsp_decay: float = 0.995
    include_random: bool = True
    include_idle: bool = False
    side_swap_prob: float = 0.5
    elo_k: float = 16.0
    initial_elo: float = 1000.0
    anchor: str = "random"
    max_cached_models: int = 8


@dataclass
class StageConfig:
    team_size: int = 2
    map_kind: str = "arena"
    max_ticks: Optional[int] = None
    promote_winrate: float = 0.8
    min_updates: int = 10
    scenario: str = "standard"
    min_episodes: int = 0


@dataclass
class CurriculumConfig:
    stages: list = field(default_factory=list)  # list[StageConfig]
    metric: str = "eval_random"  # "eval_random" | "league"
    # Share of training envs that keep playing earlier stages of the same team
    # size after a promotion, to limit forgetting. 0 disables.
    rehearsal_frac: float = 0.0


@dataclass
class TrainConfig:
    name: str = "run"
    seed: int = 0
    deterministic: bool = False
    total_updates: int = 200
    torch_threads: int = 4
    device: str = "cpu"
    out_dir: str = "runs"
    log_every: int = 1
    eval_every: int = 10
    eval_episodes: int = 64
    checkpoint_every: int = 25
    env: EnvConfig = field(default_factory=EnvConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    league: LeagueConfig = field(default_factory=LeagueConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)

    def validate(self) -> None:
        p = self.ppo
        if p.rollout_steps % p.bptt_chunk != 0:
            raise ValueError("ppo.rollout_steps must be a multiple of ppo.bptt_chunk")
        n_chunks = (p.rollout_steps // p.bptt_chunk) * self.env.n_envs
        if p.minibatches < 1 or p.minibatches > n_chunks:
            raise ValueError(f"ppo.minibatches must be in [1, {n_chunks}]")
        if self.env.kind not in ("mock", "warsim"):
            raise ValueError(f"env.kind must be 'mock' or 'warsim', got {self.env.kind!r}")
        if self.model.d_model % self.model.n_heads or self.model.critic_d % self.model.n_heads:
            raise ValueError("model.d_model and model.critic_d must be divisible by n_heads")
        if not 0.0 <= self.league.p_latest <= 1.0:
            raise ValueError("league.p_latest must be in [0, 1]")
        if self.league.pfsp_mode not in ("hard", "even", "uniform"):
            raise ValueError("league.pfsp_mode must be hard | even | uniform")
        if self.curriculum.metric not in ("eval_random", "league"):
            raise ValueError("curriculum.metric must be eval_random | league")
        if not 0.0 <= self.curriculum.rehearsal_frac < 1.0:
            raise ValueError("curriculum.rehearsal_frac must be in [0, 1)")
        if self.env.kind == "warsim":
            scenarios = {"standard", "duel", "cover", "door", "skirmish"}
            for stage in self.curriculum.stages or [StageConfig(
                self.env.team_size, self.env.map_kind, scenario=self.env.scenario
            )]:
                if stage.scenario not in scenarios:
                    raise ValueError(f"unknown scenario {stage.scenario!r}")
                expected = 2 if stage.scenario == "skirmish" else 1
                if stage.scenario != "standard" and stage.team_size != expected:
                    raise ValueError(f"scenario {stage.scenario!r} requires team_size={expected}")
                if stage.min_episodes < 0:
                    raise ValueError("stage.min_episodes must be >= 0")


def _build(cls: type, data: Any) -> Any:
    """Recursively build dataclass ``cls`` from a dict, rejecting unknown keys."""
    if not isinstance(data, dict):
        raise TypeError(f"expected a mapping for {cls.__name__}, got {type(data).__name__}")
    hints = get_type_hints(cls)
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    kwargs = {}
    for name, value in data.items():
        kwargs[name] = _coerce(hints[name], value, name)
    return cls(**kwargs)


def _coerce(hint: Any, value: Any, name: str) -> Any:
    if is_dataclass(hint):
        return _build(hint, value)
    if name == "stages":
        return [v if isinstance(v, StageConfig) else _build(StageConfig, v) for v in value]
    if hint is tuple or get_origin(hint) is tuple:
        return tuple(value)
    origin = get_origin(hint)
    args = [a for a in get_args(hint) if a is not type(None)]
    if origin is not None and len(args) == 1 and value is not None:
        hint = args[0]
    if hint is float and isinstance(value, int):
        return float(value)
    return value


def config_from_dict(data: dict) -> TrainConfig:
    cfg = _build(TrainConfig, data)
    cfg.validate()
    return cfg


def config_to_dict(cfg: TrainConfig) -> dict:
    return asdict(cfg)


def resolve_config_path(name_or_path: str) -> Path:
    path = Path(name_or_path)
    if path.is_file():
        return path
    for suffix in ("", ".json", ".yaml", ".yml"):
        candidate = CONFIG_DIR / f"{path.name}{suffix}"
        if candidate.is_file():
            return candidate
        candidate = CONFIG_DIR / f"{path.stem}{suffix}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"config not found: {name_or_path} (also looked in {CONFIG_DIR})")


def load_config_dict(name_or_path: str) -> dict:
    path = resolve_config_path(name_or_path)
    text = path.read_text(encoding="utf-8")
    if path.suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError("YAML configs need pyyaml; use the JSON config instead") from exc
        return yaml.safe_load(text) or {}
    return json.loads(text)


def _parse_value(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def apply_overrides(data: dict, overrides: list[str]) -> dict:
    """Return a new dict with ``a.b.c=value`` overrides applied (values parsed as JSON)."""
    result = json.loads(json.dumps(data))
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must look like key.path=value, got {item!r}")
        key, raw = item.split("=", 1)
        parts = key.strip().split(".")
        node = result
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise ValueError(f"cannot override inside non-mapping at {key!r}")
        node[parts[-1]] = _parse_value(raw)
    return result


def load_config(name_or_path: str, overrides: Optional[list[str]] = None) -> TrainConfig:
    return config_from_dict(apply_overrides(load_config_dict(name_or_path), overrides or []))


def replace(cfg: Any, **changes: Any) -> Any:
    """Thin alias for ``dataclasses.replace`` (returns a new config object)."""
    return dataclasses.replace(cfg, **changes)
