"""Rehearsal: after a promotion, some training envs keep playing earlier stages."""

import numpy as np
import pytest

from warsim_rl.config import StageConfig, load_config
from warsim_rl.curriculum import Curriculum
from warsim_rl.env_factory import MixedVecEnv, make_vec_env
from warsim_rl.trainer import Trainer


def tiny(tmp_path, **changes):
    cfg = load_config("mock_tiny")
    cfg.out_dir = str(tmp_path)
    for k, v in changes.items():
        setattr(cfg, k, v)
    return cfg


def same_size_stages():
    return [
        StageConfig(team_size=1, map_kind="open", promote_winrate=0.0, min_updates=1),
        StageConfig(team_size=1, map_kind="cover", promote_winrate=2.0, min_updates=1),
    ]


def test_mixed_vec_env_concatenates_sub_envs():
    cfg = load_config("mock_tiny").env
    parts = [make_vec_env(cfg, seed=1, n_envs=3, team_size=1),
             make_vec_env(cfg, seed=2, n_envs=2, team_size=1, map_kind="cover")]
    env = MixedVecEnv(parts)
    assert env.n_envs == 5 and env.n_units == 2 and env.n_current == 3
    obs = env.reset()
    assert all(v.shape[0] == 5 for v in obs.values())
    actions = np.zeros((5, env.n_units, 5), dtype=np.uint8)
    obs, rewards, dones, infos = env.step(actions)
    assert rewards.shape == (5, 2) and dones.shape == (5,)
    assert infos["winner"].shape == (5,) and infos["episode_len"].shape == (5,)
    env.close()


def test_mixed_vec_env_rejects_unit_count_mismatch():
    cfg = load_config("mock_tiny").env
    parts = [make_vec_env(cfg, seed=1, n_envs=2, team_size=1),
             make_vec_env(cfg, seed=2, n_envs=2, team_size=2)]
    with pytest.raises(ValueError, match="n_units"):
        MixedVecEnv(parts)


def test_rehearsal_stages_are_earlier_stages_of_same_team_size():
    cfg = load_config("mock_tiny")
    cfg.curriculum.stages = same_size_stages() + [
        StageConfig(team_size=2, map_kind="cover", promote_winrate=2.0, min_updates=1)]
    cur = Curriculum(cfg.curriculum, cfg.env)
    assert cur.rehearsal_stages() == []
    cur.index = 1
    assert [s.map_kind for s in cur.rehearsal_stages()] == ["open"]
    cur.index = 2
    assert cur.rehearsal_stages() == []  # team size changed, models cannot share envs


def test_rehearsal_frac_must_be_a_fraction():
    cfg = load_config("mock_tiny")
    cfg.curriculum.rehearsal_frac = 1.0
    with pytest.raises(ValueError, match="rehearsal_frac"):
        cfg.validate()


def test_trainer_mixes_earlier_stage_after_promotion(tmp_path, monkeypatch):
    cfg = tiny(tmp_path, name="reh", total_updates=3, eval_every=1)
    cfg.curriculum.stages = same_size_stages()
    cfg.curriculum.rehearsal_frac = 0.5
    t = Trainer(cfg, quiet=True)
    assert not isinstance(t.env, MixedVecEnv)  # stage 0 has nothing to rehearse
    monkeypatch.setattr(t, "evaluate_vs_random", lambda: {
        "win_rate": 1.0, "draw_rate": 0.0, "loss_rate": 0.0,
    })
    t.train()
    assert t.curriculum.index == 1
    assert t.env.n_envs == cfg.env.n_envs == 4
    assert t.env.n_current == 2
    t.close()


def test_no_rehearsal_by_default(tmp_path, monkeypatch):
    cfg = tiny(tmp_path, name="plain", total_updates=2, eval_every=1)
    cfg.curriculum.stages = same_size_stages()
    t = Trainer(cfg, quiet=True)
    monkeypatch.setattr(t, "evaluate_vs_random", lambda: {
        "win_rate": 1.0, "draw_rate": 0.0, "loss_rate": 0.0,
    })
    t.train()
    assert t.curriculum.index == 1 and t.env.n_envs == 4
    assert not hasattr(t.env, "n_current")
    t.close()


def test_warsim_small_rehearses_by_default():
    cfg = load_config("warsim_small")
    assert cfg.curriculum.rehearsal_frac == 0.25
    cfg.validate()
