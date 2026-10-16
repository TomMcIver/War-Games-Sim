"""Smoke tests against the real Rust sim. Skipped unless ``warsim`` is installed
(build it with maturin, see python/README.md)."""

import numpy as np
import pytest

warsim = pytest.importorskip("warsim")

from warsim_rl import constants  # noqa: E402
from warsim_rl.config import load_config  # noqa: E402
from warsim_rl.env_factory import make_vec_env  # noqa: E402
from warsim_rl.evaluate import play_match  # noqa: E402
from warsim_rl.policies import IdlePolicy, RandomPolicy  # noqa: E402
from warsim_rl.trainer import Trainer  # noqa: E402


def test_constants_match_bindings():
    assert warsim.TOKEN_FEATS == constants.TOKEN_FEATS
    assert warsim.MAP_CHANNELS == constants.MAP_CHANNELS
    assert warsim.LOCAL_VIEW == constants.LOCAL_VIEW
    assert warsim.STATE_FEATS == constants.STATE_FEATS
    assert tuple(warsim.ACTION_DIMS) == constants.ACTION_DIMS


@pytest.mark.parametrize("map_kind", ["arena", "building", "village"])
def test_adapter_shapes_on_real_env(map_kind):
    cfg = load_config("warsim_small").env
    env = make_vec_env(cfg, seed=0, n_envs=2, team_size=2, map_kind=map_kind, max_ticks=50)
    obs = env.reset()
    assert obs["tokens"].shape == (2, 4, 4, constants.TOKEN_FEATS)
    rng = np.random.default_rng(0)
    for _ in range(60):
        acts = np.stack([rng.integers(0, d, (2, 4)) for d in constants.ACTION_DIMS], -1)
        obs, rew, dones, infos = env.step(acts.astype(np.uint8))
        assert rew.shape == (2, 4) and dones.dtype == bool
        assert np.isfinite(obs["tokens"]).all() and np.isfinite(rew).all()
    env.close()


def test_random_vs_idle_on_real_env():
    cfg = load_config("warsim_small").env
    env = make_vec_env(cfg, seed=1, n_envs=4, team_size=1, max_ticks=200)
    res = play_match(RandomPolicy(0), IdlePolicy(), env, episodes=8)
    assert res.episodes == 8
    env.close()


def test_tiny_training_on_real_env(tmp_path):
    cfg = load_config("mock_tiny", ["env.kind=warsim", "env.max_ticks=40", "curriculum.stages=[]"])
    cfg.out_dir = str(tmp_path)
    t = Trainer(cfg, quiet=True)
    row = t.train()
    t.close()
    assert row["update"] == cfg.total_updates and np.isfinite(row["loss"])
