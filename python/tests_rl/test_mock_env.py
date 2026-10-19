"""MockVecEnv conforms to the INTERFACE.md Python API."""

import numpy as np
import pytest

from warsim_rl.constants import ACTION_DIMS, LOCAL_VIEW, MAP_CHANNELS, STATE_FEATS, TOKEN_FEATS
from warsim_rl.env_factory import VecEnvAdapter, normalise_infos
from warsim_rl.mock_env import MockVecEnv


def random_actions(rng, E, U):
    return np.stack([rng.integers(0, d, size=(E, U)) for d in ACTION_DIMS], -1).astype(np.uint8)


@pytest.mark.parametrize("team_size", [1, 2, 4])
def test_shapes_and_dtypes(team_size):
    E = 3
    env = MockVecEnv(n_envs=E, team_size=team_size, seed=1)
    U = 2 * team_size
    assert env.n_units == U and env.action_dims == [3, 3, 3, 2, 2]
    obs = env.reset()
    expected = {
        "tokens": (E, U, U, TOKEN_FEATS),
        "token_mask": (E, U, U),
        "local_map": (E, U, MAP_CHANNELS, LOCAL_VIEW, LOCAL_VIEW),
        "state": (E, U * STATE_FEATS),
        "alive": (E, U),
    }
    for key, shape in expected.items():
        assert obs[key].shape == shape, key
        assert obs[key].dtype == np.float32, key
    obs, rew, dones, infos = env.step(random_actions(np.random.default_rng(0), E, U))
    assert rew.shape == (E, U) and rew.dtype == np.float32
    assert dones.shape == (E,) and dones.dtype == bool
    assert infos["winner"].shape == (E,) and infos["episode_len"].shape == (E,)


def test_token_layout_self_allies_enemies():
    env = MockVecEnv(n_envs=2, team_size=3, seed=3)
    obs = env.reset()
    tok, mask = obs["tokens"], obs["token_mask"]
    assert np.all(mask[:, :, 0] == 1) and np.all(tok[:, :, 0, 0] == 1)  # token 0 is self
    # allies (alive at reset) come right after self
    assert np.all(tok[:, :, 1:3, 1] == 1)
    # padded tokens are all-zero; real tokens are exactly one of self/ally/enemy
    assert np.all(tok[mask == 0] == 0)
    assert np.all(tok[mask == 1][:, :3].sum(-1) == 1)
    # mask is a prefix (real tokens first, then padding)
    assert np.all(np.diff(mask, axis=-1) <= 0)


def test_auto_reset_and_terminal_rewards():
    env = MockVecEnv(n_envs=4, team_size=1, max_ticks=5, seed=0)
    env.reset()
    idle = np.zeros((4, 2, 5), dtype=np.uint8)
    for _ in range(4):
        _, _, dones, _ = env.step(idle)
        assert not dones.any()
    obs, rew, dones, infos = env.step(idle)
    assert dones.all()
    assert np.all(infos["winner"] == -1) and np.all(infos["episode_len"] == 5)
    assert np.allclose(rew, 0.0)  # draw, no shaping
    assert np.all(env.tick == 0) and np.all(obs["alive"] == 1)  # first obs of new episode


def test_win_gives_plus_minus_one():
    env = MockVecEnv(n_envs=1, team_size=1, max_ticks=50, seed=0)
    env.reset()
    # Place shooter facing the target at close range with clear sight.
    env.walls[:] = False
    env.pos[0] = [[2.0, 5.0], [4.0, 5.0]]
    env.yaw[0] = [0.0, np.pi / 2]
    env.hp[0] = [1.0, 0.3]
    act = np.zeros((1, 2, 5), dtype=np.uint8)
    act[0, 0, 3] = 1
    _, rew, dones, infos = env.step(act)
    assert dones[0] and infos["winner"][0] == 0
    assert rew[0, 0] > 1.0 and rew[0, 1] < -1.0  # terminal +-1 plus shaping


def test_determinism_given_seed_and_actions():
    def rollout(seed):
        env = MockVecEnv(n_envs=3, team_size=2, seed=seed)
        rng = np.random.default_rng(7)
        out = [env.reset()["state"]]
        for _ in range(60):
            obs, rew, _, _ = env.step(random_actions(rng, 3, 4))
            out += [obs["state"], rew]
        return out

    a, b = rollout(5), rollout(5)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert not all(np.array_equal(x, y) for x, y in zip(a, rollout(6)))


def test_invalid_actions_treated_as_zero_and_bad_shape_rejected():
    env = MockVecEnv(n_envs=1, team_size=1, seed=0)
    env.reset()
    before = env.pos.copy(), env.yaw.copy()
    env.step(np.full((1, 2, 5), 9, dtype=np.uint8))
    assert np.array_equal(env.pos, before[0]) and np.array_equal(env.yaw, before[1])
    with pytest.raises(ValueError):
        env.step(np.zeros((1, 3, 5), dtype=np.uint8))


def test_adapter_normalises_list_infos():
    dones = np.array([True, False])
    infos = normalise_infos([{"winner": 1, "episode_len": 9}, {}], 2, dones)
    assert infos["winner"].tolist() == [1, -1] and infos["episode_len"].tolist() == [9, 0]
    env = VecEnvAdapter(MockVecEnv(n_envs=2, team_size=1), 2, 1)
    obs = env.reset()
    assert set(obs) == {"tokens", "token_mask", "local_map", "state", "alive", "action_availability"}
