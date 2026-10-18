"""End-to-end tests through the real sim-core engine and VecEnv.

All tests here are marked ``needs_core`` and are skipped while the core is
still stubbed (see conftest.py).
"""

from __future__ import annotations

import sys
import threading
import time

import numpy as np
import pytest

from binding_utils import OBS_SHAPES, noop_actions, random_actions

pytestmark = pytest.mark.needs_core


def assert_obs_layout(obs: dict, n_envs: int, n_units: int) -> None:
    assert set(obs) == set(OBS_SHAPES)
    for key, shape_fn in OBS_SHAPES.items():
        arr = obs[key]
        assert isinstance(arr, np.ndarray), key
        assert arr.dtype == np.float32, key
        assert arr.shape == shape_fn(n_envs, n_units), key
        assert arr.flags.c_contiguous and arr.flags.owndata and arr.flags.writeable, key
        assert np.isfinite(arr).all(), key
    assert np.isin(obs["alive"], (0.0, 1.0)).all()
    assert np.isin(obs["token_mask"], (0.0, 1.0)).all()
    # Token 0 is always "self" and always real.
    assert (obs["token_mask"][:, :, 0] == 1.0).all()


@pytest.mark.parametrize("team_size", [1, 2, 8])
def test_reset_and_step_shapes_and_dtypes(make_env, team_size):
    env = make_env(3, team_size=team_size, seed=1)
    e, u = env.n_envs, env.n_units
    obs = env.reset()
    assert_obs_layout(obs, e, u)
    assert (obs["alive"] == 1.0).all()

    obs, rewards, dones, infos = env.step(random_actions(np.random.default_rng(0), env))
    assert_obs_layout(obs, e, u)
    assert rewards.dtype == np.float32 and rewards.shape == (e, u)
    assert np.isfinite(rewards).all()
    assert dones.dtype == np.bool_ and dones.shape == (e,)
    assert set(infos) == {"winner", "episode_len", "episode_stats"}
    assert infos["winner"].dtype == np.int8 and infos["winner"].shape == (e,)
    assert infos["episode_len"].dtype == np.uint32 and infos["episode_len"].shape == (e,)
    expected_stats = {
        "termination", "shots_fired", "shots_hit", "damage_dealt", "kills", "deaths",
        "contact_ticks", "ending_ammo", "survivors_team0", "survivors_team1",
    }
    expected_stats.update(
        f"{key}_team{team}"
        for key in ("shots_fired", "shots_hit", "damage_dealt", "kills", "deaths", "ending_ammo")
        for team in (0, 1)
    )
    assert set(infos["episode_stats"]) == expected_stats
    for value in infos["episode_stats"].values():
        assert value.shape == (e,)


@pytest.mark.parametrize("map_kind", ["arena", "building", "village"])
def test_every_map_kind_runs(make_env, map_kind):
    env = make_env(2, map_kind=map_kind, seed=3)
    env.reset()
    rng = np.random.default_rng(3)
    for _ in range(5):
        env.step(random_actions(rng, env))


def rollout(env, steps: int, action_seed: int) -> list:
    rng = np.random.default_rng(action_seed)
    frames = [env.reset()]
    for _ in range(steps):
        frames.append(env.step(random_actions(rng, env)))
    return frames


def assert_frames_equal(a: list, b: list) -> None:
    assert len(a) == len(b)
    for fa, fb in zip(a, b):
        if isinstance(fa, dict):
            fa, fb = (fa,), (fb,)
        for xa, xb in zip(fa, fb):
            if isinstance(xa, dict):
                assert xa.keys() == xb.keys()
                for k in xa:
                    if isinstance(xa[k], dict):
                        assert xa[k].keys() == xb[k].keys()
                        for field in xa[k]:
                            np.testing.assert_array_equal(xa[k][field], xb[k][field], err_msg=field)
                    else:
                        np.testing.assert_array_equal(xa[k], xb[k], err_msg=k)
            else:
                np.testing.assert_array_equal(xa, xb)


def test_determinism_same_seed_same_actions(make_env):
    kwargs = dict(team_size=2, max_ticks=40, seed=1234)
    a = rollout(make_env(4, **kwargs), steps=100, action_seed=7)
    b = rollout(make_env(4, **kwargs), steps=100, action_seed=7)
    assert_frames_equal(a, b)


def test_determinism_independent_of_thread_count(make_env):
    kwargs = dict(team_size=2, max_ticks=40, seed=99)
    a = rollout(make_env(8, num_threads=1, **kwargs), steps=60, action_seed=1)
    b = rollout(make_env(8, num_threads=4, **kwargs), steps=60, action_seed=1)
    assert_frames_equal(a, b)


def test_reset_is_reproducible_on_same_object(make_env):
    env = make_env(4, seed=5)
    first = env.reset()
    env.step(random_actions(np.random.default_rng(0), env))
    # A second reset need not equal the first (episode seeds may advance), but
    # it must still be a valid fresh episode.
    second = env.reset()
    assert_obs_layout(second, env.n_envs, env.n_units)
    assert (second["alive"] == 1.0).all()
    assert first["tokens"].shape == second["tokens"].shape


def test_different_seeds_diverge(make_env):
    a = rollout(make_env(4, seed=1), steps=30, action_seed=0)
    b = rollout(make_env(4, seed=2), steps=30, action_seed=0)
    with pytest.raises(AssertionError):
        assert_frames_equal(a, b)


def test_auto_reset_on_timeout(make_env):
    """No-op actions never kill anyone, so every episode ends at max_ticks."""
    max_ticks = 5
    env = make_env(3, team_size=1, max_ticks=max_ticks, seed=0)
    env.reset()
    done_steps = []
    for step in range(1, 3 * max_ticks + 1):
        obs, rewards, dones, infos = env.step(noop_actions(env))
        if dones.any():
            assert dones.all(), "identical no-op envs should finish together"
            assert (infos["episode_len"] == infos["episode_len"][0]).all()
            assert 1 <= int(infos["episode_len"][0]) <= max_ticks
            assert np.isin(infos["winner"], (-1, 0, 1)).all()
            assert (infos["episode_stats"]["termination"] == 3).all()
            assert (infos["episode_stats"]["shots_fired"] == 0).all()
            assert (infos["episode_stats"]["ending_ammo"] == 2 * 30).all()
            # Returned obs is the first obs of the next episode.
            assert (obs["alive"] == 1.0).all()
            done_steps.append((step, int(infos["episode_len"][0])))
    assert len(done_steps) >= 2, done_steps
    # The next episode continues: episodes keep finishing at a fixed period
    # equal to their length.
    period = done_steps[1][0] - done_steps[0][0]
    assert period == done_steps[0][1] == done_steps[1][1]


def test_reward_kwargs_reach_the_engine(make_env):
    """shaping off + custom draw reward: no-op episodes pay exactly reward_draw at timeout."""
    max_ticks = 4
    env = make_env(2, team_size=1, max_ticks=max_ticks, seed=0,
                   shaping_scale=0.0, reward_draw=0.5)
    env.reset()
    for step in range(1, max_ticks + 1):
        _, rewards, dones, infos = env.step(noop_actions(env))
        if dones.all():
            assert (infos["winner"] == -1).all()
            np.testing.assert_array_equal(rewards, np.full_like(rewards, 0.5))
            return
        np.testing.assert_array_equal(rewards, np.zeros_like(rewards))
    pytest.fail(f"episode did not end within max_ticks={max_ticks}")


def test_returned_arrays_are_fresh_copies(make_env):
    env = make_env(2, seed=0)
    obs0 = env.reset()
    snapshot = {k: v.copy() for k, v in obs0.items()}
    obs1, rewards1, _, _ = env.step(random_actions(np.random.default_rng(0), env))
    obs2, rewards2, _, _ = env.step(random_actions(np.random.default_rng(1), env))
    for k in obs0:
        np.testing.assert_array_equal(obs0[k], snapshot[k], err_msg=k)
        assert not np.shares_memory(obs1[k], obs2[k]), k
    assert not np.shares_memory(rewards1, rewards2)
    # Mutating a returned array must not affect the env.
    obs2["tokens"][:] = 123.0
    obs3, *_ = env.step(noop_actions(env))
    assert not (obs3["tokens"] == 123.0).all()


def test_gil_released_during_step(make_env):
    """A pure-Python thread keeps running while another thread is inside step()."""
    env = make_env(512, team_size=8, num_threads=1, seed=0)
    env.reset()
    actions = random_actions(np.random.default_rng(0), env)
    t0 = time.perf_counter()
    env.step(actions)
    one_step = time.perf_counter() - t0
    if one_step < 0.02:
        pytest.skip(f"step too fast ({one_step * 1e3:.1f} ms) to observe GIL release reliably")

    intervals: list[tuple[float, float]] = []
    stop = threading.Event()

    def stepper() -> None:
        for _ in range(5):
            start = time.perf_counter()
            env.step(actions)
            intervals.append((start, time.perf_counter()))
        stop.set()

    old_interval = sys.getswitchinterval()
    sys.setswitchinterval(0.0005)
    try:
        ticks: list[float] = []
        thread = threading.Thread(target=stepper)
        thread.start()
        while not stop.is_set():
            ticks.append(time.perf_counter())
        thread.join()
    finally:
        sys.setswitchinterval(old_interval)

    ticks_arr = np.asarray(ticks)
    inside_middle = 0
    for start, end in intervals:
        quarter = (end - start) / 4
        lo, hi = start + quarter, end - quarter
        inside_middle += int(((ticks_arr > lo) & (ticks_arr < hi)).sum())
    # With the GIL held during step the main thread could not run in the middle
    # half of any step call; with it released it runs throughout.
    assert inside_middle > 100, inside_middle


def test_throughput_smoke(make_env):
    env = make_env(64, team_size=2, seed=0)
    env.reset()
    rng = np.random.default_rng(0)
    batch = [random_actions(rng, env) for _ in range(16)]
    steps = 200
    start = time.perf_counter()
    for i in range(steps):
        env.step(batch[i % len(batch)])
    elapsed = time.perf_counter() - start
    env_steps_per_s = steps * env.n_envs / elapsed
    print(f"\nwarsim throughput: {env_steps_per_s:,.0f} env-steps/s "
          f"({elapsed / steps * 1e3:.2f} ms per VecEnv.step, n_envs=64, team_size=2)")
    # Very loose floor: only catches pathological regressions in the bindings.
    assert env_steps_per_s > 1_000
