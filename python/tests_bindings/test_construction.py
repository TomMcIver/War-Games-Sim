"""Constructor validation, properties and lifecycle (no sim-core stepping)."""

import math

import numpy as np
import pytest

import warsim
from binding_utils import noop_actions


def test_defaults_and_properties(make_env):
    env = make_env(8)
    assert env.n_envs == 8
    assert env.team_size == 2
    assert env.n_units == 4
    assert env.action_dims == [3, 3, 3, 2, 2]
    assert isinstance(env.action_dims, list)
    assert env.closed is False


@pytest.mark.parametrize("team_size", [1, 3, warsim.MAX_TEAM_SIZE])
def test_n_units_is_twice_team_size(make_env, team_size):
    env = make_env(2, team_size=team_size)
    assert env.n_units == 2 * team_size


def test_all_keyword_arguments_accepted(make_env):
    env = make_env(
        n_envs=3,
        team_size=1,
        map_kind="arena",
        max_ticks=10,
        seed=2**64 - 1,
        tick_seconds=0.05,
        num_threads=2,
    )
    assert env.n_envs == 3
    assert "num_threads=2" in repr(env)


def test_numpy_integer_arguments_accepted(make_env):
    env = make_env(np.int64(2), team_size=np.int32(2), seed=np.uint64(7))
    assert env.n_envs == 2


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n_envs": 0}, "n_envs"),
        ({"n_envs": -1}, "n_envs"),
        ({"n_envs": 2**40}, "n_envs"),
        ({"team_size": 0}, "team_size"),
        ({"team_size": warsim.MAX_TEAM_SIZE + 1}, "team_size"),
        ({"map_kind": "moon"}, "map_kind"),
        ({"map_kind": ""}, "map_kind"),
        ({"max_ticks": 0}, "max_ticks"),
        ({"max_ticks": 2**32}, "max_ticks"),
        ({"seed": -1}, "seed"),
        ({"seed": 2**64}, "seed"),
        ({"tick_seconds": 0.0}, "tick_seconds"),
        ({"tick_seconds": -0.1}, "tick_seconds"),
        ({"tick_seconds": math.nan}, "tick_seconds"),
        ({"tick_seconds": math.inf}, "tick_seconds"),
        ({"num_threads": 0}, "num_threads"),
    ],
)
def test_invalid_values_raise_value_error(kwargs, match):
    args = {"n_envs": 2, **kwargs}
    with pytest.raises(ValueError, match=match):
        warsim.VecEnv(**args)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_envs": "4"},
        {"n_envs": 2.5},
        {"n_envs": None},
        {"team_size": 1.0},
        {"map_kind": 3},
        {"tick_seconds": "fast"},
        {"num_threads": 1.5},
    ],
)
def test_wrong_types_raise_type_error(kwargs):
    args = {"n_envs": 2, **kwargs}
    with pytest.raises(TypeError):
        warsim.VecEnv(**args)


def test_unknown_keyword_raises_type_error():
    with pytest.raises(TypeError):
        warsim.VecEnv(2, not_a_real_option=1)


def test_missing_n_envs_raises_type_error():
    with pytest.raises(TypeError):
        warsim.VecEnv()


def test_close_is_idempotent_and_blocks_further_use(make_env):
    env = make_env(2)
    env.close()
    env.close()
    assert env.closed is True
    assert "closed" in repr(env)
    with pytest.raises(RuntimeError, match="closed"):
        env.reset()
    with pytest.raises(RuntimeError, match="closed"):
        env.step(noop_actions(env))
    # Static properties stay readable after close.
    assert env.n_units == 4


def test_context_manager_closes(make_env):
    with warsim.VecEnv(2) as env:
        assert env.closed is False
    assert env.closed is True


def test_step_before_reset_raises(make_env):
    env = make_env(2)
    with pytest.raises(RuntimeError, match="reset"):
        env.step(noop_actions(env))
