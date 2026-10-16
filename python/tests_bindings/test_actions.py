"""Validation of the `actions` argument of VecEnv.step.

Validation runs before sim-core is touched (and before the reset check), so
these tests work even while the core is stubbed.
"""

import numpy as np
import pytest

from binding_utils import noop_actions


@pytest.fixture
def env(make_env):
    return make_env(3, team_size=2)


@pytest.mark.parametrize("dtype", [np.int64, np.int32, np.int8, np.uint16, np.float32, np.bool_])
def test_wrong_dtype_raises_type_error(env, dtype):
    actions = noop_actions(env).astype(dtype)
    with pytest.raises(TypeError, match="uint8"):
        env.step(actions)


@pytest.mark.parametrize("bad", [[[0, 0, 0, 0, 0]], (0, 1), None, 3, "actions"])
def test_non_array_raises_type_error(env, bad):
    with pytest.raises(TypeError, match="numpy.ndarray"):
        env.step(bad)


@pytest.mark.parametrize(
    "shape",
    [(3, 4), (3, 4, 4), (3, 4, 6), (2, 4, 5), (4, 4, 5), (3, 3, 5), (3, 5, 5), (1, 3, 4, 5), ()],
)
def test_wrong_shape_raises_value_error(env, shape):
    with pytest.raises(ValueError, match="shape"):
        env.step(np.zeros(shape, dtype=np.uint8))


def test_non_contiguous_raises_value_error(env):
    big = np.zeros((env.n_envs * 2, env.n_units, 5), dtype=np.uint8)
    strided = big[::2]
    assert strided.shape == (env.n_envs, env.n_units, 5)
    assert not strided.flags.c_contiguous
    with pytest.raises(ValueError, match="contiguous"):
        env.step(strided)


def test_fortran_order_raises_value_error(env):
    actions = np.asfortranarray(noop_actions(env))
    assert not actions.flags.c_contiguous
    with pytest.raises(ValueError, match="contiguous"):
        env.step(actions)


@pytest.mark.parametrize(("head", "value"), [(0, 3), (1, 3), (2, 255), (3, 2), (4, 2)])
def test_out_of_range_raises_value_error_with_location(env, head, value):
    actions = noop_actions(env)
    actions[2, 1, head] = value
    with pytest.raises(ValueError, match=rf"actions\[2, 1, {head}\] = {value}"):
        env.step(actions)


def test_valid_actions_pass_validation(env):
    # Max legal value in every head: validation passes, then the reset check fires.
    actions = np.broadcast_to(
        np.asarray(env.action_dims, dtype=np.uint8) - 1, (env.n_envs, env.n_units, 5)
    ).copy()
    with pytest.raises(RuntimeError, match="reset"):
        env.step(actions)


def test_readonly_array_is_accepted(env):
    actions = noop_actions(env)
    actions.setflags(write=False)
    with pytest.raises(RuntimeError, match="reset"):
        env.step(actions)
