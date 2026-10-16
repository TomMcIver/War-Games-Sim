"""Module-level constants and metadata (no sim-core calls)."""

import re

import warsim


def test_version_is_semver_string():
    assert isinstance(warsim.__version__, str)
    assert re.fullmatch(r"\d+\.\d+\.\d+.*", warsim.__version__)


def test_shape_constants_match_interface():
    assert warsim.TOKEN_FEATS == 12
    assert warsim.MAP_CHANNELS == 4
    assert warsim.LOCAL_VIEW == 9
    assert warsim.STATE_FEATS == 10
    assert warsim.MAX_TEAM_SIZE == 8
    assert warsim.MAX_UNITS == 16
    assert warsim.ACTION_HEADS == 5


def test_action_dims_constant_is_immutable_tuple():
    assert warsim.ACTION_DIMS == (3, 3, 3, 2, 2)
    assert isinstance(warsim.ACTION_DIMS, tuple)


def test_map_kinds_include_arena():
    assert "arena" in warsim.MAP_KINDS
    assert isinstance(warsim.MAP_KINDS, tuple)
