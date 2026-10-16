"""Shared fixtures for the `warsim` binding tests.

Tests marked ``needs_core`` call into sim-core's VecEnv/Env. While those are
still ``todo!()`` stubs they are skipped (probed once, in a subprocess, so a
hard crash in the core can never take down the test session). Set
``WARSIM_REQUIRE_CORE=1`` to turn that skip into a failure (e.g. in CI once the
core is merged).
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

import warsim

_PROBE = """
import numpy as np, warsim
env = warsim.VecEnv(2, team_size=1, max_ticks=5, seed=0)
env.reset()
env.step(np.zeros((2, env.n_units, 5), dtype=np.uint8))
"""


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "needs_core: needs the real sim-core engine/VecEnv (skipped while stubbed)"
    )


def _probe_core() -> tuple[bool, str]:
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE], capture_output=True, text=True, timeout=300
    )
    if proc.returncode == 0:
        return True, ""
    tail = (proc.stderr.strip().splitlines() or ["<no output>"])[-1]
    return False, f"sim-core not usable yet (exit {proc.returncode}): {tail}"


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if not any(item.get_closest_marker("needs_core") for item in items):
        return
    ok, reason = _probe_core()
    if ok:
        return
    if os.environ.get("WARSIM_REQUIRE_CORE") == "1":
        raise pytest.UsageError(f"WARSIM_REQUIRE_CORE=1 but {reason}")
    skip = pytest.mark.skip(reason=reason)
    for item in items:
        if item.get_closest_marker("needs_core"):
            item.add_marker(skip)


@pytest.fixture
def make_env():
    """Factory for VecEnvs that are closed at teardown."""
    envs: list[warsim.VecEnv] = []

    def _make(n_envs: int = 4, **kwargs) -> warsim.VecEnv:
        env = warsim.VecEnv(n_envs, **kwargs)
        envs.append(env)
        return env

    yield _make
    for env in envs:
        env.close()
