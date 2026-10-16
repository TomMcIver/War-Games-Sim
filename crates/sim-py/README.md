# sim-py: the `warsim` Python module

PyO3 + rust-numpy bindings for `sim_core::VecEnv` (see `docs/INTERFACE.md`,
"Python API"). Built with maturin as an abi3 wheel for CPython >= 3.11.

## Build and test (Podman only)

Host Rust/Python are not used. Every command runs in the dev image via
`infra/run.sh` (build the image once with `bash infra/build-image.sh`).

The containers are `--rm`, so `maturin develop` installs into a throwaway venv.
Build and use the module **in the same** `run.sh` call:

```bash
# Build (release) + run the binding tests
bash infra/run.sh bash -c "cd crates/sim-py && maturin develop --release && pytest ../../python/tests_bindings -q"

# Build + run any Python code (e.g. the trainer) against the bindings
bash infra/run.sh bash -c "cd crates/sim-py && maturin develop --release && cd /work && python -m your.module"

# Lint (the dev image ships without clippy/rustfmt; add them per run)
bash infra/run.sh bash -c "rustup component add clippy rustfmt && cargo fmt -p sim-py -- --check && cargo clippy -p sim-py -- -D warnings"
```

To reuse a build across runs, build a wheel into the mounted checkout and
install it later:

```bash
bash infra/run.sh bash -c "cd crates/sim-py && maturin build --release -o /work/target/wheels"
bash infra/run.sh bash -c "pip install --force-reinstall /work/target/wheels/warsim-*.whl && python -c 'import warsim; print(warsim.__version__)'"
```

Tests that need the real engine are marked `needs_core`. They are skipped
automatically (with the reason shown under `-rs`) while sim-core's `VecEnv`/`Env`
are still `todo!()` stubs. Set `WARSIM_REQUIRE_CORE=1` to make that a hard failure:

```bash
bash infra/run.sh bash -c "cd crates/sim-py && maturin develop --release && WARSIM_REQUIRE_CORE=1 pytest ../../python/tests_bindings -q -rs"
```

## Usage

```python
import numpy as np
import warsim

env = warsim.VecEnv(n_envs=64, team_size=2, map_kind="arena",
                    max_ticks=600, seed=0, tick_seconds=0.1, num_threads=None,
                    shaping_scale=1.0)  # + other reward_* overrides, see below
obs = env.reset()                       # dict of float32 arrays
E, U = env.n_envs, env.n_units          # U = 2 * team_size
actions = np.zeros((E, U, 5), dtype=np.uint8)   # heads: env.action_dims == [3, 3, 3, 2, 2]
obs, rewards, dones, infos = env.step(actions)
env.close()                             # or: with warsim.VecEnv(...) as env: ...
```

| output | dtype | shape |
|---|---|---|
| `obs["tokens"]` | float32 | `[E, U, U, TOKEN_FEATS]` |
| `obs["token_mask"]` | float32 | `[E, U, U]` |
| `obs["local_map"]` | float32 | `[E, U, MAP_CHANNELS, LOCAL_VIEW, LOCAL_VIEW]` |
| `obs["state"]` | float32 | `[E, U * STATE_FEATS]` (critic only) |
| `obs["alive"]` | float32 | `[E, U]` |
| `rewards` | float32 | `[E, U]` |
| `dones` | bool | `[E]` |
| `infos["winner"]` | int8 | `[E]`, -1 draw/none else team id; valid where `dones` |
| `infos["episode_len"]` | uint32 | `[E]`, valid where `dones` |

Done envs are auto-reset: the obs returned for them is the first obs of the next
episode. Module constants: `TOKEN_FEATS, MAP_CHANNELS, LOCAL_VIEW, STATE_FEATS,
MAX_TEAM_SIZE, MAX_UNITS, ACTION_HEADS, ACTION_DIMS` (tuple), `MAP_KINDS`, `__version__`.
Type stubs ship in `warsim.pyi`.

## Behaviour notes

- **Arrays are never reused.** Every call returns freshly allocated, C-contiguous
  numpy arrays that own their memory (one memcpy from the Rust buffer each). It is
  safe to store them (e.g. in a rollout buffer) and to mutate them.
- **GIL** is released while sim-core runs `reset`/`step`, so other Python threads
  keep running. Calling methods of the *same* env from two threads at once raises
  `RuntimeError` (already borrowed) instead of racing.
- **Threads**: `num_threads=None` uses rayon's global pool (all cores, or
  `RAYON_NUM_THREADS`); an integer gives this env a dedicated pool of that size
  (`sim_core::VecEnv::set_threads`). Results are identical for any thread count.
- **Rewards**: keyword-only overrides of `sim_core::reward::RewardConfig`;
  `None` (default) keeps the core default: `reward_win`, `reward_loss`,
  `reward_draw` (any finite float), `reward_damage_dealt`, `reward_kill`,
  `reward_damage_taken`, `reward_time_penalty`, `reward_ammo_waste`,
  `shaping_scale` (finite, `>= 0`). E.g. `VecEnv(64, shaping_scale=0.0)` trains on
  the pure win/loss signal. They are fixed for the env's lifetime: to anneal
  `shaping_scale`, build a new `VecEnv` (sim-core has no setter yet).
- **Validation** (before sim-core is touched):
  - constructor: bad ranges raise `ValueError` (`n_envs` 1..2^20, `team_size`
    1..MAX_TEAM_SIZE, `map_kind` in `MAP_KINDS`, `max_ticks` 1..2^32-1,
    `seed` 0..2^64-1, `tick_seconds` finite > 0, `num_threads` 1..1024); wrong
    types or unknown keywords raise `TypeError`.
  - `step(actions)`: non-ndarray or dtype != uint8 -> `TypeError`; wrong shape,
    non-C-contiguous or any head value >= its `ACTION_DIMS` entry -> `ValueError`
    (message names the offending `[env, unit, head]`). Note the core itself would
    treat invalid values as 0; the bindings reject them instead.
  - `step()` before `reset()`, or any call after `close()` -> `RuntimeError`.
- **Panics**: the workspace release profile is `panic = "abort"`; `pyproject.toml`
  overrides it to `unwind` for this extension only, so a sim-core panic becomes a
  `RuntimeError` (and the env requires `reset()`) instead of killing Python.
