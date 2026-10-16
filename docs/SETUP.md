# Setup and daily workflow

Everything (Rust, Python, maturin, PyTorch CPU) runs in one Podman dev image,
`localhost/warsim-dev`. The host only needs Podman, Git (Git Bash on Windows)
and a browser. Host Rust/Python toolchains are not used: see
[Troubleshooting](#troubleshooting) for why.

## Architecture and ownership

```
              docs/INTERFACE.md  (the contract every component codes against)
                         |
  crates/sim-core (Rust, deterministic, headless)
    map/        tile map, floors, doors, windows, stairs, LOS, generators   map-agent
    world.rs    units, movement, collisions, doors/stairs                   engine-agent
    env.rs      one match: reset/step, termination, rewards (+combat, reward) engine-agent
    obs.rs      fog-of-war observations, centralized-critic state           obs-agent
    vec_env.rs  E envs in parallel (rayon), flat buffers, auto-reset         obs-agent
    replay.rs   per-tick frames -> replay JSON                              infra-agent
         |                                   |
  crates/sim-py (PyO3 `warsim`)        viewer/ (static web page)            infra-agent
         |                              plays replay JSON in a browser
  python/warsim_rl (MAPPO, self-play league, eval)                          rl-agent
         |
  infra/ (Podman image + scripts), .github/workflows/ci.yml                 infra-agent
```

The AI is a neural network trained by self-play RL; there are no scripted
tactics anywhere. Each component owner touches only its own files (table in
`docs/INTERFACE.md`); contract changes go through `docs/INTERFACE.md` in the
same PR.

## One-time setup (Windows)

1. Install [Podman Desktop](https://podman-desktop.io/) or the Podman CLI, and
   Git for Windows (provides Git Bash). Run the commands below in Git Bash.
2. Create and start the Podman machine:

   ```bash
   podman machine init --cpus 8 --memory 16384 --disk-size 100
   podman machine start
   podman info --format '{{.Host.CPUs}} CPUs, {{.Host.MemTotal}} bytes RAM'
   ```

3. Give training enough CPU/RAM. How depends on the machine type
   (`podman machine list`, column *VM TYPE*):
   - **hyperv / qemu / applehv:** resize in place
     ```bash
     podman machine stop
     podman machine set --cpus 8 --memory 16384   # MiB
     podman machine start
     ```
   - **wsl** (the Windows default): `podman machine set --cpus/--memory` is not
     supported; the WSL 2 VM is sized by `%UserProfile%\.wslconfig`
     (default: all logical CPUs, half of RAM):
     ```ini
     [wsl2]
     processors=8
     memory=16GB
     ```
     then `wsl --shutdown` and `podman machine start`.
4. Build the dev image (about 2 GB; torch CPU is the bulk). Rebuild whenever
   `infra/Containerfile` changes:

   ```bash
   bash infra/build-image.sh
   ```

## Daily commands

All scripts work from Git Bash on Windows and from bash on Linux/macOS; they
mount the checkout at `/work` and keep caches in Podman volumes
(`warsim-cargo` for the registry, `warsim-target-<checkout>` for `target/`,
so each worktree builds incrementally).

| task | command |
|---|---|
| run anything in the image | `bash infra/run.sh <cmd...>` |
| Rust tests | `bash infra/run.sh cargo test` (`-p sim-core` for one crate) |
| format Rust | `bash infra/run.sh cargo fmt --all` |
| everything CI checks | `bash infra/test-all.sh` (`--rust-only` to skip Python) |
| Python bindings + tests | `bash infra/run.sh bash -c "cd crates/sim-py && maturin develop --release && pytest ../../python/tests_bindings -q"` |
| train | `bash infra/train.sh --config mock_small --name exp1` |
| view replays | `bash infra/viewer.sh` then http://localhost:8000/viewer/ |

`infra/test-all.sh` runs `infra/checks.sh` in the container: `cargo fmt
--check`, `cargo clippy --workspace --all-targets -D warnings`, `cargo test
--workspace`, then (when present) builds `crates/sim-py` with maturin and runs
`python/tests_bindings` and `python/tests_rl`. Missing components are
reported as skipped. It also runs the viewer's Node tests on the host if
`node` is installed. CI (`.github/workflows/ci.yml`) runs the same steps on
`ubuntu-latest` for every push to `main` and every PR.

## Training

```bash
bash infra/train.sh --config mock_small --name exp1           # foreground
bash infra/train.sh --config warsim_small --set env.n_envs=64 --resume auto
WARSIM_DETACH=1 bash infra/train.sh --config warsim_small --name long1
podman logs -f warsim-train                                   # follow a detached run
podman stop warsim-train
```

`train.sh` rebuilds the `warsim` bindings (incremental), puts `python/` on
`PYTHONPATH` and runs `python -m warsim_rl.train <args>`. Output goes to
`/runs` in the container, which is the persistent volume `warsim-runs`
(`--out-dir /runs` is added unless you pass your own). To write to a host
folder instead, set `WARSIM_RUNS_DIR`, e.g. `WARSIM_RUNS_DIR=./runs bash
infra/train.sh ...` (`runs/` is git-ignored).

Copy results out of the volume:

```bash
podman run --rm -v warsim-runs:/runs -v "$(pwd -W 2>/dev/null || pwd):/out" \
  localhost/warsim-dev cp -r /runs /out/runs-export
```

## Viewing replays

- Quick look: open `viewer/index.html` in a browser (works from `file://`),
  then drag a replay `.json` onto the page. The bundled demo loads by default.
- Served, with training replays: `bash infra/viewer.sh [port]` serves
  `viewer/` and the runs volume read-only on `127.0.0.1:<port>`:
  - `http://localhost:8000/viewer/?replay=/runs/<run>/<file>.json`
  - `http://localhost:8000/runs/` lists the volume.
- Recording a replay from Rust:
  ```rust
  let mut w = sim_core::replay::ReplayWriter::for_env(&env, seed);
  w.record(&env.world);                 // initial state, then after every step
  w.finish().write_to_file("/runs/exp1/replays/ep0.json")?;
  ```

Format, controls and URL parameters: [viewer/README.md](../viewer/README.md).

## Troubleshooting

**Host `cargo build` fails at link time on Windows (`link: extra operand`,
`link.exe` errors).** Git for Windows ships GNU coreutils' `link` as
`C:\Program Files\Git\usr\bin\link.exe`. In Git Bash that directory is early
on `PATH`, so rustc's MSVC target invokes Git's `link.exe` instead of the
Visual Studio linker and every binary/test link fails. This is why the project
builds only in Podman. If you must build on the host anyway: run cargo from a
"Developer PowerShell for VS" (not Git Bash), or point the MSVC target at the
real linker in `%UserProfile%\.cargo\config.toml`:

```toml
[target.x86_64-pc-windows-msvc]
linker = "C:\\Program Files\\Microsoft Visual Studio\\2022\\BuildTools\\VC\\Tools\\MSVC\\<version>\\bin\\Hostx64\\x64\\link.exe"
```

**`image localhost/warsim-dev missing`.** Run `bash infra/build-image.sh`.

**`cargo fmt`/`cargo clippy`: component not installed.** Your image predates
the `rustup component add rustfmt clippy` layer; rebuild it.

**Paths look like `C:/Program Files/Git/work`.** MSYS rewrote a container path.
The infra scripts export `MSYS_NO_PATHCONV=1`; do the same for raw `podman`
commands in Git Bash.

**`http://localhost:8000` does not answer.** Check `podman ps` shows
`warsim-viewer`; try `http://127.0.0.1:8000`; pick another port with
`bash infra/viewer.sh 8080` if 8000 is taken.

**`the input device is not a TTY`.** Don't add `-it` to podman in Git Bash's
mintty; use `winpty podman ...` for an interactive shell, e.g.
`winpty bash infra/run.sh bash`.

**Slow first build / disk full.** The first build fills the cargo volumes.
Inspect with `podman system df`; drop a stale worktree's cache with
`podman volume rm warsim-target-<name>`.
