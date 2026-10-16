# War-Games-Sim

A fast, deterministic, headless tactical simulator in **Rust**, built as a
reinforcement-learning environment. Two teams of neural-network soldiers fight
in a tile-based city/village map (buildings, floors, doors, windows, stairs).
There are no scripted tactics: the AI learns from reward through self-play.

- `crates/sim-core` - the simulator (map, combat, observations, parallel VecEnv, replays)
- `crates/sim-py` - Python bindings (`warsim`)
- `python/` - MAPPO trainer, self-play league, evaluation
- `viewer/` - dependency-free browser replay viewer
- `infra/` - Podman image and scripts (test, train, serve the viewer)
- `docs/INTERFACE.md` - the contract between all components
- `docs/SETUP.md` - setup, daily workflow, architecture, troubleshooting

## Quick start

Everything builds and runs in Podman (host Rust linking is broken on Windows,
see `docs/SETUP.md`). From Git Bash or any bash:

```bash
bash infra/build-image.sh                 # once (and after Containerfile changes)
bash infra/test-all.sh                    # fmt, clippy, cargo test, sim-py + pytest
bash infra/run.sh cargo test -p sim-core  # or run any single command in the image
bash infra/train.sh --config mock_small --name exp1   # train; output in volume warsim-runs
bash infra/viewer.sh                      # http://localhost:8000/viewer/
```

Or just open `viewer/index.html` in a browser: it plays a bundled demo replay
and accepts replay JSON files by drag-and-drop.
