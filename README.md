# War-Games-Sim

A fast, deterministic, headless tactical simulator in **Rust**, built as a
reinforcement-learning environment. Two teams of neural-network soldiers fight
in a tile-based city/village map (buildings, floors, doors, windows, stairs).
There are no scripted tactics: the AI learns from reward through self-play.

- `crates/sim-core` - the simulator (map, combat, observations, parallel VecEnv)
- `crates/sim-py` - Python bindings (`warsim`)
- `python/` - MAPPO trainer, self-play league, evaluation
- `viewer/` - browser replay viewer
- `infra/` - Podman build/run tooling
- `docs/INTERFACE.md` - the contract between all components

Build and test in Podman: `bash infra/run.sh cargo test`
