# Interface contract

Every component codes against this document. If you must change it, change
it in the same PR as the code and call it out in the PR description.

## Goal
Two teams of AI soldiers fight in a tile-based 2.5D map (outdoors, buildings,
multiple floors, doors, windows, stairs). The AI is a neural network trained
with reinforcement learning (self-play). **No scripted tactics anywhere.**
The environment has rules (physics, combat, win condition); the policy has none.

## Units and indexing
- `U = 2 * team_size` units per env. Team 0 = units `0..n`, team 1 = `n..2n`.
- `team_size <= MAX_TEAM_SIZE = 8` (so `U <= MAX_UNITS = 16`).
- Tensors use the **real** `U` (not padded to 16). Per-agent token count is
  `T = U` exactly (self + n-1 allies + n enemies), so the only padding is for
  enemies the agent cannot currently see (masked out).

## Action (per unit, per tick) - multi-discrete `uint8[5]`
| head | dims | meaning |
|------|------|---------|
| forward | 3 | 0 none, 1 forward, 2 back |
| strafe | 3 | 0 none, 1 left, 2 right |
| turn | 3 | 0 none, 1 left, 2 right |
| shoot | 2 | 0/1 (hitscan along facing, needs ammo + cooldown ready) |
| interact | 2 | 0/1 (toggle adjacent door, use stairs when on them) |

Dead units' actions are ignored. Invalid values are treated as 0.

## Observation (per env, per unit) - fog of war
Everything below is `float32`, laid out `[n_envs, U, ...]`:
- `tokens`      `[E, U, T, F]`, T = `U`, F = `TOKEN_FEATS` (12).
  Token 0 = self; then visible/known allies, then visible enemies; padded.
  Features (all roughly in [-1, 1]; positions relative to self, rotated into
  self's frame): is_self, is_ally, is_enemy, rel_x, rel_y, same_floor/rel_floor,
  sin/cos of relative yaw, hp, ammo, cooldown, last-seen-age (0 if visible now).
- `token_mask`  `[E, U, T]`, 1.0 = real token, 0.0 = padding.
- `local_map`   `[E, U, C, V, V]`, C = `MAP_CHANNELS` (4), V = `LOCAL_VIEW` (9),
  rotated into the agent's frame. Channels: wall, door(open=0.5/closed=1.0),
  walkable floor, stairs/window. Tiles not in line of sight are 0 (fogged).
- `state`       `[E, S]`, S = `U * STATE_FEATS` (STATE_FEATS = 10 per unit, team 0 then team 1). Full-information
  battlefield for the **centralized critic only** (training). Never fed to actors.
- `alive`       `[E, U]` 1.0 / 0.0.

An enemy is only a token while the agent (or an ally, shared team vision)
has line of sight to it. Allies are always tokens while alive.

## Stairs and doors
`StairsUp` at `(x, y, f)` pairs with `StairsDown` at the **same `(x, y)` on
floor `f+1`**. Standing on a stairs tile with `interact=1` moves the unit to the
paired tile. `interact=1` next to a door (the tile in front) toggles it
closed/open. Windows block movement but not sight or bullets.

## Rust VecEnv API (sim-core::vec_env) - obs-agent implements, py-agent consumes
```rust
pub struct VecEnvBuffers {          // all flat, row-major, sized for the real U
    pub tokens: Vec<f32>,      // E*U*U*TOKEN_FEATS
    pub token_mask: Vec<f32>,  // E*U*U
    pub local_map: Vec<f32>,   // E*U*MAP_CHANNELS*LOCAL_VIEW*LOCAL_VIEW
    pub state: Vec<f32>,       // E*U*STATE_FEATS
    pub alive: Vec<f32>,       // E*U
    pub rewards: Vec<f32>,     // E*U
    pub dones: Vec<u8>,        // E
    pub winners: Vec<i8>,      // E  (-1 none/draw, else team id; valid when dones[e]==1)
    pub episode_len: Vec<u32>, // E  (valid when dones[e]==1)
}
impl VecEnv {
    pub fn new(n_envs: usize, cfg: EnvConfig, seed: u64) -> Self;
    pub fn n_units(&self) -> usize;                           // 2 * team_size
    pub fn reset(&mut self) -> &VecEnvBuffers;                // fills obs fields
    pub fn step(&mut self, actions: &[u8]) -> &VecEnvBuffers; // len E*U*5, auto-resets done envs
    pub fn buffers(&self) -> &VecEnvBuffers;
}
```
The stub already exists in `vec_env.rs` with `todo!()` bodies.

## Map JSON (map-agent writes, viewer reads)
`{"width":W,"height":H,"floors":F,"tiles":[...]}`; `tiles` is a flat array of
`Tile` u8 values (Void 0, Floor 1, Wall 2, DoorClosed 3, DoorOpen 4, Window 5,
StairsUp 6, StairsDown 7), index `(f*H + y)*W + x`.

## Replay JSON (infra-agent writes/reads)
`{"version":1,"config":{...},"seed":123,"map":<Map JSON>,"frames":[{"tick":0,
"units":[{id,team,x,y,floor,yaw,hp,alive}],"shots":[{shooter,x,y,floor,hit}]}]}`.
`World.shots` (filled by the engine each tick, cleared at the start of the next)
feeds `shots`.

## Reward and episode
- `rewards` `[E, U]` float32 per tick: team-shared terminal +1 win / -1 loss /
  0 draw, plus small shaping (damage dealt, kills, damage taken, per-step cost).
  Shaping weights come from `EnvConfig`/`RewardConfig` so they can be annealed to 0.
- `dones` `[E]` bool. Episode ends when a team is wiped out or `max_ticks`.
- VecEnv **auto-resets** done envs; the returned obs for a done env is the
  first obs of the new episode. `infos` carries `winner` (-1 draw/none, 0, 1)
  and `episode_len` for envs that just finished.

## Python API (module `warsim`, built from crates/sim-py)
```python
env = warsim.VecEnv(n_envs=64, team_size=2, map_kind="arena",
                    max_ticks=600, seed=0)
obs = env.reset()                    # dict of numpy arrays (see above)
obs, rewards, dones, infos = env.step(actions)   # actions: uint8 [E, U, 5]
env.close()
```
`env.n_units`, `env.action_dims == [3,3,3,2,2]` are exposed.
Envs are deterministic given `(seed, action stream)`.

## Replay
`replay.rs` writes JSON: `{config, seed, map, frames:[{tick, units:[...]}]}`;
`viewer/index.html` plays it in the browser (no server needed).

## Ownership (avoid merge conflicts - touch only your files)
| component | owner | files |
|---|---|---|
| map | map-agent | `crates/sim-core/src/map/**` |
| engine | engine-agent | `world.rs`, `env.rs` (+ `combat.rs`, `reward.rs` new) |
| obs/vec-env | obs-agent | `obs.rs`, `vec_env.rs`, `benches/` |
| python bindings | py-agent | `crates/sim-py/**` |
| RL trainer | rl-agent | `python/**` |
| infra/viewer | infra-agent | `infra/**`, `viewer/**`, `replay.rs`, `.github/**` |

## Build/test
Host Rust linking is broken on the dev laptop. **Always build/test in Podman:**
`bash infra/run.sh cargo test` (see infra/run.sh).
