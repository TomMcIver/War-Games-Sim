//! OWNER: obs-agent. Observation encoding with fog of war (an agent only
//! sees what has line of sight). Writes into caller-provided flat f32 buffers.
//! Layouts are documented in docs/INTERFACE.md and in the item docs below.
//!
//! Everything here is allocation-free: scratch state lives on the stack
//! (fixed `MAX_UNITS` arrays) or in caller-owned structs ([`ObsMemory`]).
//!
//! # Frames
//! The agent frame has its origin at the agent's position. Axis "forward"
//! is the facing direction `(cos yaw, sin yaw)`; axis "lateral" is facing
//! rotated by +90 degrees, `(-sin yaw, cos yaw)`, i.e. the agent's "left"
//! (the engine's turn-left / strafe-left direction, see world.rs). With
//! `yaw == 0` the agent frame equals the world frame.
//!
//! # Entity tokens (`[T, TOKEN_FEATS]`, `T = U = world.units.len()`)
//! Slot order: token 0 = self, then alive allies (unit-index order), then
//! enemies the team currently sees (index order), then enemies the team
//! remembers (seen within the last [`MEMORY_TICKS`] ticks; only when an
//! [`ObsMemory`] is supplied), then zero padding with mask 0. Dead units
//! (allies or enemies) never get a token. A dead agent only gets its own
//! token (hp 0) so attention never sees an all-masked row.
//! Features: see the `F_*` index constants.
//!
//! # Local map (`[MAP_CHANNELS, LOCAL_VIEW, LOCAL_VIEW]`)
//! `local_map[ch][row][col]` samples the world point
//! `pos + (col - H) * forward + (row - H) * lateral` (`H = LOCAL_VIEW / 2`) on
//! the agent's floor, so the agent sits at the centre cell facing `+col`.
//! With `yaw == 0` the patch is the world grid (`row` = y, `col` = x).
//! Cells without line of sight from the agent are all-zero (fogged).
//! Channels: see the `C_*` constants.

use crate::map::{GameMap, Tile};
use crate::types::*;
use crate::world::{Unit, World, DEFAULT_AMMO, DEFAULT_MAX_HP};

// ---- buffer sizes ---------------------------------------------------------

/// Floats per agent for entity tokens at the maximum unit count. Real
/// buffers use [`tokens_len`] (`U * TOKEN_FEATS`).
pub const TOKENS_LEN: usize = MAX_TOKENS * TOKEN_FEATS;
/// Floats per agent for the local map patch.
pub const LOCAL_MAP_LEN: usize = MAP_CHANNELS * LOCAL_VIEW * LOCAL_VIEW;
/// Global-state floats at the maximum unit count. Real buffers use
/// [`state_len`] (`U * STATE_FEATS`).
pub const STATE_LEN: usize = MAX_UNITS * STATE_FEATS;

/// Floats per agent for entity tokens with `n_units` units (T = U).
pub const fn tokens_len(n_units: usize) -> usize {
    n_units * TOKEN_FEATS
}

/// Floats of centralized-critic state with `n_units` units.
pub const fn state_len(n_units: usize) -> usize {
    n_units * STATE_FEATS
}

// ---- normalization --------------------------------------------------------

/// Relative positions (tiles) are divided by this and clamped to [-1, 1].
pub const POS_NORM: f32 = 32.0;
/// Relative floor difference is divided by this and clamped to [-1, 1].
pub const FLOOR_NORM: f32 = 2.0;
/// Hit points are divided by this (fraction of full health).
pub const HP_NORM: f32 = DEFAULT_MAX_HP;
/// Ammo is divided by this and clamped to [0, 1] (full magazine = 1).
pub const AMMO_NORM: f32 = DEFAULT_AMMO as f32;
/// Weapon cooldown ticks are divided by this and clamped to [0, 1]
/// (matches the default `WeaponStats::cooldown_ticks`).
pub const COOLDOWN_NORM: f32 = 3.0;
/// Enemies stay in team memory this many ticks after last being seen;
/// last-seen-age is `age_ticks / MEMORY_TICKS` (saturating at 1).
pub const MEMORY_TICKS: u64 = 50;

// ---- token feature indices --------------------------------------------------

/// 1.0 for the agent's own token.
pub const F_IS_SELF: usize = 0;
/// 1.0 for an ally token (not self).
pub const F_IS_ALLY: usize = 1;
/// 1.0 for an enemy token.
pub const F_IS_ENEMY: usize = 2;
/// Position along the agent's facing direction / POS_NORM.
pub const F_REL_X: usize = 3;
/// Position along facing rotated +90 deg / POS_NORM.
pub const F_REL_Y: usize = 4;
/// (floor - own floor) / FLOOR_NORM.
pub const F_REL_FLOOR: usize = 5;
/// sin(yaw - own yaw).
pub const F_SIN_YAW: usize = 6;
/// cos(yaw - own yaw).
pub const F_COS_YAW: usize = 7;
/// hp / HP_NORM.
pub const F_HP: usize = 8;
/// ammo / AMMO_NORM (0 for enemies: not observable).
pub const F_AMMO: usize = 9;
/// cooldown / COOLDOWN_NORM (0 for enemies: not observable).
pub const F_COOLDOWN: usize = 10;
/// 0 if currently seen, else ticks since last seen / MEMORY_TICKS.
pub const F_LAST_SEEN: usize = 11;

// ---- local map channels -----------------------------------------------------

/// Wall or void (opaque, impassable): 1.0.
pub const C_WALL: usize = 0;
/// Door: open 0.5, closed 1.0.
pub const C_DOOR: usize = 1;
/// Walkable (floor, open door, stairs): 1.0.
pub const C_WALKABLE: usize = 2;
/// Stairs up 1.0, stairs down -1.0, window 0.5.
pub const C_STAIRS_WINDOW: usize = 3;

// ---- global state layout ----------------------------------------------------

/// Global-state per-unit offsets (`out[unit * STATE_FEATS + S_*]`).
pub const S_X: usize = 0;
pub const S_Y: usize = 1;
pub const S_FLOOR: usize = 2;
pub const S_SIN_YAW: usize = 3;
pub const S_COS_YAW: usize = 4;
pub const S_HP: usize = 5;
pub const S_AMMO: usize = 6;
pub const S_COOLDOWN: usize = 7;
pub const S_ALIVE: usize = 8;
pub const S_TEAM: usize = 9;

const N_TEAMS: usize = 2;

// ---- shared team vision -----------------------------------------------------

/// Which units each team can currently see. A team sees its own alive units
/// and every alive enemy that at least one alive teammate has line of sight
/// to (`GameMap::line_of_sight`). Compute once per tick and share between
/// all agents of an env: `U^2 / 4` LOS checks instead of `U^3 / 4`.
#[derive(Clone, Copy, Debug)]
pub struct TeamVision {
    seen: [[bool; MAX_UNITS]; N_TEAMS],
}

impl TeamVision {
    pub fn compute(world: &World) -> Self {
        let units = &world.units;
        assert!(
            units.len() <= MAX_UNITS,
            "at most {MAX_UNITS} units supported"
        );
        let mut seen = [[false; MAX_UNITS]; N_TEAMS];
        for (i, a) in units.iter().enumerate() {
            if !a.alive || a.team as usize >= N_TEAMS {
                continue;
            }
            seen[a.team as usize][i] = true;
            for (j, b) in units.iter().enumerate().skip(i + 1) {
                if !b.alive || b.team == a.team || b.team as usize >= N_TEAMS {
                    continue;
                }
                let both_known = seen[a.team as usize][j] && seen[b.team as usize][i];
                if !both_known && world.map.line_of_sight(a.pos, b.pos) {
                    seen[a.team as usize][j] = true;
                    seen[b.team as usize][i] = true;
                }
            }
        }
        Self { seen }
    }

    /// Does `team` currently see unit `unit`?
    pub fn sees(&self, team: TeamId, unit: UnitId) -> bool {
        (team as usize) < N_TEAMS && unit < MAX_UNITS && self.seen[team as usize][unit]
    }
}

// ---- team memory ------------------------------------------------------------

#[derive(Clone, Copy, Debug, Default)]
struct Sighting {
    pos: Pos,
    yaw: f32,
    hp: f32,
    tick: u64,
    valid: bool,
}

/// Per-env, per-team memory of where enemies were last seen. Owned by the
/// caller (one per env in `VecEnv`); `clear` on episode reset, `update` once
/// per tick before writing observations. Fixed size, no heap.
#[derive(Clone, Copy, Debug, Default)]
pub struct ObsMemory {
    last: [[Sighting; MAX_UNITS]; N_TEAMS],
}

impl ObsMemory {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn clear(&mut self) {
        *self = Self::default();
    }

    /// Record every enemy each team sees right now.
    pub fn update(&mut self, world: &World, vision: &TeamVision) {
        for team in 0..N_TEAMS {
            for (i, u) in world.units.iter().enumerate() {
                if u.alive && u.team as usize != team && vision.sees(team as TeamId, i) {
                    self.last[team][i] = Sighting {
                        pos: u.pos,
                        yaw: u.yaw,
                        hp: u.hp,
                        tick: world.tick,
                        valid: true,
                    };
                }
            }
        }
    }

    /// Age in ticks of the team's last sighting of `unit`, if still remembered.
    fn recall(&self, team: TeamId, unit: UnitId, now: u64) -> Option<(Sighting, u64)> {
        let s = *self.last.get(team as usize)?.get(unit)?;
        let age = now.saturating_sub(s.tick);
        (s.valid && age < MEMORY_TICKS).then_some((s, age))
    }
}

// ---- agent observation ------------------------------------------------------

/// Simple entry point: fill `tokens` (`U * TOKEN_FEATS`), `token_mask` (`U`,
/// 1.0 = valid) and `local_map` (`LOCAL_MAP_LEN`) for `unit`, with shared
/// team vision but no memory (unseen enemies are simply absent).
/// Computes [`TeamVision`] itself; when encoding many agents of one world
/// use [`write_agent_obs_with`] or [`write_env_obs`] instead.
pub fn write_agent_obs(
    world: &World,
    unit: UnitId,
    tokens: &mut [f32],
    token_mask: &mut [f32],
    local_map: &mut [f32],
) {
    let vision = TeamVision::compute(world);
    write_agent_obs_with(world, unit, &vision, None, tokens, token_mask, local_map);
}

/// Full entry point. `memory`, when given, adds remembered (recently seen,
/// currently unseen) enemies as tokens at their last known state with a
/// non-zero last-seen-age.
pub fn write_agent_obs_with(
    world: &World,
    unit: UnitId,
    vision: &TeamVision,
    memory: Option<&ObsMemory>,
    tokens: &mut [f32],
    token_mask: &mut [f32],
    local_map: &mut [f32],
) {
    let n = world.units.len();
    assert!(n <= MAX_UNITS, "at most {MAX_UNITS} units supported");
    assert!(unit < n, "unit {unit} out of range ({n} units)");
    assert_eq!(tokens.len(), tokens_len(n), "tokens buffer length");
    assert_eq!(token_mask.len(), n, "token_mask buffer length");
    assert_eq!(local_map.len(), LOCAL_MAP_LEN, "local_map buffer length");

    tokens.fill(0.0);
    token_mask.fill(0.0);
    local_map.fill(0.0);

    let me = &world.units[unit];
    let frame = Frame::of(me);
    let mut slot = 0;
    let mut push = |view: TokenView, tokens: &mut [f32]| {
        frame.encode(
            &view,
            &mut tokens[slot * TOKEN_FEATS..(slot + 1) * TOKEN_FEATS],
        );
        token_mask[slot] = 1.0;
        slot += 1;
    };

    push(TokenView::live(me, Kind::SelfUnit), tokens);
    if !me.alive {
        return;
    }
    for (i, u) in world.units.iter().enumerate() {
        if i != unit && u.alive && u.team == me.team {
            push(TokenView::live(u, Kind::Ally), tokens);
        }
    }
    for (i, u) in world.units.iter().enumerate() {
        if u.alive && u.team != me.team && vision.sees(me.team, i) {
            push(TokenView::live(u, Kind::Enemy), tokens);
        }
    }
    if let Some(mem) = memory {
        for (i, u) in world.units.iter().enumerate() {
            if !u.alive || u.team == me.team || vision.sees(me.team, i) {
                continue;
            }
            if let Some((s, age)) = mem.recall(me.team, i, world.tick) {
                push(TokenView::remembered(&s, age), tokens);
            }
        }
    }

    write_local_map(&world.map, me, &frame, local_map);
}

/// Encode every agent of one env plus its critic state and alive flags.
/// Updates `memory` with this tick's sightings first. Buffers are the env's
/// slices of the `VecEnvBuffers` fields (`U*U*TOKEN_FEATS`, `U*U`,
/// `U*LOCAL_MAP_LEN`, `U*STATE_FEATS`, `U`).
pub fn write_env_obs(
    world: &World,
    memory: &mut ObsMemory,
    tokens: &mut [f32],
    token_mask: &mut [f32],
    local_map: &mut [f32],
    state: &mut [f32],
    alive: &mut [f32],
) {
    let n = world.units.len();
    assert!(n > 0, "world has no units");
    assert_eq!(tokens.len(), n * tokens_len(n), "env tokens buffer length");
    assert_eq!(token_mask.len(), n * n, "env token_mask buffer length");
    assert_eq!(
        local_map.len(),
        n * LOCAL_MAP_LEN,
        "env local_map buffer length"
    );
    assert_eq!(alive.len(), n, "alive buffer length");
    let vision = TeamVision::compute(world);
    memory.update(world, &vision);
    let (lmaps, _) = local_map.as_chunks_mut::<LOCAL_MAP_LEN>();
    for (u, ((tok, mask), lmap)) in tokens
        .chunks_exact_mut(tokens_len(n))
        .zip(token_mask.chunks_exact_mut(n))
        .zip(lmaps)
        .enumerate()
    {
        write_agent_obs_with(world, u, &vision, Some(memory), tok, mask, lmap);
    }
    for (a, u) in alive.iter_mut().zip(&world.units) {
        *a = flag(u.alive);
    }
    write_global_state(world, state);
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    SelfUnit,
    Ally,
    Enemy,
}

/// What the agent knows about one entity (live or remembered).
struct TokenView {
    kind: Kind,
    pos: Pos,
    yaw: f32,
    hp: f32,
    ammo: f32,
    cooldown: f32,
    age_ticks: u64,
}

impl TokenView {
    fn live(u: &Unit, kind: Kind) -> Self {
        // Enemy ammo/cooldown are not observable.
        let (ammo, cooldown) = if kind == Kind::Enemy {
            (0.0, 0.0)
        } else {
            (u.ammo as f32, u.cooldown as f32)
        };
        Self {
            kind,
            pos: u.pos,
            yaw: u.yaw,
            hp: u.hp,
            ammo,
            cooldown,
            age_ticks: 0,
        }
    }

    fn remembered(s: &Sighting, age_ticks: u64) -> Self {
        Self {
            kind: Kind::Enemy,
            pos: s.pos,
            yaw: s.yaw,
            hp: s.hp,
            ammo: 0.0,
            cooldown: 0.0,
            age_ticks: age_ticks.max(1),
        }
    }
}

/// The agent's egocentric frame.
struct Frame {
    x: f32,
    y: f32,
    floor: u8,
    yaw: f32,
    cos: f32,
    sin: f32,
}

impl Frame {
    fn of(u: &Unit) -> Self {
        let (sin, cos) = u.yaw.sin_cos();
        Self {
            x: u.pos.x,
            y: u.pos.y,
            floor: u.pos.floor,
            yaw: u.yaw,
            cos,
            sin,
        }
    }

    /// World offset -> (forward, lateral) in tiles.
    fn to_local(&self, x: f32, y: f32) -> (f32, f32) {
        let (dx, dy) = (x - self.x, y - self.y);
        (
            dx * self.cos + dy * self.sin,
            -dx * self.sin + dy * self.cos,
        )
    }

    /// Local (forward, lateral) -> world point.
    fn to_world(&self, fwd: f32, lat: f32) -> (f32, f32) {
        (
            self.x + fwd * self.cos - lat * self.sin,
            self.y + fwd * self.sin + lat * self.cos,
        )
    }

    fn encode(&self, v: &TokenView, out: &mut [f32]) {
        let (fwd, lat) = self.to_local(v.pos.x, v.pos.y);
        let (s, c) = (v.yaw - self.yaw).sin_cos();
        out[F_IS_SELF] = flag(v.kind == Kind::SelfUnit);
        out[F_IS_ALLY] = flag(v.kind == Kind::Ally);
        out[F_IS_ENEMY] = flag(v.kind == Kind::Enemy);
        out[F_REL_X] = (fwd / POS_NORM).clamp(-1.0, 1.0);
        out[F_REL_Y] = (lat / POS_NORM).clamp(-1.0, 1.0);
        out[F_REL_FLOOR] =
            ((f32::from(v.pos.floor) - f32::from(self.floor)) / FLOOR_NORM).clamp(-1.0, 1.0);
        out[F_SIN_YAW] = s;
        out[F_COS_YAW] = c;
        out[F_HP] = (v.hp / HP_NORM).clamp(0.0, 1.0);
        out[F_AMMO] = (v.ammo / AMMO_NORM).clamp(0.0, 1.0);
        out[F_COOLDOWN] = (v.cooldown / COOLDOWN_NORM).clamp(0.0, 1.0);
        out[F_LAST_SEEN] = (v.age_ticks as f32 / MEMORY_TICKS as f32).min(1.0);
    }
}

fn flag(b: bool) -> f32 {
    if b {
        1.0
    } else {
        0.0
    }
}

// ---- local map --------------------------------------------------------------

const HALF_VIEW: i32 = (LOCAL_VIEW / 2) as i32;
const CELLS: usize = LOCAL_VIEW * LOCAL_VIEW;

fn write_local_map(map: &GameMap, me: &Unit, frame: &Frame, out: &mut [f32]) {
    let floor = i32::from(me.pos.floor);
    for row in 0..LOCAL_VIEW {
        for col in 0..LOCAL_VIEW {
            let fwd = col as i32 - HALF_VIEW;
            let lat = row as i32 - HALF_VIEW;
            let (px, py) = frame.to_world(fwd as f32, lat as f32);
            if !tile_in_sight(map, (frame.x, frame.y), (px, py), floor) {
                continue;
            }
            let tile = map.tile(px.floor() as i32, py.floor() as i32, floor);
            let cell = row * LOCAL_VIEW + col;
            for (ch, v) in tile_channels(tile).into_iter().enumerate() {
                out[ch * CELLS + cell] = v;
            }
        }
    }
}

fn tile_channels(tile: Tile) -> [f32; MAP_CHANNELS] {
    let mut c = [0.0; MAP_CHANNELS];
    match tile {
        Tile::Void | Tile::Wall => c[C_WALL] = 1.0,
        Tile::DoorClosed => c[C_DOOR] = 1.0,
        Tile::DoorOpen => {
            c[C_DOOR] = 0.5;
            c[C_WALKABLE] = 1.0;
        }
        Tile::Floor => c[C_WALKABLE] = 1.0,
        Tile::Window => c[C_STAIRS_WINDOW] = 0.5,
        Tile::StairsUp => {
            c[C_WALKABLE] = 1.0;
            c[C_STAIRS_WINDOW] = 1.0;
        }
        Tile::StairsDown => {
            c[C_WALKABLE] = 1.0;
            c[C_STAIRS_WINDOW] = -1.0;
        }
    }
    c
}

/// Exact grid traversal (Amanatides-Woo) from `from` to `to` on `floor`:
/// true if no sight-blocking tile lies strictly between the start tile and
/// the tile containing `to` (so walls themselves are visible). A ray passing
/// exactly through a corner is blocked only if both side tiles block.
fn tile_in_sight(map: &GameMap, from: (f32, f32), to: (f32, f32), floor: i32) -> bool {
    let (dx, dy) = (to.0 - from.0, to.1 - from.1);
    let (mut cx, mut cy) = (from.0.floor() as i32, from.1.floor() as i32);
    let (tx, ty) = (to.0.floor() as i32, to.1.floor() as i32);
    let (step_x, mut t_max_x, t_delta_x) = axis_setup(from.0, dx, cx);
    let (step_y, mut t_max_y, t_delta_y) = axis_setup(from.1, dy, cy);
    // Each iteration crosses one or two boundaries; bound defensively.
    for _ in 0..=((tx - cx).abs() + (ty - cy).abs() + 1) {
        if (cx == tx && cy == ty) || (t_max_x > 1.0 && t_max_y > 1.0) {
            return true;
        }
        if t_max_x < t_max_y {
            cx += step_x;
            t_max_x += t_delta_x;
        } else if t_max_y < t_max_x {
            cy += step_y;
            t_max_y += t_delta_y;
        } else {
            if map.blocks_sight(cx + step_x, cy, floor) && map.blocks_sight(cx, cy + step_y, floor)
            {
                return false;
            }
            cx += step_x;
            cy += step_y;
            t_max_x += t_delta_x;
            t_max_y += t_delta_y;
        }
        if (cx != tx || cy != ty) && map.blocks_sight(cx, cy, floor) {
            return false;
        }
    }
    true
}

/// (step, t at first boundary crossing, t per tile) along one axis.
fn axis_setup(origin: f32, d: f32, cell: i32) -> (i32, f32, f32) {
    if d > 0.0 {
        (1, (cell as f32 + 1.0 - origin) / d, 1.0 / d)
    } else if d < 0.0 {
        (-1, (origin - cell as f32) / -d, 1.0 / -d)
    } else {
        (0, f32::INFINITY, f32::INFINITY)
    }
}

// ---- centralized critic state -------------------------------------------------

/// Full-information state for the centralized critic (training only; never
/// fed to actors). `out.len() == U * STATE_FEATS`; units in index order
/// (team 0 then team 1). Per unit, at offsets `S_*`:
///
/// | idx | feature | normalization |
/// |-----|---------|---------------|
/// | 0 | x | `2 x / width - 1` |
/// | 1 | y | `2 y / height - 1` |
/// | 2 | floor | `floor / max(floors - 1, 1)` |
/// | 3 | sin yaw | absolute yaw |
/// | 4 | cos yaw | absolute yaw |
/// | 5 | hp | `/ HP_NORM`, clamped [0, 1] |
/// | 6 | ammo | `/ AMMO_NORM`, clamped [0, 1] |
/// | 7 | cooldown | `/ COOLDOWN_NORM`, clamped [0, 1] |
/// | 8 | alive | 1 / 0 |
/// | 9 | team | 0 / 1 |
///
/// Dead units keep their last position/yaw with hp 0 and alive 0.
pub fn write_global_state(world: &World, out: &mut [f32]) {
    let n = world.units.len();
    assert_eq!(out.len(), state_len(n), "state buffer length");
    let map = &world.map;
    let (w, h) = (map.width.max(1) as f32, map.height.max(1) as f32);
    let floors = map.floors.saturating_sub(1).max(1) as f32;
    let (rows, _) = out.as_chunks_mut::<STATE_FEATS>();
    for (u, o) in world.units.iter().zip(rows) {
        let (s, c) = u.yaw.sin_cos();
        o[S_X] = (2.0 * u.pos.x / w - 1.0).clamp(-1.0, 1.0);
        o[S_Y] = (2.0 * u.pos.y / h - 1.0).clamp(-1.0, 1.0);
        o[S_FLOOR] = (f32::from(u.pos.floor) / floors).clamp(0.0, 1.0);
        o[S_SIN_YAW] = s;
        o[S_COS_YAW] = c;
        o[S_HP] = (u.hp / HP_NORM).clamp(0.0, 1.0);
        o[S_AMMO] = (u.ammo as f32 / AMMO_NORM).clamp(0.0, 1.0);
        o[S_COOLDOWN] = (u.cooldown as f32 / COOLDOWN_NORM).clamp(0.0, 1.0);
        o[S_ALIVE] = flag(u.alive);
        o[S_TEAM] = f32::from(u.team);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::{FRAC_PI_2, PI};

    const EPS: f32 = 1e-5;

    const TEST_AMMO: u32 = 15;
    const TEST_COOLDOWN: u32 = 2;

    fn unit(id: UnitId, team: TeamId, x: f32, y: f32, yaw: f32) -> Unit {
        let mut u = Unit::new(id, team, Pos { x, y, floor: 0 });
        u.yaw = yaw;
        u.ammo = TEST_AMMO;
        u.cooldown = TEST_COOLDOWN;
        u
    }

    fn world(units: Vec<Unit>) -> World {
        let mut w = World::new(GameMap::arena(20, 20));
        w.units = units;
        w
    }

    /// Wall column at x = `x` for y in 1..19 on floor 0.
    fn wall_column(w: &mut World, x: usize) {
        for y in 1..19 {
            w.map.set_tile(x, y, 0, Tile::Wall);
        }
    }

    struct Obs {
        tokens: Vec<f32>,
        mask: Vec<f32>,
        map: Vec<f32>,
    }

    impl Obs {
        fn tok(&self, slot: usize) -> &[f32] {
            &self.tokens[slot * TOKEN_FEATS..(slot + 1) * TOKEN_FEATS]
        }
        fn n_valid(&self) -> usize {
            self.mask.iter().filter(|&&m| m == 1.0).count()
        }
        fn cell(&self, ch: usize, row: usize, col: usize) -> f32 {
            self.map[ch * CELLS + row * LOCAL_VIEW + col]
        }
    }

    fn obs_of(w: &World, u: UnitId, mem: Option<&ObsMemory>) -> Obs {
        let n = w.units.len();
        let mut o = Obs {
            tokens: vec![9.0; tokens_len(n)],
            mask: vec![9.0; n],
            map: vec![9.0; LOCAL_MAP_LEN],
        };
        let vision = TeamVision::compute(w);
        write_agent_obs_with(w, u, &vision, mem, &mut o.tokens, &mut o.mask, &mut o.map);
        o
    }

    fn four_units() -> World {
        world(vec![
            unit(0, 0, 3.5, 3.5, 0.0),
            unit(1, 0, 3.5, 6.5, 0.0),
            unit(2, 1, 15.5, 3.5, PI),
            unit(3, 1, 15.5, 6.5, PI),
        ])
    }

    #[test]
    fn token_order_is_self_allies_enemies() {
        let w = four_units();
        let o = obs_of(&w, 1, None);
        assert_eq!(o.mask, vec![1.0; 4]);
        assert_eq!(&o.tok(0)[..3], &[1.0, 0.0, 0.0]);
        assert_eq!(&o.tok(1)[..3], &[0.0, 1.0, 0.0]);
        assert_eq!(&o.tok(2)[..3], &[0.0, 0.0, 1.0]);
        assert_eq!(&o.tok(3)[..3], &[0.0, 0.0, 1.0]);
        // Self token: zero offset, zero relative yaw, own stats.
        let s = o.tok(0);
        assert_eq!(
            (s[F_REL_X], s[F_REL_Y], s[F_SIN_YAW], s[F_COS_YAW]),
            (0.0, 0.0, 0.0, 1.0)
        );
        assert!((s[F_HP] - 1.0).abs() < EPS);
        assert!((s[F_AMMO] - TEST_AMMO as f32 / AMMO_NORM).abs() < EPS);
        assert!((s[F_COOLDOWN] - TEST_COOLDOWN as f32 / COOLDOWN_NORM).abs() < EPS);
        assert_eq!(s[F_LAST_SEEN], 0.0);
        // Ally (unit 0) is 3 tiles at -y; yaw 0 -> lateral axis is +y.
        let a = o.tok(1);
        assert!((a[F_REL_X]).abs() < EPS);
        assert!((a[F_REL_Y] + 3.0 / POS_NORM).abs() < EPS);
        // Enemy ammo/cooldown hidden, relative yaw PI.
        let e = o.tok(2);
        assert_eq!((e[F_AMMO], e[F_COOLDOWN]), (0.0, 0.0));
        assert!((e[F_COS_YAW] + 1.0).abs() < EPS);
    }

    #[test]
    fn all_features_are_bounded() {
        let w = four_units();
        for u in 0..4 {
            let o = obs_of(&w, u, None);
            assert!(o
                .tokens
                .iter()
                .chain(&o.map)
                .all(|v| (-1.0..=1.0).contains(v)));
        }
    }

    #[test]
    fn relative_coords_rotate_with_facing() {
        let mut w = world(vec![unit(0, 0, 5.5, 5.5, 0.0), unit(1, 1, 9.5, 5.5, 0.0)]);
        // Facing +x: enemy 4 tiles straight ahead.
        let t = obs_of(&w, 0, None).tok(1).to_vec();
        assert!((t[F_REL_X] - 4.0 / POS_NORM).abs() < EPS && t[F_REL_Y].abs() < EPS);
        // Facing +y: the same enemy is now on the negative lateral side.
        w.units[0].yaw = FRAC_PI_2;
        let t = obs_of(&w, 0, None).tok(1).to_vec();
        assert!(t[F_REL_X].abs() < EPS && (t[F_REL_Y] + 4.0 / POS_NORM).abs() < EPS);
        // Facing -x: behind.
        w.units[0].yaw = PI;
        let t = obs_of(&w, 0, None).tok(1).to_vec();
        assert!((t[F_REL_X] + 4.0 / POS_NORM).abs() < EPS);
        // Relative yaw: enemy yaw 0 vs own PI -> cos -1.
        assert!((t[F_COS_YAW] + 1.0).abs() < EPS);
    }

    #[test]
    fn far_positions_clamp() {
        let mut w = world(vec![unit(0, 0, 1.5, 1.5, 0.0), unit(1, 0, 90.0, 1.5, 0.0)]);
        w.map = GameMap::filled(100, 100, 1, Tile::Floor);
        let t = obs_of(&w, 0, None).tok(1).to_vec();
        assert_eq!(t[F_REL_X], 1.0);
    }

    #[test]
    fn wall_hides_enemy() {
        let mut w = world(vec![unit(0, 0, 3.5, 3.5, 0.0), unit(1, 1, 15.5, 3.5, PI)]);
        wall_column(&mut w, 9);
        let o = obs_of(&w, 0, None);
        assert_eq!(o.mask, vec![1.0, 0.0]);
        assert!(o.tok(1).iter().all(|&v| v == 0.0));
        // And symmetric for the enemy.
        assert_eq!(obs_of(&w, 1, None).n_valid(), 1);
    }

    #[test]
    fn teammate_shares_vision() {
        let mut w = world(vec![
            unit(0, 0, 3.5, 3.5, 0.0),
            unit(1, 0, 12.5, 3.5, 0.0),
            unit(2, 1, 15.5, 3.5, PI),
            unit(3, 1, 15.5, 16.5, PI),
        ]);
        // Wall between unit 0 and everyone else; units 1/2 see each other.
        wall_column(&mut w, 9);
        // Hide unit 3 from unit 1 with a wall row segment at y = 10.
        for x in 10..19 {
            w.map.set_tile(x, 10, 0, Tile::Wall);
        }
        let o = obs_of(&w, 0, None);
        // self, ally 1, enemy 2 (seen by ally); enemy 3 unseen.
        assert_eq!(o.mask, vec![1.0, 1.0, 1.0, 0.0]);
        assert_eq!(o.tok(2)[F_IS_ENEMY], 1.0);
        assert!((o.tok(2)[F_REL_X] - 12.0 / POS_NORM).abs() < EPS);
    }

    #[test]
    fn dead_units_are_masked() {
        let mut w = four_units();
        w.units[1].alive = false;
        w.units[3].alive = false;
        let o = obs_of(&w, 0, None);
        assert_eq!(o.mask, vec![1.0, 1.0, 0.0, 0.0]);
        assert_eq!(o.tok(1)[F_IS_ENEMY], 1.0);
        // Dead ally can't see for the team either.
        let mut w = world(vec![
            unit(0, 0, 3.5, 3.5, 0.0),
            unit(1, 0, 12.5, 3.5, 0.0),
            unit(2, 1, 15.5, 3.5, PI),
        ]);
        wall_column(&mut w, 9);
        w.units[1].alive = false;
        assert_eq!(obs_of(&w, 0, None).mask, vec![1.0, 0.0, 0.0]);
    }

    #[test]
    fn dead_agent_sees_only_itself() {
        let mut w = four_units();
        w.units[0].alive = false;
        w.units[0].hp = 0.0;
        let o = obs_of(&w, 0, None);
        assert_eq!(o.mask, vec![1.0, 0.0, 0.0, 0.0]);
        assert_eq!(o.tok(0)[F_IS_SELF], 1.0);
        assert_eq!(o.tok(0)[F_HP], 0.0);
        assert!(o.map.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn memory_keeps_recent_enemy_with_age() {
        let mut w = world(vec![unit(0, 0, 3.5, 3.5, 0.0), unit(1, 1, 15.5, 3.5, PI)]);
        let mut mem = ObsMemory::new();
        mem.update(&w, &TeamVision::compute(&w));
        // Enemy moves behind a wall; 10 ticks later.
        wall_column(&mut w, 9);
        w.units[1].pos.y = 8.5;
        w.tick = 10;
        mem.update(&w, &TeamVision::compute(&w));
        let o = obs_of(&w, 0, Some(&mem));
        assert_eq!(o.mask, vec![1.0, 1.0]);
        let t = o.tok(1);
        assert_eq!(t[F_IS_ENEMY], 1.0);
        assert!((t[F_LAST_SEEN] - 10.0 / MEMORY_TICKS as f32).abs() < EPS);
        // Last known position (y = 3.5), not the true one.
        assert!(t[F_REL_Y].abs() < EPS);
        // Without memory it is absent; after MEMORY_TICKS it is forgotten.
        assert_eq!(obs_of(&w, 0, None).n_valid(), 1);
        w.tick = MEMORY_TICKS;
        assert_eq!(obs_of(&w, 0, Some(&mem)).n_valid(), 1);
        mem.clear();
        w.tick = 10;
        assert_eq!(obs_of(&w, 0, Some(&mem)).n_valid(), 1);
    }

    #[test]
    fn local_map_fog_and_channels() {
        let mut w = world(vec![unit(0, 0, 5.5, 5.5, 0.0)]);
        wall_column(&mut w, 7);
        w.map.set_tile(7, 5, 0, Tile::DoorClosed);
        w.map.set_tile(5, 3, 0, Tile::Window);
        w.map.set_tile(4, 5, 0, Tile::StairsUp);
        w.map.set_tile(5, 7, 0, Tile::DoorOpen);
        let o = obs_of(&w, 0, None);
        let h = LOCAL_VIEW / 2;
        // Own cell: walkable floor.
        assert_eq!(o.cell(C_WALKABLE, h, h), 1.0);
        // Closed door 2 ahead (x = 7): visible; beyond it fogged.
        assert_eq!(o.cell(C_DOOR, h, h + 2), 1.0);
        assert_eq!(o.cell(C_WALL, h + 1, h + 2), 1.0);
        for c in 0..MAP_CHANNELS {
            assert_eq!(o.cell(c, h, h + 3), 0.0);
            assert_eq!(o.cell(c, h + 1, h + 4), 0.0);
        }
        assert_eq!(o.cell(C_STAIRS_WINDOW, h - 2, h), 0.5);
        assert_eq!(o.cell(C_STAIRS_WINDOW, h, h - 1), 1.0);
        assert_eq!(o.cell(C_WALKABLE, h, h - 1), 1.0);
        assert_eq!(o.cell(C_DOOR, h + 2, h), 0.5);
        assert_eq!(o.cell(C_WALKABLE, h + 2, h), 1.0);
        // Arena border wall at x = 0 is 5 tiles away: outside the patch.
        // Out-of-map cells at -y (y = 1..) are inside the arena: walkable.
        assert_eq!(o.cell(C_WALKABLE, 0, h), 1.0);
    }

    #[test]
    fn local_map_rotates_with_facing() {
        let mut w = world(vec![unit(0, 0, 5.5, 5.5, 0.0)]);
        w.map.set_tile(5, 7, 0, Tile::DoorClosed); // 2 tiles at +y
        let h = LOCAL_VIEW / 2;
        // yaw 0: +y is the lateral axis (row).
        assert_eq!(obs_of(&w, 0, None).cell(C_DOOR, h + 2, h), 1.0);
        // yaw +90deg: +y is straight ahead (col).
        w.units[0].yaw = FRAC_PI_2;
        let o = obs_of(&w, 0, None);
        assert_eq!(o.cell(C_DOOR, h, h + 2), 1.0);
        assert_eq!(o.cell(C_DOOR, h + 2, h), 0.0);
        // yaw 180deg: +y is the negative lateral side.
        w.units[0].yaw = PI;
        assert_eq!(obs_of(&w, 0, None).cell(C_DOOR, h - 2, h), 1.0);
    }

    #[test]
    fn local_map_ignores_other_floors() {
        let mut w = world(vec![unit(0, 0, 5.5, 5.5, 0.0)]);
        w.map = GameMap::filled(20, 20, 2, Tile::Floor);
        w.map.set_tile(6, 5, 1, Tile::Wall);
        let o = obs_of(&w, 0, None);
        let h = LOCAL_VIEW / 2;
        assert_eq!(o.cell(C_WALL, h, h + 1), 0.0);
        assert_eq!(o.cell(C_WALKABLE, h, h + 1), 1.0);
    }

    #[test]
    fn diagonal_wall_gap_blocks_sight() {
        let mut m = GameMap::filled(10, 10, 1, Tile::Floor);
        m.set_tile(3, 2, 0, Tile::Wall);
        m.set_tile(2, 3, 0, Tile::Wall);
        // Ray from (2.0+, 2.0+) diagonally through the corner (3,3).
        assert!(!tile_in_sight(&m, (2.5, 2.5), (4.5, 4.5), 0));
        m.set_tile(2, 3, 0, Tile::Floor);
        assert!(tile_in_sight(&m, (2.5, 2.5), (4.5, 4.5), 0));
        // Walls themselves are visible, tiles behind them are not.
        assert!(tile_in_sight(&m, (0.5, 2.5), (3.5, 2.5), 0));
        assert!(!tile_in_sight(&m, (0.5, 2.5), (4.5, 2.5), 0));
    }

    #[test]
    fn global_state_layout() {
        let mut w = four_units();
        w.units[3].alive = false;
        w.units[3].hp = 0.0;
        let mut s = vec![9.0; state_len(4)];
        write_global_state(&w, &mut s);
        let u2 = &s[2 * STATE_FEATS..3 * STATE_FEATS];
        assert!((u2[S_X] - (2.0 * 15.5 / 20.0 - 1.0)).abs() < EPS);
        assert!((u2[S_Y] - (2.0 * 3.5 / 20.0 - 1.0)).abs() < EPS);
        assert_eq!(u2[S_FLOOR], 0.0);
        assert!((u2[S_COS_YAW] + 1.0).abs() < EPS);
        assert!((u2[S_HP] - 1.0).abs() < EPS);
        assert!((u2[S_AMMO] - TEST_AMMO as f32 / AMMO_NORM).abs() < EPS);
        assert!((u2[S_COOLDOWN] - TEST_COOLDOWN as f32 / COOLDOWN_NORM).abs() < EPS);
        assert_eq!((u2[S_ALIVE], u2[S_TEAM]), (1.0, 1.0));
        assert_eq!(s[S_TEAM], 0.0);
        assert_eq!(s[3 * STATE_FEATS + S_ALIVE], 0.0);
        assert!(s.iter().all(|v| (-1.0..=1.0).contains(v)));
    }

    #[test]
    fn write_env_obs_fills_every_agent() {
        let w = four_units();
        let n = 4;
        let mut mem = ObsMemory::new();
        let mut tok = vec![9.0; n * tokens_len(n)];
        let mut mask = vec![9.0; n * n];
        let mut lmap = vec![9.0; n * LOCAL_MAP_LEN];
        let mut state = vec![9.0; state_len(n)];
        let mut alive = vec![9.0; n];
        write_env_obs(
            &w, &mut mem, &mut tok, &mut mask, &mut lmap, &mut state, &mut alive,
        );
        assert_eq!(alive, vec![1.0; 4]);
        assert_eq!(mask, vec![1.0; 16]);
        for u in 0..n {
            assert_eq!(tok[u * tokens_len(n) + F_IS_SELF], 1.0);
            assert_eq!(
                &tok[u * tokens_len(n)..(u + 1) * tokens_len(n)],
                &obs_of(&w, u, None).tokens[..]
            );
        }
    }

    #[test]
    #[should_panic(expected = "tokens buffer length")]
    fn rejects_wrong_buffer_size() {
        let w = four_units();
        let (mut t, mut m, mut l) = (
            vec![0.0; TOKENS_LEN],
            vec![0.0; 4],
            vec![0.0; LOCAL_MAP_LEN],
        );
        write_agent_obs(&w, 0, &mut t, &mut m, &mut l);
    }
}
