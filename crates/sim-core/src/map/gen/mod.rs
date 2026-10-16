//! Deterministic, seeded map generators (ChaCha8 RNG only).
//!
//! Every generated map is point-symmetric (180-degree rotation about the map
//! centre) so neither team gets a better side: generators only author the
//! "canonical" half and every write is mirrored by [`set_sym`]. Every
//! generated map passes [`GameMap::validate`] and
//! [`GameMap::is_fully_connected`]; an attempt that fails is retried with
//! the same RNG stream, and after `MAX_ATTEMPTS` failures the generator falls
//! back to an arena (which is valid by construction).

mod arena;
mod building;
mod rooms;
mod village;

use std::str::FromStr;

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use super::{GameMap, MapError, Tile, TileCoord};
use crate::types::MAX_TEAM_SIZE;

const MAX_ATTEMPTS: u32 = 50;
/// Spawn zones are the columns `1..=SPAWN_DEPTH` inside the low-x edge
/// (team 0) and their rotation (team 1).
const SPAWN_DEPTH: i32 = 3;

/// The map kinds understood by [`generate`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MapKind {
    /// Walled open arena with seeded cover blocks; size scales with team size.
    Arena,
    /// One two-floor multi-room building in a walled yard.
    Building,
    /// 65x65 street grid of one/two-floor houses with cover in between.
    Village,
}

impl MapKind {
    pub const NAMES: [&'static str; 3] = ["arena", "building", "village"];

    pub fn name(self) -> &'static str {
        match self {
            MapKind::Arena => "arena",
            MapKind::Building => "building",
            MapKind::Village => "village",
        }
    }
}

impl FromStr for MapKind {
    type Err = MapError;

    /// Case-insensitive, surrounding whitespace ignored.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "arena" => Ok(MapKind::Arena),
            "building" => Ok(MapKind::Building),
            "village" => Ok(MapKind::Village),
            _ => Err(MapError::UnknownKind(s.to_string())),
        }
    }
}

/// Generate a map. Same `(kind, seed, team_size)` => identical map.
/// `team_size` is clamped to `1..=MAX_TEAM_SIZE`.
///
/// Unknown kinds **fall back to "arena"** (use [`try_generate`] to get a
/// [`MapError::UnknownKind`] instead).
pub fn generate(kind: &str, seed: u64, team_size: usize) -> GameMap {
    let kind = kind.parse().unwrap_or(MapKind::Arena);
    generate_kind(kind, seed, team_size)
}

/// Like [`generate`] but rejects unknown kinds.
pub fn try_generate(kind: &str, seed: u64, team_size: usize) -> Result<GameMap, MapError> {
    Ok(generate_kind(kind.parse()?, seed, team_size))
}

fn generate_kind(kind: MapKind, seed: u64, team_size: usize) -> GameMap {
    let team_size = team_size.clamp(1, MAX_TEAM_SIZE);
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let build: fn(&mut ChaCha8Rng, usize) -> Option<GameMap> = match kind {
        MapKind::Arena => |rng, ts| Some(arena::build(rng, ts)),
        MapKind::Building => building::build,
        MapKind::Village => village::build,
    };
    for _ in 0..MAX_ATTEMPTS {
        if let Some(m) = build(&mut rng, team_size) {
            if m.validate().is_ok() && m.is_fully_connected() {
                return m;
            }
        }
    }
    arena::build(&mut rng, team_size)
}

/// Inclusive integer rectangle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Rect {
    pub x0: i32,
    pub y0: i32,
    pub x1: i32,
    pub y1: i32,
}

impl Rect {
    pub fn w(self) -> i32 {
        self.x1 - self.x0 + 1
    }
    pub fn h(self) -> i32 {
        self.y1 - self.y0 + 1
    }
    pub fn inner(self) -> Rect {
        Rect { x0: self.x0 + 1, y0: self.y0 + 1, x1: self.x1 - 1, y1: self.y1 - 1 }
    }
    pub fn cells(self) -> impl Iterator<Item = (i32, i32)> {
        (self.y0..=self.y1).flat_map(move |y| (self.x0..=self.x1).map(move |x| (x, y)))
    }
}

/// Write `t` at `(x, y, f)` and at its 180-degree rotation.
pub(super) fn set_sym(m: &mut GameMap, x: i32, y: i32, f: i32, t: Tile) {
    if !m.in_bounds(x, y, f) {
        return;
    }
    let (rx, ry, _) = m.rotate(x, y, f);
    m.set_tile(x as usize, y as usize, f as usize, t);
    m.set_tile(rx as usize, ry as usize, f as usize, t);
}

/// Is `(x, y)` in the authored half (the other half is its mirror image)?
pub(super) fn is_canonical(m: &GameMap, x: i32, y: i32) -> bool {
    let (rx, ry, _) = m.rotate(x, y, 0);
    (x, y) <= (rx, ry)
}

/// Bordered single-floor canvas (walls on the edge, floor inside).
pub(super) fn bordered(w: i32, h: i32, floors: usize) -> GameMap {
    let mut m = GameMap::filled(w as usize, h as usize, floors, Tile::Void);
    for (x, y) in (Rect { x0: 0, y0: 0, x1: w - 1, y1: h - 1 }).cells() {
        let edge = x == 0 || y == 0 || x == w - 1 || y == h - 1;
        m.set_tile(x as usize, y as usize, 0, if edge { Tile::Wall } else { Tile::Floor });
    }
    m
}

/// Ordering key for spawn tiles: closest to the centre row first, then
/// deepest-forward column.
fn zone_key(m: &GameMap, (x, y, _): TileCoord) -> (i32, i32, i32) {
    ((2 * y - (m.height as i32 - 1)).abs(), -x, y)
}

/// Walkable floor-0 tiles in columns `x_from..=x_to`, ordered for spawning.
pub(super) fn edge_zone(m: &GameMap, x_from: i32, x_to: i32) -> Vec<TileCoord> {
    let mut zone: Vec<TileCoord> = (0..m.height as i32)
        .flat_map(|y| (x_from..=x_to).map(move |x| (x, y, 0)))
        .filter(|&(x, y, f)| m.is_walkable(x, y, f))
        .collect();
    zone.sort_by_key(|&c| zone_key(m, c));
    zone
}

/// Spawn zone for maps without stored zones (hand-built or parsed from
/// JSON): walkable floor-0 tiles in the low-x third of the map.
pub(super) fn derived_zone(m: &GameMap) -> Vec<TileCoord> {
    let third = (m.width as i32 / 3).max(1);
    edge_zone(m, 0, third - 1)
}

/// Pick `n` tiles in zone order, first keeping Chebyshev spacing >= 2, then
/// filling with the remaining tiles in order. Rotation-invariant, so the
/// rotated zone yields the rotated picks.
pub(super) fn spread_pick(zone: &[TileCoord], n: usize) -> Vec<TileCoord> {
    let mut picked: Vec<TileCoord> = Vec::with_capacity(n);
    for &c in zone {
        if picked.len() == n {
            return picked;
        }
        let spaced = picked
            .iter()
            .all(|p| p.2 != c.2 || (p.0 - c.0).abs().max((p.1 - c.1).abs()) >= 2);
        if spaced {
            picked.push(c);
        }
    }
    for &c in zone {
        if picked.len() == n {
            break;
        }
        if !picked.contains(&c) {
            picked.push(c);
        }
    }
    picked
}

/// Install the standard edge spawn zones.
pub(super) fn install_edge_spawns(m: &mut GameMap) {
    let zone = edge_zone(m, 1, SPAWN_DEPTH);
    m.set_spawn_zones(zone);
}

/// Try to stamp up to `count` small cover blocks (walls, or window
/// "fences" that block movement but not sight/bullets) inside `region`
/// (canonical half). A block needs a one-tile margin of plain floor around
/// it and is reverted if it would disconnect the map.
pub(super) fn scatter_cover(m: &mut GameMap, rng: &mut ChaCha8Rng, region: Rect, count: u32) {
    let mut placed = 0;
    for _ in 0..count * 6 {
        if placed == count {
            return;
        }
        let (w, h) = match rng.gen_range(0..5) {
            0 => (1, 1),
            1 => (2, 2),
            2 => (rng.gen_range(2..=4), 1),
            3 => (1, rng.gen_range(2..=4)),
            _ => (rng.gen_range(2..=3), rng.gen_range(1..=2)),
        };
        if region.w() < w || region.h() < h {
            continue;
        }
        let x0 = rng.gen_range(region.x0..=region.x1 - w + 1);
        let y0 = rng.gen_range(region.y0..=region.y1 - h + 1);
        let block = Rect { x0, y0, x1: x0 + w - 1, y1: y0 + h - 1 };
        let tile = if rng.gen_bool(0.25) { Tile::Window } else { Tile::Wall };
        let margin = Rect { x0: x0 - 1, y0: y0 - 1, x1: block.x1 + 1, y1: block.y1 + 1 };
        if !margin.cells().all(|(x, y)| m.tile(x, y, 0) == Tile::Floor) {
            continue;
        }
        let before = m.clone();
        for (x, y) in block.cells() {
            set_sym(m, x, y, 0, tile);
        }
        if m.is_fully_connected() {
            placed += 1;
        } else {
            *m = before;
        }
    }
}
