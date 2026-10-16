//! OWNER: map-agent. Tile map with floors, walls, doors, windows, stairs;
//! generators (arena, building, village); line of sight, bullet rays,
//! circle-vs-tile movement, connectivity and JSON export.
//!
//! Coordinates: tile `(x, y)` covers the square `[x, x+1) x [y, y+1)` in
//! continuous tile units, so a [`Pos`] maps to tile `(floor(x), floor(y))`.
//! Angles are radians with direction `(cos a, sin a)` in map x/y.
//! See `README.md` in this directory for tile semantics and generators.

mod connectivity;
mod error;
mod gen;
mod json;
mod los;
mod motion;

#[cfg(test)]
mod tests;

pub use error::MapError;
pub use gen::{generate, try_generate, MapKind};
pub use los::RayHit;

use crate::types::{Pos, TeamId, MAX_TEAM_SIZE};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Tile {
    Void = 0,
    Floor = 1,
    Wall = 2,
    DoorClosed = 3,
    DoorOpen = 4,
    Window = 5,
    StairsUp = 6,
    StairsDown = 7,
}

impl Tile {
    /// Decode the contract's u8 value (see docs/INTERFACE.md, Map JSON).
    pub fn from_u8(v: u8) -> Option<Tile> {
        Some(match v {
            0 => Tile::Void,
            1 => Tile::Floor,
            2 => Tile::Wall,
            3 => Tile::DoorClosed,
            4 => Tile::DoorOpen,
            5 => Tile::Window,
            6 => Tile::StairsUp,
            7 => Tile::StairsDown,
            _ => return None,
        })
    }

    #[inline]
    pub fn as_u8(self) -> u8 {
        self as u8
    }

    /// A unit can stand on / move through this tile.
    #[inline]
    pub fn is_walkable(self) -> bool {
        matches!(
            self,
            Tile::Floor | Tile::DoorOpen | Tile::StairsUp | Tile::StairsDown
        )
    }

    /// Blocks line of sight. Windows and open doors do not.
    #[inline]
    pub fn blocks_sight(self) -> bool {
        matches!(self, Tile::Wall | Tile::DoorClosed | Tile::Void)
    }

    /// Stops a bullet. Windows do not (they are glass / openings).
    #[inline]
    pub fn blocks_bullets(self) -> bool {
        matches!(self, Tile::Wall | Tile::DoorClosed | Tile::Void)
    }

    /// Passable for reachability analysis: walkable, or a closed door
    /// (which a unit can open).
    #[inline]
    pub fn is_passable(self) -> bool {
        self.is_walkable() || self == Tile::DoorClosed
    }

    #[inline]
    pub fn is_door(self) -> bool {
        matches!(self, Tile::DoorClosed | Tile::DoorOpen)
    }

    #[inline]
    pub fn is_stairs(self) -> bool {
        matches!(self, Tile::StairsUp | Tile::StairsDown)
    }
}

/// Integer tile coordinate `(x, y, floor)`.
pub type TileCoord = (i32, i32, i32);

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GameMap {
    pub width: usize,
    pub height: usize,
    pub floors: usize,
    tiles: Vec<Tile>, // index: (f * height + y) * width + x
    /// Ordered spawn candidate tiles per team. Empty => derived on demand.
    spawn_zones: [Vec<TileCoord>; 2],
}

impl GameMap {
    pub fn filled(width: usize, height: usize, floors: usize, t: Tile) -> Self {
        Self {
            width,
            height,
            floors,
            tiles: vec![t; width * height * floors],
            spawn_zones: [Vec::new(), Vec::new()],
        }
    }

    /// Deterministic generator entry point; see [`generate`].
    pub fn generate(kind: &str, seed: u64, team_size: usize) -> Self {
        generate(kind, seed, team_size)
    }

    fn idx(&self, x: usize, y: usize, f: usize) -> usize {
        (f * self.height + y) * self.width + x
    }

    pub fn in_bounds(&self, x: i32, y: i32, f: i32) -> bool {
        x >= 0
            && y >= 0
            && f >= 0
            && (x as usize) < self.width
            && (y as usize) < self.height
            && (f as usize) < self.floors
    }

    /// Out-of-bounds reads as Void.
    #[inline]
    pub fn tile(&self, x: i32, y: i32, f: i32) -> Tile {
        if self.in_bounds(x, y, f) {
            self.tiles[self.idx(x as usize, y as usize, f as usize)]
        } else {
            Tile::Void
        }
    }

    pub fn set_tile(&mut self, x: usize, y: usize, f: usize, t: Tile) {
        let i = self.idx(x, y, f);
        self.tiles[i] = t;
    }

    /// Flat tile slice, index `(f * height + y) * width + x`.
    pub fn tiles(&self) -> &[Tile] {
        &self.tiles
    }

    /// Can a unit stand/walk through this tile?
    #[inline]
    pub fn is_walkable(&self, x: i32, y: i32, f: i32) -> bool {
        self.tile(x, y, f).is_walkable()
    }

    /// Does this tile block line of sight?
    #[inline]
    pub fn blocks_sight(&self, x: i32, y: i32, f: i32) -> bool {
        self.tile(x, y, f).blocks_sight()
    }

    /// Does this tile stop bullets?
    #[inline]
    pub fn blocks_bullets(&self, x: i32, y: i32, f: i32) -> bool {
        self.tile(x, y, f).blocks_bullets()
    }

    /// Toggle a door open <-> closed. Returns `false` (and changes nothing)
    /// if the tile is not a door. The map does not know about units: the
    /// engine should not close a door a unit currently overlaps.
    pub fn toggle_door(&mut self, x: i32, y: i32, f: i32) -> bool {
        let next = match self.tile(x, y, f) {
            Tile::DoorClosed => Tile::DoorOpen,
            Tile::DoorOpen => Tile::DoorClosed,
            _ => return false,
        };
        self.set_tile(x as usize, y as usize, f as usize, next);
        true
    }

    /// Where using the stairs at `(x, y, f)` leads: `StairsUp` at floor `f`
    /// pairs with `StairsDown` at the same `(x, y)` on floor `f + 1`.
    /// `None` if the tile is not stairs or its partner is missing.
    pub fn stairs_destination(&self, x: i32, y: i32, f: i32) -> Option<TileCoord> {
        match self.tile(x, y, f) {
            Tile::StairsUp if self.tile(x, y, f + 1) == Tile::StairsDown => Some((x, y, f + 1)),
            Tile::StairsDown if self.tile(x, y, f - 1) == Tile::StairsUp => Some((x, y, f - 1)),
            _ => None,
        }
    }

    /// Simple bordered arena on floor 0 (no cover). Spawn zones are the
    /// three columns inside each short edge.
    pub fn arena(width: usize, height: usize) -> Self {
        let mut m = Self::filled(width, height, 1, Tile::Floor);
        for x in 0..width {
            m.set_tile(x, 0, 0, Tile::Wall);
            m.set_tile(x, height - 1, 0, Tile::Wall);
        }
        for y in 0..height {
            m.set_tile(0, y, 0, Tile::Wall);
            m.set_tile(width - 1, y, 0, Tile::Wall);
        }
        let zone = gen::edge_zone(&m, 1, 3.min(width.saturating_sub(2) as i32));
        m.set_spawn_zones(zone);
        m
    }

    /// Install team 0's ordered spawn zone; team 1's is its 180-degree
    /// rotation (same order), which keeps spawns fair on symmetric maps.
    pub(crate) fn set_spawn_zones(&mut self, team0: Vec<TileCoord>) {
        let team1 = team0.iter().map(|&(x, y, f)| self.rotate(x, y, f)).collect();
        self.spawn_zones = [team0, team1];
    }

    /// 180-degree rotation about the map centre (same floor).
    #[inline]
    pub(crate) fn rotate(&self, x: i32, y: i32, f: i32) -> TileCoord {
        (self.width as i32 - 1 - x, self.height as i32 - 1 - y, f)
    }

    /// Ordered spawn candidate tiles for a team (derived from the map if the
    /// map was not built by a generator).
    pub fn spawn_zone(&self, team: TeamId) -> Vec<TileCoord> {
        let t = (team as usize).min(1);
        if !self.spawn_zones[t].is_empty() {
            return self.spawn_zones[t].clone();
        }
        let zone0 = gen::derived_zone(self);
        if t == 0 {
            zone0
        } else {
            zone0.iter().map(|&(x, y, f)| self.rotate(x, y, f)).collect()
        }
    }

    /// `n` distinct, walkable spawn points (tile centres) for a team; team 0
    /// spawns on the low-x side, team 1 on the opposite side. Points are
    /// spread out (Chebyshev spacing >= 2) when the zone allows it.
    ///
    /// Panics if `n` exceeds the zone size; every generator guarantees at
    /// least `MAX_TEAM_SIZE` candidates.
    pub fn spawn_points(&self, team: TeamId, n: usize) -> Vec<Pos> {
        let zone = self.spawn_zone(team);
        let zone: Vec<TileCoord> = zone
            .into_iter()
            .filter(|&(x, y, f)| self.is_walkable(x, y, f))
            .collect();
        assert!(
            n <= zone.len(),
            "spawn_points: asked for {n} points but team {team} zone has {} tiles (max supported {MAX_TEAM_SIZE})",
            zone.len()
        );
        let picked = gen::spread_pick(&zone, n);
        picked
            .into_iter()
            .map(|(x, y, f)| Pos {
                x: x as f32 + 0.5,
                y: y as f32 + 0.5,
                floor: f as u8,
            })
            .collect()
    }
}
