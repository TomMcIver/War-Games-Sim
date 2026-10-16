//! OWNER: map-agent. Tile map with floors, walls, doors, windows, stairs;
//! generators (arena, building, village); line of sight.

use crate::types::{Pos, TeamId};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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

#[derive(Clone, Debug)]
pub struct GameMap {
    pub width: usize,
    pub height: usize,
    pub floors: usize,
    tiles: Vec<Tile>, // index: (f * height + y) * width + x
}

impl GameMap {
    pub fn filled(width: usize, height: usize, floors: usize, t: Tile) -> Self {
        Self { width, height, floors, tiles: vec![t; width * height * floors] }
    }

    fn idx(&self, x: usize, y: usize, f: usize) -> usize {
        (f * self.height + y) * self.width + x
    }

    pub fn in_bounds(&self, x: i32, y: i32, f: i32) -> bool {
        x >= 0 && y >= 0 && f >= 0
            && (x as usize) < self.width && (y as usize) < self.height && (f as usize) < self.floors
    }

    /// Out-of-bounds reads as Void.
    pub fn tile(&self, x: i32, y: i32, f: i32) -> Tile {
        if self.in_bounds(x, y, f) { self.tiles[self.idx(x as usize, y as usize, f as usize)] } else { Tile::Void }
    }

    pub fn set_tile(&mut self, x: usize, y: usize, f: usize, t: Tile) {
        let i = self.idx(x, y, f);
        self.tiles[i] = t;
    }

    /// Can a unit stand/walk through this tile?
    pub fn is_walkable(&self, x: i32, y: i32, f: i32) -> bool {
        matches!(self.tile(x, y, f), Tile::Floor | Tile::DoorOpen | Tile::StairsUp | Tile::StairsDown)
    }

    /// Does this tile block line of sight?
    pub fn blocks_sight(&self, x: i32, y: i32, f: i32) -> bool {
        matches!(self.tile(x, y, f), Tile::Wall | Tile::DoorClosed | Tile::Void)
    }

    /// Placeholder sampled LOS. map-agent: replace with exact grid traversal.
    pub fn line_of_sight(&self, a: Pos, b: Pos) -> bool {
        if a.floor != b.floor { return false; }
        let (dx, dy) = (b.x - a.x, b.y - a.y);
        let steps = (dx.abs().max(dy.abs()) * 4.0).ceil().max(1.0) as i32;
        (1..steps).all(|i| {
            let t = i as f32 / steps as f32;
            !self.blocks_sight((a.x + dx * t).floor() as i32, (a.y + dy * t).floor() as i32, a.floor as i32)
        })
    }

    /// Simple bordered arena on floor 0.
    pub fn arena(width: usize, height: usize) -> Self {
        let mut m = Self::filled(width, height, 1, Tile::Floor);
        for x in 0..width { m.set_tile(x, 0, 0, Tile::Wall); m.set_tile(x, height - 1, 0, Tile::Wall); }
        for y in 0..height { m.set_tile(0, y, 0, Tile::Wall); m.set_tile(width - 1, y, 0, Tile::Wall); }
        m
    }

    /// `n` spawn points for a team. Placeholder: left side for team 0, right for team 1.
    pub fn spawn_points(&self, team: TeamId, n: usize) -> Vec<Pos> {
        let x = if team == 0 { 2.5 } else { self.width as f32 - 2.5 };
        (0..n).map(|i| Pos { x, y: 2.5 + i as f32, floor: 0 }).collect()
    }
}
