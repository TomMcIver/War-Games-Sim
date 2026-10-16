//! Reachability (BFS over passable tiles + stairs) and map validation.

use std::collections::VecDeque;

use super::{GameMap, MapError, Tile, TileCoord};
use crate::types::{Pos, MAX_TEAM_SIZE};

const NEIGHBOURS: [(i32, i32); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];

impl GameMap {
    fn flat(&self, (x, y, f): TileCoord) -> usize {
        self.idx(x as usize, y as usize, f as usize)
    }

    fn passable(&self, x: i32, y: i32, f: i32) -> bool {
        self.tile(x, y, f).is_passable()
    }

    /// Flood fill from `start`: `visited[flat index]` for every tile
    /// reachable by 4-neighbour steps over passable tiles (closed doors
    /// count as passable) plus stairs links. Empty fill if `start` is not
    /// passable.
    pub(crate) fn flood(&self, start: TileCoord) -> Vec<bool> {
        let mut seen = vec![false; self.tiles.len()];
        let (sx, sy, sf) = start;
        if !self.passable(sx, sy, sf) {
            return seen;
        }
        let mut queue = VecDeque::new();
        seen[self.flat(start)] = true;
        queue.push_back(start);
        while let Some((x, y, f)) = queue.pop_front() {
            let stairs = self.stairs_destination(x, y, f);
            let steps = NEIGHBOURS.iter().map(|&(dx, dy)| (x + dx, y + dy, f));
            for next in steps.chain(stairs) {
                if self.passable(next.0, next.1, next.2) && !seen[self.flat(next)] {
                    seen[self.flat(next)] = true;
                    queue.push_back(next);
                }
            }
        }
        seen
    }

    /// Can a unit walk from `a` to `b` (opening doors, using stairs)?
    pub fn is_connected(&self, a: Pos, b: Pos) -> bool {
        let to_tile = |p: Pos| -> Option<TileCoord> {
            if !(p.x.is_finite() && p.y.is_finite()) {
                return None;
            }
            let t = (p.x.floor() as i32, p.y.floor() as i32, i32::from(p.floor));
            self.passable(t.0, t.1, t.2).then_some(t)
        };
        match (to_tile(a), to_tile(b)) {
            (Some(ta), Some(tb)) => self.flood(ta)[self.flat(tb)],
            _ => false,
        }
    }

    /// True if every passable tile on the map is reachable from every other.
    pub fn is_fully_connected(&self) -> bool {
        let first = (0..self.tiles.len()).find(|&i| self.tiles[i].is_passable());
        let Some(i) = first else { return true };
        let seen = self.flood(self.coord_of(i));
        self.tiles
            .iter()
            .zip(&seen)
            .all(|(t, &s)| !t.is_passable() || s)
    }

    fn coord_of(&self, i: usize) -> TileCoord {
        let x = i % self.width;
        let y = (i / self.width) % self.height;
        let f = i / (self.width * self.height);
        (x as i32, y as i32, f as i32)
    }

    /// Check the invariants every generated map guarantees:
    /// both spawn zones hold at least `MAX_TEAM_SIZE` walkable tiles, all of
    /// them mutually reachable, and every stairs tile has its partner.
    pub fn validate(&self) -> Result<(), MapError> {
        for team in 0..2u8 {
            let zone = self.spawn_zone(team);
            let walkable = zone
                .iter()
                .filter(|&&(x, y, f)| self.is_walkable(x, y, f))
                .count();
            if walkable < MAX_TEAM_SIZE {
                return Err(MapError::Invalid(format!(
                    "team {team} spawn zone has {walkable} walkable tiles, need {MAX_TEAM_SIZE}"
                )));
            }
        }
        self.validate_stairs()?;
        let zones = [self.spawn_zone(0), self.spawn_zone(1)];
        let seen = self.flood(zones[0][0]);
        if let Some(t) = zones.iter().flatten().find(|&&c| !seen[self.flat(c)]) {
            return Err(MapError::Invalid(format!(
                "spawn tile {t:?} unreachable from {:?}",
                zones[0][0]
            )));
        }
        Ok(())
    }

    fn validate_stairs(&self) -> Result<(), MapError> {
        for (i, &t) in self.tiles.iter().enumerate() {
            if t == Tile::StairsUp || t == Tile::StairsDown {
                let (x, y, f) = self.coord_of(i);
                if self.stairs_destination(x, y, f).is_none() {
                    return Err(MapError::Invalid(format!(
                        "unpaired {t:?} at {:?}",
                        (x, y, f)
                    )));
                }
            }
        }
        Ok(())
    }
}
