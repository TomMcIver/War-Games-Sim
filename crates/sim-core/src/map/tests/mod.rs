//! Map module tests. `ascii` builds maps from text art:
//! `#` Wall, `.` Floor, `+` DoorClosed, `/` DoorOpen, `=` Window,
//! `U` StairsUp, `D` StairsDown, ` ` Void.

mod doors_stairs;
mod generators;
mod json;
mod los;
mod motion;

use super::{GameMap, Tile};
use crate::types::Pos;

/// Build a map from one ASCII grid per floor (all floors the same size).
pub(super) fn ascii(floors: &[&[&str]]) -> GameMap {
    let h = floors[0].len();
    let w = floors[0][0].len();
    let mut m = GameMap::filled(w, h, floors.len(), Tile::Void);
    for (f, rows) in floors.iter().enumerate() {
        assert_eq!(rows.len(), h, "floor {f} height");
        for (y, row) in rows.iter().enumerate() {
            assert_eq!(row.len(), w, "floor {f} row {y} width");
            for (x, c) in row.chars().enumerate() {
                let t = match c {
                    '#' => Tile::Wall,
                    '.' => Tile::Floor,
                    '+' => Tile::DoorClosed,
                    '/' => Tile::DoorOpen,
                    '=' => Tile::Window,
                    'U' => Tile::StairsUp,
                    'D' => Tile::StairsDown,
                    ' ' => Tile::Void,
                    other => panic!("bad map char {other:?}"),
                };
                m.set_tile(x, y, f, t);
            }
        }
    }
    m
}

pub(super) fn p(x: f32, y: f32) -> Pos {
    Pos { x, y, floor: 0 }
}

/// Tiny deterministic xorshift for test inputs (keeps tests independent of
/// the generator RNG).
pub(super) struct TestRng(u64);

impl TestRng {
    pub fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    /// Uniform in `[lo, hi)`.
    pub fn range(&mut self, lo: f32, hi: f32) -> f32 {
        let u = (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32;
        lo + (hi - lo) * u
    }
}
