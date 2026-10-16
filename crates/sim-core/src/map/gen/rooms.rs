//! Shared building pieces: shells, BSP room partitions, doors, windows,
//! stairs. All writes go through `set_sym` (point-symmetric maps).

use rand::seq::SliceRandom;
use rand::Rng;
use rand_chacha::ChaCha8Rng;

use super::{is_canonical, set_sym, Rect};
use crate::map::{GameMap, Tile};

/// Rooms no larger than this (both sides) may stop subdividing.
const MAX_ROOM: i32 = 7;
/// Chance a small-enough room stops subdividing.
const STOP_CHANCE: f64 = 0.35;
/// Chance an interior wall tile (between two rooms) becomes a window.
const INTERIOR_WINDOW_CHANCE: f64 = 0.06;
/// Chance an exterior wall tile (floor 0) becomes a window.
const EXTERIOR_WINDOW_CHANCE: f64 = 0.2;
/// Chance a placed door starts open.
const DOOR_OPEN_CHANCE: f64 = 0.5;

/// A straight wall line (`vertical` => constant x).
#[derive(Clone, Copy, Debug)]
pub(super) struct Seg {
    pub r: Rect,
    pub vertical: bool,
}

fn door_tile(rng: &mut ChaCha8Rng) -> Tile {
    if rng.gen_bool(DOOR_OPEN_CHANCE) {
        Tile::DoorOpen
    } else {
        Tile::DoorClosed
    }
}

/// Perimeter walls on `outer`, plain floor inside, on floor `f`.
pub(super) fn shell(m: &mut GameMap, outer: Rect, f: i32) {
    for (x, y) in outer.cells() {
        let edge = x == outer.x0 || x == outer.x1 || y == outer.y0 || y == outer.y1;
        set_sym(m, x, y, f, if edge { Tile::Wall } else { Tile::Floor });
    }
}

/// Recursively split `interior` (all floor) with one-tile walls into rooms
/// of at least `min_room` per side. Returns the walls in creation order.
pub(super) fn partition(
    m: &mut GameMap,
    rng: &mut ChaCha8Rng,
    interior: Rect,
    f: i32,
    min_room: i32,
) -> Vec<Seg> {
    let mut segs = Vec::new();
    let mut stack = vec![interior];
    while let Some(r) = stack.pop() {
        let can_v = r.w() > 2 * min_room;
        let can_h = r.h() > 2 * min_room;
        if !can_v && !can_h {
            continue;
        }
        if r.w() <= MAX_ROOM && r.h() <= MAX_ROOM && rng.gen_bool(STOP_CHANCE) {
            continue;
        }
        let vertical = match (can_v, can_h) {
            (true, true) if r.w() == r.h() => rng.gen_bool(0.5),
            (true, true) => r.w() > r.h(),
            (v, _) => v,
        };
        if vertical {
            let x = rng.gen_range(r.x0 + min_room..=r.x1 - min_room);
            let wall = Rect { x0: x, x1: x, ..r };
            stack.push(Rect { x1: x - 1, ..r });
            stack.push(Rect { x0: x + 1, ..r });
            segs.push(Seg { r: wall, vertical });
        } else {
            let y = rng.gen_range(r.y0 + min_room..=r.y1 - min_room);
            let wall = Rect { y0: y, y1: y, ..r };
            stack.push(Rect { y1: y - 1, ..r });
            stack.push(Rect { y0: y + 1, ..r });
            segs.push(Seg { r: wall, vertical });
        }
        for (x, y) in segs[segs.len() - 1].r.cells() {
            set_sym(m, x, y, f, Tile::Wall);
        }
    }
    segs
}

/// Wall tiles of `seg` (canonical half only) with plain floor on both sides.
fn seg_openings(m: &GameMap, seg: Seg, f: i32) -> Vec<(i32, i32)> {
    let (dx, dy) = if seg.vertical { (1, 0) } else { (0, 1) };
    seg.r
        .cells()
        .filter(|&(x, y)| {
            is_canonical(m, x, y)
                && m.tile(x, y, f) == Tile::Wall
                && m.tile(x - dx, y - dy, f) == Tile::Floor
                && m.tile(x + dx, y + dy, f) == Tile::Floor
        })
        .collect()
}

/// Put one door (two on long walls, sometimes) in every wall. Returns
/// `false` if some wall has no valid door position (caller retries).
pub(super) fn place_seg_doors(m: &mut GameMap, rng: &mut ChaCha8Rng, segs: &[Seg], f: i32) -> bool {
    for &seg in segs {
        let spots = seg_openings(m, seg, f);
        let Some(&(x, y)) = spots.choose(rng) else {
            return false;
        };
        let tile = door_tile(rng);
        set_sym(m, x, y, f, tile);
        if spots.len() >= 8 && rng.gen_bool(0.3) {
            let far: Vec<(i32, i32)> = spots
                .iter()
                .copied()
                .filter(|&(sx, sy)| (sx - x).abs() + (sy - y).abs() >= 4)
                .collect();
            if let Some(&(x2, y2)) = far.choose(rng) {
                let tile = door_tile(rng);
                set_sym(m, x2, y2, f, tile);
            }
        }
    }
    true
}

/// Turn a few interior wall tiles into windows (see-through, no passage).
pub(super) fn place_seg_windows(m: &mut GameMap, rng: &mut ChaCha8Rng, segs: &[Seg], f: i32) {
    for &seg in segs {
        for (x, y) in seg_openings(m, seg, f) {
            if rng.gen_bool(INTERIOR_WINDOW_CHANCE) {
                set_sym(m, x, y, f, Tile::Window);
            }
        }
    }
}

/// Perimeter wall tiles of `outer` (canonical half, not corners) with
/// plain floor inside and walkable ground outside.
fn perimeter_openings(m: &GameMap, outer: Rect, f: i32) -> Vec<(i32, i32)> {
    outer
        .cells()
        .filter(|&(x, y)| {
            let on_x = x == outer.x0 || x == outer.x1;
            let on_y = y == outer.y0 || y == outer.y1;
            if on_x == on_y || !is_canonical(m, x, y) || m.tile(x, y, f) != Tile::Wall {
                return false; // interior tile or corner
            }
            let (ox, oy) = if on_x {
                (if x == outer.x0 { -1 } else { 1 }, 0)
            } else {
                (0, if y == outer.y0 { -1 } else { 1 })
            };
            m.tile(x - ox, y - oy, f) == Tile::Floor && m.is_walkable(x + ox, y + oy, f)
        })
        .collect()
}

/// Exterior doors (`doors` of them, at least one) and windows on floor `f`.
/// Returns `false` if no door could be placed.
pub(super) fn place_exterior_openings(
    m: &mut GameMap,
    rng: &mut ChaCha8Rng,
    outer: Rect,
    f: i32,
    doors: usize,
) -> bool {
    let mut spots = perimeter_openings(m, outer, f);
    if spots.is_empty() {
        return false;
    }
    spots.shuffle(rng);
    let (door_spots, rest) = spots.split_at(doors.clamp(1, spots.len()));
    for &(x, y) in door_spots {
        set_sym(m, x, y, f, Tile::DoorClosed);
    }
    for &(x, y) in rest {
        let beside_door = [(1, 0), (-1, 0), (0, 1), (0, -1)]
            .iter()
            .any(|&(dx, dy)| m.tile(x + dx, y + dy, f).is_door());
        if !beside_door && rng.gen_bool(EXTERIOR_WINDOW_CHANCE) {
            set_sym(m, x, y, f, Tile::Window);
        }
    }
    true
}

/// Place `count` stair pairs (StairsUp on `f`, StairsDown on `f + 1`) on
/// tiles of `interior` that are plain floor on both floors and not next to
/// a door. Returns `false` if there is no room for them.
pub(super) fn place_stairs(
    m: &mut GameMap,
    rng: &mut ChaCha8Rng,
    interior: Rect,
    f: i32,
    count: usize,
) -> bool {
    for _ in 0..count {
        let spots: Vec<(i32, i32)> = interior
            .cells()
            .filter(|&(x, y)| {
                let (rx, ry, _) = m.rotate(x, y, f);
                (x, y) < (rx, ry)
                    && m.tile(x, y, f) == Tile::Floor
                    && m.tile(x, y, f + 1) == Tile::Floor
                    && [(1, 0), (-1, 0), (0, 1), (0, -1)].iter().all(|&(dx, dy)| {
                        !m.tile(x + dx, y + dy, f).is_door()
                            && !m.tile(x + dx, y + dy, f + 1).is_door()
                    })
            })
            .collect();
        let Some(&(x, y)) = spots.choose(rng) else {
            return false;
        };
        set_sym(m, x, y, f, Tile::StairsUp);
        set_sym(m, x, y, f + 1, Tile::StairsDown);
    }
    true
}
