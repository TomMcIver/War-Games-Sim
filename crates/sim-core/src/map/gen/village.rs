//! "village": 65 x 65 walled map with a street grid. Each lot in the
//! canonical half usually holds a house of random size (one or two floors,
//! BSP rooms, exterior doors and windows, stairs when two floors); cover
//! blocks are scattered over streets and empty lots. Teams spawn along
//! opposite edges. Floor 1 is Void except for upper storeys.

use rand::Rng;
use rand_chacha::ChaCha8Rng;

use super::rooms::{
    partition, place_exterior_openings, place_seg_doors, place_seg_windows, place_stairs,
    shell,
};
use super::{bordered, install_edge_spawns, scatter_cover, Rect, SPAWN_DEPTH};
use crate::map::GameMap;

const SIZE: i32 = 65;
/// Lot columns / rows in the canonical (low-x) half; 3-4 tile streets
/// between them and a 5-tile boulevard down the centre column.
const LOT_XS: [(i32, i32); 2] = [(6, 17), (21, 29)];
const LOT_YS: [(i32, i32); 4] = [(4, 15), (19, 30), (34, 45), (49, 60)];
const HOUSE_CHANCE: f64 = 0.8;
const TWO_FLOOR_CHANCE: f64 = 0.5;
const MIN_HOUSE: i32 = 5;
const MIN_ROOM: i32 = 2;

pub(super) fn build(rng: &mut ChaCha8Rng, _team_size: usize) -> Option<GameMap> {
    let mut m = bordered(SIZE, SIZE, 2);
    for &(lx0, lx1) in &LOT_XS {
        for &(ly0, ly1) in &LOT_YS {
            if !rng.gen_bool(HOUSE_CHANCE) {
                continue;
            }
            let lot = Rect { x0: lx0, y0: ly0, x1: lx1, y1: ly1 };
            if !house(&mut m, rng, lot) {
                return None;
            }
        }
    }
    let cx = (SIZE - 1) / 2;
    let region = Rect { x0: SPAWN_DEPTH + 2, y0: 1, x1: cx - 1, y1: SIZE - 2 };
    let cover = rng.gen_range(18..=26);
    scatter_cover(&mut m, rng, region, cover);
    install_edge_spawns(&mut m);
    Some(m)
}

/// Random-size house somewhere inside `lot`.
fn house(m: &mut GameMap, rng: &mut ChaCha8Rng, lot: Rect) -> bool {
    let w = rng.gen_range(MIN_HOUSE..=lot.w());
    let h = rng.gen_range(MIN_HOUSE..=lot.h());
    let x0 = rng.gen_range(lot.x0..=lot.x1 - w + 1);
    let y0 = rng.gen_range(lot.y0..=lot.y1 - h + 1);
    let outer = Rect { x0, y0, x1: x0 + w - 1, y1: y0 + h - 1 };
    let floors = if w >= 7 && h >= 7 && rng.gen_bool(TWO_FLOOR_CHANCE) { 2 } else { 1 };
    for f in 0..floors {
        shell(m, outer, f);
        let segs = partition(m, rng, outer.inner(), f, MIN_ROOM);
        if !place_seg_doors(m, rng, &segs, f) {
            return false;
        }
        place_seg_windows(m, rng, &segs, f);
    }
    let doors = rng.gen_range(1..=2);
    if !place_exterior_openings(m, rng, outer, 0, doors) {
        return false;
    }
    floors == 1 || place_stairs(m, rng, outer.inner(), 0, 1)
}
