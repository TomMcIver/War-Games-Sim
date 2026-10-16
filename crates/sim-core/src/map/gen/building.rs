//! "building": one two-floor building (25 x 21 footprint) in a walled
//! 41 x 29 yard. Each floor is split by a central spine wall (with doors)
//! and BSP rooms with doors and interior windows; the ground floor has
//! exterior doors and windows; two stair pairs link the floors. Teams spawn
//! in the yard on opposite short sides. Floor 1 outside the building is Void.

use rand::Rng;
use rand_chacha::ChaCha8Rng;

use super::rooms::{
    partition, place_exterior_openings, place_seg_doors, place_seg_windows, place_stairs, shell,
    Seg,
};
use super::{bordered, install_edge_spawns, scatter_cover, Rect, SPAWN_DEPTH};
use crate::map::GameMap;

const W: i32 = 41;
const H: i32 = 29;
/// Building outer wall rectangle (point-symmetric about the map centre).
const OUTER: Rect = Rect {
    x0: 8,
    y0: 4,
    x1: W - 1 - 8,
    y1: H - 1 - 4,
};
const MIN_ROOM: i32 = 3;

pub(super) fn build(rng: &mut ChaCha8Rng, _team_size: usize) -> Option<GameMap> {
    let mut m = bordered(W, H, 2);
    let cx = (W - 1) / 2;
    // Interior of the canonical (low-x) half, up to the spine wall.
    let half = Rect {
        x0: OUTER.x0 + 1,
        y0: OUTER.y0 + 1,
        x1: cx - 1,
        y1: OUTER.y1 - 1,
    };
    let spine = Seg {
        r: Rect {
            x0: cx,
            x1: cx,
            ..OUTER.inner()
        },
        vertical: true,
    };

    for f in 0..2 {
        shell(&mut m, OUTER, f);
        for (x, y) in spine.r.cells() {
            super::set_sym(&mut m, x, y, f, crate::map::Tile::Wall);
        }
        let mut segs = partition(&mut m, rng, half, f, MIN_ROOM);
        segs.push(spine);
        if !place_seg_doors(&mut m, rng, &segs, f) {
            return None;
        }
        place_seg_windows(&mut m, rng, &segs[..segs.len() - 1], f);
    }
    let exterior_doors = rng.gen_range(2..=3);
    if !place_exterior_openings(&mut m, rng, OUTER, 0, exterior_doors) {
        return None;
    }
    if !place_stairs(&mut m, rng, half, 0, 1) {
        return None;
    }
    let yard = Rect {
        x0: SPAWN_DEPTH + 2,
        y0: 1,
        x1: cx - 1,
        y1: H - 2,
    };
    let cover = rng.gen_range(3..=6);
    scatter_cover(&mut m, rng, yard, cover);
    install_edge_spawns(&mut m);
    Some(m)
}
