//! "arena": walled open field with seeded cover blocks.
//! Size `(23 + 2n) x (17 + 2n)` for team size `n` (always odd, so the
//! map has a centre column/row), one floor.

use rand::Rng;
use rand_chacha::ChaCha8Rng;

use super::{bordered, install_edge_spawns, scatter_cover, Rect, SPAWN_DEPTH};
use crate::map::GameMap;

pub(super) fn build(rng: &mut ChaCha8Rng, team_size: usize) -> GameMap {
    let n = team_size as i32;
    let (w, h) = (23 + 2 * n, 17 + 2 * n);
    let mut m = bordered(w, h, 1);
    let cx = (w - 1) / 2;
    // Canonical half, clear of the spawn columns.
    let region = Rect { x0: SPAWN_DEPTH + 2, y0: 1, x1: cx - 1, y1: h - 2 };
    let count = 3 + n as u32 + rng.gen_range(0..=3);
    scatter_cover(&mut m, rng, region, count);
    install_edge_spawns(&mut m);
    m
}
