//! Exact grid traversal (Amanatides & Woo DDA): line of sight and bullet rays.
//!
//! Corner rule: when a segment passes exactly through a grid vertex it is
//! blocked only if *both* tiles beside the vertex block (two walls touching
//! diagonally form a sealed line; grazing a single wall's corner does not
//! block). This makes the result symmetric in its endpoints.

use super::{GameMap, Tile};
use crate::types::Pos;

/// Tolerance (in segment parameter units) for treating two axis crossings as
/// one simultaneous vertex crossing.
const CORNER_EPS: f64 = 1e-9;

/// Result of [`GameMap::raycast`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RayHit {
    /// Distance travelled before stopping (`max_range` if nothing was hit).
    pub distance: f32,
    /// Point where the ray stopped.
    pub x: f32,
    pub y: f32,
    /// The bullet-blocking tile that stopped the ray, if any.
    pub tile: Option<(i32, i32)>,
}

impl RayHit {
    /// True if the ray was stopped by a tile before `max_range`.
    pub fn blocked(&self) -> bool {
        self.tile.is_some()
    }
}

/// Walk the tiles crossed by segment `a -> b`, excluding the start tile.
/// Returns the first tile for which `blocked` holds, with the segment
/// parameter `t` in `[0, 1]` at which the segment enters it. The end tile is
/// tested only when `include_end` is set. Allocation-free.
#[inline]
fn first_blocking<F: Fn(i32, i32) -> bool>(
    (ax, ay): (f64, f64),
    (bx, by): (f64, f64),
    include_end: bool,
    blocked: F,
) -> Option<(i32, i32, f64)> {
    let (mut cx, mut cy) = (ax.floor() as i32, ay.floor() as i32);
    let (ex, ey) = (bx.floor() as i32, by.floor() as i32);
    let (dx, dy) = (bx - ax, by - ay);
    // Step direction comes from the cell difference so the per-axis step
    // counts are exact and float drift can never overshoot the end cell.
    let sx = (ex - cx).signum();
    let sy = (ey - cy).signum();
    let (tdx, mut tmx) = axis_params(ax, dx, cx, sx);
    let (tdy, mut tmy) = axis_params(ay, dy, cy, sy);
    let (mut rx, mut ry) = ((ex - cx).abs(), (ey - cy).abs());

    while rx > 0 || ry > 0 {
        let t;
        if rx > 0 && ry > 0 && (tmx - tmy).abs() <= CORNER_EPS {
            t = tmx.min(tmy);
            if blocked(cx + sx, cy) && blocked(cx, cy + sy) {
                return Some((cx + sx, cy, t));
            }
            cx += sx;
            cy += sy;
            tmx += tdx;
            tmy += tdy;
            rx -= 1;
            ry -= 1;
        } else if ry == 0 || (rx > 0 && tmx < tmy) {
            t = tmx;
            cx += sx;
            tmx += tdx;
            rx -= 1;
        } else {
            t = tmy;
            cy += sy;
            tmy += tdy;
            ry -= 1;
        }
        if rx == 0 && ry == 0 && !include_end {
            return None;
        }
        if blocked(cx, cy) {
            return Some((cx, cy, t.clamp(0.0, 1.0)));
        }
    }
    None
}

/// `(t_delta, t_max)` for one axis: parameter spacing between grid lines and
/// the parameter of the first grid line crossed.
#[inline]
fn axis_params(a: f64, d: f64, c: i32, s: i32) -> (f64, f64) {
    match s {
        1 => (1.0 / d, (f64::from(c + 1) - a) / d),
        -1 => (-1.0 / d, (a - f64::from(c)) / -d),
        _ => (f64::INFINITY, f64::INFINITY),
    }
}

impl GameMap {
    /// Fast tile predicate on one floor (out of bounds counts as Void).
    #[inline]
    fn floor_pred(&self, f: usize, pred: fn(Tile) -> bool) -> impl Fn(i32, i32) -> bool + '_ {
        let (w, h) = (self.width as i32, self.height as i32);
        let base = f * self.width * self.height;
        move |x, y| {
            if x < 0 || y < 0 || x >= w || y >= h {
                pred(Tile::Void)
            } else {
                pred(self.tiles[base + (y * w + x) as usize])
            }
        }
    }

    /// Exact line of sight between two positions. False across floors.
    /// Walls, closed doors and Void block; windows and open doors do not.
    /// The tiles containing `a` and `b` themselves never block, so a wall or
    /// door tile is "visible" when the line reaches it.
    pub fn line_of_sight(&self, a: Pos, b: Pos) -> bool {
        if a.floor != b.floor || a.floor as usize >= self.floors {
            return false;
        }
        if !(a.x.is_finite() && a.y.is_finite() && b.x.is_finite() && b.y.is_finite()) {
            return false;
        }
        let pred = self.floor_pred(a.floor as usize, Tile::blocks_sight);
        first_blocking(
            (f64::from(a.x), f64::from(a.y)),
            (f64::from(b.x), f64::from(b.y)),
            false,
            pred,
        )
        .is_none()
    }

    /// Cast a bullet ray from `from` along `angle` (direction
    /// `(cos angle, sin angle)`) for up to `max_range` tiles. Stops at the
    /// first wall, closed door or Void tile (windows don't stop bullets).
    /// The tile containing `from` is ignored.
    pub fn raycast(&self, from: Pos, angle: f32, max_range: f32) -> RayHit {
        let miss = |d: f32| RayHit {
            distance: d,
            x: from.x + angle.cos() * d,
            y: from.y + angle.sin() * d,
            tile: None,
        };
        let valid = from.x.is_finite() && from.y.is_finite() && angle.is_finite();
        if !valid || max_range.is_nan() || max_range <= 0.0 {
            return miss(0.0);
        }
        if from.floor as usize >= self.floors {
            return RayHit {
                distance: 0.0,
                x: from.x,
                y: from.y,
                tile: None,
            };
        }
        let range = f64::from(max_range.min(1.0e6));
        let (ax, ay) = (f64::from(from.x), f64::from(from.y));
        let a = f64::from(angle);
        let (bx, by) = (ax + a.cos() * range, ay + a.sin() * range);
        let pred = self.floor_pred(from.floor as usize, Tile::blocks_bullets);
        match first_blocking((ax, ay), (bx, by), true, pred) {
            None => miss(range as f32),
            Some((tx, ty, t)) => RayHit {
                distance: (t * range) as f32,
                x: (ax + (bx - ax) * t) as f32,
                y: (ay + (by - ay) * t) as f32,
                tile: Some((tx, ty)),
            },
        }
    }
}
