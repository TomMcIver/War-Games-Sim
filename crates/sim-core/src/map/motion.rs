//! Circle-vs-tile movement collision and wall sliding.

use super::GameMap;
use crate::types::Pos;

/// Sub-step length as a fraction of the unit radius (prevents tunnelling).
const SUBSTEP_FRACTION: f32 = 0.5;
/// Smallest sub-step, so tiny radii don't explode the step count.
const MIN_SUBSTEP: f32 = 0.05;
/// Bisection iterations when sliding up to a wall.
const SLIDE_BISECT_ITERS: u32 = 10;

impl GameMap {
    /// True if a circle of `radius` centred at `(x, y)` on floor `f` overlaps
    /// no non-walkable tile (walls, windows, closed doors, Void, outside the
    /// map). Touching a tile edge exactly is allowed.
    pub fn circle_fits(&self, x: f32, y: f32, f: i32, radius: f32) -> bool {
        if !(x.is_finite() && y.is_finite()) {
            return false;
        }
        let r = radius.max(0.0);
        if !self.is_walkable(x.floor() as i32, y.floor() as i32, f) {
            return false;
        }
        let (x0, x1) = ((x - r).floor() as i32, (x + r).floor() as i32);
        let (y0, y1) = ((y - r).floor() as i32, (y + r).floor() as i32);
        let r2 = r * r;
        for ty in y0..=y1 {
            for tx in x0..=x1 {
                if self.is_walkable(tx, ty, f) {
                    continue;
                }
                // Closest point of the tile square to the circle centre.
                let cx = x.clamp(tx as f32, tx as f32 + 1.0);
                let cy = y.clamp(ty as f32, ty as f32 + 1.0);
                let (ddx, ddy) = (x - cx, y - cy);
                if ddx * ddx + ddy * ddy < r2 {
                    return false;
                }
            }
        }
        true
    }

    /// Can a circle of `radius` move in a straight line from `from` to `to`?
    /// The path is swept in sub-steps of at most `radius / 2`, so it cannot
    /// tunnel through thin walls. Floor changes are never valid moves (use
    /// [`GameMap::stairs_destination`]). The start position itself is not
    /// checked, so a unit overlapping a blocked tile can still move away if
    /// the first sub-step is free.
    pub fn can_move(&self, from: Pos, to: Pos, radius: f32) -> bool {
        if from.floor != to.floor {
            return false;
        }
        let (dx, dy) = (to.x - from.x, to.y - from.y);
        let dist = (dx * dx + dy * dy).sqrt();
        if !dist.is_finite() {
            return false;
        }
        let step = (radius * SUBSTEP_FRACTION).max(MIN_SUBSTEP);
        let n = (dist / step).ceil().max(1.0) as u32;
        let f = i32::from(from.floor);
        (1..=n).all(|i| {
            let s = i as f32 / n as f32;
            let (x, y) = if i == n {
                (to.x, to.y)
            } else {
                (from.x + dx * s, from.y + dy * s)
            };
            self.circle_fits(x, y, f, radius)
        })
    }

    /// Move by `(dx, dy)` sliding along walls: tries the full move, otherwise
    /// moves each axis separately (larger component first), advancing as far
    /// as possible toward the obstacle. Deterministic; returns the new
    /// position (possibly `from`).
    pub fn slide_move(&self, from: Pos, dx: f32, dy: f32, radius: f32) -> Pos {
        if !(dx.is_finite() && dy.is_finite()) {
            return from;
        }
        let full = Pos {
            x: from.x + dx,
            y: from.y + dy,
            ..from
        };
        if self.can_move(from, full, radius) {
            return full;
        }
        if dx.abs() >= dy.abs() {
            let p = self.advance(from, dx, 0.0, radius);
            self.advance(p, 0.0, dy, radius)
        } else {
            let p = self.advance(from, 0.0, dy, radius);
            self.advance(p, dx, 0.0, radius)
        }
    }

    /// Furthest point along `(dx, dy)` reachable from `from` (bisection).
    fn advance(&self, from: Pos, dx: f32, dy: f32, radius: f32) -> Pos {
        let at = |s: f32| Pos {
            x: from.x + dx * s,
            y: from.y + dy * s,
            ..from
        };
        if dx == 0.0 && dy == 0.0 {
            return from;
        }
        if self.can_move(from, at(1.0), radius) {
            return at(1.0);
        }
        let (mut lo, mut hi) = (0.0f32, 1.0f32);
        for _ in 0..SLIDE_BISECT_ITERS {
            let mid = 0.5 * (lo + hi);
            if self.can_move(from, at(mid), radius) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        if lo == 0.0 {
            from
        } else {
            at(lo)
        }
    }
}
