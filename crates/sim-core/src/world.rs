//! OWNER: engine-agent. Unit and World state plus per-tick physics:
//! turning, movement with circle collisions, doors and stairs.
//!
//! Conventions:
//! - Positions are continuous, in tile units; tile `(tx, ty)` covers
//!   `[tx, tx+1) x [ty, ty+1)`. Floors are discrete.
//! - `yaw` is in radians, normalised to `[-PI, PI)`. Facing vector is
//!   `(cos yaw, sin yaw)`. "Left" is `+90deg` from facing: turning left
//!   increases yaw, strafing left moves along `(-sin yaw, cos yaw)`.
//! - Units are circles of radius `UnitStats::radius` (must stay < 0.5: map
//!   doorways and corridors are one tile wide). Dead units neither move,
//!   block movement nor can be hit.
//! - Wall collision, doors and stairs use the map API (`slide_move`,
//!   `circle_fits`, `toggle_door`, `stairs_destination`); this file only
//!   adds unit-vs-unit collision and the engine rules on top.

use std::f32::consts::{PI, TAU};

use crate::map::{GameMap, Tile};
use crate::types::{Action, Pos, TeamId, UnitId};

/// Default unit hit points (also used by `Unit::new`).
pub const DEFAULT_MAX_HP: f32 = 100.0;
/// Default rounds per unit per episode (no reloads).
pub const DEFAULT_AMMO: u32 = 30;

/// Body/movement physics shared by all units. Speeds are per second and
/// get scaled by `EnvConfig::tick_seconds`.
#[derive(Clone, Debug, PartialEq)]
pub struct UnitStats {
    pub max_hp: f32,
    /// Collision and hit radius, in tiles.
    pub radius: f32,
    /// Tiles per second when moving forward.
    pub forward_speed: f32,
    /// Tiles per second when moving backward.
    pub back_speed: f32,
    /// Tiles per second when strafing.
    pub strafe_speed: f32,
    /// Radians per second.
    pub turn_rate: f32,
    /// Ticks a unit must wait after a successful `interact` before the next
    /// one (prevents door/stairs flicker when interact is held). Tracked by
    /// `Env`, not `Unit`, so `Unit` stays a plain literal-constructible struct.
    pub interact_cooldown_ticks: u32,
}

impl Default for UnitStats {
    fn default() -> Self {
        Self {
            max_hp: DEFAULT_MAX_HP,
            radius: 0.3,
            forward_speed: 3.0,
            back_speed: 1.5,
            strafe_speed: 2.0,
            turn_rate: 1.5,
            interact_cooldown_ticks: 5,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Unit {
    pub id: UnitId,
    pub team: TeamId,
    pub pos: Pos,
    /// Facing angle in radians.
    pub yaw: f32,
    pub hp: f32,
    pub ammo: u32,
    pub alive: bool,
    /// Ticks until the weapon can fire again.
    pub cooldown: u32,
}

impl Unit {
    /// A live unit with default hp/ammo, facing +x, ready to fire.
    pub fn new(id: UnitId, team: TeamId, pos: Pos) -> Self {
        Self {
            id,
            team,
            pos,
            yaw: 0.0,
            hp: DEFAULT_MAX_HP,
            ammo: DEFAULT_AMMO,
            alive: true,
            cooldown: 0,
        }
    }

    /// Unit facing vector `(cos yaw, sin yaw)`.
    pub fn facing(&self) -> (f32, f32) {
        (self.yaw.cos(), self.yaw.sin())
    }
}

/// A shot fired this tick (for replays/tracers). `hit` = unit that was struck.
#[derive(Clone, Debug)]
pub struct ShotEvent {
    pub shooter: UnitId,
    /// Where the shot ended (hit point or max range / wall impact).
    pub x: f32,
    pub y: f32,
    pub floor: u8,
    pub hit: Option<UnitId>,
}

#[derive(Clone, Debug)]
pub struct World {
    pub map: GameMap,
    pub units: Vec<Unit>,
    pub tick: u64,
    /// Shots fired during the most recent tick.
    pub shots: Vec<ShotEvent>,
}

/// What an `interact` did (for tests / debugging).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InteractOutcome {
    None,
    ChangedFloor,
    DoorOpened,
    DoorClosed,
}

impl World {
    /// Empty world on `map` at tick 0. Push units by hand for tests.
    pub fn new(map: GameMap) -> Self {
        Self {
            map,
            units: Vec::new(),
            tick: 0,
            shots: Vec::new(),
        }
    }

    /// Number of living units on `team`.
    pub fn alive_count(&self, team: TeamId) -> usize {
        self.units
            .iter()
            .filter(|u| u.alive && u.team == team)
            .count()
    }

    /// Rotate unit `i` according to the `turn` head (0 none, 1 left, 2 right).
    pub(crate) fn turn_unit(&mut self, i: usize, turn: u8, stats: &UnitStats, dt: f32) {
        let sign = match turn {
            1 => 1.0,
            2 => -1.0,
            _ => return,
        };
        let u = &mut self.units[i];
        u.yaw = wrap_angle(u.yaw + sign * stats.turn_rate * dt);
    }

    /// Move unit `i` from its forward/strafe heads, sliding along walls and
    /// other units. Returns true if the unit actually changed position.
    pub(crate) fn move_unit(&mut self, i: usize, a: &Action, stats: &UnitStats, dt: f32) -> bool {
        let (fx, fy) = self.units[i].facing();
        let fwd = match a.forward {
            1 => stats.forward_speed,
            2 => -stats.back_speed,
            _ => 0.0,
        };
        let side = match a.strafe {
            1 => stats.strafe_speed,
            2 => -stats.strafe_speed,
            _ => 0.0,
        };
        // Left of facing is (-fy, fx).
        let dx = (fwd * fx - side * fy) * dt;
        let dy = (fwd * fy + side * fx) * dt;
        let len = (dx * dx + dy * dy).sqrt();
        let max_len = stats
            .forward_speed
            .max(stats.strafe_speed)
            .max(stats.back_speed)
            * dt;
        if len <= 0.0 || !len.is_finite() {
            return false;
        }
        let scale = if len > max_len { max_len / len } else { 1.0 };
        let (dx, dy) = (dx * scale, dy * scale);

        // Sub-step so a single step never exceeds half a radius (no tunnelling).
        let step_len = (stats.radius * 0.5).max(0.05);
        let n_sub = ((len * scale) / step_len).ceil().clamp(1.0, 64.0) as usize;
        let (sx, sy) = (dx / n_sub as f32, dy / n_sub as f32);
        let start = self.units[i].pos;
        for _ in 0..n_sub {
            self.units[i].pos = self.slide_axis(i, sx, 0.0, stats.radius);
            self.units[i].pos = self.slide_axis(i, 0.0, sy, stats.radius);
        }
        let end = self.units[i].pos;
        end.x != start.x || end.y != start.y
    }

    /// Furthest free position along the axis move `(dx, dy)` from unit i's
    /// position: walls are resolved by the map (sliding up to them), then
    /// other units by bisection on the remaining wall-free segment.
    fn slide_axis(&self, i: usize, dx: f32, dy: f32, radius: f32) -> Pos {
        const BISECT_STEPS: usize = 8;
        let p = self.units[i].pos;
        if dx == 0.0 && dy == 0.0 {
            return p;
        }
        let q = self.map.slide_move(p, dx, dy, radius);
        if !self.units_block(i, q, radius) {
            return q;
        }
        let at = |t: f32| Pos {
            x: p.x + (q.x - p.x) * t,
            y: p.y + (q.y - p.y) * t,
            floor: p.floor,
        };
        let (mut lo, mut hi) = (0.0f32, 1.0f32);
        for _ in 0..BISECT_STEPS {
            let mid = 0.5 * (lo + hi);
            if self.units_block(i, at(mid), radius) {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        at(lo)
    }

    /// Would unit `i` standing at `p` overlap another living unit? Overlaps
    /// that the move does not make worse are allowed so units can never get
    /// permanently stuck together.
    fn units_block(&self, i: usize, p: Pos, radius: f32) -> bool {
        let cur = self.units[i].pos;
        let min_d2 = (2.0 * radius) * (2.0 * radius);
        self.units.iter().enumerate().any(|(j, o)| {
            if j == i || !o.alive || o.pos.floor != p.floor {
                return false;
            }
            let d2_new = dist2(p, o.pos);
            d2_new < min_d2 && d2_new < dist2(cur, o.pos)
        })
    }

    /// `interact=1` for unit `i`: use stairs when standing on them, else
    /// toggle the door in the tile directly in front. (The interact
    /// cooldown is enforced by `Env`.)
    pub(crate) fn interact(&mut self, i: usize, stats: &UnitStats) -> InteractOutcome {
        match self.use_stairs(i, stats.radius) {
            InteractOutcome::None => self.toggle_door_ahead(i, stats.radius),
            changed => changed,
        }
    }

    /// Whether interacting now would change the world. Env applies interaction
    /// before movement so this is also the mask for the next action tick.
    pub(crate) fn can_interact(&self, i: usize, radius: f32) -> bool {
        let u = &self.units[i];
        if !u.alive {
            return false;
        }
        let p = u.pos;
        let (sx, sy) = (p.x.floor() as i32, p.y.floor() as i32);
        if let Some((_, _, nf)) = self.map.stairs_destination(sx, sy, i32::from(p.floor)) {
            let floor = nf as u8;
            let exact = Pos { floor, ..p };
            let centre = Pos {
                x: sx as f32 + 0.5,
                y: sy as f32 + 0.5,
                floor,
            };
            if [exact, centre].into_iter().any(|c| {
                self.map.circle_fits(c.x, c.y, nf, radius)
                    && !self.units.iter().enumerate().any(|(j, o)| {
                        j != i
                            && o.alive
                            && o.pos.floor == floor
                            && dist2(c, o.pos) < 4.0 * radius * radius
                    })
            }) {
                return true;
            }
        }
        let (fx, fy) = u.facing();
        let reach = radius + 0.5;
        let tx = (p.x + fx * reach).floor() as i32;
        let ty = (p.y + fy * reach).floor() as i32;
        let f = i32::from(p.floor);
        match self.map.tile(tx, ty, f) {
            Tile::DoorClosed => true,
            Tile::DoorOpen => {
                let mut closed = self.map.clone();
                closed.toggle_door(tx, ty, f);
                self.units.iter().all(|o| {
                    !o.alive
                        || i32::from(o.pos.floor) != f
                        || closed.circle_fits(o.pos.x, o.pos.y, f, radius)
                })
            }
            _ => false,
        }
    }

    fn use_stairs(&mut self, i: usize, radius: f32) -> InteractOutcome {
        let p = self.units[i].pos;
        let (tx, ty) = (p.x.floor() as i32, p.y.floor() as i32);
        let Some((_, _, nf)) = self.map.stairs_destination(tx, ty, i32::from(p.floor)) else {
            return InteractOutcome::None;
        };
        let floor = nf as u8;
        // Prefer the exact same (x, y); fall back to the stairs tile centre.
        let exact = Pos { floor, ..p };
        let centre = Pos {
            x: tx as f32 + 0.5,
            y: ty as f32 + 0.5,
            floor,
        };
        let free = [exact, centre].into_iter().find(|&c| {
            self.map.circle_fits(c.x, c.y, nf, radius)
                && !self.units.iter().enumerate().any(|(j, o)| {
                    j != i
                        && o.alive
                        && o.pos.floor == floor
                        && dist2(c, o.pos) < 4.0 * radius * radius
                })
        });
        match free {
            Some(c) => {
                self.units[i].pos = c;
                InteractOutcome::ChangedFloor
            }
            None => InteractOutcome::None,
        }
    }

    fn toggle_door_ahead(&mut self, i: usize, radius: f32) -> InteractOutcome {
        let u = &self.units[i];
        let (fx, fy) = u.facing();
        let reach = radius + 0.5;
        let tx = (u.pos.x + fx * reach).floor() as i32;
        let ty = (u.pos.y + fy * reach).floor() as i32;
        let f = i32::from(u.pos.floor);
        let was = self.map.tile(tx, ty, f);
        if !self.map.toggle_door(tx, ty, f) {
            return InteractOutcome::None;
        }
        if was == Tile::DoorClosed {
            return InteractOutcome::DoorOpened;
        }
        // Never close a door on top of somebody: undo if any body overlaps it.
        let crushed = self.units.iter().any(|o| {
            o.alive
                && i32::from(o.pos.floor) == f
                && !self.map.circle_fits(o.pos.x, o.pos.y, f, radius)
        });
        if crushed {
            self.map.toggle_door(tx, ty, f);
            return InteractOutcome::None;
        }
        InteractOutcome::DoorClosed
    }
}

/// Wrap an angle into `[-PI, PI)`. Non-finite input maps to 0.
pub(crate) fn wrap_angle(a: f32) -> f32 {
    if !a.is_finite() {
        return 0.0;
    }
    let w = (a + PI).rem_euclid(TAU) - PI;
    if w >= PI {
        -PI
    } else {
        w
    }
}

fn dist2(a: Pos, b: Pos) -> f32 {
    let (dx, dy) = (a.x - b.x, a.y - b.y);
    dx * dx + dy * dy
}

#[cfg(test)]
mod tests {
    use super::*;

    const DT: f32 = 0.1;

    fn pos(x: f32, y: f32) -> Pos {
        Pos { x, y, floor: 0 }
    }

    fn world_with(units: &[(TeamId, Pos, f32)]) -> World {
        let mut w = World::new(GameMap::arena(10, 10));
        for (i, &(team, p, yaw)) in units.iter().enumerate() {
            let mut u = Unit::new(i, team, p);
            u.yaw = yaw;
            w.units.push(u);
        }
        w
    }

    fn fwd() -> Action {
        Action {
            forward: 1,
            ..Action::default()
        }
    }

    #[test]
    fn unit_new_has_defaults() {
        let u = Unit::new(3, 1, pos(2.0, 3.0));
        assert_eq!(
            (u.id, u.team, u.hp, u.ammo, u.alive),
            (3, 1, DEFAULT_MAX_HP, DEFAULT_AMMO, true)
        );
        assert_eq!(u.cooldown, 0);
    }

    /// Other crates/tests build `Unit` and `World` with struct literals; adding
    /// a field breaks them. This fails to compile if the field set changes.
    #[test]
    fn unit_and_world_literal_shape_is_stable() {
        let u = Unit {
            id: 0,
            team: 0,
            pos: pos(1.5, 1.5),
            yaw: 0.0,
            hp: 1.0,
            ammo: 1,
            alive: true,
            cooldown: 0,
        };
        let w = World {
            map: GameMap::arena(4, 4),
            units: vec![u],
            tick: 0,
            shots: Vec::new(),
        };
        assert_eq!(w.units.len(), 1);
    }

    #[test]
    fn forward_moves_along_facing_at_speed() {
        let s = UnitStats::default();
        let mut w = world_with(&[(0, pos(3.5, 5.5), 0.0)]);
        assert!(w.move_unit(0, &fwd(), &s, DT));
        assert!((w.units[0].pos.x - (3.5 + s.forward_speed * DT)).abs() < 1e-5);
        assert!((w.units[0].pos.y - 5.5).abs() < 1e-6);
    }

    #[test]
    fn strafe_left_is_plus_ninety_degrees() {
        let s = UnitStats::default();
        let mut w = world_with(&[(0, pos(5.5, 5.5), 0.0)]);
        let a = Action {
            strafe: 1,
            ..Action::default()
        };
        w.move_unit(0, &a, &s, DT);
        assert!(w.units[0].pos.y > 5.5);
        assert!((w.units[0].pos.x - 5.5).abs() < 1e-5);
    }

    #[test]
    fn movement_blocked_by_walls() {
        let s = UnitStats::default();
        // Facing -x towards the arena border wall at x in [0,1).
        let mut w = world_with(&[(0, pos(2.5, 5.5), PI)]);
        for _ in 0..50 {
            w.move_unit(0, &fwd(), &s, DT);
        }
        let x = w.units[0].pos.x;
        assert!(x >= 1.0 + s.radius - 1e-3, "walked into wall: x={x}");
        assert!(
            x < 1.0 + s.radius + 0.05,
            "should be flush against wall: x={x}"
        );
    }

    #[test]
    fn huge_timestep_does_not_tunnel_through_walls() {
        let s = UnitStats::default();
        let mut map = GameMap::arena(10, 10);
        for y in 0..10 {
            map.set_tile(5, y, 0, Tile::Wall);
        }
        let mut w = World::new(map);
        w.units.push(Unit::new(0, 0, pos(4.0, 5.5)));
        w.move_unit(0, &fwd(), &s, 5.0);
        assert!(w.units[0].pos.x <= 5.0 - s.radius + 1e-3);
    }

    #[test]
    fn units_collide_with_each_other_but_not_dead_ones() {
        let s = UnitStats::default();
        let mut w = world_with(&[(0, pos(3.5, 5.5), 0.0), (1, pos(4.5, 5.5), PI)]);
        for _ in 0..20 {
            w.move_unit(0, &fwd(), &s, DT);
        }
        let d = w.units[1].pos.x - w.units[0].pos.x;
        assert!(d >= 2.0 * s.radius - 1e-3, "units overlap: d={d}");
        w.units[1].alive = false;
        for _ in 0..5 {
            w.move_unit(0, &fwd(), &s, DT);
        }
        assert!(w.units[0].pos.x > w.units[1].pos.x - 2.0 * s.radius + 0.01);
    }

    #[test]
    fn turning_changes_yaw_by_turn_rate() {
        let s = UnitStats::default();
        let mut w = world_with(&[(0, pos(5.5, 5.5), 0.0)]);
        w.turn_unit(0, 1, &s, DT);
        assert!((w.units[0].yaw - s.turn_rate * DT).abs() < 1e-6);
        w.turn_unit(0, 2, &s, DT);
        w.turn_unit(0, 2, &s, DT);
        assert!((w.units[0].yaw + s.turn_rate * DT).abs() < 1e-6);
        w.turn_unit(0, 0, &s, DT);
        assert!((w.units[0].yaw + s.turn_rate * DT).abs() < 1e-6);
    }

    #[test]
    fn yaw_stays_wrapped() {
        let s = UnitStats::default();
        let mut w = world_with(&[(0, pos(5.5, 5.5), 3.0)]);
        for _ in 0..100 {
            w.turn_unit(0, 1, &s, DT);
            let y = w.units[0].yaw;
            assert!((-PI..PI).contains(&y), "yaw out of range: {y}");
        }
        assert_eq!(wrap_angle(f32::NAN), 0.0);
        assert!((wrap_angle(PI) + PI).abs() < 1e-6);
    }

    fn door_world() -> World {
        let mut map = GameMap::arena(10, 10);
        for y in 0..10 {
            map.set_tile(5, y, 0, Tile::Wall);
        }
        map.set_tile(5, 5, 0, Tile::DoorClosed);
        let mut w = World::new(map);
        w.units.push(Unit::new(0, 0, pos(4.5, 5.5)));
        w
    }

    #[test]
    fn door_blocks_until_opened_by_interact() {
        let s = UnitStats::default();
        let mut w = door_world();
        for _ in 0..10 {
            w.move_unit(0, &fwd(), &s, DT);
        }
        assert!(w.units[0].pos.x < 5.0);
        assert_eq!(w.interact(0, &s), InteractOutcome::DoorOpened);
        assert_eq!(w.map.tile(5, 5, 0), Tile::DoorOpen);
        for _ in 0..20 {
            w.move_unit(0, &fwd(), &s, DT);
        }
        assert!(w.units[0].pos.x > 6.0, "should walk through open door");
    }

    #[test]
    fn interact_toggles_door_back_closed() {
        let s = UnitStats::default();
        let mut w = door_world();
        assert_eq!(w.interact(0, &s), InteractOutcome::DoorOpened);
        assert_eq!(w.interact(0, &s), InteractOutcome::DoorClosed);
        assert_eq!(w.map.tile(5, 5, 0), Tile::DoorClosed);
    }

    #[test]
    fn door_does_not_close_on_a_unit() {
        let s = UnitStats::default();
        let mut w = door_world();
        w.interact(0, &s);
        w.units.push(Unit::new(1, 1, pos(5.5, 5.5)));
        assert_eq!(w.interact(0, &s), InteractOutcome::None);
        assert_eq!(w.map.tile(5, 5, 0), Tile::DoorOpen);
    }

    fn stairs_world() -> World {
        let mut map = GameMap::filled(6, 6, 2, Tile::Floor);
        map.set_tile(2, 2, 0, Tile::StairsUp);
        map.set_tile(2, 2, 1, Tile::StairsDown);
        let mut w = World::new(map);
        w.units.push(Unit::new(0, 0, pos(2.5, 2.5)));
        w
    }

    #[test]
    fn stairs_move_between_floors() {
        let s = UnitStats::default();
        let mut w = stairs_world();
        assert_eq!(w.interact(0, &s), InteractOutcome::ChangedFloor);
        assert_eq!(w.units[0].pos.floor, 1);
        assert!((w.units[0].pos.x - 2.5).abs() < 1e-6);
        assert_eq!(w.interact(0, &s), InteractOutcome::ChangedFloor);
        assert_eq!(w.units[0].pos.floor, 0);
    }

    #[test]
    fn stairs_blocked_when_destination_occupied() {
        let s = UnitStats::default();
        let mut w = stairs_world();
        w.units.push(Unit::new(
            1,
            1,
            Pos {
                x: 2.5,
                y: 2.5,
                floor: 1,
            },
        ));
        assert_eq!(w.interact(0, &s), InteractOutcome::None);
        assert_eq!(w.units[0].pos.floor, 0);
    }

    #[test]
    fn alive_count_counts_living_per_team() {
        let mut w = world_with(&[
            (0, pos(2.5, 2.5), 0.0),
            (0, pos(2.5, 4.5), 0.0),
            (1, pos(7.5, 2.5), 0.0),
        ]);
        w.units[1].alive = false;
        assert_eq!((w.alive_count(0), w.alive_count(1)), (1, 1));
    }
}
