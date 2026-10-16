//! OWNER: engine-agent. Unit and World state plus per-tick physics:
//! turning, movement with circle collisions, doors and stairs.
//!
//! Conventions:
//! - Positions are continuous, in tile units; tile `(tx, ty)` covers
//!   `[tx, tx+1) x [ty, ty+1)`. Floors are discrete.
//! - `yaw` is in radians, normalised to `[-PI, PI)`. Facing vector is
//!   `(cos yaw, sin yaw)`. "Left" is `+90deg` from facing: turning left
//!   increases yaw, strafing left moves along `(-sin yaw, cos yaw)`.
//! - Units are circles of radius `UnitStats::radius`. Dead units neither
//!   move, block movement nor can be hit.

use std::f32::consts::{PI, TAU};

use crate::map::{GameMap, Tile};
use crate::types::{Action, Pos, TeamId, UnitId};

/// Default unit hit points (also used by `Unit::new`).
pub const DEFAULT_MAX_HP: f32 = 100.0;
/// Default rounds per unit per episode (no reloads).
pub const DEFAULT_AMMO: u32 = 60;

/// Small epsilon used to keep units strictly off wall edges.
const EPS: f32 = 1e-4;

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
    /// Ticks a unit must wait between two successful `interact`s
    /// (prevents door/stairs flicker when interact is held).
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
    /// Ticks until `interact` can be used again.
    pub interact_cooldown: u32,
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
            interact_cooldown: 0,
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

    /// Furthest free position along `(dx, dy)` from unit i's position
    /// (bisection on contact so units end up flush against obstacles).
    fn slide_axis(&self, i: usize, dx: f32, dy: f32, radius: f32) -> Pos {
        const BISECT_STEPS: usize = 8;
        let p = self.units[i].pos;
        let at = |t: f32| Pos {
            x: p.x + dx * t,
            y: p.y + dy * t,
            floor: p.floor,
        };
        if dx == 0.0 && dy == 0.0 {
            return p;
        }
        if !self.blocked(i, at(1.0), radius) {
            return at(1.0);
        }
        let (mut lo, mut hi) = (0.0f32, 1.0f32);
        for _ in 0..BISECT_STEPS {
            let mid = 0.5 * (lo + hi);
            if self.blocked(i, at(mid), radius) {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        at(lo)
    }

    /// Would unit `i` standing at `p` collide with a wall or another living
    /// unit? Overlaps with units that the move does not make worse are
    /// allowed so units can never get permanently stuck together.
    fn blocked(&self, i: usize, p: Pos, radius: f32) -> bool {
        if circle_hits_blocking_tile(&self.map, p, radius) {
            return true;
        }
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

    /// Is any living unit's body overlapping tile `(tx, ty, f)`?
    fn tile_occupied(&self, tx: i32, ty: i32, f: u8, radius: f32) -> bool {
        self.units
            .iter()
            .any(|u| u.alive && u.pos.floor == f && circle_overlaps_tile(u.pos, radius, tx, ty))
    }

    /// `interact=1` for unit `i`: use stairs when standing on them, else
    /// toggle the door in the tile directly in front.
    pub(crate) fn interact(&mut self, i: usize, stats: &UnitStats) -> InteractOutcome {
        if self.units[i].interact_cooldown > 0 {
            return InteractOutcome::None;
        }
        let outcome = self.use_stairs(i, stats.radius);
        let outcome = if outcome == InteractOutcome::None {
            self.toggle_door_ahead(i, stats.radius)
        } else {
            outcome
        };
        if outcome != InteractOutcome::None {
            self.units[i].interact_cooldown = stats.interact_cooldown_ticks;
        }
        outcome
    }

    fn use_stairs(&mut self, i: usize, radius: f32) -> InteractOutcome {
        let p = self.units[i].pos;
        let Some(dest) = stairs_destination(&self.map, p) else {
            return InteractOutcome::None;
        };
        // Prefer the exact same (x, y); fall back to the stairs tile centre.
        let centre = Pos {
            x: p.x.floor() + 0.5,
            y: p.y.floor() + 0.5,
            floor: dest.floor,
        };
        let free = [dest, centre].into_iter().find(|&c| {
            !circle_hits_blocking_tile(&self.map, c, radius)
                && !self.units.iter().enumerate().any(|(j, o)| {
                    j != i
                        && o.alive
                        && o.pos.floor == c.floor
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
        let f = u.pos.floor;
        match self.map.tile(tx, ty, f as i32) {
            Tile::DoorClosed => {
                set_door(&mut self.map, tx, ty, f, true);
                InteractOutcome::DoorOpened
            }
            // Never close a door on top of somebody.
            Tile::DoorOpen if !self.tile_occupied(tx, ty, f, radius) => {
                set_door(&mut self.map, tx, ty, f, false);
                InteractOutcome::DoorClosed
            }
            _ => InteractOutcome::None,
        }
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

fn circle_overlaps_tile(p: Pos, radius: f32, tx: i32, ty: i32) -> bool {
    let cx = p.x.clamp(tx as f32, tx as f32 + 1.0);
    let cy = p.y.clamp(ty as f32, ty as f32 + 1.0);
    let (dx, dy) = (p.x - cx, p.y - cy);
    dx * dx + dy * dy < (radius + EPS) * (radius + EPS)
}

fn circle_hits_blocking_tile(map: &GameMap, p: Pos, radius: f32) -> bool {
    let f = p.floor as i32;
    let (x0, x1) = ((p.x - radius).floor() as i32, (p.x + radius).floor() as i32);
    let (y0, y1) = ((p.y - radius).floor() as i32, (p.y + radius).floor() as i32);
    (y0..=y1).any(|ty| {
        (x0..=x1)
            .any(|tx| blocks_movement(map, tx, ty, f) && circle_overlaps_tile(p, radius, tx, ty))
    })
}

// ---------------------------------------------------------------------------
// MAP-HOOKs: every map query used by movement/interaction goes through these.
// Swap in map-agent's richer API (`can_move`, `toggle_door`,
// `stairs_destination`) here when it lands.
// ---------------------------------------------------------------------------

// MAP-HOOK: tile-level movement blocking (windows, walls, closed doors, void).
fn blocks_movement(map: &GameMap, x: i32, y: i32, f: i32) -> bool {
    !map.is_walkable(x, y, f)
}

// MAP-HOOK: StairsUp at (x,y,f) pairs with StairsDown at (x,y,f+1).
fn stairs_destination(map: &GameMap, p: Pos) -> Option<Pos> {
    let (tx, ty, f) = (p.x.floor() as i32, p.y.floor() as i32, p.floor as i32);
    let target = match map.tile(tx, ty, f) {
        Tile::StairsUp => (f + 1, Tile::StairsDown),
        Tile::StairsDown => (f - 1, Tile::StairsUp),
        _ => return None,
    };
    let (nf, want) = target;
    if map.tile(tx, ty, nf) != want {
        return None;
    }
    Some(Pos {
        x: p.x,
        y: p.y,
        floor: nf as u8,
    })
}

// MAP-HOOK: open/close a door tile.
fn set_door(map: &mut GameMap, x: i32, y: i32, f: u8, open: bool) {
    let t = if open {
        Tile::DoorOpen
    } else {
        Tile::DoorClosed
    };
    map.set_tile(x as usize, y as usize, f as usize, t);
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
        assert_eq!((u.cooldown, u.interact_cooldown), (0, 0));
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
    fn interact_cooldown_and_door_close() {
        let s = UnitStats::default();
        let mut w = door_world();
        assert_eq!(w.interact(0, &s), InteractOutcome::DoorOpened);
        assert_eq!(w.units[0].interact_cooldown, s.interact_cooldown_ticks);
        // On cooldown: nothing happens.
        assert_eq!(w.interact(0, &s), InteractOutcome::None);
        w.units[0].interact_cooldown = 0;
        assert_eq!(w.interact(0, &s), InteractOutcome::DoorClosed);
        assert_eq!(w.map.tile(5, 5, 0), Tile::DoorClosed);
    }

    #[test]
    fn door_does_not_close_on_a_unit() {
        let s = UnitStats::default();
        let mut w = door_world();
        w.interact(0, &s);
        w.units[0].interact_cooldown = 0;
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
        w.units[0].interact_cooldown = 0;
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
