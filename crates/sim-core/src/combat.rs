//! OWNER: engine-agent. Hitscan combat.
//!
//! Rules (environment physics, no tactics):
//! - A unit fires when `shoot=1`, it is alive, has `ammo > 0` and
//!   `cooldown == 0`. Firing costs 1 ammo and sets `cooldown = cooldown_ticks`.
//! - The bullet is a ray along `yaw + u`, `u ~ Uniform[-s, s]` where
//!   `s = spread (+ moving_spread if the shooter moved this tick)`, drawn
//!   from the env's single ChaCha8 RNG in unit-index order.
//! - It hits the nearest living **enemy** whose body circle it crosses
//!   before any sight-blocking tile (walls, closed doors, void; windows do
//!   not stop bullets), within `max_range`, and to which the shooter has
//!   line of sight. **Friendly fire is OFF**: bullets pass through allies.
//! - Damage falls off linearly from `damage` at `falloff_start` to
//!   `damage * min_damage_frac` at `max_range`.
//! - All shots in a tick are simultaneous: hittability is decided by who
//!   was alive when the shooting phase began, so two units can kill each
//!   other in the same tick. Kill credit goes to the shot (in unit-index
//!   order) that takes the target to `hp <= 0`.

use rand::Rng;

use crate::map::GameMap;
use crate::reward::TickEvents;
use crate::types::Pos;
use crate::world::{ShotEvent, World, DEFAULT_AMMO};

#[derive(Clone, Debug, PartialEq)]
pub struct WeaponStats {
    /// Damage of one bullet up to `falloff_start`.
    pub damage: f32,
    /// Bullets travel at most this far (tiles).
    pub max_range: f32,
    /// Distance where damage falloff begins (tiles).
    pub falloff_start: f32,
    /// Fraction of `damage` dealt at `max_range`.
    pub min_damage_frac: f32,
    /// Half-angle (radians) of uniform aim error when standing still.
    pub spread: f32,
    /// Extra half-angle added when the shooter moved this tick.
    pub moving_spread: f32,
    /// Ticks between shots.
    pub cooldown_ticks: u32,
    /// Starting rounds (no reloads).
    pub ammo: u32,
}

impl Default for WeaponStats {
    fn default() -> Self {
        Self {
            damage: 25.0,
            max_range: 30.0,
            falloff_start: 8.0,
            min_damage_frac: 0.5,
            spread: 0.02,
            moving_spread: 0.04,
            cooldown_ticks: 3,
            ammo: DEFAULT_AMMO,
        }
    }
}

impl WeaponStats {
    /// Damage of a hit at distance `d`.
    pub fn damage_at(&self, d: f32) -> f32 {
        let span = self.max_range - self.falloff_start;
        if d <= self.falloff_start || span <= 0.0 {
            return self.damage;
        }
        let frac = ((d - self.falloff_start) / span).clamp(0.0, 1.0);
        self.damage * (1.0 - frac * (1.0 - self.min_damage_frac))
    }
}

/// Resolve this tick's shooting. `wants_shoot[i]` is unit i's shoot head,
/// `moved[i]` whether it moved this tick. `radius` is the unit body radius.
/// Appends to `world.shots`, applies damage/deaths and records events.
pub(crate) fn resolve_shots<R: Rng>(
    world: &mut World,
    wants_shoot: &[bool],
    moved: &[bool],
    weapon: &WeaponStats,
    radius: f32,
    rng: &mut R,
    ev: &mut TickEvents,
) {
    let n = world.units.len();
    let targetable: Vec<bool> = world.units.iter().map(|u| u.alive).collect();
    let mut pending_hp: Vec<f32> = world.units.iter().map(|u| u.hp).collect();

    for i in 0..n {
        let u = &world.units[i];
        let wants = wants_shoot.get(i).copied().unwrap_or(false);
        if !wants || !targetable[i] || u.ammo == 0 || u.cooldown > 0 {
            continue;
        }
        let is_moving = moved.get(i).copied().unwrap_or(false);
        let half = weapon.spread + if is_moving { weapon.moving_spread } else { 0.0 };
        let offset = if half.is_finite() && half > 0.0 {
            rng.gen_range(-half..=half)
        } else {
            0.0
        };
        let angle = u.yaw + offset;
        let dir = (angle.cos(), angle.sin());
        let origin = u.pos;

        let wall_t = wall_distance(&world.map, origin, dir, weapon.max_range);
        let hit = (0..n)
            .filter(|&j| {
                let o = &world.units[j];
                targetable[j] && o.team != u.team && o.pos.floor == origin.floor
            })
            .filter_map(|j| ray_circle(origin, dir, world.units[j].pos, radius).map(|t| (j, t)))
            .filter(|&(j, t)| t < wall_t && has_los(&world.map, origin, world.units[j].pos))
            .min_by(|a, b| a.1.total_cmp(&b.1));

        let end_t = hit.map_or(wall_t, |(_, t)| t);
        world.shots.push(ShotEvent {
            shooter: i,
            x: origin.x + dir.0 * end_t,
            y: origin.y + dir.1 * end_t,
            floor: origin.floor,
            hit: hit.map(|(j, _)| j),
        });
        let shooter = &mut world.units[i];
        shooter.ammo -= 1;
        shooter.cooldown = weapon.cooldown_ticks;

        match hit {
            Some((j, t)) => {
                let dmg = weapon.damage_at(t);
                let before = pending_hp[j];
                let applied = dmg.min(before.max(0.0));
                pending_hp[j] = before - dmg;
                ev.damage_dealt[i] += applied;
                ev.damage_taken[j] += applied;
                if before > 0.0 && pending_hp[j] <= 0.0 {
                    ev.kills[i] += 1;
                }
            }
            None => ev.missed_shots[i] += 1,
        }
    }

    for (u, &hp) in world.units.iter_mut().zip(&pending_hp) {
        if u.alive {
            u.hp = hp.max(0.0);
            if hp <= 0.0 {
                u.alive = false;
            }
        }
    }
}

/// Distance along the ray `origin + t*dir` to the first point inside the
/// circle `(c, r)`, if the ray crosses it (t >= 0).
fn ray_circle(origin: Pos, dir: (f32, f32), c: Pos, r: f32) -> Option<f32> {
    let (rx, ry) = (c.x - origin.x, c.y - origin.y);
    let proj = rx * dir.0 + ry * dir.1;
    let d2 = rx * rx + ry * ry;
    if d2 <= r * r {
        return Some(0.0); // muzzle inside the target's body
    }
    if proj < 0.0 {
        return None;
    }
    let perp2 = d2 - proj * proj;
    if perp2 > r * r {
        return None;
    }
    Some((proj - (r * r - perp2).max(0.0).sqrt()).max(0.0))
}

// ---------------------------------------------------------------------------
// MAP-HOOKs: every map query used by combat goes through these.
// When map-agent's `raycast(from, angle, max_range)` / exact LOS land, swap
// them in here.
// ---------------------------------------------------------------------------

// MAP-HOOK: does this tile stop bullets? (Windows don't.)
fn blocks_bullets(map: &GameMap, x: i32, y: i32, f: i32) -> bool {
    map.blocks_sight(x, y, f)
}

// MAP-HOOK: shooter-to-target line of sight.
fn has_los(map: &GameMap, a: Pos, b: Pos) -> bool {
    map.line_of_sight(a, b)
}

// MAP-HOOK: distance from `origin` along unit vector `dir` to the first
// bullet-blocking tile, capped at `max_range` (exact grid DDA).
fn wall_distance(map: &GameMap, origin: Pos, dir: (f32, f32), max_range: f32) -> f32 {
    let f = origin.floor as i32;
    let (mut cx, mut cy) = (origin.x.floor() as i32, origin.y.floor() as i32);
    if blocks_bullets(map, cx, cy, f) {
        return 0.0;
    }
    let axis = |p: f32, c: i32, d: f32| -> (i32, f32, f32) {
        if d > 0.0 {
            (1, (c as f32 + 1.0 - p) / d, 1.0 / d)
        } else if d < 0.0 {
            (-1, (p - c as f32) / -d, -1.0 / d)
        } else {
            (0, f32::INFINITY, f32::INFINITY)
        }
    };
    let (sx, mut tmx, tdx) = axis(origin.x, cx, dir.0);
    let (sy, mut tmy, tdy) = axis(origin.y, cy, dir.1);
    loop {
        let t = if tmx < tmy {
            cx += sx;
            let t = tmx;
            tmx += tdx;
            t
        } else {
            cy += sy;
            let t = tmy;
            tmy += tdy;
            t
        };
        if !t.is_finite() || t >= max_range {
            return max_range;
        }
        if blocks_bullets(map, cx, cy, f) {
            return t;
        }
    }
}

#[cfg(test)]
mod tests {
    use std::f32::consts::PI;

    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    use super::*;
    use crate::map::Tile;
    use crate::types::TeamId;
    use crate::world::Unit;

    const R: f32 = 0.3;

    fn exact() -> WeaponStats {
        WeaponStats {
            spread: 0.0,
            moving_spread: 0.0,
            ..WeaponStats::default()
        }
    }

    fn world(map: GameMap, units: &[(TeamId, f32, f32, f32)]) -> World {
        let mut w = World::new(map);
        for (i, &(team, x, y, yaw)) in units.iter().enumerate() {
            let mut u = Unit::new(i, team, Pos { x, y, floor: 0 });
            u.yaw = yaw;
            w.units.push(u);
        }
        w
    }

    fn fire(w: &mut World, shooters: &[bool], weapon: &WeaponStats) -> TickEvents {
        let mut ev = TickEvents::new(w.units.iter().map(|u| u.alive).collect());
        let moved = vec![false; w.units.len()];
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        resolve_shots(w, shooters, &moved, weapon, R, &mut rng, &mut ev);
        ev
    }

    #[test]
    fn close_hit_deals_full_damage_and_costs_ammo() {
        let mut w = world(
            GameMap::arena(12, 12),
            &[(0, 2.5, 5.5, 0.0), (1, 6.5, 5.5, PI)],
        );
        let ev = fire(&mut w, &[true, false], &exact());
        assert_eq!(w.units[1].hp, 75.0);
        assert_eq!(ev.damage_dealt[0], 25.0);
        assert_eq!(ev.damage_taken[1], 25.0);
        assert_eq!(w.units[0].ammo, crate::world::DEFAULT_AMMO - 1);
        assert_eq!(w.units[0].cooldown, 3);
        assert_eq!(w.shots.len(), 1);
        assert_eq!(w.shots[0].hit, Some(1));
        assert!(
            (w.shots[0].x - (6.5 - R)).abs() < 1e-4,
            "impact at body edge"
        );
    }

    #[test]
    fn falloff_reduces_damage_with_distance() {
        let wpn = WeaponStats::default();
        assert_eq!(wpn.damage_at(1.0), 25.0);
        assert!((wpn.damage_at(30.0) - 12.5).abs() < 1e-5);
        let mid = wpn.damage_at(19.0);
        assert!(mid < 25.0 && mid > 12.5);
    }

    #[test]
    fn no_hit_through_wall_but_through_window() {
        let mut map = GameMap::arena(12, 12);
        map.set_tile(5, 5, 0, Tile::Wall);
        let mut w = world(map.clone(), &[(0, 2.5, 5.5, 0.0), (1, 8.5, 5.5, PI)]);
        let ev = fire(&mut w, &[true, false], &exact());
        assert_eq!(w.units[1].hp, 100.0);
        assert_eq!(ev.missed_shots[0], 1);
        assert_eq!(w.shots[0].hit, None);
        assert!((w.shots[0].x - 5.0).abs() < 1e-4, "stops at wall face");

        map.set_tile(5, 5, 0, Tile::Window);
        let mut w = world(map, &[(0, 2.5, 5.5, 0.0), (1, 8.5, 5.5, PI)]);
        fire(&mut w, &[true, false], &exact());
        assert!(w.units[1].hp < 100.0, "windows don't stop bullets");
    }

    #[test]
    fn closed_door_blocks_bullets() {
        let mut map = GameMap::arena(12, 12);
        map.set_tile(5, 5, 0, Tile::DoorClosed);
        let mut w = world(map, &[(0, 2.5, 5.5, 0.0), (1, 8.5, 5.5, PI)]);
        fire(&mut w, &[true, false], &exact());
        assert_eq!(w.units[1].hp, 100.0);
    }

    #[test]
    fn out_of_range_misses() {
        let wpn = WeaponStats {
            max_range: 3.0,
            ..exact()
        };
        let mut w = world(
            GameMap::arena(12, 12),
            &[(0, 2.5, 5.5, 0.0), (1, 8.5, 5.5, PI)],
        );
        let ev = fire(&mut w, &[true, false], &wpn);
        assert_eq!(w.units[1].hp, 100.0);
        assert_eq!(ev.missed_shots[0], 1);
        assert!(
            (w.shots[0].x - 5.5).abs() < 1e-4,
            "tracer ends at max range"
        );
    }

    #[test]
    fn friendly_fire_off_bullets_pass_through_allies() {
        let mut w = world(
            GameMap::arena(12, 12),
            &[(0, 2.5, 5.5, 0.0), (0, 4.5, 5.5, 0.0), (1, 7.5, 5.5, PI)],
        );
        fire(&mut w, &[true, false, false], &exact());
        assert_eq!(w.units[1].hp, 100.0, "ally untouched");
        assert_eq!(w.units[2].hp, 75.0, "enemy behind ally is hit");
    }

    #[test]
    fn needs_ammo_and_cooldown() {
        let mut w = world(
            GameMap::arena(12, 12),
            &[(0, 2.5, 5.5, 0.0), (1, 6.5, 5.5, PI)],
        );
        w.units[0].ammo = 0;
        fire(&mut w, &[true, false], &exact());
        assert!(w.shots.is_empty());
        w.units[0].ammo = 5;
        w.units[0].cooldown = 1;
        fire(&mut w, &[true, false], &exact());
        assert!(w.shots.is_empty());
        assert_eq!(w.units[1].hp, 100.0);
        assert_eq!(w.units[0].ammo, 5);
    }

    #[test]
    fn dead_units_neither_shoot_nor_get_hit() {
        let mut w = world(
            GameMap::arena(12, 12),
            &[(0, 2.5, 5.5, 0.0), (1, 5.5, 5.5, PI), (1, 8.5, 5.5, PI)],
        );
        w.units[1].alive = false;
        w.units[1].hp = 0.0;
        fire(&mut w, &[true, true, false], &exact());
        assert_eq!(w.shots.len(), 1, "dead unit 1 must not fire");
        assert_eq!(w.shots[0].hit, Some(2), "corpse is not hittable");
    }

    #[test]
    fn simultaneous_shots_can_kill_each_other_with_kill_credit() {
        let mut w = world(
            GameMap::arena(12, 12),
            &[(0, 2.5, 5.5, 0.0), (1, 6.5, 5.5, PI)],
        );
        w.units[0].hp = 10.0;
        w.units[1].hp = 10.0;
        let ev = fire(&mut w, &[true, true], &exact());
        assert!(!w.units[0].alive && !w.units[1].alive);
        assert_eq!((w.units[0].hp, w.units[1].hp), (0.0, 0.0));
        assert_eq!(ev.kills, vec![1, 1]);
        assert_eq!(
            ev.damage_dealt,
            vec![10.0, 10.0],
            "damage credit clipped to remaining hp"
        );
    }

    #[test]
    fn different_floor_not_hittable() {
        let map = GameMap::filled(12, 12, 2, Tile::Floor);
        let mut w = world(map, &[(0, 2.5, 5.5, 0.0), (1, 6.5, 5.5, PI)]);
        w.units[1].pos.floor = 1;
        fire(&mut w, &[true, false], &exact());
        assert_eq!(w.units[1].hp, 100.0);
    }

    #[test]
    fn spread_is_seeded_and_bounded() {
        let wpn = WeaponStats {
            spread: 0.3,
            ..WeaponStats::default()
        };
        let run = |seed: u64| {
            let mut w = world(GameMap::arena(30, 30), &[(0, 2.5, 15.5, 0.0)]);
            let mut ev = TickEvents::new(vec![true]);
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            resolve_shots(&mut w, &[true], &[false], &wpn, R, &mut rng, &mut ev);
            (w.shots[0].x, w.shots[0].y)
        };
        assert_eq!(run(1), run(1));
        assert_ne!(run(1), run(2));
        let (x, y) = run(3);
        let ang = (y - 15.5).atan2(x - 2.5);
        assert!(ang.abs() <= 0.3 + 1e-4);
    }

    #[test]
    fn wall_distance_diagonal_and_axis() {
        let map = GameMap::arena(10, 10);
        let p = Pos {
            x: 5.5,
            y: 5.5,
            floor: 0,
        };
        assert!((wall_distance(&map, p, (1.0, 0.0), 50.0) - 3.5).abs() < 1e-4);
        assert!((wall_distance(&map, p, (0.0, -1.0), 50.0) - 4.5).abs() < 1e-4);
        let d = std::f32::consts::FRAC_1_SQRT_2;
        let t = wall_distance(&map, p, (d, d), 50.0);
        assert!((t - 3.5 * std::f32::consts::SQRT_2).abs() < 1e-3, "t={t}");
        assert_eq!(wall_distance(&map, p, (1.0, 0.0), 2.0), 2.0);
    }
}
