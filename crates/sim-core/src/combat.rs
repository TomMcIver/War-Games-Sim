//! OWNER: engine-agent. Hitscan combat.
//!
//! Rules (environment physics, no tactics):
//! - A unit fires when `shoot=1`, it is alive, has `ammo > 0` and
//!   `cooldown == 0`. Firing costs 1 ammo and sets `cooldown = cooldown_ticks`.
//! - The bullet is a ray along `yaw + u`, `u ~ Uniform[-s, s]` where
//!   `s = spread (+ moving_spread if the shooter moved this tick)`, drawn
//!   from the env's single ChaCha8 RNG in unit-index order.
//! - It hits the nearest living **enemy** whose body circle it crosses
//!   before the first bullet-blocking tile (`GameMap::raycast`: walls,
//!   closed doors, void; windows do not stop bullets), within `max_range`,
//!   and to which the shooter has exact `GameMap::line_of_sight`. **Friendly fire is OFF**: bullets pass through allies.
//! - Damage falls off linearly from `damage` at `falloff_start` to
//!   `damage * min_damage_frac` at `max_range`.
//! - All shots in a tick are simultaneous: hittability is decided by who
//!   was alive when the shooting phase began, so two units can kill each
//!   other in the same tick. Kill credit goes to the shot (in unit-index
//!   order) that takes the target to `hp <= 0`.

use rand::Rng;

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
///
/// Hittability and the right to fire come from `ev.was_alive` (alive when
/// the tick began; nothing kills before this phase). Damage is applied to
/// `hp` immediately, but `alive` only flips after every shot is resolved,
/// which keeps the phase simultaneous without scratch allocations.
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
    for i in 0..n {
        let u = &world.units[i];
        let wants = wants_shoot.get(i).copied().unwrap_or(false);
        let can_act = ev.was_alive.get(i).copied().unwrap_or(false);
        if !wants || !can_act || u.ammo == 0 || u.cooldown > 0 {
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

        // Exact bullet ray from the map; windows don't stop it.
        let wall = world.map.raycast(origin, angle, weapon.max_range);
        let hit = (0..n)
            .filter(|&j| {
                let o = &world.units[j];
                ev.was_alive[j] && o.team != u.team && o.pos.floor == origin.floor
            })
            .filter_map(|j| ray_circle(origin, dir, world.units[j].pos, radius).map(|t| (j, t)))
            .filter(|&(j, t)| {
                t < wall.distance && world.map.line_of_sight(origin, world.units[j].pos)
            })
            .min_by(|a, b| a.1.total_cmp(&b.1));

        let (x, y) = match hit {
            Some((_, t)) => (origin.x + dir.0 * t, origin.y + dir.1 * t),
            None => (wall.x, wall.y),
        };
        world.shots.push(ShotEvent {
            shooter: i,
            x,
            y,
            floor: origin.floor,
            hit: hit.map(|(j, _)| j),
        });
        let shooter = &mut world.units[i];
        shooter.ammo -= 1;
        shooter.cooldown = weapon.cooldown_ticks;

        match hit {
            Some((j, t)) => {
                let dmg = weapon.damage_at(t);
                let target = &mut world.units[j];
                let before = target.hp;
                target.hp = (before - dmg).max(0.0);
                let applied = dmg.min(before.max(0.0));
                ev.damage_dealt[i] += applied;
                ev.damage_taken[j] += applied;
                if before > 0.0 && before - dmg <= 0.0 {
                    ev.kills[i] += 1;
                }
            }
            None => ev.missed_shots[i] += 1,
        }
    }

    for u in world.units.iter_mut().filter(|u| u.alive && u.hp <= 0.0) {
        u.hp = 0.0;
        u.alive = false;
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

#[cfg(test)]
mod tests {
    use std::f32::consts::PI;

    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    use super::*;
    use crate::map::{GameMap, Tile};
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
    fn miss_tracer_ends_at_border_wall() {
        let mut w = world(GameMap::arena(10, 10), &[(0, 5.5, 5.5, 0.0)]);
        let ev = fire(&mut w, &[true], &exact());
        assert_eq!(ev.missed_shots[0], 1);
        assert!((w.shots[0].x - 9.0).abs() < 1e-4, "x={}", w.shots[0].x);
    }
}
