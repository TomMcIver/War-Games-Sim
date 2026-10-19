use std::f32::consts::PI;

use rand::Rng;

use super::*;
use crate::map::Tile;

fn exact_cfg() -> EnvConfig {
    EnvConfig {
        weapon: WeaponStats {
            spread: 0.0,
            moving_spread: 0.0,
            ..WeaponStats::default()
        },
        ..EnvConfig::default()
    }
}

/// 1v1 on an open 16x10 arena, facing each other along y = 5.5.
fn duel(cfg: EnvConfig, x0: f32, x1: f32) -> Env {
    let mut w = World::new(GameMap::arena(16, 10));
    w.units.push(Unit::new(
        0,
        0,
        Pos {
            x: x0,
            y: 5.5,
            floor: 0,
        },
    ));
    let mut b = Unit::new(
        1,
        1,
        Pos {
            x: x1,
            y: 5.5,
            floor: 0,
        },
    );
    b.yaw = PI - 1e-6;
    w.units.push(b);
    Env::from_world(cfg, w, 1)
}

fn shoot() -> Action {
    Action {
        shoot: 1,
        ..Action::default()
    }
}

#[test]
fn timeout_telemetry_counts_contact_and_resets() {
    let cfg = EnvConfig {
        max_ticks: 3,
        ..exact_cfg()
    };
    let mut env = duel(cfg, 2.5, 6.5);
    for _ in 0..2 {
        assert!(env.step(&[Action::default(); 2]).episode_stats.is_none());
    }
    let stats = env.step(&[Action::default(); 2]).episode_stats.unwrap();
    assert_eq!(stats.termination, TerminationReason::Timeout);
    assert_eq!(stats.contact_ticks, 3);
    assert_eq!(
        (stats.shots_fired, stats.shots_hit, stats.deaths),
        (0, 0, 0)
    );
    assert_eq!(stats.ending_ammo, 2 * crate::world::DEFAULT_AMMO);
    assert_eq!((stats.survivors_team0, stats.survivors_team1), (1, 1));
    env.reset(4);
    assert!(env.step(&[Action::default(); 2]).episode_stats.is_none());
}

#[test]
fn wipe_telemetry_counts_damage_and_ammo() {
    let mut env = duel(exact_cfg(), 2.5, 6.5);
    env.world.units[1].hp = 10.0;
    let stats = env
        .step(&[shoot(), Action::default()])
        .episode_stats
        .unwrap();
    assert_eq!(stats.termination, TerminationReason::TeamWipe);
    assert_eq!(
        (
            stats.shots_fired,
            stats.shots_hit,
            stats.kills,
            stats.deaths
        ),
        (1, 1, 1, 1)
    );
    assert_eq!(stats.damage_dealt, 10.0);
    assert_eq!(stats.ending_ammo, 2 * crate::world::DEFAULT_AMMO - 1);
    assert_eq!((stats.survivors_team0, stats.survivors_team1), (1, 0));
}

fn random_actions(rng: &mut impl Rng, n: usize, max: u8) -> Vec<Action> {
    (0..n)
        .map(|_| Action {
            forward: rng.gen_range(0..=max.max(2)),
            strafe: rng.gen_range(0..=max.max(2)),
            turn: rng.gen_range(0..=max.max(2)),
            shoot: rng.gen_range(0..=max.max(1)),
            interact: rng.gen_range(0..=max.max(1)),
        })
        .collect()
}

fn run_hashes(seed: u64, ticks: usize) -> Vec<u64> {
    let cfg = EnvConfig {
        team_size: 3,
        ..EnvConfig::default()
    };
    let mut env = Env::new(cfg, seed);
    let mut act_rng = ChaCha8Rng::seed_from_u64(99);
    (0..ticks)
        .map(|_| {
            let a = random_actions(&mut act_rng, env.n_units(), 0);
            env.step(&a);
            env.state_hash()
        })
        .collect()
}

#[test]
fn spawns_teams_in_index_order_facing_enemy() {
    let cfg = EnvConfig {
        team_size: 3,
        ..EnvConfig::default()
    };
    let env = Env::new(cfg.clone(), 5);
    let us = &env.world().units;
    assert_eq!(us.len(), 6);
    for (i, u) in us.iter().enumerate() {
        assert_eq!(u.id, i);
        assert_eq!(u.team, if i < 3 { 0 } else { 1 });
        assert!(u.alive);
        assert_eq!(
            (u.hp, u.ammo, u.cooldown),
            (cfg.unit.max_hp, cfg.weapon.ammo, 0)
        );
        let toward_enemy = if u.team == 0 {
            u.yaw.cos()
        } else {
            -u.yaw.cos()
        };
        assert!(
            toward_enemy > 0.9,
            "unit {i} yaw {} not facing enemy",
            u.yaw
        );
    }
    assert_eq!(env.world().tick, 0);
}

#[test]
#[should_panic(expected = "team_size")]
fn rejects_bad_team_size() {
    Env::new(
        EnvConfig {
            team_size: MAX_TEAM_SIZE + 1,
            ..EnvConfig::default()
        },
        0,
    );
}

#[test]
fn determinism_same_seed_and_actions() {
    assert_eq!(run_hashes(42, 300), run_hashes(42, 300));
}

#[test]
fn reset_reproduces_fresh_env() {
    let cfg = EnvConfig {
        team_size: 3,
        ..EnvConfig::default()
    };
    let mut env = Env::new(cfg.clone(), 7);
    let fresh_hash = env.state_hash();
    let mut act_rng = ChaCha8Rng::seed_from_u64(3);
    for _ in 0..50 {
        let a = random_actions(&mut act_rng, env.n_units(), 0);
        env.step(&a);
    }
    env.reset(42);
    assert_eq!(env.state_hash(), Env::new(cfg, 42).state_hash());
    env.reset(7);
    assert_eq!(env.state_hash(), fresh_hash);
    // Same actions after reset -> same trajectory as a fresh env.
    let tail = run_hashes(42, 20);
    let mut env = Env::new(
        EnvConfig {
            team_size: 3,
            ..EnvConfig::default()
        },
        1,
    );
    env.step(&[shoot(); 6]);
    env.reset(42);
    let mut act_rng = ChaCha8Rng::seed_from_u64(99);
    let replay: Vec<u64> = (0..20)
        .map(|_| {
            let a = random_actions(&mut act_rng, env.n_units(), 0);
            env.step(&a);
            env.state_hash()
        })
        .collect();
    assert_eq!(tail, replay);
}

#[test]
fn different_seeds_differ() {
    let a = run_hashes(1, 100);
    let b = run_hashes(2, 100);
    assert_ne!(a[0], b[0]);
    assert_ne!(a.last(), b.last());
}

#[test]
fn shooting_kills_target_in_los_in_expected_ticks() {
    let cfg = exact_cfg();
    let mut env = duel(cfg.clone(), 3.5, 8.5);
    let acts = [shoot(), Action::default()];
    // 4 hits of 25 needed, one every cooldown_ticks (3): ticks 1,4,7,10.
    let mut last = StepResult::default();
    for t in 1..=10 {
        assert!(!env.is_done(), "ended early at tick {t}");
        last = env.step(&acts);
    }
    assert!(!env.world().units[1].alive);
    assert_eq!(env.world().units[0].ammo, cfg.weapon.ammo - 4);
    assert!(last.done);
    assert_eq!(last.winner, Some(0));
    assert!(last.rewards[0] > 0.9, "winner reward {:?}", last.rewards);
    assert!(last.rewards[1] < -0.9, "loser reward {:?}", last.rewards);
}

#[test]
fn shots_recorded_and_cleared_each_tick() {
    let mut env = duel(exact_cfg(), 3.5, 8.5);
    env.step(&[shoot(), Action::default()]);
    assert_eq!(env.world().shots.len(), 1);
    assert_eq!(env.world().tick, 1);
    // Cooldown: next tick has no shot and the list is cleared.
    env.step(&[shoot(), Action::default()]);
    assert!(env.world().shots.is_empty());
}

#[test]
fn no_hit_through_wall_in_env() {
    let mut map = GameMap::arena(16, 10);
    for y in 1..9 {
        map.set_tile(6, y, 0, Tile::Wall);
    }
    let mut w = World::new(map);
    w.units.push(Unit::new(
        0,
        0,
        Pos {
            x: 3.5,
            y: 5.5,
            floor: 0,
        },
    ));
    w.units.push(Unit::new(
        1,
        1,
        Pos {
            x: 9.5,
            y: 5.5,
            floor: 0,
        },
    ));
    let mut env = Env::from_world(exact_cfg(), w, 0);
    for _ in 0..30 {
        env.step(&[shoot(), Action::default()]);
    }
    assert_eq!(env.world().units[1].hp, 100.0);
}

#[test]
fn dead_units_are_inert() {
    let mut w = World::new(GameMap::arena(16, 10));
    w.units.push(Unit::new(
        0,
        0,
        Pos {
            x: 3.5,
            y: 5.5,
            floor: 0,
        },
    ));
    w.units.push(Unit::new(
        1,
        1,
        Pos {
            x: 8.5,
            y: 5.5,
            floor: 0,
        },
    ));
    w.units.push(Unit::new(
        2,
        1,
        Pos {
            x: 12.5,
            y: 2.5,
            floor: 0,
        },
    ));
    w.units[1].alive = false;
    w.units[1].hp = 0.0;
    let mut env = Env::from_world(exact_cfg(), w, 0);
    let before = env.world().units[1].clone();
    let busy = Action {
        forward: 1,
        strafe: 1,
        turn: 1,
        shoot: 1,
        interact: 1,
    };
    for _ in 0..10 {
        env.step(&[Action::default(), busy, Action::default()]);
    }
    let after = &env.world().units[1];
    assert_eq!(
        (after.pos, after.yaw, after.ammo),
        (before.pos, before.yaw, before.ammo)
    );
    assert!(env.world().shots.iter().all(|s| s.shooter != 1));
}

#[test]
fn friendly_fire_is_off() {
    let mut w = World::new(GameMap::arena(16, 10));
    w.units.push(Unit::new(
        0,
        0,
        Pos {
            x: 3.5,
            y: 5.5,
            floor: 0,
        },
    ));
    w.units.push(Unit::new(
        1,
        0,
        Pos {
            x: 6.5,
            y: 5.5,
            floor: 0,
        },
    ));
    w.units.push(Unit::new(
        2,
        1,
        Pos {
            x: 12.5,
            y: 2.5,
            floor: 0,
        },
    ));
    let mut env = Env::from_world(exact_cfg(), w, 0);
    for _ in 0..20 {
        env.step(&[shoot(), Action::default(), Action::default()]);
    }
    assert_eq!(env.world().units[1].hp, 100.0);
}

#[test]
fn interact_toggles_door_in_front() {
    let mut map = GameMap::arena(16, 10);
    map.set_tile(5, 5, 0, Tile::DoorClosed);
    let mut w = World::new(map);
    w.units.push(Unit::new(
        0,
        0,
        Pos {
            x: 4.5,
            y: 5.5,
            floor: 0,
        },
    ));
    w.units.push(Unit::new(
        1,
        1,
        Pos {
            x: 12.5,
            y: 2.5,
            floor: 0,
        },
    ));
    let mut env = Env::from_world(EnvConfig::default(), w, 0);
    let h0 = env.state_hash();
    let use_ = Action {
        interact: 1,
        ..Action::default()
    };
    env.step(&[use_, Action::default()]);
    assert_eq!(env.world().map.tile(5, 5, 0), Tile::DoorOpen);
    assert_ne!(env.state_hash(), h0);
    // Held interact does not flicker the door while on cooldown.
    env.step(&[use_, Action::default()]);
    assert_eq!(env.world().map.tile(5, 5, 0), Tile::DoorOpen);
    for _ in 0..5 {
        env.step(&[use_, Action::default()]);
    }
    assert_eq!(env.world().map.tile(5, 5, 0), Tile::DoorClosed);
}

#[test]
fn both_wiped_same_tick_is_draw() {
    let cfg = EnvConfig {
        reward: RewardConfig {
            shaping_scale: 0.0,
            ..RewardConfig::default()
        },
        ..exact_cfg()
    };
    let mut env = duel(cfg, 3.5, 8.5);
    env.world_mut().units[0].hp = 10.0;
    env.world_mut().units[1].hp = 10.0;
    let r = env.step(&[shoot(), shoot()]);
    assert!(r.done);
    assert_eq!(r.winner, None);
    assert_eq!(r.rewards, vec![0.0, 0.0]);
}

#[test]
fn timeout_is_draw_at_max_ticks() {
    let cfg = EnvConfig {
        max_ticks: 5,
        ..EnvConfig::default()
    };
    let mut env = Env::new(cfg, 0);
    let idle = vec![Action::default(); env.n_units()];
    for _ in 0..4 {
        assert!(!env.step(&idle).done);
    }
    let r = env.step(&idle);
    assert!(r.done);
    assert_eq!(r.winner, None);
    assert_eq!(env.world().tick, 5);
}

#[test]
fn step_after_done_is_noop() {
    let cfg = EnvConfig {
        max_ticks: 1,
        ..EnvConfig::default()
    };
    let mut env = Env::new(cfg, 0);
    let busy = vec![
        Action {
            forward: 1,
            shoot: 1,
            ..Action::default()
        };
        env.n_units()
    ];
    assert!(env.step(&busy).done);
    let h = env.state_hash();
    let r = env.step(&busy);
    assert!(r.done);
    assert!(r.rewards.iter().all(|&x| x == 0.0));
    assert_eq!(env.state_hash(), h);
}

#[test]
fn shaping_scale_zero_gives_zero_reward_mid_episode() {
    let cfg = EnvConfig {
        reward: RewardConfig {
            shaping_scale: 0.0,
            ..RewardConfig::default()
        },
        ..exact_cfg()
    };
    let mut env = duel(cfg, 3.5, 8.5);
    let r = env.step(&[shoot(), shoot()]);
    assert!(!r.done);
    assert_eq!(r.rewards, vec![0.0, 0.0]);
    assert!(env.world().units[1].hp < 100.0, "damage did happen");
}

#[test]
fn shaping_rewards_damage_dealer() {
    let mut env = duel(exact_cfg(), 3.5, 8.5);
    let r = env.step(&[shoot(), Action::default()]);
    assert!(r.rewards[0] > 0.0 && r.rewards[1] < 0.0, "{:?}", r.rewards);
}

#[test]
fn invalid_actions_are_treated_as_zero() {
    let cfg = EnvConfig {
        team_size: 2,
        ..EnvConfig::default()
    };
    let mut a = Env::new(cfg.clone(), 11);
    let mut b = Env::new(cfg, 11);
    let bad = Action {
        forward: 3,
        strafe: 200,
        turn: 9,
        shoot: 2,
        interact: 255,
    };
    for _ in 0..20 {
        a.step(&[bad; 4]);
        b.step(&[Action::default(); 4]);
    }
    assert_eq!(a.state_hash(), b.state_hash());
    // Short action slices: missing units idle.
    a.step(&[]);
    b.step(&[Action::default(); 4]);
    assert_eq!(a.state_hash(), b.state_hash());
}

#[test]
fn stairs_via_step() {
    let mut map = GameMap::filled(8, 8, 2, Tile::Floor);
    map.set_tile(2, 2, 0, Tile::StairsUp);
    map.set_tile(2, 2, 1, Tile::StairsDown);
    let mut w = World::new(map);
    w.units.push(Unit::new(
        0,
        0,
        Pos {
            x: 2.5,
            y: 2.5,
            floor: 0,
        },
    ));
    w.units.push(Unit::new(
        1,
        1,
        Pos {
            x: 6.5,
            y: 6.5,
            floor: 0,
        },
    ));
    let mut env = Env::from_world(EnvConfig::default(), w, 0);
    env.step(&[
        Action {
            interact: 1,
            ..Action::default()
        },
        Action::default(),
    ]);
    assert_eq!(env.world().units[0].pos.floor, 1);
}

#[test]
fn fuzz_random_actions_never_panic_or_nan() {
    let cfg = EnvConfig {
        team_size: 4,
        max_ticks: 200,
        ..EnvConfig::default()
    };
    let mut env = Env::new(cfg, 123);
    let mut rng = ChaCha8Rng::seed_from_u64(456);
    let mut episodes = 0;
    for _ in 0..1000 {
        let a = random_actions(&mut rng, env.n_units(), 4);
        let r = env.step(&a);
        assert_eq!(r.rewards.len(), env.n_units());
        assert!(r.rewards.iter().all(|x| x.is_finite()), "{:?}", r.rewards);
        for u in &env.world().units {
            assert!(u.pos.x.is_finite() && u.pos.y.is_finite() && u.yaw.is_finite());
            assert!(u.hp.is_finite() && u.hp >= 0.0);
            assert!(env.world().map.is_walkable(
                u.pos.x.floor() as i32,
                u.pos.y.floor() as i32,
                u.pos.floor as i32
            ));
        }
        if r.done {
            episodes += 1;
            env.reset(rng.gen());
        }
    }
    assert!(episodes >= 1);
}

// ---------------------------------------------------------------------------
// Generated maps (arena / building / village)
// ---------------------------------------------------------------------------

const MAP_KINDS: [&str; 3] = ["arena", "building", "village"];

fn kind_cfg(kind: &str, team_size: usize) -> EnvConfig {
    EnvConfig {
        team_size,
        map_kind: kind.into(),
        ..EnvConfig::default()
    }
}

/// Every living unit's body fits the map (not inside a wall/window/door).
fn assert_units_fit(env: &Env, ctx: &str) {
    let r = env.cfg.unit.radius;
    for u in env.world().units.iter().filter(|u| u.alive) {
        assert!(
            env.world()
                .map
                .circle_fits(u.pos.x, u.pos.y, i32::from(u.pos.floor), r),
            "{ctx}: unit {} stuck in a blocked tile at {:?}",
            u.id,
            u.pos
        );
    }
}

/// Signed distance of unit 0 past the centre of door `(x, y)` along `(dx, dy)`.
fn past_door(env: &Env, (x, y, dx, dy): (i32, i32, i32, i32)) -> f32 {
    let p = env.world().units[0].pos;
    (p.x - (x as f32 + 0.5)) * dx as f32 + (p.y - (y as f32 + 0.5)) * dy as f32
}

#[test]
fn different_seeds_give_different_generated_maps() {
    for kind in ["building", "village"] {
        let a = Env::new(kind_cfg(kind, 2), 1);
        let b = Env::new(kind_cfg(kind, 2), 2);
        assert_ne!(a.world().map, b.world().map, "{kind}");
    }
}

#[test]
fn fuzz_every_map_kind_never_panics_nan_or_clips_walls() {
    for kind in MAP_KINDS {
        for seed in [3u64, 17, 2024] {
            let mut env = Env::new(kind_cfg(kind, 4), seed);
            let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0xA5A5);
            for t in 0..1000 {
                let a = random_actions(&mut rng, env.n_units(), 3);
                let r = env.step(&a);
                assert!(r.rewards.iter().all(|x| x.is_finite()), "{kind}/{seed}");
                for u in &env.world().units {
                    assert!(u.pos.x.is_finite() && u.pos.y.is_finite() && u.yaw.is_finite());
                    assert!(u.hp.is_finite() && u.hp >= 0.0);
                }
                assert_units_fit(&env, &format!("{kind}/{seed}/t{t}"));
                if r.done {
                    env.reset(rng.gen());
                    assert_units_fit(&env, &format!("{kind}/{seed}/reset"));
                }
            }
        }
    }
}

#[test]
fn every_team_size_spawns_validly_on_every_map_kind() {
    for kind in MAP_KINDS {
        for team_size in 1..=MAX_TEAM_SIZE {
            for seed in [0u64, 9] {
                let env = Env::new(kind_cfg(kind, team_size), seed);
                let ctx = format!("{kind}/n{team_size}/s{seed}");
                let us = &env.world().units;
                assert_eq!(us.len(), 2 * team_size, "{ctx}");
                assert_units_fit(&env, &ctx);
                let r = env.cfg.unit.radius;
                for (i, a) in us.iter().enumerate() {
                    assert_eq!((a.id, a.team), (i, u8::from(i >= team_size)), "{ctx}");
                    for b in &us[i + 1..] {
                        let d2 = (a.pos.x - b.pos.x).powi(2) + (a.pos.y - b.pos.y).powi(2);
                        let overlap = a.pos.floor == b.pos.floor && d2 < 4.0 * r * r;
                        assert!(!overlap, "{ctx}: units {} and {} overlap", a.id, b.id);
                    }
                }
                let mean_x = |t: u8| {
                    let team = us.iter().filter(|u| u.team == t);
                    team.clone().map(|u| u.pos.x).sum::<f32>() / team.count() as f32
                };
                assert!(mean_x(0) < mean_x(1), "{ctx}: team 0 should spawn low-x");
            }
        }
    }
}

#[test]
fn determinism_holds_per_map_kind() {
    let run = |env: &mut Env| -> Vec<u64> {
        let mut rng = ChaCha8Rng::seed_from_u64(5);
        (0..200)
            .map(|_| {
                let a = random_actions(&mut rng, env.n_units(), 0);
                env.step(&a);
                env.state_hash()
            })
            .collect()
    };
    for kind in MAP_KINDS {
        let mut a = Env::new(kind_cfg(kind, 3), 77);
        let mut b = Env::new(kind_cfg(kind, 3), 77);
        let ha = run(&mut a);
        assert_eq!(ha, run(&mut b), "{kind}: same seed must match");
        a.reset(77);
        assert_eq!(ha, run(&mut a), "{kind}: reset(seed) must replay");
    }
}

/// A door on floor 0 with plain floor on both sides along an axis:
/// `(door_x, door_y, dx, dy)`.
fn find_door(map: &GameMap) -> Option<(i32, i32, i32, i32)> {
    (0..map.height as i32)
        .flat_map(|y| (0..map.width as i32).map(move |x| (x, y)))
        .filter(|&(x, y)| map.tile(x, y, 0).is_door())
        .find_map(|(x, y)| {
            [(1, 0), (0, 1)].into_iter().find_map(|(dx, dy)| {
                let before = map.tile(x - dx, y - dy, 0);
                let after = map.tile(x + dx, y + dy, 0);
                (before == Tile::Floor && after == Tile::Floor).then_some((x, y, dx, dy))
            })
        })
}

/// Building env where only unit 0 (driven by the test) and one idle enemy
/// (unit 2, keeps the episode running) are alive.
fn solo_building(seed: u64) -> Env {
    let mut env = Env::new(kind_cfg("building", 2), seed);
    for i in [1, 3] {
        let u = &mut env.world_mut().units[i];
        u.alive = false;
        u.hp = 0.0;
    }
    env
}

#[test]
fn unit_traverses_a_door_in_generated_building() {
    let mut env = solo_building(4);
    let door = find_door(&env.world().map).expect("building has a door");
    let (x, y, dx, dy) = door;
    {
        let u = &mut env.world_mut().units[0];
        u.pos = Pos {
            x: (x - dx) as f32 + 0.5,
            y: (y - dy) as f32 + 0.5,
            floor: 0,
        };
        u.yaw = (dy as f32).atan2(dx as f32);
    }
    let idle = Action::default();
    let fwd = Action {
        forward: 1,
        ..Action::default()
    };
    let use_ = Action {
        interact: 1,
        ..Action::default()
    };
    if env.world().map.tile(x, y, 0) == Tile::DoorClosed {
        for _ in 0..5 {
            env.step(&[fwd, idle, idle, idle]);
        }
        let along = past_door(&env, door);
        assert!(along < -0.5, "closed door must block (along={along})");
        env.step(&[use_, idle, idle, idle]);
    }
    assert_eq!(env.world().map.tile(x, y, 0), Tile::DoorOpen);
    for _ in 0..12 {
        env.step(&[fwd, idle, idle, idle]);
    }
    let along = past_door(&env, door);
    assert!(along > 0.5, "unit should be past the door (along={along})");
    assert_units_fit(&env, "after door");
}

#[test]
fn unit_uses_stairs_in_generated_building() {
    let mut env = solo_building(4);
    let map = &env.world().map;
    let (sx, sy) = (0..map.height as i32)
        .flat_map(|y| (0..map.width as i32).map(move |x| (x, y)))
        .find(|&(x, y)| map.stairs_destination(x, y, 0).is_some())
        .expect("building has stairs");
    env.world_mut().units[0].pos = Pos {
        x: sx as f32 + 0.5,
        y: sy as f32 + 0.5,
        floor: 0,
    };
    let idle = Action::default();
    let use_ = Action {
        interact: 1,
        ..Action::default()
    };
    env.step(&[use_, idle, idle, idle]);
    assert_eq!(env.world().units[0].pos.floor, 1, "went upstairs");
    assert_units_fit(&env, "upstairs");
    for _ in 0..env.cfg.unit.interact_cooldown_ticks {
        env.step(&[use_, idle, idle, idle]);
    }
    assert_eq!(env.world().units[0].pos.floor, 0, "came back down");
}

#[test]
#[should_panic(expected = "unit.radius")]
fn rejects_radius_too_big_for_doorways() {
    let cfg = EnvConfig {
        unit: UnitStats {
            radius: 0.5,
            ..UnitStats::default()
        },
        ..EnvConfig::default()
    };
    Env::new(cfg, 0);
}
