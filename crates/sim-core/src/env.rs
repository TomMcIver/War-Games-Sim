//! OWNER: engine-agent. One match: reset / step with raw per-unit actions.
//! Units are indexed team 0 first (0..n), then team 1 (n..2n).
//!
//! Tick order inside `step` (all deterministic, unit-index order):
//! 1. clear `World.shots`; sanitise actions (invalid head values -> 0,
//!    missing actions -> all zeros); dead units' actions are ignored;
//! 2. weapon and interact cooldowns tick down by one;
//! 3. turn, then move (units move sequentially by index, sliding on walls
//!    and other units);
//! 4. interact (stairs if standing on them, else toggle the door in front);
//! 5. shooting, resolved simultaneously (see `combat`);
//! 6. `tick += 1`, termination check, rewards.
//!
//! Termination: a team with no living units loses (both wiped on the same
//! tick = draw). Reaching `max_ticks` is a **draw** (`winner = None`); there
//! is deliberately no hp tie-break so the policy cannot farm a timeout win.
//! Calling `step` after the episode ended is a no-op returning `done = true`
//! and zero rewards until `reset`.

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::combat::{self, WeaponStats};
use crate::map::GameMap;
use crate::reward::{compute_rewards, Outcome, RewardConfig, TickEvents};
use crate::types::{Action, Pos, TeamId, MAX_TEAM_SIZE};
use crate::world::{InteractOutcome, Unit, UnitStats, World};

/// Arena size used while the map generator is a stub.
const ARENA_WIDTH: usize = 24;
const ARENA_HEIGHT: usize = 16;

#[derive(Clone, Debug)]
pub struct EnvConfig {
    pub team_size: usize,
    pub max_ticks: u32,
    pub tick_seconds: f32,
    /// "arena" | "building" | "village" (map-agent defines kinds).
    pub map_kind: String,
    /// Body / movement physics.
    pub unit: UnitStats,
    /// Weapon physics.
    pub weapon: WeaponStats,
    /// Reward weights (terminal + annealable shaping).
    pub reward: RewardConfig,
}

impl Default for EnvConfig {
    fn default() -> Self {
        Self {
            team_size: 2,
            max_ticks: 600,
            tick_seconds: 0.1,
            map_kind: "arena".into(),
            unit: UnitStats::default(),
            weapon: WeaponStats::default(),
            reward: RewardConfig::default(),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct StepResult {
    /// One reward per unit (len = 2 * team_size).
    pub rewards: Vec<f32>,
    pub done: bool,
    pub winner: Option<TeamId>,
}

pub struct Env {
    pub cfg: EnvConfig,
    pub world: World,
    rng: ChaCha8Rng,
    done: bool,
    winner: Option<TeamId>,
    /// Per-unit ticks until `interact` works again (see
    /// `UnitStats::interact_cooldown_ticks`).
    interact_cooldown: Vec<u32>,
}

impl Env {
    /// Build the map for `cfg.map_kind` and spawn both teams.
    ///
    /// Panics if `team_size` is not in `1..=MAX_TEAM_SIZE` or `tick_seconds`
    /// is not a positive finite number.
    pub fn new(cfg: EnvConfig, seed: u64) -> Self {
        validate(&cfg);
        let world = World::new(build_map(&cfg, seed));
        let mut env = Self::from_world(cfg, world, seed);
        env.reset(seed);
        env
    }

    /// Wrap a hand-built world (tests / scenarios). Units are used as given;
    /// a later `reset` rebuilds the configured map and respawns.
    pub fn from_world(cfg: EnvConfig, world: World, seed: u64) -> Self {
        validate(&cfg);
        let n = world.units.len();
        Self {
            cfg,
            world,
            rng: ChaCha8Rng::seed_from_u64(seed),
            done: false,
            winner: None,
            interact_cooldown: vec![0; n],
        }
    }

    /// Start a new episode: rebuild the map, respawn, reseed the RNG.
    pub fn reset(&mut self, seed: u64) {
        let map = build_map(&self.cfg, seed);
        let units = spawn_units(&map, &self.cfg);
        self.world = World {
            map,
            units,
            tick: 0,
            shots: Vec::new(),
        };
        self.rng = ChaCha8Rng::seed_from_u64(seed);
        self.done = false;
        self.winner = None;
        self.interact_cooldown = vec![0; self.world.units.len()];
    }

    pub fn world(&self) -> &World {
        &self.world
    }

    pub fn world_mut(&mut self) -> &mut World {
        &mut self.world
    }

    pub fn n_units(&self) -> usize {
        self.world.units.len()
    }

    pub fn is_done(&self) -> bool {
        self.done
    }

    /// Winner of the finished episode (`None` = draw or still running).
    pub fn winner(&self) -> Option<TeamId> {
        self.winner
    }

    /// `actions.len() == 2 * team_size`. Dead units' actions are ignored.
    pub fn step(&mut self, actions: &[Action]) -> StepResult {
        let n = self.world.units.len();
        if self.done {
            return StepResult {
                rewards: vec![0.0; n],
                done: true,
                winner: self.winner,
            };
        }
        self.world.shots.clear();
        let acts: Vec<Action> = (0..n)
            .map(|i| actions.get(i).copied().map(sanitize).unwrap_or_default())
            .collect();
        let was_alive: Vec<bool> = self.world.units.iter().map(|u| u.alive).collect();
        let mut events = TickEvents::new(was_alive.clone());

        // Units may have been added through `world_mut()`.
        self.interact_cooldown.resize(n, 0);
        for u in self.world.units.iter_mut().filter(|u| u.alive) {
            u.cooldown = u.cooldown.saturating_sub(1);
        }
        for cd in &mut self.interact_cooldown {
            *cd = cd.saturating_sub(1);
        }

        let dt = self.cfg.tick_seconds;
        let mut moved = vec![false; n];
        for i in (0..n).filter(|&i| was_alive[i]) {
            self.world.turn_unit(i, acts[i].turn, &self.cfg.unit, dt);
            moved[i] = self.world.move_unit(i, &acts[i], &self.cfg.unit, dt);
        }
        for i in (0..n).filter(|&i| was_alive[i] && acts[i].interact == 1) {
            if self.interact_cooldown[i] == 0
                && self.world.interact(i, &self.cfg.unit) != InteractOutcome::None
            {
                self.interact_cooldown[i] = self.cfg.unit.interact_cooldown_ticks;
            }
        }

        let wants_shoot: Vec<bool> = (0..n).map(|i| was_alive[i] && acts[i].shoot == 1).collect();
        combat::resolve_shots(
            &mut self.world,
            &wants_shoot,
            &moved,
            &self.cfg.weapon,
            self.cfg.unit.radius,
            &mut self.rng,
            &mut events,
        );

        self.world.tick += 1;
        let outcome = self.check_outcome();
        if let Some(o) = outcome {
            self.done = true;
            self.winner = o.winner();
        }
        let teams: Vec<TeamId> = self.world.units.iter().map(|u| u.team).collect();
        let rewards = compute_rewards(
            &self.cfg.reward,
            self.cfg.unit.max_hp,
            &teams,
            &events,
            outcome,
        );
        StepResult {
            rewards,
            done: self.done,
            winner: self.winner,
        }
    }

    fn check_outcome(&self) -> Option<Outcome> {
        let (a0, a1) = (self.world.alive_count(0), self.world.alive_count(1));
        match (a0, a1) {
            (0, 0) => Some(Outcome::Draw),
            (0, _) => Some(Outcome::Win(1)),
            (_, 0) => Some(Outcome::Win(0)),
            _ if self.world.tick >= u64::from(self.cfg.max_ticks) => Some(Outcome::Draw),
            _ => None,
        }
    }

    /// FNV-1a digest of the full state: tick, episode status, RNG state,
    /// every unit field, all map tiles (door states) and this tick's shots.
    pub fn state_hash(&self) -> u64 {
        let mut h = Fnv1a::new();
        let w = &self.world;
        h.u64(w.tick);
        h.u8(u8::from(self.done));
        h.u8(self.winner.unwrap_or(u8::MAX));
        h.bytes(&self.rng.get_seed());
        h.u64(self.rng.get_stream());
        h.bytes(&self.rng.get_word_pos().to_le_bytes());
        h.u64(w.units.len() as u64);
        for u in &w.units {
            h.u64(u.id as u64);
            h.u8(u.team);
            h.pos(u.pos);
            h.f32(u.yaw);
            h.f32(u.hp);
            h.u32(u.ammo);
            h.u8(u8::from(u.alive));
            h.u32(u.cooldown);
        }
        h.u64(self.interact_cooldown.len() as u64);
        for &cd in &self.interact_cooldown {
            h.u32(cd);
        }
        hash_map_tiles(&mut h, &w.map);
        h.u64(w.shots.len() as u64);
        for s in &w.shots {
            h.u64(s.shooter as u64);
            h.f32(s.x);
            h.f32(s.y);
            h.u8(s.floor);
            h.u64(s.hit.map_or(u64::MAX, |j| j as u64));
        }
        h.finish()
    }
}

fn validate(cfg: &EnvConfig) {
    assert!(
        (1..=MAX_TEAM_SIZE).contains(&cfg.team_size),
        "team_size must be in 1..={MAX_TEAM_SIZE}, got {}",
        cfg.team_size
    );
    assert!(
        cfg.tick_seconds.is_finite() && cfg.tick_seconds > 0.0,
        "tick_seconds must be positive and finite, got {}",
        cfg.tick_seconds
    );
}

/// Head values outside `ACTION_DIMS` become 0.
fn sanitize(a: Action) -> Action {
    let fix = |v: u8, dim: u8| if v < dim { v } else { 0 };
    Action {
        forward: fix(a.forward, 3),
        strafe: fix(a.strafe, 3),
        turn: fix(a.turn, 3),
        shoot: fix(a.shoot, 2),
        interact: fix(a.interact, 2),
    }
}

/// Team 0 then team 1, each facing the centroid of the enemy spawns.
fn spawn_units(map: &GameMap, cfg: &EnvConfig) -> Vec<Unit> {
    let n = cfg.team_size;
    let centre = Pos {
        x: map.width as f32 * 0.5,
        y: map.height as f32 * 0.5,
        floor: 0,
    };
    let points: [Vec<Pos>; 2] = [0, 1].map(|t| {
        let pts = spawn_points(map, t, n);
        (0..n)
            .map(|i| pts.get(i).or(pts.last()).copied().unwrap_or(centre))
            .collect()
    });
    let centroid = |ps: &[Pos]| {
        let k = ps.len().max(1) as f32;
        (
            ps.iter().map(|p| p.x).sum::<f32>() / k,
            ps.iter().map(|p| p.y).sum::<f32>() / k,
        )
    };
    let mut units = Vec::with_capacity(2 * n);
    for team in 0..2u8 {
        let (ex, ey) = centroid(&points[1 - team as usize]);
        for &p in &points[team as usize] {
            let mut u = Unit::new(units.len(), team, p);
            u.yaw = crate::world::wrap_angle((ey - p.y).atan2(ex - p.x));
            u.hp = cfg.unit.max_hp;
            u.ammo = cfg.weapon.ammo;
            units.push(u);
        }
    }
    units
}

fn hash_map_tiles(h: &mut Fnv1a, map: &GameMap) {
    h.u64(map.width as u64);
    h.u64(map.height as u64);
    h.u64(map.floors as u64);
    for f in 0..map.floors as i32 {
        for y in 0..map.height as i32 {
            for x in 0..map.width as i32 {
                h.u8(map.tile(x, y, f) as u8);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// MAP-HOOKs
// ---------------------------------------------------------------------------

// MAP-HOOK: the single place maps are built. When map-agent's generator
// lands this becomes `GameMap::generate(&cfg.map_kind, seed, cfg.team_size)`.
fn build_map(cfg: &EnvConfig, seed: u64) -> GameMap {
    let _ = (seed, &cfg.map_kind);
    GameMap::arena(ARENA_WIDTH, ARENA_HEIGHT)
}

// MAP-HOOK: per-team spawn points.
fn spawn_points(map: &GameMap, team: TeamId, n: usize) -> Vec<Pos> {
    map.spawn_points(team, n)
}

/// 64-bit FNV-1a. Order-stable and platform-independent (little-endian
/// encoding of every field; floats hashed by bit pattern).
struct Fnv1a(u64);

impl Fnv1a {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;

    fn new() -> Self {
        Self(Self::OFFSET)
    }
    fn bytes(&mut self, b: &[u8]) {
        for &x in b {
            self.0 ^= u64::from(x);
            self.0 = self.0.wrapping_mul(Self::PRIME);
        }
    }
    fn u8(&mut self, v: u8) {
        self.bytes(&[v]);
    }
    fn u32(&mut self, v: u32) {
        self.bytes(&v.to_le_bytes());
    }
    fn u64(&mut self, v: u64) {
        self.bytes(&v.to_le_bytes());
    }
    fn f32(&mut self, v: f32) {
        self.u32(v.to_bits());
    }
    fn pos(&mut self, p: Pos) {
        self.f32(p.x);
        self.f32(p.y);
        self.u8(p.floor);
    }
    fn finish(&self) -> u64 {
        self.0
    }
}

#[cfg(test)]
mod tests {
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
}
