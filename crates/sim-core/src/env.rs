//! OWNER: engine-agent. One match: reset / step with raw per-unit actions.
//! Units are indexed team 0 first (0..n), then team 1 (n..2n).
//!
//! Tick order inside `step` (all deterministic, unit-index order):
//! 1. clear `World.shots`; sanitise actions (invalid head values -> 0,
//!    missing actions -> all zeros); dead units' actions are ignored;
//! 2. weapon and interact cooldowns tick down by one;
//! 3. interact (stairs if standing on them, else toggle the door in front);
//! 4. turn, then move (units move sequentially by index, sliding on walls
//!    and other units);
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
use crate::map::{GameMap, Tile};
use crate::reward::{compute_rewards, Outcome, RewardConfig, TickEvents};
use crate::types::{Action, Pos, TeamId, MAX_TEAM_SIZE};
use crate::world::{InteractOutcome, Unit, UnitStats, World};

#[derive(Clone, Debug)]
pub struct EnvConfig {
    pub team_size: usize,
    pub max_ticks: u32,
    pub tick_seconds: f32,
    /// "arena" | "building" | "village" (map-agent defines kinds).
    pub map_kind: String,
    /// "standard" procedural spawns, or a compact combat lesson: "duel",
    /// "cover", "door", "skirmish". Scenarios change initial conditions only.
    pub scenario: String,
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
            scenario: "standard".into(),
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
    /// Completed episode telemetry, present only on the terminal step.
    pub episode_stats: Option<EpisodeStats>,
}

/// Why an episode ended. Zero is reserved for VecEnv steps without a finish.
#[repr(u8)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TerminationReason {
    #[default]
    None = 0,
    TeamWipe = 1,
    MutualWipe = 2,
    Timeout = 3,
}

/// Match-level measurements. Counts include both teams unless named otherwise.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct EpisodeStats {
    pub termination: TerminationReason,
    pub shots_fired: u32,
    pub shots_hit: u32,
    pub damage_dealt: f32,
    pub kills: u32,
    pub deaths: u32,
    /// Ticks when at least one living opposing pair had line of sight.
    pub contact_ticks: u32,
    pub ending_ammo: u32,
    pub survivors_team0: u32,
    pub survivors_team1: u32,
    pub team_shots_fired: [u32; 2],
    pub team_shots_hit: [u32; 2],
    pub team_damage_dealt: [f32; 2],
    pub team_kills: [u32; 2],
    pub team_deaths: [u32; 2],
    pub team_ending_ammo: [u32; 2],
}

pub struct Env {
    pub cfg: EnvConfig,
    pub world: World,
    rng: ChaCha8Rng,
    done: bool,
    winner: Option<TeamId>,
    episode_stats: EpisodeStats,
    /// Per-unit ticks until `interact` works again (see
    /// `UnitStats::interact_cooldown_ticks`).
    interact_cooldown: Vec<u32>,
    /// Reused per-tick buffers (not part of the simulation state).
    scratch: StepScratch,
}

impl Env {
    /// Generate the map for `cfg.map_kind` from `seed` and spawn both teams.
    ///
    /// Panics if `team_size` is not in `1..=MAX_TEAM_SIZE`, `tick_seconds`
    /// is not a positive finite number, or `unit.radius` is not in (0, 0.5).
    pub fn new(cfg: EnvConfig, seed: u64) -> Self {
        validate(&cfg);
        let map = build_map(&cfg, seed);
        let units = spawn_units(&map, &cfg);
        let world = World {
            units,
            ..World::new(map)
        };
        Self::from_world(cfg, world, seed)
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
            episode_stats: EpisodeStats::default(),
            interact_cooldown: vec![0; n],
            scratch: StepScratch::default(),
        }
    }

    /// Start a new episode: rebuild the map, respawn, reseed the RNG.
    pub fn reset(&mut self, seed: u64) {
        let map = build_map(&self.cfg, seed);
        let units = spawn_units(&map, &self.cfg);
        self.world.map = map;
        self.world.units = units;
        self.world.tick = 0;
        self.world.shots.clear();
        self.rng = ChaCha8Rng::seed_from_u64(seed);
        self.done = false;
        self.winner = None;
        self.episode_stats = EpisodeStats::default();
        self.interact_cooldown.clear();
        self.interact_cooldown.resize(self.world.units.len(), 0);
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
    /// The only per-tick allocation is the returned reward vector.
    pub fn step(&mut self, actions: &[Action]) -> StepResult {
        let n = self.world.units.len();
        if self.done {
            return StepResult {
                rewards: vec![0.0; n],
                done: true,
                winner: self.winner,
                episode_stats: None,
            };
        }
        self.world.shots.clear();
        let s = &mut self.scratch;
        s.prepare(&self.world, actions);

        // Units may have been added through `world_mut()`.
        self.interact_cooldown.resize(n, 0);
        for u in self.world.units.iter_mut().filter(|u| u.alive) {
            u.cooldown = u.cooldown.saturating_sub(1);
        }
        for cd in &mut self.interact_cooldown {
            *cd = cd.saturating_sub(1);
        }

        let stats = &self.cfg.unit;
        for i in (0..n).filter(|&i| s.events.was_alive[i] && s.acts[i].interact == 1) {
            if self.interact_cooldown[i] == 0
                && self.world.interact(i, stats) != InteractOutcome::None
            {
                self.interact_cooldown[i] = stats.interact_cooldown_ticks;
            }
        }
        let dt = self.cfg.tick_seconds;
        for i in (0..n).filter(|&i| s.events.was_alive[i]) {
            self.world.turn_unit(i, s.acts[i].turn, stats, dt);
            s.moved[i] = self.world.move_unit(i, &s.acts[i], stats, dt);
        }

        if opposing_units_have_contact(&self.world) {
            self.episode_stats.contact_ticks += 1;
        }

        s.wants_shoot
            .extend((0..n).map(|i| s.events.was_alive[i] && s.acts[i].shoot == 1));
        combat::resolve_shots(
            &mut self.world,
            &s.wants_shoot,
            &s.moved,
            &self.cfg.weapon,
            stats.radius,
            &mut self.rng,
            &mut s.events,
        );
        self.episode_stats.shots_fired += self.world.shots.len() as u32;
        for shot in &self.world.shots {
            let team = self.world.units[shot.shooter].team as usize;
            self.episode_stats.team_shots_fired[team] += 1;
            self.episode_stats.team_shots_hit[team] += u32::from(shot.hit.is_some());
        }
        self.episode_stats.shots_hit += self
            .world
            .shots
            .iter()
            .filter(|shot| shot.hit.is_some())
            .count() as u32;
        self.episode_stats.damage_dealt += s.events.damage_dealt.iter().sum::<f32>();
        self.episode_stats.kills += s.events.kills.iter().sum::<u32>();
        for (i, unit) in self.world.units.iter().enumerate() {
            let team = unit.team as usize;
            self.episode_stats.team_damage_dealt[team] += s.events.damage_dealt[i];
            self.episode_stats.team_kills[team] += s.events.kills[i];
            self.episode_stats.team_deaths[team] += u32::from(s.events.was_alive[i] && !unit.alive);
        }
        self.episode_stats.deaths += s
            .events
            .was_alive
            .iter()
            .enumerate()
            .filter(|(i, alive)| **alive && !self.world.units[*i].alive)
            .count() as u32;

        self.world.tick += 1;
        let outcome = check_outcome(&self.world, self.cfg.max_ticks);
        if let Some(o) = outcome {
            self.done = true;
            self.winner = o.winner();
            self.episode_stats.survivors_team0 = self.world.alive_count(0) as u32;
            self.episode_stats.survivors_team1 = self.world.alive_count(1) as u32;
            self.episode_stats.ending_ammo = self.world.units.iter().map(|u| u.ammo).sum();
            for unit in &self.world.units {
                self.episode_stats.team_ending_ammo[unit.team as usize] += unit.ammo;
            }
            self.episode_stats.termination = match (
                self.episode_stats.survivors_team0,
                self.episode_stats.survivors_team1,
            ) {
                (0, 0) => TerminationReason::MutualWipe,
                (0, _) | (_, 0) => TerminationReason::TeamWipe,
                _ => TerminationReason::Timeout,
            };
        }
        let rewards = compute_rewards(&self.cfg.reward, stats.max_hp, &s.teams, &s.events, outcome);
        StepResult {
            rewards,
            done: self.done,
            winner: self.winner,
            episode_stats: outcome.map(|_| self.episode_stats),
        }
    }

    /// Physical availability for shoot and interact, two floats per unit.
    /// Cooldowns tick down before actions are resolved on the next step.
    pub fn write_action_availability(&self, out: &mut [f32]) {
        assert_eq!(out.len(), self.world.units.len() * 2);
        for (i, u) in self.world.units.iter().enumerate() {
            out[2 * i] = f32::from(u.alive && u.ammo > 0 && u.cooldown <= 1);
            out[2 * i + 1] = f32::from(
                u.alive
                    && self.interact_cooldown.get(i).copied().unwrap_or(0) <= 1
                    && self.world.can_interact(i, self.cfg.unit.radius),
            );
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

/// Reusable per-tick buffers so `step` does not allocate.
#[derive(Debug, Default)]
struct StepScratch {
    acts: Vec<Action>,
    moved: Vec<bool>,
    wants_shoot: Vec<bool>,
    teams: Vec<TeamId>,
    events: TickEvents,
}

impl StepScratch {
    /// Refill every buffer for a tick over `world`'s units. Missing actions
    /// become all-zero, invalid head values become 0.
    fn prepare(&mut self, world: &World, actions: &[Action]) {
        let n = world.units.len();
        self.acts.clear();
        self.acts
            .extend((0..n).map(|i| actions.get(i).copied().map(sanitize).unwrap_or_default()));
        self.moved.clear();
        self.moved.resize(n, false);
        self.wants_shoot.clear();
        self.teams.clear();
        self.teams.extend(world.units.iter().map(|u| u.team));
        self.events.reset(world.units.iter().map(|u| u.alive));
    }
}

/// A team with no living units loses; both wiped = draw; timeout = draw.
fn check_outcome(world: &World, max_ticks: u32) -> Option<Outcome> {
    match (world.alive_count(0), world.alive_count(1)) {
        (0, 0) => Some(Outcome::Draw),
        (0, _) => Some(Outcome::Win(1)),
        (_, 0) => Some(Outcome::Win(0)),
        _ if world.tick >= u64::from(max_ticks) => Some(Outcome::Draw),
        _ => None,
    }
}

fn opposing_units_have_contact(world: &World) -> bool {
    world.units.iter().enumerate().any(|(i, a)| {
        a.alive
            && world
                .units
                .iter()
                .skip(i + 1)
                .any(|b| b.alive && a.team != b.team && world.map.line_of_sight(a.pos, b.pos))
    })
}

/// Largest unit radius the generated maps support (1-tile doorways).
const MAX_UNIT_RADIUS: f32 = 0.5;

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
    let r = cfg.unit.radius;
    assert!(
        r > 0.0 && r < MAX_UNIT_RADIUS,
        "unit.radius must be in (0, {MAX_UNIT_RADIUS}) so units fit through doorways, got {r}"
    );
    match cfg.scenario.as_str() {
        "standard" => {}
        "duel" | "cover" | "door" => assert_eq!(cfg.team_size, 1, "scenario requires 1v1"),
        "skirmish" => assert_eq!(cfg.team_size, 2, "skirmish requires 2v2"),
        other => panic!("unknown scenario {other:?}"),
    }
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

/// Team 0 then team 1 on the map's spawn zones, each facing the centroid of
/// the enemy spawns.
fn spawn_units(map: &GameMap, cfg: &EnvConfig) -> Vec<Unit> {
    let n = cfg.team_size;
    let points: [Vec<Pos>; 2] = match cfg.scenario.as_str() {
        "duel" => [vec![scenario_pos(4.5, 4.5)], vec![scenario_pos(8.5, 4.5)]],
        "cover" => [vec![scenario_pos(5.5, 5.5)], vec![scenario_pos(13.5, 5.5)]],
        "door" => [vec![scenario_pos(7.5, 5.5)], vec![scenario_pos(11.5, 5.5)]],
        "skirmish" => [
            vec![scenario_pos(5.5, 4.5), scenario_pos(5.5, 8.5)],
            vec![scenario_pos(13.5, 4.5), scenario_pos(13.5, 8.5)],
        ],
        _ => [0, 1].map(|t| map.spawn_points(t, n)),
    };
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

fn scenario_pos(x: f32, y: f32) -> Pos {
    Pos { x, y, floor: 0 }
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

/// The single place maps are built: a seeded generator, so different seeds
/// give different layouts. Unknown kinds fall back to "arena" (the Python
/// bindings validate `map_kind` up front).
fn build_map(cfg: &EnvConfig, seed: u64) -> GameMap {
    if cfg.scenario == "standard" {
        return GameMap::generate(&cfg.map_kind, seed, cfg.team_size);
    }
    let (w, h) = match cfg.scenario.as_str() {
        "duel" => (13, 9),
        "skirmish" => (19, 13),
        _ => (19, 11),
    };
    let mut map = GameMap::filled(w, h, 1, Tile::Floor);
    for x in 0..w {
        map.set_tile(x, 0, 0, Tile::Wall);
        map.set_tile(x, h - 1, 0, Tile::Wall);
    }
    for y in 0..h {
        map.set_tile(0, y, 0, Tile::Wall);
        map.set_tile(w - 1, y, 0, Tile::Wall);
    }
    match cfg.scenario.as_str() {
        "cover" => {
            for y in 4..=6 {
                map.set_tile(9, y, 0, Tile::Wall);
            }
        }
        "door" => {
            for y in 1..h - 1 {
                map.set_tile(9, y, 0, if y == 5 { Tile::DoorClosed } else { Tile::Wall });
            }
        }
        _ => {}
    }
    map
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
#[path = "env_tests.rs"]
mod tests;
