//! OWNER: engine-agent. One match: reset / step with raw per-unit actions.
//! Units are indexed team 0 first (0..n), then team 1 (n..2n).

use crate::types::{Action, TeamId};
use crate::world::World;

#[derive(Clone, Debug)]
pub struct EnvConfig {
    pub team_size: usize,
    pub max_ticks: u32,
    pub tick_seconds: f32,
    /// "arena" | "building" | "village" (map-agent defines kinds).
    pub map_kind: String,
}

impl Default for EnvConfig {
    fn default() -> Self {
        Self { team_size: 2, max_ticks: 600, tick_seconds: 0.1, map_kind: "arena".into() }
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
}

impl Env {
    pub fn new(cfg: EnvConfig, seed: u64) -> Self {
        let _ = (seed, &cfg);
        todo!("engine-agent")
    }
    pub fn reset(&mut self, _seed: u64) { todo!("engine-agent") }
    /// `actions.len() == 2 * team_size`. Dead units' actions are ignored.
    pub fn step(&mut self, _actions: &[Action]) -> StepResult { todo!("engine-agent") }
    /// FNV/xxhash-style digest of full state, for determinism tests.
    pub fn state_hash(&self) -> u64 { todo!("engine-agent") }
}
