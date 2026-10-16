//! OWNER: obs-agent. Many envs stepped in parallel (rayon), auto-reset on
//! done, results written into flat buffers that numpy can wrap.
//! API is fixed by docs/INTERFACE.md ("Rust VecEnv API").

use crate::env::EnvConfig;

#[derive(Clone, Debug, Default)]
pub struct VecEnvBuffers {
    pub tokens: Vec<f32>,
    pub token_mask: Vec<f32>,
    pub local_map: Vec<f32>,
    pub state: Vec<f32>,
    pub alive: Vec<f32>,
    pub rewards: Vec<f32>,
    pub dones: Vec<u8>,
    pub winners: Vec<i8>,
    pub episode_len: Vec<u32>,
}

pub struct VecEnv {
    pub n_envs: usize,
    pub cfg: EnvConfig,
    buffers: VecEnvBuffers,
}

impl VecEnv {
    pub fn new(n_envs: usize, cfg: EnvConfig, seed: u64) -> Self {
        let _ = seed;
        Self { n_envs, cfg, buffers: VecEnvBuffers::default() }
    }
    /// Units per env = 2 * team_size.
    pub fn n_units(&self) -> usize { 2 * self.cfg.team_size }
    pub fn reset(&mut self) -> &VecEnvBuffers { todo!("obs-agent") }
    pub fn step(&mut self, _actions: &[u8]) -> &VecEnvBuffers { todo!("obs-agent") }
    pub fn buffers(&self) -> &VecEnvBuffers { &self.buffers }
}
