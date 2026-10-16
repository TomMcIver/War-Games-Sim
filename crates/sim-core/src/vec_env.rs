//! OWNER: obs-agent. Many envs stepped in parallel (rayon), auto-reset on
//! done, results written into flat buffers that numpy can wrap zero-copy.

use crate::env::EnvConfig;

pub struct VecEnv {
    pub n_envs: usize,
    pub cfg: EnvConfig,
}

impl VecEnv {
    pub fn new(n_envs: usize, cfg: EnvConfig, seed: u64) -> Self {
        let _ = seed;
        Self { n_envs, cfg }
    }
    /// Units per env = 2 * team_size.
    pub fn n_units(&self) -> usize { 2 * self.cfg.team_size }
}
