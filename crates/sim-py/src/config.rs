//! Validation of the Python constructor arguments into a sim-core `EnvConfig`.
//!
//! Everything here is plain Rust (no PyO3 types) so the rules are easy to read;
//! errors are returned as messages and turned into `ValueError` by the caller.

use sim_core::{EnvConfig, MAX_TEAM_SIZE};

/// Map generators known to sim-core (see `EnvConfig::map_kind`).
pub const MAP_KINDS: [&str; 3] = ["arena", "building", "village"];

/// Upper bound on `n_envs`: keeps buffer sizes far from allocation failure
/// (which would abort the interpreter instead of raising).
pub const MAX_N_ENVS: i128 = 1 << 20;

/// Upper bound on `num_threads` for the dedicated rayon pool.
pub const MAX_THREADS: i128 = 1024;

/// Raw constructor arguments, exactly as received from Python.
pub struct RawArgs<'a> {
    pub n_envs: i128,
    pub team_size: i128,
    pub map_kind: &'a str,
    pub max_ticks: i128,
    pub seed: i128,
    pub tick_seconds: f64,
    pub num_threads: Option<i128>,
}

/// Validated settings, ready to build a `sim_core::VecEnv`.
#[derive(Debug)]
pub struct Settings {
    pub n_envs: usize,
    pub cfg: EnvConfig,
    pub seed: u64,
    pub num_threads: Option<usize>,
}

fn int_in_range(name: &str, value: i128, min: i128, max: i128) -> Result<i128, String> {
    if (min..=max).contains(&value) {
        Ok(value)
    } else {
        Err(format!("{name} must be in [{min}, {max}], got {value}"))
    }
}

pub fn validate(raw: &RawArgs<'_>) -> Result<Settings, String> {
    let n_envs = int_in_range("n_envs", raw.n_envs, 1, MAX_N_ENVS)?;
    let team_size = int_in_range("team_size", raw.team_size, 1, MAX_TEAM_SIZE as i128)?;
    let max_ticks = int_in_range("max_ticks", raw.max_ticks, 1, u32::MAX as i128)?;
    let seed = int_in_range("seed", raw.seed, 0, u64::MAX as i128)?;
    let num_threads = raw
        .num_threads
        .map(|n| int_in_range("num_threads", n, 1, MAX_THREADS))
        .transpose()?;

    if !MAP_KINDS.contains(&raw.map_kind) {
        return Err(format!(
            "map_kind must be one of {MAP_KINDS:?}, got {:?}",
            raw.map_kind
        ));
    }
    let tick_seconds = raw.tick_seconds as f32;
    if !(tick_seconds.is_finite() && tick_seconds > 0.0) {
        return Err(format!(
            "tick_seconds must be a finite number > 0, got {}",
            raw.tick_seconds
        ));
    }

    // Values are range-checked above, so the narrowing casts are lossless.
    // `..default()` keeps any EnvConfig fields not exposed here at their defaults.
    #[allow(clippy::needless_update)]
    let cfg = EnvConfig {
        team_size: team_size as usize,
        max_ticks: max_ticks as u32,
        tick_seconds,
        map_kind: raw.map_kind.to_owned(),
        ..EnvConfig::default()
    };
    Ok(Settings {
        n_envs: n_envs as usize,
        cfg,
        seed: seed as u64,
        num_threads: num_threads.map(|n| n as usize),
    })
}
