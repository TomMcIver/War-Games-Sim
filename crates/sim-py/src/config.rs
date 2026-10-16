//! Validation of the Python constructor arguments into a sim-core `EnvConfig`.
//!
//! Everything here is plain Rust (no PyO3 types) so the rules are easy to read;
//! errors are returned as messages and turned into `ValueError` by the caller.

use sim_core::reward::RewardConfig;
use sim_core::{EnvConfig, MAX_TEAM_SIZE};

/// Map generators known to sim-core (see `EnvConfig::map_kind`).
pub const MAP_KINDS: [&str; 3] = ["arena", "building", "village"];

/// Upper bound on `n_envs`: keeps buffer sizes far from allocation failure
/// (which would abort the interpreter instead of raising).
pub const MAX_N_ENVS: i128 = 1 << 20;

/// Upper bound on `num_threads` for the dedicated rayon pool.
pub const MAX_THREADS: i128 = 1024;

/// Optional reward overrides (`None` = sim-core default). Python names are
/// `reward_<field>` except `shaping_scale`; see `sim_core::reward::RewardConfig`.
#[derive(Default)]
pub struct RewardArgs {
    pub win: Option<f64>,
    pub loss: Option<f64>,
    pub draw: Option<f64>,
    pub damage_dealt: Option<f64>,
    pub kill: Option<f64>,
    pub damage_taken: Option<f64>,
    pub time_penalty: Option<f64>,
    pub ammo_waste: Option<f64>,
    pub shaping_scale: Option<f64>,
}

/// Raw constructor arguments, exactly as received from Python.
pub struct RawArgs<'a> {
    pub n_envs: i128,
    pub team_size: i128,
    pub map_kind: &'a str,
    pub max_ticks: i128,
    pub seed: i128,
    pub tick_seconds: f64,
    pub num_threads: Option<i128>,
    pub reward: RewardArgs,
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

/// `value` as a finite f32 (and `>= 0` when `non_negative`).
fn finite_f32(name: &str, value: f64, non_negative: bool) -> Result<f32, String> {
    let v = value as f32;
    if !v.is_finite() {
        return Err(format!("{name} must be a finite float32, got {value}"));
    }
    if non_negative && v < 0.0 {
        return Err(format!("{name} must be >= 0, got {value}"));
    }
    Ok(v)
}

/// Apply the overrides in `args` on top of `base`.
fn reward_config(args: &RewardArgs, base: RewardConfig) -> Result<RewardConfig, String> {
    let mut r = base;
    // (python name, override, must be non-negative, target field)
    let fields: [(&str, Option<f64>, bool, &mut f32); 9] = [
        ("reward_win", args.win, false, &mut r.win),
        ("reward_loss", args.loss, false, &mut r.loss),
        ("reward_draw", args.draw, false, &mut r.draw),
        (
            "reward_damage_dealt",
            args.damage_dealt,
            true,
            &mut r.damage_dealt,
        ),
        ("reward_kill", args.kill, true, &mut r.kill),
        (
            "reward_damage_taken",
            args.damage_taken,
            true,
            &mut r.damage_taken,
        ),
        (
            "reward_time_penalty",
            args.time_penalty,
            true,
            &mut r.time_penalty,
        ),
        (
            "reward_ammo_waste",
            args.ammo_waste,
            true,
            &mut r.ammo_waste,
        ),
        (
            "shaping_scale",
            args.shaping_scale,
            true,
            &mut r.shaping_scale,
        ),
    ];
    for (name, value, non_negative, slot) in fields {
        if let Some(v) = value {
            *slot = finite_f32(name, v, non_negative)?;
        }
    }
    Ok(r)
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
    let tick_seconds = finite_f32("tick_seconds", raw.tick_seconds, true)?;
    if tick_seconds <= 0.0 {
        return Err(format!(
            "tick_seconds must be > 0, got {}",
            raw.tick_seconds
        ));
    }

    let defaults = EnvConfig::default();
    let reward = reward_config(&raw.reward, defaults.reward.clone())?;
    // Values are range-checked above, so the narrowing casts are lossless.
    let cfg = EnvConfig {
        team_size: team_size as usize,
        max_ticks: max_ticks as u32,
        tick_seconds,
        map_kind: raw.map_kind.to_owned(),
        reward,
        ..defaults
    };
    Ok(Settings {
        n_envs: n_envs as usize,
        cfg,
        seed: seed as u64,
        num_threads: num_threads.map(|n| n as usize),
    })
}
