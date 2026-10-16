//! VecEnv throughput benchmark with uniformly random actions.
//!
//! `bash infra/run.sh cargo run --release -p sim-core --example bench [-- ARGS]`
//! ARGS: `n_envs team_size steps threads map_kind` (defaults 256 4 2000 0 arena;
//! threads 0 = rayon's global pool). Also times the obs encoder alone.

use std::time::Instant;

use sim_core::obs::{self, ObsMemory, LOCAL_MAP_LEN};
use sim_core::vec_env::splitmix64;
use sim_core::{Action, EnvConfig, VecEnv, ACTION_DIMS, ACTION_HEADS};

struct Args {
    n_envs: usize,
    team_size: usize,
    steps: usize,
    threads: usize,
    map_kind: String,
}

fn parse_args() -> Args {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let num = |i: usize, default: usize| -> usize {
        a.get(i).map_or(default, |s| {
            s.parse().unwrap_or_else(|_| panic!("bad arg {s:?}"))
        })
    };
    Args {
        n_envs: num(0, 256),
        team_size: num(1, 4),
        steps: num(2, 2000),
        threads: num(3, 0),
        map_kind: a.get(4).cloned().unwrap_or_else(|| "arena".into()),
    }
}

fn fill_random(actions: &mut [u8], rng: &mut u64) {
    for (i, a) in actions.iter_mut().enumerate() {
        *rng = splitmix64(*rng);
        *a = (*rng % ACTION_DIMS[i % ACTION_HEADS] as u64) as u8;
    }
}

fn main() {
    let args = parse_args();
    let cfg = EnvConfig {
        team_size: args.team_size,
        map_kind: args.map_kind.clone(),
        ..EnvConfig::default()
    };
    let mut v = VecEnv::new(args.n_envs, cfg, 0);
    v.set_threads(args.threads).expect("thread pool");
    v.reset();
    let u = v.n_units();
    let mut actions = vec![0u8; args.n_envs * u * ACTION_HEADS];
    let mut rng = 42u64;

    // Warm up.
    for _ in 0..50 {
        fill_random(&mut actions, &mut rng);
        v.step(&actions);
    }

    let mut episodes = 0u64;
    let start = Instant::now();
    for _ in 0..args.steps {
        fill_random(&mut actions, &mut rng);
        episodes += v
            .step(&actions)
            .dones
            .iter()
            .map(|&d| u64::from(d))
            .sum::<u64>();
    }
    let secs = start.elapsed().as_secs_f64();
    let env_steps = (args.n_envs * args.steps) as f64;
    println!(
        "VecEnv: {} envs x team_size {} ({} units, map {}), {} steps, threads {}",
        args.n_envs,
        args.team_size,
        u,
        args.map_kind,
        args.steps,
        if args.threads == 0 {
            rayon::current_num_threads()
        } else {
            args.threads
        }
    );
    println!(
        "  {:.0} env-steps/s, {:.0} agent-steps/s, {:.1} us/VecEnv step, {} episodes",
        env_steps / secs,
        env_steps * u as f64 / secs,
        secs * 1e6 / args.steps as f64,
        episodes
    );
    bench_obs_only(u, args.team_size, &args.map_kind);
}

/// Single-threaded cost of encoding one env's observations.
fn bench_obs_only(u: usize, team_size: usize, map_kind: &str) {
    let cfg = EnvConfig {
        team_size,
        map_kind: map_kind.into(),
        ..EnvConfig::default()
    };
    let mut env = sim_core::Env::new(cfg, 7);
    // Play a few random ticks so the snapshot is mid-episode, not spawn.
    let mut rng = 1u64;
    let mut raw = vec![0u8; u * ACTION_HEADS];
    for _ in 0..40 {
        fill_random(&mut raw, &mut rng);
        let acts: Vec<Action> = raw.chunks(ACTION_HEADS).map(Action::from_slice).collect();
        if env.step(&acts).done {
            env.reset(8);
        }
    }
    let iters = 20_000;
    let start = Instant::now();
    for _ in 0..iters {
        std::hint::black_box(obs::TeamVision::compute(std::hint::black_box(&env.world)));
    }
    let vision = start.elapsed().as_secs_f64() / iters as f64;
    let mut mem = ObsMemory::new();
    let mut tok = vec![0.0; u * obs::tokens_len(u)];
    let mut mask = vec![0.0; u * u];
    let mut lmap = vec![0.0; u * LOCAL_MAP_LEN];
    let mut state = vec![0.0; obs::state_len(u)];
    let mut alive = vec![0.0; u];
    let start = Instant::now();
    for _ in 0..iters {
        obs::write_env_obs(
            std::hint::black_box(&env.world),
            &mut mem,
            &mut tok,
            &mut mask,
            &mut lmap,
            &mut state,
            &mut alive,
        );
        std::hint::black_box((&tok, &mask, &lmap, &state, &alive));
    }
    let per = start.elapsed().as_secs_f64() / iters as f64;
    println!(
        "  obs encode alone (1 thread): {:.2} us/env ({:.2} us team vision), {:.0} envs/s",
        per * 1e6,
        vision * 1e6,
        1.0 / per
    );
}
