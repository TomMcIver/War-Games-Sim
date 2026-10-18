//! OWNER: obs-agent. Many envs stepped in parallel (rayon), auto-reset on
//! done, results written into flat buffers that numpy can wrap.
//! API is fixed by docs/INTERFACE.md ("Rust VecEnv API").
//!
//! Determinism: episode `k` of env `i` is seeded with
//! [`episode_seed`]`(seed, i, k)`, and every env is stepped independently,
//! so the buffers depend only on `(seed, n_envs, cfg, call sequence)` and
//! never on the thread count or scheduling.
//!
//! Episodes: `reset()` starts the next episode of every env (the first
//! `reset()` after `new` plays episode 0). `step()` auto-resets envs that
//! finish; their `rewards`/`dones`/`winners`/`episode_len` describe the
//! finished episode while the observation fields already show the first
//! observation of the next one.

use rayon::prelude::*;

use crate::env::{Env, EnvConfig, EpisodeStats};
use crate::obs::{self, ObsMemory, LOCAL_MAP_LEN};
use crate::types::*;

/// Flat row-major buffers sized for the real `U` (see docs/INTERFACE.md).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct VecEnvBuffers {
    /// `E*U*U*TOKEN_FEATS`
    pub tokens: Vec<f32>,
    /// `E*U*U`
    pub token_mask: Vec<f32>,
    /// `E*U*MAP_CHANNELS*LOCAL_VIEW*LOCAL_VIEW`
    pub local_map: Vec<f32>,
    /// `E*U*STATE_FEATS`
    pub state: Vec<f32>,
    /// `E*U`
    pub alive: Vec<f32>,
    /// `E*U*2`: shoot and interact physically available on the next tick.
    pub action_availability: Vec<f32>,
    /// `E*U`
    pub rewards: Vec<f32>,
    /// `E`
    pub dones: Vec<u8>,
    /// `E` (-1 none/draw, else team id; valid when `dones[e] == 1`)
    pub winners: Vec<i8>,
    /// `E` (valid when `dones[e] == 1`)
    pub episode_len: Vec<u32>,
    /// `E` (default on non-terminal steps; finished episode on done steps).
    pub episode_stats: Vec<EpisodeStats>,
}

impl VecEnvBuffers {
    fn new(n_envs: usize, n_units: usize) -> Self {
        Self {
            tokens: vec![0.0; n_envs * n_units * obs::tokens_len(n_units)],
            token_mask: vec![0.0; n_envs * n_units * n_units],
            local_map: vec![0.0; n_envs * n_units * LOCAL_MAP_LEN],
            state: vec![0.0; n_envs * obs::state_len(n_units)],
            alive: vec![0.0; n_envs * n_units],
            action_availability: vec![0.0; n_envs * n_units * 2],
            rewards: vec![0.0; n_envs * n_units],
            dones: vec![0; n_envs],
            winners: vec![-1; n_envs],
            episode_len: vec![0; n_envs],
            episode_stats: vec![EpisodeStats::default(); n_envs],
        }
    }
}

/// SplitMix64 finalizer (Steele et al.), a strong 64-bit mixer.
pub fn splitmix64(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Seed of episode `episode` of env `env_index` in a VecEnv seeded `base`.
pub fn episode_seed(base: u64, env_index: usize, episode: u64) -> u64 {
    splitmix64(splitmix64(splitmix64(base) ^ env_index as u64) ^ episode)
}

/// One env plus its per-env scratch, so a rayon task owns everything it
/// touches.
struct EnvSlot {
    index: usize,
    env: Env,
    memory: ObsMemory,
    actions: Vec<Action>,
    /// Episodes started so far (the next reset uses this as its index).
    episode: u64,
    /// Steps taken in the current episode.
    steps: u32,
}

impl EnvSlot {
    fn start_episode(&mut self, base_seed: u64) {
        self.env
            .reset(episode_seed(base_seed, self.index, self.episode));
        self.episode += 1;
        self.steps = 0;
        self.memory.clear();
    }
}

/// One env's slices of every buffer.
struct EnvOut<'a> {
    tokens: &'a mut [f32],
    token_mask: &'a mut [f32],
    local_map: &'a mut [f32],
    state: &'a mut [f32],
    alive: &'a mut [f32],
    action_availability: &'a mut [f32],
    rewards: &'a mut [f32],
    done: &'a mut u8,
    winner: &'a mut i8,
    episode_len: &'a mut u32,
    episode_stats: &'a mut EpisodeStats,
}

impl EnvOut<'_> {
    fn write_obs(&mut self, slot: &mut EnvSlot) {
        obs::write_env_obs(
            &slot.env.world,
            &mut slot.memory,
            self.tokens,
            self.token_mask,
            self.local_map,
            self.state,
            self.alive,
        );
        slot.env.write_action_availability(self.action_availability);
    }

    fn clear_step_info(&mut self) {
        self.rewards.fill(0.0);
        *self.done = 0;
        *self.winner = -1;
        *self.episode_len = 0;
        *self.episode_stats = EpisodeStats::default();
    }
}

pub struct VecEnv {
    pub n_envs: usize,
    pub cfg: EnvConfig,
    seed: u64,
    slots: Vec<EnvSlot>,
    buffers: VecEnvBuffers,
    pool: Option<rayon::ThreadPool>,
    started: bool,
}

impl VecEnv {
    /// Panics if `n_envs == 0` or `team_size` is not in `1..=MAX_TEAM_SIZE`.
    pub fn new(n_envs: usize, cfg: EnvConfig, seed: u64) -> Self {
        assert!(n_envs > 0, "n_envs must be > 0");
        assert!(
            (1..=MAX_TEAM_SIZE).contains(&cfg.team_size),
            "team_size must be in 1..={MAX_TEAM_SIZE}, got {}",
            cfg.team_size
        );
        let n_units = 2 * cfg.team_size;
        let slots = (0..n_envs)
            .map(|index| EnvSlot {
                index,
                env: Env::new(cfg.clone(), episode_seed(seed, index, 0)),
                memory: ObsMemory::new(),
                actions: vec![Action::default(); n_units],
                episode: 0,
                steps: 0,
            })
            .collect();
        Self {
            n_envs,
            buffers: VecEnvBuffers::new(n_envs, n_units),
            cfg,
            seed,
            slots,
            pool: None,
            started: false,
        }
    }

    /// Units per env = 2 * team_size.
    pub fn n_units(&self) -> usize {
        2 * self.cfg.team_size
    }

    /// Run `step`/`reset` on a dedicated pool of `n` threads; `n == 0`
    /// returns to rayon's global pool. Results are identical either way.
    pub fn set_threads(&mut self, n: usize) -> Result<(), rayon::ThreadPoolBuildError> {
        self.pool = match n {
            0 => None,
            n => Some(rayon::ThreadPoolBuilder::new().num_threads(n).build()?),
        };
        Ok(())
    }

    /// Start a new episode in every env and fill the observation fields.
    /// `rewards`/`dones`/`winners`/`episode_len` are cleared.
    pub fn reset(&mut self) -> &VecEnvBuffers {
        let base_seed = self.seed;
        self.for_each_env(&[], move |slot, _, out| {
            slot.start_episode(base_seed);
            out.clear_step_info();
            out.write_obs(slot);
        });
        self.started = true;
        &self.buffers
    }

    /// `actions`: `E*U*ACTION_HEADS` values, row-major `[E, U, 5]`. Values
    /// outside `ACTION_DIMS` are treated as 0. Done envs auto-reset.
    /// Panics on a wrong length or if `reset` was never called.
    pub fn step(&mut self, actions: &[u8]) -> &VecEnvBuffers {
        assert!(self.started, "VecEnv::reset must be called before step");
        let expected = self.n_envs * self.n_units() * ACTION_HEADS;
        assert_eq!(
            actions.len(),
            expected,
            "actions must have n_envs * n_units * {ACTION_HEADS} = {expected} values"
        );
        let base_seed = self.seed;
        self.for_each_env(actions, move |slot, acts, out| {
            step_one(slot, acts, out, base_seed);
        });
        &self.buffers
    }

    pub fn buffers(&self) -> &VecEnvBuffers {
        &self.buffers
    }

    /// Run `f` for every env in parallel with its slot, action slice (empty
    /// if `actions` is empty) and buffer slices. No heap allocation here.
    fn for_each_env<F>(&mut self, actions: &[u8], f: F)
    where
        F: Fn(&mut EnvSlot, &[u8], &mut EnvOut<'_>) + Sync + Send,
    {
        let u = self.n_units();
        let act_chunk = if actions.is_empty() {
            1
        } else {
            u * ACTION_HEADS
        };
        let b = &mut self.buffers;
        let slots = &mut self.slots;
        let mut job = || {
            (
                slots.par_iter_mut(),
                b.tokens.par_chunks_mut(u * obs::tokens_len(u)),
                b.token_mask.par_chunks_mut(u * u),
                b.local_map.par_chunks_mut(u * LOCAL_MAP_LEN),
                b.state.par_chunks_mut(obs::state_len(u)),
                b.alive.par_chunks_mut(u),
                b.action_availability.par_chunks_mut(u * 2),
                b.rewards.par_chunks_mut(u),
                b.dones.par_iter_mut(),
                b.winners.par_iter_mut(),
                b.episode_len.par_iter_mut(),
                b.episode_stats.par_iter_mut(),
            )
                .into_par_iter()
                .enumerate()
                .for_each(
                    |(
                        i,
                        (
                            slot,
                            tokens,
                            token_mask,
                            local_map,
                            state,
                            alive,
                            action_availability,
                            rewards,
                            done,
                            winner,
                            episode_len,
                            episode_stats,
                        ),
                    )| {
                        let acts = if actions.is_empty() {
                            &[][..]
                        } else {
                            &actions[i * act_chunk..(i + 1) * act_chunk]
                        };
                        let mut out = EnvOut {
                            tokens,
                            token_mask,
                            local_map,
                            state,
                            alive,
                            action_availability,
                            rewards,
                            done,
                            winner,
                            episode_len,
                            episode_stats,
                        };
                        f(slot, acts, &mut out);
                    },
                );
        };
        match &self.pool {
            Some(pool) => pool.install(job),
            None => job(),
        }
    }
}

/// Decode one head, mapping out-of-range values to 0 (no-op).
fn sanitize(value: u8, head: usize) -> u8 {
    if (value as usize) < ACTION_DIMS[head] {
        value
    } else {
        0
    }
}

fn step_one(slot: &mut EnvSlot, acts: &[u8], out: &mut EnvOut<'_>, base_seed: u64) {
    let (rows, _) = acts.as_chunks::<ACTION_HEADS>();
    for (a, &[forward, strafe, turn, shoot, interact]) in slot.actions.iter_mut().zip(rows) {
        *a = Action {
            forward: sanitize(forward, 0),
            strafe: sanitize(strafe, 1),
            turn: sanitize(turn, 2),
            shoot: sanitize(shoot, 3),
            interact: sanitize(interact, 4),
        };
    }
    let result = slot.env.step(&slot.actions);
    slot.steps += 1;
    assert_eq!(
        result.rewards.len(),
        out.rewards.len(),
        "Env::step reward count"
    );
    out.rewards.copy_from_slice(&result.rewards);
    if result.done {
        *out.done = 1;
        *out.winner = result.winner.map_or(-1, |t| t as i8);
        *out.episode_len = slot.steps;
        *out.episode_stats = result.episode_stats.expect("terminal episode stats");
        slot.start_episode(base_seed);
    } else {
        *out.done = 0;
        *out.winner = -1;
        *out.episode_len = 0;
        *out.episode_stats = EpisodeStats::default();
    }
    out.write_obs(slot);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(team_size: usize, max_ticks: u32) -> EnvConfig {
        EnvConfig {
            team_size,
            max_ticks,
            ..EnvConfig::default()
        }
    }

    /// Deterministic pseudo-random action stream (valid values only).
    fn random_actions(n: usize, rng: &mut u64) -> Vec<u8> {
        (0..n)
            .map(|i| {
                *rng = splitmix64(*rng);
                (*rng % ACTION_DIMS[i % ACTION_HEADS] as u64) as u8
            })
            .collect()
    }

    fn assert_shapes(v: &VecEnv) {
        let (e, u) = (v.n_envs, v.n_units());
        let b = v.buffers();
        assert_eq!(b.tokens.len(), e * u * u * TOKEN_FEATS);
        assert_eq!(b.token_mask.len(), e * u * u);
        assert_eq!(
            b.local_map.len(),
            e * u * MAP_CHANNELS * LOCAL_VIEW * LOCAL_VIEW
        );
        assert_eq!(b.state.len(), e * u * STATE_FEATS);
        assert_eq!(b.alive.len(), e * u);
        assert_eq!(b.rewards.len(), e * u);
        assert_eq!(b.episode_stats.len(), e);
        assert_eq!(
            (b.dones.len(), b.winners.len(), b.episode_len.len()),
            (e, e, e)
        );
    }

    #[test]
    fn splitmix_and_episode_seeds_are_distinct() {
        assert_ne!(splitmix64(0), splitmix64(1));
        let seeds: std::collections::HashSet<u64> = (0..64)
            .flat_map(|i| (0..8).map(move |k| episode_seed(7, i, k)))
            .collect();
        assert_eq!(seeds.len(), 64 * 8);
        assert_ne!(episode_seed(1, 0, 0), episode_seed(2, 0, 0));
        assert_eq!(episode_seed(3, 4, 5), episode_seed(3, 4, 5));
    }

    #[test]
    fn sanitize_maps_out_of_range_to_zero() {
        assert_eq!(sanitize(2, 0), 2);
        assert_eq!(sanitize(3, 0), 0);
        assert_eq!(sanitize(1, 3), 1);
        assert_eq!(sanitize(2, 3), 0);
        assert_eq!(sanitize(255, 4), 0);
    }

    #[test]
    #[should_panic(expected = "team_size must be in")]
    fn rejects_oversized_team() {
        VecEnv::new(1, cfg(MAX_TEAM_SIZE + 1, 10), 0);
    }

    #[test]
    fn shapes_and_reset_obs() {
        for team_size in [1, 2, 4] {
            let mut v = VecEnv::new(3, cfg(team_size, 50), 1);
            assert_eq!(v.n_units(), 2 * team_size);
            assert_shapes(&v);
            let u = v.n_units();
            let b = v.reset().clone();
            assert_shapes(&v);
            assert!(b.alive.iter().all(|&a| a == 1.0));
            assert!(b.dones.iter().all(|&d| d == 0));
            // Every agent's token 0 is itself.
            for agent in 0..3 * u {
                assert_eq!(b.token_mask[agent * u], 1.0);
                assert_eq!(b.tokens[agent * u * TOKEN_FEATS + obs::F_IS_SELF], 1.0);
            }
            v.step(&vec![0u8; 3 * u * ACTION_HEADS]);
            assert_shapes(&v);
        }
    }

    #[test]
    #[should_panic(expected = "actions must have")]
    fn wrong_action_length_panics() {
        let mut v = VecEnv::new(2, cfg(2, 50), 1);
        v.reset();
        v.step(&[0; 7]);
    }

    #[test]
    #[should_panic(expected = "reset must be called")]
    fn step_before_reset_panics() {
        let mut v = VecEnv::new(1, cfg(1, 50), 1);
        v.step(&[0; 2 * ACTION_HEADS]);
    }

    #[test]
    fn invalid_action_values_act_as_noop() {
        let mut a = VecEnv::new(2, cfg(2, 50), 9);
        let mut b = VecEnv::new(2, cfg(2, 50), 9);
        a.reset();
        b.reset();
        let n = 2 * 4 * ACTION_HEADS;
        for _ in 0..10 {
            let a_out = a.step(&vec![200u8; n]).clone();
            assert_eq!(&a_out, b.step(&vec![0u8; n]));
        }
    }

    fn run(threads: usize, steps: usize) -> Vec<VecEnvBuffers> {
        let mut v = VecEnv::new(8, cfg(3, 40), 1234);
        v.set_threads(threads).expect("thread pool");
        let mut out = vec![v.reset().clone()];
        let mut rng = 99;
        let n = v.n_envs * v.n_units() * ACTION_HEADS;
        for _ in 0..steps {
            let acts = random_actions(n, &mut rng);
            out.push(v.step(&acts).clone());
        }
        out
    }

    #[test]
    fn deterministic_across_thread_counts() {
        let one = run(1, 100);
        assert_eq!(one, run(4, 100));
        assert_eq!(one, run(0, 100));
    }

    /// Encode a fresh env playing episode `episode` of env `e`.
    fn fresh_obs(v: &VecEnv, e: usize, episode: u64) -> (Vec<f32>, Vec<f32>) {
        let mut env = Env::new(v.cfg.clone(), 0);
        env.reset(episode_seed(v.seed, e, episode));
        let u = v.n_units();
        let mut tok = vec![0.0; u * obs::tokens_len(u)];
        let mut mask = vec![0.0; u * u];
        let mut lmap = vec![0.0; u * LOCAL_MAP_LEN];
        let mut state = vec![0.0; obs::state_len(u)];
        let mut alive = vec![0.0; u];
        let mut mem = ObsMemory::new();
        obs::write_env_obs(
            &env.world, &mut mem, &mut tok, &mut mask, &mut lmap, &mut state, &mut alive,
        );
        (tok, state)
    }

    #[test]
    fn auto_reset_reports_finished_episode_and_new_obs() {
        let max_ticks = 15;
        let mut v = VecEnv::new(4, cfg(2, max_ticks), 5);
        v.reset();
        let (u, n) = (v.n_units(), v.n_envs * v.n_units() * ACTION_HEADS);
        let (tl, sl) = (u * obs::tokens_len(u), obs::state_len(u));
        let mut rng = 3;
        let mut finished = 0;
        for _ in 0..3 * max_ticks {
            let acts = random_actions(n, &mut rng);
            let b = v.step(&acts).clone();
            for e in 0..v.n_envs {
                if b.dones[e] == 0 {
                    assert_eq!((b.winners[e], b.episode_len[e]), (-1, 0));
                    assert_eq!(b.episode_stats[e], EpisodeStats::default());
                    continue;
                }
                finished += 1;
                assert!((1..=max_ticks).contains(&b.episode_len[e]));
                assert!((-1..=1).contains(&b.winners[e]));
                assert_ne!(
                    b.episode_stats[e].termination,
                    crate::env::TerminationReason::None
                );
                assert!(b.episode_stats[e].contact_ticks <= b.episode_len[e]);
                // Obs already show the first obs of the next episode.
                let slot = &v.slots[e];
                assert_eq!(slot.steps, 0);
                let (tok, state) = fresh_obs(&v, e, slot.episode - 1);
                assert_eq!(&b.tokens[e * tl..(e + 1) * tl], &tok[..]);
                assert_eq!(&b.state[e * sl..(e + 1) * sl], &state[..]);
                assert!(b.alive[e * u..(e + 1) * u].iter().all(|&a| a == 1.0));
            }
        }
        assert!(
            finished >= v.n_envs,
            "every env should finish within 3 * max_ticks"
        );
    }
}
