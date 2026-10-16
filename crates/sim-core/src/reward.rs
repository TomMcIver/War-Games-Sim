//! OWNER: engine-agent. Per-tick reward computation.
//!
//! Reward for unit `i` on a tick:
//! ```text
//! r_i = terminal(team_i)
//!     + shaping_scale * ( damage_dealt * dealt_i / max_hp
//!                       + kill         * kills_i
//!                       - damage_taken * taken_i / max_hp
//!                       - time_penalty * [i alive at tick start]
//!                       - ammo_waste   * missed_shots_i )
//! ```
//! Terminal rewards are team-shared and given to every unit of the team,
//! including dead ones. All shaping weights are non-negative magnitudes;
//! set `shaping_scale = 0` to train on the pure win/loss signal.

use crate::types::TeamId;

#[derive(Clone, Debug, PartialEq)]
pub struct RewardConfig {
    /// Terminal reward for every unit of the winning team.
    pub win: f32,
    /// Terminal reward for every unit of the losing team.
    pub loss: f32,
    /// Terminal reward for everyone on a draw (incl. timeout).
    pub draw: f32,
    /// Per full `max_hp` of damage dealt to enemies.
    pub damage_dealt: f32,
    /// Per enemy killed (last hit).
    pub kill: f32,
    /// Penalty per full `max_hp` of damage taken.
    pub damage_taken: f32,
    /// Penalty per tick for each unit alive at the start of the tick.
    pub time_penalty: f32,
    /// Penalty per shot fired that hit nobody.
    pub ammo_waste: f32,
    /// Multiplier on all shaping terms (anneal to 0). Terminal rewards are
    /// not scaled.
    pub shaping_scale: f32,
}

impl Default for RewardConfig {
    fn default() -> Self {
        Self {
            win: 1.0,
            loss: -1.0,
            draw: 0.0,
            damage_dealt: 0.2,
            kill: 0.2,
            damage_taken: 0.1,
            time_penalty: 0.0005,
            ammo_waste: 0.001,
            shaping_scale: 1.0,
        }
    }
}

/// How an episode ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    Win(TeamId),
    Draw,
}

impl Outcome {
    pub fn winner(self) -> Option<TeamId> {
        match self {
            Outcome::Win(t) => Some(t),
            Outcome::Draw => None,
        }
    }
}

/// Everything that happened to each unit during one tick.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TickEvents {
    pub damage_dealt: Vec<f32>,
    pub damage_taken: Vec<f32>,
    pub kills: Vec<u32>,
    pub missed_shots: Vec<u32>,
    /// Alive at the start of the tick.
    pub was_alive: Vec<bool>,
}

impl TickEvents {
    pub fn new(was_alive: Vec<bool>) -> Self {
        let n = was_alive.len();
        Self {
            damage_dealt: vec![0.0; n],
            damage_taken: vec![0.0; n],
            kills: vec![0; n],
            missed_shots: vec![0; n],
            was_alive,
        }
    }
}

/// One reward per unit. `teams[i]` is unit i's team.
pub fn compute_rewards(
    cfg: &RewardConfig,
    max_hp: f32,
    teams: &[TeamId],
    ev: &TickEvents,
    outcome: Option<Outcome>,
) -> Vec<f32> {
    let hp_norm = if max_hp > 0.0 { 1.0 / max_hp } else { 0.0 };
    teams
        .iter()
        .enumerate()
        .map(|(i, &team)| {
            let alive_term = if ev.was_alive[i] {
                cfg.time_penalty
            } else {
                0.0
            };
            let shaping = cfg.damage_dealt * ev.damage_dealt[i] * hp_norm
                + cfg.kill * ev.kills[i] as f32
                - cfg.damage_taken * ev.damage_taken[i] * hp_norm
                - alive_term
                - cfg.ammo_waste * ev.missed_shots[i] as f32;
            let terminal = match outcome {
                None => 0.0,
                Some(Outcome::Draw) => cfg.draw,
                Some(Outcome::Win(w)) if w == team => cfg.win,
                Some(Outcome::Win(_)) => cfg.loss,
            };
            terminal + cfg.shaping_scale * shaping
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEAMS: [TeamId; 4] = [0, 0, 1, 1];

    fn busy_events() -> TickEvents {
        let mut ev = TickEvents::new(vec![true, false, true, true]);
        ev.damage_dealt[0] = 50.0;
        ev.kills[0] = 1;
        ev.damage_taken[2] = 50.0;
        ev.missed_shots[3] = 2;
        ev
    }

    #[test]
    fn terminal_win_loss_signs_and_dead_units_included() {
        let cfg = RewardConfig {
            shaping_scale: 0.0,
            ..RewardConfig::default()
        };
        let r = compute_rewards(&cfg, 100.0, &TEAMS, &busy_events(), Some(Outcome::Win(0)));
        assert_eq!(r, vec![1.0, 1.0, -1.0, -1.0]);
        let r = compute_rewards(&cfg, 100.0, &TEAMS, &busy_events(), Some(Outcome::Draw));
        assert_eq!(r, vec![0.0; 4]);
    }

    #[test]
    fn shaping_scale_zero_gives_zero_non_terminal_reward() {
        let cfg = RewardConfig {
            shaping_scale: 0.0,
            ..RewardConfig::default()
        };
        let r = compute_rewards(&cfg, 100.0, &TEAMS, &busy_events(), None);
        assert!(r.iter().all(|&x| x == 0.0), "{r:?}");
    }

    #[test]
    fn shaping_signs() {
        let cfg = RewardConfig::default();
        let r = compute_rewards(&cfg, 100.0, &TEAMS, &busy_events(), None);
        assert!(r[0] > 0.0, "damage+kill should be positive: {r:?}");
        assert_eq!(r[1], 0.0, "dead, idle unit gets nothing");
        assert!(r[2] < 0.0, "damage taken is a penalty");
        assert!(r[3] < 0.0, "missed shots + time cost are penalties");
        let expected0 = 0.2 * 0.5 + 0.2 - 0.0005;
        assert!((r[0] - expected0).abs() < 1e-6);
    }

    #[test]
    fn shaping_scale_is_linear() {
        let full = compute_rewards(
            &RewardConfig::default(),
            100.0,
            &TEAMS,
            &busy_events(),
            None,
        );
        let half_cfg = RewardConfig {
            shaping_scale: 0.5,
            ..RewardConfig::default()
        };
        let half = compute_rewards(&half_cfg, 100.0, &TEAMS, &busy_events(), None);
        for (f, h) in full.iter().zip(&half) {
            assert!((f * 0.5 - h).abs() < 1e-6);
        }
    }
}
