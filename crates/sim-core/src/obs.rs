//! OWNER: obs-agent. Observation encoding with fog of war (an agent only
//! sees what has line of sight). Writes into caller-provided flat f32 buffers.
//! Layouts are documented in docs/INTERFACE.md.

use crate::types::*;
use crate::world::World;

/// Floats per agent for entity tokens: MAX_TOKENS * TOKEN_FEATS.
pub const TOKENS_LEN: usize = MAX_TOKENS * TOKEN_FEATS;
/// Floats per agent for the local map patch.
pub const LOCAL_MAP_LEN: usize = MAP_CHANNELS * LOCAL_VIEW * LOCAL_VIEW;
/// Floats for the centralized critic state (whole battlefield, both teams).
pub const STATE_LEN: usize = MAX_UNITS * STATE_FEATS;

/// Fill `tokens` (TOKENS_LEN), `token_mask` (MAX_TOKENS, 1.0 = valid) and
/// `local_map` (LOCAL_MAP_LEN) for `unit`.
pub fn write_agent_obs(
    _world: &World,
    _unit: UnitId,
    _tokens: &mut [f32],
    _token_mask: &mut [f32],
    _local_map: &mut [f32],
) {
    todo!("obs-agent")
}

/// Full-information state for the centralized critic (training only).
pub fn write_global_state(_world: &World, _out: &mut [f32]) {
    todo!("obs-agent")
}
