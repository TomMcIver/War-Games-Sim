//! OWNER: engine-agent. Unit and World state.

use crate::map::GameMap;
use crate::types::{Pos, TeamId, UnitId};

#[derive(Clone, Debug)]
pub struct Unit {
    pub id: UnitId,
    pub team: TeamId,
    pub pos: Pos,
    /// Facing angle in radians.
    pub yaw: f32,
    pub hp: f32,
    pub ammo: u32,
    pub alive: bool,
    /// Ticks until the weapon can fire again.
    pub cooldown: u32,
}

/// A shot fired this tick (for replays/tracers). `hit` = unit that was struck.
#[derive(Clone, Debug)]
pub struct ShotEvent {
    pub shooter: UnitId,
    /// Where the shot ended (hit point or max range / wall impact).
    pub x: f32,
    pub y: f32,
    pub floor: u8,
    pub hit: Option<UnitId>,
}

#[derive(Clone, Debug)]
pub struct World {
    pub map: GameMap,
    pub units: Vec<Unit>,
    pub tick: u64,
    /// Shots fired during the most recent tick.
    pub shots: Vec<ShotEvent>,
}
