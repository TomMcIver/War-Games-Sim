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

#[derive(Clone, Debug)]
pub struct World {
    pub map: GameMap,
    pub units: Vec<Unit>,
    pub tick: u64,
}
