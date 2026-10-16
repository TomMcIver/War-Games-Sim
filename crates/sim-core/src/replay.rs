//! OWNER: viewer-agent. Compact replay frames + JSON writer, consumed by
//! viewer/ (static web page). A replay = config + seed + per-tick frames.

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UnitFrame {
    pub id: usize,
    pub team: u8,
    pub x: f32,
    pub y: f32,
    pub floor: u8,
    pub yaw: f32,
    pub hp: f32,
    pub alive: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Frame {
    pub tick: u64,
    pub units: Vec<UnitFrame>,
}
