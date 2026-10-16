//! Shared constants and plain types. This file is the contract surface.
//! Changing a constant here changes the Python tensor shapes: update
//! docs/INTERFACE.md in the same PR.

pub type TeamId = u8;
pub type UnitId = usize;

/// Largest supported team. Tensors are padded to this.
pub const MAX_TEAM_SIZE: usize = 8;
pub const MAX_UNITS: usize = 2 * MAX_TEAM_SIZE;

/// Entity tokens per agent observation: self + allies + enemies (padded).
pub const MAX_TOKENS: usize = MAX_UNITS;
pub const TOKEN_FEATS: usize = 12;

/// Local tile patch around the agent (square, odd side length).
pub const LOCAL_VIEW: usize = 9;
pub const MAP_CHANNELS: usize = 4;

/// Per-unit feature count in the centralized critic's global state.
pub const STATE_FEATS: usize = 10;

/// Multi-discrete action: forward, strafe, turn, shoot, interact.
pub const ACTION_HEADS: usize = 5;
pub const ACTION_DIMS: [usize; ACTION_HEADS] = [3, 3, 3, 2, 2];

/// Continuous position: x/y in tile units, discrete floor.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Pos {
    pub x: f32,
    pub y: f32,
    pub floor: u8,
}

/// One unit's action for one tick.
/// forward: 0 none, 1 forward, 2 back
/// strafe:  0 none, 1 left,    2 right
/// turn:    0 none, 1 left,    2 right
/// shoot:   0/1
/// interact: 0/1 (open/close door, use stairs)
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Action {
    pub forward: u8,
    pub strafe: u8,
    pub turn: u8,
    pub shoot: u8,
    pub interact: u8,
}

impl Action {
    pub fn from_slice(a: &[u8]) -> Self {
        Self { forward: a[0], strafe: a[1], turn: a[2], shoot: a[3], interact: a[4] }
    }
    pub fn to_array(self) -> [u8; ACTION_HEADS] {
        [self.forward, self.strafe, self.turn, self.shoot, self.interact]
    }
    /// True if every head is within ACTION_DIMS.
    pub fn is_valid(self) -> bool {
        self.to_array().iter().zip(ACTION_DIMS).all(|(&v, d)| (v as usize) < d)
    }
}
