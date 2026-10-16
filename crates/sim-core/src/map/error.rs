//! Map error type.

use std::fmt;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MapError {
    /// `generate`/`try_generate` got a kind it does not know.
    UnknownKind(String),
    /// Map JSON could not be parsed or has the wrong shape.
    InvalidJson(String),
    /// A tile value outside the contract's 0..=7 range.
    InvalidTile { index: usize, value: u8 },
    /// A map failed validation (spawns, stairs pairing, connectivity).
    Invalid(String),
}

impl fmt::Display for MapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MapError::UnknownKind(k) => write!(
                f,
                "unknown map kind {k:?} (expected one of: {})",
                super::MapKind::NAMES.join(", ")
            ),
            MapError::InvalidJson(msg) => write!(f, "invalid map JSON: {msg}"),
            MapError::InvalidTile { index, value } => {
                write!(f, "invalid tile value {value} at index {index}")
            }
            MapError::Invalid(msg) => write!(f, "invalid map: {msg}"),
        }
    }
}

impl std::error::Error for MapError {}
