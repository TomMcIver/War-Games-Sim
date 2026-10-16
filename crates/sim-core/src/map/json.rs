//! Map JSON per docs/INTERFACE.md:
//! `{"width":W,"height":H,"floors":F,"tiles":[u8...]}`, index `(f*H + y)*W + x`.

use serde::{Deserialize, Serialize};

use super::{GameMap, MapError, Tile};

#[derive(Serialize)]
struct MapJsonOut<'a> {
    width: usize,
    height: usize,
    floors: usize,
    tiles: &'a [u8],
}

#[derive(Deserialize)]
struct MapJsonIn {
    width: usize,
    height: usize,
    floors: usize,
    tiles: Vec<u8>,
}

impl GameMap {
    /// Tile values as contract u8s, index `(f * height + y) * width + x`.
    pub fn tile_bytes(&self) -> Vec<u8> {
        self.tiles.iter().map(|t| t.as_u8()).collect()
    }

    /// Serialize to the contract's Map JSON.
    pub fn to_json(&self) -> String {
        let tiles = self.tile_bytes();
        let out = MapJsonOut {
            width: self.width,
            height: self.height,
            floors: self.floors,
            tiles: &tiles,
        };
        // Serializing plain integers and a byte slice cannot fail.
        serde_json::to_string(&out).expect("map JSON serialization")
    }

    /// Parse the contract's Map JSON. Spawn zones are not part of the JSON,
    /// so a parsed map derives them from its layout (see `spawn_zone`).
    pub fn from_json(s: &str) -> Result<GameMap, MapError> {
        let raw: MapJsonIn =
            serde_json::from_str(s).map_err(|e| MapError::InvalidJson(e.to_string()))?;
        if raw.width == 0 || raw.height == 0 || raw.floors == 0 {
            return Err(MapError::InvalidJson("dimensions must be non-zero".into()));
        }
        let expected = raw
            .width
            .checked_mul(raw.height)
            .and_then(|n| n.checked_mul(raw.floors))
            .ok_or_else(|| MapError::InvalidJson("dimensions overflow".into()))?;
        if raw.tiles.len() != expected {
            return Err(MapError::InvalidJson(format!(
                "expected {expected} tiles for {}x{}x{}, got {}",
                raw.width,
                raw.height,
                raw.floors,
                raw.tiles.len()
            )));
        }
        let tiles = raw
            .tiles
            .iter()
            .enumerate()
            .map(|(index, &value)| {
                Tile::from_u8(value).ok_or(MapError::InvalidTile { index, value })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut m = GameMap::filled(raw.width, raw.height, raw.floors, Tile::Void);
        m.tiles = tiles;
        Ok(m)
    }
}
