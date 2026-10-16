use super::ascii;
use crate::map::{generate, GameMap, MapError, Tile};

#[test]
fn json_has_contract_shape_and_index_order() {
    let mut m = GameMap::filled(4, 3, 2, Tile::Floor);
    m.set_tile(3, 2, 1, Tile::StairsDown);
    m.set_tile(1, 0, 0, Tile::Window);
    let v: serde_json::Value = serde_json::from_str(&m.to_json()).unwrap();
    let obj = v.as_object().unwrap();
    let mut keys: Vec<&str> = obj.keys().map(String::as_str).collect();
    keys.sort_unstable();
    assert_eq!(keys, ["floors", "height", "tiles", "width"]);
    assert_eq!((v["width"].as_u64(), v["height"].as_u64(), v["floors"].as_u64()), (Some(4), Some(3), Some(2)));
    let tiles = v["tiles"].as_array().unwrap();
    assert_eq!(tiles.len(), 24);
    let at = |x: usize, y: usize, f: usize| tiles[(f * 3 + y) * 4 + x].as_u64().unwrap();
    assert_eq!(at(3, 2, 1), 7);
    assert_eq!(at(1, 0, 0), 5);
    assert_eq!(at(0, 0, 0), 1);
}

#[test]
fn json_round_trips_generated_maps() {
    for kind in ["arena", "building", "village"] {
        let m = generate(kind, 21, 3);
        let s = m.to_json();
        let back = GameMap::from_json(&s).unwrap();
        assert_eq!(back.tiles(), m.tiles());
        assert_eq!((back.width, back.height, back.floors), (m.width, m.height, m.floors));
        assert_eq!(back.to_json(), s);
        // Spawn zones are derived for parsed maps and still valid.
        let pts = back.spawn_points(0, 8);
        assert!(pts.iter().all(|q| back.is_walkable(q.x as i32, q.y as i32, 0)));
    }
}

#[test]
fn tile_u8_values_match_contract() {
    for v in 0..=7u8 {
        assert_eq!(Tile::from_u8(v).unwrap().as_u8(), v);
    }
    assert_eq!(Tile::from_u8(8), None);
    let m = ascii(&[&[" .#+/=UD"]]);
    let bytes = m.tile_bytes();
    assert_eq!(bytes, [0, 1, 2, 3, 4, 5, 6, 7]);
}

#[test]
fn from_json_rejects_bad_input() {
    assert!(matches!(GameMap::from_json("nope"), Err(MapError::InvalidJson(_))));
    let short = r#"{"width":2,"height":2,"floors":1,"tiles":[1,1,1]}"#;
    assert!(matches!(GameMap::from_json(short), Err(MapError::InvalidJson(_))));
    let zero = r#"{"width":0,"height":2,"floors":1,"tiles":[]}"#;
    assert!(matches!(GameMap::from_json(zero), Err(MapError::InvalidJson(_))));
    let bad = r#"{"width":2,"height":1,"floors":1,"tiles":[1,9]}"#;
    assert_eq!(GameMap::from_json(bad), Err(MapError::InvalidTile { index: 1, value: 9 }));
}
