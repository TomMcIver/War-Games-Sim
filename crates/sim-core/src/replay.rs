//! OWNER: infra-agent. Compact replay frames + JSON writer/reader, consumed by
//! `viewer/` (static web page). Format: docs/INTERFACE.md "Replay JSON".
//!
//! A replay = version + config + seed + map + per-tick frames. Floats are
//! rounded to 3 decimals on write to keep files small; non-finite floats are
//! written as `0`.
//!
//! ```no_run
//! use sim_core::{replay::ReplayWriter, Action, Env, EnvConfig};
//! let seed = 42;
//! let mut env = Env::new(EnvConfig::default(), seed);
//! let mut writer = ReplayWriter::for_env(&env, seed);
//! writer.record(&env.world); // initial state
//! while !env.step(&vec![Action::default(); env.n_units()]).done {
//!     writer.record(&env.world); // once per tick, after stepping
//! }
//! writer.record(&env.world);
//! writer.finish().write_to_file("runs/replays/ep0.json").unwrap();
//! ```
//!
//! Known limitation (format v1): the map is stored once, so doors toggled
//! during the episode are drawn in their initial state.

use std::fmt;
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;

use serde::{Deserialize, Serialize, Serializer};
use serde_json::{json, Value};

use crate::env::{Env, EnvConfig};
use crate::map::GameMap;
use crate::world::{ShotEvent, Unit, World};

/// Replay format version written by this crate and accepted by `from_json`.
pub const REPLAY_VERSION: u32 = 1;

/// Decimal places kept for floats in replay JSON.
const FLOAT_SCALE: f64 = 1000.0;

/// Round to 3 decimals in f64 so the JSON text stays short (`1.234`, not
/// `1.2339999675750732`).
fn round3(v: f32) -> f64 {
    if v.is_finite() {
        (f64::from(v) * FLOAT_SCALE).round() / FLOAT_SCALE
    } else {
        0.0
    }
}

fn ser_f32<S: Serializer>(v: &f32, s: S) -> Result<S::Ok, S::Error> {
    s.serialize_f64(round3(*v))
}

/// One unit's state in one frame.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct UnitFrame {
    pub id: usize,
    pub team: u8,
    #[serde(serialize_with = "ser_f32")]
    pub x: f32,
    #[serde(serialize_with = "ser_f32")]
    pub y: f32,
    pub floor: u8,
    /// Facing angle in radians.
    #[serde(serialize_with = "ser_f32")]
    pub yaw: f32,
    #[serde(serialize_with = "ser_f32")]
    pub hp: f32,
    pub alive: bool,
}

impl From<&Unit> for UnitFrame {
    fn from(u: &Unit) -> Self {
        Self {
            id: u.id,
            team: u.team,
            x: u.pos.x,
            y: u.pos.y,
            floor: u.pos.floor,
            yaw: u.yaw,
            hp: u.hp,
            alive: u.alive,
        }
    }
}

/// A shot fired during this frame's tick. `(x, y, floor)` is where it ended;
/// the viewer draws a tracer from the shooter's position to that point.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ShotFrame {
    pub shooter: usize,
    #[serde(serialize_with = "ser_f32")]
    pub x: f32,
    #[serde(serialize_with = "ser_f32")]
    pub y: f32,
    pub floor: u8,
    /// Unit that was struck, `null` for a miss.
    #[serde(default)]
    pub hit: Option<usize>,
}

impl From<&ShotEvent> for ShotFrame {
    fn from(s: &ShotEvent) -> Self {
        Self {
            shooter: s.shooter,
            x: s.x,
            y: s.y,
            floor: s.floor,
            hit: s.hit,
        }
    }
}

/// Snapshot of the world after one tick.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Frame {
    pub tick: u64,
    pub units: Vec<UnitFrame>,
    #[serde(default)]
    pub shots: Vec<ShotFrame>,
}

impl Frame {
    /// Capture units and the shots fired during the most recent tick.
    pub fn from_world(world: &World) -> Self {
        Self {
            tick: world.tick,
            units: world.units.iter().map(UnitFrame::from).collect(),
            shots: world.shots.iter().map(ShotFrame::from).collect(),
        }
    }
}

/// A whole recorded episode. `config` and `map` are opaque JSON so the replay
/// does not depend on how other modules serialise themselves; `map` must follow
/// docs/INTERFACE.md "Map JSON" for the viewer to draw it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Replay {
    pub version: u32,
    pub config: Value,
    pub seed: u64,
    pub map: Value,
    pub frames: Vec<Frame>,
}

impl Replay {
    /// Compact JSON (no whitespace).
    pub fn to_json(&self) -> Result<String, ReplayError> {
        Ok(serde_json::to_string(self)?)
    }

    /// Parse and check the version.
    pub fn from_json(s: &str) -> Result<Self, ReplayError> {
        let replay: Replay = serde_json::from_str(s)?;
        replay.check_version()
    }

    pub fn write_to_file(&self, path: impl AsRef<Path>) -> Result<(), ReplayError> {
        let mut out = BufWriter::new(File::create(path)?);
        serde_json::to_writer(&mut out, self)?;
        out.flush()?;
        Ok(())
    }

    pub fn read_from_file(path: impl AsRef<Path>) -> Result<Self, ReplayError> {
        let replay: Replay = serde_json::from_reader(BufReader::new(File::open(path)?))?;
        replay.check_version()
    }

    fn check_version(self) -> Result<Self, ReplayError> {
        if self.version == REPLAY_VERSION {
            Ok(self)
        } else {
            Err(ReplayError::Version {
                found: self.version,
            })
        }
    }
}

/// Accumulates frames for one episode.
#[derive(Clone, Debug)]
pub struct ReplayWriter {
    replay: Replay,
}

impl ReplayWriter {
    /// `config_json`: the env config as JSON; `map_json`: Map JSON (see
    /// [`map_json`] or `GameMap::to_json`).
    pub fn new(config_json: Value, seed: u64, map_json: Value) -> Self {
        Self {
            replay: Replay {
                version: REPLAY_VERSION,
                config: config_json,
                seed,
                map: map_json,
                frames: Vec::new(),
            },
        }
    }

    pub fn push(&mut self, frame: Frame) {
        self.replay.frames.push(frame);
    }

    /// Shorthand for `push(Frame::from_world(world))`.
    pub fn record(&mut self, world: &World) {
        self.push(Frame::from_world(world));
    }

    pub fn len(&self) -> usize {
        self.replay.frames.len()
    }

    pub fn is_empty(&self) -> bool {
        self.replay.frames.is_empty()
    }

    pub fn to_json(&self) -> Result<String, ReplayError> {
        self.replay.to_json()
    }

    pub fn write_to_file(&self, path: impl AsRef<Path>) -> Result<(), ReplayError> {
        self.replay.write_to_file(path)
    }

    pub fn finish(self) -> Replay {
        self.replay
    }
}

/// Map JSON (docs/INTERFACE.md) as a `Value`, same content as
/// `GameMap::to_json`: `{"width","height","floors","tiles"}` with
/// `tiles[(f*H + y)*W + x]` = Tile u8.
pub fn map_json(map: &GameMap) -> Value {
    json!({
        "width": map.width,
        "height": map.height,
        "floors": map.floors,
        "tiles": map.tile_bytes(),
    })
}

/// The parts of `EnvConfig` the viewer and analysis tools care about.
/// (`EnvConfig` itself is not `Serialize`.)
pub fn config_json(cfg: &EnvConfig) -> Value {
    json!({
        "team_size": cfg.team_size,
        "max_ticks": cfg.max_ticks,
        "tick_seconds": cfg.tick_seconds,
        "map_kind": cfg.map_kind,
        "max_hp": cfg.unit.max_hp,
    })
}

impl ReplayWriter {
    /// Writer for an episode of `env`, taking config and map from it. Call
    /// right after `Env::new`/`reset` and `record` the initial world first.
    pub fn for_env(env: &Env, seed: u64) -> Self {
        Self::new(config_json(&env.cfg), seed, map_json(&env.world.map))
    }
}

#[derive(Debug)]
pub enum ReplayError {
    Io(std::io::Error),
    Json(serde_json::Error),
    /// The file was written by an incompatible format version.
    Version {
        found: u32,
    },
}

impl fmt::Display for ReplayError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "replay I/O error: {e}"),
            Self::Json(e) => write!(f, "replay JSON error: {e}"),
            Self::Version { found } => {
                write!(
                    f,
                    "unsupported replay version {found} (expected {REPLAY_VERSION})"
                )
            }
        }
    }
}

impl std::error::Error for ReplayError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::Json(e) => Some(e),
            Self::Version { .. } => None,
        }
    }
}

impl From<std::io::Error> for ReplayError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<serde_json::Error> for ReplayError {
    fn from(e: serde_json::Error) -> Self {
        Self::Json(e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::map::Tile;
    use crate::types::{Action, Pos};

    fn world() -> World {
        World {
            map: GameMap::filled(6, 4, 1, Tile::Floor),
            units: vec![],
            tick: 7,
            shots: vec![ShotEvent {
                shooter: 0,
                x: 4.4,
                y: 2.5,
                floor: 0,
                hit: Some(1),
            }],
        }
    }

    fn unit_frame(id: usize, team: u8, alive: bool) -> UnitFrame {
        UnitFrame {
            id,
            team,
            x: 1.5 + id as f32,
            y: 2.25,
            floor: 0,
            yaw: -0.5,
            hp: if alive { 100.0 } else { 0.0 },
            alive,
        }
    }

    fn sample_replay() -> Replay {
        let w = world();
        let mut writer = ReplayWriter::new(
            json!({"team_size": 1, "map_kind": "arena"}),
            99,
            map_json(&w.map),
        );
        writer.push(Frame {
            units: vec![unit_frame(0, 0, true), unit_frame(1, 1, false)],
            ..Frame::from_world(&w)
        });
        writer.push(Frame {
            tick: 8,
            units: vec![],
            shots: vec![],
        });
        writer.finish()
    }

    #[test]
    fn frame_from_world_copies_tick_and_shots() {
        let f = Frame::from_world(&world());
        assert_eq!(f.tick, 7);
        assert!(f.units.is_empty());
        assert_eq!(
            f.shots,
            vec![ShotFrame {
                shooter: 0,
                x: 4.4,
                y: 2.5,
                floor: 0,
                hit: Some(1)
            }]
        );
    }

    #[test]
    fn unit_frame_copies_unit_state() {
        let mut w = world();
        let mut u = Unit::new(
            3,
            1,
            Pos {
                x: 2.5,
                y: 1.25,
                floor: 1,
            },
        );
        u.yaw = -1.0;
        u.hp = 40.0;
        w.units.push(u);
        let f = Frame::from_world(&w);
        assert_eq!(
            f.units,
            vec![UnitFrame {
                id: 3,
                team: 1,
                x: 2.5,
                y: 1.25,
                floor: 1,
                yaw: -1.0,
                hp: 40.0,
                alive: true
            }]
        );
    }

    fn record_episode(seed: u64, max_ticks: u32) -> Replay {
        let cfg = EnvConfig {
            max_ticks,
            ..EnvConfig::default()
        };
        let mut env = Env::new(cfg, seed);
        let mut writer = ReplayWriter::for_env(&env, seed);
        writer.record(&env.world);
        // Everyone turns and fires every tick: exercises shots and kills
        // without any tactics (this is a fixture, not a policy).
        let n = env.n_units();
        let act = Action::from_slice(&[1, 0, 1, 1, 0]);
        loop {
            let done = env.step(&vec![act; n]).done;
            writer.record(&env.world);
            if done {
                break;
            }
        }
        writer.finish()
    }

    #[test]
    fn records_a_real_episode_deterministically() {
        let r = record_episode(5, 120);
        assert_eq!(r.config["team_size"], 2);
        check_map(&r.map);
        assert!(r.frames.len() > 1 && r.frames.len() <= 121);
        assert!(r.frames.windows(2).all(|p| p[1].tick == p[0].tick + 1));
        let back = Replay::from_json(&r.to_json().unwrap()).unwrap();
        assert_eq!(back.frames.len(), r.frames.len());
        assert_eq!(
            record_episode(5, 120).to_json().unwrap(),
            r.to_json().unwrap(),
            "same seed + actions must give the same replay"
        );
    }

    /// Dump an engine replay for eyeballing in the viewer:
    /// `bash infra/run.sh env WARSIM_REPLAY_OUT=/work/target/demo.json
    ///  cargo test -p sim-core write_engine_replay -- --ignored`
    #[test]
    #[ignore = "writes a file; run on demand"]
    fn write_engine_replay() {
        let out = std::env::var("WARSIM_REPLAY_OUT").expect("set WARSIM_REPLAY_OUT");
        record_episode(3, 600).write_to_file(out).unwrap();
    }

    #[test]
    fn writer_accumulates_frames() {
        let mut w = ReplayWriter::new(Value::Null, 1, Value::Null);
        assert!(w.is_empty());
        w.record(&world());
        w.record(&world());
        assert_eq!(w.len(), 2);
        assert_eq!(w.finish().frames.len(), 2);
    }

    #[test]
    fn json_round_trip_preserves_replay() {
        let r = sample_replay();
        let back = Replay::from_json(&r.to_json().unwrap()).unwrap();
        assert_eq!(back, r);
    }

    #[test]
    fn json_has_documented_top_level_shape() {
        let v: Value = serde_json::from_str(&sample_replay().to_json().unwrap()).unwrap();
        assert_eq!(v["version"], 1);
        assert_eq!(v["seed"], 99);
        assert_eq!(v["config"]["map_kind"], "arena");
        let u = &v["frames"][0]["units"][0];
        for key in ["id", "team", "x", "y", "floor", "yaw", "hp", "alive"] {
            assert!(u.get(key).is_some(), "unit frame missing {key}");
        }
        let s = &v["frames"][0]["shots"][0];
        assert_eq!(s["hit"], 1);
        assert_eq!(v["frames"][1]["shots"], json!([]));
    }

    #[test]
    fn floats_are_rounded_to_three_decimals() {
        let mut f = sample_replay().frames[0].clone();
        f.units[0].x = 1.234_567;
        f.units[0].yaw = std::f32::consts::PI;
        f.units[0].hp = f32::NAN;
        let s = serde_json::to_string(&f).unwrap();
        assert!(s.contains("\"x\":1.235"), "{s}");
        assert!(s.contains("\"yaw\":3.142"), "{s}");
        assert!(s.contains("\"hp\":0.0"), "{s}");
        // Re-serialising a parsed frame is stable (no float drift).
        let back: Frame = serde_json::from_str(&s).unwrap();
        assert_eq!(serde_json::to_string(&back).unwrap(), s);
    }

    #[test]
    fn missing_shots_and_hit_default() {
        let f: Frame =
            serde_json::from_str(r#"{"tick":3,"units":[]}"#).expect("shots should default");
        assert!(f.shots.is_empty());
        let s: ShotFrame = serde_json::from_str(r#"{"shooter":2,"x":1,"y":2,"floor":0}"#).unwrap();
        assert_eq!(s.hit, None);
    }

    #[test]
    fn rejects_unknown_version_and_bad_json() {
        let mut v: Value = serde_json::from_str(&sample_replay().to_json().unwrap()).unwrap();
        v["version"] = json!(2);
        let err = Replay::from_json(&v.to_string()).unwrap_err();
        assert!(matches!(err, ReplayError::Version { found: 2 }), "{err}");
        assert!(matches!(Replay::from_json("{"), Err(ReplayError::Json(_))));
    }

    #[test]
    fn file_round_trip() {
        let r = sample_replay();
        let path = std::env::temp_dir().join(format!("warsim-replay-{}.json", std::process::id()));
        r.write_to_file(&path).unwrap();
        let back = Replay::read_from_file(&path);
        std::fs::remove_file(&path).unwrap();
        assert_eq!(back.unwrap(), r);
        assert!(matches!(
            Replay::read_from_file(&path),
            Err(ReplayError::Io(_))
        ));
    }

    #[test]
    fn map_json_matches_interface_layout() {
        let mut m = GameMap::filled(3, 2, 2, Tile::Floor);
        m.set_tile(2, 1, 1, Tile::StairsDown);
        m.set_tile(0, 1, 0, Tile::Window);
        let v = map_json(&m);
        assert_eq!(
            (v["width"].clone(), v["height"].clone(), v["floors"].clone()),
            (json!(3), json!(2), json!(2))
        );
        let tiles = v["tiles"].as_array().unwrap();
        assert_eq!(tiles.len(), 12);
        assert_eq!(tiles[(2 + 1) * 3 + 2], 7); // (f=1, y=1, x=2)
        assert_eq!(tiles[3], 5); // (f=0, y=1, x=0)
    }

    /// Validates a Map JSON value the way the viewer consumes it.
    fn check_map(map: &Value) -> (usize, usize, usize) {
        let dim = |k: &str| map[k].as_u64().unwrap_or_else(|| panic!("map.{k}")) as usize;
        let (w, h, f) = (dim("width"), dim("height"), dim("floors"));
        let tiles = map["tiles"].as_array().expect("map.tiles");
        assert_eq!(tiles.len(), w * h * f, "tiles length");
        assert!(
            tiles.iter().all(|t| t.as_u64().is_some_and(|t| t <= 7)),
            "tile values"
        );
        (w, h, f)
    }

    /// The bundled viewer sample must stay loadable by this crate.
    #[test]
    fn bundled_viewer_sample_is_valid() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../viewer/sample_replay.json");
        let r = Replay::read_from_file(&path).expect("viewer/sample_replay.json");
        let (w, h, floors) = check_map(&r.map);
        assert!(floors >= 2, "sample should show multiple floors");
        assert!(!r.frames.is_empty());
        let n = r.frames[0].units.len();
        let mut saw_shot = false;
        for pair in r.frames.windows(2) {
            assert!(pair[1].tick > pair[0].tick, "ticks must increase");
        }
        for f in &r.frames {
            assert_eq!(f.units.len(), n, "unit count must be constant");
            for (i, u) in f.units.iter().enumerate() {
                assert_eq!(u.id, i);
                assert!(u.x >= 0.0 && u.x <= w as f32 && u.y >= 0.0 && u.y <= h as f32);
                assert!((u.floor as usize) < floors);
            }
            for s in &f.shots {
                saw_shot = true;
                assert!(s.shooter < n && s.hit.is_none_or(|h| h < n));
            }
        }
        assert!(saw_shot, "sample should contain shots");
    }
}
