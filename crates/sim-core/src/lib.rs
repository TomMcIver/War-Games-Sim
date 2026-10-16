//! War-Games-Sim core: a deterministic, headless, fast 2.5D tactical sim
//! built to be a reinforcement-learning environment.
//!
//! The AI gets NO scripted tactics. It sees observations, emits low-level
//! actions, and learns from reward. See docs/INTERFACE.md for the contract.

pub mod env;
pub mod map;
pub mod obs;
pub mod replay;
pub mod types;
pub mod vec_env;
pub mod world;

pub use env::{Env, EnvConfig, StepResult};
pub use types::*;
pub use vec_env::VecEnv;
