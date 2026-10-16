//! Python bindings for War-Games-Sim, exposed as the module `warsim`.
//! The API is fixed by docs/INTERFACE.md ("Python API"); see README.md here.

mod config;
mod convert;
mod py_vec_env;

use pyo3::prelude::*;
use pyo3::types::PyTuple;
use sim_core::{
    ACTION_DIMS, ACTION_HEADS, LOCAL_VIEW, MAP_CHANNELS, MAX_TEAM_SIZE, MAX_UNITS, STATE_FEATS,
    TOKEN_FEATS,
};

#[pymodule]
fn warsim(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("TOKEN_FEATS", TOKEN_FEATS)?;
    m.add("MAP_CHANNELS", MAP_CHANNELS)?;
    m.add("LOCAL_VIEW", LOCAL_VIEW)?;
    m.add("STATE_FEATS", STATE_FEATS)?;
    m.add("MAX_TEAM_SIZE", MAX_TEAM_SIZE)?;
    m.add("MAX_UNITS", MAX_UNITS)?;
    m.add("ACTION_HEADS", ACTION_HEADS)?;
    // Tuples (immutable) so callers cannot accidentally mutate module constants.
    m.add("ACTION_DIMS", PyTuple::new(m.py(), ACTION_DIMS)?)?;
    m.add("MAP_KINDS", PyTuple::new(m.py(), config::MAP_KINDS)?)?;
    m.add_class::<py_vec_env::PyVecEnv>()?;
    Ok(())
}
