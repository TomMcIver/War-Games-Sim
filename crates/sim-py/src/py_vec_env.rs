//! `warsim.VecEnv`: thin, safe wrapper around `sim_core::VecEnv`.
//!
//! - The GIL is released while sim-core runs (`reset`/`step`).
//! - Inputs are validated before reaching sim-core; sim-core panics are caught
//!   (when built with `panic = "unwind"`, see pyproject.toml) and re-raised as
//!   `RuntimeError`, never unwinding across the FFI boundary.
//! - Every returned array is a fresh copy of the Rust buffer (never reused or
//!   aliased between calls), so callers may store them freely.

use std::any::Any;
use std::panic::{catch_unwind, AssertUnwindSafe};

use numpy::{PyArray1, PyArrayDyn};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use sim_core::{
    VecEnv, VecEnvBuffers, ACTION_DIMS, ACTION_HEADS, LOCAL_VIEW, MAP_CHANNELS, STATE_FEATS,
    TOKEN_FEATS,
};

use crate::config::{validate, RawArgs, Settings};
use crate::convert::{read_actions, to_bool_numpy, to_numpy};

#[pyclass(name = "VecEnv", module = "warsim")]
pub struct PyVecEnv {
    /// `None` once `close()` has been called.
    inner: Option<VecEnv>,
    /// Dedicated rayon pool when `num_threads` is given, else rayon's global pool.
    pool: Option<rayon::ThreadPool>,
    n_envs: usize,
    team_size: usize,
    n_units: usize,
    map_kind: String,
    max_ticks: u32,
    num_threads: Option<usize>,
    /// Preallocated action buffer (len `n_envs * n_units * ACTION_HEADS`):
    /// actions are copied here so the GIL can be released safely.
    actions: Vec<u8>,
    /// True until a successful `reset()` (and again after a sim-core panic).
    needs_reset: bool,
}

#[pymethods]
impl PyVecEnv {
    #[new]
    #[pyo3(signature = (
        n_envs, team_size=2, map_kind="arena", max_ticks=600, seed=0,
        tick_seconds=0.1, num_threads=None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        n_envs: i128,
        team_size: i128,
        map_kind: &str,
        max_ticks: i128,
        seed: i128,
        tick_seconds: f64,
        num_threads: Option<i128>,
    ) -> PyResult<Self> {
        let raw = RawArgs {
            n_envs,
            team_size,
            map_kind,
            max_ticks,
            seed,
            tick_seconds,
            num_threads,
        };
        let settings = validate(&raw).map_err(PyValueError::new_err)?;
        Self::build(py, settings)
    }

    /// Reset every env. Returns the observation dict.
    fn reset<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let pool = self.pool.as_ref();
        let inner = self.inner.as_mut().ok_or_else(closed_error)?;
        let outcome = py.detach(|| {
            run_guarded(pool, || {
                inner.reset();
            })
        });
        self.needs_reset = outcome.is_err();
        outcome.map_err(|msg| core_panic_error("reset", &msg))?;
        let buffers = self.buffers()?;
        self.obs_dict(py, buffers)
    }

    /// Step every env with `actions` (`uint8 [n_envs, n_units, 5]`).
    /// Returns `(obs, rewards, dones, infos)`; done envs are auto-reset.
    #[allow(clippy::type_complexity)]
    fn step<'py>(
        &mut self,
        py: Python<'py>,
        actions: &Bound<'py, PyAny>,
    ) -> PyResult<(
        Bound<'py, PyDict>,
        Bound<'py, PyArrayDyn<f32>>,
        Bound<'py, PyArray1<bool>>,
        Bound<'py, PyDict>,
    )> {
        if self.inner.is_none() {
            return Err(closed_error());
        }
        read_actions(actions, self.n_envs, self.n_units, &mut self.actions)?;
        if self.needs_reset {
            return Err(PyRuntimeError::new_err(
                "VecEnv.step() called before reset() (or after a failed call); call reset() first",
            ));
        }
        let pool = self.pool.as_ref();
        let acts = self.actions.as_slice();
        let inner = self.inner.as_mut().ok_or_else(closed_error)?;
        let outcome = py.detach(|| {
            run_guarded(pool, || {
                inner.step(acts);
            })
        });
        self.needs_reset = outcome.is_err();
        outcome.map_err(|msg| core_panic_error("step", &msg))?;

        let buffers = self.buffers()?;
        let (e, u) = (self.n_envs, self.n_units);
        let obs = self.obs_dict(py, buffers)?;
        let rewards = to_numpy(py, "rewards", &buffers.rewards, &[e, u])?;
        let dones = to_bool_numpy(py, "dones", &buffers.dones, e)?;
        let infos = PyDict::new(py);
        infos.set_item("winner", to_numpy(py, "winners", &buffers.winners, &[e])?)?;
        infos.set_item(
            "episode_len",
            to_numpy(py, "episode_len", &buffers.episode_len, &[e])?,
        )?;
        Ok((obs, rewards, dones, infos))
    }

    /// Release the sim (idempotent). Further `reset`/`step` calls raise.
    fn close(&mut self) {
        self.inner = None;
        self.pool = None;
        self.actions = Vec::new();
    }

    fn __enter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    fn __exit__(
        &mut self,
        _exc_type: &Bound<'_, PyAny>,
        _exc: &Bound<'_, PyAny>,
        _tb: &Bound<'_, PyAny>,
    ) -> bool {
        self.close();
        false
    }

    fn __repr__(&self) -> String {
        format!(
            "VecEnv(n_envs={}, team_size={}, map_kind={:?}, max_ticks={}, num_threads={}{})",
            self.n_envs,
            self.team_size,
            self.map_kind,
            self.max_ticks,
            self.num_threads
                .map_or_else(|| "None".to_owned(), |n| n.to_string()),
            if self.inner.is_none() { ", closed" } else { "" },
        )
    }

    #[getter]
    fn n_envs(&self) -> usize {
        self.n_envs
    }

    /// Units per env: `2 * team_size`.
    #[getter]
    fn n_units(&self) -> usize {
        self.n_units
    }

    #[getter]
    fn team_size(&self) -> usize {
        self.team_size
    }

    /// Multi-discrete action dims per head: `[3, 3, 3, 2, 2]`.
    #[getter]
    fn action_dims(&self) -> Vec<usize> {
        ACTION_DIMS.to_vec()
    }

    #[getter]
    fn closed(&self) -> bool {
        self.inner.is_none()
    }
}

impl PyVecEnv {
    fn build(py: Python<'_>, s: Settings) -> PyResult<Self> {
        let pool = s
            .num_threads
            .map(|n| {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(n)
                    .thread_name(|i| format!("warsim-{i}"))
                    .build()
                    .map_err(|e| PyRuntimeError::new_err(format!("cannot start thread pool: {e}")))
            })
            .transpose()?;
        let (n_envs, seed) = (s.n_envs, s.seed);
        let cfg = s.cfg.clone();
        let inner = py
            .detach(|| run_guarded(pool.as_ref(), move || VecEnv::new(n_envs, cfg, seed)))
            .map_err(|msg| core_panic_error("VecEnv::new", &msg))?;
        let n_units = inner.n_units();
        if n_units != 2 * s.cfg.team_size {
            return Err(PyRuntimeError::new_err(format!(
                "internal error: sim-core reports n_units={n_units}, expected {}",
                2 * s.cfg.team_size
            )));
        }
        Ok(Self {
            inner: Some(inner),
            pool,
            n_envs,
            team_size: s.cfg.team_size,
            n_units,
            map_kind: s.cfg.map_kind,
            max_ticks: s.cfg.max_ticks,
            num_threads: s.num_threads,
            actions: vec![0; n_envs * n_units * ACTION_HEADS],
            needs_reset: true,
        })
    }

    fn buffers(&self) -> PyResult<&VecEnvBuffers> {
        self.inner
            .as_ref()
            .map(VecEnv::buffers)
            .ok_or_else(closed_error)
    }

    fn obs_dict<'py>(&self, py: Python<'py>, b: &VecEnvBuffers) -> PyResult<Bound<'py, PyDict>> {
        let (e, u) = (self.n_envs, self.n_units);
        let obs = PyDict::new(py);
        obs.set_item(
            "tokens",
            to_numpy(py, "tokens", &b.tokens, &[e, u, u, TOKEN_FEATS])?,
        )?;
        obs.set_item(
            "token_mask",
            to_numpy(py, "token_mask", &b.token_mask, &[e, u, u])?,
        )?;
        obs.set_item(
            "local_map",
            to_numpy(
                py,
                "local_map",
                &b.local_map,
                &[e, u, MAP_CHANNELS, LOCAL_VIEW, LOCAL_VIEW],
            )?,
        )?;
        obs.set_item(
            "state",
            to_numpy(py, "state", &b.state, &[e, u * STATE_FEATS])?,
        )?;
        obs.set_item("alive", to_numpy(py, "alive", &b.alive, &[e, u])?)?;
        Ok(obs)
    }
}

/// Run `f` (on the dedicated pool if any), converting a panic into its message.
fn run_guarded<T: Send>(
    pool: Option<&rayon::ThreadPool>,
    f: impl FnOnce() -> T + Send,
) -> Result<T, String> {
    // AssertUnwindSafe: after a panic the env is flagged `needs_reset` and never
    // observed in its possibly-inconsistent state.
    catch_unwind(AssertUnwindSafe(move || match pool {
        Some(p) => p.install(f),
        None => f(),
    }))
    .map_err(panic_message)
}

fn panic_message(payload: Box<dyn Any + Send>) -> String {
    payload
        .downcast_ref::<&str>()
        .map(|s| (*s).to_owned())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "<non-string panic payload>".to_owned())
}

fn core_panic_error(what: &str, msg: &str) -> PyErr {
    PyRuntimeError::new_err(format!(
        "sim-core panicked in {what}: {msg} (env state is invalid; call reset())"
    ))
}

fn closed_error() -> PyErr {
    PyRuntimeError::new_err("VecEnv is closed")
}
