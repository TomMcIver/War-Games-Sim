//! numpy <-> Rust buffer conversion: copying sim-core output buffers into
//! fresh numpy arrays, and validating/copying the incoming action array.

use numpy::prelude::*;
use numpy::{dtype, Element, PyArray1, PyArrayDyn, PyUntypedArray};
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use sim_core::{ACTION_DIMS, ACTION_HEADS};

const HEAD_NAMES: [&str; ACTION_HEADS] = ["forward", "strafe", "turn", "shoot", "interact"];

/// Copy `src` into a newly allocated C-contiguous numpy array of shape `dims`.
///
/// Exactly one allocation and one memcpy. The returned array owns its memory,
/// so the caller may keep it after later `step()` calls.
pub fn to_numpy<'py, T: Element + Copy>(
    py: Python<'py>,
    name: &str,
    src: &[T],
    dims: &[usize],
) -> PyResult<Bound<'py, PyArrayDyn<T>>> {
    let expected: usize = dims.iter().product();
    if src.len() != expected {
        return Err(PyRuntimeError::new_err(format!(
            "internal error: sim-core buffer `{name}` has {} elements, expected {expected} \
             for shape {dims:?}",
            src.len()
        )));
    }
    // SAFETY: `new` allocates an uninitialised, C-contiguous array of exactly
    // `expected` elements that no other code can observe yet. We fully
    // initialise it through a raw pointer (no reference to uninit memory is
    // created) before handing it to Python. `src` and the new array cannot overlap.
    unsafe {
        let arr = PyArrayDyn::<T>::new(py, dims, false);
        std::ptr::copy_nonoverlapping(src.as_ptr(), arr.data(), expected);
        Ok(arr)
    }
}

/// `dones` is `u8` in Rust; Python gets a `bool` array.
pub fn to_bool_numpy<'py>(
    py: Python<'py>,
    name: &str,
    src: &[u8],
    len: usize,
) -> PyResult<Bound<'py, PyArray1<bool>>> {
    if src.len() != len {
        return Err(PyRuntimeError::new_err(format!(
            "internal error: sim-core buffer `{name}` has {} elements, expected {len}",
            src.len()
        )));
    }
    Ok(PyArray1::from_iter(py, src.iter().map(|&v| v != 0)))
}

/// Validate a Python `actions` object (`numpy.uint8`, shape `[n_envs, n_units, 5]`,
/// C-contiguous, every head within `ACTION_DIMS`) and copy it into `out`.
///
/// `out` must already have length `n_envs * n_units * 5`. On error `out` is
/// left untouched.
pub fn read_actions(
    actions: &Bound<'_, PyAny>,
    n_envs: usize,
    n_units: usize,
    out: &mut [u8],
) -> PyResult<()> {
    let py = actions.py();
    let expected_shape = [n_envs, n_units, ACTION_HEADS];
    let arr = actions.cast::<PyUntypedArray>().map_err(|_| {
        PyTypeError::new_err(format!(
            "actions must be a numpy.ndarray of dtype uint8 and shape {expected_shape:?}, \
             got {}",
            type_name(actions)
        ))
    })?;
    let dt = arr.dtype();
    if !dt.is_equiv_to(&dtype::<u8>(py)) {
        return Err(PyTypeError::new_err(format!(
            "actions must have dtype uint8, got {dt}; convert with actions.astype(np.uint8)"
        )));
    }
    if arr.shape() != expected_shape {
        return Err(PyValueError::new_err(format!(
            "actions must have shape {expected_shape:?} (n_envs, n_units, action heads), \
             got {:?}",
            arr.shape()
        )));
    }
    if !arr.is_c_contiguous() {
        return Err(PyValueError::new_err(
            "actions must be C-contiguous; use np.ascontiguousarray(actions)",
        ));
    }
    let typed = arr.cast::<PyArrayDyn<u8>>()?;
    let readonly = typed
        .try_readonly()
        .map_err(|e| PyValueError::new_err(format!("actions array is not readable: {e}")))?;
    let src = readonly
        .as_slice()
        .map_err(|e| PyValueError::new_err(format!("actions array is not contiguous: {e}")))?;
    check_action_range(src, n_units)?;
    out.copy_from_slice(src);
    Ok(())
}

fn check_action_range(src: &[u8], n_units: usize) -> PyResult<()> {
    let bad = src
        .iter()
        .enumerate()
        .find(|&(i, &v)| usize::from(v) >= ACTION_DIMS[i % ACTION_HEADS]);
    match bad {
        None => Ok(()),
        Some((i, &v)) => {
            let head = i % ACTION_HEADS;
            let unit = (i / ACTION_HEADS) % n_units;
            let env = i / (ACTION_HEADS * n_units);
            Err(PyValueError::new_err(format!(
                "actions[{env}, {unit}, {head}] = {v} is out of range for head '{}' \
                 (valid: 0..{}); ACTION_DIMS = {ACTION_DIMS:?}",
                HEAD_NAMES[head], ACTION_DIMS[head]
            )))
        }
    }
}

fn type_name(obj: &Bound<'_, PyAny>) -> String {
    obj.get_type()
        .name()
        .map(|n| n.to_string())
        .unwrap_or_else(|_| "<unknown type>".to_owned())
}
