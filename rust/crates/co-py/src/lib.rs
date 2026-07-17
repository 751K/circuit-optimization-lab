//! co-py - the `circuitopt_core` PyO3 extension module.
//!
//! Two surfaces live here:
//!
//! * A **C ABI** (`co_bsim4_*`, `#[unsafe(no_mangle)] extern "C"`) that is
//!   signature- and semantics-identical to the historical `host.c` entry points.
//!   `circuitopt/compact_models/bsim4/native.py` loads this compiled module with
//!   `ctypes.CDLL(circuitopt_core.__file__)` and binds these symbols exactly as
//!   it binds the runtime-`cc` library, so the Numba transient bridge picks up
//!   `co_bsim4_eval_vp` as a plain machine-word function pointer.
//! * A **PyO3** surface: build metadata, BSIM4 and OTFT device access, and
//!   GIL-free whole-analysis MNA/LTI/transient entry points. Large inputs are
//!   borrowed from read-only C-contiguous NumPy arrays; waveform and matrix
//!   outputs are returned as NumPy arrays without Python-list materialization.
//!
//! The BSIM4 logic itself lives in `co-bsim4`; this crate is a thin boundary
//! that catches panics so nothing unwinds across the FFI edge.

// The C ABI shims below are `unsafe extern "C"` and forward raw pointers; an
// inner `unsafe {}` per dereference would only add noise. Their contract is the
// documented host.c ABI (valid handle, correctly sized output buffers), so the
// per-function `# Safety` sections would be pure boilerplate.
#![allow(unsafe_op_in_unsafe_fn)]
#![allow(clippy::missing_safety_doc)]

use std::collections::HashMap;
use std::ffi::CString;
use std::os::raw::{c_char, c_double, c_int, c_uint, c_void};
use std::panic::catch_unwind;
use std::sync::Arc;

use numpy::ndarray::{Array1, Array2, Array3};
use numpy::{
    Complex64, IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2,
    PyReadonlyArray3, PyReadonlyArray4,
};
use pyo3::create_exception;
use pyo3::exceptions::{PyKeyError, PyOSError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use co_bsim4::CoBsim4;
use co_core::{bsim_transient, lti, mna, otft, periodic, transient};
use co_pdk::{CompiledPdk as PdkCompiled, NumericCard, PdkError};
use co_spice::{
    ErrorKind as SpiceErrorKind, EvalCtx, LibrarySection, NumericModel, ParamValue,
    ParameterAssignment, ScopeInner, SpiceError, SpiceModelLibrary, Statement, Subcircuit,
    elaborate_library, logical_lines, parse_assignments, parse_spice_library_text,
    parse_spice_number, select_library_sections,
};

// SPICE expression engine exceptions. The hierarchy mirrors the Python
// reference exactly (all are `ValueError` subclasses), so the parity harness
// can match on `type(exc).__name__`:
//   SpiceExpressionError < ValueError
//   UnknownSymbolError   < SpiceExpressionError
//   ParameterCycleError  < SpiceExpressionError
create_exception!(circuitopt_core, SpiceExpressionError, PyValueError);
create_exception!(circuitopt_core, UnknownSymbolError, SpiceExpressionError);
create_exception!(circuitopt_core, ParameterCycleError, SpiceExpressionError);
// Deck-parser / elaborator exceptions (direct `ValueError` subclasses):
//   SpiceSyntaxError      < ValueError
//   SpiceElaborationError < ValueError
create_exception!(circuitopt_core, SpiceSyntaxError, PyValueError);
create_exception!(circuitopt_core, SpiceElaborationError, PyValueError);

const OK: c_int = 0;
/// Vague internal error (ngspice `E_PANIC`); also what we return if a Rust panic
/// is caught at the FFI boundary.
const E_PANIC: c_int = 1;

fn core_error(error: co_core::CoreError) -> PyErr {
    match error {
        co_core::CoreError::Singular { .. } => PyRuntimeError::new_err(error.to_string()),
        co_core::CoreError::InvalidTopology { .. } | co_core::CoreError::InvalidInput { .. } => {
            PyValueError::new_err(error.to_string())
        }
    }
}

// ---------------------------------------------------------------------------
// C ABI — identical to host.c. Every body is wrapped in `catch_unwind` so a
// panic becomes an ordinary status code / null pointer instead of unwinding
// into C (which is undefined behaviour).
// ---------------------------------------------------------------------------

#[unsafe(no_mangle)]
pub extern "C" fn co_bsim4_abi_version() -> c_uint {
    co_bsim4::abi_version() as c_uint
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn co_bsim4_create(polarity: c_int, temperature_k: c_double) -> *mut c_void {
    catch_unwind(|| co_bsim4::create(polarity, temperature_k) as *mut c_void)
        .unwrap_or(std::ptr::null_mut())
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn co_bsim4_destroy(device: *mut c_void) {
    let _ = catch_unwind(|| co_bsim4::destroy(device as *mut CoBsim4));
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn co_bsim4_set_model(
    device: *mut c_void,
    name: *const c_char,
    value: c_double,
) -> c_int {
    catch_unwind(|| co_bsim4::set_model(device as *mut CoBsim4, name, value)).unwrap_or(E_PANIC)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn co_bsim4_set_instance(
    device: *mut c_void,
    name: *const c_char,
    value: c_double,
) -> c_int {
    catch_unwind(|| co_bsim4::set_instance(device as *mut CoBsim4, name, value)).unwrap_or(E_PANIC)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn co_bsim4_setup(device: *mut c_void) -> c_int {
    catch_unwind(|| co_bsim4::setup(device as *mut CoBsim4)).unwrap_or(E_PANIC)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn co_bsim4_node_count(device: *const c_void) -> c_int {
    catch_unwind(|| co_bsim4::node_count(device as *const CoBsim4)).unwrap_or(0)
}

#[allow(clippy::too_many_arguments)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn co_bsim4_dc(
    device: *mut c_void,
    terminals: *const c_double,
    currents: *mut c_double,
    conductance: *mut c_double,
    charges: *mut c_double,
    capacitance: *mut c_double,
    op: *mut c_double,
) -> c_int {
    catch_unwind(|| {
        co_bsim4::dc(
            device as *mut CoBsim4,
            terminals,
            currents,
            conductance,
            charges,
            capacitance,
            op,
        )
    })
    .unwrap_or(E_PANIC)
}

#[allow(clippy::too_many_arguments)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn co_bsim4_eval(
    device: *mut c_void,
    terminals: *const c_double,
    currents: *mut c_double,
    conductance: *mut c_double,
    charges: *mut c_double,
    capacitance: *mut c_double,
    op: *mut c_double,
) -> c_int {
    catch_unwind(|| {
        co_bsim4::eval(
            device as *mut CoBsim4,
            terminals,
            currents,
            conductance,
            charges,
            capacitance,
            op,
        )
    })
    .unwrap_or(E_PANIC)
}

/// All-`void*` evaluation ABI (host.c `co_bsim4_eval_vp`) — the entry the Numba
/// bridge takes the address of and calls as a machine-word function pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn co_bsim4_eval_vp(
    device: *mut c_void,
    terminals: *mut c_void,
    currents: *mut c_void,
    conductance: *mut c_void,
    charges: *mut c_void,
    capacitance: *mut c_void,
) -> c_int {
    catch_unwind(|| {
        co_bsim4::eval_vp(
            device as *mut CoBsim4,
            terminals as *const c_double,
            currents as *mut c_double,
            conductance as *mut c_double,
            charges as *mut c_double,
            capacitance as *mut c_double,
        )
    })
    .unwrap_or(E_PANIC)
}

#[allow(clippy::too_many_arguments)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn co_bsim4_eval_batch(
    devices: *const *mut c_void,
    count: usize,
    terminals: *const c_double,
    currents: *mut c_double,
    conductance: *mut c_double,
    charges: *mut c_double,
    capacitance: *mut c_double,
    statuses: *mut c_int,
) -> c_int {
    catch_unwind(|| {
        co_bsim4::eval_batch(
            devices as *const *mut CoBsim4,
            count,
            terminals,
            currents,
            conductance,
            charges,
            capacitance,
            statuses,
        )
    })
    .unwrap_or(E_PANIC)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn co_bsim4_noise(
    device: *mut c_void,
    frequency_hz: c_double,
    total_real: *mut c_double,
    total_imag: *mut c_double,
    flicker_real: *mut c_double,
    flicker_imag: *mut c_double,
) -> c_int {
    catch_unwind(|| {
        co_bsim4::noise(
            device as *mut CoBsim4,
            frequency_hz,
            total_real,
            total_imag,
            flicker_real,
            flicker_imag,
        )
    })
    .unwrap_or(E_PANIC)
}

#[allow(clippy::too_many_arguments)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn co_bsim4_noise_batch(
    devices: *const *mut c_void,
    count: usize,
    frequencies: *const c_double,
    frequency_count: usize,
    total_real: *mut c_double,
    total_imag: *mut c_double,
    flicker_real: *mut c_double,
    flicker_imag: *mut c_double,
    statuses: *mut c_int,
) -> c_int {
    catch_unwind(|| {
        co_bsim4::noise_batch(
            devices as *const *mut CoBsim4,
            count,
            frequencies,
            frequency_count,
            total_real,
            total_imag,
            flicker_real,
            flicker_imag,
            statuses,
        )
    })
    .unwrap_or(E_PANIC)
}

// ---------------------------------------------------------------------------
// PyO3 surface
// ---------------------------------------------------------------------------

/// Touch both path-dependency crates so the workspace link is real, not
/// nominal. Returns a deterministic scalar the Python tests can pin.
fn workspace_probe() -> f64 {
    co_core::core_probe(co_bsim4::model_probe(1.0))
}

/// Build/runtime metadata for the compiled core.
#[pyfunction]
fn engine_info(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let info = PyDict::new(py);
    info.set_item("version", env!("CARGO_PKG_VERSION"))?;
    info.set_item(
        "profile",
        if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        },
    )?;
    info.set_item("rustc", env!("CO_RUSTC_VERSION"))?;
    info.set_item("core", co_core::version())?;
    info.set_item("bsim4", co_bsim4::version())?;
    // R2: the BSIM4 core is live — surface its four-terminal ABI version so the
    // Python side can gate on it exactly as it gates the runtime-cc library.
    info.set_item("bsim4_abi", co_bsim4::abi_version())?;
    info.set_item("bsim4_backend", "berkeley-bsim4v5-native")?;
    info.set_item("probe", workspace_probe())?;
    Ok(info)
}

/// Raw address of the `co_bsim4_eval_vp` C ABI entry, for callers that prefer
/// not to reopen the module with `ctypes.CDLL`.
#[pyfunction]
fn bsim4_eval_vp_address() -> usize {
    // Coerce the fn item to a fn pointer first, then to an address.
    let entry: unsafe extern "C" fn(
        *mut c_void,
        *mut c_void,
        *mut c_void,
        *mut c_void,
        *mut c_void,
        *mut c_void,
    ) -> c_int = co_bsim4_eval_vp;
    entry as usize
}

fn otft_params_slice(values: &[f64]) -> PyResult<otft::Params> {
    otft::Params::from_slice(values).ok_or_else(|| {
        PyRuntimeError::new_err(format!(
            "OTFT parameter vector must contain exactly {} values, got {}",
            otft::Params::LEN,
            values.len()
        ))
    })
}

fn otft_params(values: Vec<f64>) -> PyResult<otft::Params> {
    otft_params_slice(&values)
}

/// Low-overhead production wrapper for scalar OTFT model operations.
#[pyclass(frozen)]
struct OtftModel {
    params: otft::Params,
}

#[pymethods]
impl OtftModel {
    #[new]
    fn new(params: Vec<f64>) -> PyResult<Self> {
        Ok(Self {
            params: otft_params(params)?,
        })
    }

    fn eval_currents(&self, vs: f64, vd: f64, vg: f64, vs1: f64, vd1: f64) -> Vec<f64> {
        otft::eval_currents(&self.params, vs, vd, vg, vs1, vd1).to_vec()
    }

    #[pyo3(signature = (vs, vd, vg, x0s, x0d, tol=1e-12, maxit=40))]
    #[allow(clippy::too_many_arguments)]
    fn newton_internal(
        &self,
        vs: f64,
        vd: f64,
        vg: f64,
        x0s: f64,
        x0d: f64,
        tol: f64,
        maxit: usize,
    ) -> (bool, f64, f64) {
        let result = otft::newton_internal_fast(&self.params, vs, vd, vg, x0s, x0d, tol, maxit);
        (result.converged, result.vs1, result.vd1)
    }

    fn capacitance_charges(&self, vs: f64, vd: f64, vg: f64, vs1: f64, vd1: f64) -> Vec<f64> {
        otft::capacitance_charges(&self.params, vs, vd, vg, vs1, vd1).to_vec()
    }

    #[pyo3(signature = (vs, vd, vg, vs1, vd1, need_gm=true, need_gds=true, use_abs=false, hh=1e-3))]
    #[allow(clippy::too_many_arguments)]
    fn terminal_derivatives(
        &self,
        vs: f64,
        vd: f64,
        vg: f64,
        vs1: f64,
        vd1: f64,
        need_gm: bool,
        need_gds: bool,
        use_abs: bool,
        hh: f64,
    ) -> (bool, f64, f64) {
        let jac = otft::residual_pair_jac_internal(&self.params, vs, vd, vg, vs1, vd1);
        let idc0 = jac[1] - (vs1 - vd1) / 0.1;
        otft::terminal_derivatives_from_jac(
            &self.params,
            vs,
            vd,
            vg,
            vs1,
            vd1,
            jac[0],
            jac[1],
            idc0,
            [jac[2], jac[3], jac[4], jac[5]],
            need_gm,
            need_gds,
            use_abs,
            hh,
        )
    }
}

/// Evaluate solved OTFT operating points in one GIL-free batch.
///
/// Each point is `(Vs, Vd, Vg, Vs1, Vd1)`. The returned tuple contains
/// currents `[5]`, charge/capacitance values `[4]`, residual/Jacobian values
/// `[6]`, and analytic `(ok, gm, gds)` records.
#[pyfunction]
#[pyo3(signature = (params, points, need_gm=true, need_gds=true, use_abs=true, hh=1e-3))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn otft_device_batch<'py>(
    py: Python<'py>,
    params: PyReadonlyArray1<'py, f64>,
    points: PyReadonlyArray2<'py, f64>,
    need_gm: bool,
    need_gds: bool,
    use_abs: bool,
    hh: f64,
) -> PyResult<(
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
)> {
    let params = params
        .as_slice()
        .map_err(|_| PyValueError::new_err("params must be a contiguous float64 array"))?;
    let params = otft_params_slice(params)?;
    let point_view = points.as_array();
    let point_shape = point_view.dim();
    if point_shape.1 != 5 {
        return Err(PyValueError::new_err("points must have shape (N, 5)"));
    }
    if !point_view.is_standard_layout() {
        return Err(PyValueError::new_err(
            "points must be a C-contiguous float64 array",
        ));
    }
    let points = points
        .as_slice()
        .map_err(|_| PyValueError::new_err("points must be a C-contiguous float64 array"))?;
    let count = point_shape.0;
    let (currents, charges, jacobians, derivatives) = py.detach(move || {
        let mut currents = Vec::with_capacity(count * 5);
        let mut charges = Vec::with_capacity(count * 4);
        let mut jacobians = Vec::with_capacity(count * 6);
        let mut derivatives = Vec::with_capacity(count * 3);
        for point in points.chunks_exact(5) {
            let [vs, vd, vg, vs1, vd1] = [point[0], point[1], point[2], point[3], point[4]];
            let current = otft::eval_currents(&params, vs, vd, vg, vs1, vd1);
            let charge = otft::capacitance_charges(&params, vs, vd, vg, vs1, vd1);
            let jac = otft::residual_pair_jac_internal(&params, vs, vd, vg, vs1, vd1);
            let idc0 = jac[1] - (vs1 - vd1) / 0.1;
            let derivative = otft::terminal_derivatives_from_jac(
                &params,
                vs,
                vd,
                vg,
                vs1,
                vd1,
                jac[0],
                jac[1],
                idc0,
                [jac[2], jac[3], jac[4], jac[5]],
                need_gm,
                need_gds,
                use_abs,
                hh,
            );
            currents.extend_from_slice(&current);
            charges.extend_from_slice(&charge);
            jacobians.extend_from_slice(&jac);
            derivatives.extend_from_slice(&[f64::from(derivative.0), derivative.1, derivative.2]);
        }
        (currents, charges, jacobians, derivatives)
    });
    let make = |values, width, name: &str| {
        Array2::from_shape_vec((count, width), values)
            .map(|array| array.into_pyarray(py))
            .map_err(|error| PyRuntimeError::new_err(format!("invalid {name} shape: {error}")))
    };
    Ok((
        make(currents, 5, "OTFT currents")?,
        make(charges, 4, "OTFT charges")?,
        make(jacobians, 6, "OTFT Jacobians")?,
        make(derivatives, 3, "OTFT derivatives")?,
    ))
}

/// Evaluate the finite-difference reference terminal derivative path.
#[pyfunction]
#[pyo3(signature = (params, points, need_gm=true, need_gds=true, use_abs=true, hh=1e-3, hx=1e-6))]
#[allow(clippy::too_many_arguments)]
fn otft_terminal_derivatives_batch<'py>(
    py: Python<'py>,
    params: PyReadonlyArray1<'py, f64>,
    points: PyReadonlyArray2<'py, f64>,
    need_gm: bool,
    need_gds: bool,
    use_abs: bool,
    hh: f64,
    hx: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let params = params
        .as_slice()
        .map_err(|_| PyValueError::new_err("params must be a contiguous float64 array"))?;
    let params = otft_params_slice(params)?;
    let point_view = points.as_array();
    let point_shape = point_view.dim();
    if point_shape.1 != 5 {
        return Err(PyValueError::new_err("points must have shape (N, 5)"));
    }
    if !point_view.is_standard_layout() {
        return Err(PyValueError::new_err(
            "points must be a C-contiguous float64 array",
        ));
    }
    let points = points
        .as_slice()
        .map_err(|_| PyValueError::new_err("points must be a C-contiguous float64 array"))?;
    let count = point_shape.0;
    let values = py.detach(move || {
        let mut values = Vec::with_capacity(count * 3);
        for point in points.chunks_exact(5) {
            let result = otft::terminal_derivatives(
                &params, point[0], point[1], point[2], point[3], point[4], need_gm, need_gds,
                use_abs, hh, hx,
            );
            values.extend_from_slice(&[f64::from(result.0), result.1, result.2]);
        }
        values
    });
    let array = Array2::from_shape_vec((count, 3), values).map_err(|error| {
        PyRuntimeError::new_err(format!("invalid OTFT derivative shape: {error}"))
    })?;
    Ok(array.into_pyarray(py))
}

/// Solve OTFT internal nodes for a batch of external biases and initial guesses.
#[pyfunction]
#[pyo3(signature = (params, points, tol=1e-12, maxit=40, analytic=true))]
fn otft_newton_batch<'py>(
    py: Python<'py>,
    params: PyReadonlyArray1<'py, f64>,
    points: PyReadonlyArray2<'py, f64>,
    tol: f64,
    maxit: usize,
    analytic: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let params = params
        .as_slice()
        .map_err(|_| PyValueError::new_err("params must be a contiguous float64 array"))?;
    let params = otft_params_slice(params)?;
    let point_view = points.as_array();
    let point_shape = point_view.dim();
    if point_shape.1 != 5 {
        return Err(PyValueError::new_err("points must have shape (N, 5)"));
    }
    if !point_view.is_standard_layout() {
        return Err(PyValueError::new_err(
            "points must be a C-contiguous float64 array",
        ));
    }
    let points = points
        .as_slice()
        .map_err(|_| PyValueError::new_err("points must be a C-contiguous float64 array"))?;
    let count = point_shape.0;
    let values = py.detach(move || {
        let mut values = Vec::with_capacity(count * 5);
        for point in points.chunks_exact(5) {
            let [vs, vd, vg, x0s, x0d] = [point[0], point[1], point[2], point[3], point[4]];
            let result = if analytic {
                otft::newton_internal_fast(&params, vs, vd, vg, x0s, x0d, tol, maxit)
            } else {
                otft::newton_internal(&params, vs, vd, vg, x0s, x0d, tol, maxit)
            };
            values.extend_from_slice(&[
                f64::from(result.converged),
                result.vs1,
                result.vd1,
                result.iterations as f64,
                result.fd_fallbacks as f64,
            ]);
        }
        values
    });
    let array = Array2::from_shape_vec((count, 5), values)
        .map_err(|error| PyRuntimeError::new_err(format!("invalid OTFT Newton shape: {error}")))?;
    Ok(array.into_pyarray(py))
}

/// Resolve compiled topology terms in one GIL-free batch.
#[pyfunction]
fn mna_term_values<'py>(
    py: Python<'py>,
    kinds: PyReadonlyArray1<'py, i64>,
    references: PyReadonlyArray1<'py, i64>,
    values: PyReadonlyArray1<'py, f64>,
    state: PyReadonlyArray1<'py, f64>,
    inputs: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let kinds = kinds
        .as_slice()
        .map_err(|_| PyValueError::new_err("kinds must be a contiguous int64 array"))?;
    let references = references
        .as_slice()
        .map_err(|_| PyValueError::new_err("references must be a contiguous int64 array"))?;
    let values = values
        .as_slice()
        .map_err(|_| PyValueError::new_err("values must be a contiguous float64 array"))?;
    let state = state
        .as_slice()
        .map_err(|_| PyValueError::new_err("state must be a contiguous float64 array"))?;
    let inputs = inputs
        .as_slice()
        .map_err(|_| PyValueError::new_err("inputs must be a contiguous float64 array"))?;
    if kinds.len() != references.len() || kinds.len() != values.len() {
        return Err(PyRuntimeError::new_err(
            "MNA term kind/reference/value arrays must have equal lengths",
        ));
    }
    if references.iter().any(|reference| *reference < 0) {
        return Err(PyValueError::new_err(
            "MNA term references must be non-negative",
        ));
    }
    let result: PyResult<Vec<f64>> = py.detach(move || {
        kinds
            .iter()
            .zip(references.iter())
            .zip(values.iter())
            .map(|((kind, reference), value)| {
                mna::Term {
                    kind: *kind,
                    reference: *reference as usize,
                    value: *value,
                }
                .resolve(state, inputs)
                .ok_or_else(|| PyRuntimeError::new_err("MNA term reference is out of bounds"))
            })
            .collect()
    });
    Ok(result?.into_pyarray(py))
}

/// Apply the solver's order-preserving in-place GEPP to `A*x = -b`.
#[pyfunction]
#[allow(clippy::type_complexity)]
fn dense_neg_solve<'py>(
    py: Python<'py>,
    matrix: PyReadonlyArray2<'py, f64>,
    rhs: PyReadonlyArray1<'py, f64>,
) -> PyResult<(bool, Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>)> {
    let matrix_view = matrix.as_array();
    let matrix_shape = matrix_view.dim();
    if !matrix_view.is_standard_layout() {
        return Err(PyValueError::new_err(
            "matrix must be a C-contiguous float64 array",
        ));
    }
    let matrix = matrix
        .as_slice()
        .map_err(|_| PyValueError::new_err("matrix must be a C-contiguous float64 array"))?;
    let rhs = rhs
        .as_slice()
        .map_err(|_| PyValueError::new_err("rhs must be a contiguous float64 array"))?;
    let n = rhs.len();
    if matrix_shape != (n, n) {
        return Err(PyRuntimeError::new_err(
            "dense solve matrix must be square and match rhs length",
        ));
    }
    let (solved, matrix, rhs) = py.detach(move || {
        let mut matrix = matrix.to_vec();
        let mut rhs = rhs.to_vec();
        let solved = n == 0 || mna::solve_dense_neg_rhs_in_place(&mut matrix, &mut rhs);
        (solved, matrix, rhs)
    });
    let matrix = Array2::from_shape_vec((n, n), matrix)
        .map_err(|error| PyRuntimeError::new_err(format!("invalid dense matrix shape: {error}")))?
        .into_pyarray(py);
    Ok((solved, matrix, rhs.into_pyarray(py)))
}

/// Assemble dense harmonic-balance G+jwC and C conversion blocks.
#[pyfunction]
#[allow(clippy::type_complexity)]
fn periodic_hb_blocks<'py>(
    py: Python<'py>,
    gf: PyReadonlyArray3<'py, Complex64>,
    cf: PyReadonlyArray3<'py, Complex64>,
    sidebands: usize,
    fundamental: f64,
    charge_caps: bool,
) -> PyResult<(
    Bound<'py, PyArray2<Complex64>>,
    Bound<'py, PyArray2<Complex64>>,
)> {
    let gf_view = gf.as_array();
    let cf_view = cf.as_array();
    let shape = gf_view.dim();
    if shape != cf_view.dim() || shape.0 == 0 || shape.1 == 0 || shape.1 != shape.2 {
        return Err(PyValueError::new_err(
            "gf and cf must have matching shape (samples, state, state)",
        ));
    }
    if !gf_view.is_standard_layout() || !cf_view.is_standard_layout() {
        return Err(PyValueError::new_err(
            "gf and cf must be C-contiguous complex128 arrays",
        ));
    }
    if !fundamental.is_finite() || fundamental < 0.0 {
        return Err(PyValueError::new_err(
            "fundamental must be a finite non-negative frequency",
        ));
    }
    let gf = gf
        .as_slice()
        .map_err(|_| PyValueError::new_err("gf must be C-contiguous complex128"))?;
    let cf = cf
        .as_slice()
        .map_err(|_| PyValueError::new_err("cf must be C-contiguous complex128"))?;
    let result = py
        .detach(move || {
            periodic::hb_blocks(
                gf,
                cf,
                shape.0,
                shape.1,
                sidebands,
                fundamental,
                charge_caps,
            )
        })
        .ok_or_else(|| PyValueError::new_err("periodic HB dimensions overflow"))?;
    let admittance = Array2::from_shape_vec((result.size, result.size), result.admittance)
        .map_err(|error| PyRuntimeError::new_err(format!("invalid HB shape: {error}")))?;
    let capacitance = Array2::from_shape_vec((result.size, result.size), result.capacitance)
        .map_err(|error| PyRuntimeError::new_err(format!("invalid HB shape: {error}")))?;
    Ok((admittance.into_pyarray(py), capacitance.into_pyarray(py)))
}

/// Fold scalar thermal and flicker sources through periodic adjoints.
#[pyfunction]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn periodic_fold_psd<'py>(
    py: Python<'py>,
    adjs: PyReadonlyArray2<'py, Complex64>,
    frequencies: PyReadonlyArray1<'py, f64>,
    sidebands: usize,
    fundamental: f64,
    p_indices: PyReadonlyArray2<'py, i64>,
    q_indices: PyReadonlyArray2<'py, i64>,
    thermal: PyReadonlyArray3<'py, Complex64>,
    flicker: PyReadonlyArray2<'py, Complex64>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray2<f64>>)> {
    let adj_shape = adjs.as_array().dim();
    let p_shape = p_indices.as_array().dim();
    let q_shape = q_indices.as_array().dim();
    let thermal_shape = thermal.as_array().dim();
    let flicker_shape = flicker.as_array().dim();
    let harmonics = sidebands
        .checked_mul(2)
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| PyValueError::new_err("sideband count overflows"))?;
    let modulation_width = sidebands
        .checked_mul(4)
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| PyValueError::new_err("sideband count overflows"))?;
    let frequency_count = frequencies.len()?;
    let source_count = p_shape.0;
    if adj_shape.0 != frequency_count
        || p_shape != (source_count, harmonics)
        || q_shape != p_shape
        || thermal_shape != (source_count, harmonics, harmonics)
        || flicker_shape != (source_count, modulation_width)
    {
        return Err(PyValueError::new_err(
            "periodic noise arrays have inconsistent dimensions",
        ));
    }
    if !fundamental.is_finite() || fundamental < 0.0 {
        return Err(PyValueError::new_err(
            "fundamental must be a finite non-negative frequency",
        ));
    }
    let adjs = adjs
        .as_slice()
        .map_err(|_| PyValueError::new_err("adjs must be C-contiguous complex128"))?;
    let frequencies = frequencies
        .as_slice()
        .map_err(|_| PyValueError::new_err("frequencies must be contiguous float64"))?;
    let p_indices = p_indices
        .as_slice()
        .map_err(|_| PyValueError::new_err("p_indices must be C-contiguous int64"))?;
    let q_indices = q_indices
        .as_slice()
        .map_err(|_| PyValueError::new_err("q_indices must be C-contiguous int64"))?;
    let thermal = thermal
        .as_slice()
        .map_err(|_| PyValueError::new_err("thermal must be C-contiguous complex128"))?;
    let flicker = flicker
        .as_slice()
        .map_err(|_| PyValueError::new_err("flicker must be C-contiguous complex128"))?;
    let result = py
        .detach(move || {
            periodic::fold_psd(
                adjs,
                frequencies,
                adj_shape.1,
                sidebands,
                fundamental,
                p_indices,
                q_indices,
                source_count,
                thermal,
                flicker,
            )
        })
        .ok_or_else(|| PyValueError::new_err("invalid periodic noise array contents"))?;
    let output = Array1::from_vec(result.output);
    let devices = Array2::from_shape_vec((result.sources, result.frequencies), result.devices)
        .map_err(|error| PyRuntimeError::new_err(format!("invalid PSD shape: {error}")))?;
    Ok((output.into_pyarray(py), devices.into_pyarray(py)))
}

type PacValueRecord = (u8, usize, f64);
type PacStampRecord = (u8, usize);
type PacDeviceRecord = (
    PacValueRecord,
    PacValueRecord,
    PacValueRecord,
    PacStampRecord,
    PacStampRecord,
    PacStampRecord,
    Vec<f64>,
    Option<(usize, f64, f64)>,
);
type PacPassiveRecord = (PacStampRecord, PacStampRecord, f64);
type PacDenseRecord = [PacStampRecord; 4];

fn pac_value(record: PacValueRecord) -> periodic::ValueTerm {
    periodic::ValueTerm {
        kind: record.0,
        reference: record.1,
        value: record.2,
    }
}

fn pac_stamp(record: PacStampRecord) -> periodic::StampTerm {
    periodic::StampTerm {
        kind: record.0,
        reference: record.1,
    }
}

/// Immutable PAC topology and OTFT parameter pack compiled once from Python.
#[pyclass(frozen)]
struct PeriodicLinearizationProblem {
    problem: periodic::PacProblem,
}

#[pymethods]
impl PeriodicLinearizationProblem {
    #[new]
    fn new(spec: &Bound<'_, PyDict>) -> PyResult<Self> {
        let devices: Vec<PacDeviceRecord> = required(spec, "devices")?;
        let dense_devices: Vec<PacDenseRecord> = required(spec, "dense_devices")?;
        let resistors: Vec<PacPassiveRecord> = required(spec, "resistors")?;
        let capacitors: Vec<PacPassiveRecord> = required(spec, "capacitors")?;
        let devices = devices
            .into_iter()
            .map(|record| {
                Ok(periodic::OtftDevice {
                    value_d: pac_value(record.0),
                    value_g: pac_value(record.1),
                    value_s: pac_value(record.2),
                    stamp_d: pac_stamp(record.3),
                    stamp_g: pac_stamp(record.4),
                    stamp_s: pac_stamp(record.5),
                    params: otft_params(record.6)?,
                    gate1: record.7,
                })
            })
            .collect::<PyResult<Vec<_>>>()?;
        let passives = |records: Vec<PacPassiveRecord>| {
            records
                .into_iter()
                .map(|record| periodic::Passive {
                    a: pac_stamp(record.0),
                    b: pac_stamp(record.1),
                    value: record.2,
                })
                .collect::<Vec<_>>()
        };
        let problem = periodic::PacProblem {
            node_count: required(spec, "node_count")?,
            state_count: required(spec, "state_count")?,
            input_count: required(spec, "input_count")?,
            drive_count: required(spec, "drive_count")?,
            devices,
            dense_devices: dense_devices
                .into_iter()
                .map(|record| periodic::DenseDevice {
                    terminals: record.map(pac_stamp),
                })
                .collect(),
            resistors: passives(resistors),
            capacitors: passives(capacitors),
            gmin: required(spec, "gmin")?,
            fd_step: required(spec, "fd_step")?,
        };
        if !problem.validate() {
            return Err(PyValueError::new_err(
                "invalid periodic linearization topology",
            ));
        }
        Ok(Self { problem })
    }

    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn linearize<'py>(
        &self,
        py: Python<'py>,
        node_wave: PyReadonlyArray2<'py, f64>,
        input_wave: PyReadonlyArray2<'py, f64>,
        node_dot: PyReadonlyArray2<'py, f64>,
        input_dot: PyReadonlyArray2<'py, f64>,
        dense_conductance: PyReadonlyArray4<'py, f64>,
        dense_capacitance: PyReadonlyArray4<'py, f64>,
    ) -> PyResult<(
        Bound<'py, PyArray3<f64>>,
        Bound<'py, PyArray3<f64>>,
        Bound<'py, PyArray3<f64>>,
        Bound<'py, PyArray3<f64>>,
    )> {
        let node_shape = node_wave.as_array().dim();
        let input_shape = input_wave.as_array().dim();
        if node_shape.1 != self.problem.node_count
            || node_dot.as_array().dim() != node_shape
            || input_shape != (self.problem.input_count, node_shape.0)
            || input_dot.as_array().dim() != input_shape
            || dense_conductance.as_array().dim()
                != (node_shape.0, self.problem.dense_devices.len(), 4, 4)
            || dense_capacitance.as_array().dim() != dense_conductance.as_array().dim()
        {
            return Err(PyValueError::new_err(
                "periodic orbit arrays have inconsistent dimensions",
            ));
        }
        let node_wave = node_wave
            .as_slice()
            .map_err(|_| PyValueError::new_err("node_wave must be C-contiguous float64"))?;
        let input_wave = input_wave
            .as_slice()
            .map_err(|_| PyValueError::new_err("input_wave must be C-contiguous float64"))?;
        let node_dot = node_dot
            .as_slice()
            .map_err(|_| PyValueError::new_err("node_dot must be C-contiguous float64"))?;
        let input_dot = input_dot
            .as_slice()
            .map_err(|_| PyValueError::new_err("input_dot must be C-contiguous float64"))?;
        let dense_conductance = dense_conductance
            .as_slice()
            .map_err(|_| PyValueError::new_err("dense_conductance must be C-contiguous float64"))?;
        let dense_capacitance = dense_capacitance
            .as_slice()
            .map_err(|_| PyValueError::new_err("dense_capacitance must be C-contiguous float64"))?;
        let problem = &self.problem;
        let result = py
            .detach(move || {
                periodic::linearize_otft_orbit(
                    problem,
                    node_wave,
                    input_wave,
                    node_dot,
                    input_dot,
                    dense_conductance,
                    dense_capacitance,
                    node_shape.0,
                )
            })
            .ok_or_else(|| PyRuntimeError::new_err("periodic orbit linearization failed"))?;
        let matrix_shape = (result.samples, result.state_count, result.state_count);
        let input_shape = (result.samples, result.state_count, result.drive_count);
        let array3 = |values, shape, name: &str| {
            Array3::from_shape_vec(shape, values)
                .map_err(|error| PyRuntimeError::new_err(format!("invalid {name} shape: {error}")))
                .map(|array| array.into_pyarray(py))
        };
        Ok((
            array3(result.conductance, matrix_shape, "conductance")?,
            array3(result.capacitance, matrix_shape, "capacitance")?,
            array3(result.input_conductance, input_shape, "input conductance")?,
            array3(result.input_capacitance, input_shape, "input capacitance")?,
        ))
    }
}

type TermRecord = (i64, usize, f64);
type DeviceRecord = (
    TermRecord,
    TermRecord,
    TermRecord,
    i64,
    i64,
    i64,
    bool,
    Vec<f64>,
);
type ResistorRecord = (TermRecord, TermRecord, i64, i64, f64);
type CapacitorRecord = (TermRecord, TermRecord, i64, i64, f64);
type CurrentSourceRecord = (i64, i64, f64);
type DynamicSourceRecord = (i64, i64, usize);
type VccsRecord = (i64, i64, TermRecord, TermRecord, i64, i64, f64);
type VoltageSourceRecord = (TermRecord, TermRecord, i64, i64, usize, f64, i64);
type VcvsRecord = (
    TermRecord,
    TermRecord,
    TermRecord,
    TermRecord,
    i64,
    i64,
    i64,
    i64,
    usize,
    f64,
);
type CccsRecord = (i64, i64, usize, f64);
type CcvsRecord = (TermRecord, TermRecord, i64, i64, usize, usize, f64);
type LtiDenseRecord = (Vec<TermRecord>, Vec<Vec<f64>>, Vec<Vec<f64>>);
type LtiMosRecord = (TermRecord, TermRecord, TermRecord, f64, f64, f64, f64);
type LtiBranchRecord = (TermRecord, TermRecord, f64);
type LtiVccsRecord = (TermRecord, TermRecord, TermRecord, TermRecord, f64);
type LtiVoltageRecord = (TermRecord, TermRecord, usize, f64, f64);
type LtiVcvsRecord = (TermRecord, TermRecord, TermRecord, TermRecord, usize, f64);
type LtiCccsRecord = (TermRecord, TermRecord, usize, f64);
type LtiCcvsRecord = (TermRecord, TermRecord, usize, usize, f64);

fn optional_index(value: i64) -> PyResult<Option<usize>> {
    if value < 0 {
        Ok(None)
    } else {
        usize::try_from(value)
            .map(Some)
            .map_err(|_| PyValueError::new_err("MNA index does not fit usize"))
    }
}

fn term(record: TermRecord) -> mna::Term {
    mna::Term {
        kind: record.0,
        reference: record.1,
        value: record.2,
    }
}

fn required<'py, T>(spec: &Bound<'py, PyDict>, key: &str) -> PyResult<T>
where
    T: FromPyObjectOwned<'py>,
{
    let value = spec
        .get_item(key)?
        .ok_or_else(|| PyKeyError::new_err(key.to_string()))?
        .extract::<T>()
        .map_err(Into::into)?;
    Ok(value)
}

fn flatten_square(matrix: Vec<Vec<f64>>, width: usize, name: &str) -> PyResult<Vec<f64>> {
    if matrix.len() != width || matrix.iter().any(|row| row.len() != width) {
        return Err(PyValueError::new_err(format!(
            "{name} must be a {width}x{width} matrix"
        )));
    }
    Ok(matrix.into_iter().flatten().collect())
}

fn rows_into_array2<'py>(
    py: Python<'py>,
    rows: Vec<Vec<f64>>,
    width: usize,
    name: &str,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let height = rows.len();
    if rows.iter().any(|row| row.len() != width) {
        return Err(PyRuntimeError::new_err(format!(
            "{name} returned inconsistent row widths"
        )));
    }
    let values = rows.into_iter().flatten().collect();
    let array = Array2::from_shape_vec((height, width), values)
        .map_err(|error| PyRuntimeError::new_err(format!("invalid {name} shape: {error}")))?;
    Ok(array.into_pyarray(py))
}

fn complex_rows_into_array3<'py>(
    py: Python<'py>,
    rows: Vec<Vec<lti::Complex>>,
    width: usize,
    name: &str,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    let height = rows.len();
    if rows.iter().any(|row| row.len() != width) {
        return Err(PyRuntimeError::new_err(format!(
            "{name} returned inconsistent row widths"
        )));
    }
    let mut values = Vec::with_capacity(height.saturating_mul(width).saturating_mul(2));
    for value in rows.into_iter().flatten() {
        values.push(value.re);
        values.push(value.im);
    }
    let array = Array3::from_shape_vec((height, width, 2), values)
        .map_err(|error| PyRuntimeError::new_err(format!("invalid {name} shape: {error}")))?;
    Ok(array.into_pyarray(py))
}

#[pyclass(frozen)]
struct LtiProblem {
    system: lti::System,
}

#[pymethods]
impl LtiProblem {
    #[new]
    fn new(spec: &Bound<'_, PyDict>) -> PyResult<Self> {
        let size: usize = required(spec, "size")?;
        let dense: Vec<LtiDenseRecord> = required(spec, "dense_devices")?;
        let mos: Vec<LtiMosRecord> = required(spec, "mos_devices")?;
        let capacitors: Vec<LtiBranchRecord> = required(spec, "capacitors")?;
        let resistors: Vec<LtiBranchRecord> = required(spec, "resistors")?;
        let vccs: Vec<LtiVccsRecord> = required(spec, "vccs")?;
        let voltage_sources: Vec<LtiVoltageRecord> = required(spec, "voltage_sources")?;
        let vcvs: Vec<LtiVcvsRecord> = required(spec, "vcvs")?;
        let cccs: Vec<LtiCccsRecord> = required(spec, "cccs")?;
        let ccvs: Vec<LtiCcvsRecord> = required(spec, "ccvs")?;
        let problem = lti::Problem {
            size,
            dense_devices: dense
                .into_iter()
                .map(|(terms, conductance, capacitance)| {
                    let width = terms.len();
                    Ok(lti::DenseDevice {
                        terms: terms.into_iter().map(term).collect(),
                        conductance: flatten_square(conductance, width, "conductance")?,
                        capacitance: flatten_square(capacitance, width, "capacitance")?,
                    })
                })
                .collect::<PyResult<_>>()?,
            mos_devices: mos
                .into_iter()
                .map(|record| lti::MosDevice {
                    drain: term(record.0),
                    gate: term(record.1),
                    source: term(record.2),
                    gm: record.3,
                    gds: record.4,
                    cgs: record.5,
                    cgd: record.6,
                })
                .collect(),
            capacitors: capacitors
                .into_iter()
                .map(|record| lti::Branch {
                    a: term(record.0),
                    b: term(record.1),
                    value: record.2,
                })
                .collect(),
            resistors: resistors
                .into_iter()
                .map(|record| lti::Branch {
                    a: term(record.0),
                    b: term(record.1),
                    value: record.2,
                })
                .collect(),
            vccs: vccs
                .into_iter()
                .map(|record| lti::Vccs {
                    p: term(record.0),
                    q: term(record.1),
                    cp: term(record.2),
                    cn: term(record.3),
                    gm: record.4,
                })
                .collect(),
            voltage_sources: voltage_sources
                .into_iter()
                .map(|record| lti::VoltageSource {
                    p: term(record.0),
                    q: term(record.1),
                    branch: record.2,
                    emf_re: record.3,
                    emf_im: record.4,
                })
                .collect(),
            vcvs: vcvs
                .into_iter()
                .map(|record| lti::Vcvs {
                    p: term(record.0),
                    q: term(record.1),
                    cp: term(record.2),
                    cn: term(record.3),
                    branch: record.4,
                    mu: record.5,
                })
                .collect(),
            cccs: cccs
                .into_iter()
                .map(|record| lti::Cccs {
                    p: term(record.0),
                    q: term(record.1),
                    control_branch: record.2,
                    beta: record.3,
                })
                .collect(),
            ccvs: ccvs
                .into_iter()
                .map(|record| lti::Ccvs {
                    p: term(record.0),
                    q: term(record.1),
                    control_branch: record.2,
                    branch: record.3,
                    gamma: record.4,
                })
                .collect(),
        };
        let system = problem.try_assemble().map_err(core_error)?;
        Ok(Self { system })
    }

    #[allow(clippy::type_complexity)]
    fn matrices(&self) -> (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<f64>, Vec<f64>) {
        let n = self.system.size;
        (
            self.system
                .conductance
                .chunks(n)
                .map(<[f64]>::to_vec)
                .collect(),
            self.system
                .capacitance
                .chunks(n)
                .map(<[f64]>::to_vec)
                .collect(),
            self.system.rhs_g.clone(),
            self.system.rhs_c.clone(),
        )
    }

    fn solve<'py>(
        &self,
        py: Python<'py>,
        frequencies: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray3<f64>>> {
        let frequencies = frequencies
            .as_slice()
            .map_err(|_| PyValueError::new_err("frequencies must be a contiguous float64 array"))?;
        let system = self.system.clone();
        let width = system.size;
        let rows = py.detach(move || {
            system
                .try_solve_frequencies(frequencies)
                .map_err(core_error)
        })?;
        complex_rows_into_array3(py, rows, width, "LTI solve")
    }

    fn solve_transpose<'py>(
        &self,
        py: Python<'py>,
        frequencies: PyReadonlyArray1<'py, f64>,
        sense: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray3<f64>>> {
        let frequencies = frequencies
            .as_slice()
            .map_err(|_| PyValueError::new_err("frequencies must be a contiguous float64 array"))?;
        let sense = sense
            .as_slice()
            .map_err(|_| PyValueError::new_err("sense must be a contiguous float64 array"))?;
        let system = self.system.clone();
        let width = system.size;
        let rows = py.detach(move || {
            system
                .try_solve_transpose(frequencies, sense)
                .map_err(core_error)
        })?;
        complex_rows_into_array3(py, rows, width, "transposed LTI solve")
    }
}

/// Immutable OTFT transient topology. Runtime caches are per solve call.
#[pyclass(frozen)]
struct OtftTransientProblem {
    problem: transient::Problem,
}

struct RustBsimEvaluator {
    handles: Vec<usize>,
}

impl bsim_transient::Evaluator for RustBsimEvaluator {
    fn evaluate(
        &mut self,
        index: usize,
        terminals: [f64; 4],
    ) -> Option<bsim_transient::Evaluation> {
        let handle = *self.handles.get(index)? as *mut CoBsim4;
        let mut currents = [0.0; 4];
        let mut conductance = [0.0; 16];
        let mut charges = [0.0; 4];
        let mut capacitance = [0.0; 16];
        let status = unsafe {
            co_bsim4::eval_vp(
                handle,
                terminals.as_ptr(),
                currents.as_mut_ptr(),
                conductance.as_mut_ptr(),
                charges.as_mut_ptr(),
                capacitance.as_mut_ptr(),
            )
        };
        (status == OK).then_some(bsim_transient::Evaluation {
            currents,
            conductance,
            charges,
            capacitance,
        })
    }
}

/// Non-owning circuit/grid solver for Rust-backed BSIM4 handles.
#[pyclass(frozen)]
struct Bsim4TransientProblem {
    circuit: transient::Problem,
    devices: Vec<bsim_transient::Device>,
    handles: Vec<usize>,
}

#[pymethods]
impl Bsim4TransientProblem {
    #[new]
    fn new(
        circuit: PyRef<'_, OtftTransientProblem>,
        devices: Vec<(Vec<TermRecord>, Vec<i64>)>,
        handles: Vec<usize>,
    ) -> PyResult<Self> {
        if devices.len() != handles.len() {
            return Err(PyValueError::new_err(
                "BSIM4 devices and handle arrays must have equal length",
            ));
        }
        let devices = devices
            .into_iter()
            .enumerate()
            .map(|(index, (terms, rows))| {
                if terms.len() != 4 || rows.len() != 4 {
                    return Err(PyValueError::new_err(
                        "each BSIM4 transient device needs four terms and rows",
                    ));
                }
                Ok(bsim_transient::Device {
                    terms: [
                        term(terms[0]),
                        term(terms[1]),
                        term(terms[2]),
                        term(terms[3]),
                    ],
                    rows: [
                        optional_index(rows[0])?,
                        optional_index(rows[1])?,
                        optional_index(rows[2])?,
                        optional_index(rows[3])?,
                    ],
                    evaluator_index: index,
                })
            })
            .collect::<PyResult<Vec<_>>>()?;
        let mut circuit = circuit.problem.clone();
        circuit.devices.clear();
        let valid_devices = devices.iter().all(|device| {
            device
                .terms
                .iter()
                .all(|value| value.is_valid(circuit.node_count, true))
                && device
                    .rows
                    .iter()
                    .flatten()
                    .all(|index| *index < circuit.node_count)
        });
        if !circuit.validate() || !valid_devices || handles.contains(&0) {
            return Err(PyValueError::new_err(
                "invalid BSIM4 transient topology or handle",
            ));
        }
        Ok(Self {
            circuit,
            devices,
            handles,
        })
    }

    #[pyo3(signature = (
        initial, inputs, max_iterations=80, voltage_tolerance=1e-10,
        step_limit=0.25, gmin=1e-12
    ))]
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn solve_dc<'py>(
        &self,
        py: Python<'py>,
        initial: PyReadonlyArray1<'py, f64>,
        inputs: PyReadonlyArray1<'py, f64>,
        max_iterations: usize,
        voltage_tolerance: f64,
        step_limit: f64,
        gmin: f64,
    ) -> PyResult<(bool, Bound<'py, PyArray1<f64>>, usize, f64)> {
        let initial = initial
            .as_slice()
            .map_err(|_| PyValueError::new_err("initial must be a contiguous float64 array"))?;
        let inputs = inputs
            .as_slice()
            .map_err(|_| PyValueError::new_err("inputs must be a contiguous float64 array"))?;
        let circuit = self.circuit.clone();
        let devices = self.devices.clone();
        let handles = self.handles.clone();
        let result = py.detach(move || {
            let mut evaluator = RustBsimEvaluator { handles };
            bsim_transient::solve_dc(
                &circuit,
                &devices,
                &mut evaluator,
                initial,
                inputs,
                bsim_transient::DcOptions {
                    max_iterations,
                    voltage_tolerance,
                    step_limit,
                    gmin,
                },
            )
        });
        Ok((
            result.converged,
            result.state.into_pyarray(py),
            result.iterations,
            result.residual_inf,
        ))
    }

    #[pyo3(signature = (
        initial, times, inputs, integration_method="be", max_iterations=40,
        voltage_tolerance=1e-8, step_limit=0.25, gmin=1e-12
    ))]
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn solve_fixed_grid<'py>(
        &self,
        py: Python<'py>,
        initial: PyReadonlyArray1<'py, f64>,
        times: PyReadonlyArray1<'py, f64>,
        inputs: PyReadonlyArray2<'py, f64>,
        integration_method: &str,
        max_iterations: usize,
        voltage_tolerance: f64,
        step_limit: f64,
        gmin: f64,
    ) -> PyResult<(bool, Bound<'py, PyArray2<f64>>, usize, i64)> {
        let gear2 = match integration_method {
            "be" => false,
            "gear2" | "bdf2" => true,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown BSIM4 integration method {other:?}"
                )));
            }
        };
        let initial = initial
            .as_slice()
            .map_err(|_| PyValueError::new_err("initial must be a contiguous float64 array"))?;
        let times = times
            .as_slice()
            .map_err(|_| PyValueError::new_err("times must be a contiguous float64 array"))?;
        let input_view = inputs.as_array();
        let input_shape = input_view.dim();
        if !input_view.is_standard_layout() {
            return Err(PyValueError::new_err(
                "inputs must be a C-contiguous float64 array",
            ));
        }
        let inputs = inputs
            .as_slice()
            .map_err(|_| PyValueError::new_err("inputs must be a C-contiguous float64 array"))?;
        let waveforms = transient::Waveforms::new(inputs, input_shape.0, input_shape.1)
            .ok_or_else(|| PyValueError::new_err("invalid inputs shape"))?;
        bsim_transient::validate_fixed_grid_input(
            &self.circuit,
            &self.devices,
            initial,
            times,
            waveforms,
        )
        .map_err(core_error)?;
        let circuit = self.circuit.clone();
        let devices = self.devices.clone();
        let handles = self.handles.clone();
        let width = circuit.size;
        let result = py.detach(move || {
            let mut evaluator = RustBsimEvaluator { handles };
            bsim_transient::solve_fixed_grid(
                &circuit,
                &devices,
                &mut evaluator,
                initial,
                times,
                waveforms,
                bsim_transient::Options {
                    gear2,
                    max_iterations,
                    voltage_tolerance,
                    step_limit,
                    gmin,
                },
            )
        });
        let states = rows_into_array2(py, result.states, width, "BSIM4 transient states")?;
        Ok((
            result.completed,
            states,
            result.failures,
            result.first_failure.map_or(-1, |value| value as i64),
        ))
    }
}

#[pymethods]
impl OtftTransientProblem {
    #[new]
    fn new(spec: &Bound<'_, PyDict>) -> PyResult<Self> {
        let devices: Vec<DeviceRecord> = required(spec, "devices")?;
        let resistors: Vec<ResistorRecord> = required(spec, "resistors")?;
        let capacitors: Vec<CapacitorRecord> = required(spec, "capacitors")?;
        let current_sources: Vec<CurrentSourceRecord> = required(spec, "current_sources")?;
        let dynamic_sources: Vec<DynamicSourceRecord> = required(spec, "dynamic_sources")?;
        let vccs: Vec<VccsRecord> = required(spec, "vccs")?;
        let voltage_sources: Vec<VoltageSourceRecord> = required(spec, "voltage_sources")?;
        let vcvs: Vec<VcvsRecord> = required(spec, "vcvs")?;
        let cccs: Vec<CccsRecord> = required(spec, "cccs")?;
        let ccvs: Vec<CcvsRecord> = required(spec, "ccvs")?;
        let problem = transient::Problem {
            node_count: required(spec, "node_count")?,
            size: required(spec, "size")?,
            devices: devices
                .into_iter()
                .map(|record| {
                    Ok(transient::Device {
                        drain: term(record.0),
                        gate: term(record.1),
                        source: term(record.2),
                        di: optional_index(record.3)?,
                        gi: optional_index(record.4)?,
                        si: optional_index(record.5)?,
                        use_abs: record.6,
                        params: otft_params(record.7)?,
                    })
                })
                .collect::<PyResult<_>>()?,
            resistors: resistors
                .into_iter()
                .map(|record| {
                    Ok(transient::Resistor {
                        a: term(record.0),
                        b: term(record.1),
                        ai: optional_index(record.2)?,
                        bi: optional_index(record.3)?,
                        conductance: record.4,
                    })
                })
                .collect::<PyResult<_>>()?,
            capacitors: capacitors
                .into_iter()
                .map(|record| {
                    Ok(transient::Capacitor {
                        a: term(record.0),
                        b: term(record.1),
                        ai: optional_index(record.2)?,
                        bi: optional_index(record.3)?,
                        capacitance: record.4,
                    })
                })
                .collect::<PyResult<_>>()?,
            current_sources: current_sources
                .into_iter()
                .map(|record| {
                    Ok(transient::CurrentSource {
                        pi: optional_index(record.0)?,
                        qi: optional_index(record.1)?,
                        value: record.2,
                    })
                })
                .collect::<PyResult<_>>()?,
            dynamic_current_sources: dynamic_sources
                .into_iter()
                .map(|record| {
                    Ok(transient::DynamicCurrentSource {
                        pi: optional_index(record.0)?,
                        qi: optional_index(record.1)?,
                        input_index: record.2,
                    })
                })
                .collect::<PyResult<_>>()?,
            vccs: vccs
                .into_iter()
                .map(|record| {
                    Ok(transient::Vccs {
                        pi: optional_index(record.0)?,
                        qi: optional_index(record.1)?,
                        cp: term(record.2),
                        cn: term(record.3),
                        cpi: optional_index(record.4)?,
                        cni: optional_index(record.5)?,
                        gm: record.6,
                    })
                })
                .collect::<PyResult<_>>()?,
            voltage_sources: voltage_sources
                .into_iter()
                .map(|record| {
                    Ok(transient::VoltageSource {
                        a: term(record.0),
                        b: term(record.1),
                        pi: optional_index(record.2)?,
                        qi: optional_index(record.3)?,
                        branch: record.4,
                        emf: record.5,
                        input_index: optional_index(record.6)?,
                    })
                })
                .collect::<PyResult<_>>()?,
            vcvs: vcvs
                .into_iter()
                .map(|record| {
                    Ok(transient::Vcvs {
                        a: term(record.0),
                        b: term(record.1),
                        cp: term(record.2),
                        cn: term(record.3),
                        pi: optional_index(record.4)?,
                        qi: optional_index(record.5)?,
                        cpi: optional_index(record.6)?,
                        cni: optional_index(record.7)?,
                        branch: record.8,
                        mu: record.9,
                    })
                })
                .collect::<PyResult<_>>()?,
            cccs: cccs
                .into_iter()
                .map(|record| {
                    Ok(transient::Cccs {
                        pi: optional_index(record.0)?,
                        qi: optional_index(record.1)?,
                        control_branch: record.2,
                        beta: record.3,
                    })
                })
                .collect::<PyResult<_>>()?,
            ccvs: ccvs
                .into_iter()
                .map(|record| {
                    Ok(transient::Ccvs {
                        a: term(record.0),
                        b: term(record.1),
                        pi: optional_index(record.2)?,
                        qi: optional_index(record.3)?,
                        branch: record.4,
                        control_branch: record.5,
                        gamma: record.6,
                    })
                })
                .collect::<PyResult<_>>()?,
        };
        problem.try_validate().map_err(core_error)?;
        Ok(Self { problem })
    }

    #[pyo3(signature = (
        state, previous_state, input_now, input_previous, h,
        cap_mode=0, bdf=(1.0, -1.0, 0.0), previous2_state=None,
        input_previous2=None, gmin=1e-12, hh=1e-3
    ))]
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn stamp(
        &self,
        py: Python<'_>,
        state: Vec<f64>,
        previous_state: Vec<f64>,
        input_now: Vec<f64>,
        input_previous: Vec<f64>,
        h: f64,
        cap_mode: i64,
        bdf: (f64, f64, f64),
        previous2_state: Option<Vec<f64>>,
        input_previous2: Option<Vec<f64>>,
        gmin: f64,
        hh: f64,
    ) -> PyResult<(
        bool,
        Vec<f64>,
        Vec<Vec<f64>>,
        Vec<(bool, f64, f64)>,
        (usize, usize, usize, usize, usize),
    )> {
        if state.len() != self.problem.size
            || previous_state.len() != self.problem.size
            || previous2_state
                .as_ref()
                .is_some_and(|value| value.len() != self.problem.size)
            || !h.is_finite()
            || h <= 0.0
        {
            return Err(PyValueError::new_err(
                "transient stamp state lengths must match topology and h must be positive",
            ));
        }
        let problem = self.problem.clone();
        Ok(py.detach(move || {
            let mut caches = vec![transient::DeviceCache::default(); problem.devices.len()];
            let mut history = transient::HistoryTerms::new(&problem);
            let history_ok = transient::fill_history_terms(
                &problem,
                &mut caches,
                &previous_state,
                &input_previous,
                cap_mode,
                &mut history,
            );
            let mut history2 = transient::HistoryTerms::new(&problem);
            let history2_ok = match previous2_state {
                Some(previous2) => {
                    let mut cache2 = vec![transient::DeviceCache::default(); problem.devices.len()];
                    transient::fill_history_terms(
                        &problem,
                        &mut cache2,
                        &previous2,
                        input_previous2.as_deref().unwrap_or(&input_previous),
                        cap_mode,
                        &mut history2,
                    )
                }
                None => {
                    history2.clone_from(&history);
                    true
                }
            };
            let mut system = mna::DenseSystem::new(problem.size);
            let mut stats = transient::DeviceSolveStats::default();
            let ok = history_ok
                && history2_ok
                && transient::stamp_system(
                    &problem,
                    &mut caches,
                    &state,
                    &input_now,
                    &history,
                    &history2,
                    transient::StampOptions {
                        h,
                        gmin,
                        hh,
                        cap_mode,
                        bdf: [bdf.0, bdf.1, bdf.2],
                    },
                    &mut system,
                    &mut stats,
                );
            let matrix = system
                .jacobian
                .chunks(problem.size)
                .map(<[f64]>::to_vec)
                .collect();
            let caches = caches
                .into_iter()
                .map(|cache| (cache.valid, cache.vs1, cache.vd1))
                .collect();
            (
                ok,
                system.residual,
                matrix,
                caches,
                (
                    stats.solves,
                    stats.attempts,
                    stats.iterations,
                    stats.fd_fallbacks,
                    stats.terminal_fd_fallbacks,
                ),
            )
        }))
    }

    #[pyo3(signature = (
        seed, previous_state, input_now, input_previous, h,
        previous2_state=None, input_previous2=None, h_previous=0.0,
        max_iterations=30, step_limit=5.0, voltage_tolerance=1e-8,
        fallback_accept=false, fallback_tolerance=1e-9,
        clip_lo=f64::INFINITY, clip_hi=f64::NEG_INFINITY,
        gmin=1e-12, hh=1e-3, cap_mode=0
    ))]
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn newton_step(
        &self,
        py: Python<'_>,
        seed: Vec<f64>,
        previous_state: Vec<f64>,
        input_now: Vec<f64>,
        input_previous: Vec<f64>,
        h: f64,
        previous2_state: Option<Vec<f64>>,
        input_previous2: Option<Vec<f64>>,
        h_previous: f64,
        max_iterations: usize,
        step_limit: f64,
        voltage_tolerance: f64,
        fallback_accept: bool,
        fallback_tolerance: f64,
        clip_lo: f64,
        clip_hi: f64,
        gmin: f64,
        hh: f64,
        cap_mode: i64,
    ) -> PyResult<(
        Vec<f64>,
        usize,
        bool,
        bool,
        f64,
        f64,
        (usize, usize, usize, usize, usize),
    )> {
        if seed.len() != self.problem.size
            || previous_state.len() != self.problem.size
            || previous2_state
                .as_ref()
                .is_some_and(|value| value.len() != self.problem.size)
            || !h.is_finite()
            || h <= 0.0
        {
            return Err(PyValueError::new_err(
                "transient Newton state lengths must match topology and h must be positive",
            ));
        }
        let problem = self.problem.clone();
        Ok(py.detach(move || {
            let previous2 = previous2_state.unwrap_or_else(|| previous_state.clone());
            let input2 = input_previous2.unwrap_or_else(|| input_previous.clone());
            let mut caches = vec![transient::DeviceCache::default(); problem.devices.len()];
            let mut state = vec![0.0; problem.size];
            let mut system = mna::DenseSystem::new(problem.size);
            let mut stats = transient::DeviceSolveStats::default();
            let result = transient::newton_step(
                &problem,
                &mut caches,
                &seed,
                &previous_state,
                &input_now,
                &input_previous,
                h,
                &previous2,
                &input2,
                h_previous,
                transient::NewtonOptions {
                    max_iterations,
                    step_limit,
                    voltage_tolerance,
                    fallback_accept,
                    fallback_tolerance,
                    clip_lo,
                    clip_hi,
                    gmin,
                    hh,
                    cap_mode,
                },
                &mut state,
                &mut system,
                &mut stats,
            );
            (
                state,
                result.iterations,
                result.converged,
                result.usable,
                result.residual_inf,
                result.step_inf,
                (
                    stats.solves,
                    stats.attempts,
                    stats.iterations,
                    stats.fd_fallbacks,
                    stats.terminal_fd_fallbacks,
                ),
            )
        }))
    }

    #[pyo3(signature = (
        initial, times, inputs, edge_mask=Vec::new(), integration_method="be",
        max_step=-1.0, flat_max_step=-1.0, max_retry_subdivisions=0,
        max_iterations=30, step_limit=5.0, voltage_tolerance=1e-8,
        fallback_accept=false, fallback_tolerance=1e-9,
        clip_lo=f64::INFINITY, clip_hi=f64::NEG_INFINITY,
        gmin=1e-12, hh=1e-3, cap_mode=0, profile=false
    ))]
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn solve_fixed_grid<'py>(
        &self,
        py: Python<'py>,
        initial: PyReadonlyArray1<'py, f64>,
        times: PyReadonlyArray1<'py, f64>,
        inputs: PyReadonlyArray2<'py, f64>,
        edge_mask: Vec<bool>,
        integration_method: &str,
        max_step: f64,
        flat_max_step: f64,
        max_retry_subdivisions: usize,
        max_iterations: usize,
        step_limit: f64,
        voltage_tolerance: f64,
        fallback_accept: bool,
        fallback_tolerance: f64,
        clip_lo: f64,
        clip_hi: f64,
        gmin: f64,
        hh: f64,
        cap_mode: i64,
        profile: bool,
    ) -> PyResult<(
        bool,
        Bound<'py, PyArray2<f64>>,
        usize,
        i64,
        Vec<usize>,
        Bound<'py, PyArray1<f64>>,
    )> {
        let gear2 = match integration_method {
            "be" => false,
            "gear2" => true,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown Rust integration method {other:?}"
                )));
            }
        };
        let initial = initial
            .as_slice()
            .map_err(|_| PyValueError::new_err("initial must be a contiguous float64 array"))?;
        let times = times
            .as_slice()
            .map_err(|_| PyValueError::new_err("times must be a contiguous float64 array"))?;
        let input_view = inputs.as_array();
        let input_shape = input_view.dim();
        if !input_view.is_standard_layout() {
            return Err(PyValueError::new_err(
                "inputs must be a C-contiguous float64 array",
            ));
        }
        let inputs = inputs
            .as_slice()
            .map_err(|_| PyValueError::new_err("inputs must be a C-contiguous float64 array"))?;
        let waveforms = transient::Waveforms::new(inputs, input_shape.0, input_shape.1)
            .ok_or_else(|| PyValueError::new_err("invalid inputs shape"))?;
        transient::validate_fixed_grid_input(&self.problem, initial, times, waveforms, &edge_mask)
            .map_err(core_error)?;
        let problem = self.problem.clone();
        let width = problem.size;
        let result = py.detach(move || {
            transient::solve_fixed_grid(
                &problem,
                initial,
                times,
                waveforms,
                &edge_mask,
                transient::FixedGridOptions {
                    newton: transient::NewtonOptions {
                        max_iterations,
                        step_limit,
                        voltage_tolerance,
                        fallback_accept,
                        fallback_tolerance,
                        clip_lo,
                        clip_hi,
                        gmin,
                        hh,
                        cap_mode,
                    },
                    gear2,
                    max_step,
                    flat_max_step,
                    max_retry_subdivisions,
                    profile,
                },
            )
        });
        let states = rows_into_array2(py, result.states, width, "OTFT transient states")?;
        let profile = result.profile.to_vec().into_pyarray(py);
        Ok((
            result.completed,
            states,
            result.substeps,
            result.failed_index.map_or(-1, |value| value as i64),
            result.failed_intervals,
            profile,
        ))
    }

    #[pyo3(signature = (
        initial, source_times, source_inputs, max_step=-1.0,
        reltol=1e-4, voltage_abstol=1e-6, current_abstol=1e-12,
        max_steps=200000, initial_step=-1.0,
        max_iterations=30, step_limit=5.0, voltage_tolerance=1e-8,
        fallback_accept=false, fallback_tolerance=1e-9,
        clip_lo=f64::INFINITY, clip_hi=f64::NEG_INFINITY,
        gmin=1e-12, hh=1e-3, cap_mode=0, profile=false
    ))]
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn solve_adaptive_gear2<'py>(
        &self,
        py: Python<'py>,
        initial: PyReadonlyArray1<'py, f64>,
        source_times: PyReadonlyArray1<'py, f64>,
        source_inputs: PyReadonlyArray2<'py, f64>,
        max_step: f64,
        reltol: f64,
        voltage_abstol: f64,
        current_abstol: f64,
        max_steps: usize,
        initial_step: f64,
        max_iterations: usize,
        step_limit: f64,
        voltage_tolerance: f64,
        fallback_accept: bool,
        fallback_tolerance: f64,
        clip_lo: f64,
        clip_hi: f64,
        gmin: f64,
        hh: f64,
        cap_mode: i64,
        profile: bool,
    ) -> PyResult<(
        bool,
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray2<f64>>,
        Bound<'py, PyArray2<f64>>,
        usize,
        usize,
        Bound<'py, PyArray1<f64>>,
    )> {
        let initial = initial
            .as_slice()
            .map_err(|_| PyValueError::new_err("initial must be a contiguous float64 array"))?;
        let source_times = source_times.as_slice().map_err(|_| {
            PyValueError::new_err("source_times must be a contiguous float64 array")
        })?;
        let input_view = source_inputs.as_array();
        let input_shape = input_view.dim();
        if !input_view.is_standard_layout() {
            return Err(PyValueError::new_err(
                "source_inputs must be a C-contiguous float64 array",
            ));
        }
        let source_inputs = source_inputs.as_slice().map_err(|_| {
            PyValueError::new_err("source_inputs must be a C-contiguous float64 array")
        })?;
        let waveforms = transient::Waveforms::new(source_inputs, input_shape.0, input_shape.1)
            .ok_or_else(|| PyValueError::new_err("invalid source_inputs shape"))?;
        transient::validate_adaptive_input(&self.problem, initial, source_times, waveforms)
            .map_err(core_error)?;
        let problem = self.problem.clone();
        let state_width = problem.size;
        let input_width = input_shape.0;
        let result = py.detach(move || {
            transient::solve_adaptive_gear2(
                &problem,
                initial,
                source_times,
                waveforms,
                transient::AdaptiveOptions {
                    newton: transient::NewtonOptions {
                        max_iterations,
                        step_limit,
                        voltage_tolerance,
                        fallback_accept,
                        fallback_tolerance,
                        clip_lo,
                        clip_hi,
                        gmin,
                        hh,
                        cap_mode,
                    },
                    max_step,
                    reltol,
                    voltage_abstol,
                    current_abstol,
                    max_steps,
                    initial_step,
                    profile,
                },
            )
        });
        let times = result.times.into_pyarray(py);
        let states = rows_into_array2(py, result.states, state_width, "adaptive OTFT states")?;
        let inputs = rows_into_array2(py, result.inputs, input_width, "adaptive OTFT inputs")?;
        let profile = result.profile.to_vec().into_pyarray(py);
        Ok((
            result.completed,
            times,
            states,
            inputs,
            result.substeps,
            result.rejected,
            profile,
        ))
    }
}

fn check(status: c_int, action: &str) -> PyResult<()> {
    if status == OK {
        Ok(())
    } else {
        Err(PyRuntimeError::new_err(format!(
            "BSIM4 {action} failed with status {status}"
        )))
    }
}

/// In-process Berkeley BSIM4.5 device handle. Terminal order is always
/// `(drain, gate, source, bulk)`.
#[pyclass(unsendable)]
struct Bsim4Device {
    handle: *mut CoBsim4,
}

#[pymethods]
impl Bsim4Device {
    #[new]
    #[pyo3(signature = (polarity, temperature=300.15))]
    fn new(polarity: c_int, temperature: f64) -> PyResult<Self> {
        let handle = unsafe { co_bsim4::create(polarity, temperature) };
        if handle.is_null() {
            return Err(PyRuntimeError::new_err("BSIM4 device allocation failed"));
        }
        Ok(Self { handle })
    }

    fn set_model_param(&self, name: &str, value: f64) -> PyResult<()> {
        let cname = CString::new(name)
            .map_err(|_| PyRuntimeError::new_err("parameter name has an interior NUL"))?;
        check(
            unsafe { co_bsim4::set_model(self.handle, cname.as_ptr(), value) },
            "model setup",
        )
    }

    fn set_instance_param(&self, name: &str, value: f64) -> PyResult<()> {
        let cname = CString::new(name)
            .map_err(|_| PyRuntimeError::new_err("parameter name has an interior NUL"))?;
        check(
            unsafe { co_bsim4::set_instance(self.handle, cname.as_ptr(), value) },
            "instance setup",
        )
    }

    fn setup(&self) -> PyResult<()> {
        check(unsafe { co_bsim4::setup(self.handle) }, "setup")
    }

    fn node_count(&self) -> i32 {
        unsafe { co_bsim4::node_count(self.handle) }
    }

    /// Evaluate at one bias; returns `(I[4], G[4][4], Q[4], C[4][4], op[8])`.
    #[pyo3(signature = (drain, gate, source, bulk=0.0))]
    #[allow(clippy::type_complexity)]
    fn eval(
        &self,
        drain: f64,
        gate: f64,
        source: f64,
        bulk: f64,
    ) -> PyResult<(Vec<f64>, Vec<Vec<f64>>, Vec<f64>, Vec<Vec<f64>>, Vec<f64>)> {
        let terminals = [drain, gate, source, bulk];
        let mut currents = [0.0f64; 4];
        let mut conductance = [0.0f64; 16];
        let mut charges = [0.0f64; 4];
        let mut capacitance = [0.0f64; 16];
        let mut op = [0.0f64; 8];
        let status = unsafe {
            co_bsim4::eval(
                self.handle,
                terminals.as_ptr(),
                currents.as_mut_ptr(),
                conductance.as_mut_ptr(),
                charges.as_mut_ptr(),
                capacitance.as_mut_ptr(),
                op.as_mut_ptr(),
            )
        };
        check(status, "evaluation")?;
        Ok((
            currents.to_vec(),
            reshape4(&conductance),
            charges.to_vec(),
            reshape4(&capacitance),
            op.to_vec(),
        ))
    }

    /// Terminal-current cross-spectral density at `frequency_hz`; returns
    /// `(total_real[4][4], total_imag[4][4], flicker_real[4][4], flicker_imag[4][4])`.
    #[allow(clippy::type_complexity)]
    fn noise(
        &self,
        frequency_hz: f64,
    ) -> PyResult<(Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>)> {
        let mut total_real = [0.0f64; 16];
        let mut total_imag = [0.0f64; 16];
        let mut flicker_real = [0.0f64; 16];
        let mut flicker_imag = [0.0f64; 16];
        let status = unsafe {
            co_bsim4::noise(
                self.handle,
                frequency_hz,
                total_real.as_mut_ptr(),
                total_imag.as_mut_ptr(),
                flicker_real.as_mut_ptr(),
                flicker_imag.as_mut_ptr(),
            )
        };
        check(status, "noise evaluation")?;
        Ok((
            reshape4(&total_real),
            reshape4(&total_imag),
            reshape4(&flicker_real),
            reshape4(&flicker_imag),
        ))
    }
}

impl Drop for Bsim4Device {
    fn drop(&mut self) {
        unsafe { co_bsim4::destroy(self.handle) };
    }
}

/// Reshape a row-major 16-element buffer into a 4x4 list-of-lists.
fn reshape4(flat: &[f64; 16]) -> Vec<Vec<f64>> {
    (0..4).map(|r| flat[r * 4..r * 4 + 4].to_vec()).collect()
}

// ---------------------------------------------------------------------------
// SPICE expression engine (co-spice) — parity surface.
//
// A thin PyO3 wrapper over `co_spice::ScopeInner`, exposed for differential
// verification against `circuitopt.spice.EvaluationScope`. This is not wired
// into production (the Python expression path stays live); downstream `co-pdk`
// will consume the Rust evaluator directly.
// ---------------------------------------------------------------------------

/// Map a `co-spice` error onto the matching Python exception class by kind.
fn spice_error_to_py(error: SpiceError) -> PyErr {
    match error.kind {
        SpiceErrorKind::Expression => SpiceExpressionError::new_err(error.message),
        SpiceErrorKind::UnknownSymbol => UnknownSymbolError::new_err(error.message),
        SpiceErrorKind::ParameterCycle => ParameterCycleError::new_err(error.message),
        SpiceErrorKind::Syntax => SpiceSyntaxError::new_err(error.message),
        SpiceErrorKind::Elaboration => SpiceElaborationError::new_err(error.message),
    }
}

/// Lower-case the keys of an optional seed map, mirroring the Python
/// `EvaluationScope` constructor (`{str(name).lower(): float(value)}`).
fn lower_keys(values: Option<HashMap<String, f64>>) -> HashMap<String, f64> {
    values
        .map(|map| {
            map.into_iter()
                .map(|(name, value)| (name.to_lowercase(), value))
                .collect()
        })
        .unwrap_or_default()
}

/// Case-insensitive lazy HSPICE parameter scope — the Rust analogue of
/// `circuitopt.spice.EvaluationScope`.
///
/// Every method takes `&self` (interior mutability) and releases the GIL for
/// the core work, so a single scope can be shared and resolved concurrently.
#[pyclass]
struct SpiceScope {
    inner: Arc<ScopeInner>,
}

#[pymethods]
impl SpiceScope {
    #[new]
    #[pyo3(signature = (values=None))]
    fn new(values: Option<HashMap<String, f64>>) -> Self {
        Self {
            inner: ScopeInner::new_root(lower_keys(values)),
        }
    }

    /// Bind `name` to a lazily-evaluated expression.
    fn define(&self, name: &str, expression: &str) {
        self.inner.define(name, expression);
    }

    /// Bind `name` to an eager numeric value.
    fn set_value(&self, name: &str, value: f64) {
        self.inner.set_value(name, value);
    }

    /// Register a user-defined parameter function with `formals` -> `expression`.
    #[pyo3(signature = (name, formals, expression))]
    fn define_function(&self, name: &str, formals: Vec<String>, expression: &str) {
        self.inner.define_function(name, &formals, expression);
    }

    /// Resolve a symbol to its numeric value.
    fn resolve_symbol(&self, py: Python<'_>, name: &str) -> PyResult<f64> {
        let inner = self.inner.clone();
        let name = name.to_string();
        py.detach(move || {
            let mut ctx = EvalCtx::new();
            inner.resolve_symbol(&name, &mut ctx)
        })
        .map_err(spice_error_to_py)
    }

    /// Evaluate a free-standing expression in this scope.
    fn evaluate(&self, py: Python<'_>, expression: &str) -> PyResult<f64> {
        let inner = self.inner.clone();
        let expression = expression.to_string();
        py.detach(move || inner.evaluate(&expression))
            .map_err(spice_error_to_py)
    }

    /// Resolve every lazy parameter; return a snapshot of all values.
    fn evaluate_all(&self, py: Python<'_>) -> PyResult<HashMap<String, f64>> {
        let inner = self.inner.clone();
        py.detach(move || inner.evaluate_all())
            .map_err(spice_error_to_py)
    }
}

/// One-shot evaluation: `SpiceScope(values).evaluate(expression)`.
#[pyfunction]
#[pyo3(signature = (expression, values=None))]
fn spice_eval(
    py: Python<'_>,
    expression: &str,
    values: Option<HashMap<String, f64>>,
) -> PyResult<f64> {
    let values = lower_keys(values);
    let expression = expression.to_string();
    py.detach(move || co_spice::spice_eval(&expression, values))
        .map_err(spice_error_to_py)
}

// ---------------------------------------------------------------------------
// SPICE deck parser + elaborator (co-spice) — parity surface.
//
// A thin PyO3 wrapper over the co-spice deck parser and elaborator, exposed for
// differential verification against `circuitopt.spice`. Not wired into
// production; downstream `co-pdk` consumes the Rust deck/elaborator directly.
// Canonical trees mirror the Python dataclass field names 1:1
// (kind/location/text/name/arguments/parameters/terminals/statements/
// subcircuits/sections/top_level/path); `location` is `(path, first, last)`.
// ---------------------------------------------------------------------------

/// Read a SPICE library file as the reference does (`encoding="ascii"`,
/// `errors="strict"`): reject any non-ASCII byte.
fn read_ascii_file(path: &str) -> PyResult<String> {
    let bytes = std::fs::read(path).map_err(|e| PyOSError::new_err(format!("{path}: {e}")))?;
    if let Some(offset) = bytes.iter().position(|b| *b >= 0x80) {
        return Err(PyValueError::new_err(format!(
            "'ascii' codec can't decode byte in {path} at position {offset}"
        )));
    }
    Ok(String::from_utf8(bytes).expect("ascii bytes are valid utf-8"))
}

fn assignment_to_py<'py>(
    py: Python<'py>,
    assignment: &ParameterAssignment,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("name", &assignment.name)?;
    dict.set_item("expression", &assignment.expression)?;
    dict.set_item("formal_parameters", assignment.formal_parameters.clone())?;
    Ok(dict)
}

fn statement_to_py<'py>(py: Python<'py>, statement: &Statement) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("kind", &statement.kind)?;
    dict.set_item(
        "location",
        (
            statement.location.path.as_str(),
            statement.location.first_line,
            statement.location.last_line,
        ),
    )?;
    dict.set_item("text", &statement.text)?;
    dict.set_item("name", statement.name.clone())?;
    dict.set_item("arguments", statement.arguments.clone())?;
    let parameters = statement
        .parameters
        .iter()
        .map(|a| assignment_to_py(py, a))
        .collect::<PyResult<Vec<_>>>()?;
    dict.set_item("parameters", PyList::new(py, parameters)?)?;
    Ok(dict)
}

fn subcircuit_to_py<'py>(py: Python<'py>, sub: &Subcircuit) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("name", &sub.name)?;
    dict.set_item(
        "location",
        (
            sub.location.path.as_str(),
            sub.location.first_line,
            sub.location.last_line,
        ),
    )?;
    dict.set_item("terminals", sub.terminals.clone())?;
    let parameters = sub
        .parameters
        .iter()
        .map(|a| assignment_to_py(py, a))
        .collect::<PyResult<Vec<_>>>()?;
    dict.set_item("parameters", PyList::new(py, parameters)?)?;
    let statements = sub
        .statements
        .iter()
        .map(|s| statement_to_py(py, s))
        .collect::<PyResult<Vec<_>>>()?;
    dict.set_item("statements", PyList::new(py, statements)?)?;
    Ok(dict)
}

fn section_to_py<'py>(py: Python<'py>, section: &LibrarySection) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("name", &section.name)?;
    dict.set_item(
        "location",
        (
            section.location.path.as_str(),
            section.location.first_line,
            section.location.last_line,
        ),
    )?;
    let statements = section
        .statements
        .iter()
        .map(|s| statement_to_py(py, s))
        .collect::<PyResult<Vec<_>>>()?;
    dict.set_item("statements", PyList::new(py, statements)?)?;
    let subcircuits = PyDict::new(py);
    for (key, sub) in section.subcircuits.iter() {
        subcircuits.set_item(key, subcircuit_to_py(py, sub)?)?;
    }
    dict.set_item("subcircuits", subcircuits)?;
    Ok(dict)
}

fn library_to_py<'py>(
    py: Python<'py>,
    library: &SpiceModelLibrary,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("path", &library.path)?;
    dict.set_item("top_level", section_to_py(py, &library.top_level)?)?;
    let sections = PyDict::new(py);
    for (key, section) in library.sections.iter() {
        sections.set_item(key, section_to_py(py, section)?)?;
    }
    dict.set_item("sections", sections)?;
    Ok(dict)
}

fn numeric_model_to_py<'py>(py: Python<'py>, model: &NumericModel) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("name", &model.name)?;
    dict.set_item("model_type", &model.model_type)?;
    let parameters = PyDict::new(py);
    for (key, value) in &model.parameters {
        parameters.set_item(key, *value)?;
    }
    dict.set_item("parameters", parameters)?;
    Ok(dict)
}

/// `parse_spice_number(text)` — a SPICE numeric literal to `f64`.
#[pyfunction]
fn spice_parse_number(py: Python<'_>, text: String) -> PyResult<f64> {
    py.detach(move || parse_spice_number(&text))
        .map_err(spice_error_to_py)
}

/// One logical line: its joined text plus a `(path, first_line, last_line)` tuple.
type LogicalLine = (String, (String, usize, usize));

/// `logical_lines(text, path)` -> `[(text, (path, first, last)), ...]`.
#[pyfunction]
#[pyo3(signature = (text, path=String::from("<string>")))]
fn spice_logical_lines(py: Python<'_>, text: String, path: String) -> PyResult<Vec<LogicalLine>> {
    let lines = py
        .detach(move || logical_lines(&text, &path))
        .map_err(spice_error_to_py)?;
    Ok(lines
        .into_iter()
        .map(|(text, loc)| (text, (loc.path, loc.first_line, loc.last_line)))
        .collect())
}

/// `parse_assignments(text)` -> `[{name, expression, formal_parameters}, ...]`.
#[pyfunction]
fn spice_parse_assignments(py: Python<'_>, text: String) -> PyResult<Bound<'_, PyList>> {
    let parsed = py
        .detach(move || parse_assignments(&text))
        .map_err(spice_error_to_py)?;
    let items = parsed
        .iter()
        .map(|a| assignment_to_py(py, a))
        .collect::<PyResult<Vec<_>>>()?;
    PyList::new(py, items)
}

/// `parse_spice_library(path)` -> canonical library tree.
#[pyfunction]
fn spice_parse_library(py: Python<'_>, path: String) -> PyResult<Bound<'_, PyDict>> {
    let text = read_ascii_file(&path)?;
    let library = py
        .detach(move || parse_spice_library_text(&text, &path))
        .map_err(spice_error_to_py)?;
    library_to_py(py, &library)
}

/// `parse_spice_library_text(text, path)` -> canonical library tree.
#[pyfunction]
#[pyo3(signature = (text, path=String::from("<string>")))]
fn spice_parse_library_text(
    py: Python<'_>,
    text: String,
    path: String,
) -> PyResult<Bound<'_, PyDict>> {
    let library = py
        .detach(move || parse_spice_library_text(&text, &path))
        .map_err(spice_error_to_py)?;
    library_to_py(py, &library)
}

/// `select_library_sections(path, sections)` -> ordered, de-duplicated names.
#[pyfunction]
fn spice_select_sections(
    py: Python<'_>,
    path: String,
    sections: Vec<String>,
) -> PyResult<Vec<String>> {
    let text = read_ascii_file(&path)?;
    py.detach(move || -> Result<Vec<String>, SpiceError> {
        let library = parse_spice_library_text(&text, &path)?;
        let selection = select_library_sections(&library, &sections, true)?;
        Ok(selection.names)
    })
    .map_err(spice_error_to_py)
}

/// `elaborate(path, sections, overrides)` -> `{model_name: {name, model_type,
/// parameters}}` for the section-level `.model` statements.
#[pyfunction]
#[pyo3(signature = (path, sections, overrides=None))]
fn spice_elaborate(
    py: Python<'_>,
    path: String,
    sections: Vec<String>,
    overrides: Option<HashMap<String, f64>>,
) -> PyResult<Bound<'_, PyDict>> {
    let text = read_ascii_file(&path)?;
    let initial = lower_keys(overrides);
    let models = py
        .detach(move || -> Result<Vec<(String, NumericModel)>, SpiceError> {
            let library = parse_spice_library_text(&text, &path)?;
            let elaborated = elaborate_library(&library, &sections, initial, true)?;
            let mut out = Vec::new();
            for (key, statement) in elaborated.models.iter() {
                out.push((key.clone(), elaborated.numeric_model(statement)?));
            }
            Ok(out)
        })
        .map_err(spice_error_to_py)?;
    let dict = PyDict::new(py);
    for (key, model) in &models {
        dict.set_item(key, numeric_model_to_py(py, model)?)?;
    }
    Ok(dict)
}

/// Extract a `Mapping[str, float | str]` into ordered `(name, ParamValue)` pairs.
fn extract_param_values(params: Option<&Bound<'_, PyDict>>) -> PyResult<Vec<(String, ParamValue)>> {
    let mut out = Vec::new();
    if let Some(dict) = params {
        for (key, value) in dict.iter() {
            let name: String = key.extract()?;
            let value = if let Ok(number) = value.extract::<f64>() {
                ParamValue::Num(number)
            } else {
                ParamValue::Str(value.extract::<String>()?)
            };
            out.push((name, value));
        }
    }
    Ok(out)
}

/// `elaborate_instance(path, sections, subckt, params, overrides)` -> the
/// numericized `.model` statements and elements of one subcircuit instance.
#[pyfunction]
#[pyo3(signature = (path, sections, subckt, params=None, overrides=None))]
fn spice_elaborate_instance<'py>(
    py: Python<'py>,
    path: String,
    sections: Vec<String>,
    subckt: String,
    params: Option<Bound<'_, PyDict>>,
    overrides: Option<HashMap<String, f64>>,
) -> PyResult<Bound<'py, PyDict>> {
    let text = read_ascii_file(&path)?;
    let initial = lower_keys(overrides);
    let param_values = extract_param_values(params.as_ref())?;
    // (kind, name, numeric parameters) for one numericized element statement.
    type ElementNumeric = (String, String, HashMap<String, f64>);
    let (models, elements): (Vec<NumericModel>, Vec<ElementNumeric>) = py
        .detach(move || -> Result<_, SpiceError> {
            let library = parse_spice_library_text(&text, &path)?;
            let elaborated = elaborate_library(&library, &sections, initial, true)?;
            let instance = elaborated.instantiate(&subckt, &param_values)?;
            let mut model_out = Vec::new();
            for statement in instance.model_statements() {
                model_out.push(instance.numeric_model(statement, None)?);
            }
            let mut element_out = Vec::new();
            for statement in instance.elements() {
                let parameters = instance.numeric_parameters(statement, None)?;
                element_out.push((
                    statement.kind.clone(),
                    statement.name.clone().unwrap_or_default(),
                    parameters,
                ));
            }
            Ok((model_out, element_out))
        })
        .map_err(spice_error_to_py)?;
    let dict = PyDict::new(py);
    let model_items = models
        .iter()
        .map(|m| numeric_model_to_py(py, m))
        .collect::<PyResult<Vec<_>>>()?;
    dict.set_item("models", PyList::new(py, model_items)?)?;
    let mut element_items = Vec::new();
    for (kind, name, parameters) in &elements {
        let element = PyDict::new(py);
        element.set_item("kind", kind)?;
        element.set_item("name", name)?;
        let params = PyDict::new(py);
        for (key, value) in parameters {
            params.set_item(key, *value)?;
        }
        element.set_item("parameters", params)?;
        element_items.push(element);
    }
    dict.set_item("elements", PyList::new(py, element_items)?)?;
    Ok(dict)
}

// ---------------------------------------------------------------------------
// PDK compilers (co-pdk) — parity surface.
//
// A thin PyO3 wrapper over `co_pdk::CompiledPdk`, exposed for differential
// verification against `circuitopt.pdk.{freepdk45,sky130,tsmc28}`. Not wired
// into production (the Python PDK adapters stay live). D12: only numeric card
// values and path/section identifiers cross the boundary — never card text.
// ---------------------------------------------------------------------------

/// Map a `co-pdk` error onto a Python exception. Errors that originated in
/// `co-spice` re-raise the matching class; PDK-specific model errors (the
/// Python `*ModelError`, all `ValueError` subclasses) become `ValueError`.
fn pdk_error_to_py(error: PdkError) -> PyErr {
    match error.kind {
        Some(kind) => spice_error_to_py(SpiceError {
            kind,
            message: error.message,
        }),
        None => PyValueError::new_err(error.message),
    }
}

fn params_to_py<'py>(
    py: Python<'py>,
    params: &HashMap<String, f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    for (key, value) in params {
        dict.set_item(key, *value)?;
    }
    Ok(dict)
}

fn numeric_card_to_py<'py>(py: Python<'py>, card: &NumericCard) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item(
        "model_parameters",
        params_to_py(py, &card.model_parameters)?,
    )?;
    dict.set_item(
        "instance_parameters",
        params_to_py(py, &card.instance_parameters)?,
    )?;
    dict.set_item("model_name", &card.model_name)?;
    dict.set_item("model_type", &card.model_type)?;
    dict.set_item("source_version", card.source_version)?;
    match &card.bin {
        Some(bin) => {
            let bin_dict = PyDict::new(py);
            bin_dict.set_item("name", &bin.name)?;
            bin_dict.set_item("lmin", bin.lmin)?;
            bin_dict.set_item("lmax", bin.lmax)?;
            bin_dict.set_item("wmin", bin.wmin)?;
            bin_dict.set_item("wmax", bin.wmax)?;
            dict.set_item("bin", bin_dict)?;
        }
        None => dict.set_item("bin", py.None())?,
    }
    let source = PyDict::new(py);
    source.set_item("pdk", &card.source.pdk)?;
    source.set_item("polarity", &card.source.polarity)?;
    source.set_item("corner", &card.source.corner)?;
    source.set_item("path", &card.source.path)?;
    source.set_item("temperature_c", card.source.temperature_c)?;
    source.set_item("macro_name", card.source.macro_name.clone())?;
    source.set_item("bin_name", card.source.bin_name.clone())?;
    dict.set_item("source", source)?;
    Ok(dict)
}

/// An immutable PDK compiler with a thread-safe in-memory card/program cache.
#[pyclass(name = "CompiledPdk")]
struct PyCompiledPdk {
    inner: Arc<PdkCompiled>,
}

#[pymethods]
impl PyCompiledPdk {
    #[new]
    #[pyo3(signature = (pdk, root=None))]
    fn new(pdk: &str, root: Option<String>) -> PyResult<Self> {
        let inner = PdkCompiled::new(pdk, root).map_err(pdk_error_to_py)?;
        Ok(Self {
            inner: Arc::new(inner),
        })
    }

    /// Compile one numeric BSIM4 card.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (polarity, corner, temp_c, w_um=None, l_um=None, nf=1, mult=1, mismatch=None))]
    fn numeric_card<'py>(
        &self,
        py: Python<'py>,
        polarity: String,
        corner: String,
        temp_c: f64,
        w_um: Option<f64>,
        l_um: Option<f64>,
        nf: i64,
        mult: i64,
        mismatch: Option<f64>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let inner = self.inner.clone();
        let card = py
            .detach(move || {
                inner.numeric_card(&polarity, &corner, temp_c, w_um, l_um, nf, mult, mismatch)
            })
            .map_err(pdk_error_to_py)?;
        numeric_card_to_py(py, &card)
    }
}

#[pymodule]
fn circuitopt_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(engine_info, m)?)?;
    m.add_function(wrap_pyfunction!(bsim4_eval_vp_address, m)?)?;
    m.add_function(wrap_pyfunction!(otft_device_batch, m)?)?;
    m.add_function(wrap_pyfunction!(otft_terminal_derivatives_batch, m)?)?;
    m.add_function(wrap_pyfunction!(otft_newton_batch, m)?)?;
    m.add_function(wrap_pyfunction!(mna_term_values, m)?)?;
    m.add_function(wrap_pyfunction!(dense_neg_solve, m)?)?;
    m.add_function(wrap_pyfunction!(periodic_hb_blocks, m)?)?;
    m.add_function(wrap_pyfunction!(periodic_fold_psd, m)?)?;
    m.add_class::<OtftModel>()?;
    m.add_class::<Bsim4Device>()?;
    m.add_class::<OtftTransientProblem>()?;
    m.add_class::<Bsim4TransientProblem>()?;
    m.add_class::<LtiProblem>()?;
    m.add_class::<PeriodicLinearizationProblem>()?;
    // SPICE expression engine parity surface.
    m.add(
        "SpiceExpressionError",
        m.py().get_type::<SpiceExpressionError>(),
    )?;
    m.add(
        "UnknownSymbolError",
        m.py().get_type::<UnknownSymbolError>(),
    )?;
    m.add(
        "ParameterCycleError",
        m.py().get_type::<ParameterCycleError>(),
    )?;
    m.add("SpiceSyntaxError", m.py().get_type::<SpiceSyntaxError>())?;
    m.add(
        "SpiceElaborationError",
        m.py().get_type::<SpiceElaborationError>(),
    )?;
    m.add_class::<SpiceScope>()?;
    m.add_function(wrap_pyfunction!(spice_eval, m)?)?;
    // SPICE deck parser + elaborator parity surface.
    m.add_function(wrap_pyfunction!(spice_parse_number, m)?)?;
    m.add_function(wrap_pyfunction!(spice_logical_lines, m)?)?;
    m.add_function(wrap_pyfunction!(spice_parse_assignments, m)?)?;
    m.add_function(wrap_pyfunction!(spice_parse_library, m)?)?;
    m.add_function(wrap_pyfunction!(spice_parse_library_text, m)?)?;
    m.add_function(wrap_pyfunction!(spice_select_sections, m)?)?;
    m.add_function(wrap_pyfunction!(spice_elaborate, m)?)?;
    m.add_function(wrap_pyfunction!(spice_elaborate_instance, m)?)?;
    // PDK compilers parity surface.
    m.add_class::<PyCompiledPdk>()?;
    Ok(())
}
