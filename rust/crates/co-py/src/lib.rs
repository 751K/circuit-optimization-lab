//! co-py — the `circuitopt_core` PyO3 extension module (R2).
//!
//! Two surfaces live here:
//!
//! * A **C ABI** (`co_bsim4_*`, `#[unsafe(no_mangle)] extern "C"`) that is
//!   signature- and semantics-identical to the historical `host.c` entry points.
//!   `circuitopt/compact_models/bsim4/native.py` loads this compiled module with
//!   `ctypes.CDLL(circuitopt_core.__file__)` and binds these symbols exactly as
//!   it binds the runtime-`cc` library, so the Numba transient bridge picks up
//!   `co_bsim4_eval_vp` as a plain machine-word function pointer.
//! * A **PyO3** surface: `engine_info()` (build metadata + BSIM4 ABI),
//!   `bsim4_eval_vp_address()` (the raw address of the eval entry), and a
//!   `Bsim4Device` convenience class.
//!
//! The BSIM4 logic itself lives in `co-bsim4`; this crate is a thin boundary
//! that catches panics so nothing unwinds across the FFI edge.

// The C ABI shims below are `unsafe extern "C"` and forward raw pointers; an
// inner `unsafe {}` per dereference would only add noise. Their contract is the
// documented host.c ABI (valid handle, correctly sized output buffers), so the
// per-function `# Safety` sections would be pure boilerplate.
#![allow(unsafe_op_in_unsafe_fn)]
#![allow(clippy::missing_safety_doc)]

use std::ffi::CString;
use std::os::raw::{c_char, c_double, c_int, c_uint, c_void};
use std::panic::catch_unwind;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use co_bsim4::CoBsim4;

const OK: c_int = 0;
/// Vague internal error (ngspice `E_PANIC`); also what we return if a Rust panic
/// is caught at the FFI boundary.
const E_PANIC: c_int = 1;

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

#[pymodule]
fn circuitopt_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(engine_info, m)?)?;
    m.add_function(wrap_pyfunction!(bsim4_eval_vp_address, m)?)?;
    m.add_class::<Bsim4Device>()?;
    Ok(())
}
