//! co-py — the `circuitopt_core` PyO3 extension module (R1 scaffold).
//!
//! R1 (this commit): exposes `engine_info()` so the Python side
//! (`circuitopt/_engine.py`) can (a) confirm the compiled core is importable
//! when `CIRCUIT_ENGINE=rust`, and (b) report build metadata. The workspace
//! link is exercised for real — `engine_info()` calls into both `co-core` and
//! `co-bsim4`.
//!
//! The numeric path still runs through numba in R1; R3 replaces that dispatch
//! with calls into `co-core` / `co-bsim4`.

use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Touch both path-dependency crates so the workspace link is real, not
/// nominal. Returns a deterministic scalar the Python tests can pin.
fn workspace_probe() -> f64 {
    co_core::core_probe(co_bsim4::model_probe(1.0))
}

/// Build/runtime metadata for the compiled core.
///
/// Keys: `version` (Cargo package version), `profile` ("release" | "debug"),
/// `rustc` (toolchain that built the wheel), plus the two placeholder crate
/// versions and the workspace probe value.
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
    info.set_item("probe", workspace_probe())?;
    Ok(info)
}

#[pymodule]
fn circuitopt_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(engine_info, m)?)?;
    Ok(())
}
