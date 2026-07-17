//! co-core — CircuitOpt numerical solver core (R1 scaffold).
//!
//! R1 (this commit): placeholder only. The crate exists so the workspace
//! links end-to-end and `co-py` has a real path dependency to call.
//!
//! R3 will port the numerical hot paths here — the MNA stamping, the Newton
//! loop, and the transient integration currently living in
//! `circuitopt/numba_kernels.py` and the `*_solver.py` modules — as
//! allocation-light Rust behind a stable, FFI-friendly surface that `co-py`
//! can dispatch to in place of the numba kernels.

/// Canonical crate version, surfaced to Python through `engine_info()`.
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Placeholder numeric probe: proves the crate compiles, links, and is
/// callable across the workspace. R3 replaces it with the real solver entry
/// points (stamp / solve / step).
pub fn core_probe(x: f64) -> f64 {
    2.0 * x + 1.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_is_nonempty() {
        assert!(!version().is_empty());
    }

    #[test]
    fn core_probe_is_affine() {
        assert_eq!(core_probe(2.0), 5.0);
    }
}
