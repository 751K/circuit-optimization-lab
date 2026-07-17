//! co-core - CircuitOpt numerical solver core.
//!
//! The crate owns the AT4000TG OTFT device equations, terminal resolution,
//! dense MNA/GEPP, circuit Newton, fixed and adaptive transient integration,
//! complex LTI solves, and the BSIM4 fixed-grid orchestration used by `co-py`.

pub mod bsim_transient;
pub mod error;
pub mod lti;
pub mod mna;
pub mod otft;
pub mod transient;

pub use error::CoreError;

/// Canonical crate version, surfaced to Python through `engine_info()`.
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Minimal numeric ABI probe retained for build and linkage diagnostics.
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
