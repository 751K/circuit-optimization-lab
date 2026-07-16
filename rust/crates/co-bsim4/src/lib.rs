//! co-bsim4 — Rust host for the Berkeley BSIM4.5 compact model (R1 scaffold).
//!
//! R1 (this commit): placeholder only. No C is vendored yet.
//!
//! R2 will vendor the Berkeley BSIM4.5 reference C — the same sources that
//! `circuitopt/compact_models/bsim4/native_src/` already builds for the Python
//! extension — and compile them from a `build.rs`, then port the host adapter
//! layer (parameter binding, terminal load/eval, charge & conductance
//! extraction) to Rust so the model can be evaluated without CPython in the
//! inner loop.

/// Canonical crate version, surfaced to Python through `engine_info()`.
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Placeholder terminal-eval probe: proves the crate compiles, links, and is
/// callable across the workspace. R2 replaces it with the BSIM4 device
/// evaluation entry point.
pub fn model_probe(x: f64) -> f64 {
    x + 1.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_is_nonempty() {
        assert!(!version().is_empty());
    }

    #[test]
    fn model_probe_offsets_by_one() {
        assert_eq!(model_probe(1.0), 2.0);
    }
}
