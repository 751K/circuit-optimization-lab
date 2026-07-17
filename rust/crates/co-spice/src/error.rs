//! Error type mirroring the Python exception hierarchy.
//!
//! The Python reference raises five exception classes, all `ValueError`
//! subclasses:
//!
//! * `SpiceExpressionError` — base parse/evaluation failure (expressions).
//! * `UnknownSymbolError`   — a symbol or function is absent from scope.
//! * `ParameterCycleError`  — a lazy-parameter dependency cycle.
//! * `SpiceSyntaxError`     — malformed structural SPICE syntax (deck parser).
//! * `SpiceElaborationError`— a structurally valid library cannot elaborate.
//!
//! We carry the same distinction in [`ErrorKind`] so the `co-py` boundary can
//! map each variant to the matching Python class by exact name.

use std::error::Error;
use std::fmt;

/// Which Python exception a [`SpiceError`] maps to at the `co-py` boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorKind {
    /// Maps to `SpiceExpressionError` (the base; a `ValueError` subclass).
    Expression,
    /// Maps to `UnknownSymbolError` (a `SpiceExpressionError` subclass).
    UnknownSymbol,
    /// Maps to `ParameterCycleError` (a `SpiceExpressionError` subclass).
    ParameterCycle,
    /// Maps to `SpiceSyntaxError` (a direct `ValueError` subclass).
    Syntax,
    /// Maps to `SpiceElaborationError` (a direct `ValueError` subclass).
    Elaboration,
}

/// A parse/evaluation failure carrying its target Python exception class.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpiceError {
    pub kind: ErrorKind,
    pub message: String,
}

impl SpiceError {
    pub fn expression(message: impl Into<String>) -> Self {
        Self {
            kind: ErrorKind::Expression,
            message: message.into(),
        }
    }

    pub fn unknown(message: impl Into<String>) -> Self {
        Self {
            kind: ErrorKind::UnknownSymbol,
            message: message.into(),
        }
    }

    pub fn cycle(message: impl Into<String>) -> Self {
        Self {
            kind: ErrorKind::ParameterCycle,
            message: message.into(),
        }
    }

    pub fn syntax(message: impl Into<String>) -> Self {
        Self {
            kind: ErrorKind::Syntax,
            message: message.into(),
        }
    }

    pub fn elaboration(message: impl Into<String>) -> Self {
        Self {
            kind: ErrorKind::Elaboration,
            message: message.into(),
        }
    }
}

impl fmt::Display for SpiceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl Error for SpiceError {}

/// Convenience result alias used throughout the crate.
pub type SpiceResult<T> = Result<T, SpiceError>;
