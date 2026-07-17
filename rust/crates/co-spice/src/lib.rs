//! co-spice — the HSPICE parameter expression engine.
//!
//! A 1:1 Rust port of `circuitopt/spice/expressions.py` (the frozen Python
//! reference). It provides:
//!
//! * a SPICE-number aware lexer ([`tokenize`]),
//! * a Pratt parser producing an immutable AST ([`Expr`]) behind a thread-safe
//!   compile cache ([`compile_expression`]),
//! * a deterministic evaluator and a case-insensitive lazy parameter scope
//!   ([`ScopeInner`]) with user-defined functions, cycle detection, and lexical
//!   parent fallback,
//! * a convenience one-shot evaluator ([`spice_eval`]).
//!
//! Errors carry an [`ErrorKind`] so the PyO3 boundary in `co-py` can raise the
//! matching Python class (`SpiceExpressionError` / `UnknownSymbolError` /
//! `ParameterCycleError`).
//!
//! The port keeps the reference's operation order (no fast-math, no reordering)
//! so results match bit-for-bit, except libm `pow` where the gate is a relative
//! error of 1e-14. Downstream (`co-pdk`) will consume this evaluator entirely
//! within Rust; the Python expression path stays in production for now.

mod ast;
mod error;
mod lexer;
mod parser;
mod scope;

pub use ast::{BinaryOp, Expr, UnaryOp};
pub use error::{ErrorKind, SpiceError, SpiceResult};
pub use lexer::{Token, TokenKind, spice_number_value, tokenize};
pub use parser::compile_expression;
pub use scope::{EvalCtx, ScopeInner, spice_eval};
