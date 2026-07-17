//! Evaluator, lazy scope, and built-in functions.
//!
//! A 1:1 port of the Python `EvaluationScope`, `Expression.evaluate` methods,
//! and `_builtin`. The delicate behaviours are reproduced deliberately:
//!
//! * `&&` / `||` short-circuit; the left operand is always evaluated first.
//! * Booleans are `0.0` / `1.0`; truthiness is `value != 0.0`.
//! * Division by zero and `pow` domain/overflow/`0**-n` cases raise *eagerly*,
//!   matching Python's `ZeroDivisionError` / `OverflowError` — so an
//!   intermediate `inf`/`NaN` cannot be silently swallowed by a comparison.
//! * `math.sqrt`/`exp`/`log10` domain/range errors raise eagerly too.
//! * Non-finite results are rejected only at the two Python boundaries
//!   (`evaluate` and a lazy parameter's resolution), never mid-expression.
//! * Cycle detection falls back to the lexical parent before declaring a cycle
//!   (the `w=w` / `param=param+delta` override idiom).
//!
//! Thread-safety: the reference relies on the GIL. Here each scope guards its
//! caches with locks and the per-evaluation "resolving" stack lives on the
//! caller's stack ([`EvalCtx`]), keyed by scope identity, so concurrent
//! resolution of a shared scope is race-free and cycle detection stays
//! per-thread.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};

use crate::ast::{BinaryOp, Expr, UnaryOp};
use crate::error::{ErrorKind, SpiceError, SpiceResult};
use crate::parser::compile_expression;

#[inline]
fn truthy(value: f64) -> bool {
    value != 0.0
}

#[inline]
fn boolean(flag: bool) -> f64 {
    if flag { 1.0 } else { 0.0 }
}

/// Per-evaluation resolution state, threaded through the call chain instead of
/// being stored on the (shared) scope. Entries are `(scope_id, key)` so the same
/// parameter name in different scopes never collides — mirroring the reference's
/// per-scope `_resolving` list, but without shared mutable state.
pub struct EvalCtx {
    stack: Vec<(usize, String)>,
}

impl Default for EvalCtx {
    fn default() -> Self {
        Self::new()
    }
}

impl EvalCtx {
    pub fn new() -> Self {
        Self { stack: Vec::new() }
    }

    fn is_resolving(&self, scope_id: usize, key: &str) -> bool {
        self.stack
            .iter()
            .any(|(id, name)| *id == scope_id && name == key)
    }

    fn push(&mut self, scope_id: usize, key: String) {
        self.stack.push((scope_id, key));
    }

    fn pop(&mut self) {
        self.stack.pop();
    }

    /// Build the `a -> b -> a` chain from the current scope's substack, matching
    /// `self._resolving[first:] + [key]` in the reference.
    fn cycle_chain(&self, scope_id: usize, key: &str) -> String {
        let substack: Vec<&String> = self
            .stack
            .iter()
            .filter(|(id, _)| *id == scope_id)
            .map(|(_, name)| name)
            .collect();
        let first = substack
            .iter()
            .position(|name| name.as_str() == key)
            .unwrap_or(0);
        let mut chain: Vec<String> = substack[first..]
            .iter()
            .map(|name| (*name).clone())
            .collect();
        chain.push(key.to_string());
        chain.join(" -> ")
    }
}

/// A case-insensitive lazy parameter scope with user-defined functions.
///
/// Held behind an `Arc` so child scopes (function-call frames) can reference
/// their lexical parent and so the scope can be shared across threads.
pub struct ScopeInner {
    parent: Option<Arc<ScopeInner>>,
    values: Mutex<HashMap<String, f64>>,
    expressions: RwLock<HashMap<String, String>>,
    function_defs: RwLock<HashMap<String, (Vec<String>, String)>>,
}

impl ScopeInner {
    /// Create a root scope seeded with eager values (keys already lower-cased by
    /// the caller, mirroring the Python constructor).
    pub fn new_root(values: HashMap<String, f64>) -> Arc<Self> {
        Arc::new(Self {
            parent: None,
            values: Mutex::new(values),
            expressions: RwLock::new(HashMap::new()),
            function_defs: RwLock::new(HashMap::new()),
        })
    }

    pub(crate) fn new_child(parent: Arc<ScopeInner>, values: HashMap<String, f64>) -> Arc<Self> {
        Arc::new(Self {
            parent: Some(parent),
            values: Mutex::new(values),
            expressions: RwLock::new(HashMap::new()),
            function_defs: RwLock::new(HashMap::new()),
        })
    }

    fn id(self: &Arc<Self>) -> usize {
        Arc::as_ptr(self) as *const () as usize
    }

    /// Define a lazy parameter (`define`): store the expression, drop any cached
    /// value for the name.
    pub fn define(&self, name: &str, expression: &str) {
        let key = name.to_lowercase();
        self.expressions
            .write()
            .unwrap_or_else(|p| p.into_inner())
            .insert(key.clone(), expression.to_string());
        self.values
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .remove(&key);
    }

    /// Set an eager value (`set_value`): drop any lazy expression for the name.
    pub fn set_value(&self, name: &str, value: f64) {
        let key = name.to_lowercase();
        self.values
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .insert(key.clone(), value);
        self.expressions
            .write()
            .unwrap_or_else(|p| p.into_inner())
            .remove(&key);
    }

    /// Register a user-defined parameter function (`define_function`). Formal
    /// parameter names are lower-cased, like the reference.
    pub fn define_function(&self, name: &str, formals: &[String], expression: &str) {
        let key = name.to_lowercase();
        let formals: Vec<String> = formals.iter().map(|f| f.to_lowercase()).collect();
        self.function_defs
            .write()
            .unwrap_or_else(|p| p.into_inner())
            .insert(key, (formals, expression.to_string()));
    }

    /// Resolve a symbol to its numeric value.
    pub fn resolve_symbol(self: &Arc<Self>, name: &str, ctx: &mut EvalCtx) -> SpiceResult<f64> {
        let key = name.to_lowercase();

        if let Some(value) = self
            .values
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .get(&key)
            .copied()
        {
            return Ok(value);
        }

        let expression = self
            .expressions
            .read()
            .unwrap_or_else(|p| p.into_inner())
            .get(&key)
            .cloned();
        if let Some(expression) = expression {
            let scope_id = self.id();
            if ctx.is_resolving(scope_id, &key) {
                // SPICE instance/model overrides commonly use `w=w` or
                // `param=param+delta`: the RHS refers to the enclosing scope.
                // Fall back lexically before declaring a real local cycle.
                if let Some(parent) = &self.parent {
                    match parent.resolve_symbol(name, ctx) {
                        Ok(value) => return Ok(value),
                        Err(err) if err.kind == ErrorKind::UnknownSymbol => {}
                        Err(err) => return Err(err),
                    }
                }
                return Err(SpiceError::cycle(format!(
                    "parameter dependency cycle: {}",
                    ctx.cycle_chain(scope_id, &key)
                )));
            }

            ctx.push(scope_id, key.clone());
            let evaluated =
                compile_expression(&expression).and_then(|tree| evaluate(&tree, self, ctx));
            ctx.pop();
            let value = evaluated?;

            if !value.is_finite() {
                return Err(SpiceError::expression(format!(
                    "parameter {name:?} evaluated to non-finite value {value}"
                )));
            }
            // `setdefault`: under concurrent resolution the first completed
            // (deterministic) value wins; every writer computes the same number.
            let stored = *self
                .values
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .entry(key)
                .or_insert(value);
            return Ok(stored);
        }

        if let Some(parent) = &self.parent {
            return parent.resolve_symbol(name, ctx);
        }

        match key.as_str() {
            "pi" => Ok(std::f64::consts::PI),
            "e" => Ok(std::f64::consts::E),
            "true" => Ok(1.0),
            "false" => Ok(0.0),
            _ => Err(SpiceError::unknown(format!(
                "unknown HSPICE symbol {name:?}"
            ))),
        }
    }

    /// Invoke a function: user definitions first, then the lexical parent, then
    /// built-ins — exactly the reference's `call_function` chain.
    pub fn call_function(
        self: &Arc<Self>,
        name: &str,
        arguments: &[f64],
        ctx: &mut EvalCtx,
    ) -> SpiceResult<f64> {
        let key = name.to_lowercase();

        let definition = self
            .function_defs
            .read()
            .unwrap_or_else(|p| p.into_inner())
            .get(&key)
            .cloned();
        if let Some((formals, expression)) = definition {
            if arguments.len() != formals.len() {
                return Err(SpiceError::expression(format!(
                    "{} expects {} arguments, received {}",
                    name,
                    formals.len(),
                    arguments.len()
                )));
            }
            let child_values: HashMap<String, f64> =
                formals.into_iter().zip(arguments.iter().copied()).collect();
            let child = ScopeInner::new_child(self.clone(), child_values);
            let tree = compile_expression(&expression)?;
            return evaluate(&tree, &child, ctx);
        }

        if let Some(parent) = &self.parent {
            match parent.call_function(name, arguments, ctx) {
                Ok(value) => return Ok(value),
                Err(err) if err.kind == ErrorKind::UnknownSymbol => {}
                Err(err) => return Err(err),
            }
        }

        builtin(name, arguments)
    }

    /// Evaluate a free-standing expression in this scope (`evaluate`).
    pub fn evaluate(self: &Arc<Self>, expression: &str) -> SpiceResult<f64> {
        let mut ctx = EvalCtx::new();
        let tree = compile_expression(expression)?;
        let value = evaluate(&tree, self, &mut ctx)?;
        if !value.is_finite() {
            return Err(SpiceError::expression(format!(
                "expression {expression:?} evaluated to non-finite value {value}"
            )));
        }
        Ok(value)
    }

    /// Resolve every lazy parameter and return a snapshot of all values
    /// (`evaluate_all`).
    pub fn evaluate_all(self: &Arc<Self>) -> SpiceResult<HashMap<String, f64>> {
        let names: Vec<String> = self
            .expressions
            .read()
            .unwrap_or_else(|p| p.into_inner())
            .keys()
            .cloned()
            .collect();
        for name in names {
            let mut ctx = EvalCtx::new();
            self.resolve_symbol(&name, &mut ctx)?;
        }
        Ok(self
            .values
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clone())
    }
}

/// Evaluate one AST node against a scope.
fn evaluate(expr: &Expr, scope: &Arc<ScopeInner>, ctx: &mut EvalCtx) -> SpiceResult<f64> {
    match expr {
        Expr::Number(value) => Ok(*value),
        Expr::Name(name) => scope.resolve_symbol(name, ctx),
        Expr::Unary { op, operand } => {
            let value = evaluate(operand, scope, ctx)?;
            Ok(match op {
                UnaryOp::Pos => value,
                UnaryOp::Neg => -value,
                UnaryOp::Not => boolean(!truthy(value)),
            })
        }
        Expr::Binary { op, left, right } => {
            let left_value = evaluate(left, scope, ctx)?;
            match op {
                BinaryOp::And => {
                    if !truthy(left_value) {
                        Ok(0.0)
                    } else {
                        Ok(boolean(truthy(evaluate(right, scope, ctx)?)))
                    }
                }
                BinaryOp::Or => {
                    if truthy(left_value) {
                        Ok(1.0)
                    } else {
                        Ok(boolean(truthy(evaluate(right, scope, ctx)?)))
                    }
                }
                _ => {
                    let right_value = evaluate(right, scope, ctx)?;
                    match op {
                        BinaryOp::Add => Ok(left_value + right_value),
                        BinaryOp::Sub => Ok(left_value - right_value),
                        BinaryOp::Mul => Ok(left_value * right_value),
                        BinaryOp::Div => {
                            if right_value == 0.0 {
                                Err(SpiceError::expression("float division by zero"))
                            } else {
                                Ok(left_value / right_value)
                            }
                        }
                        BinaryOp::Pow => spice_pow(left_value, right_value),
                        BinaryOp::Lt => Ok(boolean(left_value < right_value)),
                        BinaryOp::Le => Ok(boolean(left_value <= right_value)),
                        BinaryOp::Gt => Ok(boolean(left_value > right_value)),
                        BinaryOp::Ge => Ok(boolean(left_value >= right_value)),
                        BinaryOp::Eq => Ok(boolean(left_value == right_value)),
                        BinaryOp::Ne => Ok(boolean(left_value != right_value)),
                        BinaryOp::And | BinaryOp::Or => unreachable!(),
                    }
                }
            }
        }
        Expr::Conditional {
            condition,
            when_true,
            when_false,
        } => {
            if truthy(evaluate(condition, scope, ctx)?) {
                evaluate(when_true, scope, ctx)
            } else {
                evaluate(when_false, scope, ctx)
            }
        }
        Expr::Call { name, arguments } => {
            let mut values = Vec::with_capacity(arguments.len());
            for argument in arguments {
                values.push(evaluate(argument, scope, ctx)?);
            }
            scope.call_function(name, &values, ctx)
        }
    }
}

/// `left ** right` with the reference's Python-`float.__pow__` semantics:
///
/// * `x ** 0 == 1` for every `x`;
/// * a zero base with a negative exponent raises (Python `ZeroDivisionError`);
/// * a negative base with a fractional exponent is complex in Python — there it
///   propagates as a `TypeError` from `float(complex)`; we raise a
///   `SpiceExpressionError` (both raise; see the crate notes on this one corner);
/// * a finite-operand overflow to `inf` raises (Python `OverflowError`).
fn spice_pow(base: f64, exponent: f64) -> SpiceResult<f64> {
    if exponent == 0.0 {
        return Ok(1.0);
    }
    if base == 0.0 {
        if exponent < 0.0 {
            return Err(SpiceError::expression(
                "0.0 cannot be raised to a negative power",
            ));
        }
        return Ok(base.powf(exponent));
    }
    if base < 0.0 && exponent.fract() != 0.0 {
        return Err(SpiceError::expression(
            "negative number cannot be raised to a fractional power",
        ));
    }
    let result = base.powf(exponent);
    if result.is_infinite() && base.is_finite() && exponent.is_finite() {
        return Err(SpiceError::expression("math range error"));
    }
    Ok(result)
}

fn require_arity(name: &str, arguments: &[f64], count: usize) -> SpiceResult<()> {
    if arguments.len() != count {
        return Err(SpiceError::expression(format!(
            "{} expects {} arguments, received {}",
            name,
            count,
            arguments.len()
        )));
    }
    Ok(())
}

/// Wrap a libm-style unary result the way CPython's `math_1` does: a `NaN` from
/// a non-`NaN` input is a domain error; an `inf` from a finite input is a range
/// error. Both map to `SpiceExpressionError` (the reference distinguishes
/// `ValueError` vs `OverflowError`, but both are caught into the same class).
fn checked_math(x: f64, result: f64) -> SpiceResult<f64> {
    if result.is_nan() && !x.is_nan() {
        return Err(SpiceError::expression("math domain error"));
    }
    if result.is_infinite() && x.is_finite() {
        return Err(SpiceError::expression("math range error"));
    }
    Ok(result)
}

/// The `_builtin` table. Names are matched case-insensitively.
fn builtin(name: &str, arguments: &[f64]) -> SpiceResult<f64> {
    let key = name.to_lowercase();
    match key.as_str() {
        "abs" => {
            require_arity(name, arguments, 1)?;
            Ok(arguments[0].abs())
        }
        "sqrt" => {
            require_arity(name, arguments, 1)?;
            checked_math(arguments[0], arguments[0].sqrt())
        }
        "exp" => {
            require_arity(name, arguments, 1)?;
            checked_math(arguments[0], arguments[0].exp())
        }
        "log10" => {
            require_arity(name, arguments, 1)?;
            checked_math(arguments[0], arguments[0].log10())
        }
        "int" => {
            require_arity(name, arguments, 1)?;
            let x = arguments[0];
            if !x.is_finite() {
                // Python: math.trunc(inf|nan) raises Overflow/ValueError.
                return Err(SpiceError::expression(
                    "cannot truncate non-finite value to an integer",
                ));
            }
            let truncated = x.trunc();
            // `float(math.trunc(x))` yields +0.0 for a truncated-to-zero result;
            // `f64::trunc` keeps -0.0, so normalise the sign of zero.
            Ok(if truncated == 0.0 { 0.0 } else { truncated })
        }
        "sgn" => {
            require_arity(name, arguments, 1)?;
            let x = arguments[0];
            Ok((i32::from(x > 0.0) - i32::from(x < 0.0)) as f64)
        }
        "max" | "min" => {
            if arguments.is_empty() {
                return Err(SpiceError::expression(format!(
                    "{name} expects at least one argument"
                )));
            }
            let mut best = arguments[0];
            if key == "max" {
                for &value in &arguments[1..] {
                    if value > best {
                        best = value;
                    }
                }
            } else {
                for &value in &arguments[1..] {
                    if value < best {
                        best = value;
                    }
                }
            }
            Ok(best)
        }
        "selmin" => {
            if arguments.is_empty() {
                return Err(SpiceError::expression(
                    "selmin expects at least one argument",
                ));
            }
            let mut best = arguments[0];
            for &value in &arguments[1..] {
                if value < best {
                    best = value;
                }
            }
            Ok(best)
        }
        "pwr" | "pow" => {
            require_arity(name, arguments, 2)?;
            spice_pow(arguments[0], arguments[1])
        }
        "agauss" => {
            if arguments.len() != 2 && arguments.len() != 3 {
                return Err(SpiceError::expression(format!(
                    "agauss expects 2 or 3 arguments, received {}",
                    arguments.len()
                )));
            }
            // Native PVT elaboration is deterministic nominal evaluation.
            Ok(arguments[0])
        }
        _ => Err(SpiceError::unknown(format!(
            "unknown HSPICE function {name:?}"
        ))),
    }
}

/// Convenience: evaluate one expression in a fresh root scope seeded with
/// `values` (already lower-cased by the caller). Equivalent to
/// `ScopeInner::new_root(values).evaluate(expression)`.
pub fn spice_eval(expression: &str, values: HashMap<String, f64>) -> SpiceResult<f64> {
    ScopeInner::new_root(values).evaluate(expression)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn root() -> Arc<ScopeInner> {
        ScopeInner::new_root(HashMap::new())
    }

    fn eval(expr: &str) -> SpiceResult<f64> {
        root().evaluate(expr)
    }

    fn ok(expr: &str) -> f64 {
        eval(expr).unwrap()
    }

    #[test]
    fn precedence_and_associativity() {
        assert_eq!(ok("1 + 2 * 3"), 7.0);
        assert_eq!(ok("2**3**2"), 512.0);
        assert_eq!(ok("2.5k + 500"), 3000.0);
        assert_eq!(ok("(1 + 2) * 3"), 9.0);
        assert_eq!(ok("1 - 2 - 3"), -4.0);
    }

    #[test]
    fn conditionals() {
        assert_eq!(ok("1 ? 2 : 3"), 2.0);
        assert_eq!(ok("0 ? 2 : 3"), 3.0);
        assert_eq!(ok("-2 < -1 ? max(4, 3) : 0"), 4.0);
        // Deeply nested, right-associative ternary.
        assert_eq!(ok("0 ? 1 : 0 ? 2 : 1 ? 3 : 4"), 3.0);
        // The untaken branch is never evaluated (would divide by zero).
        assert_eq!(ok("1 ? 5 : 1/0"), 5.0);
        assert_eq!(ok("0 ? 1/0 : 5"), 5.0);
    }

    #[test]
    fn boolean_semantics() {
        assert_eq!(ok("!0"), 1.0);
        assert_eq!(ok("!5"), 0.0);
        // Negative numbers are truthy.
        assert_eq!(ok("!(-3)"), 0.0);
        assert_eq!(ok("2 && 3"), 1.0);
        assert_eq!(ok("0 && 3"), 0.0);
        assert_eq!(ok("0 || -1"), 1.0);
        // Short-circuit: the divide-by-zero right operand is never evaluated.
        assert_eq!(ok("0 && (1/0)"), 0.0);
        assert_eq!(ok("1 || (1/0)"), 1.0);
        // Comparisons return 0.0 / 1.0 floats.
        assert_eq!(ok("3 == 3"), 1.0);
        assert_eq!(ok("3 != 3"), 0.0);
    }

    #[test]
    fn builtins() {
        assert_eq!(ok("sgn(-3) + int(2.9)"), 1.0);
        assert_eq!(ok("pwr(3, 2) + sqrt(16)"), 13.0);
        assert_eq!(ok("agauss(5, 2, 3)"), 5.0);
        assert_eq!(ok("agauss(5, 2)"), 5.0);
        assert_eq!(ok("int(-2.9)"), -2.0);
        assert_eq!(ok("sgn(0)"), 0.0);
        assert_eq!(ok("min(3, 1, 2)"), 1.0);
        assert_eq!(ok("selmin(3, 1, 2)"), 1.0);
        assert_eq!(ok("abs(-4.5)"), 4.5);
        assert_eq!(ok("exp(0)"), 1.0);
        assert_eq!(ok("log10(1000)"), 3.0);
        // int truncates toward zero and normalises the sign of zero.
        assert!(ok("int(-0.5)").to_bits() == 0.0f64.to_bits());
    }

    #[test]
    fn builtin_arity_and_unknown_errors() {
        assert_eq!(eval("agauss(5)").unwrap_err().kind, ErrorKind::Expression);
        assert_eq!(
            eval("agauss(5,2,3,4)").unwrap_err().kind,
            ErrorKind::Expression
        );
        assert_eq!(eval("sqrt(1, 2)").unwrap_err().kind, ErrorKind::Expression);
        assert_eq!(eval("max()").unwrap_err().kind, ErrorKind::Expression);
        // Unknown symbol / function.
        assert_eq!(
            eval("missing + 1").unwrap_err().kind,
            ErrorKind::UnknownSymbol
        );
        assert_eq!(eval("v(0)").unwrap_err().kind, ErrorKind::UnknownSymbol);
    }

    #[test]
    fn arithmetic_domain_errors_are_eager() {
        // Division by zero raises before a comparison can swallow the inf.
        assert_eq!(eval("1 / 0").unwrap_err().kind, ErrorKind::Expression);
        assert_eq!(eval("1/0 < 5").unwrap_err().kind, ErrorKind::Expression);
        assert_eq!(eval("sqrt(-1)").unwrap_err().kind, ErrorKind::Expression);
        assert_eq!(
            eval("sqrt(-1) < 5").unwrap_err().kind,
            ErrorKind::Expression
        );
        assert_eq!(eval("log10(0)").unwrap_err().kind, ErrorKind::Expression);
        assert_eq!(eval("log10(-1)").unwrap_err().kind, ErrorKind::Expression);
        assert_eq!(eval("exp(1000)").unwrap_err().kind, ErrorKind::Expression);
        assert_eq!(
            eval("exp(1000) < 5").unwrap_err().kind,
            ErrorKind::Expression
        );
        // 0 ** -n and negative base ** fractional exponent.
        assert_eq!(eval("0 ** -1").unwrap_err().kind, ErrorKind::Expression);
        assert_eq!(
            eval("pwr(-8, 0.5)").unwrap_err().kind,
            ErrorKind::Expression
        );
        assert_eq!(eval("10 ** 400").unwrap_err().kind, ErrorKind::Expression);
    }

    #[test]
    fn pow_edge_values() {
        assert_eq!(ok("0 ** 0"), 1.0);
        assert_eq!(ok("5 ** 0"), 1.0);
        assert_eq!(ok("0 ** 2"), 0.0);
        assert_eq!(ok("(-2) ** 2"), 4.0);
        assert_eq!(ok("(-2) ** 3"), -8.0);
    }

    #[test]
    fn nonfinite_boundary_guard() {
        // A finite-operand overflow via multiply is only rejected at the boundary.
        assert_eq!(
            eval("1e300 * 1e300").unwrap_err().kind,
            ErrorKind::Expression
        );
        // ...but as a comparison operand it flows through, matching Python.
        assert_eq!(ok("1e300 * 1e300 < 5"), 0.0);
        assert_eq!(
            eval("int(1e300 * 1e300)").unwrap_err().kind,
            ErrorKind::Expression
        );
    }

    #[test]
    fn constants() {
        assert_eq!(ok("pi").to_bits(), std::f64::consts::PI.to_bits());
        assert_eq!(ok("e").to_bits(), std::f64::consts::E.to_bits());
        assert_eq!(ok("true"), 1.0);
        assert_eq!(ok("FALSE"), 0.0);
    }

    #[test]
    fn lazy_case_insensitive_forward_reference() {
        let mut values = HashMap::new();
        values.insert("w".to_string(), 2e-6);
        let scope = ScopeInner::new_root(values);
        scope.define("area", "w * length");
        scope.define("LENGTH", "30n");
        let mut ctx = EvalCtx::new();
        let area = scope.resolve_symbol("AREA", &mut ctx).unwrap();
        assert!((area - 60e-15).abs() <= 60e-15 * 1e-14);
    }

    #[test]
    fn define_and_set_value_are_mutually_exclusive() {
        let scope = ScopeInner::new_root(HashMap::new());
        scope.define("x", "2 + 3");
        assert_eq!(scope.resolve_symbol("x", &mut EvalCtx::new()).unwrap(), 5.0);
        scope.set_value("x", 42.0);
        assert_eq!(
            scope.resolve_symbol("x", &mut EvalCtx::new()).unwrap(),
            42.0
        );
        scope.define("x", "1 + 1");
        assert_eq!(scope.resolve_symbol("x", &mut EvalCtx::new()).unwrap(), 2.0);
    }

    #[test]
    fn cycle_detection_reports_chain() {
        let scope = ScopeInner::new_root(HashMap::new());
        scope.define("a", "b + 1");
        scope.define("b", "a + 1");
        let err = scope.resolve_symbol("a", &mut EvalCtx::new()).unwrap_err();
        assert_eq!(err.kind, ErrorKind::ParameterCycle);
        assert!(err.message.contains("a -> b -> a"), "{}", err.message);
    }

    #[test]
    fn reentrant_expression_falls_back_to_lexical_parent() {
        // The `w=w` override idiom: a child re-defines `w` in terms of the
        // parent's `w`, which must resolve via the lexical parent, not cycle.
        let mut parent_values = HashMap::new();
        parent_values.insert("w".to_string(), 5.0);
        let parent = ScopeInner::new_root(parent_values);
        let child = ScopeInner::new_child(parent, HashMap::new());
        child.define("w", "w");
        assert_eq!(child.resolve_symbol("w", &mut EvalCtx::new()).unwrap(), 5.0);

        // `param = param + delta` also resolves against the parent.
        let mut pv = HashMap::new();
        pv.insert("param".to_string(), 10.0);
        pv.insert("delta".to_string(), 1.5);
        let parent = ScopeInner::new_root(pv);
        let child = ScopeInner::new_child(parent, HashMap::new());
        child.define("param", "param + delta");
        assert_eq!(
            child.resolve_symbol("param", &mut EvalCtx::new()).unwrap(),
            11.5
        );
    }

    #[test]
    fn user_defined_function_uses_lexical_parent() {
        let mut values = HashMap::new();
        values.insert("offset".to_string(), 2.0);
        let scope = ScopeInner::new_root(values);
        scope.define_function("shift", &["x".into(), "gain".into()], "x * gain + offset");
        assert_eq!(scope.evaluate("SHIFT(3, 4)").unwrap(), 14.0);
        // Arity mismatch on a user function.
        assert_eq!(
            scope.evaluate("shift(1)").unwrap_err().kind,
            ErrorKind::Expression
        );
    }

    #[test]
    fn evaluate_all_snapshots_values() {
        let mut values = HashMap::new();
        values.insert("a".to_string(), 1.0);
        let scope = ScopeInner::new_root(values);
        scope.define("b", "a + 1");
        scope.define("c", "b * 10");
        let all = scope.evaluate_all().unwrap();
        assert_eq!(all.get("a"), Some(&1.0));
        assert_eq!(all.get("b"), Some(&2.0));
        assert_eq!(all.get("c"), Some(&20.0));
    }

    #[test]
    fn negative_fractional_power_raises() {
        // Python returns complex here, then float() raises TypeError; we raise a
        // SpiceExpressionError. Documented divergence: both raise.
        assert_eq!(
            eval("pwr(-8, 1.0/3.0)").unwrap_err().kind,
            ErrorKind::Expression
        );
    }

    #[test]
    fn shared_scope_resolves_concurrently() {
        use std::thread;
        let scope = ScopeInner::new_root(HashMap::new());
        scope.set_value("base", 2.0);
        for i in 0..64 {
            scope.define(
                &format!("p{i}"),
                &if i == 0 {
                    "base + 1".to_string()
                } else {
                    format!("p{} * 2", i - 1)
                },
            );
        }
        // Serial reference.
        let serial = scope.resolve_symbol("p63", &mut EvalCtx::new()).unwrap();

        let mut handles = Vec::new();
        for _ in 0..8 {
            let scope = scope.clone();
            handles.push(thread::spawn(move || {
                let mut last = 0.0;
                for i in 0..64 {
                    last = scope
                        .resolve_symbol(&format!("p{i}"), &mut EvalCtx::new())
                        .unwrap();
                }
                last
            }));
        }
        for handle in handles {
            assert_eq!(handle.join().unwrap(), serial);
        }
    }
}
