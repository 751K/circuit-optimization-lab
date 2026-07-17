//! Pratt parser and the compiled-expression cache.
//!
//! A direct port of the Python `_Parser` (operator binding powers, the
//! right-associative power operator, the right-associative ternary, and prefix
//! parsing of unary operators / parentheses / calls). [`compile_expression`]
//! mirrors the reference's `@lru_cache` with a thread-safe global AST cache.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use crate::ast::{BinaryOp, Expr, UnaryOp};
use crate::error::{SpiceError, SpiceResult};
use crate::lexer::{Token, TokenKind, spice_number_value, tokenize};

/// `_BINDING_POWER`: infix precedence. `None` means "not an infix operator",
/// which stops the binary loop.
fn binding_power(op: &str) -> Option<i32> {
    Some(match op {
        "||" => 10,
        "&&" => 20,
        "==" | "!=" | "<" | "<=" | ">" | ">=" => 30,
        "+" | "-" => 40,
        "*" | "/" => 50,
        "^" | "**" => 60,
        _ => return None,
    })
}

fn unary_op(text: &str) -> UnaryOp {
    match text {
        "+" => UnaryOp::Pos,
        "-" => UnaryOp::Neg,
        "!" => UnaryOp::Not,
        _ => unreachable!("unary_op called with {text:?}"),
    }
}

fn binary_op(text: &str) -> BinaryOp {
    match text {
        "||" => BinaryOp::Or,
        "&&" => BinaryOp::And,
        "==" => BinaryOp::Eq,
        "!=" => BinaryOp::Ne,
        "<" => BinaryOp::Lt,
        "<=" => BinaryOp::Le,
        ">" => BinaryOp::Gt,
        ">=" => BinaryOp::Ge,
        "+" => BinaryOp::Add,
        "-" => BinaryOp::Sub,
        "*" => BinaryOp::Mul,
        "/" => BinaryOp::Div,
        "^" | "**" => BinaryOp::Pow,
        _ => unreachable!("binary_op called with {text:?}"),
    }
}

struct Parser<'a> {
    tokens: Vec<Token>,
    source: &'a str,
    index: usize,
}

impl<'a> Parser<'a> {
    fn new(tokens: Vec<Token>, source: &'a str) -> Self {
        Self {
            tokens,
            source,
            index: 0,
        }
    }

    fn current(&self) -> &Token {
        &self.tokens[self.index]
    }

    fn accept(&mut self, text: &str) -> bool {
        if self.current().text == text {
            self.index += 1;
            true
        } else {
            false
        }
    }

    fn expect(&mut self, text: &str) -> SpiceResult<()> {
        if self.accept(text) {
            Ok(())
        } else {
            Err(SpiceError::expression(format!(
                "expected {:?} at offset {} in {:?}",
                text,
                self.current().offset,
                self.source
            )))
        }
    }

    fn parse(&mut self) -> SpiceResult<Expr> {
        let expression = self.parse_conditional()?;
        if self.current().kind != TokenKind::Eof {
            return Err(SpiceError::expression(format!(
                "unexpected token {:?} at offset {} in {:?}",
                self.current().text,
                self.current().offset,
                self.source
            )));
        }
        Ok(expression)
    }

    fn parse_conditional(&mut self) -> SpiceResult<Expr> {
        let condition = self.parse_binary(0)?;
        if !self.accept("?") {
            return Ok(condition);
        }
        let when_true = self.parse_conditional()?;
        self.expect(":")?;
        let when_false = self.parse_conditional()?;
        Ok(Expr::Conditional {
            condition: Box::new(condition),
            when_true: Box::new(when_true),
            when_false: Box::new(when_false),
        })
    }

    fn parse_binary(&mut self, minimum_binding: i32) -> SpiceResult<Expr> {
        let mut left = self.parse_prefix()?;
        loop {
            let operator = self.current().text.clone();
            let binding = match binding_power(&operator) {
                Some(binding) if binding >= minimum_binding => binding,
                _ => return Ok(left),
            };
            self.index += 1;
            // Power is right-associative; all other operators are left-associative.
            let next_binding = if operator == "^" || operator == "**" {
                binding
            } else {
                binding + 1
            };
            let right = self.parse_binary(next_binding)?;
            left = Expr::Binary {
                op: binary_op(&operator),
                left: Box::new(left),
                right: Box::new(right),
            };
        }
    }

    fn parse_prefix(&mut self) -> SpiceResult<Expr> {
        let text = self.current().text.clone();
        if text == "+" || text == "-" || text == "!" {
            self.index += 1;
            let operand = self.parse_prefix()?;
            return Ok(Expr::Unary {
                op: unary_op(&text),
                operand: Box::new(operand),
            });
        }
        if self.accept("(") {
            let expression = self.parse_conditional()?;
            self.expect(")")?;
            return Ok(expression);
        }
        match self.current().kind {
            TokenKind::Number => {
                let text = self.current().text.clone();
                self.index += 1;
                Ok(Expr::Number(spice_number_value(&text)?))
            }
            TokenKind::Identifier => {
                let name = self.current().text.clone();
                self.index += 1;
                if !self.accept("(") {
                    return Ok(Expr::Name(name));
                }
                let mut arguments = Vec::new();
                if !self.accept(")") {
                    loop {
                        arguments.push(self.parse_conditional()?);
                        if self.accept(")") {
                            break;
                        }
                        self.expect(",")?;
                    }
                }
                Ok(Expr::Call { name, arguments })
            }
            _ => Err(SpiceError::expression(format!(
                "expected expression at offset {} in {:?}",
                self.current().offset,
                self.source
            ))),
        }
    }
}

fn parse_expression(expression: &str) -> SpiceResult<Expr> {
    let tokens = tokenize(expression)?;
    Parser::new(tokens, expression).parse()
}

/// Compile one HSPICE expression to a reusable immutable tree.
///
/// Mirrors the Python `@lru_cache(maxsize=32768)` with a process-global,
/// thread-safe cache. Parse failures are not cached (repeated calls re-raise),
/// exactly as `lru_cache` never memoises exceptions. The cache never changes
/// numerical results; it only spares repeated parsing under corner/MC fan-out.
pub fn compile_expression(expression: &str) -> SpiceResult<Arc<Expr>> {
    static CACHE: OnceLock<Mutex<HashMap<String, Arc<Expr>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    {
        let guard = cache.lock().unwrap_or_else(|poison| poison.into_inner());
        if let Some(compiled) = guard.get(expression) {
            return Ok(compiled.clone());
        }
    }
    let compiled = Arc::new(parse_expression(expression)?);
    let mut guard = cache.lock().unwrap_or_else(|poison| poison.into_inner());
    Ok(guard
        .entry(expression.to_string())
        .or_insert(compiled)
        .clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(expr: &str) -> Expr {
        (*compile_expression(expr).unwrap()).clone()
    }

    #[test]
    fn power_is_right_associative() {
        // 2 ** 3 ** 2 -> 2 ** (3 ** 2)
        match parse("2**3**2") {
            Expr::Binary {
                op: BinaryOp::Pow,
                left,
                right,
            } => {
                assert_eq!(*left, Expr::Number(2.0));
                assert!(matches!(
                    *right,
                    Expr::Binary {
                        op: BinaryOp::Pow,
                        ..
                    }
                ));
            }
            other => panic!("unexpected tree: {other:?}"),
        }
    }

    #[test]
    fn additive_is_left_associative() {
        // 1 - 2 - 3 -> (1 - 2) - 3
        match parse("1-2-3") {
            Expr::Binary {
                op: BinaryOp::Sub,
                left,
                right,
            } => {
                assert!(matches!(
                    *left,
                    Expr::Binary {
                        op: BinaryOp::Sub,
                        ..
                    }
                ));
                assert_eq!(*right, Expr::Number(3.0));
            }
            other => panic!("unexpected tree: {other:?}"),
        }
    }

    #[test]
    fn unary_minus_binds_tighter_than_power_operand() {
        // Reference quirk: -2**2 parses as (-2) ** 2, not -(2 ** 2).
        match parse("-2**2") {
            Expr::Binary {
                op: BinaryOp::Pow,
                left,
                ..
            } => assert!(matches!(
                *left,
                Expr::Unary {
                    op: UnaryOp::Neg,
                    ..
                }
            )),
            other => panic!("unexpected tree: {other:?}"),
        }
    }

    #[test]
    fn ternary_is_right_associative() {
        // 1 ? 2 : 0 ? 3 : 4 -> 1 ? 2 : (0 ? 3 : 4)
        match parse("1?2:0?3:4") {
            Expr::Conditional { when_false, .. } => {
                assert!(matches!(*when_false, Expr::Conditional { .. }));
            }
            other => panic!("unexpected tree: {other:?}"),
        }
    }

    #[test]
    fn parse_errors_are_expression_kind() {
        assert!(compile_expression("1 +").is_err());
        assert!(compile_expression("(1").is_err());
        assert!(compile_expression("1 2").is_err());
        assert!(compile_expression("max(1,").is_err());
    }

    #[test]
    fn cache_returns_shared_tree() {
        let a = compile_expression("1 + 2 * 3").unwrap();
        let b = compile_expression("1 + 2 * 3").unwrap();
        assert!(Arc::ptr_eq(&a, &b));
    }
}
