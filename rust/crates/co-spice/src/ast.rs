//! Immutable expression tree — the Rust analogue of the Python `Expression`
//! dataclasses (`Number`, `Name`, `Unary`, `Binary`, `Conditional`, `Call`).
//!
//! The tree is produced once by the parser and shared (behind an `Arc`) by the
//! compile cache; evaluation only ever borrows it.

/// Prefix operators: `+`, `-`, `!`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    /// `+operand` — identity.
    Pos,
    /// `-operand` — negation.
    Neg,
    /// `!operand` — logical not (`float(not bool(operand))`).
    Not,
}

/// Infix operators, spanning the full binding-power table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    Or,
    And,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    Add,
    Sub,
    Mul,
    Div,
    /// Both `^` and `**` map here; both are `left ** right`.
    Pow,
}

/// One node of the compiled expression tree.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    /// A literal SPICE number, already resolved to its `f64` value.
    Number(f64),
    /// A symbol reference, resolved lazily through the scope.
    Name(String),
    Unary {
        op: UnaryOp,
        operand: Box<Expr>,
    },
    Binary {
        op: BinaryOp,
        left: Box<Expr>,
        right: Box<Expr>,
    },
    Conditional {
        condition: Box<Expr>,
        when_true: Box<Expr>,
        when_false: Box<Expr>,
    },
    Call {
        name: String,
        arguments: Vec<Expr>,
    },
}
