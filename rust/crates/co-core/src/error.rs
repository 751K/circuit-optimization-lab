//! Stable errors for failures at the coarse solver boundary.

use thiserror::Error;

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum CoreError {
    #[error("invalid {analysis}")]
    InvalidTopology { analysis: &'static str },
    #[error("invalid {analysis} input: {detail}")]
    InvalidInput {
        analysis: &'static str,
        detail: &'static str,
    },
    #[error("singular {analysis} system")]
    Singular { analysis: &'static str },
}
