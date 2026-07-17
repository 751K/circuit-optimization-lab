//! Tokenizer for the supported HSPICE expression subset.
//!
//! Hand-written to reproduce the exact behaviour of the Python reference's
//! regular expressions (`_TOKEN_NUMBER`, `_TOKEN_IDENT`, `_SPICE_NUMBER`) and
//! operator sets, so token boundaries and SPICE-number values match bit-for-bit
//! without pulling in a regex dependency.
//!
//! The scanner works on a `Vec<char>` (Unicode code points) so error offsets
//! align with Python's code-point indexing.

use crate::error::{SpiceError, SpiceResult};

/// Token classes, mirroring the Python `Token.kind` strings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenKind {
    Op,
    Number,
    Identifier,
    Eof,
}

/// A lexeme: its class, source text, and code-point offset.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Token {
    pub kind: TokenKind,
    pub text: String,
    pub offset: usize,
}

/// Two-character operators, matched before single characters.
const TWO_CHAR_OPERATORS: [&str; 7] = ["**", "<=", ">=", "==", "!=", "&&", "||"];

/// `_ONE_CHAR_OPERATORS = set("+-*/^(),?:<>!")`.
fn is_one_char_operator(c: char) -> bool {
    matches!(
        c,
        '+' | '-' | '*' | '/' | '^' | '(' | ')' | ',' | '?' | ':' | '<' | '>' | '!'
    )
}

/// Strip a single pair of matching outer quotes (`'` or `"`), like the Python
/// `_strip_expression_quotes`.
fn strip_expression_quotes(expression: &str) -> String {
    let body = expression.trim();
    let chars: Vec<char> = body.chars().collect();
    if chars.len() >= 2 {
        let first = chars[0];
        let last = chars[chars.len() - 1];
        if first == last && (first == '\'' || first == '"') {
            let inner: String = chars[1..chars.len() - 1].iter().collect();
            return inner.trim().to_string();
        }
    }
    body.to_string()
}

/// Tokenize the supported HSPICE expression subset.
pub fn tokenize(expression: &str) -> SpiceResult<Vec<Token>> {
    let body: Vec<char> = strip_expression_quotes(expression).chars().collect();
    let n = body.len();
    let mut tokens = Vec::new();
    let mut index = 0usize;

    while index < n {
        let c = body[index];
        if c.is_whitespace() {
            index += 1;
            continue;
        }
        if index + 1 < n {
            let pair: String = [body[index], body[index + 1]].iter().collect();
            if TWO_CHAR_OPERATORS.contains(&pair.as_str()) {
                tokens.push(Token {
                    kind: TokenKind::Op,
                    text: pair,
                    offset: index,
                });
                index += 2;
                continue;
            }
        }
        if is_one_char_operator(c) {
            tokens.push(Token {
                kind: TokenKind::Op,
                text: c.to_string(),
                offset: index,
            });
            index += 1;
            continue;
        }
        if let Some(end) = scan_number(&body, index) {
            tokens.push(Token {
                kind: TokenKind::Number,
                text: body[index..end].iter().collect(),
                offset: index,
            });
            index = end;
            continue;
        }
        if let Some(end) = scan_identifier(&body, index) {
            tokens.push(Token {
                kind: TokenKind::Identifier,
                text: body[index..end].iter().collect(),
                offset: index,
            });
            index = end;
            continue;
        }
        return Err(SpiceError::expression(format!(
            "unsupported character {:?} at offset {} in {:?}",
            c, index, expression
        )));
    }

    tokens.push(Token {
        kind: TokenKind::Eof,
        text: String::new(),
        offset: n,
    });
    Ok(tokens)
}

fn is_ascii_digit(c: char) -> bool {
    c.is_ascii_digit()
}

/// Reproduce `_TOKEN_NUMBER.match(body, start)`: return the end index of the
/// longest number token starting at `start`, or `None` if none matches.
///
/// Grammar: `(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?(?:meg|mil|[tgkmunpf])?(?:[A-Za-z]+)?`.
fn scan_number(body: &[char], start: usize) -> Option<usize> {
    let n = body.len();
    let mut i = start;

    // Mantissa: `\d+(?:\.\d*)?` or `\.\d+`.
    if i < n && is_ascii_digit(body[i]) {
        while i < n && is_ascii_digit(body[i]) {
            i += 1;
        }
        if i < n && body[i] == '.' {
            i += 1;
            while i < n && is_ascii_digit(body[i]) {
                i += 1;
            }
        }
    } else if i < n && body[i] == '.' {
        if i + 1 < n && is_ascii_digit(body[i + 1]) {
            i += 1; // consume '.'
            while i < n && is_ascii_digit(body[i]) {
                i += 1;
            }
        } else {
            return None;
        }
    } else {
        return None;
    }

    // Exponent: `(?:[eE][+-]?\d+)?` — only consumed when digits follow.
    if i < n && (body[i] == 'e' || body[i] == 'E') {
        let mut j = i + 1;
        if j < n && (body[j] == '+' || body[j] == '-') {
            j += 1;
        }
        if j < n && is_ascii_digit(body[j]) {
            while j < n && is_ascii_digit(body[j]) {
                j += 1;
            }
            i = j;
        }
    }

    // Suffix `(?:meg|mil|[tgkmunpf])?` then unit `(?:[A-Za-z]+)?`.
    i = consume_suffix(body, i);
    while i < n && body[i].is_ascii_alphabetic() {
        i += 1;
    }

    Some(i)
}

/// Consume an optional SPICE magnitude suffix, ordered `meg|mil` before the
/// single-character class — exactly as the regex alternation is tried.
fn consume_suffix(body: &[char], i: usize) -> usize {
    let n = body.len();
    if i + 3 <= n {
        let s: String = body[i..i + 3]
            .iter()
            .collect::<String>()
            .to_ascii_lowercase();
        if s == "meg" || s == "mil" {
            return i + 3;
        }
    }
    if i < n && is_single_suffix(body[i]) {
        return i + 1;
    }
    i
}

fn is_single_suffix(c: char) -> bool {
    matches!(
        c.to_ascii_lowercase(),
        't' | 'g' | 'k' | 'm' | 'u' | 'n' | 'p' | 'f'
    )
}

/// Reproduce `_TOKEN_IDENT.match`: `[A-Za-z_][A-Za-z0-9_.$]*`.
fn scan_identifier(body: &[char], start: usize) -> Option<usize> {
    let n = body.len();
    let mut i = start;
    if i < n && (body[i].is_ascii_alphabetic() || body[i] == '_') {
        i += 1;
        while i < n
            && (body[i].is_ascii_alphanumeric()
                || body[i] == '_'
                || body[i] == '.'
                || body[i] == '$')
        {
            i += 1;
        }
        Some(i)
    } else {
        None
    }
}

/// The magnitude factor for a (lower-cased) suffix — the `_SUFFIX` table.
///
/// The literal texts are copied verbatim from the reference so the compiler
/// rounds them to the same `f64` bit patterns.
fn suffix_factor(suffix: &str) -> f64 {
    match suffix {
        "" => 1.0,
        "t" => 1e12,
        "g" => 1e9,
        "meg" => 1e6,
        "k" => 1e3,
        "mil" => 25.4e-6,
        "m" => 1e-3,
        "u" => 1e-6,
        "n" => 1e-9,
        "p" => 1e-12,
        "f" => 1e-15,
        _ => 1.0,
    }
}

/// Reproduce `_number` + `_SPICE_NUMBER.fullmatch`: parse a number token's text
/// into its `f64` value, preserving the reference's `float(number) * factor`
/// evaluation order (which affects rounding).
pub fn spice_number_value(text: &str) -> SpiceResult<f64> {
    let invalid = || SpiceError::expression(format!("invalid SPICE number {:?}", text));
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    let mut i = 0usize;

    // Number group: mantissa (+ optional exponent).
    let number_start = i;
    if i < n && is_ascii_digit(chars[i]) {
        while i < n && is_ascii_digit(chars[i]) {
            i += 1;
        }
        if i < n && chars[i] == '.' {
            i += 1;
            while i < n && is_ascii_digit(chars[i]) {
                i += 1;
            }
        }
    } else if i < n && chars[i] == '.' {
        if i + 1 < n && is_ascii_digit(chars[i + 1]) {
            i += 1;
            while i < n && is_ascii_digit(chars[i]) {
                i += 1;
            }
        } else {
            return Err(invalid());
        }
    } else {
        return Err(invalid());
    }
    if i < n && (chars[i] == 'e' || chars[i] == 'E') {
        let mut j = i + 1;
        if j < n && (chars[j] == '+' || chars[j] == '-') {
            j += 1;
        }
        if j < n && is_ascii_digit(chars[j]) {
            while j < n && is_ascii_digit(chars[j]) {
                j += 1;
            }
            i = j;
        }
    }
    let number_str: String = chars[number_start..i].iter().collect();

    // Suffix group (greedy `meg|mil` then single char).
    let mut suffix = String::new();
    if i + 3 <= n {
        let s: String = chars[i..i + 3]
            .iter()
            .collect::<String>()
            .to_ascii_lowercase();
        if s == "meg" || s == "mil" {
            suffix = s;
            i += 3;
        }
    }
    if suffix.is_empty() && i < n && is_single_suffix(chars[i]) {
        suffix.push(chars[i].to_ascii_lowercase());
        i += 1;
    }

    // Unit group `[A-Za-z]*`, then the whole text must be consumed (`$`).
    while i < n && chars[i].is_ascii_alphabetic() {
        i += 1;
    }
    if i != n {
        return Err(invalid());
    }

    let mantissa = parse_mantissa(&number_str).ok_or_else(invalid)?;
    Ok(mantissa * suffix_factor(&suffix))
}

/// Parse the mantissa+exponent significand exactly as Python's `float()` would,
/// normalising the leading/trailing decimal point (value-preserving) so Rust's
/// correctly-rounded parser accepts every form the grammar can emit.
fn parse_mantissa(s: &str) -> Option<f64> {
    let (significand, exponent) = match s.find(['e', 'E']) {
        Some(pos) => (&s[..pos], Some(&s[pos..])),
        None => (s, None),
    };
    let mut normalized = String::new();
    if significand.starts_with('.') {
        normalized.push('0');
    }
    normalized.push_str(significand);
    if normalized.ends_with('.') {
        normalized.push('0');
    }
    if let Some(exponent) = exponent {
        normalized.push_str(exponent);
    }
    normalized.parse::<f64>().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kinds(expr: &str) -> Vec<(TokenKind, String)> {
        tokenize(expr)
            .unwrap()
            .into_iter()
            .map(|t| (t.kind, t.text))
            .collect()
    }

    #[test]
    fn two_char_operators_beat_single() {
        assert_eq!(
            kinds("2**3"),
            vec![
                (TokenKind::Number, "2".into()),
                (TokenKind::Op, "**".into()),
                (TokenKind::Number, "3".into()),
                (TokenKind::Eof, String::new()),
            ]
        );
        for op in ["<=", ">=", "==", "!=", "&&", "||"] {
            let toks = kinds(&format!("1{op}2"));
            assert_eq!(toks[1], (TokenKind::Op, op.to_string()));
        }
    }

    #[test]
    fn identifier_allows_dot_dollar_underscore() {
        assert_eq!(
            kinds("a.b$c_d"),
            vec![
                (TokenKind::Identifier, "a.b$c_d".into()),
                (TokenKind::Eof, String::new()),
            ]
        );
    }

    #[test]
    fn strips_matching_outer_quotes() {
        assert_eq!(kinds("'1+2'"), kinds("1+2"));
        assert_eq!(kinds("\"1+2\""), kinds("1+2"));
        // Mismatched quotes are not stripped; the leading quote is then an
        // unsupported character (matching the reference).
        assert!(tokenize("'1+2\"").is_err());
    }

    #[test]
    fn unsupported_character_errors() {
        assert!(tokenize("1 @ 2").is_err());
        assert!(tokenize("&").is_err());
        assert!(tokenize(".").is_err());
    }

    #[test]
    fn suffix_values_match_reference_table() {
        let cases = [
            ("1t", 1.0 * 1e12),
            ("1g", 1.0 * 1e9),
            ("1meg", 1.0 * 1e6),
            ("1k", 1.0 * 1e3),
            ("1mil", 1.0 * 25.4e-6),
            ("1m", 1.0 * 1e-3),
            ("1u", 1.0 * 1e-6),
            ("1n", 1.0 * 1e-9),
            ("1p", 1.0 * 1e-12),
            ("1f", 1.0 * 1e-15),
            ("1", 1.0),
        ];
        for (text, expected) in cases {
            assert_eq!(spice_number_value(text).unwrap(), expected, "{text}");
        }
    }

    #[test]
    fn meg_beats_single_m_and_case_insensitive() {
        assert_eq!(spice_number_value("2.5MEG").unwrap(), 2.5 * 1e6);
        assert_eq!(spice_number_value("2.5Meg").unwrap(), 2.5 * 1e6);
        // "2.5m" is milli, not mega.
        assert_eq!(spice_number_value("2.5m").unwrap(), 2.5 * 1e-3);
        // "mil" is a distinct 3-char suffix.
        assert_eq!(spice_number_value("2.5mil").unwrap(), 2.5 * 25.4e-6);
    }

    #[test]
    fn trailing_unit_letters_are_ignored() {
        assert_eq!(spice_number_value("2.5mA").unwrap(), 2.5 * 1e-3);
        assert_eq!(spice_number_value("2.5abc").unwrap(), 2.5);
        assert_eq!(spice_number_value("1e3k").unwrap(), 1000.0 * 1e3);
    }

    #[test]
    fn evaluation_order_is_mantissa_times_factor() {
        // The reference multiplies float(number) by the factor; keep that order.
        assert_eq!(spice_number_value("1.1u").unwrap(), 1.1f64 * 1e-6);
        assert_eq!(spice_number_value("2.5k").unwrap(), 2.5f64 * 1e3);
    }

    #[test]
    fn leading_and_trailing_dot_parse() {
        assert_eq!(spice_number_value(".5").unwrap(), 0.5);
        assert_eq!(spice_number_value("2.").unwrap(), 2.0);
        assert_eq!(spice_number_value("2.e3").unwrap(), 2000.0);
        assert_eq!(spice_number_value(".5e2").unwrap(), 50.0);
    }

    #[test]
    fn exponent_without_digits_is_unit() {
        // "2e" is not scientific notation; the 'e' becomes an ignored unit.
        assert_eq!(spice_number_value("2e").unwrap(), 2.0);
        assert_eq!(kinds("2e")[0], (TokenKind::Number, "2e".into()));
    }

    #[test]
    fn number_token_boundaries() {
        // "1k2" -> number "1k" then number "2".
        assert_eq!(
            kinds("1k2"),
            vec![
                (TokenKind::Number, "1k".into()),
                (TokenKind::Number, "2".into()),
                (TokenKind::Eof, String::new()),
            ]
        );
        // "1.2.3" -> "1.2" then ".3".
        assert_eq!(
            kinds("1.2.3"),
            vec![
                (TokenKind::Number, "1.2".into()),
                (TokenKind::Number, ".3".into()),
                (TokenKind::Eof, String::new()),
            ]
        );
    }
}
