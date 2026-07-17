//! SPICE/HSPICE model-library deck parser.
//!
//! A 1:1 port of `circuitopt/spice/parser.py` (the frozen Python reference). It
//! turns a model delivery into an in-memory AST without evaluating foundry
//! expressions: continuations, inline/line comments, library sections, model
//! cards, and subcircuits. The hand-written scanners reproduce the reference's
//! regular-expression and string-slicing semantics exactly (`_NAME`,
//! `_ASSIGNMENT`, `_NUMBER`, `str.splitlines`, `str.find`, quote/paren-aware
//! word splitting) so the parsed structure matches field-for-field, with no
//! third-party dependency.
//!
//! Structural failures raise [`ErrorKind::Syntax`] (the Python `SpiceSyntaxError`,
//! a `ValueError` subclass).

use crate::error::{SpiceError, SpiceResult};

// ---------------------------------------------------------------------------
// Ordered maps (Python dicts preserve insertion order).
// ---------------------------------------------------------------------------

/// An insertion-ordered string-keyed map, mirroring a Python `dict`. Re-inserting
/// an existing key overwrites the value in place while keeping its position, just
/// like `d[k] = v`.
#[derive(Debug, Clone)]
pub struct OrderedMap<V> {
    entries: Vec<(String, V)>,
}

impl<V> Default for OrderedMap<V> {
    fn default() -> Self {
        Self {
            entries: Vec::new(),
        }
    }
}

impl<V> OrderedMap<V> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, key: String, value: V) {
        if let Some(slot) = self.entries.iter_mut().find(|(k, _)| *k == key) {
            slot.1 = value;
        } else {
            self.entries.push((key, value));
        }
    }

    pub fn get(&self, key: &str) -> Option<&V> {
        self.entries.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    pub fn get_mut(&mut self, key: &str) -> Option<&mut V> {
        self.entries
            .iter_mut()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v)
    }

    pub fn contains(&self, key: &str) -> bool {
        self.entries.iter().any(|(k, _)| k == key)
    }

    pub fn iter(&self) -> impl Iterator<Item = &(String, V)> {
        self.entries.iter()
    }

    pub fn values(&self) -> impl Iterator<Item = &V> {
        self.entries.iter().map(|(_, v)| v)
    }

    pub fn keys(&self) -> impl Iterator<Item = &String> {
        self.entries.iter().map(|(k, _)| k)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

// ---------------------------------------------------------------------------
// AST data structures (mirror the Python dataclasses field-for-field).
// ---------------------------------------------------------------------------

/// `SourceLocation(path, first_line, last_line)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceLocation {
    pub path: String,
    pub first_line: usize,
    pub last_line: usize,
}

/// `ParameterAssignment(name, expression, formal_parameters)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParameterAssignment {
    pub name: String,
    pub expression: String,
    pub formal_parameters: Vec<String>,
}

impl ParameterAssignment {
    pub fn is_function(&self) -> bool {
        !self.formal_parameters.is_empty()
    }
}

/// One logical SPICE statement. `kind` is lower-case and, for directives, omits
/// the leading dot; element statements use their lower-case first character.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Statement {
    pub kind: String,
    pub location: SourceLocation,
    pub text: String,
    pub name: Option<String>,
    pub arguments: Vec<String>,
    pub parameters: Vec<ParameterAssignment>,
}

impl Statement {
    /// `parameter_map`: lower-cased name -> expression (last wins).
    pub fn parameter_map(&self) -> OrderedMap<String> {
        let mut map = OrderedMap::new();
        for item in &self.parameters {
            map.insert(item.name.to_lowercase(), item.expression.clone());
        }
        map
    }
}

/// A parsed `.subckt ... .ends` template.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Subcircuit {
    pub name: String,
    pub location: SourceLocation,
    pub terminals: Vec<String>,
    pub parameters: Vec<ParameterAssignment>,
    pub statements: Vec<Statement>,
}

/// A `.lib ... .endl` section (or the synthetic `<top>` section).
#[derive(Debug, Clone)]
pub struct LibrarySection {
    pub name: String,
    pub location: SourceLocation,
    pub statements: Vec<Statement>,
    pub subcircuits: OrderedMap<Subcircuit>,
}

impl LibrarySection {
    fn new(name: impl Into<String>, location: SourceLocation) -> Self {
        Self {
            name: name.into(),
            location,
            statements: Vec::new(),
            subcircuits: OrderedMap::new(),
        }
    }

    /// `models`: lower-name -> `.model` statement. Mirrors the Python dict
    /// comprehension (a repeated key keeps its first position, last value).
    pub fn models(&self) -> OrderedMap<Statement> {
        let mut map = OrderedMap::new();
        for statement in &self.statements {
            if statement.kind == "model"
                && let Some(name) = &statement.name
            {
                map.insert(name.to_lowercase(), statement.clone());
            }
        }
        map
    }
}

/// A parsed SPICE model library.
#[derive(Debug, Clone)]
pub struct SpiceModelLibrary {
    pub path: String,
    pub top_level: LibrarySection,
    pub sections: OrderedMap<LibrarySection>,
}

impl SpiceModelLibrary {
    /// `library.section(name)`; the `Err` carries the Python `KeyError` message
    /// body so callers can wrap it (the elaborator raises `SpiceElaborationError`).
    pub fn section(&self, name: &str) -> Result<&LibrarySection, String> {
        let key = name.to_lowercase();
        self.sections.get(&key).ok_or_else(|| {
            let mut names: Vec<&String> = self.sections.keys().collect();
            names.sort();
            let available: Vec<String> = names.iter().map(|s| (*s).clone()).collect();
            format!(
                "unknown library section {}; available sections: {}",
                py_repr(name),
                available.join(", ")
            )
        })
    }
}

// ---------------------------------------------------------------------------
// SPICE-number literal (`parse_spice_number`).
// ---------------------------------------------------------------------------

/// The `_SUFFIX` magnitude table (copied verbatim so the compiler rounds the
/// literals to the same `f64` bit patterns as the Python reference).
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

fn is_single_suffix(c: char) -> bool {
    matches!(
        c.to_ascii_lowercase(),
        't' | 'g' | 'k' | 'm' | 'u' | 'n' | 'p' | 'f'
    )
}

/// Parse a SPICE number, including `m`/`meg` and trailing-unit handling
/// (`1kOhm == 1k`). Mirrors `parse_spice_number` and its `_NUMBER` regex, which
/// — unlike the expression tokenizer — allows a leading sign and surrounding
/// `[ \t]`.
pub fn parse_spice_number(value: &str) -> SpiceResult<f64> {
    let chars: Vec<char> = value.chars().collect();
    let n = chars.len();
    let err = || SpiceError::expression(format!("not a SPICE numeric literal: {}", py_repr(value)));

    // ^[ \t]* ... [ \t]$
    let mut i = 0usize;
    while i < n && (chars[i] == ' ' || chars[i] == '\t') {
        i += 1;
    }
    let mut end = n;
    while end > i && (chars[end - 1] == ' ' || chars[end - 1] == '\t') {
        end -= 1;
    }

    let number_start = i;
    // optional sign
    if i < end && (chars[i] == '+' || chars[i] == '-') {
        i += 1;
    }
    // mantissa: \d+(?:\.\d*)? | \.\d+
    if i < end && chars[i].is_ascii_digit() {
        while i < end && chars[i].is_ascii_digit() {
            i += 1;
        }
        if i < end && chars[i] == '.' {
            i += 1;
            while i < end && chars[i].is_ascii_digit() {
                i += 1;
            }
        }
    } else if i < end && chars[i] == '.' {
        if i + 1 < end && chars[i + 1].is_ascii_digit() {
            i += 1;
            while i < end && chars[i].is_ascii_digit() {
                i += 1;
            }
        } else {
            return Err(err());
        }
    } else {
        return Err(err());
    }
    // exponent (only if digits follow)
    if i < end && (chars[i] == 'e' || chars[i] == 'E') {
        let mut j = i + 1;
        if j < end && (chars[j] == '+' || chars[j] == '-') {
            j += 1;
        }
        if j < end && chars[j].is_ascii_digit() {
            while j < end && chars[j].is_ascii_digit() {
                j += 1;
            }
            i = j;
        }
    }
    let number_str: String = chars[number_start..i].iter().collect();

    // suffix (greedy meg|mil then single char), case-insensitive
    let mut suffix = String::new();
    if i + 3 <= end {
        let s: String = chars[i..i + 3]
            .iter()
            .collect::<String>()
            .to_ascii_lowercase();
        if s == "meg" || s == "mil" {
            suffix = s;
            i += 3;
        }
    }
    if suffix.is_empty() && i < end && is_single_suffix(chars[i]) {
        suffix.push(chars[i].to_ascii_lowercase());
        i += 1;
    }
    // unit [A-Za-z]* then the whole (stripped) text must be consumed
    while i < end && chars[i].is_ascii_alphabetic() {
        i += 1;
    }
    if i != end {
        return Err(err());
    }

    let mantissa = parse_number_literal(&number_str).ok_or_else(err)?;
    Ok(mantissa * suffix_factor(&suffix))
}

/// Parse a signed mantissa (`[+-]?...` with optional leading/trailing decimal
/// point and optional exponent) exactly as Python's `float()` would.
fn parse_number_literal(s: &str) -> Option<f64> {
    let (sign, rest) = if let Some(stripped) = s.strip_prefix('-') {
        ("-", stripped)
    } else if let Some(stripped) = s.strip_prefix('+') {
        ("", stripped)
    } else {
        ("", s)
    };
    let (significand, exponent) = match rest.find(['e', 'E']) {
        Some(pos) => (&rest[..pos], Some(&rest[pos..])),
        None => (rest, None),
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
    format!("{sign}{normalized}").parse::<f64>().ok()
}

// ---------------------------------------------------------------------------
// Logical lines (`logical_lines`, `_strip_inline_comment`, `splitlines`).
// ---------------------------------------------------------------------------

/// Reproduce `str.splitlines()` for the ASCII line-boundary set (`\n`, `\r`,
/// `\r\n`, and the vertical-tab / form-feed / file-group-record separators that
/// vendor HSPICE deliveries occasionally use as page breaks).
fn splitlines(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    let mut lines: Vec<String> = Vec::new();
    let mut start = 0usize;
    let mut i = 0usize;
    while i < n {
        let c = chars[i];
        let is_break = matches!(
            c,
            '\n' | '\r'
                | '\u{0b}'
                | '\u{0c}'
                | '\u{1c}'
                | '\u{1d}'
                | '\u{1e}'
                | '\u{85}'
                | '\u{2028}'
                | '\u{2029}'
        );
        if is_break {
            lines.push(chars[start..i].iter().collect());
            if c == '\r' && i + 1 < n && chars[i + 1] == '\n' {
                i += 2;
            } else {
                i += 1;
            }
            start = i;
        } else {
            i += 1;
        }
    }
    if start < n {
        lines.push(chars[start..].iter().collect());
    }
    lines
}

fn is_py_space(c: char) -> bool {
    matches!(
        c,
        ' ' | '\t' | '\n' | '\r' | '\u{0b}' | '\u{0c}' | '\u{1c}' | '\u{1d}' | '\u{1e}' | '\u{1f}'
    )
}

/// `line.rstrip()` — strip trailing Python-whitespace.
fn rstrip(s: &str) -> String {
    let chars: Vec<char> = s.chars().collect();
    let mut end = chars.len();
    while end > 0 && is_py_space(chars[end - 1]) {
        end -= 1;
    }
    chars[..end].iter().collect()
}

/// `line.lstrip()` — strip leading Python-whitespace.
fn lstrip(s: &str) -> String {
    let chars: Vec<char> = s.chars().collect();
    let mut start = 0;
    while start < chars.len() && is_py_space(chars[start]) {
        start += 1;
    }
    chars[start..].iter().collect()
}

/// `s.strip()` — both sides.
fn strip(s: &str) -> String {
    rstrip(&lstrip(s))
}

/// Remove an HSPICE `$` inline comment outside quotes (`_strip_inline_comment`).
fn strip_inline_comment(line: &str) -> String {
    let chars: Vec<char> = line.chars().collect();
    let mut quote: Option<char> = None;
    for (index, &c) in chars.iter().enumerate() {
        if let Some(q) = quote {
            if c == q {
                quote = None;
            }
            continue;
        }
        if c == '\'' || c == '"' {
            quote = Some(c);
        } else if c == '$' {
            return chars[..index].iter().collect();
        }
    }
    line.to_string()
}

/// Join `+` continuations and attach source locations (`logical_lines`).
pub fn logical_lines(text: &str, path: &str) -> SpiceResult<Vec<(String, SourceLocation)>> {
    let mut out: Vec<(String, SourceLocation)> = Vec::new();
    let mut current: Vec<String> = Vec::new();
    let mut first = 0usize;
    let mut last = 0usize;

    // flush(): join stripped non-empty parts with single spaces.
    let flush = |current: &mut Vec<String>,
                 first: usize,
                 last: usize|
     -> Option<(String, SourceLocation)> {
        if current.is_empty() {
            return None;
        }
        let joined = current
            .iter()
            .map(|part| strip(part))
            .filter(|part| !part.is_empty())
            .collect::<Vec<_>>()
            .join(" ");
        current.clear();
        Some((
            joined,
            SourceLocation {
                path: path.to_string(),
                first_line: first,
                last_line: last,
            },
        ))
    };

    for (index, physical) in splitlines(text).into_iter().enumerate() {
        let line_number = index + 1;
        let line = rstrip(&strip_inline_comment(&physical));
        let stripped = lstrip(&line);
        if stripped.is_empty() || stripped.starts_with('*') {
            continue;
        }
        if stripped.starts_with('+') {
            if current.is_empty() {
                return Err(SpiceError::syntax(format!(
                    "{path}:{line_number}: continuation without a previous statement"
                )));
            }
            current.push(stripped.chars().skip(1).collect());
            last = line_number;
            continue;
        }
        if let Some(pending) = flush(&mut current, first, last) {
            out.push(pending);
        }
        current = vec![line];
        first = line_number;
        last = line_number;
    }
    if let Some(pending) = flush(&mut current, first, last) {
        out.push(pending);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Assignment scanning (`_balanced_outer_parentheses`, `_assignment_starts`,
// `parse_assignments`), quote/paren-aware word splitting (`_split_words`).
// ---------------------------------------------------------------------------

fn balanced_outer_parentheses(chars: &[char]) -> bool {
    let n = chars.len();
    if !(n >= 1 && chars[0] == '(' && chars[n - 1] == ')') {
        return false;
    }
    let mut depth = 0i64;
    let mut quote: Option<char> = None;
    for (index, &c) in chars.iter().enumerate() {
        if let Some(q) = quote {
            if c == q {
                quote = None;
            }
            continue;
        }
        if c == '\'' || c == '"' {
            quote = Some(c);
        } else if c == '(' {
            depth += 1;
        } else if c == ')' {
            depth -= 1;
            if depth == 0 && index != n - 1 {
                return false;
            }
            if depth < 0 {
                return false;
            }
        }
    }
    depth == 0 && quote.is_none()
}

fn is_ident_start(c: char) -> bool {
    c.is_ascii_alphabetic() || c == '_'
}

fn is_ident_rest(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_' || c == '.' || c == '$'
}

/// Match `\s*=(?!=)` starting at `start`; return the index just past `=`.
fn match_equals(chars: &[char], start: usize) -> Option<usize> {
    let n = chars.len();
    let mut j = start;
    while j < n && is_py_space(chars[j]) {
        j += 1;
    }
    if j < n && chars[j] == '=' && !(j + 1 < n && chars[j + 1] == '=') {
        Some(j + 1)
    } else {
        None
    }
}

/// Match `\s*\(([^()]*)\)` then `\s*=(?!=)` starting at `start`; return
/// `(index_past_equals, inner_text)`.
fn match_paren_then_equals(chars: &[char], start: usize) -> Option<(usize, String)> {
    let n = chars.len();
    let mut j = start;
    while j < n && is_py_space(chars[j]) {
        j += 1;
    }
    if !(j < n && chars[j] == '(') {
        return None;
    }
    j += 1;
    let inner_start = j;
    while j < n && chars[j] != '(' && chars[j] != ')' {
        j += 1;
    }
    if !(j < n && chars[j] == ')') {
        return None;
    }
    let inner: String = chars[inner_start..j].iter().collect();
    j += 1;
    let end = match_equals(chars, j)?;
    Some((end, inner))
}

/// Parse the formal-parameter list of a function definition:
/// `tuple(item.strip() for item in inner.split(",") if item.strip())`.
fn parse_formals(inner: &str) -> Vec<String> {
    inner
        .split(',')
        .map(strip)
        .filter(|item| !item.is_empty())
        .collect()
}

/// One `_ASSIGNMENT` match anchored at `index`: returns
/// `(end_past_equals, name, formals)`.
fn match_assignment(chars: &[char], index: usize) -> Option<(usize, String, Vec<String>)> {
    let n = chars.len();
    if !(index < n && is_ident_start(chars[index])) {
        return None;
    }
    let mut i = index + 1;
    while i < n && is_ident_rest(chars[i]) {
        i += 1;
    }
    let name: String = chars[index..i].iter().collect();
    // Greedy optional `(...)` group; backtrack to the bare `=` on failure.
    if let Some((end, inner)) = match_paren_then_equals(chars, i) {
        return Some((end, name, parse_formals(&inner)));
    }
    if let Some(end) = match_equals(chars, i) {
        return Some((end, name, Vec::new()));
    }
    None
}

/// Top-level `name=` spans, ignoring function arguments and quoted text.
/// Returns `(start, value_begin, name, formals)`.
fn assignment_starts(chars: &[char]) -> Vec<(usize, usize, String, Vec<String>)> {
    let n = chars.len();
    let mut starts = Vec::new();
    let mut depth = 0i64;
    let mut quote: Option<char> = None;
    let mut index = 0usize;
    while index < n {
        let c = chars[index];
        if let Some(q) = quote {
            if c == q {
                quote = None;
            }
            index += 1;
            continue;
        }
        if c == '\'' || c == '"' {
            quote = Some(c);
            index += 1;
            continue;
        }
        if c == '(' || c == '{' || c == '[' {
            depth += 1;
            index += 1;
            continue;
        }
        if c == ')' || c == '}' || c == ']' {
            depth = (depth - 1).max(0);
            index += 1;
            continue;
        }
        if depth == 0
            && let Some((end, name, formals)) = match_assignment(chars, index)
        {
            starts.push((index, end, name, formals));
            index = end;
            continue;
        }
        index += 1;
    }
    starts
}

/// Parse a whitespace-separated HSPICE assignment list (`parse_assignments`).
pub fn parse_assignments(text: &str) -> SpiceResult<Vec<ParameterAssignment>> {
    let mut body: Vec<char> = strip(text).chars().collect();
    while balanced_outer_parentheses(&body) {
        let inner: String = body[1..body.len() - 1].iter().collect();
        body = strip(&inner).chars().collect();
    }
    let starts = assignment_starts(&body);
    let mut assignments = Vec::new();
    for (idx, (_begin, value_begin, name, formals)) in starts.iter().enumerate() {
        let value_end = if idx + 1 < starts.len() {
            starts[idx + 1].0
        } else {
            body.len()
        };
        let raw: String = body[*value_begin..value_end].iter().collect();
        let expression = rstrip_commas(&strip(&raw));
        if expression.is_empty() {
            return Err(SpiceError::syntax(format!(
                "missing value for parameter {}",
                py_repr(name)
            )));
        }
        assignments.push(ParameterAssignment {
            name: name.clone(),
            expression,
            formal_parameters: formals.clone(),
        });
    }
    Ok(assignments)
}

/// `s.rstrip(",")`.
fn rstrip_commas(s: &str) -> String {
    s.trim_end_matches(',').to_string()
}

/// Split words while preserving quoted strings and parenthesized expressions
/// (`_split_words`).
fn split_words(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    let mut words = Vec::new();
    let mut start: Option<usize> = None;
    let mut depth = 0i64;
    let mut quote: Option<char> = None;
    for index in 0..n {
        let c = chars[index];
        if let Some(q) = quote {
            if c == q {
                quote = None;
            }
            continue;
        }
        if c == '\'' || c == '"' {
            quote = Some(c);
            if start.is_none() {
                start = Some(index);
            }
        } else if c == '(' || c == '{' || c == '[' {
            depth += 1;
            if start.is_none() {
                start = Some(index);
            }
        } else if c == ')' || c == '}' || c == ']' {
            depth = (depth - 1).max(0);
        } else if is_py_space(c) && depth == 0 {
            if let Some(s) = start {
                words.push(chars[s..index].iter().collect());
                start = None;
            }
        } else if start.is_none() {
            start = Some(index);
        }
    }
    if let Some(s) = start {
        words.push(chars[s..].iter().collect());
    }
    words
}

// ---------------------------------------------------------------------------
// Statement construction (`_statement`, `_subckt_header`).
// ---------------------------------------------------------------------------

/// `str.find`: first char index of `needle` in `haystack`, or `None`.
fn char_find(haystack: &[char], needle: &[char]) -> Option<usize> {
    if needle.is_empty() {
        return Some(0);
    }
    if needle.len() > haystack.len() {
        return None;
    }
    for start in 0..=(haystack.len() - needle.len()) {
        if haystack[start..start + needle.len()] == *needle {
            return Some(start);
        }
    }
    None
}

fn statement(text: &str, location: &SourceLocation) -> SpiceResult<Statement> {
    let stripped = strip(text);
    let stripped_chars: Vec<char> = stripped.chars().collect();
    if stripped.starts_with('.') {
        let after_dot: String = stripped_chars[1..].iter().collect();
        let words = split_words(&after_dot);
        if words.is_empty() {
            // `_split_words("")` -> []; `words[0]` would IndexError. The reference
            // never reaches this with a lone "."; treat as malformed directive.
            return Err(SpiceError::syntax(format!(
                "{}:{}: malformed directive",
                location.path, location.first_line
            )));
        }
        let kind = words[0].to_lowercase();
        // stripped[1 + len(words[0]):].strip()
        let cut = (1 + words[0].chars().count()).min(stripped_chars.len());
        let rest = strip(&stripped_chars[cut..].iter().collect::<String>());
        if kind == "model" {
            let head = split_words(&rest);
            if head.len() < 2 {
                return Err(SpiceError::syntax(format!(
                    "{}:{}: malformed .model",
                    location.path, location.first_line
                )));
            }
            let name = head[0].clone();
            let model_type = head[1].clone();
            let rest_chars: Vec<char> = rest.chars().collect();
            let type_chars: Vec<char> = model_type.chars().collect();
            let found = char_find(&rest_chars, &type_chars);
            let offset = match found {
                Some(pos) => pos + type_chars.len(),
                // str.find returns -1; offset = -1 + len(type).
                None => type_chars.len().saturating_sub(1),
            };
            let tail: String = rest_chars[offset.min(rest_chars.len())..].iter().collect();
            return Ok(Statement {
                kind,
                location: location.clone(),
                text: text.to_string(),
                name: Some(name),
                arguments: vec![model_type],
                parameters: parse_assignments(&tail)?,
            });
        }
        if kind == "param" {
            return Ok(Statement {
                kind,
                location: location.clone(),
                text: text.to_string(),
                name: None,
                arguments: Vec::new(),
                parameters: parse_assignments(&rest)?,
            });
        }
        return Ok(Statement {
            kind,
            location: location.clone(),
            text: text.to_string(),
            name: None,
            arguments: words[1..].to_vec(),
            parameters: Vec::new(),
        });
    }

    let words = split_words(&stripped);
    if words.is_empty() || !is_ident_start(words[0].chars().next().unwrap_or(' ')) {
        return Err(SpiceError::syntax(format!(
            "{}:{}: malformed element statement",
            location.path, location.first_line
        )));
    }
    let name = words[0].clone();
    let kind = name
        .chars()
        .next()
        .map(|c| c.to_ascii_lowercase().to_string())
        .unwrap_or_default();
    let cut = name.chars().count().min(stripped_chars.len());
    let tail: String = stripped_chars[cut..].iter().collect();
    Ok(Statement {
        kind,
        location: location.clone(),
        text: text.to_string(),
        name: Some(name),
        arguments: words[1..].to_vec(),
        parameters: parse_assignments(&tail)?,
    })
}

/// `_subckt_header`: `(name, terminals, parameters)` from a `.subckt` directive.
fn subckt_header(stmt: &Statement) -> SpiceResult<(String, Vec<String>, Vec<ParameterAssignment>)> {
    if stmt.arguments.is_empty() {
        return Err(SpiceError::syntax(format!(
            "{}:{}: missing .subckt name",
            stmt.location.path, stmt.location.first_line
        )));
    }
    let name = stmt.arguments[0].clone();
    let tail = stmt.arguments[1..].join(" ");
    let assignments = parse_assignments(&tail)?;
    let tail_chars: Vec<char> = tail.chars().collect();
    let starts = assignment_starts(&tail_chars);
    let terminal_text: String = if let Some(first) = starts.first() {
        tail_chars[..first.0].iter().collect()
    } else {
        tail.clone()
    };
    let terminals: Vec<String> = split_words(&terminal_text)
        .into_iter()
        .filter(|word| {
            let lower = word.to_lowercase();
            lower != "params:" && lower != "param:"
        })
        .collect();
    Ok((name, terminals, assignments))
}

/// Strip leading/trailing `'`/`"` (Python `str.strip("'\"")`).
fn strip_quotes(s: &str) -> String {
    s.trim_matches(|c| c == '\'' || c == '"').to_string()
}

// ---------------------------------------------------------------------------
// Library parsing (`parse_spice_library_text`).
// ---------------------------------------------------------------------------

enum Cur {
    Top,
    Section(String),
}

/// Parse a SPICE/HSPICE model library from text.
pub fn parse_spice_library_text(text: &str, path: &str) -> SpiceResult<SpiceModelLibrary> {
    let total_lines = splitlines(text).len().max(1);
    let mut top = LibrarySection::new(
        "<top>",
        SourceLocation {
            path: path.to_string(),
            first_line: 1,
            last_line: total_lines,
        },
    );
    let mut sections: OrderedMap<LibrarySection> = OrderedMap::new();
    let mut cur = Cur::Top;
    // (key, original-name) of the open subcircuit, if any.
    let mut current_subckt: Option<(String, String)> = None;

    for (raw, location) in logical_lines(text, path)? {
        let stmt = statement(&raw, &location)?;

        if stmt.kind == "lib" && stmt.arguments.len() == 1 {
            if !matches!(cur, Cur::Top) {
                return Err(SpiceError::syntax(format!(
                    "{}:{}: nested .lib section",
                    path, location.first_line
                )));
            }
            let name = strip_quotes(&stmt.arguments[0]).to_lowercase();
            if sections.contains(&name) {
                return Err(SpiceError::syntax(format!(
                    "{}:{}: duplicate .lib section {}",
                    path,
                    location.first_line,
                    py_repr(&name)
                )));
            }
            sections.insert(
                name.clone(),
                LibrarySection::new(name.clone(), location.clone()),
            );
            cur = Cur::Section(name);
            continue;
        }
        if stmt.kind == "endl" {
            if matches!(cur, Cur::Top) {
                return Err(SpiceError::syntax(format!(
                    "{}:{}: .endl outside a .lib section",
                    path, location.first_line
                )));
            }
            if let Some((_, name)) = &current_subckt {
                return Err(SpiceError::syntax(format!(
                    "{}:{}: .endl inside .subckt {}",
                    path,
                    location.first_line,
                    py_repr(name)
                )));
            }
            cur = Cur::Top;
            continue;
        }
        if stmt.kind == "subckt" {
            if current_subckt.is_some() {
                return Err(SpiceError::syntax(format!(
                    "{}:{}: nested .subckt",
                    path, location.first_line
                )));
            }
            let (name, terminals, parameters) = subckt_header(&stmt)?;
            let key = name.to_lowercase();
            let sub = Subcircuit {
                name: name.clone(),
                location: location.clone(),
                terminals,
                parameters,
                statements: Vec::new(),
            };
            match &cur {
                Cur::Top => top.subcircuits.insert(key.clone(), sub),
                Cur::Section(sk) => sections
                    .get_mut(sk)
                    .expect("open section exists")
                    .subcircuits
                    .insert(key.clone(), sub),
            }
            current_subckt = Some((key, name));
            continue;
        }
        if stmt.kind == "ends" {
            if current_subckt.is_none() {
                return Err(SpiceError::syntax(format!(
                    "{}:{}: .ends outside a .subckt",
                    path, location.first_line
                )));
            }
            current_subckt = None;
            continue;
        }

        // Ordinary statement: append to the open subcircuit or the current section.
        if let Some((key, _)) = &current_subckt {
            let section = match &cur {
                Cur::Top => &mut top,
                Cur::Section(sk) => sections.get_mut(sk).expect("open section exists"),
            };
            section
                .subcircuits
                .get_mut(key)
                .expect("open subckt exists")
                .statements
                .push(stmt);
        } else {
            match &cur {
                Cur::Top => top.statements.push(stmt),
                Cur::Section(sk) => sections
                    .get_mut(sk)
                    .expect("open section exists")
                    .statements
                    .push(stmt),
            }
        }
    }

    if let Some((_, name)) = &current_subckt {
        return Err(SpiceError::syntax(format!(
            "{}: unterminated .subckt {}",
            path,
            py_repr(name)
        )));
    }
    if let Cur::Section(sk) = &cur {
        let name = sections.get(sk).map(|s| s.name.clone()).unwrap_or_default();
        return Err(SpiceError::syntax(format!(
            "{}: unterminated .lib section {}",
            path,
            py_repr(&name)
        )));
    }
    Ok(SpiceModelLibrary {
        path: path.to_string(),
        top_level: top,
        sections,
    })
}

/// Python `repr()` of a string: single-quoted with the minimal escapes CPython
/// applies for the characters that occur in SPICE identifiers/paths.
fn py_repr(s: &str) -> String {
    let use_double = s.contains('\'') && !s.contains('"');
    let quote = if use_double { '"' } else { '\'' };
    let mut out = String::new();
    out.push(quote);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spice_number_suffixes_and_units() {
        // The reference computes `float(number) * factor`; keep that exact order
        // (e.g. `5 * 1e-6` is not the literal `5e-6`).
        assert_eq!(parse_spice_number("1").unwrap(), 1.0);
        assert_eq!(parse_spice_number("2.5k").unwrap(), 2.5f64 * 1e3);
        assert_eq!(parse_spice_number("3meg").unwrap(), 3.0f64 * 1e6);
        assert_eq!(parse_spice_number("4m").unwrap(), 4.0f64 * 1e-3);
        assert_eq!(parse_spice_number("5uF").unwrap(), 5.0f64 * 1e-6);
        assert_eq!(parse_spice_number("6mil").unwrap(), 6.0f64 * 25.4e-6);
        assert_eq!(parse_spice_number("  -2.5e3  ").unwrap(), -2500.0);
        assert_eq!(parse_spice_number("+.5").unwrap(), 0.5);
        assert!(parse_spice_number("abc").is_err());
    }

    #[test]
    fn logical_lines_join_and_locations() {
        let source = "* card\n.param a=1\n+ b='max(2, 3)' $ trailing comment\n\n.model nch nmos (\n+ level=54 version=4.5)\n";
        let lines = logical_lines(source, "<string>").unwrap();
        let texts: Vec<&str> = lines.iter().map(|(t, _)| t.as_str()).collect();
        assert_eq!(
            texts,
            vec![
                ".param a=1 b='max(2, 3)'",
                ".model nch nmos ( level=54 version=4.5)"
            ]
        );
        assert_eq!(lines[0].1.first_line, 2);
        assert_eq!(lines[0].1.last_line, 3);
    }

    #[test]
    fn orphan_continuation_errors() {
        let err = logical_lines("+ orphan", "<string>").unwrap_err();
        assert_eq!(err.kind, crate::error::ErrorKind::Syntax);
        assert!(err.message.contains("continuation"));
    }

    #[test]
    fn assignments_preserve_nested_and_quoted() {
        let parsed = parse_assignments(
            "(level=54 version=4.50 vth0='base + pwr(l, 2)' rdsw=max(0, r0 + delta))",
        )
        .unwrap();
        let pairs: Vec<(String, String)> = parsed
            .iter()
            .map(|a| (a.name.clone(), a.expression.clone()))
            .collect();
        assert_eq!(
            pairs,
            vec![
                ("level".into(), "54".into()),
                ("version".into(), "4.50".into()),
                ("vth0".into(), "'base + pwr(l, 2)'".into()),
                ("rdsw".into(), "max(0, r0 + delta)".into()),
            ]
        );
    }

    #[test]
    fn function_definition_formals() {
        let parsed = parse_assignments(
            "selbin(par1, par2, par3, par4)='max(par1, min(par2, par3))' scale=1",
        )
        .unwrap();
        assert_eq!(parsed[0].name, "selbin");
        assert_eq!(
            parsed[0].formal_parameters,
            vec!["par1", "par2", "par3", "par4"]
        );
        assert!(parsed[0].is_function());
        assert!(parsed[1].formal_parameters.is_empty());
    }

    #[test]
    fn model_subcircuit_and_lib_reference() {
        let source = "\n.lib tt\n.param scale=1 corner='max(0, process)'\n.model nch nmos level=54 version=4.5 vth0='0.4 + dvth'\n.subckt nch_mac d g s b w=1u l=30n nf=1\nm0 d g s b nch w=w l=l nf=nf\nr0 d s 10k\n.ends nch_mac\n.endl tt\n.lib \"other.lib\" support\n";
        let library = parse_spice_library_text(source, "<string>").unwrap();
        let tt = library.section("TT").unwrap();
        let models = tt.models();
        let nch = models.get("nch").unwrap();
        assert_eq!(nch.arguments, vec!["nmos"]);
        assert_eq!(nch.parameter_map().get("version").unwrap(), "4.5");
        let macro_ = tt.subcircuits.get("nch_mac").unwrap();
        assert_eq!(macro_.terminals, vec!["d", "g", "s", "b"]);
        assert_eq!(macro_.parameters[0].name, "w");
        let kinds: Vec<&str> = macro_.statements.iter().map(|s| s.kind.as_str()).collect();
        assert_eq!(kinds, vec!["m", "r"]);
        assert_eq!(library.top_level.statements[0].kind, "lib");
        assert_eq!(
            library.top_level.statements[0].arguments,
            vec!["\"other.lib\"", "support"]
        );
    }

    #[test]
    fn unterminated_structures_error() {
        assert_eq!(
            parse_spice_library_text(".subckt x a b\nm0 a b\n", "<string>")
                .unwrap_err()
                .kind,
            crate::error::ErrorKind::Syntax
        );
        assert_eq!(
            parse_spice_library_text(".lib tt\n.param a=1\n", "<string>")
                .unwrap_err()
                .kind,
            crate::error::ErrorKind::Syntax
        );
    }
}
