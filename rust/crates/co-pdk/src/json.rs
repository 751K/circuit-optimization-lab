//! A minimal reader for the flat `{ "name": number, ... }` SKY130 resolved-card
//! JSON, matching `json.loads(...)` followed by `float(value)`.
//!
//! Only what the card format uses is supported: a single top-level object whose
//! values are JSON numbers (booleans are accepted as `1.0`/`0.0`, mirroring
//! `float(bool)`). Numbers are parsed with Rust's correctly-rounded `f64` parser,
//! which agrees bit-for-bit with CPython's `float()` on the same token. Anything
//! else is an error, exactly as `float(value)` would raise for the Python adapter.

/// Parse a flat JSON object into `(raw_key, value)` pairs, preserving order.
pub fn parse_flat_object(text: &str) -> Result<Vec<(String, f64)>, String> {
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    let mut i = 0usize;

    skip_ws(&chars, &mut i);
    expect(&chars, &mut i, '{')?;
    skip_ws(&chars, &mut i);

    let mut out: Vec<(String, f64)> = Vec::new();
    if i < n && chars[i] == '}' {
        i += 1;
        skip_ws(&chars, &mut i);
        if i != n {
            return Err("trailing content after JSON object".to_string());
        }
        return Ok(out);
    }

    loop {
        skip_ws(&chars, &mut i);
        let key = parse_string(&chars, &mut i)?;
        skip_ws(&chars, &mut i);
        expect(&chars, &mut i, ':')?;
        skip_ws(&chars, &mut i);
        let value = parse_number_value(&chars, &mut i)?;
        out.push((key, value));
        skip_ws(&chars, &mut i);
        if i >= n {
            return Err("unterminated JSON object".to_string());
        }
        match chars[i] {
            ',' => {
                i += 1;
                continue;
            }
            '}' => {
                i += 1;
                break;
            }
            other => return Err(format!("unexpected character {other:?} in JSON object")),
        }
    }
    skip_ws(&chars, &mut i);
    if i != n {
        return Err("trailing content after JSON object".to_string());
    }
    Ok(out)
}

fn skip_ws(chars: &[char], i: &mut usize) {
    while *i < chars.len() && matches!(chars[*i], ' ' | '\t' | '\n' | '\r') {
        *i += 1;
    }
}

fn expect(chars: &[char], i: &mut usize, c: char) -> Result<(), String> {
    if *i < chars.len() && chars[*i] == c {
        *i += 1;
        Ok(())
    } else {
        Err(format!("expected {c:?} in JSON"))
    }
}

fn parse_string(chars: &[char], i: &mut usize) -> Result<String, String> {
    let n = chars.len();
    expect(chars, i, '"')?;
    let mut out = String::new();
    while *i < n {
        let c = chars[*i];
        *i += 1;
        match c {
            '"' => return Ok(out),
            '\\' => {
                if *i >= n {
                    return Err("unterminated JSON escape".to_string());
                }
                let esc = chars[*i];
                *i += 1;
                match esc {
                    '"' => out.push('"'),
                    '\\' => out.push('\\'),
                    '/' => out.push('/'),
                    'b' => out.push('\u{0008}'),
                    'f' => out.push('\u{000c}'),
                    'n' => out.push('\n'),
                    'r' => out.push('\r'),
                    't' => out.push('\t'),
                    'u' => {
                        if *i + 4 > n {
                            return Err("truncated \\u escape".to_string());
                        }
                        let hex: String = chars[*i..*i + 4].iter().collect();
                        let code = u32::from_str_radix(&hex, 16)
                            .map_err(|_| "invalid \\u escape".to_string())?;
                        *i += 4;
                        out.push(char::from_u32(code).unwrap_or('\u{fffd}'));
                    }
                    other => return Err(format!("invalid JSON escape \\{other}")),
                }
            }
            other => out.push(other),
        }
    }
    Err("unterminated JSON string".to_string())
}

/// Parse a JSON value that must be numeric (`float(value)` in the reference).
fn parse_number_value(chars: &[char], i: &mut usize) -> Result<f64, String> {
    let n = chars.len();
    if *i >= n {
        return Err("expected a JSON value".to_string());
    }
    // Accept booleans as `float(bool)` does; reject null (would raise in Python).
    if matches_keyword(chars, *i, "true") {
        *i += 4;
        return Ok(1.0);
    }
    if matches_keyword(chars, *i, "false") {
        *i += 5;
        return Ok(0.0);
    }
    let start = *i;
    if chars[*i] == '-' || chars[*i] == '+' {
        *i += 1;
    }
    let mut saw_digit = false;
    while *i < n && chars[*i].is_ascii_digit() {
        *i += 1;
        saw_digit = true;
    }
    if *i < n && chars[*i] == '.' {
        *i += 1;
        while *i < n && chars[*i].is_ascii_digit() {
            *i += 1;
            saw_digit = true;
        }
    }
    if !saw_digit {
        return Err("expected a JSON number".to_string());
    }
    if *i < n && (chars[*i] == 'e' || chars[*i] == 'E') {
        *i += 1;
        if *i < n && (chars[*i] == '+' || chars[*i] == '-') {
            *i += 1;
        }
        let mut saw_exp = false;
        while *i < n && chars[*i].is_ascii_digit() {
            *i += 1;
            saw_exp = true;
        }
        if !saw_exp {
            return Err("malformed JSON exponent".to_string());
        }
    }
    let token: String = chars[start..*i].iter().collect();
    token
        .parse::<f64>()
        .map_err(|_| format!("invalid JSON number {token:?}"))
}

fn matches_keyword(chars: &[char], i: usize, keyword: &str) -> bool {
    let kw: Vec<char> = keyword.chars().collect();
    i + kw.len() <= chars.len() && chars[i..i + kw.len()] == kw[..]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_flat_numeric_object() {
        let text = "{\n \"a0\": 1.5,\n \"b1\": 0.0,\n \"alpha0\": 3.75982e-08,\n \"bgidl\": 2300000000.0,\n \"cf\": 1.4067e-12\n}\n";
        let parsed = parse_flat_object(text).unwrap();
        assert_eq!(parsed.len(), 5);
        assert_eq!(parsed[0], ("a0".to_string(), 1.5));
        assert_eq!(parsed[2].1, 3.75982e-08);
        assert_eq!(parsed[3].1, 2300000000.0);
    }

    #[test]
    fn number_tokens_match_rust_parse() {
        for token in ["1", "2.0", "-3.5", "1e-13", "2.44907e-10", "0.955177"] {
            let text = format!("{{\"x\": {token}}}");
            let parsed = parse_flat_object(&text).unwrap();
            assert_eq!(parsed[0].1, token.parse::<f64>().unwrap());
        }
    }

    #[test]
    fn rejects_malformed() {
        assert!(parse_flat_object("{\"a\": }").is_err());
        assert!(parse_flat_object("{\"a\": 1}extra").is_err());
        assert!(parse_flat_object("[1, 2]").is_err());
    }
}
