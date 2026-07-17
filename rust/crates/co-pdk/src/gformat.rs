//! Python `format(x, "g")` (default precision 6), used to build SKY130 card
//! filenames (`W{width:g}_L{length:g}`) exactly as `circuitopt.pdk.sky130`.
//!
//! The C `%g` rule: with precision `P = 6` (treated as 1 if 0), format with `%e`
//! to find the decimal exponent `X`; if `-4 <= X < P` use `%f` with precision
//! `P-1-X`, otherwise `%e` with precision `P-1`; then strip trailing zeros and a
//! trailing decimal point (no `#` flag). Exponents are printed Python-style:
//! a sign and at least two digits (`1e-05`).

/// Strip trailing zeros (and a dangling decimal point) from a fixed-point string.
fn strip_fixed(s: &str) -> String {
    if !s.contains('.') {
        return s.to_string();
    }
    let trimmed = s.trim_end_matches('0');
    let trimmed = trimmed.strip_suffix('.').unwrap_or(trimmed);
    trimmed.to_string()
}

/// Format a `%e` string's exponent the way Python does: `e`, a sign, and at
/// least two digits; strip trailing zeros from the mantissa.
fn format_scientific(sci: &str) -> String {
    let e = sci
        .find(['e', 'E'])
        .expect("scientific form has an exponent");
    let mantissa = strip_fixed(&sci[..e]);
    let exp: i32 = sci[e + 1..].parse().expect("valid exponent");
    let sign = if exp < 0 { '-' } else { '+' };
    format!("{mantissa}e{sign}{:02}", exp.abs())
}

/// `format(x, "g")` with the default precision of 6.
pub fn format_g(x: f64) -> String {
    if x == 0.0 {
        return if x.is_sign_negative() {
            "-0".to_string()
        } else {
            "0".to_string()
        };
    }
    if x.is_nan() {
        return "nan".to_string();
    }
    if x.is_infinite() {
        return if x < 0.0 { "-inf".into() } else { "inf".into() };
    }
    let negative = x < 0.0;
    let ax = x.abs();
    const P: i32 = 6;
    // Decimal exponent X from an `%e` render with P-1 fractional digits.
    let probe = format!("{:.*e}", (P - 1) as usize, ax);
    let exp: i32 = probe[probe.find('e').expect("exponent") + 1..]
        .parse()
        .expect("valid exponent");
    let body = if (-4..P).contains(&exp) {
        let precision = (P - 1 - exp).max(0) as usize;
        strip_fixed(&format!("{ax:.precision$}"))
    } else {
        format_scientific(&format!("{:.*e}", (P - 1) as usize, ax))
    };
    if negative { format!("-{body}") } else { body }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_python_g_for_geometry_values() {
        // Values drawn from the bundled SKY130 card filenames.
        let cases = [
            (1.0, "1"),
            (1.5, "1.5"),
            (10.0, "10"),
            (0.5, "0.5"),
            (0.15, "0.15"),
            (12.056, "12.056"),
            (18.7239, "18.7239"),
            (58.3602, "58.3602"),
            (0.666667, "0.666667"),
            (1.33333, "1.33333"),
            (1.66667, "1.66667"),
            (10.2721, "10.2721"),
            (24.0, "24"),
            (40.0, "40"),
            (2.0, "2"),
            (100.0, "100"),
        ];
        for (value, expected) in cases {
            assert_eq!(format_g(value), expected, "format_g({value})");
        }
    }

    #[test]
    fn scientific_and_edge_forms() {
        // X < -4 or X >= 6 selects the %e branch.
        assert_eq!(format_g(1e-5), "1e-05");
        assert_eq!(format_g(1.5e-5), "1.5e-05");
        assert_eq!(format_g(1e6), "1e+06");
        assert_eq!(format_g(1234567.0), "1.23457e+06");
        assert_eq!(format_g(0.0), "0");
        assert_eq!(format_g(-2.5), "-2.5");
        assert_eq!(format_g(123456.0), "123456");
        assert_eq!(format_g(1000000.0), "1e+06");
    }
}
