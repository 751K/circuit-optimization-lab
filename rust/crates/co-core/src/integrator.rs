//! The single home of the transient integration formulas and step policy.
//!
//! Every solver in this crate advances `q' ~ (a0*q_n + a1*q_{n-1} + a2*q_{n-2})`
//! with one shared BDF family walk: the first step (and any step without usable
//! history) is backward Euler, later steps use variable-step BDF2 ("gear2"),
//! and a step growing past [`GEAR2_MAX_STEP_RATIO`] falls back to backward
//! Euler for that sample. The BDF3-minus-BDF2 defect row drives the adaptive
//! local-error estimate.
//!
//! Two scaling conventions exist for historical reasons and both are kept
//! bit-exact here rather than unified numerically:
//!
//! * **scaled** rows carry the `1/h` factor (`[a0, a1, a2]` in 1/s) — the BSIM
//!   fixed-grid and adaptive solvers, and the rows exported to Python.
//! * **dimensionless** rows leave `1/h` to the stamp site — the OTFT solver
//!   and the LTI PSS/PAC companion maps.
//!
//! Converting one convention into the other reorders IEEE operations and moves
//! the last bits of every frozen golden result, so each caller keeps the exact
//! expression tree it was frozen with; the property tests at the bottom pin
//! both closed forms to the Lagrange generator that also produces the BDF3
//! defect weights.

/// A growing step keeps variable-step BDF2 only while `h/h_prev` stays at or
/// below this ratio; beyond it the sample falls back to backward Euler.
/// (Variable-step BDF2 loses zero-stability near ratio 1+sqrt(2); 2.0 is the
/// conservative bound every engine froze on.)
pub const GEAR2_MAX_STEP_RATIO: f64 = 2.0;

/// The Newton *seed* extrapolation tolerates larger growth than the corrector:
/// a poor seed only costs iterations, never accuracy, so the predictor stays
/// active in the ratio band (2, 4] where the corrector has already dropped to
/// backward Euler.
pub const PREDICTOR_MAX_STEP_RATIO: f64 = 4.0;
/// Fraction of quadratic input curvature blended into the predictor seed.
pub const PREDICTOR_INPUT_CURVATURE: f64 = 0.25;

// Step-control policy shared by every adaptive driver in this crate.
pub(crate) const ADAPTIVE_ACCEPT_WRMS: f64 = 1.0;
pub(crate) const ADAPTIVE_DONE_ABS: f64 = 1e-18;
pub(crate) const ADAPTIVE_DONE_REL: f64 = 1e-13;
pub(crate) const ADAPTIVE_ERR_FLOOR: f64 = 1e-12;
pub(crate) const ADAPTIVE_GROWTH_MAX: f64 = 2.0;
pub(crate) const ADAPTIVE_GROWTH_MIN: f64 = 0.2;
pub(crate) const ADAPTIVE_INITIAL_MIN_DENOM: usize = 16;
pub(crate) const ADAPTIVE_MIN_H_ABS: f64 = 1e-18;
pub(crate) const ADAPTIVE_MIN_H_REL: f64 = 1e-15;
pub(crate) const ADAPTIVE_SAFETY: f64 = 0.9;
pub(crate) const ADAPTIVE_SCALE_FLOOR: f64 = 1e-30;
pub(crate) const ADAPTIVE_STEP_ORDER: f64 = 3.0;

/// Backward-Euler derivative row with the `1/h` factor applied.
#[inline]
pub fn backward_euler_scaled(h: f64) -> [f64; 3] {
    [1.0 / h, -1.0 / h, 0.0]
}

/// Variable-step BDF2 derivative row with the `1/h` factor applied.
///
/// Callers must have checked [`gear2_ratio_ok`]; the expression tree is the
/// one the BSIM solvers froze on.
#[inline]
pub fn gear2_scaled(h: f64, h_previous: f64) -> [f64; 3] {
    [
        (2.0 * h + h_previous) / (h * (h + h_previous)),
        -(h + h_previous) / (h * h_previous),
        h / (h_previous * (h + h_previous)),
    ]
}

/// Backward-Euler derivative row in the dimensionless convention.
pub const BACKWARD_EULER_DIMENSIONLESS: [f64; 3] = [1.0, -1.0, 0.0];

/// Variable-step BDF2 derivative row in the dimensionless convention
/// (`ratio = h / h_previous`; the stamp site multiplies by `1/h`).
///
/// The expression tree is the one the OTFT solver and the LTI PSS/PAC maps
/// froze on.
#[inline]
pub fn gear2_dimensionless(ratio: f64) -> [f64; 3] {
    [
        (1.0 + 2.0 * ratio) / (1.0 + ratio),
        -(1.0 + ratio),
        ratio * ratio / (1.0 + ratio),
    ]
}

/// Whether a step may use variable-step BDF2 against its predecessor.
///
/// `h_previous <= 0` encodes "no usable history" (start of a solve or a
/// post-restart step); a non-finite or over-limit ratio also falls back.
#[inline]
pub fn gear2_ratio_ok(h: f64, h_previous: f64) -> bool {
    h_previous > 0.0 && h / h_previous <= GEAR2_MAX_STEP_RATIO
}

/// The scaled derivative row for one step: BDF2 when history and ratio allow,
/// backward Euler otherwise. This is the whole per-step method selection for
/// the scaled-convention solvers.
#[inline]
pub fn gear2_or_be_scaled(h: f64, h_previous: f64) -> [f64; 3] {
    if gear2_ratio_ok(h, h_previous) {
        gear2_scaled(h, h_previous)
    } else {
        backward_euler_scaled(h)
    }
}

/// The dimensionless derivative row for one step, with the same selection rule.
#[inline]
pub fn gear2_or_be_dimensionless(h: f64, h_previous: f64) -> [f64; 3] {
    if gear2_ratio_ok(h, h_previous) {
        gear2_dimensionless(h / h_previous)
    } else {
        BACKWARD_EULER_DIMENSIONLESS
    }
}

/// First-derivative weights at `t = 0` for four sample nodes (Lagrange form).
///
/// This is the generator behind the BDF family: three nodes give the BDF2 row,
/// four give the BDF3 row used for the defect estimate. The closed forms above
/// are pinned to it by the property tests below.
pub fn first_derivative_weights_at_zero(nodes: [f64; 4]) -> Option<[f64; 4]> {
    let mut weights = [0.0; 4];
    for (index, &node) in nodes.iter().enumerate() {
        let mut denominator = 1.0;
        for (other, &other_node) in nodes.iter().enumerate() {
            if other != index {
                denominator *= node - other_node;
            }
        }
        if denominator == 0.0 || !denominator.is_finite() {
            return None;
        }
        let mut numerator = 0.0;
        for (omitted, _) in nodes.iter().enumerate() {
            if omitted == index {
                continue;
            }
            let mut product = 1.0;
            for (other, &other_node) in nodes.iter().enumerate() {
                if other != index && other != omitted {
                    product *= -other_node;
                }
            }
            numerator += product;
        }
        weights[index] = numerator / denominator;
        if !weights[index].is_finite() {
            return None;
        }
    }
    Some(weights)
}

/// BDF3-minus-BDF2 defect weights over the last four samples.
///
/// `None` when fewer than three previous steps exist — the caller has no
/// error estimate for that trial and must control the step another way.
pub fn charge_defect_weights(
    step: f64,
    previous_step: f64,
    previous2_step: f64,
    bdf2: [f64; 3],
) -> Option<[f64; 4]> {
    if step <= 0.0 || previous_step <= 0.0 || previous2_step <= 0.0 {
        return None;
    }
    let ratio1 = previous_step / step;
    let ratio2 = previous2_step / step;
    let nodes = [0.0, -1.0, -(1.0 + ratio1), -(1.0 + ratio1 + ratio2)];
    let mut bdf3 = first_derivative_weights_at_zero(nodes)?;
    for weight in &mut bdf3 {
        *weight /= step;
    }
    Some([
        bdf3[0] - bdf2[0],
        bdf3[1] - bdf2[1],
        bdf3[2] - bdf2[2],
        bdf3[3],
    ])
}

/// Dimensionless derivative rows for every sample of a fixed grid.
///
/// Row zero has no derivative and is `[0, 0, 0]`; row one is backward Euler;
/// later rows follow the shared BDF2/backward-Euler selection. `None` when the
/// grid is not finite and strictly increasing. The LTI PSS/PAC companion maps
/// consume these rows; the stamp site supplies the `1/h` factor.
pub fn integration_rows_dimensionless(times: &[f64], gear2: bool) -> Option<Vec<[f64; 3]>> {
    if times.len() < 2
        || times.iter().any(|value| !value.is_finite())
        || times.windows(2).any(|window| window[1] <= window[0])
    {
        return None;
    }
    let mut rows = vec![[0.0; 3]; times.len()];
    for sample in 1..times.len() {
        let h = times[sample] - times[sample - 1];
        rows[sample] = if !gear2 || sample < 2 {
            BACKWARD_EULER_DIMENSIONLESS
        } else {
            gear2_or_be_dimensionless(h, times[sample - 1] - times[sample - 2])
        };
    }
    Some(rows)
}

/// The startup (and post-restart) trial step for an adaptive solve.
///
/// `fraction` of the smaller of the mean source interval and the smallest
/// *steppable* source interval. Merged event grids can hold two samples a few
/// ULP apart when an event nearly coincides with a grid point; such a gap says
/// nothing about how fast the stimulus moves and must not drive the startup
/// step below `min_step`, so intervals at or below `min_step` are ignored.
///
/// The two engines froze on different fractions (see
/// [`BSIM_STARTUP_FRACTION`] and [`OTFT_STARTUP_FRACTION`]); only the
/// degenerate-interval rule is shared.
pub fn startup_step(source_times: &[f64], span: f64, min_step: f64, fraction: f64) -> f64 {
    let denominator = (source_times.len() - 1).max(ADAPTIVE_INITIAL_MIN_DENOM);
    let smallest_source_step = source_times
        .windows(2)
        .map(|window| window[1] - window[0])
        .filter(|delta| *delta > min_step)
        .fold(span, f64::min);
    fraction * (span / denominator as f64).min(smallest_source_step)
}

/// The BSIM adaptive engine starts at a quarter of the source scale.
pub const BSIM_STARTUP_FRACTION: f64 = 0.25;
/// The OTFT adaptive engine starts at the full source scale; its step trial is
/// a step-halving comparison that shrinks on its own when the guess is coarse.
pub const OTFT_STARTUP_FRACTION: f64 = 1.0;

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() <= 1e-14 * expected.abs().max(1.0),
            "{actual} != {expected}"
        );
    }

    #[test]
    fn uniform_gear2_rows_match_the_classic_coefficients() {
        let scaled = gear2_scaled(2.0, 2.0);
        for (actual, expected) in scaled.into_iter().zip([0.75, -1.0, 0.25]) {
            assert_close(actual, expected);
        }
        let dimensionless = gear2_dimensionless(1.0);
        for (actual, expected) in dimensionless.into_iter().zip([1.5, -2.0, 0.5]) {
            assert_close(actual, expected);
        }
    }

    #[test]
    fn scaled_and_dimensionless_rows_agree_up_to_the_step_factor() {
        for (h, h_previous) in [(1.0, 1.0), (2.0, 1.5), (1e-11, 0.7e-11), (3.0, 1.5)] {
            let scaled = gear2_scaled(h, h_previous);
            let dimensionless = gear2_dimensionless(h / h_previous);
            for (a, b) in scaled.into_iter().zip(dimensionless) {
                assert_close(a * h, b);
            }
        }
    }

    /// First-derivative weights at 0 over three Lagrange nodes.
    fn three_node_derivative_weights(nodes: [f64; 3]) -> [f64; 3] {
        let mut weights = [0.0f64; 3];
        for (index, &node) in nodes.iter().enumerate() {
            let mut denominator = 1.0;
            for (other, &other_node) in nodes.iter().enumerate() {
                if other != index {
                    denominator *= node - other_node;
                }
            }
            let mut numerator = 0.0;
            for omitted in 0..nodes.len() {
                if omitted == index {
                    continue;
                }
                let mut product = 1.0;
                for (other, &other_node) in nodes.iter().enumerate() {
                    if other != index && other != omitted {
                        product *= -other_node;
                    }
                }
                numerator += product;
            }
            weights[index] = numerator / denominator;
        }
        weights
    }

    #[test]
    fn closed_forms_match_the_lagrange_generator() {
        for (h, h_previous) in [(1.0, 1.0), (2.0, 1.5), (0.5, 1.0), (1e-11, 0.9e-11)] {
            let ratio1 = h_previous / h;
            let lagrange = three_node_derivative_weights([0.0, -1.0, -(1.0 + ratio1)]);
            for (closed_value, lagrange_value) in
                gear2_scaled(h, h_previous).into_iter().zip(lagrange)
            {
                assert_close(closed_value, lagrange_value / h);
            }
            for (closed_value, lagrange_value) in gear2_dimensionless(h / h_previous)
                .into_iter()
                .zip(lagrange)
            {
                assert_close(closed_value, lagrange_value);
            }
            // Backward Euler is the two-node member: weights [1, -1] on [0, -1].
            assert_close(backward_euler_scaled(h)[0], 1.0 / h);
            assert_close(backward_euler_scaled(h)[1], -1.0 / h);
            assert_eq!(backward_euler_scaled(h)[2], 0.0);
        }
    }

    #[test]
    fn ratio_guard_matches_every_frozen_call_site() {
        assert!(gear2_ratio_ok(2.0, 1.0));
        assert!(!gear2_ratio_ok(2.0000001, 1.0));
        assert!(!gear2_ratio_ok(1.0, 0.0));
        assert!(!gear2_ratio_ok(1.0, -1.0));
        assert!(!gear2_ratio_ok(f64::INFINITY, 1.0));
        assert!(!gear2_ratio_ok(f64::NAN, 1.0));
        assert_eq!(gear2_or_be_scaled(2.0, -1.0), [0.5, -0.5, 0.0]);
        assert_eq!(
            gear2_or_be_dimensionless(5.0, 1.0),
            BACKWARD_EULER_DIMENSIONLESS
        );
    }

    #[test]
    fn defect_weights_need_three_previous_steps() {
        assert!(charge_defect_weights(1.0, -1.0, 1.0, gear2_scaled(1.0, 1.0)).is_none());
        assert!(charge_defect_weights(1.0, 1.0, -1.0, gear2_scaled(1.0, 1.0)).is_none());
        let weights = charge_defect_weights(2.0, 2.0, 2.0, gear2_scaled(2.0, 2.0)).unwrap();
        let expected = [1.0 / 6.0, -0.5, 0.5, -1.0 / 6.0];
        for (actual, expected) in weights.into_iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-14);
        }
        assert!(weights.into_iter().sum::<f64>().abs() < 1e-14);
    }

    #[test]
    fn startup_step_ignores_degenerate_source_intervals() {
        let times = [0.0, 1e-11, 1e-11 + 3e-27, 2e-11, 5e-9];
        let span = 5e-9;
        let min_step = 1e-18;
        let step = startup_step(&times, span, min_step, BSIM_STARTUP_FRACTION);
        // The 3e-27 gap is filtered; the smallest real interval is 1e-11 and
        // the mean-interval bound span/16 dominates it downward.
        assert!(step > min_step * 100.0, "startup collapsed: {step}");
        assert_close(step, BSIM_STARTUP_FRACTION * (span / 16.0).min(1e-11));
    }
}
