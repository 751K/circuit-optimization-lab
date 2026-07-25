//! Shared stimulus sampling for the adaptive drivers.
//!
//! Both adaptive transient engines (OTFT and BSIM) read the same
//! piecewise-linear source description: `inputs_at_time` interpolates the
//! sampled waveform rows at an arbitrary solve time, and `critical_times`
//! finds the interior samples where an input's slope breaks hard enough that
//! the stepper should land on them exactly and restart its history.

use crate::integrator::{ADAPTIVE_MIN_H_ABS, ADAPTIVE_MIN_H_REL};
use crate::transient::Waveforms;

/// Fraction of the steepest global slope a slope discontinuity must reach to
/// count as a breakpoint worth restarting on.
pub(crate) const INPUT_SLOPE_BREAK_FRACTION: f64 = 0.1;

/// Linearly interpolate every input row at `time` (clamped to the grid ends).
pub fn inputs_at_time(times: &[f64], inputs: Waveforms<'_>, time: f64) -> Vec<f64> {
    if time <= times[0] {
        return inputs.sample(0).unwrap_or_default();
    }
    let last = times.len() - 1;
    if time >= times[last] {
        return inputs.sample(last).unwrap_or_default();
    }
    let upper = times.partition_point(|value| *value < time);
    let interval = upper.saturating_sub(1).min(last - 1);
    let fraction = (time - times[interval]) / (times[interval + 1] - times[interval]);
    (0..inputs.rows())
        .map(|row| {
            let values = inputs.row(row).unwrap_or_default();
            values[interval] + (values[interval + 1] - values[interval]) * fraction
        })
        .collect()
}

/// Interior sample times whose slope jump exceeds
/// [`INPUT_SLOPE_BREAK_FRACTION`] of the steepest slope of any input.
///
/// Intervals narrower than the stepper's own minimum step are grid artefacts,
/// not stimulus features, and are excluded from every slope: merging an event
/// time into a requested grid can leave two samples a few ULP apart when the
/// event all but coincides with a sample. A zero-rise edge sampled across such
/// a pair swings full scale over a gap of ~3e-27 s, and the implied ~1e26 V/s
/// would otherwise set the scale that every real edge is compared against,
/// hiding all of them. A degenerate pair whose value does move is a genuine
/// discontinuity, so its later time is reported directly; one whose value does
/// not move says nothing at all and is ignored. On a grid with no degenerate
/// interval this is exactly the plain neighbour-slope test.
pub fn critical_times(times: &[f64], inputs: Waveforms<'_>) -> Vec<f64> {
    if inputs.rows() == 0 || times.len() < 3 {
        return Vec::new();
    }
    let span = times[times.len() - 1] - times[0];
    let min_step = ADAPTIVE_MIN_H_ABS.max(span * ADAPTIVE_MIN_H_REL);
    let steppable = |interval: usize| times[interval + 1] - times[interval] > min_step;
    let slope_over = |interval: usize| {
        let dt = times[interval + 1] - times[interval];
        (0..inputs.rows())
            .map(|row| {
                let values = inputs.row(row).unwrap_or_default();
                (values[interval + 1] - values[interval]) / dt
            })
            .collect::<Vec<f64>>()
    };
    let moved_across = |interval: usize| {
        (0..inputs.rows()).any(|row| {
            let values = inputs.row(row).unwrap_or_default();
            values[interval + 1] != values[interval]
        })
    };

    let mut global_slope = 1.0f64;
    for interval in 0..times.len() - 1 {
        if !steppable(interval) {
            continue;
        }
        for slope in slope_over(interval) {
            global_slope = global_slope.max(slope.abs());
        }
    }

    let mut critical = Vec::new();
    for position in 1..times.len() - 1 {
        if !steppable(position - 1) {
            if moved_across(position - 1) {
                critical.push(times[position]);
            }
            continue;
        }
        // Compare against the nearest interval on each side that the stepper
        // could actually take, so a degenerate pair between two real segments
        // cannot fabricate or mask a slope break.
        let Some(before) = (0..position).rev().find(|index| steppable(*index)) else {
            continue;
        };
        let Some(after) = (position..times.len() - 1).find(|index| steppable(*index)) else {
            continue;
        };
        let entering = slope_over(before);
        let leaving = slope_over(after);
        let jump = entering
            .into_iter()
            .zip(leaving)
            .fold(0.0f64, |worst, (slope0, slope1)| {
                worst.max((slope1 - slope0).abs())
            });
        if jump > INPUT_SLOPE_BREAK_FRACTION * global_slope {
            critical.push(times[position]);
        }
    }
    critical
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interpolation_matches_the_piecewise_linear_source() {
        let times = [0.0, 1.0, 3.0];
        let data = [0.0, 2.0, -2.0];
        let inputs = Waveforms::new(&data, 1, 3).unwrap();
        assert_eq!(inputs_at_time(&times, inputs, -1.0), vec![0.0]);
        assert_eq!(inputs_at_time(&times, inputs, 0.5), vec![1.0]);
        assert_eq!(inputs_at_time(&times, inputs, 1.0), vec![2.0]);
        assert_eq!(inputs_at_time(&times, inputs, 2.0), vec![0.0]);
        assert_eq!(inputs_at_time(&times, inputs, 9.0), vec![-2.0]);
    }

    #[test]
    fn slope_breaks_are_flagged_against_the_global_scale() {
        // A ramp corner at t=1 and a flat tail: the corner is critical.
        let times = [0.0, 1.0, 2.0, 3.0];
        let data = [0.0, 1.0, 1.0, 1.0];
        let inputs = Waveforms::new(&data, 1, 4).unwrap();
        assert_eq!(critical_times(&times, inputs), vec![1.0]);
    }

    /// A caller may hand the adaptive solver any grid, including one whose
    /// event time landed a few ULP from a sample. A zero-rise edge across such
    /// a pair used to set `global_slope` to ~1e26 V/s and hide every genuine
    /// edge below the threshold.
    #[test]
    fn a_degenerate_pair_cannot_poison_the_slope_scale() {
        let edge = 2e-11f64;
        let below = f64::from_bits(edge.to_bits() - 1);
        let times = [0.0, 1e-11, below, edge, 3e-11, 4e-11, 5e-11];
        assert!(edge - below < 1e-25, "expected a degenerate pair");
        // Row 0: a zero-rise square edge straddling the degenerate pair.
        // Row 1: a genuine finite-rise ramp between 4e-11 and 5e-11.
        let mut data = Vec::new();
        data.extend(times.iter().map(|t| if *t >= edge { 0.9 } else { 0.0 }));
        data.extend(
            times
                .iter()
                .map(|t| ((t - 4e-11) / 1e-11).clamp(0.0, 1.0) * 0.9),
        );
        let inputs = Waveforms::new(&data, 2, times.len()).unwrap();

        let critical = critical_times(&times, inputs);
        // The discontinuity itself is reported, at the time the new value
        // takes effect...
        assert!(critical.contains(&edge), "discontinuity lost: {critical:?}");
        // ...and the real ramp corner is still found, which is what the
        // poisoned scale used to suppress.
        assert!(
            critical.contains(&4e-11),
            "genuine ramp edge missed: {critical:?}"
        );
        assert!(
            critical.windows(2).all(|pair| pair[1] > pair[0]),
            "critical times must stay sorted and unique: {critical:?}"
        );
    }

    /// A degenerate pair the stimulus does not move across carries no
    /// information and must not become a restart point.
    #[test]
    fn a_degenerate_pair_without_a_value_change_is_ignored() {
        let edge = 2e-11f64;
        let times = [0.0, 1e-11, edge, f64::from_bits(edge.to_bits() + 1), 5e-11];
        let data: Vec<f64> = times
            .iter()
            .map(|t| if *t >= 1e-11 { 0.9 } else { 0.0 })
            .collect();
        let inputs = Waveforms::new(&data, 1, times.len()).unwrap();

        let critical = critical_times(&times, inputs);
        assert!(
            !critical.iter().any(|t| (*t - edge).abs() < 1e-25),
            "a flat degenerate pair became critical: {critical:?}"
        );
    }
}
