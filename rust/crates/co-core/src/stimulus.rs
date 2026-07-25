//! Shared stimulus sampling for the adaptive drivers.
//!
//! Both adaptive transient engines (OTFT and BSIM) read the same
//! piecewise-linear source description: `inputs_at_time` interpolates the
//! sampled waveform rows at an arbitrary solve time, and `critical_times`
//! finds the interior samples where an input's slope breaks hard enough that
//! the stepper should land on them exactly and restart its history.

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
pub fn critical_times(times: &[f64], inputs: Waveforms<'_>) -> Vec<f64> {
    if inputs.rows() == 0 || times.len() < 3 {
        return Vec::new();
    }
    let mut global_slope = 1.0f64;
    for interval in 0..times.len() - 1 {
        let dt = times[interval + 1] - times[interval];
        for row in 0..inputs.rows() {
            let values = inputs.row(row).unwrap_or_default();
            global_slope = global_slope.max(((values[interval + 1] - values[interval]) / dt).abs());
        }
    }
    let mut critical = Vec::new();
    for position in 1..times.len() - 1 {
        let dt0 = times[position] - times[position - 1];
        let dt1 = times[position + 1] - times[position];
        let mut jump = 0.0f64;
        for row in 0..inputs.rows() {
            let values = inputs.row(row).unwrap_or_default();
            let slope0 = (values[position] - values[position - 1]) / dt0;
            let slope1 = (values[position + 1] - values[position]) / dt1;
            jump = jump.max((slope1 - slope0).abs());
        }
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
}
