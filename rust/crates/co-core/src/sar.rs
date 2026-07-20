//! Closed-loop SAR conversion driver (rewrite step R8).
//!
//! Ports the per-bit conversion loop of `circuitopt.sar.run_sar_conversion` into
//! the compiled core so a whole N-bit conversion — and, via
//! `co-py::sar_campaign`, a whole mismatch Monte-Carlo — runs under one
//! `py.detach` with no per-bit Python callback.
//!
//! Every numeric step mirrors the frozen Python path exactly:
//!
//!   * PWL waveform assembly (`_wave` + `sar_input_waveforms`): event sort +
//!     equal-time compaction + `np.interp`, in the same evaluation order, for
//!     the differential/single-ended and optional clocked-comparator forms.
//!   * grid expansion (`compact_models.bsim4.transient._expanded_grid`): each
//!     original interval is subdivided to at most `max_step`, waveforms are
//!     re-interpolated onto the expanded grid, and the solved trajectory is
//!     down-sampled back to the original grid before the comparator read.
//!   * the transient itself is `bsim_transient::solve_fixed_grid` — the same
//!     kernel the Python rust-engine path calls through `Bsim4TransientProblem`.
//!   * the comparator decision reads `np.interp(decision_time, tgrid, vout)` and
//!     applies the `high_means_clear` polarity, exactly as `run_sar_conversion`.
//!
//! Because bit decisions are discrete, the loop reproduces the frozen path's
//! codes bit-for-bit (the waveform/trajectory parity is `<= 1e-12`; a decision
//! only differs if a comparator sample sits within that band of the threshold).

use crate::bsim_transient::{self, Device, Evaluator, Options};
use crate::transient::{Problem as CircuitProblem, Waveforms};

/// The role a single marshalled input row plays in the SAR stimulus, in the
/// exact insertion order `sar_input_waveforms` emits its dict keys.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    /// `sample_input`: the top-plate sampling switch drive.
    Sample,
    /// `sample_bar_input`: `vref - sample` (computed on the original grid).
    SampleBar,
    /// `bit_inputs[k]`: the CDAC bit-k drive.
    BitInput(usize),
    /// `bit_inputs_bar[k]`: the complementary CDAC bit-k drive (differential).
    BitInputBar(usize),
    /// `dummy_input`: the differential dummy top plate.
    Dummy,
    /// `dummy_input_bar`: the complementary dummy top plate.
    DummyBar,
    /// `clock.input`: the strobe for a clocked (StrongARM) comparator.
    Clock,
}

/// Resolved `adc.clock` strobe block (see `sar._clock_config`).
#[derive(Clone, Copy, Debug)]
pub struct ClockConfig {
    pub high: f64,
    pub low: f64,
    pub eval_before: f64,
    pub reset_hold: f64,
}

/// Resolved SAR conversion parameters + waveform plan shared by every bit and
/// every conversion of a template.
#[derive(Clone, Debug)]
pub struct SarConfig {
    pub n_bits: usize,
    pub vref: f64,
    pub sample_end: f64,
    pub bit_period: f64,
    pub edge_time: f64,
    pub input_common_mode: f64,
    pub comparator_threshold: f64,
    pub high_means_clear: bool,
    pub differential: bool,
    /// Solved-node index of `comparator_node` in the state vector.
    pub comparator_index: usize,
    /// `tgrid[-1]` (last time point), the flat tail all waveforms hold to.
    pub tstop: f64,
    pub clock: Option<ClockConfig>,
    /// One entry per marshalled input row, aligned with the circuit's
    /// `input_index` wiring.
    pub roles: Vec<Role>,
    /// Newton/integration controls for `bsim_transient::solve_fixed_grid`.
    pub newton: Options,
}

/// Scalar `numpy.interp(x, xp, fp)` for strictly increasing `xp`.
///
/// Bit-for-bit port of the CPython `compiled_interp` branch: strict-`<`/`>`
/// clamping to the endpoints, the last-interval and exact-hit fast paths, the
/// `slope*(x-xp[j]) + fp[j]` form, and the NaN fall-back. `xp` is strictly
/// increasing (SAR grids and `_wave`'s compacted event times both are).
fn np_interp(x: f64, xp: &[f64], fp: &[f64]) -> f64 {
    let n = xp.len();
    if n == 1 {
        return fp[0];
    }
    if x > xp[n - 1] {
        return fp[n - 1];
    }
    if x < xp[0] {
        return fp[0];
    }
    // Largest j with xp[j] <= x (xp strictly increasing, x within [xp0, xpN-1]).
    let j = match xp.binary_search_by(|probe| probe.partial_cmp(&x).unwrap()) {
        Ok(hit) => hit,
        Err(insert) => insert - 1,
    };
    if j == n - 1 {
        return fp[n - 1];
    }
    if xp[j] == x {
        return fp[j];
    }
    let slope = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j]);
    let mut res = slope * (x - xp[j]) + fp[j];
    if res.is_nan() {
        res = slope * (x - xp[j + 1]) + fp[j + 1];
        if res.is_nan() && fp[j] == fp[j + 1] {
            res = fp[j];
        }
    }
    res
}

/// `circuitopt.sar._wave`: sort events by `(t, v)`, compact equal times keeping
/// the last, and `np.interp` onto `tgrid`.
fn wave(tgrid: &[f64], mut events: Vec<(f64, f64)>) -> Vec<f64> {
    events.sort_by(|a, b| a.partial_cmp(b).expect("SAR waveform events are finite"));
    let mut xs: Vec<f64> = Vec::with_capacity(events.len());
    let mut ys: Vec<f64> = Vec::with_capacity(events.len());
    for (t, v) in events {
        if xs.last().is_some_and(|&last| last == t) {
            *ys.last_mut().unwrap() = v;
        } else {
            xs.push(t);
            ys.push(v);
        }
    }
    tgrid.iter().map(|&x| np_interp(x, &xs, &ys)).collect()
}

/// Build the stage-1 (original-grid) waveform rows for the conversion of
/// `trial_index`, given `vin` and the decisions resolved so far.
///
/// Mirrors `sar_input_waveforms`: `decisions[b] == Some(0)` marks a cleared bit
/// (`b < trial_index`); the bit under trial (`b == trial_index`) is `None`.
fn build_original_rows(
    cfg: &SarConfig,
    vin: f64,
    decisions: &[Option<i32>],
    trial_index: usize,
    tgrid: &[f64],
) -> Vec<Vec<f64>> {
    let vref = cfg.vref;
    let edge = cfg.edge_time;
    let period = cfg.bit_period;
    let sample_end = cfg.sample_end;
    let tstop = cfg.tstop;
    let common_mode = cfg.input_common_mode;
    let hold_start = sample_end + edge;
    let hold_done = sample_end + 2.0 * edge;
    let differential = cfg.differential;
    let sampled_p = if differential {
        common_mode + 0.5 * vin
    } else {
        vin
    };
    let sampled_n = common_mode - 0.5 * vin;

    // Sample row (needed both for Role::Sample and Role::SampleBar).
    let sample = wave(
        tgrid,
        vec![
            (0.0, vref),
            (sample_end - edge, vref),
            (sample_end, 0.0),
            (tstop, 0.0),
        ],
    );

    let bit_row = |bit: usize| -> Vec<f64> {
        let baseline = if differential { common_mode } else { 0.0 };
        let mut events = vec![
            (0.0, sampled_p),
            (hold_start, sampled_p),
            (hold_done, baseline),
        ];
        if bit <= trial_index {
            let trial_start = sample_end + (bit as f64 + 0.5) * period;
            let decision_time = sample_end + (bit as f64 + 1.0) * period;
            events.push((trial_start, baseline));
            events.push((trial_start + edge, vref));
            if decisions[bit] == Some(0) {
                events.push((decision_time, vref));
                events.push((decision_time + edge, baseline));
            }
        }
        events.push((tstop, events.last().unwrap().1));
        wave(tgrid, events)
    };

    let bit_bar_row = |bit: usize| -> Vec<f64> {
        let mut events = vec![
            (0.0, sampled_n),
            (hold_start, sampled_n),
            (hold_done, common_mode),
        ];
        if bit <= trial_index {
            let trial_start = sample_end + (bit as f64 + 0.5) * period;
            let decision_time = sample_end + (bit as f64 + 1.0) * period;
            events.push((trial_start, common_mode));
            events.push((trial_start + edge, 0.0));
            if decisions[bit] == Some(0) {
                events.push((decision_time, 0.0));
                events.push((decision_time + edge, common_mode));
            }
        }
        events.push((tstop, events.last().unwrap().1));
        wave(tgrid, events)
    };

    let clock_row = || -> Vec<f64> {
        let ck = cfg.clock.expect("clock role requires clock config");
        let mut events = vec![(0.0, ck.low)];
        for bit in 0..cfg.n_bits {
            let decision_time = sample_end + (bit as f64 + 1.0) * period;
            let rise = decision_time - ck.eval_before;
            let fall = decision_time + ck.reset_hold;
            events.push((rise - edge, ck.low));
            events.push((rise, ck.high));
            events.push((fall, ck.high));
            events.push((fall + edge, ck.low));
        }
        events.push((tstop, ck.low));
        wave(tgrid, events)
    };

    cfg.roles
        .iter()
        .map(|role| match *role {
            Role::Sample => sample.clone(),
            Role::SampleBar => sample.iter().map(|&s| vref - s).collect(),
            Role::BitInput(bit) => bit_row(bit),
            Role::BitInputBar(bit) => bit_bar_row(bit),
            Role::Dummy => wave(
                tgrid,
                vec![
                    (0.0, sampled_p),
                    (hold_start, sampled_p),
                    (hold_done, common_mode),
                    (tstop, common_mode),
                ],
            ),
            Role::DummyBar => wave(
                tgrid,
                vec![
                    (0.0, sampled_n),
                    (hold_start, sampled_n),
                    (hold_done, common_mode),
                    (tstop, common_mode),
                ],
            ),
            Role::Clock => clock_row(),
        })
        .collect()
}

/// The constant (per-template) expanded time grid + down-sample map produced by
/// `_expanded_grid(tgrid, .., max_step=edge_time)`.
#[derive(Clone, Debug)]
pub struct ExpandedGrid {
    pub times: Vec<f64>,
    /// For each original time index, its position in `times`.
    pub requested_index: Vec<usize>,
}

impl ExpandedGrid {
    /// Port of `_expanded_grid`: subdivide each interval to at most `max_step`
    /// with `numpy.linspace` endpoints. `max_step <= 0` disables subdivision.
    pub fn build(tgrid: &[f64], max_step: f64) -> Self {
        if max_step <= 0.0 || max_step.is_nan() || tgrid.len() < 2 {
            return Self {
                times: tgrid.to_vec(),
                requested_index: (0..tgrid.len()).collect(),
            };
        }
        let mut times = vec![tgrid[0]];
        let mut requested_index = vec![0usize];
        for k in 1..tgrid.len() {
            let a = tgrid[k - 1];
            let b = tgrid[k];
            let count = ((b - a) / max_step).ceil().max(1.0) as usize;
            let step = (b - a) / count as f64;
            for i in 1..=count {
                // numpy.linspace sets only the final point exactly to `b`.
                let t = if i == count { b } else { (i as f64) * step + a };
                times.push(t);
            }
            requested_index.push(times.len() - 1);
        }
        Self {
            times,
            requested_index,
        }
    }
}

/// Outcome of one closed-loop conversion.
#[derive(Clone, Debug)]
pub struct Conversion {
    pub code: u64,
    pub bits: Vec<i32>,
}

/// Run one closed-loop SAR conversion for `vin`, returning the code and per-bit
/// decisions, or `None` if any bit's transient fails to complete (the frozen
/// Python path raises there; the caller maps `None` to a per-trial error).
#[allow(clippy::too_many_arguments)]
pub fn run_conversion<E: Evaluator>(
    circuit: &CircuitProblem,
    devices: &[Device],
    evaluator: &mut E,
    cfg: &SarConfig,
    v0: &[f64],
    tgrid: &[f64],
    grid: &ExpandedGrid,
    vin: f64,
) -> Option<Conversion> {
    let n_bits = cfg.n_bits;
    let n_rows = cfg.roles.len();
    let exp_n = grid.times.len();
    let mut decisions: Vec<Option<i32>> = vec![None; n_bits];
    let mut comparator = vec![0.0f64; tgrid.len()];
    let mut flat = vec![0.0f64; n_rows * exp_n];

    for bit in 0..n_bits {
        let orig_rows = build_original_rows(cfg, vin, &decisions, bit, tgrid);
        // Stage 2: re-interpolate each original-grid row onto the expanded grid.
        for (r, orig) in orig_rows.iter().enumerate() {
            let base = r * exp_n;
            for (m, &te) in grid.times.iter().enumerate() {
                flat[base + m] = np_interp(te, tgrid, orig);
            }
        }
        let waveforms = Waveforms::new(&flat, n_rows, exp_n)?;
        let result = bsim_transient::solve_fixed_grid(
            circuit,
            devices,
            evaluator,
            v0,
            &grid.times,
            waveforms,
            cfg.newton,
        );
        if !result.completed {
            return None;
        }
        // Down-sample the comparator node back to the original grid, then read
        // it at the decision instant (np.interp over the original grid).
        for (i, &ri) in grid.requested_index.iter().enumerate() {
            comparator[i] = result.states[ri][cfg.comparator_index];
        }
        let decision_time = cfg.sample_end + (bit as f64 + 1.0) * cfg.bit_period;
        let comparator_v = np_interp(decision_time, tgrid, &comparator);
        let high = comparator_v >= cfg.comparator_threshold;
        decisions[bit] = Some(if cfg.high_means_clear {
            i32::from(!high)
        } else {
            i32::from(high)
        });
    }

    let mut code: u64 = 0;
    let bits: Vec<i32> = (0..n_bits)
        .map(|bit| {
            let d = decisions[bit].unwrap_or(0);
            if d != 0 {
                code |= 1u64 << (n_bits - 1 - bit);
            }
            d
        })
        .collect();
    Some(Conversion { code, bits })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn np_interp_matches_grid_points_and_clamps() {
        let xp = [0.0, 1.0, 2.0, 3.0];
        let fp = [10.0, 20.0, 40.0, 80.0];
        assert_eq!(np_interp(-1.0, &xp, &fp), 10.0); // left clamp
        assert_eq!(np_interp(4.0, &xp, &fp), 80.0); // right clamp
        assert_eq!(np_interp(0.0, &xp, &fp), 10.0); // first point
        assert_eq!(np_interp(3.0, &xp, &fp), 80.0); // last point (j==n-1)
        assert_eq!(np_interp(1.0, &xp, &fp), 20.0); // exact hit
        assert_eq!(np_interp(0.5, &xp, &fp), 15.0); // interpolated
        assert_eq!(np_interp(2.5, &xp, &fp), 60.0);
    }

    #[test]
    fn wave_is_piecewise_linear_with_compaction() {
        let tgrid = [0.0, 0.5, 1.0, 1.5, 2.0];
        // Duplicate time at t=1.0: `sorted` orders (1,5) before (1,10), so the
        // larger value (10.0) is kept after equal-time compaction (matches the
        // Python `_wave` "last after sort wins" rule).
        let events = vec![(0.0, 0.0), (1.0, 10.0), (1.0, 5.0), (2.0, 5.0)];
        let got = wave(&tgrid, events);
        assert_eq!(got, vec![0.0, 5.0, 10.0, 7.5, 5.0]);
    }

    #[test]
    fn expanded_grid_no_subdivision_when_step_large() {
        let tgrid = [0.0, 1.0, 2.0];
        let grid = ExpandedGrid::build(&tgrid, 10.0);
        assert_eq!(grid.times, vec![0.0, 1.0, 2.0]);
        assert_eq!(grid.requested_index, vec![0, 1, 2]);
    }

    #[test]
    fn expanded_grid_subdivides_and_maps_back() {
        let tgrid = [0.0, 1.0];
        let grid = ExpandedGrid::build(&tgrid, 0.4); // ceil(1/0.4)=3 subintervals
        assert_eq!(grid.times.len(), 4);
        assert_eq!(grid.times[0], 0.0);
        assert_eq!(*grid.times.last().unwrap(), 1.0);
        assert_eq!(grid.requested_index, vec![0, 3]);
    }
}
