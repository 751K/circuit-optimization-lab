//! Device-agnostic batch **campaign** executor (rewrite step R5-C).
//!
//! A campaign evaluates a matrix of design candidates (each a size/geometry
//! vector, optionally at several process corners and mismatch samples) through a
//! device-build -> DC -> AC -> noise pipeline and reduces each to a small set of
//! scalar metrics. This module owns only the *orchestration and reductions*:
//!
//!   * a single Rayon pool sized to the requested worker count,
//!   * an adaptive choice between candidate-level and frequency-level
//!     parallelism (never both at once, so the one pool is never
//!     oversubscribed),
//!   * candidate-index-ordered write-back (a fixed seed and worker count give
//!     byte-identical, index-ordered output),
//!   * atomic progress + cooperative cancellation that never feed a numeric
//!     reduction,
//!   * the metric reductions `bw_from_gain` and `band_rms`, ported from the
//!     frozen Python scalar path.
//!
//! The device physics lives behind the [`CandidateEvaluator`] trait; concrete
//! evaluators (OTFT, silicon PDK) compose the `otft` / `lti` / `mna` kernels.
//! Nothing here touches Python: the whole batch runs under one `py.detach` on
//! the caller side.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use rayon::ThreadPoolBuilder;
use rayon::prelude::*;

/// -3 dB bandwidth from a gain-vs-frequency curve.
///
/// Bit-for-bit port of `circuitopt.ac_solver.bw_from_gain`: the peak is the
/// array maximum, the first grid point past the peak that drops to `peak/sqrt2`
/// brackets the crossing, and the crossing is interpolated in log-frequency
/// (falling back to linear when a bracket touches DC). Returns the last
/// frequency when the curve never crosses.
pub fn bw_from_gain(freqs: &[f64], gains: &[f64]) -> f64 {
    if freqs.is_empty() || gains.is_empty() {
        return f64::NAN;
    }
    // numpy `max` / first-`argmax` semantics.
    let mut peak = gains[0];
    let mut ipk = 0usize;
    for (i, &g) in gains.iter().enumerate() {
        if g > peak {
            peak = g;
            ipk = i;
        }
    }
    let a3 = peak / std::f64::consts::SQRT_2;
    let mut bw = freqs[freqs.len() - 1];
    for i in (ipk + 1)..gains.len() {
        if gains[i] <= a3 {
            let (f0, f1) = (freqs[i - 1], freqs[i]);
            let (g0, g1) = (gains[i - 1], gains[i]);
            if g1 == g0 {
                bw = f1;
            } else if f0 > 0.0 && f1 > 0.0 {
                let (x0, x1) = (f0.log10(), f1.log10());
                let x = x0 + (a3 - g0) * (x1 - x0) / (g1 - g0);
                let lo = x0.min(x1);
                let hi = x0.max(x1);
                bw = 10f64.powf(x.clamp(lo, hi));
            } else {
                bw = f0 + (a3 - g0) * (f1 - f0) / (g1 - g0);
            }
            break;
        }
    }
    bw
}

/// Band-limited RMS of a PSD via trapezoid integration over `[f_lo, f_hi]`.
///
/// Port of `circuitopt.noise_solver.band_rms`: keep the grid points with
/// `f_lo <= f <= f_hi`, trapezoid-integrate `psd` against `freqs`, and take the
/// square root. The sum is a straight sequential accumulation; NumPy's
/// `trapezoid` reduces through `np.sum` (8-way unrolled + pairwise), so the two
/// agree to a relative error well inside the `1e-12` parity gate rather than
/// bit-for-bit (documented deviation).
pub fn band_rms(freqs: &[f64], psd: &[f64], f_lo: f64, f_hi: f64) -> f64 {
    debug_assert_eq!(freqs.len(), psd.len());
    // Masked (freq, psd) pairs, preserving grid order.
    let mut xs: Vec<f64> = Vec::new();
    let mut ys: Vec<f64> = Vec::new();
    for (&f, &p) in freqs.iter().zip(psd.iter()) {
        if f >= f_lo && f <= f_hi {
            xs.push(f);
            ys.push(p);
        }
    }
    if xs.len() < 2 {
        return 0.0;
    }
    let mut acc = 0.0;
    for i in 0..xs.len() - 1 {
        let d = xs[i + 1] - xs[i];
        acc += d * ((ys[i] + ys[i + 1]) / 2.0);
    }
    acc.max(0.0).sqrt()
}

/// Which axis a single batch parallelizes over.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParallelAxis {
    /// Parallelize across candidates; each candidate solves its frequencies
    /// serially. Preferred when candidates outnumber workers.
    Candidate,
    /// Run candidates serially; each candidate may parallelize its frequency
    /// sweep. Preferred when there are few candidates but many frequencies.
    Frequency,
    /// Pick `Candidate` when there is candidate-level parallelism to exploit
    /// (`n >= workers` and `workers > 1`), else `Frequency`.
    Auto,
}

/// Batch scheduling knobs.
#[derive(Clone, Copy, Debug)]
pub struct BatchConfig {
    /// Worker threads for the (single) pool this batch installs. Clamped to `>= 1`.
    pub workers: usize,
    /// Parallelization axis (see [`ParallelAxis`]).
    pub axis: ParallelAxis,
}

impl BatchConfig {
    pub fn new(workers: usize) -> Self {
        Self {
            workers: workers.max(1),
            axis: ParallelAxis::Auto,
        }
    }

    fn resolved_axis(&self, n: usize) -> ParallelAxis {
        match self.axis {
            ParallelAxis::Auto => {
                if self.workers > 1 && n >= self.workers {
                    ParallelAxis::Candidate
                } else {
                    ParallelAxis::Frequency
                }
            }
            other => other,
        }
    }

    /// Whether an evaluator should solve its inner frequency sweep in parallel.
    /// True only on the `Frequency` axis, so the single pool is never nested.
    pub fn inner_parallel(&self, n: usize) -> bool {
        self.resolved_axis(n) == ParallelAxis::Frequency
    }
}

/// Shared, atomic progress counter + cooperative cancel flag.
///
/// Both are pure control signals: read for monitoring/cancellation only, never
/// mixed into a numeric reduction, so results stay independent of scheduling.
#[derive(Debug, Default)]
pub struct BatchProgress {
    completed: AtomicUsize,
    cancelled: AtomicBool,
}

impl BatchProgress {
    pub fn new() -> Self {
        Self::default()
    }

    /// Candidates whose compute has finished so far (monotonic).
    pub fn completed(&self) -> usize {
        self.completed.load(Ordering::Relaxed)
    }

    /// Request cooperative cancellation. Candidates already in flight finish;
    /// not-yet-started candidates are skipped (`None` in the output).
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Relaxed);
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Relaxed)
    }

    fn bump(&self) {
        self.completed.fetch_add(1, Ordering::Relaxed);
    }
}

/// One candidate's outcome: a success payload, or a per-candidate error message
/// (a single bad candidate must not sink the batch).
pub type CandidateOutcome<T> = Result<T, String>;

/// The per-candidate compute. Implementors must be `Sync` (the engine may call
/// `evaluate` from several worker threads) and must be deterministic in
/// `index`. `inner_parallel` tells the evaluator whether it may parallelize its
/// own frequency sweep (true only when the engine is running candidates
/// serially), which keeps the single Rayon pool free of nested oversubscription.
pub trait CandidateEvaluator: Sync {
    type Output: Send;

    fn evaluate(&self, index: usize, inner_parallel: bool) -> CandidateOutcome<Self::Output>;
}

/// Run `evaluator` over candidates `0..n`, returning results in candidate-index
/// order. A cancelled slot is `None`; a completed slot is `Some(Ok|Err)`.
///
/// The whole batch installs on one freshly built Rayon pool of `config.workers`
/// threads. On the `Candidate` axis candidates run in parallel and each solves
/// serially; on the `Frequency` axis candidates run serially and each may use
/// the pool for its frequency sweep. Either way exactly one pool is live and it
/// is never nested, so a fixed seed yields byte-identical, index-ordered output
/// for any worker count.
pub fn evaluate_batch<E: CandidateEvaluator>(
    evaluator: &E,
    n: usize,
    config: BatchConfig,
    progress: &BatchProgress,
) -> Vec<Option<CandidateOutcome<E::Output>>> {
    let workers = config.workers.max(1);
    let axis = config.resolved_axis(n);
    let inner_parallel = axis == ParallelAxis::Frequency;

    let run = || -> Vec<Option<CandidateOutcome<E::Output>>> {
        match axis {
            ParallelAxis::Candidate => (0..n)
                .into_par_iter()
                .map(|index| run_one(evaluator, index, inner_parallel, progress))
                .collect(),
            // Frequency / (Auto already resolved): candidates serial, inner may parallelize.
            _ => (0..n)
                .map(|index| run_one(evaluator, index, inner_parallel, progress))
                .collect(),
        }
    };

    // A dedicated, single pool gives deterministic worker control and keeps the
    // frequency-level `lti` par-iter on the same pool (no second pool, no
    // oversubscription). Fall back to the ambient pool if construction fails.
    match ThreadPoolBuilder::new().num_threads(workers).build() {
        Ok(pool) => pool.install(run),
        Err(_) => run(),
    }
}

fn run_one<E: CandidateEvaluator>(
    evaluator: &E,
    index: usize,
    inner_parallel: bool,
    progress: &BatchProgress,
) -> Option<CandidateOutcome<E::Output>> {
    if progress.is_cancelled() {
        return None;
    }
    let outcome = evaluator.evaluate(index, inner_parallel);
    progress.bump();
    Some(outcome)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bw_from_gain_flat_curve_returns_last_frequency() {
        let freqs = [1.0, 10.0, 100.0, 1000.0];
        let gains = [1.0, 1.0, 1.0, 1.0];
        assert_eq!(bw_from_gain(&freqs, &gains), 1000.0);
    }

    #[test]
    fn bw_from_gain_interpolates_in_log_space() {
        // Peak at f=1 (gain 2). -3 dB level = 2/sqrt(2) ~= 1.41421356.
        // Crossing between f=10 (2.0) and f=100 (1.0): log-space interpolation.
        let freqs = [1.0, 10.0, 100.0];
        let gains = [2.0, 2.0, 1.0];
        let a3 = 2.0 / std::f64::consts::SQRT_2;
        let (x0, x1) = (10f64.log10(), 100f64.log10());
        let x = x0 + (a3 - 2.0) * (x1 - x0) / (1.0 - 2.0);
        let expected = 10f64.powf(x.clamp(x0, x1));
        assert_eq!(bw_from_gain(&freqs, &gains), expected);
    }

    #[test]
    fn band_rms_trapezoid_matches_hand_value() {
        // psd = 1 everywhere over [0,1,2,3]; band [0,3] -> integral 3 -> sqrt(3).
        let freqs = [0.0, 1.0, 2.0, 3.0];
        let psd = [1.0, 1.0, 1.0, 1.0];
        let got = band_rms(&freqs, &psd, 0.0, 3.0);
        assert!((got - 3f64.sqrt()).abs() < 1e-15);
    }

    #[test]
    fn band_rms_masks_out_of_band_points() {
        let freqs = [0.01, 0.1, 1.0, 10.0, 1000.0];
        let psd = [5.0, 1.0, 1.0, 1.0, 5.0];
        // band [0.1, 10]: trapezoid over x=[0.1,1,10], y=[1,1,1] -> 9.9 -> sqrt.
        let got = band_rms(&freqs, &psd, 0.1, 10.0);
        let expected: f64 = ((0.9 * 1.0) + (9.0 * 1.0f64)).sqrt();
        assert!(
            (got - expected).abs() < 1e-12,
            "got {got}, expected {expected}"
        );
    }

    /// Deterministic mock: metric = index scaled, so any reorder is visible.
    struct MockEvaluator {
        scale: f64,
        fail_at: Option<usize>,
    }

    impl CandidateEvaluator for MockEvaluator {
        type Output = f64;

        fn evaluate(&self, index: usize, _inner_parallel: bool) -> CandidateOutcome<f64> {
            if self.fail_at == Some(index) {
                return Err(format!("candidate {index} forced failure"));
            }
            // A little arithmetic so parallel scheduling has something to race on.
            let mut acc = 0.0;
            for k in 0..1000 {
                acc += ((index * 1000 + k) as f64).sqrt();
            }
            Ok(self.scale * index as f64 + acc * 0.0)
        }
    }

    fn collect_ok(results: &[Option<CandidateOutcome<f64>>]) -> Vec<f64> {
        results
            .iter()
            .map(|slot| *slot.as_ref().unwrap().as_ref().unwrap())
            .collect()
    }

    #[test]
    fn batch_is_ordered_and_worker_count_invariant() {
        let evaluator = MockEvaluator {
            scale: 3.5,
            fail_at: None,
        };
        let n = 64;
        let baseline = evaluate_batch(
            &evaluator,
            n,
            BatchConfig {
                workers: 1,
                axis: ParallelAxis::Candidate,
            },
            &BatchProgress::new(),
        );
        let baseline = collect_ok(&baseline);
        // index order preserved.
        for (i, value) in baseline.iter().enumerate() {
            assert_eq!(*value, 3.5 * i as f64);
        }
        for workers in [1usize, 2, 8] {
            for axis in [
                ParallelAxis::Candidate,
                ParallelAxis::Frequency,
                ParallelAxis::Auto,
            ] {
                let progress = BatchProgress::new();
                let results =
                    evaluate_batch(&evaluator, n, BatchConfig { workers, axis }, &progress);
                assert_eq!(
                    collect_ok(&results),
                    baseline,
                    "workers={workers} axis={axis:?}"
                );
                assert_eq!(progress.completed(), n);
            }
        }
    }

    #[test]
    fn per_candidate_error_does_not_sink_batch() {
        let evaluator = MockEvaluator {
            scale: 1.0,
            fail_at: Some(7),
        };
        let results = evaluate_batch(&evaluator, 16, BatchConfig::new(4), &BatchProgress::new());
        for (i, slot) in results.iter().enumerate() {
            let outcome = slot.as_ref().unwrap();
            if i == 7 {
                assert!(outcome.is_err());
            } else {
                assert!(outcome.is_ok());
            }
        }
    }

    #[test]
    fn cancellation_is_cooperative_and_marks_skipped() {
        let evaluator = MockEvaluator {
            scale: 1.0,
            fail_at: None,
        };
        let progress = BatchProgress::new();
        progress.cancel();
        // Serial axis so cancellation is observed before the first candidate.
        let results = evaluate_batch(
            &evaluator,
            32,
            BatchConfig {
                workers: 1,
                axis: ParallelAxis::Frequency,
            },
            &progress,
        );
        assert!(results.iter().all(Option::is_none));
        assert_eq!(progress.completed(), 0);
    }
}
