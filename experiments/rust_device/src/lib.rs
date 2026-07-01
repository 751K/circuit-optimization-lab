use pyo3::prelude::*;
use std::f64;

// ── Pure math helpers ────────────────────────────────────────────────

/// ln(1 + exp(x)), overflow-safe.
#[inline(always)]
fn softplus(x: f64) -> f64 {
    if x > 0.0 {
        x + (-x).exp().ln_1p()
    } else {
        x.exp().ln_1p()
    }
}

/// 1/(1+exp(-x)), overflow-safe.
#[inline(always)]
fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        let z = (-x).exp();
        1.0 / (1.0 + z)
    } else {
        let z = x.exp();
        z / (1.0 + z)
    }
}

#[inline(always)]
fn fast_pow(x: f64, exp: f64) -> f64 {
    // x > 0 always for our use case — use the log/exp identity
    // which is faster than the general-purpose powf on most platforms.
    (x.ln() * exp).exp()
}

/// The 8 precomputed scalars that the hot path needs.
/// Created once per device from Python (`_precompute_constants`), then
/// called millions of times from transient / Newton loops.
#[pyclass]
struct PmosHot {
    vfb: f64,
    vss: f64,
    lc: f64,
    lambda: f64,
    contact_scale: f64,
    channel_exponent: f64,
    current_scale: f64,
    inv_rleak: f64,
}

#[pymethods]
impl PmosHot {
    #[pyo3(signature = (vfb, vss, lc, lambd, contact_scale, channel_exponent, current_scale, inv_rleak))]
    #[new]
    fn new(
        vfb: f64, vss: f64, lc: f64, lambd: f64,
        contact_scale: f64, channel_exponent: f64,
        current_scale: f64, inv_rleak: f64,
    ) -> Self {
        PmosHot { vfb, vss, lc, lambda: lambd, contact_scale, channel_exponent, current_scale, inv_rleak }
    }

    // ── Fast Ich-only channel (for residual evaluation) ──────────────

    /// `_eval_channel_ich_sorted(v_d, v_d1, Vg)`  →  scalar Ich
    fn eval_channel_ich_sorted(&self, v_d: f64, v_d1: f64, vg: f64) -> f64 {
        let arg_d1 = (v_d1 - vg + self.vfb) / self.vss;
        let arg_d  = (v_d  - vg + self.vfb) / self.vss;
        let vods = self.vss * softplus(arg_d1);
        let vodd = self.vss * softplus(arg_d);
        let chmod = 1.0 + self.lambda * (v_d1 - v_d);
        self.current_scale * (fast_pow(vods, self.channel_exponent) - fast_pow(vodd, self.channel_exponent)) * chmod
    }

    // ── Full branch currents ─────────────────────────────────────────

    /// `_eval_currents(Vs, Vd, Vg, Vs1, Vd1)`  →  (I_s_s1, I_s1_d1, I_d1_d, Ich, I_d1_d_leak)
    fn eval_currents(&self, vs: f64, vd: f64, vg: f64, vs1: f64, vd1: f64)
        -> (f64, f64, f64, f64, f64)
    {
        // Verilog-A node sorting
        let v_s  = if vs > vs1 { vs } else { vs1 };
        let v_s1 = if vs > vs1 { vs1 } else { vs };
        let v_d  = if vd > vd1 { vd } else { vd1 };
        let v_d1 = if vd > vd1 { vd1 } else { vd };

        // --- Contact model (between s and s1) ---
        let vt = -(0.0045 * (v_s - vg).powi(2) + 0.7125 * (v_s - vg) + 0.9625);
        let vods1 = self.vss * softplus((v_s - vg + vt) / self.vss);
        let vodd1 = self.vss * softplus((v_s1 - vg + vt) / self.vss);
        let ecsat = 0.85 * 20.0 / ((v_s - vg).abs() + 0.1);
        let lambdac = 1.0 / (self.lc * ecsat);
        let cmod = 1.0 + lambdac * (v_s - v_s1);
        let icont = self.contact_scale
            * (fast_pow(vods1, self.channel_exponent) - fast_pow(vodd1, self.channel_exponent))
            * cmod;
        let i_s_s1 = if vs > vs1 { icont } else { -icont };

        // --- Channel model (between d1 and d) ---
        let ich = self.eval_channel_ich_sorted(v_d, v_d1, vg);
        let i_d1_d_ch = if vs1 > vd { ich } else { -ich };
        let i_d1_d_leak = (vd1 - vd + 0.1) * self.inv_rleak;
        let i_d1_d = i_d1_d_ch + i_d1_d_leak;

        // --- Internal resistor (between s1 and d1) ---
        let i_s1_d1 = (vs1 - vd1) / 0.1;

        (i_s_s1, i_s1_d1, i_d1_d, ich, i_d1_d_leak)
    }

    // ── 2×2 Newton on internal nodes (s1, d1) ────────────────────────

    /// `_newton_internal(Vs, Vd, Vg, x0)`  →  (converged, Vs1, Vd1)
    fn newton_internal(&self, vs: f64, vd: f64, vg: f64,
                       x0_s1: f64, x0_d1: f64, tol: f64, maxit: u32)
        -> (bool, f64, f64)
    {
        let mut s1 = x0_s1;
        let mut d1 = x0_d1;
        let hj = 1e-6_f64;

        for _ in 0..maxit {
            // residuals at center
            let (i_s_s1, i_s1_d1, i_d1_d, _, _) = self.eval_currents(vs, vd, vg, s1, d1);
            let r1 = i_s_s1 - i_s1_d1;
            let r2 = i_s1_d1 - i_d1_d;

            if r1.abs() < tol && r2.abs() < tol {
                return (true, s1, d1);
            }

            // finite-difference 2×2 Jacobian
            let (i_s_s1_p, i_s1_d1_p, i_d1_d_p, _, _) = self.eval_currents(vs, vd, vg, s1 + hj, d1);
            let r1p = i_s_s1_p - i_s1_d1_p;
            let r2p = i_s1_d1_p - i_d1_d_p;

            let (i_s_s1_q, i_s1_d1_q, i_d1_d_q, _, _) = self.eval_currents(vs, vd, vg, s1, d1 + hj);
            let r1q = i_s_s1_q - i_s1_d1_q;
            let r2q = i_s1_d1_q - i_d1_d_q;

            let j00 = (r1p - r1) / hj;
            let j01 = (r1q - r1) / hj;
            let j10 = (r2p - r2) / hj;
            let j11 = (r2q - r2) / hj;

            // analytic 2×2 inverse:  [j11 -j01; -j10 j00] / det
            let det = j00 * j11 - j01 * j10;
            if det.abs() < 1e-30 {
                return (false, s1, d1);
            }
            let inv_det = 1.0 / det;
            let ds1 = -(j11 * r1 - j01 * r2) * inv_det;
            let dd1 = -(-j10 * r1 + j00 * r2) * inv_det;

            // damped step (max factor 1.0, min 0.0625)
            let mut alpha = 1.0_f64;
            for _ in 0..5 {
                let s1_try = s1 + alpha * ds1;
                let d1_try = d1 + alpha * dd1;
                let (is, id1_, id_, _, _) = self.eval_currents(vs, vd, vg, s1_try, d1_try);
                let r1n = is - id1_;
                let r2n = id1_ - id_;
                if r1n.abs() + r2n.abs() < r1.abs() + r2.abs() + 1e-30 {
                    s1 = s1_try;
                    d1 = d1_try;
                    break;
                }
                alpha *= 0.5;
            }
            if alpha < 0.07 {
                return (false, s1, d1);
            }
        }
        (false, s1, d1)
    }
}

// ── Scalar softplus/sigmoid (kept for testing) ────────────────────────

#[pyfunction]
fn softplus_scalar(x: f64) -> f64 { softplus(x) }

#[pyfunction]
fn sigmoid_scalar(x: f64) -> f64 { sigmoid(x) }

// ── Module ────────────────────────────────────────────────────────────

#[pymodule]
fn rust_device(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PmosHot>()?;
    m.add_function(wrap_pyfunction!(softplus_scalar, m)?)?;
    m.add_function(wrap_pyfunction!(sigmoid_scalar, m)?)?;
    Ok(())
}
