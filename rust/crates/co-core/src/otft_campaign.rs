//! AFE OTFT campaign evaluator (rewrite step R5-C).
//!
//! Composes the native `otft` device kernels, the dense `mna` solver, and the
//! complex `lti` MNA into a single per-candidate device-build -> DC -> AC ->
//! noise pipeline, driven by the device-agnostic [`crate::campaign`] engine.
//! Every metric mirrors the frozen Python scalar path
//! (`ac_solver.ac_solve` / `noise_solver.noise_analysis` / `corners.metrics`)
//! with the same kernels the Python "rust" engine already dispatches to, so a
//! shared DC operating point yields bit-identical AC/noise metrics.
//!
//! Only the AFE topology is required here (no external C dependency), which
//! makes it the reference vertical for parity. Silicon PDK families plug the
//! same engine in from `co-py` where the BSIM4 host is available.

use crate::campaign::{CandidateEvaluator, CandidateOutcome, band_rms, bw_from_gain};
use crate::{lti, mna, otft};

/// AT4000TG fixed model constants (the `PMOS_TFT` construction defaults). Only
/// geometry, `nf`, and the corner/mismatch shifts vary per candidate.
#[derive(Clone, Copy, Debug)]
pub struct OtftConstants {
    pub vt: f64,
    pub ci: f64,
    pub roff: f64,
    pub reg: f64,
    pub c1: f64,
    pub c2: f64,
    pub c3: f64,
    pub c4: f64,
    pub kv: f64,
    pub kh: f64,
    pub temperature: f64,
}

impl Default for OtftConstants {
    fn default() -> Self {
        // Mirrors `PMOS_TFT.__init__` defaults.
        Self {
            vt: -3.03,
            ci: 2.4,
            roff: 1.0,
            reg: 1.0,
            c1: 37.5,
            c2: 50.0,
            c3: 35.0,
            c4: 35.0,
            kv: 1.0,
            kh: 1.0,
            temperature: 300.15,
        }
    }
}

/// A compiled DC/bias terminal token: `Solved(index)` reads a node voltage,
/// `Rail(value)` is a fixed bias. Mirrors `compiled_topology` TERM_SOLVED /
/// TERM_RAIL (TERM_INPUT is transient-only and unused here).
#[derive(Clone, Copy, Debug)]
pub enum BiasTerm {
    Solved(usize),
    Rail(f64),
}

impl BiasTerm {
    #[inline]
    fn eval(self, x: &[f64]) -> f64 {
        match self {
            BiasTerm::Solved(i) => x[i],
            BiasTerm::Rail(v) => v,
        }
    }
}

/// Immutable per-device topology template (candidate-invariant).
#[derive(Clone, Debug)]
pub struct DeviceTemplate {
    // DC bias tokens (Vs, Vd, Vg) and the KCL rows this device stamps.
    pub dc_d: BiasTerm,
    pub dc_g: BiasTerm,
    pub dc_s: BiasTerm,
    pub di: Option<usize>,
    pub si: Option<usize>,
    // AC small-signal terminals (kind 0 node / kind 2 fixed).
    pub ac_d: mna::Term,
    pub ac_g: mna::Term,
    pub ac_s: mna::Term,
    // Noise injection node indices (drain, source) — derived from the AC terms.
    pub noise_d: Option<usize>,
    pub noise_s: Option<usize>,
}

/// Immutable campaign template: topology arrays + analysis plan. Built once from
/// `CompiledTopology(AFE_TOPO, bias)` on the Python side and shared read-only
/// across the whole batch.
#[derive(Clone, Debug)]
pub struct OtftTemplate {
    pub n_aug: usize,
    pub n_nodes: usize,
    pub consts: OtftConstants,
    pub devices: Vec<DeviceTemplate>,
    /// AC load capacitors `(a_term, b_term, value)`.
    pub ac_caps: Vec<(mna::Term, mna::Term, f64)>,
    /// Output combination `(node_index, weight)` — e.g. VOP:+1, VON:-1.
    pub output_weights: Vec<(usize, f64)>,
    /// Transposed-noise sense vector (length `n_aug`).
    pub sense: Vec<f64>,
    pub vin_norm: f64,
    pub freqs: Vec<f64>,
    pub band_lo: f64,
    pub band_hi: f64,
    pub gmin: f64,
    pub dc_tol: f64,
    /// Cold-solve DC guess vectors (from `topo.dc_guess_vectors(bias)`).
    pub dc_guesses: Vec<Vec<f64>>,
    /// Output node pair for the latch metric `|V0 - V1|`, if two outputs.
    pub latch_nodes: Option<(usize, usize)>,
}

/// Per-device geometry + corner/mismatch shift for one candidate.
#[derive(Clone, Copy, Debug)]
pub struct DeviceGeom {
    pub w: f64,
    pub l: f64,
    pub nf: f64,
    pub pvt0: f64,
    pub mvt0: f64,
    pub pbeta0: f64,
    pub mbeta0: f64,
}

/// One candidate: per-device geometry, an optional DC seed, and whether to trust
/// that seed as the operating point directly (bit-exact AC/noise isolation) or
/// refine it with the Rust Newton (the production behaviour).
#[derive(Clone, Debug)]
pub struct OtftCandidate {
    pub devices: Vec<DeviceGeom>,
    pub seed: Option<Vec<f64>>,
    pub trust_seed_as_op: bool,
}

/// The scalar metrics for one candidate — keys mirror `corners.metrics`.
#[derive(Clone, Debug)]
pub struct OtftMetrics {
    pub gain_peak_db: f64,
    pub bw_hz: f64,
    pub irn_uv: f64,
    pub latch_dv: f64,
    pub dc_op: Vec<f64>,
    pub dc_iterations: usize,
    pub dc_from_seed: bool,
}

/// A built device instance: the 16-scalar OTFT ABI plus the extra constants the
/// noise PSD needs.
#[derive(Clone, Copy, Debug)]
struct BuiltDevice {
    params: otft::Params,
    ci: f64,
    w_m: f64,
    l_m: f64,
    temperature: f64,
}

/// Order-preserving port of `PMOS_TFT._precompute_constants`: geometry + corner
/// shifts -> the 16-scalar `otft::Params` ABI (+ noise constants).
fn build_device(geom: &DeviceGeom, c: &OtftConstants) -> BuiltDevice {
    use std::f64::consts::PI;
    // Fixed physical constants (PMOS_TFT.__init__).
    let q: f64 = 1.6e-19;
    let kbe: f64 = 8.617343e-5;
    let e0: f64 = 8.85418781e-14;
    let ks: f64 = 3.0;
    let alfa: f64 = 4.5455e7;
    let ve: f64 = 3.6;
    let nt: f64 = 1.0000e21;
    let nss: f64 = 2.0006e11;
    let s0: f64 = 5.9658e6;
    let tt: f64 = 304.9889;
    let bc: f64 = 2.8;

    let (w, l, nf) = (geom.w, geom.l, geom.nf);
    let t = 295.0 - (c.temperature - 295.0) / 4.0;
    let w_m = w * 1e-6;
    let l_m = l * 1e-6;
    let two_over_pi = 2.0 / PI;
    let cpvt = 1.0 + geom.pvt0 + geom.mvt0 / (w * l * 1e-12).sqrt();
    let vfb = c.vt * (3.25 / c.ci) * cpvt;
    let beta0 = 1.0 + geom.mbeta0 + geom.pbeta0;
    let ceff = (c.ci / 3.25 * 5.000) * 1e-9;
    let ci_eff = c.ci / cpvt * 1e-9;
    let es = e0 * ks;
    let g0 = s0 * ((PI * (tt / t).powi(3) * nt) / (bc * (2.0 * alfa).powi(3))).powf(tt / t);
    let rleak = (2.0 * 1e12 / w) * l * c.roff;
    let inv_rleak = 1.0 / rleak;
    let term1 = es * g0 / ceff;
    let term2 = (kbe * t) * (t / (2.0 * tt - t));
    let term3 =
        ((ceff * ceff * (PI * t / tt).sin()) / (PI * nt * 2.0 * q * kbe * t * es)).powf(tt / t);
    let beta = beta0 * 0.8 * term1 * term2 * term3;
    let exponent = 2.0 * tt / t;
    let current_scale = (w / l) * beta;
    let vss = 2.0 * kbe * tt * (1.0 + q * nss / ceff) * 2.0 * tt / (2.0 * tt - t);
    let lambda = 1.0 / (l * ve * (c.ci / 3.25));
    let lc = 5.0;
    let k0 = 1.1e-8;
    let contact_scale = (w / lc) * k0;
    let fw = w / nf;
    let cl = 10.0;
    let osc_o1 = c.c2 - c.c3 + c.c4;
    // Python nests ceil(ceil(A)/2); reproduce exactly.
    let a_ox = ((cl + l) * nf + cl + 2.0 * osc_o1 + c.kv * c.c2) / c.c1;
    let edge_ox = 2.0 * c.c3 + 2.0 * c.c1 * (a_ox.ceil() / 2.0).ceil();
    let a_oy = (fw + 2.0 * osc_o1 + (c.kh - 1.0) * c.c2) / c.c1;
    let edge_oy = 2.0 * c.c3 + 2.0 * c.c1 * (a_oy.ceil() / 2.0).ceil();
    let aosc = edge_ox * edge_oy;
    let cap_cgs1 = ((nf / 2.0).floor() + 1.0) * (fw + 210.0) * cl * ci_eff;
    let cap_cgd1 = (nf / 2.0).ceil() * (fw + 210.0) * cl * ci_eff;
    let cap_wl_ci = w * l * ci_eff;
    let cap_half_wl_ci = 0.5 * w * l * ci_eff;
    let cap_cgs3_base = 0.5 * aosc * ci_eff - 1.43 * cap_wl_ci;
    let cap_cgd3_base = 0.5 * aosc * ci_eff - 0.33 * cap_wl_ci;
    let k1 = match c.reg as i64 {
        1 => 1.0,
        2 => 0.0,
        3 => 0.1,
        4 => 0.001,
        _ => 0.0,
    };
    let r_cap2 = 1e12 * cpvt;
    let gate_leak_g = 1.0 / r_cap2;

    let params = otft::Params {
        vfb,
        vss,
        lc,
        lambda,
        contact_scale,
        exponent,
        current_scale,
        inv_rleak,
        two_over_pi,
        cap_cgs1,
        cap_cgd1,
        cap_half_wl_ci,
        cap_cgs3_base,
        cap_cgd3_base,
        k1,
        gate_leak_g,
    };
    BuiltDevice {
        params,
        ci: ci_eff,
        w_m,
        l_m,
        temperature: c.temperature,
    }
}

#[inline]
fn inf_norm(v: &[f64]) -> f64 {
    v.iter().fold(0.0, |acc, x| acc.max(x.abs()))
}

/// Internal-node OP `(Vs1, Vd1)` via the warm/cold Newton, mirroring
/// `PMOS_TFT._solve_internal` cold path (cold seed then a robust guess sweep,
/// with the Rust `newton_internal_fast` standing in for the scipy fallback).
fn solve_internal(bd: &BuiltDevice, vs: f64, vd: f64, vg: f64) -> Option<(f64, f64)> {
    let cold = (vs - 0.01 * (vs - vd), vd + 0.01 * (vs - vd));
    let r = otft::newton_internal_fast(&bd.params, vs, vd, vg, cold.0, cold.1, 1e-12, 40);
    if r.converged {
        return Some((r.vs1, r.vd1));
    }
    let guesses = [
        (vs - 0.01 * (vs - vd), vd + 0.01 * (vs - vd)),
        (vs, vd),
        ((vs + vd) / 2.0, (vs + vd) / 2.0),
        (vs, vs),
        (vd, vd),
    ];
    for (x0s, x0d) in guesses {
        let r = otft::newton_internal_fast(&bd.params, vs, vd, vg, x0s, x0d, 1e-12, 40);
        if r.converged {
            return Some((r.vs1, r.vd1));
        }
    }
    None
}

/// Signed DC drain current `get_Idc = -I_d1_d`, or `None` if the internal solve
/// fails.
fn get_idc(bd: &BuiltDevice, vs: f64, vd: f64, vg: f64) -> Option<f64> {
    let (vs1, vd1) = solve_internal(bd, vs, vd, vg)?;
    let currents = otft::eval_currents(&bd.params, vs, vd, vg, vs1, vd1);
    Some(-currents[2])
}

/// Small-signal `(gm, gds, cgs, cgd)` at a bias — mirror of
/// `PMOS_TFT.get_ss_params` (analytic terminal derivatives with a finite-diff
/// fallback under the small-current guard).
fn ss_params(
    bd: &BuiltDevice,
    vs: f64,
    vd: f64,
    vg: f64,
) -> Option<(f64, f64, f64, f64, f64, f64)> {
    let (vs1, vd1) = solve_internal(bd, vs, vd, vg)?;
    let currents = otft::eval_currents(&bd.params, vs, vd, vg, vs1, vd1);
    let idc0 = -currents[2];
    let cc = otft::capacitance_charges(&bd.params, vs, vd, vg, vs1, vd1);
    let (cgs, cgd) = (cc[2], cc[3]);
    if idc0.abs() < 1e-10 {
        // Pure central-difference fallback on get_Idc (h = 1e-3).
        let h = 1e-3;
        let gm = (get_idc(bd, vs, vd, vg + h)? - get_idc(bd, vs, vd, vg - h)?) / (2.0 * h);
        let gds = (get_idc(bd, vs, vd + h, vg)? - get_idc(bd, vs, vd - h, vg)?) / (2.0 * h);
        return Some((gm, gds, cgs, cgd, vs1, vd1));
    }
    let jac = otft::residual_pair_jac_internal(&bd.params, vs, vd, vg, vs1, vd1);
    let idc0b = jac[1] - (vs1 - vd1) / 0.1;
    let (ok, gm_neg, gds_neg) = otft::terminal_derivatives_from_jac(
        &bd.params,
        vs,
        vd,
        vg,
        vs1,
        vd1,
        jac[0],
        jac[1],
        idc0b,
        [jac[2], jac[3], jac[4], jac[5]],
        true,
        true,
        false,
        1e-3,
    );
    if ok && gm_neg.is_finite() && gds_neg.is_finite() {
        return Some((-gm_neg, -gds_neg, cgs, cgd, vs1, vd1));
    }
    let h = 1e-3;
    let gm = (get_idc(bd, vs, vd, vg + h)? - get_idc(bd, vs, vd, vg - h)?) / (2.0 * h);
    let gds = (get_idc(bd, vs, vd + h, vg)? - get_idc(bd, vs, vd - h, vg)?) / (2.0 * h);
    Some((gm, gds, cgs, cgd, vs1, vd1))
}

/// The AFE OTFT batch evaluator.
pub struct OtftEvaluator<'a> {
    pub template: &'a OtftTemplate,
    pub candidates: &'a [OtftCandidate],
    /// When false, the noise (transpose) stage is skipped and `irn_uv` is NaN —
    /// the AC-only prefilter path.
    pub compute_noise: bool,
}

impl<'a> OtftEvaluator<'a> {
    /// KCL residual (length `n_aug`) at node vector `x`, gmin on node rows.
    fn residual(&self, built: &[BuiltDevice], x: &[f64]) -> Vec<f64> {
        let t = self.template;
        let mut res = vec![0.0; t.n_aug];
        for (dev, bd) in t.devices.iter().zip(built) {
            let vs = dev.dc_s.eval(x);
            let vd = dev.dc_d.eval(x);
            let vg = dev.dc_g.eval(x);
            // Id = kcl_sign(+1) * abs(get_Idc); a failed internal solve -> 1e-18.
            let i = match get_idc(bd, vs, vd, vg) {
                Some(v) => v.abs(),
                None => 1e-18,
            };
            if let Some(di) = dev.di {
                res[di] += i;
            }
            if let Some(si) = dev.si {
                res[si] -= i;
            }
        }
        for k in 0..t.n_nodes {
            res[k] -= x[k] * t.gmin;
        }
        res
    }

    /// Damped Newton on the node system from one guess. Returns `(x, iters)` on
    /// convergence (∞-norm residual < `dc_tol`).
    fn newton(&self, built: &[BuiltDevice], x0: &[f64]) -> Option<(Vec<f64>, usize)> {
        let t = self.template;
        let n = t.n_aug;
        let mut x = x0.to_vec();
        let maxit = 200usize;
        for it in 0..maxit {
            let r = self.residual(built, &x);
            let rnorm = inf_norm(&r);
            if rnorm < t.dc_tol {
                return Some((x, it));
            }
            // Forward-difference Jacobian (row-major n*n).
            let mut jac = vec![0.0; n * n];
            for col in 0..n {
                let h = 1e-6 * x[col].abs().max(1.0);
                let mut xp = x.clone();
                xp[col] += h;
                let rp = self.residual(built, &xp);
                for row in 0..n {
                    jac[row * n + col] = (rp[row] - r[row]) / h;
                }
            }
            let mut rhs = r.clone();
            if !mna::solve_dense_neg_rhs_in_place(&mut jac, &mut rhs) {
                return None;
            }
            // rhs now holds dx = J^{-1}(-r). Backtracking line search.
            let mut alpha = 1.0;
            let mut accepted = false;
            for _ in 0..24 {
                let xt: Vec<f64> = x.iter().zip(&rhs).map(|(xi, d)| xi + alpha * d).collect();
                if xt.iter().all(|v| v.is_finite()) {
                    let rt = self.residual(built, &xt);
                    if inf_norm(&rt) < rnorm {
                        x = xt;
                        accepted = true;
                        break;
                    }
                }
                alpha *= 0.5;
            }
            if !accepted {
                for k in 0..n {
                    x[k] += rhs[k];
                }
                if !x.iter().all(|v| v.is_finite()) {
                    return None;
                }
            }
        }
        let r = self.residual(built, &x);
        if inf_norm(&r) < t.dc_tol {
            Some((x, maxit))
        } else {
            None
        }
    }

    /// DC operating point for one candidate.
    fn solve_dc(
        &self,
        cand: &OtftCandidate,
        built: &[BuiltDevice],
    ) -> Result<(Vec<f64>, usize, bool), String> {
        let t = self.template;
        if let (true, Some(seed)) = (cand.trust_seed_as_op, &cand.seed) {
            if seed.len() != t.n_aug {
                return Err(format!("seed length {} != n_aug {}", seed.len(), t.n_aug));
            }
            return Ok((seed.clone(), 0, true));
        }
        let mut guesses: Vec<Vec<f64>> = Vec::new();
        if let Some(seed) = &cand.seed {
            if seed.len() != t.n_aug {
                return Err(format!("seed length {} != n_aug {}", seed.len(), t.n_aug));
            }
            guesses.push(seed.clone());
        }
        for g in &t.dc_guesses {
            guesses.push(g.clone());
        }
        for (gi, g) in guesses.iter().enumerate() {
            if let Some((x, iters)) = self.newton(built, g) {
                let from_seed = gi == 0 && cand.seed.is_some();
                return Ok((x, iters, from_seed));
            }
        }
        Err("DC did not converge from any guess".to_string())
    }
}

impl CandidateEvaluator for OtftEvaluator<'_> {
    type Output = OtftMetrics;

    fn evaluate(&self, index: usize, inner_parallel: bool) -> CandidateOutcome<OtftMetrics> {
        let t = self.template;
        let cand = self
            .candidates
            .get(index)
            .ok_or_else(|| format!("candidate index {index} out of range"))?;
        if cand.devices.len() != t.devices.len() {
            return Err(format!(
                "candidate {index} has {} devices, template has {}",
                cand.devices.len(),
                t.devices.len()
            ));
        }
        // 1. Build device instances (geometry + corner/mismatch -> ABI params).
        let built: Vec<BuiltDevice> = cand
            .devices
            .iter()
            .map(|g| build_device(g, &t.consts))
            .collect();

        // 2. DC operating point.
        let (dc_op, dc_iterations, dc_from_seed) = self.solve_dc(cand, &built)?;

        // 3. Per-device small-signal params at the DC bias.
        let mut mos = Vec::with_capacity(t.devices.len());
        let mut internal = Vec::with_capacity(t.devices.len());
        for (dev, bd) in t.devices.iter().zip(&built) {
            let vs = dev.dc_s.eval(&dc_op);
            let vd = dev.dc_d.eval(&dc_op);
            let vg = dev.dc_g.eval(&dc_op);
            let (gm, gds, cgs, cgd, vs1, vd1) = ss_params(bd, vs, vd, vg)
                .ok_or_else(|| format!("candidate {index}: small-signal solve failed"))?;
            mos.push(lti::MosDevice {
                drain: dev.ac_d,
                gate: dev.ac_g,
                source: dev.ac_s,
                gm,
                gds,
                cgs,
                cgd,
            });
            internal.push((vs, vd, vg, vs1, vd1));
        }

        // 4. Assemble and solve the small-signal MNA (AC).
        let problem = lti::Problem {
            size: t.n_aug,
            dense_devices: Vec::new(),
            mos_devices: mos,
            capacitors: t
                .ac_caps
                .iter()
                .map(|(a, b, v)| lti::Branch {
                    a: *a,
                    b: *b,
                    value: *v,
                })
                .collect(),
            resistors: Vec::new(),
            vccs: Vec::new(),
            voltage_sources: Vec::new(),
            vcvs: Vec::new(),
            cccs: Vec::new(),
            ccvs: Vec::new(),
        };
        let system = problem
            .try_assemble()
            .map_err(|e| format!("candidate {index}: AC assembly failed: {e}"))?;

        let v = if inner_parallel {
            system.solve_frequencies_parallel(&t.freqs)
        } else {
            system.solve_frequencies_serial(&t.freqs)
        }
        .ok_or_else(|| format!("candidate {index}: AC solve singular"))?;

        // Response -> gains.
        let mut gains = Vec::with_capacity(t.freqs.len());
        for row in &v {
            let mut re = 0.0;
            let mut im = 0.0;
            for &(idx, w) in &t.output_weights {
                re += w * row[idx].re;
                im += w * row[idx].im;
            }
            let (re, im) = (re / t.vin_norm, im / t.vin_norm);
            gains.push(re.hypot(im));
        }
        let peak = gains.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let gain_peak_db = 20.0 * peak.max(1e-9).log10();
        let bw_hz = bw_from_gain(&t.freqs, &gains);

        // 5. Noise: transposed LTI solve reused from the same system. Skipped on
        // the AC-only prefilter path (irn_uV = NaN).
        let irn_uv = if self.compute_noise {
            let tvec = if inner_parallel {
                system.solve_transpose_parallel(&t.freqs, &t.sense)
            } else {
                system.solve_transpose_serial(&t.freqs, &t.sense)
            }
            .ok_or_else(|| format!("candidate {index}: noise transpose solve singular"))?;

            let nfreq = t.freqs.len();
            let mut out_psd = vec![0.0; nfreq];
            for ((dev, bd), (vs, vd, vg, vs1, vd1)) in
                t.devices.iter().zip(&built).zip(internal.iter().copied())
            {
                // Drain-current noise PSD split at 1 Hz, then 1/f scaled.
                let (s_th, s_fl1) = otft::noise_psd(
                    &bd.params,
                    vs,
                    vd,
                    vg,
                    vs1,
                    vd1,
                    bd.temperature,
                    bd.ci,
                    bd.w_m,
                    bd.l_m,
                    1.0,
                );
                for (fi, &f) in t.freqs.iter().enumerate() {
                    let s = s_th + s_fl1 / f;
                    // Transimpedance Z = tvec[:,d] - tvec[:,s] (node terminals only).
                    let mut zre = 0.0;
                    let mut zim = 0.0;
                    if let Some(di) = dev.noise_d {
                        zre += tvec[fi][di].re;
                        zim += tvec[fi][di].im;
                    }
                    if let Some(si) = dev.noise_s {
                        zre -= tvec[fi][si].re;
                        zim -= tvec[fi][si].im;
                    }
                    let zabs = zre.hypot(zim);
                    out_psd[fi] += zabs * zabs * s;
                }
            }
            let mut irn_psd = vec![0.0; nfreq];
            for fi in 0..nfreq {
                let h2 = (gains[fi] * gains[fi]).max(1e-300);
                irn_psd[fi] = out_psd[fi] / h2;
            }
            band_rms(&t.freqs, &irn_psd, t.band_lo, t.band_hi) * 1e6
        } else {
            f64::NAN
        };

        // 6. Latch metric.
        let latch_dv = match t.latch_nodes {
            Some((a, b)) => (dc_op[a] - dc_op[b]).abs(),
            None => 0.0,
        };

        Ok(OtftMetrics {
            gain_peak_db,
            bw_hz,
            irn_uv,
            latch_dv,
            dc_op,
            dc_iterations,
            dc_from_seed,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::campaign::{BatchConfig, BatchProgress, ParallelAxis, evaluate_batch};

    // A minimal single-node OTFT-ish template is not physically meaningful, so
    // these tests exercise device-build determinism and the batch plumbing on a
    // trivially-constructed template rather than full AFE numerics (which are
    // covered bit-for-bit by the Python parity tests).

    fn tiny_template() -> OtftTemplate {
        // One device between node 0 (drain) and rail, gate rail, source rail.
        OtftTemplate {
            n_aug: 1,
            n_nodes: 1,
            consts: OtftConstants::default(),
            devices: vec![DeviceTemplate {
                dc_d: BiasTerm::Solved(0),
                dc_g: BiasTerm::Rail(20.0),
                dc_s: BiasTerm::Rail(40.0),
                di: Some(0),
                si: None,
                ac_d: mna::Term {
                    kind: 0,
                    reference: 0,
                    value: 0.0,
                },
                ac_g: mna::Term {
                    kind: 2,
                    reference: 0,
                    value: 0.0,
                },
                ac_s: mna::Term {
                    kind: 2,
                    reference: 0,
                    value: 40.0,
                },
                noise_d: Some(0),
                noise_s: None,
            }],
            ac_caps: vec![(
                mna::Term {
                    kind: 0,
                    reference: 0,
                    value: 0.0,
                },
                mna::Term {
                    kind: 2,
                    reference: 0,
                    value: 0.0,
                },
                1e-12,
            )],
            output_weights: vec![(0, 1.0)],
            sense: vec![1.0],
            vin_norm: 1.0,
            freqs: vec![1.0, 10.0, 100.0, 1000.0],
            band_lo: 1.0,
            band_hi: 1000.0,
            gmin: 1e-12,
            dc_tol: 1e-10,
            dc_guesses: vec![vec![20.0], vec![30.0]],
            latch_nodes: None,
        }
    }

    fn geom() -> DeviceGeom {
        DeviceGeom {
            w: 1000.0,
            l: 20.0,
            nf: 1.0,
            pvt0: 0.0,
            mvt0: 0.0,
            pbeta0: 0.0,
            mbeta0: 0.0,
        }
    }

    #[test]
    fn device_build_is_finite_and_deterministic() {
        let c = OtftConstants::default();
        let a = build_device(&geom(), &c);
        let b = build_device(&geom(), &c);
        assert_eq!(a.params.vfb, b.params.vfb);
        assert_eq!(a.params.vss, b.params.vss);
        assert!(a.params.vfb.is_finite() && a.params.current_scale.is_finite());
        assert!(a.ci > 0.0 && a.w_m > 0.0 && a.l_m > 0.0);
    }

    #[test]
    fn trusted_seed_makes_batch_worker_invariant() {
        let template = tiny_template();
        // Trust-as-op with a fixed seed removes any DC solver dependence, so the
        // whole pipeline is a pure function of the candidate -> byte identical
        // across worker counts and axes.
        let candidates: Vec<OtftCandidate> = (0..12)
            .map(|k| OtftCandidate {
                devices: vec![DeviceGeom {
                    w: 1000.0 + k as f64,
                    ..geom()
                }],
                seed: Some(vec![30.0]),
                trust_seed_as_op: true,
            })
            .collect();
        let evaluator = OtftEvaluator {
            template: &template,
            candidates: &candidates,
            compute_noise: true,
        };
        let baseline = evaluate_batch(
            &evaluator,
            candidates.len(),
            BatchConfig {
                workers: 1,
                axis: ParallelAxis::Frequency,
            },
            &BatchProgress::new(),
        );
        let base_vals: Vec<(f64, f64, f64)> = baseline
            .iter()
            .map(|o| {
                let m = o.as_ref().unwrap().as_ref().unwrap();
                (m.gain_peak_db, m.bw_hz, m.irn_uv)
            })
            .collect();
        for workers in [1usize, 2, 8] {
            for axis in [
                ParallelAxis::Candidate,
                ParallelAxis::Frequency,
                ParallelAxis::Auto,
            ] {
                let got = evaluate_batch(
                    &evaluator,
                    candidates.len(),
                    BatchConfig { workers, axis },
                    &BatchProgress::new(),
                );
                let vals: Vec<(f64, f64, f64)> = got
                    .iter()
                    .map(|o| {
                        let m = o.as_ref().unwrap().as_ref().unwrap();
                        (m.gain_peak_db, m.bw_hz, m.irn_uv)
                    })
                    .collect();
                assert_eq!(vals, base_vals, "workers={workers} axis={axis:?}");
            }
        }
    }
}
