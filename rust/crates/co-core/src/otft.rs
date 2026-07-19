//! AT4000TG OTFT compact-model kernels.
//!
//! This is a direct, order-preserving port of the device-level `_impl`
//! functions in `circuitopt/numba_kernels.py`. Circuit-level stamping and
//! integration intentionally remain outside this module so device parity can
//! be established before the solver is moved.

/// Canonical 16-scalar parameter ABI shared with Python's `NumbaParams`.
///
/// `reference` is an out-of-band mode flag (not part of the 16-scalar ABI). When
/// `true`, the scalar threshold `Vt` is squared with `powf(2.0)` (libm `pow`,
/// matching the retired Python `_impl` reference oracle's `x ** 2`) instead of the
/// production `powi(2)` (`x * x`). This 1-ULP `Vt` difference is the root-selection
/// recovery lever the rust engine relies on for bifurcation-edge OTFT screens
/// (`pmos_tft_model.rust_otft_reference_mode`); see `docs/rust_core_rewrite_plan.md`
/// §4-D4. Production (`reference == false`) is bit-frozen against the golden corpus.
#[derive(Clone, Copy, Debug)]
pub struct Params {
    pub vfb: f64,
    pub vss: f64,
    pub lc: f64,
    pub lambda: f64,
    pub contact_scale: f64,
    pub exponent: f64,
    pub current_scale: f64,
    pub inv_rleak: f64,
    pub two_over_pi: f64,
    pub cap_cgs1: f64,
    pub cap_cgd1: f64,
    pub cap_half_wl_ci: f64,
    pub cap_cgs3_base: f64,
    pub cap_cgd3_base: f64,
    pub k1: f64,
    pub gate_leak_g: f64,
    pub reference: bool,
}

impl Params {
    pub const LEN: usize = 16;

    pub fn from_slice(values: &[f64]) -> Option<Self> {
        if values.len() != Self::LEN {
            return None;
        }
        Some(Self {
            vfb: values[0],
            vss: values[1],
            lc: values[2],
            lambda: values[3],
            contact_scale: values[4],
            exponent: values[5],
            current_scale: values[6],
            inv_rleak: values[7],
            two_over_pi: values[8],
            cap_cgs1: values[9],
            cap_cgd1: values[10],
            cap_half_wl_ci: values[11],
            cap_cgs3_base: values[12],
            cap_cgd3_base: values[13],
            k1: values[14],
            gate_leak_g: values[15],
            reference: false,
        })
    }
}

unsafe extern "C" {
    /// System libm `pow`. CPython's `float ** float` calls this exact symbol, so
    /// routing the reference square through it reproduces `x ** 2` bit-for-bit.
    /// `f64::powf(2.0)` cannot be used here: LLVM constant-folds a literal `2.0`
    /// exponent to `x * x`, which is identical to the production `powi(2)` and
    /// would erase the 1-ULP reference divergence.
    safe fn pow(base: f64, exp: f64) -> f64;
}

/// Square the contact-threshold argument. Production uses `powi(2)` (`x * x`); the
/// reference recovery path uses the system libm `pow(x, 2.0)` to reproduce the
/// Python `_impl` oracle's `x ** 2` bit-for-bit. The exponent is hidden behind
/// `black_box` so LLVM's libcall simplifier cannot prove it is `2.0` and fold the
/// call back into `x * x` (which would erase the reference divergence).
#[inline]
fn dv_squared(p: &Params, dv: f64) -> f64 {
    if p.reference {
        pow(dv, std::hint::black_box(2.0))
    } else {
        dv.powi(2)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct NewtonResult {
    pub converged: bool,
    pub vs1: f64,
    pub vd1: f64,
    pub iterations: usize,
    pub fd_fallbacks: usize,
}

#[inline]
fn softplus(x: f64) -> f64 {
    if x > 0.0 {
        x + (-x).exp().ln_1p()
    } else {
        x.exp().ln_1p()
    }
}

#[inline]
fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        let z = (-x).exp();
        1.0 / (1.0 + z)
    } else {
        let z = x.exp();
        z / (1.0 + z)
    }
}

/// Return `(I_s_s1, I_s1_d1, I_d1_d, Ich, I_d1_d_leak)`.
pub fn eval_currents(p: &Params, vs: f64, vd: f64, vg: f64, vs1: f64, vd1: f64) -> [f64; 5] {
    let v_s = if vs > vs1 { vs } else { vs1 };
    let v_s1 = if vs > vs1 { vs1 } else { vs };
    let v_d = if vd1 > vd { vd } else { vd1 };
    let v_d1 = if vd1 > vd { vd1 } else { vd };

    let vt = -(0.0045 * dv_squared(p, v_s - vg) + 0.7125 * (v_s - vg) + 0.9625);
    let vods1 = p.vss * softplus((v_s - vg + vt) / p.vss);
    let vodd1 = p.vss * softplus((v_s1 - vg + vt) / p.vss);

    let ecsat = 17.0 / ((v_s - vg).abs() + 0.1);
    let lambdac = 1.0 / (p.lc * ecsat);
    let cmod = 1.0 + lambdac * (v_s - v_s1);
    let icont = p.contact_scale * (vods1.powf(p.exponent) - vodd1.powf(p.exponent)) * cmod;
    let i_s_s1 = if vs > vs1 { icont } else { -icont };

    let arg_d1 = (v_d1 - vg + p.vfb) / p.vss;
    let arg_d = (v_d - vg + p.vfb) / p.vss;
    let vods = p.vss * softplus(arg_d1);
    let vodd = p.vss * softplus(arg_d);
    let chmod = 1.0 + p.lambda * (v_d1 - v_d);
    let ich = p.current_scale * (vods.powf(p.exponent) - vodd.powf(p.exponent)) * chmod;

    let i_d1_d_ch = if vs1 > vd { ich } else { -ich };
    let i_d1_d_leak = (vd1 - vd + 0.1) * p.inv_rleak;
    let i_d1_d = i_d1_d_ch + i_d1_d_leak;
    let i_s1_d1 = (vs1 - vd1) / 0.1;
    [i_s_s1, i_s1_d1, i_d1_d, ich, i_d1_d_leak]
}

#[inline]
pub fn residual_pair(p: &Params, vs: f64, vd: f64, vg: f64, vs1: f64, vd1: f64) -> [f64; 2] {
    let i = eval_currents(p, vs, vd, vg, vs1, vd1);
    [i[0] - i[1], i[1] - i[2]]
}

fn residual_pair_fd_jac(
    p: &Params,
    vs: f64,
    vd: f64,
    vg: f64,
    vs1: f64,
    vd1: f64,
    hj: f64,
) -> [f64; 6] {
    let r0 = residual_pair(p, vs, vd, vg, vs1, vd1);
    let r1 = residual_pair(p, vs, vd, vg, vs1 + hj, vd1);
    let r2 = residual_pair(p, vs, vd, vg, vs1, vd1 + hj);
    [
        r0[0],
        r0[1],
        (r1[0] - r0[0]) / hj,
        (r2[0] - r0[0]) / hj,
        (r1[1] - r0[1]) / hj,
        (r2[1] - r0[1]) / hj,
    ]
}

pub fn residual_pair_jac_internal(
    p: &Params,
    vs: f64,
    vd: f64,
    vg: f64,
    vs1: f64,
    vd1: f64,
) -> [f64; 6] {
    if (vs - vs1).abs() < 1e-10 || (vd1 - vd).abs() < 1e-10 || (vs1 - vd).abs() < 1e-10 {
        return residual_pair_fd_jac(p, vs, vd, vg, vs1, vd1, 1e-6);
    }

    let (i_s_s1, diss_dvs1) = if vs > vs1 {
        let v_s = vs;
        let v_s1 = vs1;
        let dv = v_s - vg;
        let vt = -(0.0045 * dv_squared(p, dv) + 0.7125 * dv + 0.9625);
        let arg_a = (v_s - vg + vt) / p.vss;
        let arg_b = (v_s1 - vg + vt) / p.vss;
        let a = p.vss * softplus(arg_a);
        let b = p.vss * softplus(arg_b);
        let ap = a.powf(p.exponent);
        let bp = b.powf(p.exponent);
        let bem1 = if b != 0.0 { bp / b } else { 0.0 };
        let ecsat = 17.0 / (dv.abs() + 0.1);
        let lambdac = 1.0 / (p.lc * ecsat);
        let cmod = 1.0 + lambdac * (v_s - v_s1);
        let icont = p.contact_scale * (ap - bp) * cmod;
        let db = sigmoid(arg_b);
        let dicont = p.contact_scale * (-p.exponent * bem1 * db * cmod - (ap - bp) * lambdac);
        (icont, dicont)
    } else {
        let v_s = vs1;
        let v_s1 = vs;
        let dv = v_s - vg;
        let vt = -(0.0045 * dv_squared(p, dv) + 0.7125 * dv + 0.9625);
        let dvt = -(0.009 * dv + 0.7125);
        let arg_a = (v_s - vg + vt) / p.vss;
        let arg_b = (v_s1 - vg + vt) / p.vss;
        let a = p.vss * softplus(arg_a);
        let b = p.vss * softplus(arg_b);
        let ap = a.powf(p.exponent);
        let bp = b.powf(p.exponent);
        let aem1 = if a != 0.0 { ap / a } else { 0.0 };
        let bem1 = if b != 0.0 { bp / b } else { 0.0 };
        let ecsat = 17.0 / (dv.abs() + 0.1);
        let lambdac = 1.0 / (p.lc * ecsat);
        let cmod = 1.0 + lambdac * (v_s - v_s1);
        let icont = p.contact_scale * (ap - bp) * cmod;
        let sign_dv = if dv > 0.0 { 1.0 } else { -1.0 };
        let dlambdac = sign_dv / (17.0 * p.lc);
        let da = sigmoid(arg_a) * (1.0 + dvt);
        let db = sigmoid(arg_b) * dvt;
        let dcmod = dlambdac * (v_s - v_s1) + lambdac;
        let dicont = p.contact_scale
            * ((p.exponent * aem1 * da - p.exponent * bem1 * db) * cmod + (ap - bp) * dcmod);
        (-icont, -dicont)
    };

    let (ich, dich) = if vd1 > vd {
        let v_d = vd;
        let v_d1 = vd1;
        let arg_a = (v_d1 - vg + p.vfb) / p.vss;
        let arg_b = (v_d - vg + p.vfb) / p.vss;
        let a = p.vss * softplus(arg_a);
        let b = p.vss * softplus(arg_b);
        let ap = a.powf(p.exponent);
        let bp = b.powf(p.exponent);
        let aem1 = if a != 0.0 { ap / a } else { 0.0 };
        let chmod = 1.0 + p.lambda * (v_d1 - v_d);
        let ich = p.current_scale * (ap - bp) * chmod;
        let da = sigmoid(arg_a);
        let dich = p.current_scale * (p.exponent * aem1 * da * chmod + (ap - bp) * p.lambda);
        (ich, dich)
    } else {
        let v_d = vd1;
        let v_d1 = vd;
        let arg_a = (v_d1 - vg + p.vfb) / p.vss;
        let arg_b = (v_d - vg + p.vfb) / p.vss;
        let a = p.vss * softplus(arg_a);
        let b = p.vss * softplus(arg_b);
        let ap = a.powf(p.exponent);
        let bp = b.powf(p.exponent);
        let bem1 = if b != 0.0 { bp / b } else { 0.0 };
        let chmod = 1.0 + p.lambda * (v_d1 - v_d);
        let ich = p.current_scale * (ap - bp) * chmod;
        let db = sigmoid(arg_b);
        let dich = p.current_scale * (-p.exponent * bem1 * db * chmod - (ap - bp) * p.lambda);
        (ich, dich)
    };

    let sign_ch = if vs1 > vd { 1.0 } else { -1.0 };
    let i_d1_d = sign_ch * ich + (vd1 - vd + 0.1) * p.inv_rleak;
    let did_dvd1 = sign_ch * dich + p.inv_rleak;
    let i_s1_d1 = (vs1 - vd1) / 0.1;
    [
        i_s_s1 - i_s1_d1,
        i_s1_d1 - i_d1_d,
        diss_dvs1 - 10.0,
        10.0,
        10.0,
        -10.0 - did_dvd1,
    ]
}

#[allow(clippy::too_many_arguments)]
pub fn newton_internal(
    p: &Params,
    vs: f64,
    vd: f64,
    vg: f64,
    x0s: f64,
    x0d: f64,
    tol: f64,
    maxit: usize,
) -> NewtonResult {
    let mut vs1 = x0s;
    let mut vd1 = x0d;
    for it in 0..maxit {
        let r = residual_pair_fd_jac(p, vs, vd, vg, vs1, vd1, 1e-6);
        if r[0].abs() + r[1].abs() < tol {
            return NewtonResult {
                converged: true,
                vs1,
                vd1,
                iterations: it + 1,
                fd_fallbacks: it + 1,
            };
        }
        let det = r[2] * r[5] - r[3] * r[4];
        if det == 0.0 || !det.is_finite() {
            return NewtonResult {
                converged: false,
                vs1,
                vd1,
                iterations: it + 1,
                fd_fallbacks: it + 1,
            };
        }
        let mut d0 = -(r[5] * r[0] - r[3] * r[1]) / det;
        let mut d1 = -(-r[4] * r[0] + r[2] * r[1]) / det;
        let mut mx = d0.abs().max(d1.abs());
        if mx > 2.0 {
            let scale = 2.0 / mx;
            d0 *= scale;
            d1 *= scale;
            mx = 2.0;
        }
        vs1 += d0;
        vd1 += d1;
        if mx < 1e-13 {
            let converged = r[0].abs() + r[1].abs() < 1e-9;
            return NewtonResult {
                converged,
                vs1,
                vd1,
                iterations: it + 1,
                fd_fallbacks: it + 1,
            };
        }
    }
    NewtonResult {
        converged: false,
        vs1,
        vd1,
        iterations: maxit,
        fd_fallbacks: maxit,
    }
}

#[allow(clippy::too_many_arguments)]
pub fn newton_internal_fast(
    p: &Params,
    vs: f64,
    vd: f64,
    vg: f64,
    x0s: f64,
    x0d: f64,
    tol: f64,
    maxit: usize,
) -> NewtonResult {
    let mut vs1 = x0s;
    let mut vd1 = x0d;
    let mut fd_fallbacks = 0;
    for it in 0..maxit {
        if (vs - vs1).abs() < 1e-10 || (vd1 - vd).abs() < 1e-10 || (vs1 - vd).abs() < 1e-10 {
            fd_fallbacks += 1;
        }
        let r = residual_pair_jac_internal(p, vs, vd, vg, vs1, vd1);
        if r[0].abs() + r[1].abs() < tol {
            return NewtonResult {
                converged: true,
                vs1,
                vd1,
                iterations: it + 1,
                fd_fallbacks,
            };
        }
        let det = r[2] * r[5] - r[3] * r[4];
        if det == 0.0 || !det.is_finite() {
            return NewtonResult {
                converged: false,
                vs1,
                vd1,
                iterations: it + 1,
                fd_fallbacks,
            };
        }
        let mut d0 = -(r[5] * r[0] - r[3] * r[1]) / det;
        let mut d1 = -(-r[4] * r[0] + r[2] * r[1]) / det;
        let mut mx = d0.abs().max(d1.abs());
        if mx > 2.0 {
            let scale = 2.0 / mx;
            d0 *= scale;
            d1 *= scale;
            mx = 2.0;
        }
        vs1 += d0;
        vd1 += d1;
        if mx < 1e-13 {
            let converged = r[0].abs() + r[1].abs() < 1e-9;
            return NewtonResult {
                converged,
                vs1,
                vd1,
                iterations: it + 1,
                fd_fallbacks,
            };
        }
    }
    NewtonResult {
        converged: false,
        vs1,
        vd1,
        iterations: maxit,
        fd_fallbacks,
    }
}

pub fn capacitances(p: &Params, vs: f64, vd: f64, vg: f64, vs1: f64, vd1: f64) -> [f64; 2] {
    let v_s = if vs > vs1 { vs } else { vs1 };
    let v_d = if vd1 > vd { vd } else { vd1 };
    let arg_gs = v_s - vg + p.vfb;
    let cgs2 = 1.43 * p.cap_half_wl_ci * (p.two_over_pi * (arg_gs * 0.6).atan() + 1.0);
    let cgd2 = 0.33 * p.cap_half_wl_ci * (p.two_over_pi * (arg_gs * 2.01).atan() + 1.0);
    let arg_gd = -vg + p.vfb + v_d;
    let cgs3 = 0.34 * p.cap_cgs3_base * (p.two_over_pi * (arg_gd * 0.21).atan() + 1.0);
    let cgd3 = 0.52 * p.cap_cgd3_base * (p.two_over_pi * (arg_gd * 0.42).atan() + 1.0);
    [
        p.k1 * (p.cap_cgs1 + cgs2 + cgs3) * 1e4 * 1e-12,
        p.k1 * (p.cap_cgd1 + cgd2 + cgd3) * 1e4 * 1e-12,
    ]
}

/// Channel transconductance from `_eval_channel['gm']` (analytic), used only by
/// the drain-current noise PSD. Order-preserving port of
/// `PMOS_TFT._eval_channel` (`circuitopt/pmos_tft_model.py`).
pub fn channel_gm(p: &Params, vs: f64, vd: f64, vg: f64, vs1: f64, vd1: f64) -> f64 {
    let _ = (vs, vs1);
    let v_d = if vd1 > vd { vd } else { vd1 };
    let v_d1 = if vd1 > vd { vd1 } else { vd };
    let arg_d1 = (v_d1 - vg + p.vfb) / p.vss;
    let arg_d = (v_d - vg + p.vfb) / p.vss;
    let vods = p.vss * softplus(arg_d1);
    let vodd = p.vss * softplus(arg_d);
    let chmod = 1.0 + p.lambda * (v_d1 - v_d);
    p.current_scale
        * p.exponent
        * (vods.powf(p.exponent - 1.0) * sigmoid(arg_d1)
            - vodd.powf(p.exponent - 1.0) * sigmoid(arg_d))
        * chmod
}

/// Drain-current noise PSD split `(S_thermal, S_flicker_at(frequency))` in
/// A^2/Hz. Order-preserving port of `PMOS_TFT.get_noise_psd`. The physical
/// constants match the device model's own truncated values (`q = 1.6e-19`,
/// `Kb = 1.38064e-23`), which differ from the resistor-noise Boltzmann constant.
/// `ci`, `w_m = W*1e-6`, `l_m = L*1e-6` and `temperature` come from the same
/// geometry/corner build that produced `p`. Call with `frequency = 1.0` to get
/// the 1 Hz flicker coefficient (`device_psd` then scales it by `1/f`).
#[allow(clippy::too_many_arguments)]
pub fn noise_psd(
    p: &Params,
    vs: f64,
    vd: f64,
    vg: f64,
    vs1: f64,
    vd1: f64,
    temperature: f64,
    ci: f64,
    w_m: f64,
    l_m: f64,
    frequency: f64,
) -> (f64, f64) {
    const Q: f64 = 1.6e-19;
    const KB: f64 = 1.38064e-23;
    let currents = eval_currents(p, vs, vd, vg, vs1, vd1);
    let ich = currents[3];
    let ioff = currents[4];
    let gm = channel_gm(p, vs, vd, vg, vs1, vd1);
    let s_th1 = 2.0 * Q * (ich + ioff);
    let s_th2 = 4.0 * KB * temperature * gm * 2.0 / 3.0;
    let s_thermal = s_th1 + s_th2;
    let hooge = 0.05;
    // `_va_sorted_nodes` v_d1 = max(Vd1, Vd).
    let v_d1 = if vd1 > vd { vd1 } else { vd };
    let denom = w_m * l_m * ci * 1e4 * (v_d1 - vg + p.vfb);
    let s_flicker_1hz = (hooge * Q * ich * ich) / denom.abs();
    (s_thermal, s_flicker_1hz / frequency)
}

#[inline]
fn atan_cap_integral(y: f64, scale: f64, two_over_pi: f64) -> f64 {
    let ay = scale * y;
    y + two_over_pi * (y * ay.atan() - 0.5 * (ay * ay).ln_1p() / scale)
}

/// Return `(qgs, qgd, Cgs, Cgd)`.
pub fn capacitance_charges(p: &Params, vs: f64, vd: f64, vg: f64, vs1: f64, vd1: f64) -> [f64; 4] {
    let v_s = if vs > vs1 { vs } else { vs1 };
    let v_d = if vd1 > vd { vd } else { vd1 };
    let y_s = v_s - vg + p.vfb;
    let y_d = v_d - vg + p.vfb;
    let x_gs = vg - vs;
    let x_gd = vg - vd;
    let cgs2_coeff = 1.43 * p.cap_half_wl_ci;
    let cgd2_coeff = 0.33 * p.cap_half_wl_ci;
    let cgs3_coeff = 0.34 * p.cap_cgs3_base;
    let cgd3_coeff = 0.52 * p.cap_cgd3_base;
    let f_s_060 = p.two_over_pi * (y_s * 0.6).atan() + 1.0;
    let f_s_201 = p.two_over_pi * (y_s * 2.01).atan() + 1.0;
    let f_d_021 = p.two_over_pi * (y_d * 0.21).atan() + 1.0;
    let f_d_042 = p.two_over_pi * (y_d * 0.42).atan() + 1.0;
    let cgs_cross = cgs3_coeff * f_d_021;
    let cgd_cross = cgd2_coeff * f_s_201;
    let qscale = p.k1 * 1e4 * 1e-12;
    let qgs = qscale
        * (p.cap_cgs1 * x_gs - cgs2_coeff * atan_cap_integral(y_s, 0.6, p.two_over_pi)
            + cgs_cross * x_gs);
    let qgd = qscale
        * (p.cap_cgd1 * x_gd + cgd_cross * x_gd
            - cgd3_coeff * atan_cap_integral(y_d, 0.42, p.two_over_pi));
    let cgss = qscale * (p.cap_cgs1 + cgs2_coeff * f_s_060 + cgs_cross);
    let cgdd = qscale * (p.cap_cgd1 + cgd_cross + cgd3_coeff * f_d_042);
    [qgs, qgd, cgss, cgdd]
}

#[inline]
fn eval_at(p: &Params, vs: f64, vd: f64, vg: f64, vs1: f64, vd1: f64) -> [f64; 3] {
    let i = eval_currents(p, vs, vd, vg, vs1, vd1);
    [i[0] - i[1], i[1] - i[2], -i[2]]
}

#[allow(clippy::too_many_arguments)]
fn terminal_deriv_one(
    p: &Params,
    vp: [f64; 3],
    vm: [f64; 3],
    vs1: f64,
    vd1: f64,
    j: [f64; 4],
    ix: [f64; 2],
    det: f64,
    sign: f64,
    hh: f64,
) -> f64 {
    let ep = eval_at(p, vp[0], vp[1], vp[2], vs1, vd1);
    let em = eval_at(p, vm[0], vm[1], vm[2], vs1, vd1);
    let fu0 = (ep[0] - em[0]) / (2.0 * hh);
    let fu1 = (ep[1] - em[1]) / (2.0 * hh);
    let iu = (ep[2] - em[2]) / (2.0 * hh);
    let y0 = (j[3] * fu0 - j[1] * fu1) / det;
    let y1 = (-j[2] * fu0 + j[0] * fu1) / det;
    sign * (iu - ix[0] * y0 - ix[1] * y1)
}

#[allow(clippy::too_many_arguments)]
fn terminal_derivatives_from_base(
    p: &Params,
    vs: f64,
    vd: f64,
    vg: f64,
    vs1: f64,
    vd1: f64,
    f0a: f64,
    f0b: f64,
    idc0: f64,
    need_gm: bool,
    need_gds: bool,
    use_abs: bool,
    hh: f64,
    hx: f64,
) -> (bool, f64, f64) {
    if !need_gm && !need_gds {
        return (true, 0.0, 0.0);
    }
    if use_abs && idc0.abs() < 1e-30 {
        return (false, 0.0, 0.0);
    }
    let ep0 = eval_at(p, vs, vd, vg, vs1 + hx, vd1);
    let j00 = (ep0[0] - f0a) / hx;
    let j10 = (ep0[1] - f0b) / hx;
    let ix0 = (ep0[2] - idc0) / hx;
    let ep1 = eval_at(p, vs, vd, vg, vs1, vd1 + hx);
    let j01 = (ep1[0] - f0a) / hx;
    let j11 = (ep1[1] - f0b) / hx;
    let ix1 = (ep1[2] - idc0) / hx;
    let det = j00 * j11 - j01 * j10;
    if det == 0.0 || !det.is_finite() {
        return (false, 0.0, 0.0);
    }
    let sign = if idc0 > 0.0 { 1.0 } else { -1.0 };
    let current_sign = if use_abs { sign } else { -1.0 };
    let j = [j00, j01, j10, j11];
    let ix = [ix0, ix1];
    let gm = if need_gm {
        terminal_deriv_one(
            p,
            [vs, vd, vg + hh],
            [vs, vd, vg - hh],
            vs1,
            vd1,
            j,
            ix,
            det,
            current_sign,
            hh,
        )
    } else {
        0.0
    };
    let gds = if need_gds {
        terminal_deriv_one(
            p,
            [vs, vd + hh, vg],
            [vs, vd - hh, vg],
            vs1,
            vd1,
            j,
            ix,
            det,
            current_sign,
            hh,
        )
    } else {
        0.0
    };
    (true, gm, gds)
}

#[allow(clippy::too_many_arguments)]
fn terminal_derivatives_from_jac_fdterm(
    p: &Params,
    vs: f64,
    vd: f64,
    vg: f64,
    vs1: f64,
    vd1: f64,
    idc0: f64,
    j: [f64; 4],
    need_gm: bool,
    need_gds: bool,
    use_abs: bool,
    hh: f64,
) -> (bool, f64, f64) {
    if !need_gm && !need_gds {
        return (true, 0.0, 0.0);
    }
    if use_abs && idc0.abs() < 1e-30 {
        return (false, 0.0, 0.0);
    }
    let det = j[0] * j[3] - j[1] * j[2];
    if det == 0.0 || !det.is_finite() {
        return (false, 0.0, 0.0);
    }
    let ix = [j[2] - 10.0, j[3] + 10.0];
    let sign = if idc0 > 0.0 { 1.0 } else { -1.0 };
    let current_sign = if use_abs { sign } else { -1.0 };
    let gm = if need_gm {
        terminal_deriv_one(
            p,
            [vs, vd, vg + hh],
            [vs, vd, vg - hh],
            vs1,
            vd1,
            j,
            ix,
            det,
            current_sign,
            hh,
        )
    } else {
        0.0
    };
    let gds = if need_gds {
        terminal_deriv_one(
            p,
            [vs, vd + hh, vg],
            [vs, vd - hh, vg],
            vs1,
            vd1,
            j,
            ix,
            det,
            current_sign,
            hh,
        )
    } else {
        0.0
    };
    (true, gm, gds)
}

fn contact_diss_dvg(p: &Params, vs: f64, vg: f64, vs1: f64) -> f64 {
    let (v_s, v_s1, outer_sign) = if vs > vs1 {
        (vs, vs1, 1.0)
    } else {
        (vs1, vs, -1.0)
    };
    let dv = v_s - vg;
    let vt = -(0.0045 * dv_squared(p, dv) + 0.7125 * dv + 0.9625);
    let dvt = 0.009 * dv + 0.7125;
    let arg_a = (v_s - vg + vt) / p.vss;
    let arg_b = (v_s1 - vg + vt) / p.vss;
    let a = p.vss * softplus(arg_a);
    let b = p.vss * softplus(arg_b);
    let ap = a.powf(p.exponent);
    let bp = b.powf(p.exponent);
    let aem1 = if a != 0.0 { ap / a } else { 0.0 };
    let bem1 = if b != 0.0 { bp / b } else { 0.0 };
    let ecsat = 17.0 / (dv.abs() + 0.1);
    let lambdac = 1.0 / (p.lc * ecsat);
    let cmod = 1.0 + lambdac * (v_s - v_s1);
    let sign_dv = if dv > 0.0 { 1.0 } else { -1.0 };
    let dlambdac = -sign_dv / (17.0 * p.lc);
    let darg_num = -1.0 + dvt;
    let da = sigmoid(arg_a) * darg_num;
    let db = sigmoid(arg_b) * darg_num;
    let dcmod = dlambdac * (v_s - v_s1);
    outer_sign
        * p.contact_scale
        * ((p.exponent * aem1 * da - p.exponent * bem1 * db) * cmod + (ap - bp) * dcmod)
}

fn channel_partials(p: &Params, vs1: f64, vd: f64, vg: f64, vd1: f64) -> [f64; 2] {
    let (did_vg, did_vd) = if vd1 > vd {
        let v_d = vd;
        let v_d1 = vd1;
        let arg_a = (v_d1 - vg + p.vfb) / p.vss;
        let arg_b = (v_d - vg + p.vfb) / p.vss;
        let a = p.vss * softplus(arg_a);
        let b = p.vss * softplus(arg_b);
        let ap = a.powf(p.exponent);
        let bp = b.powf(p.exponent);
        let aem1 = if a != 0.0 { ap / a } else { 0.0 };
        let bem1 = if b != 0.0 { bp / b } else { 0.0 };
        let chmod = 1.0 + p.lambda * (v_d1 - v_d);
        let dvg =
            p.current_scale * p.exponent * (-aem1 * sigmoid(arg_a) + bem1 * sigmoid(arg_b)) * chmod;
        let dvd =
            p.current_scale * (-p.exponent * bem1 * sigmoid(arg_b) * chmod - (ap - bp) * p.lambda);
        (dvg, dvd)
    } else {
        let v_d = vd1;
        let v_d1 = vd;
        let arg_a = (v_d1 - vg + p.vfb) / p.vss;
        let arg_b = (v_d - vg + p.vfb) / p.vss;
        let a = p.vss * softplus(arg_a);
        let b = p.vss * softplus(arg_b);
        let ap = a.powf(p.exponent);
        let bp = b.powf(p.exponent);
        let aem1 = if a != 0.0 { ap / a } else { 0.0 };
        let bem1 = if b != 0.0 { bp / b } else { 0.0 };
        let chmod = 1.0 + p.lambda * (v_d1 - v_d);
        let dvg =
            p.current_scale * p.exponent * (-aem1 * sigmoid(arg_a) + bem1 * sigmoid(arg_b)) * chmod;
        let dvd =
            p.current_scale * (p.exponent * aem1 * sigmoid(arg_a) * chmod + (ap - bp) * p.lambda);
        (dvg, dvd)
    };
    let sign_ch = if vs1 > vd { 1.0 } else { -1.0 };
    [sign_ch * did_vg, sign_ch * did_vd - p.inv_rleak]
}

#[inline]
fn terminal_deriv_from_partials(
    fu: [f64; 2],
    iu: f64,
    j: [f64; 4],
    ix: [f64; 2],
    det: f64,
    current_sign: f64,
) -> f64 {
    let y0 = (j[3] * fu[0] - j[1] * fu[1]) / det;
    let y1 = (-j[2] * fu[0] + j[0] * fu[1]) / det;
    current_sign * (iu - ix[0] * y0 - ix[1] * y1)
}

#[allow(clippy::too_many_arguments)]
pub fn terminal_derivatives_from_jac(
    p: &Params,
    vs: f64,
    vd: f64,
    vg: f64,
    vs1: f64,
    vd1: f64,
    f0a: f64,
    f0b: f64,
    idc0: f64,
    j: [f64; 4],
    need_gm: bool,
    need_gds: bool,
    use_abs: bool,
    hh: f64,
) -> (bool, f64, f64) {
    if (vs - vs1).abs() < 1e-10 || (vd1 - vd).abs() < 1e-10 || (vs1 - vd).abs() < 1e-10 {
        let _ = (f0a, f0b);
        return terminal_derivatives_from_jac_fdterm(
            p, vs, vd, vg, vs1, vd1, idc0, j, need_gm, need_gds, use_abs, hh,
        );
    }
    if !need_gm && !need_gds {
        return (true, 0.0, 0.0);
    }
    if use_abs && idc0.abs() < 1e-30 {
        return (false, 0.0, 0.0);
    }
    let det = j[0] * j[3] - j[1] * j[2];
    if det == 0.0 || !det.is_finite() {
        return (false, 0.0, 0.0);
    }
    let ix = [j[2] - 10.0, j[3] + 10.0];
    let sign = if idc0 > 0.0 { 1.0 } else { -1.0 };
    let current_sign = if use_abs { sign } else { -1.0 };
    let partials = channel_partials(p, vs1, vd, vg, vd1);
    let gm = if need_gm {
        let diss_dvg = contact_diss_dvg(p, vs, vg, vs1);
        terminal_deriv_from_partials(
            [diss_dvg, -partials[0]],
            -partials[0],
            j,
            ix,
            det,
            current_sign,
        )
    } else {
        0.0
    };
    let gds = if need_gds {
        terminal_deriv_from_partials([0.0, -partials[1]], -partials[1], j, ix, det, current_sign)
    } else {
        0.0
    };
    (true, gm, gds)
}

#[allow(clippy::too_many_arguments)]
pub fn terminal_derivatives(
    p: &Params,
    vs: f64,
    vd: f64,
    vg: f64,
    vs1: f64,
    vd1: f64,
    need_gm: bool,
    need_gds: bool,
    use_abs: bool,
    hh: f64,
    hx: f64,
) -> (bool, f64, f64) {
    let e = eval_at(p, vs, vd, vg, vs1, vd1);
    terminal_derivatives_from_base(
        p, vs, vd, vg, vs1, vd1, e[0], e[1], e[2], need_gm, need_gds, use_abs, hh, hx,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> Params {
        Params {
            vfb: -3.0,
            vss: 0.5,
            lc: 5.0,
            lambda: 0.01,
            contact_scale: 2.2e-6,
            exponent: 4.0,
            current_scale: 1e-8,
            inv_rleak: 1e-12,
            two_over_pi: 2.0 / std::f64::consts::PI,
            cap_cgs1: 1e-8,
            cap_cgd1: 2e-8,
            cap_half_wl_ci: 3e-8,
            cap_cgs3_base: 4e-8,
            cap_cgd3_base: 5e-8,
            k1: 1.0,
            gate_leak_g: 1e-12,
            reference: false,
        }
    }

    #[test]
    fn reference_flips_vt_square_but_not_channel() {
        // Production squares Vt with powi(2)=x*x; reference with powf(2.0)=pow.
        // eval_currents differs only where Vt matters (contact branch I_s_s1);
        // the channel current Ich (no Vt) stays bit-identical between the modes.
        let mut pp = params();
        pp.reference = false;
        let mut pr = params();
        pr.reference = true;
        let (vs, vd, vg, vs1, vd1) = (5.783, 4.348, 43.975, 6.766, 4.117);
        let prod = eval_currents(&pp, vs, vd, vg, vs1, vd1);
        let refr = eval_currents(&pr, vs, vd, vg, vs1, vd1);
        assert_eq!(
            prod[3], refr[3],
            "Ich must not depend on the Vt square mode"
        );
        // Caps carry no Vt term -> identical in both modes.
        assert_eq!(
            capacitance_charges(&pp, vs, vd, vg, vs1, vd1),
            capacitance_charges(&pr, vs, vd, vg, vs1, vd1)
        );
    }

    #[test]
    fn parameter_schema_is_exact() {
        let raw: Vec<f64> = (0..Params::LEN).map(|x| x as f64).collect();
        let p = Params::from_slice(&raw).unwrap();
        assert_eq!(p.vfb, 0.0);
        assert_eq!(p.gate_leak_g, 15.0);
        assert!(Params::from_slice(&raw[..15]).is_none());
    }

    #[test]
    fn device_outputs_are_finite() {
        let p = params();
        let currents = eval_currents(&p, 40.0, 0.0, 20.0, 39.6, 0.4);
        let caps = capacitance_charges(&p, 40.0, 0.0, 20.0, 39.6, 0.4);
        assert!(currents.iter().all(|x| x.is_finite()));
        assert!(caps.iter().all(|x| x.is_finite()));
    }
}
