//! Silicon (BSIM4) campaign evaluator — rewrite step R5-C.
//!
//! Composes `co_pdk::CompiledPdk` numeric cards, `co_bsim4` device handles,
//! `co_core::bsim_transient::solve_dc`, and the complex `lti` MNA into the
//! per-candidate device-build -> DC -> AC -> noise pipeline for the
//! freepdk45 / sky130 / tsmc28 families, driven by `co_core::campaign`.
//!
//! Every step mirrors the frozen Python scalar path exactly:
//!
//! * card -> BSIM ABI: `Bsim4ModelCard.__post_init__` semantics (lower-cased
//!   keys, `level`/`version` dropped) and, for TSMC28 only, the
//!   `to_bsim4_cards()` `mulu0 -> u0` mobility fold (pop `mulu0`; if != 1.0
//!   multiply the model's `u0`; error when `u0` is absent).
//! * handle build: `co_bsim4::create(polarity, temp_K)` + `set_model*` +
//!   `set_instance*` + `setup` (the `_NativeDevice.__init__` sequence).
//! * DC: `bsim_transient::solve_dc` — the same kernel the Python rust-engine
//!   path calls through `Bsim4TransientProblem.solve_dc`.
//! * small-signal: one `eval_vp` per device at the operating point; the raw
//!   4x4 conductance/capacitance are the `get_terminal_linearization` output
//!   and the eval leaves the handle biased at the op, which `co_bsim4::noise`
//!   requires (the evaluate-then-noise call order of `noise_solver`).
//! * noise: per-device 4x4 total spectral density at each frequency,
//!   `max(Re(z S z*), 0)` with the transposed-solve transimpedance vector,
//!   plus resistor thermal noise `4 k T / R` at `T = 300.15 K`.

use std::collections::HashMap;
use std::ffi::CString;
use std::sync::Arc;

use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use co_bsim4::CoBsim4;
use co_core::campaign::{CandidateEvaluator, CandidateOutcome, band_rms, bw_from_gain};
use co_core::{bsim_transient, lti, mna, transient};
use co_pdk::CompiledPdk as PdkCompiled;

use crate::{
    LtiBranchRecord, LtiCccsRecord, LtiCcvsRecord, LtiVccsRecord, LtiVcvsRecord, LtiVoltageRecord,
    TermRecord, optional_field, optional_index, required, term,
};

/// Boltzmann constant + resistor noise temperature — `noise_solver._KB/_TEMP`.
const KB: f64 = 1.380649e-23;
const RES_TEMP: f64 = 300.15;

/// Candidate-invariant static description of one device slot.
#[derive(Clone, Debug)]
pub struct DeviceStatic {
    /// "nmos" / "pmos" — the `CompiledPdk` polarity request.
    pub polarity: String,
    /// +1 NMOS / -1 PMOS — `Bsim4ModelCard.polarity`.
    pub polarity_sign: i32,
    /// Bulk rail voltage (`wrapper.vb`).
    pub vb: f64,
    /// Handle temperature in kelvin (`wrapper.temperature`).
    pub temperature_k: f64,
    /// Card-selection temperature in Celsius (TSMC28; ignored elsewhere).
    pub temp_c: f64,
    /// SKY130 `extract_w` card-bin width in µm (`None` = bin on the actual width).
    /// Candidate-invariant: it is the device's `extract_w` kwarg / class default,
    /// not a swept geometry. `None` for FreePDK45/TSMC28 (they bin on corner).
    pub reference_width_um: Option<f64>,
    /// AC terminals (d, g, s) — bulk is a fixed `("v", 0.0)` fourth terminal.
    pub ac_d: mna::Term,
    pub ac_g: mna::Term,
    pub ac_s: mna::Term,
    /// Noise transimpedance node indices (solved-node terminals only).
    pub noise_nodes: [Option<usize>; 4],
}

/// Immutable silicon campaign template.
pub struct SiliconTemplate {
    pub pdk: Arc<PdkCompiled>,
    /// True for TSMC28: apply the `mulu0 -> u0` fold when building cards.
    pub fold_mulu0: bool,
    pub circuit: transient::Problem,
    pub dc_devices: Vec<bsim_transient::Device>,
    pub devices: Vec<DeviceStatic>,
    /// LTI element base: everything except the per-candidate dense devices.
    pub lti_base: lti::Problem,
    /// Resistor thermal-noise injections `(a_term, b_term, 4kT/R)`.
    pub resistor_noise: Vec<(mna::Term, mna::Term, f64)>,
    pub output_weights: Vec<(usize, f64)>,
    pub sense: Vec<f64>,
    pub vin_norm: f64,
    pub freqs: Vec<f64>,
    pub band_lo: f64,
    pub band_hi: f64,
    pub dc_guesses: Vec<Vec<f64>>,
    pub dc_options: bsim_transient::DcOptions,
    pub latch_nodes: Option<(usize, usize)>,
}

/// One candidate device geometry: `w`/`l` in µm, `nf`/`mult` counts, `delvto`
/// mismatch volts.
#[derive(Clone, Copy, Debug)]
pub struct SiliconGeom {
    pub w_um: f64,
    pub l_um: f64,
    pub nf: i64,
    pub mult: i64,
    pub delvto: f64,
}

#[derive(Clone, Debug)]
pub struct SiliconCandidate {
    pub devices: Vec<SiliconGeom>,
    pub corner: String,
    pub seed: Option<Vec<f64>>,
    pub trust_seed_as_op: bool,
}

/// Metrics with the same key shape as the AFE family.
#[derive(Clone, Debug)]
pub struct SiliconMetrics {
    pub gain_peak_db: f64,
    pub bw_hz: f64,
    pub irn_uv: f64,
    pub latch_dv: f64,
    pub dc_op: Vec<f64>,
    pub dc_iterations: usize,
    pub dc_from_seed: bool,
}

/// Owned native BSIM4 handle (freed on drop). Created, used, and destroyed on
/// one worker thread — it never crosses threads.
struct HandleGuard(*mut CoBsim4);

impl Drop for HandleGuard {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe { co_bsim4::destroy(self.0) };
            self.0 = std::ptr::null_mut();
        }
    }
}

/// `Bsim4ModelCard.__post_init__` normalization: lower-case keys, drop
/// `level`/`version`, reject non-finite values.
fn normalize_model(params: &HashMap<String, f64>) -> Result<HashMap<String, f64>, String> {
    let mut out = HashMap::with_capacity(params.len());
    for (name, value) in params {
        let key = name.to_lowercase();
        if key == "level" || key == "version" {
            continue;
        }
        if !value.is_finite() {
            return Err(format!("model parameter {key:?} is non-finite: {value}"));
        }
        out.insert(key, *value);
    }
    Ok(out)
}

/// `Bsim4InstanceCard.__post_init__` normalization (lower-case + finite).
fn normalize_instance(params: &HashMap<String, f64>) -> Result<HashMap<String, f64>, String> {
    let mut out = HashMap::with_capacity(params.len());
    for (name, value) in params {
        if !value.is_finite() {
            return Err(format!(
                "instance parameter {name:?} is non-finite: {value}"
            ));
        }
        out.insert(name.to_lowercase(), *value);
    }
    Ok(out)
}

/// Build one native handle from a compiled numeric card, applying the TSMC28
/// `mulu0 -> u0` fold when requested. Mirrors `to_bsim4_cards()` +
/// `_NativeDevice.__init__`.
fn build_handle(
    template: &SiliconTemplate,
    stat: &DeviceStatic,
    geom: &SiliconGeom,
    corner: &str,
) -> Result<HandleGuard, String> {
    let card = template
        .pdk
        .numeric_card(
            &stat.polarity,
            corner,
            stat.temp_c,
            Some(geom.w_um),
            Some(geom.l_um),
            geom.nf,
            geom.mult,
            Some(geom.delvto),
            stat.reference_width_um,
        )
        .map_err(|error| format!("numeric card failed: {}", error.message))?;
    let mut model = normalize_model(&card.model_parameters)?;
    let mut instance = normalize_instance(&card.instance_parameters)?;
    if template.fold_mulu0 {
        co_pdk::apply_mulu0_fold(&mut model, &mut instance).map_err(|error| error.message)?;
    }

    let handle = unsafe { co_bsim4::create(stat.polarity_sign, stat.temperature_k) };
    if handle.is_null() {
        return Err("BSIM4 native device allocation failed".to_string());
    }
    let guard = HandleGuard(handle);
    for (name, value) in &model {
        let cname = CString::new(name.as_str())
            .map_err(|_| format!("model parameter name {name:?} contains NUL"))?;
        let status = unsafe { co_bsim4::set_model(guard.0, cname.as_ptr(), *value) };
        if status != 0 {
            return Err(format!("model setup failed for {name:?} (status {status})"));
        }
    }
    for (name, value) in &instance {
        let cname = CString::new(name.as_str())
            .map_err(|_| format!("instance parameter name {name:?} contains NUL"))?;
        let status = unsafe { co_bsim4::set_instance(guard.0, cname.as_ptr(), *value) };
        if status != 0 {
            return Err(format!(
                "instance setup failed for {name:?} (status {status})"
            ));
        }
    }
    let status = unsafe { co_bsim4::setup(guard.0) };
    if status != 0 {
        return Err(format!("BSIM4 temperature/setup failed (status {status})"));
    }
    Ok(guard)
}

struct GuardEvaluator<'a> {
    handles: &'a [HandleGuard],
}

impl bsim_transient::Evaluator for GuardEvaluator<'_> {
    fn evaluate(
        &mut self,
        index: usize,
        terminals: [f64; 4],
    ) -> Option<bsim_transient::Evaluation> {
        let handle = self.handles.get(index)?.0;
        let mut evaluation = bsim_transient::Evaluation::default();
        let status = unsafe {
            co_bsim4::eval_vp(
                handle,
                terminals.as_ptr(),
                evaluation.currents.as_mut_ptr(),
                evaluation.conductance.as_mut_ptr(),
                evaluation.charges.as_mut_ptr(),
                evaluation.capacitance.as_mut_ptr(),
            )
        };
        (status == 0).then_some(evaluation)
    }

    /// DC-Newton eval (D6 acLoad-skip): skip the small-signal capacitance/charge
    /// extraction unless `CIRCUIT_BSIM4_FULL_EVAL` forces the full path. `solve_dc`
    /// consumes only currents+conductance, which are bit-for-bit identical.
    fn evaluate_dc(
        &mut self,
        index: usize,
        terminals: [f64; 4],
    ) -> Option<bsim_transient::Evaluation> {
        let handle = self.handles.get(index)?.0;
        let mut evaluation = bsim_transient::Evaluation::default();
        let status = unsafe {
            co_bsim4::eval_vp_dc(
                handle,
                terminals.as_ptr(),
                evaluation.currents.as_mut_ptr(),
                evaluation.conductance.as_mut_ptr(),
                evaluation.charges.as_mut_ptr(),
                evaluation.capacitance.as_mut_ptr(),
                co_bsim4::full_eval_forced(),
            )
        };
        (status == 0).then_some(evaluation)
    }
}

pub struct SiliconEvaluator<'a> {
    pub template: &'a SiliconTemplate,
    pub candidates: &'a [SiliconCandidate],
    pub compute_noise: bool,
}

impl SiliconEvaluator<'_> {
    fn solve_dc(
        &self,
        cand: &SiliconCandidate,
        handles: &[HandleGuard],
        index: usize,
    ) -> Result<(Vec<f64>, usize, bool), String> {
        let t = self.template;
        if let (true, Some(seed)) = (cand.trust_seed_as_op, &cand.seed) {
            if seed.len() != t.circuit.size {
                return Err(format!(
                    "candidate {index}: seed length {} != n_aug {}",
                    seed.len(),
                    t.circuit.size
                ));
            }
            return Ok((seed.clone(), 0, true));
        }
        let mut guesses: Vec<&[f64]> = Vec::new();
        if let Some(seed) = &cand.seed {
            if seed.len() != t.circuit.size {
                return Err(format!(
                    "candidate {index}: seed length {} != n_aug {}",
                    seed.len(),
                    t.circuit.size
                ));
            }
            guesses.push(seed.as_slice());
        }
        for guess in &t.dc_guesses {
            guesses.push(guess.as_slice());
        }
        for (gi, guess) in guesses.iter().enumerate() {
            let mut evaluator = GuardEvaluator { handles };
            let result = bsim_transient::solve_dc(
                &t.circuit,
                &t.dc_devices,
                &mut evaluator,
                guess,
                &[],
                t.dc_options,
            );
            if result.converged {
                let from_seed = gi == 0 && cand.seed.is_some();
                return Ok((result.state, result.iterations, from_seed));
            }
        }
        Err(format!(
            "candidate {index}: DC did not converge from any guess"
        ))
    }
}

impl CandidateEvaluator for SiliconEvaluator<'_> {
    type Output = SiliconMetrics;

    fn evaluate(&self, index: usize, inner_parallel: bool) -> CandidateOutcome<SiliconMetrics> {
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

        // 1. Device build: numeric cards -> native handles (fresh per candidate).
        let mut handles = Vec::with_capacity(t.devices.len());
        for (stat, geom) in t.devices.iter().zip(&cand.devices) {
            handles.push(
                build_handle(t, stat, geom, &cand.corner)
                    .map_err(|error| format!("candidate {index}: {error}"))?,
            );
        }

        // 2. DC operating point.
        let (dc_op, dc_iterations, dc_from_seed) = self.solve_dc(cand, &handles, index)?;

        // 3. One eval per device at the op: the terminal linearization AND the
        //    bias state the noise call reads (evaluate-then-noise order). The
        //    bias comes from the DC terminal tokens (rails carry their true DC
        //    voltage there — the AC tokens are small-signal values).
        let mut problem = t.lti_base.clone();
        for (slot, (stat, handle)) in t.devices.iter().zip(&handles).enumerate() {
            let dc_terms = &t.dc_devices[slot].terms;
            let resolve = |term: &mna::Term| term.resolve(&dc_op, &[]).unwrap_or(term.value);
            let vd = resolve(&dc_terms[0]);
            let vg = resolve(&dc_terms[1]);
            let vs = resolve(&dc_terms[2]);
            let mut evaluation = bsim_transient::Evaluation::default();
            let status = unsafe {
                co_bsim4::eval_vp(
                    handle.0,
                    [vd, vg, vs, stat.vb].as_ptr(),
                    evaluation.currents.as_mut_ptr(),
                    evaluation.conductance.as_mut_ptr(),
                    evaluation.charges.as_mut_ptr(),
                    evaluation.capacitance.as_mut_ptr(),
                )
            };
            if status != 0 {
                return Err(format!(
                    "candidate {index}: BSIM4 evaluation failed at the operating point"
                ));
            }
            problem.dense_devices.push(lti::DenseDevice {
                terms: vec![
                    stat.ac_d,
                    stat.ac_g,
                    stat.ac_s,
                    mna::Term {
                        kind: 2,
                        reference: 0,
                        value: 0.0,
                    },
                ],
                conductance: evaluation.conductance.to_vec(),
                capacitance: evaluation.capacitance.to_vec(),
            });
        }

        // 4. AC solve + gain reductions.
        let system = problem
            .try_assemble()
            .map_err(|error| format!("candidate {index}: AC assembly failed: {error}"))?;
        let v = if inner_parallel {
            system.solve_frequencies_parallel(&t.freqs)
        } else {
            system.solve_frequencies_serial(&t.freqs)
        }
        .ok_or_else(|| format!("candidate {index}: AC solve singular"))?;
        let mut gains = Vec::with_capacity(t.freqs.len());
        for row in &v {
            let mut re = 0.0;
            let mut im = 0.0;
            for &(node, weight) in &t.output_weights {
                re += weight * row[node].re;
                im += weight * row[node].im;
            }
            gains.push((re / t.vin_norm).hypot(im / t.vin_norm));
        }
        let peak = gains.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let gain_peak_db = 20.0 * peak.max(1e-9).log10();
        let bw_hz = bw_from_gain(&t.freqs, &gains);

        // 5. Noise: transposed solve + per-device BSIM4 noise matrices.
        let irn_uv = if self.compute_noise {
            let tvec = if inner_parallel {
                system.solve_transpose_parallel(&t.freqs, &t.sense)
            } else {
                system.solve_transpose_serial(&t.freqs, &t.sense)
            }
            .ok_or_else(|| format!("candidate {index}: noise transpose solve singular"))?;

            let nfreq = t.freqs.len();
            let mut out_psd = vec![0.0; nfreq];
            let mut total_re = [0.0f64; 16];
            let mut total_im = [0.0f64; 16];
            let mut flicker_re = [0.0f64; 16];
            let mut flicker_im = [0.0f64; 16];
            for (stat, handle) in t.devices.iter().zip(&handles) {
                for (fi, &frequency) in t.freqs.iter().enumerate() {
                    let status = unsafe {
                        co_bsim4::noise(
                            handle.0,
                            frequency,
                            total_re.as_mut_ptr(),
                            total_im.as_mut_ptr(),
                            flicker_re.as_mut_ptr(),
                            flicker_im.as_mut_ptr(),
                        )
                    };
                    if status != 0 {
                        return Err(format!(
                            "candidate {index}: BSIM4 noise evaluation failed (status {status})"
                        ));
                    }
                    // z_t = tvec[fi][node] for solved-node terminals, else 0.
                    let mut z = [lti::Complex { re: 0.0, im: 0.0 }; 4];
                    for (slot, node) in stat.noise_nodes.iter().enumerate() {
                        if let Some(node) = node {
                            z[slot] = tvec[fi][*node];
                        }
                    }
                    // contribution = max(Re(z S z*), 0) over the total matrix.
                    let mut acc = 0.0;
                    for (row, zi) in z.iter().enumerate() {
                        for (col, zj) in z.iter().enumerate() {
                            let s_re = total_re[row * 4 + col];
                            let s_im = total_im[row * 4 + col];
                            // Re(zi * S * conj(zj))
                            let zr = zi.re * s_re - zi.im * s_im;
                            let zi_im = zi.re * s_im + zi.im * s_re;
                            acc += zr * zj.re + zi_im * zj.im;
                        }
                    }
                    out_psd[fi] += acc.max(0.0);
                }
            }
            for (a, b, s_th) in &t.resistor_noise {
                for (fi, psd) in out_psd.iter_mut().enumerate() {
                    let mut zre = 0.0f64;
                    let mut zim = 0.0f64;
                    if a.kind == 0 {
                        zre += tvec[fi][a.reference].re;
                        zim += tvec[fi][a.reference].im;
                    }
                    if b.kind == 0 {
                        zre -= tvec[fi][b.reference].re;
                        zim -= tvec[fi][b.reference].im;
                    }
                    let zabs = zre.hypot(zim);
                    *psd += zabs * zabs * s_th;
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

        let latch_dv = match t.latch_nodes {
            Some((a, b)) => (dc_op[a] - dc_op[b]).abs(),
            None => 0.0,
        };

        Ok(SiliconMetrics {
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

// ---------------------------------------------------------------------------
// Template / candidate marshalling (mirrors the AFE family's record formats).
// ---------------------------------------------------------------------------

type SilDeviceRecord = (
    String, // polarity "nmos"/"pmos"
    f64,    // vb
    f64,    // temperature_k
    f64,    // temp_c (card selection)
    TermRecord,
    TermRecord,
    TermRecord,  // ac d, g, s
    Option<f64>, // reference_width_um (SKY130 extract_w; None elsewhere)
);
type SilDcDeviceRecord = (Vec<TermRecord>, Vec<i64>);
type SilResNoiseRecord = (TermRecord, TermRecord, f64);

pub(crate) fn build_silicon_template(
    spec: &Bound<'_, PyDict>,
    circuit: transient::Problem,
) -> PyResult<SiliconTemplate> {
    let pdk_name: String = required(spec, "pdk")?;
    let root: Option<String> = optional_field(spec, "root")?;
    let pdk = PdkCompiled::new(&pdk_name, root).map_err(crate::pdk_error_to_py)?;
    let fold_mulu0 = pdk_name == "tsmc28";

    let dc_records: Vec<SilDcDeviceRecord> = required(spec, "dc_devices")?;
    let mut dc_devices = Vec::with_capacity(dc_records.len());
    for (index, (terms, rows)) in dc_records.into_iter().enumerate() {
        if terms.len() != 4 || rows.len() != 4 {
            return Err(PyValueError::new_err(
                "each silicon DC device needs four terms and four rows",
            ));
        }
        dc_devices.push(bsim_transient::Device {
            terms: [
                term(terms[0]),
                term(terms[1]),
                term(terms[2]),
                term(terms[3]),
            ],
            rows: [
                optional_index(rows[0])?,
                optional_index(rows[1])?,
                optional_index(rows[2])?,
                optional_index(rows[3])?,
            ],
            evaluator_index: index,
        });
    }

    let device_records: Vec<SilDeviceRecord> = required(spec, "devices")?;
    if device_records.len() != dc_devices.len() {
        return Err(PyValueError::new_err(
            "devices and dc_devices must have equal length",
        ));
    }
    let mut devices = Vec::with_capacity(device_records.len());
    for record in device_records {
        let polarity = record.0.to_lowercase();
        let polarity_sign = match polarity.as_str() {
            "nmos" => 1,
            "pmos" => -1,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unsupported polarity {other:?} (expected nmos/pmos)"
                )));
            }
        };
        let ac_d = term(record.4);
        let ac_g = term(record.5);
        let ac_s = term(record.6);
        let reference_width_um = record.7;
        if reference_width_um.is_some_and(|width| !(width.is_finite() && width > 0.0)) {
            return Err(PyValueError::new_err(
                "reference_width_um must be a positive finite width or None",
            ));
        }
        let node_of = |value: mna::Term| (value.kind == 0).then_some(value.reference);
        devices.push(DeviceStatic {
            polarity,
            polarity_sign,
            vb: record.1,
            temperature_k: record.2,
            temp_c: record.3,
            reference_width_um,
            ac_d,
            ac_g,
            ac_s,
            noise_nodes: [node_of(ac_d), node_of(ac_g), node_of(ac_s), None],
        });
    }

    let size: usize = required(spec, "n_aug")?;
    let capacitors: Vec<LtiBranchRecord> = required(spec, "ac_caps")?;
    let resistors: Vec<LtiBranchRecord> = required(spec, "ac_resistors")?;
    let vccs: Vec<LtiVccsRecord> = required(spec, "ac_vccs")?;
    let voltage_sources: Vec<LtiVoltageRecord> = required(spec, "ac_vsources")?;
    let vcvs: Vec<LtiVcvsRecord> = required(spec, "ac_vcvs")?;
    let cccs: Vec<LtiCccsRecord> = required(spec, "ac_cccs")?;
    let ccvs: Vec<LtiCcvsRecord> = required(spec, "ac_ccvs")?;
    let lti_base = lti::Problem {
        size,
        dense_devices: Vec::new(),
        mos_devices: Vec::new(),
        capacitors: capacitors
            .into_iter()
            .map(|record| lti::Branch {
                a: term(record.0),
                b: term(record.1),
                value: record.2,
            })
            .collect(),
        resistors: resistors
            .into_iter()
            .map(|record| lti::Branch {
                a: term(record.0),
                b: term(record.1),
                value: record.2,
            })
            .collect(),
        vccs: vccs
            .into_iter()
            .map(|record| lti::Vccs {
                p: term(record.0),
                q: term(record.1),
                cp: term(record.2),
                cn: term(record.3),
                gm: record.4,
            })
            .collect(),
        voltage_sources: voltage_sources
            .into_iter()
            .map(|record| lti::VoltageSource {
                p: term(record.0),
                q: term(record.1),
                branch: record.2,
                emf_re: record.3,
                emf_im: record.4,
            })
            .collect(),
        vcvs: vcvs
            .into_iter()
            .map(|record| lti::Vcvs {
                p: term(record.0),
                q: term(record.1),
                cp: term(record.2),
                cn: term(record.3),
                branch: record.4,
                mu: record.5,
            })
            .collect(),
        cccs: cccs
            .into_iter()
            .map(|record| lti::Cccs {
                p: term(record.0),
                q: term(record.1),
                control_branch: record.2,
                beta: record.3,
            })
            .collect(),
        ccvs: ccvs
            .into_iter()
            .map(|record| lti::Ccvs {
                p: term(record.0),
                q: term(record.1),
                control_branch: record.2,
                branch: record.3,
                gamma: record.4,
            })
            .collect(),
    };

    let resistor_noise: Vec<SilResNoiseRecord> = required(spec, "resistor_noise")?;
    let resistor_noise = resistor_noise
        .into_iter()
        .map(|(a, b, resistance)| (term(a), term(b), 4.0 * KB * RES_TEMP / resistance))
        .collect();

    let band: Vec<f64> = required(spec, "band")?;
    if band.len() != 2 {
        return Err(PyValueError::new_err("band must be [f_lo, f_hi]"));
    }
    let dc_opts: Vec<f64> = required(spec, "dc_options")?;
    if dc_opts.len() != 4 {
        return Err(PyValueError::new_err(
            "dc_options must be [max_iterations, voltage_tolerance, step_limit, gmin]",
        ));
    }

    Ok(SiliconTemplate {
        pdk: Arc::new(pdk),
        fold_mulu0,
        circuit,
        dc_devices,
        devices,
        lti_base,
        resistor_noise,
        output_weights: required(spec, "output_weights")?,
        sense: required(spec, "sense")?,
        vin_norm: required(spec, "vin_norm")?,
        freqs: required(spec, "freqs")?,
        band_lo: band[0],
        band_hi: band[1],
        dc_guesses: required(spec, "dc_guesses")?,
        dc_options: bsim_transient::DcOptions {
            max_iterations: dc_opts[0] as usize,
            voltage_tolerance: dc_opts[1],
            step_limit: dc_opts[2],
            gmin: dc_opts[3],
        },
        latch_nodes: optional_field(spec, "latch_nodes")?,
    })
}

pub(crate) fn parse_silicon_candidate(item: &Bound<'_, PyAny>) -> PyResult<SiliconCandidate> {
    let dict = item.cast::<PyDict>()?;
    let corner: String = required(dict, "corner")?;
    let geoms: Vec<Vec<f64>> = required(dict, "devices")?;
    let mut devices = Vec::with_capacity(geoms.len());
    for geom in geoms {
        if geom.len() != 5 {
            return Err(PyValueError::new_err(
                "each silicon candidate device must be [w_um, l_um, nf, mult, delvto]",
            ));
        }
        devices.push(SiliconGeom {
            w_um: geom[0],
            l_um: geom[1],
            nf: geom[2] as i64,
            mult: geom[3] as i64,
            delvto: geom[4],
        });
    }
    let seed: Option<Vec<f64>> = optional_field(dict, "seed")?;
    let trust_seed_as_op: bool = optional_field(dict, "trust_seed_as_op")?.unwrap_or(false);
    Ok(SiliconCandidate {
        devices,
        corner,
        seed,
        trust_seed_as_op,
    })
}

/// The `template` key must reference an `OtftTransientProblem`-compatible
/// passive circuit under `"circuit"`; extract it and clear device slots.
pub(crate) fn extract_circuit(spec: &Bound<'_, PyDict>) -> PyResult<transient::Problem> {
    let circuit_obj = spec
        .get_item("circuit")?
        .ok_or_else(|| PyKeyError::new_err("circuit"))?;
    let circuit_ref = circuit_obj.extract::<PyRef<'_, crate::OtftTransientProblem>>()?;
    let mut circuit = circuit_ref.problem.clone();
    circuit.devices.clear();
    if !circuit.validate() {
        return Err(PyValueError::new_err("invalid silicon campaign circuit"));
    }
    Ok(circuit)
}
