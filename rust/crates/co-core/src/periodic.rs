//! Periodic small-signal assembly and cyclostationary-noise kernels.

use num_complex::Complex64;

use crate::otft::{self, Params};
use crate::transient::{self, DeviceCache};

#[derive(Clone, Copy, Debug)]
pub struct ValueTerm {
    /// 0 = solved node, 1 = periodic input, 2 = constant.
    pub kind: u8,
    pub reference: usize,
    pub value: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct StampTerm {
    /// 0 = solved state, 1 = small-signal drive, 2 = known/ground.
    pub kind: u8,
    pub reference: usize,
}

#[derive(Clone, Debug)]
pub struct OtftDevice {
    pub value_d: ValueTerm,
    pub value_g: ValueTerm,
    pub value_s: ValueTerm,
    pub stamp_d: StampTerm,
    pub stamp_g: StampTerm,
    pub stamp_s: StampTerm,
    pub params: Params,
    /// `(state index, R_cap, R_cap2)` for the retained gate1 formulation.
    pub gate1: Option<(usize, f64, f64)>,
}

#[derive(Clone, Debug)]
pub struct DenseDevice {
    /// Terminal order is drain, gate, source, bulk.
    pub terminals: [StampTerm; 4],
}

#[derive(Clone, Copy, Debug)]
pub struct Passive {
    pub a: StampTerm,
    pub b: StampTerm,
    pub value: f64,
}

#[derive(Clone, Debug)]
pub struct PacProblem {
    pub node_count: usize,
    pub state_count: usize,
    pub input_count: usize,
    pub drive_count: usize,
    pub devices: Vec<OtftDevice>,
    pub dense_devices: Vec<DenseDevice>,
    pub resistors: Vec<Passive>,
    pub capacitors: Vec<Passive>,
    pub gmin: f64,
    pub fd_step: f64,
}

pub struct PacLinearization {
    pub samples: usize,
    pub state_count: usize,
    pub drive_count: usize,
    pub conductance: Vec<f64>,
    pub capacitance: Vec<f64>,
    pub input_conductance: Vec<f64>,
    pub input_capacitance: Vec<f64>,
}

pub struct HbBlocks {
    pub size: usize,
    pub admittance: Vec<Complex64>,
    pub capacitance: Vec<Complex64>,
}

pub struct FoldedPsd {
    pub frequencies: usize,
    pub sources: usize,
    pub output: Vec<f64>,
    pub devices: Vec<f64>,
}

#[inline]
fn checked_product(values: &[usize]) -> Option<usize> {
    values
        .iter()
        .try_fold(1usize, |acc, value| acc.checked_mul(*value))
}

/// Convert Fourier coefficients of G(t) and C(t) into dense harmonic-balance blocks.
///
/// `gf` and `cf` are row-major `(samples, state, state)` arrays. The returned
/// matrices are row-major `(harmonics * state, harmonics * state)` arrays.
pub fn hb_blocks(
    gf: &[Complex64],
    cf: &[Complex64],
    samples: usize,
    state: usize,
    sidebands: usize,
    fundamental: f64,
    charge_caps: bool,
) -> Option<HbBlocks> {
    let input_len = checked_product(&[samples, state, state])?;
    if samples == 0 || state == 0 || gf.len() != input_len || cf.len() != input_len {
        return None;
    }
    let harmonics = sidebands.checked_mul(2)?.checked_add(1)?;
    let size = harmonics.checked_mul(state)?;
    let output_len = size.checked_mul(size)?;
    let mut admittance = vec![Complex64::new(0.0, 0.0); output_len];
    let mut capacitance = vec![Complex64::new(0.0, 0.0); output_len];

    for row_harmonic in 0..harmonics {
        let kr = row_harmonic as isize - sidebands as isize;
        let row_base = row_harmonic * state;
        for column_harmonic in 0..harmonics {
            let kc = column_harmonic as isize - sidebands as isize;
            let column_base = column_harmonic * state;
            let delta = kr - kc;
            let coefficient = delta.rem_euclid(samples as isize) as usize;
            let omega_harmonic = if charge_caps { kr } else { kc };
            let derivative = Complex64::new(
                0.0,
                2.0 * std::f64::consts::PI * omega_harmonic as f64 * fundamental,
            );
            let coefficient_base = coefficient * state * state;
            for row in 0..state {
                for column in 0..state {
                    let source = coefficient_base + row * state + column;
                    let target = (row_base + row) * size + column_base + column;
                    let c_value = cf[source];
                    admittance[target] = gf[source] + derivative * c_value;
                    capacitance[target] = c_value;
                }
            }
        }
    }
    Some(HbBlocks {
        size,
        admittance,
        capacitance,
    })
}

/// Fold scalar cyclostationary thermal and 1/f sources through HB adjoints.
#[allow(clippy::too_many_arguments)]
pub fn fold_psd(
    adjs: &[Complex64],
    frequencies: &[f64],
    adjoint_width: usize,
    sidebands: usize,
    fundamental: f64,
    p_indices: &[i64],
    q_indices: &[i64],
    source_count: usize,
    thermal: &[Complex64],
    flicker: &[Complex64],
) -> Option<FoldedPsd> {
    let frequency_count = frequencies.len();
    let harmonics = sidebands.checked_mul(2)?.checked_add(1)?;
    let modulation_width = sidebands.checked_mul(4)?.checked_add(1)?;
    if adjoint_width == 0
        || adjs.len() != checked_product(&[frequency_count, adjoint_width])?
        || p_indices.len() != checked_product(&[source_count, harmonics])?
        || q_indices.len() != checked_product(&[source_count, harmonics])?
        || thermal.len() != checked_product(&[source_count, harmonics, harmonics])?
        || flicker.len() != checked_product(&[source_count, modulation_width])?
    {
        return None;
    }
    for &index in p_indices.iter().chain(q_indices) {
        if index < -1 || index >= adjoint_width as i64 {
            return None;
        }
    }

    let mut output = vec![0.0; frequency_count];
    let mut devices = vec![0.0; source_count * frequency_count];
    let mut inverse_frequency = vec![0.0; harmonics];
    let mut z = vec![Complex64::new(0.0, 0.0); harmonics];
    for (frequency_index, frequency) in frequencies.iter().copied().enumerate() {
        for (a, inverse) in inverse_frequency.iter_mut().enumerate() {
            let nu = (frequency + (a as isize - sidebands as isize) as f64 * fundamental)
                .abs()
                .max(1e-9);
            *inverse = 1.0 / nu;
        }
        let adjoint_base = frequency_index * adjoint_width;
        for source in 0..source_count {
            let index_base = source * harmonics;
            for r in 0..harmonics {
                let mut value = Complex64::new(0.0, 0.0);
                let p = p_indices[index_base + r];
                let q = q_indices[index_base + r];
                if p >= 0 {
                    value += adjs[adjoint_base + p as usize];
                }
                if q >= 0 {
                    value -= adjs[adjoint_base + q as usize];
                }
                z[r] = value;
            }

            let thermal_base = source * harmonics * harmonics;
            let mut contribution = 0.0;
            for r in 0..harmonics {
                let mut accumulated = Complex64::new(0.0, 0.0);
                for c in 0..harmonics {
                    accumulated += thermal[thermal_base + r * harmonics + c] * z[c].conj();
                }
                contribution += (z[r] * accumulated).re;
            }

            let modulation_base = source * modulation_width;
            for (a, inverse) in inverse_frequency.iter().copied().enumerate() {
                let base = 2 * sidebands - a;
                let mut u = Complex64::new(0.0, 0.0);
                for r in 0..harmonics {
                    u += z[r] * flicker[modulation_base + base + r];
                }
                contribution += u.norm_sqr() * inverse;
            }
            contribution = contribution.max(0.0);
            devices[source * frequency_count + frequency_index] = contribution;
            output[frequency_index] += contribution;
        }
    }
    Some(FoldedPsd {
        frequencies: frequency_count,
        sources: source_count,
        output,
        devices,
    })
}

#[inline]
fn value_term(
    term: ValueTerm,
    sample: usize,
    node_wave: &[f64],
    node_count: usize,
    input_wave: &[f64],
    samples: usize,
) -> Option<f64> {
    match term.kind {
        0 if term.reference < node_count => Some(node_wave[sample * node_count + term.reference]),
        1 => Some(input_wave[term.reference.checked_mul(samples)? + sample]),
        2 => Some(term.value),
        _ => None,
    }
}

#[inline]
fn term_derivative(
    term: ValueTerm,
    sample: usize,
    node_dot: &[f64],
    node_count: usize,
    input_dot: &[f64],
    samples: usize,
) -> Option<f64> {
    match term.kind {
        0 if term.reference < node_count => Some(node_dot[sample * node_count + term.reference]),
        1 => Some(input_dot[term.reference.checked_mul(samples)? + sample]),
        2 => Some(0.0),
        _ => None,
    }
}

#[inline]
fn stamp_coeff(
    matrix: &mut [f64],
    input_matrix: &mut [f64],
    state_count: usize,
    drive_count: usize,
    row: StampTerm,
    column: StampTerm,
    coefficient: f64,
) {
    if row.kind != 0 || coefficient == 0.0 {
        return;
    }
    if column.kind == 0 {
        matrix[row.reference * state_count + column.reference] += coefficient;
    } else if column.kind == 1 {
        input_matrix[row.reference * drive_count + column.reference] += coefficient;
    }
}

#[inline]
fn stamp_admittance(
    matrix: &mut [f64],
    input_matrix: &mut [f64],
    state_count: usize,
    drive_count: usize,
    p: StampTerm,
    q: StampTerm,
    value: f64,
) {
    if value == 0.0 {
        return;
    }
    if p.kind == 0 {
        matrix[p.reference * state_count + p.reference] += value;
        stamp_coeff(matrix, input_matrix, state_count, drive_count, p, q, -value);
    }
    if q.kind == 0 {
        matrix[q.reference * state_count + q.reference] += value;
        stamp_coeff(matrix, input_matrix, state_count, drive_count, q, p, -value);
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn stamp_vccs(
    matrix: &mut [f64],
    input_matrix: &mut [f64],
    state_count: usize,
    drive_count: usize,
    d: StampTerm,
    g: StampTerm,
    s: StampTerm,
    gm: f64,
) {
    stamp_coeff(matrix, input_matrix, state_count, drive_count, d, g, gm);
    stamp_coeff(matrix, input_matrix, state_count, drive_count, d, s, -gm);
    stamp_coeff(matrix, input_matrix, state_count, drive_count, s, g, -gm);
    stamp_coeff(matrix, input_matrix, state_count, drive_count, s, s, gm);
}

fn solved_idc(params: &Params, vs: f64, vd: f64, vg: f64, cache: DeviceCache) -> Option<f64> {
    let (ok, solved, ..) = transient::solve_internal_with_guesses(params, vs, vd, vg, cache);
    if !ok {
        return None;
    }
    let jac = otft::residual_pair_jac_internal(params, vs, vd, vg, solved.vs1, solved.vd1);
    Some(jac[1] - (solved.vs1 - solved.vd1) / 0.1)
}

fn operating_point(
    device: &OtftDevice,
    vs: f64,
    vd: f64,
    vg: f64,
    cache: DeviceCache,
) -> Option<(DeviceCache, f64, f64, f64, f64)> {
    let (ok, solved, ..) =
        transient::solve_internal_with_guesses(&device.params, vs, vd, vg, cache);
    if !ok {
        return None;
    }
    let jac = otft::residual_pair_jac_internal(&device.params, vs, vd, vg, solved.vs1, solved.vd1);
    let idc = jac[1] - (solved.vs1 - solved.vd1) / 0.1;
    let mut finite_difference = idc.abs() < 1e-10;
    let (mut gm, mut gds) = (0.0, 1e-12);
    if !finite_difference {
        let derivative = otft::terminal_derivatives_from_jac(
            &device.params,
            vs,
            vd,
            vg,
            solved.vs1,
            solved.vd1,
            jac[0],
            jac[1],
            idc,
            [jac[2], jac[3], jac[4], jac[5]],
            true,
            true,
            false,
            1e-3,
        );
        if derivative.0 && derivative.1.is_finite() && derivative.2.is_finite() {
            gm = -derivative.1;
            gds = -derivative.2;
        } else {
            finite_difference = true;
        }
    }
    if finite_difference {
        let h = 1e-3;
        let id_gp = solved_idc(&device.params, vs, vd, vg + h, solved)?;
        let id_gm = solved_idc(&device.params, vs, vd, vg - h, solved)?;
        let id_dp = solved_idc(&device.params, vs, vd + h, vg, solved)?;
        let id_dm = solved_idc(&device.params, vs, vd - h, vg, solved)?;
        gm = ((id_gp - id_gm) / (2.0 * h)).max(0.0);
        gds = ((id_dp - id_dm) / (2.0 * h)).max(1e-12);
        if !gm.is_finite() || !gds.is_finite() {
            return None;
        }
    }
    let caps = otft::capacitances(&device.params, vs, vd, vg, solved.vs1, solved.vd1);
    Some((solved, gm, gds, caps[0], caps[1]))
}

fn validate_stamp(term: StampTerm, state_count: usize, drive_count: usize) -> bool {
    match term.kind {
        0 => term.reference < state_count,
        1 => term.reference < drive_count,
        2 => true,
        _ => false,
    }
}

impl PacProblem {
    pub fn validate(&self) -> bool {
        self.node_count > 0
            && self.state_count >= self.node_count
            && self.gmin.is_finite()
            && self.gmin >= 0.0
            && self.fd_step.is_finite()
            && self.fd_step > 0.0
            && self.devices.iter().all(|device| {
                [device.stamp_d, device.stamp_g, device.stamp_s]
                    .into_iter()
                    .all(|term| validate_stamp(term, self.state_count, self.drive_count))
                    && device.gate1.is_none_or(|(reference, r1, r2)| {
                        reference < self.state_count
                            && r1.is_finite()
                            && r1 > 0.0
                            && r2.is_finite()
                            && r2 > 0.0
                    })
            })
            && self.dense_devices.iter().all(|device| {
                device
                    .terminals
                    .into_iter()
                    .all(|term| validate_stamp(term, self.state_count, self.drive_count))
            })
            && self
                .resistors
                .iter()
                .chain(&self.capacitors)
                .all(|passive| {
                    validate_stamp(passive.a, self.state_count, self.drive_count)
                        && validate_stamp(passive.b, self.state_count, self.drive_count)
                        && passive.value.is_finite()
                })
    }
}

/// Linearize an OTFT periodic orbit, including retained gate1 dynamic states.
#[allow(clippy::too_many_arguments)]
pub fn linearize_otft_orbit(
    problem: &PacProblem,
    node_wave: &[f64],
    input_wave: &[f64],
    node_dot: &[f64],
    input_dot: &[f64],
    dense_conductance: &[f64],
    dense_capacitance: &[f64],
    samples: usize,
) -> Option<PacLinearization> {
    let dense_len = checked_product(&[samples, problem.dense_devices.len(), 4, 4])?;
    if !problem.validate()
        || samples == 0
        || node_wave.len() != checked_product(&[samples, problem.node_count])?
        || node_dot.len() != node_wave.len()
        || input_wave.len() != checked_product(&[problem.input_count, samples])?
        || input_dot.len() != input_wave.len()
        || dense_conductance.len() != dense_len
        || dense_capacitance.len() != dense_len
    {
        return None;
    }
    let matrix_stride = problem.state_count.checked_mul(problem.state_count)?;
    let input_stride = problem.state_count.checked_mul(problem.drive_count)?;
    let mut conductance = vec![0.0; samples.checked_mul(matrix_stride)?];
    let mut capacitance = vec![0.0; samples.checked_mul(matrix_stride)?];
    let mut input_conductance = vec![0.0; samples.checked_mul(input_stride)?];
    let mut input_capacitance = vec![0.0; samples.checked_mul(input_stride)?];
    let mut caches = vec![DeviceCache::default(); problem.devices.len()];

    for sample in 0..samples {
        let matrix_base = sample * matrix_stride;
        let input_base = sample * input_stride;
        let gm = &mut conductance[matrix_base..matrix_base + matrix_stride];
        let cm = &mut capacitance[matrix_base..matrix_base + matrix_stride];
        let gim = &mut input_conductance[input_base..input_base + input_stride];
        let cim = &mut input_capacitance[input_base..input_base + input_stride];
        for state in 0..problem.node_count {
            gm[state * problem.state_count + state] += problem.gmin;
        }
        for resistor in &problem.resistors {
            stamp_admittance(
                gm,
                gim,
                problem.state_count,
                problem.drive_count,
                resistor.a,
                resistor.b,
                resistor.value,
            );
        }
        for capacitor in &problem.capacitors {
            stamp_admittance(
                cm,
                cim,
                problem.state_count,
                problem.drive_count,
                capacitor.a,
                capacitor.b,
                capacitor.value,
            );
        }
        for (position, device) in problem.dense_devices.iter().enumerate() {
            let base = (sample * problem.dense_devices.len() + position) * 16;
            for row in 0..4 {
                for column in 0..4 {
                    let offset = base + row * 4 + column;
                    stamp_coeff(
                        gm,
                        gim,
                        problem.state_count,
                        problem.drive_count,
                        device.terminals[row],
                        device.terminals[column],
                        dense_conductance[offset],
                    );
                    stamp_coeff(
                        cm,
                        cim,
                        problem.state_count,
                        problem.drive_count,
                        device.terminals[row],
                        device.terminals[column],
                        dense_capacitance[offset],
                    );
                }
            }
        }
        for (position, device) in problem.devices.iter().enumerate() {
            let vs = value_term(
                device.value_s,
                sample,
                node_wave,
                problem.node_count,
                input_wave,
                samples,
            )?;
            let vd = value_term(
                device.value_d,
                sample,
                node_wave,
                problem.node_count,
                input_wave,
                samples,
            )?;
            let vg = value_term(
                device.value_g,
                sample,
                node_wave,
                problem.node_count,
                input_wave,
                samples,
            )?;
            let (solved, transconductance, output_conductance, cgs, cgd) =
                operating_point(device, vs, vd, vg, caches[position])?;
            caches[position] = solved;
            stamp_admittance(
                gm,
                gim,
                problem.state_count,
                problem.drive_count,
                device.stamp_d,
                device.stamp_s,
                output_conductance,
            );
            stamp_vccs(
                gm,
                gim,
                problem.state_count,
                problem.drive_count,
                device.stamp_d,
                device.stamp_g,
                device.stamp_s,
                transconductance,
            );
            if let Some((gate1_reference, r_cap, r_cap2)) = device.gate1 {
                let gate1 = StampTerm {
                    kind: 0,
                    reference: gate1_reference,
                };
                let inv_r_cap = 1.0 / r_cap;
                let inv_r_cap2 = 1.0 / r_cap2;
                stamp_admittance(
                    gm,
                    gim,
                    problem.state_count,
                    problem.drive_count,
                    gate1,
                    device.stamp_g,
                    inv_r_cap,
                );
                stamp_admittance(
                    gm,
                    gim,
                    problem.state_count,
                    problem.drive_count,
                    device.stamp_s,
                    gate1,
                    inv_r_cap2,
                );
                stamp_admittance(
                    gm,
                    gim,
                    problem.state_count,
                    problem.drive_count,
                    device.stamp_d,
                    gate1,
                    inv_r_cap2,
                );
                stamp_admittance(
                    cm,
                    cim,
                    problem.state_count,
                    problem.drive_count,
                    device.stamp_s,
                    gate1,
                    cgs,
                );
                stamp_admittance(
                    cm,
                    cim,
                    problem.state_count,
                    problem.drive_count,
                    device.stamp_d,
                    gate1,
                    cgd,
                );

                let dvs = term_derivative(
                    device.value_s,
                    sample,
                    node_dot,
                    problem.node_count,
                    input_dot,
                    samples,
                )?;
                let dvd = term_derivative(
                    device.value_d,
                    sample,
                    node_dot,
                    problem.node_count,
                    input_dot,
                    samples,
                )?;
                let dvg = term_derivative(
                    device.value_g,
                    sample,
                    node_dot,
                    problem.node_count,
                    input_dot,
                    samples,
                )?;
                let denominator = inv_r_cap + 2.0 * inv_r_cap2;
                let dvg1 = (dvg * inv_r_cap + (dvs + dvd) * inv_r_cap2) / denominator;
                let voltage_dot_s = dvs - dvg1;
                let voltage_dot_d = dvd - dvg1;
                if voltage_dot_s.abs() >= 1e-30 || voltage_dot_d.abs() >= 1e-30 {
                    for axis in 0..3 {
                        let h = problem.fd_step;
                        let mut plus = [vs, vd, vg];
                        let mut minus = [vs, vd, vg];
                        plus[axis] += h;
                        minus[axis] -= h;
                        let (ok_plus, solved_plus, ..) = transient::solve_internal_with_guesses(
                            &device.params,
                            plus[0],
                            plus[1],
                            plus[2],
                            solved,
                        );
                        let (ok_minus, solved_minus, ..) = transient::solve_internal_with_guesses(
                            &device.params,
                            minus[0],
                            minus[1],
                            minus[2],
                            solved,
                        );
                        if !ok_plus || !ok_minus {
                            return None;
                        }
                        let caps_plus = otft::capacitances(
                            &device.params,
                            plus[0],
                            plus[1],
                            plus[2],
                            solved_plus.vs1,
                            solved_plus.vd1,
                        );
                        let caps_minus = otft::capacitances(
                            &device.params,
                            minus[0],
                            minus[1],
                            minus[2],
                            solved_minus.vs1,
                            solved_minus.vd1,
                        );
                        let dcgs = (caps_plus[0] - caps_minus[0]) / (2.0 * h);
                        let dcgd = (caps_plus[1] - caps_minus[1]) / (2.0 * h);
                        let control = match axis {
                            0 => device.stamp_s,
                            1 => device.stamp_d,
                            _ => device.stamp_g,
                        };
                        if voltage_dot_s != 0.0 && dcgs != 0.0 {
                            let coefficient = dcgs * voltage_dot_s;
                            stamp_coeff(
                                gm,
                                gim,
                                problem.state_count,
                                problem.drive_count,
                                device.stamp_s,
                                control,
                                coefficient,
                            );
                            stamp_coeff(
                                gm,
                                gim,
                                problem.state_count,
                                problem.drive_count,
                                gate1,
                                control,
                                -coefficient,
                            );
                        }
                        if voltage_dot_d != 0.0 && dcgd != 0.0 {
                            let coefficient = dcgd * voltage_dot_d;
                            stamp_coeff(
                                gm,
                                gim,
                                problem.state_count,
                                problem.drive_count,
                                device.stamp_d,
                                control,
                                coefficient,
                            );
                            stamp_coeff(
                                gm,
                                gim,
                                problem.state_count,
                                problem.drive_count,
                                gate1,
                                control,
                                -coefficient,
                            );
                        }
                    }
                }
            } else {
                stamp_admittance(
                    cm,
                    cim,
                    problem.state_count,
                    problem.drive_count,
                    device.stamp_g,
                    device.stamp_s,
                    cgs,
                );
                stamp_admittance(
                    cm,
                    cim,
                    problem.state_count,
                    problem.drive_count,
                    device.stamp_g,
                    device.stamp_d,
                    cgd,
                );
            }
        }
    }
    Some(PacLinearization {
        samples,
        state_count: problem.state_count,
        drive_count: problem.drive_count,
        conductance,
        capacitance,
        input_conductance,
        input_capacitance,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dc_hb_block_repeats_static_matrix() {
        let g = vec![Complex64::new(2.0, 0.0)];
        let c = vec![Complex64::new(3.0, 0.0)];
        let blocks = hb_blocks(&g, &c, 1, 1, 1, 10.0, false).unwrap();
        assert_eq!(blocks.size, 3);
        assert_eq!(blocks.capacitance, vec![c[0]; 9]);
        assert_eq!(blocks.admittance[4], g[0]);
        assert_eq!(blocks.admittance[0].im, -60.0 * std::f64::consts::PI);
        assert_eq!(blocks.admittance[8].im, 60.0 * std::f64::consts::PI);
    }

    #[test]
    fn one_white_source_folds_to_adjoint_norm() {
        let result = fold_psd(
            &[Complex64::new(2.0, 0.0)],
            &[1.0],
            1,
            0,
            10.0,
            &[0],
            &[-1],
            1,
            &[Complex64::new(3.0, 0.0)],
            &[Complex64::new(0.0, 0.0)],
        )
        .unwrap();
        assert_eq!(result.output, vec![12.0]);
        assert_eq!(result.devices, vec![12.0]);
    }
}
