//! Linear time-invariant MNA assembly and complex frequency solves.

use crate::{CoreError, mna::Term};
use rayon::prelude::*;
use std::ops::{Add, AddAssign, Div, Mul, Sub, SubAssign};

// Rayon task dispatch costs more than the tiny LU for the common 61-point,
// ~10-state sweeps. Use cubic matrix work as a conservative crossover proxy.
const PARALLEL_LTI_WORK_MIN: usize = 1_000_000;

fn use_parallel_frequency_solve(size: usize, frequency_count: usize) -> bool {
    frequency_count > 1
        && frequency_count
            .saturating_mul(size)
            .saturating_mul(size)
            .saturating_mul(size)
            >= PARALLEL_LTI_WORK_MIN
}

#[derive(Clone, Debug)]
pub struct DenseDevice {
    pub terms: Vec<Term>,
    pub conductance: Vec<f64>,
    pub capacitance: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct MosDevice {
    pub drain: Term,
    pub gate: Term,
    pub source: Term,
    pub gm: f64,
    pub gds: f64,
    pub cgs: f64,
    pub cgd: f64,
}

#[derive(Clone, Debug)]
pub struct Branch {
    pub a: Term,
    pub b: Term,
    pub value: f64,
}

#[derive(Clone, Debug)]
pub struct Vccs {
    pub p: Term,
    pub q: Term,
    pub cp: Term,
    pub cn: Term,
    pub gm: f64,
}

#[derive(Clone, Debug)]
pub struct VoltageSource {
    pub p: Term,
    pub q: Term,
    pub branch: usize,
    pub emf_re: f64,
    pub emf_im: f64,
}

#[derive(Clone, Debug)]
pub struct Vcvs {
    pub p: Term,
    pub q: Term,
    pub cp: Term,
    pub cn: Term,
    pub branch: usize,
    pub mu: f64,
}

#[derive(Clone, Debug)]
pub struct Cccs {
    pub p: Term,
    pub q: Term,
    pub control_branch: usize,
    pub beta: f64,
}

#[derive(Clone, Debug)]
pub struct Ccvs {
    pub p: Term,
    pub q: Term,
    pub control_branch: usize,
    pub branch: usize,
    pub gamma: f64,
}

#[derive(Clone, Debug)]
pub struct Problem {
    pub size: usize,
    pub dense_devices: Vec<DenseDevice>,
    pub mos_devices: Vec<MosDevice>,
    pub capacitors: Vec<Branch>,
    pub resistors: Vec<Branch>,
    pub vccs: Vec<Vccs>,
    pub voltage_sources: Vec<VoltageSource>,
    pub vcvs: Vec<Vcvs>,
    pub cccs: Vec<Cccs>,
    pub ccvs: Vec<Ccvs>,
}

#[derive(Clone, Debug)]
pub struct System {
    pub size: usize,
    pub conductance: Vec<f64>,
    pub capacitance: Vec<f64>,
    pub rhs_g: Vec<f64>,
    pub rhs_g_im: Vec<f64>,
    pub rhs_c: Vec<f64>,
}

fn solved(term: Term) -> Option<usize> {
    (term.kind == 0).then_some(term.reference)
}

fn known(term: Term) -> f64 {
    term.value
}

fn add_matrix(matrix: &mut [f64], size: usize, row: usize, col: usize, value: f64) {
    matrix[row * size + col] += value;
}

fn stamp_admittance(
    matrix: &mut [f64],
    rhs: &mut [f64],
    size: usize,
    a: Term,
    b: Term,
    value: f64,
) {
    if let Some(ai) = solved(a) {
        add_matrix(matrix, size, ai, ai, value);
        if let Some(bi) = solved(b) {
            add_matrix(matrix, size, ai, bi, -value);
        } else {
            rhs[ai] += value * known(b);
        }
    }
    if let Some(bi) = solved(b) {
        add_matrix(matrix, size, bi, bi, value);
        if let Some(ai) = solved(a) {
            add_matrix(matrix, size, bi, ai, -value);
        } else {
            rhs[bi] += value * known(a);
        }
    }
}

fn stamp_vccs(matrix: &mut [f64], rhs: &mut [f64], size: usize, source: &Vccs) {
    for (row_term, sign) in [(source.p, 1.0), (source.q, -1.0)] {
        let Some(row) = solved(row_term) else {
            continue;
        };
        for (control, coefficient) in [
            (source.cp, sign * source.gm),
            (source.cn, -sign * source.gm),
        ] {
            if let Some(column) = solved(control) {
                add_matrix(matrix, size, row, column, coefficient);
            } else {
                rhs[row] -= coefficient * known(control);
            }
        }
    }
}

fn stamp_voltage_incidence(
    matrix: &mut [f64],
    rhs: &mut [f64],
    size: usize,
    p: Term,
    q: Term,
    branch: usize,
    emf: f64,
) {
    let mut branch_rhs = emf;
    if let Some(pi) = solved(p) {
        add_matrix(matrix, size, pi, branch, 1.0);
        add_matrix(matrix, size, branch, pi, 1.0);
    } else {
        branch_rhs -= known(p);
    }
    if let Some(qi) = solved(q) {
        add_matrix(matrix, size, qi, branch, -1.0);
        add_matrix(matrix, size, branch, qi, -1.0);
    } else {
        branch_rhs += known(q);
    }
    rhs[branch] += branch_rhs;
}

impl Problem {
    fn validate(&self) -> bool {
        let term = |value: Term| value.is_valid(self.size, false);
        let branch = |index: usize| index < self.size;

        self.size > 0
            && self.dense_devices.iter().all(|device| {
                let width = device.terms.len();
                device.terms.iter().copied().all(term)
                    && device.conductance.len() == width * width
                    && device.capacitance.len() == width * width
            })
            && self
                .mos_devices
                .iter()
                .all(|device| term(device.drain) && term(device.gate) && term(device.source))
            && self
                .capacitors
                .iter()
                .chain(&self.resistors)
                .all(|item| term(item.a) && term(item.b))
            && self
                .vccs
                .iter()
                .all(|item| term(item.p) && term(item.q) && term(item.cp) && term(item.cn))
            && self
                .voltage_sources
                .iter()
                .all(|item| term(item.p) && term(item.q) && branch(item.branch))
            && self.vcvs.iter().all(|item| {
                term(item.p)
                    && term(item.q)
                    && term(item.cp)
                    && term(item.cn)
                    && branch(item.branch)
            })
            && self
                .cccs
                .iter()
                .all(|item| term(item.p) && term(item.q) && branch(item.control_branch))
            && self.ccvs.iter().all(|item| {
                term(item.p) && term(item.q) && branch(item.control_branch) && branch(item.branch)
            })
    }

    pub fn assemble(&self) -> Option<System> {
        if !self.validate() {
            return None;
        }
        let mut system = System {
            size: self.size,
            conductance: vec![0.0; self.size * self.size],
            capacitance: vec![0.0; self.size * self.size],
            rhs_g: vec![0.0; self.size],
            rhs_g_im: vec![0.0; self.size],
            rhs_c: vec![0.0; self.size],
        };
        for device in &self.dense_devices {
            let width = device.terms.len();
            for (source_row, row_term) in device.terms.iter().copied().enumerate() {
                let Some(row) = solved(row_term) else {
                    continue;
                };
                for (source_col, col_term) in device.terms.iter().copied().enumerate() {
                    let offset = source_row * width + source_col;
                    let g = device.conductance[offset];
                    let c = device.capacitance[offset];
                    if let Some(column) = solved(col_term) {
                        add_matrix(&mut system.conductance, self.size, row, column, g);
                        add_matrix(&mut system.capacitance, self.size, row, column, c);
                    } else {
                        system.rhs_g[row] -= g * known(col_term);
                        system.rhs_c[row] -= c * known(col_term);
                    }
                }
            }
        }
        for device in &self.mos_devices {
            stamp_admittance(
                &mut system.conductance,
                &mut system.rhs_g,
                self.size,
                device.drain,
                device.source,
                device.gds,
            );
            stamp_admittance(
                &mut system.capacitance,
                &mut system.rhs_c,
                self.size,
                device.gate,
                device.source,
                device.cgs,
            );
            stamp_admittance(
                &mut system.capacitance,
                &mut system.rhs_c,
                self.size,
                device.gate,
                device.drain,
                device.cgd,
            );
            stamp_vccs(
                &mut system.conductance,
                &mut system.rhs_g,
                self.size,
                &Vccs {
                    p: device.drain,
                    q: device.source,
                    cp: device.gate,
                    cn: device.source,
                    gm: device.gm,
                },
            );
        }
        for branch in &self.capacitors {
            stamp_admittance(
                &mut system.capacitance,
                &mut system.rhs_c,
                self.size,
                branch.a,
                branch.b,
                branch.value,
            );
        }
        for branch in &self.resistors {
            stamp_admittance(
                &mut system.conductance,
                &mut system.rhs_g,
                self.size,
                branch.a,
                branch.b,
                branch.value,
            );
        }
        for source in &self.vccs {
            stamp_vccs(
                &mut system.conductance,
                &mut system.rhs_g,
                self.size,
                source,
            );
        }
        for source in &self.voltage_sources {
            stamp_voltage_incidence(
                &mut system.conductance,
                &mut system.rhs_g,
                self.size,
                source.p,
                source.q,
                source.branch,
                source.emf_re,
            );
            system.rhs_g_im[source.branch] += source.emf_im;
        }
        for source in &self.vcvs {
            stamp_voltage_incidence(
                &mut system.conductance,
                &mut system.rhs_g,
                self.size,
                source.p,
                source.q,
                source.branch,
                0.0,
            );
            for (control, coefficient) in [(source.cp, -source.mu), (source.cn, source.mu)] {
                if let Some(column) = solved(control) {
                    add_matrix(
                        &mut system.conductance,
                        self.size,
                        source.branch,
                        column,
                        coefficient,
                    );
                } else {
                    system.rhs_g[source.branch] -= coefficient * known(control);
                }
            }
        }
        for source in &self.cccs {
            for (output, coefficient) in [(source.p, source.beta), (source.q, -source.beta)] {
                if let Some(row) = solved(output) {
                    add_matrix(
                        &mut system.conductance,
                        self.size,
                        row,
                        source.control_branch,
                        coefficient,
                    );
                }
            }
        }
        for source in &self.ccvs {
            stamp_voltage_incidence(
                &mut system.conductance,
                &mut system.rhs_g,
                self.size,
                source.p,
                source.q,
                source.branch,
                0.0,
            );
            add_matrix(
                &mut system.conductance,
                self.size,
                source.branch,
                source.control_branch,
                -source.gamma,
            );
        }
        Some(system)
    }

    pub fn try_assemble(&self) -> Result<System, CoreError> {
        self.assemble().ok_or(CoreError::InvalidTopology {
            analysis: "LTI MNA problem",
        })
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Complex {
    pub re: f64,
    pub im: f64,
}

impl Complex {
    fn norm_sqr(self) -> f64 {
        self.re * self.re + self.im * self.im
    }
}

impl Add for Complex {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            re: self.re + rhs.re,
            im: self.im + rhs.im,
        }
    }
}
impl AddAssign for Complex {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}
impl Sub for Complex {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self {
            re: self.re - rhs.re,
            im: self.im - rhs.im,
        }
    }
}
impl SubAssign for Complex {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}
impl Mul for Complex {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }
}
impl Div for Complex {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        let denominator = rhs.norm_sqr();
        Self {
            re: (self.re * rhs.re + self.im * rhs.im) / denominator,
            im: (self.im * rhs.re - self.re * rhs.im) / denominator,
        }
    }
}

fn solve_complex(matrix: &mut [Complex], rhs: &mut [Complex], size: usize) -> bool {
    for pivot_col in 0..size {
        let mut pivot_row = pivot_col;
        let mut pivot_abs = matrix[pivot_col * size + pivot_col].norm_sqr();
        for row in pivot_col + 1..size {
            let candidate = matrix[row * size + pivot_col].norm_sqr();
            if candidate > pivot_abs {
                pivot_abs = candidate;
                pivot_row = row;
            }
        }
        if pivot_abs < 1e-60 || !pivot_abs.is_finite() {
            return false;
        }
        if pivot_row != pivot_col {
            for col in 0..size {
                matrix.swap(pivot_col * size + col, pivot_row * size + col);
            }
            rhs.swap(pivot_col, pivot_row);
        }
        let pivot = matrix[pivot_col * size + pivot_col];
        for row in pivot_col + 1..size {
            let factor = matrix[row * size + pivot_col] / pivot;
            matrix[row * size + pivot_col] = Complex::default();
            for col in pivot_col + 1..size {
                let pivot_value = matrix[pivot_col * size + col];
                matrix[row * size + col] -= factor * pivot_value;
            }
            let pivot_rhs = rhs[pivot_col];
            rhs[row] -= factor * pivot_rhs;
        }
    }
    for row in (0..size).rev() {
        let mut value = rhs[row];
        for col in row + 1..size {
            value -= matrix[row * size + col] * rhs[col];
        }
        rhs[row] = value / matrix[row * size + row];
    }
    rhs.iter()
        .all(|value| value.re.is_finite() && value.im.is_finite())
}

impl System {
    fn matrix_at(&self, omega: f64, transpose: bool) -> Vec<Complex> {
        let mut matrix = vec![Complex::default(); self.size * self.size];
        for row in 0..self.size {
            for col in 0..self.size {
                let source = if transpose {
                    col * self.size + row
                } else {
                    row * self.size + col
                };
                matrix[row * self.size + col] = Complex {
                    re: self.conductance[source],
                    im: omega * self.capacitance[source],
                };
            }
        }
        matrix
    }

    fn solve_frequency(&self, frequency: f64) -> Option<Vec<Complex>> {
        let omega = std::f64::consts::TAU * frequency;
        let mut matrix = self.matrix_at(omega, false);
        let mut rhs = self
            .rhs_g
            .iter()
            .zip(&self.rhs_g_im)
            .zip(&self.rhs_c)
            .map(|((g, g_im), c)| Complex {
                re: *g,
                im: g_im + omega * c,
            })
            .collect::<Vec<_>>();
        solve_complex(&mut matrix, &mut rhs, self.size).then_some(rhs)
    }

    pub fn solve_frequencies_serial(&self, frequencies: &[f64]) -> Option<Vec<Vec<Complex>>> {
        frequencies
            .iter()
            .map(|frequency| self.solve_frequency(*frequency))
            .collect()
    }

    pub fn solve_frequencies_parallel(&self, frequencies: &[f64]) -> Option<Vec<Vec<Complex>>> {
        frequencies
            .par_iter()
            .map(|frequency| self.solve_frequency(*frequency))
            .collect()
    }

    pub fn solve_frequencies(&self, frequencies: &[f64]) -> Option<Vec<Vec<Complex>>> {
        if use_parallel_frequency_solve(self.size, frequencies.len()) {
            self.solve_frequencies_parallel(frequencies)
        } else {
            self.solve_frequencies_serial(frequencies)
        }
    }

    pub fn try_solve_frequencies(
        &self,
        frequencies: &[f64],
    ) -> Result<Vec<Vec<Complex>>, CoreError> {
        if frequencies.iter().any(|value| !value.is_finite()) {
            return Err(CoreError::InvalidInput {
                analysis: "LTI frequency",
                detail: "frequencies must be finite",
            });
        }
        self.solve_frequencies(frequencies)
            .ok_or(CoreError::Singular {
                analysis: "LTI frequency",
            })
    }

    fn solve_transpose_frequency(&self, frequency: f64, sense: &[f64]) -> Option<Vec<Complex>> {
        let omega = std::f64::consts::TAU * frequency;
        let mut matrix = self.matrix_at(omega, true);
        let mut rhs = sense
            .iter()
            .map(|value| Complex {
                re: *value,
                im: 0.0,
            })
            .collect::<Vec<_>>();
        solve_complex(&mut matrix, &mut rhs, self.size).then_some(rhs)
    }

    pub fn solve_transpose_serial(
        &self,
        frequencies: &[f64],
        sense: &[f64],
    ) -> Option<Vec<Vec<Complex>>> {
        frequencies
            .iter()
            .map(|frequency| self.solve_transpose_frequency(*frequency, sense))
            .collect()
    }

    pub fn solve_transpose_parallel(
        &self,
        frequencies: &[f64],
        sense: &[f64],
    ) -> Option<Vec<Vec<Complex>>> {
        frequencies
            .par_iter()
            .map(|frequency| self.solve_transpose_frequency(*frequency, sense))
            .collect()
    }

    pub fn solve_transpose(&self, frequencies: &[f64], sense: &[f64]) -> Option<Vec<Vec<Complex>>> {
        if sense.len() != self.size {
            return None;
        }
        if use_parallel_frequency_solve(self.size, frequencies.len()) {
            self.solve_transpose_parallel(frequencies, sense)
        } else {
            self.solve_transpose_serial(frequencies, sense)
        }
    }

    pub fn try_solve_transpose(
        &self,
        frequencies: &[f64],
        sense: &[f64],
    ) -> Result<Vec<Vec<Complex>>, CoreError> {
        if sense.len() != self.size {
            return Err(CoreError::InvalidInput {
                analysis: "transposed LTI",
                detail: "sense length must match system size",
            });
        }
        if frequencies.iter().any(|value| !value.is_finite())
            || sense.iter().any(|value| !value.is_finite())
        {
            return Err(CoreError::InvalidInput {
                analysis: "transposed LTI",
                detail: "frequencies and sense must be finite",
            });
        }
        self.solve_transpose(frequencies, sense)
            .ok_or(CoreError::Singular {
                analysis: "transposed LTI",
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(index: usize) -> Term {
        Term {
            kind: 0,
            reference: index,
            value: 0.0,
        }
    }
    fn known(value: f64) -> Term {
        Term {
            kind: 2,
            reference: 0,
            value,
        }
    }

    #[test]
    fn rc_lowpass_matches_closed_form() {
        let problem = Problem {
            size: 1,
            dense_devices: Vec::new(),
            mos_devices: Vec::new(),
            capacitors: vec![Branch {
                a: node(0),
                b: known(0.0),
                value: 1e-6,
            }],
            resistors: vec![Branch {
                a: node(0),
                b: known(1.0),
                value: 1e-3,
            }],
            vccs: Vec::new(),
            voltage_sources: Vec::new(),
            vcvs: Vec::new(),
            cccs: Vec::new(),
            ccvs: Vec::new(),
        };
        let system = problem.assemble().unwrap();
        let value = system.solve_frequencies(&[1e3]).unwrap()[0][0];
        let expected = Complex { re: 1e-3, im: 0.0 }
            / Complex {
                re: 1e-3,
                im: std::f64::consts::TAU * 1e3 * 1e-6,
            };
        assert!((value.re - expected.re).abs() < 1e-15);
        assert!((value.im - expected.im).abs() < 1e-15);
    }

    #[test]
    fn frequency_parallelism_is_gated_and_bitwise_deterministic() {
        assert!(!use_parallel_frequency_solve(10, 61));
        assert!(use_parallel_frequency_solve(64, 4));

        let problem = Problem {
            size: 1,
            dense_devices: Vec::new(),
            mos_devices: Vec::new(),
            capacitors: vec![Branch {
                a: node(0),
                b: known(0.0),
                value: 1e-9,
            }],
            resistors: vec![Branch {
                a: node(0),
                b: known(1.0),
                value: 1e-3,
            }],
            vccs: Vec::new(),
            voltage_sources: Vec::new(),
            vcvs: Vec::new(),
            cccs: Vec::new(),
            ccvs: Vec::new(),
        };
        let system = problem.assemble().unwrap();
        let frequencies = (0..257)
            .map(|index| 10_f64.powf(index as f64 / 32.0))
            .collect::<Vec<_>>();
        let serial = system.solve_frequencies_serial(&frequencies).unwrap();
        let parallel = system.solve_frequencies_parallel(&frequencies).unwrap();
        for (serial_row, parallel_row) in serial.iter().zip(parallel) {
            for (serial_value, parallel_value) in serial_row.iter().zip(parallel_row) {
                assert_eq!(serial_value.re.to_bits(), parallel_value.re.to_bits());
                assert_eq!(serial_value.im.to_bits(), parallel_value.im.to_bits());
            }
        }
    }
}
