//! OTFT transient stamping and circuit-level Newton primitives.

use crate::otft::{self, Params};
use crate::{
    CoreError,
    mna::{DenseSystem, Term, solve_dense_neg_rhs_in_place},
};

#[derive(Clone, Debug)]
pub struct Device {
    pub drain: Term,
    pub gate: Term,
    pub source: Term,
    pub di: Option<usize>,
    pub gi: Option<usize>,
    pub si: Option<usize>,
    pub use_abs: bool,
    pub params: Params,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct DeviceCache {
    pub valid: bool,
    pub vs1: f64,
    pub vd1: f64,
}

#[derive(Clone, Debug)]
pub struct Resistor {
    pub a: Term,
    pub b: Term,
    pub ai: Option<usize>,
    pub bi: Option<usize>,
    pub conductance: f64,
}

#[derive(Clone, Debug)]
pub struct Capacitor {
    pub a: Term,
    pub b: Term,
    pub ai: Option<usize>,
    pub bi: Option<usize>,
    pub capacitance: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct CurrentSource {
    pub pi: Option<usize>,
    pub qi: Option<usize>,
    pub value: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct DynamicCurrentSource {
    pub pi: Option<usize>,
    pub qi: Option<usize>,
    pub input_index: usize,
}

#[derive(Clone, Debug)]
pub struct Vccs {
    pub pi: Option<usize>,
    pub qi: Option<usize>,
    pub cp: Term,
    pub cn: Term,
    pub cpi: Option<usize>,
    pub cni: Option<usize>,
    pub gm: f64,
}

#[derive(Clone, Debug)]
pub struct VoltageSource {
    pub a: Term,
    pub b: Term,
    pub pi: Option<usize>,
    pub qi: Option<usize>,
    pub branch: usize,
    pub emf: f64,
    pub input_index: Option<usize>,
}

#[derive(Clone, Debug)]
pub struct Vcvs {
    pub a: Term,
    pub b: Term,
    pub cp: Term,
    pub cn: Term,
    pub pi: Option<usize>,
    pub qi: Option<usize>,
    pub cpi: Option<usize>,
    pub cni: Option<usize>,
    pub branch: usize,
    pub mu: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct Cccs {
    pub pi: Option<usize>,
    pub qi: Option<usize>,
    pub control_branch: usize,
    pub beta: f64,
}

#[derive(Clone, Debug)]
pub struct Ccvs {
    pub a: Term,
    pub b: Term,
    pub pi: Option<usize>,
    pub qi: Option<usize>,
    pub branch: usize,
    pub control_branch: usize,
    pub gamma: f64,
}

#[derive(Clone, Debug)]
pub struct Problem {
    pub node_count: usize,
    pub size: usize,
    pub devices: Vec<Device>,
    pub resistors: Vec<Resistor>,
    pub capacitors: Vec<Capacitor>,
    pub current_sources: Vec<CurrentSource>,
    pub dynamic_current_sources: Vec<DynamicCurrentSource>,
    pub vccs: Vec<Vccs>,
    pub voltage_sources: Vec<VoltageSource>,
    pub vcvs: Vec<Vcvs>,
    pub cccs: Vec<Cccs>,
    pub ccvs: Vec<Ccvs>,
}

/// Borrowed row-major waveform matrix (`rows` inputs by `columns` samples).
#[derive(Clone, Copy, Debug)]
pub struct Waveforms<'a> {
    data: &'a [f64],
    rows: usize,
    columns: usize,
}

impl<'a> Waveforms<'a> {
    pub fn new(data: &'a [f64], rows: usize, columns: usize) -> Option<Self> {
        (rows.checked_mul(columns)? == data.len()).then_some(Self {
            data,
            rows,
            columns,
        })
    }

    pub fn rows(self) -> usize {
        self.rows
    }

    pub fn columns(self) -> usize {
        self.columns
    }

    pub fn row(self, index: usize) -> Option<&'a [f64]> {
        if index >= self.rows {
            return None;
        }
        let start = index * self.columns;
        Some(&self.data[start..start + self.columns])
    }

    pub fn sample(self, index: usize) -> Option<Vec<f64>> {
        (index < self.columns).then(|| {
            (0..self.rows)
                .map(|row| self.data[row * self.columns + index])
                .collect()
        })
    }
}

impl Problem {
    pub fn validate(&self) -> bool {
        let row = |index: Option<usize>| index.is_none_or(|value| value < self.node_count);
        let term = |value: Term| value.is_valid(self.node_count, true);
        let branch = |index: usize| index >= self.node_count && index < self.size;

        self.node_count <= self.size
            && self.devices.iter().all(|d| {
                term(d.drain)
                    && term(d.gate)
                    && term(d.source)
                    && [d.di, d.gi, d.si].into_iter().all(row)
            })
            && self
                .resistors
                .iter()
                .all(|item| term(item.a) && term(item.b) && row(item.ai) && row(item.bi))
            && self
                .capacitors
                .iter()
                .all(|item| term(item.a) && term(item.b) && row(item.ai) && row(item.bi))
            && self
                .current_sources
                .iter()
                .all(|item| row(item.pi) && row(item.qi))
            && self
                .dynamic_current_sources
                .iter()
                .all(|item| row(item.pi) && row(item.qi))
            && self.vccs.iter().all(|item| {
                term(item.cp)
                    && term(item.cn)
                    && row(item.pi)
                    && row(item.qi)
                    && row(item.cpi)
                    && row(item.cni)
            })
            && self.voltage_sources.iter().all(|source| {
                term(source.a)
                    && term(source.b)
                    && row(source.pi)
                    && row(source.qi)
                    && branch(source.branch)
            })
            && self.vcvs.iter().all(|source| {
                term(source.a)
                    && term(source.b)
                    && term(source.cp)
                    && term(source.cn)
                    && row(source.pi)
                    && row(source.qi)
                    && row(source.cpi)
                    && row(source.cni)
                    && branch(source.branch)
            })
            && self.ccvs.iter().all(|source| {
                term(source.a)
                    && term(source.b)
                    && row(source.pi)
                    && row(source.qi)
                    && branch(source.branch)
                    && branch(source.control_branch)
            })
            && self
                .cccs
                .iter()
                .all(|source| row(source.pi) && row(source.qi) && branch(source.control_branch))
    }

    pub fn try_validate(&self) -> Result<(), CoreError> {
        self.validate()
            .then_some(())
            .ok_or(CoreError::InvalidTopology {
                analysis: "OTFT transient topology",
            })
    }
}

#[derive(Clone, Debug)]
pub struct HistoryTerms {
    pub vs: Vec<f64>,
    pub vd: Vec<f64>,
    pub vg: Vec<f64>,
    pub cgs: Vec<f64>,
    pub cgd: Vec<f64>,
    pub capacitor_dv: Vec<f64>,
}

impl HistoryTerms {
    pub fn new(problem: &Problem) -> Self {
        let device_count = problem.devices.len();
        Self {
            vs: vec![0.0; device_count],
            vd: vec![0.0; device_count],
            vg: vec![0.0; device_count],
            cgs: vec![0.0; device_count],
            cgd: vec![0.0; device_count],
            capacitor_dv: vec![0.0; problem.capacitors.len()],
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct StampOptions {
    pub h: f64,
    pub gmin: f64,
    pub hh: f64,
    pub cap_mode: i64,
    pub bdf: [f64; 3],
}

#[derive(Clone, Copy, Debug)]
pub struct NewtonOptions {
    pub max_iterations: usize,
    pub step_limit: f64,
    pub voltage_tolerance: f64,
    pub fallback_accept: bool,
    pub fallback_tolerance: f64,
    pub clip_lo: f64,
    pub clip_hi: f64,
    pub gmin: f64,
    pub hh: f64,
    pub cap_mode: i64,
}

#[derive(Clone, Copy, Debug)]
pub struct NewtonResult {
    pub iterations: usize,
    pub converged: bool,
    pub usable: bool,
    pub residual_inf: f64,
    pub step_inf: f64,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct DeviceSolveStats {
    pub solves: usize,
    pub attempts: usize,
    pub iterations: usize,
    pub fd_fallbacks: usize,
    pub terminal_fd_fallbacks: usize,
}

pub(crate) fn solve_internal_with_guesses(
    params: &Params,
    vs: f64,
    vd: f64,
    vg: f64,
    cache: DeviceCache,
) -> (bool, DeviceCache, usize, usize, usize) {
    let mut attempts = 0;
    let mut iterations = 0;
    let mut fd_fallbacks = 0;
    if cache.valid {
        attempts += 1;
        let result =
            otft::newton_internal_fast(params, vs, vd, vg, cache.vs1, cache.vd1, 1e-12, 40);
        iterations += result.iterations;
        fd_fallbacks += result.fd_fallbacks;
        if result.converged {
            return (
                true,
                DeviceCache {
                    valid: true,
                    vs1: result.vs1,
                    vd1: result.vd1,
                },
                attempts,
                iterations,
                fd_fallbacks,
            );
        }
    }
    let span = vs - vd;
    let guesses = [
        (vs - 0.01 * span, vd + 0.01 * span),
        (vs, vd),
        (0.5 * (vs + vd), 0.5 * (vs + vd)),
        (vs, vs),
        (vd, vd),
    ];
    for (x0s, x0d) in guesses {
        attempts += 1;
        let result = otft::newton_internal_fast(params, vs, vd, vg, x0s, x0d, 1e-12, 40);
        iterations += result.iterations;
        fd_fallbacks += result.fd_fallbacks;
        if result.converged {
            return (
                true,
                DeviceCache {
                    valid: true,
                    vs1: result.vs1,
                    vd1: result.vd1,
                },
                attempts,
                iterations,
                fd_fallbacks,
            );
        }
    }
    (false, cache, attempts, iterations, fd_fallbacks)
}

pub fn fill_history_terms(
    problem: &Problem,
    caches: &mut [DeviceCache],
    state: &[f64],
    inputs: &[f64],
    cap_mode: i64,
    history: &mut HistoryTerms,
) -> bool {
    if caches.len() != problem.devices.len()
        || state.len() != problem.size
        || history.vs.len() != problem.devices.len()
        || history.capacitor_dv.len() != problem.capacitors.len()
    {
        return false;
    }
    for (position, device) in problem.devices.iter().enumerate() {
        let Some(vs) = device.source.resolve(state, inputs) else {
            return false;
        };
        let Some(vd) = device.drain.resolve(state, inputs) else {
            return false;
        };
        let Some(vg) = device.gate.resolve(state, inputs) else {
            return false;
        };
        history.vs[position] = vs;
        history.vd[position] = vd;
        history.vg[position] = vg;
        let (ok, solved, _, _, _) =
            solve_internal_with_guesses(&device.params, vs, vd, vg, caches[position]);
        if !ok {
            return false;
        }
        caches[position] = solved;
        let charge = otft::capacitance_charges(&device.params, vs, vd, vg, solved.vs1, solved.vd1);
        history.cgs[position] = if cap_mode == 1 { charge[2] } else { charge[0] };
        history.cgd[position] = if cap_mode == 1 { charge[3] } else { charge[1] };
    }
    for (position, capacitor) in problem.capacitors.iter().enumerate() {
        let Some(va) = capacitor.a.resolve(state, inputs) else {
            return false;
        };
        let Some(vb) = capacitor.b.resolve(state, inputs) else {
            return false;
        };
        history.capacitor_dv[position] = va - vb;
    }
    true
}

#[allow(clippy::too_many_arguments)]
pub fn stamp_system(
    problem: &Problem,
    caches: &mut [DeviceCache],
    state: &[f64],
    input_now: &[f64],
    history: &HistoryTerms,
    history2: &HistoryTerms,
    options: StampOptions,
    system: &mut DenseSystem,
    device_stats: &mut DeviceSolveStats,
) -> bool {
    if !problem.validate()
        || caches.len() != problem.devices.len()
        || state.len() != problem.size
        || system.size() != problem.size
        || options.h <= 0.0
        || !options.h.is_finite()
    {
        return false;
    }
    system.residual.fill(0.0);
    system.jacobian.fill(0.0);
    let inv_h = 1.0 / options.h;

    for (position, device) in problem.devices.iter().enumerate() {
        let Some(vs) = device.source.resolve(state, input_now) else {
            return false;
        };
        let Some(vd) = device.drain.resolve(state, input_now) else {
            return false;
        };
        let Some(vg) = device.gate.resolve(state, input_now) else {
            return false;
        };
        let (ok, solved, attempts, iterations, fd_fallbacks) =
            solve_internal_with_guesses(&device.params, vs, vd, vg, caches[position]);
        device_stats.solves += 1;
        device_stats.attempts += attempts;
        device_stats.iterations += iterations;
        device_stats.fd_fallbacks += fd_fallbacks;
        if !ok {
            return false;
        }
        caches[position] = solved;
        let jac =
            otft::residual_pair_jac_internal(&device.params, vs, vd, vg, solved.vs1, solved.vd1);
        let charge = otft::capacitance_charges(&device.params, vs, vd, vg, solved.vs1, solved.vd1);
        let qgs = charge[0];
        let qgd = charge[1];
        let cgs = charge[2];
        let cgd = charge[3];
        let idc0 = jac[1] - (solved.vs1 - solved.vd1) / 0.1;
        let current = if device.use_abs { idc0.abs() } else { -idc0 };
        if let Some(row) = device.di {
            system.residual[row] += current;
        }
        if let Some(row) = device.si {
            system.residual[row] -= current;
        }

        let leak_g = device.params.gate_leak_g;
        if leak_g != 0.0 {
            let i_sg = (vs - vg) * leak_g;
            if let Some(row) = device.si {
                system.residual[row] -= i_sg;
            }
            if let Some(row) = device.gi {
                system.residual[row] += i_sg;
            }
            let i_dg = (vd - vg) * leak_g;
            if let Some(row) = device.di {
                system.residual[row] -= i_dg;
            }
            if let Some(row) = device.gi {
                system.residual[row] += i_dg;
            }
        }

        if cgs != 0.0 {
            let displacement = if options.cap_mode == 1 {
                0.5 * (cgs + history.cgs[position])
                    * ((vg - vs) - (history.vg[position] - history.vs[position]))
                    * inv_h
            } else {
                (options.bdf[0] * qgs
                    + options.bdf[1] * history.cgs[position]
                    + options.bdf[2] * history2.cgs[position])
                    * inv_h
            };
            if let Some(row) = device.gi {
                system.residual[row] -= displacement;
            }
            if let Some(row) = device.si {
                system.residual[row] += displacement;
            }
        }
        if cgd != 0.0 {
            let displacement = if options.cap_mode == 1 {
                0.5 * (cgd + history.cgd[position])
                    * ((vg - vd) - (history.vg[position] - history.vd[position]))
                    * inv_h
            } else {
                (options.bdf[0] * qgd
                    + options.bdf[1] * history.cgd[position]
                    + options.bdf[2] * history2.cgd[position])
                    * inv_h
            };
            if let Some(row) = device.gi {
                system.residual[row] -= displacement;
            }
            if let Some(row) = device.di {
                system.residual[row] += displacement;
            }
        }

        let need_gm = device.gi.is_some() || device.si.is_some();
        let need_gds = device.di.is_some() || device.si.is_some();
        if (need_gm || need_gds)
            && ((vs - solved.vs1).abs() < 1e-10
                || (solved.vd1 - vd).abs() < 1e-10
                || (solved.vs1 - vd).abs() < 1e-10)
        {
            device_stats.terminal_fd_fallbacks += 1;
        }
        let derivatives = otft::terminal_derivatives_from_jac(
            &device.params,
            vs,
            vd,
            vg,
            solved.vs1,
            solved.vd1,
            jac[0],
            jac[1],
            idc0,
            [jac[2], jac[3], jac[4], jac[5]],
            need_gm,
            need_gds,
            device.use_abs,
            options.hh,
        );
        if !derivatives.0 {
            return false;
        }
        let gm = derivatives.1;
        let gds = derivatives.2;
        let did_vs = -(gm + gds);
        if let Some(row) = device.di {
            if let Some(col) = device.di {
                system.add_jacobian(row, col, gds);
            }
            if let Some(col) = device.gi {
                system.add_jacobian(row, col, gm);
            }
            if let Some(col) = device.si {
                system.add_jacobian(row, col, did_vs);
            }
        }
        if let Some(row) = device.si {
            if let Some(col) = device.di {
                system.add_jacobian(row, col, -gds);
            }
            if let Some(col) = device.gi {
                system.add_jacobian(row, col, -gm);
            }
            if let Some(col) = device.si {
                system.add_jacobian(row, col, -did_vs);
            }
        }

        if leak_g != 0.0 {
            if let Some(row) = device.si {
                system.add_jacobian(row, row, -leak_g);
                if let Some(col) = device.gi {
                    system.add_jacobian(row, col, leak_g);
                }
            }
            if let Some(row) = device.di {
                system.add_jacobian(row, row, -leak_g);
                if let Some(col) = device.gi {
                    system.add_jacobian(row, col, leak_g);
                }
            }
            if let Some(row) = device.gi {
                let mut count = 0;
                if let Some(col) = device.si {
                    system.add_jacobian(row, col, leak_g);
                    count += 1;
                }
                if let Some(col) = device.di {
                    system.add_jacobian(row, col, leak_g);
                    count += 1;
                }
                system.add_jacobian(row, row, -leak_g * f64::from(count));
            }
        }
        if cgs != 0.0 {
            let gc = options.bdf[0] * cgs * inv_h;
            if let Some(row) = device.gi {
                system.add_jacobian(row, row, -gc);
                if let Some(col) = device.si {
                    system.add_jacobian(row, col, gc);
                }
            }
            if let Some(row) = device.si {
                system.add_jacobian(row, row, -gc);
                if let Some(col) = device.gi {
                    system.add_jacobian(row, col, gc);
                }
            }
        }
        if cgd != 0.0 {
            let gc = options.bdf[0] * cgd * inv_h;
            if let Some(row) = device.gi {
                system.add_jacobian(row, row, -gc);
                if let Some(col) = device.di {
                    system.add_jacobian(row, col, gc);
                }
            }
            if let Some(row) = device.di {
                system.add_jacobian(row, row, -gc);
                if let Some(col) = device.gi {
                    system.add_jacobian(row, col, gc);
                }
            }
        }
    }

    system.stamp_gmin(state, problem.node_count, options.gmin);
    for resistor in &problem.resistors {
        if !system.stamp_resistor(
            state,
            input_now,
            resistor.a,
            resistor.b,
            resistor.ai,
            resistor.bi,
            resistor.conductance,
        ) {
            return false;
        }
    }
    for source in &problem.current_sources {
        system.stamp_current_source(source.pi, source.qi, source.value);
    }
    for source in &problem.dynamic_current_sources {
        let Some(value) = input_now.get(source.input_index) else {
            return false;
        };
        system.stamp_current_source(source.pi, source.qi, *value);
    }
    for (position, capacitor) in problem.capacitors.iter().enumerate() {
        if !system.stamp_capacitor(
            state,
            input_now,
            capacitor.a,
            capacitor.b,
            capacitor.ai,
            capacitor.bi,
            capacitor.capacitance,
            inv_h,
            options.bdf,
            history.capacitor_dv[position],
            history2.capacitor_dv[position],
        ) {
            return false;
        }
    }
    for source in &problem.vccs {
        let Some(vcp) = source.cp.resolve(state, input_now) else {
            return false;
        };
        let Some(vcn) = source.cn.resolve(state, input_now) else {
            return false;
        };
        let current = source.gm * (vcp - vcn);
        if let Some(row) = source.pi {
            system.residual[row] += current;
            if let Some(col) = source.cpi {
                system.add_jacobian(row, col, source.gm);
            }
            if let Some(col) = source.cni {
                system.add_jacobian(row, col, -source.gm);
            }
        }
        if let Some(row) = source.qi {
            system.residual[row] -= current;
            if let Some(col) = source.cpi {
                system.add_jacobian(row, col, -source.gm);
            }
            if let Some(col) = source.cni {
                system.add_jacobian(row, col, source.gm);
            }
        }
    }
    for source in &problem.voltage_sources {
        let emf = match source.input_index {
            Some(index) => match input_now.get(index) {
                Some(value) => *value,
                None => return false,
            },
            None => source.emf,
        };
        if !system.stamp_voltage_source(
            state,
            input_now,
            source.a,
            source.b,
            source.pi,
            source.qi,
            source.branch,
            emf,
        ) {
            return false;
        }
    }
    for source in &problem.vcvs {
        if !system.stamp_vcvs(
            state,
            input_now,
            source.a,
            source.b,
            source.cp,
            source.cn,
            source.pi,
            source.qi,
            source.cpi,
            source.cni,
            source.branch,
            source.mu,
        ) {
            return false;
        }
    }
    for source in &problem.cccs {
        system.stamp_cccs(
            state,
            source.pi,
            source.qi,
            source.control_branch,
            source.beta,
        );
    }
    for source in &problem.ccvs {
        if !system.stamp_ccvs(
            state,
            input_now,
            source.a,
            source.b,
            source.pi,
            source.qi,
            source.branch,
            source.control_branch,
            source.gamma,
        ) {
            return false;
        }
    }
    true
}

#[allow(clippy::too_many_arguments)]
pub fn newton_step(
    problem: &Problem,
    caches: &mut [DeviceCache],
    seed: &[f64],
    previous_state: &[f64],
    input_now: &[f64],
    input_previous: &[f64],
    h: f64,
    previous2_state: &[f64],
    input_previous2: &[f64],
    h_previous: f64,
    options: NewtonOptions,
    state: &mut [f64],
    system: &mut DenseSystem,
    device_stats: &mut DeviceSolveStats,
) -> NewtonResult {
    if seed.len() != problem.size || state.len() != problem.size || h <= 0.0 || !h.is_finite() {
        return NewtonResult {
            iterations: 0,
            converged: false,
            usable: false,
            residual_inf: f64::INFINITY,
            step_inf: f64::INFINITY,
        };
    }
    state.copy_from_slice(seed);
    let mut history = HistoryTerms::new(problem);
    if !fill_history_terms(
        problem,
        caches,
        previous_state,
        input_previous,
        options.cap_mode,
        &mut history,
    ) {
        return NewtonResult {
            iterations: 0,
            converged: false,
            usable: false,
            residual_inf: f64::INFINITY,
            step_inf: f64::INFINITY,
        };
    }
    let mut history2 = HistoryTerms::new(problem);
    if h_previous > 0.0 {
        let mut second_cache = vec![DeviceCache::default(); problem.devices.len()];
        if !fill_history_terms(
            problem,
            &mut second_cache,
            previous2_state,
            input_previous2,
            options.cap_mode,
            &mut history2,
        ) {
            return NewtonResult {
                iterations: 0,
                converged: false,
                usable: false,
                residual_inf: f64::INFINITY,
                step_inf: f64::INFINITY,
            };
        }
    } else {
        history2.clone_from(&history);
    }
    let bdf = if h_previous > 0.0 && h / h_previous <= 2.0 {
        let ratio = h / h_previous;
        [
            (1.0 + 2.0 * ratio) / (1.0 + ratio),
            -(1.0 + ratio),
            ratio * ratio / (1.0 + ratio),
        ]
    } else {
        [1.0, -1.0, 0.0]
    };
    let mut previous_step = f64::INFINITY;
    let mut last_residual = f64::INFINITY;
    let mut last_step = f64::INFINITY;
    for iteration in 0..options.max_iterations {
        let stamped = stamp_system(
            problem,
            caches,
            state,
            input_now,
            &history,
            &history2,
            StampOptions {
                h,
                gmin: options.gmin,
                hh: options.hh,
                cap_mode: options.cap_mode,
                bdf,
            },
            system,
            device_stats,
        );
        if !stamped {
            return NewtonResult {
                iterations: iteration + 1,
                converged: false,
                usable: false,
                residual_inf: last_residual,
                step_inf: last_step,
            };
        }
        last_residual = system
            .residual
            .iter()
            .fold(0.0f64, |current, value| current.max(value.abs()));
        if options.fallback_accept && last_residual < options.fallback_tolerance {
            return NewtonResult {
                iterations: iteration + 1,
                converged: true,
                usable: true,
                residual_inf: last_residual,
                step_inf: last_step,
            };
        }
        if !solve_dense_neg_rhs_in_place(&mut system.jacobian, &mut system.residual) {
            return NewtonResult {
                iterations: iteration + 1,
                converged: false,
                usable: true,
                residual_inf: last_residual,
                step_inf: last_step,
            };
        }
        last_step = system
            .residual
            .iter()
            .fold(0.0f64, |current, value| current.max(value.abs()));
        let mut applied_step = last_step;
        if applied_step > options.step_limit {
            let scale = options.step_limit / applied_step;
            for value in &mut system.residual {
                *value *= scale;
            }
            applied_step = options.step_limit;
        }
        for (value, delta) in state.iter_mut().zip(&system.residual) {
            *value += delta;
            if options.clip_lo <= options.clip_hi {
                *value = value.clamp(options.clip_lo, options.clip_hi);
            }
        }
        if applied_step < options.voltage_tolerance {
            if options.fallback_accept {
                if last_residual < options.fallback_tolerance.max(1e-6) {
                    return NewtonResult {
                        iterations: iteration + 1,
                        converged: true,
                        usable: true,
                        residual_inf: last_residual,
                        step_inf: applied_step,
                    };
                }
                previous_step = applied_step;
                continue;
            }
            return NewtonResult {
                iterations: iteration + 1,
                converged: true,
                usable: true,
                residual_inf: last_residual,
                step_inf: applied_step,
            };
        }
        if iteration >= 4 && applied_step >= previous_step && applied_step < 1e-5 {
            if options.fallback_accept {
                if last_residual < options.fallback_tolerance.max(1e-6) {
                    return NewtonResult {
                        iterations: iteration + 1,
                        converged: true,
                        usable: true,
                        residual_inf: last_residual,
                        step_inf: applied_step,
                    };
                }
                previous_step = applied_step;
                continue;
            }
            return NewtonResult {
                iterations: iteration + 1,
                converged: true,
                usable: true,
                residual_inf: last_residual,
                step_inf: applied_step,
            };
        }
        previous_step = applied_step;
    }
    NewtonResult {
        iterations: options.max_iterations,
        converged: false,
        usable: true,
        residual_inf: last_residual,
        step_inf: last_step,
    }
}

pub const PROFILE_LEN: usize = 24;
const PROFILE_NEWTON_ITERS: usize = 0;
const PROFILE_PMOS_OP_SOLVES: usize = 1;
const PROFILE_PMOS_INTERNAL_NEWTON_ATTEMPTS: usize = 2;
const PROFILE_PMOS_INTERNAL_NEWTON_ITERS: usize = 3;
const PROFILE_INTERNAL_FD_JAC_FALLBACKS: usize = 4;
const PROFILE_TERMINAL_FD_JAC_FALLBACKS: usize = 5;
const PROFILE_EDGE_SUBSTEPS: usize = 6;
const PROFILE_FLAT_SUBSTEPS: usize = 7;
const PROFILE_EDGE_NEWTON_ITERS: usize = 8;
const PROFILE_FLAT_NEWTON_ITERS: usize = 9;
const PROFILE_FAILED_SUBSTEPS: usize = 10;
const PROFILE_INTERVALS: usize = 11;
const PROFILE_SUBSTEPS: usize = 12;
const PROFILE_FAILED_INTERVALS: usize = 13;
const PROFILE_FAILED_EDGE_INTERVALS: usize = 14;
const PROFILE_FAILED_FLAT_INTERVALS: usize = 15;
const PROFILE_FAILED_LAST_RESIDUAL_INF: usize = 16;
const PROFILE_FAILED_MAX_RESIDUAL_INF: usize = 17;
const PROFILE_FAILED_LAST_STEP_INF: usize = 18;
const PROFILE_FAILED_MAX_STEP_INF: usize = 19;
const PROFILE_FAILED_STAMP_OR_PREV_COUNT: usize = 20;
const PROFILE_FAILED_LINEAR_SOLVE_COUNT: usize = 21;
const PROFILE_FAILED_MAXIT_COUNT: usize = 22;

#[derive(Clone, Copy, Debug)]
pub struct FixedGridOptions {
    pub newton: NewtonOptions,
    pub gear2: bool,
    pub max_step: f64,
    pub flat_max_step: f64,
    pub max_retry_subdivisions: usize,
    pub profile: bool,
}

#[derive(Clone, Debug)]
pub struct FixedGridResult {
    pub completed: bool,
    pub states: Vec<Vec<f64>>,
    pub substeps: usize,
    pub failed_index: Option<usize>,
    pub failed_intervals: Vec<usize>,
    pub profile: [f64; PROFILE_LEN],
}

pub fn validate_fixed_grid_input(
    problem: &Problem,
    initial: &[f64],
    times: &[f64],
    inputs: Waveforms<'_>,
    edge_mask: &[bool],
) -> Result<(), CoreError> {
    problem.try_validate()?;
    let valid = !times.is_empty()
        && initial.len() == problem.size
        && inputs.columns() == times.len()
        && (edge_mask.is_empty() || edge_mask.len() == times.len())
        && initial.iter().all(|value| value.is_finite())
        && times.iter().all(|value| value.is_finite())
        && times.windows(2).all(|window| window[1] > window[0]);
    valid.then_some(()).ok_or(CoreError::InvalidInput {
        analysis: "fixed-grid transient",
        detail: "state, time, input, or edge-mask dimensions are invalid",
    })
}

fn interpolated_inputs(start: &[f64], end: &[f64], fraction: f64) -> Vec<f64> {
    start
        .iter()
        .zip(end)
        .map(|(a, b)| a + (b - a) * fraction)
        .collect()
}

fn record_newton_profile(
    profile: &mut [f64; PROFILE_LEN],
    result: NewtonResult,
    device_before: DeviceSolveStats,
    device_after: DeviceSolveStats,
    edge: bool,
) {
    profile[PROFILE_NEWTON_ITERS] += result.iterations as f64;
    profile[PROFILE_PMOS_OP_SOLVES] += (device_after.solves - device_before.solves) as f64;
    profile[PROFILE_PMOS_INTERNAL_NEWTON_ATTEMPTS] +=
        (device_after.attempts - device_before.attempts) as f64;
    profile[PROFILE_PMOS_INTERNAL_NEWTON_ITERS] +=
        (device_after.iterations - device_before.iterations) as f64;
    profile[PROFILE_INTERNAL_FD_JAC_FALLBACKS] +=
        (device_after.fd_fallbacks - device_before.fd_fallbacks) as f64;
    profile[PROFILE_TERMINAL_FD_JAC_FALLBACKS] +=
        (device_after.terminal_fd_fallbacks - device_before.terminal_fd_fallbacks) as f64;
    if result.converged {
        let substep_slot = if edge {
            PROFILE_EDGE_SUBSTEPS
        } else {
            PROFILE_FLAT_SUBSTEPS
        };
        let iteration_slot = if edge {
            PROFILE_EDGE_NEWTON_ITERS
        } else {
            PROFILE_FLAT_NEWTON_ITERS
        };
        profile[substep_slot] += 1.0;
        profile[iteration_slot] += result.iterations as f64;
    } else {
        profile[PROFILE_FAILED_SUBSTEPS] += 1.0;
        profile[PROFILE_FAILED_LAST_RESIDUAL_INF] = result.residual_inf;
        profile[PROFILE_FAILED_MAX_RESIDUAL_INF] =
            profile[PROFILE_FAILED_MAX_RESIDUAL_INF].max(result.residual_inf);
        profile[PROFILE_FAILED_LAST_STEP_INF] = result.step_inf;
        profile[PROFILE_FAILED_MAX_STEP_INF] =
            profile[PROFILE_FAILED_MAX_STEP_INF].max(result.step_inf);
        if !result.usable {
            profile[PROFILE_FAILED_STAMP_OR_PREV_COUNT] += 1.0;
        } else if result.iterations >= 1 && result.step_inf.is_infinite() {
            profile[PROFILE_FAILED_LINEAR_SOLVE_COUNT] += 1.0;
        } else {
            profile[PROFILE_FAILED_MAXIT_COUNT] += 1.0;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn fixed_substep(
    problem: &Problem,
    caches: &mut [DeviceCache],
    seed: &[f64],
    previous: &[f64],
    previous2: &[f64],
    input_now: &[f64],
    input_previous: &[f64],
    input_previous2: &[f64],
    h: f64,
    h_previous: f64,
    options: NewtonOptions,
    system: &mut DenseSystem,
    device_stats: &mut DeviceSolveStats,
) -> (NewtonResult, Vec<f64>) {
    let mut state = vec![0.0; problem.size];
    let result = newton_step(
        problem,
        caches,
        seed,
        previous,
        input_now,
        input_previous,
        h,
        previous2,
        input_previous2,
        h_previous,
        options,
        &mut state,
        system,
        device_stats,
    );
    (result, state)
}

/// Solve a user-supplied fixed output grid using BE or variable-step BDF2.
///
/// Internal ``max_step`` slicing and failed-step binary subdivision preserve
/// the accepted-state/history order of the existing Numba implementation.
pub fn solve_fixed_grid(
    problem: &Problem,
    initial: &[f64],
    times: &[f64],
    inputs: Waveforms<'_>,
    edge_mask: &[bool],
    options: FixedGridOptions,
) -> FixedGridResult {
    let count = times.len();
    let mut profile = [0.0; PROFILE_LEN];
    if validate_fixed_grid_input(problem, initial, times, inputs, edge_mask).is_err() {
        return FixedGridResult {
            completed: false,
            states: Vec::new(),
            substeps: 0,
            failed_index: Some(0),
            failed_intervals: Vec::new(),
            profile,
        };
    }
    let input_at = |index: usize| inputs.sample(index).unwrap_or_default();
    let mut states = vec![vec![0.0; problem.size]; count];
    states[0].copy_from_slice(initial);
    let mut current = initial.to_vec();
    let mut previous2 = initial.to_vec();
    let mut input_current = input_at(0);
    let mut input_previous2 = input_current.clone();
    let mut h_previous = 0.0;
    let mut caches = vec![DeviceCache::default(); problem.devices.len()];
    let mut system = DenseSystem::new(problem.size);
    let mut device_stats = DeviceSolveStats::default();
    let mut substeps = 0usize;
    let mut failed_intervals = Vec::new();

    for interval in 1..count {
        if options.gear2 {
            current.clone_from(&states[interval - 1]);
            input_current = input_at(interval - 1);
            if interval >= 2 {
                previous2.clone_from(&states[interval - 2]);
                input_previous2 = input_at(interval - 2);
                h_previous = times[interval - 1] - times[interval - 2];
            } else {
                previous2.clone_from(&current);
                input_previous2.clone_from(&input_current);
                h_previous = 0.0;
            }
        }
        let interval_h = times[interval] - times[interval - 1];
        if interval_h <= 0.0 || !interval_h.is_finite() {
            return FixedGridResult {
                completed: false,
                states,
                substeps,
                failed_index: Some(interval),
                failed_intervals,
                profile,
            };
        }
        let edge = edge_mask.len() == count && (edge_mask[interval - 1] || edge_mask[interval]);
        let local_max = if !edge && options.flat_max_step > 0.0 {
            options.flat_max_step
        } else {
            options.max_step
        };
        let pieces = if local_max > 0.0 {
            (interval_h / local_max).ceil().max(1.0) as usize
        } else {
            1
        };
        let piece_h = interval_h / pieces as f64;
        let input_start = input_at(interval - 1);
        let input_end = input_at(interval);
        let substeps_before_interval = substeps;
        let mut interval_failed = false;

        for piece in 0..pieces {
            let piece_input_start = input_current.clone();
            let piece_input_end =
                interpolated_inputs(&input_start, &input_end, (piece + 1) as f64 / pieces as f64);
            let device_before = device_stats;
            let (result, candidate) = fixed_substep(
                problem,
                &mut caches,
                &current,
                &current,
                &previous2,
                &piece_input_end,
                &input_current,
                &input_previous2,
                piece_h,
                if options.gear2 { h_previous } else { 0.0 },
                options.newton,
                &mut system,
                &mut device_stats,
            );
            if options.profile {
                record_newton_profile(&mut profile, result, device_before, device_stats, edge);
            }
            if result.converged {
                previous2.clone_from(&current);
                current = candidate;
                input_previous2.clone_from(&input_current);
                input_current = piece_input_end;
                h_previous = piece_h;
                substeps += 1;
                continue;
            }

            let retry_count = 1usize << options.max_retry_subdivisions;
            if retry_count <= 1 {
                if options.newton.fallback_accept {
                    return FixedGridResult {
                        completed: false,
                        states,
                        substeps: substeps_before_interval,
                        failed_index: Some(interval),
                        failed_intervals,
                        profile,
                    };
                }
                interval_failed = true;
                break;
            }
            let retry_h = piece_h / retry_count as f64;
            let mut retry_ok = true;
            for retry in 0..retry_count {
                let retry_input = interpolated_inputs(
                    &piece_input_start,
                    &piece_input_end,
                    (retry + 1) as f64 / retry_count as f64,
                );
                let device_before = device_stats;
                let (retry_result, retry_candidate) = fixed_substep(
                    problem,
                    &mut caches,
                    &current,
                    &current,
                    &previous2,
                    &retry_input,
                    &input_current,
                    &input_previous2,
                    retry_h,
                    if options.gear2 { h_previous } else { 0.0 },
                    options.newton,
                    &mut system,
                    &mut device_stats,
                );
                if options.profile {
                    record_newton_profile(
                        &mut profile,
                        retry_result,
                        device_before,
                        device_stats,
                        edge,
                    );
                }
                if !retry_result.converged {
                    retry_ok = false;
                    break;
                }
                previous2.clone_from(&current);
                current = retry_candidate;
                input_previous2.clone_from(&input_current);
                input_current = retry_input;
                h_previous = retry_h;
                substeps += 1;
            }
            if !retry_ok {
                if options.newton.fallback_accept {
                    return FixedGridResult {
                        completed: false,
                        states,
                        substeps: substeps_before_interval,
                        failed_index: Some(interval),
                        failed_intervals,
                        profile,
                    };
                }
                interval_failed = true;
                break;
            }
        }
        if interval_failed {
            failed_intervals.push(interval);
            if options.profile {
                profile[PROFILE_FAILED_INTERVALS] += 1.0;
                let slot = if edge {
                    PROFILE_FAILED_EDGE_INTERVALS
                } else {
                    PROFILE_FAILED_FLAT_INTERVALS
                };
                profile[slot] += 1.0;
            }
        }
        states[interval].copy_from_slice(&current);
    }
    if options.profile {
        profile[PROFILE_INTERVALS] = count.saturating_sub(1) as f64;
        profile[PROFILE_SUBSTEPS] = substeps as f64;
    }
    FixedGridResult {
        completed: true,
        states,
        substeps,
        failed_index: None,
        failed_intervals,
        profile,
    }
}

const ADAPTIVE_ACCEPT_WRMS: f64 = 1.0;
const ADAPTIVE_DONE_ABS: f64 = 1e-18;
const ADAPTIVE_DONE_REL: f64 = 1e-13;
const ADAPTIVE_ERR_FLOOR: f64 = 1e-12;
const ADAPTIVE_GROWTH_MAX: f64 = 2.0;
const ADAPTIVE_GROWTH_MIN: f64 = 0.2;
const ADAPTIVE_INITIAL_MIN_DENOM: usize = 16;
const ADAPTIVE_INPUT_SLOPE_BREAK_FRACTION: f64 = 0.1;
const ADAPTIVE_LTE_DIVISOR: f64 = 3.0;
const ADAPTIVE_MIN_H_ABS: f64 = 1e-18;
const ADAPTIVE_MIN_H_REL: f64 = 1e-15;
const ADAPTIVE_SAFETY: f64 = 0.9;
const ADAPTIVE_SCALE_FLOOR: f64 = 1e-30;
const ADAPTIVE_STEP_ORDER: f64 = 3.0;

#[derive(Clone, Copy, Debug)]
pub struct AdaptiveOptions {
    pub newton: NewtonOptions,
    pub max_step: f64,
    pub reltol: f64,
    pub voltage_abstol: f64,
    pub current_abstol: f64,
    pub max_steps: usize,
    pub initial_step: f64,
    pub profile: bool,
}

#[derive(Clone, Debug)]
pub struct AdaptiveResult {
    pub completed: bool,
    pub times: Vec<f64>,
    pub states: Vec<Vec<f64>>,
    pub inputs: Vec<Vec<f64>>,
    pub substeps: usize,
    pub rejected: usize,
    pub profile: [f64; PROFILE_LEN],
}

pub fn validate_adaptive_input(
    problem: &Problem,
    initial: &[f64],
    source_times: &[f64],
    source_inputs: Waveforms<'_>,
) -> Result<(), CoreError> {
    problem.try_validate()?;
    let valid = source_times.len() >= 2
        && initial.len() == problem.size
        && source_inputs.columns() == source_times.len()
        && initial.iter().all(|value| value.is_finite())
        && source_times.iter().all(|value| value.is_finite())
        && source_times.windows(2).all(|window| window[1] > window[0]);
    valid.then_some(()).ok_or(CoreError::InvalidInput {
        analysis: "adaptive transient",
        detail: "state, time, or input dimensions are invalid",
    })
}

fn inputs_at_time(times: &[f64], inputs: Waveforms<'_>, time: f64) -> Vec<f64> {
    if time <= times[0] {
        return inputs.sample(0).unwrap_or_default();
    }
    let last = times.len() - 1;
    if time >= times[last] {
        return inputs.sample(last).unwrap_or_default();
    }
    let mut interval = 0usize;
    for position in 0..last {
        if times[position] <= time && time <= times[position + 1] {
            interval = position;
            break;
        }
    }
    let fraction = (time - times[interval]) / (times[interval + 1] - times[interval]);
    (0..inputs.rows())
        .map(|row| {
            let values = inputs.row(row).unwrap_or_default();
            values[interval] + (values[interval + 1] - values[interval]) * fraction
        })
        .collect()
}

fn adaptive_error(
    half: &[f64],
    full: &[f64],
    node_count: usize,
    reltol: f64,
    voltage_abstol: f64,
    current_abstol: f64,
) -> f64 {
    let mut sum = 0.0;
    for (index, (half_value, full_value)) in half.iter().zip(full).enumerate() {
        let abstol = if index < node_count {
            voltage_abstol
        } else {
            current_abstol
        };
        let scale =
            (reltol * half_value.abs().max(full_value.abs()) + abstol).max(ADAPTIVE_SCALE_FLOOR);
        let normalized = ((half_value - full_value) / ADAPTIVE_LTE_DIVISOR) / scale;
        sum += normalized * normalized;
    }
    (sum / half.len() as f64).sqrt()
}

fn adaptive_next_step(step: f64, error: f64) -> f64 {
    let factor = if error <= 0.0 {
        ADAPTIVE_GROWTH_MAX
    } else if !error.is_finite() {
        ADAPTIVE_GROWTH_MIN
    } else {
        (ADAPTIVE_SAFETY * error.powf(-1.0 / ADAPTIVE_STEP_ORDER))
            .clamp(ADAPTIVE_GROWTH_MIN, ADAPTIVE_GROWTH_MAX)
    };
    step * factor
}

fn adaptive_critical_times(times: &[f64], inputs: Waveforms<'_>) -> Vec<f64> {
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
        if jump > ADAPTIVE_INPUT_SLOPE_BREAK_FRACTION * global_slope {
            critical.push(times[position]);
        }
    }
    critical
}

fn add_adaptive_profile(
    profile: &mut [f64; PROFILE_LEN],
    iterations: usize,
    before: DeviceSolveStats,
    after: DeviceSolveStats,
) {
    profile[PROFILE_NEWTON_ITERS] += iterations as f64;
    profile[PROFILE_PMOS_OP_SOLVES] += (after.solves - before.solves) as f64;
    profile[PROFILE_PMOS_INTERNAL_NEWTON_ATTEMPTS] += (after.attempts - before.attempts) as f64;
    profile[PROFILE_PMOS_INTERNAL_NEWTON_ITERS] += (after.iterations - before.iterations) as f64;
    profile[PROFILE_INTERNAL_FD_JAC_FALLBACKS] += (after.fd_fallbacks - before.fd_fallbacks) as f64;
    profile[PROFILE_TERMINAL_FD_JAC_FALLBACKS] +=
        (after.terminal_fd_fallbacks - before.terminal_fd_fallbacks) as f64;
}

/// Adaptive variable-step Gear2 using full-step versus two-half-step LTE.
pub fn solve_adaptive_gear2(
    problem: &Problem,
    initial: &[f64],
    source_times: &[f64],
    source_inputs: Waveforms<'_>,
    options: AdaptiveOptions,
) -> AdaptiveResult {
    let mut profile = [0.0; PROFILE_LEN];
    if validate_adaptive_input(problem, initial, source_times, source_inputs).is_err() {
        return AdaptiveResult {
            completed: false,
            times: Vec::new(),
            states: Vec::new(),
            inputs: Vec::new(),
            substeps: 0,
            rejected: 0,
            profile,
        };
    }
    let start = source_times[0];
    let end = source_times[source_times.len() - 1];
    let span = end - start;
    let max_step = if options.max_step > 0.0 {
        options.max_step.min(span)
    } else {
        span
    };
    let mut step = if options.initial_step > 0.0 {
        options.initial_step
    } else {
        let denominator = (source_times.len() - 1).max(ADAPTIVE_INITIAL_MIN_DENOM);
        let smallest_source_step = source_times
            .windows(2)
            .map(|window| window[1] - window[0])
            .fold(span, f64::min);
        (span / denominator as f64).min(smallest_source_step)
    };
    if step <= 0.0 || !step.is_finite() {
        step = span / 100.0;
    }
    step = step.min(max_step);
    let min_step = ADAPTIVE_MIN_H_ABS.max(span * ADAPTIVE_MIN_H_REL);
    let done_tolerance = ADAPTIVE_DONE_ABS.max(span * ADAPTIVE_DONE_REL);
    let critical_times = adaptive_critical_times(source_times, source_inputs);

    let mut times = Vec::with_capacity(options.max_steps + 1);
    let mut states = Vec::with_capacity(options.max_steps + 1);
    let mut input_history = Vec::with_capacity(options.max_steps + 1);
    let mut current = initial.to_vec();
    let mut previous2 = initial.to_vec();
    let mut input_current = inputs_at_time(source_times, source_inputs, start);
    let mut input_previous2 = input_current.clone();
    times.push(start);
    states.push(current.clone());
    input_history.push(input_current.clone());
    let mut current_time = start;
    let mut previous_step = -1.0;
    let mut accepted = 0usize;
    let mut substeps = 0usize;
    let mut rejected = 0usize;
    let mut caches = vec![DeviceCache::default(); problem.devices.len()];
    let mut system = DenseSystem::new(problem.size);
    let mut device_stats = DeviceSolveStats::default();

    while accepted < options.max_steps && current_time < end - done_tolerance {
        if previous_step > 0.0 {
            step = step.min(ADAPTIVE_GROWTH_MAX * previous_step);
        }
        step = step.min(max_step).min(end - current_time);
        for critical in &critical_times {
            if *critical > current_time + min_step {
                if *critical < current_time + step {
                    step = *critical - current_time;
                }
                break;
            }
        }
        if step <= min_step {
            break;
        }
        let input_now = inputs_at_time(source_times, source_inputs, current_time + step);
        let input_mid = inputs_at_time(source_times, source_inputs, current_time + 0.5 * step);
        let device_before = device_stats;
        let (full_result, full) = fixed_substep(
            problem,
            &mut caches,
            &current,
            &current,
            &previous2,
            &input_now,
            &input_current,
            &input_previous2,
            step,
            previous_step,
            options.newton,
            &mut system,
            &mut device_stats,
        );
        let (mid_result, mid) = fixed_substep(
            problem,
            &mut caches,
            &current,
            &current,
            &previous2,
            &input_mid,
            &input_current,
            &input_previous2,
            0.5 * step,
            previous_step,
            options.newton,
            &mut system,
            &mut device_stats,
        );
        let (half_result, half) = if mid_result.converged {
            fixed_substep(
                problem,
                &mut caches,
                &mid,
                &mid,
                &current,
                &input_now,
                &input_mid,
                &input_current,
                0.5 * step,
                0.5 * step,
                options.newton,
                &mut system,
                &mut device_stats,
            )
        } else {
            (
                NewtonResult {
                    iterations: 0,
                    converged: false,
                    usable: false,
                    residual_inf: f64::INFINITY,
                    step_inf: f64::INFINITY,
                },
                vec![0.0; problem.size],
            )
        };
        substeps += 3;
        if options.profile {
            add_adaptive_profile(
                &mut profile,
                full_result.iterations + mid_result.iterations + half_result.iterations,
                device_before,
                device_stats,
            );
        }
        let error = if full_result.converged && mid_result.converged && half_result.converged {
            adaptive_error(
                &half,
                &full,
                problem.node_count,
                options.reltol,
                options.voltage_abstol,
                options.current_abstol,
            )
        } else {
            f64::INFINITY
        };
        if error <= ADAPTIVE_ACCEPT_WRMS {
            current_time += step;
            accepted += 1;
            previous2.clone_from(&current);
            current = half;
            input_previous2.clone_from(&input_current);
            input_current = input_now;
            times.push(current_time);
            states.push(current.clone());
            input_history.push(input_current.clone());
            let critical_tolerance = min_step.max(done_tolerance);
            let hit_critical = critical_times
                .iter()
                .any(|critical| (*critical - current_time).abs() <= critical_tolerance);
            if hit_critical {
                previous2[..problem.node_count].copy_from_slice(&current[..problem.node_count]);
                input_previous2.clone_from(&input_current);
                previous_step = -1.0;
            } else {
                previous_step = step;
            }
            step = adaptive_next_step(step, error.max(ADAPTIVE_ERR_FLOOR));
        } else {
            rejected += 1;
            step = adaptive_next_step(step, error).max(min_step);
        }
    }
    AdaptiveResult {
        completed: current_time >= end - done_tolerance,
        times,
        states,
        inputs: input_history,
        substeps,
        rejected,
        profile,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rail(value: f64) -> Term {
        Term {
            kind: 2,
            reference: 0,
            value,
        }
    }

    fn solved(reference: usize) -> Term {
        Term {
            kind: 0,
            reference,
            value: 0.0,
        }
    }

    #[test]
    fn passive_rc_newton_step_matches_closed_form_be() {
        let problem = Problem {
            node_count: 1,
            size: 1,
            devices: Vec::new(),
            resistors: vec![Resistor {
                a: solved(0),
                b: rail(0.0),
                ai: Some(0),
                bi: None,
                conductance: 1e-3,
            }],
            capacitors: vec![Capacitor {
                a: solved(0),
                b: rail(0.0),
                ai: Some(0),
                bi: None,
                capacitance: 1e-6,
            }],
            current_sources: vec![CurrentSource {
                pi: None,
                qi: Some(0),
                value: 1e-3,
            }],
            dynamic_current_sources: Vec::new(),
            vccs: Vec::new(),
            voltage_sources: Vec::new(),
            vcvs: Vec::new(),
            cccs: Vec::new(),
            ccvs: Vec::new(),
        };
        let mut state = vec![0.0];
        let mut system = DenseSystem::new(1);
        let result = newton_step(
            &problem,
            &mut [],
            &[0.0],
            &[0.0],
            &[],
            &[],
            1e-3,
            &[0.0],
            &[],
            0.0,
            NewtonOptions {
                max_iterations: 4,
                step_limit: 5.0,
                voltage_tolerance: 1e-12,
                fallback_accept: false,
                fallback_tolerance: 1e-9,
                clip_lo: f64::INFINITY,
                clip_hi: f64::NEG_INFINITY,
                gmin: 0.0,
                hh: 1e-3,
                cap_mode: 0,
            },
            &mut state,
            &mut system,
            &mut DeviceSolveStats::default(),
        );
        assert!(result.converged);
        assert!((state[0] - 0.5).abs() < 1e-12);
    }
}
