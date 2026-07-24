//! Fixed-grid four-terminal compact-model transient orchestration.

use crate::transient::{HistoryTerms, Problem as CircuitProblem, Waveforms, fill_history_terms};
use crate::{
    CoreError,
    mna::{DenseSystem, Term, solve_dense_neg_rhs_in_place},
};

#[derive(Clone, Copy, Debug)]
pub struct Device {
    pub terms: [Term; 4],
    pub rows: [Option<usize>; 4],
    pub evaluator_index: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Evaluation {
    pub currents: [f64; 4],
    pub conductance: [f64; 16],
    pub charges: [f64; 4],
    pub capacitance: [f64; 16],
}

pub trait Evaluator {
    fn evaluate(&mut self, index: usize, terminals: [f64; 4]) -> Option<Evaluation>;

    fn evaluate_batch(
        &mut self,
        indices: &[usize],
        terminals: &[[f64; 4]],
        evaluations: &mut [Evaluation],
    ) -> BatchStatus {
        if indices.len() != terminals.len() || indices.len() != evaluations.len() {
            return BatchStatus::default();
        }
        let mut attempted = 0;
        for ((&index, &terminal), output) in indices.iter().zip(terminals).zip(evaluations) {
            attempted += 1;
            let Some(evaluation) = self.evaluate(index, terminal) else {
                return BatchStatus {
                    completed: false,
                    attempted,
                };
            };
            *output = evaluation;
        }
        BatchStatus {
            completed: true,
            attempted,
        }
    }

    /// DC-Newton variant used inside `solve_dc`, which consumes only the currents
    /// and conductance. A small-signal device backend may skip capacitance/charge
    /// extraction here (D6 acLoad-skip) — the returned `charges`/`capacitance` are
    /// then unspecified and must not be read. The default runs the full
    /// `evaluate`, so implementors that do not override this are unaffected.
    fn evaluate_dc(&mut self, index: usize, terminals: [f64; 4]) -> Option<Evaluation> {
        self.evaluate(index, terminals)
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct BatchStatus {
    pub completed: bool,
    pub attempted: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct Options {
    pub gear2: bool,
    pub max_iterations: usize,
    pub voltage_tolerance: f64,
    pub step_limit: f64,
    pub gmin: f64,
    pub record_device_history: bool,
    pub profile: bool,
}

#[derive(Clone, Debug, Default)]
pub struct Profile {
    pub newton_iterations: usize,
    pub bsim_evaluations: usize,
    pub bsim_batches: usize,
    pub gear2_predictor_steps: usize,
    pub lte_estimates: usize,
    pub lte_linear_solves: usize,
    pub lte_rejections: usize,
    pub newton_rejections: usize,
    pub failed_steps: Vec<usize>,
}

#[derive(Clone, Debug)]
pub struct Result {
    pub completed: bool,
    pub states: Vec<Vec<f64>>,
    /// Sample-major terminal histories with one `[d, g, s, b]` row per device.
    /// Empty when `Options::record_device_history` is false.
    pub device_currents: Vec<[f64; 4]>,
    pub device_charges: Vec<[f64; 4]>,
    pub failures: usize,
    pub first_failure: Option<usize>,
    pub profile: Profile,
}

#[derive(Clone, Copy, Debug)]
pub struct DcOptions {
    pub max_iterations: usize,
    pub voltage_tolerance: f64,
    pub step_limit: f64,
    pub gmin: f64,
}

#[derive(Clone, Debug)]
pub struct DcResult {
    pub converged: bool,
    pub state: Vec<f64>,
    pub iterations: usize,
    pub residual_inf: f64,
}

pub fn validate_fixed_grid_input(
    circuit: &CircuitProblem,
    devices: &[Device],
    initial: &[f64],
    times: &[f64],
    inputs: Waveforms<'_>,
) -> std::result::Result<(), CoreError> {
    let topology_valid = circuit.validate()
        && circuit.devices.is_empty()
        && devices.iter().all(|device| {
            device
                .terms
                .iter()
                .all(|term| term.is_valid(circuit.node_count, true))
                && device
                    .rows
                    .iter()
                    .flatten()
                    .all(|row| *row < circuit.node_count)
        });
    if !topology_valid {
        return Err(CoreError::InvalidTopology {
            analysis: "BSIM4 transient topology",
        });
    }
    let input_valid = times.len() >= 2
        && initial.len() == circuit.size
        && inputs.columns() == times.len()
        && initial.iter().all(|value| value.is_finite())
        && times.iter().all(|value| value.is_finite())
        && times.windows(2).all(|window| window[1] > window[0]);
    input_valid.then_some(()).ok_or(CoreError::InvalidInput {
        analysis: "BSIM4 transient",
        detail: "state, time, or input dimensions are invalid",
    })
}

#[allow(clippy::too_many_arguments)]
fn evaluate_devices<E: Evaluator>(
    evaluator: &mut E,
    devices: &[Device],
    evaluator_indices: &[usize],
    state: &[f64],
    inputs: &[f64],
    terminals: &mut [[f64; 4]],
    evaluations: &mut [Evaluation],
    profile: &mut Profile,
    profile_enabled: bool,
) -> bool {
    if devices.len() != evaluator_indices.len()
        || devices.len() != terminals.len()
        || devices.len() != evaluations.len()
    {
        return false;
    }
    for (device, terminal_values) in devices.iter().zip(terminals.iter_mut()) {
        for (position, term) in device.terms.into_iter().enumerate() {
            let Some(value) = term.resolve(state, inputs) else {
                return false;
            };
            terminal_values[position] = value;
        }
    }
    let status = evaluator.evaluate_batch(evaluator_indices, terminals, evaluations);
    if profile_enabled {
        profile.bsim_batches += 1;
        profile.bsim_evaluations += status.attempted;
    }
    status.completed
}

/// `evaluate_device` for the DC-Newton path: routes through `Evaluator::evaluate_dc`
/// so a backend can skip capacitance/charge extraction (D6 acLoad-skip).
fn evaluate_device_dc<E: Evaluator>(
    evaluator: &mut E,
    device: Device,
    state: &[f64],
    inputs: &[f64],
) -> Option<Evaluation> {
    let mut terminals = [0.0; 4];
    for (position, term) in device.terms.into_iter().enumerate() {
        terminals[position] = term.resolve(state, inputs)?;
    }
    evaluator.evaluate_dc(device.evaluator_index, terminals)
}

fn history_for(circuit: &CircuitProblem, state: &[f64], inputs: &[f64]) -> Option<HistoryTerms> {
    let mut history = HistoryTerms::new(circuit);
    if fill_history_terms(circuit, &mut [], state, inputs, 0, &mut history) {
        Some(history)
    } else {
        None
    }
}

#[allow(clippy::too_many_arguments)]
fn stamp_linear_elements(
    circuit: &CircuitProblem,
    state: &[f64],
    input_now: &[f64],
    coefficients: [f64; 3],
    history: &HistoryTerms,
    history2: &HistoryTerms,
    gmin: f64,
    system: &mut DenseSystem,
) -> bool {
    for resistor in &circuit.resistors {
        let Some(va) = resistor.a.resolve(state, input_now) else {
            return false;
        };
        let Some(vb) = resistor.b.resolve(state, input_now) else {
            return false;
        };
        let current = resistor.conductance * (va - vb);
        if let Some(row) = resistor.ai {
            system.residual[row] += current;
            system.add_jacobian(row, row, resistor.conductance);
            if let Some(column) = resistor.bi {
                system.add_jacobian(row, column, -resistor.conductance);
            }
        }
        if let Some(row) = resistor.bi {
            system.residual[row] -= current;
            system.add_jacobian(row, row, resistor.conductance);
            if let Some(column) = resistor.ai {
                system.add_jacobian(row, column, -resistor.conductance);
            }
        }
    }

    for (position, capacitor) in circuit.capacitors.iter().enumerate() {
        let Some(va) = capacitor.a.resolve(state, input_now) else {
            return false;
        };
        let Some(vb) = capacitor.b.resolve(state, input_now) else {
            return false;
        };
        let current = capacitor.capacitance
            * (coefficients[0] * (va - vb)
                + coefficients[1] * history.capacitor_dv[position]
                + coefficients[2] * history2.capacitor_dv[position]);
        let admittance = capacitor.capacitance * coefficients[0];
        if let Some(row) = capacitor.ai {
            system.residual[row] += current;
            system.add_jacobian(row, row, admittance);
            if let Some(column) = capacitor.bi {
                system.add_jacobian(row, column, -admittance);
            }
        }
        if let Some(row) = capacitor.bi {
            system.residual[row] -= current;
            system.add_jacobian(row, row, admittance);
            if let Some(column) = capacitor.ai {
                system.add_jacobian(row, column, -admittance);
            }
        }
    }

    for source in &circuit.current_sources {
        if let Some(row) = source.pi {
            system.residual[row] += source.value;
        }
        if let Some(row) = source.qi {
            system.residual[row] -= source.value;
        }
    }
    for source in &circuit.dynamic_current_sources {
        let Some(current) = input_now.get(source.input_index) else {
            return false;
        };
        if let Some(row) = source.pi {
            system.residual[row] += current;
        }
        if let Some(row) = source.qi {
            system.residual[row] -= current;
        }
    }

    for source in &circuit.voltage_sources {
        let branch_current = state[source.branch];
        let emf = match source.input_index {
            Some(index) => match input_now.get(index) {
                Some(value) => *value,
                None => return false,
            },
            None => source.emf,
        };
        let mut vp = 0.0;
        let mut vq = 0.0;
        if let Some(row) = source.pi {
            vp = state[row];
            system.residual[row] += branch_current;
            system.add_jacobian(row, source.branch, 1.0);
        }
        if let Some(row) = source.qi {
            vq = state[row];
            system.residual[row] -= branch_current;
            system.add_jacobian(row, source.branch, -1.0);
        }
        system.residual[source.branch] += vp - vq - emf;
        if let Some(column) = source.pi {
            system.add_jacobian(source.branch, column, 1.0);
        }
        if let Some(column) = source.qi {
            system.add_jacobian(source.branch, column, -1.0);
        }
    }

    for source in &circuit.vccs {
        let Some(vcp) = source.cp.resolve(state, input_now) else {
            return false;
        };
        let Some(vcn) = source.cn.resolve(state, input_now) else {
            return false;
        };
        let current = source.gm * (vcp - vcn);
        if let Some(row) = source.pi {
            system.residual[row] -= current;
            if let Some(column) = source.cpi {
                system.add_jacobian(row, column, -source.gm);
            }
            if let Some(column) = source.cni {
                system.add_jacobian(row, column, source.gm);
            }
        }
        if let Some(row) = source.qi {
            system.residual[row] += current;
            if let Some(column) = source.cpi {
                system.add_jacobian(row, column, source.gm);
            }
            if let Some(column) = source.cni {
                system.add_jacobian(row, column, -source.gm);
            }
        }
    }

    for source in &circuit.vcvs {
        let branch_current = state[source.branch];
        if let Some(row) = source.pi {
            system.residual[row] += branch_current;
            system.add_jacobian(row, source.branch, 1.0);
        }
        if let Some(row) = source.qi {
            system.residual[row] -= branch_current;
            system.add_jacobian(row, source.branch, -1.0);
        }
        let Some(vp) = source.a.resolve(state, input_now) else {
            return false;
        };
        let Some(vq) = source.b.resolve(state, input_now) else {
            return false;
        };
        let Some(vcp) = source.cp.resolve(state, input_now) else {
            return false;
        };
        let Some(vcn) = source.cn.resolve(state, input_now) else {
            return false;
        };
        system.residual[source.branch] += vp - vq - source.mu * (vcp - vcn);
        if let Some(column) = source.pi {
            system.add_jacobian(source.branch, column, 1.0);
        }
        if let Some(column) = source.qi {
            system.add_jacobian(source.branch, column, -1.0);
        }
        if let Some(column) = source.cpi {
            system.add_jacobian(source.branch, column, -source.mu);
        }
        if let Some(column) = source.cni {
            system.add_jacobian(source.branch, column, source.mu);
        }
    }

    for source in &circuit.cccs {
        let current = source.beta * state[source.control_branch];
        if let Some(row) = source.pi {
            system.residual[row] -= current;
            system.add_jacobian(row, source.control_branch, -source.beta);
        }
        if let Some(row) = source.qi {
            system.residual[row] += current;
            system.add_jacobian(row, source.control_branch, source.beta);
        }
    }

    for source in &circuit.ccvs {
        let branch_current = state[source.branch];
        if let Some(row) = source.pi {
            system.residual[row] += branch_current;
            system.add_jacobian(row, source.branch, 1.0);
        }
        if let Some(row) = source.qi {
            system.residual[row] -= branch_current;
            system.add_jacobian(row, source.branch, -1.0);
        }
        let Some(vp) = source.a.resolve(state, input_now) else {
            return false;
        };
        let Some(vq) = source.b.resolve(state, input_now) else {
            return false;
        };
        system.residual[source.branch] += vp - vq - source.gamma * state[source.control_branch];
        if let Some(column) = source.pi {
            system.add_jacobian(source.branch, column, 1.0);
        }
        if let Some(column) = source.qi {
            system.add_jacobian(source.branch, column, -1.0);
        }
        system.add_jacobian(source.branch, source.control_branch, -source.gamma);
    }

    for (row, voltage) in state.iter().copied().take(circuit.node_count).enumerate() {
        system.residual[row] += gmin * voltage;
        system.add_jacobian(row, row, gmin);
    }
    true
}

pub fn solve_dc<E: Evaluator>(
    circuit: &CircuitProblem,
    devices: &[Device],
    evaluator: &mut E,
    initial: &[f64],
    inputs: &[f64],
    options: DcOptions,
) -> DcResult {
    let topology_valid = circuit.validate()
        && circuit.devices.is_empty()
        && initial.len() == circuit.size
        && initial.iter().all(|value| value.is_finite())
        && devices.iter().all(|device| {
            device
                .terms
                .iter()
                .all(|term| term.is_valid(circuit.node_count, true))
                && device
                    .rows
                    .iter()
                    .flatten()
                    .all(|row| *row < circuit.node_count)
        });
    if !topology_valid {
        return DcResult {
            converged: false,
            state: initial.to_vec(),
            iterations: 0,
            residual_inf: f64::INFINITY,
        };
    }

    let mut state = initial.to_vec();
    let mut system = DenseSystem::new(circuit.size);
    let history = HistoryTerms::new(circuit);
    let mut residual_inf = f64::INFINITY;
    for iteration in 0..options.max_iterations {
        system.residual.fill(0.0);
        system.jacobian.fill(0.0);
        let mut evaluation_failed = false;
        for device in devices.iter().copied() {
            let Some(evaluation) = evaluate_device_dc(evaluator, device, &state, inputs) else {
                evaluation_failed = true;
                break;
            };
            for terminal_row in 0..4 {
                let Some(row) = device.rows[terminal_row] else {
                    continue;
                };
                system.residual[row] += evaluation.currents[terminal_row];
                for terminal_col in 0..4 {
                    let Some(column) = device.rows[terminal_col] else {
                        continue;
                    };
                    system.add_jacobian(
                        row,
                        column,
                        evaluation.conductance[terminal_row * 4 + terminal_col],
                    );
                }
            }
        }
        let stamped = !evaluation_failed
            && stamp_linear_elements(
                circuit,
                &state,
                inputs,
                [0.0; 3],
                &history,
                &history,
                options.gmin,
                &mut system,
            );
        if !stamped {
            return DcResult {
                converged: false,
                state,
                iterations: iteration + 1,
                residual_inf,
            };
        }
        residual_inf = system
            .residual
            .iter()
            .fold(0.0f64, |peak, value| peak.max(value.abs()));
        if residual_inf <= options.voltage_tolerance {
            return DcResult {
                converged: true,
                state,
                iterations: iteration + 1,
                residual_inf,
            };
        }
        if !solve_dense_neg_rhs_in_place(&mut system.jacobian, &mut system.residual) {
            return DcResult {
                converged: false,
                state,
                iterations: iteration + 1,
                residual_inf,
            };
        }
        let peak_step = system
            .residual
            .iter()
            .fold(0.0f64, |peak, value| peak.max(value.abs()));
        if !peak_step.is_finite() {
            return DcResult {
                converged: false,
                state,
                iterations: iteration + 1,
                residual_inf,
            };
        }
        if peak_step > options.step_limit {
            let scale = options.step_limit / peak_step;
            for value in &mut system.residual {
                *value *= scale;
            }
        }
        for (value, delta) in state.iter_mut().zip(&system.residual) {
            *value += delta;
        }
    }
    DcResult {
        converged: false,
        state,
        iterations: options.max_iterations,
        residual_inf,
    }
}

const GEAR2_PREDICTOR_MAX_STEP_RATIO: f64 = 4.0;
const GEAR2_PREDICTOR_INPUT_CURVATURE: f64 = 0.25;

fn gear2_predictor_enabled() -> bool {
    std::env::var("CIRCUITOPT_BSIM_GEAR2_PREDICTOR")
        .map(|value| {
            !matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "0" | "false" | "off"
            )
        })
        .unwrap_or(true)
}

/// Seed one variable-step Gear2 solve by linearly extrapolating accepted state.
///
/// A visible input-slope discontinuity disables prediction for the edge and the
/// following sample. This avoids carrying a pre-edge state slope across clock
/// or DAC-code transitions while retaining the predictor on smooth waveforms.
#[allow(clippy::too_many_arguments)]
fn predict_gear2_state(
    state: &mut [f64],
    previous: &[f64],
    previous2: &[f64],
    input_now: &[f64],
    input_previous: &[f64],
    input_previous2: &[f64],
    h: f64,
    h_previous: f64,
) -> bool {
    if state.len() != previous.len()
        || state.len() != previous2.len()
        || input_now.len() != input_previous.len()
        || input_now.len() != input_previous2.len()
        || !h.is_finite()
        || !h_previous.is_finite()
        || h <= 0.0
        || h_previous <= 0.0
    {
        return false;
    }
    let ratio = h / h_previous;
    if !ratio.is_finite() || ratio > GEAR2_PREDICTOR_MAX_STEP_RATIO {
        return false;
    }
    for ((&now, &previous_input), &previous2_input) in
        input_now.iter().zip(input_previous).zip(input_previous2)
    {
        let actual_delta = now - previous_input;
        let predicted_delta = ratio * (previous_input - previous2_input);
        let curvature = (actual_delta - predicted_delta).abs();
        let activity = actual_delta.abs() + predicted_delta.abs();
        if !curvature.is_finite()
            || curvature > GEAR2_PREDICTOR_INPUT_CURVATURE * activity.max(1e-12)
        {
            return false;
        }
    }
    for ((output, &previous_value), &previous2_value) in
        state.iter_mut().zip(previous).zip(previous2)
    {
        *output = previous_value + ratio * (previous_value - previous2_value);
        if !output.is_finite() {
            state.copy_from_slice(previous);
            return false;
        }
    }
    true
}

const ADAPTIVE_ACCEPT_WRMS: f64 = 1.0;
const ADAPTIVE_DONE_ABS: f64 = 1e-18;
const ADAPTIVE_DONE_REL: f64 = 1e-13;
const ADAPTIVE_ERR_FLOOR: f64 = 1e-12;
const ADAPTIVE_GROWTH_MAX: f64 = 2.0;
const ADAPTIVE_GROWTH_MIN: f64 = 0.2;
const ADAPTIVE_INITIAL_MIN_DENOM: usize = 16;
const ADAPTIVE_INPUT_SLOPE_BREAK_FRACTION: f64 = 0.1;
const ADAPTIVE_MIN_H_ABS: f64 = 1e-18;
const ADAPTIVE_MIN_H_REL: f64 = 1e-15;
const ADAPTIVE_NEWTON_REJECT_FACTOR: f64 = 0.25;
const ADAPTIVE_PI_INTEGRAL: f64 = 0.4 / ADAPTIVE_STEP_ORDER;
const ADAPTIVE_PI_PROPORTIONAL: f64 = 0.7 / ADAPTIVE_STEP_ORDER;
const ADAPTIVE_POST_REJECT_GROWTH_MAX: f64 = 1.0;
const ADAPTIVE_REJECT_GROWTH_MAX: f64 = 0.8;
const ADAPTIVE_SAFETY: f64 = 0.9;
const ADAPTIVE_SCALE_FLOOR: f64 = 1e-30;
const ADAPTIVE_STARTUP_FRACTION: f64 = 0.25;
const ADAPTIVE_STEP_ORDER: f64 = 3.0;

#[derive(Clone, Copy, Debug)]
pub struct AdaptiveOptions {
    pub newton: Options,
    pub max_step: f64,
    pub reltol: f64,
    pub voltage_abstol: f64,
    /// Reserved for future dynamic current states such as inductor currents.
    /// Algebraic MNA voltage-source branch currents are not LTE states.
    pub current_abstol: f64,
    pub max_steps: usize,
    pub initial_step: f64,
}

#[derive(Clone, Debug)]
pub struct AdaptiveResult {
    pub completed: bool,
    pub times: Vec<f64>,
    pub states: Vec<Vec<f64>>,
    pub inputs: Vec<Vec<f64>>,
    pub device_currents: Vec<[f64; 4]>,
    pub device_charges: Vec<[f64; 4]>,
    pub coefficients: Vec<[f64; 3]>,
    pub accepted_steps: usize,
    pub rejected_steps: usize,
    pub trial_solves: usize,
    pub profile: Profile,
}

fn adaptive_inputs_at_time(times: &[f64], inputs: Waveforms<'_>, time: f64) -> Vec<f64> {
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

fn first_derivative_weights_at_zero(nodes: [f64; 4]) -> Option<[f64; 4]> {
    let mut weights = [0.0; 4];
    for (index, &node) in nodes.iter().enumerate() {
        let mut denominator = 1.0;
        for (other, &other_node) in nodes.iter().enumerate() {
            if other != index {
                denominator *= node - other_node;
            }
        }
        if denominator == 0.0 || !denominator.is_finite() {
            return None;
        }
        let mut numerator = 0.0;
        for (omitted, _) in nodes.iter().enumerate() {
            if omitted == index {
                continue;
            }
            let mut product = 1.0;
            for (other, &other_node) in nodes.iter().enumerate() {
                if other != index && other != omitted {
                    product *= -other_node;
                }
            }
            numerator += product;
        }
        weights[index] = numerator / denominator;
        if !weights[index].is_finite() {
            return None;
        }
    }
    Some(weights)
}

fn charge_defect_weights(
    step: f64,
    previous_step: f64,
    previous2_step: f64,
    bdf2: [f64; 3],
) -> Option<[f64; 4]> {
    if step <= 0.0 || previous_step <= 0.0 || previous2_step <= 0.0 {
        return None;
    }
    let ratio1 = previous_step / step;
    let ratio2 = previous2_step / step;
    let nodes = [0.0, -1.0, -(1.0 + ratio1), -(1.0 + ratio1 + ratio2)];
    let mut bdf3 = first_derivative_weights_at_zero(nodes)?;
    for weight in &mut bdf3 {
        *weight /= step;
    }
    Some([
        bdf3[0] - bdf2[0],
        bdf3[1] - bdf2[1],
        bdf3[2] - bdf2[2],
        bdf3[3],
    ])
}

fn projected_lte_wrms(
    correction: &[f64],
    state: &[f64],
    previous: &[f64],
    node_count: usize,
    reltol: f64,
    voltage_abstol: f64,
) -> f64 {
    let controlled_count = node_count
        .min(correction.len())
        .min(state.len())
        .min(previous.len());
    if controlled_count == 0 {
        return f64::INFINITY;
    }
    let mut sum = 0.0;
    for index in 0..controlled_count {
        let scale = (reltol * state[index].abs().max(previous[index].abs()) + voltage_abstol)
            .max(ADAPTIVE_SCALE_FLOOR);
        let normalized = correction[index] / scale;
        sum += normalized * normalized;
    }
    (sum / controlled_count as f64).sqrt()
}

fn adaptive_pi_accepted_step(
    step: f64,
    error: f64,
    previous_accepted_error: Option<f64>,
    follows_rejection: bool,
) -> f64 {
    let error = error.max(ADAPTIVE_ERR_FLOOR);
    let previous = previous_accepted_error
        .unwrap_or(error)
        .max(ADAPTIVE_ERR_FLOOR);
    let mut factor = ADAPTIVE_SAFETY
        * error.powf(-ADAPTIVE_PI_PROPORTIONAL)
        * previous.powf(ADAPTIVE_PI_INTEGRAL);
    factor = factor.clamp(ADAPTIVE_GROWTH_MIN, ADAPTIVE_GROWTH_MAX);
    if follows_rejection {
        factor = factor.min(ADAPTIVE_POST_REJECT_GROWTH_MAX);
    }
    step * factor
}

fn adaptive_rejected_step(step: f64, error: f64, newton_failure: bool) -> f64 {
    let factor = if newton_failure || !error.is_finite() {
        ADAPTIVE_NEWTON_REJECT_FACTOR
    } else {
        (ADAPTIVE_SAFETY * error.powf(-1.0 / ADAPTIVE_STEP_ORDER))
            .clamp(ADAPTIVE_GROWTH_MIN, ADAPTIVE_REJECT_GROWTH_MAX)
    };
    step * factor
}

fn adaptive_coefficients(step: f64, previous_step: f64) -> [f64; 3] {
    if previous_step > 0.0 && step / previous_step <= 2.0 {
        [
            (2.0 * step + previous_step) / (step * (step + previous_step)),
            -(step + previous_step) / (step * previous_step),
            step / (previous_step * (step + previous_step)),
        ]
    } else {
        [1.0 / step, -1.0 / step, 0.0]
    }
}

#[allow(clippy::too_many_arguments)]
fn stamp_adaptive_candidate_system(
    circuit: &CircuitProblem,
    devices: &[Device],
    state: &[f64],
    input_now: &[f64],
    evaluations: &[Evaluation],
    previous_charges: &[[f64; 4]],
    previous2_charges: &[[f64; 4]],
    coefficients: [f64; 3],
    history: &HistoryTerms,
    history2: &HistoryTerms,
    gmin: f64,
    system: &mut DenseSystem,
) -> bool {
    system.residual.fill(0.0);
    system.jacobian.fill(0.0);
    if devices.len() != evaluations.len()
        || devices.len() != previous_charges.len()
        || devices.len() != previous2_charges.len()
    {
        return false;
    }
    for (position, (device, evaluation)) in devices
        .iter()
        .copied()
        .zip(evaluations.iter().copied())
        .enumerate()
    {
        for terminal_row in 0..4 {
            let Some(row) = device.rows[terminal_row] else {
                continue;
            };
            system.residual[row] += evaluation.currents[terminal_row]
                + coefficients[0] * evaluation.charges[terminal_row]
                + coefficients[1] * previous_charges[position][terminal_row]
                + coefficients[2] * previous2_charges[position][terminal_row];
            for terminal_col in 0..4 {
                let Some(column) = device.rows[terminal_col] else {
                    continue;
                };
                let offset = terminal_row * 4 + terminal_col;
                system.add_jacobian(
                    row,
                    column,
                    evaluation.conductance[offset]
                        + coefficients[0] * evaluation.capacitance[offset],
                );
            }
        }
    }
    stamp_linear_elements(
        circuit,
        state,
        input_now,
        coefficients,
        history,
        history2,
        gmin,
        system,
    )
}

#[allow(clippy::too_many_arguments)]
fn charge_defect_lte(
    circuit: &CircuitProblem,
    devices: &[Device],
    state: &[f64],
    previous_state: &[f64],
    input_now: &[f64],
    evaluations: &[Evaluation],
    charges1: &[[f64; 4]],
    charges2: &[[f64; 4]],
    charges3: &[[f64; 4]],
    history_now: &HistoryTerms,
    history1: &HistoryTerms,
    history2: &HistoryTerms,
    history3: &HistoryTerms,
    coefficients: [f64; 3],
    defect_weights: [f64; 4],
    options: AdaptiveOptions,
    system: &mut DenseSystem,
) -> Option<f64> {
    if !stamp_adaptive_candidate_system(
        circuit,
        devices,
        state,
        input_now,
        evaluations,
        charges1,
        charges2,
        coefficients,
        history1,
        history2,
        options.newton.gmin,
        system,
    ) {
        return None;
    }
    system.residual.fill(0.0);
    if devices.len() != charges3.len() {
        return None;
    }
    for (position, (device, evaluation)) in devices
        .iter()
        .copied()
        .zip(evaluations.iter().copied())
        .enumerate()
    {
        for terminal in 0..4 {
            let Some(row) = device.rows[terminal] else {
                continue;
            };
            system.residual[row] += defect_weights[0] * evaluation.charges[terminal]
                + defect_weights[1] * charges1[position][terminal]
                + defect_weights[2] * charges2[position][terminal]
                + defect_weights[3] * charges3[position][terminal];
        }
    }
    for (position, capacitor) in circuit.capacitors.iter().enumerate() {
        let defect = capacitor.capacitance
            * (defect_weights[0] * history_now.capacitor_dv[position]
                + defect_weights[1] * history1.capacitor_dv[position]
                + defect_weights[2] * history2.capacitor_dv[position]
                + defect_weights[3] * history3.capacitor_dv[position]);
        if let Some(row) = capacitor.ai {
            system.residual[row] += defect;
        }
        if let Some(row) = capacitor.bi {
            system.residual[row] -= defect;
        }
    }
    if !system.residual.iter().all(|value| value.is_finite())
        || !solve_dense_neg_rhs_in_place(&mut system.jacobian, &mut system.residual)
    {
        return None;
    }
    let error = projected_lte_wrms(
        &system.residual,
        state,
        previous_state,
        circuit.node_count,
        options.reltol,
        options.voltage_abstol,
    );
    error.is_finite().then_some(error)
}

#[allow(clippy::too_many_arguments)]
fn solve_adaptive_candidate<E: Evaluator>(
    circuit: &CircuitProblem,
    devices: &[Device],
    evaluator: &mut E,
    evaluator_indices: &[usize],
    state: &mut [f64],
    previous_charges: &[[f64; 4]],
    previous2_charges: &[[f64; 4]],
    input_now: &[f64],
    coefficients: [f64; 3],
    history: &HistoryTerms,
    history2: &HistoryTerms,
    options: Options,
    terminal_batch: &mut [[f64; 4]],
    evaluation_batch: &mut [Evaluation],
    system: &mut DenseSystem,
    profile: &mut Profile,
) -> bool {
    for _iteration in 0..options.max_iterations {
        if options.profile {
            profile.newton_iterations += 1;
        }
        let evaluated = evaluate_devices(
            evaluator,
            devices,
            evaluator_indices,
            state,
            input_now,
            terminal_batch,
            evaluation_batch,
            profile,
            options.profile,
        );
        let stamped = evaluated
            && stamp_adaptive_candidate_system(
                circuit,
                devices,
                state,
                input_now,
                evaluation_batch,
                previous_charges,
                previous2_charges,
                coefficients,
                history,
                history2,
                options.gmin,
                system,
            );
        if !stamped || !solve_dense_neg_rhs_in_place(&mut system.jacobian, &mut system.residual) {
            return false;
        }
        let mut peak = 0.0f64;
        for value in &system.residual[..circuit.node_count] {
            if !value.is_finite() {
                return false;
            }
            peak = peak.max(value.abs());
        }
        if peak <= options.voltage_tolerance {
            return true;
        }
        if peak > options.step_limit {
            let scale = options.step_limit / peak;
            for value in &mut system.residual {
                *value *= scale;
            }
        }
        for (value, delta) in state.iter_mut().zip(&system.residual) {
            *value += delta;
        }
    }
    false
}

pub fn solve_fixed_grid<E: Evaluator>(
    circuit: &CircuitProblem,
    devices: &[Device],
    evaluator: &mut E,
    initial: &[f64],
    times: &[f64],
    inputs: Waveforms<'_>,
    options: Options,
) -> Result {
    if validate_fixed_grid_input(circuit, devices, initial, times, inputs).is_err() {
        return Result {
            completed: false,
            states: Vec::new(),
            device_currents: Vec::new(),
            device_charges: Vec::new(),
            failures: 0,
            first_failure: Some(0),
            profile: Profile::default(),
        };
    }
    let mut profile = Profile::default();
    let input_at = |index: usize| inputs.sample(index).unwrap_or_default();
    let mut states = vec![vec![0.0; circuit.size]; times.len()];
    states[0].copy_from_slice(initial);
    let mut state = initial.to_vec();
    let initial_inputs = input_at(0);
    let mut charge1 = vec![[0.0; 4]; devices.len()];
    let mut charge2 = vec![[0.0; 4]; devices.len()];
    let evaluator_indices: Vec<usize> = devices
        .iter()
        .map(|device| device.evaluator_index)
        .collect();
    let mut terminal_batch = vec![[0.0; 4]; devices.len()];
    let mut evaluation_batch = vec![Evaluation::default(); devices.len()];
    let history_len = times.len().saturating_mul(devices.len());
    let mut device_currents = if options.record_device_history {
        vec![[0.0; 4]; history_len]
    } else {
        Vec::new()
    };
    let mut device_charges = if options.record_device_history {
        vec![[0.0; 4]; history_len]
    } else {
        Vec::new()
    };
    if !evaluate_devices(
        evaluator,
        devices,
        &evaluator_indices,
        &state,
        &initial_inputs,
        &mut terminal_batch,
        &mut evaluation_batch,
        &mut profile,
        options.profile,
    ) {
        if options.profile {
            profile.failed_steps.push(0);
        }
        return Result {
            completed: false,
            states,
            device_currents,
            device_charges,
            failures: 0,
            first_failure: Some(0),
            profile,
        };
    }
    for (position, evaluation) in evaluation_batch.iter().copied().enumerate() {
        charge1[position] = evaluation.charges;
        charge2[position] = evaluation.charges;
        if options.record_device_history {
            device_currents[position] = evaluation.currents;
            device_charges[position] = evaluation.charges;
        }
    }
    let mut system = DenseSystem::new(circuit.size);
    let mut failures = 0usize;
    let mut first_failure = None;
    let mut converged_streak = 1usize;
    let predictor_enabled = options.gear2 && gear2_predictor_enabled();

    for sample in 1..times.len() {
        let h = times[sample] - times[sample - 1];
        let coefficients = if options.gear2 && sample >= 2 {
            let h_previous = times[sample - 1] - times[sample - 2];
            [
                (2.0 * h + h_previous) / (h * (h + h_previous)),
                -(h + h_previous) / (h * h_previous),
                h / (h_previous * (h + h_previous)),
            ]
        } else {
            [1.0 / h, -1.0 / h, 0.0]
        };
        let input_now = input_at(sample);
        let input_previous = input_at(sample - 1);
        let input_previous2 = if sample >= 2 {
            input_at(sample - 2)
        } else {
            input_previous.clone()
        };
        let Some(history) = history_for(circuit, &states[sample - 1], &input_previous) else {
            if options.profile {
                profile.failed_steps.push(sample);
            }
            return Result {
                completed: false,
                states,
                device_currents,
                device_charges,
                failures,
                first_failure: Some(sample),
                profile,
            };
        };
        let history2_state = if sample >= 2 {
            &states[sample - 2]
        } else {
            &states[sample - 1]
        };
        let Some(history2) = history_for(circuit, history2_state, &input_previous2) else {
            if options.profile {
                profile.failed_steps.push(sample);
            }
            return Result {
                completed: false,
                states,
                device_currents,
                device_charges,
                failures,
                first_failure: Some(sample),
                profile,
            };
        };
        state.clone_from(&states[sample - 1]);
        if predictor_enabled
            && sample >= 2
            && converged_streak >= 2
            && predict_gear2_state(
                &mut state,
                &states[sample - 1],
                &states[sample - 2],
                &input_now,
                &input_previous,
                &input_previous2,
                h,
                times[sample - 1] - times[sample - 2],
            )
            && options.profile
        {
            profile.gear2_predictor_steps += 1;
        }
        let mut converged = false;
        for _iteration in 0..options.max_iterations {
            if options.profile {
                profile.newton_iterations += 1;
            }
            system.residual.fill(0.0);
            system.jacobian.fill(0.0);
            let evaluated = evaluate_devices(
                evaluator,
                devices,
                &evaluator_indices,
                &state,
                &input_now,
                &mut terminal_batch,
                &mut evaluation_batch,
                &mut profile,
                options.profile,
            );
            if evaluated {
                for (position, (device, evaluation)) in devices
                    .iter()
                    .copied()
                    .zip(evaluation_batch.iter().copied())
                    .enumerate()
                {
                    for terminal_row in 0..4 {
                        let Some(row) = device.rows[terminal_row] else {
                            continue;
                        };
                        let current = evaluation.currents[terminal_row]
                            + coefficients[0] * evaluation.charges[terminal_row]
                            + coefficients[1] * charge1[position][terminal_row]
                            + coefficients[2] * charge2[position][terminal_row];
                        system.residual[row] += current;
                        for terminal_col in 0..4 {
                            let Some(column) = device.rows[terminal_col] else {
                                continue;
                            };
                            let offset = terminal_row * 4 + terminal_col;
                            system.add_jacobian(
                                row,
                                column,
                                evaluation.conductance[offset]
                                    + coefficients[0] * evaluation.capacitance[offset],
                            );
                        }
                    }
                }
            }
            let stamped = evaluated
                && stamp_linear_elements(
                    circuit,
                    &state,
                    &input_now,
                    coefficients,
                    &history,
                    &history2,
                    options.gmin,
                    &mut system,
                );
            if !stamped || !solve_dense_neg_rhs_in_place(&mut system.jacobian, &mut system.residual)
            {
                break;
            }
            let mut peak = 0.0f64;
            let mut finite = true;
            for value in &system.residual[..circuit.node_count] {
                if !value.is_finite() {
                    finite = false;
                    break;
                }
                peak = peak.max(value.abs());
            }
            if !finite {
                break;
            }
            if peak <= options.voltage_tolerance {
                converged = true;
                break;
            }
            if peak > options.step_limit {
                let scale = options.step_limit / peak;
                for value in &mut system.residual {
                    *value *= scale;
                }
            }
            for (value, delta) in state.iter_mut().zip(&system.residual) {
                *value += delta;
            }
        }
        if !converged {
            failures += 1;
            first_failure.get_or_insert(sample);
            converged_streak = 0;
            if options.profile {
                profile.failed_steps.push(sample);
            }
        } else {
            converged_streak = converged_streak.saturating_add(1);
        }
        states[sample].copy_from_slice(&state);
        // A converged Newton round does not apply its already-sub-tolerance
        // correction, so evaluation_batch is exactly the accepted-state I/G/Q/C.
        // Failed rounds may have updated state after their last evaluation and
        // must refresh before recording history.
        if !converged
            && !evaluate_devices(
                evaluator,
                devices,
                &evaluator_indices,
                &state,
                &input_now,
                &mut terminal_batch,
                &mut evaluation_batch,
                &mut profile,
                options.profile,
            )
        {
            if options.profile && profile.failed_steps.last().copied() != Some(sample) {
                profile.failed_steps.push(sample);
            }
            return Result {
                completed: false,
                states,
                device_currents,
                device_charges,
                failures,
                first_failure: Some(sample),
                profile,
            };
        }
        for (position, evaluation) in evaluation_batch.iter().copied().enumerate() {
            charge2[position] = charge1[position];
            charge1[position] = evaluation.charges;
            if options.record_device_history {
                let offset = sample * devices.len() + position;
                device_currents[offset] = evaluation.currents;
                device_charges[offset] = evaluation.charges;
            }
        }
    }
    Result {
        completed: true,
        states,
        device_currents,
        device_charges,
        failures,
        first_failure,
        profile,
    }
}

/// Adaptive variable-step Gear2 using a charge-history defect LTE estimate.
pub fn solve_adaptive_gear2<E: Evaluator>(
    circuit: &CircuitProblem,
    devices: &[Device],
    evaluator: &mut E,
    initial: &[f64],
    source_times: &[f64],
    source_inputs: Waveforms<'_>,
    options: AdaptiveOptions,
) -> AdaptiveResult {
    let valid_options = options.reltol > 0.0
        && options.reltol.is_finite()
        && options.voltage_abstol > 0.0
        && options.voltage_abstol.is_finite()
        && options.current_abstol > 0.0
        && options.current_abstol.is_finite()
        && options.max_steps > 0
        && options.max_step.is_finite()
        && options.initial_step.is_finite()
        && options.newton.max_iterations > 0
        && options.newton.voltage_tolerance > 0.0
        && options.newton.voltage_tolerance.is_finite()
        && options.newton.step_limit > 0.0
        && options.newton.step_limit.is_finite()
        && options.newton.gmin >= 0.0
        && options.newton.gmin.is_finite();
    if !valid_options
        || validate_fixed_grid_input(circuit, devices, initial, source_times, source_inputs)
            .is_err()
    {
        return AdaptiveResult {
            completed: false,
            times: Vec::new(),
            states: Vec::new(),
            inputs: Vec::new(),
            device_currents: Vec::new(),
            device_charges: Vec::new(),
            coefficients: Vec::new(),
            accepted_steps: 0,
            rejected_steps: 0,
            trial_solves: 0,
            profile: Profile::default(),
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
    let smallest_source_step = source_times
        .windows(2)
        .map(|window| window[1] - window[0])
        .fold(span, f64::min);
    let mut step = if options.initial_step > 0.0 {
        options.initial_step
    } else {
        let denominator = (source_times.len() - 1).max(ADAPTIVE_INITIAL_MIN_DENOM);
        ADAPTIVE_STARTUP_FRACTION * (span / denominator as f64).min(smallest_source_step)
    };
    if step <= 0.0 || !step.is_finite() {
        step = span / 100.0;
    }
    step = step.min(max_step);
    let restart_step = step;
    let min_step = ADAPTIVE_MIN_H_ABS.max(span * ADAPTIVE_MIN_H_REL);
    let done_tolerance = ADAPTIVE_DONE_ABS.max(span * ADAPTIVE_DONE_REL);
    let critical_times = adaptive_critical_times(source_times, source_inputs);

    let capacity = source_times.len().max(16);
    let mut times = Vec::with_capacity(capacity);
    let mut states = Vec::with_capacity(capacity);
    let mut input_history = Vec::with_capacity(capacity);
    let mut coefficients_history = Vec::with_capacity(capacity);
    let mut device_currents = Vec::with_capacity(capacity.saturating_mul(devices.len()));
    let mut device_charges = Vec::with_capacity(capacity.saturating_mul(devices.len()));
    let mut current = initial.to_vec();
    let mut previous2 = initial.to_vec();
    let mut input_current = adaptive_inputs_at_time(source_times, source_inputs, start);
    let mut input_previous2 = input_current.clone();
    let mut current_time = start;
    let mut previous_step = -1.0;
    let mut previous2_step = -1.0;
    let mut previous_accepted_error = None;
    let mut follows_rejection = false;
    let mut accepted_steps = 0usize;
    let mut rejected_steps = 0usize;
    let mut trial_solves = 0usize;
    let mut profile = Profile::default();
    let evaluator_indices: Vec<usize> = devices
        .iter()
        .map(|device| device.evaluator_index)
        .collect();
    let mut terminal_batch = vec![[0.0; 4]; devices.len()];
    let mut evaluation_batch = vec![Evaluation::default(); devices.len()];
    let mut candidate = current.clone();
    let mut system = DenseSystem::new(circuit.size);

    if !evaluate_devices(
        evaluator,
        devices,
        &evaluator_indices,
        &current,
        &input_current,
        &mut terminal_batch,
        &mut evaluation_batch,
        &mut profile,
        options.newton.profile,
    ) {
        return AdaptiveResult {
            completed: false,
            times,
            states,
            inputs: input_history,
            device_currents,
            device_charges,
            coefficients: coefficients_history,
            accepted_steps,
            rejected_steps,
            trial_solves,
            profile,
        };
    }
    let mut current_charges: Vec<[f64; 4]> = evaluation_batch
        .iter()
        .map(|evaluation| evaluation.charges)
        .collect();
    let mut previous2_charges = current_charges.clone();
    let mut previous3_charges = current_charges.clone();
    let Some(mut history_current) = history_for(circuit, &current, &input_current) else {
        return AdaptiveResult {
            completed: false,
            times,
            states,
            inputs: input_history,
            device_currents,
            device_charges,
            coefficients: coefficients_history,
            accepted_steps,
            rejected_steps,
            trial_solves,
            profile,
        };
    };
    let mut history2 = history_current.clone();
    let mut history3 = history_current.clone();
    times.push(start);
    states.push(current.clone());
    input_history.push(input_current.clone());
    coefficients_history.push([0.0; 3]);
    for evaluation in &evaluation_batch {
        device_currents.push(evaluation.currents);
        device_charges.push(evaluation.charges);
    }

    while accepted_steps + rejected_steps < options.max_steps && current_time < end - done_tolerance
    {
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

        let endpoint = current_time + step;
        let input_now = adaptive_inputs_at_time(source_times, source_inputs, endpoint);
        let coefficients = adaptive_coefficients(step, previous_step);
        candidate.clone_from(&current);
        if previous_step > 0.0
            && predict_gear2_state(
                &mut candidate,
                &current,
                &previous2,
                &input_now,
                &input_current,
                &input_previous2,
                step,
                previous_step,
            )
            && options.newton.profile
        {
            profile.gear2_predictor_steps += 1;
        }
        trial_solves += 1;
        let converged = solve_adaptive_candidate(
            circuit,
            devices,
            evaluator,
            &evaluator_indices,
            &mut candidate,
            &current_charges,
            &previous2_charges,
            &input_now,
            coefficients,
            &history_current,
            &history2,
            options.newton,
            &mut terminal_batch,
            &mut evaluation_batch,
            &mut system,
            &mut profile,
        );
        if !converged {
            rejected_steps += 1;
            follows_rejection = true;
            if options.newton.profile {
                profile.newton_rejections += 1;
            }
            step = adaptive_rejected_step(step, f64::INFINITY, true).max(min_step);
            continue;
        }
        let Some(candidate_history) = history_for(circuit, &candidate, &input_now) else {
            break;
        };
        let defect_weights =
            charge_defect_weights(step, previous_step, previous2_step, coefficients);
        let error = if let Some(weights) = defect_weights {
            if options.newton.profile {
                profile.lte_estimates += 1;
            }
            let estimate = charge_defect_lte(
                circuit,
                devices,
                &candidate,
                &current,
                &input_now,
                &evaluation_batch,
                &current_charges,
                &previous2_charges,
                &previous3_charges,
                &candidate_history,
                &history_current,
                &history2,
                &history3,
                coefficients,
                weights,
                options,
                &mut system,
            );
            if estimate.is_some() && options.newton.profile {
                profile.lte_linear_solves += 1;
            }
            estimate.unwrap_or(f64::INFINITY)
        } else {
            0.0
        };
        if defect_weights.is_none() || error <= ADAPTIVE_ACCEPT_WRMS {
            current_time = endpoint;
            accepted_steps += 1;
            previous3_charges.clone_from(&previous2_charges);
            previous2.clone_from(&current);
            previous2_charges.clone_from(&current_charges);
            std::mem::swap(&mut current, &mut candidate);
            for (charges, evaluation) in current_charges.iter_mut().zip(&evaluation_batch) {
                *charges = evaluation.charges;
            }
            history3.clone_from(&history2);
            history2.clone_from(&history_current);
            history_current = candidate_history;
            input_previous2.clone_from(&input_current);
            input_current = input_now;
            times.push(current_time);
            states.push(current.clone());
            input_history.push(input_current.clone());
            coefficients_history.push(coefficients);
            for evaluation in &evaluation_batch {
                device_currents.push(evaluation.currents);
                device_charges.push(evaluation.charges);
            }

            let critical_tolerance = min_step.max(done_tolerance);
            let hit_critical = critical_times
                .iter()
                .any(|critical| (*critical - current_time).abs() <= critical_tolerance);
            if hit_critical {
                previous2.clone_from(&current);
                previous2_charges.clone_from(&current_charges);
                previous3_charges.clone_from(&current_charges);
                history2.clone_from(&history_current);
                history3.clone_from(&history_current);
                input_previous2.clone_from(&input_current);
                previous_step = -1.0;
                previous2_step = -1.0;
                previous_accepted_error = None;
            } else {
                previous2_step = previous_step;
                previous_step = step;
            }
            step = if hit_critical {
                restart_step.min(max_step)
            } else if defect_weights.is_some() {
                let next = adaptive_pi_accepted_step(
                    step,
                    error,
                    previous_accepted_error,
                    follows_rejection,
                );
                previous_accepted_error = Some(error.max(ADAPTIVE_ERR_FLOOR));
                next
            } else {
                step
            };
            follows_rejection = false;
        } else {
            rejected_steps += 1;
            follows_rejection = true;
            if options.newton.profile {
                profile.lte_rejections += 1;
            }
            step = adaptive_rejected_step(step, error, false).max(min_step);
        }
    }

    AdaptiveResult {
        completed: current_time >= end - done_tolerance,
        times,
        states,
        inputs: input_history,
        device_currents,
        device_charges,
        coefficients: coefficients_history,
        accepted_steps,
        rejected_steps,
        trial_solves,
        profile,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn linear_evaluation(terminals: [f64; 4]) -> Evaluation {
        let current = terminals[0] - terminals[2];
        let mut conductance = [0.0; 16];
        conductance[0] = 1.0;
        conductance[2] = -1.0;
        conductance[8] = -1.0;
        conductance[10] = 1.0;
        Evaluation {
            currents: [current, 0.0, -current, 0.0],
            conductance,
            ..Evaluation::default()
        }
    }

    struct LinearEvaluator;

    impl Evaluator for LinearEvaluator {
        fn evaluate(&mut self, _index: usize, terminals: [f64; 4]) -> Option<Evaluation> {
            Some(linear_evaluation(terminals))
        }
    }

    struct BatchOnlyEvaluator {
        calls: usize,
    }

    impl Evaluator for BatchOnlyEvaluator {
        fn evaluate(&mut self, _index: usize, _terminals: [f64; 4]) -> Option<Evaluation> {
            panic!("fixed-grid transient fell back to scalar device evaluation")
        }

        fn evaluate_batch(
            &mut self,
            indices: &[usize],
            terminals: &[[f64; 4]],
            evaluations: &mut [Evaluation],
        ) -> BatchStatus {
            assert_eq!(indices.len(), terminals.len());
            assert_eq!(indices.len(), evaluations.len());
            self.calls += 1;
            for (&terminal, output) in terminals.iter().zip(evaluations) {
                *output = linear_evaluation(terminal);
            }
            BatchStatus {
                completed: true,
                attempted: terminals.len(),
            }
        }
    }

    fn linear_problem() -> (CircuitProblem, Vec<Device>) {
        let circuit = CircuitProblem {
            node_count: 1,
            size: 1,
            devices: Vec::new(),
            resistors: Vec::new(),
            capacitors: Vec::new(),
            current_sources: Vec::new(),
            dynamic_current_sources: Vec::new(),
            vccs: Vec::new(),
            voltage_sources: Vec::new(),
            vcvs: Vec::new(),
            cccs: Vec::new(),
            ccvs: Vec::new(),
        };
        let rail = Term {
            kind: 2,
            reference: 0,
            value: 0.0,
        };
        let devices = vec![Device {
            terms: [
                Term {
                    kind: 0,
                    reference: 0,
                    value: 0.0,
                },
                rail,
                rail,
                rail,
            ],
            rows: [Some(0), None, None, None],
            evaluator_index: 0,
        }];
        (circuit, devices)
    }

    fn rc_problem() -> CircuitProblem {
        use crate::transient::{Capacitor, CurrentSource, Resistor};

        let node = Term {
            kind: 0,
            reference: 0,
            value: 0.0,
        };
        let ground = Term {
            kind: 2,
            reference: 0,
            value: 0.0,
        };
        CircuitProblem {
            node_count: 1,
            size: 1,
            devices: Vec::new(),
            resistors: vec![Resistor {
                a: node,
                b: ground,
                ai: Some(0),
                bi: None,
                conductance: 1e-3,
            }],
            capacitors: vec![Capacitor {
                a: node,
                b: ground,
                ai: Some(0),
                bi: None,
                capacitance: 1e-9,
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
        }
    }

    fn solve_adaptive_rc(reltol: f64, voltage_abstol: f64) -> AdaptiveResult {
        let circuit = rc_problem();
        let waveforms = Waveforms::new(&[], 0, 2).unwrap();
        solve_adaptive_gear2(
            &circuit,
            &[],
            &mut LinearEvaluator,
            &[0.0],
            &[0.0, 5e-6],
            waveforms,
            AdaptiveOptions {
                newton: Options {
                    gear2: true,
                    max_iterations: 4,
                    voltage_tolerance: 1e-12,
                    step_limit: 10.0,
                    gmin: 0.0,
                    record_device_history: false,
                    profile: true,
                },
                max_step: 1e-6,
                reltol,
                voltage_abstol,
                current_abstol: 1e-12,
                max_steps: 10_000,
                initial_step: 1e-9,
            },
        )
    }

    fn options(max_iterations: usize, profile: bool) -> Options {
        Options {
            gear2: false,
            max_iterations,
            voltage_tolerance: 1e-12,
            step_limit: 10.0,
            gmin: 0.0,
            record_device_history: false,
            profile,
        }
    }

    #[test]
    fn fixed_grid_profile_counts_newton_and_device_evaluations() {
        let (circuit, devices) = linear_problem();
        let waveforms = Waveforms::new(&[], 0, 2).unwrap();
        let result = solve_fixed_grid(
            &circuit,
            &devices,
            &mut LinearEvaluator,
            &[1.0],
            &[0.0, 1.0],
            waveforms,
            options(4, true),
        );

        assert!(result.completed);
        assert_eq!(result.failures, 0);
        assert_eq!(result.profile.newton_iterations, 2);
        assert_eq!(result.profile.bsim_evaluations, 3);
        assert_eq!(result.profile.bsim_batches, 3);
        assert!(result.profile.failed_steps.is_empty());
    }

    #[test]
    fn fixed_grid_uses_one_batch_call_per_newton_or_history_evaluation() {
        let (circuit, devices) = linear_problem();
        let waveforms = Waveforms::new(&[], 0, 2).unwrap();
        let mut evaluator = BatchOnlyEvaluator { calls: 0 };
        let mut solve_options = options(4, true);
        solve_options.record_device_history = true;
        let result = solve_fixed_grid(
            &circuit,
            &devices,
            &mut evaluator,
            &[1.0],
            &[0.0, 1.0],
            waveforms,
            solve_options,
        );

        assert!(result.completed);
        assert_eq!(evaluator.calls, 3);
        assert_eq!(result.profile.bsim_batches, evaluator.calls);
        assert_eq!(result.profile.bsim_evaluations, evaluator.calls);
        assert_eq!(
            result.device_currents,
            vec![[1.0, 0.0, -1.0, 0.0], [0.0; 4]]
        );
    }

    #[test]
    fn fixed_grid_profile_records_failed_step_and_respects_disable() {
        let (circuit, devices) = linear_problem();
        let waveforms = Waveforms::new(&[], 0, 2).unwrap();
        let failed = solve_fixed_grid(
            &circuit,
            &devices,
            &mut LinearEvaluator,
            &[1.0],
            &[0.0, 1.0],
            waveforms,
            options(1, true),
        );
        assert_eq!(failed.failures, 1);
        assert_eq!(failed.profile.newton_iterations, 1);
        assert_eq!(failed.profile.bsim_evaluations, 3);
        assert_eq!(failed.profile.bsim_batches, 3);
        assert_eq!(failed.profile.failed_steps, vec![1]);

        let disabled = solve_fixed_grid(
            &circuit,
            &devices,
            &mut LinearEvaluator,
            &[1.0],
            &[0.0, 1.0],
            waveforms,
            options(4, false),
        );
        assert_eq!(disabled.profile.newton_iterations, 0);
        assert_eq!(disabled.profile.bsim_evaluations, 0);
        assert_eq!(disabled.profile.bsim_batches, 0);
        assert!(disabled.profile.failed_steps.is_empty());
    }

    #[test]
    fn gear2_predictor_extrapolates_variable_step_state() {
        let mut state = [0.0, 0.0];
        assert!(predict_gear2_state(
            &mut state,
            &[2.0, -1.0],
            &[1.0, -0.5],
            &[0.6],
            &[0.4],
            &[0.3],
            2.0,
            1.0,
        ));
        assert_eq!(state, [4.0, -2.0]);
    }

    #[test]
    fn gear2_predictor_rejects_input_edges_and_extreme_step_growth() {
        let previous = [2.0];
        let previous2 = [1.0];
        let mut edge_state = previous;
        assert!(!predict_gear2_state(
            &mut edge_state,
            &previous,
            &previous2,
            &[1.0],
            &[0.0],
            &[0.0],
            1.0,
            1.0,
        ));
        assert_eq!(edge_state, previous);

        let mut growth_state = previous;
        assert!(!predict_gear2_state(
            &mut growth_state,
            &previous,
            &previous2,
            &[],
            &[],
            &[],
            5.0,
            1.0,
        ));
        assert_eq!(growth_state, previous);
    }

    #[test]
    fn adaptive_gear2_returns_nonuniform_history_and_profile() {
        let (circuit, devices) = linear_problem();
        let waveforms = Waveforms::new(&[], 0, 3).unwrap();
        let result = solve_adaptive_gear2(
            &circuit,
            &devices,
            &mut LinearEvaluator,
            &[1.0],
            &[0.0, 0.4, 1.0],
            waveforms,
            AdaptiveOptions {
                newton: Options {
                    gear2: true,
                    max_iterations: 4,
                    voltage_tolerance: 1e-12,
                    step_limit: 10.0,
                    gmin: 0.0,
                    record_device_history: true,
                    profile: true,
                },
                max_step: 0.3,
                reltol: 1e-4,
                voltage_abstol: 1e-8,
                current_abstol: 1e-12,
                max_steps: 100,
                initial_step: 0.05,
            },
        );

        assert!(result.completed);
        assert_eq!(result.times.first().copied(), Some(0.0));
        assert_eq!(result.times.last().copied(), Some(1.0));
        assert!(result.times.windows(2).all(|pair| pair[1] > pair[0]));
        assert_eq!(result.accepted_steps, result.times.len() - 1);
        assert_eq!(result.coefficients.len(), result.times.len());
        assert_eq!(result.states.len(), result.times.len());
        assert_eq!(result.inputs.len(), result.times.len());
        assert_eq!(
            result.device_currents.len(),
            result.times.len() * devices.len()
        );
        assert_eq!(
            result.device_charges.len(),
            result.times.len() * devices.len()
        );
        assert_eq!(
            result.trial_solves,
            result.accepted_steps + result.rejected_steps
        );
        assert!(result.profile.newton_iterations > 0);
        assert!(result.profile.bsim_batches > 0);
        assert!(result.profile.lte_estimates > 0);
        assert_eq!(
            result.profile.lte_estimates,
            result.profile.lte_linear_solves
        );
        assert!(
            result
                .states
                .iter()
                .flatten()
                .all(|value| value.is_finite())
        );
    }

    #[test]
    fn adaptive_gear2_rejects_invalid_tolerances() {
        let (circuit, devices) = linear_problem();
        let waveforms = Waveforms::new(&[], 0, 2).unwrap();
        let result = solve_adaptive_gear2(
            &circuit,
            &devices,
            &mut LinearEvaluator,
            &[1.0],
            &[0.0, 1.0],
            waveforms,
            AdaptiveOptions {
                newton: options(4, true),
                max_step: -1.0,
                reltol: 0.0,
                voltage_abstol: 1e-8,
                current_abstol: 1e-12,
                max_steps: 100,
                initial_step: -1.0,
            },
        );

        assert!(!result.completed);
        assert!(result.times.is_empty());
    }

    #[test]
    fn charge_defect_lte_tracks_explicit_rc_error_and_tolerance() {
        let loose = solve_adaptive_rc(1e-3, 1e-6);
        let tight = solve_adaptive_rc(1e-5, 1e-8);

        assert!(loose.completed);
        assert!(tight.completed);
        assert!(loose.profile.lte_estimates > 0);
        assert!(tight.profile.lte_estimates > 0);
        let trajectory_error = |result: &AdaptiveResult| {
            result
                .times
                .iter()
                .zip(&result.states)
                .map(|(&time, state)| {
                    let exact = 1.0 - (-time / 1e-6).exp();
                    (state[0] - exact).abs()
                })
                .fold(0.0, f64::max)
        };
        let loose_error = trajectory_error(&loose);
        let tight_error = trajectory_error(&tight);

        assert!(loose_error < 5e-3, "loose error = {loose_error:e}");
        assert!(tight_error < 5e-4, "tight error = {tight_error:e}");
        assert!(
            tight_error < loose_error / 3.0,
            "loose error = {loose_error:e}, tight error = {tight_error:e}"
        );
        assert!(tight.accepted_steps > loose.accepted_steps);
        assert_eq!(
            tight.trial_solves,
            tight.accepted_steps + tight.rejected_steps
        );
    }

    #[test]
    fn charge_defect_weights_match_uniform_bdf3_minus_bdf2() {
        let weights =
            charge_defect_weights(2.0, 2.0, 2.0, adaptive_coefficients(2.0, 2.0)).unwrap();
        let expected = [1.0 / 6.0, -0.5, 0.5, -1.0 / 6.0];
        for (actual, expected) in weights.into_iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-14);
        }
        assert!(weights.into_iter().sum::<f64>().abs() < 1e-14);
    }

    #[test]
    fn projected_lte_excludes_algebraic_branch_currents() {
        let correction = [1e-7, 1.0e-3];
        let state = [0.5, 1.0e-3];
        let previous = [0.5, -1.0e-3];
        let error = projected_lte_wrms(&correction, &state, &previous, 1, 1e-4, 1e-6);

        assert!(error < ADAPTIVE_ACCEPT_WRMS);
    }

    #[test]
    fn pi_controller_suppresses_post_rejection_growth() {
        let free = adaptive_pi_accepted_step(1.0, 0.01, Some(0.02), false);
        let guarded = adaptive_pi_accepted_step(1.0, 0.01, Some(0.02), true);

        assert!(free > 1.0);
        assert_eq!(guarded, 1.0);
        assert!(adaptive_rejected_step(1.0, 8.0, false) < 1.0);
        assert_eq!(
            adaptive_rejected_step(1.0, f64::INFINITY, true),
            ADAPTIVE_NEWTON_REJECT_FACTOR
        );
    }
}
