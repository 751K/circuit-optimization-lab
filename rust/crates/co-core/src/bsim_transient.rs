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
            if options.profile {
                profile.failed_steps.push(sample);
            }
        }
        states[sample].copy_from_slice(&state);
        if !evaluate_devices(
            evaluator,
            devices,
            &evaluator_indices,
            &state,
            &input_now,
            &mut terminal_batch,
            &mut evaluation_batch,
            &mut profile,
            options.profile,
        ) {
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
        assert_eq!(result.profile.bsim_evaluations, 4);
        assert_eq!(result.profile.bsim_batches, 4);
        assert!(result.profile.failed_steps.is_empty());
    }

    #[test]
    fn fixed_grid_uses_one_batch_call_per_newton_or_history_evaluation() {
        let (circuit, devices) = linear_problem();
        let waveforms = Waveforms::new(&[], 0, 2).unwrap();
        let mut evaluator = BatchOnlyEvaluator { calls: 0 };
        let result = solve_fixed_grid(
            &circuit,
            &devices,
            &mut evaluator,
            &[1.0],
            &[0.0, 1.0],
            waveforms,
            options(4, true),
        );

        assert!(result.completed);
        assert_eq!(evaluator.calls, 4);
        assert_eq!(result.profile.bsim_batches, evaluator.calls);
        assert_eq!(result.profile.bsim_evaluations, evaluator.calls);
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
}
