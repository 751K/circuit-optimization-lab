//! Compiled SAR conversion batch (rewrite step R8).
//!
//! Marshals one SAR circuit template once and runs a whole batch of closed-loop
//! conversions — a mismatch Monte-Carlo's trial sweep, or a single static sweep
//! — under one `py.detach`, parallelized across trials on the single
//! `co_core::campaign` Rayon pool with candidate-index-ordered write-back.
//!
//! Each trial builds its own native BSIM4 handles (the vendored core is not
//! thread-safe, so handles are created, used, and dropped on one worker thread,
//! exactly as `silicon_campaign` does) with the trial's per-device `delvto`
//! offset, patches the trial's perturbed CDAC capacitor values onto a clone of
//! the shared circuit, and drives `co_core::sar::run_conversion` for every
//! code-center input. The device cards themselves are the frozen
//! `build_devices` model/instance parameters marshalled from Python, so the
//! handle a trial builds is byte-identical to the one the reference path builds
//! (only `delvto` varies per sample).

use std::ffi::CString;
use std::sync::Arc;

use numpy::IntoPyArray;
use numpy::ndarray::Array2;
use pyo3::exceptions::{PyKeyError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use co_bsim4::CoBsim4;
use co_core::bsim_transient::{self, Options};
use co_core::campaign::{
    BatchConfig, BatchProgress, CandidateEvaluator, CandidateOutcome, evaluate_batch,
};
use co_core::sar::{ClockConfig, ExpandedGrid, Role, SarConfig};
use co_core::transient;

use crate::{RustBsimEvaluator, TermRecord, optional_index, required, term};

/// Per-device frozen card: create/setup parameters for one native handle.
#[derive(Clone, Debug)]
struct DeviceCard {
    polarity: i32,
    temperature_k: f64,
    model: Vec<(String, f64)>,
    instance: Vec<(String, f64)>,
}

/// Immutable SAR conversion template shared by every trial of a batch.
pub struct SarTemplate {
    circuit: transient::Problem,
    devices: Vec<bsim_transient::Device>,
    cards: Vec<DeviceCard>,
    config: SarConfig,
    v0: Vec<f64>,
    tgrid: Vec<f64>,
    grid: ExpandedGrid,
    vins: Vec<f64>,
}

/// One trial's mismatch draw: per-device `delvto` and perturbed CDAC values.
#[derive(Clone, Debug)]
struct SarTrial {
    delvto: Vec<f64>,
    cap_values: Vec<f64>,
}

/// Owned native BSIM4 handle, freed on drop. Created, used, and destroyed on a
/// single worker thread — it never crosses threads.
struct HandleGuard(*mut CoBsim4);

impl Drop for HandleGuard {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe { co_bsim4::destroy(self.0) };
            self.0 = std::ptr::null_mut();
        }
    }
}

/// `bsim_transient::Evaluator` owning one trial's native handles (built once and
/// reused across the trial's conversions — `eval_vp` recomputes from terminals
/// and resets its per-handle scratch, so reuse is bit-identical to a fresh
/// handle). Created, used, and dropped on one worker thread.
struct GuardEvaluator {
    _handles: Vec<HandleGuard>,
    evaluator: RustBsimEvaluator,
}

impl GuardEvaluator {
    fn new(cards: &[DeviceCard], delvto: &[f64]) -> Result<Self, String> {
        let mut handles = Vec::with_capacity(cards.len());
        for (card, &delvto) in cards.iter().zip(delvto) {
            handles.push(build_handle(card, delvto)?);
        }
        let raw_handles = handles.iter().map(|handle| handle.0 as usize).collect();
        // Each guard owns one freshly created non-null handle and keeps it alive
        // for the evaluator's complete lifetime.
        let evaluator = unsafe { RustBsimEvaluator::new(raw_handles) };
        Ok(Self {
            _handles: handles,
            evaluator,
        })
    }
}

impl bsim_transient::Evaluator for GuardEvaluator {
    fn evaluate(
        &mut self,
        index: usize,
        terminals: [f64; 4],
    ) -> Option<bsim_transient::Evaluation> {
        <RustBsimEvaluator as bsim_transient::Evaluator>::evaluate(
            &mut self.evaluator,
            index,
            terminals,
        )
    }

    fn evaluate_batch(
        &mut self,
        indices: &[usize],
        terminals: &[[f64; 4]],
        evaluations: &mut [bsim_transient::Evaluation],
    ) -> bsim_transient::BatchStatus {
        <RustBsimEvaluator as bsim_transient::Evaluator>::evaluate_batch(
            &mut self.evaluator,
            indices,
            terminals,
            evaluations,
        )
    }
}

/// Set one parameter, mapping a nonzero status to an error string.
fn set_param(
    handle: *mut CoBsim4,
    name: &str,
    value: f64,
    kind: &str,
    setter: unsafe fn(*mut CoBsim4, *const std::os::raw::c_char, f64) -> std::os::raw::c_int,
) -> Result<(), String> {
    let cname = CString::new(name).map_err(|_| format!("{kind} param {name:?} has NUL"))?;
    let status = unsafe { setter(handle, cname.as_ptr(), value) };
    if status != 0 {
        return Err(format!(
            "{kind} setup failed for {name:?} (status {status})"
        ));
    }
    Ok(())
}

/// Build one native handle from a frozen card, then append the trial's per-device
/// `delvto` offset iff it is nonzero.
///
/// The reference path only writes `delvto` when `mismatch_v` is truthy
/// (`freepdk45.library.device_card`: `if mismatch_v: parameters["delvto"] = ...`),
/// appending it after the geometry parameters — so a zero offset (a device the
/// mismatch draw never touched) leaves the instance card at its nominal shape,
/// and a nonzero offset appends `delvto` last, exactly as the frozen card does.
fn build_handle(card: &DeviceCard, delvto: f64) -> Result<HandleGuard, String> {
    let handle = unsafe { co_bsim4::create(card.polarity, card.temperature_k) };
    if handle.is_null() {
        return Err("BSIM4 native device allocation failed".to_string());
    }
    let guard = HandleGuard(handle);
    for (name, value) in &card.model {
        set_param(guard.0, name, *value, "model", co_bsim4::set_model)?;
    }
    for (name, value) in &card.instance {
        set_param(guard.0, name, *value, "instance", co_bsim4::set_instance)?;
    }
    if delvto != 0.0 {
        set_param(
            guard.0,
            "delvto",
            delvto,
            "instance",
            co_bsim4::set_instance,
        )?;
    }
    let status = unsafe { co_bsim4::setup(guard.0) };
    if status != 0 {
        return Err(format!("BSIM4 setup failed (status {status})"));
    }
    Ok(guard)
}

struct SarEvaluator<'a> {
    template: &'a SarTemplate,
    trials: &'a [SarTrial],
    /// Whether the batch runs trials concurrently, so a trial should leave the
    /// shared device pool alone.
    outer_parallel: bool,
}

impl CandidateEvaluator for SarEvaluator<'_> {
    /// One trial's code-center sweep: the integer code per input in `vins`.
    type Output = Vec<i64>;

    fn evaluate(&self, index: usize, inner_parallel: bool) -> CandidateOutcome<Vec<i64>> {
        // Trials are the parallel level unless the batch handed the pool to the
        // inner sweep, so a trial's device batches are evaluated inline rather
        // than contending for the shared BSIM pool. A single-worker batch has no
        // outer parallelism to protect and keeps the pool.
        let _serial =
            (self.outer_parallel && !inner_parallel).then(co_bsim4::SerialEvalGuard::enter);
        let t = self.template;
        let trial = self
            .trials
            .get(index)
            .ok_or_else(|| format!("trial index {index} out of range"))?;
        if trial.delvto.len() != t.cards.len() {
            return Err(format!(
                "trial {index} has {} delvto entries, template has {} devices",
                trial.delvto.len(),
                t.cards.len()
            ));
        }
        if trial.cap_values.len() != t.circuit.capacitors.len() {
            return Err(format!(
                "trial {index} has {} cap values, template has {}",
                trial.cap_values.len(),
                t.circuit.capacitors.len()
            ));
        }

        // Patch the trial's CDAC capacitor perturbation onto a circuit clone.
        let mut circuit = t.circuit.clone();
        for (capacitor, &value) in circuit.capacitors.iter_mut().zip(&trial.cap_values) {
            capacitor.capacitance = value;
        }

        // This trial's handle set (fresh, delvto-adjusted) on this thread; the
        // evaluator rebuilds it per bit inside run_conversion.
        let mut evaluator = GuardEvaluator::new(&t.cards, &trial.delvto)
            .map_err(|error| format!("trial {index}: {error}"))?;
        let mut codes = Vec::with_capacity(t.vins.len());
        for &vin in &t.vins {
            let conversion = co_core::sar::run_conversion(
                &circuit,
                &t.devices,
                &mut evaluator,
                &t.config,
                &t.v0,
                &t.tgrid,
                &t.grid,
                vin,
                false,
            )
            .ok_or_else(|| format!("trial {index}: SAR conversion failed to converge"))?;
            codes.push(conversion.code as i64);
        }
        Ok(codes)
    }
}

struct SarTrajectoryEvaluator {
    template: Arc<SarTemplate>,
    trial: Arc<SarTrial>,
    ranges: Vec<core::ops::Range<usize>>,
    outer_parallel: bool,
    record_trajectory: bool,
}

fn balanced_ranges(item_count: usize, requested_workers: usize) -> Vec<core::ops::Range<usize>> {
    let chunk_count = requested_workers.max(1).min(item_count);
    (0..chunk_count)
        .map(|index| {
            let start = index * item_count / chunk_count;
            let end = (index + 1) * item_count / chunk_count;
            start..end
        })
        .collect()
}

impl CandidateEvaluator for SarTrajectoryEvaluator {
    type Output = Vec<co_core::sar::Conversion>;

    fn evaluate(&self, index: usize, _inner_parallel: bool) -> CandidateOutcome<Self::Output> {
        let _serial = self.outer_parallel.then(co_bsim4::SerialEvalGuard::enter);
        let range = self
            .ranges
            .get(index)
            .cloned()
            .ok_or_else(|| format!("SAR trajectory chunk {index} out of range"))?;
        let t = &self.template;
        let trial = &self.trial;
        let mut circuit = t.circuit.clone();
        for (capacitor, &value) in circuit.capacitors.iter_mut().zip(&trial.cap_values) {
            capacitor.capacitance = value;
        }
        let mut evaluator = GuardEvaluator::new(&t.cards, &trial.delvto)
            .map_err(|error| format!("trajectory chunk {index}: {error}"))?;
        range
            .map(|vin_index| {
                co_core::sar::run_conversion(
                    &circuit,
                    &t.devices,
                    &mut evaluator,
                    &t.config,
                    &t.v0,
                    &t.tgrid,
                    &t.grid,
                    t.vins[vin_index],
                    self.record_trajectory,
                )
                .ok_or_else(|| format!("SAR conversion {vin_index} failed to converge"))
            })
            .collect()
    }
}

fn evaluate_trial_conversions(
    template: Arc<SarTemplate>,
    trial: SarTrial,
    workers: usize,
    record_trajectory: bool,
) -> Result<Vec<co_core::sar::Conversion>, String> {
    if trial.delvto.len() != template.cards.len() {
        return Err(format!(
            "trial has {} delvto entries, template has {} devices",
            trial.delvto.len(),
            template.cards.len()
        ));
    }
    if trial.cap_values.len() != template.circuit.capacitors.len() {
        return Err(format!(
            "trial has {} cap values, template has {}",
            trial.cap_values.len(),
            template.circuit.capacitors.len()
        ));
    }
    // Produce exactly one non-empty, contiguous range per usable worker. The
    // former ceil-sized step occasionally produced fewer ranges than workers
    // (64 inputs / 12 workers became 11 ranges); campaign Auto then selected
    // its serial-candidate axis and turned that worker count into a performance
    // cliff. Quotient boundaries are also maximally balanced.
    let ranges = balanced_ranges(template.vins.len(), workers);
    let evaluator = SarTrajectoryEvaluator {
        template,
        trial: Arc::new(trial),
        outer_parallel: ranges.len() > 1,
        record_trajectory,
        ranges,
    };
    let progress = BatchProgress::new();
    let effective_workers = evaluator.ranges.len();
    let outcomes = evaluate_batch(
        &evaluator,
        evaluator.ranges.len(),
        BatchConfig::new(effective_workers),
        &progress,
    );
    let mut conversions = Vec::new();
    for (index, slot) in outcomes.into_iter().enumerate() {
        match slot {
            Some(Ok(mut chunk)) => conversions.append(&mut chunk),
            Some(Err(message)) => {
                return Err(format!("SAR trajectory chunk {index}: {message}"));
            }
            None => return Err(format!("SAR trajectory chunk {index} was skipped")),
        }
    }
    Ok(conversions)
}

#[cfg(test)]
mod tests {
    use super::balanced_ranges;

    #[test]
    fn sar_ranges_are_exactly_counted_balanced_and_contiguous() {
        for (items, workers, expected_count) in [(64, 12, 12), (64, 10, 10), (8, 16, 8), (1, 0, 1)]
        {
            let ranges = balanced_ranges(items, workers);
            assert_eq!(ranges.len(), expected_count);
            assert_eq!(ranges.first().unwrap().start, 0);
            assert_eq!(ranges.last().unwrap().end, items);
            assert!(ranges.iter().all(|range| !range.is_empty()));
            assert!(ranges.windows(2).all(|pair| pair[0].end == pair[1].start));
            let lengths = ranges.iter().map(core::ops::Range::len).collect::<Vec<_>>();
            assert!(lengths.iter().max().unwrap() - lengths.iter().min().unwrap() <= 1);
        }
    }
}

/// A compiled SAR conversion template exposed to Python.
#[pyclass(name = "CompiledSarConversion")]
pub struct PyCompiledSarConversion {
    template: Arc<SarTemplate>,
}

#[pymethods]
impl PyCompiledSarConversion {
    #[new]
    fn new(spec: &Bound<'_, PyDict>) -> PyResult<Self> {
        Ok(Self {
            template: Arc::new(build_sar_template(spec)?),
        })
    }

    /// Number of code-center inputs each trial converts.
    fn levels(&self) -> usize {
        self.template.vins.len()
    }

    /// Expanded integration grid shared by every conversion in this template.
    fn expanded_times(&self) -> Vec<f64> {
        self.template.grid.times.clone()
    }

    /// Expanded-grid positions corresponding to the public requested time grid.
    fn requested_index(&self) -> Vec<usize> {
        self.template.grid.requested_index.clone()
    }

    /// Exact Gear2/BE coefficients used on the expanded integration grid.
    fn integration_coefficients(&self) -> PyResult<Vec<Vec<f64>>> {
        bsim_transient::fixed_grid_integration_coefficients(
            &self.template.grid.times,
            self.template.config.newton.gear2,
        )
        .map(|rows| rows.into_iter().map(|row| row.to_vec()).collect::<Vec<_>>())
        .ok_or_else(|| PyRuntimeError::new_err("invalid SAR expanded time grid"))
    }

    /// Lightweight counter profile for one input in one physical trial.
    #[pyo3(signature = (trial, vin_index=0))]
    fn profile_conversion<'py>(
        &self,
        py: Python<'py>,
        trial: &Bound<'py, PyDict>,
        vin_index: usize,
    ) -> PyResult<Bound<'py, PyDict>> {
        let trial = parse_trial(trial)?;
        if vin_index >= self.template.vins.len() {
            return Err(PyValueError::new_err("vin_index out of range"));
        }
        if trial.delvto.len() != self.template.cards.len()
            || trial.cap_values.len() != self.template.circuit.capacitors.len()
        {
            return Err(PyValueError::new_err("trial shape does not match template"));
        }
        let template = Arc::clone(&self.template);
        let conversion = py
            .detach(move || -> Result<co_core::sar::Conversion, String> {
                let mut circuit = template.circuit.clone();
                for (capacitor, &value) in circuit.capacitors.iter_mut().zip(&trial.cap_values) {
                    capacitor.capacitance = value;
                }
                let mut evaluator = GuardEvaluator::new(&template.cards, &trial.delvto)?;
                let mut config = template.config.clone();
                config.newton.profile = true;
                co_core::sar::run_conversion(
                    &circuit,
                    &template.devices,
                    &mut evaluator,
                    &config,
                    &template.v0,
                    &template.tgrid,
                    &template.grid,
                    template.vins[vin_index],
                    false,
                )
                .ok_or_else(|| format!("SAR conversion {vin_index} failed to converge"))
            })
            .map_err(PyValueError::new_err)?;
        let profile = conversion.profile;
        let result = PyDict::new(py);
        result.set_item("code", conversion.code)?;
        result.set_item("grid_points", self.template.grid.times.len())?;
        result.set_item("newton_iterations", profile.newton_iterations)?;
        result.set_item("bsim_evaluations", profile.bsim_evaluations)?;
        result.set_item("bsim_batches", profile.bsim_batches)?;
        result.set_item("gear2_predictor_steps", profile.gear2_predictor_steps)?;
        result.set_item("failed_steps", profile.failed_steps)?;
        Ok(result)
    }

    /// Evaluate a batch of mismatch trials, returning each trial's code sweep in
    /// trial-index order (byte-identical for any worker count).
    #[pyo3(signature = (trials, workers=1))]
    fn evaluate_batch(
        &self,
        py: Python<'_>,
        trials: Vec<Bound<'_, PyDict>>,
        workers: usize,
    ) -> PyResult<Vec<Vec<i64>>> {
        let parsed: Vec<SarTrial> = trials.iter().map(parse_trial).collect::<PyResult<_>>()?;
        let template = Arc::clone(&self.template);
        let outcomes = py.detach(move || {
            let evaluator = SarEvaluator {
                template: &template,
                trials: &parsed,
                outer_parallel: workers > 1,
            };
            let progress = BatchProgress::new();
            evaluate_batch(
                &evaluator,
                parsed.len(),
                BatchConfig::new(workers),
                &progress,
            )
        });
        let mut results = Vec::with_capacity(outcomes.len());
        for (index, slot) in outcomes.into_iter().enumerate() {
            match slot {
                Some(Ok(codes)) => results.push(codes),
                Some(Err(message)) => {
                    return Err(PyValueError::new_err(format!(
                        "SAR trial {index}: {message}"
                    )));
                }
                None => {
                    return Err(PyValueError::new_err(format!(
                        "SAR trial {index} was skipped"
                    )));
                }
            }
        }
        Ok(results)
    }

    /// Evaluate one trial's input sweep without waveform history. Unlike
    /// `evaluate_batch`, this parallelizes within the trial when workers > 1.
    #[pyo3(signature = (trial, workers=1))]
    fn evaluate_codes(
        &self,
        py: Python<'_>,
        trial: &Bound<'_, PyDict>,
        workers: usize,
    ) -> PyResult<Vec<i64>> {
        if workers == 0 {
            return Err(PyValueError::new_err("workers must be positive"));
        }
        let trial = parse_trial(trial)?;
        let template = Arc::clone(&self.template);
        py.detach(move || evaluate_trial_conversions(template, trial, workers, false))
            .map(|conversions| {
                conversions
                    .into_iter()
                    .map(|conversion| conversion.code as i64)
                    .collect()
            })
            .map_err(PyValueError::new_err)
    }

    /// Evaluate one physical trial and return each input's complete accepted
    /// trajectory. This is the public ADC path: code-only campaign/MC callers
    /// continue to use `evaluate_batch` and allocate no waveform history.
    #[pyo3(signature = (trial, workers=1))]
    fn evaluate_trajectories<'py>(
        &self,
        py: Python<'py>,
        trial: &Bound<'py, PyDict>,
        workers: usize,
    ) -> PyResult<Bound<'py, PyList>> {
        if workers == 0 {
            return Err(PyValueError::new_err("workers must be positive"));
        }
        let trial = parse_trial(trial)?;
        let template = Arc::clone(&self.template);
        let sample_count = template.grid.times.len();
        let state_width = template.circuit.size;
        let device_count = template.devices.len();
        let input_count = template.config.roles.len();
        let conversions = py
            .detach(move || evaluate_trial_conversions(template, trial, workers, true))
            .map_err(PyValueError::new_err)?;

        let results = PyList::empty(py);
        for conversion in conversions {
            let trajectory = conversion.trajectory.ok_or_else(|| {
                PyRuntimeError::new_err("SAR detailed solve omitted its trajectory")
            })?;
            let inputs = Array2::from_shape_vec((input_count, sample_count), trajectory.inputs)
                .map_err(|error| {
                    PyRuntimeError::new_err(format!("invalid SAR input history shape: {error}"))
                })?
                .into_pyarray(py);
            let states = crate::rows_into_array2(
                py,
                trajectory.states,
                state_width,
                "SAR transient states",
            )?;
            let currents = crate::terminal_history_into_array3(
                py,
                trajectory.device_currents,
                sample_count,
                device_count,
                "SAR transient device currents",
            )?;
            let charges = crate::terminal_history_into_array3(
                py,
                trajectory.device_charges,
                sample_count,
                device_count,
                "SAR transient device charges",
            )?;
            let item = PyDict::new(py);
            item.set_item("code", conversion.code)?;
            item.set_item("bits", conversion.bits)?;
            item.set_item("inputs", inputs)?;
            item.set_item("states", states)?;
            item.set_item("device_currents", currents)?;
            item.set_item("device_charges", charges)?;
            item.set_item("failures", trajectory.failures)?;
            item.set_item(
                "first_failure",
                trajectory.first_failure.map_or(-1, |value| value as i64),
            )?;
            results.append(item)?;
        }
        Ok(results)
    }
}

fn parse_trial(item: &Bound<'_, PyDict>) -> PyResult<SarTrial> {
    Ok(SarTrial {
        delvto: required(item, "delvto")?,
        cap_values: required(item, "cap_values")?,
    })
}

type BsimDeviceRecord = (Vec<TermRecord>, Vec<i64>);
type DeviceCardRecord = (i32, f64, Vec<(String, f64)>, Vec<(String, f64)>);

fn parse_role(record: (i64, i64)) -> PyResult<Role> {
    Ok(match record.0 {
        0 => Role::Sample,
        1 => Role::SampleBar,
        2 => Role::BitInput(record.1 as usize),
        3 => Role::BitInputBar(record.1 as usize),
        4 => Role::Dummy,
        5 => Role::DummyBar,
        6 => Role::Clock,
        7 => Role::ClockBar,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown SAR role kind {other}"
            )));
        }
    })
}

fn build_sar_template(spec: &Bound<'_, PyDict>) -> PyResult<SarTemplate> {
    // Passive circuit (base capacitor values), with device slots cleared.
    let circuit_obj = spec
        .get_item("circuit")?
        .ok_or_else(|| PyKeyError::new_err("circuit"))?;
    let circuit_ref = circuit_obj.extract::<PyRef<'_, crate::OtftTransientProblem>>()?;
    let mut circuit = circuit_ref.problem.clone();
    circuit.devices.clear();
    if !circuit.validate() {
        return Err(PyValueError::new_err("invalid SAR conversion circuit"));
    }

    // BSIM device topology (d, g, s, bulk terms + solved rows).
    let device_records: Vec<BsimDeviceRecord> = required(spec, "bsim_devices")?;
    let mut devices = Vec::with_capacity(device_records.len());
    for (index, (terms, rows)) in device_records.into_iter().enumerate() {
        if terms.len() != 4 || rows.len() != 4 {
            return Err(PyValueError::new_err(
                "each SAR BSIM device needs four terms and four rows",
            ));
        }
        devices.push(bsim_transient::Device {
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

    // Frozen device cards (model/instance parameters), one per device.
    let card_records: Vec<DeviceCardRecord> = required(spec, "device_cards")?;
    if card_records.len() != devices.len() {
        return Err(PyValueError::new_err(
            "device_cards and bsim_devices must have equal length",
        ));
    }
    let mut cards = Vec::with_capacity(card_records.len());
    for (polarity, temperature_k, model, instance) in card_records {
        if polarity != 1 && polarity != -1 {
            return Err(PyValueError::new_err("device polarity must be +1 or -1"));
        }
        cards.push(DeviceCard {
            polarity,
            temperature_k,
            model,
            instance,
        });
    }

    let role_records: Vec<(i64, i64)> = required(spec, "roles")?;
    let roles = role_records
        .into_iter()
        .map(parse_role)
        .collect::<PyResult<Vec<_>>>()?;

    let newton: Vec<f64> = required(spec, "newton")?;
    if newton.len() != 5 {
        return Err(PyValueError::new_err(
            "newton must be [gear2, max_iterations, voltage_tolerance, step_limit, gmin]",
        ));
    }
    let options = Options {
        gear2: newton[0] != 0.0,
        max_iterations: newton[1] as usize,
        voltage_tolerance: newton[2],
        step_limit: newton[3],
        gmin: newton[4],
        record_device_history: false,
        profile: false,
    };

    let clock: Option<Vec<f64>> = crate::optional_field(spec, "clock")?;
    let clock = match clock {
        Some(values) if values.len() == 4 => Some(ClockConfig {
            high: values[0],
            low: values[1],
            eval_before: values[2],
            reset_hold: values[3],
        }),
        Some(_) => {
            return Err(PyValueError::new_err(
                "clock must be [high, low, eval_before, reset_hold]",
            ));
        }
        None => None,
    };

    let tgrid: Vec<f64> = required(spec, "tgrid")?;
    if tgrid.len() < 2 {
        return Err(PyValueError::new_err("tgrid needs at least two points"));
    }
    let comparator_index: usize = required(spec, "comparator_index")?;
    if comparator_index >= circuit.node_count {
        return Err(PyValueError::new_err("comparator_index out of range"));
    }

    let config = SarConfig {
        n_bits: required(spec, "n_bits")?,
        vref: required(spec, "vref")?,
        sample_end: required(spec, "sample_end")?,
        bit_period: required(spec, "bit_period")?,
        edge_time: required(spec, "edge_time")?,
        input_common_mode: required(spec, "input_common_mode")?,
        comparator_threshold: required(spec, "comparator_threshold")?,
        high_means_clear: required(spec, "high_means_clear")?,
        differential: required(spec, "differential")?,
        comparator_index,
        tstop: tgrid[tgrid.len() - 1],
        clock,
        roles,
        newton: options,
    };
    if config.roles.len() != circuit.node_count && config.roles.is_empty() {
        return Err(PyValueError::new_err(
            "SAR template needs at least one input role",
        ));
    }

    let v0: Vec<f64> = required(spec, "v0")?;
    if v0.len() != circuit.size {
        return Err(PyValueError::new_err("v0 length must equal circuit size"));
    }
    let vins: Vec<f64> = required(spec, "vins")?;
    if vins.is_empty() {
        return Err(PyValueError::new_err("vins must be non-empty"));
    }

    let grid = ExpandedGrid::build(&tgrid, config.edge_time);
    // Validate the fixed-grid topology once, up front, against a zero stimulus.
    let zero_waveform = vec![0.0; config.roles.len() * grid.times.len()];
    let waveforms = transient::Waveforms::new(&zero_waveform, config.roles.len(), grid.times.len())
        .ok_or_else(|| PyValueError::new_err("invalid SAR waveform shape"))?;
    bsim_transient::validate_fixed_grid_input(&circuit, &devices, &v0, &grid.times, waveforms)
        .map_err(crate::core_error)?;

    Ok(SarTemplate {
        circuit,
        devices,
        cards,
        config,
        v0,
        tgrid,
        grid,
        vins,
    })
}
