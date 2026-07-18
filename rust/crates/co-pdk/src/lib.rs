//! co-pdk — FreePDK45 / SKY130 / TSMC28 PDK compilers.
//!
//! Each adapter normalizes corner/temperature/polarity, resolves a local card,
//! applies the geometry-bin and `nf`/`mult`/`mismatch` instance rules, and
//! returns a numeric BSIM4 card ([`NumericCard`]). All deck parsing, elaboration
//! and expression evaluation reuse `co-spice`; no second parser or evaluator is
//! introduced. The compilers are 1:1 ports of the frozen Python adapters
//! (`circuitopt.pdk.{freepdk45,sky130,tsmc28}`).
//!
//! A [`CompiledPdk`] is immutable configuration (PDK kind + optional root) with
//! an interior, thread-safe in-memory cache. Cache keys follow D12: a canonical
//! path plus its mtime/size, and — for TSMC28 — the elaborated section set and
//! temperature. No card content is ever persisted; the cache is process-local.

mod freepdk45;
mod gformat;
mod json;
mod sky130;
mod tsmc28;

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex};
use std::time::UNIX_EPOCH;

use co_spice::ErrorKind as SpiceErrorKind;

/// A PDK compilation failure. `kind` is `Some` when it originated in `co-spice`
/// (so the boundary can re-raise the matching Python class); `None` marks a
/// PDK-specific model error (a plain `ValueError` in the reference adapters).
#[derive(Debug, Clone)]
pub struct PdkError {
    pub kind: Option<SpiceErrorKind>,
    pub message: String,
}

impl PdkError {
    /// A PDK-specific model error (`Freepdk45ModelError` / `Sky130ModelError` /
    /// `Tsmc28ModelError` in the reference — all `ValueError` subclasses).
    pub fn model(message: impl Into<String>) -> Self {
        Self {
            kind: None,
            message: message.into(),
        }
    }
}

impl fmt::Display for PdkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for PdkError {}

impl From<co_spice::SpiceError> for PdkError {
    fn from(error: co_spice::SpiceError) -> Self {
        Self {
            kind: Some(error.kind),
            message: error.message,
        }
    }
}

pub type PdkResult<T> = Result<T, PdkError>;

/// Optional geometry-bin descriptor (TSMC28 only).
#[derive(Debug, Clone)]
pub struct Bin {
    pub name: String,
    pub lmin: f64,
    pub lmax: f64,
    pub wmin: f64,
    pub wmax: f64,
}

/// Provenance for a numeric card: paths and section/bin identifiers only — never
/// card parameter text.
#[derive(Debug, Clone, Default)]
pub struct Source {
    pub pdk: String,
    pub polarity: String,
    pub corner: String,
    pub path: String,
    pub temperature_c: Option<f64>,
    pub macro_name: Option<String>,
    pub bin_name: Option<String>,
}

/// A compiled numeric BSIM4 card: model parameters, the instance parameters, and
/// provenance. Mirrors the Python `*Card` dataclasses' `model_parameters` /
/// `instance_parameters` (pre-`to_bsim4_cards`; the TSMC28 `mulu0 -> u0` fold is
/// a documented downstream step).
#[derive(Debug, Clone)]
pub struct NumericCard {
    pub model_parameters: HashMap<String, f64>,
    pub instance_parameters: HashMap<String, f64>,
    pub model_name: String,
    pub model_type: String,
    pub source_version: f64,
    pub bin: Option<Bin>,
    pub source: Source,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PdkKind {
    Freepdk45,
    Sky130,
    Tsmc28,
}

/// `(canonical path, mtime_ns, size)` — the D12 file identity.
pub(crate) type FileKey = (String, u128, u64);

#[derive(Default)]
pub(crate) struct Cache {
    pub(crate) fp45: Mutex<HashMap<FileKey, Arc<freepdk45::Card>>>,
    pub(crate) sky: Mutex<HashMap<FileKey, Arc<HashMap<String, f64>>>>,
    pub(crate) tsmc: Mutex<tsmc28::TsmcState>,
}

/// Immutable PDK configuration with a thread-safe in-memory card/program cache.
pub struct CompiledPdk {
    kind: PdkKind,
    root: Option<String>,
    cache: Cache,
}

impl CompiledPdk {
    /// Build a compiler for `pdk` (`"freepdk45"`, `"sky130"`, or `"tsmc28"`).
    ///
    /// `root` locates the local delivery:
    /// * freepdk45 — the `PDK_ROOT` directory (holds `freepdk45/models_*/`);
    /// * sky130 — the resolved card directory (holds `*.json`);
    /// * tsmc28 — the HSPICE model directory (holds the `.l` delivery).
    pub fn new(pdk: &str, root: Option<String>) -> PdkResult<Self> {
        let kind = match pdk.to_lowercase().as_str() {
            "freepdk45" => PdkKind::Freepdk45,
            "sky130" => PdkKind::Sky130,
            "tsmc28" | "tsmc28hpcp" => PdkKind::Tsmc28,
            other => {
                return Err(PdkError::model(format!(
                    "unknown PDK {other:?}; expected freepdk45, sky130, or tsmc28"
                )));
            }
        };
        Ok(Self {
            kind,
            root,
            cache: Cache::default(),
        })
    }

    pub(crate) fn root(&self) -> Option<&str> {
        self.root.as_deref()
    }

    pub(crate) fn cache(&self) -> &Cache {
        &self.cache
    }

    /// Compile one numeric card. `temp_c` is used only by TSMC28; `w_um`/`l_um`
    /// are required by every PDK (a positive geometry). `mismatch` is `None` for
    /// no threshold offset.
    #[allow(clippy::too_many_arguments)]
    pub fn numeric_card(
        &self,
        polarity: &str,
        corner: &str,
        temp_c: f64,
        w_um: Option<f64>,
        l_um: Option<f64>,
        nf: i64,
        mult: i64,
        mismatch: Option<f64>,
    ) -> PdkResult<NumericCard> {
        match self.kind {
            PdkKind::Freepdk45 => {
                freepdk45::numeric_card(self, polarity, corner, w_um, l_um, nf, mult, mismatch)
            }
            PdkKind::Sky130 => {
                sky130::numeric_card(self, polarity, corner, w_um, l_um, nf, mult, mismatch)
            }
            PdkKind::Tsmc28 => tsmc28::numeric_card(
                self, polarity, corner, temp_c, w_um, l_um, nf, mult, mismatch,
            ),
        }
    }
}

/// The D12 file identity: `(path, mtime_ns, size)`. Errors if the file is
/// missing, with the caller-supplied "not found" message.
pub(crate) fn file_key(path: &str, missing: impl FnOnce() -> String) -> PdkResult<FileKey> {
    let meta = match std::fs::metadata(path) {
        Ok(meta) if meta.is_file() => meta,
        _ => return Err(PdkError::model(missing())),
    };
    let mtime = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    Ok((path.to_string(), mtime, meta.len()))
}

/// Read a SPICE deck as the reference does (`encoding="ascii"`): reject any
/// non-ASCII byte.
pub(crate) fn read_ascii_file(path: &str, missing: impl FnOnce() -> String) -> PdkResult<String> {
    let bytes = std::fs::read(path).map_err(|_| PdkError::model(missing()))?;
    if bytes.iter().any(|b| *b >= 0x80) {
        return Err(PdkError::model(format!(
            "{path}: non-ASCII byte in model card"
        )));
    }
    Ok(String::from_utf8(bytes).expect("ascii bytes are valid utf-8"))
}

/// `os.path.expanduser` for a leading `~` / `~/`, then leave the path as given
/// (the reference resolvers already hand us absolute roots).
pub(crate) fn expanduser(path: &str) -> String {
    if path == "~"
        && let Ok(home) = std::env::var("HOME")
    {
        return home;
    }
    if let Some(rest) = path.strip_prefix("~/")
        && let Ok(home) = std::env::var("HOME")
    {
        return format!("{home}/{rest}");
    }
    path.to_string()
}

/// The TSMC28 `to_bsim4_cards()` `mulu0 -> u0` mobility fold
/// (`circuitopt/pdk/tsmc28/library.py`): HSPICE's instance-only `mulu0`
/// extension is always removed from the instance parameters; when it is
/// non-unity the selected model's `u0` is multiplied by it (an error when the
/// card carries no `u0`). Exposed for the compiled-campaign device build.
pub fn apply_mulu0_fold(
    model_parameters: &mut HashMap<String, f64>,
    instance_parameters: &mut HashMap<String, f64>,
) -> PdkResult<()> {
    let mobility_multiplier = instance_parameters.remove("mulu0").unwrap_or(1.0);
    if mobility_multiplier != 1.0 {
        match model_parameters.get_mut("u0") {
            Some(u0) => *u0 *= mobility_multiplier,
            None => {
                return Err(PdkError::model(
                    "mulu0 is non-unity but the selected BSIM4 card has no u0",
                ));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod mulu0_tests {
    use super::*;

    #[test]
    fn non_unity_mulu0_multiplies_u0_exactly() {
        let mut model = HashMap::from([("u0".to_string(), 0.018202628)]);
        let mut instance = HashMap::from([("mulu0".to_string(), 1.07), ("w".to_string(), 3e-6)]);
        apply_mulu0_fold(&mut model, &mut instance).unwrap();
        assert_eq!(model["u0"], 0.018202628 * 1.07); // exact IEEE product
        assert!(!instance.contains_key("mulu0"));
        assert_eq!(instance["w"], 3e-6);
    }

    #[test]
    fn unity_mulu0_is_popped_but_u0_untouched() {
        let mut model = HashMap::from([("u0".to_string(), 0.03)]);
        let mut instance = HashMap::from([("mulu0".to_string(), 1.0)]);
        apply_mulu0_fold(&mut model, &mut instance).unwrap();
        assert_eq!(model["u0"], 0.03);
        assert!(!instance.contains_key("mulu0"));
    }

    #[test]
    fn absent_mulu0_is_a_no_op() {
        let mut model = HashMap::from([("u0".to_string(), 0.03)]);
        let mut instance = HashMap::from([("w".to_string(), 1e-6)]);
        apply_mulu0_fold(&mut model, &mut instance).unwrap();
        assert_eq!(model["u0"], 0.03);
    }

    #[test]
    fn non_unity_mulu0_without_u0_errors() {
        let mut model = HashMap::new();
        let mut instance = HashMap::from([("mulu0".to_string(), 1.2)]);
        let error = apply_mulu0_fold(&mut model, &mut instance).unwrap_err();
        assert!(error.message.contains("mulu0"));
    }
}
