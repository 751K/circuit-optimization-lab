//! FreePDK45 flat BSIM4 card loading — a 1:1 port of
//! `circuitopt.pdk.freepdk45.library`.
//!
//! Each corner/polarity resolves one flat `.inc` card containing exactly one
//! `.model`. Its parameters are read with `parse_spice_number` (no expression
//! evaluation), and the instance parameters follow the same arithmetic and
//! `delvto` mismatch rule as `Freepdk45Library.device_card`.

use std::collections::HashMap;
use std::sync::Arc;

use co_spice::{parse_spice_library_text, parse_spice_number};

use crate::{
    CompiledPdk, NumericCard, PdkError, PdkResult, Source, expanduser, file_key, read_ascii_file,
};

/// A parsed, validated flat card (the cached `Freepdk45Library` payload).
#[derive(Debug, Clone)]
pub struct Card {
    pub model_name: String,
    pub model_parameters: HashMap<String, f64>,
    pub source_version: f64,
}

fn model_name(polarity: &str) -> &'static str {
    match polarity {
        "nmos" => "NMOS_VTG",
        "pmos" => "PMOS_VTG",
        _ => unreachable!("polarity is normalized"),
    }
}

/// `_CORNER_DIRS`: `(nmos_dir, pmos_dir)` per corner.
fn corner_dirs(corner: &str) -> (&'static str, &'static str) {
    match corner {
        "nom" | "tt" => ("nom", "nom"),
        "ss" => ("ss", "ss"),
        "ff" => ("ff", "ff"),
        "sf" => ("ss", "ff"),
        "fs" => ("ff", "ss"),
        _ => unreachable!("corner is normalized"),
    }
}

/// `normalize_corner`: empty -> `nom`; validated against `_CORNER_DIRS`.
fn normalize_corner(corner: &str) -> PdkResult<String> {
    if corner.is_empty() {
        return Ok("nom".to_string());
    }
    let key = corner.to_lowercase();
    if matches!(key.as_str(), "nom" | "tt" | "ss" | "ff" | "sf" | "fs") {
        Ok(key)
    } else {
        Err(PdkError::model(format!(
            "unknown FreePDK45 corner {corner:?}; expected one of \
             ['ff', 'fs', 'nom', 'sf', 'ss', 'tt']"
        )))
    }
}

/// `normalize_polarity`: aliases n/nfet/p/pfet, else error.
fn normalize_polarity(polarity: &str) -> PdkResult<&'static str> {
    match polarity.to_lowercase().as_str() {
        "nmos" | "n" | "nfet" => Ok("nmos"),
        "pmos" | "p" | "pfet" => Ok("pmos"),
        _ => Err(PdkError::model(format!(
            "unknown FreePDK45 polarity {polarity:?}; expected nmos or pmos"
        ))),
    }
}

/// `freepdk45_card_path`: `{root}/freepdk45/models_{dir}/{MODEL}.inc`.
fn card_path(root: &str, polarity: &str, corner: &str) -> String {
    let (nmos_dir, pmos_dir) = corner_dirs(corner);
    let dir = if polarity == "nmos" {
        nmos_dir
    } else {
        pmos_dir
    };
    format!("{root}/freepdk45/models_{dir}/{}.inc", model_name(polarity))
}

fn parse_and_validate(path: &str, polarity: &str) -> PdkResult<Card> {
    let missing = || format!("FreePDK45 model card not found: {path}; set PDK_ROOT");
    let text = read_ascii_file(path, missing)?;
    let library = parse_spice_library_text(&text, path)?;
    let models = library.top_level.models();
    if models.len() != 1 {
        return Err(PdkError::model(format!(
            "{path} must contain exactly one .model statement"
        )));
    }
    let statement = models.values().next().expect("one model");
    let expected = model_name(polarity);
    let model_type = statement
        .arguments
        .first()
        .map(|s| s.to_lowercase())
        .unwrap_or_default();
    if statement.name.as_deref() != Some(expected) || model_type != polarity {
        return Err(PdkError::model(format!(
            "{path} defines {:?}/{:?}, expected {:?}/{:?}",
            statement.name.clone().unwrap_or_default(),
            model_type,
            expected,
            polarity
        )));
    }
    let mut parameters: HashMap<String, f64> = HashMap::new();
    for assignment in &statement.parameters {
        let value = parse_spice_number(&assignment.expression).map_err(|_| {
            // D12: name only, never the parameter value/expression text.
            PdkError::model(format!(
                "{path} has non-numeric model parameter {}",
                assignment.name
            ))
        })?;
        parameters.insert(assignment.name.to_lowercase(), value);
    }
    // int(parameters.get("level", -1)) != 54  (int() truncates toward zero).
    let level = parameters.get("level").copied().unwrap_or(-1.0) as i64;
    if level != 54 {
        return Err(PdkError::model(format!(
            "{path} is not a BSIM4 level-54 card"
        )));
    }
    let version = parameters.get("version").copied().unwrap_or(-1.0);
    if version != 4.0 {
        return Err(PdkError::model(format!(
            "{path} uses unsupported BSIM4 version {version}"
        )));
    }
    Ok(Card {
        model_name: expected.to_string(),
        model_parameters: parameters,
        source_version: version,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn numeric_card(
    pdk: &CompiledPdk,
    polarity: &str,
    corner: &str,
    w_um: Option<f64>,
    l_um: Option<f64>,
    nf: i64,
    mult: i64,
    mismatch: Option<f64>,
) -> PdkResult<NumericCard> {
    let polarity = normalize_polarity(polarity)?;
    let corner = normalize_corner(corner)?;
    let root = pdk
        .root()
        .ok_or_else(|| PdkError::model("FreePDK45 requires a PDK root (set PDK_ROOT)"))?;
    let path = expanduser(&card_path(root, polarity, &corner));

    let key = file_key(&path, || {
        format!("FreePDK45 model card not found: {path}; set PDK_ROOT")
    })?;
    let card = {
        let mut cache = pdk.cache().fp45.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(existing) = cache.get(&key) {
            existing.clone()
        } else {
            let parsed = Arc::new(parse_and_validate(&path, polarity)?);
            cache.insert(key, parsed.clone());
            parsed
        }
    };

    let (width_um, length_um) = require_geometry(w_um, l_um)?;
    if width_um <= 0.0 || length_um <= 0.0 {
        return Err(PdkError::model(
            "FreePDK45 width and length must be positive",
        ));
    }
    if nf < 1 || mult < 1 {
        return Err(PdkError::model(
            "FreePDK45 nf and mult must be positive integers",
        ));
    }
    let mut instance: HashMap<String, f64> = HashMap::new();
    instance.insert("w".into(), width_um * 1e-6);
    instance.insert("l".into(), length_um * 1e-6);
    instance.insert("nf".into(), nf as f64);
    instance.insert("m".into(), mult as f64);
    let mismatch_v = mismatch.unwrap_or(0.0);
    if mismatch_v != 0.0 {
        instance.insert("delvto".into(), mismatch_v);
    }

    Ok(NumericCard {
        model_parameters: card.model_parameters.clone(),
        instance_parameters: instance,
        model_name: card.model_name.clone(),
        model_type: polarity.to_string(),
        source_version: card.source_version,
        bin: None,
        source: Source {
            pdk: "freepdk45".into(),
            polarity: polarity.into(),
            corner,
            path,
            ..Source::default()
        },
    })
}

/// Unpack a required geometry (positivity is checked per-PDK, in reference order).
pub(crate) fn require_geometry(w_um: Option<f64>, l_um: Option<f64>) -> PdkResult<(f64, f64)> {
    let width = w_um.ok_or_else(|| PdkError::model("w_um is required"))?;
    let length = l_um.ok_or_else(|| PdkError::model("l_um is required"))?;
    Ok((width, length))
}
