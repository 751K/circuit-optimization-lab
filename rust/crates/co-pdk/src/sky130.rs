//! SKY130 resolved-card loading — a 1:1 port of `circuitopt.pdk.sky130.library`.
//!
//! The model parameters are read from a bundled flat JSON card whose filename
//! encodes polarity/corner/geometry (`W{width:g}_L{length:g}`). Instance
//! parameters follow the same arithmetic and `delvto` mismatch rule as
//! `load_sky130_card`.

use std::collections::HashMap;
use std::sync::Arc;

use crate::freepdk45::require_geometry;
use crate::gformat::format_g;
use crate::json::parse_flat_object;
use crate::{CompiledPdk, NumericCard, PdkError, PdkResult, Source, expanduser, file_key};

fn subckt(polarity: &str) -> &'static str {
    match polarity {
        "nmos" => "sky130_fd_pr__nfet_01v8",
        "pmos" => "sky130_fd_pr__pfet_01v8",
        _ => unreachable!("polarity is normalized"),
    }
}

/// `normalize_corner`: `None`/`""`/`nom` -> `tt`; validated against the five
/// SKY130 corners.
fn normalize_corner(corner: &str) -> PdkResult<String> {
    let key = if matches!(corner, "" | "nom") {
        "tt".to_string()
    } else {
        corner.to_lowercase()
    };
    if matches!(key.as_str(), "tt" | "ss" | "ff" | "sf" | "fs") {
        Ok(key)
    } else {
        Err(PdkError::model(format!(
            "unknown SKY130 corner {corner:?}; expected one of \
             ('tt', 'ss', 'ff', 'sf', 'fs')"
        )))
    }
}

fn normalize_polarity(polarity: &str) -> PdkResult<&'static str> {
    match polarity.to_lowercase().as_str() {
        "nmos" | "n" | "nfet" => Ok("nmos"),
        "pmos" | "p" | "pfet" => Ok("pmos"),
        _ => Err(PdkError::model(format!(
            "unknown SKY130 polarity {polarity:?}; expected nmos or pmos"
        ))),
    }
}

/// `sky130_card_filename`.
fn card_filename(polarity: &str, corner: &str, width_um: f64, length_um: f64) -> String {
    format!(
        "{}_{corner}_W{}_L{}.json",
        subckt(polarity),
        format_g(width_um),
        format_g(length_um)
    )
}

fn read_model_parameters(path: &str) -> PdkResult<HashMap<String, f64>> {
    let text = std::fs::read_to_string(path)
        .map_err(|_| PdkError::model(format!("invalid resolved SKY130 card: {path}")))?;
    let pairs = parse_flat_object(&text)
        .map_err(|_| PdkError::model(format!("invalid resolved SKY130 card: {path}")))?;
    let mut parameters: HashMap<String, f64> = HashMap::new();
    for (name, value) in pairs {
        parameters.insert(name.to_lowercase(), value);
    }
    if !parameters.contains_key("vth0") {
        return Err(PdkError::model(format!(
            "resolved SKY130 card has no vth0 parameter: {path}"
        )));
    }
    let version = parameters.get("version").copied().unwrap_or(-1.0);
    if version != 4.5 {
        return Err(PdkError::model(format!(
            "{path} uses unsupported BSIM4 version {version}"
        )));
    }
    Ok(parameters)
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
    let (width_um, length_um) = require_geometry(w_um, l_um)?;
    // reference_width defaults to width; card filename bins on the reference width.
    let reference_width_um = width_um;
    if width_um <= 0.0 || length_um <= 0.0 || reference_width_um <= 0.0 {
        return Err(PdkError::model(
            "SKY130 widths and lengths must be positive",
        ));
    }
    if nf < 1 || mult < 1 {
        return Err(PdkError::model(
            "SKY130 nf and mult must be positive integers",
        ));
    }

    let dir = pdk
        .root()
        .ok_or_else(|| PdkError::model("SKY130 requires a card directory root"))?;
    let filename = card_filename(polarity, &corner, reference_width_um, length_um);
    let path = expanduser(&format!("{dir}/{filename}"));
    let key = file_key(&path, || {
        format!(
            "resolved SKY130 BSIM4 card {filename:?} was not found in {dir}. \
             Use a bundled geometry, set SKY130_CARD_DIR, or explicitly generate \
             an oracle card."
        )
    })?;

    let model_parameters = {
        let mut cache = pdk.cache().sky.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(existing) = cache.get(&key) {
            existing.clone()
        } else {
            let parsed = Arc::new(read_model_parameters(&path)?);
            cache.insert(key, parsed.clone());
            parsed
        }
    };

    let mut instance: HashMap<String, f64> = HashMap::new();
    instance.insert("w".into(), width_um * 1e-6);
    instance.insert("l".into(), length_um * 1e-6);
    instance.insert("nf".into(), nf as f64);
    instance.insert("m".into(), mult as f64);
    let mismatch_v = mismatch.unwrap_or(0.0);
    if mismatch_v != 0.0 {
        instance.insert("delvto".into(), mismatch_v);
    }

    let model_name = filename
        .strip_suffix(".json")
        .unwrap_or(&filename)
        .to_string();

    Ok(NumericCard {
        model_parameters: (*model_parameters).clone(),
        instance_parameters: instance,
        model_name,
        model_type: polarity.to_string(),
        source_version: 4.5,
        bin: None,
        source: Source {
            pdk: "sky130".into(),
            polarity: polarity.into(),
            corner,
            path,
            ..Source::default()
        },
    })
}
