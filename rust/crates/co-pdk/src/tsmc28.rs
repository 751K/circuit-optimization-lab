//! TSMC28HPC+ core-MOS elaboration — a 1:1 port of
//! `circuitopt.pdk.tsmc28.library`.
//!
//! The licensed HSPICE `.l` delivery is parsed once and elaborated per
//! (corner, temperature). A macro (`nch_mac`/`pch_mac`) is instantiated at the
//! requested geometry, its single MOS element supplies the instance parameters,
//! and its `.model` bins are selected by the same half-open (then inclusive)
//! width/length rule as `Tsmc28CoreLibrary._select_bin`.

use std::collections::HashMap;
use std::sync::Arc;

use co_spice::{
    ElaboratedLibrary, ErrorKind as SpiceErrorKind, ParamValue, SpiceModelLibrary, Statement,
    SubcircuitInstance, elaborate_library, parse_spice_library_text,
};

use crate::{
    Bin, CompiledPdk, FileKey, NumericCard, PdkError, PdkResult, Source, expanduser, file_key,
    read_ascii_file,
};

const MODEL_FILE: &str = "cln28hpcp_1d8_elk_v1d0_2p2.l";
const BIN_PARAMETERS: [&str; 4] = ["lmin", "lmax", "wmin", "wmax"];

/// Cached parsed library plus per-(corner, temperature) elaborated programs.
#[derive(Default)]
pub struct TsmcState {
    lib_key: Option<FileKey>,
    library: Option<Arc<SpiceModelLibrary>>,
    programs: HashMap<(String, u64), Arc<ElaboratedLibrary>>,
}

fn macro_name(polarity: &str) -> &'static str {
    match polarity {
        "nmos" => "nch_mac",
        "pmos" => "pch_mac",
        _ => unreachable!("polarity is normalized"),
    }
}

/// `_normalize_corner`: `nom` -> `tt`; validated against the five core corners.
fn normalize_corner(corner: &str) -> PdkResult<String> {
    let mut key = corner.to_lowercase();
    if key == "nom" {
        key = "tt".to_string();
    }
    if matches!(key.as_str(), "tt" | "ss" | "ff" | "sf" | "fs") {
        Ok(key)
    } else {
        Err(PdkError::model(format!(
            "unknown TSMC28 core corner {corner:?}; expected one of \
             ('tt', 'ss', 'ff', 'sf', 'fs')"
        )))
    }
}

fn normalize_polarity(polarity: &str) -> PdkResult<&'static str> {
    match polarity.to_lowercase().as_str() {
        "nmos" | "n" | "nfet" => Ok("nmos"),
        "pmos" | "p" | "pfet" => Ok("pmos"),
        _ => Err(PdkError::model(format!(
            "unknown TSMC28 core polarity {polarity:?}; expected nmos or pmos"
        ))),
    }
}

/// `_select_bin`: half-open `lmin<=l<lmax`, `wmin<=w<wmax`; if none match and
/// exactly one bin is inclusive-closed, use it. Exactly one bin must remain.
fn select_bin(
    instance: &SubcircuitInstance,
    width_m: f64,
    length_m: f64,
) -> PdkResult<(&Statement, HashMap<String, f64>)> {
    let names: Vec<String> = BIN_PARAMETERS.iter().map(|s| s.to_string()).collect();
    let mut candidates: Vec<(&Statement, HashMap<String, f64>)> = Vec::new();
    let mut inclusive: Vec<(&Statement, HashMap<String, f64>)> = Vec::new();
    for statement in instance.model_statements() {
        let bounds = instance.numeric_model(statement, Some(&names))?.parameters;
        let lmin = *bounds
            .get("lmin")
            .ok_or_else(|| PdkError::model("bin lacks lmin"))?;
        let lmax = *bounds
            .get("lmax")
            .ok_or_else(|| PdkError::model("bin lacks lmax"))?;
        let wmin = *bounds
            .get("wmin")
            .ok_or_else(|| PdkError::model("bin lacks wmin"))?;
        let wmax = *bounds
            .get("wmax")
            .ok_or_else(|| PdkError::model("bin lacks wmax"))?;
        let half_open = lmin <= length_m && length_m < lmax && wmin <= width_m && width_m < wmax;
        let closed = lmin <= length_m && length_m <= lmax && wmin <= width_m && width_m <= wmax;
        if half_open {
            candidates.push((statement, bounds.clone()));
        }
        if closed {
            inclusive.push((statement, bounds));
        }
    }
    if candidates.is_empty() && inclusive.len() == 1 {
        candidates = inclusive;
    }
    if candidates.len() != 1 {
        let names: Vec<String> = candidates
            .iter()
            .map(|(s, _)| s.name.clone().unwrap_or_default())
            .collect();
        return Err(PdkError::model(format!(
            "geometry W={width_m} m L={length_m} m selects {} bins ({:?}); \
             expected exactly one",
            candidates.len(),
            names
        )));
    }
    let (statement, bounds) = candidates.into_iter().next().expect("one bin");
    Ok((statement, bounds))
}

#[allow(clippy::too_many_arguments)]
pub fn numeric_card(
    pdk: &CompiledPdk,
    polarity: &str,
    corner: &str,
    temp_c: f64,
    w_um: Option<f64>,
    l_um: Option<f64>,
    nf: i64,
    mult: i64,
    mismatch: Option<f64>,
) -> PdkResult<NumericCard> {
    let polarity = normalize_polarity(polarity)?;
    let corner = normalize_corner(corner)?;
    let width_um = w_um.ok_or_else(|| PdkError::model("w_um is required"))?;
    let length_um = l_um.ok_or_else(|| PdkError::model("l_um is required"))?;
    if width_um <= 0.0 || length_um <= 0.0 {
        return Err(PdkError::model(
            "core MOS width and length must be positive",
        ));
    }
    if nf < 1 || mult < 1 {
        return Err(PdkError::model(
            "core MOS nf and mult must be positive integers",
        ));
    }

    let root = pdk
        .root()
        .ok_or_else(|| PdkError::model("TSMC28 requires a model directory root"))?;
    let path = expanduser(&format!("{root}/{MODEL_FILE}"));
    fn missing_message(path: &str) -> String {
        format!(
            "TSMC28HPC+ HSPICE model not found: {path}; set \
             TSMC28_MODEL_DIR/TSMC28_PDK_ROOT or install the local model"
        )
    }
    let key = file_key(&path, || missing_message(&path))?;

    // Parse once per file identity; elaborate once per (corner, temperature).
    let program = {
        let mut state = pdk.cache().tsmc.lock().unwrap_or_else(|p| p.into_inner());
        if state.lib_key.as_ref() != Some(&key) {
            let text = read_ascii_file(&path, || missing_message(&path))?;
            let library = Arc::new(parse_spice_library_text(&text, &path)?);
            state.lib_key = Some(key);
            state.library = Some(library);
            state.programs.clear();
        }
        let library = state.library.clone().expect("library set");
        let program_key = (corner.clone(), temp_c.to_bits());
        if !state.programs.contains_key(&program_key) {
            let sections: Vec<String> = vec![
                "setup".into(),
                corner.clone(),
                "global".into(),
                "total".into(),
                "stat".into(),
            ];
            let mut initial = HashMap::new();
            initial.insert("temper".to_string(), temp_c);
            let elaborated = elaborate_library(&library, &sections, initial, true)?;
            state
                .programs
                .insert(program_key.clone(), Arc::new(elaborated));
        }
        state
            .programs
            .get(&program_key)
            .expect("program set")
            .clone()
    };

    let macro_id = macro_name(polarity);
    let mismatch_v = mismatch.unwrap_or(0.0);
    let params = vec![
        ("w".to_string(), ParamValue::Num(width_um * 1e-6)),
        ("l".to_string(), ParamValue::Num(length_um * 1e-6)),
        ("nf".to_string(), ParamValue::Num(nf as f64)),
        ("multi".to_string(), ParamValue::Num(mult as f64)),
        ("_delvto".to_string(), ParamValue::Num(mismatch_v)),
    ];
    let instance = program.instantiate(macro_id, &params).map_err(|e| {
        // The reference wraps only a SpiceElaborationError into Tsmc28ModelError.
        if e.kind == SpiceErrorKind::Elaboration {
            PdkError::model(e.message)
        } else {
            PdkError::from(e)
        }
    })?;

    let mos_elements: Vec<&Statement> = instance
        .elements()
        .into_iter()
        .filter(|s| s.kind == "m")
        .collect();
    let unsupported: Vec<String> = instance
        .elements()
        .into_iter()
        .filter(|s| s.kind != "m")
        .map(|s| s.kind.clone())
        .collect();
    if mos_elements.len() != 1 || !unsupported.is_empty() {
        return Err(PdkError::model(format!(
            "{macro_id} must expand to exactly one MOS and no other active \
             elements; mos={}, unsupported={:?}",
            mos_elements.len(),
            unsupported
        )));
    }
    let element = mos_elements[0];
    let instance_parameters = instance.numeric_parameters(element, None)?;

    let inst_w = *instance_parameters
        .get("w")
        .ok_or_else(|| PdkError::model("instance has no w"))?;
    let inst_nf = *instance_parameters
        .get("nf")
        .ok_or_else(|| PdkError::model("instance has no nf"))?;
    let inst_l = *instance_parameters
        .get("l")
        .ok_or_else(|| PdkError::model("instance has no l"))?;
    // BSIM4 bins on effective width per finger.
    let (selected, bounds) = select_bin(&instance, inst_w / inst_nf, inst_l)?;

    let numeric = instance.numeric_model(selected, None)?;
    if numeric.model_type.to_lowercase() != polarity {
        return Err(PdkError::model(format!(
            "{macro_id} selected {:?}, expected {:?}",
            numeric.model_type, polarity
        )));
    }
    let level = numeric.parameters.get("level").copied().unwrap_or(-1.0) as i64;
    if level != 54 {
        return Err(PdkError::model(format!(
            "{} is not a BSIM4 level-54 model",
            selected.name.clone().unwrap_or_default()
        )));
    }
    // version in {4.5, 4.50} (identical floats); absent -> unsupported.
    let version = numeric.parameters.get("version").copied();
    if version != Some(4.5) {
        return Err(PdkError::model(format!(
            "{} uses unsupported BSIM4 version {:?}",
            selected.name.clone().unwrap_or_default(),
            version
        )));
    }

    let bin_name = selected.name.clone().unwrap_or_default();
    let model_type = numeric.model_type.to_lowercase();
    Ok(NumericCard {
        model_parameters: numeric.parameters,
        instance_parameters,
        model_name: bin_name.clone(),
        model_type,
        source_version: 4.5,
        bin: Some(Bin {
            name: bin_name.clone(),
            lmin: bounds.get("lmin").copied().unwrap_or_default(),
            lmax: bounds.get("lmax").copied().unwrap_or_default(),
            wmin: bounds.get("wmin").copied().unwrap_or_default(),
            wmax: bounds.get("wmax").copied().unwrap_or_default(),
        }),
        source: Source {
            pdk: "tsmc28".into(),
            polarity: polarity.into(),
            corner,
            path,
            temperature_c: Some(temp_c),
            macro_name: Some(macro_id.to_string()),
            bin_name: Some(bin_name),
        },
    })
}
