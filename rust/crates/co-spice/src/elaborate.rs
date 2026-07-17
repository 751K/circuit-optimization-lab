//! Library-section, parameter-scope and subcircuit elaboration.
//!
//! A 1:1 port of `circuitopt/spice/elaborator.py` (the frozen Python reference).
//! It resolves the structural semantics shared by SPICE/HSPICE model libraries
//! and reuses the crate's [`ScopeInner`] evaluator for all numericization — no
//! second expression engine is introduced.
//!
//! Section-dependency and subcircuit failures raise [`ErrorKind::Elaboration`]
//! (the Python `SpiceElaborationError`); the duplicate model/subckt name check
//! raises [`ErrorKind::Syntax`] (`SpiceSyntaxError`), exactly as the reference.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::deck::{
    LibrarySection, OrderedMap, ParameterAssignment, SpiceModelLibrary, Statement, Subcircuit,
};
use crate::error::{SpiceError, SpiceResult};
use crate::scope::{EvalCtx, ScopeInner};

/// A parameter override value passed to `instantiate`: a number (`set_value`) or
/// an expression string (`define`), mirroring `Mapping[str, float | str]`.
#[derive(Debug, Clone)]
pub enum ParamValue {
    Num(f64),
    Str(String),
}

/// `apply_assignments`: apply values/functions to a scope in declaration order.
pub fn apply_assignments(scope: &Arc<ScopeInner>, assignments: &[ParameterAssignment]) {
    for assignment in assignments {
        if assignment.is_function() {
            scope.define_function(
                &assignment.name,
                &assignment.formal_parameters,
                &assignment.expression,
            );
        } else {
            scope.define(&assignment.name, &assignment.expression);
        }
    }
}

/// `apply_parameter_statements`: apply the assignments of every `.param`
/// statement, in order.
fn apply_parameter_statements(scope: &Arc<ScopeInner>, statements: &[Statement]) {
    for statement in statements {
        if statement.kind == "param" {
            apply_assignments(scope, &statement.parameters);
        }
    }
}

/// POSIX `os.path.basename`: the path component after the last `/`.
fn basename(path: &str) -> &str {
    match path.rfind('/') {
        Some(pos) => &path[pos + 1..],
        None => path,
    }
}

/// Strip leading/trailing `'`/`"` (Python `str.strip("'\"")`).
fn strip_quotes(s: &str) -> String {
    s.trim_matches(|c| c == '\'' || c == '"').to_string()
}

/// `_same_library`. The reference's `abspath(a) == abspath(b)` term is subsumed
/// by the case-insensitive basename term (equal absolute paths always share a
/// basename), so this CWD-free form is exactly equivalent — and free of the
/// working-directory dependence `os.path.abspath` would introduce.
fn same_library(reference: &str, library_path: &str) -> bool {
    let requested = strip_quotes(reference);
    if requested.is_empty() {
        return false;
    }
    basename(&requested).to_lowercase() == basename(library_path).to_lowercase()
}

/// One ordered, de-duplicated set of library sections (`SectionSelection`).
/// Borrows the parsed library; consumed by [`elaborate_library`] in the same
/// call, so no section data is cloned.
#[derive(Debug)]
pub struct SectionSelection<'a> {
    pub names: Vec<String>,
    pub sections: Vec<&'a LibrarySection>,
}

impl SectionSelection<'_> {
    /// `statements`: every non-`.lib` statement across the selected sections.
    pub fn statements(&self) -> Vec<&Statement> {
        self.sections
            .iter()
            .flat_map(|section| section.statements.iter())
            .filter(|statement| statement.kind != "lib")
            .collect()
    }

    /// `subcircuits`: merged across sections; later sections override earlier.
    pub fn subcircuits(&self) -> OrderedMap<Subcircuit> {
        let mut result = OrderedMap::new();
        for section in &self.sections {
            for (key, sub) in section.subcircuits.iter() {
                result.insert(key.clone(), sub.clone());
            }
        }
        result
    }
}

#[allow(clippy::too_many_arguments)]
fn visit<'a>(
    library: &'a SpiceModelLibrary,
    name: &str,
    follow_same_file_references: bool,
    ordered: &mut Vec<&'a LibrarySection>,
    completed: &mut HashSet<String>,
    visiting: &mut Vec<String>,
) -> SpiceResult<()> {
    let key = name.to_lowercase();
    if completed.contains(&key) {
        return Ok(());
    }
    if visiting.contains(&key) {
        let first = visiting.iter().position(|k| *k == key).unwrap_or(0);
        let mut cycle: Vec<String> = visiting[first..].to_vec();
        cycle.push(key.clone());
        return Err(SpiceError::elaboration(format!(
            "library-section dependency cycle: {}",
            cycle.join(" -> ")
        )));
    }
    let section = library
        .section(&key)
        .map_err(|body| SpiceError::elaboration(py_repr(&body)))?;
    visiting.push(key.clone());
    if follow_same_file_references {
        for statement in &section.statements {
            if statement.kind != "lib" || statement.arguments.len() != 2 {
                continue;
            }
            let reference = &statement.arguments[0];
            let target = &statement.arguments[1];
            if same_library(reference, &library.path) {
                visit(
                    library,
                    &strip_quotes(target),
                    follow_same_file_references,
                    ordered,
                    completed,
                    visiting,
                )?;
            }
        }
    }
    visiting.pop();
    completed.insert(key);
    ordered.push(section);
    Ok(())
}

/// Resolve requested sections and same-file `.lib file section` references
/// (`select_library_sections`).
pub fn select_library_sections<'a>(
    library: &'a SpiceModelLibrary,
    names: &[String],
    follow_same_file_references: bool,
) -> SpiceResult<SectionSelection<'a>> {
    let mut ordered: Vec<&LibrarySection> = Vec::new();
    let mut completed: HashSet<String> = HashSet::new();
    let mut visiting: Vec<String> = Vec::new();
    for requested in names {
        visit(
            library,
            requested,
            follow_same_file_references,
            &mut ordered,
            &mut completed,
            &mut visiting,
        )?;
    }
    Ok(SectionSelection {
        names: ordered.iter().map(|s| s.name.clone()).collect(),
        sections: ordered,
    })
}

/// A fully numeric `.model` view (`NumericModel`).
#[derive(Debug, Clone)]
pub struct NumericModel {
    pub name: String,
    pub model_type: String,
    pub parameters: HashMap<String, f64>,
}

/// One parameterized instance of a parsed subcircuit template
/// (`SubcircuitInstance`).
pub struct SubcircuitInstance {
    template: Subcircuit,
    pub scope: Arc<ScopeInner>,
}

impl SubcircuitInstance {
    /// Build the instance scope: template parameters, then instance overrides
    /// (`define` for strings, `set_value` for numbers), then `.param` statements.
    pub fn new(
        template: Subcircuit,
        parent_scope: Arc<ScopeInner>,
        parameters: &[(String, ParamValue)],
    ) -> Self {
        let scope = ScopeInner::new_child(parent_scope, HashMap::new());
        apply_assignments(&scope, &template.parameters);
        for (name, value) in parameters {
            match value {
                ParamValue::Str(text) => scope.define(name, text),
                ParamValue::Num(number) => scope.set_value(name, *number),
            }
        }
        apply_parameter_statements(&scope, &template.statements);
        Self { template, scope }
    }

    /// `elements`: statements that are neither `.param` nor `.model`.
    pub fn elements(&self) -> Vec<&Statement> {
        self.template
            .statements
            .iter()
            .filter(|s| s.kind != "param" && s.kind != "model")
            .collect()
    }

    /// `model_statements`: the `.model` statements of the template.
    pub fn model_statements(&self) -> Vec<&Statement> {
        self.template
            .statements
            .iter()
            .filter(|s| s.kind == "model")
            .collect()
    }

    /// `statement_scope`: child scope holding one statement's parameters.
    fn statement_scope(&self, statement: &Statement) -> Arc<ScopeInner> {
        let scope = ScopeInner::new_child(self.scope.clone(), HashMap::new());
        apply_assignments(&scope, &statement.parameters);
        scope
    }

    /// `numeric_parameters`: resolve a statement's parameters. With `names`,
    /// resolve exactly those (lower-cased keys); otherwise resolve all.
    pub fn numeric_parameters(
        &self,
        statement: &Statement,
        names: Option<&[String]>,
    ) -> SpiceResult<HashMap<String, f64>> {
        let scope = self.statement_scope(statement);
        match names {
            None => scope.evaluate_all(),
            Some(names) => {
                let mut out = HashMap::new();
                for name in names {
                    let mut ctx = EvalCtx::new();
                    let value = scope.resolve_symbol(name, &mut ctx)?;
                    out.insert(name.to_lowercase(), value);
                }
                Ok(out)
            }
        }
    }

    /// `numeric_model`: numericize a `.model` statement.
    pub fn numeric_model(
        &self,
        statement: &Statement,
        names: Option<&[String]>,
    ) -> SpiceResult<NumericModel> {
        let parameters = self.numeric_parameters(statement, names)?;
        Ok(NumericModel {
            name: statement.name.clone().unwrap_or_default(),
            model_type: statement.arguments.first().cloned().unwrap_or_default(),
            parameters,
        })
    }
}

/// Global parameter/model/subcircuit views for selected sections
/// (`ElaboratedLibrary`).
pub struct ElaboratedLibrary {
    pub selection_names: Vec<String>,
    pub global_scope: Arc<ScopeInner>,
    pub models: OrderedMap<Statement>,
    pub subcircuits: OrderedMap<Subcircuit>,
}

impl ElaboratedLibrary {
    /// `instantiate`: build one instance of a named subcircuit template.
    pub fn instantiate(
        &self,
        name: &str,
        parameters: &[(String, ParamValue)],
    ) -> SpiceResult<SubcircuitInstance> {
        let key = name.to_lowercase();
        let template = self.subcircuits.get(&key).cloned().ok_or_else(|| {
            let mut available: Vec<&String> = self.subcircuits.keys().collect();
            available.sort();
            let joined: Vec<String> = available.iter().map(|s| (*s).clone()).collect();
            SpiceError::elaboration(format!(
                "unknown subcircuit {}; available: {}",
                py_repr(name),
                joined.join(", ")
            ))
        })?;
        Ok(SubcircuitInstance::new(
            template,
            self.global_scope.clone(),
            parameters,
        ))
    }
}

/// Build global parameter/model/subcircuit views for selected sections
/// (`elaborate_library`).
pub fn elaborate_library(
    library: &SpiceModelLibrary,
    section_names: &[String],
    initial_values: HashMap<String, f64>,
    follow_same_file_references: bool,
) -> SpiceResult<ElaboratedLibrary> {
    let selection = select_library_sections(library, section_names, follow_same_file_references)?;
    let scope = ScopeInner::new_root(initial_values);
    let mut models: OrderedMap<Statement> = OrderedMap::new();
    for section in &selection.sections {
        apply_parameter_statements(&scope, &section.statements);
        for statement in &section.statements {
            if statement.kind == "model"
                && let Some(name) = &statement.name
            {
                models.insert(name.to_lowercase(), statement.clone());
            }
        }
    }
    let subcircuits = selection.subcircuits();
    // Duplicate model/subckt name check (a SpiceSyntaxError in the reference).
    let mut duplicate: Vec<String> = models
        .keys()
        .filter(|k| subcircuits.contains(k))
        .cloned()
        .collect();
    if !duplicate.is_empty() {
        duplicate.sort();
        return Err(SpiceError::syntax(format!(
            "names used by both models and subcircuits: {}",
            duplicate.join(", ")
        )));
    }
    Ok(ElaboratedLibrary {
        selection_names: selection.names.clone(),
        global_scope: scope,
        models,
        subcircuits,
    })
}

/// Python `repr()` for the strings that appear in elaboration error messages.
fn py_repr(s: &str) -> String {
    let use_double = s.contains('\'') && !s.contains('"');
    let quote = if use_double { '"' } else { '\'' };
    let mut out = String::new();
    out.push(quote);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deck::parse_spice_library_text;
    use crate::error::ErrorKind;

    fn library() -> SpiceModelLibrary {
        parse_spice_library_text(
            "\n.lib setup\n.param base=2\n.endl setup\n.lib corner\n.lib \"models.l\" setup\n.param shift=3\n.endl corner\n.lib devices\n.subckt core d g s b w=1u l=30n\n.param area=w*l\n.param scale(x)=x*base+shift\n.model bin.1 nmos level=54 lmin=20n lmax=40n wmin=0.5u wmax=2u\n+ vth0=scale(0.2)\nm0 d g s b bin w=w l=l\n.ends core\n.endl devices\n",
            "/tmp/models.l",
        )
        .unwrap()
    }

    #[test]
    fn sections_ordered_and_deduplicated() {
        let lib = library();
        let names = vec![
            "setup".to_string(),
            "corner".to_string(),
            "devices".to_string(),
        ];
        let selected = select_library_sections(&lib, &names, true).unwrap();
        assert_eq!(selected.names, vec!["setup", "corner", "devices"]);
    }

    #[test]
    fn section_cycle_and_missing_fail() {
        let lib = parse_spice_library_text(
            "\n.lib a\n.lib \"x.l\" b\n.endl a\n.lib b\n.lib \"x.l\" a\n.endl b\n",
            "/tmp/x.l",
        )
        .unwrap();
        let err = select_library_sections(&lib, &["a".to_string()], true).unwrap_err();
        assert_eq!(err.kind, ErrorKind::Elaboration);
        assert!(err.message.contains("a -> b -> a"), "{}", err.message);
        let err = select_library_sections(&lib, &["missing".to_string()], true).unwrap_err();
        assert_eq!(err.kind, ErrorKind::Elaboration);
        assert!(err.message.contains("unknown library section"));
    }

    #[test]
    fn subcircuit_scope_and_numeric_model() {
        let lib = library();
        let elaborated = elaborate_library(
            &lib,
            &["corner".to_string(), "devices".to_string()],
            HashMap::new(),
            true,
        )
        .unwrap();
        let instance = elaborated
            .instantiate("CORE", &[("W".to_string(), ParamValue::Num(1.5e-6))])
            .unwrap();
        let area = instance
            .scope
            .resolve_symbol("area", &mut EvalCtx::new())
            .unwrap();
        assert!((area - 45e-15).abs() <= 45e-15 * 1e-12);
        let model_statements = instance.model_statements();
        let model = instance.numeric_model(model_statements[0], None).unwrap();
        assert_eq!(model.model_type, "nmos");
        assert!((model.parameters["vth0"] - 3.4).abs() <= 3.4 * 1e-12);
        assert!((model.parameters["wmin"] - 0.5e-6).abs() <= 0.5e-6 * 1e-12);
        let elements = instance.elements();
        assert_eq!(elements[0].kind, "m");
        let wl = instance
            .numeric_parameters(elements[0], Some(&["w".to_string(), "l".to_string()]))
            .unwrap();
        assert!((wl["w"] - 1.5e-6).abs() <= 1.5e-6 * 1e-12);
        assert!((wl["l"] - 30e-9).abs() <= 30e-9 * 1e-12);
    }

    #[test]
    fn instance_override_breaks_default_cycle() {
        let lib = parse_spice_library_text(
            "\n.lib devices\n.subckt bad d g s b w=l l=w\nm0 d g s b nch w=w l=l\n.ends bad\n.endl devices\n",
            "<string>",
        )
        .unwrap();
        let elaborated =
            elaborate_library(&lib, &["devices".to_string()], HashMap::new(), true).unwrap();
        let instance = elaborated.instantiate("bad", &[]).unwrap();
        let err = instance
            .scope
            .resolve_symbol("w", &mut EvalCtx::new())
            .unwrap_err();
        assert_eq!(err.kind, ErrorKind::ParameterCycle);
        let instance = elaborated
            .instantiate(
                "bad",
                &[
                    ("w".to_string(), ParamValue::Num(1e-6)),
                    ("l".to_string(), ParamValue::Num(30e-9)),
                ],
            )
            .unwrap();
        assert_eq!(
            instance
                .scope
                .resolve_symbol("w", &mut EvalCtx::new())
                .unwrap(),
            1e-6
        );
    }
}
