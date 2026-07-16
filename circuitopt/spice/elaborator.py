"""Library-section, parameter-scope and subcircuit elaboration.

This layer is process-neutral. It resolves the structural semantics shared by
SPICE/HSPICE model libraries, but leaves process-specific section selection and
geometry-bin policy to the PDK adapter.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Mapping

from .expressions import EvaluationScope
from .parser import (
    LibrarySection,
    ParameterAssignment,
    SpiceModelLibrary,
    SpiceSyntaxError,
    Statement,
    Subcircuit,
)


class SpiceElaborationError(ValueError):
    """A structurally valid library cannot be elaborated."""


def apply_assignments(
    scope: EvaluationScope,
    assignments: Iterable[ParameterAssignment],
) -> None:
    """Apply parameter values/functions to *scope* in declaration order."""
    for assignment in assignments:
        if assignment.is_function:
            scope.define_function(
                assignment.name,
                assignment.formal_parameters,
                assignment.expression,
            )
        else:
            scope.define(assignment.name, assignment.expression)


def apply_parameter_statements(
    scope: EvaluationScope,
    statements: Iterable[Statement],
) -> None:
    for statement in statements:
        if statement.kind == "param":
            apply_assignments(scope, statement.parameters)


def _same_library(reference: str, library_path: str) -> bool:
    requested = reference.strip("'\"")
    if not requested:
        return False
    return (
        os.path.abspath(requested) == os.path.abspath(library_path)
        or os.path.basename(requested).lower() == os.path.basename(library_path).lower()
    )


@dataclass(frozen=True)
class SectionSelection:
    """One ordered, de-duplicated set of library sections."""

    names: tuple[str, ...]
    sections: tuple[LibrarySection, ...]

    @property
    def statements(self) -> tuple[Statement, ...]:
        return tuple(
            statement
            for section in self.sections
            for statement in section.statements
            if statement.kind != "lib"
        )

    @property
    def subcircuits(self) -> dict[str, Subcircuit]:
        result = {}
        for section in self.sections:
            result.update(section.subcircuits)
        return result


def select_library_sections(
    library: SpiceModelLibrary,
    names: Iterable[str],
    *,
    follow_same_file_references: bool = True,
) -> SectionSelection:
    """Resolve requested sections and same-file ``.lib file section`` references.

    Each section is emitted once, at the position of its first reference. Cycles
    and references to unknown same-file sections fail explicitly.
    """
    ordered: list[LibrarySection] = []
    completed: set[str] = set()
    visiting: list[str] = []

    def visit(name: str) -> None:
        key = name.lower()
        if key in completed:
            return
        if key in visiting:
            first = visiting.index(key)
            cycle = visiting[first:] + [key]
            raise SpiceElaborationError(
                "library-section dependency cycle: " + " -> ".join(cycle))
        try:
            section = library.section(key)
        except KeyError as exc:
            raise SpiceElaborationError(str(exc)) from exc
        visiting.append(key)
        if follow_same_file_references:
            for statement in section.statements:
                if statement.kind != "lib" or len(statement.arguments) != 2:
                    continue
                reference, target = statement.arguments
                if _same_library(reference, library.path):
                    visit(target.strip("'\""))
        visiting.pop()
        completed.add(key)
        ordered.append(section)

    for requested in names:
        visit(str(requested))
    return SectionSelection(
        names=tuple(section.name for section in ordered),
        sections=tuple(ordered),
    )


@dataclass
class NumericModel:
    name: str
    model_type: str
    parameters: dict[str, float]
    statement: Statement


class SubcircuitInstance:
    """One parameterized instance of a parsed subcircuit template."""

    def __init__(
        self,
        template: Subcircuit,
        parent_scope: EvaluationScope,
        parameters: Mapping[str, float | str] | None = None,
    ):
        self.template = template
        self.scope = EvaluationScope(parent=parent_scope)
        apply_assignments(self.scope, template.parameters)
        for name, value in (parameters or {}).items():
            if isinstance(value, str):
                self.scope.define(name, value)
            else:
                self.scope.set_value(name, value)
        apply_parameter_statements(self.scope, template.statements)

    @property
    def elements(self) -> tuple[Statement, ...]:
        return tuple(
            statement
            for statement in self.template.statements
            if statement.kind not in {"param", "model"}
        )

    @property
    def model_statements(self) -> tuple[Statement, ...]:
        return tuple(
            statement
            for statement in self.template.statements
            if statement.kind == "model"
        )

    def model_scope(self, statement: Statement) -> EvaluationScope:
        if statement.kind != "model":
            raise TypeError("model_scope requires a .model statement")
        scope = EvaluationScope(parent=self.scope)
        apply_assignments(scope, statement.parameters)
        return scope

    def statement_scope(self, statement: Statement) -> EvaluationScope:
        """Child scope containing one element/model statement's parameters."""
        scope = EvaluationScope(parent=self.scope)
        apply_assignments(scope, statement.parameters)
        return scope

    def numeric_parameters(
        self,
        statement: Statement,
        names: Iterable[str] | None = None,
    ) -> dict[str, float]:
        scope = self.statement_scope(statement)
        if names is None:
            return scope.evaluate_all()
        return {
            str(name).lower(): scope.resolve_symbol(str(name))
            for name in names
        }

    def numeric_model(
        self,
        statement: Statement,
        *,
        names: Iterable[str] | None = None,
    ) -> NumericModel:
        parameters = self.numeric_parameters(statement, names)
        return NumericModel(
            name=statement.name or "",
            model_type=statement.arguments[0] if statement.arguments else "",
            parameters=parameters,
            statement=statement,
        )


@dataclass
class ElaboratedLibrary:
    library: SpiceModelLibrary
    selection: SectionSelection
    global_scope: EvaluationScope
    models: dict[str, Statement]
    subcircuits: dict[str, Subcircuit]

    def instantiate(
        self,
        name: str,
        parameters: Mapping[str, float | str] | None = None,
    ) -> SubcircuitInstance:
        key = name.lower()
        try:
            template = self.subcircuits[key]
        except KeyError as exc:
            raise SpiceElaborationError(
                f"unknown subcircuit {name!r}; available: "
                f"{', '.join(sorted(self.subcircuits))}") from exc
        return SubcircuitInstance(template, self.global_scope, parameters)


def elaborate_library(
    library: SpiceModelLibrary,
    section_names: Iterable[str],
    *,
    initial_values: Mapping[str, float] | None = None,
    follow_same_file_references: bool = True,
) -> ElaboratedLibrary:
    """Build global parameter/model/subcircuit views for selected sections."""
    selection = select_library_sections(
        library,
        section_names,
        follow_same_file_references=follow_same_file_references,
    )
    scope = EvaluationScope(values=initial_values)
    models = {}
    for section in selection.sections:
        apply_parameter_statements(scope, section.statements)
        for statement in section.statements:
            if statement.kind == "model" and statement.name is not None:
                models[statement.name.lower()] = statement
    subcircuits = selection.subcircuits
    duplicate = set(models) & set(subcircuits)
    if duplicate:
        raise SpiceSyntaxError(
            "names used by both models and subcircuits: " + ", ".join(sorted(duplicate)))
    return ElaboratedLibrary(
        library=library,
        selection=selection,
        global_scope=scope,
        models=models,
        subcircuits=subcircuits,
    )
