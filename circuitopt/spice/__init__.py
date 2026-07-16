"""SPICE syntax and elaboration support used by native compact-model backends.

The package owns netlist/model-library semantics only.  Numerical compact models
live under :mod:`circuitopt.compact_models`, while PDK-specific selection and
registration live under :mod:`circuitopt.pdk`.
"""

from .parser import (
    LibrarySection,
    ParameterAssignment,
    SourceLocation,
    SpiceModelLibrary,
    SpiceSyntaxError,
    Statement,
    Subcircuit,
    logical_lines,
    parse_assignments,
    parse_spice_library,
    parse_spice_library_text,
    parse_spice_number,
)
from .expressions import (
    EvaluationScope,
    ParameterCycleError,
    SpiceExpressionError,
    UnknownSymbolError,
    compile_expression,
    tokenize,
)
from .elaborator import (
    ElaboratedLibrary,
    NumericModel,
    SectionSelection,
    SpiceElaborationError,
    SubcircuitInstance,
    apply_assignments,
    elaborate_library,
    select_library_sections,
)

__all__ = [
    "ElaboratedLibrary",
    "EvaluationScope",
    "LibrarySection",
    "NumericModel",
    "ParameterAssignment",
    "ParameterCycleError",
    "SectionSelection",
    "SourceLocation",
    "SpiceElaborationError",
    "SpiceExpressionError",
    "SpiceModelLibrary",
    "SpiceSyntaxError",
    "Statement",
    "Subcircuit",
    "SubcircuitInstance",
    "UnknownSymbolError",
    "apply_assignments",
    "compile_expression",
    "elaborate_library",
    "logical_lines",
    "parse_assignments",
    "parse_spice_library",
    "parse_spice_library_text",
    "parse_spice_number",
    "select_library_sections",
    "tokenize",
]
