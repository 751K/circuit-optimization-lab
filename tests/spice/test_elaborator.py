"""Section dependency, parameter scope and subcircuit elaboration tests."""
from __future__ import annotations

import pytest

from circuitopt.spice import (
    ParameterCycleError,
    SpiceElaborationError,
    elaborate_library,
    parse_spice_library_text,
    select_library_sections,
)


def _library():
    return parse_spice_library_text(
        """
.lib setup
.param base=2
.endl setup
.lib corner
.lib "models.l" setup
.param shift=3
.endl corner
.lib devices
.subckt core d g s b w=1u l=30n
.param area=w*l
.param scale(x)=x*base+shift
.model bin.1 nmos level=54 lmin=20n lmax=40n wmin=0.5u wmax=2u
+ vth0=scale(0.2)
m0 d g s b bin w=w l=l
.ends core
.endl devices
""",
        path="/tmp/models.l",
    )


def test_section_references_are_ordered_and_deduplicated():
    selected = select_library_sections(_library(), ("setup", "corner", "devices"))
    assert selected.names == ("setup", "corner", "devices")


def test_section_cycles_and_missing_sections_fail():
    library = parse_spice_library_text(
        """
.lib a
.lib "x.l" b
.endl a
.lib b
.lib "x.l" a
.endl b
""",
        path="/tmp/x.l",
    )
    with pytest.raises(SpiceElaborationError, match="a -> b -> a"):
        select_library_sections(library, ("a",))
    with pytest.raises(SpiceElaborationError, match="unknown library section"):
        select_library_sections(library, ("missing",))


def test_subcircuit_instance_scope_and_numeric_model():
    elaborated = elaborate_library(_library(), ("corner", "devices"))
    instance = elaborated.instantiate("CORE", {"W": 1.5e-6})
    assert instance.scope.resolve_symbol("area") == pytest.approx(45e-15)
    model = instance.numeric_model(instance.model_statements[0])
    assert model.model_type == "nmos"
    assert model.parameters["vth0"] == pytest.approx(3.4)
    assert model.parameters["wmin"] == pytest.approx(0.5e-6)
    assert instance.elements[0].kind == "m"
    assert instance.numeric_parameters(
        instance.elements[0], names=("w", "l")) == pytest.approx(
            {"w": 1.5e-6, "l": 30e-9})


def test_instance_override_breaks_default_dependency_cycle_safely():
    library = parse_spice_library_text(
        """
.lib devices
.subckt bad d g s b w=l l=w
m0 d g s b nch w=w l=l
.ends bad
.endl devices
"""
    )
    elaborated = elaborate_library(library, ("devices",))
    with pytest.raises(ParameterCycleError):
        elaborated.instantiate("bad").scope.resolve_symbol("w")
    instance = elaborated.instantiate("bad", {"w": 1e-6, "l": 30e-9})
    assert instance.scope.resolve_symbol("w") == pytest.approx(1e-6)
