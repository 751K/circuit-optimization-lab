"""Syntax-layer tests for the in-process SPICE/HSPICE model parser."""
from __future__ import annotations

import os

import pytest

from circuitopt.spice.parser import (
    SpiceSyntaxError,
    logical_lines,
    parse_assignments,
    parse_spice_library,
    parse_spice_library_text,
    parse_spice_number,
)


def test_logical_lines_join_continuations_and_strip_comments():
    source = """* card
.param a=1
+ b='max(2, 3)' $ trailing comment

.model nch nmos (
+ level=54 version=4.5)
"""
    lines = list(logical_lines(source))
    assert [line for line, _location in lines] == [
        ".param a=1 b='max(2, 3)'",
        ".model nch nmos ( level=54 version=4.5)",
    ]
    assert lines[0][1].first_line == 2
    assert lines[0][1].last_line == 3


def test_continuation_without_statement_is_rejected():
    with pytest.raises(SpiceSyntaxError, match="continuation"):
        list(logical_lines("+ orphan"))


def test_parse_assignments_preserves_nested_and_quoted_expressions():
    parsed = parse_assignments(
        "(level=54 version=4.50 vth0='base + pwr(l, 2)' "
        "rdsw=max(0, r0 + delta))"
    )
    assert [(item.name, item.expression) for item in parsed] == [
        ("level", "54"),
        ("version", "4.50"),
        ("vth0", "'base + pwr(l, 2)'"),
        ("rdsw", "max(0, r0 + delta)"),
    ]


def test_parse_parameter_function_definition():
    parsed = parse_assignments(
        "selbin(par1, par2, par3, par4)='max(par1, min(par2, par3))' scale=1"
    )
    assert parsed[0].name == "selbin"
    assert parsed[0].formal_parameters == ("par1", "par2", "par3", "par4")
    assert parsed[0].is_function
    assert parsed[1].formal_parameters == ()


@pytest.mark.parametrize(
    ("literal", "expected"),
    [
        ("1", 1.0),
        ("2.5k", 2.5e3),
        ("3meg", 3e6),
        ("4m", 4e-3),
        ("5uF", 5e-6),
        ("6mil", 6 * 25.4e-6),
    ],
)
def test_parse_spice_number(literal, expected):
    assert parse_spice_number(literal) == pytest.approx(expected)


def test_library_model_and_subcircuit_ast():
    source = """
.lib tt
.param scale=1 corner='max(0, process)'
.model nch nmos level=54 version=4.5 vth0='0.4 + dvth'
.subckt nch_mac d g s b w=1u l=30n nf=1
m0 d g s b nch w=w l=l nf=nf
r0 d s 10k
.ends nch_mac
.endl tt
.lib "other.lib" support
"""
    library = parse_spice_library_text(source)
    tt = library.section("TT")
    assert tt.models["nch"].arguments == ("nmos",)
    assert tt.models["nch"].parameter_map["version"] == "4.5"
    macro = tt.subcircuits["nch_mac"]
    assert macro.terminals == ("d", "g", "s", "b")
    assert macro.parameters[0].name == "w"
    assert [statement.kind for statement in macro.statements] == ["m", "r"]
    assert library.top_level.statements[0].kind == "lib"
    assert library.top_level.statements[0].arguments == ('"other.lib"', "support")


def test_real_tsmc_library_syntax_when_locally_installed():
    from circuitopt.toolchain import tsmc28_model_dir

    path = os.path.join(
        tsmc28_model_dir(), "cln28hpcp_1d8_elk_v1d0_2p2.l")
    if not os.path.isfile(path):
        pytest.skip("licensed TSMC28HPC+ model is not installed")

    library = parse_spice_library(path)
    for section in ("setup", "tt", "ss", "ff", "sf", "fs", "global", "total", "stat"):
        assert section in library.sections
    model_statements = [
        statement
        for section in library.sections.values()
        for statement in (
            list(section.statements)
            + [
                child
                for subcircuit in section.subcircuits.values()
                for child in subcircuit.statements
            ]
        )
        if statement.kind == "model"
    ]
    assert len(model_statements) >= 400
    macros = {
        name
        for section in library.sections.values()
        for name in section.subcircuits
    }
    assert {"nch_mac", "pch_mac"} <= macros
    core_cards = [
        model
        for model in model_statements
        if model.arguments and model.arguments[0].lower() in {"nmos", "pmos"}
        and model.parameter_map.get("level") == "54"
    ]
    assert core_cards
    assert any(
        model.parameter_map.get("version", "").strip("'\"") in {"4.5", "4.50"}
        for model in core_cards
    )
