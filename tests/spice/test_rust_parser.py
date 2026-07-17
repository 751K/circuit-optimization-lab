"""Differential parity: the Rust co-spice deck parser vs the frozen Python one.

Every assertion compares the compiled-core output against the frozen Python
reference in :mod:`circuitopt.spice.parser`. The Python side is the oracle; the
Rust side must match its canonical tree field-for-field and its
``parse_spice_number`` bit-for-bit.

The licensed TSMC28 delivery is treated as a differential fixture only: nothing
about its parameter text or values is written to this file, a golden, or a log.
The real-library assertions compare structure and counts, and on any mismatch
raise a fixed message (never echoing card content).
"""
from __future__ import annotations

import os

import pytest

from circuitopt.spice import parser as P

cc = pytest.importorskip("circuitopt_core")

if not hasattr(cc, "spice_parse_library"):
    pytest.skip(
        "compiled core lacks the SPICE deck-parser parity surface",
        allow_module_level=True,
    )


# --- canonicalizers: Python dataclasses -> the same dict shape the Rust side emits


def _canon_assignment(a):
    return {
        "name": a.name,
        "expression": a.expression,
        "formal_parameters": list(a.formal_parameters),
    }


def _canon_statement(s):
    return {
        "kind": s.kind,
        "location": (s.location.path, s.location.first_line, s.location.last_line),
        "text": s.text,
        "name": s.name,
        "arguments": list(s.arguments),
        "parameters": [_canon_assignment(a) for a in s.parameters],
    }


def _canon_subckt(sub):
    return {
        "name": sub.name,
        "location": (sub.location.path, sub.location.first_line, sub.location.last_line),
        "terminals": list(sub.terminals),
        "parameters": [_canon_assignment(a) for a in sub.parameters],
        "statements": [_canon_statement(s) for s in sub.statements],
    }


def _canon_section(sec):
    return {
        "name": sec.name,
        "location": (sec.location.path, sec.location.first_line, sec.location.last_line),
        "statements": [_canon_statement(s) for s in sec.statements],
        "subcircuits": {k: _canon_subckt(v) for k, v in sec.subcircuits.items()},
    }


def _canon_library(lib):
    return {
        "path": lib.path,
        "top_level": _canon_section(lib.top_level),
        "sections": {k: _canon_section(v) for k, v in lib.sections.items()},
    }


# --- parse_spice_number: bit-for-bit


@pytest.mark.parametrize(
    "literal",
    [
        "1", "0", "100", "0.5", ".5", "2.", "-3.2", "+.5",
        "2.5k", "3meg", "4m", "5uF", "6mil", "1e3", "2.5e-3", "1e3k",
        "2.5MEG", "2.5Meg", ".5e2", "2.e3", "1t", "1g", "1n", "1p", "1f",
        "1mil", "1.5meg", "  -2.5e3  ", "1kOhm", "10uA",
    ],
)
def test_parse_spice_number_bit_exact(literal):
    assert cc.spice_parse_number(literal).hex() == P.parse_spice_number(literal).hex()


@pytest.mark.parametrize("bad", ["abc", "", "1 2", "meg", "1..2", "1 k"])
def test_parse_spice_number_invalid_raises(bad):
    with pytest.raises(ValueError):
        P.parse_spice_number(bad)
    with pytest.raises(ValueError):
        cc.spice_parse_number(bad)


# --- logical lines, comments, continuations


def test_logical_lines_join_comments_and_continuations():
    source = (
        "* card\n"
        ".param a=1\n"
        "+ b='max(2, 3)' $ trailing comment\n"
        "\n"
        ".model nch nmos (\n"
        "+ level=54 version=4.5)\n"
    )
    rust = cc.spice_logical_lines(source)
    ref = [
        (t, (loc.path, loc.first_line, loc.last_line))
        for t, loc in P.logical_lines(source)
    ]
    assert rust == ref


def test_orphan_continuation_error_parity():
    with pytest.raises(P.SpiceSyntaxError, match="continuation"):
        list(P.logical_lines("+ orphan"))
    with pytest.raises(cc.SpiceSyntaxError):
        cc.spice_logical_lines("+ orphan")
    assert issubclass(cc.SpiceSyntaxError, ValueError)


# --- parse_assignments: nested/quoted/function definitions


@pytest.mark.parametrize(
    "text",
    [
        "(level=54 version=4.50 vth0='base + pwr(l, 2)' rdsw=max(0, r0 + delta))",
        "selbin(par1, par2, par3, par4)='max(par1, min(par2, par3))' scale=1",
        "w=1u l=30n nf=1",
        "k= x >= 3 j=1",
        "f()=1 g=2",
        "a=b c(x)=x*2 d='q'",
    ],
)
def test_parse_assignments_parity(text):
    rust = cc.spice_parse_assignments(text)
    ref = [_canon_assignment(a) for a in P.parse_assignments(text)]
    assert rust == ref


# --- full canonical tree for a synthetic all-syntax deck


_FULL_SYNTAX = """* header comment
.param g1=1 g2='max(2, 3)' fn(a,b)=a*b+g1  $ inline
+ g3=2.5k
.lib tt
.param local=10 corner='max(0, process)'
.model nch nmos level=54 version=4.5 vth0='0.4 + dvth' rdsw=max(0, r0)
.subckt nch_mac d g s b params: w=1u l=30n nf=1
m0 d g s b nch w=w l=l nf=nf
r0 d s 10k
.model binmod nmos level=54 lmin=20n lmax=40n wmin=0.5u wmax=2u
+ vth0='0.3 + fn(1, 2)'
.ends nch_mac
.endl tt
.lib "self.l" tt
xsub 1 2 3 mysub scale=2
"""


def test_full_syntax_canonical_tree():
    path = "/tmp/self.l"
    rust = cc.spice_parse_library_text(_FULL_SYNTAX, path)
    ref = _canon_library(P.parse_spice_library_text(_FULL_SYNTAX, path=path))
    assert rust == ref
    # Spot-checks that exercise the tricky corners explicitly.
    tt = rust["sections"]["tt"]
    assert tt["subcircuits"]["nch_mac"]["terminals"] == ["d", "g", "s", "b"]
    top_kinds = [s["kind"] for s in rust["top_level"]["statements"]]
    assert top_kinds == ["param", "lib", "x"]
    # The `.lib "self.l" tt` file-reference form keeps quoted arguments verbatim.
    assert rust["top_level"]["statements"][1]["arguments"] == ['"self.l"', "tt"]


@pytest.mark.parametrize(
    "source",
    [
        ".lib a\n.lib b\n.endl b\n.endl a\n",          # nested .lib
        ".endl\n",                                       # .endl outside section
        ".subckt x a b\n.subckt y c d\n",                # nested .subckt
        ".ends\n",                                       # .ends outside subckt
        ".subckt x a b\nm0 a b nch\n",                   # unterminated subckt
        ".lib tt\n.param a=1\n",                         # unterminated section
        "123 bad element\n",                             # malformed element
    ],
)
def test_structural_errors_parity(source):
    with pytest.raises(P.SpiceSyntaxError):
        P.parse_spice_library_text(source, path="/tmp/x.l")
    with pytest.raises(cc.SpiceSyntaxError):
        cc.spice_parse_library_text(source, "/tmp/x.l")


# --- real TSMC28 library: structure and counts only (D12: no card text emitted)


def _tsmc_path():
    from circuitopt.toolchain import tsmc28_model_dir

    path = os.path.join(tsmc28_model_dir(), "cln28hpcp_1d8_elk_v1d0_2p2.l")
    return path if os.path.isfile(path) else None


def test_tsmc_real_library_structural_parity():
    path = _tsmc_path()
    if path is None:
        pytest.skip("licensed TSMC28HPC+ model is not installed")

    rust = cc.spice_parse_library(path)
    ref = _canon_library(P.parse_spice_library(path))

    # Section set identical.
    if set(rust["sections"]) != set(ref["sections"]):
        raise AssertionError("TSMC section set mismatch (details suppressed for D12)")
    for core in ("setup", "tt", "ss", "ff", "sf", "fs", "global", "total", "stat"):
        assert core in rust["sections"], core

    # Full canonical tree identical — the strongest structural check. On any
    # mismatch, raise a fixed message so no PDK card content reaches the log.
    if rust != ref:
        raise AssertionError("TSMC canonical tree mismatch (details suppressed for D12)")

    # Structural counts (numbers only).
    py = P.parse_spice_library(path)
    model_statements = [
        s
        for sec in py.sections.values()
        for s in (
            list(sec.statements)
            + [c for sub in sec.subcircuits.values() for c in sub.statements]
        )
        if s.kind == "model"
    ]
    assert len(model_statements) >= 400
    macros = {n for sec in py.sections.values() for n in sec.subcircuits}
    assert {"nch_mac", "pch_mac"} <= macros

    # level=54 card-name set consistent between Rust and Python (names, not text).
    def level54_names(library_tree):
        names = set()
        for sec in library_tree["sections"].values():
            groups = [sec["statements"]]
            groups += [sub["statements"] for sub in sec["subcircuits"].values()]
            for stmts in groups:
                for st in stmts:
                    if st["kind"] != "model":
                        continue
                    pmap = {a["name"].lower(): a["expression"] for a in st["parameters"]}
                    if pmap.get("level") == "54":
                        names.add(st["name"])
        return names

    if level54_names(rust) != level54_names(ref):
        raise AssertionError("TSMC level-54 card set mismatch (details suppressed for D12)")
    assert len(level54_names(rust)) >= 1
