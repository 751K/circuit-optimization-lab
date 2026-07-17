"""Differential parity: the Rust co-spice elaborator vs the frozen Python one.

Section selection, parameter-scope filling and model/element numericization are
compared against :mod:`circuitopt.spice.elaborator` (the oracle). Numeric
results must match bit-for-bit or within a relative error of 1e-14; error types
must match.

The licensed TSMC28 delivery is a differential fixture only: no parameter text
or value is written to this file, a golden, or a log. The real-library
assertions report counts and the worst observed relative error.
"""
from __future__ import annotations

import os

import pytest

from circuitopt.spice import expressions as E
from circuitopt.spice import parser as P
from circuitopt.spice.elaborator import (
    SpiceElaborationError,
    apply_assignments,
    elaborate_library,
    select_library_sections,
)

cc = pytest.importorskip("circuitopt_core")

if not hasattr(cc, "spice_elaborate"):
    pytest.skip(
        "compiled core lacks the SPICE elaborator parity surface",
        allow_module_level=True,
    )


_TOL = 1e-14


def _worst_rel(rust_params: dict, py_params: dict, where: str) -> float:
    assert set(rust_params) == set(py_params), where
    worst = 0.0
    for key, pv in py_params.items():
        rv = rust_params[key]
        if rv == pv:
            continue
        if pv == 0.0:
            raise AssertionError(f"{where}:{key} expected 0 got nonzero")
        rel = abs(rv - pv) / abs(pv)
        worst = max(worst, rel)
        assert rel <= _TOL, f"{where}:{key} rel={rel:.3e}"
    return worst


def _ref_section_models(library, sections, overrides):
    """Frozen-Python reference for ``spice_elaborate``: numericize each
    section-level ``.model`` in a child of the global scope."""
    elaborated = elaborate_library(library, sections, initial_values=overrides)
    out = {}
    for key, stmt in elaborated.models.items():
        scope = E.EvaluationScope(parent=elaborated.global_scope)
        apply_assignments(scope, stmt.parameters)
        out[key] = {
            "name": stmt.name or "",
            "model_type": stmt.arguments[0] if stmt.arguments else "",
            "parameters": scope.evaluate_all(),
        }
    return out


_SYNTH = """
.lib setup
.param base=2 vdd=1.8
.endl setup
.lib corner
.lib "models.l" setup
.param shift=3 temp_adj='temper*0.01'
.endl corner
.lib devices
.param mult=base+shift
.model top_nmos nmos level=54 version=4.5 vth0='base*0.1 + shift*0.01' u0='mult*1e-3' k1=sqrt(4)
.model top_pmos pmos level=54 vth0='0 - (base*0.1)' tox=1.4n
.subckt core d g s b w=1u l=30n nf=1 mulu0=1
.param area=w*l scale(x)=x*base+shift
.model bin.1 nmos level=54 lmin=20n lmax=40n wmin=0.5u wmax=2u vth0='scale(0.2) + area*1e12'
m0 d g s b bin.1 w=w l=l nf=nf mulu0=mulu0
.ends core
.endl devices
"""


@pytest.fixture()
def synth_path(tmp_path):
    # Basename must be "models.l" so the `.lib "models.l" setup` reference resolves.
    path = tmp_path / "models.l"
    path.write_text(_SYNTH, encoding="ascii")
    return str(path)


def test_select_sections_parity(synth_path):
    library = P.parse_spice_library(synth_path)
    for request in (["corner", "devices"], ["devices", "setup", "corner"], ["setup"]):
        rust = cc.spice_select_sections(synth_path, request)
        ref = list(select_library_sections(library, request).names)
        assert rust == ref, request


def test_synth_section_level_models(synth_path):
    overrides = {"temper": 27.0, "process": 0.0}
    rust = cc.spice_elaborate(synth_path, ["corner", "devices"], overrides)
    ref = _ref_section_models(P.parse_spice_library(synth_path), ["corner", "devices"], overrides)
    assert set(rust) == set(ref)
    worst = 0.0
    total = 0
    for key in ref:
        assert rust[key]["name"] == ref[key]["name"]
        assert rust[key]["model_type"] == ref[key]["model_type"]
        worst = max(worst, _worst_rel(rust[key]["parameters"], ref[key]["parameters"], key))
        total += len(ref[key]["parameters"])
    assert total > 0
    assert worst <= _TOL


def test_synth_instance_numericization(synth_path):
    overrides = {"temper": 27.0, "process": 0.0}
    params = {"w": 1.5e-6, "l": 30e-9, "nf": 1, "mulu0": 1.2}
    rust = cc.spice_elaborate_instance(synth_path, ["corner", "devices"], "core", params, overrides)

    library = P.parse_spice_library(synth_path)
    elaborated = elaborate_library(library, ["corner", "devices"], initial_values=overrides)
    instance = elaborated.instantiate("core", dict(params))
    ref_models = [instance.numeric_model(s) for s in instance.model_statements]
    ref_elems = [(s.kind, s.name or "", instance.numeric_parameters(s)) for s in instance.elements]

    assert len(rust["models"]) == len(ref_models)
    for rm, pm in zip(rust["models"], ref_models):
        assert rm["name"] == (pm.name or "")
        assert rm["model_type"] == pm.model_type
        _worst_rel(rm["parameters"], pm.parameters, rm["name"])
    assert len(rust["elements"]) == len(ref_elems)
    for re_, (kind, name, pp) in zip(rust["elements"], ref_elems):
        assert re_["kind"] == kind and re_["name"] == name
        _worst_rel(re_["parameters"], pp, name)


def test_parameter_cycle_error_parity(tmp_path):
    path = tmp_path / "bad.l"
    path.write_text(
        ".lib devices\n.subckt bad d g s b w=l l=w\nm0 d g s b nch w=w l=l\n.ends bad\n.endl devices\n",
        encoding="ascii",
    )
    library = P.parse_spice_library(str(path))
    elaborated = elaborate_library(library, ["devices"])
    with pytest.raises(E.ParameterCycleError):
        elaborated.instantiate("bad").scope.resolve_symbol("w")
    with pytest.raises(cc.ParameterCycleError):
        cc.spice_elaborate_instance(str(path), ["devices"], "bad", None, None)


def test_section_cycle_and_missing_parity(tmp_path):
    path = tmp_path / "x.l"
    path.write_text(
        ".lib a\n.lib \"x.l\" b\n.endl a\n.lib b\n.lib \"x.l\" a\n.endl b\n",
        encoding="ascii",
    )
    library = P.parse_spice_library(str(path))
    with pytest.raises(SpiceElaborationError, match="a -> b -> a"):
        select_library_sections(library, ["a"])
    with pytest.raises(cc.SpiceElaborationError):
        cc.spice_select_sections(str(path), ["a"])
    with pytest.raises(cc.SpiceElaborationError):
        cc.spice_select_sections(str(path), ["missing"])
    assert issubclass(cc.SpiceElaborationError, ValueError)


def test_duplicate_model_subckt_name_parity(tmp_path):
    path = tmp_path / "dup.l"
    path.write_text(
        ".lib devices\n.model dup nmos level=54\n.subckt dup a b\nm0 a b nch\n.ends dup\n.endl devices\n",
        encoding="ascii",
    )
    library = P.parse_spice_library(str(path))
    with pytest.raises(P.SpiceSyntaxError, match="both models and subcircuits"):
        elaborate_library(library, ["devices"])
    with pytest.raises(cc.SpiceSyntaxError):
        cc.spice_elaborate(str(path), ["devices"], None)


# --- real TSMC28 library: deep numericization (D12: counts + worst_rel only)


def _tsmc_path():
    from circuitopt.toolchain import tsmc28_model_dir

    path = os.path.join(tsmc28_model_dir(), "cln28hpcp_1d8_elk_v1d0_2p2.l")
    return path if os.path.isfile(path) else None


_TSMC_SECTIONS = ["setup", "tt", "global", "total", "stat"]


def test_tsmc_real_instance_numericization():
    path = _tsmc_path()
    if path is None:
        pytest.skip("licensed TSMC28HPC+ model is not installed")

    library = P.parse_spice_library(path)
    overrides = {"temper": 27.0}
    elaborated = elaborate_library(library, _TSMC_SECTIONS, initial_values=overrides)

    worst = 0.0
    param_count = 0
    for macro in ("nch_mac", "pch_mac"):
        for (w_um, l_um) in ((1.0, 0.03), (0.15, 0.5), (5.0, 0.1)):
            params = {"w": w_um * 1e-6, "l": l_um * 1e-6, "nf": 1, "multi": 1, "_delvto": 0.0}
            rust = cc.spice_elaborate_instance(path, _TSMC_SECTIONS, macro, params, overrides)
            instance = elaborated.instantiate(macro, dict(params))
            ref_models = [instance.numeric_model(s) for s in instance.model_statements]
            ref_elems = [
                (s.kind, s.name or "", instance.numeric_parameters(s))
                for s in instance.elements
            ]
            assert len(rust["models"]) == len(ref_models)
            for rm, pm in zip(rust["models"], ref_models):
                assert rm["name"] == (pm.name or "")
                assert rm["model_type"] == pm.model_type
                worst = max(worst, _worst_rel(rm["parameters"], pm.parameters, rm["name"]))
                param_count += len(pm.parameters)
            assert len(rust["elements"]) == len(ref_elems)
            for re_, (kind, name, pp) in zip(rust["elements"], ref_elems):
                assert re_["kind"] == kind and re_["name"] == name
                worst = max(worst, _worst_rel(re_["parameters"], pp, name))
                param_count += len(pp)

    assert param_count >= 400
    assert worst <= _TOL


def test_tsmc_real_section_level_models():
    path = _tsmc_path()
    if path is None:
        pytest.skip("licensed TSMC28HPC+ model is not installed")

    overrides = {"temper": 27.0}
    rust = cc.spice_elaborate(path, _TSMC_SECTIONS, overrides)
    ref = _ref_section_models(P.parse_spice_library(path), _TSMC_SECTIONS, overrides)
    assert set(rust) == set(ref)
    worst = 0.0
    count = 0
    for key in ref:
        assert rust[key]["name"] == ref[key]["name"]
        assert rust[key]["model_type"] == ref[key]["model_type"]
        worst = max(worst, _worst_rel(rust[key]["parameters"], ref[key]["parameters"], key))
        count += len(ref[key]["parameters"])
    assert worst <= _TOL
