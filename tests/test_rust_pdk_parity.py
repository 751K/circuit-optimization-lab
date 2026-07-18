"""Differential parity: the Rust co-pdk compilers vs the frozen Python adapters.

For each PDK, `CompiledPdk.numeric_card` is compared against the Python
reference path (`circuitopt.pdk.{freepdk45,sky130,tsmc28}`). Model and instance
parameters must match bit-for-bit or within a relative error of 1e-14; when a
geometry does not bin, both sides must raise.

D12: no PDK parameter value is written to this file, a golden, or a log. Failure
messages carry only a parameter name and a relative error (never a card value);
the summary reports counts and the worst observed relative error. Each PDK is
skipped when its local delivery is absent.
"""
from __future__ import annotations

import os

import pytest

cc = pytest.importorskip("circuitopt_core")

if not hasattr(cc, "CompiledPdk"):
    pytest.skip(
        "compiled core lacks the PDK-compiler parity surface",
        allow_module_level=True,
    )

_TOL = 1e-14


def _cmp_params(rust: dict, ref: dict, where: str) -> float:
    """Compare two parameter maps; return the worst relative error. Raises a
    D12-safe message (parameter name + relative error only) on any mismatch."""
    if set(rust) != set(k.lower() for k in ref):
        raise AssertionError(f"{where}: parameter-name set mismatch")
    worst = 0.0
    for name, ref_value in ref.items():
        key = name.lower()
        rust_value = rust[key]
        ref_value = float(ref_value)
        if rust_value == ref_value:
            continue
        if ref_value == 0.0:
            raise AssertionError(f"{where}:{key} expected 0, got nonzero")
        rel = abs(rust_value - ref_value) / abs(ref_value)
        worst = max(worst, rel)
        if rel > _TOL:
            raise AssertionError(f"{where}:{key} rel={rel:.3e}")
    return worst


def _run(callable_):
    try:
        return callable_(), None
    except Exception as exc:  # noqa: BLE001 - differential error-parity check
        return None, exc


def _assert_card_parity(rust_card, ref_card, ref_getters, where):
    """Compare a Rust card dict and a Python `*Card` (or assert both errored)."""
    (rust_value, rust_err) = rust_card
    (ref_value, ref_err) = ref_card
    if rust_err is not None or ref_err is not None:
        assert (rust_err is not None) == (ref_err is not None), (
            f"{where}: error parity (rust={type(rust_err).__name__ if rust_err else None}, "
            f"python={type(ref_err).__name__ if ref_err else None})"
        )
        assert isinstance(ref_err, ValueError)
        return None
    worst = 0.0
    worst = max(worst, _cmp_params(rust_value["model_parameters"], ref_value.model_parameters, f"{where}.model"))
    worst = max(worst, _cmp_params(rust_value["instance_parameters"], ref_value.instance_parameters, f"{where}.instance"))
    for field, getter in ref_getters.items():
        assert rust_value[field] == getter(ref_value), f"{where}.{field}"
    return worst


# --------------------------------------------------------------------------
# FreePDK45
# --------------------------------------------------------------------------


def test_freepdk45_numeric_card_parity():
    from circuitopt.pdk.freepdk45.library import FREEPDK45_CORNERS, load_freepdk45_library
    from circuitopt.toolchain import pdk_root

    root = pdk_root()
    if not os.path.isfile(os.path.join(root, "freepdk45", "models_nom", "NMOS_VTG.inc")):
        pytest.skip("FreePDK45 cards not present")

    pdk = cc.CompiledPdk("freepdk45", root)
    getters = {"model_name": lambda c: c.model_name, "source_version": lambda c: c.source_version}
    worst = 0.0
    count = 0
    for polarity in ("nmos", "pmos"):
        for corner in FREEPDK45_CORNERS:
            for (w_um, l_um) in ((0.09, 0.05), (1.0, 0.1), (5.0, 0.2)):
                for (nf, mult) in ((1, 1), (2, 3)):
                    for mismatch in (0.0, 0.02):
                        where = f"freepdk45[{polarity},{corner},{w_um},{l_um},nf{nf},m{mult},mm{mismatch}]"
                        rust = _run(lambda p=polarity, c=corner, w=w_um, l=l_um, n=nf, m=mult, mm=mismatch:
                                    pdk.numeric_card(p, c, 27.0, w, l, n, m, mm or None))
                        ref = _run(lambda p=polarity, c=corner, w=w_um, l=l_um, n=nf, m=mult, mm=mismatch:
                                   load_freepdk45_library(p, c).device_card(
                                       width_um=w, length_um=l, nf=n, mult=m, mismatch_v=mm))
                        result = _assert_card_parity(rust, ref, getters, where)
                        if result is not None:
                            worst = max(worst, result)
                            count += 1
    assert count > 0
    assert worst <= _TOL


# --------------------------------------------------------------------------
# SKY130 (bundled cards)
# --------------------------------------------------------------------------


def _sky130_geometries(card_dir):
    for filename in sorted(f for f in os.listdir(card_dir) if f.endswith(".json")):
        stem = filename[:-5]
        if "nfet_01v8" in stem:
            polarity, prefix = "nmos", "sky130_fd_pr__nfet_01v8_"
        elif "pfet_01v8" in stem:
            polarity, prefix = "pmos", "sky130_fd_pr__pfet_01v8_"
        else:
            continue
        corner, w_token, l_token = stem[len(prefix):].split("_")
        yield polarity, corner, float(w_token[1:]), float(l_token[1:])


def test_sky130_numeric_card_parity():
    from circuitopt.pdk.sky130.library import _BUNDLED_CARD_DIR, load_sky130_card

    card_dir = str(_BUNDLED_CARD_DIR)
    if not os.path.isdir(card_dir):
        pytest.skip("SKY130 bundled cards not present")

    pdk = cc.CompiledPdk("sky130", card_dir)
    getters = {"model_name": lambda c: c.path.stem, "source_version": lambda c: c.source_version}
    worst = 0.0
    count = 0
    for (polarity, corner, w_um, l_um) in _sky130_geometries(card_dir):
        for (nf, mult) in ((1, 1), (2, 2)):
            for mismatch in (0.0, 0.01):
                where = f"sky130[{polarity},{corner},{w_um},{l_um},nf{nf},m{mult}]"
                rust = _run(lambda p=polarity, c=corner, w=w_um, l=l_um, n=nf, m=mult, mm=mismatch:
                            pdk.numeric_card(p, c, 27.0, w, l, n, m, mm or None))
                ref = _run(lambda p=polarity, c=corner, w=w_um, l=l_um, n=nf, m=mult, mm=mismatch:
                           load_sky130_card(p, width_um=w, length_um=l, nf=n, mult=m,
                                            corner=c, mismatch_v=mm))
                result = _assert_card_parity(rust, ref, getters, where)
                if result is not None:
                    worst = max(worst, result)
                    count += 1
    assert count > 100  # all bundled geometries
    assert worst <= _TOL


def test_sky130_extract_w_reference_width_parity():
    """`reference_width_um` (the `extract_w` path): the card bins on the reference
    width while the instance `w` keeps the *actual* width. Bit-for-bit vs the
    frozen `load_sky130_card(reference_width_um=...)`; the actual width is chosen
    to differ from the card-stem width so an accidental fall-through to the
    actual-width bin would either mis-select the card or corrupt instance `w`."""
    from circuitopt.pdk.sky130.library import _BUNDLED_CARD_DIR, load_sky130_card

    card_dir = str(_BUNDLED_CARD_DIR)
    if not os.path.isdir(card_dir):
        pytest.skip("SKY130 bundled cards not present")

    pdk = cc.CompiledPdk("sky130", card_dir)
    getters = {"model_name": lambda c: c.path.stem, "source_version": lambda c: c.source_version}
    worst = 0.0
    count = 0
    for (polarity, corner, ref_w_um, l_um) in _sky130_geometries(card_dir):
        # Actual width deliberately off the card-stem (reference) width.
        for actual_w in (ref_w_um * 2.5, ref_w_um * 0.4 + 0.123):
            for (nf, mult) in ((1, 1), (2, 2)):
                for mismatch in (0.0, 0.01):
                    where = (f"sky130.extract_w[{polarity},{corner},ref{ref_w_um},"
                             f"w{actual_w:.4g},{l_um},nf{nf},m{mult}]")
                    rust = _run(lambda p=polarity, c=corner, w=actual_w, rw=ref_w_um,
                                l=l_um, n=nf, m=mult, mm=mismatch:
                                pdk.numeric_card(p, c, 27.0, w, l, n, m, mm or None,
                                                 reference_width_um=rw))
                    ref = _run(lambda p=polarity, c=corner, w=actual_w, rw=ref_w_um,
                               l=l_um, n=nf, m=mult, mm=mismatch:
                               load_sky130_card(p, width_um=w, length_um=l, nf=n, mult=m,
                                                corner=c, reference_width_um=rw,
                                                mismatch_v=mm))
                    result = _assert_card_parity(rust, ref, getters, where)
                    if result is not None:
                        worst = max(worst, result)
                        count += 1
                        # The instance width must be the ACTUAL width, not the
                        # reference the card was binned on.
                        assert rust[0]["instance_parameters"]["w"] == actual_w * 1e-6, where
    assert count > 100  # every bundled geometry, both off-reference widths
    assert worst <= _TOL


# --------------------------------------------------------------------------
# TSMC28 (geometry grid + bin-boundary points)
# --------------------------------------------------------------------------


def _tsmc_dir():
    from circuitopt.toolchain import tsmc28_model_dir

    mdir = tsmc28_model_dir()
    if os.path.isfile(os.path.join(mdir, "cln28hpcp_1d8_elk_v1d0_2p2.l")):
        return mdir
    return None


def test_tsmc28_numeric_card_parity():
    mdir = _tsmc_dir()
    if mdir is None:
        pytest.skip("licensed TSMC28HPC+ model is not installed")

    from circuitopt.pdk.tsmc28.library import TSMC28_CORE_CORNERS, load_tsmc28_core_library

    pdk = cc.CompiledPdk("tsmc28", mdir)
    lib = load_tsmc28_core_library()
    getters = {"model_name": lambda c: c.bin_name, "model_type": lambda c: c.model_type}

    def ref_card(polarity, corner, temp, w, l, nf, mult, mm):
        return lib.core_card(polarity, width_um=w, length_um=l, nf=nf, mult=mult,
                             corner=corner, temperature_c=temp, mismatch_v=mm)

    worst = 0.0
    count = 0
    both_err = 0
    bins = {}
    grid = [(1.0, 0.03), (0.15, 0.5), (5.0, 0.1), (0.2, 0.03), (10.0, 1.0), (0.3, 0.03)]
    for polarity in ("nmos", "pmos"):
        for corner in TSMC28_CORE_CORNERS:
            for (w_um, l_um) in grid:
                for (nf, mult) in ((1, 1), (2, 4)):
                    for temp in (27.0, 85.0):
                        for mismatch in (0.0, 0.01):
                            where = f"tsmc28[{polarity},{corner},{w_um},{l_um},nf{nf},t{temp},mm{mismatch}]"
                            rust = _run(lambda p=polarity, c=corner, t=temp, w=w_um, l=l_um, n=nf, m=mult, mm=mismatch:
                                        pdk.numeric_card(p, c, t, w, l, n, m, mm or None))
                            ref = _run(lambda p=polarity, c=corner, t=temp, w=w_um, l=l_um, n=nf, m=mult, mm=mismatch:
                                       ref_card(p, c, t, w, l, n, m, mm))
                            result = _assert_card_parity(rust, ref, getters, where)
                            if result is None:
                                both_err += 1
                            else:
                                worst = max(worst, result)
                                count += 1
                                if rust[0] is not None:
                                    b = rust[0]["bin"]
                                    if b is not None:
                                        bins[b["name"]] = (b["lmin"], b["lmax"], b["wmin"], b["wmax"])

    # Bin-boundary points: probe each discovered bin exactly at its edges. The
    # half-open rule (l<lmax, w<wmax) must place edge geometries identically in
    # Rust and Python (the bounds are bit-identical between them).
    boundary_count = 0
    for (name, (lmin, lmax, wmin, wmax)) in bins.items():
        w_mid_um = (wmin + wmax) / 2.0 * 1e6
        l_lo_um = lmin * 1e6
        l_hi_um = lmax * 1e6
        w_lo_um = wmin * 1e6
        w_hi_um = wmax * 1e6
        l_mid_um = (lmin + lmax) / 2.0 * 1e6
        probes = [
            (w_mid_um, l_lo_um), (w_mid_um, l_hi_um),
            (w_lo_um, l_mid_um), (w_hi_um, l_mid_um),
        ]
        for (w_um, l_um) in probes:
            for polarity in ("nmos", "pmos"):
                where = f"tsmc28.boundary[{name},{polarity},{w_um:.6g},{l_um:.6g}]"
                rust = _run(lambda p=polarity, w=w_um, l=l_um: pdk.numeric_card(p, "tt", 27.0, w, l, 1, 1, None))
                ref = _run(lambda p=polarity, w=w_um, l=l_um: ref_card(p, "tt", 27.0, w, l, 1, 1, 0.0))
                result = _assert_card_parity(rust, ref, getters, where)
                if result is not None:
                    worst = max(worst, result)
                boundary_count += 1

    assert count > 0
    assert len(bins) >= 1
    assert boundary_count > 0
    assert worst <= _TOL
