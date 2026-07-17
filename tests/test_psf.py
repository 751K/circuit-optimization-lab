"""Direct unit tests for the generic PSFASCII parser (:mod:`circuitopt.psf`).

``circuitopt.psf`` is the read side of the whole Cadence calibration chain: the
byte-gate in :mod:`circuitopt.calibration` loads every Spectre reference file through
these parsers, so a parser regression surfaces as a mysterious "calibration
FAIL" far from its root cause. Until now the parser had only *indirect* coverage
via ``test_calibration.py`` (which also runs the full solver stack). These tests
exercise the parser in isolation against the real Spectre PSFASCII fixtures
checked into ``calibration/``, asserting the return *structure* plus a handful of
hand-picked anchor values read straight out of the fixture files -- so a break in
the parsing layer points here, not at a solver.

Anchor values are transcribed from the raw fixtures (line numbers noted inline);
they are exact Spectre doubles, so equality is checked with a tight rtol.

Pure local-file parsing -- no external simulator or PDK, no skips: the fixtures live
in the repo and are always available.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from circuitopt import psf

# Fixture roots, anchored off this file so cwd doesn't matter.
_CAL = Path(__file__).resolve().parent.parent / "calibration"
AMP = _CAL / "amp_design3_typical"
CHOP = _CAL / "chopper_design3_typical"
SC = _CAL / "sc_lpf"

AC = AMP / "ac.ac"
DC = AMP / "dcOp.dc"
NOISE = AMP / "noiseAnal.noise"
CHOP_PAC = CHOP / "pac.0.pac"
CHOP_PNOISE = CHOP / "pnoise.pnoise"
SC_PAC = SC / "pac.0.pac"
SC_PNOISE = SC / "pnoise.pnoise"
SC_PSS = SC / "pss.td.pss"


def _all_files():
    return [AC, DC, NOISE, CHOP_PAC, CHOP_PNOISE, SC_PAC, SC_PNOISE, SC_PSS]


def test_fixtures_present():
    """Guard: the tests below are meaningless if a fixture went missing."""
    for p in _all_files():
        assert p.is_file(), f"missing fixture {p}"


# ── HEADER / provenance ──────────────────────────────────────────────────────

def test_parse_header_amp_dc():
    """HEADER key->value: strings unquoted, numerics coerced to float.
    Anchors from dcOp.dc lines 2-15."""
    h = psf.parse_header(str(DC))
    assert h["PSFversion"] == "1.00"          # quoted string stays a str
    assert h["simulator"] == "spectre"
    assert h["version"] == "24.1.0.078"
    assert h["analysis type"] == "dc"
    assert h["analysis name"] == "dcOp"
    # bare numeric -> float (line 15: "temp" 2.7e+01)
    assert isinstance(h["temp"], float)
    assert h["temp"] == pytest.approx(27.0)


def test_provenance_amp_dc():
    """Compact provenance dict mapped from the HEADER."""
    p = psf.provenance(str(DC))
    assert p["psf_version"] == "1.00"
    assert p["simulator"] == "spectre"
    assert p["spectre_version"] == "24.1.0.078"
    assert p["analysis_type"] == "dc"
    assert p["analysis_name"] == "dcOp"
    assert p["fundamental"] is None            # DC has no periodic fundamental


def test_provenance_chopper_pac_fundamental():
    """Periodic analyses surface the fundamental in provenance().

    The chopper PAC HEADER spells the key ``"fundamental frequency"`` (pac.0.pac
    line 12 = 225 Hz), and provenance() reads exactly that (with a fallback to the
    bare ``"fundamental"`` spelling), so the compact dict now carries the true value
    -- matching the docstring promise "(for periodic analyses) the fundamental".
    parse_header exposes the same value under the real key."""
    p = psf.provenance(str(CHOP_PAC))
    assert p["analysis_type"] == "pac"
    assert p["fundamental"] == pytest.approx(225.0)        # surfaced, no longer None
    h = psf.parse_header(str(CHOP_PAC))                    # same value under real key
    assert h["fundamental frequency"] == pytest.approx(225.0)


# ── DC operating point ───────────────────────────────────────────────────────

def test_parse_dc_amp():
    """DC op -> {signal: float}. Anchors from dcOp.dc VALUE block (lines 145-148)."""
    dc = psf.parse_dc(str(DC))
    assert set(dc) == {"VOP", "VON", "vip", "vin"}
    for v in dc.values():
        assert isinstance(v, float) and np.isfinite(v)
    assert dc["VOP"] == pytest.approx(2.907924045932123e+01, rel=1e-12)
    assert dc["VON"] == pytest.approx(2.907924045932124e+01, rel=1e-12)
    assert dc["vip"] == pytest.approx(30.65, rel=1e-12)
    assert dc["vin"] == pytest.approx(30.65, rel=1e-12)


# ── AC / PAC (complex) ───────────────────────────────────────────────────────

def test_parse_ac_amp_structure_and_anchors():
    """AC -> (freqs, {signal: complex array}). 121-point dec sweep 0.01..10k Hz.
    Anchors: first freq point VOP (ac.ac lines 153-154), last VOP (tail)."""
    freqs, sig = psf.parse_ac(str(AC))
    assert freqs.shape == (121,)
    assert freqs.dtype == np.float64
    assert set(sig) == {"VOP", "VON", "vip", "vin"}
    for name, arr in sig.items():
        assert arr.shape == (121,)
        assert np.iscomplexobj(arr), name
    # first frequency point
    assert freqs[0] == pytest.approx(0.01, rel=1e-12)
    assert sig["VOP"][0] == pytest.approx(
        complex(-6.977473507513711e+00, 1.359962934080744e-04), rel=1e-12)
    assert sig["vip"][0] == pytest.approx(complex(0.5, 0.0), rel=1e-12)
    # last frequency point (10 kHz)
    assert freqs[-1] == pytest.approx(1.0e4, rel=1e-12)
    assert sig["VOP"][-1] == pytest.approx(
        complex(3.536466840790150e-01, 3.649525841554707e-01), rel=1e-12)


def test_parse_ac_freqs_monotonic():
    """The sweep axis is strictly ascending (xVecSorted 'ascending')."""
    freqs, _ = psf.parse_ac(str(AC))
    assert np.all(np.diff(freqs) > 0)


def test_parse_pac_is_parse_ac():
    """parse_pac is the same callable -- documents the alias contract."""
    assert psf.parse_pac is psf.parse_ac


def test_parse_pac_chopper_complex():
    """Chopper PAC parses the same (real imag) complex form.
    Anchors: first freq (10 mHz) voutp_f (pac.0.pac lines 159-162)."""
    freqs, sig = psf.parse_pac(str(CHOP_PAC))
    assert freqs.shape == (51,)
    # calibration.compare_pac reads these output/input node names
    for name in ("voutp_f", "voutn_f", "vinp", "vinn"):
        assert name in sig
        assert np.iscomplexobj(sig[name])
    assert freqs[0] == pytest.approx(0.01, rel=1e-12)
    assert sig["voutp_f"][0] == pytest.approx(
        complex(-5.916593426800734e+00, 8.249046632684048e-04), rel=1e-12)
    assert sig["vinp"][0] == pytest.approx(
        complex(5.000000000040556e-01, -2.405336694688234e-15), rel=1e-9)


def test_parse_pac_sc_lpf_single_ended():
    """SC-LPF PAC: single-ended VOUT/VIN. Anchors from pac.0.pac lines 156-158."""
    freqs, sig = psf.parse_pac(str(SC_PAC))
    assert freqs.shape == (41,)
    for name in ("VIN", "VOUT", "VMID", "CLK1", "CLK2"):
        assert name in sig
    assert freqs[0] == pytest.approx(0.1, rel=1e-12)
    assert sig["VIN"][0] == pytest.approx(
        complex(1.000000000000001e+00, 2.436799834301976e-20), rel=1e-12)
    assert sig["VOUT"][0] == pytest.approx(
        complex(1.002571096199836e+00, -6.040776152553803e-03), rel=1e-12)


# ── noise / pnoise ───────────────────────────────────────────────────────────

def test_parse_noise_amp_structure_and_anchors():
    """noise -> (freqs, out_asd, {device: (Nf, 3)}). 121-point sweep.
    Anchors: first freq, first 'out' ASD (noiseAnal.noise line 114:
    1.914713604956926e-03), last 'out' (4.043066091256795e-07), and the M15
    device tuple at f0 (lines 64-68: flicker, thermal, total)."""
    freqs, out, dev = psf.parse_noise(str(NOISE))
    assert freqs.shape == (121,)
    assert out.shape == (121,)
    assert freqs.dtype == np.float64 and out.dtype == np.float64
    # out ASD (V/sqrt(Hz)) positive, finite
    assert np.all(out > 0) and np.all(np.isfinite(out))
    assert out[0] == pytest.approx(1.914713604956926e-03, rel=1e-12)
    assert out[-1] == pytest.approx(4.043066091256795e-07, rel=1e-12)
    assert freqs[0] == pytest.approx(0.01, rel=1e-12)
    assert freqs[-1] == pytest.approx(1.0e4, rel=1e-12)
    # per-device (flicker, thermal, total) multi-line struct
    assert "M15" in dev
    assert dev["M15"].shape == (121, 3)
    np.testing.assert_allclose(
        dev["M15"][0],
        [2.183532345873407e-07, 1.674285874072218e-12, 2.183549088732148e-07],
        rtol=1e-12)
    # 'total' column >= 'flicker' column here (total = flicker + thermal-ish)
    assert dev["M15"][0, 2] >= dev["M15"][0, 0]


def test_parse_noise_all_devices_shape():
    """On the amp fixture every device is a pmos_TFT_behavioral struct, so every
    contribution is (Nf, 3) = flicker/thermal/total -- the layout calibration relies
    on when it column-selects. (The chopper pnoise fixture mixes 3- and 2-column
    structs; see test_parse_pnoise_chopper_anchors.)"""
    freqs, _out, dev = psf.parse_noise(str(NOISE))
    assert len(dev) > 1
    for name, arr in dev.items():
        assert arr.shape == (freqs.shape[0], 3), name
        assert np.all(np.isfinite(arr)), name


def test_parse_pnoise_is_parse_noise():
    assert psf.parse_pnoise is psf.parse_noise


def test_parse_pnoise_chopper_anchors():
    """Chopper pnoise: 37-point band. Anchor: first 'out' ASD
    (pnoise.pnoise line 187: 1.602306476120335e-05)."""
    freqs, out, dev = psf.parse_pnoise(str(CHOP_PNOISE))
    assert freqs.shape == (37,)
    assert out.shape == (37,)
    assert freqs[0] == pytest.approx(0.05, rel=1e-12)
    assert out[0] == pytest.approx(1.602306476120335e-05, rel=1e-12)
    # Struct width follows the TYPE declaration, NOT a fixed 3. MOSFETs
    # (pmos_TFT_behavioral) are (Nf, 3) = flicker/thermal/total; resistors here
    # (RLP_N/RLP_P, master 'resistor') declare only a single 'rn' field, so their
    # struct is (Nf, 2) in this fixture. Assert the actual mix so a change in how
    # ragged structs are handled is caught.  CURRENT BEHAVIOR.
    widths = {arr.shape[1] for arr in dev.values()}
    assert widths == {2, 3}, widths
    mos = [n for n, a in dev.items() if a.shape[1] == 3]
    res = [n for n, a in dev.items() if a.shape[1] == 2]
    assert mos and res
    for arr in dev.values():
        assert arr.shape[0] == 37


def test_parse_pnoise_sc_lpf_anchors():
    """SC-LPF pnoise: 34-point band. Anchors: first freq, first 'out'
    (pnoise.pnoise line 69: 4.695597451800385e-06) and the M2 struct at f0
    (lines 59-63)."""
    freqs, out, dev = psf.parse_pnoise(str(SC_PNOISE))
    assert freqs.shape == (34,)
    assert out.shape == (34,)
    assert freqs[0] == pytest.approx(0.1, rel=1e-12)
    assert out[0] == pytest.approx(4.695597451800385e-06, rel=1e-12)
    assert "M2" in dev
    np.testing.assert_allclose(
        dev["M2"][0],
        [1.379000733660288e-11, 1.595886656894498e-14, 1.380596620317182e-11],
        rtol=1e-12)


# ── transient / PSS time-domain ──────────────────────────────────────────────

def test_parse_pss_is_parse_tran():
    assert psf.parse_pss is psf.parse_tran


def test_parse_tran_sc_lpf_structure_and_anchors():
    """PSS td -> (time, {signal: real array}). 510-point period.
    Anchors: t0 signals (pss.td.pss lines 177-181), monotone time, final time."""
    time, sig = psf.parse_tran(str(SC_PSS))
    assert time.shape == (510,)
    assert time.dtype == np.float64
    for name in ("VIN", "VOUT", "VMID", "CLK1", "CLK2"):
        assert name in sig
        assert sig[name].shape == (510,)
        assert sig[name].dtype == np.float64
        assert not np.iscomplexobj(sig[name])
    # first time sample
    assert time[0] == pytest.approx(0.0, abs=1e-18)
    assert sig["VIN"][0] == pytest.approx(20.0, rel=1e-12)
    assert sig["VOUT"][0] == pytest.approx(2.003215816622399e+01, rel=1e-12)
    assert sig["CLK1"][0] == pytest.approx(0.0, abs=1e-18)
    # time axis strictly increasing; last point is one clock period (1 ms)
    assert np.all(np.diff(time) > 0)
    assert time[-1] == pytest.approx(1.0e-3, rel=1e-9)


def test_parse_tran_signals_filter():
    """The optional ``signals`` arg restricts which traces are kept."""
    time, sig = psf.parse_tran(str(SC_PSS), signals=["VOUT"])
    assert set(sig) == {"VOUT"}
    assert sig["VOUT"].shape == time.shape


# ── error / edge paths (documents *current* behavior) ────────────────────────

def test_missing_file_raises(tmp_path):
    """A path that doesn't exist -> OSError from the open()."""
    missing = tmp_path / "nope.ac"
    with pytest.raises((FileNotFoundError, OSError)):
        psf.parse_ac(str(missing))


def test_no_value_section_raises(tmp_path):
    """A file with a HEADER but no VALUE marker -> ValueError with a clear message
    (psf._value_lines guards this explicitly)."""
    stub = tmp_path / "stub.dc"
    stub.write_text('HEADER\n"PSFversion" "1.00"\nTYPE\n"V" FLOAT DOUBLE\n')
    with pytest.raises(ValueError, match="no VALUE section"):
        psf.parse_dc(str(stub))


def test_empty_file_raises(tmp_path):
    """A completely empty file also trips the no-VALUE guard."""
    stub = tmp_path / "empty.dc"
    stub.write_text("")
    with pytest.raises(ValueError, match="no VALUE section"):
        psf.parse_dc(str(stub))


def test_truncated_ac_returns_partial_current_behavior(tmp_path):
    """Truncating a VALUE block mid-file (no END marker, no trailing data).

    CURRENT BEHAVIOR, NOT A CONTRACT: _value_lines falls back to end-of-file when
    no END marker is found, so the parser returns whatever complete points it read
    without raising. Here we keep only the first frequency point's records, so the
    result is a well-formed 1-point sweep. Documented so a future change to the
    truncation policy is caught deliberately, not by surprise."""
    lines = AC.read_text().splitlines()
    # VALUE at index 151 (line 152); keep header..first point's 5 records, drop END.
    i0 = lines.index("VALUE")
    truncated = lines[:i0 + 1 + 5]            # VALUE + freq + VOP/VON/vip/vin
    stub = tmp_path / "trunc.ac"
    stub.write_text("\n".join(truncated) + "\n")

    freqs, sig = psf.parse_ac(str(stub))
    assert freqs.shape == (1,)                # no raise; partial data returned
    assert freqs[0] == pytest.approx(0.01, rel=1e-12)
    assert sig["VOP"][0] == pytest.approx(
        complex(-6.977473507513711e+00, 1.359962934080744e-04), rel=1e-12)


# ── contract with circuitopt.calibration ───────────────────────────────────────────

def test_calibration_contract_amp_ac_transfer():
    """The differential transfer calibration.compare_ac builds from parse_ac
    output is finite and non-trivial on the amp fixture."""
    _f, sig = psf.parse_ac(str(AC))
    H = np.abs((sig["VOP"] - sig["VON"]) / (sig["vip"] - sig["vin"]))
    assert H.shape == (121,)
    assert np.all(np.isfinite(H)) and np.all(H > 0)
    # DC gain (low-freq) clearly above unity for this amp
    assert H[0] > 1.0


def test_calibration_contract_amp_noise_asd():
    """The 'out' ASD that compare_noise refers to input is non-empty and finite."""
    fr, out, _dev = psf.parse_noise(str(NOISE))
    assert out.shape == fr.shape
    assert np.all(np.isfinite(out)) and np.all(out > 0)


def test_calibration_contract_chopper_pac_baseband_gain():
    """compare_pac reads sideband-0 complex nodes to form the baseband gain."""
    _f, sig = psf.parse_pac(str(CHOP_PAC))
    gain = np.abs((sig["voutp_f"][0] - sig["voutn_f"][0])
                  / (sig["vinp"][0] - sig["vinn"][0]))
    assert np.isfinite(gain) and gain > 0
