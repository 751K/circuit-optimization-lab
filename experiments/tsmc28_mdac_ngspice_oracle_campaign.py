#!/usr/bin/env python3
"""Explicit ngspice-oracle 45-point TSMC28HPC+ MDAC OTA regression.

This specializes the established MDAC campaign for TT/SS/FF/SF/FS,
-40/27/125 C, and 0.85/0.90/0.95 V.  In addition to the five lumped-residue
levels it runs the split-CDAC 0111->1000 major-carry transition.  Transient
device operating vectors are used for the settled saturation check so sampled
charge is not lost through an invalid replacement DC operating point.
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))
sys.path.insert(0, str(ROOT / "experiments"))

import freepdk45_mdac_ngspice_oracle_campaign as campaign  # noqa: E402
import tsmc28_mdac_ota_gen as G  # noqa: E402
from circuitopt.circuit_loader import circuit_from_dict  # noqa: E402
from circuitopt.ngspice_ac import (  # noqa: E402
    ac_ngspice as _ac_ngspice,
    loop_gain_ngspice as _loop_gain_ngspice,
    noise_ngspice as _noise_ngspice,
    op_ngspice as _op_ngspice,
)
from circuitopt.ngspice_transient import (  # noqa: E402
    transient_ngspice as _transient_ngspice,
    transient_ngspice_chain as _transient_ngspice_chain,
)


campaign.G = G
campaign.CORNERS = ["tt", "ss", "ff", "sf", "fs"]
campaign.TEMPS_C = [-40.0, 27.0, 125.0]
campaign.SUPPLIES = [0.85, 0.90, 0.95]
campaign.SPEC_GAIN_DB = 80.0
campaign.OUT_CSV = ROOT / "results" / "tsmc28_mdac_ota_pvt45.csv"
# The generator owns the authoritative list of saturation-checked core devices.
# When a colleague collapses the parallel instances (M0B/M0C, M9B, ... folded into an
# M-multiplier on M0/M9/...) they will export ``CORE_SAT_DEVICES``; consuming it via
# getattr keeps this campaign correct both before and after that refactor.
_CORE_DEVS_DEFAULT = [
    "M0", "M0B", "M0C", *[f"M{i}" for i in range(1, 13)], "M9B", "M10B",
    "M11B", "M11C", "M12B", "M12C",
]
campaign.CORE_DEVS = list(getattr(G, "CORE_SAT_DEVICES", _CORE_DEVS_DEFAULT))

# Risk-first order: the corners most likely to fail (slow/cold or fast/hot rails)
# run before the bulk of the grid so a broken build surfaces early.
campaign.GRID_PRIORITY = [
    ("ss", 125.0, 0.85),
    ("sf", -40.0, 0.85),
    ("ss", -40.0, 0.85),
    ("ff", 125.0, 0.95),
    ("fs", -40.0, 0.95),
    ("tt", 27.0, 0.90),
]

EXTRA_FIELDS = [
    "settle_time_worst_ns",
    "cm_static_signed_mv", "cm5_min_signed_mv", "cm5_max_signed_mv",
    "noise_wideband_onoise_uv", "noise_wideband_adc_uv",
    "code_transition_pct", "code_settle_ns", "code_peak_glitch_pct", "code_cm5_mv",
    "code_cm5_signed_mv",
    "pass_code_transition",
    "smoke",
]
campaign.CSV_FIELDS = campaign.CSV_FIELDS + EXTRA_FIELDS

_tls = threading.local()
_base_run_point = campaign.run_point
_base_transient_batch = campaign.transient_batch


def _with_timeout(function, timeout):
    def wrapped(*args, **kwargs):
        kwargs.setdefault("timeout", timeout)
        return function(*args, **kwargs)
    return wrapped


campaign.ac_ngspice = _with_timeout(_ac_ngspice, 900.0)

# Prefer the Tian double-injection loop-gain probe when the colleague's version is
# present (same signature/return shape); otherwise fall back to the Middlebrook
# single-injection one.  Applies to the DM and both CMFB loops, which all route
# through campaign.loop_gain_ngspice.
try:
    from circuitopt.ngspice_ac import loop_gain_tian_ngspice as _loop_gain_impl
except ImportError:
    _loop_gain_impl = _loop_gain_ngspice
campaign.loop_gain_ngspice = _with_timeout(_loop_gain_impl, 900.0)


def _hold_phase_noise(*args, **kwargs):
    """Closed-loop hold-phase output noise for the MDAC first stage.

    Two quantities come out of one sweep:

    * The WIDEBAND integral 1e5-2e10 (``_tls.noise_wideband``) is the sign-off
      quantity.  The next pipeline stage samples the full-bandwidth output noise, so
      aliasing folds the entire PSD into the Nyquist band; the SPEC_NOISE_UV budget
      was derived for exactly that integral.  The lower bound is pushed down to
      1e5 Hz to capture the 0.1-10 MHz flicker tail while staying above the ~30 kHz
      DC-helper artifact corner of this testbench.
    * The 10-50 MHz narrowband rms (``result["onoise_rms"]`` via ``band``) is
      REPORT-ONLY (the ``noise_onoise_uv`` / ``noise_adc_uv`` columns).  It is NOT
      the pass/fail gate.

    The sweep starts at 1e5 with >=20 points/decade so the trapezoid over
    1e5-2e10 is trustworthy (the oracle's ``points`` argument is per decade)."""
    kwargs["fstart"] = 1e5
    kwargs["fstop"] = 2e10
    kwargs["band"] = (1e7, 5e7)          # narrowband report window (10-50 MHz)
    kwargs["points"] = max(int(kwargs.get("points", 20)), 20)   # >=20 pts/decade
    kwargs["noiseless_resistors"] = {"RDC1", "RDC2"}
    kwargs.setdefault("timeout", 1000.0)
    result = _noise_ngspice(*args, **kwargs)
    mask = (result["freq"] >= 1e5) & (result["freq"] <= 2e10)
    wideband = float(np.sqrt(np.trapezoid(
        result["onoise_psd"][mask], result["freq"][mask]
    )))
    _tls.noise_wideband = wideband
    return result


campaign.noise_ngspice = _hold_phase_noise


def _transient_with_op(*args, **kwargs):
    kwargs.setdefault("timeout", 1200.0)
    kwargs["op_devices"] = campaign.CORE_DEVS
    inputs = dict(kwargs.get("inputs") or {})
    inputs.update(G.hold_clock_inputs(args[2], args[1]["VDD"]))
    kwargs["inputs"] = inputs
    result = _transient_ngspice(*args, **kwargs)
    _tls.transients.append(result)
    return result


def _transient_batch_with_op(spec_args, cases, **shared_kwargs):
    """Batch counterpart of :func:`_transient_with_op` — same per-level contracts.

    Chaining off: route through the base per-case loop, whose calls hit the
    ``campaign.transient_ngspice`` override above, so every level still gets the
    hold-clock inputs, ``op_devices=CORE_DEVS``, the 1200 s timeout, and its own
    ``_tls.transients`` append — byte-for-byte today's behaviour.

    Chaining on: merge the hold-clock inputs (identical across levels, so their
    PWL sources are never altered) into every case, thread ``op_devices`` and a
    whole-chain timeout, run ONE ngspice process for all levels, and append the
    results to ``_tls.transients`` in case order (``campaign.RESIDUE_LEVELS``
    order — the settled-saturation shim answers call 0 from ``transients[0]``
    and call 1 from ``transients[-1]`` exactly as before)."""
    if not campaign.ngspice_chain_enabled():
        return campaign._transient_batch_loop(spec_args, cases, **shared_kwargs)
    hold_inputs = G.hold_clock_inputs(spec_args[2], spec_args[1]["VDD"])
    chain_cases = [
        dict(case, inputs={**dict(case["inputs"] or {}), **hold_inputs})
        for case in cases
    ]
    kwargs = dict(shared_kwargs)
    kwargs.setdefault("timeout", 1200.0 * max(1, len(chain_cases)))
    kwargs["op_devices"] = campaign.CORE_DEVS
    results = _transient_ngspice_chain(*spec_args, cases=chain_cases, **kwargs)
    _tls.transients.extend(results)
    return results


def _op_with_transient_settled(*args, **kwargs):
    """Answer the settled-saturation .op checks from the real .tran device vectors.

    The base run_point now folds the STATIC saturation check into the power .op
    (``_supply_current`` prints vds/vdsat on the adapter path), so on TSMC28 the
    static ``op_ngspice`` call no longer happens and every call that reaches this
    shim is a settled check.  The two settled checks arrive in the base loop order
    ``(-FS/16, +FS/16)`` -> map call 0 to the -FS/16 transient (``_tls.transients[0]``)
    and call 1 to the +FS/16 transient (``_tls.transients[-1]``).

    If a static call ever does reach here (e.g. the base falls back to a real static
    ``op_ngspice`` because no adapter was resolved), ``_tls.transients`` is still
    empty, so run the real oracle instead of indexing an empty list."""
    call = _tls.op_calls
    _tls.op_calls += 1
    if not _tls.transients:
        kwargs.setdefault("timeout", 900.0)
        return _op_ngspice(*args, **kwargs)

    trace_index = 0 if call == 0 else -1
    final = _tls.transients[trace_index]["device_op_final"]
    return {
        name: dict(values, region_ok=bool(abs(values["vds"]) >= abs(values["vdsat"])))
        for name, values in final.items()
    }


campaign.transient_ngspice = _transient_with_op
campaign.transient_batch = _transient_batch_with_op
campaign.op_ngspice = _op_with_transient_settled


def _code_transition(corner, tk, vdd):
    spec = circuit_from_dict(G.build_code_transition(vdd))
    binding = spec.binding()
    base_kwargs = binding.device_kwargs or {}
    device_kwargs = {
        name: dict(base_kwargs.get(name, {}), temperature=tk)
        for name, *_ in spec.topology.devices
    }
    tgrid = np.linspace(0.0, 5e-9, 501)
    initial = spec.topology.dc_guesses[0]
    v0 = np.array([initial.get(name, 0.0) for name in spec.topology.solved])
    start = time.time()
    result = _transient_ngspice(
        spec.sizes, spec.bias, tgrid, topo=spec.topology, nf=spec.nf,
        model_types=binding.model_types, device_kwargs=device_kwargs, corner=corner,
        V0=v0, inputs=G.code_transition_inputs(tgrid, vdd),
        extra_options=campaign.TIGHT, max_step=10e-12, timeout=1500.0,
        op_devices=campaign.CORE_DEVS,
    )
    vod = result["nodes"]["OUTP"] - result["nodes"]["OUTN"]
    vcm = (result["nodes"]["OUTP"] + result["nodes"]["OUTN"]) / 2.0
    error = np.abs(vod + 0.45)
    inside = error <= 0.00045
    settle = float("nan")
    for pos in range(len(tgrid)):
        if np.all(inside[pos:]):
            settle = float(tgrid[pos])
            break
    final = result["device_op_final"]
    bad = [name for name in campaign.CORE_DEVS
           if abs(final[name]["vds"]) < abs(final[name]["vdsat"])]
    return {
        "code_transition_pct": float(error[-1] / 0.45 * 100.0),
        "code_settle_ns": settle * 1e9,
        "code_peak_glitch_pct": float(np.max(error) / 0.45 * 100.0),
        "code_cm5_mv": float(abs(vcm[-1] - vdd / 2.0) * 1e3),
        "code_cm5_signed_mv": float((vcm[-1] - vdd / 2.0) * 1e3),
        "code_sat_bad": bad,
        "code_runtime_s": time.time() - start,
    }


def run_point(corner, temp_c, vdd):
    _tls.transients = []
    _tls.op_calls = 0
    _tls.noise_wideband = float("nan")
    row = _base_run_point(corner, temp_c, vdd)
    row["noise_wideband_onoise_uv"] = _tls.noise_wideband * 1e6
    row["noise_wideband_adc_uv"] = _tls.noise_wideband / 8.0 * 1e6

    # Report the first time every later sample remains inside the 0.1%-FS band.
    settle_times = []
    for residue, trace in zip(campaign.RESIDUE_LEVELS, _tls.transients, strict=True):
        vod = trace["nodes"]["OUTP"] - trace["nodes"]["OUTN"]
        inside = np.abs(vod + 8.0 * residue) <= 0.00045
        settle = float("nan")
        for pos in range(len(trace["t"])):
            if np.all(inside[pos:]):
                settle = float(trace["t"][pos])
                break
        settle_times.append(settle)
    # A level that never enters the error band contributes nan; the WORST settle
    # time must then be nan too, not a finite max().  Bare ``max()`` with a nan is
    # position-dependent and can silently drop the never-settled level, so gate on
    # nan explicitly.
    if any(np.isnan(t) for t in settle_times):
        row["settle_time_worst_ns"] = float("nan")
    else:
        row["settle_time_worst_ns"] = max(settle_times) * 1e9

    cm_static = (_tls.transients[0]["nodes"]["OUTP"][0]
                 + _tls.transients[0]["nodes"]["OUTN"][0]) / 2.0 - vdd / 2.0
    cm5_signed = [
        (trace["nodes"]["OUTP"][-1] + trace["nodes"]["OUTN"][-1]) / 2.0
        - vdd / 2.0
        for trace in _tls.transients
    ]
    row["cm_static_signed_mv"] = float(cm_static * 1e3)
    row["cm5_min_signed_mv"] = float(min(cm5_signed) * 1e3)
    row["cm5_max_signed_mv"] = float(max(cm5_signed) * 1e3)

    # Noise sign-off is the WIDEBAND integral (sampled/aliased into the next stage's
    # Nyquist band), NOT the 10-50 MHz narrowband report the base loop stored in
    # ``noise_onoise_uv``.  Override pass_noise here, BEFORE pass_all is recomputed
    # below, so the campaign gates on the physically meaningful quantity.  In smoke
    # mode noise never ran, so it cannot sign off -> False (and it is dropped from
    # pass_all further down).
    row["pass_noise"] = bool(
        (not campaign.SKIP_NOISE)
        and np.isfinite(row["noise_wideband_onoise_uv"])
        and row["noise_wideband_onoise_uv"] <= campaign.SPEC_NOISE_UV
    )

    if campaign.SKIP_CODE_TRANSITION:
        # Slowest per-point run; skipped in smoke mode.  Columns are nan and the
        # spec is NOT signed off (pass_code_transition False, dropped from pass_all).
        for column in ("code_transition_pct", "code_settle_ns", "code_peak_glitch_pct",
                       "code_cm5_mv", "code_cm5_signed_mv"):
            row[column] = float("nan")
        row["pass_code_transition"] = False
        code_bad = []
    else:
        code = _code_transition(corner, temp_c + 273.15, vdd)
        row["code_transition_pct"] = code["code_transition_pct"]
        row["code_settle_ns"] = code["code_settle_ns"]
        row["code_peak_glitch_pct"] = code["code_peak_glitch_pct"]
        row["code_cm5_mv"] = code["code_cm5_mv"]
        row["code_cm5_signed_mv"] = code["code_cm5_signed_mv"]
        row["pass_code_transition"] = bool(
            code["code_transition_pct"] < campaign.SPEC_SETTLE_PCT
            and code["code_settle_ns"] <= 5.0
            and code["code_cm5_mv"] < campaign.SPEC_CM_MV
            and not code["code_sat_bad"]
        )
        code_bad = [f"{name}@0111->1000" for name in code["code_sat_bad"]]

    all_bad = [item for item in row["sat_bad"].split(";") if item] + code_bad
    row["sat_bad"] = ";".join(dict.fromkeys(all_bad))
    row["pass_sat"] = bool(row["sat_static_ok"] and row["sat_settled_ok"]
                           and not code_bad)
    # The output-CM requirement applies both at quiescence and after every real
    # residue level has settled, not merely to the initial operating point.
    row["pass_cm"] = bool(max(row["cm_static_mv"], row["cm5_worst_mv"])
                          < campaign.SPEC_CM_MV)

    # pass_all covers only the specs actually MEASURED at this point.  In smoke mode
    # the skipped specs (noise, code transition) are excluded so the aggregate is an
    # honest reflection of the measured subset rather than a false green; the
    # ``smoke`` column records which regime produced the row.
    row["smoke"] = 1 if (campaign.SKIP_NOISE or campaign.SKIP_CODE_TRANSITION) else 0
    measured = ["pass_gain", "pass_dmpm", "pass_cmfb1pm", "pass_cmfb2pm",
                "pass_settle", "pass_cm", "pass_sat"]
    if not campaign.SKIP_NOISE:
        measured.append("pass_noise")
    if not campaign.SKIP_CODE_TRANSITION:
        measured.append("pass_code_transition")
    row["pass_all"] = all(bool(row[name]) for name in measured)
    return row


campaign.run_point = run_point


if __name__ == "__main__":
    campaign.main()
