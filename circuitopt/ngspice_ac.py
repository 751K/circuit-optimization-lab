"""Full-circuit model-card oracles through ngspice: ``.ac`` / ``.noise`` / ``.op``.

The default FreePDK45 and TSMC28HPC+ paths are native. These helpers render the
complete circuit and let ngspice run the original model deck as an independent
reference for AC bandwidth, phase margin, noise, and operating-region checks.
They also reproduce historical FreePDK45 grid-based design records.

All four entry points share the deck renderer in :mod:`circuitopt.ngspice_render`
(the same one the ``.tran`` backend uses), so device M-lines, R/C, controlled sources,
rails, per-polarity corner routing (nom/tt/ss/ff + mixed sf/fs), temperature and the
supply bias are rendered identically across analyses.

Public API
----------
``ac_ngspice``      — complex transfer to nodes over an ``.ac dec`` sweep.
``noise_ngspice``   — output / input-referred noise PSD + integrated band rms.
``op_ngspice``      — per-device operating point with a saturation-region check.
``loop_gain_ngspice`` — Middlebrook single-voltage-injection loop gain + PM/GM.
``loop_gain_tian_ngspice`` — Tian double-injection loop gain (exact at capacitive
breaks where the single-injection high-Z premise fails).
plus response helpers ``dc_gain_db`` / ``unity_gain_freq`` / ``phase_margin`` /
``gain_margin_db`` / ``ac_response``.
"""
from __future__ import annotations

import os
import subprocess
import tempfile

import numpy as np

from .device_factory import apply_silicon_corner
from .ngspice_char import _ngspice, _run_ngspice, ngspice_chain_enabled
from .ngspice_render import (
    _element, build_node_map, nodeset_line, render_controlled,
    render_devices, render_passives, render_rail_sources, resolve_common_temperature,
    resolve_ngspice_preamble, seed_vector)


# ── shared network deck (everything above the `.control` block) ─────────────────
def _network_deck(topo, sizes, bias, *, header, nf, model_types, device_kwargs,
                  corner, temperature, x0_guess, ac=None):
    """Render the circuit network (includes, options, rails, devices, passives,
    controlled sources, ``.nodeset`` seed) shared by ``.ac``/``.noise``/``.op``.

    ``ac`` maps an AC-stimulus source name (a rail name or an ideal-vsource name) to
    ``(magnitude, phase_deg)``. Returns ``(lines, node_map, node, adapter)``."""
    model_types = dict(model_types or {})
    device_kwargs, solver_corner = apply_silicon_corner(model_types, device_kwargs, corner)
    if solver_corner not in (None, {}):
        raise ValueError(f"FreePDK45 ngspice analysis needs a silicon corner, got {corner!r}")
    device_kwargs = {k: dict(v) for k, v in (device_kwargs or {}).items()}
    device_names = {name for name, *_ in topo.devices}

    adapter, _corner, preamble = resolve_ngspice_preamble(
        model_types, device_kwargs, device_names)
    temp_c = resolve_common_temperature(device_kwargs, device_names, temperature)
    node_map, node = build_node_map(topo, bias, node_inputs=None)

    lines = [header, *preamble, f".options temp={temp_c:g}"]
    rail_lines, _bv = render_rail_sources(topo, bias, None, node, ac=ac)
    lines.extend(rail_lines)
    dev_lines, _b2 = render_devices(topo, sizes, bias, None, node, nf=nf,
                                    model_types=model_types, device_kwargs=device_kwargs,
                                    adapter=adapter)
    lines.extend(dev_lines)
    lines.extend(render_passives(topo, node))
    ctrl_lines, _b3, _names = render_controlled(topo, node, ac=ac)
    lines.extend(ctrl_lines)

    seed = seed_vector(topo, x0_guess)
    ns = nodeset_line(topo, node_map, seed)
    if ns is not None:
        lines.append(ns)
    return lines, node_map, node, adapter


def _resolve_source_name(topo, name: str) -> str:
    """Deck element name for an AC stimulus / noise input reference: an ideal
    vsource ``name`` renders as ``_element('V', name)``; a rail renders as
    ``_element('V', 'rail_'+name)``. Accepts either topology name."""
    if any(vs[0] == name for vs in topo.vsources):
        return name          # render_rail_sources/render_controlled key AC by topo name
    if name in topo.rails:
        return name
    raise ValueError(f"AC/noise source {name!r} is neither a vsource nor a rail")


def _stimulus_deck_element(topo, name: str) -> str:
    """The actual SPICE element name (for ``.noise <src>``) of a stimulus source."""
    if any(vs[0] == name for vs in topo.vsources):
        return _element("V", name)
    if name in topo.rails:
        return _element("V", "rail_" + name)
    raise ValueError(f"source {name!r} is neither a vsource nor a rail")


# ── .ac ─────────────────────────────────────────────────────────────────────────
def ac_ngspice(sizes, bias, *, topo, acmag, fstart, fstop, points=20,
               out_nodes=None, nf=None, model_types=None, device_kwargs=None,
               corner=None, temperature=None, x0_guess=None, timeout=300.0):
    """Full-circuit ngspice ``.ac dec`` — complex node transfer over frequency.

    ``acmag`` maps a stimulus source (a rail name or an ideal-vsource name) to
    ``(magnitude, phase_deg)``; a differential drive is two sources with opposite
    phase (e.g. ``{"VINP": (0.5, 0), "VINN": (0.5, 180)}``). The DC operating point
    is found from the circuit's bias (seed the multistable OTA via ``x0_guess`` →
    ``.nodeset``), then ``.ac dec <points> <fstart> <fstop>`` runs.

    Returns ``{"freq": f, "nodes": {name: complex[]}, "acmag": ...}`` — one complex
    array per recorded node (default every solved node, or ``out_nodes``). Combine
    with :func:`ac_response` and the ``dc_gain_db`` / ``unity_gain_freq`` /
    ``phase_margin`` helpers. Temperature (K), corner (incl. sf/fs) and supply bias
    are all honored.
    """
    acmag = {k: (float(v[0]), float(v[1])) for k, v in dict(acmag or {}).items()}
    ac = {_resolve_source_name(topo, k): v for k, v in acmag.items()}
    record = list(out_nodes) if out_nodes is not None else list(topo.solved)
    for name in record:
        if name not in topo.idx:
            raise ValueError(f"ac_ngspice out node {name!r} is not a solved node")

    lines, node_map, _node, adapter = _network_deck(
        topo, sizes, bias, header="* circuitopt FreePDK45 full-circuit .ac",
        nf=nf, model_types=model_types, device_kwargs=device_kwargs, corner=corner,
        temperature=temperature, x0_guess=x0_guess, ac=ac)

    with tempfile.TemporaryDirectory(prefix="circuitopt-fp45-ac-") as td:
        out_path = os.path.join(td, "ac.dat")
        deck_path = os.path.join(td, "deck.cir")
        vecs = []
        for name in record:
            vecs.append(f"real(v({node_map[name]}))")
            vecs.append(f"imag(v({node_map[name]}))")
        lines.extend([
            ".control", "set filetype=ascii", "set wr_singlescale", "set wr_vecnames",
            f"ac dec {int(points):d} {float(fstart):.17g} {float(fstop):.17g}",
            f"wrdata {out_path} " + " ".join(vecs),
            ".endc", ".end",
        ])
        with open(deck_path, "w", encoding="ascii") as fh:
            fh.write("\n".join(lines) + "\n")
        _run_ngspice(
            deck_path, out_path, timeout=timeout, what="full-circuit .ac",
            extra_args=adapter.command_args if adapter is not None else ())
        raw = np.loadtxt(out_path, skiprows=1, ndmin=2)

    freq = raw[:, 0]
    nodes = {}
    for i, name in enumerate(record):
        re = raw[:, 1 + 2 * i]
        im = raw[:, 2 + 2 * i]
        nodes[name] = re + 1j * im
    return {"freq": freq, "nodes": nodes, "acmag": acmag}


def ac_response(result, out, ref=None, *, vin=1.0):
    """Complex transfer ``(V(out) [- V(ref)]) / vin`` from an :func:`ac_ngspice` result.

    ``out``/``ref`` name recorded solved nodes (``ref`` gives a differential output);
    ``vin`` normalises by the differential input magnitude (1.0 leaves node voltages)."""
    nodes = result["nodes"]
    resp = np.asarray(nodes[out], complex)
    if ref is not None:
        resp = resp - np.asarray(nodes[ref], complex)
    return resp / float(vin)


# ── response metrics ────────────────────────────────────────────────────────────
def dc_gain_db(freq, resp):
    """Low-frequency gain in dB (magnitude of *resp* at the lowest swept frequency).

    NB: for an AC-coupled input this is on the high-pass slope, not the passband —
    use :func:`peak_gain_db` for the mid-band gain of a bandpass response."""
    resp = np.asarray(resp, complex)
    mag = float(np.abs(resp[int(np.argmin(np.asarray(freq, float)))]))
    return 20.0 * np.log10(max(mag, 1e-300))


def peak_gain_db(freq, resp):
    """Maximum magnitude of *resp* over the sweep, in dB — the passband gain of a
    band-pass (AC-coupled) or low-pass response."""
    return 20.0 * np.log10(max(float(np.max(np.abs(np.asarray(resp, complex)))), 1e-300))


def unity_gain_freq(freq, resp):
    """Unity-gain (0 dB) crossover frequency [Hz] — first |resp|=1 crossing above the
    peak, log-interpolated. ``nan`` if the response never reaches unity."""
    freq = np.asarray(freq, float)
    mag = np.abs(np.asarray(resp, complex))
    order = np.argsort(freq)
    freq, mag = freq[order], mag[order]
    ipk = int(np.argmax(mag))
    if mag[ipk] < 1.0:
        return float("nan")
    for i in range(ipk + 1, len(mag)):
        if mag[i] <= 1.0:
            f0, f1 = freq[i - 1], freq[i]
            g0, g1 = np.log10(max(mag[i - 1], 1e-300)), np.log10(max(mag[i], 1e-300))
            if f0 <= 0 or f1 <= 0 or g1 == g0:
                return float(f1)
            x0, x1 = np.log10(f0), np.log10(f1)
            x = x0 + (0.0 - g0) * (x1 - x0) / (g1 - g0)   # log10|H| = 0
            return float(10.0 ** np.clip(x, min(x0, x1), max(x0, x1)))
    return float("nan")


def _phase_deg_unwrapped(freq, resp):
    freq = np.asarray(freq, float)
    order = np.argsort(freq)
    ph = np.unwrap(np.angle(np.asarray(resp, complex)[order]))
    return freq[order], np.degrees(ph)


def phase_margin(freq, resp):
    """Phase margin [deg] of a forward or loop response *resp* under unity feedback.

    ``PM = 180 + (phase(resp) at UGBW - phase(resp) at the peak-gain frequency)``. The
    passband/peak phase is the reference so an inverting (passband phase ≈ ±180°) or
    AC-coupled band-pass amplifier reports the physical margin — e.g. the FD-OTA
    passband phase is -179° and PM comes out ~84°, not ~-95°. For a plain low-pass
    loop gain the peak sits at DC with phase 0, so this reduces to ``180 + phase(UGBW)``.
    ``nan`` if there is no unity crossing."""
    fu = unity_gain_freq(freq, resp)
    if not np.isfinite(fu):
        return float("nan")
    f, ph = _phase_deg_unwrapped(freq, resp)
    mag = np.abs(np.asarray(resp, complex))[np.argsort(np.asarray(freq, float))]
    ph_ref = ph[int(np.argmax(mag))]
    ph_u = float(np.interp(np.log10(fu), np.log10(f), ph))
    return 180.0 + (ph_u - ph_ref)


def gain_margin_db(freq, resp):
    """Gain margin [dB] = -20log10|resp| at the -180° phase crossing. ``nan`` if the
    phase never reaches -180° within the sweep."""
    freq = np.asarray(freq, float)
    resp = np.asarray(resp, complex)
    order = np.argsort(freq)
    ph = np.degrees(np.unwrap(np.angle(resp[order])))
    mag_db = 20.0 * np.log10(np.maximum(np.abs(resp[order]), 1e-300))
    for i in range(1, len(ph)):
        if (ph[i - 1] + 180.0) * (ph[i] + 180.0) <= 0.0 and ph[i] != ph[i - 1]:
            w = (-180.0 - ph[i - 1]) / (ph[i] - ph[i - 1])
            return float(-(mag_db[i - 1] + w * (mag_db[i] - mag_db[i - 1])))
    return float("nan")


# ── .noise ──────────────────────────────────────────────────────────────────────
def noise_ngspice(sizes, bias, *, topo, out, src, fstart, fstop, points=20,
                  ref=None, band=None, nf=None, model_types=None, device_kwargs=None,
                  corner=None, temperature=None, x0_guess=None, timeout=300.0,
                  noiseless_resistors=None):
    """Full-circuit ngspice ``.noise`` — output & input-referred PSD + band rms.

    ``out``/``ref`` are solved-node names — ``v(out)`` single-ended or ``v(out,ref)``
    differential. ``src`` names the input source (a rail or an ideal vsource); it is
    driven ``ac 1`` so ngspice's ``inoise`` is meaningful. ``band=(f_lo, f_hi)`` (Hz)
    sets the integration band for the reported rms (defaults to the full sweep).

    Returns ``{"freq", "onoise_psd" [V^2/Hz], "inoise_psd", "onoise_rms", "inoise_rms",
    "band"}``. PSD = ngspice spectrum² (its ``*noise_spectrum`` vectors are amplitude
    densities, V/√Hz or A/√Hz — squared here to a power density). rms is
    ``sqrt(∫ PSD df)`` over ``band`` (trapezoid on the swept grid).

    ``noiseless_resistors`` names testbench-only DC helper resistors that retain
    their operating-point conductance but render with ngspice ``noisy=0``.  It
    must not be used to suppress physical DUT resistor noise.
    """
    ac = {_resolve_source_name(topo, src): (1.0, 0.0)}
    for name in (out, ref):
        if name is not None and name not in topo.idx:
            raise ValueError(f"noise_ngspice node {name!r} is not a solved node")

    lines, node_map, _node, adapter = _network_deck(
        topo, sizes, bias, header="* circuitopt FreePDK45 full-circuit .noise",
        nf=nf, model_types=model_types, device_kwargs=device_kwargs, corner=corner,
        temperature=temperature, x0_guess=x0_guess, ac=ac)

    noiseless = {str(name) for name in (noiseless_resistors or ())}
    known_resistors = {name for name, *_ in topo.resistors}
    unknown = noiseless - known_resistors
    if unknown:
        raise ValueError(f"unknown noiseless resistor(s): {sorted(unknown)}")
    noiseless_elements = {_element("R", name).lower() for name in noiseless}
    for pos, line in enumerate(lines):
        fields = line.split(None, 1)
        if fields and fields[0].lower() in noiseless_elements:
            lines[pos] = line + " noisy=0"

    out_expr = f"v({node_map[out]}" + (f",{node_map[ref]})" if ref is not None else ")")
    src_elem = _stimulus_deck_element(topo, src)
    with tempfile.TemporaryDirectory(prefix="circuitopt-fp45-noise-") as td:
        out_path = os.path.join(td, "noise.dat")
        deck_path = os.path.join(td, "deck.cir")
        lines.extend([
            ".control", "set filetype=ascii", "set wr_singlescale", "set wr_vecnames",
            f"noise {out_expr} {src_elem} dec {int(points):d} "
            f"{float(fstart):.17g} {float(fstop):.17g}",
            "setplot noise1",
            f"wrdata {out_path} onoise_spectrum inoise_spectrum",
            ".endc", ".end",
        ])
        with open(deck_path, "w", encoding="ascii") as fh:
            fh.write("\n".join(lines) + "\n")
        _run_ngspice(deck_path, out_path, timeout=timeout,
                     what="full-circuit .noise",
                     extra_args=adapter.command_args if adapter is not None else ())
        raw = np.loadtxt(out_path, skiprows=1, ndmin=2)

    freq = raw[:, 0]
    onoise_psd = raw[:, 1] ** 2       # spectrum is amplitude density → square to PSD
    inoise_psd = raw[:, 2] ** 2
    lo, hi = (float(band[0]), float(band[1])) if band else (float(freq[0]), float(freq[-1]))
    mask = (freq >= lo) & (freq <= hi)
    onoise_rms = float(np.sqrt(np.trapezoid(onoise_psd[mask], freq[mask]))) if mask.sum() > 1 else 0.0
    inoise_rms = float(np.sqrt(np.trapezoid(inoise_psd[mask], freq[mask]))) if mask.sum() > 1 else 0.0
    return {"freq": freq, "onoise_psd": onoise_psd, "inoise_psd": inoise_psd,
            "onoise_rms": onoise_rms, "inoise_rms": inoise_rms, "band": (lo, hi)}


# ── .op ──────────────────────────────────────────────────────────────────────────
_OP_PARAMS = ("vds", "vgs", "vdsat", "id", "gm", "gds")


def _run_ngspice_capture(deck_path: str, timeout: float, what: str,
                         extra_args=()) -> str:
    proc = subprocess.run([_ngspice(), *extra_args, "-b", deck_path], capture_output=True,
                          text=True, timeout=timeout)
    if proc.returncode != 0:
        tail = ((proc.stderr or "") + (proc.stdout or "")).strip()[-800:]
        raise RuntimeError(f"ngspice {what} failed (rc {proc.returncode})\n{tail}")
    return (proc.stdout or "") + "\n" + (proc.stderr or "")


def op_ngspice(sizes, bias, *, topo, margin=0.0, nf=None, model_types=None,
               device_kwargs=None, corner=None, temperature=None, x0_guess=None,
               timeout=120.0):
    """Full-circuit ngspice ``.op`` — per-device operating point + saturation check.

    Runs ``.op`` (seed the DC with ``x0_guess`` → ``.nodeset`` for a multistable OTA)
    and harvests ngspice op-vars per transistor. Returns
    ``{device: {"vds", "vgs", "vdsat", "id", "gm", "gds", "region_ok"}}`` where
    ``region_ok = |vds| >= |vdsat| + margin`` — the saturation-region test, with the
    absolute values handling NMOS/PMOS polarity uniformly (``margin`` in volts,
    default 0). Devices whose op-vars ngspice does not report are omitted."""
    lines, _node_map, _node, adapter = _network_deck(
        topo, sizes, bias, header="* circuitopt FreePDK45 full-circuit .op",
        nf=nf, model_types=model_types, device_kwargs=device_kwargs, corner=corner,
        temperature=temperature, x0_guess=x0_guess)

    dev_names = [name for name, *_ in topo.devices]
    prints = []
    vectors = {}
    for name in dev_names:
        elem = _element("X" if adapter is not None else "M", name).lower()
        for p in _OP_PARAMS:
            vector = (adapter.op_vector(elem, p) if adapter is not None
                      else f"@{elem}[{p}]")
            vectors.setdefault(name, {})[p] = vector[1:].lower()
            prints.append(f"print {vector}")
    lines.extend([".control", "op", *prints, ".endc", ".end"])

    import re as _re
    with tempfile.TemporaryDirectory(prefix="circuitopt-fp45-op-") as td:
        deck_path = os.path.join(td, "deck.cir")
        with open(deck_path, "w", encoding="ascii") as fh:
            fh.write("\n".join(lines) + "\n")
        text = _run_ngspice_capture(
            deck_path, timeout=timeout, what="full-circuit .op",
            extra_args=adapter.command_args if adapter is not None else ())

    # ngspice prints a scalar as "@m1[vds] = 3.5e-01"
    pat = _re.compile(r"@([^\s=]+)\s*=\s*([-+0-9.eEnaN]+)")
    raw = {}
    for m in pat.finditer(text):
        vector, val = m.group(1).lower(), m.group(2)
        try:
            raw[vector] = float(val)
        except ValueError:
            continue
    out = {}
    for name in dev_names:
        vals = {p: raw.get(vector) for p, vector in vectors[name].items()}
        if vals.get("vds") is None or vals.get("vdsat") is None:
            continue
        rec = {p: vals[p] if vals[p] is not None else float("nan") for p in _OP_PARAMS}
        rec["region_ok"] = bool(abs(rec["vds"]) >= abs(rec["vdsat"]) + float(margin))
        out[name] = rec
    return out


# ── loop gain (Middlebrook single voltage injection) ────────────────────────────
def loop_gain_ngspice(sizes, bias, *, topo, inject, fstart, fstop, points=20,
                      nf=None, model_types=None, device_kwargs=None, corner=None,
                      temperature=None, x0_guess=None, timeout=300.0):
    """Loop gain T(f) by Middlebrook single voltage injection through ``inject``.

    ``inject`` names an ideal vsource placed IN SERIES in the loop, between the
    driver-side node (its ``q`` terminal — low impedance, e.g. a stage output) and the
    DUT-input node (its ``p`` terminal — high impedance, e.g. a transistor gate); its
    DC value is 0 so the loop is closed for biasing. This one testbench-side ideal
    source is all that is needed — no loop-breaking inductor. The source is driven
    ``ac 1`` and the loop gain is read as ``T = -V(p_side)/V(q_side)`` at the break
    (exact when the injection sits at a high-Z/low-Z boundary).

    Use it for a feedback amplifier's differential loop (break at an input-pair gate)
    and for a CMFB loop (break at the common-mode control gate) — see
    ``docs/ngspice_oracles.md``. Returns ``{"freq", "loop_gain" (complex T),
    "gain_db", "phase_deg", "ugf", "pm", "gm_db"}``."""
    vs = next((v for v in topo.vsources if v[0] == inject), None)
    if vs is None:
        raise ValueError(f"loop_gain_ngspice: {inject!r} is not an ideal vsource in the topology")
    _name, p, q, _val = vs
    if p not in topo.idx or q not in topo.idx:
        raise ValueError("loop injection source must break between two solved nodes")

    result = ac_ngspice(
        sizes, bias, topo=topo, acmag={inject: (1.0, 0.0)}, fstart=fstart, fstop=fstop,
        points=points, out_nodes=[p, q], nf=nf, model_types=model_types,
        device_kwargs=device_kwargs, corner=corner, temperature=temperature,
        x0_guess=x0_guess, timeout=timeout)
    freq = result["freq"]
    vp = result["nodes"][p]
    vq = result["nodes"][q]
    T = -vq / vp
    return {
        "freq": freq, "loop_gain": T,
        "gain_db": 20.0 * np.log10(np.maximum(np.abs(T), 1e-300)),
        "phase_deg": _phase_deg_unwrapped(freq, T)[1],
        "ugf": unity_gain_freq(freq, T),
        "pm": phase_margin(freq, T),
        "gm_db": gain_margin_db(freq, T),
    }


def loop_gain_tian_ngspice(sizes, bias, *, topo, inject, fstart, fstop, points=20,
                           nf=None, model_types=None, device_kwargs=None, corner=None,
                           temperature=None, x0_guess=None, timeout=300.0, chain=None):
    """Loop gain T(f) by Tian/Middlebrook DOUBLE injection at the vsource ``inject``.

    :func:`loop_gain_ngspice`'s single voltage injection is exact only while the
    ``p`` side of the break stays high-impedance relative to the ``q`` side. At a
    MOS gate that premise dies at RF (Cgg of a large input pair reaches the kOhm →
    hundreds-of-ohm range right around loop crossover) and the reported PM becomes
    an artifact of the probe, not a property of the loop. This oracle removes the
    impedance condition with a second, current-injection run.

    Conventions (``inject`` = ideal 0 V vsource between ``p`` = forward-path input
    node, ``q`` = feedback-network drive node, rendered ``v<inj> n_p n_q 0`` so
    ``i(v<inj>)`` is the branch current flowing p → q). Both runs record the SAME
    two observables — ``v = v(p)`` (the injection-side terminal) and
    ``i = i(v<inj>)``:

    * run 1 — voltage injection: ``inject`` driven ``ac 1``; measure ``v1, i1``.
    * run 2 — current injection: ``inject`` at 0 V; a testbench current source
      drives ``ac 1`` from ground INTO node ``p``; measure ``v2, i2``.
    * combination (Tian et al., "Striving for small-signal stability", IEEE
      Circuits & Devices 17(1), 2001; rational form per F. Wiedmann's reference
      implementation)::

          T = -1 / (1 - 1/(2*(i1*v2 - v1*i2) + v1 + i2))

      The result is the return ratio for arbitrary impedances on both sides of
      the break and is orientation-symmetric per the paper. This exact wiring +
      sign convention (v and Iinj on the vsource's N+ terminal, ngspice branch
      current N+ → N-) reproduces a two-pole analytic reference loop to machine
      precision across five frequency decades (see tests).

    Differential double-break probes (e.g. the MDAC dm-loop testbench: ``Vinj``
    on one input gate plus a unity VCVS mirroring the break voltage onto the other
    gate) are detected automatically: a VCVS whose control pair equals the break
    pair gets an anti-phase (or in-phase, matching the mirror polarity) current
    injected into its output-side node during run 2, keeping the excitation inside
    the mirrored signal subspace so the ``p``-side measurements are exactly the
    half-circuit's Tian quantities. Without this, a single-ended current injection
    excites the orthogonal mode and the extracted T is meaningless.

    Same signature and return shape as :func:`loop_gain_ngspice`, plus ``chain``:
    ``None`` (default) resolves :func:`~circuitopt.ngspice_char.ngspice_chain_enabled`
    from ``CIRCUITOPT_NGSPICE_CHAIN`` at call time; ``True``/``False`` overrides it.
    Chained, ONE ngspice process renders both source sets — ``inject`` driven
    ``ac 1`` while the Tian current sources sit in the deck at ``ac 0`` (a 0-amp
    AC current source is electrically inert; the mirror's phase is pre-set on its
    element line) — runs the v-injection sweep, flips the magnitudes with
    ``alter @src[acmag]`` (no re-parse, so the deck's fixed model-expansion cost
    is paid once instead of twice), and runs the i-injection sweep. Unchained,
    the historical two-process path runs verbatim and each AC run pays the
    model-expansion cost (2x a single-injection run). Both paths parse the same
    per-run vectors and feed the identical Tian combination."""
    vs = next((v for v in topo.vsources if v[0] == inject), None)
    if vs is None:
        raise ValueError(
            f"loop_gain_tian_ngspice: {inject!r} is not an ideal vsource in the topology")
    _name, p, q, value = vs
    if isinstance(value, str) or float(value) != 0.0:
        raise ValueError("loop injection source must be a constant 0 V break element")
    if p not in topo.idx or q not in topo.idx:
        raise ValueError("loop injection source must break between two solved nodes")

    # Differential probe: a unity-gain VCVS slaved to the break voltage mirrors the
    # injection onto a second break. (cp, cn) == (q, p) copies -e (anti-symmetric
    # differential probe -> counter current ac 1 <180); (p, q) copies +e -> <0.
    counter = None
    for _vname, p2, _q2, cp, cn, mu in topo.vcvs:
        if float(mu) == 1.0 and {cp, cn} == {p, q} and p2 in topo.idx:
            counter = (p2, 180.0 if cp == q else 0.0)
            break

    velem = _element("V", inject)
    sweeps = {}
    if ngspice_chain_enabled(chain):
        # ONE process, both source sets from the start: Vinj drives ac 1 while the
        # Tian current sources are rendered at ac 0 (electrically inert; the
        # mirror's anti-/in-phase is pre-set so only MAGNITUDES are altered).
        # After the v-injection sweep, `alter @src[acmag]` swaps the drive without
        # re-parsing the deck — the fixed model-expansion cost is paid once.
        lines, node_map, _node, adapter = _network_deck(
            topo, sizes, bias,
            header="* circuitopt Tian loop gain (chained v+i injection)",
            nf=nf, model_types=model_types, device_kwargs=device_kwargs, corner=corner,
            temperature=temperature, x0_guess=x0_guess,
            ac={_resolve_source_name(topo, inject): (1.0, 0.0)})
        lines.append(f"itianprobe 0 {node_map[p]} dc 0 ac 0 0")
        if counter is not None:
            lines.append(
                f"itianmirror 0 {node_map[counter[0]]} dc 0 ac 0 {counter[1]:g}")
        with tempfile.TemporaryDirectory(prefix="circuitopt-tian-") as td:
            out_paths = {mode: os.path.join(td, f"ac_{mode}.dat") for mode in ("v", "i")}
            deck_path = os.path.join(td, "deck.cir")
            vecs = []
            for expr in (f"v({node_map[p]})", f"v({node_map[q]})", f"i({velem})"):
                vecs.extend([f"real({expr})", f"imag({expr})"])
            sweep = f"ac dec {int(points):d} {float(fstart):.17g} {float(fstop):.17g}"
            lines.extend([
                ".control", "set filetype=ascii", "set wr_singlescale", "set wr_vecnames",
                sweep,
                f"wrdata {out_paths['v']} " + " ".join(vecs),
                f"alter @{velem.lower()}[acmag]=0",
                "alter @itianprobe[acmag]=1",
            ])
            if counter is not None:
                lines.append("alter @itianmirror[acmag]=1")
            lines.extend([
                sweep,
                f"wrdata {out_paths['i']} " + " ".join(vecs),
                ".endc", ".end",
            ])
            with open(deck_path, "w", encoding="ascii") as fh:
                fh.write("\n".join(lines) + "\n")
            _run_ngspice(
                deck_path, out_paths["i"], timeout=timeout,
                what="Tian loop gain (chained v+i injection) .ac",
                extra_args=adapter.command_args if adapter is not None else ())
            for mode in ("v", "i"):
                if not os.path.exists(out_paths[mode]):
                    raise RuntimeError(
                        f"Tian loop gain chained run wrote no {mode}-injection sweep")
                raw = np.loadtxt(out_paths[mode], skiprows=1, ndmin=2)
                sweeps[mode] = {
                    "freq": raw[:, 0],
                    "vp": raw[:, 1] + 1j * raw[:, 2],
                    "vq": raw[:, 3] + 1j * raw[:, 4],
                    "ib": raw[:, 5] + 1j * raw[:, 6],
                }
    else:
        for mode in ("v", "i"):
            ac = {inject: (1.0, 0.0)} if mode == "v" else {}
            lines, node_map, _node, adapter = _network_deck(
                topo, sizes, bias, header=f"* circuitopt Tian loop gain ({mode}-injection)",
                nf=nf, model_types=model_types, device_kwargs=device_kwargs, corner=corner,
                temperature=temperature, x0_guess=x0_guess,
                ac={_resolve_source_name(topo, k): v for k, v in ac.items()})
            if mode == "i":
                lines.append(f"itianprobe 0 {node_map[p]} dc 0 ac 1 0")
                if counter is not None:
                    lines.append(
                        f"itianmirror 0 {node_map[counter[0]]} dc 0 ac 1 {counter[1]:g}")
            with tempfile.TemporaryDirectory(prefix="circuitopt-tian-") as td:
                out_path = os.path.join(td, "ac.dat")
                deck_path = os.path.join(td, "deck.cir")
                vecs = []
                for expr in (f"v({node_map[p]})", f"v({node_map[q]})", f"i({velem})"):
                    vecs.extend([f"real({expr})", f"imag({expr})"])
                lines.extend([
                    ".control", "set filetype=ascii", "set wr_singlescale", "set wr_vecnames",
                    f"ac dec {int(points):d} {float(fstart):.17g} {float(fstop):.17g}",
                    f"wrdata {out_path} " + " ".join(vecs),
                    ".endc", ".end",
                ])
                with open(deck_path, "w", encoding="ascii") as fh:
                    fh.write("\n".join(lines) + "\n")
                _run_ngspice(
                    deck_path, out_path, timeout=timeout,
                    what=f"Tian loop gain ({mode}-injection) .ac",
                    extra_args=adapter.command_args if adapter is not None else ())
                raw = np.loadtxt(out_path, skiprows=1, ndmin=2)
            sweeps[mode] = {
                "freq": raw[:, 0],
                "vp": raw[:, 1] + 1j * raw[:, 2],
                "vq": raw[:, 3] + 1j * raw[:, 4],
                "ib": raw[:, 5] + 1j * raw[:, 6],
            }

    freq = sweeps["v"]["freq"]
    if not np.allclose(freq, sweeps["i"]["freq"], rtol=1e-9, atol=0.0):
        raise RuntimeError("Tian loop gain: the two injection sweeps disagree on frequency")
    v1, i1 = sweeps["v"]["vp"], sweeps["v"]["ib"]
    v2, i2 = sweeps["i"]["vp"], sweeps["i"]["ib"]
    T = -1.0 / (1.0 - 1.0 / (2.0 * (i1 * v2 - v1 * i2) + v1 + i2))
    return {
        "freq": freq, "loop_gain": T,
        "gain_db": 20.0 * np.log10(np.maximum(np.abs(T), 1e-300)),
        "phase_deg": _phase_deg_unwrapped(freq, T)[1],
        "ugf": unity_gain_freq(freq, T),
        "pm": phase_margin(freq, T),
        "gm_db": gain_margin_db(freq, T),
    }
