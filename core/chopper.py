"""Differential chopper analyses.

Two complementary levels are supported:

1. Ideal LPTV analysis. The physical eight-switch implementation (two switches at
   each differential input and output port) is modeled as two synchronized ideal
   commutators:

    input differential signal  -> multiplied by m(t) = +/-1
    amplifier output           -> multiplied by the same m(t)

For a periodic square wave, this is a linear periodically time-varying system.
The baseband response is the harmonic transfer sum:

    H_chop(f) = sum_k |c_k|^2 H_amp(f + k*f_chop)

where c_k are the complex Fourier coefficients of m(t). Internal amplifier noise
is similarly folded by the output chopper:

    S_out,chop(f) = sum_k |c_k|^2 S_out,amp(|f + k*f_chop|)

This captures the core chopper effect: low-frequency amplifier noise is moved away
from baseband, while baseband signal/noise is recovered by synchronous demodulation.
Switch Ron/Roff, charge injection, clock feedthrough, finite edge time, and switch
thermal noise are not included in this ideal analysis.

2. PMOS switch topology. The default AFE can be wrapped in eight real PMOS_TFT
   pass devices driven by two complementary clock nodes. Static phase AC/noise
   analysis includes switch Ron, nonlinear capacitances, and switch thermal/flicker
   noise at each clock phase. It is not a full LPTV/PNoise solver, so clock-edge
   effects and sideband folding from time-varying switch operating points are still
   outside this function.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock

import numpy as np

from .ac_solver import bw_from_gain, ac_solve
from .dc_solver import is_afe_topology
from .device_factory import dev_corner, dev_nf
from .adaptive_config import resolve_adaptive_config
from .device_model import create_device, get_default_model_type
from .noise_solver import band_rms, noise_analysis
from .pac_solver import pac_solve
from .pnoise_solver import pnoise_solve
from .pss_solver import pss_solve
from .topology import AFE_TOPO, Topology
from .transient_solver import transient


# The quasi-static `pmos_chopper_lptv_analysis` below is a fast FIRST-ORDER
# sideband-sum estimate; it underestimates PAC gain by ~10% because it omits the
# higher-order LPTV conversion (and a small cyclostationary noise increment). For
# Cadence-grade accuracy use the first-principles harmonic-balance path
# (`pmos_chopper_pss` -> `pmos_chopper_pac`/`pmos_chopper_pnoise`), which carries
# NO empirical constants and is what `core/calibration.py` validates. The old
# Cadence-fit conversion-phase (24.93 deg) and noise-PSD-scale (1.0355) constants
# were retired 2026-06-22 — they only patched this first-order approximation.
_PMOS_CHOPPER_BARE_DC_SEED_CACHE_MAX = 64
_PMOS_CHOPPER_BARE_DC_SEED_CACHE = OrderedDict()
_PMOS_CHOPPER_BARE_DC_SEED_CACHE_LOCK = RLock()


@dataclass(frozen=True)
class PMOSChopperBuild:
    """Topology metadata for the eight-PMOS AFE chopper wrapper."""

    topology: Topology
    switch_sizes: dict
    switch_nf: dict
    switch_names: tuple
    input_nodes: tuple
    amp_input_nodes: tuple
    amp_output_nodes: tuple
    output_nodes: tuple
    sense_output_nodes: tuple
    clock_nodes: tuple
    split_input_pair: bool = False


def square_chopper_harmonics(max_harmonic=31):
    """Odd signed harmonics and |c_k|^2 weights for a +/-1, 50% duty square wave.

    `max_harmonic` is the largest odd harmonic index included. Larger values
    reduce the ideal square-wave truncation error at the cost of more sideband
    AC/noise points.
    """
    if max_harmonic < 1:
        raise ValueError("max_harmonic must be >= 1")
    max_harmonic = int(max_harmonic)
    odds = np.arange(1, max_harmonic + 1, 2, dtype=int)
    harmonics = np.concatenate((-odds[::-1], odds))
    weights = 4.0 / (np.pi ** 2 * harmonics.astype(float) ** 2)
    return harmonics, weights


def _interp_complex_response(query_signed, table_freqs, table_response):
    """Lookup response at signed frequencies using H(-f)=conj(H(f))."""
    q = np.asarray(query_signed, float)
    mag_freq = np.abs(q)
    real = np.interp(mag_freq, table_freqs, table_response.real)
    imag = np.interp(mag_freq, table_freqs, table_response.imag)
    out = real + 1j * imag
    out = np.where(q < 0.0, np.conjugate(out), out)
    return out


def _interp_psd(query_signed, table_freqs, table_psd):
    return np.interp(np.abs(query_signed), table_freqs, table_psd)


def _dedup(seq):
    out = []
    seen = set()
    for item in seq:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


def _smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _periodic_pulse_weight(phase, start, stop, edge_fraction):
    phase = np.asarray(phase, float) % 1.0
    start = float(start) % 1.0
    stop = float(stop) % 1.0
    duration = (stop - start) % 1.0
    if duration <= 0.0:
        return np.zeros_like(phase)
    x = (phase - start) % 1.0
    inside = x < duration
    w = np.zeros_like(phase)
    if edge_fraction <= 0.0:
        w[inside] = 1.0
        return w
    edge = min(float(edge_fraction), 0.5 * duration)
    xi = x[inside]
    wi = np.ones_like(xi)
    rising = xi < edge
    falling = xi > duration - edge
    wi[rising] = _smoothstep(xi[rising] / edge)
    wi[falling] = _smoothstep((duration - xi[falling]) / edge)
    w[inside] = wi
    return w


def finite_edge_clock_pair(tgrid, f_chop, *, v_low=0.0, v_high=40.0,
                           edge_time=0.0, dead_time=0.0, phase_offset=0.0):
    """Complementary PMOS gate clocks with finite edge and optional dead time.

    Returned `clk_a` is low/on in the first half-cycle, `clk_b` is low/on in the
    second half-cycle. The waveforms are break-before-make when `dead_time > 0`.
    """
    tgrid = np.asarray(tgrid, float)
    f_chop = float(f_chop)
    if f_chop <= 0.0:
        raise ValueError("f_chop must be positive")
    period = 1.0 / f_chop
    edge_fraction = max(0.0, float(edge_time) / period)
    dead_fraction = max(0.0, float(dead_time) / period)
    if dead_fraction >= 0.5:
        raise ValueError("dead_time must be less than half the chopper period")

    phase = (tgrid * f_chop + float(phase_offset)) % 1.0
    if dead_fraction <= 0.0 and edge_fraction > 0.0:
        edge = min(edge_fraction, 0.5)
        half_edge = 0.5 * edge
        a_on = np.where(phase < 0.5, 1.0, 0.0)
        rise = phase < half_edge
        rise_wrap = phase >= 1.0 - half_edge
        fall = (phase >= 0.5 - half_edge) & (phase < 0.5 + half_edge)
        a_on[rise] = _smoothstep((phase[rise] + half_edge) / edge)
        a_on[rise_wrap] = _smoothstep((phase[rise_wrap] - (1.0 - half_edge)) / edge)
        a_on[fall] = 1.0 - _smoothstep((phase[fall] - (0.5 - half_edge)) / edge)
        b_on = 1.0 - a_on
    else:
        a_on = _periodic_pulse_weight(phase, 0.5 * dead_fraction,
                                      0.5 - 0.5 * dead_fraction, edge_fraction)
        b_on = _periodic_pulse_weight(phase, 0.5 + 0.5 * dead_fraction,
                                      1.0 - 0.5 * dead_fraction, edge_fraction)
    vspan = float(v_high) - float(v_low)
    clk_a = float(v_high) - a_on * vspan
    clk_b = float(v_high) - b_on * vspan
    return clk_a, clk_b, a_on, b_on


def spectre_pulse_clock_pair(tgrid, f_chop, *, v_low=0.0, v_high=40.0,
                             edge_time=0.0):
    """Complementary PMOS gate clocks matching Spectre ``type=pulse`` timing.

    The Cadence verification netlist uses:

    ``delay=T/2 width=T/2 period=T rise=edge_time fall=edge_time``.

    Spectre applies the finite rise before starting the high-width interval, so
    the falling edge starts at ``T + edge_time`` for the first cycle.  This
    helper intentionally follows that source semantics rather than the ideal
    centered-edge periodic chopper waveform used by the frequency-domain helper.
    """
    tgrid = np.asarray(tgrid, float)
    f_chop = float(f_chop)
    if f_chop <= 0.0:
        raise ValueError("f_chop must be positive")
    period = 1.0 / f_chop
    half = 0.5 * period
    edge = max(0.0, float(edge_time))
    low = float(v_low)
    high = float(v_high)

    def pulse(val0, val1):
        out = np.full_like(tgrid, float(val0), dtype=float)
        active = tgrid >= half
        if not np.any(active):
            return out
        tau = np.mod(tgrid[active] - half, period)
        vals = np.full_like(tau, float(val0), dtype=float)
        if edge <= 0.0:
            vals[tau < half] = float(val1)
        else:
            rise = tau < edge
            high_region = (tau >= edge) & (tau < edge + half)
            fall = (tau >= edge + half) & (tau < 2.0 * edge + half)
            vals[rise] = val0 + (val1 - val0) * (tau[rise] / edge)
            vals[high_region] = val1
            vals[fall] = val1 + (val0 - val1) * ((tau[fall] - edge - half) / edge)
        out[active] = vals
        return out

    clk_a = pulse(low, high)
    clk_b = pulse(high, low)
    vspan = high - low
    if vspan == 0.0:
        a_on = np.zeros_like(clk_a)
        b_on = np.zeros_like(clk_b)
    else:
        a_on = np.clip((high - clk_a) / vspan, 0.0, 1.0)
        b_on = np.clip((high - clk_b) / vspan, 0.0, 1.0)
    return clk_a, clk_b, a_on, b_on


def finite_edge_chopper_harmonics(max_harmonic=31, *, edge_fraction=0.0,
                                  dead_fraction=0.0, samples=4096):
    """Fourier coefficients of a finite-edge differential chopper waveform.

    The modulation is `m(t)=a_on(t)-b_on(t)`: +1 for straight phase, -1 for
    crossed phase, and intermediate values during finite edges/dead time.
    """
    if max_harmonic < 1:
        raise ValueError("max_harmonic must be >= 1")
    samples = int(samples)
    if samples < 16:
        raise ValueError("samples must be >= 16")
    odds = np.arange(1, int(max_harmonic) + 1, 2, dtype=int)
    harmonics = np.concatenate((-odds[::-1], odds))
    phase = (np.arange(samples) + 0.5) / samples
    clk_a, clk_b, a_on, b_on = finite_edge_clock_pair(
        phase, 1.0, v_low=0.0, v_high=1.0, edge_time=edge_fraction,
        dead_time=dead_fraction)
    del clk_a, clk_b
    mod = a_on - b_on
    coeffs = np.array([
        np.mean(mod * np.exp(-2j * np.pi * k * phase))
        for k in harmonics
    ])
    return harmonics, coeffs, np.abs(coeffs) ** 2


def build_afe_pmos_chopper(*, switch_size=(20000.0, 80.0), switch_nf=1,
                           base_topo=AFE_TOPO, prefix="CH",
                           output_filter=None, split_input_pair=False):
    """Wrap the default AFE with eight PMOS_TFT pass switches.

    The wrapper creates external differential ports and internal amplifier ports:

    - `CH_VIP`, `CH_VIN`: external driven input rails.
    - `CH_INP`, `CH_INN`: internal amplifier input nodes.
    - `CH_AMP_OP`, `CH_AMP_ON`: internal amplifier output nodes.
    - `CH_VOP`, `CH_VON`: external sensed output nodes.
    - `CH_CLK_A`, `CH_CLK_B`: complementary PMOS gate clocks.

    Phase A turns on the straight paths, phase B turns on the crossed paths.
    PMOS is on when its gate is low, so use `pmos_chopper_phase_bias(..., "A")`
    or `"B"` to generate the matching clock bias.
    """
    if not is_afe_topology(base_topo):
        raise NotImplementedError("build_afe_pmos_chopper currently rewires the canonical AFE")

    vip = f"{prefix}_VIP"
    vin = f"{prefix}_VIN"
    inp = f"{prefix}_INP"
    inn = f"{prefix}_INN"
    amp_op = f"{prefix}_AMP_OP"
    amp_on = f"{prefix}_AMP_ON"
    vop = f"{prefix}_VOP"
    von = f"{prefix}_VON"
    vop_f = f"{prefix}_VOP_F"
    von_f = f"{prefix}_VON_F"
    clk_a = f"{prefix}_CLK_A"
    clk_b = f"{prefix}_CLK_B"
    filter_rc = None
    if output_filter is not None:
        if isinstance(output_filter, dict):
            filter_rc = (float(output_filter["R"]), float(output_filter["C"]))
        else:
            filter_rc = tuple(float(x) for x in output_filter)
        if len(filter_rc) != 2 or filter_rc[0] <= 0.0 or filter_rc[1] < 0.0:
            raise ValueError("output_filter must be (R_ohm, C_farad)")

    core_output_map = {"VOP": amp_op, "VON": amp_on}
    external_output_map = {"VOP": vop, "VON": von}

    def map_core_node(node):
        return core_output_map.get(node, node)

    def map_load_node(node):
        return external_output_map.get(node, node)

    split_input_pair = bool(split_input_pair)
    devices = []
    for name, drain, gate, source in base_topo.devices:
        drain = map_core_node(drain)
        source = map_core_node(source)
        if name == "M7":
            gate = inp
        elif name == "M8":
            gate = inn
        else:
            gate = map_core_node(gate)
        if split_input_pair and name == "M7":
            devices.append((name, drain, gate, source))
            devices.append(("M16", drain, gate, source))
        elif split_input_pair and name == "M8":
            devices.append((name, drain, gate, source))
            devices.append(("M17", drain, gate, source))
        else:
            devices.append((name, drain, gate, source))

    def switch(name, source, drain, gate):
        # PMOS_TFT tuples are (name, drain, gate, source).
        return (name, drain, gate, source)

    switches = [
        switch(f"{prefix}_SW_INP_A", vip, inp, clk_a),
        switch(f"{prefix}_SW_INN_A", vin, inn, clk_a),
        switch(f"{prefix}_SW_INP_B", vip, inn, clk_b),
        switch(f"{prefix}_SW_INN_B", vin, inp, clk_b),
        switch(f"{prefix}_SW_OUTP_A", amp_op, vop, clk_a),
        switch(f"{prefix}_SW_OUTN_A", amp_on, von, clk_a),
        switch(f"{prefix}_SW_OUTP_B", amp_op, von, clk_b),
        switch(f"{prefix}_SW_OUTN_B", amp_on, vop, clk_b),
    ]
    devices.extend(switches)
    switch_names = tuple(name for name, *_ in switches)

    filtered_nodes = [vop_f, von_f] if filter_rc is not None else []
    solved = _dedup([inp, inn] + [map_core_node(n) for n in base_topo.solved] +
                    [vop, von] + filtered_nodes)
    rails = dict(base_topo.rails)
    rails.update({vip: "VIP", vin: "VIN", clk_a: "CLK_A", clk_b: "CLK_B"})

    load_caps = [(map_load_node(a), map_load_node(b), cap)
                 for a, b, cap in base_topo.load_caps]
    resistors = [(name, map_core_node(a), map_core_node(b), R)
                 for name, a, b, R in base_topo.resistors]
    capacitors = [(name, map_load_node(a), map_load_node(b), C)
                  for name, a, b, C in base_topo.capacitors]
    sense_outputs = (vop, von)
    if filter_rc is not None:
        r_filter, c_filter = filter_rc
        resistors.extend((
            (f"{prefix}_RLP_P", vop, vop_f, r_filter),
            (f"{prefix}_RLP_N", von, von_f, r_filter),
        ))
        if c_filter:
            capacitors.extend((
                (f"{prefix}_CLP_P", vop_f, "GND", c_filter),
                (f"{prefix}_CLP_N", von_f, "GND", c_filter),
            ))
        sense_outputs = (vop_f, von_f)
    isources = [(name, map_core_node(p), map_core_node(q), I)
                for name, p, q, I in base_topo.isources]

    def guess(bias):
        vcm = bias["VCM"]
        return {
            inp: vcm,
            inn: vcm,
            amp_op: vcm - 4.0,
            amp_on: vcm - 4.0,
            vop: vcm - 4.0,
            von: vcm - 4.0,
            vop_f: vcm - 4.0,
            von_f: vcm - 4.0,
            "VFBP": vcm - 8.0,
            "VFBN": vcm - 8.0,
            "NET20": vcm + 15.0,
            "NET2": vcm + 7.0,
        }

    def switch_guess(bias):
        vcm = bias["VCM"]
        vdd = bias["VDD"]
        vout = vcm - 1.6
        vfb = max(vcm - 25.15, 0.5)
        return {
            inp: vcm,
            inn: vcm,
            amp_op: vout,
            amp_on: vout,
            vop: vout,
            von: vout,
            vop_f: vout,
            von_f: vout,
            "VFBP": vfb,
            "VFBN": vfb,
            "NET20": min(vcm + 7.45, vdd - 0.5),
            "NET2": min(vcm + 5.7, vdd - 0.5),
        }

    aliases = {key: map_core_node(node) for key, node in base_topo.aliases.items()}
    aliases.update({
        "VOP": sense_outputs[0],
        "VON": sense_outputs[1],
        "vop": sense_outputs[0],
        "von": sense_outputs[1],
        "vop_raw": vop,
        "von_raw": von,
        "vop_f": vop_f,
        "von_f": von_f,
        "vop_core": amp_op,
        "von_core": amp_on,
        "vinp_core": inp,
        "vinn_core": inn,
    })

    topo = Topology(
        solved=solved,
        devices=devices,
        rails=rails,
        outputs=sense_outputs,
        input_drives={},
        ac_drives={vip: +0.5, vin: -0.5},
        load_caps=load_caps,
        dc_guesses=(guess, switch_guess),
        aliases=aliases,
        transient_inputs={},
        resistors=resistors,
        capacitors=capacitors,
        isources=isources,
        dc_tol=1e-8,
        require_dc_in_box=True,
    )
    return PMOSChopperBuild(
        topology=topo,
        switch_sizes={name: tuple(switch_size) for name in switch_names},
        switch_nf={name: int(switch_nf) for name in switch_names},
        switch_names=switch_names,
        input_nodes=(vip, vin),
        amp_input_nodes=(inp, inn),
        amp_output_nodes=(amp_op, amp_on),
        output_nodes=(vop, von),
        sense_output_nodes=sense_outputs,
        clock_nodes=(clk_a, clk_b),
        split_input_pair=split_input_pair,
    )


def pmos_chopper_phase_bias(bias, phase="A", *, input_common_mode=None,
                            input_diff=0.0, clk_low=0.0, clk_high=None):
    """Bias dictionary for one static phase of the PMOS chopper wrapper.

    PMOS pass switches are on at `clk_low` and off at `clk_high`. The external
    input rails default to the AFE common-mode voltage with optional DC
    differential offset.
    """
    phase = str(phase).upper()
    if phase not in ("A", "B"):
        raise ValueError("phase must be 'A' or 'B'")
    out = dict(bias)
    vcm = bias["VCM"] if input_common_mode is None else float(input_common_mode)
    high = bias["VDD"] if clk_high is None else float(clk_high)
    low = float(clk_low)
    diff = float(input_diff)
    out["VIP"] = vcm + 0.5 * diff
    out["VIN"] = vcm - 0.5 * diff
    out["CLK_A"] = low if phase == "A" else high
    out["CLK_B"] = high if phase == "A" else low
    return out


def _with_switch_maps(sizes, nf, build):
    sizes_out = dict(sizes)
    if getattr(build, "split_input_pair", False):
        for parent, child in (("M7", "M16"), ("M8", "M17")):
            if parent in sizes_out:
                W, L = sizes_out[parent]
                half = (float(W) * 0.5, L)
                sizes_out[parent] = half
                sizes_out[child] = half
    sizes_out.update(build.switch_sizes)
    if nf is None:
        nf_out = dict(build.switch_nf)
    elif isinstance(nf, dict):
        nf_out = dict(nf)
        if getattr(build, "split_input_pair", False):
            for parent, child in (("M7", "M16"), ("M8", "M17")):
                parent_nf = int(nf_out.get(parent, 1))
                half_nf = max(1, int(round(parent_nf * 0.5)))
                nf_out[parent] = half_nf
                nf_out[child] = half_nf
        nf_out.update(build.switch_nf)
    else:
        nf_out = {name: int(nf) for name in sizes}
        if getattr(build, "split_input_pair", False):
            for parent, child in (("M7", "M16"), ("M8", "M17")):
                parent_nf = int(nf_out.get(parent, int(nf)))
                half_nf = max(1, int(round(parent_nf * 0.5)))
                nf_out[parent] = half_nf
                nf_out[child] = half_nf
        nf_out.update(build.switch_nf)
    return sizes_out, nf_out


def _dc_get(dc, *names, default=None):
    for name in names:
        if name in dc:
            return dc[name]
    return default


def _freeze_mapping(mapping):
    if mapping is None:
        return None
    if isinstance(mapping, dict):
        return tuple((str(k), repr(v)) for k, v in sorted(mapping.items()))
    return repr(mapping)


def _topology_seed_key(topo):
    return (
        tuple(topo.solved),
        tuple(topo.devices),
        tuple(sorted(topo.rails.items())),
        tuple(topo.resistors),
        tuple(topo.cap_list()),
        tuple(topo.isources),
        tuple(topo.outputs),
        tuple(sorted(topo.input_drives.items())),
        tuple(sorted(topo.ac_drives.items())),
    )


def _bare_dc_seed_cache_key(sizes, bias, nf, corner, base_topo):
    return (
        tuple((str(k), float(v[0]), float(v[1])) for k, v in sorted(sizes.items())),
        tuple((str(k), float(v)) for k, v in sorted(bias.items())),
        _freeze_mapping(nf),
        _freeze_mapping(corner),
        _topology_seed_key(base_topo),
    )


def _get_cached_bare_dc_seed(key):
    with _PMOS_CHOPPER_BARE_DC_SEED_CACHE_LOCK:
        if key not in _PMOS_CHOPPER_BARE_DC_SEED_CACHE:
            return None
        value = _PMOS_CHOPPER_BARE_DC_SEED_CACHE.pop(key)
        _PMOS_CHOPPER_BARE_DC_SEED_CACHE[key] = value
    return None if value is None else dict(value)


def _store_cached_bare_dc_seed(key, dc_op):
    if dc_op is None:
        return
    with _PMOS_CHOPPER_BARE_DC_SEED_CACHE_LOCK:
        _PMOS_CHOPPER_BARE_DC_SEED_CACHE[key] = dict(dc_op)
        _PMOS_CHOPPER_BARE_DC_SEED_CACHE.move_to_end(key)
        while len(_PMOS_CHOPPER_BARE_DC_SEED_CACHE) > _PMOS_CHOPPER_BARE_DC_SEED_CACHE_MAX:
            _PMOS_CHOPPER_BARE_DC_SEED_CACHE.popitem(last=False)


def _pmos_chopper_seed_from_core_dc(build, bias, phase, core_dc):
    """Map a bare-AFE DC operating point onto the PMOS chopper wrapper nodes."""
    if not isinstance(core_dc, dict):
        return core_dc

    topo = build.topology
    if all(node in core_dc for node in topo.solved):
        return core_dc

    vcm = bias.get("VCM", 0.5 * (bias.get("VIP", 0.0) + bias.get("VIN", 0.0)))
    vip = bias.get("VIP", vcm)
    vin = bias.get("VIN", vcm)
    vop_core = _dc_get(core_dc, "VOP", "vop", "vop_core")
    von_core = _dc_get(core_dc, "VON", "von", "von_core", default=vop_core)
    vfbp = _dc_get(core_dc, "VFBP", "vfbp", "vfb")
    vfbn = _dc_get(core_dc, "VFBN", "vfbn", "vfb", default=vfbp)
    net20 = _dc_get(core_dc, "NET20", "n20")
    net2 = _dc_get(core_dc, "NET2", "net2")
    if any(v is None for v in (vop_core, von_core, vfbp, vfbn, net20, net2)):
        return None

    phase = str(phase).upper()
    inp, inn = (vip, vin) if phase == "A" else (vin, vip)
    vop, von = (vop_core, von_core) if phase == "A" else (von_core, vop_core)
    seed = {
        build.amp_input_nodes[0]: inp,
        build.amp_input_nodes[1]: inn,
        build.amp_output_nodes[0]: vop_core,
        build.amp_output_nodes[1]: von_core,
        build.output_nodes[0]: vop,
        build.output_nodes[1]: von,
        "VFBP": vfbp,
        "VFBN": vfbn,
        "NET20": net20,
        "NET2": net2,
    }
    if getattr(build, "sense_output_nodes", build.output_nodes) != build.output_nodes:
        seed[build.sense_output_nodes[0]] = vop
        seed[build.sense_output_nodes[1]] = von
    return seed


def _pmos_chopper_auto_seed(sizes, bias, phase, build, *, nf=None, corner=None,
                            x0_guess=None, base_topo=AFE_TOPO):
    """Return a chopper-topology DC seed, deriving one from bare AFE DC if needed."""
    mapped = _pmos_chopper_seed_from_core_dc(build, bias, phase, x0_guess)
    if mapped is not None:
        return mapped
    cache_key = None
    if x0_guess is None:
        cache_key = _bare_dc_seed_cache_key(sizes, bias, nf, corner, base_topo)
        bare_dc = _get_cached_bare_dc_seed(cache_key)
        mapped = _pmos_chopper_seed_from_core_dc(build, bias, phase, bare_dc)
        if mapped is not None:
            return mapped
    bare = ac_solve(sizes, bias, np.array([1.0]), topo=base_topo,
                    nf=nf, corner=corner)
    if bare is None:
        return None
    if cache_key is not None:
        _store_cached_bare_dc_seed(cache_key, bare["dc_op"])
    return _pmos_chopper_seed_from_core_dc(build, bias, phase, bare["dc_op"])


def _pmos_chopper_one_phase(sizes, bias, freqs, phase, *, switch_size,
                            switch_nf, nf, corner, x0_guess, band, base_topo,
                            output_filter, split_input_pair=False):
    build = build_afe_pmos_chopper(switch_size=switch_size, switch_nf=switch_nf,
                                   base_topo=base_topo, output_filter=output_filter,
                                   split_input_pair=split_input_pair)
    all_sizes, all_nf = _with_switch_maps(sizes, nf, build)
    pbias = pmos_chopper_phase_bias(bias, phase)
    seed = _pmos_chopper_auto_seed(
        sizes, pbias, phase, build, nf=nf, corner=corner,
        x0_guess=x0_guess, base_topo=base_topo)
    noise = noise_analysis(all_sizes, pbias, freqs, corner=corner,
                           x0_guess=seed, topo=build.topology, nf=all_nf)
    if noise is None:
        return None, build

    response = noise.get("response")
    gains = np.asarray(noise["Hmag"], float) if response is None else np.abs(response)
    out_psd = noise["out_psd"]
    denom = np.maximum(gains ** 2, 1e-300)
    irn_psd = out_psd / denom
    return {
        "phase": phase,
        "bias": pbias,
        "freqs": freqs,
        "response": response,
        "gains": gains,
        "Av_dc_dB": 20 * np.log10(max(float(gains[0]), 1e-300)),
        "peak_dB": 20 * np.log10(max(float(np.max(gains)), 1e-300)),
        "bw_Hz": bw_from_gain(freqs, gains),
        "out_psd": out_psd,
        "irn_psd": irn_psd,
        "irn_uV_band": band_rms(freqs, irn_psd, band[0], band[1]) * 1e6,
        "dc": noise["dc"],
        "noise": noise,
    }, build


def pmos_chopper_analysis(sizes, bias, freqs, *, switch_size=(20000.0, 80.0),
                          switch_nf=1, nf=None, corner=None, x0_guess=None,
                          band=(0.05, 100.0), phases=("A", "B"),
                          base_topo=AFE_TOPO, output_filter=None,
                          split_input_pair=False):
    """Static-phase gain/BW/noise of the AFE wrapped with eight PMOS switches.

    This is the PMOS-device counterpart to `chopper_analysis`, but it is an LTI
    phase analysis rather than a full LPTV/PNoise computation. It is useful for
    sizing the switch devices because it includes Ron loading, device
    capacitances, and switch noise. For chopped flicker-noise folding, use the
    ideal `chopper_analysis` until a full periodic-noise solver is added.
    """
    freqs = np.asarray(freqs, float)
    if np.any(freqs <= 0.0):
        raise ValueError("freqs must be positive for PMOS switch noise analysis")
    phase_list = tuple(str(p).upper() for p in phases)
    if not phase_list:
        raise ValueError("at least one phase is required")

    phase_results = {}
    build = None
    for phase in phase_list:
        result, build = _pmos_chopper_one_phase(
            sizes, bias, freqs, phase, switch_size=switch_size,
            switch_nf=switch_nf, nf=nf, corner=corner, x0_guess=x0_guess,
            band=band, base_topo=base_topo, output_filter=output_filter,
            split_input_pair=split_input_pair)
        if result is None:
            return None
        phase_results[phase] = result

    responses = [r["response"] for r in phase_results.values()]
    if all(r is not None for r in responses):
        response = np.mean(np.vstack(responses), axis=0)
        gains = np.abs(response)
    else:
        response = None
        gains = np.mean(np.vstack([r["gains"] for r in phase_results.values()]), axis=0)
    out_psd = np.mean(np.vstack([r["out_psd"] for r in phase_results.values()]), axis=0)
    irn_psd = out_psd / np.maximum(gains ** 2, 1e-300)
    peak = float(np.max(gains))

    return {
        "freqs": freqs,
        "topology": build.topology,
        "switch_sizes": build.switch_sizes,
        "switch_nf": build.switch_nf,
        "switch_names": build.switch_names,
        "split_input_pair": bool(build.split_input_pair),
        "phases": phase_results,
        "response": response,
        "gains": gains,
        "Av_dc_dB": 20 * np.log10(max(float(gains[0]), 1e-300)),
        "peak_dB": 20 * np.log10(max(peak, 1e-300)),
        "bw_Hz": bw_from_gain(freqs, gains),
        "out_psd": out_psd,
        "irn_psd": irn_psd,
        "irn_uV_band": band_rms(freqs, irn_psd, band[0], band[1]) * 1e6,
        "analysis_note": (
            "Static phase average of the PMOS-switch topology; includes switch "
            "Ron/cap/noise but not full LPTV sideband folding or clock-edge effects."
        ),
    }


def _node_bias_value(topo, bias, dc, node):
    if node in dc:
        return dc[node]
    ref = topo.rails[node]
    return bias[ref] if isinstance(ref, str) else float(ref)


def _clock_positive_slew_current_shape(tgrid, clock):
    tgrid = np.asarray(tgrid, float)
    clock = np.asarray(clock, float)
    if len(tgrid) < 2:
        return np.zeros_like(clock)
    dclk = np.gradient(clock, tgrid)
    return np.maximum(dclk, 0.0)


def _edge_phase_intervals(edge_fraction, dead_fraction):
    a0 = 0.5 * dead_fraction
    a1 = 0.5 - 0.5 * dead_fraction
    b0 = 0.5 + 0.5 * dead_fraction
    b1 = 1.0 - 0.5 * dead_fraction
    edge = max(0.0, float(edge_fraction))
    return (
        (a0, a0 + edge),
        (a1 - edge, a1),
        (b0, b0 + edge),
        (b1 - edge, b1),
    )


def refine_chopper_tgrid(tgrid, f_chop, *, edge_time=0.0, dead_time=0.0,
                         phase_offset=0.0, edge_points=17,
                         hard_edge_window=None):
    """Return a time grid with extra points around chopper clock edges."""
    tgrid = np.asarray(tgrid, float)
    if len(tgrid) < 2:
        return tgrid.copy()
    f_chop = float(f_chop)
    if f_chop <= 0.0:
        raise ValueError("f_chop must be positive")
    period = 1.0 / f_chop
    edge_time = max(0.0, float(edge_time))
    dead_time = max(0.0, float(dead_time))
    dead_fraction = dead_time / period
    if dead_fraction >= 0.5:
        raise ValueError("dead_time must be less than half the chopper period")

    if edge_time > 0.0:
        edge_window = edge_time
    elif hard_edge_window is not None:
        edge_window = float(hard_edge_window)
    else:
        edge_window = 0.0
    if edge_window <= 0.0:
        return np.unique(tgrid)

    edge_fraction = min(edge_window / period, 0.5 * (0.5 - dead_fraction))
    points = [tgrid]
    t0 = float(tgrid[0])
    t1 = float(tgrid[-1])
    edge_points = max(3, int(edge_points))
    phase_offset = float(phase_offset)
    for p0, p1 in _edge_phase_intervals(edge_fraction, dead_fraction):
        m_min = int(np.floor(t0 * f_chop + phase_offset - p1)) - 1
        m_max = int(np.ceil(t1 * f_chop + phase_offset - p0)) + 1
        for m in range(m_min, m_max + 1):
            te0 = (p0 - phase_offset + m) / f_chop
            te1 = (p1 - phase_offset + m) / f_chop
            lo = max(t0, te0)
            hi = min(t1, te1)
            if hi >= lo:
                points.append(np.linspace(lo, hi, edge_points))
    refined = np.unique(np.concatenate(points))
    return refined[(refined >= t0) & (refined <= t1)]


def refine_pulse_clock_tgrid(tgrid, f_chop, *, edge_time=0.0,
                             edge_points=17, hard_edge_window=None,
                             time_shift=0.0):
    """Refine around Spectre pulse-source clock edges."""
    tgrid = np.asarray(tgrid, float)
    if len(tgrid) < 2:
        return tgrid.copy()
    f_chop = float(f_chop)
    if f_chop <= 0.0:
        raise ValueError("f_chop must be positive")
    period = 1.0 / f_chop
    half = 0.5 * period
    edge_time = max(0.0, float(edge_time))
    if edge_time > 0.0:
        edge_window = edge_time
    elif hard_edge_window is not None:
        edge_window = float(hard_edge_window)
    else:
        edge_window = 0.0
    if edge_window <= 0.0:
        return np.unique(tgrid)

    t0 = float(tgrid[0])
    t1 = float(tgrid[-1])
    edge_points = max(3, int(edge_points))
    time_shift = float(time_shift)
    starts = []
    k_min = int(np.floor((t0 - half - 2.0 * edge_window) / period)) - 1
    k_max = int(np.ceil((t1 - half + 2.0 * edge_window) / period)) + 1
    for k in range(k_min, k_max + 1):
        starts.append(half + k * period - time_shift)
        starts.append(half + edge_time + half + k * period - time_shift)

    points = [tgrid]
    for start in starts:
        if edge_time > 0.0:
            lo = start
            hi = start + edge_time
        else:
            lo = start - 0.5 * edge_window
            hi = start + 0.5 * edge_window
        lo = max(t0, lo)
        hi = min(t1, hi)
        if hi >= lo:
            points.append(np.linspace(lo, hi, edge_points))
    refined = np.unique(np.concatenate(points))
    return refined[(refined >= t0) & (refined <= t1)]


def _charge_injection_sources(build, all_sizes, all_nf, bias, tgrid, clk_a, clk_b,
                              *, charge_scale=1.0, source_split=0.5,
                              corner=None):
    """Build charge-injection current-source waveforms for PMOS turn-off edges."""
    if charge_scale == 0.0:
        return {}, []
    source_split = float(source_split)
    if not 0.0 <= source_split <= 1.0:
        raise ValueError("source_split must be in [0, 1]")

    topo = build.topology
    phase_dc = {}
    for phase in ("A", "B"):
        pbias = pmos_chopper_phase_bias(bias, phase)
        seed = _pmos_chopper_auto_seed(
            all_sizes, pbias, phase, build, nf=all_nf, corner=corner,
            base_topo=AFE_TOPO)
        ac = ac_solve(all_sizes, pbias, np.array([1.0]), topo=topo,
                      nf=all_nf, corner=corner, x0_guess=seed)
        if ac is None:
            phase_dc[phase] = None
        else:
            phase_dc[phase] = ac["dc_op"]

    clock_by_node = {build.clock_nodes[0]: clk_a, build.clock_nodes[1]: clk_b}
    phase_by_clock = {build.clock_nodes[0]: "A", build.clock_nodes[1]: "B"}
    devices = {name: (drain, gate, source) for name, drain, gate, source in topo.devices}

    waveforms = {}
    current_inputs = []
    for name in build.switch_names:
        drain, gate, source = devices[name]
        phase = phase_by_clock[gate]
        dc = phase_dc[phase]
        if dc is None:
            continue
        pbias = pmos_chopper_phase_bias(bias, phase)
        Vs = _node_bias_value(topo, pbias, dc, source)
        Vd = _node_bias_value(topo, pbias, dc, drain)
        Vg_on = pbias["CLK_A"] if phase == "A" else pbias["CLK_B"]
        dev = create_device(get_default_model_type(),
            W=all_sizes[name][0], L=all_sizes[name][1],
            NF=dev_nf(all_nf, name), **dev_corner(corner, name))
        qch = float(charge_scale) * dev.estimate_channel_charge(Vs, Vd, Vg_on)
        if qch <= 0.0:
            continue
        slew = _clock_positive_slew_current_shape(tgrid, clock_by_node[gate])
        vspan = max(float(np.max(clock_by_node[gate]) - np.min(clock_by_node[gate])), 1e-30)
        total_current = qch * slew / vspan
        if not np.any(total_current):
            continue

        key_s = f"qinj_{name}_s"
        key_d = f"qinj_{name}_d"
        waveforms[key_s] = source_split * total_current
        waveforms[key_d] = (1.0 - source_split) * total_current
        current_inputs.append({"p": gate, "q": source, "input": key_s})
        current_inputs.append({"p": gate, "q": drain, "input": key_d})
    return waveforms, current_inputs


def pmos_chopper_transient(sizes, bias, tgrid, f_chop, *, input_diff=0.0,
                           input_common_mode=None, vip=None, vin=None,
                           edge_time=0.0, dead_time=0.0,
                           clock_style="pulse",
                           clock_phase_offset=0.25,
                           switch_size=(20000.0, 80.0), switch_nf=1, nf=None,
                           corner=None,
                           V0=None, charge_injection=True,
                           charge_scale=1.0, charge_source_split=0.5,
                           refine_edges=True, edge_points=9,
                           signed_switches=True,
                           transient_max_step=None, max_retry_subdivisions=1,
                           transient_flat_max_step=None,
                           newton_maxit=60, newton_step_limit=2.0,
                           newton_vtol=1e-8,
                           fallback_full_jacobian=False,
                           fallback_least_squares=True, fallback_tol=1e-10,
                           base_topo=AFE_TOPO, output_filter=None,
                           rail_margin=2.0, profile=False,
                           split_input_pair=False, integration_method="be"):
    """Transient of the eight-PMOS chopper with finite-edge clocks.

    Clock feedthrough is produced by the PMOS model's own Cgss/Cgdd displacement
    currents. Optional charge injection adds turn-off current pulses derived from
    the PDK capacitance equations and the switch on-state operating point.
    """
    requested_tgrid = np.asarray(tgrid, float)
    if len(requested_tgrid) < 2:
        raise ValueError("tgrid must contain at least two points")
    tgrid = requested_tgrid
    f_chop = float(f_chop)
    period = 1.0 / f_chop
    clock_style = str(clock_style).lower()
    pulse_clock = clock_style in {"pulse", "spectre", "spectre_pulse"}
    if clock_style not in {"phase", "finite_edge", "legacy", "pulse", "spectre", "spectre_pulse"}:
        raise ValueError("clock_style must be 'pulse' or 'phase'")
    if pulse_clock and dead_time:
        raise ValueError("dead_time is only supported with clock_style='phase'")
    build = build_afe_pmos_chopper(switch_size=switch_size, switch_nf=switch_nf,
                                   base_topo=base_topo, output_filter=output_filter,
                                   split_input_pair=split_input_pair)
    all_sizes, all_nf = _with_switch_maps(sizes, nf, build)
    vcm = bias["VCM"] if input_common_mode is None else float(input_common_mode)
    if vip is None or vin is None:
        diff = np.asarray(input_diff, float)
        if diff.ndim == 0:
            diff = np.full_like(requested_tgrid, float(diff))
        if len(diff) != len(requested_tgrid):
            raise ValueError("input_diff waveform length must match tgrid")
        vip = vcm + 0.5 * diff
        vin = vcm - 0.5 * diff
    else:
        vip = np.asarray(vip, float)
        vin = np.asarray(vin, float)
    if len(vip) != len(requested_tgrid) or len(vin) != len(requested_tgrid):
        raise ValueError("vip/vin waveform length must match tgrid")

    if refine_edges:
        if pulse_clock:
            tgrid = refine_pulse_clock_tgrid(
                requested_tgrid, f_chop, edge_time=edge_time,
                edge_points=edge_points, hard_edge_window=period / 200.0)
        else:
            tgrid = refine_chopper_tgrid(
                requested_tgrid, f_chop, edge_time=edge_time, dead_time=dead_time,
                phase_offset=clock_phase_offset, edge_points=edge_points,
                hard_edge_window=period / 200.0)
        vip = np.interp(tgrid, requested_tgrid, vip)
        vin = np.interp(tgrid, requested_tgrid, vin)

    if pulse_clock:
        clk_a, clk_b, a_on, b_on = spectre_pulse_clock_pair(
            tgrid, f_chop, v_low=0.0, v_high=bias["VDD"],
            edge_time=edge_time)
    else:
        clk_a, clk_b, a_on, b_on = finite_edge_clock_pair(
            tgrid, f_chop, v_low=0.0, v_high=bias["VDD"],
            edge_time=edge_time, dead_time=dead_time,
            phase_offset=clock_phase_offset)
    edge_intervals = ((np.abs(np.diff(clk_a)) > 1e-12) |
                      (np.abs(np.diff(clk_b)) > 1e-12))
    transient_edge_mask = np.zeros(len(tgrid), dtype=bool)
    transient_edge_mask[:-1] |= edge_intervals
    transient_edge_mask[1:] |= edge_intervals
    tbias = dict(bias)
    tbias.update({"VIP": float(vip[0]), "VIN": float(vin[0]),
                  "CLK_A": float(clk_a[0]), "CLK_B": float(clk_b[0])})
    inputs = {"vip": vip, "vin": vin, "clk_a": clk_a, "clk_b": clk_b}
    node_inputs = {
        build.input_nodes[0]: "vip",
        build.input_nodes[1]: "vin",
        build.clock_nodes[0]: "clk_a",
        build.clock_nodes[1]: "clk_b",
    }

    current_inputs = []
    qinj_waveforms = {}
    if charge_injection:
        qinj_waveforms, current_inputs = _charge_injection_sources(
            build, all_sizes, all_nf, bias, tgrid, clk_a, clk_b,
            charge_scale=charge_scale, source_split=charge_source_split,
            corner=corner)
        inputs.update(qinj_waveforms)

    if transient_max_step is None:
        transient_max_step = (float(edge_time) / 20.0
                              if edge_time else period / 200.0)
    if transient_flat_max_step is None:
        transient_flat_max_step = (max(float(transient_max_step), float(edge_time) / 18.0)
                                   if edge_time else 0.0)
    if V0 is None:
        start_phase = "A" if float(a_on[0]) >= float(b_on[0]) else "B"
        seed = _pmos_chopper_auto_seed(
            sizes, tbias, start_phase, build, nf=nf, corner=corner,
            base_topo=base_topo)
        if seed is not None:
            ac0 = ac_solve(all_sizes, tbias, np.array([1.0]), topo=build.topology,
                           nf=all_nf, corner=corner, x0_guess=seed)
            dc0 = ac0["dc_op"] if ac0 is not None else seed
            V0 = np.array(build.topology.guess_vector(
                dc0, default=build.topology.default_guess_value(tbias)), dtype=float)
    if signed_switches is True:
        signed_devices = build.switch_names
    elif signed_switches is False or signed_switches is None:
        signed_devices = ()
    else:
        signed_devices = tuple(signed_switches)

    result = transient(all_sizes, tbias, tgrid, topo=build.topology, inputs=inputs,
                       node_inputs=node_inputs, current_inputs=current_inputs,
                       nf=all_nf, corner=corner, V0=V0,
                       max_step=transient_max_step,
                       flat_max_step=transient_flat_max_step,
                       max_retry_subdivisions=max_retry_subdivisions,
                       newton_maxit=newton_maxit,
                       newton_step_limit=newton_step_limit,
                       newton_vtol=newton_vtol,
                       fallback_full_jacobian=fallback_full_jacobian,
                       fallback_least_squares=fallback_least_squares,
                       fallback_tol=fallback_tol,
                       signed_devices=signed_devices,
                       rail_margin=rail_margin,
                       profile=profile, edge_mask=transient_edge_mask,
                       integration_method=integration_method)
    requested_output = np.interp(requested_tgrid, result["t"], result["output"])
    requested_nodes = {
        name: np.interp(requested_tgrid, result["t"], vals)
        for name, vals in result["nodes"].items()
    }
    result.update({
        "topology": build.topology,
        "switch_names": build.switch_names,
        "switch_sizes": build.switch_sizes,
        "switch_nf": build.switch_nf,
        "split_input_pair": bool(build.split_input_pair),
        "all_sizes": all_sizes,
        "all_nf": all_nf,
        "requested_tgrid": requested_tgrid,
        "requested_output": requested_output,
        "requested_nodes": requested_nodes,
        "refined_edges": bool(refine_edges),
        "refined_point_count": int(len(tgrid)),
        "transient_max_step": transient_max_step,
        "transient_flat_max_step": transient_flat_max_step,
        "clk_a": clk_a,
        "clk_b": clk_b,
        "a_on": a_on,
        "b_on": b_on,
        "charge_injection_currents": qinj_waveforms,
        "charge_injection_sources": current_inputs,
        "node_inputs": node_inputs,
        "inputs": inputs,
        "bias": tbias,
        "clock_style": "pulse" if pulse_clock else "phase",
        "signed_devices": signed_devices,
        "corner": corner,
    })
    return result


def pmos_chopper_pss(sizes, bias, f_chop, *, input_diff=0.0,
                     input_common_mode=None, edge_time=0.0, dead_time=0.0,
                     clock_style="pulse", clock_phase_offset=0.25,
                     pulse_time_shift=None,
                     switch_size=(20000.0, 80.0), switch_nf=1, nf=None,
                     corner=None,
                     V0=None, charge_injection=True, charge_scale=1.0,
                     charge_source_split=0.5, refine_edges=True, edge_points=9,
                     signed_switches=True, tgrid=None, n_points=161,
                     tstab_periods=1, max_shooting_iters=8,
                     residual_tol=1e-7, fd_step=1e-5,
                     rail_margin=2.0,
                     transient_max_step=None, transient_flat_max_step=None,
                     max_retry_subdivisions=1, newton_maxit=60,
                     newton_step_limit=2.0, newton_vtol=1e-8,
                     fallback_full_jacobian=False,
                     fallback_least_squares=False, fallback_tol=1e-10,
                     analytic_jacobian=True,
                     base_topo=AFE_TOPO, output_filter=None, profile=False,
                     split_input_pair=False, integration_method="gear2",
                     cap_mode="average",
                     adaptive=False, adaptive_reltol=1e-4, adaptive_vabstol=1e-6,
                     adaptive_iabstol=1e-12, adaptive_max_steps=200000,
                     adaptive_h0=None, adaptive_freeze_factor=10.0,
                     adaptive_config=None):
    """Periodic steady state of the eight-PMOS chopper.

    This is a shooting PSS wrapper around the same hard-switched topology and
    transient stamps used by :func:`pmos_chopper_transient`.  It returns one
    periodic orbit; PAC/PNoise can be built on top of the returned trajectory.

    ``cap_mode`` selects the parasitic-cap transient operator for the orbit.
    Default ``"average"`` (trapezoidal ``0.5*(C(Vn)+C(Vn-1))*dV``): a STABLE,
    non-conservative discretization that matches Cadence's commutation
    feedthrough on the high-Z internal nodes -- closing the slow PAC gap to
    ~0% (the conservative ``"charge"`` Q-stamp over-swings the feedthrough ~26%,
    leaving a slow-corner +1% residual). ``"charge"`` stays the global default
    for stiff tau>>T circuits (e.g. SC-LPF) where the trapezoidal rule rings.

    ``adaptive=True`` is rejected on the chopper wrapper: the LTE step controller
    currently collapses the step to ~0 at hard switch edges, producing a one-point
    orbit and invalid PAC/PNoise. Use the validated fixed edge-refined grid until
    chopper-specific adaptive stepping is implemented.
    """
    adaptive_config = resolve_adaptive_config(
        adaptive_config,
        adaptive_reltol=adaptive_reltol,
        adaptive_vabstol=adaptive_vabstol,
        adaptive_iabstol=adaptive_iabstol,
        adaptive_max_steps=adaptive_max_steps,
        adaptive_h0=adaptive_h0,
        adaptive_freeze_factor=adaptive_freeze_factor,
    )
    if adaptive:
        raise ValueError(
            "pmos_chopper_pss does not support adaptive=True on the hard-switched "
            "chopper topology yet; use the fixed edge-refined grid, or call the "
            "generic pss_solve adaptive path on a topology where the LTE controller "
            "is validated."
        )
    _CAP_MODE_IDS = {"charge": 0, "q": 0, "average": 1, "avg": 1, "trapezoid": 1,
                     "qstamp": 0, "q-stamp": 0, "trap": 1}
    if cap_mode is None:
        cap_mode_id = None
    else:
        key = str(cap_mode).lower()
        if key not in _CAP_MODE_IDS:
            raise ValueError(
                "cap_mode must be 'charge' or 'average' "
                "(aliases: q/qstamp/q-stamp, avg/trapezoid/trap)")
        cap_mode_id = _CAP_MODE_IDS[key]
    f_chop = float(f_chop)
    if f_chop <= 0.0:
        raise ValueError("f_chop must be positive")
    period = 1.0 / f_chop
    if tgrid is None:
        requested_tgrid = np.linspace(0.0, period, max(2, int(n_points)))
    else:
        requested_tgrid = np.asarray(tgrid, float)
    if len(requested_tgrid) < 2:
        raise ValueError("tgrid must contain at least two points")
    if not np.isclose(requested_tgrid[0], 0.0, rtol=0.0, atol=period * 1e-14):
        raise ValueError("PSS tgrid must start at 0")
    if not np.isclose(requested_tgrid[-1], period, rtol=1e-12, atol=period * 1e-12):
        raise ValueError("PSS tgrid must end at one chopper period")

    clock_style = str(clock_style).lower()
    pulse_clock = clock_style in {"pulse", "spectre", "spectre_pulse"}
    if clock_style not in {"phase", "finite_edge", "legacy", "pulse", "spectre", "spectre_pulse"}:
        raise ValueError("clock_style must be 'pulse' or 'phase'")
    if pulse_clock and dead_time:
        raise ValueError("dead_time is only supported with clock_style='phase'")

    build = build_afe_pmos_chopper(switch_size=switch_size, switch_nf=switch_nf,
                                   base_topo=base_topo, output_filter=output_filter,
                                   split_input_pair=split_input_pair)
    all_sizes, all_nf = _with_switch_maps(sizes, nf, build)
    vcm = bias["VCM"] if input_common_mode is None else float(input_common_mode)

    t_period = requested_tgrid
    diff = input_diff(t_period) if callable(input_diff) else input_diff
    diff = np.asarray(diff, float)
    if diff.ndim == 0:
        diff = np.full_like(t_period, float(diff))
    if len(diff) != len(t_period):
        raise ValueError("input_diff waveform length must match tgrid")
    vip = vcm + 0.5 * diff
    vin = vcm - 0.5 * diff

    tgrid_work = t_period
    if refine_edges:
        if pulse_clock:
            shift = 0.5 * period if pulse_time_shift is None else float(pulse_time_shift)
            tgrid_work = refine_pulse_clock_tgrid(
                t_period, f_chop, edge_time=edge_time,
                edge_points=edge_points, hard_edge_window=period / 200.0,
                time_shift=shift)
        else:
            tgrid_work = refine_chopper_tgrid(
                t_period, f_chop, edge_time=edge_time, dead_time=dead_time,
                phase_offset=clock_phase_offset, edge_points=edge_points,
                hard_edge_window=period / 200.0)
        vip = np.interp(tgrid_work, t_period, vip)
        vin = np.interp(tgrid_work, t_period, vin)

    if pulse_clock:
        # Shift by T/2 by default so the PSS waveform starts at the Spectre pulse
        # edge, matching Cadence's PSS time-domain phase while keeping endpoints
        # periodic.
        shift = 0.5 * period if pulse_time_shift is None else float(pulse_time_shift)
        clk_a, clk_b, a_on, b_on = spectre_pulse_clock_pair(
            tgrid_work + shift, f_chop, v_low=0.0, v_high=bias["VDD"],
            edge_time=edge_time)
    else:
        clk_a, clk_b, a_on, b_on = finite_edge_clock_pair(
            tgrid_work, f_chop, v_low=0.0, v_high=bias["VDD"],
            edge_time=edge_time, dead_time=dead_time,
            phase_offset=clock_phase_offset)

    edge_intervals = ((np.abs(np.diff(clk_a)) > 1e-12) |
                      (np.abs(np.diff(clk_b)) > 1e-12))
    edge_mask = np.zeros(len(tgrid_work), dtype=bool)
    edge_mask[:-1] |= edge_intervals
    edge_mask[1:] |= edge_intervals

    tbias = dict(bias)
    tbias.update({"VIP": float(vip[0]), "VIN": float(vin[0]),
                  "CLK_A": float(clk_a[0]), "CLK_B": float(clk_b[0])})
    inputs = {"vip": vip, "vin": vin, "clk_a": clk_a, "clk_b": clk_b}
    node_inputs = {
        build.input_nodes[0]: "vip",
        build.input_nodes[1]: "vin",
        build.clock_nodes[0]: "clk_a",
        build.clock_nodes[1]: "clk_b",
    }

    current_inputs = []
    qinj_waveforms = {}
    if charge_injection:
        qinj_waveforms, current_inputs = _charge_injection_sources(
            build, all_sizes, all_nf, bias, tgrid_work, clk_a, clk_b,
            charge_scale=charge_scale, source_split=charge_source_split,
            corner=corner)
        inputs.update(qinj_waveforms)

    if transient_max_step is None:
        transient_max_step = (float(edge_time) / 20.0
                              if edge_time else period / 200.0)
    if transient_flat_max_step is None:
        transient_flat_max_step = (max(float(transient_max_step), float(edge_time) / 18.0)
                                   if edge_time else 0.0)
    if V0 is None:
        start_phase = "A" if float(a_on[0]) >= float(b_on[0]) else "B"
        seed = _pmos_chopper_auto_seed(
            sizes, tbias, start_phase, build, nf=nf, corner=corner,
            base_topo=base_topo)
        if seed is not None:
            V0 = np.array(build.topology.guess_vector(
                seed, default=build.topology.default_guess_value(tbias)), dtype=float)

    if signed_switches is True:
        signed_devices = build.switch_names
    elif signed_switches is False or signed_switches is None:
        signed_devices = ()
    else:
        signed_devices = tuple(signed_switches)

    result = pss_solve(
        all_sizes, tbias, period, topo=build.topology, nf=all_nf,
        corner=corner,
        tgrid=tgrid_work, inputs=inputs, node_inputs=node_inputs,
        current_inputs=current_inputs, V0=V0, tstab_periods=tstab_periods,
        max_step=transient_max_step, flat_max_step=transient_flat_max_step,
        max_retry_subdivisions=max_retry_subdivisions,
        newton_maxit=newton_maxit, newton_step_limit=newton_step_limit,
        newton_vtol=newton_vtol, fallback_full_jacobian=fallback_full_jacobian,
        fallback_least_squares=fallback_least_squares, fallback_tol=fallback_tol,
        signed_devices=signed_devices, residual_tol=residual_tol,
        max_shooting_iters=max_shooting_iters, fd_step=fd_step,
        analytic_jacobian=analytic_jacobian,
        rail_margin=rail_margin, check_periodic_inputs=False, profile=profile,
        edge_mask=edge_mask, integration_method=integration_method,
        cap_mode_id=cap_mode_id,
        adaptive=adaptive, adaptive_config=adaptive_config,
    )
    requested_output = np.interp(requested_tgrid, result["t"], result["output"])
    requested_nodes = {
        name: np.interp(requested_tgrid, result["t"], vals)
        for name, vals in result["nodes"].items()
    }
    result.update({
        "topology": build.topology,
        "switch_names": build.switch_names,
        "switch_sizes": build.switch_sizes,
        "switch_nf": build.switch_nf,
        "split_input_pair": bool(build.split_input_pair),
        "all_sizes": all_sizes,
        "all_nf": all_nf,
        "requested_tgrid": requested_tgrid,
        "requested_output": requested_output,
        "requested_nodes": requested_nodes,
        "refined_edges": bool(refine_edges),
        "refined_point_count": int(len(tgrid_work)),
        "transient_max_step": transient_max_step,
        "transient_flat_max_step": transient_flat_max_step,
        "clk_a": clk_a,
        "clk_b": clk_b,
        "a_on": a_on,
        "b_on": b_on,
        "charge_injection_currents": qinj_waveforms,
        "charge_injection_sources": current_inputs,
        "node_inputs": node_inputs,
        "inputs": inputs,
        "bias": tbias,
        "clock_style": "pulse" if pulse_clock else "phase",
        "signed_devices": signed_devices,
        "corner": corner,
    })
    return result


def pmos_chopper_pac(sizes, bias, freqs, f_chop, *, pss_result=None,
                     nf=None, corner=None,
                     pacmag=1.0, fd_state_step=1e-4, fd_input_step=1e-4,
                     pss_kwargs=None, transient_kwargs=None,
                     cache_linearization=True, cache_forcing=True,
                     compute_condition=None, lti_fast_path=True,
                     analytic=True, n_period_samples=384, max_sideband=64,
                     time_domain=True, td_integration="gear2",
                     td_n_period_samples=768,
                     profile=False, debug=False):
    """PSS-assisted small-signal PAC for the PMOS chopper.

    Builds the chopper PSS orbit when needed, then delegates to
    :func:`core.pac_solver.pac_solve`. By default this uses the time-domain
    shooting PAC kernel, which keeps the PMOS_TFT internal gate1 small-signal
    states and avoids HB sideband truncation. Set ``time_domain=False`` to use
    the analytic-adjoint HB conversion matrix as an explicit comparison path, or
    ``analytic=False`` for the finite-difference shooting kernel.

    ``max_sideband`` defaults to 64 here (vs. 10 for smooth orbits) for the
    optional HB path: the hard switch edges spread the input/output commutation
    across many sidebands, so the baseband conversion gain only converges to
    Spectre PAC once many sidebands are kept.

    Do not lower the HB ``max_sideband`` default when using
    ``time_domain=False`` as a Cadence comparison path. The per-frequency cost is
    ~((2K+1)*n)^3 so a smaller K is tempting, but a 3-corner calibration K-sweep
    (2026-06-26) showed K=64 was load-bearing for the old collapsed HB model.
    The production default no longer pays that cost because it uses the
    truncation-free time-domain PAC path.
    """
    pss_kwargs = dict(pss_kwargs or {})
    transient_kwargs = dict(transient_kwargs or {})
    if pss_result is None:
        pss_defaults = dict(charge_injection=False, max_shooting_iters=0,
                            tstab_periods=1, fallback_least_squares=False)
        if nf is not None:
            pss_defaults["nf"] = nf
        if corner is not None:
            pss_defaults["corner"] = corner
        pss_defaults.update(pss_kwargs)
        pss_result = pmos_chopper_pss(sizes, bias, f_chop, **pss_defaults)
    if corner is None:
        corner = pss_result.get("corner")
    return pac_solve(
        sizes, bias, freqs, pss_result=pss_result,
        input_drive={"vip": 0.5, "vin": -0.5}, nf=nf, corner=corner,
        fd_state_step=fd_state_step, fd_input_step=fd_input_step,
        transient_kwargs=transient_kwargs, pacmag=pacmag,
        rail_margin=pss_kwargs.get("rail_margin", 2.0),
        cache_linearization=cache_linearization,
        cache_forcing=cache_forcing,
        compute_condition=compute_condition,
        lti_fast_path=lti_fast_path,
        analytic=analytic, n_period_samples=n_period_samples,
        max_sideband=max_sideband,
        time_domain=time_domain, td_integration=td_integration,
        td_n_period_samples=td_n_period_samples,
        profile=profile, debug=debug,
    )


def pmos_chopper_pnoise(sizes, bias, freqs, f_chop, *, pss_result=None, nf=None,
                        corner=None,
                        max_sideband=32, n_period_samples=384,
                        time_domain=True,
                        band=(0.05, 100.0), gains=None, pac_result=None,
                        noise_devices=None, switch_noise_conductance_gated=True,
                        cache_linearization=True, lti_fast_path=True,
                        hb_solver="auto", hb_sparse_min_size=384,
                        hb_sparse_max_density=0.12, hb_sparse_drop_tol=0.0,
                        iterative_tol=1e-10, iterative_maxiter=10,
                        profile=False, pss_kwargs=None, base_topo=AFE_TOPO):
    """PSS-based LPTV periodic noise for the eight-PMOS chopper.

    This wrapper builds the chopper PSS orbit when needed, then delegates the
    periodic-noise conversion to :func:`core.pnoise_solver.pnoise_solve`.

    The default ``time_domain=True`` uses the sparse Floquet-adjoint PNoise path,
    which removes the HB adjoint sideband-truncation error seen on the hard
    chopper. Set ``time_domain=False`` to run the harmonic-balance path as an
    explicit comparison/fallback. In that mode ``max_sideband`` defaults to 32
    because the local HB noise-conversion truncation converges more slowly than
    Spectre's shooting PNoise for hard switch edges.
    """
    f_chop = float(f_chop)
    if f_chop <= 0.0:
        raise ValueError("f_chop must be positive")

    pss_kwargs = dict(pss_kwargs or {})
    if pss_result is None:
        pss_defaults = dict(charge_injection=False, tstab_periods=1,
                            fallback_least_squares=False)
        if nf is not None:
            pss_defaults["nf"] = nf
        if corner is not None:
            pss_defaults["corner"] = corner
        pss_defaults.update(pss_kwargs)
        pss_result = pmos_chopper_pss(sizes, bias, f_chop, base_topo=base_topo,
                                      **pss_defaults)
    if corner is None:
        corner = pss_result.get("corner")
    gds_noise_devices = pss_result.get("switch_names", ())
    # Input-refer with the chopper's own PAC gain (default time-domain PAC with
    # PMOS gate1 states). Otherwise pnoise_solve falls back to pac_solve's
    # generic HB defaults, which are not tuned for hard-switch chopper gain.
    if gains is None and pac_result is None:
        pac_result = pmos_chopper_pac(
            sizes, bias, freqs, f_chop, pss_result=pss_result, nf=nf,
            corner=corner)
    return pnoise_solve(
        sizes, bias, freqs, pss_result=pss_result, fundamental=f_chop,
        nf=nf, corner=corner, max_sideband=max_sideband,
        n_period_samples=n_period_samples, time_domain=time_domain,
        band=band, gains=gains, pac_result=pac_result,
        input_drive={"vip": 0.5, "vin": -0.5},
        noise_devices=noise_devices, gds_noise_devices=gds_noise_devices,
        switch_noise_conductance_gated=switch_noise_conductance_gated,
        cache_linearization=cache_linearization,
        lti_fast_path=lti_fast_path,
        hb_solver=hb_solver,
        hb_sparse_min_size=hb_sparse_min_size,
        hb_sparse_max_density=hb_sparse_max_density,
        hb_sparse_drop_tol=hb_sparse_drop_tol,
        iterative_tol=iterative_tol,
        iterative_maxiter=iterative_maxiter,
        profile=profile,
    )


def pmos_chopper_lptv_analysis(sizes, bias, freqs, f_chop, *,
                               switch_size=(20000.0, 80.0), switch_nf=1,
                               nf=None, corner=None, x0_guess=None,
                               band=(0.05, 100.0), max_harmonic=31,
                               edge_time=0.0, dead_time=0.0,
                               conversion_phase_rad=0.0,
                               periodic_noise_psd_scale=1.0,
                               harmonic_samples=4096, base_topo=AFE_TOPO,
                               output_filter=None):
    """PMOS-switch sideband folding with finite-edge clock weights.

    This folds the PMOS-switch static phase response/noise at sideband
    frequencies using the actual finite-edge modulation harmonics. It captures
    PMOS Ron/cap/noise plus finite-edge spectral weights.

    This is a fast FIRST-ORDER estimate: it sums the frozen-phase sideband
    responses with the chopper's finite-edge harmonic weights, so it omits the
    higher-order LPTV conversion and underestimates the baseband gain by ~10%.
    For Cadence-grade gain/noise use the first-principles harmonic-balance path
    (``pmos_chopper_pss`` -> ``pmos_chopper_pac``/``pmos_chopper_pnoise``), which
    carries no empirical constants. ``conversion_phase_rad`` and
    ``periodic_noise_psd_scale`` remain as optional manual knobs (default 0 / 1,
    i.e. the raw first-order sum); the old Cadence-fit defaults were retired.
    """
    freqs = np.asarray(freqs, float)
    if np.any(freqs < 0.0):
        raise ValueError("freqs must be non-negative baseband frequencies")
    f_chop = float(f_chop)
    if f_chop <= 0.0:
        raise ValueError("f_chop must be positive")

    if edge_time or dead_time:
        harmonics, coeffs, weights = finite_edge_chopper_harmonics(
            max_harmonic,
            edge_fraction=float(edge_time) * f_chop,
            dead_fraction=float(dead_time) * f_chop,
            samples=harmonic_samples,
        )
    else:
        harmonics, weights = square_chopper_harmonics(max_harmonic)
        coeffs = None

    signed_sidebands = freqs[:, None] + harmonics[None, :] * f_chop
    sideband_freqs = np.unique(np.abs(signed_sidebands).ravel())
    sideband_freqs = sideband_freqs[sideband_freqs > 0.0]
    if sideband_freqs.size == 0:
        raise ValueError("sideband frequency set is empty")

    side = pmos_chopper_analysis(
        sizes, bias, sideband_freqs, switch_size=switch_size,
        switch_nf=switch_nf, nf=nf, corner=corner, x0_guess=x0_guess,
        band=band, phases=("A", "B"), base_topo=base_topo,
        output_filter=output_filter)
    if side is None:
        return None
    if side.get("response") is None:
        raise RuntimeError("pmos_chopper_analysis did not return complex AC response")

    # Optional manual knobs; default (0, 1) = the raw first-order sideband sum.
    conversion_phase_rad = float(conversion_phase_rad)
    periodic_noise_psd_scale = float(periodic_noise_psd_scale)
    if periodic_noise_psd_scale <= 0.0:
        raise ValueError("periodic_noise_psd_scale must be positive")

    h_sb = _interp_complex_response(signed_sidebands, sideband_freqs, side["response"])
    raw_h_chop = np.sum(h_sb * weights[None, :], axis=1)
    phase_weights = weights[None, :] * np.exp(
        1j * harmonics[None, :] * conversion_phase_rad
    )
    h_chop = np.sum(h_sb * phase_weights, axis=1)
    out_psd_sb = _interp_psd(signed_sidebands, sideband_freqs, side["out_psd"])
    raw_out_psd = np.sum(out_psd_sb * weights[None, :], axis=1)
    out_psd = raw_out_psd * periodic_noise_psd_scale
    gains = np.abs(h_chop)
    raw_gains = np.abs(raw_h_chop)
    irn_psd = out_psd / np.maximum(gains ** 2, 1e-300)
    raw_irn_psd = raw_out_psd / np.maximum(raw_gains ** 2, 1e-300)
    peak = float(np.max(gains))
    return {
        "freqs": freqs,
        "f_chop": f_chop,
        "harmonics": harmonics,
        "harmonic_coefficients": coeffs,
        "harmonic_weights": weights,
        "harmonic_weight_sum": float(np.sum(weights)),
        "edge_time": float(edge_time),
        "dead_time": float(dead_time),
        "conversion_phase_rad": conversion_phase_rad,
        "periodic_noise_psd_scale": periodic_noise_psd_scale,
        "sideband_freqs": sideband_freqs,
        "response": h_chop,
        "gains": gains,
        "Av_dc_dB": 20 * np.log10(max(float(gains[0]), 1e-300)),
        "peak_dB": 20 * np.log10(max(peak, 1e-300)),
        "bw_Hz": bw_from_gain(freqs, gains),
        "out_psd": out_psd,
        "irn_psd": irn_psd,
        "irn_uV_band": band_rms(freqs, irn_psd, band[0], band[1]) * 1e6,
        "raw_quasi_response": raw_h_chop,
        "raw_quasi_gains": raw_gains,
        "raw_quasi_Av_dc_dB": 20 * np.log10(max(float(raw_gains[0]), 1e-300)),
        "raw_quasi_bw_Hz": bw_from_gain(freqs, raw_gains),
        "raw_quasi_out_psd": raw_out_psd,
        "raw_quasi_irn_psd": raw_irn_psd,
        "raw_quasi_irn_uV_band": (
            band_rms(freqs, raw_irn_psd, band[0], band[1]) * 1e6
        ),
        "pmos_sideband": side,
        "analysis_note": (
            "First-order quasi-static PMOS-switch sideband folding with finite-edge "
            "harmonic weights; underestimates gain ~10% (omits higher-order LPTV "
            "conversion) and is not a correlated periodic-noise solve. Use the "
            "harmonic-balance pmos_chopper_pac / pmos_chopper_pnoise for accuracy."
        ),
    }


def chopper_analysis(sizes, bias, freqs, f_chop, *, topo=AFE_TOPO, nf=None,
                     corner=None, x0_guess=None, band=(0.05, 100.0),
                     max_harmonic=31, edge_time=0.0, dead_time=0.0,
                     harmonic_samples=4096):
    """Gain/BW/noise for an ideal synchronized input+output differential chopper.

    Parameters
    ----------
    sizes, bias, freqs, topo, nf, corner, x0_guess
        Same meaning as `noise_analysis` / `ac_solve`.
    f_chop
        Chopper clock frequency in Hz.
    band
        Integration band for `irn_uV_band`.
    max_harmonic
        Largest odd square-wave harmonic included in the sideband sum.
    edge_time, dead_time
        Optional finite clock edge / break-before-make time. When either is
        non-zero, sideband weights come from the Fourier coefficients of the
        finite-edge modulation waveform instead of the ideal square wave.
    """
    freqs = np.asarray(freqs, float)
    if np.any(freqs < 0.0):
        raise ValueError("freqs must be non-negative baseband frequencies")
    f_chop = float(f_chop)
    if f_chop <= 0.0:
        raise ValueError("f_chop must be positive")

    if edge_time or dead_time:
        harmonics, coeffs, weights = finite_edge_chopper_harmonics(
            max_harmonic,
            edge_fraction=float(edge_time) * f_chop,
            dead_fraction=float(dead_time) * f_chop,
            samples=harmonic_samples,
        )
    else:
        harmonics, weights = square_chopper_harmonics(max_harmonic)
        coeffs = None
    signed_sidebands = freqs[:, None] + harmonics[None, :] * f_chop
    # Avoid duplicated sideband solves; noise_analysis expects positive freqs.
    sideband_freqs = np.unique(np.abs(signed_sidebands).ravel())
    sideband_freqs = sideband_freqs[sideband_freqs > 0.0]
    if sideband_freqs.size == 0:
        raise ValueError("sideband frequency set is empty")

    side = noise_analysis(sizes, bias, sideband_freqs, corner=corner,
                          x0_guess=x0_guess, topo=topo, nf=nf)
    if side is None:
        return None
    if side.get("response") is None:
        raise RuntimeError("noise_analysis did not return complex AC response")

    h_sb = _interp_complex_response(signed_sidebands, sideband_freqs, side["response"])
    h_chop = np.sum(h_sb * weights[None, :], axis=1)

    out_psd_sb = _interp_psd(signed_sidebands, sideband_freqs, side["out_psd"])
    out_psd = np.sum(out_psd_sb * weights[None, :], axis=1)
    gains = np.abs(h_chop)
    denom = np.maximum(gains ** 2, 1e-300)
    irn_psd = out_psd / denom

    peak = float(np.max(gains))
    return {
        "freqs": freqs,
        "f_chop": f_chop,
        "harmonics": harmonics,
        "harmonic_coefficients": coeffs,
        "harmonic_weights": weights,
        "harmonic_weight_sum": float(np.sum(weights)),
        "edge_time": float(edge_time),
        "dead_time": float(dead_time),
        "sideband_freqs": sideband_freqs,
        "response": h_chop,
        "gains": gains,
        "Av_dc_dB": 20 * np.log10(max(float(gains[0]), 1e-300)),
        "peak_dB": 20 * np.log10(max(peak, 1e-300)),
        "bw_Hz": bw_from_gain(freqs, gains),
        "out_psd": out_psd,
        "irn_psd": irn_psd,
        "irn_uV_band": band_rms(freqs, irn_psd, band[0], band[1]) * 1e6,
        "dc": side["dc"],
        "baseband_noise": side,
    }
