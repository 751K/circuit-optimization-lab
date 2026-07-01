"""Ideal voltage source (true MNA) tests.

Each voltage source adds a branch-current unknown and a constraint row V_p - V_q = E,
growing the system from n to n_aug = n + m. Cases check closed-form results: a resistive
divider (exact node voltages and branch current), a floating source (V_p - V_q == E), the
AC short / AC-stimulus paths, a time-varying transient source (RC step + sine), the JSON
round-trip + loader validation, the "no thermal noise" property, and the PSS guard.
"""
import numpy as np
import pytest

from core.ac_solver import ac_solve
from core.circuit_loader import circuit_from_dict, load_circuit_json
from core.noise_solver import noise_analysis
from core.pac_solver import pac_solve
from core.pnoise_solver import pnoise_solve
from core.pss_solver import pss_solve
from core.topology import Topology
from core.transient_solver import transient

_KB = 1.380649e-23
_TEMP = 300.15


def _divider(E=2.0, R1=1e3, R2=1e3, **kw):
    """V1 (EMF E) from IN to GND; R1 IN-MID, R2 MID-GND. V_MID = E*R2/(R1+R2)."""
    return Topology(solved=["IN", "MID"], devices=[], rails={"GND": 0.0},
                    outputs=("MID",),
                    resistors=[("R1", "IN", "MID", R1), ("R2", "MID", "GND", R2)],
                    vsources=[("V1", "IN", "GND", E)], **kw)


# ── DC ──────────────────────────────────────────────────────────────────────

def test_vsource_dc_divider_exact():
    # True MNA pins V_IN = E exactly and gives the exact divider voltage (no Rs error).
    ac = ac_solve({}, {}, np.array([1.0]), topo=_divider(E=2.0, R1=1e3, R2=3e3))
    assert ac["dc_op"]["IN"] == pytest.approx(2.0, abs=1e-9)         # pinned by constraint
    # MID via KCL; the only departure from exact is the 1e-12 gmin leakage (~1e-9 V).
    assert ac["dc_op"]["MID"] == pytest.approx(2.0 * 3e3 / 4e3, abs=1e-6)
    # branch current magnitude = E / (R1 + R2)
    assert abs(ac["branch_currents"]["V1"]) == pytest.approx(2.0 / 4e3, rel=1e-6)


def test_vsource_floating_constraint_exact():
    # Source between two solved nodes (neither is a rail): V_A - V_B == E exactly.
    topo = Topology(solved=["A", "B"], devices=[], rails={"GND": 0.0}, outputs=("A",),
                    resistors=[("RA", "A", "GND", 1e3), ("RB", "B", "GND", 2e3)],
                    vsources=[("V1", "A", "B", 1.5)])
    ac = ac_solve({}, {}, np.array([1.0]), topo=topo)
    assert ac["dc_op"]["A"] - ac["dc_op"]["B"] == pytest.approx(1.5, abs=1e-9)


def test_vsource_dimensions():
    topo = _divider()
    assert (topo.n, topo.n_branches, topo.n_aug) == (2, 1, 3)
    assert topo.vsource_index == {"V1": 2}


# ── AC ──────────────────────────────────────────────────────────────────────

def test_vsource_ac_short_no_coupling():
    # A DC source is an AC short: with no stimulus it pins IN to AC ground, so nothing
    # drives MID -> zero response (and the bordered Y stays nonsingular).
    g0 = ac_solve({}, {}, np.array([1.0]), topo=_divider())["gains"][0]
    assert g0 == pytest.approx(0.0, abs=1e-12)


def test_vsource_as_ac_stimulus():
    # Driving the source as a 1 V AC stimulus gives the divider ratio R2/(R1+R2)=0.5.
    topo = _divider(R1=1e3, R2=1e3, ac_drives={"V1": 1.0})
    g = ac_solve({}, {}, np.array([1.0]), topo=topo)["gains"][0]
    assert g == pytest.approx(0.5, rel=1e-9)


# ── Noise ─────────────────────────────────────────────────────────────────--

@pytest.mark.filterwarnings("ignore:divide by zero")  # passive net: no gain -> IRN inf
def test_vsource_carries_no_thermal_noise():
    # The ideal source is a short but NOT a noise source; only the resistors contribute.
    nz = noise_analysis({}, {}, np.array([1.0, 10.0]), topo=_divider())
    assert np.all(np.isfinite(nz["out_psd"]))
    assert "V1" not in nz["dev_psd"]
    assert "R1" in nz["dev_psd"] and "R2" in nz["dev_psd"]


# ── Transient ─────────────────────────────────────────────────────────────--

def test_vsource_transient_constant_divider():
    # Static source -> output sits at the divider value at every timestep. Since P4 the
    # numba gear2 grid handles the augmented (n_aug>n) vsource system directly.
    tr = transient({}, {}, np.linspace(0, 1e-3, 51), topo=_divider(E=2.0),
                   integration_method="gear2")
    assert tr["nfail"] == 0
    assert tr["numba_grid_solver"] is True
    assert np.allclose(tr["output"], 1.0, atol=1e-6)


def test_vsource_transient_timevarying_rc_step():
    # value="vsrc" -> time-varying EMF from the input waveform. RC lowpass step response:
    # MID(t) = E*(1 - exp(-t/RC)), RC = 1 ms (DC op starts at 0 since waveform EMF=0 in DC).
    topo = Topology(solved=["IN", "MID"], devices=[], rails={"GND": 0.0}, outputs=("MID",),
                    resistors=[("R1", "IN", "MID", 1e3)], load_caps=[("MID", "GND", 1e-6)],
                    vsources=[("V1", "IN", "GND", "vsrc")])
    N = 2001
    t = np.linspace(0, 5e-3, N)
    tr = transient({}, {}, t, topo=topo, inputs={"vsrc": np.ones(N)},
                   integration_method="be")
    assert tr["nfail"] == 0 and tr["numba_grid_solver"] is True
    expected = 1.0 - np.exp(-t / 1e-3)
    assert np.max(np.abs(tr["output"] - expected)) < 2e-3


def test_vsource_transient_timevarying_sine():
    # Sine EMF through the same RC: steady-state amplitude = 1/sqrt(1+(2*pi*f*RC)^2).
    topo = Topology(solved=["IN", "MID"], devices=[], rails={"GND": 0.0}, outputs=("MID",),
                    resistors=[("R1", "IN", "MID", 1e3)], load_caps=[("MID", "GND", 1e-6)],
                    vsources=[("V1", "IN", "GND", "vsrc")])
    f, tau = 50.0, 1e-3
    N = 20001
    t = np.linspace(0, 10.0 / f, N)
    tr = transient({}, {}, t, topo=topo, inputs={"vsrc": np.sin(2 * np.pi * f * t)},
                   integration_method="be")
    last = tr["output"][t >= 9.0 / f]                       # final period (steady state)
    amp = 0.5 * (last.max() - last.min())
    assert amp == pytest.approx(1.0 / np.sqrt(1 + (2 * np.pi * f * tau) ** 2), rel=5e-3)


# ── JSON loader / schema ─────────────────────────────────────────────────────

def test_vsource_example_json_runs():
    spec = load_circuit_json("examples/voltage_divider.json")
    ac = ac_solve(spec.sizes, spec.bias, np.array([1.0]), topo=spec.topology, nf=spec.nf)
    assert ac["dc_op"]["MID"] == pytest.approx(1.0, abs=1e-6)        # E=2, equal R -> 1.0
    assert "V1" in ac["branch_currents"]


def test_vsource_loader_roundtrip_object_and_tuple():
    data = {"solved": ["IN", "MID"], "rails": {"GND": 0.0}, "devices": [],
            "resistors": [["R1", "IN", "MID", 1e3], ["R2", "MID", "GND", 1e3]],
            "vsources": [{"name": "V1", "p": "IN", "q": "GND", "value": 2.0}],
            "outputs": ["MID"]}
    spec = circuit_from_dict(data)
    assert spec.topology.vsources == [("V1", "IN", "GND", 2.0)]
    assert (spec.topology.n_branches, spec.topology.n_aug) == (1, 3)
    data["vsources"] = [["V1", "IN", "GND", "vsrc"]]                 # tuple + waveform key
    assert circuit_from_dict(data).topology.vsources == [("V1", "IN", "GND", "vsrc")]


def test_vsource_loader_rejects_unknown_node():
    bad = {"solved": ["IN"], "rails": {"GND": 0.0}, "devices": [],
           "vsources": [["V1", "IN", "NOPE", 1.0]], "outputs": ["IN"]}
    with pytest.raises(ValueError, match="unknown node"):
        circuit_from_dict(bad)


def test_vsource_loader_rejects_both_rails():
    bad = {"solved": ["MID"], "rails": {"VDD": "VDD", "GND": 0.0}, "devices": [],
           "resistors": [["R1", "MID", "GND", 1e3]],
           "vsources": [["V1", "VDD", "GND", 1.0]], "outputs": ["MID"]}
    with pytest.raises(ValueError, match="at least one solved node"):
        circuit_from_dict(bad)


def test_vsource_loader_rejects_identical_terminals():
    bad = {"solved": ["IN"], "rails": {"GND": 0.0}, "devices": [],
           "vsources": [["V1", "IN", "IN", 1.0]], "outputs": ["IN"]}
    with pytest.raises(ValueError, match="identical terminals"):
        circuit_from_dict(bad)


# ── Periodic analyses (PSS / PAC / PNoise) ───────────────────────────────────

def _rc_vsource_topo(R=1e5, C=1e-9):
    """VIN -(R1)- MID =(V1 short)= OUT -(C1)- GND. VIN is driven; the vsource shorts
    MID to OUT so the VIN->OUT transfer is the RC lowpass 1/(1+jw R C)."""
    return Topology(solved=["MID", "OUT"], devices=[], rails={"VIN": "VIN", "GND": 0.0},
                    outputs=("OUT",), resistors=[("R1", "VIN", "MID", R)],
                    capacitors=[("C1", "OUT", "GND", C)],
                    vsources=[("V1", "MID", "OUT", 0.0)])


def test_vsource_pss_periodic_orbit():
    # Linear RC driven by a periodic vsource E(t)=sin: the converged PSS orbit equals the
    # sinusoidal steady state, amplitude 1/sqrt(1+(2*pi f RC)^2). gear2 is exact.
    f, R, C = 50.0, 1e3, 1e-6
    period = 1.0 / f
    topo = Topology(solved=["IN", "MID"], devices=[], rails={"GND": 0.0}, outputs=("MID",),
                    resistors=[("R1", "IN", "MID", R)], load_caps=[("MID", "GND", C)],
                    vsources=[("V1", "IN", "GND", "vsrc")])
    t = np.linspace(0.0, period, 401)
    pss = pss_solve({}, {}, period, topo=topo, n_points=401, tgrid=t,
                    inputs={"vsrc": np.sin(2 * np.pi * f * t)}, integration_method="gear2",
                    rail_margin=None, max_shooting_iters=20)
    assert pss["converged"] and pss["nfail"] == 0
    mid = pss["nodes"]["MID"]
    amp = 0.5 * (mid.max() - mid.min())
    assert amp == pytest.approx(1.0 / np.sqrt(1 + (2 * np.pi * f * R * C) ** 2), rel=1e-3)


def test_vsource_pac_matches_rc_transfer():
    # For a linear circuit PAC sideband-0 == AC. Both the LTI fast path and the bordered
    # harmonic-balance path must reproduce the RC transfer through the vsource short.
    R, C, period = 1e5, 1e-9, 1e-3
    topo = _rc_vsource_topo(R, C)
    t = np.linspace(0.0, period, 401)
    pss = pss_solve({}, {"VIN": 0.0}, period, topo=topo, tgrid=t,
                    inputs={"vin": np.zeros_like(t)}, node_inputs={"VIN": "vin"},
                    V0=np.array([0.0, 0.0]), residual_tol=1e-12, max_shooting_iters=3)
    freqs = np.array([100.0, 500.0, 2000.0])
    expected = np.abs(1.0 / (1.0 + 2j * np.pi * freqs * R * C))
    for lti in (True, False):
        pac = pac_solve({}, {"VIN": 0.0}, freqs, pss_result=pss, input_drive={"vin": 1.0},
                        lti_fast_path=lti, transient_kwargs={"max_retry_subdivisions": 0})
        np.testing.assert_allclose(np.abs(pac["response"]), expected, rtol=1e-5)


@pytest.mark.filterwarnings("ignore:divide by zero")  # internal IRN of the no-drive net
def test_vsource_pnoise_resistor_noise_through_short():
    # Resistor thermal noise filtered through the vsource short: out_psd = 4kTR/(1+(wRC)^2).
    # The ideal source is a short but carries no noise (absent from dev_psd). Both the LTI
    # fast path and the bordered harmonic-balance path must agree with the closed form.
    R, C, period = 1e5, 1e-9, 1e-3
    topo = _rc_vsource_topo(R, C)
    t = np.linspace(0.0, period, 401)
    pss = pss_solve({}, {"VIN": 0.0}, period, topo=topo, tgrid=t,
                    inputs={"vin": np.zeros_like(t)}, node_inputs={"VIN": "vin"},
                    V0=np.array([0.0, 0.0]), residual_tol=1e-12, max_shooting_iters=3)
    freqs = np.array([100.0, 1000.0])
    expected = 4.0 * _KB * _TEMP * R / (1.0 + (2 * np.pi * freqs * R * C) ** 2)
    for lti in (True, False):
        nz = pnoise_solve({}, {"VIN": 0.0}, freqs, pss_result=pss, fundamental=1.0 / period,
                          input_drive={"vin": 1.0}, lti_fast_path=lti)
        np.testing.assert_allclose(nz["out_psd"], expected, rtol=1e-6)
        assert "R1" in nz["dev_psd"] and "V1" not in nz["dev_psd"]


@pytest.mark.cadence_regression
@pytest.mark.slow_regression
def test_sc_lpf_pss_converges_to_physical_orbit():
    # Switched-capacitor LPF (2-phase PMOS switches, vsource clocks): the reverse-
    # biased switch used to pump VMID/VOUT off a thin basin to a spurious ~40 V
    # (rail-clipped) orbit. With the signed drain current + robust shooting it must
    # converge to the physical ~20 V orbit and never report a runaway as converged.
    import examples.sc_lpf as L
    topo = L.build_sc_topo()
    sizes = {"M1": (L.W_SW, L.L_SW), "M2": (L.W_SW, L.L_SW)}
    n_points = 201
    t = np.linspace(0.0, L.PERIOD, n_points + 1)[:-1]
    inputs = L.build_inputs(t)
    # A generous tstab used to drift into the runaway; it must not anymore.
    pss = pss_solve(sizes, {}, L.PERIOD, topo=topo, n_points=n_points, inputs=inputs,
                    tstab_periods=60, residual_tol=2e-2, max_shooting_iters=20,
                    min_damping=1.0 / 256.0, integration_method="be")
    assert not pss["diverged"], f"PSS reported a runaway orbit: {pss['pss_status']}"
    vout = float(np.mean(pss["nodes"]["VOUT"]))
    assert 19.0 < vout < 21.0, f"VOUT settled to {vout:.2f} V, not the physical ~20 V"
    # PAC baseband transfer ~0 dB and -3 dB BW within ~10% of Cadence (16.96 Hz).
    pf = np.logspace(-1, 3, 41)
    pac = pac_solve(sizes, {}, pf, pss_result=pss, input_drive={"vin": 1.0},
                    fd_state_step=1e-4, fd_input_step=1e-4)
    g = np.asarray(pac["gains"], float)
    assert abs(g[0] - 1.0) < 0.05, f"PAC DC gain {g[0]:.3f} (expected ~1.0)"
    thr = g[0] / np.sqrt(2)
    bw = float(pf[-1])
    for i in range(1, len(g)):
        if g[i] < thr:
            bw = float(np.interp(thr, [g[i], g[i - 1]], [pf[i], pf[i - 1]])); break
    assert abs(bw - 16.96) / 16.96 < 0.12, f"PAC BW {bw:.2f} Hz vs Cadence 16.96 (>12%)"


@pytest.mark.cadence_regression
@pytest.mark.slow_regression
def test_sc_lpf_pac_is_integration_method_independent():
    # Stiff tau>>T switched-cap PAC must come from the analytic-adjoint HB (built from
    # the continuous small-signal G(t)/C(t) along the orbit), NOT the x0-sensitive
    # finite-difference shooting whose near-singular (I-Phi)^-1 turned a 0.003 V
    # be-vs-gear2 orbit difference into a 24x baseband gain. gear2 and be must agree,
    # both ~1, via the analytic adjoint (the vsource small-signal drive is coupled into
    # the bordered HB branch row, so the path no longer bails to FD shooting).
    import examples.sc_lpf as L
    topo = L.build_sc_topo()
    sizes = {"M1": (L.W_SW, L.L_SW), "M2": (L.W_SW, L.L_SW)}
    n_points = 201
    t = np.linspace(0.0, L.PERIOD, n_points + 1)[:-1]
    inputs = L.build_inputs(t)
    pf = np.logspace(-1, 3, 41)
    g0 = {}
    for method in ("be", "gear2"):
        pss = pss_solve(sizes, {}, L.PERIOD, topo=topo, n_points=n_points, inputs=inputs,
                        tstab_periods=60, residual_tol=2e-2, max_shooting_iters=20,
                        min_damping=1.0 / 256.0, integration_method=method)
        pac = pac_solve(sizes, {}, pf, pss_result=pss, input_drive={"vin": 1.0})
        assert pac["method"] == "pss_analytic_adjoint", (
            f"{method}: PAC used {pac['method']!r}, not the robust analytic adjoint "
            "(x0-sensitive FD shooting is unsafe for stiff tau>>T switched-cap)")
        g0[method] = float(np.abs(np.asarray(pac["gains"], float))[0])
    assert abs(g0["be"] - 1.0) < 0.05 and abs(g0["gear2"] - 1.0) < 0.05, g0
    assert abs(g0["gear2"] - g0["be"]) < 1e-2, (
        f"PAC baseband gain depends on the integration method: "
        f"be={g0['be']:.4f} gear2={g0['gear2']:.4f} (the 24x gear2 blowup is back)")
