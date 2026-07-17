"""Private bridge from ``CompiledTopology`` AC metadata to Rust LTI MNA."""
from __future__ import annotations

import numpy as np


def _term_record(term):
    kind, value = term
    if kind == "n":
        return 0, int(value), 0.0
    return 2, 0, float(value)


def build_lti_problem(plan, devices, device_instances, bias_points, ss,
                      capacitors, resistors, vccs, ac_drives=None):
    import circuitopt_core

    dense_devices = []
    mos_devices = []
    for name, drain, gate, source in devices:
        device = device_instances[name]
        if getattr(device, "HAS_TERMINAL_LINEARIZATION", False):
            conductance, capacitance = device.get_terminal_linearization(
                *bias_points[name])
            dense_devices.append((
                [_term_record(value) for value in
                 (drain, gate, source, ("v", 0.0))],
                np.asarray(conductance, float).tolist(),
                np.asarray(capacitance, float).tolist(),
            ))
        else:
            params = ss[name]
            mos_devices.append((
                _term_record(drain), _term_record(gate), _term_record(source),
                float(params["gm"]), float(params["gds"]),
                float(params["Cgs"]), float(params["Cgd"]),
            ))

    spec = {
        "size": int(plan.n_aug),
        "dense_devices": dense_devices,
        "mos_devices": mos_devices,
        "capacitors": [
            (_term_record(a), _term_record(b), float(value))
            for a, b, value in capacitors
        ],
        "resistors": [
            (_term_record(a), _term_record(b), float(conductance))
            for _name, a, b, _resistance, conductance in resistors
        ],
        "vccs": [
            (_term_record(p), _term_record(q), _term_record(cp),
             _term_record(cn), float(gm))
            for p, q, cp, cn, gm in vccs
        ],
        "voltage_sources": [
            (_term_record(p), _term_record(q), int(branch),
             float(complex(emf).real), float(complex(emf).imag))
            for p, q, branch, emf in plan.ac_vsources(ac_drives)
        ],
        "vcvs": [
            (_term_record(p), _term_record(q), _term_record(cp),
             _term_record(cn), int(branch), float(mu))
            for p, q, cp, cn, branch, mu in plan.ac_vcvs(ac_drives)
        ],
        "cccs": [
            (_term_record(p), _term_record(q), int(control), float(beta))
            for p, q, control, beta in plan.ac_cccs(ac_drives)
        ],
        "ccvs": [
            (_term_record(p), _term_record(q), int(control), int(branch),
             float(gamma))
            for p, q, control, branch, gamma in plan.ac_ccvs(ac_drives)
        ],
    }
    return circuitopt_core.LtiProblem(spec)


def complex_array(pairs):
    """Convert the stable PyO3 ``(real, imag)`` ABI to a NumPy complex array."""
    values = np.asarray(pairs, float)
    return values[..., 0] + 1j * values[..., 1]
