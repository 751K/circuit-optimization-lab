#!/usr/bin/env python3
"""Compare CircuitOpt's native BSIM4 path with ngspice on one TSMC28 5T OTA."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from circuitopt import ac_solve, load_circuit_json, noise_analysis, transient  # noqa: E402
from circuitopt.device_factory import build_devices  # noqa: E402
from circuitopt.ngspice_ac import (  # noqa: E402
    ac_ngspice,
    ac_response,
    noise_ngspice,
    op_ngspice,
)


CONFIG = ROOT / "examples" / "tsmc28hpcp_5t_ota.json"


def _oracle_models(model_types):
    return {
        name: model_type.replace(
            "tsmc28hpcp.", "tsmc28hpcp_ngspice.", 1)
        for name, model_type in model_types.items()
    }


def _relative_error(actual, reference, floor=1e-30):
    return np.abs(actual - reference) / np.maximum(np.abs(reference), floor)


def compare() -> dict:
    spec = load_circuit_json(CONFIG)
    oracle_models = _oracle_models(spec.model_types)
    shared_oracle = {
        "topo": spec.topology,
        "nf": spec.nf,
        "model_types": oracle_models,
        "device_kwargs": spec.device_kwargs,
        "corner": "tt",
        "x0_guess": spec.topology.dc_guesses[0],
    }
    report = {"config": str(CONFIG.relative_to(ROOT)), "corner": "tt"}

    started = time.perf_counter()
    oracle_ac = ac_ngspice(
        spec.sizes,
        spec.bias,
        acmag={"vinp": (0.5, 0.0), "vinn": (0.5, 180.0)},
        fstart=1e3,
        fstop=1e11,
        points=15,
        out_nodes=["tail", "n1", "vout"],
        **shared_oracle,
    )
    native_ac = ac_solve(
        spec.sizes,
        spec.bias,
        oracle_ac["freq"],
        binding=spec.binding(),
        corner="tt",
    )
    if native_ac is None:
        raise RuntimeError("native AC/DC solve failed")
    oracle_h = ac_response(oracle_ac, "vout", vin=1.0)
    native_h = native_ac["response"]
    useful = np.maximum(np.abs(oracle_h), np.abs(native_h)) > 1e-6
    gain_error_db = np.abs(
        20.0 * np.log10(np.maximum(np.abs(native_h[useful]), 1e-300))
        - 20.0 * np.log10(np.maximum(np.abs(oracle_h[useful]), 1e-300))
    )
    report["dc_native_v"] = {
        name: float(native_ac["dc_op"][name])
        for name in spec.topology.solved
    }
    report["ac"] = {
        "native_gain_db_1khz": float(
            20.0 * np.log10(abs(native_h[0]))),
        "ngspice_gain_db_1khz": float(
            20.0 * np.log10(abs(oracle_h[0]))),
        "max_gain_error_db": float(np.max(gain_error_db)),
    }

    native_devices = build_devices(
        spec.sizes,
        nf=spec.nf,
        corner=None,
        topo=spec.topology,
        model_types=spec.model_types,
        device_kwargs=spec.device_kwargs,
    )
    from circuitopt.compiled_topology import CompiledTopology

    plan = CompiledTopology(spec.topology, spec.bias)
    bias_points = plan.bias_points(report["dc_native_v"])
    oracle_op = op_ngspice(spec.sizes, spec.bias, **shared_oracle)
    device_rows = {}
    for name, dev in native_devices.items():
        native_ss = dev.get_ss_params(*bias_points[name])
        native_id = abs(dev.get_Idc(*bias_points[name]))
        reference = oracle_op[name]
        device_rows[name] = {
            "id_rel_error": float(_relative_error(
                native_id, abs(reference["id"]))),
            "gm_rel_error": float(_relative_error(
                native_ss["gm"], abs(reference["gm"]))),
            "gds_rel_error": float(_relative_error(
                native_ss["gds"], abs(reference["gds"]))),
        }
    report["device_op"] = device_rows

    oracle_noise = noise_ngspice(
        spec.sizes,
        spec.bias,
        out="vout",
        src="vinp",
        fstart=1e3,
        fstop=1e10,
        points=15,
        band=(1e3, 1e10),
        **shared_oracle,
    )
    native_noise = noise_analysis(
        spec.sizes,
        spec.bias,
        oracle_noise["freq"],
        binding=spec.binding(),
        corner="tt",
    )
    if native_noise is None:
        raise RuntimeError("native noise solve failed")
    native_noise_rms = float(np.sqrt(np.trapezoid(
        native_noise["out_psd"], oracle_noise["freq"])))
    report["noise"] = {
        "native_output_rms_v": native_noise_rms,
        "ngspice_output_rms_v": float(oracle_noise["onoise_rms"]),
        "rms_rel_error": float(_relative_error(
            native_noise_rms, oracle_noise["onoise_rms"])),
    }

    tgrid = np.linspace(0.0, 10e-9, 201)
    differential_step = np.where(tgrid < 1e-9, 0.0, 1e-3)
    waveforms = {
        "vinp": spec.bias["VCM"] + differential_step,
        "vinn": spec.bias["VCM"] - differential_step,
    }
    node_inputs = {"vinp": "vinp", "vinn": "vinn"}
    v0 = np.asarray([
        native_ac["dc_op"][name] for name in spec.topology.solved
    ])
    native_transient = transient(
        spec.sizes,
        spec.bias,
        tgrid,
        binding=spec.binding(),
        inputs=waveforms,
        node_inputs=node_inputs,
        V0=v0,
        corner="tt",
        integration_method="gear2",
        max_step=20e-12,
    )
    oracle_transient = transient(
        spec.sizes,
        spec.bias,
        tgrid,
        topo=spec.topology,
        nf=spec.nf,
        model_types=oracle_models,
        device_kwargs=spec.device_kwargs,
        inputs=waveforms,
        node_inputs=node_inputs,
        V0=v0,
        corner="tt",
        integration_method="gear2",
        max_step=20e-12,
    )
    transient_rows = {}
    for name in spec.topology.solved:
        error = (
            native_transient["nodes"][name]
            - oracle_transient["nodes"][name]
        )
        transient_rows[name] = {
            "max_abs_error_v": float(np.max(np.abs(error))),
            "rms_error_v": float(np.sqrt(np.mean(error * error))),
            "final_error_v": float(error[-1]),
        }
    report["transient"] = {
        "native_nfail": int(native_transient["nfail"]),
        "native_nnear": int(native_transient.get("nnear", 0)),
        "nodes": transient_rows,
    }
    max_device_error = max(
        value
        for row in device_rows.values()
        for value in row.values()
    )
    max_transient_error = max(
        row["max_abs_error_v"] for row in transient_rows.values())
    checks = {
        "ac_gain_error_le_0p01_db": (
            report["ac"]["max_gain_error_db"] <= 0.01),
        "device_op_error_le_0p1_percent": max_device_error <= 1e-3,
        "noise_rms_error_le_1_percent": (
            report["noise"]["rms_rel_error"] <= 0.01),
        "transient_node_error_le_0p2_mv": max_transient_error <= 2e-4,
        "transient_has_no_failed_steps": (
            report["transient"]["native_nfail"] == 0),
    }
    report["checks"] = checks
    report["passed"] = all(checks.values())
    report["runtime_s"] = float(time.perf_counter() - started)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON report path; the report is always printed.",
    )
    args = parser.parse_args()
    report = compare()
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="ascii")


if __name__ == "__main__":
    main()
