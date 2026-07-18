"""Thin, non-wired bridge to ``circuitopt_core.CompiledCampaign`` (rewrite R5-C).

Marshals the frozen AFE OTFT topology + analysis plan into the Rust compiled
campaign and expands a candidate matrix into the flat, index-ordered candidate
list the executor consumes. Random mismatch draws are made **up front** (numpy,
same rule as :func:`corners.mismatch_corner`) so the detached Rust batch never
calls back into Python.

This module is **not** wired into any production path. It exists only for the
R5-C parity/determinism tests and the later R5-D workflow integration; the
Python analysis paths (``ac_solver`` / ``noise_solver`` / ``corners``) stay the
single source of truth.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .compiled_topology import TERM_SOLVED, CompiledTopology
from .device_factory import CORNERS, dev_nf
from .topology import AFE_TOPO

# AT4000TG ``PMOS_TFT.__init__`` construction defaults, in the order the Rust
# ``OtftConstants`` expects: vt, ci, roff, reg, c1, c2, c3, c4, kv, kh, temp.
_OTFT_CONSTS = [-3.03, 2.4, 1.0, 1.0, 37.5, 50.0, 35.0, 35.0, 1.0, 1.0, 300.15]


def _dc_term(token) -> tuple[int, int, float]:
    """compiled_topology ``(kind, ref_or_value)`` -> ``(kind, ref, value)``."""
    kind, payload = token
    if kind == TERM_SOLVED:  # 0 -> solved node index
        return (0, int(payload), 0.0)
    return (2, 0, float(payload))  # TERM_RAIL (TERM_INPUT is transient-only)


def _ac_term(token) -> tuple[int, int, float]:
    """AC token ``("n", idx)`` / ``("v", value)`` -> ``(kind, ref, value)``."""
    tag, payload = token
    if tag == "n":
        return (0, int(payload), 0.0)
    return (2, 0, float(payload))


def _vin_norm(input_drives: Mapping[str, float], ac_drives: Mapping[str, float]) -> float:
    """Reproduce the gain normalization in ``ac_solver.ac_solve``."""
    norm_vals = list(ac_drives.values()) if ac_drives else list(input_drives.values())
    if not norm_vals:
        return 1.0
    if len(norm_vals) > 1 and max(norm_vals) > min(norm_vals):
        return max(norm_vals) - min(norm_vals)
    return max(abs(v) for v in norm_vals) or 1.0


class AfeOtftCampaign:
    """Compiled AFE OTFT campaign over one bias + analysis plan."""

    def __init__(self, bias: Mapping[str, float], freqs: Sequence[float],
                 band: tuple[float, float] = (0.05, 100.0), topo: Any = AFE_TOPO):
        import circuitopt_core

        self.topo = topo
        self.bias = dict(bias)
        self.plan = CompiledTopology(topo, bias)
        self.solved = tuple(self.plan.solved)
        self.freqs = [float(f) for f in np.asarray(freqs, float)]
        self.band = (float(band[0]), float(band[1]))
        self.default_guess = float(topo.default_guess_value(bias))
        self.device_names = tuple(name for name, *_ in topo.devices)

        drive = getattr(topo, "input_drives", {}) or {}
        node_drives = getattr(topo, "ac_drives", {}) or {}
        ac_devs = {name: (d, g, s)
                   for name, d, g, s in self.plan.ac_devices(drive=drive, node_drives=node_drives)}

        devices = []
        for dp in self.plan.devices:
            acd, acg, acs = ac_devs[dp.name]
            devices.append((
                _dc_term(dp.d), _dc_term(dp.g), _dc_term(dp.s),
                -1 if dp.di is None else int(dp.di),
                -1 if dp.si is None else int(dp.si),
                _ac_term(acd), _ac_term(acg), _ac_term(acs),
            ))

        ac_caps = [(_ac_term(a), _ac_term(b), float(v))
                   for a, b, v in self.plan.ac_capacitors()]
        output_weights = [(int(self.plan.idx[node]), float(w))
                          for node, w in self.plan.output_weights.items()]
        sense = [float(v) for v in self.plan.output_sense(dtype=float)]
        outs = topo.outputs
        latch_nodes = ((int(self.plan.idx[outs[0]]), int(self.plan.idx[outs[1]]))
                       if len(outs) == 2 else None)

        template = {
            "n_aug": int(self.plan.n_aug),
            "n_nodes": int(self.plan.n),
            "consts": list(_OTFT_CONSTS),
            "devices": devices,
            "ac_caps": ac_caps,
            "output_weights": output_weights,
            "sense": sense,
            "vin_norm": float(_vin_norm(drive, node_drives)),
            "freqs": self.freqs,
            "band": [self.band[0], self.band[1]],
            "gmin": 1e-12,
            "dc_tol": float(getattr(topo, "dc_tol", None) or 1e-10),
            "dc_guesses": [[float(v) for v in g] for g in topo.dc_guess_vectors(bias)],
            "latch_nodes": latch_nodes,
        }
        self.core = circuitopt_core.CompiledCampaign({"family": "afe_otft", "template": template})

    def seed_vector(self, dc_op: Mapping[str, float]) -> list[float]:
        """Solved-order DC seed vector from a ``{node: V}`` operating point."""
        return [float(dc_op.get(node, self.default_guess)) for node in self.solved]

    def candidate(self, sizes: Mapping[str, tuple[float, float]], corner=None,
                  mismatch: Mapping[str, Mapping[str, float]] | None = None,
                  nf=None, seed=None, trust_seed_as_op: bool = False) -> dict:
        """Build one marshalled candidate for the given sizes/corner/mismatch."""
        base = CORNERS[corner] if isinstance(corner, str) else dict(corner or {})
        pvt0 = float(base.get("pvt0", 0.0))
        pbeta0 = float(base.get("pbeta0", 0.0))
        mismatch = mismatch or {}
        devices = []
        for name in self.device_names:
            w, l = sizes[name]
            mm = mismatch.get(name, {})
            devices.append([
                float(w), float(l), float(dev_nf(nf, name)),
                pvt0, float(mm.get("mvt0", 0.0)),
                pbeta0, float(mm.get("mbeta0", 0.0)),
            ])
        out = {"devices": devices, "trust_seed_as_op": bool(trust_seed_as_op)}
        if seed is not None:
            out["seed"] = (self.seed_vector(seed) if isinstance(seed, Mapping)
                           else [float(v) for v in seed])
        return out

    def evaluate_batch(self, candidates: Sequence[dict], workers: int = 1,
                       analyses: Sequence[str] = ("dc", "ac", "noise")) -> list[dict]:
        """Run the compiled batch; results are candidate-index ordered."""
        return self.core.evaluate_batch(list(candidates), workers, list(analyses))
