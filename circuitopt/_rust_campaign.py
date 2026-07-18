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


def _reference_width_um(dev) -> float | None:
    """The device's SKY130 ``extract_w`` card-bin width, or ``None``.

    Mirrors the sky130 device wrapper exactly (``extract_w`` kwarg, else the class
    ``EXTRACT_W`` default): the card is binned on this width while the instance
    ``w`` keeps the actual geometry. ``None`` for FreePDK45/TSMC28 (no reference-
    width binning), so the Rust silicon pipeline bins on the actual width there.
    """
    ref = getattr(dev, "extract_w", None)
    if ref is None:
        ref = getattr(type(dev), "EXTRACT_W", None)
    return None if ref is None else float(ref)


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


def _silicon_pdk_of(model_types: Mapping[str, str]) -> str:
    """The single silicon PDK family a circuit's model types belong to."""
    families = {str(m).split(".", 1)[0] for m in model_types.values()}
    if len(families) != 1:
        raise ValueError(f"expected one silicon PDK family, got {sorted(families)}")
    family = families.pop()
    # Device-registry name -> CompiledPdk name.
    return {"tsmc28hpcp": "tsmc28"}.get(family, family)


def silicon_pdk_root(pdk: str) -> str:
    """Card root for :class:`circuitopt_core.CompiledPdk`, per PARITY.md."""
    if pdk == "freepdk45":
        from .toolchain import pdk_root

        return pdk_root()
    if pdk == "sky130":
        from .pdk.sky130.library import _BUNDLED_CARD_DIR

        return str(_BUNDLED_CARD_DIR)
    if pdk == "tsmc28":
        from .toolchain import tsmc28_model_dir

        return tsmc28_model_dir()
    raise ValueError(f"unknown silicon pdk {pdk!r}")


class SiliconCampaign:
    """Compiled silicon (BSIM4) campaign over one circuit spec + analysis plan.

    ``spec`` is a loaded circuit (:func:`circuit_loader.load_circuit_json`).
    The template captures everything candidate-invariant — passive circuit,
    per-device polarity/vb/temperature, LTI element records, analysis plan —
    while candidates carry geometry (+ per-candidate process corner and
    optional ``delvto`` mismatch volts).
    """

    def __init__(self, spec, freqs: Sequence[float],
                 band: tuple[float, float] = (1e3, 1e6)):
        import circuitopt_core

        from ._rust_transient import passive_problem_spec
        from .device_factory import build_devices
        from .dc_solver import DC_FALLBACK_TOL

        topo = spec.topology
        bias = dict(spec.bias)
        binding = spec.binding()
        self.topo = topo
        self.bias = bias
        self.model_types = dict(binding.model_types or {})
        self.device_kwargs = {name: dict(kw)
                              for name, kw in (binding.device_kwargs or {}).items()}
        self.base_sizes = dict(spec.sizes)
        self.nf = spec.nf
        self.pdk = _silicon_pdk_of(self.model_types)
        self.plan = CompiledTopology(topo, bias)
        self.solved = tuple(self.plan.solved)
        self.freqs = [float(f) for f in np.asarray(freqs, float)]
        self.band = (float(band[0]), float(band[1]))
        self.default_guess = float(topo.default_guess_value(bias))
        self.device_names = tuple(name for name, *_ in topo.devices)

        # Built once only to extract candidate-invariant statics (vb,
        # temperature, polarity, mult); cards themselves are compiled in Rust.
        built = build_devices(self.base_sizes, nf=self.nf, corner=None, topo=topo,
                              model_types=self.model_types,
                              device_kwargs=self.device_kwargs)
        self._mult = {name: int(getattr(built[name], "mult", 1))
                      for name in self.device_names}

        drive = getattr(topo, "input_drives", {}) or {}
        node_drives = getattr(topo, "ac_drives", {}) or {}
        ac_devs = {name: (d, g, s)
                   for name, d, g, s in self.plan.ac_devices(drive=drive,
                                                             node_drives=node_drives)}

        dc_devices = []
        devices = []
        for dp in self.plan.devices:
            dev = built[dp.name]
            dc_devices.append((
                [_dc_term(dp.d), _dc_term(dp.g), _dc_term(dp.s),
                 (2, 0, float(dev.vb))],
                [-1 if dp.di is None else int(dp.di),
                 -1 if dp.gi is None else int(dp.gi),
                 -1 if dp.si is None else int(dp.si), -1],
            ))
            acd, acg, acs = ac_devs[dp.name]
            devices.append((
                str(dev.POLARITY), float(dev.vb), float(dev.temperature),
                float(dev.temperature) - 273.15,
                _ac_term(acd), _ac_term(acg), _ac_term(acs),
                _reference_width_um(dev),
            ))

        dc_tol = float(getattr(topo, "dc_tol", None) or DC_FALLBACK_TOL)
        rail_span = max((abs(float(v)) for v in bias.values()), default=1.0)
        outs = topo.outputs
        latch_nodes = ((int(self.plan.idx[outs[0]]), int(self.plan.idx[outs[1]]))
                       if len(outs) == 2 else None)

        template = {
            "pdk": self.pdk,
            "root": silicon_pdk_root(self.pdk),
            "circuit": circuitopt_core.OtftTransientProblem(
                passive_problem_spec(self.plan)),
            "n_aug": int(self.plan.n_aug),
            "dc_devices": dc_devices,
            "devices": devices,
            "ac_caps": [(_ac_term(a), _ac_term(b), float(v))
                        for a, b, v in self.plan.ac_capacitors(node_drives)],
            "ac_resistors": [(_ac_term(a), _ac_term(b), float(g))
                             for _n, a, b, _r, g in self.plan.ac_resistors(node_drives)],
            "ac_vccs": [(_ac_term(p), _ac_term(q), _ac_term(cp), _ac_term(cn), float(gm))
                        for p, q, cp, cn, gm in self.plan.ac_vccs(node_drives)],
            "ac_vsources": [(_ac_term(p), _ac_term(q), int(bi),
                             float(complex(e).real), float(complex(e).imag))
                            for p, q, bi, e in self.plan.ac_vsources(node_drives)],
            "ac_vcvs": [(_ac_term(p), _ac_term(q), _ac_term(cp), _ac_term(cn),
                         int(bi), float(mu))
                        for p, q, cp, cn, bi, mu in self.plan.ac_vcvs(node_drives)],
            "ac_cccs": [(_ac_term(p), _ac_term(q), int(cb), float(beta))
                        for p, q, cb, beta in self.plan.ac_cccs(node_drives)],
            "ac_ccvs": [(_ac_term(p), _ac_term(q), int(cb), int(bi), float(gamma))
                        for p, q, cb, bi, gamma in self.plan.ac_ccvs(node_drives)],
            "resistor_noise": [(_ac_term(a), _ac_term(b), float(r))
                               for _n, a, b, r, _g in self.plan.ac_resistors()],
            "output_weights": [(int(self.plan.idx[node]), float(w))
                               for node, w in self.plan.output_weights.items()],
            "sense": [float(v) for v in self.plan.output_sense(dtype=float)],
            "vin_norm": float(_vin_norm(drive, node_drives)),
            "freqs": self.freqs,
            "band": [self.band[0], self.band[1]],
            "dc_guesses": [[float(v) for v in g]
                           for g in topo.dc_guess_vectors(bias)],
            "dc_options": [100.0, min(dc_tol, 1e-10),
                           max(0.25, rail_span / 4.0), 1e-12],
            "latch_nodes": latch_nodes,
        }
        self.core = circuitopt_core.CompiledCampaign(
            {"family": "silicon_bsim4", "template": template})

    def seed_vector(self, dc_op: Mapping[str, float]) -> list[float]:
        """Solved-order DC seed vector from a ``{node: V}`` operating point."""
        return [float(dc_op.get(node, self.default_guess)) for node in self.solved]

    def candidate(self, sizes: Mapping[str, tuple[float, float]], corner: str,
                  mismatch: Mapping[str, float] | None = None, nf=None,
                  seed=None, trust_seed_as_op: bool = False) -> dict:
        """One marshalled candidate. ``mismatch`` maps device -> delvto volts."""
        nf = self.nf if nf is None else nf
        mismatch = mismatch or {}
        devices = []
        for name in self.device_names:
            w, l = sizes[name]
            devices.append([
                float(w), float(l), float(dev_nf(nf, name)),
                float(self._mult[name]), float(mismatch.get(name, 0.0)),
            ])
        out = {"devices": devices, "corner": str(corner).lower(),
               "trust_seed_as_op": bool(trust_seed_as_op)}
        if seed is not None:
            out["seed"] = (self.seed_vector(seed) if isinstance(seed, Mapping)
                           else [float(v) for v in seed])
        return out

    def evaluate_batch(self, candidates: Sequence[dict], workers: int = 1,
                       analyses: Sequence[str] = ("dc", "ac", "noise")) -> list[dict]:
        """Run the compiled batch; results are candidate-index ordered."""
        return self.core.evaluate_batch(list(candidates), workers, list(analyses))
