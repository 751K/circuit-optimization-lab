"""Python bridge to ``circuitopt_core.CompiledCampaign`` (rewrite R5-C/R5-D).

Marshals a generic BSIM ``CircuitSpec`` or the legacy AFE OTFT topology plus an
analysis plan into the Rust compiled campaign, then expands a candidate matrix
into the flat, index-ordered list the executor consumes. Each candidate may
override any named topology bias; symbolic rails remain candidate-input slots
through the Rust DC solve. Random mismatch draws and candidate-specific DC
guesses are prepared **up front** so the detached Rust batch never calls back
into Python. BSIM campaigns may additionally retain prepared DC/linearization/
AC state and resume noise for an index subset.

The shared :mod:`circuitopt._campaign_sweep` dispatcher wires this bridge into
eligible dataset, corner, mismatch, and benchmark batches. Scalar Python
analysis paths remain the compatibility reference and fallback.
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


def _bias_schema(topo, bias):
    """Return stable candidate-bias names/defaults and their slot lookup."""
    names = tuple(str(name) for name in bias)
    defaults = [float(bias[name]) for name in names]
    if not all(np.isfinite(defaults)):
        raise ValueError("campaign bias defaults must be finite")
    slots = {name: index for index, name in enumerate(names)}
    for rail, reference in topo.rails.items():
        if isinstance(reference, str) and reference not in slots:
            raise ValueError(
                f"symbolic rail {rail!r} references missing bias {reference!r}")
    return names, defaults, slots


def _candidate_dc_term(topo, slots, node, token) -> tuple[int, int, float]:
    """Encode symbolic rails as candidate-input slots for the Rust DC solver."""
    if token[0] == TERM_SOLVED:
        return _dc_term(token)
    reference = topo.rails.get(node)
    if isinstance(reference, str):
        return (1, int(slots[reference]), 0.0)
    return _dc_term(token)


def _candidate_bias(base, override):
    """Merge and validate one named candidate-bias override."""
    if override is None:
        return None
    unknown = sorted(set(override) - set(base))
    if unknown:
        raise ValueError(f"unknown candidate bias key(s): {unknown}")
    merged = dict(base)
    merged.update({str(name): float(value) for name, value in override.items()})
    if not all(np.isfinite(tuple(merged.values()))):
        raise ValueError("candidate bias values must be finite")
    return merged


def _campaign_source_power(topo, plan, bias, row, bulk_metadata=None):
    """Reduce native terminal/branch currents into the scalar source-power contract."""
    from .run_contract import SimulationInvalid

    state = [float(value) for value in row["dc_op"]]
    if len(state) != plan.n_aug:
        raise SimulationInvalid(
            "invalid_result_shape",
            f"compiled campaign returned {len(state)} DC values; expected {plan.n_aug}",
            analysis="dc_power",
        )
    terminal_currents = row.get("terminal_currents") or ()
    if len(terminal_currents) != len(topo.devices):
        raise SimulationInvalid(
            "invalid_result_shape",
            "compiled campaign returned an incomplete terminal-current matrix",
            analysis="dc_power",
        )
    node_values = {name: state[index] for index, name in enumerate(plan.solved)}
    rail_values = {name: float(value) for name, value in topo.rail_values(bias).items()}
    delivered = {name: 0.0 for name in rail_values}

    def add(node, current):
        if node in delivered:
            delivered[node] += float(current)

    def node_v(node):
        return float(node_values[node]) if node in node_values else rail_values[node]

    for index, ((name, drain, gate, source), currents) in enumerate(
            zip(topo.devices, terminal_currents)):
        values = np.asarray(currents, dtype=float)
        if values.shape != (4,) or not np.all(np.isfinite(values)):
            raise SimulationInvalid(
                "non_finite_result",
                f"{name}: compiled campaign returned invalid terminal currents",
                analysis="dc_power",
            )
        add(drain, values[0])
        add(gate, values[1])
        add(source, values[2])
        if bulk_metadata is None:
            continue
        meta = bulk_metadata[index]
        matches = [
            rail for rail, voltage in rail_values.items()
            if np.isclose(voltage, meta["vb"], rtol=0.0, atol=1e-12)
        ]
        explicit = meta["explicit"]
        if explicit is not None:
            if explicit not in matches:
                raise SimulationInvalid(
                    "invalid_bulk_supply",
                    f"{name}: bulk rail {explicit!r} does not match vb={meta['vb']:g} V",
                    analysis="dc_power",
                )
            bulk_rail = explicit
        elif meta["source"] in matches:
            bulk_rail = meta["source"]
        else:
            preferred = (
                ("GND", "VSS", "VSSA", "VSSD")
                if meta["polarity"] == "nmos"
                else ("VDD", "VCC", "VDDA", "VDDD")
            )
            named = [rail for rail in preferred if rail in matches]
            bulk_rail = named[0] if len(named) == 1 else (
                matches[0] if len(matches) == 1 else None)
        if bulk_rail is None and values[3] != 0.0:
            raise SimulationInvalid(
                "unresolved_bulk_supply",
                f"{name}: non-zero bulk current cannot be assigned to one explicit rail",
                analysis="dc_power",
            )
        if bulk_rail is not None:
            add(bulk_rail, values[3])

    for _name, a, b, resistance in topo.resistors:
        current = (node_v(a) - node_v(b)) / float(resistance)
        add(a, current)
        add(b, -current)
    for _name, p, q, current in topo.isources:
        add(p, current)
        add(q, -current)
    for _name, p, q, cp, cn, gm in topo.vccs:
        current = float(gm) * (node_v(cp) - node_v(cn))
        add(p, -current)
        add(q, current)

    branch_currents = {}
    for item in (*plan.vsources, *plan.vcvs, *plan.ccvs):
        branch_currents[item.name] = state[item.bi]
        add(item.p_node, state[item.bi])
        add(item.q_node, -state[item.bi])
    for item in plan.cccs:
        current = float(item.beta) * branch_currents[item.ctrl_name]
        add(item.p_node, -current)
        add(item.q_node, current)

    per_source = {
        name: rail_values[name] * current for name, current in delivered.items()
    }
    if not all(np.isfinite(tuple(per_source.values()))):
        raise SimulationInvalid(
            "non_finite_result", "compiled campaign source power is non-finite",
            analysis="dc_power",
        )
    return {
        "per_source_w": per_source,
        "source_currents_a": delivered,
        "source_voltages_v": rail_values,
        "total_w": float(sum(per_source.values())),
    }


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
        self.bias_names, bias_defaults, bias_slots = _bias_schema(topo, self.bias)
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
                _candidate_dc_term(topo, bias_slots, dp.d_node, dp.d),
                _candidate_dc_term(topo, bias_slots, dp.g_node, dp.g),
                _candidate_dc_term(topo, bias_slots, dp.s_node, dp.s),
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
            "bias_names": list(self.bias_names),
            "bias_defaults": bias_defaults,
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
                  nf=None, seed=None, trust_seed_as_op: bool = False,
                  bias: Mapping[str, float] | None = None) -> dict:
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
        candidate_bias = _candidate_bias(self.bias, bias)
        if candidate_bias is not None:
            out["bias"] = candidate_bias
            out["dc_guesses"] = [
                [float(value) for value in guess]
                for guess in self.topo.dc_guess_vectors(candidate_bias)
            ]
        if seed is not None:
            out["seed"] = (self.seed_vector(seed) if isinstance(seed, Mapping)
                           else [float(v) for v in seed])
        return out

    def evaluate_batch(self, candidates: Sequence[dict], workers: int = 1,
                       analyses: Sequence[str] = ("dc", "ac", "noise")) -> list[dict]:
        """Run the compiled batch; results are candidate-index ordered."""
        return self.core.evaluate_batch(list(candidates), workers, list(analyses))

    def reduce_result(self, row, sizes, bias, nf=None):
        """Return exact topology-level area and source power without device rebuilds."""
        c1, c2, c3, c4, kv, kh = 37.5, 50.0, 35.0, 35.0, 1.0, 1.0
        area = 0.0
        for name in self.device_names:
            width, length = sizes[name]
            fingers = float(dev_nf(nf, name))
            fw = float(width) / fingers
            osc_o1 = c2 - c3 + c4
            edge_x_expr = (
                (10.0 + float(length)) * fingers + 10.0
                + 2 * osc_o1 + kv * c2
            )
            edge_ox = 2 * c3 + 2 * c1 * np.ceil(
                np.ceil(edge_x_expr / c1) / 2
            )
            edge_oy = 2 * c3 + 2 * c1 * np.ceil(
                np.ceil((fw + 2 * osc_o1 + (kh - 1) * c2) / c1) / 2
            )
            area += (edge_ox + 2 * c1) * (edge_oy + 2 * c1)
        return {
            "area": float(area),
            "source_power": _campaign_source_power(
                self.topo, self.plan, bias, row),
        }


def _silicon_pdk_of(model_types: Mapping[str, str]) -> str:
    """The single silicon PDK family a circuit's model types belong to."""
    families = {str(m).split(".", 1)[0] for m in model_types.values()}
    if len(families) != 1:
        raise ValueError(f"expected one silicon PDK family, got {sorted(families)}")
    family = families.pop()
    if family not in {"sky130", "freepdk45", "tsmc28hpcp"}:
        raise ValueError(f"unsupported native silicon PDK family {family!r}")
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


class BsimCampaign:
    """Compiled BSIM4 campaign over a generic circuit spec + analysis plan.

    ``spec`` is a loaded circuit (:func:`circuit_loader.load_circuit_json`).
    Device count and connectivity are unrestricted. The template captures the
    topology's MOS terminals, resistors, capacitors, independent sources,
    controlled sources, output projection, per-device process binding, and
    analysis plan. Candidates carry geometry, bias, process corner, NF, and
    optional ``delvto`` mismatch volts.
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
        self.bias_names, bias_defaults, bias_slots = _bias_schema(topo, self.bias)
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
        self._bulk_metadata = [
            {
                "vb": float(built[dp.name].vb),
                "polarity": str(built[dp.name].POLARITY).lower(),
                "source": dp.s_node,
                "explicit": (getattr(built[dp.name], "binding", {}) or {}).get(
                    "bulk_rail"),
            }
            for dp in self.plan.devices
        ]
        # The corner a ``corner=None`` scalar build resolves to (device-class
        # default: freepdk45 ``nom``, sky130/tsmc28 ``tt``) — what a nominal
        # size-sweep candidate must stamp so the campaign matches the scalar path.
        self.nominal_corner = str(getattr(built[self.device_names[0]], "corner", "tt"))

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
                [_candidate_dc_term(topo, bias_slots, dp.d_node, dp.d),
                 _candidate_dc_term(topo, bias_slots, dp.g_node, dp.g),
                 _candidate_dc_term(topo, bias_slots, dp.s_node, dp.s),
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
                passive_problem_spec(
                    self.plan,
                    term_record=lambda node, token: _candidate_dc_term(
                        topo, bias_slots, node, token),
                )),
            "n_aug": int(self.plan.n_aug),
            "bias_names": list(self.bias_names),
            "bias_defaults": bias_defaults,
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
                  seed=None, trust_seed_as_op: bool = False,
                  bias: Mapping[str, float] | None = None) -> dict:
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
        candidate_bias = _candidate_bias(self.bias, bias)
        if candidate_bias is not None:
            out["bias"] = candidate_bias
            out["dc_guesses"] = [
                [float(value) for value in guess]
                for guess in self.topo.dc_guess_vectors(candidate_bias)
            ]
        if seed is not None:
            out["seed"] = (self.seed_vector(seed) if isinstance(seed, Mapping)
                           else [float(v) for v in seed])
        return out

    def evaluate_batch(self, candidates: Sequence[dict], workers: int = 1,
                       analyses: Sequence[str] = ("dc", "ac", "noise")) -> list[dict]:
        """Run the compiled batch; results are candidate-index ordered."""
        return self.core.evaluate_batch(list(candidates), workers, list(analyses))

    def prepare_batch(self, candidates: Sequence[dict], workers: int = 1):
        """Retain DC, device linearization, MNA, and forward-AC state."""
        return PreparedBsimCampaign(
            self.core.prepare_batch(list(candidates), int(workers)))

    def reduce_result(self, row, sizes, bias, nf=None):
        """Return exact topology-level area and source power without device rebuilds."""
        area = sum(
            float(sizes[name][0]) * float(sizes[name][1]) * self._mult[name]
            for name in self.device_names
        )
        return {
            "area": float(area),
            "source_power": _campaign_source_power(
                self.topo, self.plan, bias, row, self._bulk_metadata),
        }


class PreparedBsimCampaign:
    """Opaque reusable state returned by the native BSIM campaign.

    The Rust object owns only numeric state, never native C handles. A noise
    continuation therefore re-biases fresh handles at the retained operating
    point while reusing DC, the device linearization, assembled MNA system, and
    the forward AC response.
    """

    def __init__(self, core):
        self.core = core

    @property
    def count(self) -> int:
        return int(self.core.count)

    @property
    def prepared_count(self) -> int:
        return int(self.core.prepared_count)

    @property
    def profile(self) -> dict:
        return dict(self.core.profile)

    def evaluate_batch(self, indices=None, workers: int = 1,
                       analyses: Sequence[str] = ("dc", "ac")) -> list[dict]:
        """Evaluate retained states in ``indices`` order without rerunning DC/AC."""
        selected = None if indices is None else [int(index) for index in indices]
        return self.core.evaluate_batch(
            selected, int(workers), list(analyses))


# Compatibility name retained for existing callers. The implementation is not
# tied to a 5T OTA or a particular silicon PDK.
SiliconCampaign = BsimCampaign
