"""Full-circuit model-card transient simulation through ngspice.

The fast :mod:`circuitopt.ngspice_device` adapters store DC and small-signal
characterisation grids, not the four-terminal BSIM charge state required by a
large-signal transient.  This backend keeps ngspice as the silicon-process
oracle: it renders the complete :class:`~circuitopt.topology.Topology`, runs
``.tran`` with the original model cards, and maps the resulting waveforms back
to circuitopt's standard transient result shape.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from .device_factory import apply_silicon_corner
from .ngspice_char import _run_ngspice
from .ngspice_render import (
    _current_input, _element, _ident, _pwl_lines, build_node_map,
    nodeset_line, render_controlled, render_devices, render_passives,
    render_rail_sources, resolve_common_temperature, resolve_ngspice_preamble)


@dataclass(frozen=True)
class RenderedTransient:
    netlist: str
    node_names: tuple[str, ...]
    branch_names: tuple[str, ...]
    op_vectors: tuple[tuple[str, str], ...] = ()
    command_args: tuple[str, ...] = ()
    process: str = "freepdk45"


def render_ngspice_transient_netlist(
    sizes: Mapping[str, tuple[float, float]],
    bias: Mapping[str, float],
    tgrid: Sequence[float],
    *,
    topo,
    output_path: str,
    nf: int | Mapping[str, int] | None = None,
    V0=None,
    inputs: Mapping[str, Any] | None = None,
    node_inputs: Mapping[str, str] | None = None,
    current_inputs: Sequence[Any] | None = None,
    corner: str | Mapping[str, Any] | None = None,
    model_types: Mapping[str, str] | None = None,
    device_kwargs: Mapping[str, Mapping[str, Any]] | None = None,
    integration_method: str = "be",
    max_step: float | None = None,
    mismatch: Mapping[str, float] | None = None,
    extra_options: Mapping[str, Any] | None = None,
    op_devices: Sequence[str] | None = None,
    uic: bool = False,
) -> RenderedTransient:
    """Render a complete model-card-backed ``.tran`` deck and its column map.

    ``mismatch`` maps a device name to a threshold-voltage offset in volts, emitted
    as the BSIM4 instance parameter ``delvto`` on that transistor's M-line. This is
    the injection hook for per-instance Vth mismatch Monte-Carlo (see
    :mod:`circuitopt.sar_mc`): ``delvto`` shifts the flat-band/Vth of that one
    instance without touching the shared model card, so paired devices stay
    independent. ``None`` (or an all-zero map) renders the byte-identical nominal
    deck — zero offsets are skipped rather than emitted as ``delvto=0`` so a
    zero-sigma trial reproduces the nominal netlist exactly.
    """
    tgrid = np.asarray(tgrid, float)
    if tgrid.ndim != 1 or len(tgrid) < 2 or tgrid[0] < 0.0 or np.any(np.diff(tgrid) <= 0.0):
        raise ValueError("tgrid must be one-dimensional, non-negative, and strictly increasing")
    inputs = {str(k): np.asarray(v, float) for k, v in (inputs or {}).items()}
    node_inputs = {str(k): str(v) for k, v in (node_inputs or {}).items()}
    current_inputs = tuple(current_inputs or ())
    model_types = dict(model_types or {})
    device_kwargs, solver_corner = apply_silicon_corner(
        model_types, device_kwargs, corner)
    if solver_corner not in (None, {}):
        raise ValueError(
            f"ngspice transient requires a supported silicon corner, got {corner!r}")
    device_kwargs = {k: dict(v) for k, v in (device_kwargs or {}).items()}
    mismatch = {str(k): float(v) for k, v in (mismatch or {}).items()}

    device_names = {name for name, *_ in topo.devices}
    op_devices = tuple(str(name) for name in (op_devices or ()))
    unknown_op = sorted(set(op_devices) - device_names)
    if unknown_op:
        raise ValueError(
            f"op_devices references unknown devices: {', '.join(unknown_op)}")
    unknown_mismatch = sorted(set(mismatch) - device_names)
    if unknown_mismatch:
        raise ValueError(
            f"mismatch references unknown devices: {', '.join(unknown_mismatch)}")

    # Per-polarity corner card resolution (nom/tt/ss/ff + mixed sf/fs) and the single
    # common circuit temperature — shared with the .ac/.noise/.op oracles.
    adapter, _corner, preamble = resolve_ngspice_preamble(
        model_types, device_kwargs, device_names)
    temp_c = resolve_common_temperature(device_kwargs, device_names)

    node_map, node = build_node_map(topo, bias, node_inputs)

    process = adapter.name if adapter is not None else "FreePDK45"
    lines = [f"* circuitopt {process} full-charge transient"]
    lines.extend(preamble)
    method = str(integration_method).lower()
    if method not in {"be", "gear2"}:
        raise ValueError(f"integration_method must be 'be' or 'gear2', got {method!r}")
    lines.append(f".options temp={temp_c:g} method=gear maxord={1 if method == 'be' else 2}")
    if extra_options:
        # e.g. {"reltol": 1e-5, "vntol": 1e-9} — tighter solver tolerances for
        # sub-0.1% settling measurements (ngspice's default reltol=1e-3 leaves a
        # ~100 uV numerical band on ~0.5 V nodes). None/{} renders byte-identically.
        opts = " ".join(f"{k}={v:g}" if isinstance(v, (int, float)) else f"{k}={v}"
                        for k, v in extra_options.items())
        lines.append(f".options {opts}")

    rail_lines, branch_vectors = render_rail_sources(topo, bias, node_inputs, node)
    lines.extend(rail_lines)

    def waveform(key: str):
        if key not in inputs:
            raise ValueError(f"transient source references missing input waveform {key!r}")
        values = inputs[key]
        if values.shape != tgrid.shape:
            raise ValueError(f"input waveform {key!r} length differs from tgrid")
        return values

    for driven_node, key in node_inputs.items():
        if driven_node not in node_map:
            raise ValueError(f"node_inputs references unknown node {driven_node!r}")
        source = _element("V", "node_" + driven_node)
        lines.extend(_pwl_lines(source, node_map[driven_node], "0", tgrid, waveform(key)))
        branch_vectors.append((f"node:{driven_node}", source))

    gate_nodes = {}
    for name, *_ in topo.devices:
        key = topo.transient_inputs.get(name)
        if key is None:
            continue
        gate_nodes[name] = "n_gate_" + _ident(name)
        source = _element("V", "gate_" + name)
        lines.extend(_pwl_lines(source, gate_nodes[name], "0", tgrid, waveform(str(key))))
        branch_vectors.append((f"gate:{name}", source))

    dev_lines, dev_branches = render_devices(
        topo, sizes, bias, node_inputs, node, nf=nf, model_types=model_types,
        device_kwargs=device_kwargs, mismatch=mismatch, gate_nodes=gate_nodes,
        adapter=adapter)
    lines.extend(dev_lines)
    branch_vectors.extend(dev_branches)

    lines.extend(render_passives(topo, node))

    ctrl_lines, ctrl_branches, _names = render_controlled(
        topo, node, tgrid=tgrid, waveform_fn=waveform)
    lines.extend(ctrl_lines)
    branch_vectors.extend(ctrl_branches)

    for pos, item in enumerate(current_inputs):
        p, q, key = _current_input(item)
        source = _element("I", f"wave_{pos}")
        lines.extend(_pwl_lines(source, node(p), node(q), tgrid, waveform(key)))

    initial = nodeset_line(topo, node_map, V0)
    if uic:
        if initial is None:
            raise ValueError("uic=True requires V0 for every solved node")
        lines.append(initial.replace(".nodeset ", ".ic ", 1))
    elif initial is not None:
        lines.append(initial)

    print_step = float(np.min(np.diff(tgrid)))
    tmax = print_step if max_step is None else float(max_step)
    if tmax <= 0.0:
        raise ValueError("max_step must be positive")
    vectors = [f"v({node_map[name]})" for name in topo.solved]
    vectors.extend(f"i({source})" for _, source in branch_vectors)
    op_vectors = []
    for name in op_devices:
        elem = _element("X" if adapter is not None else "M", name).lower()
        for variable in ("vds", "vgs", "vdsat", "id", "gm", "gds"):
            vector = (adapter.op_vector(elem, variable) if adapter is not None
                      else f"@{elem}[{variable}]")
            vectors.append(vector)
            op_vectors.append((name, variable))
    lines.extend([
        ".control",
        "set wr_singlescale",
        "set wr_vecnames",
        f"tran {print_step:.17g} {tgrid[-1]:.17g} 0 {tmax:.17g}"
        + (" uic" if uic else ""),
        f"wrdata {output_path} " + " ".join(vectors),
        ".endc",
        ".end",
    ])
    return RenderedTransient(
        netlist="\n".join(lines) + "\n",
        node_names=tuple(topo.solved),
        branch_names=tuple(name for name, _ in branch_vectors),
        op_vectors=tuple(op_vectors),
        command_args=adapter.command_args if adapter is not None else (),
        process=process,
    )


def render_freepdk45_transient_netlist(*args, **kwargs) -> RenderedTransient:
    """Compatibility name for the generic ngspice transient renderer."""
    return render_ngspice_transient_netlist(*args, **kwargs)


def _transient_result(rendered: RenderedTransient, raw, requested_t, *, topo, uic):
    """Map one ``wrdata`` table back to the standard transient result dict.

    Shared verbatim by :func:`transient_ngspice` (one analysis per process) and
    :func:`transient_ngspice_chain` (several analyses per process) so the two
    backends cannot drift: column-count check, time sort, duplicate-breakpoint
    dedup (keep the FINAL sample), tgrid coverage check, interpolation onto the
    requested grid, and the result/alias assembly are one code path."""
    expected_cols = (1 + len(rendered.node_names) + len(rendered.branch_names)
                     + len(rendered.op_vectors))
    if raw.shape[1] != expected_cols:
        raise RuntimeError(
            f"ngspice transient returned {raw.shape[1]} columns, expected {expected_cols}")
    order = np.argsort(raw[:, 0], kind="stable")
    raw = raw[order]
    # Keep the final sample at duplicate breakpoints.
    _, reverse_pos = np.unique(raw[::-1, 0], return_index=True)
    keep = np.sort(len(raw) - 1 - reverse_pos)
    raw = raw[keep]
    sim_t = raw[:, 0]
    starts_too_late = requested_t[0] < sim_t[0] - 1e-18 and not uic
    if starts_too_late or requested_t[-1] > sim_t[-1] + 1e-15:
        raise RuntimeError(
            f"ngspice transient range [{sim_t[0]}, {sim_t[-1]}] does not cover tgrid")

    pos = 1
    nodes = {}
    for name in rendered.node_names:
        nodes[name] = np.interp(requested_t, sim_t, raw[:, pos])
        pos += 1
    branch_currents = {}
    for name in rendered.branch_names:
        branch_currents[name] = np.interp(requested_t, sim_t, raw[:, pos])
        pos += 1
    device_op = {}
    for name, variable in rendered.op_vectors:
        device_op.setdefault(name, {})[variable] = np.interp(
            requested_t, sim_t, raw[:, pos])
        pos += 1
    if topo.outputs:
        output = sum(nodes[name] * weight for name, weight in topo.output_weights().items())
    else:
        output = nodes[topo.solved[0]]
    result = {
        "t": requested_t,
        "nodes": nodes,
        "output": output,
        "vout": output,
        "branch_currents": branch_currents,
        "device_op": device_op,
        "device_op_final": {
            name: {variable: float(values[-1]) for variable, values in variables.items()}
            for name, variables in device_op.items()
        },
        "nfail": 0,
        "backend": "ngspice",
        "ngspice_transient": True,
        "process": rendered.process,
    }
    for alias, node_name in topo.aliases.items():
        if node_name in nodes:
            result[alias] = nodes[node_name]
    if "VOP" in nodes:
        result["vop"] = nodes["VOP"]
    if "VON" in nodes:
        result["von"] = nodes["VON"]
    return result


def transient_ngspice(
    sizes, bias, tgrid, *, topo, nf=None, V0=None, inputs=None,
    node_inputs=None, current_inputs=None, corner=None, model_types=None,
    device_kwargs=None, integration_method="be", max_step=None,
    mismatch=None, extra_options=None, timeout: float = 300.0,
    op_devices=None, uic: bool = False,
):
    """Run a model-card full-charge transient and return circuitopt waveforms.

    ``mismatch`` is threaded straight to
    :func:`render_freepdk45_transient_netlist` as per-device ``delvto`` offsets.
    When ``op_devices`` is supplied, the returned ``device_op`` mapping contains
    time-domain ``vds``, ``vgs``, ``vdsat``, ``id``, ``gm``, and ``gds`` vectors;
    ``device_op_final`` contains the corresponding final-sample scalars.
    """
    requested_t = np.asarray(tgrid, float)
    with tempfile.TemporaryDirectory(prefix="circuitopt-ng-tran-") as td:
        output_path = os.path.join(td, "waveforms.dat")
        deck_path = os.path.join(td, "deck.cir")
        rendered = render_ngspice_transient_netlist(
            sizes, bias, requested_t, topo=topo, output_path=output_path,
            nf=nf, V0=V0, inputs=inputs, node_inputs=node_inputs,
            current_inputs=current_inputs, corner=corner, model_types=model_types,
            device_kwargs=device_kwargs, integration_method=integration_method,
            max_step=max_step, mismatch=mismatch, extra_options=extra_options,
            op_devices=op_devices, uic=uic,
        )
        with open(deck_path, "w", encoding="ascii") as fh:
            fh.write(rendered.netlist)
        _run_ngspice(deck_path, output_path, timeout=timeout,
                     what=f"{rendered.process} full-circuit transient",
                     extra_args=rendered.command_args)
        raw = np.loadtxt(output_path, skiprows=1, ndmin=2)

    return _transient_result(rendered, raw, requested_t, topo=topo, uic=uic)


# ── chained multi-case transient (one ngspice process, alter-pwl between runs) ──
# ngspice-46's interactive command parser silently corrupts a control line past
# ~1000 words (measured on this binary: a 984-word `alter @v[pwl] = [ ... ]` is
# applied exactly; a 1004-word one leaves a WRONG waveform with no error), so
# chained decks refuse to emit anything near that cliff and fall back to the
# exact constant-run compression below, or fail loudly.
_NGSPICE_CMD_MAX_WORDS = 800


def _pwl_blocks(lines):
    """Locate every rendered PWL source block: ``{elem: (start, stop, tokens)}``.

    A block is the ``<name> <p> <q> PWL(`` header plus its ``+ ...`` continuation
    lines (:func:`circuitopt.ngspice_render._pwl_lines` layout); ``tokens`` are the
    time/value literals exactly as rendered (``%.17g`` strings — token equality is
    float equality because ``%.17g`` round-trips doubles), ``lines[start:stop]``
    is the block's line span. Elements are keyed lower-case (SPICE is
    case-insensitive)."""
    blocks = {}
    pos = 0
    while pos < len(lines):
        line = lines[pos]
        if not line.startswith("+") and line.rstrip().endswith("PWL("):
            elem = line.split()[0].lower()
            stop = pos + 1
            tokens = []
            while stop < len(lines) and lines[stop].startswith("+"):
                chunk = lines[stop][1:].strip()
                stop += 1
                if chunk.endswith(")"):
                    tokens.extend(chunk[:-1].split())
                    break
                tokens.extend(chunk.split())
            if elem in blocks:
                raise ValueError(f"duplicate PWL source {elem!r} in rendered deck")
            blocks[elem] = (pos, stop, tuple(tokens))
            pos = stop
        else:
            pos += 1
    return blocks


def _pwl_skeleton(lines, blocks):
    """The netlist with every PWL block collapsed to a ``<pwl elem>`` marker.

    Chained cases must be byte-identical here — any difference outside the PWL
    sources means the cases do not share a deck and cannot be chained."""
    spans = {start: (stop, elem) for elem, (start, stop, _tokens) in blocks.items()}
    out = []
    pos = 0
    while pos < len(lines):
        if pos in spans:
            stop, elem = spans[pos]
            out.append(f"<pwl {elem}>")
            pos = stop
        else:
            out.append(lines[pos])
            pos += 1
    return out


def _compress_pwl_tokens(tokens):
    """Drop interior PWL points inside constant runs — an EXACT compression.

    A point is dropped only when its value literal equals both neighbours'
    (token equality == float equality under ``%.17g``); linear interpolation
    between the kept run endpoints reproduces every dropped sample bit-exactly,
    so the compressed source is the IDENTICAL piecewise-linear function. Sloped
    segments are never touched (collinear-float reconstruction is not bit-safe).
    Kept points keep their original literals — no re-formatting."""
    times, values = tokens[0::2], tokens[1::2]
    keep = [0]
    keep.extend(pos for pos in range(1, len(times) - 1)
                if not (values[pos] == values[pos - 1] == values[pos + 1]))
    keep.append(len(times) - 1)
    out = []
    for pos in keep:
        out.extend((times[pos], values[pos]))
    return tuple(out)


def _alter_pwl_line(elem, tokens):
    """``alter @<elem>[pwl] = [ t0 v0 t1 v1 ... ]`` under the command-word budget.

    Prefers the dense rendered literals (bit-identical to a fresh deck's PWL);
    compresses constant runs when the line would blow ngspice's silent ~1000-word
    command limit, and raises rather than ever emitting a line that limit would
    mangle."""
    words = len(tokens) + 4          # alter  @elem[pwl]  =  [ ... ]
    if words > _NGSPICE_CMD_MAX_WORDS:
        tokens = _compress_pwl_tokens(tokens)
        words = len(tokens) + 4
    if words > _NGSPICE_CMD_MAX_WORDS:
        raise RuntimeError(
            f"chained transient: replacement PWL for {elem!r} needs {words} command "
            f"words even after exact compression; ngspice's command buffer silently "
            f"corrupts lines past ~1000 words. Run this case unchained "
            f"(CIRCUITOPT_NGSPICE_CHAIN=0) or coarsen the waveform grid.")
    return f"alter @{elem}[pwl] = [ " + " ".join(tokens) + " ]"


def transient_ngspice_chain(
    sizes, bias, tgrid, *, topo, cases, nf=None, V0=None,
    node_inputs=None, current_inputs=None, corner=None, model_types=None,
    device_kwargs=None, integration_method="be", max_step=None,
    mismatch=None, extra_options=None, timeout: "float | None" = None,
    op_devices=None, uic: bool = False,
):
    """Run SEVERAL same-topology transients in ONE ngspice process.

    ``cases`` is a sequence of mappings, each carrying (exactly) the per-case
    ``{"inputs": {...}}`` waveform mapping; every other kwarg is shared. This is
    the S4 speed lever for foundry decks whose macro expansion dominates process
    startup (~2.9 s per TSMC28 macro instance at parse time): the case-0 deck is
    parsed once, then each subsequent case re-drives ONLY the PWL sources whose
    waveforms changed via ``alter @src[pwl] = [...]`` (no re-parse, no macro
    re-expansion) and runs a fresh ``tran``. The circuit-level ``.nodeset`` stays
    in force for every ``tran``'s own DC solve, so each case starts from the same
    seeded operating point a standalone process would find.

    Safety: every case is fully rendered and validated; the rendered decks must
    be byte-identical outside their PWL source blocks (anything else differing
    raises ``ValueError``), and the ``alter`` replacement literals are the exact
    ``%.17g`` strings a fresh deck would carry — measured bit-identical waveforms
    against per-process runs. Replacements that would exceed ngspice's silent
    command-word limit are exactly compressed (constant runs only) or refused.

    Returns a list of per-case dicts, each exactly what :func:`transient_ngspice`
    returns for that case (same keys, same post-processing). ``timeout`` is for
    the whole chained process; ``None`` scales the single-run default to
    ``300 s * len(cases)``.
    """
    cases = list(cases)
    if not cases:
        raise ValueError("transient_ngspice_chain requires at least one case")
    for pos, case in enumerate(cases):
        if not isinstance(case, Mapping) or "inputs" not in case:
            raise ValueError(f"chained case {pos} must be a mapping with an 'inputs' key")
        extra_keys = set(case) - {"inputs"}
        if extra_keys:
            raise ValueError(
                f"chained case {pos} carries non-chainable keys {sorted(extra_keys)}; "
                "only the input waveforms may vary along a chain")
    requested_t = np.asarray(tgrid, float)
    if timeout is None:
        timeout = 300.0 * len(cases)

    with tempfile.TemporaryDirectory(prefix="circuitopt-ng-tran-chain-") as td:
        out_paths = [os.path.join(td, f"waveforms_{pos}.dat")
                     for pos in range(len(cases))]
        deck_path = os.path.join(td, "deck.cir")
        rendered = [
            render_ngspice_transient_netlist(
                sizes, bias, requested_t, topo=topo, output_path=out_paths[0],
                nf=nf, V0=V0, inputs=case["inputs"], node_inputs=node_inputs,
                current_inputs=current_inputs, corner=corner, model_types=model_types,
                device_kwargs=device_kwargs, integration_method=integration_method,
                max_step=max_step, mismatch=mismatch, extra_options=extra_options,
                op_devices=op_devices, uic=uic,
            )
            for case in cases
        ]
        base = rendered[0]
        line_sets = [r.netlist.splitlines() for r in rendered]
        blocks = [_pwl_blocks(lines) for lines in line_sets]
        skeleton0 = _pwl_skeleton(line_sets[0], blocks[0])
        for pos in range(1, len(cases)):
            if (set(blocks[pos]) != set(blocks[0])
                    or _pwl_skeleton(line_sets[pos], blocks[pos]) != skeleton0):
                raise ValueError(
                    "chained transient cases must render byte-identical decks outside "
                    f"their PWL input sources, but case {pos} differs from case 0")

        # Splice the chained control block onto the case-0 network: reuse the
        # rendered `tran` command and wrdata vector list verbatim, one output
        # file per case, altering only the PWL sources that changed.
        idx_control = line_sets[0].index(".control")
        tran_line = next(line for line in line_sets[0][idx_control:]
                         if line.startswith("tran "))
        wr_line = next(line for line in line_sets[0][idx_control:]
                       if line.startswith("wrdata "))
        wr_prefix = f"wrdata {out_paths[0]} "
        if not wr_line.startswith(wr_prefix):
            raise RuntimeError("chained transient could not locate the wrdata vector list")
        vec_str = wr_line[len(wr_prefix):]
        control = [".control", "set wr_singlescale", "set wr_vecnames",
                   tran_line, f"wrdata {out_paths[0]} {vec_str}"]
        for pos in range(1, len(cases)):
            for elem in sorted(blocks[pos]):
                if blocks[pos][elem][2] != blocks[pos - 1][elem][2]:
                    control.append(_alter_pwl_line(elem, blocks[pos][elem][2]))
            control.append(tran_line)
            control.append(f"wrdata {out_paths[pos]} {vec_str}")
        control.extend([".endc", ".end"])

        with open(deck_path, "w", encoding="ascii") as fh:
            fh.write("\n".join(line_sets[0][:idx_control] + control) + "\n")
        _run_ngspice(
            deck_path, out_paths[-1], timeout=timeout,
            what=f"{base.process} full-circuit transient chain ({len(cases)} analyses)",
            extra_args=base.command_args)
        raws = []
        for path in out_paths:
            if not os.path.exists(path):
                raise RuntimeError(
                    f"{base.process} transient chain produced no output {path}")
            raws.append(np.loadtxt(path, skiprows=1, ndmin=2))

    return [_transient_result(base, raw, requested_t, topo=topo, uic=uic)
            for raw in raws]
