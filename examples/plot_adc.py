"""SAR ADC figures — static linearity, dynamic spectrum, one conversion, mismatch MC.

Four public functions, each returns the saved PNG :class:`~pathlib.Path` and takes
an ``out_dir`` like :mod:`examples.plot_bode` / :mod:`examples.plot_transient`:

* :func:`plot_sar_static`     — transfer staircase + DNL-per-code + INL-per-transition
  (from :func:`circuitopt.sar.run_sar_sweep` / ``adc.static_ramp_metrics``).
* :func:`plot_sar_spectrum`   — output-code power spectrum in dBc with SNDR/SNR/SFDR/ENOB
  (from :func:`circuitopt.sar.run_sar_signal` / ``adc.dynamic_metrics``).
* :func:`plot_sar_conversion` — logic-analyzer view of one conversion: sample controls,
  per-bit CDAC drives (+clk when present), CDAC top plates, comparator node with the
  physical decision instants marked (from :func:`circuitopt.sar.run_sar_conversion`).
* :func:`plot_sar_mc`         — histograms of max|DNL|/max|INL|/offset with yield and
  threshold lines (from :func:`circuitopt.sar_mc.sar_mismatch_mc`).

Design notes
------------
* **Palette** — brand-neutral validated hues from the ``dataviz`` skill: blue is the
  primary series, aqua the second, orange the highlight (harmonics / clk), red the
  limit/overflow. Text stays ink-colored; a colored mark beside it carries identity.
* **Spectrum frequency axis is linear** — a coherent short record has only tens of
  FFT bins and the harmonics land on integer multiples of the fundamental, which read
  naturally on a linear scale; a log axis would crowd the low bins and invent dynamic
  range the record does not have.
* **Mismatch inf handling** — a non-monotonic trial has undefined DNL/INL (recorded as
  ``inf`` upstream). Those are clipped into a visibly-labeled red overflow bin at the
  right edge rather than dropped, so the histogram stays honest about failures.

Run the 6-bit showcase (renders all four into ``results/``)::

    python -m examples.plot_adc --circuit examples/freepdk45_sar6.json
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping

import matplotlib
matplotlib.use("Agg")                                   # headless: write PNGs, no display
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path(__file__).resolve().parent.parent / "results"

# ── palette (dataviz reference instance, light surface) ───────────────────────
BLUE = "#2a78d6"        # series 1 / measured
AQUA = "#1baf7a"        # series 2
ORANGE = "#eb6834"      # highlight: harmonics, clk
VIOLET = "#4a3aa7"      # series 5
CRITICAL = "#d03b3b"    # limit lines / overflow / cleared
GOOD = "#0ca30c"        # kept
INK = "#0b0b0b"
MUTED = "#898781"       # axes / reference chrome
GRID = "#e1e0d9"

plt.rcParams.update({
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.alpha": 0.7,
    "font.size": 9,
})


def _save(fig, out_dir, filename, dpi=150) -> Path:
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    path = path / filename
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


# ── 1. static linearity ───────────────────────────────────────────────────────
def plot_sar_static(sweep_result: Mapping, out_dir: Path | str = RESULTS,
                    filename: str = "adc_sar_static.png",
                    note: str | None = None, dpi: int = 150) -> Path:
    """Transfer staircase + DNL-per-code + INL-per-transition.

    ``sweep_result`` is a :func:`circuitopt.sar.run_sar_sweep` payload (``vin``,
    ``codes``, ``metrics``, ``n_bits``, ``vref``). ``note`` is stamped on the figure
    (e.g. to flag a sub-sampled sweep).
    """
    vin = np.asarray(sweep_result["vin"], float)
    codes = np.asarray(sweep_result["codes"], np.int64)
    m = sweep_result["metrics"]
    n_bits = int(sweep_result["n_bits"])
    vref = float(sweep_result["vref"])
    levels = 1 << n_bits
    lsb = float(m["lsb"])
    dnl = np.asarray(m["dnl"], float)
    inl = np.asarray(m["inl"], float)
    missing_codes = np.asarray(m["missing_codes"], np.int64)

    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(9, 9))
    title = f"SAR static linearity — {n_bits}-bit (LSB = {lsb * 1e3:.3g} mV)"
    if note:
        title += f"\n{note}"
    ax0.set_title(title)

    # (a) transfer staircase + ideal line
    ax0.step(vin, codes, where="post", color=BLUE, lw=2, label="measured")
    ideal = np.clip(np.round(vin / vref * levels - 0.5), 0, levels - 1)
    ax0.plot(vin, ideal, color=MUTED, lw=1.2, ls="--", label="ideal")
    if missing_codes.size:
        ax0.scatter(np.full(missing_codes.size, vin[0]), missing_codes,
                    marker="_", s=60, color=CRITICAL, zorder=5,
                    label=f"missing code ×{missing_codes.size}")
    ax0.set_xlabel("Vin [V]"); ax0.set_ylabel("output code")
    ax0.set_xlim(0, vref); ax0.set_ylim(-0.5, levels - 0.5)
    ax0.legend(loc="upper left", framealpha=0.9)

    # (b) DNL per code
    code_idx = np.arange(levels)
    dnl_plot = np.where(np.isfinite(dnl), dnl, 0.0)
    ax1.bar(code_idx, dnl_plot, width=0.85, color=BLUE)
    for y in (-0.5, 0.5):
        ax1.axhline(y, color=CRITICAL, ls="--", lw=1)
    ax1.axhline(0, color=MUTED, lw=0.8)
    if missing_codes.size:
        ax1.scatter(missing_codes, np.zeros(missing_codes.size), marker="x",
                    s=40, color=CRITICAL, zorder=5, label="missing")
        ax1.legend(loc="lower right", framealpha=0.9)
    maxdnl = m.get("max_abs_dnl", np.nan)
    ax1.set_ylabel("DNL [LSB]")
    ax1.set_xlim(-0.5, levels - 0.5)
    ax1.annotate(f"max|DNL| = {maxdnl:.3f} LSB", xy=(0.01, 0.94),
                 xycoords="axes fraction", ha="left", va="top", color=INK,
                 fontsize=9, bbox=dict(boxstyle="round", fc="white", ec=GRID))

    # (c) INL per transition
    trans_idx = np.arange(1, levels)
    inl_plot = np.where(np.isfinite(inl), inl, np.nan)
    ax2.step(trans_idx, inl_plot, where="mid", color=AQUA, lw=2)
    ax2.scatter(trans_idx, inl_plot, s=10, color=AQUA)
    for y in (-0.5, 0.5):
        ax2.axhline(y, color=CRITICAL, ls="--", lw=1)
    ax2.axhline(0, color=MUTED, lw=0.8)
    maxinl = m.get("max_abs_inl", np.nan)
    ax2.set_xlabel("transition (code boundary)"); ax2.set_ylabel("INL [LSB]")
    ax2.set_xlim(0.5, levels - 0.5)
    ax2.annotate(f"max|INL| = {maxinl:.3f} LSB", xy=(0.01, 0.94),
                 xycoords="axes fraction", ha="left", va="top", color=INK,
                 fontsize=9, bbox=dict(boxstyle="round", fc="white", ec=GRID))

    fig.tight_layout()
    return _save(fig, out_dir, filename, dpi)


# ── 2. dynamic spectrum ────────────────────────────────────────────────────────
def plot_sar_spectrum(signal_result: Mapping, out_dir: Path | str = RESULTS,
                      filename: str = "adc_sar_spectrum.png", dpi: int = 150) -> Path:
    """Output-code power spectrum in dBc with the fundamental + harmonics 2..5 marked.

    ``signal_result`` is a :func:`circuitopt.sar.run_sar_signal` payload; its
    ``metrics`` come from ``adc.dynamic_metrics`` (``spectrum_power``, ``frequencies``,
    ``fundamental_bin``, ``harmonics``, ``sndr_db``/``snr_db``/``sfdr_db``/``enob``).
    Frequency axis is **linear** — see the module docstring for the rationale.
    """
    m = signal_result["metrics"]
    power = np.asarray(m["spectrum_power"], float)
    freqs = np.asarray(m["frequencies"], float)
    fund = int(m["fundamental_bin"])
    fs = float(m["sample_rate"])

    ref = power[fund] if power[fund] > 0 else np.max(power)
    tiny = np.finfo(float).tiny
    dbc = 10.0 * np.log10(np.maximum(power, tiny) / max(ref, tiny))
    floor = max(np.min(dbc[np.isfinite(dbc)]), -160.0)
    dbc = np.clip(dbc, floor, 5.0)
    fscale = 1e-6                                        # Hz -> MHz
    fx = freqs * fscale

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.vlines(fx, floor, dbc, color=BLUE, lw=1.4)
    ax.scatter(fx[fund], dbc[fund], s=55, color=BLUE, zorder=5, label="fundamental")
    ax.annotate("f₀", xy=(fx[fund], dbc[fund]), xytext=(0, 6),
                textcoords="offset points", ha="center", color=BLUE, fontsize=9)

    for order, center, _hp in m["harmonics"]:
        c = int(center)
        if not 0 < c < len(dbc):
            continue
        ax.scatter(fx[c], dbc[c], s=45, marker="v", color=ORANGE, zorder=5)
        ax.annotate(f"{order}f₀", xy=(fx[c], dbc[c]), xytext=(0, 6),
                    textcoords="offset points", ha="center", color=ORANGE, fontsize=8)
    # legend proxy for the harmonic marker
    ax.scatter([], [], marker="v", s=45, color=ORANGE, label="harmonics 2..5")

    ax.set_xlim(0, fx[-1]); ax.set_ylim(floor, 5.0)
    ax.set_xlabel("frequency [MHz]"); ax.set_ylabel("magnitude [dBc]")
    ax.set_title(f"SAR output spectrum — {int(m['n_samples'])}-pt coherent FFT, "
                 f"fs = {fs * 1e-6:.3g} MHz")
    ax.legend(loc="upper right", framealpha=0.9)

    box = (f"SNDR = {m['sndr_db']:.2f} dB\nSNR  = {m['snr_db']:.2f} dB\n"
           f"SFDR = {m['sfdr_db']:.2f} dB\nENOB = {m['enob']:.2f} bit")
    ax.annotate(box, xy=(0.015, 0.03), xycoords="axes fraction", ha="left", va="bottom",
                fontsize=9, family="monospace", color=INK,
                bbox=dict(boxstyle="round", fc="white", ec=GRID))
    fig.tight_layout()
    return _save(fig, out_dir, filename, dpi)


# ── 3. one conversion ──────────────────────────────────────────────────────────
def _resolve_conversion_keys(conversion_result: Mapping, adc: Mapping | None) -> dict:
    """Derive the sample/bit/clk/comparator/top-plate names, preferring the adc block.

    Never hardcodes the 3-bit vs 6-bit split: the clk key is taken from ``adc.clock``
    (absent for the static-comparator 3-bit case), so both render from the same code.
    Falls back to conservative guesses when ``adc`` is missing so a bare result dict
    still plots (used by the pure-Python no-crash test).
    """
    waves = conversion_result.get("input_waveforms", {})
    nodes = conversion_result.get("transient", {}).get("nodes", {})
    adc = dict(adc or {})
    sample = adc.get("sample_input", "sample")
    sample_b = adc.get("sample_bar_input", "sample_b")
    bit_inputs = list(adc.get("bit_inputs") or
                      [k for k in waves if k not in (sample, sample_b)])
    clk = (adc.get("clock") or {}).get("input")
    comp = adc.get("comparator_node")
    if comp is None:                                    # last-resort guess
        comp = next((n for n in ("OUTN", "vout", "OUTP") if n in nodes), None)
    top = [n for n in ("TOPP", "TOPN") if n in nodes]
    return {"sample": sample, "sample_b": sample_b, "bit_inputs": bit_inputs,
            "clk": clk, "comparator": comp, "top": top,
            "threshold": adc.get("comparator_threshold")}


def plot_sar_conversion(conversion_result: Mapping, adc: Mapping | None = None,
                        out_dir: Path | str = RESULTS,
                        filename: str = "adc_sar_conversion.png", dpi: int = 150) -> Path:
    """Logic-analyzer view of one physical SAR conversion.

    Tracks: sample/sample_b, the per-bit CDAC drives (+ clk strobe when the spec has
    an ``adc.clock`` block), the CDAC top plates, and the comparator node with each
    bit's physical decision instant marked (kept vs cleared). ``adc`` is the spec's
    ``adc`` block used to resolve node/key names; missing optional keys degrade
    gracefully (the panel is simply skipped).
    """
    k = _resolve_conversion_keys(conversion_result, adc)
    t = np.asarray(conversion_result["t"], float)
    tn = t * 1e9                                         # ns
    waves = conversion_result.get("input_waveforms", {})
    nodes = conversion_result.get("transient", {}).get("nodes", {})
    decisions = conversion_result.get("decisions", [])
    code = conversion_result.get("code")
    bits = conversion_result.get("bits", [])

    dec_t = [d["decision_time"] * 1e9 for d in decisions]

    fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    ax_s, ax_b, ax_t, ax_c = axes
    bit_str = "".join(str(int(v)) for v in bits) if len(bits) else "?"
    ax_s.set_title(f"SAR conversion — Vin = {conversion_result.get('vin', float('nan')):.4g} V"
                   f"  →  code {code} (bits {bit_str})")

    def _decision_lines(ax):
        for x in dec_t:
            ax.axvline(x, color=MUTED, ls=":", lw=0.8, zorder=0)

    # (a) sampling controls
    for key, col, lbl in ((k["sample"], BLUE, "sample"), (k["sample_b"], AQUA, "sample_b")):
        if key in waves:
            ax_s.plot(tn, np.asarray(waves[key], float), color=col, lw=1.6, label=lbl)
    _decision_lines(ax_s)
    ax_s.set_ylabel("sampling [V]"); ax_s.legend(loc="upper right", ncol=2, framealpha=0.9)

    # (b) per-bit CDAC drives (+clk) as offset logic-analyzer tracks (MSB on top)
    tracks = [(key, BLUE) for key in k["bit_inputs"]]
    if k["clk"] and k["clk"] in waves:
        tracks.append((k["clk"], ORANGE))
    n_tr = len(tracks)
    yticks, ylabels = [], []
    for i, (key, col) in enumerate(tracks):
        base = (n_tr - 1 - i)
        if key in waves:
            w = np.asarray(waves[key], float)
            span = max(np.max(w) - np.min(w), 1e-12)
            ax_b.plot(tn, base + 0.72 * (w - np.min(w)) / span, color=col, lw=1.3)
        yticks.append(base + 0.36); ylabels.append(key)
    _decision_lines(ax_b)
    ax_b.set_yticks(yticks); ax_b.set_yticklabels(ylabels, fontsize=7)
    ax_b.set_ylim(-0.2, n_tr); ax_b.set_ylabel("CDAC drives")
    ax_b.grid(axis="x")
    ax_b.annotate("clk strobe" if (k["clk"] and k["clk"] in waves) else "static comparator",
                  xy=(0.99, 0.02), xycoords="axes fraction", ha="right", va="bottom",
                  fontsize=7, color=MUTED)

    # (c) CDAC top plates
    for key, col in zip(k["top"], (BLUE, AQUA)):
        ax_t.plot(tn, np.asarray(nodes[key], float), color=col, lw=1.5, label=key)
    _decision_lines(ax_t)
    ax_t.set_ylabel("top plate [V]")
    if k["top"]:
        ax_t.legend(loc="upper right", ncol=2, framealpha=0.9)

    # (d) comparator node + decision instants
    comp = k["comparator"]
    if comp and comp in nodes:
        cv = np.asarray(nodes[comp], float)
        ax_c.plot(tn, cv, color=VIOLET, lw=1.6, label=comp)
    if k["threshold"] is not None:
        ax_c.axhline(float(k["threshold"]), color=MUTED, ls="--", lw=1, label="threshold")
    for d in decisions:
        x = d["decision_time"] * 1e9
        ax_c.axvline(x, color=MUTED, ls=":", lw=0.8, zorder=0)
        kept = d.get("kept")
        ax_c.scatter([x], [d.get("comparator_v", 0.0)], s=45, zorder=6,
                     color=GOOD if kept else CRITICAL)
        ax_c.annotate(f"b{d['bit']}={'1' if kept else '0'}",
                      xy=(x, d.get("comparator_v", 0.0)), xytext=(0, 8),
                      textcoords="offset points", ha="center", fontsize=7,
                      color=GOOD if kept else CRITICAL)
    ax_c.set_ylabel("comparator [V]"); ax_c.set_xlabel("time [ns]")
    # legend proxies for the kept/cleared markers
    ax_c.scatter([], [], s=45, color=GOOD, label="kept (1)")
    ax_c.scatter([], [], s=45, color=CRITICAL, label="cleared (0)")
    ax_c.legend(loc="upper right", ncol=2, fontsize=7, framealpha=0.9)
    ax_c.set_xlim(tn[0], tn[-1])

    fig.tight_layout()
    return _save(fig, out_dir, filename, dpi)


# ── 4. mismatch Monte Carlo ────────────────────────────────────────────────────
def _hist_with_overflow(ax, values, threshold, color, xlabel, over_label, bins=16):
    """Histogram finite values; non-finite (∞/NaN) fall into a labeled red overflow bar.

    ``over_label`` names what the overflow captures — ``inf`` DNL/INL means a
    non-monotonic trial, while a ``NaN`` offset means the first transition never fired,
    so the caller passes the right word rather than mislabeling one as the other.
    """
    values = np.asarray(values, float)
    finite = values[np.isfinite(values)]
    n_over = int(np.sum(~np.isfinite(values)))
    if finite.size:
        lo = min(0.0, float(finite.min()))
        hi = max(float(finite.max()), threshold or 0.0)
        edges = np.linspace(lo, hi * 1.05 + 1e-12, bins + 1)
        ax.hist(finite, bins=edges, color=color, edgecolor="white", linewidth=0.4)
        width = edges[1] - edges[0]
        over_x = edges[-1] + width * 0.8
    else:
        over_x, width = 1.0, 1.0
    if n_over:
        ax.bar(over_x, n_over, width=width * 0.9, color=CRITICAL,
               edgecolor="white", hatch="//", label=f"{over_label} ×{n_over}")
        ax.axvline(over_x - width * 0.6, color=MUTED, ls=":", lw=0.8)
        ax.annotate("∞/NaN", xy=(over_x, 0), xytext=(0, -13), textcoords="offset points",
                    ha="center", va="top", fontsize=7, color=CRITICAL, annotation_clip=False)
    if threshold is not None:
        ax.axvline(threshold, color=INK, ls="--", lw=1.2,
                   label=f"limit {threshold:g}")
    ax.set_xlabel(xlabel); ax.set_ylabel("trials")
    ax.margins(y=0.15)                                 # headroom so the legend clears the bars
    if ax.get_legend_handles_labels()[1]:              # only when something is labeled
        ax.legend(loc="upper right", fontsize=7, framealpha=0.9)


def plot_sar_mc(mc_result: Mapping, out_dir: Path | str = RESULTS,
                filename: str = "adc_sar_mc.png", dpi: int = 150) -> Path:
    """Histograms of max|DNL|, max|INL|, offset_lsb with thresholds + yield annotation.

    ``mc_result`` is a :func:`circuitopt.sar_mc.sar_mismatch_mc` payload. Non-monotonic
    trials (``inf`` DNL/INL) are clipped into a labeled overflow bin so they stay
    visible and count against the yield honestly.
    """
    arr = mc_result["arrays"]
    s = mc_result["summary"]
    dnl_th = s.get("dnl_threshold")
    inl_th = s.get("inl_threshold")

    fig, (a0, a1, a2) = plt.subplots(1, 3, figsize=(13, 4.5))
    _hist_with_overflow(a0, arr["max_abs_dnl"], dnl_th, BLUE, "max|DNL| [LSB]",
                        "non-monotonic")
    _hist_with_overflow(a1, arr["max_abs_inl"], inl_th, AQUA, "max|INL| [LSB]",
                        "non-monotonic")
    _hist_with_overflow(a2, arr["offset_lsb"], None, VIOLET, "offset [LSB]", "undefined")
    a2.axvline(0.0, color=MUTED, lw=0.8)

    yld = s.get("yield", float("nan"))
    n = s.get("n", len(np.asarray(arr["max_abs_dnl"])))
    mono = s.get("monotonic_rate", float("nan"))
    fig.suptitle(f"SAR mismatch Monte-Carlo — n = {n},  yield = {yld * 100:.1f}%  "
                 f"(|DNL|≤{dnl_th:g} & |INL|≤{inl_th:g} & no missing),  "
                 f"monotonic = {mono * 100:.0f}%", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _save(fig, out_dir, filename, dpi)


# ── showcase driver ────────────────────────────────────────────────────────────
def _render_showcase(circuit: str, out_dir: Path | str, *, static_points: int,
                     sine_points: int, tone_bin: int, sample_rate: float, mc_n: int,
                     mc_sigma_vth: float, mc_sigma_cu: float, seed: int, workers: int,
                     corner: str | None) -> list[Path]:
    from circuitopt.circuit_loader import load_circuit_json
    from circuitopt.sar import run_sar_conversion, run_sar_signal, run_sar_sweep
    from circuitopt.sar_mc import sar_mismatch_mc

    spec = load_circuit_json(circuit)
    adc = spec.adc
    vref = float(adc["vref"])
    levels = 1 << int(adc["n_bits"])
    out = []

    conv = run_sar_conversion(spec, 0.7 * vref, corner=corner)
    out.append(plot_sar_conversion(conv, adc=adc, out_dir=out_dir))
    print(f"  conversion: code {conv['code']}  ->  {out[-1]}")

    vin = (np.arange(static_points) + 0.5) * vref / static_points
    sweep = run_sar_sweep(spec, vin, corner=corner, workers=workers)
    note = (f"sub-sampled: {static_points} ramp points over {levels} codes"
            if static_points < levels else None)
    out.append(plot_sar_static(sweep, out_dir=out_dir, note=note))
    print(f"  static: max|DNL|={sweep['metrics']['max_abs_dnl']:.3f}  ->  {out[-1]}")

    phase = 2.0 * np.pi * tone_bin * np.arange(sine_points) / sine_points
    sig_in = 0.5 * vref + 0.45 * vref * np.sin(phase)
    sig = run_sar_signal(spec, sig_in, sample_rate, corner=corner,
                         fundamental_bin=tone_bin, workers=workers)
    out.append(plot_sar_spectrum(sig, out_dir=out_dir))
    print(f"  spectrum: SNDR={sig['metrics']['sndr_db']:.2f} dB  ->  {out[-1]}")

    # Stress mismatch config so the histograms show spread (the JSON block is mild).
    mc_cfg = {"sigma_vth0": mc_sigma_vth, "sigma_vth0_nmos": mc_sigma_vth,
              "sigma_vth0_pmos": mc_sigma_vth, "sigma_cu": mc_sigma_cu}
    mc = sar_mismatch_mc(spec, n=mc_n, seed=seed, corner=corner, workers=workers,
                         config=mc_cfg)
    out.append(plot_sar_mc(mc, out_dir=out_dir))
    print(f"  mc: yield={mc['summary']['yield'] * 100:.1f}%  ->  {out[-1]}")
    return out


def build_arg_parser(ap: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    ap = ap or argparse.ArgumentParser(description="Render the four SAR ADC showcase figures")
    ap.add_argument("--circuit", default=str(RESULTS.parent / "examples" / "freepdk45_sar6.json"),
                    help="circuit JSON carrying an 'adc' block")
    ap.add_argument("--out-dir", default=str(RESULTS), help="output directory (default: results/)")
    ap.add_argument("--static-points", type=int, default=16, help="ramp samples (default: 16)")
    ap.add_argument("--sine-points", type=int, default=64, help="sine samples (default: 64)")
    ap.add_argument("--tone-bin", type=int, default=5, help="coherent sine bin (default: 5)")
    ap.add_argument("--sample-rate", type=float, default=10e6, help="reported fs [Hz]")
    ap.add_argument("--mc-n", type=int, default=8, help="mismatch MC trials (default: 8)")
    ap.add_argument("--mc-sigma-vth", type=float, default=0.03,
                    help="stress Vth sigma [V] for the MC (default: 0.03)")
    ap.add_argument("--mc-sigma-cu", type=float, default=0.04,
                    help="stress unit-cap relative sigma for the MC (default: 0.04)")
    ap.add_argument("--seed", type=int, default=3, help="mismatch MC seed (default: 3)")
    ap.add_argument("--workers", type=int, default=8, help="parallel workers (default: 8)")
    ap.add_argument("--corner", default=None, choices=["nom", "ss", "ff"], help="process corner")
    return ap


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    print(f"Rendering SAR ADC showcase for {args.circuit}")
    _render_showcase(args.circuit, args.out_dir, static_points=args.static_points,
                     sine_points=args.sine_points, tone_bin=args.tone_bin,
                     sample_rate=args.sample_rate, mc_n=args.mc_n,
                     mc_sigma_vth=args.mc_sigma_vth, mc_sigma_cu=args.mc_sigma_cu,
                     seed=args.seed, workers=args.workers, corner=args.corner)


if __name__ == "__main__":
    main()
