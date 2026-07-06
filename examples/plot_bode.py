"""Bode (gain + phase vs frequency) for the plain AC amplifier and the chopper PAC.

Two figures (saved under ``results/`` by default):

* ``bode_ac.png``  — plain AC small-signal response of the AFE core
  (``amp_design3_typical``): the amplifier WITHOUT chopping. Low-pass:
  DC gain ≈ 22.9 dB, −3 dB ≈ 550 Hz (matches the amp calibration).
* ``bode_pac.png`` — the chopper's periodic-AC baseband **conversion** gain
  (``chopper_design3_typical``): the SAME amplifier WITH chopping. The conversion
  gain vs. modulation frequency, band-limited by the output RC filter.

Both magnitudes are complex (``result["response"]``) so phase is real, not a stub.

Run (also available as ``python -m circuitopt plot {ac,pac,bode}``):

    python examples/plot_bode.py                    # both
    python examples/plot_bode.py --ac --npts 401
    python examples/plot_bode.py --pac --f-chop 200 --fmin 0.1 --fmax 3000
    python examples/plot_bode.py --out-dir /tmp/plots

Use the conda ``daily`` env (Numba) for the PAC sweep — see
docs/environment_performance.md.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path(__file__).resolve().parent.parent / "results"


def _bode_axes(title):
    fig, (axm, axp) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axm.set_title(title)
    axm.set_ylabel("gain [dB]"); axm.grid(alpha=0.3, which="both")
    axp.set_ylabel("phase [deg]"); axp.set_xlabel("frequency [Hz]")
    axp.grid(alpha=0.3, which="both")
    axp.set_xscale("log")
    return fig, axm, axp


def _bw_3db(freqs, mag):
    """−3 dB frequency relative to the low-frequency (DC/baseband) gain."""
    ref = mag[0]
    below = np.where(mag <= ref / np.sqrt(2.0))[0]
    if len(below) == 0:
        return None
    i = below[0]
    if i == 0:
        return float(freqs[0])
    x0, x1 = np.log10(freqs[i - 1]), np.log10(freqs[i])          # log-freq interp
    y0, y1 = mag[i - 1], mag[i]
    t = (ref / np.sqrt(2.0) - y0) / (y1 - y0)
    return float(10 ** (x0 + t * (x1 - x0)))


def _draw_bode(freqs, H, title, color, path):
    mag = np.abs(H)
    mag_dB = 20 * np.log10(np.maximum(mag, 1e-12))
    phase = np.unwrap(np.angle(H)) * 180.0 / np.pi
    bw = _bw_3db(freqs, mag)
    fig, axm, axp = _bode_axes(title + (f", −3 dB ≈ {bw:.0f} Hz" if bw else ""))
    axm.semilogx(freqs, mag_dB, color=color)
    axm.axhline(mag_dB[0] - 3.0, color="gray", ls=":", lw=0.8)
    if bw:
        axm.axvline(bw, color="gray", ls=":", lw=0.8)
    axp.semilogx(freqs, phase, color=color)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    return mag, mag_dB, bw


# ── plain AC: AFE amplifier frequency response ────────────────────────────────
def plot_ac(case: str = "amp_design3_typical", fmin: float = 1e-2, fmax: float = 1e4,
            npts: int = 241, out_dir: Path | str = RESULTS) -> Path:
    from circuitopt.ac_solver import ac_solve
    from circuitopt.calibration import _sizes, load_reference

    md = load_reference(RESULTS.parent / "calibration" / case)["metadata"]
    sizes = _sizes(md); bias = dict(md["circuit"]["bias"]); nf = md["circuit"].get("nf", 1)
    corner = md.get("corner", "typical")
    freqs = np.logspace(np.log10(fmin), np.log10(fmax), npts)
    ac = ac_solve(sizes, bias, freqs, nf=nf, corner=corner)
    H = np.asarray(ac["response"], complex)
    path = Path(out_dir); path.mkdir(parents=True, exist_ok=True)
    path = path / "bode_ac.png"
    mag, mag_dB, bw = _draw_bode(
        freqs, H, f"Plain AC — AFE amplifier ({case})  DC gain {20*np.log10(abs(H[0])):.1f} dB",
        "tab:blue", path)
    print(f"AC: DC gain {mag_dB[0]:.2f} dB, −3 dB {bw:.1f} Hz  ->  {path}")
    return path


# ── PAC: chopper baseband conversion gain vs frequency ────────────────────────
def plot_pac(case: str = "chopper_design3_typical", f_chop: float | None = None,
             fmin: float = 1e-1, fmax: float = 2000.0, npts: int = 61,
             out_dir: Path | str = RESULTS) -> Path:
    from circuitopt.calibration import _sizes, load_reference, resolve_adaptive_config
    from circuitopt.chopper import pmos_chopper_pac, pmos_chopper_pss

    md = load_reference(RESULTS.parent / "calibration" / case)["metadata"]
    c = md["circuit"]; s = md.get("solver", {})
    sizes = _sizes(md); bias = dict(c["bias"]); nf = c.get("nf", 1)
    corner = md.get("corner", "typical")
    f_chop = float(c.get("f_chop", 225.0)) if f_chop is None else float(f_chop)
    pss = pmos_chopper_pss(
        sizes, bias, f_chop, switch_size=tuple(c.get("switch_size", (5000, 30))),
        switch_nf=int(c.get("switch_nf", 1)), nf=nf,
        edge_time=float(c.get("edge_time", 20e-6)), input_diff=0.0,
        input_common_mode=float(c.get("input_common_mode", bias["VCM"])),
        charge_injection=bool(c.get("charge_injection", False)),
        output_filter=tuple(c.get("output_filter", (1e6, 680e-12))),
        tstab_periods=int(s.get("tstab_periods", 2)), n_points=int(s.get("n_points", 321)),
        max_shooting_iters=int(s.get("max_shooting_iters", 5)),
        integration_method=s.get("integration_method", "gear2"),
        analytic_jacobian=True, fallback_least_squares=False, corner=corner,
        adaptive_config=resolve_adaptive_config(s))

    freqs = np.logspace(np.log10(fmin), np.log10(fmax), npts)
    pac = pmos_chopper_pac(sizes, bias, freqs, f_chop, pss_result=pss, nf=nf, corner=corner)
    H = np.asarray(pac["response"], complex)
    path = Path(out_dir); path.mkdir(parents=True, exist_ok=True)
    path = path / "bode_pac.png"
    mag, mag_dB, bw = _draw_bode(
        freqs, H,
        f"Chopper PAC — baseband conversion gain ({case})  f_chop={f_chop:.0f} Hz\n"
        f"baseband {abs(H[0]):.2f} ({20*np.log10(abs(H[0])):.1f} dB)",
        "tab:green", path)
    print(f"PAC: baseband {mag[0]:.3f} ({mag_dB[0]:.2f} dB), −3 dB {bw:.1f} Hz  ->  {path}")
    return path


def build_arg_parser(ap: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    """Attach the Bode-plot options to ``ap`` (or a fresh parser)."""
    ap = ap or argparse.ArgumentParser(description="Bode plots: plain AC and chopper PAC")
    ap.add_argument("--ac", action="store_true", help="only the plain AC amplifier")
    ap.add_argument("--pac", action="store_true", help="only the chopper PAC")
    ap.add_argument("--fmin", type=float, default=None, help="sweep start [Hz] (per-plot default)")
    ap.add_argument("--fmax", type=float, default=None, help="sweep stop [Hz] (per-plot default)")
    ap.add_argument("--npts", type=int, default=None, help="frequency points (per-plot default)")
    ap.add_argument("--f-chop", type=float, default=None,
                    help="chopper frequency for PAC [Hz] (default: case value)")
    ap.add_argument("--ac-case", default="amp_design3_typical", help="AC calibration case dir")
    ap.add_argument("--pac-case", default="chopper_design3_typical", help="PAC calibration case dir")
    ap.add_argument("--out-dir", default=str(RESULTS), help="output directory (default: results/)")
    return ap


def run(args) -> list[Path]:
    both = not (args.ac or args.pac)
    out = []
    if args.ac or both:
        kw = {"case": args.ac_case, "out_dir": args.out_dir}
        if args.fmin is not None: kw["fmin"] = args.fmin
        if args.fmax is not None: kw["fmax"] = args.fmax
        if args.npts is not None: kw["npts"] = args.npts
        out.append(plot_ac(**kw))
    if args.pac or both:
        kw = {"case": args.pac_case, "f_chop": args.f_chop, "out_dir": args.out_dir}
        if args.fmin is not None: kw["fmin"] = args.fmin
        if args.fmax is not None: kw["fmax"] = args.fmax
        if args.npts is not None: kw["npts"] = args.npts
        out.append(plot_pac(**kw))
    return out


def main(argv=None):
    run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    main()
