"""
Step-6 full Monte-Carlo: 3 corners (typical/slow/fast) x 500 runs, mismatch-only.

Per run: gain (dB), -3dB BW (Hz), IRN (uVrms, 0.05-100 Hz).
Two yield definitions:
  (A) Step-6 criterion: FAIL if ANY metric deviates >20% from the DESIGN value
      (gain measured on linear V/V; BW on Hz; IRN on uV).
  (B) absolute spec:    gain>=20 dB & BW>=100 Hz & IRN<=44.5 uV.
Reports mu/sigma/min/max per corner + yields, and a 3-corner histogram figure.
"""
import glob, os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from validate_sweep import parse_ac_nodes, parse_noise, band_rms, bw_interp

CORNERS = ["typical", "slow", "fast"]
BASE = "/tmp/mc_out"
GAIN_MIN, BW_MIN, IRN_MAX = 20.0, 100.0, 44.5
# nominal design values (Cadence-verified typical) for the >20% deviation test
# BW = interpolated -3dB (grid method under-read it as 501)
GD_dB, BD, ID = 22.91, 549.0, 37.1
GD_lin = 10 ** (GD_dB / 20)


def run_metrics(ac, noise):
    fac, nd = parse_ac_nodes(ac, ["vop", "von", "vip", "vin"])
    H = np.abs((nd["vop"] - nd["von"]) / (nd["vip"] - nd["vin"]))
    g = 20 * np.log10(H.max())
    bw = bw_interp(fac, H)          # interpolated -3dB crossing (no grid quantization)
    fr, out = parse_noise(noise)
    Hn = np.interp(np.log10(fr), np.log10(fac), H)
    irn = band_rms(fr, out ** 2 / Hn ** 2) * 1e6
    return g, bw, irn


def collect(corner):
    d = f"{BASE}/{corner}"
    rows = []
    for nf in sorted(glob.glob(f"{d}/mc-*_noimc.noise")):
        run = re.search(r"mc-(\d+)_", nf).group(1)
        af = f"{d}/mc-{run}_acmc.ac"
        if not os.path.exists(af):
            continue
        try:
            rows.append(run_metrics(af, nf))
        except Exception:
            pass
    nom = None
    if os.path.exists(f"{d}/acmc.ac") and os.path.exists(f"{d}/noimc.noise"):
        try:
            nom = run_metrics(f"{d}/acmc.ac", f"{d}/noimc.noise")
        except Exception:
            pass
    return np.array(rows), nom


def dev_vs(arr, gref_lin, bref, iref):
    """FAIL if ANY of gain(lin)/BW/IRN deviates >20% from the given reference."""
    g, bw, irn = arr[:, 0], arr[:, 1], arr[:, 2]
    return ((np.abs(10 ** (g / 20) - gref_lin) / gref_lin <= 0.20) &
            (np.abs(bw - bref) / bref <= 0.20) &
            (np.abs(irn - iref) / iref <= 0.20))


def yields(arr):
    dev20 = dev_vs(arr, GD_lin, BD, ID)                       # vs global typical design
    g, bw, irn = arr[:, 0], arr[:, 1], arr[:, 2]
    spec = (g >= GAIN_MIN) & (bw >= BW_MIN) & (irn <= IRN_MAX)
    return dev20, spec


def main():
    data = {}
    print("=== Step-6 Monte-Carlo: 3 corners x 500, mismatch-only ===")
    print(f"design ref: gain {GD_dB}dB ({GD_lin:.2f}V/V), BW {BD}Hz, IRN {ID}uV")
    print(f"(A) >20% deviation band: gain[{GD_lin*0.8:.1f}-{GD_lin*1.2:.1f}]V/V "
          f"BW[{BD*0.8:.0f}-{BD*1.2:.0f}]Hz IRN[{ID*0.8:.1f}-{ID*1.2:.1f}]uV\n")
    for c in CORNERS:
        arr, nom = collect(c)
        data[c] = (arr, nom)
        n = len(arr)
        dev20, spec = yields(arr)
        names = ["gain dB", "BW Hz", "IRN uV"]
        print(f"--- {c.upper()} corner ({n} runs)" +
              (f"  nominal: gain {nom[0]:.2f}dB BW {nom[1]:.0f}Hz IRN {nom[2]:.1f}uV" if nom else "") + " ---")
        print(f"{'metric':<9}{'mean':>9}{'std':>9}{'min':>9}{'max':>9}")
        for j, nm in enumerate(names):
            col = arr[:, j]
            print(f"{nm:<9}{col.mean():>9.2f}{col.std():>9.2f}{col.min():>9.2f}{col.max():>9.2f}")
        print(f"  yield (A, >20% dev vs TYPICAL design): {dev20.sum()}/{n} = {dev20.mean()*100:.1f}%")
        if nom:
            devc = dev_vs(arr, 10 ** (nom[0] / 20), nom[1], nom[2])
            print(f"  yield (A', >20% dev vs THIS corner's nominal = pure mismatch): "
                  f"{devc.sum()}/{n} = {devc.mean()*100:.1f}%")
        print(f"  yield (B, abs spec gain≥20/BW≥100/IRN≤44.5): {spec.sum()}/{n} = {spec.mean()*100:.1f}%\n")

    # ---- figure: 3 metrics x corners overlaid ----
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    cols = {"typical": "C0", "slow": "C1", "fast": "C2"}
    specs = [(0, "gain [dB]", GAIN_MIN, True), (1, "-3dB BW [Hz]", BW_MIN, True),
             (2, "IRN [µVrms]", IRN_MAX, False)]
    for j, label, spec, geq in specs:
        for c in CORNERS:
            arr = data[c][0]
            ax[j].hist(arr[:, j], bins=25, alpha=0.5, color=cols[c],
                       label=f"{c} (μ={arr[:,j].mean():.1f})")
        ax[j].axvline(spec, color="r", ls="--", lw=1.6, label=f"spec {'≥' if geq else '≤'}{spec}")
        ax[j].set_xlabel(label); ax[j].set_ylabel("runs"); ax[j].legend(fontsize=7)
        ax[j].set_title(label, fontsize=9)
    # overall yields
    allarr = np.vstack([data[c][0] for c in CORNERS])
    d20, sp = yields(allarr)
    fig.suptitle(f"Step-6 MC: 3 corners x 500 (mismatch) — spec-yield "
                 f"{sp.mean()*100:.1f}%  |  >20%-dev-yield {d20.mean()*100:.1f}%  "
                 f"(n={len(allarr)})", fontsize=11)
    fig.tight_layout()
    fig.savefig("../figures/mc_corners.png", dpi=130)
    print(f"TOTAL ({len(allarr)} runs): spec-yield {sp.mean()*100:.1f}%, "
          f">20%-dev-yield {d20.mean()*100:.1f}%")
    print("saved ../figures/mc_corners.png")


if __name__ == "__main__":
    main()
