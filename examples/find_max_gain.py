"""
Find the VIN bias for max small-signal gain, and plot gain vs VIN for each W/L.

Usage:
    python examples/find_max_gain.py
"""
import numpy as np
import matplotlib.pyplot as plt
from circuitopt.circuit_loader import load_circuit_json
from circuitopt.device_model import create_device
from circuitopt.compiled_topology import CompiledTopology
from scipy.optimize import fsolve


def _in_rail_box(topo, nv, bias):
    """Check that all solved node voltages are within [0, VDD]."""
    VDD = float(bias.get("VDD", 40))
    for n in topo.solved:
        v = nv.get(n)
        if v is None or not (-0.5 <= v <= VDD + 0.5):
            return False
    return True


def dc_solve_one(topo, bias, sizes, nf_map, prev_nv=None, gmin=1e-12):
    """DC solve for a single bias point. Rejects non-physical (out-of-rail) roots."""
    plan = CompiledTopology(topo, bias)
    dev_inst = {
        name: create_device("pmos_tft", W=sizes[name][0], L=sizes[name][1],
                            NF=nf_map.get(name, 1))
        for name, *_ in topo.devices
    }

    def Id(name, Vs, Vd, Vg):
        try: return abs(dev_inst[name].get_Idc(Vs, Vd, Vg))
        except: return 1e-18

    residuals = lambda x: plan.dc_residuals(x, Id, gmin)
    VCM = topo.default_guess_value(bias)
    guesses = topo.dc_guess_vectors(bias)

    # warm-start from previous solution if available
    if prev_nv is not None and _in_rail_box(topo, prev_nv, bias):
        wv = topo.guess_vector(prev_nv, default=VCM)
        guesses.insert(0, wv)

    # try fsolve with each guess; reject out-of-rail solutions
    for x0 in guesses:
        try:
            sol, _, ier, _ = fsolve(residuals, x0, full_output=True,
                                    xtol=1e-12, maxfev=3000)
            nv = topo.node_vals(sol)
            if _in_rail_box(topo, nv, bias) and \
               np.linalg.norm(residuals(sol), ord=np.inf) < 1e-10:
                return nv, dev_inst
        except Exception:
            pass

    # ── fallback: source-ramp continuation from zero ──
    base = np.array(guesses[0] if guesses else [VCM] * topo.n)
    x = base * 0.05
    for lam in np.linspace(0.05, 1.0, 25):
        bl = {k: (v * lam if isinstance(v, (int, float)) else v) for k, v in bias.items()}
        rp = CompiledTopology(topo, bl)
        try:
            s, _, ier, _ = fsolve(lambda z: rp.dc_residuals(z, Id, gmin),
                                  x, full_output=True, xtol=1e-12, maxfev=4000)
            if np.linalg.norm(rp.dc_residuals(s, Id, gmin), ord=np.inf) < 1e-10:
                x = s
            else:
                return None, dev_inst
        except Exception:
            return None, dev_inst
    nv = topo.node_vals(x)
    if _in_rail_box(topo, nv, bias):
        return nv, dev_inst
    return None, dev_inst


def _resolve_v(topo, node_name, nv, bias):
    """Resolve a node name to a voltage: solved node → nv; rail → bias or constant."""
    if node_name in topo.idx:             # solved node
        return float(nv[node_name])
    rail = topo.rails[node_name]          # rail: bias key or numeric constant
    if isinstance(rail, (int, float)):
        return float(rail)
    return float(bias[rail])              # bias key → value


def get_device_op(topo, nv, bias, dname, dev_inst):
    """Return (Vs, Vd, Vg) for a device from the solved node voltages."""
    for tname, dn, gn, sn in topo.devices:
        if tname == dname:
            return (_resolve_v(topo, sn, nv, bias),
                    _resolve_v(topo, dn, nv, bias),
                    _resolve_v(topo, gn, nv, bias))
    return 0.0, 0.0, 0.0


def main():
    spec = load_circuit_json("examples/single_stage.json")
    topo = spec.topology
    VDD = spec.bias["VDD"]

    # resolve NF
    nf_map = {}
    for name, *_ in topo.devices:
        if isinstance(spec.nf, dict):
            nf_map[name] = spec.nf.get(name, 1)
        elif isinstance(spec.nf, int):
            nf_map[name] = spec.nf
        else:
            nf_map[name] = 1

    combos = [
        ("ref W2k/1.5k",  {"MPU": (2000, 80), "MLD": (1500, 80)}),
        ("W3k/1k",         {"MPU": (3000, 80), "MLD": (1000, 80)}),
        ("W1.5k/2k",       {"MPU": (1500, 80), "MLD": (2000, 80)}),
        ("W4k/0.8k",       {"MPU": (4000, 80), "MLD": (800,  80)}),
        ("W2k/3k",         {"MPU": (2000, 80), "MLD": (3000, 80)}),
        ("W1k/4k",         {"MPU": (1000, 80), "MLD": (4000, 80)}),
    ]

    n_vin = 101
    vin_sweep = np.linspace(0, VDD, n_vin)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax_transfer, ax_gain = axes[0, 0], axes[0, 1]
    ax_detail, ax_table = axes[1, 0], axes[1, 1]

    summary_rows = []

    for label, sizes in combos:
        vout_arr = np.full(n_vin, np.nan)
        gain_lin = np.full(n_vin, np.nan)
        gain_db_arr = np.full(n_vin, np.nan)
        gm_in_arr = np.full(n_vin, np.nan)
        gm_ld_arr = np.full(n_vin, np.nan)

        best_gain = -1.0
        best_idx = 0
        prev_nv = None

        for i, vin in enumerate(vin_sweep):
            bias = {"VDD": VDD, "VIN": vin}
            nv, dev_inst = dc_solve_one(topo, bias, sizes, nf_map, prev_nv=prev_nv)
            if nv is None:
                continue
            prev_nv = nv
            vout_arr[i] = nv["OUT"]

            # Small-signal params
            Vs_in, Vd_in, Vg_in = get_device_op(topo, nv, bias, "MPU", dev_inst)
            Vs_ld, Vd_ld, Vg_ld = get_device_op(topo, nv, bias, "MLD", dev_inst)
            ss_in = dev_inst["MPU"].get_ss_params(Vs_in, Vd_in, Vg_in)
            ss_ld = dev_inst["MLD"].get_ss_params(Vs_ld, Vd_ld, Vg_ld)
            gm_in = ss_in["gm"]
            gds_in = ss_in["gds"]
            gm_ld = ss_ld["gm"]
            gds_ld = ss_ld["gds"]

            # Gain = gm_in / (gds_in + gm_ld + gds_ld)  ← common-source + diode load
            g = gm_in / (gds_in + gm_ld + gds_ld + 1e-20)
            gain_lin[i] = g
            gain_db_arr[i] = 20 * np.log10(max(g, 1e-12))
            gm_in_arr[i] = gm_in * 1e6
            gm_ld_arr[i] = gm_ld * 1e6

            if g > best_gain:
                best_gain = g
                best_idx = i

        best_vin = vin_sweep[best_idx]
        best_vout = vout_arr[best_idx]
        best_gain_db = gain_db_arr[best_idx]
        best_gm_in = gm_in_arr[best_idx]
        best_gm_ld = gm_ld_arr[best_idx]

        # Current at best bias
        bias_best = {"VDD": VDD, "VIN": best_vin}
        nv_best, dev_inst = dc_solve_one(topo, bias_best, sizes, nf_map)
        if nv_best:
            Vs_in, Vd_in, Vg_in = get_device_op(topo, nv_best, bias_best, "MPU", dev_inst)
            Idd = abs(dev_inst["MPU"].get_Idc(Vs_in, Vd_in, Vg_in))
        else:
            Idd = np.nan

        summary_rows.append((label, best_vin, best_vout, best_gain, best_gain_db,
                             Idd * 1e6, Idd * 1e6 * VDD, best_gm_in, best_gm_ld))

        # ── Plot ──
        mask = ~np.isnan(vout_arr)
        ax_transfer.plot(vin_sweep[mask], vout_arr[mask], label=label, lw=1.5)
        ax_gain.plot(vin_sweep[mask], gain_db_arr[mask], lw=1.3)
        # Mark max-gain point
        ax_gain.plot(best_vin, best_gain_db, 'o', markersize=6)

        # Detail subplot: gain + gm for the reference combo
        if "ref" in label:
            ax_detail.plot(vin_sweep[mask], gain_db_arr[mask], 'b-', lw=1.5, label="Gain (dB)")
            ax_detail.plot(vin_sweep[mask], gm_in_arr[mask], 'r--', lw=1, label="gm_in (µS)")
            ax_detail.plot(vin_sweep[mask], gm_ld_arr[mask], 'g--', lw=1, label="gm_ld (µS)")
            ax_detail.axvline(best_vin, color='gray', ls=':', alpha=0.5)
            ax_detail.legend(fontsize=8)
            ax_detail.set_xlabel("VIN (V)")
            ax_detail.set_title("ref W2k/1.5k — gain & gm breakdown")
            ax_detail.grid(True, alpha=0.25)

    # ── Transfer curve panel ──
    ax_transfer.plot([0, VDD], [0, VDD], "k--", alpha=0.15)
    ax_transfer.set_ylabel("VOUT (V)")
    ax_transfer.set_title("DC Transfer: VIN → VOUT")
    ax_transfer.legend(fontsize=7, ncol=2)
    ax_transfer.grid(True, alpha=0.25)

    # ── Gain panel ──
    ax_gain.axhline(y=0, color="k", ls="--", alpha=0.15)
    ax_gain.set_xlabel("VIN (V)")
    ax_gain.set_ylabel("Gain (dB)")
    ax_gain.set_title("Small-Signal Gain vs VIN")
    ax_gain.grid(True, alpha=0.25)

    # ── Table panel ──
    ax_table.axis("off")
    headers = ["Combo", "VIN opt", "VOUT", "Gain", "dB", "Idd µA", "P µW", "gm_in", "gm_ld"]
    col_widths = [0.17, 0.09, 0.08, 0.08, 0.07, 0.09, 0.09, 0.08, 0.08]
    table_data = []
    for r in summary_rows:
        table_data.append([
            r[0], f"{r[1]:.2f}", f"{r[2]:.2f}", f"{r[3]:.2f}",
            f"{r[4]:.1f}", f"{r[5]:.3f}", f"{r[6]:.2f}",
            f"{r[7]:.2f}", f"{r[8]:.2f}"
        ])
    tbl = ax_table.table(cellText=table_data, colLabels=headers, colWidths=col_widths,
                         loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    tbl.scale(1.0, 1.35)
    ax_table.set_title("Optimal Bias Point Summary", y=0.72)

    fig.suptitle("PMOS Inverter Amplifier — Max-Gain Bias Exploration", fontsize=13, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig("results/max_gain_analysis.png", dpi=150)
    plt.show()

    # ── Console summary ──
    print(f"\n{'Combo':<18s} {'VIN_opt':>7s} {'VOUT':>7s} {'Gain':>7s} {'dB':>6s} {'Idd(uA)':>9s} {'P(uW)':>8s} {'gm_in':>7s} {'gm_ld':>7s}")
    print("-" * 90)
    for r in summary_rows:
        print(f"{r[0]:<18s} {r[1]:>7.2f} {r[2]:>7.2f} {r[3]:>7.2f} {r[4]:>6.1f} {r[5]:>9.4f} {r[6]:>8.3f} {r[7]:>7.3f} {r[8]:>7.3f}")

    print("\nNote: Low gain (1-2 V/V) is inherent to diode-connected PMOS load —")
    print("output impedance ≈ 1/gm_load, so Av ≈ gm_in / gm_load.")
    print("For higher gain: use current-source load or cascode.")

    print("\nSaved → results/max_gain_analysis.png")


if __name__ == "__main__":
    main()
