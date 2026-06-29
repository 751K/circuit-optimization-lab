"""Cadence vs local calibration of the chopper switch PMOS in its on-state.

The chopper commutators are pmos_TFT W=5000 L=30 devices that operate in deep
triode (gate at the clock rail, source/drain near the signal common-mode ~31.4 V,
|Vds|~0).  That triode/on-state region is NEVER exercised by the amplifier AC/noise
tests (those transistors sit in saturation), so its on-resistance and overlap caps
are unvalidated -- and the residual ~1.7% chopper PAC gap traces to the switch
loading (Ron x Cin pole ~2 kHz sits right in the 200/600/1000 Hz chop harmonics).

Workflow (mirrors verify_design3.py):

    python tools/calibrate_switch.py gen      # write netlist + runner to /tmp/sw_cal
    # scp /tmp/sw_cal/* flex:~/afe_swcal/ ; ssh flex 'cd afe_swcal && bash run.sh'
    # pull ~/afe_swcal/sw_cal.raw/*  ->  /tmp/sw_cal_out/
    python tools/calibrate_switch.py parse    # parse PSF + compare to local model

The local baseline (PMOS_TFT, 5000/30 NF=1, Vs=31.4, full-on Vg=0):
    Ron ~ 70.9 kOhm,  Cgs ~ 14.8 pF,  Cgd ~ 19.7 pF.
"""
import os
import re
import sys

import numpy as np

# Device under test: identical to the eight chopper commutators.
SW_W, SW_L, SW_NF = 5000.0, 30.0, 1
VT = -3.03
VS_CM = 31.4          # signal common-mode the switch sits at
VDS_ON = -5e-3        # tiny drain-source drop used for the on-state probe
VG_LIST = (0.0, 5.0, 10.0, 16.0, 20.0)   # gate sweep: 0 = full on -> off

DEV_PARAMS = (f"pmos_TFT W={SW_W:.0f} L={SW_L:.0f} VT={VT} Roff=1 NF={SW_NF} "
              "Reg=1 C1=37.5 C2=50 C3=35 C4=35 Ci=2.4 kv=1 kh=1")

NETLIST = f"""// chopper switch (5000/30 pmos_TFT) on-state calibration
simulator lang=spectre
global 0
parameters vs_dc={VS_CM} vd_dc={VS_CM} vg_dc=0
include "/cadappl_sde/iclibs/AT_4000TG/AT_4000TG/monte.scs" section=typical
ahdl_include "/cadappl_sde/iclibs/AT_4000TG/AT_4000TG/pmos_TFT/veriloga/veriloga.va"

M0 (s d g) {DEV_PARAMS}
Vs (s 0) vsource dc=vs_dc
Vg (g 0) vsource dc=vg_dc mag=1
Vd (d 0) vsource dc=vd_dc

simulatorOptions options reltol=1e-5 vabstol=1e-9 iabstol=1e-16 temp=27 tnom=27 gmin=1e-12

// (1) Ron at full-on: Id vs Vds, sweep drain around the source rail (Vg=0)
ronvds dc dev=Vd param=dc start={VS_CM - 0.4:.3f} stop={VS_CM + 0.4:.3f} step=0.01

// (2) Ron vs gate: Id at fixed small Vds=-5mV while the gate goes 0 -> 40 V
altd alter param=vd_dc value={VS_CM + VDS_ON:.4f}
ronvgs dc dev=Vg param=dc start=0 stop=40 step=0.5

// (3) on-state overlap caps: AC on the gate (mag=1) at Vg=0, drain back to rail
altd2 alter param=vd_dc value={VS_CM:.3f}
altg  alter param=vg_dc value=0
capac ac start=10 stop=10k dec=4

save Vd:p Vg:p Vs:p
"""

RUNNER = """#!/bin/bash
export LM_LICENSE_FILE=5280@ankaramy.ele.tue.nl
source /eda/cadence/2024-25/scripts/SPECTRE_24.10.078_RHELx86.sh
cd ~/afe_swcal
spectre sw_cal.scs +escchars +log sw_cal.log -format psfascii -raw ./sw_cal.raw 2>&1 | tail -3
echo ALL_DONE
"""


def gen():
    d = "/tmp/sw_cal"
    os.makedirs(d, exist_ok=True)
    with open(f"{d}/sw_cal.scs", "w") as f:
        f.write(NETLIST)
    with open(f"{d}/run.sh", "w") as f:
        f.write(RUNNER)
    print("wrote sw_cal.scs + run.sh to", d)
    print("next: scp /tmp/sw_cal/* flex:~/afe_swcal/ ; ssh flex 'cd afe_swcal && bash run.sh'")
    print("then: pull ~/afe_swcal/sw_cal.raw/* -> /tmp/sw_cal_out/ ; python tools/calibrate_switch.py parse")


def _psf_value_block(path):
    with open(path) as fh:
        lines = fh.read().splitlines()
    i0 = next(i for i, l in enumerate(lines) if l.strip() == "VALUE")
    iend = next(i for i in range(i0 + 1, len(lines)) if lines[i].strip() == "END")
    return lines[i0 + 1:iend]


def parse_dc(path, current="Vd:p"):
    sweep, idata = [], []
    for l in _psf_value_block(path):
        l = l.strip()
        m = re.match(r'^"(?:sweep|dc)"\s+([-\d.eE+]+)', l)
        if m:
            sweep.append(float(m.group(1)))
            continue
        m = re.match(rf'^"{re.escape(current)}"\s+([-\d.eE+]+)\s*$', l)
        if m:
            idata.append(float(m.group(1)))
    return np.array(sweep), np.array(idata)


def parse_ac(path):
    freq, data = [], {}
    for l in _psf_value_block(path):
        l = l.strip()
        m = re.match(r'^"freq"\s+([-\d.eE+]+)', l)
        if m:
            freq.append(float(m.group(1)))
            continue
        m = re.match(r'^"([^"]+)"\s+\(\s*([-\d.eE+]+)\s+([-\d.eE+]+)\s*\)', l)
        if m:
            data.setdefault(m.group(1), []).append(
                complex(float(m.group(2)), float(m.group(3))))
    return np.array(freq), {k: np.array(v) for k, v in data.items()}


def _find(*cands):
    for p in cands:
        if os.path.exists(p):
            return p
    return None


def _local():
    from core.ac_solver import get_ss_params
    from core.device_model import create_device
    inst = create_device("pmos_tft", W=SW_W, L=SW_L, NF=SW_NF)
    return get_ss_params, inst


def parse():
    raw = "/tmp/sw_cal_out"
    get_ss_params, inst = _local()

    # (1) Ron at full-on from the Vds sweep
    p = _find(f"{raw}/ronvds.dc", f"{raw}/sw_cal.raw/ronvds.dc")
    if p:
        vd, idd = parse_dc(p)
        vds = vd - VS_CM
        k = int(np.argmin(np.abs(vds)))
        # central-difference conductance around Vds=0 (|.| drops the PSF current-sign convention)
        gds_c = abs((idd[k + 1] - idd[k - 1]) / (vd[k + 1] - vd[k - 1]))
        pl = abs(get_ss_params(SW_W, SW_L, VS_CM, VS_CM + VDS_ON, 0.0, nf=SW_NF, dev_inst=inst)["gds"])
        print("(1) Full-on Ron (Vg=0, Vs=31.4):")
        print(f"    Cadence  gds={gds_c:.4e} S   Ron={1/gds_c:9.1f} ohm")
        print(f"    Local    gds={pl:.4e} S   Ron={1/pl:9.1f} ohm"
              f"   ({(pl/gds_c-1)*100:+.2f}%)")
    else:
        print("(1) ronvds.dc not found")

    # (2) Ron vs Vgs
    p = _find(f"{raw}/ronvgs.dc", f"{raw}/sw_cal.raw/ronvgs.dc")
    if p:
        vg, idd = parse_dc(p)
        print("\n(2) Conduction vs gate (Vds=-5mV):  gds = Id/Vds")
        print(f"    {'Vg':>5}{'cad_gds':>12}{'loc_gds':>12}{'cad_Ron':>11}{'loc_Ron':>11}{'d%':>7}")
        for vgt in VG_LIST:
            i = int(np.argmin(np.abs(vg - vgt)))
            gc = abs(idd[i] / VDS_ON)
            pl = abs(get_ss_params(SW_W, SW_L, VS_CM, VS_CM + VDS_ON, vg[i], nf=SW_NF, dev_inst=inst)["gds"])
            print(f"    {vg[i]:>5.1f}{gc:>12.3e}{pl:>12.3e}"
                  f"{1/gc:>11.0f}{1/pl:>11.0f}{(pl/gc-1)*100:>7.2f}")
    else:
        print("\n(2) ronvgs.dc not found")

    # (3) on-state overlap caps from AC
    p = _find(f"{raw}/capac.ac", f"{raw}/sw_cal.raw/capac.ac")
    if p:
        fr, d = parse_ac(p)
        i = int(np.argmin(np.abs(fr - 1000.0)))
        w = 2 * np.pi * fr[i]
        cgd = abs(d["Vd:p"][i].imag) / w if "Vd:p" in d else float("nan")
        cgs = abs(d["Vs:p"][i].imag) / w if "Vs:p" in d else float("nan")
        pl = get_ss_params(SW_W, SW_L, VS_CM, VS_CM, 0.0, nf=SW_NF, dev_inst=inst)
        print("\n(3) On-state overlap caps (Vg=0) @1kHz:")
        print(f"    Cadence  Cgs={cgs*1e12:6.2f} pF   Cgd={cgd*1e12:6.2f} pF")
        print(f"    Local    Cgs={pl['Cgs']*1e12:6.2f} pF   Cgd={pl['Cgd']*1e12:6.2f} pF")
    else:
        print("\n(3) capac.ac not found")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "parse"
    {"gen": gen, "parse": parse}[cmd]()
