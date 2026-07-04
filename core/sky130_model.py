"""SKY130 PDK — nfet/pfet as OSDI BSIM4 devices, params resolved via ngspice.

SKY130's FET models are binned BSIM4 subckts with a huge ``.param``/Monte-Carlo web
(63 bins, 2000+ expression params, a 45k-line corner file). Rather than reimplement
that resolution, we let **ngspice resolve it** — instantiate the device, run an op,
and ``showmod`` the fully-resolved BSIM4 model card — then feed that flat card to our
OpenVAF-compiled ``bsim4va`` (see the ``silicon-pdk-openvaf`` memory). SKY130 uses
built-in BSIM4.5; our VA is 4.8, so the result is *"SKY130-parameterized BSIM4.8"* —
a realistic 130 nm process, not SkyWater's bit-exact sign-off model.

Registered as the ``"sky130"`` PDK with ``default=False`` — the AT4000TG OTFT stays the
default process, so this is purely additive (the amp/chopper byte-gate is untouched).

Extraction needs the SKY130 PDK + OSDI-ngspice (external drive); resolved cards are
cached under ``data/pdk/sky130/`` so re-use needs no toolchain. Override locations with
``PDK_ROOT`` / ``OPENVAF_ROOT`` / ``NGSPICE_BIN``.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from typing import Dict

from .device_model import register_pdk
from .osdi_device import OsdiDevice

_PDK_ROOT = os.environ.get("PDK_ROOT", "/Volumes/MacoutDsik/pdk")
_NGSPICE_LIB = os.path.join(_PDK_ROOT, "sky130A/libs.tech/ngspice/sky130.lib.spice")
_VAF_ROOT = os.environ.get("OPENVAF_ROOT", "/Volumes/MacoutDsik/Code/VAF/OpenVAF-Reloaded")
_RUN_NGSPICE = os.path.join(_VAF_ROOT, ".claude/skills/run-osdi-ngspice/scripts/run-ngspice.sh")
_BSIM4_VA = os.path.join(_VAF_ROOT, "integration_tests/BSIM4/bsim4.va")
_CARD_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/pdk/sky130")

_SUBCKT = {"nmos": "sky130_fd_pr__nfet_01v8", "pmos": "sky130_fd_pr__pfet_01v8"}
# showmod emits reliability voltage-limit params bsim4va does not model → drop.
_DROP = {"vbd_max", "vbdr_max", "vbs_max", "vbsr_max", "vds_max", "vgb_max",
         "vgbr_max", "vgd_max", "vgdr_max", "vgs_max", "vgsr_max"}
_LINE = re.compile(r"\s*([a-zA-Z]\w*)\s+([-+]?[\d.]+(?:[eE][-+]?\d+)?)\s*$")


def _ngspice():
    return _RUN_NGSPICE if os.path.exists(_RUN_NGSPICE) else os.environ.get("NGSPICE_BIN", "ngspice")


def extract_sky130_card(polarity: str, W: float, L: float,
                        corner: str = "tt") -> Dict[str, float]:
    """Resolved BSIM4 model params for a SKY130 FET at (W, L)[µm], via ngspice.

    Instantiates the SKY130 subckt at the given size/corner, runs an op, and parses
    ``showmod``'s fully-resolved card. Cached to ``data/pdk/sky130/*.json``.
    """
    subckt = _SUBCKT[polarity]
    os.makedirs(_CARD_DIR, exist_ok=True)
    cache = os.path.join(_CARD_DIR, f"{subckt}_{corner}_W{W:g}_L{L:g}.json")
    if os.path.exists(cache):
        with open(cache) as fh:
            return json.load(fh)
    if not os.path.exists(_NGSPICE_LIB):
        raise RuntimeError(f"SKY130 PDK ngspice lib not found at {_NGSPICE_LIB}; set PDK_ROOT")
    # SKY130 ngspice models set `.option scale=1u`, so W/L are bare numbers in µm.
    net = (f"* extract {subckt}\n.lib \"{_NGSPICE_LIB}\" {corner}\n"
           f"xn d g s b {subckt} w={W:g} l={L:g}\n"
           f"vd d 0 1.8\nvg g 0 1.8\nvs s 0 0\nvb b 0 0\n"
           f".control\nop\nset width=200\nshowmod m.xn.m{subckt}\n.endc\n.end\n")
    with tempfile.NamedTemporaryFile("w", suffix=".cir", delete=False) as fh:
        fh.write(net)
        cir = fh.name
    try:
        out = subprocess.run([_ngspice(), "-b", cir], capture_output=True, text=True).stdout
    finally:
        os.unlink(cir)
    card: Dict[str, float] = {}
    for line in out.splitlines():
        m = _LINE.match(line)
        if m:
            name = m.group(1).lower()
            if name not in _DROP:
                try:
                    card[name] = float(m.group(2))
                except ValueError:
                    pass
    if "vth0" not in card:
        raise RuntimeError(f"SKY130 param extraction failed for {subckt}:\n...{out[-600:]}")
    with open(cache, "w") as fh:
        json.dump(card, fh, indent=1, sort_keys=True)
    return card


class _Sky130Fet(OsdiDevice):
    """Base for SKY130 fets: pull the resolved card, then behave as an OsdiDevice."""
    VA_PATH = _BSIM4_VA
    MODULE = "bsim4va"
    POLARITY = "nmos"
    EXTRACT_W: float = None    # if set (µm), resolve the card at this fixed W and let
    #                            bsim4va scale the actual W — fast + smooth for W sweeps
    #                            (e.g. optimization); default resolves per instance W.

    def __init__(self, W: float = 1.0, L: float = 0.15, NF: int = 1, *,
                 corner: str = "tt", vb: float = 0.0, temperature: float = 300.15,
                 extract_w: float = None, **_ignored):
        # corner is a SKY130 discrete process corner (baked into the extracted card),
        # not a bsim4va param shift; extra registry kwargs (pvt0/…) don't apply here.
        # extract_w (per instance) pins the card-extraction reference W so a W sweep
        # resolves one card and lets bsim4va scale W (fast + smooth for optimization);
        # it overrides the class-level EXTRACT_W. Absent → resolve per instance W.
        ref = extract_w if extract_w is not None else self.EXTRACT_W
        w_card = ref if ref is not None else W
        self.BASE_CARD = extract_sky130_card(self.POLARITY, w_card, L, corner)
        self.corner = corner
        super().__init__(W=W, L=L, NF=NF, vb=vb, temperature=temperature)


class Sky130Nfet(_Sky130Fet):
    POLARITY = "nmos"
    TYPE = 1


class Sky130Pfet(_Sky130Fet):
    POLARITY = "pmos"
    TYPE = -1


# Register the SKY130 process (nmos + pmos) — additive; AT4000TG stays default.
register_pdk("sky130", {"nmos": Sky130Nfet, "pmos": Sky130Pfet}, default=False)
