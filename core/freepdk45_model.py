"""FreePDK45 PDK — nfet/pfet as ngspice-C-evaluated BSIM4 devices.

FreePDK45 (NCSU, 45 nm predictive kit, "Customized PTM 45") ships flat BSIM4
level-54 ``.model`` cards marked ``version = 4.0`` with nom/ss/ff corners. Its
oracle is ngspice's built-in C-BSIM4: that version is what the cards were tuned
for, and our OpenVAF BSIM4.8 VA computes ~30 % different I-V on these aggressive
45 nm cards (version-independently — proven by mutating the card version). So
FreePDK45 binds to :class:`core.ngspice_device.NgspiceDevice` (ngspice-C via a
cached characterisation grid), NOT the OSDI host used by SKY130.

Registered as the ``"freepdk45"`` PDK with ``default=False`` — additive, the OTFT
stays default and SKY130 keeps its OSDI path. Corners ``nom``/``ss``/``ff`` select
the matching card directory. Cards live on the external drive
(``PDK_ROOT/freepdk45/models_<corner>/``); characterisation is lazy + cached under
``data/pdk/freepdk45/`` so reuse needs no ngspice.
"""
from __future__ import annotations

import os

from .device_model import register_pdk
from .ngspice_device import NgspiceDevice

_PDK_ROOT = os.environ.get("PDK_ROOT", "/Volumes/MacoutDsik/pdk")
_FP45_DIR = os.path.join(_PDK_ROOT, "freepdk45")
_VDD = 1.0                      # FreePDK45 nominal supply

# SS/FF are shipped as separate card directories (not a param shift): the corner
# name selects which models_<corner>/ file the device characterises against.
_CORNERS = ("nom", "ss", "ff")


def _card_path(polarity: str, corner: str) -> str:
    dev = "NMOS_VTG" if polarity == "nmos" else "PMOS_VTG"
    return os.path.join(_FP45_DIR, f"models_{corner}", f"{dev}.inc")


class _Fp45Fet(NgspiceDevice):
    """Base for FreePDK45 fets: resolve the corner card, then behave as an
    NgspiceDevice. ``corner`` (nom/ss/ff) picks the card directory; unknown
    corners fall back to ``nom`` so a generic sweep never crashes."""
    POLARITY = "nmos"
    MODEL_NAME = "NMOS_VTG"
    VDD = _VDD

    def __init__(self, W: float = 0.09, L: float = 0.05, NF: int = 1, *,
                 corner: str = "nom", vb: float = 0.0, temperature: float = 300.15,
                 extract_w: float = None, **_ignored):
        corner = corner if corner in _CORNERS else "nom"
        self.CARD_PATH = _card_path(self.POLARITY, corner)
        super().__init__(W=W, L=L, NF=NF, vb=vb, corner=corner,
                         temperature=temperature, extract_w=extract_w)


class Fp45Nfet(_Fp45Fet):
    POLARITY = "nmos"
    MODEL_NAME = "NMOS_VTG"


class Fp45Pfet(_Fp45Fet):
    POLARITY = "pmos"
    MODEL_NAME = "PMOS_VTG"


# Register the FreePDK45 process (nmos + pmos) — additive; OTFT stays default.
register_pdk("freepdk45", {"nmos": Fp45Nfet, "pmos": Fp45Pfet}, default=False)
