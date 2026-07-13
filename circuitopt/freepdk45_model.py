"""FreePDK45 PDK — nfet/pfet as ngspice-C-evaluated BSIM4 devices.

FreePDK45 (NCSU, 45 nm predictive kit, "Customized PTM 45") ships flat BSIM4
level-54 ``.model`` cards marked ``version = 4.0`` with nom/ss/ff corners. Its
oracle is ngspice's built-in C-BSIM4: that version is what the cards were tuned
for, and our OpenVAF BSIM4.8 VA computes ~30 % different I-V on these aggressive
45 nm cards (version-independently — proven by mutating the card version). So
FreePDK45 binds to :class:`circuitopt.ngspice_device.NgspiceDevice` (ngspice-C via a
cached characterisation grid), NOT the OSDI host used by SKY130.

Registered as the ``"freepdk45"`` PDK with ``default=False`` — additive, the OTFT
stays default and SKY130 keeps its OSDI path. Corners ``nom``/``tt``/``ss``/``ff``
plus the mixed ``sf``/``fs`` select the card directory per polarity (case-insensitive;
unknown names raise — see :func:`normalize_corner`). Cards resolve under
``PDK_ROOT/freepdk45/models_<corner>/`` or the active/project virtual environment;
characterisation is lazy + cached under
``data/pdk/freepdk45/`` so reuse needs no ngspice. Full-circuit transient bypasses
the grid and runs the same cards directly through ngspice for complete BSIM4 charge.
"""
from __future__ import annotations

import os

from .device_model import register_pdk
from .ngspice_device import NgspiceDevice
from .toolchain import pdk_root

_PDK_ROOT = pdk_root()
_FP45_DIR = os.path.join(_PDK_ROOT, "freepdk45")
_VDD = 1.0                      # FreePDK45 nominal supply

# SS/FF are shipped as separate card directories (not a param shift): the corner
# name selects which models_<corner>/ file the device characterises against. The
# mixed corners sf/fs pick a DIFFERENT directory per polarity (nmos vs pmos): sf =
# NMOS slow + PMOS fast, fs = the reverse. tt is a plain alias of nom. This mapping
# is the single source of truth for "corner name -> models_<dir> per polarity" and
# is reused by the full-circuit ngspice render (see :mod:`circuitopt.ngspice_render`).
_CORNER_DIRS = {
    #        (nmos_dir, pmos_dir)
    "nom": ("nom", "nom"),
    "tt":  ("nom", "nom"),   # tt is an alias of the nominal (typical) corner
    "ss":  ("ss",  "ss"),
    "ff":  ("ff",  "ff"),
    "sf":  ("ss",  "ff"),    # NMOS slow, PMOS fast
    "fs":  ("ff",  "ss"),    # NMOS fast, PMOS slow
}
# Every corner name the FreePDK45 device / render paths accept.
FREEPDK45_CORNERS = tuple(_CORNER_DIRS)


def normalize_corner(corner) -> str:
    """Canonical FreePDK45 corner name for *corner*.

    ``None`` / ``""`` (no corner requested) map to ``"nom"``; strings are
    case-normalized, so ``"SF"`` behaves as ``"sf"`` and ``"TT"`` as ``"tt"``.
    Anything else — a typo like ``"sx"``, or a non-string — raises
    :class:`ValueError` naming the valid set. There is deliberately NO silent nom
    fallback: in a PVT campaign a misspelled corner silently producing nominal
    data poisons every downstream number, so unknown names fail loudly (matching
    the hard error the full-circuit ngspice render path raises)."""
    if corner is None or corner == "":
        return "nom"
    key = corner.lower() if isinstance(corner, str) else corner
    if key not in _CORNER_DIRS:
        raise ValueError(
            f"unknown FreePDK45 corner {corner!r}; expected one of {sorted(_CORNER_DIRS)}")
    return key


def corner_card_dir(polarity: str, corner: str) -> str:
    """``models_<dir>`` sub-directory name for *polarity* at silicon *corner*.

    ``nmos``/``pmos`` resolve independently so the mixed corners work: ``sf`` gives
    ``ss`` for nmos and ``ff`` for pmos; ``fs`` the reverse; ``nom``/``tt``/``ss``/``ff``
    give the same directory for both. *corner* goes through
    :func:`normalize_corner`, so case-insensitive spellings are accepted and an
    unknown corner raises :class:`ValueError` naming the valid set."""
    nmos_dir, pmos_dir = _CORNER_DIRS[normalize_corner(corner)]
    return nmos_dir if polarity == "nmos" else pmos_dir


def _card_path(polarity: str, corner: str) -> str:
    dev = "NMOS_VTG" if polarity == "nmos" else "PMOS_VTG"
    return os.path.join(_FP45_DIR, f"models_{corner_card_dir(polarity, corner)}", f"{dev}.inc")


class _Fp45Fet(NgspiceDevice):
    """Base for FreePDK45 fets: resolve the corner card, then behave as an
    NgspiceDevice. ``corner`` picks the card directory PER POLARITY via
    :func:`corner_card_dir`: nom/tt/ss/ff give the same directory for nmos and
    pmos, while the mixed corners sf (NMOS slow + PMOS fast) and fs (the reverse)
    resolve different directories for the two device classes. The corner NAME (not
    the resolved directory) is threaded into the characterisation-grid cache key, so
    an sf NMOS grid (built from the ss card) is cached separately from — and never
    collides with — the ss grid. Corner names are validated by
    :func:`normalize_corner`: case-insensitive, ``None``/``""`` mean ``nom``, and an
    UNKNOWN name raises :class:`ValueError` instead of silently falling back to
    nom (a typo must never produce nominal data in a corner sweep)."""
    POLARITY = "nmos"
    MODEL_NAME = "NMOS_VTG"
    VDD = _VDD

    def __init__(self, W: float = 0.09, L: float = 0.05, NF: int = 1, *,
                 corner: str = "nom", vb: float = 0.0, temperature: float = 300.15,
                 extract_w: float = None, **_ignored):
        corner = normalize_corner(corner)
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
