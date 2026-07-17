"""SKY130 compatibility exports and explicit ngspice card-extraction oracle.

Normal ``sky130.nmos`` and ``sky130.pmos`` simulation uses the in-process native
C BSIM4 backend from :mod:`circuitopt.pdk.sky130`. This module keeps the old
public class imports working and provides an explicit tool for resolving a new
geometry/corner card with a local SKY130 ngspice installation.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Dict

from .pdk.sky130 import Sky130Nfet, Sky130Pfet
from .pdk.sky130.library import (
    normalize_corner,
    normalize_polarity,
    sky130_card_dirs,
    sky130_card_filename,
)
from .toolchain import ngspice_binary, pdk_root


_SUBCKT = {
    "nmos": "sky130_fd_pr__nfet_01v8",
    "pmos": "sky130_fd_pr__pfet_01v8",
}
_DROP = {
    "vbd_max", "vbdr_max", "vbs_max", "vbsr_max", "vds_max", "vgb_max",
    "vgbr_max", "vgd_max", "vgdr_max", "vgs_max", "vgsr_max",
}
_LINE = re.compile(
    r"\s*([a-zA-Z]\w*)\s+([-+]?[\d.]+(?:[eE][-+]?\d+)?)\s*$")
_CARD_MEMO: Dict[tuple, Dict[str, float]] = {}


def _oracle_output_dir(output_dir: str | os.PathLike[str] | None) -> Path:
    if output_dir is not None:
        return Path(output_dir).expanduser()
    override = os.environ.get("SKY130_CARD_DIR")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".cache" / "circuitopt" / "sky130"


def extract_sky130_card(
    polarity: str,
    W: float,
    L: float,
    corner: str = "tt",
    *,
    output_dir: str | os.PathLike[str] | None = None,
) -> Dict[str, float]:
    """Resolve and cache one flat BSIM4 card using ngspice explicitly.

    This function is an oracle/preparation utility. Device construction never
    calls it automatically.
    """
    polarity = normalize_polarity(polarity)
    corner = normalize_corner(corner)
    memo_key = (polarity, float(W), float(L), corner)
    memo = _CARD_MEMO.get(memo_key)
    if memo is not None:
        return dict(memo)

    filename = sky130_card_filename(polarity, W, L, corner)
    for directory in (*sky130_card_dirs(), _oracle_output_dir(output_dir)):
        cache = directory / filename
        if cache.is_file():
            card = json.loads(cache.read_text(encoding="utf-8"))
            _CARD_MEMO[memo_key] = card
            return dict(card)

    library = (
        Path(pdk_root()) / "sky130A" / "libs.tech" / "ngspice" /
        "sky130.lib.spice"
    )
    if not library.is_file():
        raise RuntimeError(
            f"SKY130 ngspice library not found at {library}; set PDK_ROOT")
    simulator = ngspice_binary()
    if simulator is None:
        raise RuntimeError(
            "ngspice is required only for explicit SKY130 card extraction")

    subckt = _SUBCKT[polarity]
    netlist = (
        f"* extract {subckt}\n.lib \"{library}\" {corner}\n"
        f"xn d g s b {subckt} w={float(W):g} l={float(L):g}\n"
        "vd d 0 1.8\nvg g 0 1.8\nvs s 0 0\nvb b 0 0\n"
        ".control\nop\nset width=200\n"
        f"showmod m.xn.m{subckt}\n.endc\n.end\n"
    )
    with tempfile.NamedTemporaryFile(
        "w", suffix=".cir", delete=False, encoding="utf-8"
    ) as handle:
        handle.write(netlist)
        netlist_path = handle.name
    try:
        result = subprocess.run(
            [simulator, "-b", netlist_path],
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        os.unlink(netlist_path)
    output = result.stdout + result.stderr
    card: Dict[str, float] = {}
    for line in output.splitlines():
        match = _LINE.match(line)
        if match:
            name = match.group(1).lower()
            if name not in _DROP:
                card[name] = float(match.group(2))
    if "vth0" not in card:
        raise RuntimeError(
            f"SKY130 parameter extraction failed for {subckt}:\n"
            f"...{output[-600:]}")

    cache_dir = _oracle_output_dir(output_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / filename).write_text(
        json.dumps(card, indent=1, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _CARD_MEMO[memo_key] = card
    return dict(card)


__all__ = [
    "Sky130Nfet",
    "Sky130Pfet",
    "extract_sky130_card",
]
