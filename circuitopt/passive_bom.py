"""Passive bill of materials with silicon-area estimates for a circuit design.

Netlist review catches wrong values; nothing catches a value that is *correct
and unbuildable*.  A 40 pF compensation capacitor simulates perfectly and costs
~20 000 um2 -- an order of magnitude more area than every transistor in the
amplifier combined -- and that only becomes visible when somebody adds it up.
This adds it up.

Two things make the report trustworthy:

* **DUT vs testbench is derived, not declared.**  A generator emits several
  testbenches around one amplifier; an element belonging to the amplifier
  appears in ALL of them, while bias helpers, AC coupling and loop probes
  appear in some.  Membership across the deck set is therefore the
  classification rule, and it needs no annotation to stay correct as
  testbenches are added.

* **Process constants are explicit and overridable.**  Sheet resistance and
  capacitor density vary by flavour (unsilicided poly vs N-well, MOM vs MIM),
  so the defaults are stated in the output and every one can be overridden.
  The point of the number is its order of magnitude and its ranking, not
  three significant figures.

The deck set comes from a signoff manifest (which already names every
testbench) or from explicit paths, so this reads the same committed JSON as
every other command -- no generator module and no Python import required.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


# Defaults for a 28 nm bulk CMOS flow.  Stated in the report so a reader never
# has to guess which flavour produced the areas.
DEFAULTS = {
    "sheet_ohm_sq": 300.0,     # unsilicided poly resistor
    "resistor_width_um": 0.8,  # drawn width chosen for matching, not minimum
    "cap_ff_um2": 2.0,         # multi-layer MOM
}


def classify(decks):
    """``{element_name: ("R"|"C", value, is_dut)}`` from ``{deck_name: deck}``.

    An element present in every deck belongs to the device under test; one
    present in only some belongs to a testbench that wraps it."""
    seen: dict[str, set[str]] = {}
    kind: dict[str, tuple[str, float]] = {}
    for deck_name, deck in decks.items():
        for section, letter in (("resistors", "R"), ("capacitors", "C")):
            for element in deck.get(section, ()) or ():
                name = element["name"]
                seen.setdefault(name, set()).add(deck_name)
                kind[name] = (letter, float(element[letter]))
    total_decks = set(decks)
    return {
        name: (kind[name][0], kind[name][1], seen[name] == total_decks)
        for name in kind
    }


def resistor_area_um2(ohms, *, sheet_ohm_sq, resistor_width_um):
    """Serpentine body area: squares x width^2 (no head/contact overhead)."""
    squares = float(ohms) / float(sheet_ohm_sq)
    return squares * resistor_width_um * resistor_width_um


def capacitor_area_um2(farads, *, cap_ff_um2):
    return float(farads) * 1e15 / float(cap_ff_um2)


def dut_devices(decks):
    """Transistors present in every deck, i.e. the amplifier's own devices.

    Same membership rule as the passives: a testbench switch or probe device
    appears in some decks, an amplifier device in all of them."""
    seen: dict[str, set[str]] = {}
    geometry: dict[str, tuple[float, float, int]] = {}
    for deck_name, deck in decks.items():
        for device in deck.get("devices", ()) or ():
            name = device["name"]
            seen.setdefault(name, set()).add(deck_name)
            geometry[name] = (float(device.get("W", 0.0)),
                              float(device.get("L", 0.0)),
                              int(device.get("M", 1)))
    total = set(decks)
    return {name: geometry[name] for name in geometry if seen[name] == total}


def transistor_gate_area_um2(decks):
    """Total drawn gate area of the DUT transistors.

    Gate area alone understates the real active footprint (diffusion, contacts
    and routing typically make it 2-3x), so the passive/active ratio printed
    against it is a lower bound on how passive-dominated the layout is.  Each
    device is counted once from its own deck entry -- both halves of a
    differential pair are separate instances and must not be doubled again."""
    devices = dut_devices(decks)
    if not devices:
        return None
    return sum(width * length * mult for width, length, mult in devices.values())


def load_decks(paths) -> dict[str, Any]:
    """``{filename: deck}`` from deck paths, or from a signoff manifest.

    A manifest already names every testbench wrapped around one DUT, which is
    exactly the deck set the membership rule needs."""
    resolved: dict[str, Any] = {}
    for path in paths:
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, Mapping) and "cases" in data and "pvt" in data:
            root = path.parent
            for case in data["cases"]:
                circuit = root / case["circuit"]
                resolved[circuit.name] = json.loads(
                    circuit.read_text(encoding="utf-8"))
            continue
        resolved[path.name] = data
    if not resolved:
        raise SystemExit("no decks to inventory")
    if len(resolved) < 2:
        raise SystemExit(
            "DUT-vs-testbench classification needs at least two decks around "
            "the same design; pass a signoff manifest or several deck files")
    return resolved


def build_report(decks, *, exclude=(), **process):
    elements = classify(decks)
    excluded = set(exclude)
    rows = []
    for name, (letter, value, is_dut) in elements.items():
        if letter == "R":
            area = resistor_area_um2(
                value, sheet_ohm_sq=process["sheet_ohm_sq"],
                resistor_width_um=process["resistor_width_um"])
        else:
            area = capacitor_area_um2(value, cap_ff_um2=process["cap_ff_um2"])
        rows.append({
            "name": name, "kind": letter, "value": value, "area_um2": area,
            "dut": is_dut, "counted": is_dut and name not in excluded,
        })
    rows.sort(key=lambda r: (-r["area_um2"], r["name"]))
    counted = [r for r in rows if r["counted"]]
    gate_area = transistor_gate_area_um2(decks)
    return {
        "decks": sorted(decks),
        "process": dict(process),
        "rows": rows,
        "passive_area_um2": sum(r["area_um2"] for r in counted),
        "resistor_area_um2": sum(r["area_um2"] for r in counted if r["kind"] == "R"),
        "capacitor_area_um2": sum(r["area_um2"] for r in counted if r["kind"] == "C"),
        "gate_area_um2": gate_area,
        "excluded": sorted(excluded),
    }


def _format_value(row):
    return (f"{row['value']:.4g} ohm" if row["kind"] == "R"
            else f"{row['value'] * 1e12:.4g} pF")


def print_report(report, *, top=None):
    process = report["process"]
    print(f"decks: {', '.join(report['decks'])}")
    print(f"process: {process['sheet_ohm_sq']:g} ohm/sq poly at "
          f"{process['resistor_width_um']:g} um wide, "
          f"{process['cap_ff_um2']:g} fF/um2 capacitor")
    if report["excluded"]:
        print(f"excluded from totals: {', '.join(report['excluded'])}")
    rows = [r for r in report["rows"] if r["counted"]]
    shown = rows if top is None else rows[:top]
    width = max((len(r["name"]) for r in report["rows"]), default=4)
    print(f"\nDUT passives ({len(rows)} elements"
          + (f", top {len(shown)} shown" if len(shown) < len(rows) else "") + "):")
    print(f"  {'name':<{width}}  {'value':>12}  {'area':>12}  share")
    for row in shown:
        share = (row["area_um2"] / report["passive_area_um2"] * 100.0
                 if report["passive_area_um2"] else 0.0)
        print(f"  {row['name']:<{width}}  {_format_value(row):>12}  "
              f"{row['area_um2']:>9.0f} um2  {share:4.1f}%")

    testbench = [r for r in report["rows"] if not r["dut"]]
    if testbench:
        print(f"\ntestbench-only elements (not counted): "
              f"{', '.join(r['name'] for r in testbench)}")

    print(f"\nresistors  ~{report['resistor_area_um2']:>9.0f} um2")
    print(f"capacitors ~{report['capacitor_area_um2']:>9.0f} um2")
    print(f"passives   ~{report['passive_area_um2']:>9.0f} um2")
    gate = report["gate_area_um2"]
    if gate:
        print(f"gate area  ~{gate:>9.0f} um2  (drawn W*L; real active is 2-3x)")
        print(f"passive/active ratio ~{report['passive_area_um2'] / gate:.1f}x")
