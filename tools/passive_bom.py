#!/usr/bin/env python3
"""Passive bill of materials with silicon-area estimates for a generated design.

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

Example::

    tools/passive_bom.py --generator tsmc28_mdac_ota_gen \\
        --exclude CL1,CL2 --top 8
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "examples", ROOT / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


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


def transistor_gate_area_um2(generator):
    """Total drawn gate area over the generator's sizing table.

    Gate area alone understates the real active footprint (diffusion, contacts
    and routing typically make it 2-3x), so the passive/active ratio printed
    against it is a lower bound on how passive-dominated the layout is."""
    sizes = getattr(generator, "SZ", None)
    if not isinstance(sizes, dict):
        return None
    mult = getattr(generator, "MULT", {}) or {}
    return sum(width * length * int(mult.get(name, 1))
               for name, (width, length) in sizes.items())


def build_report(generator, *, exclude=(), **process):
    decks = generator.all_testbenches()
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
    gate_area = transistor_gate_area_um2(generator)
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


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--generator", required=True,
                        help="importable generator module exposing all_testbenches()")
    parser.add_argument("--exclude", default="",
                        help="comma-separated DUT elements to leave out of the "
                             "totals (e.g. a specified external load)")
    parser.add_argument("--top", type=int, help="only list the N largest")
    parser.add_argument("--sheet-ohm-sq", type=float,
                        default=DEFAULTS["sheet_ohm_sq"])
    parser.add_argument("--resistor-width-um", type=float,
                        default=DEFAULTS["resistor_width_um"])
    parser.add_argument("--cap-ff-um2", type=float, default=DEFAULTS["cap_ff_um2"])
    parser.add_argument("--json", action="store_true", help="emit JSON instead")
    args = parser.parse_args(argv)

    generator = importlib.import_module(args.generator)
    if not hasattr(generator, "all_testbenches"):
        raise SystemExit(
            f"{args.generator} exposes no all_testbenches(); nothing to inventory")
    report = build_report(
        generator,
        exclude=[n.strip() for n in args.exclude.split(",") if n.strip()],
        sheet_ohm_sq=args.sheet_ohm_sq,
        resistor_width_um=args.resistor_width_um,
        cap_ff_um2=args.cap_ff_um2)
    if args.json:
        import json
        print(json.dumps(report, indent=2))
    else:
        print_report(report, top=args.top)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
