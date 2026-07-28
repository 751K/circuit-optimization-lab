/**
 * Port-side geometry tests.
 *
 * The failure this guards against is silent and purely visual: a handle on the
 * wrong side does not error, it just makes the wire double back around the node
 * body, which is exactly the spaghetti the schematic layout exists to remove.
 * So the assertions are about direction — the side always faces what the port is
 * wired to.
 */
import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import type { CircuitGraph, CircuitJson, GraphNode, Position } from "../model";
import { circuitJsonToGraph } from "../model";
import { portSides } from "./sides";

const FIX_DIR = join(dirname(fileURLToPath(import.meta.url)), "..", "..", "..", "examples");

/** A rail wired to one mosfet's source, with both positions under test control. */
function railAt(rail: Position, dev: Position): CircuitGraph {
  const nodes: GraphNode[] = [
    { id: "R", kind: "rail", net: "R", railValue: 1.8, ports: [{ id: "net" }], position: rail },
    {
      id: "M", kind: "mosfet", name: "M", W: 1, L: 1,
      ports: [{ id: "D" }, { id: "G" }, { id: "S" }], position: dev,
    },
  ];
  return {
    nodes,
    edges: [{ id: "e", source: { node: "R", port: "net" }, target: { node: "M", port: "S" } }],
  };
}

describe("a single-port node faces what it is wired to", () => {
  it("a supply drawn above the circuit sends its wire down", () => {
    expect(portSides(railAt([0, 0], [0, 300])).get("R")).toEqual({ net: "bottom" });
  });

  it("a ground drawn below the circuit sends its wire up", () => {
    // The case the fixed bottom handle got wrong: the wire used to leave the
    // underside of a ground symbol sitting under the whole drawing and loop back.
    expect(portSides(railAt([0, 900], [0, 300])).get("R")).toEqual({ net: "top" });
  });

  it("a bias tap beside its device sends its wire sideways", () => {
    expect(portSides(railAt([0, 300], [220, 300])).get("R")).toEqual({ net: "right" });
    expect(portSides(railAt([440, 300], [220, 300])).get("R")).toEqual({ net: "left" });
  });

  it("takes the dominant axis when the offset is diagonal", () => {
    // 30 across, 300 down -> vertical wins.
    expect(portSides(railAt([0, 0], [30, 300])).get("R")).toEqual({ net: "bottom" });
    // 300 across, 30 down -> horizontal wins.
    expect(portSides(railAt([0, 0], [300, 30])).get("R")).toEqual({ net: "right" });
  });

  it("says nothing about a port with no wire, so the symbol keeps its default", () => {
    const g = railAt([0, 0], [0, 300]);
    expect(portSides({ nodes: g.nodes, edges: [] }).get("R")).toBeUndefined();
  });
});

describe("a two-terminal element is drawn along the axis it actually spans", () => {
  /** Capacitor C between a node at `top` (port a) and one at `bottom` (port b). */
  function bridge(pa: Position, pb: Position): CircuitGraph {
    const nodes: GraphNode[] = [
      { id: "A", kind: "rail", net: "A", railValue: 1.0, ports: [{ id: "net" }], position: pa },
      { id: "B", kind: "rail", net: "B", railValue: 0.0, ports: [{ id: "net" }], position: pb },
      {
        id: "C", kind: "capacitor", name: "C", C: 1e-12, origin: "capacitors",
        ports: [{ id: "a" }, { id: "b" }], position: [0, 0],
      },
    ];
    return {
      nodes,
      edges: [
        { id: "e1", source: { node: "A", port: "net" }, target: { node: "C", port: "a" } },
        { id: "e2", source: { node: "C", port: "b" }, target: { node: "B", port: "net" } },
      ],
    };
  }

  it("turns vertical for a load capacitor hanging from a node down to ground", () => {
    expect(portSides(bridge([0, 0], [0, 600])).get("C")).toEqual({ a: "top", b: "bottom" });
  });

  it("flips a and b when the b side is the higher one", () => {
    expect(portSides(bridge([0, 600], [0, 0])).get("C")).toEqual({ a: "bottom", b: "top" });
  });

  it("stays horizontal for a coupling capacitor between two stages", () => {
    expect(portSides(bridge([0, 0], [600, 0])).get("C")).toEqual({ a: "left", b: "right" });
  });

  it("stays horizontal when the two ends are equally far apart on both axes", () => {
    // A tie must resolve to the symbol's usual look rather than flipping on noise.
    expect(portSides(bridge([0, 0], [400, 400])).get("C")).toEqual({ a: "left", b: "right" });
  });
});

describe("over a real circuit", () => {
  it("a 5T OTA's ground wires upward and its load capacitor stands vertical", () => {
    const json = JSON.parse(
      readFileSync(join(FIX_DIR, "sky130_5t_ota.json"), "utf-8"),
    ) as CircuitJson;
    const { graph } = circuitJsonToGraph(json);
    const sides = portSides(graph);
    expect(sides.get("GND")).toEqual({ net: "top" });
    expect(sides.get("VDD")).toEqual({ net: "bottom" });
    // The 2 pF load bridges vout down to ground.
    expect(sides.get("__loadcap_0")).toEqual({ a: "top", b: "bottom" });
  });
});
