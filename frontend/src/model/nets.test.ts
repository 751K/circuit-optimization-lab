/**
 * Net-name resolution unit tests: auto-naming, rail-name priority, original-
 * name preservation, and the double-rail conflict error.
 */
import { describe, expect, it } from "vitest";
import type { CircuitGraph, GraphEdge, GraphNode } from "./graph";
import { portKey, resolveNets } from "./toJson";

/** Look up a port's resolved net using the canonical key encoding. */
function netAt(
  res: ReturnType<typeof resolveNets>,
  node: string,
  port: string,
): string | undefined {
  return res.portNet.get(portKey(node, port));
}

function railNode(name: string): GraphNode {
  return {
    id: name,
    kind: "rail",
    net: name,
    railValue: name === "GND" ? 0.0 : name,
    ports: [{ id: "net", originalNet: name }],
    position: [0, 0],
  };
}

function mosfet(id: string, drain: string, gate: string, source: string): GraphNode {
  return {
    id,
    kind: "mosfet",
    name: id,
    W: 1,
    L: 1,
    ports: [
      { id: "D", originalNet: drain },
      { id: "G", originalNet: gate },
      { id: "S", originalNet: source },
    ],
    position: [0, 0],
  };
}

function edge(sn: string, sp: string, tn: string, tp: string): GraphEdge {
  return { id: `${sn}.${sp}->${tn}.${tp}`, source: { node: sn, port: sp }, target: { node: tn, port: tp } };
}

describe("resolveNets", () => {
  it("preserves a rail net name over an auto name", () => {
    const g: CircuitGraph = {
      nodes: [railNode("VDD"), mosfet("M1", "OUT", "IN", "VDD")],
      // tie M1.S to the VDD rail
      edges: [edge("VDD", "net", "M1", "S")],
    };
    const res = resolveNets(g);
    expect(netAt(res, "M1", "S")).toBe("VDD");
    expect(netAt(res, "VDD", "net")).toBe("VDD");
  });

  it("reuses a preserved original net name (no rail in component)", () => {
    // Two mosfets whose drains share an original name "tail" with an edge.
    const g: CircuitGraph = {
      nodes: [mosfet("M1", "tail", "g1", "s1"), mosfet("M2", "tail", "g2", "s2")],
      edges: [edge("M1", "D", "M2", "D")],
    };
    const res = resolveNets(g);
    expect(netAt(res, "M1", "D")).toBe("tail");
    expect(netAt(res, "M2", "D")).toBe("tail");
  });

  it("auto-names an internal net with no original name", () => {
    // A port that never carried an original net name gets n1.
    const g: CircuitGraph = {
      nodes: [
        {
          id: "M1",
          kind: "mosfet",
          name: "M1",
          W: 1,
          L: 1,
          ports: [
            { id: "D" }, // no originalNet
            { id: "G" },
            { id: "S" },
          ],
          position: [0, 0],
        },
      ],
      edges: [],
    };
    const res = resolveNets(g);
    // three distinct components, auto-named n1..n3 in a deterministic order.
    const names = new Set([netAt(res, "M1", "D"), netAt(res, "M1", "G"), netAt(res, "M1", "S")]);
    expect(names.size).toBe(3);
    for (const n of names) expect(n).toMatch(/^n\d+$/);
  });

  it("auto names skip an already-taken name (n1 collision)", () => {
    // One component keeps original name "n1"; the anonymous one must not reuse it.
    const g: CircuitGraph = {
      nodes: [
        mosfet("M1", "n1", "gA", "sA"), // "n1" is a hand-authored original name
        {
          id: "M2",
          kind: "mosfet",
          name: "M2",
          W: 1,
          L: 1,
          ports: [{ id: "D" }, { id: "G" }, { id: "S" }], // all anonymous
          position: [0, 0],
        },
      ],
      edges: [],
    };
    const res = resolveNets(g);
    expect(netAt(res, "M1", "D")).toBe("n1");
    const autos = [netAt(res, "M2", "D"), netAt(res, "M2", "G"), netAt(res, "M2", "S")];
    expect(autos).not.toContain("n1");
    for (const n of autos) expect(n).toMatch(/^n\d+$/);
  });

  it("throws on a double-rail conflict in one component", () => {
    const g: CircuitGraph = {
      nodes: [railNode("VDD"), railNode("GND"), mosfet("M1", "VDD", "g", "GND")],
      edges: [
        edge("VDD", "net", "M1", "D"),
        edge("GND", "net", "M1", "S"),
        // tie the two rails together via the device D-S? Force one component:
        edge("M1", "D", "M1", "S"),
      ],
    };
    expect(() => resolveNets(g)).toThrow(/net conflict/i);
  });
});
