/**
 * A graph built from scratch (not via import) must export to a circuit JSON the
 * backend accepts. We build a minimal resistor divider by hand and assert the
 * exported JSON is structurally well-formed (the same shape as the
 * voltage_divider example): required blocks present, nets resolved, elements
 * wired to the intended nets.
 *
 * (The live-backend `validate:true` proof lives in scripts/backend_check.mjs.)
 */
import { describe, expect, it } from "vitest";
import type { CircuitGraph } from "./graph";
import { graphToCircuitJson } from "./toJson";

/**
 * Build:  GND rail --R2-- MID --R1-- IN,  CL on MID, output = MID.
 * (No vsource node kind in v1; a real driver would be added via `rest`.)
 */
function buildDivider(): CircuitGraph {
  const nodes: CircuitGraph["nodes"] = [
    {
      id: "GND",
      kind: "rail",
      net: "GND",
      railValue: 0.0,
      ports: [{ id: "net" }],
      position: [0, 0],
    },
    {
      id: "R1",
      kind: "resistor",
      name: "R1",
      R: 1000,
      ports: [{ id: "a" }, { id: "b" }],
      position: [240, 0],
    },
    {
      id: "R2",
      kind: "resistor",
      name: "R2",
      R: 1000,
      ports: [{ id: "a" }, { id: "b" }],
      position: [240, 120],
    },
    {
      id: "C1",
      kind: "capacitor",
      name: "C1",
      C: 1e-9,
      origin: "capacitors",
      ports: [{ id: "a" }, { id: "b" }],
      position: [480, 0],
    },
    {
      id: "__out_0",
      kind: "output",
      order: 0,
      ports: [{ id: "out" }],
      position: [640, 0],
    },
  ];
  // Nets:  IN = R1.a ;  MID = R1.b = R2.a = C1.a = out ;  GND = R2.b = C1.b = GND.net
  const edges: CircuitGraph["edges"] = [
    // MID star
    { id: "m1", source: { node: "R1", port: "b" }, target: { node: "R2", port: "a" } },
    { id: "m2", source: { node: "R1", port: "b" }, target: { node: "C1", port: "a" } },
    { id: "m3", source: { node: "R1", port: "b" }, target: { node: "__out_0", port: "out" } },
    // GND star
    { id: "g1", source: { node: "GND", port: "net" }, target: { node: "R2", port: "b" } },
    { id: "g2", source: { node: "GND", port: "net" }, target: { node: "C1", port: "b" } },
  ];
  return { nodes, edges };
}

describe("new-graph export", () => {
  const out = graphToCircuitJson(buildDivider(), { name: "hand_divider", bias: {} });

  it("has the required top-level blocks", () => {
    expect(Array.isArray(out.solved)).toBe(true);
    expect(typeof out.rails).toBe("object");
    expect(Array.isArray(out.devices)).toBe(true);
  });

  it("names the rail net GND and keeps it out of solved", () => {
    expect(out.rails.GND).toBe(0.0);
    expect(out.solved).not.toContain("GND");
  });

  it("auto-names the two internal nets (IN, MID) as n1/n2", () => {
    // no original names -> auto n1, n2 ; both are solved (non-rail) nets
    expect(out.solved.length).toBe(2);
    for (const s of out.solved) expect(s).toMatch(/^n\d+$/);
  });

  it("wires R1/R2/C1 to a shared internal (MID) net and to GND", () => {
    const r1 = out.resistors!.find((r) => (r as { name: string }).name === "R1") as {
      a: string;
      b: string;
    };
    const r2 = out.resistors!.find((r) => (r as { name: string }).name === "R2") as {
      a: string;
      b: string;
    };
    const c1 = (out.capacitors![0] as { a: string; b: string });
    // R1.b, R2.a, C1.a all the same net (MID); output equals it too.
    expect(r1.b).toBe(r2.a);
    expect(c1.a).toBe(r1.b);
    expect(out.outputs).toEqual([r1.b]);
    // R2.b and C1.b are GND
    expect(r2.b).toBe("GND");
    expect(c1.b).toBe("GND");
  });

  it("merges passthrough rest (name, bias) and writes ui.positions", () => {
    expect(out.name).toBe("hand_divider");
    expect(out.bias).toEqual({});
    expect(out.ui?.positions?.["R1"]).toEqual([240, 0]);
  });
});
