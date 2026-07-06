/**
 * Adapter identity: domain -> RF -> domain reproduces the node/edge exactly
 * (id, position, and every domain field). Net labels are display-only and do
 * not affect the reverse mapping. Run over hand-built nodes covering all 5
 * kinds and over a real fixture graph.
 */
import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { circuitJsonToGraph } from "../model/toGraph";
import { deepEqual } from "../model/util";
import type { CircuitGraph, CircuitJson, GraphNode } from "../model";
import {
  domainToRf,
  domainToRfEdge,
  domainToRfNode,
  netClass,
  rfToDomainEdge,
  rfToDomainNode,
} from "./adapter";

const FIX_DIR = join(dirname(fileURLToPath(import.meta.url)), "..", "model", "__fixtures__");

function sampleNodes(): GraphNode[] {
  return [
    {
      id: "M1",
      kind: "mosfet",
      name: "M1",
      W: 24,
      L: 0.5,
      hasEmbeddedWL: true,
      model: "sky130.pmos",
      modelKwargs: { vb: 1.8, extract_w: 12 },
      inputDrive: 1,
      ports: [{ id: "D", originalNet: "vout" }, { id: "G", originalNet: "vinp" }, { id: "S", originalNet: "VDD" }],
      position: [10, 20],
    },
    {
      id: "VDD",
      kind: "rail",
      net: "VDD",
      railValue: "VDD",
      biasValue: 1.8,
      ports: [{ id: "net", originalNet: "VDD" }],
      position: [0, 0],
    },
    {
      id: "R1",
      kind: "resistor",
      name: "R1",
      R: 1000,
      ports: [{ id: "a" }, { id: "b" }],
      position: [100, 100],
    },
    {
      id: "C1",
      kind: "capacitor",
      name: "C1",
      C: 2e-12,
      origin: "load_caps",
      ports: [{ id: "a" }, { id: "b" }],
      position: [200, 100],
    },
    {
      id: "__out_0",
      kind: "output",
      order: 0,
      ports: [{ id: "out", originalNet: "vout" }],
      position: [300, 50],
    },
  ];
}

describe("adapter node identity", () => {
  for (const node of sampleNodes()) {
    it(`round-trips a ${node.kind} node`, () => {
      const rf = domainToRfNode(node, { D: "vout" });
      const back = rfToDomainNode(rf);
      const r = deepEqual(node, back);
      if (!r.equal) throw new Error(`diverged at ${r.diff}`);
      expect(r.equal).toBe(true);
      // id + position passthrough
      expect(rf.id).toBe(node.id);
      expect(rf.position).toEqual({ x: node.position[0], y: node.position[1] });
      expect(rf.type).toBe(node.kind);
    });
  }

  it("reflects a moved RF position back into the domain node", () => {
    const [m] = sampleNodes();
    const rf = domainToRfNode(m!);
    const moved = { ...rf, position: { x: 999, y: 888 } };
    const back = rfToDomainNode(moved);
    expect(back.position).toEqual([999, 888]);
  });
});

describe("adapter edge identity", () => {
  it("round-trips an edge (data path)", () => {
    const edge = { id: "e1", source: { node: "M1", port: "D" }, target: { node: "R1", port: "a" } };
    const rf = domainToRfEdge(edge, "vout", false);
    expect(rf.source).toBe("M1");
    expect(rf.sourceHandle).toBe("D");
    expect(rf.target).toBe("R1");
    expect(rf.targetHandle).toBe("a");
    const back = rfToDomainEdge(rf);
    expect(deepEqual(edge, back).equal).toBe(true);
  });

  it("rebuilds an edge from RF fields when data is absent", () => {
    const rf = {
      id: "e2",
      source: "A",
      target: "B",
      sourceHandle: "D",
      targetHandle: "G",
    } as ReturnType<typeof domainToRfEdge>;
    const back = rfToDomainEdge(rf);
    expect(back).toEqual({
      id: "e2",
      source: { node: "A", port: "D" },
      target: { node: "B", port: "G" },
    });
  });

  it("flags a conflict edge and tags kind/net classes", () => {
    const edge = { id: "e", source: { node: "A", port: "x" }, target: { node: "B", port: "y" } };
    // A conflict edge carries the conflict class (plus the signal/net tags now
    // emitted for every edge so the CSS hover/dim rules can key off them).
    const conflict = domainToRfEdge(edge, undefined, true).className!.split(" ");
    expect(conflict).toContain("edge-conflict");
    // A plain signal edge is no longer classless: it gets edge-signal + net-<n>.
    const signal = domainToRfEdge(edge, "n1", false).className!.split(" ");
    expect(signal).not.toContain("edge-conflict");
    expect(signal).toContain("edge-signal");
    expect(signal).toContain("net-n1");
    // A rail-net edge is tagged edge-rail (dimmed) instead of edge-signal.
    const railEdge = domainToRfEdge(edge, "VDD", false, true).className!.split(" ");
    expect(railEdge).toContain("edge-rail");
    expect(railEdge).not.toContain("edge-signal");
  });
});

describe("netClass token", () => {
  it("is undefined for no net", () => {
    expect(netClass(undefined)).toBeUndefined();
  });
  it("passes css-safe net names through unchanged (prefixed)", () => {
    expect(netClass("vout")).toBe("net-vout");
    expect(netClass("n1")).toBe("net-n1");
    expect(netClass("VDD_core")).toBe("net-VDD_core");
    expect(netClass("bias-2")).toBe("net-bias-2");
  });
  it("escapes chars that would break a CSS class selector", () => {
    // A dot or bracket in a net name must not leak into the class as a combinator.
    const cls = netClass("out.p");
    expect(cls?.startsWith("net-")).toBe(true);
    expect(cls).not.toContain(".");
    // Deterministic & round-trippable per input.
    expect(netClass("out.p")).toBe(netClass("out.p"));
    expect(netClass("out.p")).not.toBe(netClass("outXp"));
  });
});

describe("whole-graph adapter over a fixture", () => {
  it("maps every node/edge and reverses each node identically", () => {
    const json = JSON.parse(
      readFileSync(join(FIX_DIR, "sky130_5t_ota.json"), "utf-8"),
    ) as CircuitJson;
    const { graph } = circuitJsonToGraph(json);
    const { nodes, edges } = domainToRf(graph);
    expect(nodes.length).toBe(graph.nodes.length);
    expect(edges.length).toBe(graph.edges.length);
    // reverse-map every node and compare to the original domain node
    const byId = new Map(graph.nodes.map((n) => [n.id, n]));
    for (const rf of nodes) {
      const orig = byId.get(rf.id)!;
      const back = rfToDomainNode(rf);
      const r = deepEqual(orig, back);
      if (!r.equal) throw new Error(`node ${rf.id} diverged at ${r.diff}`);
      expect(r.equal).toBe(true);
    }
    // every edge got a net label (this fixture has no net conflict)
    for (const e of edges) expect(typeof e.label).toBe("string");
  });

  it("labels edges with the resolved net", () => {
    const g: CircuitGraph = {
      nodes: [
        { id: "VDD", kind: "rail", net: "VDD", railValue: 1.8, ports: [{ id: "net" }], position: [0, 0] },
        {
          id: "M1",
          kind: "mosfet",
          name: "M1",
          W: 1,
          L: 1,
          ports: [{ id: "D" }, { id: "G" }, { id: "S" }],
          position: [0, 0],
        },
      ],
      edges: [{ id: "e1", source: { node: "VDD", port: "net" }, target: { node: "M1", port: "S" } }],
    };
    const { edges } = domainToRf(g);
    expect(edges[0]!.label).toBe("VDD");
  });
});
