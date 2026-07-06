/**
 * Editor store action tests: add/move/update/connect/delete, undo/redo, rename
 * (edge-endpoint rewrite + no id collision), net-conflict detection, and the
 * fixture -> store -> export round-trip staying deep-equal to the source.
 */
import { beforeEach, describe, expect, it } from "vitest";
import { readFileSync, readdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { useEditor } from "./store";
import { circuitJsonToGraph } from "../model/toGraph";
import { graphToCircuitJson } from "../model/toJson";
import { deepEqual } from "../model/util";
import type { CircuitJson, MosfetNode } from "../model";

const FIX_DIR = join(dirname(fileURLToPath(import.meta.url)), "..", "model", "__fixtures__");

/** Reset the singleton store to a clean document before each test. */
function reset(): void {
  useEditor.setState({
    graph: { nodes: [], edges: [] },
    rest: {},
    selection: { nodes: [], edges: [] },
    past: [],
    future: [],
    netError: null,
  });
}

const s = () => useEditor.getState();

describe("store: add / move / update / delete", () => {
  beforeEach(reset);

  it("addNode appends a node, selects it, and auto-names without collision", () => {
    const id1 = s().addNode("mosfet", [0, 0]);
    expect(id1).toBe("M1");
    const id2 = s().addNode("mosfet", [0, 0]);
    expect(id2).toBe("M2");
    expect(s().graph.nodes.map((n) => n.id)).toEqual(["M1", "M2"]);
    expect(s().selection.nodes).toEqual(["M2"]);
  });

  it("addNode of each kind uses the right prefix", () => {
    s().addNode("resistor", [0, 0]);
    s().addNode("capacitor", [0, 0]);
    s().addNode("rail", [0, 0]);
    s().addNode("output", [0, 0]);
    const ids = s().graph.nodes.map((n) => n.id);
    expect(ids).toContain("R1");
    expect(ids).toContain("C1");
    expect(ids).toContain("V1");
    expect(ids).toContain("__out_0");
  });

  it("moveNode updates a node's position", () => {
    const id = s().addNode("resistor", [0, 0]);
    s().moveNode(id, [123, 456]);
    expect(s().graph.nodes.find((n) => n.id === id)!.position).toEqual([123, 456]);
  });

  it("updateNodeProps patches props but not id/name", () => {
    const id = s().addNode("mosfet", [0, 0]);
    s().updateNodeProps(id, { W: 42, L: 0.25 } as Partial<MosfetNode>);
    // an id in the patch is ignored (rename must go through renameNode)
    s().updateNodeProps(id, { id: "hacked" } as never);
    const n = s().graph.nodes.find((x) => x.id === id) as MosfetNode;
    expect(n.W).toBe(42);
    expect(n.L).toBe(0.25);
    expect(s().graph.nodes.some((x) => x.id === "hacked")).toBe(false);
  });

  it("deleteNodes removes a node and its incident edges", () => {
    const a = s().addNode("resistor", [0, 0]);
    const b = s().addNode("resistor", [0, 0]);
    s().connect({ node: a, port: "b" }, { node: b, port: "a" });
    expect(s().graph.edges.length).toBe(1);
    s().deleteNodes([a]);
    expect(s().graph.nodes.map((n) => n.id)).toEqual([b]);
    expect(s().graph.edges.length).toBe(0);
  });

  it("deleteSelection clears the current selection's nodes and edges", () => {
    const a = s().addNode("resistor", [0, 0]);
    const b = s().addNode("resistor", [0, 0]);
    s().connect({ node: a, port: "b" }, { node: b, port: "a" });
    const edgeId = s().graph.edges[0]!.id;
    s().setSelection({ nodes: [a], edges: [edgeId] });
    s().deleteSelection();
    expect(s().graph.nodes.map((n) => n.id)).toEqual([b]);
    expect(s().graph.edges.length).toBe(0);
  });
});

describe("store: connect", () => {
  beforeEach(reset);

  it("adds an edge between two ports", () => {
    const a = s().addNode("resistor", [0, 0]);
    const b = s().addNode("resistor", [0, 0]);
    s().connect({ node: a, port: "b" }, { node: b, port: "a" });
    expect(s().graph.edges.length).toBe(1);
    const e = s().graph.edges[0]!;
    expect(e.source).toEqual({ node: a, port: "b" });
    expect(e.target).toEqual({ node: b, port: "a" });
  });

  it("rejects a duplicate edge (either direction)", () => {
    const a = s().addNode("resistor", [0, 0]);
    const b = s().addNode("resistor", [0, 0]);
    s().connect({ node: a, port: "b" }, { node: b, port: "a" });
    s().connect({ node: a, port: "b" }, { node: b, port: "a" });
    s().connect({ node: b, port: "a" }, { node: a, port: "b" }); // reversed
    expect(s().graph.edges.length).toBe(1);
  });

  it("rejects a self-loop on the same port", () => {
    const a = s().addNode("resistor", [0, 0]);
    s().connect({ node: a, port: "a" }, { node: a, port: "a" });
    expect(s().graph.edges.length).toBe(0);
  });
});

describe("store: rename", () => {
  beforeEach(reset);

  it("renames a node, its name, and every edge endpoint", () => {
    const a = s().addNode("mosfet", [0, 0]);
    const b = s().addNode("resistor", [0, 0]);
    s().connect({ node: a, port: "D" }, { node: b, port: "a" });
    s().setSelection({ nodes: [a], edges: [] });
    const eff = s().renameNode(a, "MPU");
    expect(eff).toBe("MPU");
    const n = s().graph.nodes.find((x) => x.id === "MPU") as MosfetNode;
    expect(n.name).toBe("MPU");
    expect(s().graph.edges[0]!.source.node).toBe("MPU");
    // selection followed the rename
    expect(s().selection.nodes).toEqual(["MPU"]);
  });

  it("refuses to rename onto an existing id (keeps the old id)", () => {
    const a = s().addNode("resistor", [0, 0]); // R1
    const b = s().addNode("resistor", [0, 0]); // R2
    const eff = s().renameNode(b, a); // collide with R1
    expect(eff).toBe(b);
    expect(s().graph.nodes.map((n) => n.id).sort()).toEqual(["R1", "R2"]);
  });

  it("keeps a rail's net in sync when renamed", () => {
    const v = s().addNode("rail", [0, 0]); // V1
    s().renameNode(v, "VDD");
    const rail = s().graph.nodes.find((n) => n.id === "VDD")!;
    expect(rail.kind).toBe("rail");
    if (rail.kind === "rail") expect(rail.net).toBe("VDD");
  });
});

describe("store: undo / redo", () => {
  beforeEach(reset);

  it("undo reverts the last edit; redo reapplies it", () => {
    s().addNode("resistor", [0, 0]); // R1
    s().addNode("resistor", [0, 0]); // R2
    expect(s().graph.nodes.length).toBe(2);
    s().undo();
    expect(s().graph.nodes.map((n) => n.id)).toEqual(["R1"]);
    s().undo();
    expect(s().graph.nodes.length).toBe(0);
    s().redo();
    expect(s().graph.nodes.map((n) => n.id)).toEqual(["R1"]);
    s().redo();
    expect(s().graph.nodes.map((n) => n.id)).toEqual(["R1", "R2"]);
  });

  it("a new edit after undo clears the redo stack", () => {
    s().addNode("resistor", [0, 0]); // R1
    s().addNode("resistor", [0, 0]); // R2
    s().undo(); // back to [R1]
    s().addNode("capacitor", [0, 0]); // C1 -> future cleared
    expect(s().future.length).toBe(0);
    s().redo(); // no-op
    expect(s().graph.nodes.map((n) => n.id)).toEqual(["R1", "C1"]);
  });

  it("undo is a no-op on empty history", () => {
    expect(() => s().undo()).not.toThrow();
    expect(s().graph.nodes.length).toBe(0);
  });
});

describe("store: net-conflict detection", () => {
  beforeEach(reset);

  it("flags a double-rail short and records the offending edges", () => {
    const vdd = s().addNode("rail", [0, 0]); // V1
    s().renameNode(vdd, "VDD");
    const gnd = s().addNode("rail", [0, 0]); // V1 again (VDD taken) -> actually V1
    s().renameNode(gnd, "GND");
    const r = s().addNode("resistor", [0, 0]); // R1
    // tie both rails to the same resistor terminal -> one component, two rails
    s().connect({ node: "VDD", port: "net" }, { node: r, port: "a" });
    s().connect({ node: "GND", port: "net" }, { node: r, port: "a" });
    expect(s().netError).not.toBeNull();
    expect(s().netError!.message).toMatch(/net conflict/i);
    expect(s().netError!.edgeIds.length).toBeGreaterThan(0);
  });

  it("clears the net error once the short is removed", () => {
    const vdd = s().addNode("rail", [0, 0]);
    s().renameNode(vdd, "VDD");
    const gnd = s().addNode("rail", [0, 0]);
    s().renameNode(gnd, "GND");
    const r = s().addNode("resistor", [0, 0]);
    s().connect({ node: "VDD", port: "net" }, { node: r, port: "a" });
    s().connect({ node: "GND", port: "net" }, { node: r, port: "a" });
    expect(s().netError).not.toBeNull();
    // delete one shorting edge
    const bad = s().netError!.edgeIds[0]!;
    s().deleteEdges([bad]);
    expect(s().netError).toBeNull();
  });
});

describe("store: load / new", () => {
  beforeEach(reset);

  it("loadCircuit populates the graph and is undo-able", () => {
    const json = JSON.parse(
      readFileSync(join(FIX_DIR, "voltage_divider.json"), "utf-8"),
    ) as CircuitJson;
    s().loadCircuit(json);
    expect(s().graph.nodes.length).toBeGreaterThan(0);
    expect(s().rest.name).toBe("ideal_vsource_divider");
    s().undo();
    expect(s().graph.nodes.length).toBe(0);
  });

  it("newCircuit clears to an empty doc with the given name", () => {
    s().addNode("resistor", [0, 0]);
    s().newCircuit("blank");
    expect(s().graph.nodes.length).toBe(0);
    expect(s().rest.name).toBe("blank");
  });
});

describe("store: fixture -> store -> export round-trip", () => {
  const files = readdirSync(FIX_DIR)
    .filter((f) => f.endsWith(".json"))
    .sort();

  for (const f of files) {
    it(`${f} exports deep-equal to source (ignoring ui)`, () => {
      reset();
      const json = JSON.parse(readFileSync(join(FIX_DIR, f), "utf-8")) as CircuitJson;
      s().loadCircuit(json);
      const out = s().exportJson();
      const r = deepEqual(json, out, { ignoreTopLevelKeys: ["ui"] });
      if (!r.equal) throw new Error(`round-trip diverged at ${r.diff}`);
      expect(r.equal).toBe(true);
    });
  }

  it("matches the raw model-layer round-trip exactly", () => {
    // store export must equal the F1 mapping composed directly.
    reset();
    const json = JSON.parse(
      readFileSync(join(FIX_DIR, "sky130_fd_ota.json"), "utf-8"),
    ) as CircuitJson;
    s().loadCircuit(json);
    const viaStore = s().exportJson();
    const { graph, rest } = circuitJsonToGraph(json);
    const viaModel = graphToCircuitJson(graph, rest);
    expect(deepEqual(viaStore, viaModel).equal).toBe(true);
  });
});
