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
import { isCircuitJson } from "../model/examples";
import type { CircuitJson, MosfetNode } from "../model";

const FIX_DIR = join(dirname(fileURLToPath(import.meta.url)), "..", "..", "..", "examples");

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
  // `examples/` also holds explore configs and signoff manifests; select the
  // circuits by shape rather than by an allow-list that would go stale.
  const files = readdirSync(FIX_DIR)
    .filter((f) => f.endsWith(".json"))
    .sort()
    .filter((f) =>
      isCircuitJson(JSON.parse(readFileSync(join(FIX_DIR, f), "utf-8"))));

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

describe("store: relayout", () => {
  beforeEach(reset);

  function ota(): CircuitJson {
    return JSON.parse(
      readFileSync(join(FIX_DIR, "sky130_5t_ota.json"), "utf-8"),
    ) as CircuitJson;
  }
  const yOf = (id: string): number =>
    s().graph.nodes.find((n) => n.id === id)!.position[1];

  it("restores the supply ranking after the positions have been scrambled", () => {
    s().loadCircuit(ota());
    // Pile every node onto one point, as a hand-edited (or legacy kind-column)
    // circuit effectively does to the ranking.
    useEditor.setState({
      graph: {
        nodes: s().graph.nodes.map((n) => ({ ...n, position: [0, 0] as [number, number] })),
        edges: s().graph.edges,
      },
    });
    s().relayout();
    expect(yOf("VDD")).toBeLessThan(yOf("M3"));
    expect(yOf("M3")).toBeLessThan(yOf("M1"));
    expect(yOf("M1")).toBeLessThan(yOf("M5"));
    expect(yOf("M5")).toBeLessThan(yOf("GND"));
  });

  it("reproduces the import layout exactly, and is undo-able", () => {
    s().loadCircuit(ota());
    const onImport = s().graph.nodes.map((n) => `${n.id}:${n.position}`);
    useEditor.setState({
      graph: {
        nodes: s().graph.nodes.map((n) => ({ ...n, position: [7, 7] as [number, number] })),
        edges: s().graph.edges,
      },
    });
    s().relayout();
    // Idempotent: the action reads the resolved nets while import reads each
    // port's remembered `originalNet`, and the two must agree — otherwise
    // pressing Tidy on an untouched circuit would move it.
    expect(s().graph.nodes.map((n) => `${n.id}:${n.position}`)).toEqual(onImport);
    s().undo();
    expect(s().graph.nodes.every((n) => n.position[0] === 7)).toBe(true);
  });

  it("changes nothing when the circuit has no rail to rank against", () => {
    // Two bare devices and no rails: there is no supply stack to draw, so the
    // action must decline rather than pile everything on the origin.
    s().addNode("mosfet", [10, 20]);
    s().addNode("mosfet", [30, 40]);
    const before = s().graph.nodes.map((n) => `${n.id}:${n.position}`);
    s().relayout();
    expect(s().graph.nodes.map((n) => `${n.id}:${n.position}`)).toEqual(before);
  });

  it("bumps viewEpoch so the canvas refits, and only on a wholesale change", () => {
    s().loadCircuit(ota());
    const afterLoad = s().viewEpoch;
    s().moveNode("M1", [1, 2]);
    expect(s().viewEpoch).toBe(afterLoad); // an edit must not yank the viewport
    s().relayout();
    expect(s().viewEpoch).toBeGreaterThan(afterLoad);
  });
});

describe("store: mirrorNodes", () => {
  beforeEach(reset);

  function ota(): CircuitJson {
    return JSON.parse(
      readFileSync(join(FIX_DIR, "sky130_5t_ota.json"), "utf-8"),
    ) as CircuitJson;
  }
  const mos = (id: string): MosfetNode =>
    s().graph.nodes.find((n) => n.id === id) as MosfetNode;

  it("flips a device and back, and is undo-able", () => {
    s().loadCircuit(ota());
    expect(mos("M2").mirrored).toBeUndefined();
    s().mirrorNodes(["M2"]);
    expect(mos("M2").mirrored).toBe(true);
    s().mirrorNodes(["M2"]);
    expect(mos("M2").mirrored).toBeUndefined();
    s().undo();
    expect(mos("M2").mirrored).toBe(true);
  });

  it("moves a mixed selection as one group instead of alternating it", () => {
    s().loadCircuit(ota());
    s().mirrorNodes(["M1"]);
    // M1 mirrored, M2 not. Flipping both must make them agree, not swap them.
    s().mirrorNodes(["M1", "M2"]);
    expect(mos("M1").mirrored).toBe(true);
    expect(mos("M2").mirrored).toBe(true);
    s().mirrorNodes(["M1", "M2"]);
    expect(mos("M1").mirrored).toBeUndefined();
    expect(mos("M2").mirrored).toBeUndefined();
  });

  it("ignores ids that are not devices, and never touches the netlist", () => {
    s().loadCircuit(ota());
    const before = JSON.stringify(s().exportJson().devices);
    s().mirrorNodes(["VDD", "__loadcap_0", "M5"]);
    expect(s().graph.nodes.find((n) => n.id === "VDD")).not.toHaveProperty("mirrored");
    expect(mos("M5").mirrored).toBe(true);
    // Orientation is display only: the exported devices block is byte-identical.
    expect(JSON.stringify(s().exportJson().devices)).toBe(before);
  });

  it("round-trips the orientation through ui.mirrored", () => {
    s().loadCircuit(ota());
    s().mirrorNodes(["M4", "M2"]);
    const exported = s().exportJson();
    expect(exported.ui?.mirrored).toEqual(["M2", "M4"]); // sorted, for a stable diff
    s().newCircuit();
    s().loadCircuit(exported);
    expect(mos("M2").mirrored).toBe(true);
    expect(mos("M4").mirrored).toBe(true);
    expect(mos("M1").mirrored).toBeUndefined();
  });

  it("omits ui.mirrored entirely when nothing is mirrored", () => {
    // A circuit nobody flipped must export the bytes it did before the feature.
    s().loadCircuit(ota());
    expect(s().exportJson().ui).not.toHaveProperty("mirrored");
    s().mirrorNodes(["M1"]);
    s().mirrorNodes(["M1"]);
    expect(s().exportJson().ui).not.toHaveProperty("mirrored");
  });
});

describe("store: setAcStimulus", () => {
  beforeEach(reset);

  function ota(): CircuitJson {
    return JSON.parse(
      readFileSync(join(FIX_DIR, "sky130_5t_ota.json"), "utf-8"),
    ) as CircuitJson;
  }
  const mos = (id: string): MosfetNode =>
    s().graph.nodes.find((n) => n.id === id) as MosfetNode;

  it("writes gate drives onto the devices and exports them as input_drives", () => {
    s().loadCircuit(ota());
    s().setAcStimulus({ M1: 0.5, M2: -0.5 }, {});
    expect(mos("M1").inputDrive).toBe(0.5);
    expect(s().exportJson().input_drives).toEqual({ M1: 0.5, M2: -0.5 });
  });

  it("clears a drive the new stimulus does not name", () => {
    // Leaving half of a previous stimulus in place would excite a port nobody
    // selected — the run would be differential when a single-ended one was asked
    // for, and nothing would say so.
    s().loadCircuit(ota());
    expect(mos("M2").inputDrive).toBe(-1);
    s().setAcStimulus({ M1: 1 }, {});
    expect(mos("M1").inputDrive).toBe(1);
    expect(mos("M2").inputDrive).toBeUndefined();
    expect(s().exportJson().input_drives).toEqual({ M1: 1 });
  });

  it("writes node drives to ac_drives, and removes the block when empty", () => {
    s().loadCircuit(ota());
    s().setAcStimulus({}, { vinp: 1 });
    expect(s().exportJson().ac_drives).toEqual({ vinp: 1 });
    s().setAcStimulus({ M1: 1 }, {});
    expect(s().exportJson()).not.toHaveProperty("ac_drives");
  });

  it("is one undo step for the whole stimulus", () => {
    s().loadCircuit(ota());
    s().setAcStimulus({ M1: 0.25 }, { vinn: 2 });
    s().undo();
    expect(mos("M1").inputDrive).toBe(1);
    expect(mos("M2").inputDrive).toBe(-1);
    expect(s().exportJson()).not.toHaveProperty("ac_drives");
  });
});
