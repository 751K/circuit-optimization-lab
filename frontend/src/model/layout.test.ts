/**
 * Schematic auto-layout tests.
 *
 * The layout is a heuristic, so these do not pin coordinates. They assert the
 * properties that make the drawing readable and that a refactor could silently
 * lose:
 *
 *  - the supply stack really is the vertical axis (VDD above its loads, above
 *    the input pair, above the tail, above ground) — the whole point of the
 *    module, checked against a 5T OTA whose correct drawing is not in dispute;
 *  - a single-gate tap lands beside the gate it drives, not across the drawing;
 *  - no two nodes are placed on top of each other, over the whole example
 *    corpus — a stacked pair is invisible on the canvas, not merely untidy;
 *  - it is deterministic, since the same circuit must not redraw differently
 *    between loads;
 *  - it declines rather than guesses when there is no rail to rank against.
 *
 * `isotonic` / `placeRow` are the numerical core and are checked directly.
 */
import { describe, expect, it } from "vitest";
import { readFileSync, readdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import type { CircuitJson } from "./circuit";
import { isCircuitJson } from "./examples";
import { circuitJsonToGraph } from "./toGraph";
import { COL_W, isotonic, placeRow, schematicLayout, type LayoutNode } from "./layout";

const FIX_DIR = join(dirname(fileURLToPath(import.meta.url)), "..", "..", "..", "examples");

function examples(): { name: string; json: CircuitJson }[] {
  return readdirSync(FIX_DIR)
    .filter((f) => f.endsWith(".json"))
    .sort()
    .map((f) => ({ name: f, json: JSON.parse(readFileSync(join(FIX_DIR, f), "utf-8")) as unknown }))
    .filter((e): e is { name: string; json: CircuitJson } => isCircuitJson(e.json));
}

describe("isotonic (pool-adjacent-violators)", () => {
  it("returns a non-decreasing sequence", () => {
    const out = isotonic([5, 1, 4, 2, 8, 3]);
    for (let i = 1; i < out.length; i++) expect(out[i]!).toBeGreaterThanOrEqual(out[i - 1]!);
  });

  it("leaves an already non-decreasing sequence untouched", () => {
    expect(isotonic([1, 2, 2, 7])).toEqual([1, 2, 2, 7]);
  });

  it("pools a violating pair to their mean, conserving the sum", () => {
    // The L2 projection of [3, 1] onto {a <= b} is [2, 2], not [1, 1] or [3, 3].
    expect(isotonic([3, 1])).toEqual([2, 2]);
    const v = [9, 1, 5, 2];
    const sum = (xs: number[]): number => xs.reduce((s, x) => s + x, 0);
    expect(sum(isotonic(v))).toBeCloseTo(sum(v), 9);
  });

  it("beats every order-preserving alternative in squared error", () => {
    const desired = [40, 10, 30];
    const fit = isotonic(desired);
    const err = (xs: number[]): number =>
      xs.reduce((s, x, i) => s + (x - desired[i]!) ** 2, 0);
    // Brute-force a grid of monotone candidates; none may do better.
    for (let a = 0; a <= 50; a += 2) {
      for (let b = a; b <= 50; b += 2) {
        for (let c = b; c <= 50; c += 2) {
          expect(err(fit)).toBeLessThanOrEqual(err([a, b, c]) + 1e-9);
        }
      }
    }
  });
});

describe("placeRow", () => {
  it("honours the minimum gap even when every node wants the same column", () => {
    const out = placeRow([100, 100, 100], 220);
    for (let i = 1; i < out.length; i++) {
      expect(out[i]! - out[i - 1]!).toBeGreaterThanOrEqual(220 - 1e-9);
    }
  });

  it("leaves a row that already clears the gap exactly where it asked to be", () => {
    expect(placeRow([0, 300, 600], 220)).toEqual([0, 300, 600]);
  });

  it("keeps the row's order — a node never overtakes the one before it", () => {
    const out = placeRow([500, 0, 250], 100);
    for (let i = 1; i < out.length; i++) expect(out[i]!).toBeGreaterThan(out[i - 1]!);
  });
});

/** The 5T OTA, whose correct drawing is not a matter of taste. */
function fiveT(): { json: CircuitJson; y: (id: string) => number; x: (id: string) => number } {
  const json = JSON.parse(
    readFileSync(join(FIX_DIR, "sky130_5t_ota.json"), "utf-8"),
  ) as CircuitJson;
  const { graph } = circuitJsonToGraph(json);
  const pos = new Map(graph.nodes.map((n) => [n.id, n.position]));
  return { json, y: (id) => pos.get(id)![1], x: (id) => pos.get(id)![0] };
}

describe("schematicLayout ranks a 5T OTA by its supply stack", () => {
  it("draws VDD, the PMOS mirror, the input pair, the tail and GND in that order", () => {
    const { y } = fiveT();
    // VDD -> M3/M4 (pmos load) -> M1/M2 (input pair) -> M5 (tail) -> GND.
    expect(y("VDD")).toBeLessThan(y("M3"));
    expect(y("M3")).toBe(y("M4"));
    expect(y("M3")).toBeLessThan(y("M1"));
    expect(y("M1")).toBe(y("M2"));
    expect(y("M1")).toBeLessThan(y("M5"));
    expect(y("M5")).toBeLessThan(y("GND"));
  });

  it("puts each half of the pair under the load that feeds it", () => {
    const { x } = fiveT();
    // M3 mirrors into M1 through n1; M4 into M2 through vout. Whichever way
    // round the two branches come out, each load sits over its own device.
    expect(Math.abs(x("M3") - x("M1"))).toBeLessThan(COL_W);
    expect(Math.abs(x("M4") - x("M2"))).toBeLessThan(COL_W);
    expect(x("M1")).not.toBe(x("M2"));
  });

  it("places a single-gate tap one column from the gate it drives, on its row", () => {
    const { x, y } = fiveT();
    // vinp -> M1.G, vinn -> M2.G, vbias -> M5.G. A tap that drifts anywhere else
    // draws a wire across the whole schematic for a one-net connection.
    for (const [tap, dev] of [["vinp", "M1"], ["vinn", "M2"], ["vbias", "M5"]] as const) {
      expect(y(tap)).toBe(y(dev));
      expect(x(dev) - x(tap)).toBeCloseTo(COL_W, 6);
    }
  });

  it("puts the tap on the other side of a mirrored device", () => {
    // Mirroring moves the gate to the right, so a tap left of the device would
    // have to loop around the body to reach it — the exact wire a tap exists to
    // avoid. Drawn as a pair, M1 keeps its input on the left and M2 takes its
    // own on the right.
    const json = JSON.parse(
      readFileSync(join(FIX_DIR, "sky130_5t_ota.json"), "utf-8"),
    ) as CircuitJson;
    json.ui = { ...(json.ui ?? {}), mirrored: ["M2"] };
    const { graph } = circuitJsonToGraph(json);
    const pos = new Map(graph.nodes.map((n) => [n.id, n.position]));
    expect(pos.get("vinp")![0]).toBeLessThan(pos.get("M1")![0]);
    expect(pos.get("vinn")![0]).toBeGreaterThan(pos.get("M2")![0]);
    expect(pos.get("vinn")![1]).toBe(pos.get("M2")![1]);
  });
});

describe("a rail that only drives gates is a tap, not a supply", () => {
  // sky130_chopper declares CLK at 1.8 V — the same potential as VDD — but it
  // switches gates, it does not source current. Ranked as a supply it ties VDD
  // for the top of the drawing, and the two end up centred on the same column
  // with nothing to separate them.
  const chopped: LayoutNode[] = [
    { id: "VDD", kind: "rail", ports: [{ id: "net", net: "VDD" }] },
    { id: "CLK", kind: "rail", ports: [{ id: "net", net: "CLK" }] },
    { id: "GND", kind: "rail", ports: [{ id: "net", net: "GND" }] },
    { id: "MP", kind: "mosfet", ports: [{ id: "D", net: "o" }, { id: "G", net: "CLK" }, { id: "S", net: "VDD" }] },
    { id: "MN", kind: "mosfet", ports: [{ id: "D", net: "o" }, { id: "G", net: "CLK" }, { id: "S", net: "GND" }] },
  ];
  const potentials = new Map([["VDD", 1.8], ["CLK", 1.8], ["GND", 0]]);

  it("does not put the clock on the supply bus beside VDD", () => {
    const placed = schematicLayout(chopped, potentials)!;
    expect(placed.get("CLK")![1]).not.toBe(placed.get("VDD")![1]);
    expect(placed.get("VDD")![1]).toBeLessThan(placed.get("CLK")![1]);
  });

  it("keeps a split supply's two buses apart by grouping what they feed", () => {
    // Nothing forces the two buses apart directly. They separate because the
    // ordering sweeps pull the devices on one rail into adjacent columns, so
    // the two medians end up in different places. Interleaving the consumers in
    // the input is the adversarial case: if the sweeps left them interleaved,
    // both medians would land on the same column and one bus would sit on top
    // of the other.
    const split: LayoutNode[] = [
      { id: "VDDA", kind: "rail", ports: [{ id: "net", net: "VDDA" }] },
      { id: "VDDB", kind: "rail", ports: [{ id: "net", net: "VDDB" }] },
      { id: "GND", kind: "rail", ports: [{ id: "net", net: "GND" }] },
      { id: "MA", kind: "mosfet", ports: [{ id: "D", net: "o" }, { id: "G", net: "g" }, { id: "S", net: "VDDA" }] },
      { id: "MB", kind: "mosfet", ports: [{ id: "D", net: "o" }, { id: "G", net: "g" }, { id: "S", net: "VDDB" }] },
      { id: "MC", kind: "mosfet", ports: [{ id: "D", net: "o" }, { id: "G", net: "g" }, { id: "S", net: "VDDB" }] },
      { id: "MD", kind: "mosfet", ports: [{ id: "D", net: "o" }, { id: "G", net: "g" }, { id: "S", net: "VDDA" }] },
      { id: "MN", kind: "mosfet", ports: [{ id: "D", net: "o" }, { id: "G", net: "g" }, { id: "S", net: "GND" }] },
    ];
    const placed = schematicLayout(split, new Map([["VDDA", 1.8], ["VDDB", 1.8], ["GND", 0]]))!;
    expect(placed.get("VDDA")![1]).toBe(placed.get("VDDB")![1]);
    expect(Math.abs(placed.get("VDDA")![0] - placed.get("VDDB")![0]))
      .toBeGreaterThanOrEqual(COL_W - 1e-6);
  });
});

describe("schematicLayout invariants over the example corpus", () => {
  const corpus = examples();

  it("finds circuits to lay out", () => {
    expect(corpus.length).toBeGreaterThan(20);
  });

  // Roughly a rendered node's box. Two nodes closer than this on both axes
  // overlap on the canvas, which hides one of them entirely — the interesting
  // failure, and one that "no two share a position" would miss by a pixel.
  const NODE_W = 150;
  const NODE_H = 76;

  for (const { name, json } of corpus) {
    it(`${name}: no two nodes overlap`, () => {
      const { graph } = circuitJsonToGraph(json);
      for (let i = 0; i < graph.nodes.length; i++) {
        for (let k = i + 1; k < graph.nodes.length; k++) {
          const a = graph.nodes[i]!;
          const b = graph.nodes[k]!;
          const dx = Math.abs(a.position[0] - b.position[0]);
          const dy = Math.abs(a.position[1] - b.position[1]);
          expect(
            dx < NODE_W && dy < NODE_H,
            `${a.id} at ${a.position} overlaps ${b.id} at ${b.position}`,
          ).toBe(false);
        }
      }
    });
  }

  it("is deterministic: the same circuit twice gives identical coordinates", () => {
    for (const { name, json } of corpus) {
      const a = circuitJsonToGraph(json).graph.nodes.map((n) => `${n.id}:${n.position}`);
      const b = circuitJsonToGraph(json).graph.nodes.map((n) => `${n.id}:${n.position}`);
      expect(a, name).toEqual(b);
    }
  });
});

describe("schematicLayout declines when there is nothing to rank against", () => {
  const twoDevices: LayoutNode[] = [
    { id: "M1", kind: "mosfet", ports: [{ id: "D", net: "a" }, { id: "G", net: "g" }, { id: "S", net: "b" }] },
    { id: "M2", kind: "mosfet", ports: [{ id: "D", net: "b" }, { id: "G", net: "g" }, { id: "S", net: "c" }] },
  ];

  it("returns null with no rail potentials at all", () => {
    expect(schematicLayout(twoDevices, new Map())).toBeNull();
  });

  it("returns null when the only rail is a name nothing in the circuit touches", () => {
    // sc_lpf declares a VDD no element references. Anchoring to it would rank
    // the drawing against a net that is not in it.
    expect(schematicLayout(twoDevices, new Map([["VDD", 1.8]]))).toBeNull();
  });

  it("anchors to the highest rail an element touches, not the highest declared", () => {
    // A deck can carry a rails entry nothing references — a leftover, or a
    // supply used by a sibling testbench. Left in the running it wins the
    // "highest potential" contest, has no conduction path to anything, and the
    // real supply gets demoted from the top bus to a slot beside the device it
    // feeds. So the anchor is chosen among the rails that are actually wired.
    const stack: LayoutNode[] = [
      { id: "VDDX", kind: "rail", ports: [{ id: "net", net: "VDDX" }] },
      { id: "VDDA", kind: "rail", ports: [{ id: "net", net: "VDDA" }] },
      { id: "GND", kind: "rail", ports: [{ id: "net", net: "GND" }] },
      { id: "M1", kind: "mosfet", ports: [{ id: "D", net: "mid" }, { id: "G", net: "g" }, { id: "S", net: "VDDA" }] },
      { id: "M2", kind: "mosfet", ports: [{ id: "D", net: "mid" }, { id: "G", net: "g" }, { id: "S", net: "GND" }] },
    ];
    const placed = schematicLayout(stack, new Map([["VDDX", 5], ["VDDA", 1], ["GND", 0]]))!;
    expect(placed).not.toBeNull();
    // VDDA is the supply: a bus above the stack, not a node level with M1.
    expect(placed.get("VDDA")![1]).toBeLessThan(placed.get("M1")![1]);
    expect(placed.get("M1")![1]).toBeLessThan(placed.get("M2")![1]);
    expect(placed.get("M2")![1]).toBeLessThan(placed.get("GND")![1]);
  });

  it("still ranks against a lone ground, which is a real anchor", () => {
    const placed = schematicLayout(twoDevices, new Map([["c", 0]]));
    expect(placed).not.toBeNull();
    // M2 returns to ground, so it sits below M1.
    expect(placed!.get("M2")![1]).toBeGreaterThan(placed!.get("M1")![1]);
  });
});
