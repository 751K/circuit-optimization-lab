/**
 * Core round-trip invariant, run over all 12 example fixtures:
 *
 *   graphToCircuitJson(circuitJsonToGraph(x)) ≈ x
 *
 * "≈" is semantic deep-equality (util.deepEqual): object key order is ignored,
 * numbers compare with a small relative tolerance, and the export-only `ui`
 * block is excluded. Never a string comparison.
 */
import { describe, expect, it } from "vitest";
import { readFileSync, readdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import type { CircuitJson } from "./circuit";
import { circuitJsonToGraph } from "./toGraph";
import { graphToCircuitJson } from "./toJson";
import { deepEqual } from "./util";

const FIX_DIR = join(dirname(fileURLToPath(import.meta.url)), "__fixtures__");

function loadFixtures(): { name: string; json: CircuitJson }[] {
  return readdirSync(FIX_DIR)
    .filter((f) => f.endsWith(".json"))
    .sort()
    .map((f) => ({
      name: f,
      json: JSON.parse(readFileSync(join(FIX_DIR, f), "utf-8")) as CircuitJson,
    }));
}

const fixtures = loadFixtures();

/** Round-trip a circuit JSON through the graph and back. */
function roundtrip(json: CircuitJson): CircuitJson {
  const { graph, rest } = circuitJsonToGraph(json);
  return graphToCircuitJson(graph, rest);
}

describe("fixture inventory", () => {
  it("finds all 12 example fixtures", () => {
    expect(fixtures.map((f) => f.name)).toEqual([
      "afe_explore.json",
      "freepdk45_5t_ota.json",
      "freepdk45_fd_ota.json",
      "periodic_rc.json",
      "resistor_load_stage.json",
      "sc_lpf.json",
      "single_stage.json",
      "sky130_5t_ota.json",
      "sky130_chopper.json",
      "sky130_fd_ota.json",
      "vcvs_amplifier.json",
      "voltage_divider.json",
    ]);
  });
});

describe("round-trip semantic equivalence (allowing only the added ui block)", () => {
  for (const { name, json } of fixtures) {
    it(name, () => {
      const out = roundtrip(json);
      const r = deepEqual(json, out, { ignoreTopLevelKeys: ["ui"] });
      if (!r.equal) throw new Error(`round-trip diverged at ${r.diff}`);
      expect(r.equal).toBe(true);
      // and the export always carries a ui.positions block
      expect(out.ui?.positions).toBeDefined();
    });
  }
});

describe("second-round idempotence  f(g(f(g(x)))) ≈ f(g(x))", () => {
  for (const { name, json } of fixtures) {
    it(name, () => {
      const once = roundtrip(json);
      const twice = roundtrip(once);
      // Compare including ui: positions are deterministic, so the second pass
      // reproduces them exactly.
      const r = deepEqual(once, twice);
      if (!r.equal) throw new Error(`not idempotent at ${r.diff}`);
      expect(r.equal).toBe(true);
    });
  }
});

describe("passthrough: unmodeled blocks survive verbatim", () => {
  it("keeps vsources / vcvs / periodic / analyses / explore / aliases", () => {
    const byName = new Map(fixtures.map((f) => [f.name, f.json]));

    const sc = byName.get("sc_lpf.json")!;
    const scOut = roundtrip(sc);
    expect(deepEqual(sc.vsources, scOut.vsources).equal).toBe(true);
    expect(deepEqual(sc.periodic, scOut.periodic).equal).toBe(true);
    expect(deepEqual(sc.analyses, scOut.analyses).equal).toBe(true);

    const vcvs = byName.get("vcvs_amplifier.json")!;
    const vcvsOut = roundtrip(vcvs);
    expect(deepEqual(vcvs.vcvs, vcvsOut.vcvs).equal).toBe(true);
    expect(deepEqual(vcvs.sizes, vcvsOut.sizes).equal).toBe(true); // empty {} preserved

    const afe = byName.get("afe_explore.json")!;
    const afeOut = roundtrip(afe);
    expect(deepEqual(afe.explore, afeOut.explore).equal).toBe(true);
    expect(deepEqual(afe.aliases, afeOut.aliases).equal).toBe(true);
    expect(deepEqual(afe.transient_inputs, afeOut.transient_inputs).equal).toBe(true);

    const chop = byName.get("sky130_chopper.json")!;
    const chopOut = roundtrip(chop);
    expect(deepEqual(chop.description, chopOut.description).equal).toBe(true);
  });
});
