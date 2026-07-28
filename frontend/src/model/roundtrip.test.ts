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
import { isCircuitJson } from "./examples";
import { circuitJsonToGraph } from "./toGraph";
import { graphToCircuitJson } from "./toJson";
import { deepEqual } from "./util";

const FIX_DIR = join(dirname(fileURLToPath(import.meta.url)), "..", "..", "..", "examples");

function loadFixtures(): { name: string; json: CircuitJson }[] {
  // `examples/` also holds explore configs and signoff manifests, which are not
  // circuits and carry no `solved` array. Select by shape so a new deck is
  // covered automatically and a new manifest is not mistaken for one.
  return readdirSync(FIX_DIR)
    .filter((f) => f.endsWith(".json"))
    .sort()
    .map((f) => ({
      name: f,
      json: JSON.parse(readFileSync(join(FIX_DIR, f), "utf-8")) as unknown,
    }))
    .filter((entry): entry is { name: string; json: CircuitJson } =>
      isCircuitJson(entry.json));
}

const fixtures = loadFixtures();

/** Round-trip a circuit JSON through the graph and back. */
function roundtrip(json: CircuitJson): CircuitJson {
  const { graph, rest } = circuitJsonToGraph(json);
  return graphToCircuitJson(graph, rest);
}

describe("fixture inventory", () => {
  it("covers every circuit in examples/, not a curated subset", () => {
    // The previous corpus was a copy under src/model/__fixtures__/. It drifted
    // from its originals and never gained the circuits added after it was made,
    // so both the palette and this gate silently stopped covering them. Reading
    // the directory means a new deck is covered the day it lands.
    const names = fixtures.map((f) => f.name);
    expect(names.length).toBeGreaterThanOrEqual(25);
    for (const expected of [
      "afe_explore.json",          // OTFT
      "sky130_fd_ota.json",        // SKY130
      "freepdk45_fd_ota.json",     // FreePDK45
      "tsmc28hpcp_5t_ota.json",    // TSMC28 — absent from the old copy entirely
      "tsmc28hpcp_chopper.json",
    ]) {
      expect(names).toContain(expected);
    }
  });

  it("excludes files that are not circuits", () => {
    // Explore configs and signoff manifests live in the same directory and have
    // no `solved` array; loading one as a circuit throws.
    const names = fixtures.map((f) => f.name);
    expect(names).not.toContain("freepdk45_sar6_explore.json");
    expect(names).not.toContain("tsmc28hpcp_mdac_ota_signoff.json");
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

describe("strict MOS binding export", () => {
  it("rejects a MOS without an explicit model", () => {
    const { graph, rest } = circuitJsonToGraph(fixtures.find(
      (f) => f.name === "single_stage.json",
    )!.json);
    const mos = graph.nodes.find((node) => node.kind === "mosfet")!;
    if (mos.kind === "mosfet") mos.model = undefined;
    expect(() => graphToCircuitJson(graph, rest)).toThrow(
      /every MOS requires an explicit PDK\/model binding/,
    );
  });

  it("rejects an unqualified model key", () => {
    const { graph, rest } = circuitJsonToGraph(fixtures.find(
      (f) => f.name === "single_stage.json",
    )!.json);
    const mos = graph.nodes.find((node) => node.kind === "mosfet")!;
    if (mos.kind === "mosfet") mos.model = "pmos";
    expect(() => graphToCircuitJson(graph, rest)).toThrow(
      /fully qualified "pdk\.model"/,
    );
  });
});
