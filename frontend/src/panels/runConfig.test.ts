/**
 * Tests for the request-only analysis-default injection (runConfig.ts):
 * defaults for ac/noise, null for the rest; injection never overrides a
 * circuit's own config; no-default analyses are reported (blocked client-side);
 * and the input circuit — what Export JSON serializes — is never mutated.
 */
import { describe, expect, it } from "vitest";
import type { CircuitJson } from "../model/circuit";
import {
  DEFAULT_FREQS,
  defaultAnalysisConfig,
  missingConfigMessage,
  prepareSolveCircuit,
} from "./runConfig";

/** A minimal circuit without an analyses block (like sky130_5t_ota). */
function bareCircuit(): CircuitJson {
  return {
    name: "bare",
    solved: ["OUT"],
    rails: { VDD: "VDD", GND: 0 },
    devices: [],
    resistors: [{ name: "R1", a: "VDD", b: "OUT", R: 1000 }],
    bias: { VDD: 1.8 },
    outputs: ["OUT"],
  };
}

describe("defaultAnalysisConfig", () => {
  it("gives ac and noise the wide log sweep", () => {
    expect(defaultAnalysisConfig("ac")).toEqual({ freqs: { ...DEFAULT_FREQS } });
    expect(defaultAnalysisConfig("noise")).toEqual({ freqs: { ...DEFAULT_FREQS } });
  });
  it("has no default for time/periodic analyses", () => {
    for (const name of ["transient", "pss", "pac", "pnoise", "unknown_future"]) {
      expect(defaultAnalysisConfig(name)).toBeNull();
    }
  });
  it("returns a fresh freqs object each call (no shared mutable state)", () => {
    const a = defaultAnalysisConfig("ac")!;
    const b = defaultAnalysisConfig("ac")!;
    expect(a.freqs).not.toBe(b.freqs);
  });
});

describe("prepareSolveCircuit", () => {
  it("injects the ac default when the circuit has no analyses block", () => {
    const circuit = bareCircuit();
    const prep = prepareSolveCircuit(circuit, ["ac"]);
    expect(prep.injected).toEqual(["ac"]);
    expect(prep.missing).toEqual([]);
    const analyses = prep.circuit.analyses as Record<string, unknown>;
    expect(analyses.ac).toEqual({ freqs: { ...DEFAULT_FREQS } });
  });

  it("never overrides a circuit's own analysis config", () => {
    const own = { freqs: { start: 1, stop: 10, num: 3, scale: "log" } };
    const circuit: CircuitJson = { ...bareCircuit(), analyses: { ac: own } };
    const prep = prepareSolveCircuit(circuit, ["ac"]);
    expect(prep.injected).toEqual([]);
    expect(prep.missing).toEqual([]);
    // untouched: same circuit reference, same config object
    expect(prep.circuit).toBe(circuit);
    expect((prep.circuit.analyses as Record<string, unknown>).ac).toBe(own);
  });

  it("mixes: injects only the unconfigured analysis, keeps the configured one", () => {
    const ownNoise = { freqs: { start: 1, stop: 10, num: 3, scale: "log" }, band: [1, 10] };
    const circuit: CircuitJson = { ...bareCircuit(), analyses: { noise: ownNoise } };
    const prep = prepareSolveCircuit(circuit, ["ac", "noise"]);
    expect(prep.injected).toEqual(["ac"]);
    const analyses = prep.circuit.analyses as Record<string, unknown>;
    expect(analyses.noise).toBe(ownNoise); // untouched, same reference
    expect(analyses.ac).toEqual({ freqs: { ...DEFAULT_FREQS } });
  });

  it("reports no-default analyses as missing (caller blocks the run)", () => {
    const prep = prepareSolveCircuit(bareCircuit(), ["ac", "transient", "pss"]);
    expect(prep.missing).toEqual(["transient", "pss"]);
    expect(prep.injected).toEqual(["ac"]);
  });

  it("a configured no-default analysis is not missing", () => {
    const circuit: CircuitJson = {
      ...bareCircuit(),
      analyses: { transient: { tstop: 1e-3, n_points: 64 } },
    };
    const prep = prepareSolveCircuit(circuit, ["transient"]);
    expect(prep.missing).toEqual([]);
    expect(prep.injected).toEqual([]);
    expect(prep.circuit).toBe(circuit);
  });

  it("never mutates the input circuit (Export JSON stays clean)", () => {
    const circuit = bareCircuit();
    const before = JSON.parse(JSON.stringify(circuit));
    const prep = prepareSolveCircuit(circuit, ["ac", "noise"]);
    expect(circuit).toEqual(before); // original untouched — no analyses key appeared
    expect(circuit.analyses).toBeUndefined();
    expect(prep.circuit).not.toBe(circuit); // the patched copy is a new object
  });

  it("copy-on-inject also leaves an existing analyses object unmutated", () => {
    const analyses: Record<string, unknown> = { noise: { freqs: [1, 2], band: [1, 2] } };
    const circuit: CircuitJson = { ...bareCircuit(), analyses };
    prepareSolveCircuit(circuit, ["ac"]);
    expect(Object.keys(analyses)).toEqual(["noise"]); // no ac key leaked in
  });

  it("passes a malformed non-object analyses through untouched", () => {
    const circuit: CircuitJson = { ...bareCircuit(), analyses: "bogus" };
    const prep = prepareSolveCircuit(circuit, ["ac"]);
    expect(prep.circuit).toBe(circuit);
    expect(prep.injected).toEqual([]);
    expect(prep.missing).toEqual([]);
  });
});

describe("missingConfigMessage", () => {
  it("names the blocked analyses and what they need", () => {
    const msg = missingConfigMessage(["transient"]);
    expect(msg).toContain("transient");
    expect(msg).toContain("tstop");
    expect(missingConfigMessage(["pss", "pac"])).toContain("pss, pac");
  });
});
