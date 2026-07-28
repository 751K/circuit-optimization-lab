/**
 * Stimulus reading and transient-waveform derivation.
 *
 * The failure being guarded against is silent by construction: a transient with
 * no periodic block converges, returns a flat line, and reports nothing wrong.
 * So the assertions are about the two halves of the fix — that the AC ports are
 * read back out of whichever block they were written in, and that the derived
 * waveform sits on the right rail at the right level with the right phase.
 */
import { describe, expect, it } from "vitest";
import { readFileSync, readdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import type { CircuitJson } from "../model/circuit";
import { isCircuitJson } from "../model/examples";
import {
  acStimulus,
  deriveBlocker,
  deriveTransientPeriodic,
  periodicFor,
  railLevel,
  stimulusReport,
} from "./stimulus";

const FIX_DIR = join(dirname(fileURLToPath(import.meta.url)), "..", "..", "..", "examples");
const load = (name: string): CircuitJson =>
  JSON.parse(readFileSync(join(FIX_DIR, `${name}.json`), "utf-8")) as CircuitJson;

describe("acStimulus reads the input ports whichever block declares them", () => {
  it("resolves an input_drives device to the net on its gate", () => {
    // sky130_5t_ota: {M1: +1, M2: -1}. The *signal* is on the devices; the nets
    // vinp/vinn look like plain DC rails at VCM, which is exactly why reading
    // the JSON makes it look as though only a common-mode level was specified.
    const ports = acStimulus(load("sky130_5t_ota"));
    expect(ports.map((p) => [p.net, p.magnitude, p.via, p.device]))
      .toEqual([["vinp", 1, "gate", "M1"], ["vinn", -1, "gate", "M2"]]);
    expect(ports.every((p) => p.onRail)).toBe(true);
  });

  it("takes an ac_drives entry as a node port", () => {
    const ports = acStimulus(load("periodic_rc"));
    expect(ports).toEqual([
      { net: "VIN", magnitude: 1, via: "node", onRail: true },
    ]);
  });

  it("returns nothing for a circuit driven only through its periodic block", () => {
    // The chopper's excitation is the clock; it has no AC path at all.
    expect(acStimulus(load("sky130_chopper"))).toEqual([]);
  });

  it("orders ports by descending magnitude so a differential pair reads + then −", () => {
    const ports = acStimulus(load("freepdk45_fd_ota"));
    expect(ports.length).toBeGreaterThan(1);
    for (let i = 1; i < ports.length; i++) {
      expect(ports[i - 1]!.magnitude).toBeGreaterThanOrEqual(ports[i]!.magnitude);
    }
  });
});

describe("railLevel", () => {
  it("resolves a bias-key rail through the bias block", () => {
    expect(railLevel(load("sky130_5t_ota"), "vinp")).toBe(0.9); // "VCM" -> 0.9
  });
  it("takes a numeric rail directly", () => {
    expect(railLevel(load("sky130_5t_ota"), "GND")).toBe(0);
  });
  it("is undefined for a net that is not a rail", () => {
    expect(railLevel(load("sky130_5t_ota"), "tail")).toBeUndefined();
  });
});

describe("deriveTransientPeriodic", () => {
  const ota = load("sky130_5t_ota");

  it("drives each AC port with a sine on that rail's own DC level", () => {
    const p = deriveTransientPeriodic(ota, { frequency: 1e4, amplitude: 5e-3 })!;
    expect(p.frequency).toBe(1e4);
    expect(p.node_inputs).toEqual({ vinp: "stim_vinp", vinn: "stim_vinn" });
    const inputs = p.inputs as Record<string, Record<string, unknown>>;
    // Centred on VCM, so the transient starts at the operating point the design
    // was solved for rather than slewing away from zero.
    expect(inputs["stim_vinp"]).toEqual({
      type: "sine", dc: 0.9, amplitude: 5e-3, phase: 0,
    });
    // The negative half of the pair is the same sine, inverted.
    expect(inputs["stim_vinn"]).toEqual({
      type: "sine", dc: 0.9, amplitude: 5e-3, phase: Math.PI,
    });
  });

  it("scales the amplitude by the port's relative magnitude", () => {
    const half: CircuitJson = { ...ota, input_drives: { M1: 0.5, M2: -0.5 } };
    const inputs = deriveTransientPeriodic(half, { frequency: 1e3, amplitude: 2e-3 })!
      .inputs as Record<string, Record<string, unknown>>;
    expect(inputs["stim_vinp"]!.amplitude).toBe(1e-3);
    expect(inputs["stim_vinn"]!.amplitude).toBe(1e-3);
  });

  it("emits one waveform per net, keeping the largest drive on it", () => {
    // Two devices on one gate net is a single-ended drive, not two inputs. The
    // magnitudes deliberately differ: emitting the net twice would let the
    // second write silently overwrite the first, and the run would be excited at
    // a quarter of the requested amplitude with nothing to show for it.
    const shared: CircuitJson = {
      ...ota,
      input_drives: { M1: 1, M2: 0.25 },
      devices: (ota.devices as Record<string, unknown>[]).map((d) =>
        (d.name === "M2" ? { ...d, gate: "vinp" } : d)) as CircuitJson["devices"],
    };
    const p = deriveTransientPeriodic(shared, { frequency: 1e3, amplitude: 4e-3 })!;
    expect(Object.keys(p.node_inputs as object)).toEqual(["vinp"]);
    const inputs = p.inputs as Record<string, Record<string, unknown>>;
    expect(Object.keys(inputs)).toEqual(["stim_vinp"]);
    expect(inputs["stim_vinp"]!.amplitude).toBe(4e-3); // 1.0 x 4 mV, not 0.25 x
  });

  it("declines, with a reason, when there is no AC input to copy", () => {
    const chopper = load("sky130_chopper");
    expect(deriveTransientPeriodic(chopper, { frequency: 1e3, amplitude: 1e-3 })).toBeNull();
    expect(deriveBlocker(chopper)).toMatch(/no AC input/);
  });

  it("declines when the AC input drives an internal node rather than a rail", () => {
    // Forcing a waveform onto a node the circuit solves for would override the
    // circuit instead of driving it.
    const internal: CircuitJson = { ...load("sky130_5t_ota"), input_drives: {}, ac_drives: { tail: 1 } };
    expect(deriveTransientPeriodic(internal, { frequency: 1e3, amplitude: 1e-3 })).toBeNull();
    expect(deriveBlocker(internal)).toMatch(/internal node/);
  });
});

describe("periodicFor", () => {
  it("merges an analyses-scoped block over the top-level one", () => {
    const c: CircuitJson = {
      ...load("sky130_5t_ota"),
      periodic: { frequency: 1, n_points: 11 },
      analyses: { transient: { periodic: { frequency: 999 } } },
    };
    expect(periodicFor(c, "transient")).toEqual({ frequency: 999, n_points: 11 });
    // …and leaves the other analyses on the top-level block.
    expect(periodicFor(c, "pss")).toEqual({ frequency: 1, n_points: 11 });
  });

  it("is null when neither exists", () => {
    expect(periodicFor(load("sky130_5t_ota"), "transient")).toBeNull();
  });
});

describe("stimulusReport flags a run that would be silent", () => {
  it("calls out an AC run with no drives", () => {
    const r = stimulusReport(load("sky130_chopper"), "ac");
    expect(r.silent).toBe(true);
    expect(r.detail).toMatch(/-180 dB/);
  });

  it("calls out a transient with no periodic block", () => {
    const r = stimulusReport(load("sky130_5t_ota"), "transient");
    expect(r.silent).toBe(true);
    expect(r.detail).toMatch(/flat line/);
  });

  it("is quiet once a stimulus exists", () => {
    expect(stimulusReport(load("sky130_5t_ota"), "ac").silent).toBe(false);
    expect(stimulusReport(load("sky130_chopper"), "transient").silent).toBe(false);
  });
});

describe("over the example corpus", () => {
  const corpus = readdirSync(FIX_DIR)
    .filter((f) => f.endsWith(".json"))
    .sort()
    .map((f) => ({ name: f, json: JSON.parse(readFileSync(join(FIX_DIR, f), "utf-8")) as unknown }))
    .filter((e): e is { name: string; json: CircuitJson } => isCircuitJson(e.json));

  it("never derives a waveform onto a net that is not a rail", () => {
    // The one way this could corrupt a circuit: forcing an internal node.
    for (const { name, json } of corpus) {
      const p = deriveTransientPeriodic(json, { frequency: 1e3, amplitude: 1e-3 });
      if (!p) continue;
      const rails = new Set(Object.keys(json.rails ?? {}));
      for (const net of Object.keys(p.node_inputs as object)) {
        expect(rails.has(net), `${name}: ${net} is not a rail`).toBe(true);
      }
    }
  });

  it("derives for every circuit that declares an AC input on a rail", () => {
    const derivable = corpus.filter(({ json }) => deriveBlocker(json) === null);
    expect(derivable.length).toBeGreaterThan(3);
    for (const { name, json } of derivable) {
      expect(deriveTransientPeriodic(json, { frequency: 1e3, amplitude: 1e-3 }), name)
        .not.toBeNull();
    }
  });
});
