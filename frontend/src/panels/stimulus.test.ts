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
  buildStimulus,
  candidateInputs,
  differentialPairs,
  periodicFor,
  periodicOwner,
  railLevel,
  stimulusOptions,
  stimulusReport,
} from "./stimulus";

/** The block a chosen option writes, for a `periodic` target. */
function periodicFrom(
  circuit: CircuitJson,
  analysis: string,
  match: (label: string) => boolean,
  opts: { frequency: number; amplitude: number },
): Record<string, unknown> | null {
  const option = stimulusOptions(circuit, analysis).find((o) => match(o.label));
  if (!option) return null;
  const patch = buildStimulus(circuit, option, analysis, opts);
  return patch.target === "periodic" ? patch.periodic : null;
}

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

describe("buildStimulus writes a periodic block from a chosen source", () => {
  const ota = load("sky130_5t_ota");
  const AC_PORTS = (l: string) => l.startsWith("Sine on the AC input ports");

  it("drives each AC port with a sine on that rail's own DC level", () => {
    const p = periodicFrom(ota, "transient", AC_PORTS, { frequency: 1e4, amplitude: 5e-3 })!;
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
    const inputs = periodicFrom(half, "transient", AC_PORTS, { frequency: 1e3, amplitude: 2e-3 })!
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
    const p = periodicFrom(shared, "transient", AC_PORTS, { frequency: 1e3, amplitude: 4e-3 })!;
    expect(Object.keys(p.node_inputs as object)).toEqual(["vinp"]);
    const inputs = p.inputs as Record<string, Record<string, unknown>>;
    expect(Object.keys(inputs)).toEqual(["stim_vinp"]);
    expect(inputs["stim_vinp"]!.amplitude).toBe(4e-3); // 1.0 x 4 mV, not 0.25 x
  });

  it("swings a pulse about the rail level, with finite edges", () => {
    const p = periodicFrom(ota, "transient", (l) => l.startsWith("Pulse / clock — vbias"),
      { frequency: 1e5, amplitude: 0.2 })!;
    const w = (p.inputs as Record<string, Record<string, unknown>>)["stim_vbias"]!;
    expect(w.type).toBe("pulse");
    expect(w.low).toBeCloseTo(0.78 - 0.2, 12);   // vbias = VB = 0.78
    expect(w.high).toBeCloseTo(0.78 + 0.2, 12);
    // An ideal step gives the solver no timescale to resolve; the adaptive
    // driver also inserts breakpoints at these edges.
    expect(w.rise as number).toBeGreaterThan(0);
    expect(w.fall).toBe(w.rise);
  });

  it("inverts a pulse by swapping its levels, not by shifting phase", () => {
    const p = periodicFrom(ota, "transient", (l) => l.startsWith("Differential sine"),
      { frequency: 1e3, amplitude: 1e-3 });
    expect(p).not.toBeNull();
    // The differential *pulse* option does not exist, so build one directly to
    // check the inverting half of a two-port pulse.
    const option = stimulusOptions(ota, "transient")
      .find((o) => o.label.startsWith("Pulse / clock — vinn"))!;
    const patch = buildStimulus(ota, { ...option, ports: [{ net: "vinn", magnitude: -1 }] },
      "transient", { frequency: 1e3, amplitude: 0.1 });
    const w = (patch as { periodic: Record<string, unknown> }).periodic
      .inputs as Record<string, Record<string, unknown>>;
    expect(w["stim_vinn"]!.low).toBeCloseTo(1.0, 12);   // 0.9 + 0.1
    expect(w["stim_vinn"]!.high).toBeCloseTo(0.8, 12);  // 0.9 - 0.1
  });
});

describe("stimulusOptions builds a menu from the circuit's own rails", () => {
  it("offers the differential pair, then each rail single-ended", () => {
    const labels = stimulusOptions(load("sky130_5t_ota"), "ac").map((o) => o.label);
    expect(labels[0]).toBe("Differential — vinn / vinp (±1)");
    expect(labels).toContain("Single-ended — vbias (M5.G)");
    // The supplies are not inputs and must never be offered as one.
    expect(labels.join(" ")).not.toMatch(/VDD|GND/);
  });

  it("targets the gates for a rail that gates devices, and the node otherwise", () => {
    const gate = stimulusOptions(load("sky130_5t_ota"), "ac")
      .find((o) => o.label.startsWith("Single-ended — vbias"))!;
    expect(gate.ports).toEqual([{ net: "vbias", magnitude: 1, device: "M5" }]);
    const node = stimulusOptions(load("periodic_rc"), "ac")
      .find((o) => o.label.startsWith("Single-ended — VIN"))!;
    expect(node.ports).toEqual([{ net: "VIN", magnitude: 1 }]);
  });

  it("leads with the existing AC ports for a time-domain analysis", () => {
    const labels = stimulusOptions(load("sky130_5t_ota"), "transient").map((o) => o.label);
    expect(labels[0]).toMatch(/^Sine on the AC input ports/);
    expect(labels).toContain("Pulse / clock — vinp (M1.G)");
  });

  it("offers nothing when every rail is a supply", () => {
    // Nothing to drive that would not override the circuit.
    const bare: CircuitJson = { ...load("sky130_5t_ota"), rails: { VDD: "VDD", GND: 0 } };
    expect(stimulusOptions(bare, "ac")).toEqual([]);
    expect(stimulusOptions(bare, "transient")).toEqual([]);
  });

  it("builds an AC patch that splits gate drives from node drives", () => {
    const ota = load("sky130_5t_ota");
    const diff = stimulusOptions(ota, "ac").find((o) => o.label.startsWith("Differential"))!;
    const patch = buildStimulus(ota, diff, "ac", { frequency: 1e3, amplitude: 1e-3 });
    expect(patch).toEqual({
      target: "ac",
      inputDrives: { M2: 1, M1: -1 },
      acDrives: {},
    });
  });
});

describe("candidateInputs / differentialPairs", () => {
  it("excludes the supply rails and keeps the inputs and biases", () => {
    const c = candidateInputs(load("sky130_5t_ota"));
    expect(c.map((x) => x.net)).toEqual(["vbias", "vinn", "vinp"]);
    expect(c.find((x) => x.net === "vinp")).toEqual({
      net: "vinp", level: 0.9, devices: ["M1"],
    });
  });

  it("pairs two rails at one level whose devices share a source", () => {
    // vinp/vinn both sit at VCM and gate M1/M2, which share `tail`.
    expect(differentialPairs(load("sky130_5t_ota"))).toEqual([["vinn", "vinp"]]);
  });

  it("offers a clock even though it sits at exactly the supply level", () => {
    // sky130_chopper's CLK is at VCK = 1.8 V, the same as VDD, and CLKB at 0 V,
    // the same as ground — but they only gate switches. Excluding them on
    // potential alone hides the obvious stimulus for the whole testbench.
    const nets = candidateInputs(load("sky130_chopper")).map((c) => c.net);
    expect(nets).toContain("CLK");
    expect(nets).toContain("CLKB");
    // …while the supplies, which do carry current, stay out.
    expect(nets).not.toContain("VDD");
    expect(nets).not.toContain("GND");
  });

  it("does not pair rails at the same level that are not a pair", () => {
    // vbias also sits on a gate, but M5's source is GND, not the pair's tail.
    const ota = load("sky130_5t_ota");
    const flat: CircuitJson = { ...ota, bias: { ...(ota.bias as object), VB: 0.9 } };
    expect(differentialPairs(flat)).toEqual([["vinn", "vinp"]]);
  });
});

describe("periodicOwner", () => {
  it("sends pac and pnoise to the pss block they are excited through", () => {
    expect(periodicOwner("transient")).toBe("transient");
    expect(periodicOwner("pss")).toBe("pss");
    expect(periodicOwner("pac")).toBe("pss");
    expect(periodicOwner("pnoise")).toBe("pss");
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

  it("never offers a source that would force a net the circuit solves for", () => {
    // The one way this could corrupt a circuit rather than excite it. Every
    // option, for every analysis, over every deck.
    for (const { name, json } of corpus) {
      const rails = new Set(Object.keys(json.rails ?? {}));
      for (const analysis of ["ac", "noise", "transient", "pss", "pac", "pnoise"]) {
        for (const option of stimulusOptions(json, analysis)) {
          for (const port of option.ports) {
            expect(rails.has(port.net), `${name}/${analysis}: ${port.net} is not a rail`)
              .toBe(true);
          }
          const patch = buildStimulus(json, option, analysis, { frequency: 1e3, amplitude: 1e-3 });
          if (patch.target !== "periodic") continue;
          for (const net of Object.keys(patch.periodic.node_inputs as object)) {
            expect(rails.has(net), `${name}/${analysis}: ${net} is not a rail`).toBe(true);
          }
        }
      }
    }
  });

  it("offers a source for most decks, and says so plainly for the rest", () => {
    const withOptions = corpus.filter(({ json }) => stimulusOptions(json, "transient").length > 0);
    expect(withOptions.length).toBeGreaterThan(corpus.length / 2);
  });

  it("gives every silent analysis in the corpus something to pick, or nothing at all", () => {
    // A menu that is present but empty is the one outcome with no explanation
    // attached; the panel renders a reason instead, so an empty list must mean
    // the circuit genuinely has no drivable rail.
    for (const { name, json } of corpus) {
      if (stimulusOptions(json, "transient").length > 0) continue;
      expect(candidateInputs(json), `${name} has candidates but no options`).toEqual([]);
    }
  });
});
