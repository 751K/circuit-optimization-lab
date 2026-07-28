/**
 * What drives a circuit, per analysis — and how to give a transient one.
 *
 * A circuit JSON declares its excitation in several unrelated blocks, and each
 * analysis reads a *different* one:
 *
 *   ac / noise        `input_drives` (a device gate) and `ac_drives` (a node)
 *   transient         nothing at all unless a `periodic` block exists
 *   pss / pac / pnoise `periodic`
 *
 * That asymmetry is the trap. `sky130_5t_ota` declares `input_drives`
 * `{M1: +1, M2: -1}` and no `periodic`, so its AC run is properly excited while
 * its transient runs with every source pinned at DC: the solver converges, every
 * gate is a flat line, and nothing anywhere reports that the stimulus was
 * missing. Reading the JSON in the editor shows the same thing — the input nets
 * are rails held at `VCM`, because the *signal* on them lives on the devices,
 * not on the rails.
 *
 * So this module does two things:
 *
 *  - {@link acStimulus} reads the AC excitation back out into one list of
 *    (net, magnitude) ports, whichever block it was written in, so the panel can
 *    show what will actually drive the run rather than leaving it implicit.
 *  - {@link deriveTransientPeriodic} turns that same port list into a `periodic`
 *    block: a sine per input port, sitting on that rail's own DC level, with the
 *    relative amplitude and the 180 degrees of a differential pair taken from
 *    the sign of the AC drive. The AC and transient stimulus then describe one
 *    input, and a transient of an AC-ready circuit stops being a flat line.
 *
 * The derived block is written under `analyses.transient.periodic`, which the
 * dispatcher merges over any top-level `periodic` — so it configures the
 * transient without touching what PSS/PAC/PNoise use.
 */
import type { CircuitJson, Device } from "../model/circuit";

/** One port of the AC excitation, resolved to the net it actually drives. */
export interface StimulusPort {
  /** The net carrying the drive. */
  net: string;
  /** Relative AC magnitude; the sign carries differential phase. */
  magnitude: number;
  /** Whether it was declared on a device gate or directly on a node. */
  via: "gate" | "node";
  /** Device name, when the drive was declared on a gate. */
  device?: string;
  /** True when the net is a rail, i.e. a node a waveform may legitimately drive. */
  onRail: boolean;
}

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

/** A device's gate net, from either the object or the array form. */
function gateOf(d: Device): { name: string; gate: string } {
  return Array.isArray(d)
    ? { name: d[0], gate: d[2] }
    : { name: d.name, gate: d.gate };
}

/**
 * Every AC input port the circuit declares, in a single list.
 *
 * `input_drives` names a *device* whose gate is driven; `ac_drives` names a
 * *node*. Both mean "this is where the signal comes in", and the panel should
 * not make the user know which spelling a given deck happened to use. Sorted by
 * descending magnitude then by net, so a differential pair reads +then- and the
 * order does not depend on JSON key order.
 */
export function acStimulus(circuit: CircuitJson): StimulusPort[] {
  const rails = new Set(Object.keys(circuit.rails ?? {}));
  const gateByDevice = new Map<string, string>();
  for (const d of circuit.devices ?? []) {
    const { name, gate } = gateOf(d);
    gateByDevice.set(name, gate);
  }

  const ports: StimulusPort[] = [];
  const drives = circuit.input_drives;
  if (isRecord(drives)) {
    for (const [device, value] of Object.entries(drives)) {
      const net = gateByDevice.get(device);
      if (net === undefined || typeof value !== "number") continue;
      ports.push({ net, magnitude: value, via: "gate", device, onRail: rails.has(net) });
    }
  }
  const acDrives = circuit.ac_drives;
  if (isRecord(acDrives)) {
    for (const [net, value] of Object.entries(acDrives)) {
      if (typeof value !== "number") continue;
      ports.push({ net, magnitude: value, via: "node", onRail: rails.has(net) });
    }
  }
  return ports.sort((a, b) =>
    (b.magnitude !== a.magnitude ? b.magnitude - a.magnitude : a.net.localeCompare(b.net)));
}

/** The DC level a rail sits at, resolved through `bias` when it is a key. */
export function railLevel(circuit: CircuitJson, net: string): number | undefined {
  const value = (circuit.rails ?? {})[net];
  if (typeof value === "number") return value;
  if (typeof value !== "string") return undefined;
  const bias = circuit.bias;
  const resolved = isRecord(bias) ? bias[value] : undefined;
  return typeof resolved === "number" ? resolved : undefined;
}

/** The `periodic` block an analysis would actually run with, if any. */
export function periodicFor(
  circuit: CircuitJson,
  analysis: string,
): Record<string, unknown> | null {
  const top = isRecord(circuit.periodic) ? circuit.periodic : null;
  const analyses = isRecord(circuit.analyses) ? circuit.analyses : {};
  const cfg = isRecord(analyses[analysis]) ? analyses[analysis] : {};
  const own = isRecord(cfg.periodic) ? cfg.periodic : null;
  if (!top && !own) return null;
  return { ...(top ?? {}), ...(own ?? {}) };
}

export interface DeriveOptions {
  /** Fundamental of the derived sine, in Hz. */
  frequency: number;
  /** Peak amplitude for a port of unit magnitude, in volts. */
  amplitude: number;
  /** Samples per period written into the block. */
  points?: number;
}

/** Why a transient stimulus could not be derived, or null when it can be. */
export function deriveBlocker(circuit: CircuitJson): string | null {
  const ports = acStimulus(circuit);
  if (ports.length === 0) {
    return "This circuit declares no AC input — there is no input_drives or "
      + "ac_drives entry to take the port list from. Add one, or write a "
      + "periodic block by hand.";
  }
  if (!ports.some((p) => p.onRail)) {
    return "The AC input drives an internal node, not a rail. A waveform can "
      + "only be forced onto a fixed-potential net, so the transient stimulus "
      + "has to be written by hand for this testbench.";
  }
  return null;
}

/**
 * Build a `periodic` block that excites the same ports the AC run uses.
 *
 * Each port becomes a sine centred on its rail's own DC level — the operating
 * point the circuit was designed around, so the transient starts settled — with
 * amplitude scaled by the port's magnitude and phase flipped for a negative one.
 * Ports on internal nodes are skipped: forcing a waveform onto a node the
 * circuit solves for would override the circuit rather than drive it.
 *
 * Returns null when nothing can be derived; see {@link deriveBlocker} for why.
 */
export function deriveTransientPeriodic(
  circuit: CircuitJson,
  opts: DeriveOptions,
): Record<string, unknown> | null {
  const ports = acStimulus(circuit).filter((p) => p.onRail);
  if (ports.length === 0) return null;

  const inputs: Record<string, unknown> = {};
  const nodeInputs: Record<string, string> = {};
  // One waveform per *net*: two devices sharing a gate net are one input, and
  // emitting it twice would let the second silently win.
  for (const port of ports) {
    if (port.net in nodeInputs) continue;
    const key = `stim_${port.net}`;
    inputs[key] = {
      type: "sine",
      dc: railLevel(circuit, port.net) ?? 0,
      amplitude: opts.amplitude * Math.abs(port.magnitude),
      phase: port.magnitude < 0 ? Math.PI : 0,
    };
    nodeInputs[port.net] = key;
  }
  return {
    frequency: opts.frequency,
    n_points: opts.points ?? 201,
    inputs,
    node_inputs: nodeInputs,
  };
}

/** What the Simulate panel says about `analysis`'s stimulus. */
export interface StimulusReport {
  /** Blocks that actually drive this analysis, for display. */
  kind: "ac" | "periodic" | "none";
  /** True when the analysis will run with no excitation at all. */
  silent: boolean;
  ports: StimulusPort[];
  /** One-line explanation, always populated. */
  detail: string;
}

/**
 * Report the stimulus `analysis` will run with.
 *
 * The point is to say it *before* the run. A silent AC returns the -180 dB gain
 * floor and a silent transient returns a flat line, and both look like results.
 */
export function stimulusReport(circuit: CircuitJson, analysis: string): StimulusReport {
  if (analysis === "ac" || analysis === "noise") {
    const ports = acStimulus(circuit);
    return {
      kind: ports.length > 0 ? "ac" : "none",
      silent: ports.length === 0,
      ports,
      detail: ports.length > 0
        ? `${ports.length} AC input port${ports.length > 1 ? "s" : ""} from `
          + `${circuit.input_drives ? "input_drives" : ""}`
            .concat(circuit.input_drives && circuit.ac_drives ? " + " : "")
            .concat(circuit.ac_drives ? "ac_drives" : "")
        : "No input_drives or ac_drives: this run has no stimulus and will "
          + "return the -180 dB gain floor.",
    };
  }
  const periodic = periodicFor(circuit, analysis);
  if (periodic) {
    const nodes = isRecord(periodic.node_inputs) ? Object.keys(periodic.node_inputs) : [];
    const freq = typeof periodic.frequency === "number" ? periodic.frequency : undefined;
    return {
      kind: "periodic",
      silent: false,
      ports: [],
      detail: `periodic block${freq ? ` at ${freq} Hz` : ""}`
        + (nodes.length > 0 ? ` driving ${nodes.join(", ")}` : ""),
    };
  }
  return {
    kind: "none",
    silent: true,
    ports: acStimulus(circuit),
    detail: analysis === "transient"
      ? "No periodic block: every source stays at DC and the run returns a flat line."
      : "No periodic block: pss/pac/pnoise cannot run without one.",
  };
}
