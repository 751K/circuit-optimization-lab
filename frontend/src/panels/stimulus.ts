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
 * So this module does three things:
 *
 *  - {@link acStimulus} reads the AC excitation back out into one list of
 *    (net, magnitude) ports, whichever block it was written in, so the panel can
 *    show what will actually drive the run rather than leaving it implicit.
 *  - {@link stimulusOptions} builds, from the circuit's own rails, the list of
 *    stimuli an analysis with none *could* be given — a differential pair, a
 *    single-ended input, a clock — so the answer to "this run has no stimulus"
 *    is a menu rather than an instruction to go and edit JSON.
 *  - {@link buildStimulus} realises a chosen one: an `input_drives`/`ac_drives`
 *    patch for AC, or a `periodic` block for everything else, with each waveform
 *    centred on the driven rail's own DC level.
 *
 * A periodic block is written under `analyses.<owner>.periodic`, which the
 * dispatcher merges over any top-level `periodic` — so configuring the transient
 * leaves what PSS/PAC/PNoise are excited by alone, and vice versa.
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

/** A net a stimulus could legitimately be applied to. */
export interface InputCandidate {
  net: string;
  /** DC level the rail sits at, which a waveform is centred on. */
  level: number;
  /** Mosfets this net gates, if any — an `input_drives` target. */
  devices: string[];
}

/**
 * Nets on which some element terminal draws current — as opposed to a net every
 * element port of which is a mosfet gate, which controls rather than supplies.
 */
function suppliesCurrent(circuit: CircuitJson): Set<string> {
  const out = new Set<string>();
  for (const d of circuit.devices ?? []) {
    if (Array.isArray(d)) {
      out.add(d[1]);
      out.add(d[3]);
    } else {
      out.add(d.drain);
      out.add(d.source);
    }
  }
  const twoTerminal = [
    ...(circuit.resistors ?? []),
    ...(circuit.capacitors ?? []),
  ];
  for (const el of twoTerminal) {
    if (Array.isArray(el)) {
      out.add(el[1]);
      out.add(el[2]);
    } else {
      out.add(el.a);
      out.add(el.b);
    }
  }
  for (const el of circuit.load_caps ?? []) {
    if (Array.isArray(el)) {
      out.add(el[0]);
      out.add(el[1]);
    } else {
      out.add(el.a);
      out.add(el.b);
    }
  }
  return out;
}

/**
 * The nets that could take a stimulus: the rails, minus the supplies.
 *
 * A waveform can only be forced onto a fixed-potential net — driving a node the
 * circuit solves for would override the circuit rather than excite it — and the
 * supply and ground are not what anyone means by an input. What is left is the
 * inputs, the clocks and the biases, and a bias is a legitimate thing to wobble.
 *
 * "Supply" is the *extreme* potentials, but only for a rail that actually
 * carries current: a clock sits at exactly the supply level and is one of the
 * most likely things to want to drive, so excluding it on potential alone hides
 * the obvious stimulus for every switched-capacitor and chopper testbench.
 */
export function candidateInputs(circuit: CircuitJson): InputCandidate[] {
  const rails = circuit.rails ?? {};
  const levels = new Map<string, number>();
  for (const net of Object.keys(rails)) {
    const level = railLevel(circuit, net);
    if (level !== undefined) levels.set(net, level);
  }
  if (levels.size === 0) return [];
  const all = [...levels.values()];
  const hi = Math.max(...all);
  const lo = Math.min(...all);
  const carries = suppliesCurrent(circuit);

  const gatesOf = new Map<string, string[]>();
  for (const d of circuit.devices ?? []) {
    const { name, gate } = gateOf(d);
    (gatesOf.get(gate) ?? gatesOf.set(gate, []).get(gate)!).push(name);
  }

  const out: InputCandidate[] = [];
  for (const [net, level] of levels) {
    // With one distinct rail potential there is no supply to exclude.
    const atExtreme = hi > lo && (level === hi || level === lo);
    if (atExtreme && carries.has(net)) continue;
    out.push({ net, level, devices: [...(gatesOf.get(net) ?? [])].sort() });
  }
  return out.sort((a, b) => a.net.localeCompare(b.net));
}

/**
 * Candidate pairs that look like the two halves of a differential input: two
 * rails sitting at the same level, each gating exactly one device, and those
 * two devices sharing a source net. Offering the pair as one choice is the
 * difference between a usable menu and asking the user to configure `vinp` and
 * `vinn` separately and remember to invert one.
 */
export function differentialPairs(circuit: CircuitJson): [string, string][] {
  const sourceOf = new Map<string, string>();
  for (const d of circuit.devices ?? []) {
    if (Array.isArray(d)) sourceOf.set(d[0], d[3]);
    else sourceOf.set(d.name, d.source);
  }
  const singles = candidateInputs(circuit).filter((c) => c.devices.length === 1);
  const pairs: [string, string][] = [];
  for (let i = 0; i < singles.length; i++) {
    for (let k = i + 1; k < singles.length; k++) {
      const a = singles[i]!;
      const b = singles[k]!;
      if (a.level !== b.level) continue;
      const sa = sourceOf.get(a.devices[0]!);
      if (sa === undefined || sa !== sourceOf.get(b.devices[0]!)) continue;
      pairs.push([a.net, b.net]);
    }
  }
  return pairs;
}

/** One stimulus the panel can offer for an analysis that has none. */
export interface StimulusOption {
  /** Stable value for the select. */
  id: string;
  label: string;
  /** Which block applying it writes. */
  target: "ac" | "periodic";
  /** Waveform shape, for a periodic target. */
  wave?: "sine" | "pulse";
  /** Ports driven, with relative magnitude; a negative one is the inverting half. */
  ports: { net: string; magnitude: number; device?: string }[];
}

/** Whether an analysis is excited by an AC drive or by a periodic waveform. */
function targetOf(analysis: string): "ac" | "periodic" {
  return analysis === "ac" || analysis === "noise" ? "ac" : "periodic";
}

/**
 * Where the periodic block for `analysis` has to be written.
 *
 * The dispatcher merges `analyses.<owner>.periodic` over the top-level block, so
 * writing it under the owner configures that analysis alone. PAC and PNoise are
 * excited through the PSS they run on, so all three share `analyses.pss`.
 */
export function periodicOwner(analysis: string): string {
  return analysis === "pac" || analysis === "pnoise" ? "pss" : analysis;
}

/**
 * The stimuli a circuit could be given for `analysis`, built from its own rails.
 *
 * Empty when there is no rail that could take one, which the panel reports
 * rather than showing an empty menu.
 */
export function stimulusOptions(circuit: CircuitJson, analysis: string): StimulusOption[] {
  const target = targetOf(analysis);
  const candidates = candidateInputs(circuit);
  const byNet = new Map(candidates.map((c) => [c.net, c]));
  const options: StimulusOption[] = [];

  const portsFor = (net: string, magnitude: number) => {
    const c = byNet.get(net);
    // Drive the gates when the net gates devices — that is `input_drives`, the
    // form the decks use — and the node itself otherwise.
    return (c && c.devices.length > 0)
      ? c.devices.map((device) => ({ net, magnitude, device }))
      : [{ net, magnitude }];
  };

  // An existing AC input is the most likely thing to want in the time domain
  // too, and reuses a port list the user already agreed with.
  if (target === "periodic") {
    const ac = acStimulus(circuit).filter((p) => p.onRail);
    if (ac.length > 0) {
      const nets = [...new Set(ac.map((p) => p.net))];
      options.push({
        id: "periodic:ac-ports",
        label: `Sine on the AC input ports (${nets.join(", ")})`,
        target, wave: "sine",
        ports: ac.map((p) => ({ net: p.net, magnitude: p.magnitude, device: p.device })),
      });
    }
  }

  for (const [a, b] of differentialPairs(circuit)) {
    const ports = [...portsFor(a, 1), ...portsFor(b, -1)];
    if (target === "ac") {
      options.push({ id: `ac:diff:${a}:${b}`, label: `Differential — ${a} / ${b} (±1)`, target, ports });
    } else {
      options.push({
        id: `periodic:sine:diff:${a}:${b}`,
        label: `Differential sine — ${a} / ${b}`, target, wave: "sine", ports,
      });
    }
  }

  for (const c of candidates) {
    const ports = portsFor(c.net, 1);
    const where = c.devices.length > 0 ? `${c.devices.join(", ")}.G` : "node";
    if (target === "ac") {
      options.push({
        id: `ac:single:${c.net}`,
        label: `Single-ended — ${c.net} (${where})`, target, ports,
      });
    } else {
      options.push({
        id: `periodic:sine:${c.net}`,
        label: `Sine — ${c.net} (${where})`, target, wave: "sine", ports,
      });
      options.push({
        id: `periodic:pulse:${c.net}`,
        label: `Pulse / clock — ${c.net} (${where})`, target, wave: "pulse", ports,
      });
    }
  }
  return options;
}

/** What applying a chosen stimulus writes. */
export type StimulusPatch =
  | { target: "ac"; inputDrives: Record<string, number>; acDrives: Record<string, number> }
  | { target: "periodic"; owner: string; periodic: Record<string, unknown> };

/**
 * Turn a chosen option into the blocks that realise it.
 *
 * A periodic waveform is centred on the driven rail's own DC level, so the run
 * starts from the operating point the circuit was designed around instead of
 * slewing away from zero, and a negative port is the same waveform inverted —
 * a sine by half a period of phase, a pulse by swapping its levels.
 */
export function buildStimulus(
  circuit: CircuitJson,
  option: StimulusOption,
  analysis: string,
  opts: DeriveOptions,
): StimulusPatch {
  if (option.target === "ac") {
    const inputDrives: Record<string, number> = {};
    const acDrives: Record<string, number> = {};
    for (const p of option.ports) {
      if (p.device) inputDrives[p.device] = p.magnitude;
      else acDrives[p.net] = p.magnitude;
    }
    return { target: "ac", inputDrives, acDrives };
  }

  const inputs: Record<string, unknown> = {};
  const nodeInputs: Record<string, string> = {};
  for (const p of option.ports) {
    if (p.net in nodeInputs) continue;
    const key = `stim_${p.net}`;
    const dc = railLevel(circuit, p.net) ?? 0;
    const swing = opts.amplitude * Math.abs(p.magnitude);
    const inverted = p.magnitude < 0;
    inputs[key] = option.wave === "pulse"
      ? {
        type: "pulse",
        low: inverted ? dc + swing : dc - swing,
        high: inverted ? dc - swing : dc + swing,
        duty: 0.5,
        // Finite edges: an ideal step has no timescale for the solver to
        // resolve, and the adaptive driver inserts breakpoints at these.
        rise: 0.02 / opts.frequency,
        fall: 0.02 / opts.frequency,
      }
      : { type: "sine", dc, amplitude: swing, phase: inverted ? Math.PI : 0 };
    nodeInputs[p.net] = key;
  }
  return {
    target: "periodic",
    owner: periodicOwner(analysis),
    periodic: {
      frequency: opts.frequency,
      n_points: opts.points ?? 201,
      inputs,
      node_inputs: nodeInputs,
    },
  };
}

export interface DeriveOptions {
  /** Fundamental of the derived sine, in Hz. */
  frequency: number;
  /** Peak amplitude for a port of unit magnitude, in volts. */
  amplitude: number;
  /** Samples per period written into the block. */
  points?: number;
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
