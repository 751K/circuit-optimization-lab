/**
 * circuit JSON  ->  editor graph  (+ a `rest` bag of everything not modeled).
 *
 * See graph.ts for the net model. The blocks this consumes into nodes/edges:
 *   solved, rails, devices, sizes, nf, models, resistors, capacitors,
 *   load_caps, outputs, input_drives.
 * Every other block (name, bias, analyses, explore, periodic, vsources,
 * controlled sources, aliases, dc_guesses, transient_inputs, ac_drives, ...)
 * is copied verbatim into `rest` and re-merged on export — no byte is dropped.
 * `bias` in particular stays in `rest` (rail nodes only reference its keys), so
 * it round-trips exactly even for bias keys no rail references.
 */
import type {
  CapacitorNode,
  CircuitGraph,
  GraphEdge,
  GraphNode,
  MosfetNode,
  OutputNode,
  Port,
  Position,
  RailNode,
  ResistorNode,
} from "./graph";
import type { CircuitJson, ModelEntry } from "./circuit";
import {
  autoPosition,
  barycenterReorder,
  normalizeDevice,
  normalizeLoadCap,
  normalizeTwoTerminal,
} from "./util";

// Blocks the graph reconstructs on export; everything else goes to `rest`.
//
// `sizes` and `nf` are deliberately NOT reconstructed — they are top-level
// override blocks (a `sizes[name]` beats embedded W/L; a top-level `nf` beats
// an embedded NF). Rebuilding them risks moving W/L between the embedded and
// the `sizes` form, a structural round-trip drift. Instead they ride through
// `rest` verbatim, and device W/L/nf are read (for graph display) with the
// override applied but re-emitted only in their original embedded slot. This is
// byte-safe: an override block is preserved exactly, and the embedded W/L it
// overrode is preserved exactly too, so the loader resolves identically.
const MODELED_KEYS = new Set<string>([
  "solved",
  "rails",
  "devices",
  "models",
  "resistors",
  "capacitors",
  "load_caps",
  "outputs",
  "input_drives",
  "ui",
]);

export interface ToGraphResult {
  graph: CircuitGraph;
  /**
   * Everything not represented in the graph, preserved verbatim for export.
   * Includes `bias` (rails reference its keys), all analysis/source/aliasing
   * blocks, `name`, `sizes`/`nf` remnants, and any unknown top-level key.
   */
  rest: Record<string, unknown>;
}

/**
 * The W/L carried on a mosfet node. This is the value re-emitted on export in
 * the device object's *embedded* slot, so it mirrors the embedded W/L (the only
 * form the 12 fixtures use). A top-level `sizes` override, when present, rides
 * through `rest` untouched and the loader still resolves it over the embedded
 * value — so we don't fold it into the node value (that would drop the embedded
 * slot on export). When a device has neither an embedded W/L nor a `sizes`
 * entry (invalid, but be graceful) the node value is 0.
 */
function embeddedSize(
  embeddedW: number | undefined,
  embeddedL: number | undefined,
): { W: number; L: number } {
  return { W: embeddedW ?? 0, L: embeddedL ?? 0 };
}

/**
 * Total order over net ports, used to lay a net out as a nearest-neighbor
 * chain: sort primarily by the host node's canvas position (x, then y), then
 * break ties deterministically by node id and port id. Purely geometric — it
 * changes the edge *shape* only; the set of ports (and thus the connected
 * component / resolved net) is untouched.
 */
function cmpPort(
  a: { node: string; port: string },
  b: { node: string; port: string },
  posOf: Map<string, Position>,
): number {
  const pa = posOf.get(a.node) ?? [0, 0];
  const pb = posOf.get(b.node) ?? [0, 0];
  if (pa[0] !== pb[0]) return pa[0] - pb[0];
  if (pa[1] !== pb[1]) return pa[1] - pb[1];
  if (a.node !== b.node) return a.node < b.node ? -1 : 1;
  if (a.port !== b.port) return a.port < b.port ? -1 : 1;
  return 0;
}

export function circuitJsonToGraph(json: CircuitJson): ToGraphResult {
  const nodes: GraphNode[] = [];
  const edges: GraphEdge[] = [];

  // `rest` = every top-level key we don't reconstruct from the graph.
  const rest: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(json)) {
    if (!MODELED_KEYS.has(k)) rest[k] = v;
  }

  // Capture source ordering of order-significant blocks so export replays it
  // (solved -> MNA vector order; devices / rails -> author order). Stashed under
  // `ui.order`; export merges its computed positions into the same `ui` block.
  const nameOf2t = (el: unknown): string =>
    Array.isArray(el) ? (el[0] as string) : (el as { name: string }).name;
  const order: NonNullable<CircuitJson["ui"]>["order"] = {
    solved: [...json.solved],
    devices: json.devices.map((d) => (Array.isArray(d) ? d[0] : d.name)),
    rails: Object.keys(json.rails),
    resistors: (json.resistors ?? []).map(nameOf2t),
    capacitors: (json.capacitors ?? []).map(nameOf2t),
  };
  rest.ui = { order };

  const positions = json.ui?.positions ?? {};
  // Ids that fell back to the auto-layout (no explicit ui.positions entry);
  // only these are eligible for the barycenter reorder pass below.
  const autoIds = new Set<string>();
  // Per-kind counters drive the deterministic fallback layout.
  const kindCounts: Record<string, number> = {};
  const placeAt = (id: string, kind: string): [number, number] => {
    const explicit = positions[id];
    if (explicit) return [explicit[0], explicit[1]];
    autoIds.add(id);
    const idx = kindCounts[kind] ?? 0;
    kindCounts[kind] = idx + 1;
    return autoPosition(kind, idx);
  };

  // Every port that carries a given net; used to synthesize connectivity edges.
  // net name -> list of {node, port}
  const netPorts = new Map<string, { node: string; port: string }[]>();
  const registerPort = (net: string, node: string, port: string): void => {
    let arr = netPorts.get(net);
    if (!arr) {
      arr = [];
      netPorts.set(net, arr);
    }
    arr.push({ node, port });
  };
  const mkPort = (id: string, net: string): Port => ({ id, originalNet: net });

  // ── rails (each rail is a one-port net node) ──────────────────────────
  const bias = (json.bias ?? {}) as Record<string, number>;
  for (const [railName, railValue] of Object.entries(json.rails)) {
    const node: RailNode = {
      id: railName,
      kind: "rail",
      net: railName,
      railValue,
      ports: [mkPort("net", railName)],
      position: placeAt(railName, "rail"),
    };
    if (typeof railValue === "string" && railValue in bias) {
      node.biasValue = bias[railValue];
    }
    nodes.push(node);
    registerPort(railName, railName, "net");
  }

  // ── devices (mosfets) ─────────────────────────────────────────────────
  const inputDrives = (json.input_drives ?? {}) as Record<string, number>;
  const models = (json.models ?? {}) as Record<string, ModelEntry>;
  for (const raw of json.devices) {
    const d = normalizeDevice(raw);
    const hasEmbeddedWL = d.W !== undefined && d.L !== undefined;
    const { W, L } = embeddedSize(d.W, d.L);
    const node: MosfetNode = {
      id: d.name,
      kind: "mosfet",
      name: d.name,
      W,
      L,
      ports: [
        mkPort("D", d.drain),
        mkPort("G", d.gate),
        mkPort("S", d.source),
      ],
      position: placeAt(d.name, "mosfet"),
    };
    // Only emit W/L back on the device object if the source did. When W/L came
    // solely from a top-level `sizes` override (never in the fixtures), the
    // override rides through `rest` and we must not add an embedded slot.
    node.hasEmbeddedWL = hasEmbeddedWL;
    if (d.NF !== undefined) node.nf = d.NF;
    if (Object.keys(d.extra).length > 0) {
      // Unknown device-object keys (rare) ride along so they survive export.
      node.modelKwargs = { ...(node.modelKwargs ?? {}), ...d.extra };
    }
    const m = models[d.name];
    if (m) {
      const { type, ...kwargs } = m;
      if (type !== undefined) node.model = type;
      if (Object.keys(kwargs).length > 0) {
        node.modelKwargs = { ...(node.modelKwargs ?? {}), ...kwargs };
      }
    }
    if (d.name in inputDrives) node.inputDrive = inputDrives[d.name];
    nodes.push(node);
    registerPort(d.drain, d.name, "D");
    registerPort(d.gate, d.name, "G");
    registerPort(d.source, d.name, "S");
  }

  // ── resistors ─────────────────────────────────────────────────────────
  for (const raw of json.resistors ?? []) {
    const r = normalizeTwoTerminal(raw);
    const node: ResistorNode = {
      id: r.name,
      kind: "resistor",
      name: r.name,
      R: r.value,
      ports: [mkPort("a", r.a), mkPort("b", r.b)],
      position: placeAt(r.name, "resistor"),
    };
    nodes.push(node);
    registerPort(r.a, r.name, "a");
    registerPort(r.b, r.name, "b");
  }

  // ── capacitors (named `capacitors` block) ─────────────────────────────
  for (const raw of json.capacitors ?? []) {
    const c = normalizeTwoTerminal(raw);
    const node: CapacitorNode = {
      id: c.name,
      kind: "capacitor",
      name: c.name,
      C: c.value,
      origin: "capacitors",
      ports: [mkPort("a", c.a), mkPort("b", c.b)],
      position: placeAt(c.name, "capacitor"),
    };
    nodes.push(node);
    registerPort(c.a, c.name, "a");
    registerPort(c.b, c.name, "b");
  }

  // ── load_caps (nameless capacitors) ───────────────────────────────────
  // Synthesize a stable id so ui.positions can key it and export can find the
  // block back. The synthetic name is NOT exported (see toJson.ts).
  json.load_caps?.forEach((raw, i) => {
    const c = normalizeLoadCap(raw);
    const id = `__loadcap_${i}`;
    const node: CapacitorNode = {
      id,
      kind: "capacitor",
      name: id,
      C: c.C,
      origin: "load_caps",
      ports: [mkPort("a", c.a), mkPort("b", c.b)],
      position: placeAt(id, "capacitor"),
    };
    nodes.push(node);
    registerPort(c.a, id, "a");
    registerPort(c.b, id, "b");
  });

  // ── outputs (single-port markers; order preserved for differential) ───
  (json.outputs ?? []).forEach((net, order) => {
    const id = `__out_${order}`;
    const node: OutputNode = {
      id,
      kind: "output",
      order,
      ports: [mkPort("out", net)],
      position: placeAt(id, "output"),
    };
    nodes.push(node);
    registerPort(net, id, "out");
  });

  // ── synthesize edges from shared net names ────────────────────────────
  // Ports sharing an original net name are electrically one net. We connect
  // them into a *nearest-neighbor chain* (each port to the next after sorting
  // by host-node position) rather than a star hub. A chain over N ports is N-1
  // edges spanning all of them, so its connected component is identical to the
  // star's — the export-side union-find reconstructs the exact same net — but
  // the geometry is a readable spine instead of long diagonals fanning from one
  // hub. Deterministic: the sort is total (position, then node id, then port).
  const posOf = new Map<string, Position>(nodes.map((n) => [n.id, n.position]));
  for (const [net, ports] of netPorts) {
    // Sort ports by their host node's canvas position (column x, then row y),
    // falling back to node/port id so the order is fully deterministic and
    // stable even when two nodes share a position.
    const chain = [...ports].sort((a, b) => cmpPort(a, b, posOf));
    for (let i = 1; i < chain.length; i++) {
      const prev = chain[i - 1]!;
      const p = chain[i]!;
      edges.push({
        id: `e:${net}:${i}`,
        source: { node: prev.node, port: prev.port },
        target: { node: p.node, port: p.port },
      });
    }
  }

  // ── barycenter layout tidy (auto-placed nodes only) ───────────────────
  // Build node-level adjacency from the synthesized edges, then run one
  // deterministic barycenter sweep to shorten the total wire length. Nodes with
  // a stored ui.positions entry are pinned (their whole column is skipped), so
  // an archived layout never shifts.
  const adj = new Map<string, Set<string>>();
  const link = (a: string, b: string): void => {
    if (a === b) return;
    (adj.get(a) ?? adj.set(a, new Set()).get(a)!).add(b);
    (adj.get(b) ?? adj.set(b, new Set()).get(b)!).add(a);
  };
  for (const e of edges) link(e.source.node, e.target.node);
  barycenterReorder(nodes, adj, autoIds);

  return { graph: { nodes, edges }, rest };
}
