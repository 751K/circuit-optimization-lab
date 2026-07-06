/**
 * editor graph (+ `rest`)  ->  circuit JSON.
 *
 * The heart of the mapping: a connected-component pass over (ports + edges)
 * resolves every port to a single net name, then each block is rebuilt from the
 * nodes using those names, and `rest` is merged back on top. Layout is written
 * to `ui.positions`. Output is deterministic (elements sorted by name) so two
 * exports of the same graph diff cleanly.
 *
 * Net-name resolution per component (priority order):
 *   1. a rail port in the component  -> that rail's net name
 *      (two *different* rails in one component is a hard error).
 *   2. any port remembers an original net name -> reuse it (the earliest such
 *      name in a stable node scan; keeps hand-authored names like "tail").
 *   3. otherwise -> an auto name n1, n2, ... skipping any name already taken.
 */
import type {
  CapacitorNode,
  CircuitGraph,
  MosfetNode,
  OutputNode,
  RailNode,
  ResistorNode,
} from "./graph";
import type {
  CapacitorObject,
  CircuitJson,
  CircuitUi,
  LoadCapArray,
  ModelEntry,
  RailValue,
  ResistorObject,
} from "./circuit";
import { deviceToObject } from "./util";

// ── union-find over ports ────────────────────────────────────────────────

class UnionFind {
  private parent = new Map<string, string>();
  find(x: string): string {
    let root = this.parent.get(x);
    if (root === undefined) {
      this.parent.set(x, x);
      return x;
    }
    // path-halving
    while (root !== x) {
      const gp = this.parent.get(root) ?? root;
      this.parent.set(x, gp);
      x = gp;
      root = this.parent.get(x) ?? x;
    }
    return x;
  }
  union(a: string, b: string): void {
    const ra = this.find(a);
    const rb = this.find(b);
    if (ra !== rb) this.parent.set(ra, rb);
  }
}

// A port is addressed by node id + port id. We join with a unit-separator
// control char that can never appear in an id, so the key is unambiguous.
const PORT_KEY_SEP = String.fromCharCode(31);
const portKey = (node: string, port: string): string =>
  `${node}${PORT_KEY_SEP}${port}`;

export interface NetResolution {
  /** port key -> resolved net name. */
  portNet: Map<string, string>;
  /** component root -> resolved net name. */
  rootNet: Map<string, string>;
}

/**
 * Resolve every port to a net name via connected components. Exported for the
 * net-resolution unit tests. Throws on a two-rail conflict in one component.
 */
export function resolveNets(graph: CircuitGraph): NetResolution {
  const uf = new UnionFind();
  // Register every port so isolated ports get their own component.
  for (const n of graph.nodes) {
    for (const p of n.ports) uf.find(portKey(n.id, p.id));
  }
  for (const e of graph.edges) {
    uf.union(
      portKey(e.source.node, e.source.port),
      portKey(e.target.node, e.target.port),
    );
  }

  // Gather per-component: rail nets present, and candidate original names.
  // A stable node scan (graph.nodes order) makes name selection deterministic.
  interface Comp {
    rails: Set<string>;
    original?: string; // first original net name seen
  }
  const comps = new Map<string, Comp>();
  const compOf = (key: string): Comp => {
    const root = uf.find(key);
    let c = comps.get(root);
    if (!c) {
      c = { rails: new Set() };
      comps.set(root, c);
    }
    return c;
  };

  for (const n of graph.nodes) {
    for (const p of n.ports) {
      const c = compOf(portKey(n.id, p.id));
      if (n.kind === "rail") c.rails.add((n as RailNode).net);
      if (p.originalNet !== undefined && c.original === undefined) {
        c.original = p.originalNet;
      }
    }
  }

  const rootNet = new Map<string, string>();
  const used = new Set<string>();
  // First pass: assign rail- and original-named components (reserve names).
  for (const [root, c] of comps) {
    if (c.rails.size > 1) {
      const names = [...c.rails].sort().join(", ");
      throw new Error(
        `net conflict: a single net is tied to multiple rails (${names})`,
      );
    }
    if (c.rails.size === 1) {
      const net = [...c.rails][0]!;
      rootNet.set(root, net);
      used.add(net);
    }
  }
  for (const [root, c] of comps) {
    if (rootNet.has(root)) continue;
    if (c.original !== undefined && !used.has(c.original)) {
      rootNet.set(root, c.original);
      used.add(c.original);
    }
  }
  // Second pass: auto-name the rest (n1, n2, ...), skipping taken names.
  let counter = 1;
  const nextAuto = (): string => {
    let name = `n${counter++}`;
    while (used.has(name)) name = `n${counter++}`;
    used.add(name);
    return name;
  };
  // Deterministic order: sort remaining roots by their first port key.
  const remaining: string[] = [];
  for (const root of comps.keys()) {
    if (!rootNet.has(root)) remaining.push(root);
  }
  remaining.sort();
  for (const root of remaining) rootNet.set(root, nextAuto());

  const portNet = new Map<string, string>();
  for (const n of graph.nodes) {
    for (const p of n.ports) {
      const key = portKey(n.id, p.id);
      const net = rootNet.get(uf.find(key));
      if (net !== undefined) portNet.set(key, net);
    }
  }
  return { portNet, rootNet };
}

// ── reconstruction ───────────────────────────────────────────────────────

/**
 * Rebuild circuit JSON from the graph and the preserved `rest`. `rest` blocks
 * (bias, analyses, explore, periodic, vsources, aliases, dc_guesses, name, ...)
 * are merged in verbatim. Graph-derived blocks (solved, rails, devices, sizes,
 * nf, models, resistors, capacitors, load_caps, outputs, input_drives) are
 * written from the resolved nets and node data.
 */
export function graphToCircuitJson(
  graph: CircuitGraph,
  rest: Record<string, unknown> = {},
): CircuitJson {
  const { portNet } = resolveNets(graph);
  // Source ordering hints (see toGraph): replay original block order, appending
  // anything new deterministically (sorted).
  const uiIn = (rest.ui ?? {}) as CircuitUi;
  const orderHint = uiIn.order ?? {};

  /** Order `items` by `hint` (kept if still present), then append the rest sorted. */
  const applyOrder = (items: string[], hint: string[] | undefined): string[] => {
    if (!hint) return [...items].sort();
    const present = new Set(items);
    const seen = new Set<string>();
    const out: string[] = [];
    for (const k of hint) {
      if (present.has(k) && !seen.has(k)) {
        out.push(k);
        seen.add(k);
      }
    }
    for (const k of [...items].sort()) {
      if (!seen.has(k)) out.push(k);
    }
    return out;
  };

  const netAt = (nodeId: string, portId: string): string => {
    const net = portNet.get(portKey(nodeId, portId));
    if (net === undefined) {
      throw new Error(`unresolved net for ${nodeId}.${portId}`);
    }
    return net;
  };

  const rails: Record<string, RailValue> = {};
  const devicesRaw: MosfetNode[] = [];
  const resistorsRaw: ResistorNode[] = [];
  const capsRaw: CapacitorNode[] = [];
  const loadCapsRaw: CapacitorNode[] = [];
  const outputsRaw: OutputNode[] = [];

  for (const n of graph.nodes) {
    switch (n.kind) {
      case "rail":
        rails[n.net] = n.railValue;
        break;
      case "mosfet":
        devicesRaw.push(n);
        break;
      case "resistor":
        resistorsRaw.push(n);
        break;
      case "capacitor":
        if (n.origin === "load_caps") loadCapsRaw.push(n);
        else capsRaw.push(n);
        break;
      case "output":
        outputsRaw.push(n);
        break;
    }
  }

  // ── rails: replay source key order when recorded, else sorted.
  const orderedRails: Record<string, RailValue> = {};
  const railKeys = applyOrder(Object.keys(rails), orderHint.rails);
  for (const k of railKeys) orderedRails[k] = rails[k]!;

  // ── devices (source order replayed, else sorted by name) + models/drives.
  // `modelKwargs` may hold two merged sources: recognized `models` kwargs (vb,
  // corner, ...) that go back into the `models` block, and rare unknown
  // device-object extras that go back onto the device object. Split them here.
  {
    const byName = new Map(devicesRaw.map((m) => [m.name, m]));
    const ordered = applyOrder([...byName.keys()], orderHint.devices);
    devicesRaw.length = 0;
    for (const name of ordered) devicesRaw.push(byName.get(name)!);
  }
  const devices = devicesRaw.map((m) => {
    const extra: Record<string, unknown> = {};
    if (m.modelKwargs) {
      for (const [k, v] of Object.entries(m.modelKwargs)) {
        if (!MODEL_KWARG_KEYS.has(k)) extra[k] = v;
      }
    }
    const d: Parameters<typeof deviceToObject>[0] = {
      name: m.name,
      drain: netAt(m.id, "D"),
      gate: netAt(m.id, "G"),
      source: netAt(m.id, "S"),
      extra,
    };
    // Re-emit embedded W/L only if the source device object carried it.
    if (m.hasEmbeddedWL !== false) {
      d.W = m.W;
      d.L = m.L;
    }
    if (m.nf !== undefined) d.NF = m.nf;
    return deviceToObject(d);
  });

  const models: Record<string, ModelEntry> = {};
  for (const m of devicesRaw) {
    const entry: ModelEntry = {};
    if (m.model !== undefined) entry.type = m.model;
    if (m.modelKwargs) {
      for (const [k, v] of Object.entries(m.modelKwargs)) {
        if (MODEL_KWARG_KEYS.has(k)) entry[k] = v;
      }
    }
    if (Object.keys(entry).length > 0) models[m.name] = entry;
  }

  const inputDrives: Record<string, number> = {};
  for (const m of devicesRaw) {
    if (m.inputDrive !== undefined) inputDrives[m.name] = m.inputDrive;
  }

  // ── resistors / capacitors / load_caps (source order replayed, else sorted;
  // objects for stability).
  const orderNamed = <T extends { name: string }>(
    arr: T[],
    hint: string[] | undefined,
  ): T[] => {
    const byName = new Map(arr.map((x) => [x.name, x]));
    return applyOrder([...byName.keys()], hint).map((n) => byName.get(n)!);
  };

  const resistors: ResistorObject[] = orderNamed(resistorsRaw, orderHint.resistors).map(
    (r) => ({ name: r.name, a: netAt(r.id, "a"), b: netAt(r.id, "b"), R: r.R }),
  );

  const capacitors: CapacitorObject[] = orderNamed(capsRaw, orderHint.capacitors).map(
    (c) => ({ name: c.name, a: netAt(c.id, "a"), b: netAt(c.id, "b"), C: c.C }),
  );

  // load_caps keep import order (id suffix __loadcap_<i>) for a stable diff.
  loadCapsRaw.sort((a, b) => loadCapIndex(a.id) - loadCapIndex(b.id));
  const loadCaps: LoadCapArray[] = loadCapsRaw.map((c) => [
    netAt(c.id, "a"),
    netAt(c.id, "b"),
    c.C,
  ]);

  // ── outputs: preserve the differential ordering via `order`.
  outputsRaw.sort((a, b) => a.order - b.order);
  const outputs = outputsRaw.map((o) => netAt(o.id, "out"));

  // ── solved: every net that is not a rail net (matches loader semantics:
  // solved = internal nodes the solver must find). Source order (== MNA vector
  // order) replayed when recorded, else sorted.
  const railNets = new Set(Object.keys(rails));
  const allNets = new Set(portNet.values());
  const internalNets = [...allNets].filter((net) => !railNets.has(net));
  const solved = applyOrder(internalNets, orderHint.solved);

  // ── ui.positions from node layout (always written; an allowed round-trip
  // difference). Skip synthetic load_cap / output ids? No — keep them so F2 can
  // reposition; they are harmless top-level `ui` data the backend ignores.
  const positions: Record<string, [number, number]> = {};
  const posNodes = [...graph.nodes].sort((a, b) => a.id.localeCompare(b.id));
  for (const n of posNodes) positions[n.id] = [n.position[0], n.position[1]];

  // ── assemble. `rest` first (name, bias, analyses, ...), then graph blocks.
  const out: CircuitJson = {
    ...(rest as Partial<CircuitJson>),
    solved,
    rails: orderedRails,
    devices,
  } as CircuitJson;

  if (outputs.length > 0) out.outputs = outputs;
  else delete out.outputs;

  if (resistors.length > 0) out.resistors = resistors;
  if (capacitors.length > 0) out.capacitors = capacitors;
  if (loadCaps.length > 0) out.load_caps = loadCaps;
  if (Object.keys(models).length > 0) out.models = models;
  if (Object.keys(inputDrives).length > 0) out.input_drives = inputDrives;

  out.ui = { ...(out.ui ?? {}), positions };

  return out;
}

// ── device-kwarg bookkeeping ─────────────────────────────────────────────
// The recognized `models` ctor kwargs (mirrors circuit_loader._MODEL_KWARGS).
// Anything in a node's `modelKwargs` outside this set came from a rare unknown
// device-object extra and is re-emitted on the device object instead.
const MODEL_KWARG_KEYS = new Set(["vb", "corner", "extract_w", "temperature", "NF"]);

function loadCapIndex(id: string): number {
  const m = /__loadcap_(\d+)$/.exec(id);
  return m ? Number(m[1]) : 0;
}

// re-export for tests
export { portKey };
