/**
 * The editor state layer (zustand). Owns the domain `graph` + preserved `rest`,
 * the current selection, an undo/redo snapshot stack, and the backend
 * capabilities probe. Every mutating action goes through `commit()` so history
 * is captured uniformly.
 *
 * F3 note: the Run panel reads `graph`/`rest` here and calls the api/client
 * `solve`/`validate` directly with `graphToCircuitJson(graph, rest)`; it does
 * not need new store actions, but `capabilities` (analyses/corners) is already
 * cached here for its dropdowns.
 */
import { create } from "zustand";
import {
  circuitJsonToGraph,
  graphToCircuitJson,
  resolveNets,
  schematicLayout,
  type CircuitGraph,
  type CircuitJson,
  type GraphEdge,
  type GraphNode,
  type LayoutNode,
  type Position,
} from "../model";
import { capabilities, type CapabilitiesResponse } from "../api/client";
import { newNode, type NewNodeOptions } from "./factory";

/** What is currently selected on the canvas. */
export interface Selection {
  nodes: string[];
  edges: string[];
}

const EMPTY_SELECTION: Selection = { nodes: [], edges: [] };

/** An empty starting circuit (used by newCircuit). */
function emptyGraph(): CircuitGraph {
  return { nodes: [], edges: [] };
}

const HISTORY_LIMIT = 50;

interface Snapshot {
  graph: CircuitGraph;
  rest: Record<string, unknown>;
}

export interface EditorState {
  // ── document ──────────────────────────────────────────────────────────
  graph: CircuitGraph;
  /** Verbatim passthrough blocks (bias, analyses, name, ...) for lossless export. */
  rest: Record<string, unknown>;

  // ── selection ─────────────────────────────────────────────────────────
  selection: Selection;

  // ── history ───────────────────────────────────────────────────────────
  past: Snapshot[];
  future: Snapshot[];

  // ── backend capabilities ──────────────────────────────────────────────
  caps: CapabilitiesResponse | null;
  capsError: string | null;
  capsLoading: boolean;

  // ── net-resolution error (double-rail short) ─────────────────────────
  /** Message + offending edge ids when resolveNets throws; null when clean. */
  netError: { message: string; edgeIds: string[] } | null;

  // ── this circuit's corner set (from /validate) ────────────────────────
  /**
   * The corners *this* circuit admits, or null before the first validation.
   * Not a slice of `caps.corners`: a circuit belongs to one model family, and
   * the OTFT process names and silicon card corners are disjoint sets. The
   * corner menu is driven from here so it can never offer a name that cannot
   * resolve. `circuitSilicon` says whether the PVT temperature / supply axes
   * exist at all for this circuit.
   */
  circuitCorners: string[] | null;
  circuitSilicon: boolean;
  setCircuitCorners: (corners: string[] | null, silicon: boolean) => void;

  // ── actions ───────────────────────────────────────────────────────────
  addNode: (kind: GraphNode["kind"], position: Position, opts?: NewNodeOptions) => string;
  moveNode: (id: string, position: Position) => void;
  updateNodeProps: (id: string, patch: Partial<GraphNode>) => void;
  /** Rename a node: rewrites its id + name and every edge endpoint. Returns the
   *  effective new id (falls back to the old id if `newId` collides or is empty). */
  renameNode: (id: string, newId: string) => string;
  connect: (
    source: { node: string; port: string },
    target: { node: string; port: string },
  ) => void;
  deleteSelection: () => void;
  deleteNodes: (ids: string[]) => void;
  deleteEdges: (ids: string[]) => void;
  setSelection: (sel: Selection) => void;
  loadCircuit: (json: CircuitJson) => void;
  newCircuit: (name?: string) => void;
  exportJson: () => CircuitJson;
  /**
   * Re-run the schematic auto-layout over the whole graph, discarding the
   * current positions. On import the layout runs for you, but a circuit that
   * has since been edited by hand — or one whose archived `ui.positions` came
   * from the old kind-column layout — needs a way to ask for it again. A no-op
   * when the circuit has no rail to rank against; undo-able either way.
   */
  relayout: () => void;
  /**
   * Flip the given devices horizontally — gate on the other side. Display only:
   * no port, net or edge changes, and the exported netlist is identical bar the
   * `ui.mirrored` list. Ids that are not mosfets are ignored, so this can be
   * handed a mixed selection.
   */
  mirrorNodes: (ids: string[]) => void;
  /**
   * Bumped whenever the document is replaced wholesale (load / new / undo /
   * redo / relayout) rather than edited. The canvas watches it to re-fit the
   * viewport, which an ordinary edit must not do.
   */
  viewEpoch: number;
  /**
   * Write (or, with `null`, remove) one analysis's config in the circuit's
   * `analyses` block. This is a real document edit — undo-able and exported —
   * unlike the request-only defaults the Run panel injects, which never touch
   * the document.
   */
  setAnalysisConfig: (name: string, config: Record<string, unknown> | null) => void;

  undo: () => void;
  redo: () => void;

  fetchCapabilities: () => Promise<void>;
}

/** Cheap structural clone for snapshots (nodes/edges are plain JSON-ish). */
function clone<T>(v: T): T {
  return structuredClone(v);
}

/**
 * Recompute the net-resolution error by attempting resolveNets. A double-rail
 * short throws; we surface the message and try to pin down which edges caused it
 * (best-effort: the edges inside components that touch >1 rail).
 */
function computeNetError(graph: CircuitGraph): EditorState["netError"] {
  try {
    resolveNets(graph);
    return null;
  } catch (e) {
    return { message: e instanceof Error ? e.message : String(e), edgeIds: findConflictEdges(graph) };
  }
}

/**
 * Best-effort: find edges whose endpoints reach two different rails. We do a
 * union-find ourselves (mirroring resolveNets) and flag every edge in a
 * component that contains ports from >1 distinct rail net.
 */
function findConflictEdges(graph: CircuitGraph): string[] {
  const parent = new Map<string, string>();
  const find = (x: string): string => {
    let r = parent.get(x);
    if (r === undefined) {
      parent.set(x, x);
      return x;
    }
    while (r !== x) {
      const gp = parent.get(r) ?? r;
      parent.set(x, gp);
      x = gp;
      r = parent.get(x) ?? x;
    }
    return x;
  };
  const union = (a: string, b: string): void => {
    const ra = find(a);
    const rb = find(b);
    if (ra !== rb) parent.set(ra, rb);
  };
  const key = (n: string, p: string): string => `${n}${p}`;
  for (const n of graph.nodes) for (const p of n.ports) find(key(n.id, p.id));
  for (const e of graph.edges) union(key(e.source.node, e.source.port), key(e.target.node, e.target.port));

  // component root -> set of rail nets present
  const railsIn = new Map<string, Set<string>>();
  for (const n of graph.nodes) {
    if (n.kind !== "rail") continue;
    for (const p of n.ports) {
      const root = find(key(n.id, p.id));
      let s = railsIn.get(root);
      if (!s) {
        s = new Set();
        railsIn.set(root, s);
      }
      s.add(n.net);
    }
  }
  const badRoots = new Set<string>();
  for (const [root, s] of railsIn) if (s.size > 1) badRoots.add(root);
  const out: string[] = [];
  for (const e of graph.edges) {
    const root = find(key(e.source.node, e.source.port));
    if (badRoots.has(root)) out.push(e.id);
  }
  return out;
}

export const useEditor = create<EditorState>((set, get) => {
  /** Push the current doc onto the undo stack, apply `next`, clear redo. */
  const commit = (next: Snapshot): void => {
    const { graph, rest, past } = get();
    const snapshot: Snapshot = { graph: clone(graph), rest: clone(rest) };
    const trimmed = past.length >= HISTORY_LIMIT ? past.slice(past.length - HISTORY_LIMIT + 1) : past;
    set({
      graph: next.graph,
      rest: next.rest,
      past: [...trimmed, snapshot],
      future: [],
      netError: computeNetError(next.graph),
    });
  };

  return {
    graph: emptyGraph(),
    rest: {},
    selection: EMPTY_SELECTION,
    past: [],
    future: [],
    caps: null,
    capsError: null,
    capsLoading: false,
    netError: null,
    circuitCorners: null,
    circuitSilicon: false,
    viewEpoch: 0,

    setCircuitCorners: (corners, silicon) =>
      set({ circuitCorners: corners, circuitSilicon: silicon }),

    addNode: (kind, position, opts) => {
      const { graph, rest } = get();
      const node = newNode(kind, graph, position, opts);
      commit({ graph: { nodes: [...graph.nodes, node], edges: graph.edges }, rest });
      set({ selection: { nodes: [node.id], edges: [] } });
      return node.id;
    },

    moveNode: (id, position) => {
      // Position drags are frequent; still commit (undo-able) but keep it cheap.
      const { graph, rest } = get();
      const nodes = graph.nodes.map((n) => (n.id === id ? ({ ...n, position } as GraphNode) : n));
      commit({ graph: { nodes, edges: graph.edges }, rest });
    },

    updateNodeProps: (id, patch) => {
      const { graph, rest } = get();
      // Guard: id/name changes go through renameNode (it also fixes edges).
      const { id: _dropId, ...safe } = patch as Record<string, unknown>;
      void _dropId;
      const nodes = graph.nodes.map((n) => {
        if (n.id !== id) return n;
        return { ...n, ...safe, kind: n.kind, id: n.id } as GraphNode;
      });
      commit({ graph: { nodes, edges: graph.edges }, rest });
    },

    renameNode: (id, rawNewId) => {
      const { graph, rest } = get();
      const newId = rawNewId.trim();
      // Reject empty or colliding names; keep the old id.
      if (!newId || newId === id) return id;
      if (graph.nodes.some((n) => n.id === newId)) return id;
      const nodes = graph.nodes.map((n) => {
        if (n.id !== id) return n;
        const patched = { ...n, id: newId } as GraphNode;
        // Keep the human-visible name in sync where the kind carries one.
        if ("name" in patched) (patched as { name: string }).name = newId;
        if (patched.kind === "rail") (patched as { net: string }).net = newId;
        return patched;
      });
      const edges = graph.edges.map((e) => ({
        ...e,
        source: e.source.node === id ? { ...e.source, node: newId } : e.source,
        target: e.target.node === id ? { ...e.target, node: newId } : e.target,
      }));
      commit({ graph: { nodes, edges }, rest });
      // Move selection to the renamed node.
      const sel = get().selection;
      if (sel.nodes.includes(id)) {
        set({ selection: { nodes: sel.nodes.map((s) => (s === id ? newId : s)), edges: sel.edges } });
      }
      return newId;
    },

    connect: (source, target) => {
      const { graph, rest } = get();
      // No self-loop on the same port; no duplicate edge.
      if (source.node === target.node && source.port === target.port) return;
      const dup = graph.edges.some(
        (e) =>
          (e.source.node === source.node &&
            e.source.port === source.port &&
            e.target.node === target.node &&
            e.target.port === target.port) ||
          (e.source.node === target.node &&
            e.source.port === target.port &&
            e.target.node === source.node &&
            e.target.port === source.port),
      );
      if (dup) return;
      const edge: GraphEdge = {
        id: `e_${source.node}.${source.port}-${target.node}.${target.port}_${Date.now().toString(36)}`,
        source,
        target,
      };
      commit({ graph: { nodes: graph.nodes, edges: [...graph.edges, edge] }, rest });
    },

    deleteNodes: (ids) => {
      const { graph, rest } = get();
      const drop = new Set(ids);
      const nodes = graph.nodes.filter((n) => !drop.has(n.id));
      // Drop any edge touching a removed node.
      const edges = graph.edges.filter((e) => !drop.has(e.source.node) && !drop.has(e.target.node));
      commit({ graph: { nodes, edges }, rest });
      set({ selection: EMPTY_SELECTION });
    },

    deleteEdges: (ids) => {
      const { graph, rest } = get();
      const drop = new Set(ids);
      const edges = graph.edges.filter((e) => !drop.has(e.id));
      commit({ graph: { nodes: graph.nodes, edges }, rest });
      set({ selection: EMPTY_SELECTION });
    },

    deleteSelection: () => {
      const { selection } = get();
      if (selection.nodes.length === 0 && selection.edges.length === 0) return;
      const dropN = new Set(selection.nodes);
      const dropE = new Set(selection.edges);
      const { graph, rest } = get();
      const nodes = graph.nodes.filter((n) => !dropN.has(n.id));
      const edges = graph.edges.filter(
        (e) => !dropE.has(e.id) && !dropN.has(e.source.node) && !dropN.has(e.target.node),
      );
      commit({ graph: { nodes, edges }, rest });
      set({ selection: EMPTY_SELECTION });
    },

    setSelection: (sel) => set({ selection: sel }),

    loadCircuit: (json) => {
      const { graph, rest } = get();
      // Snapshot current before replacing so a load is undo-able.
      const snapshot: Snapshot = { graph: clone(graph), rest: clone(rest) };
      const { graph: g, rest: r } = circuitJsonToGraph(json);
      set({
        graph: g,
        rest: r,
        selection: EMPTY_SELECTION,
        past: [...get().past, snapshot].slice(-HISTORY_LIMIT),
        future: [],
        netError: computeNetError(g),
        viewEpoch: get().viewEpoch + 1,
      });
    },

    newCircuit: (name) => {
      const { graph, rest } = get();
      const snapshot: Snapshot = { graph: clone(graph), rest: clone(rest) };
      const newRest: Record<string, unknown> = name ? { name } : {};
      set({
        graph: emptyGraph(),
        rest: newRest,
        selection: EMPTY_SELECTION,
        past: [...get().past, snapshot].slice(-HISTORY_LIMIT),
        future: [],
        netError: null,
        viewEpoch: get().viewEpoch + 1,
      });
    },

    exportJson: () => {
      const { graph, rest } = get();
      return graphToCircuitJson(graph, rest);
    },

    mirrorNodes: (ids) => {
      const { graph, rest } = get();
      const flip = new Set(ids);
      const targets = graph.nodes.filter((n) => flip.has(n.id) && n.kind === "mosfet");
      if (targets.length === 0) return;
      // A mixed selection flips as one: if any target is unmirrored they all
      // become mirrored, so repeated presses toggle the group rather than
      // scrambling it into alternating orientations.
      const next = !targets.every((n) => (n as GraphNode & { mirrored?: boolean }).mirrored === true);
      const nodes = graph.nodes.map((n) => {
        if (!flip.has(n.id) || n.kind !== "mosfet") return n;
        const patched = { ...n } as GraphNode & { mirrored?: boolean };
        if (next) patched.mirrored = true;
        else delete patched.mirrored;
        return patched as GraphNode;
      });
      commit({ graph: { nodes, edges: graph.edges }, rest });
    },

    relayout: () => {
      const { graph, rest } = get();
      if (graph.nodes.length === 0) return;
      // Feed the layout the *resolved* nets rather than each port's remembered
      // originalNet: a port wired up in the editor has no original net, and
      // laying out what the user has actually built is the whole point here.
      let portNet: Map<string, string>;
      try {
        ({ portNet } = resolveNets(graph));
      } catch {
        return; // a double-rail short has no coherent netlist to lay out
      }
      const SEP = String.fromCharCode(31);
      const layoutNodes: LayoutNode[] = graph.nodes.map((n) => ({
        id: n.id,
        kind: n.kind,
        ports: n.ports
          .map((p) => ({ id: p.id, net: portNet.get(`${n.id}${SEP}${p.id}`) }))
          .filter((p): p is { id: string; net: string } => p.net !== undefined),
        ...(n.kind === "mosfet" && n.mirrored === true ? { mirrored: true } : {}),
      }));
      const potentials = new Map<string, number>();
      for (const n of graph.nodes) {
        if (n.kind !== "rail") continue;
        const v = typeof n.railValue === "number" ? n.railValue : n.biasValue;
        if (typeof v === "number" && Number.isFinite(v)) potentials.set(n.net, v);
      }
      const placed = schematicLayout(layoutNodes, potentials);
      if (!placed) return;
      const nodes = graph.nodes.map((n) => {
        const p = placed.get(n.id);
        return p ? ({ ...n, position: [p[0], p[1]] as Position } as GraphNode) : n;
      });
      commit({ graph: { nodes, edges: graph.edges }, rest });
      set({ viewEpoch: get().viewEpoch + 1 });
    },

    setAnalysisConfig: (name, config) => {
      const { graph, rest } = get();
      const current = rest.analyses;
      // A malformed non-object `analyses` is passed through untouched everywhere
      // else in this codebase; rewriting bytes we do not understand would be the
      // one place a GUI could silently corrupt a hand-written circuit.
      if (current !== undefined
        && (typeof current !== "object" || current === null || Array.isArray(current))) {
        return;
      }
      const analyses: Record<string, unknown> = { ...(current ?? {}) };
      if (config === null) delete analyses[name];
      else analyses[name] = config;
      const nextRest = { ...rest };
      if (Object.keys(analyses).length === 0) delete nextRest.analyses;
      else nextRest.analyses = analyses;
      commit({ graph, rest: nextRest });
    },

    undo: () => {
      const { past, future, graph, rest } = get();
      if (past.length === 0) return;
      const prev = past[past.length - 1]!;
      const current: Snapshot = { graph: clone(graph), rest: clone(rest) };
      set({
        graph: prev.graph,
        rest: prev.rest,
        past: past.slice(0, -1),
        future: [current, ...future].slice(0, HISTORY_LIMIT),
        selection: EMPTY_SELECTION,
        netError: computeNetError(prev.graph),
        viewEpoch: get().viewEpoch + 1,
      });
    },

    redo: () => {
      const { past, future, graph, rest } = get();
      if (future.length === 0) return;
      const next = future[0]!;
      const current: Snapshot = { graph: clone(graph), rest: clone(rest) };
      set({
        graph: next.graph,
        rest: next.rest,
        past: [...past, current].slice(-HISTORY_LIMIT),
        future: future.slice(1),
        selection: EMPTY_SELECTION,
        netError: computeNetError(next.graph),
        viewEpoch: get().viewEpoch + 1,
      });
    },

    fetchCapabilities: async () => {
      set({ capsLoading: true, capsError: null });
      try {
        const caps = await capabilities();
        set({ caps, capsLoading: false, capsError: null });
      } catch (e) {
        set({ capsLoading: false, capsError: e instanceof Error ? e.message : String(e) });
      }
    },
  };
});
