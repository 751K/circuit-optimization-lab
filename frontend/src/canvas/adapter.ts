/**
 * RF adapter layer: domain GraphNode/GraphEdge  <->  React Flow Node/Edge.
 *
 * This is the ONLY place @xyflow/react types touch the domain model — the F1
 * contract keeps model/ framework-free, so all RF glue lives here. id and
 * position pass straight through; the whole domain node is stashed on
 * `rfNode.data.node` (plus resolved net labels for the custom node components).
 *
 * The adapter is a *pure projection*: identity holds in the sense the tests
 * assert — rfToDomainNode(domainToRfNode(n)) deep-equals n, and likewise for the
 * whole graph. Net labels are display-only and never round-trip back.
 */
import type { Edge, Node } from "@xyflow/react";
import {
  resolveNets,
  type CircuitGraph,
  type GraphEdge,
  type GraphNode,
} from "../model";
import { portSides, type PortSide } from "./sides";

/** Net label per port for a node, keyed by port id (e.g. { D: "vout", G: "vinp" }). */
export type PortNets = Record<string, string>;

/** Data carried on every RF node: the domain node + its per-port resolved nets. */
export interface RfNodeData extends Record<string, unknown> {
  node: GraphNode;
  portNets: PortNets;
  /** Port ids that carry >=2 edges -> render a junction dot (schematic tee). */
  junctions?: string[];
  /**
   * Preferred handle side per port, from the layout geometry (see sides.ts).
   * Display only — a ground below the drawing wires upward, a vertical load
   * capacitor draws vertically. Absent means "use the symbol's own default".
   */
  sides?: Record<string, PortSide>;
}

export type RfNode = Node<RfNodeData>;

/**
 * What a wire is for, which is what it is drawn like: `rail` is a power bus
 * (dimmed — a schematic does not draw long supply wires), `bias` is a
 * fixed-potential net feeding gates (drawn to be traced), `signal` is everything
 * else. See {@link railKinds}.
 */
export type EdgeRole = "signal" | "bias" | "rail";

/** Edge label = the resolved net name the edge lies on (display only). */
export interface RfEdgeData extends Record<string, unknown> {
  edge: GraphEdge;
  net?: string;
  /** true when this edge is flagged as part of a net-conflict short. */
  conflict?: boolean;
  /** what the wire carries; drives the stroke style. */
  rail?: EdgeRole;
}

export type RfEdge = Edge<RfEdgeData>;

const SEP = String.fromCharCode(31);
const pk = (node: string, port: string): string => `${node}${SEP}${port}`;

/**
 * Wire length past which an edge is labelled with its net. Roughly a column and
 * a half of the schematic layout — long enough that the port labels at the two
 * ends are no longer in the same glance.
 */
const LABEL_MIN_LENGTH = 320;

/** Map each node's ports to resolved net names (best-effort; empty on conflict). */
function resolvePortNets(graph: CircuitGraph): Map<string, PortNets> {
  const out = new Map<string, PortNets>();
  let portNet: Map<string, string>;
  try {
    portNet = resolveNets(graph).portNet;
  } catch {
    // A net conflict throws; fall back to no labels rather than crash the canvas.
    portNet = new Map();
  }
  for (const n of graph.nodes) {
    const pn: PortNets = {};
    for (const p of n.ports) {
      const net = portNet.get(pk(n.id, p.id));
      if (net !== undefined) pn[p.id] = net;
    }
    out.set(n.id, pn);
  }
  return out;
}

/** Single domain node -> RF node (position/id passthrough, domain node on data). */
export function domainToRfNode(
  node: GraphNode,
  portNets: PortNets = {},
  junctions: string[] = [],
  sides?: Record<string, PortSide>,
): RfNode {
  const data: RfNodeData = { node, portNets };
  if (junctions.length > 0) data.junctions = junctions;
  if (sides && Object.keys(sides).length > 0) data.sides = sides;
  return {
    id: node.id,
    type: node.kind,
    position: { x: node.position[0], y: node.position[1] },
    data,
  };
}

/**
 * Count edges incident on each (node, port) and return, per node, the set of
 * port ids that touch >=2 edges — a schematic tee where a junction dot belongs.
 */
export function junctionPortsByNode(
  edges: GraphEdge[],
): Map<string, Set<string>> {
  const counts = new Map<string, Map<string, number>>();
  const bump = (node: string, port: string): void => {
    let m = counts.get(node);
    if (!m) {
      m = new Map();
      counts.set(node, m);
    }
    m.set(port, (m.get(port) ?? 0) + 1);
  };
  for (const e of edges) {
    bump(e.source.node, e.source.port);
    bump(e.target.node, e.target.port);
  }
  const out = new Map<string, Set<string>>();
  for (const [node, m] of counts) {
    const dots = new Set<string>();
    for (const [port, c] of m) if (c >= 2) dots.add(port);
    if (dots.size > 0) out.set(node, dots);
  }
  return out;
}

/** RF node -> domain node. Reads id/position from RF, everything else from data. */
export function rfToDomainNode(rf: RfNode): GraphNode {
  // Rehydrate id + position from the RF envelope so a dragged node's new
  // position is reflected; the rest is the stored domain node.
  return {
    ...rf.data.node,
    id: rf.id,
    position: [rf.position.x, rf.position.y],
  } as GraphNode;
}

/** A net-name -> css-safe token map, so a class can key off the net. */
export function netClass(net: string | undefined): string | undefined {
  if (net === undefined) return undefined;
  // Encode to a class-safe suffix; any non [A-Za-z0-9_-] char -> its code point.
  const safe = net.replace(/[^A-Za-z0-9_-]/g, (c) => `_${c.codePointAt(0)}_`);
  return `net-${safe}`;
}

/**
 * Domain edge -> RF edge (label carries the resolved net).
 *
 * `rail` marks the edge as belonging to a fixed-potential (power/ground) net so
 * the canvas can dim it — real schematics never draw long power wires, so
 * dimming them makes the signal path legible. Every edge also gets a stable
 * per-net class (`net-<safe>`) so a net-level hover highlight is pure CSS and
 * never rebuilds the edge objects.
 */
export function domainToRfEdge(
  edge: GraphEdge,
  net: string | undefined,
  conflict: boolean,
  rail: EdgeRole = "signal",
  labelled = true,
): RfEdge {
  const classes = [
    conflict ? "edge-conflict" : undefined,
    `edge-${rail}`,
    netClass(net),
  ].filter((c): c is string => c !== undefined);
  // `pathOptions` is a smoothstep-variant field not on the base Edge type; RF
  // reads it at render, so we attach it via a narrow cast rather than widening
  // RfEdge to the whole discriminated edge union.
  return {
    id: edge.id,
    source: edge.source.node,
    target: edge.target.node,
    sourceHandle: edge.source.port,
    targetHandle: edge.target.port,
    type: "smoothstep",
    pathOptions: { borderRadius: 8 },
    label: labelled ? net : undefined,
    data: { edge, net, conflict, rail },
    className: classes.length > 0 ? classes.join(" ") : undefined,
  } as RfEdge;
}

/** RF edge -> domain edge. */
export function rfToDomainEdge(rf: RfEdge): GraphEdge {
  if (rf.data?.edge) return rf.data.edge;
  // Rebuild from RF fields when data isn't present (e.g. edge created by RF).
  return {
    id: rf.id,
    source: { node: rf.source, port: rf.sourceHandle ?? "" },
    target: { node: rf.target, port: rf.targetHandle ?? "" },
  };
}

/** Net names that are fixed-potential rails (a rail node names one net each). */
export function railNetsOf(graph: CircuitGraph): Set<string> {
  const rails = new Set<string>();
  for (const n of graph.nodes) {
    if (n.kind === "rail") rails.add(n.net);
  }
  return rails;
}

/**
 * Split the rail nets into power buses and bias taps.
 *
 * Both used to be dimmed to a thin dashed grey, on the reasoning that real
 * schematics do not draw long power wires. That is true of VDD and ground, and
 * wrong for a bias: `vbias`, `vinp`, `VB_CN` set the operating point of exactly
 * one device each, they are short wires you are meant to trace, and rendering
 * them as faint dashes made them the hardest thing on the canvas to read. The
 * distinction is structural, not a name list — a rail every element port of
 * which is a mosfet *gate* controls, it does not supply.
 */
export function railKinds(
  graph: CircuitGraph,
  portNetsByNode: Map<string, PortNets>,
): { buses: Set<string>; taps: Set<string> } {
  const railNets = railNetsOf(graph);
  const buses = new Set<string>();
  const taps = new Set<string>();
  const carries = new Map<string, boolean>(); // net -> some port is not a gate
  for (const n of graph.nodes) {
    if (n.kind === "rail") continue;
    const nets = portNetsByNode.get(n.id) ?? {};
    for (const [port, net] of Object.entries(nets)) {
      if (!railNets.has(net)) continue;
      const control = n.kind === "mosfet" && port === "G";
      carries.set(net, (carries.get(net) ?? false) || !control);
    }
  }
  for (const net of railNets) (carries.get(net) ? buses : taps).add(net);
  return { buses, taps };
}

/** Whole domain graph -> RF nodes/edges, with resolved net labels + conflict flags. */
export function domainToRf(
  graph: CircuitGraph,
  conflictEdgeIds: Set<string> = new Set(),
): { nodes: RfNode[]; edges: RfEdge[] } {
  const portNetsByNode = resolvePortNets(graph);
  const { buses, taps } = railKinds(graph, portNetsByNode);
  const junctions = junctionPortsByNode(graph.edges);
  const sides = portSides(graph);
  const nodes = graph.nodes.map((n) =>
    domainToRfNode(
      n,
      portNetsByNode.get(n.id) ?? {},
      [...(junctions.get(n.id) ?? [])],
      sides.get(n.id),
    ),
  );
  const pos = new Map(graph.nodes.map((n) => [n.id, n.position]));
  const edges = graph.edges.map((e) => {
    // The edge's net = the net of its source port.
    const net = portNetsByNode.get(e.source.node)?.[e.source.port];
    const rail: EdgeRole = net === undefined ? "signal"
      : buses.has(net) ? "rail"
        : taps.has(net) ? "bias" : "signal";
    // Every port already prints its own net beside the handle, so labelling a
    // short wire as well just says the same thing three times in one square
    // inch. A long run is different: there the label is the only way to tell
    // what a wire crossing half the drawing carries.
    const a = pos.get(e.source.node);
    const b = pos.get(e.target.node);
    const long = a === undefined || b === undefined
      || Math.hypot(b[0] - a[0], b[1] - a[1]) > LABEL_MIN_LENGTH;
    return domainToRfEdge(e, net, conflictEdgeIds.has(e.id), rail, long);
  });
  return { nodes, edges };
}
