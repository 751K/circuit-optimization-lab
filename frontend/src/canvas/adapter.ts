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

/** Net label per port for a node, keyed by port id (e.g. { D: "vout", G: "vinp" }). */
export type PortNets = Record<string, string>;

/** Data carried on every RF node: the domain node + its per-port resolved nets. */
export interface RfNodeData extends Record<string, unknown> {
  node: GraphNode;
  portNets: PortNets;
  /** Port ids that carry >=2 edges -> render a junction dot (schematic tee). */
  junctions?: string[];
}

export type RfNode = Node<RfNodeData>;

/** Edge label = the resolved net name the edge lies on (display only). */
export interface RfEdgeData extends Record<string, unknown> {
  edge: GraphEdge;
  net?: string;
  /** true when this edge is flagged as part of a net-conflict short. */
  conflict?: boolean;
  /** true when the edge's net is a fixed-potential (rail) net -> dimmed. */
  rail?: boolean;
}

export type RfEdge = Edge<RfEdgeData>;

const SEP = String.fromCharCode(31);
const pk = (node: string, port: string): string => `${node}${SEP}${port}`;

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
): RfNode {
  return {
    id: node.id,
    type: node.kind,
    position: { x: node.position[0], y: node.position[1] },
    data: junctions.length > 0 ? { node, portNets, junctions } : { node, portNets },
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
  rail = false,
): RfEdge {
  const classes = [
    conflict ? "edge-conflict" : undefined,
    rail ? "edge-rail" : "edge-signal",
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
    label: net,
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

/** Whole domain graph -> RF nodes/edges, with resolved net labels + conflict flags. */
export function domainToRf(
  graph: CircuitGraph,
  conflictEdgeIds: Set<string> = new Set(),
): { nodes: RfNode[]; edges: RfEdge[] } {
  const portNetsByNode = resolvePortNets(graph);
  const railNets = railNetsOf(graph);
  const junctions = junctionPortsByNode(graph.edges);
  const nodes = graph.nodes.map((n) =>
    domainToRfNode(
      n,
      portNetsByNode.get(n.id) ?? {},
      [...(junctions.get(n.id) ?? [])],
    ),
  );
  const edges = graph.edges.map((e) => {
    // The edge's net = the net of its source port.
    const net = portNetsByNode.get(e.source.node)?.[e.source.port];
    const rail = net !== undefined && railNets.has(net);
    return domainToRfEdge(e, net, conflictEdgeIds.has(e.id), rail);
  });
  return { nodes, edges };
}
