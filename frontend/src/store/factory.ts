/**
 * Fresh-node factories for palette "add" actions.
 *
 * These live in the store layer (not model/) because creating a blank device is
 * an editor concern, not part of the F1 mapping contract. Every factory returns
 * a domain GraphNode with editor-safe defaults and ports matching the kind's
 * fixed terminal set (mosfet: D/G/S, R/C: a/b, rail/output: one port). Names are
 * auto-assigned with a prefix (M/R/C) that does not collide with existing ids.
 */
import type {
  CapacitorNode,
  CircuitGraph,
  GraphNode,
  MosfetNode,
  OutputNode,
  Position,
  RailNode,
  ResistorNode,
} from "../model";

/**
 * Lowest free "<prefix><n>" name (n >= 1) not already used as a node id. Used so
 * a palette add never shadows an existing element.
 */
export function nextName(graph: CircuitGraph, prefix: string): string {
  const used = new Set(graph.nodes.map((n) => n.id));
  let i = 1;
  while (used.has(`${prefix}${i}`)) i += 1;
  return `${prefix}${i}`;
}

/** Lowest free rail net name (V1, V2, ...) not used as a node id. */
export function nextRailName(graph: CircuitGraph): string {
  const used = new Set(graph.nodes.map((n) => n.id));
  let i = 1;
  while (used.has(`V${i}`)) i += 1;
  return `V${i}`;
}

export interface NewNodeOptions {
  /** Default mosfet model type key (first capabilities.models key at runtime). */
  defaultModel?: string;
}

export function newMosfet(
  graph: CircuitGraph,
  position: Position,
  opts: NewNodeOptions = {},
): MosfetNode {
  const name = nextName(graph, "M");
  const node: MosfetNode = {
    id: name,
    kind: "mosfet",
    name,
    W: 10,
    L: 0.5,
    hasEmbeddedWL: true,
    ports: [{ id: "D" }, { id: "G" }, { id: "S" }],
    position,
  };
  if (opts.defaultModel) node.model = opts.defaultModel;
  return node;
}

export function newResistor(graph: CircuitGraph, position: Position): ResistorNode {
  const name = nextName(graph, "R");
  return {
    id: name,
    kind: "resistor",
    name,
    R: 1000,
    ports: [{ id: "a" }, { id: "b" }],
    position,
  };
}

export function newCapacitor(graph: CircuitGraph, position: Position): CapacitorNode {
  const name = nextName(graph, "C");
  return {
    id: name,
    kind: "capacitor",
    name,
    C: 1e-12,
    origin: "capacitors",
    ports: [{ id: "a" }, { id: "b" }],
    position,
  };
}

export function newRail(graph: CircuitGraph, position: Position): RailNode {
  const net = nextRailName(graph);
  return {
    id: net,
    kind: "rail",
    net,
    railValue: 0.0,
    ports: [{ id: "net" }],
    position,
  };
}

/** New output marker. `order` is the next free slot in the outputs array. */
export function newOutput(graph: CircuitGraph, position: Position): OutputNode {
  const orders = graph.nodes
    .filter((n): n is OutputNode => n.kind === "output")
    .map((n) => n.order);
  const order = orders.length ? Math.max(...orders) + 1 : 0;
  return {
    id: `__out_${order}`,
    kind: "output",
    order,
    ports: [{ id: "out" }],
    position,
  };
}

export function newNode(
  kind: GraphNode["kind"],
  graph: CircuitGraph,
  position: Position,
  opts: NewNodeOptions = {},
): GraphNode {
  switch (kind) {
    case "mosfet":
      return newMosfet(graph, position, opts);
    case "resistor":
      return newResistor(graph, position);
    case "capacitor":
      return newCapacitor(graph, position);
    case "rail":
      return newRail(graph, position);
    case "output":
      return newOutput(graph, position);
  }
}
