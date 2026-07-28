/**
 * Which side of a node each wire should leave from.
 *
 * A mosfet has a fixed schematic geometry — gate on the left, channel along the
 * vertical — so its sides are decided by the symbol itself. Everything else does
 * not: a ground symbol below the drawing must send its wire *up*, a supply above
 * it must send one *down*, and a load capacitor bridging a signal node to ground
 * is a vertical element, not the horizontal one the symbol defaults to. Fixing
 * the handle side is what turns those wires from a loop around the node body
 * into a straight segment.
 *
 * None of that is knowable from the netlist alone — it depends on where the
 * layout actually put things — so it is derived here, from geometry, after the
 * positions exist. It is presentation only: no port, edge, or net is changed,
 * and nothing here round-trips.
 */
import { Position as RfPosition } from "@xyflow/react";
import type { CircuitGraph, GraphEdge, Position } from "../model";

export type PortSide = "top" | "bottom" | "left" | "right";

/** The React Flow handle position for a side. */
export const RF_SIDE: Record<PortSide, RfPosition> = {
  top: RfPosition.Top,
  bottom: RfPosition.Bottom,
  left: RfPosition.Left,
  right: RfPosition.Right,
};

/** Preferred side per port id, per node id. Absent = the symbol's own default. */
export type SidesByNode = Map<string, Record<string, PortSide>>;

/** Mean position of everything a given port is wired to. */
function neighbourCentre(
  nodeId: string,
  portId: string,
  edges: GraphEdge[],
  pos: Map<string, Position>,
): Position | undefined {
  let sx = 0;
  let sy = 0;
  let n = 0;
  for (const e of edges) {
    let other: { node: string; port: string } | undefined;
    if (e.source.node === nodeId && e.source.port === portId) other = e.target;
    else if (e.target.node === nodeId && e.target.port === portId) other = e.source;
    if (!other) continue;
    const p = pos.get(other.node);
    if (!p) continue;
    sx += p[0];
    sy += p[1];
    n += 1;
  }
  return n === 0 ? undefined : [sx / n, sy / n];
}

/**
 * Compute the preferred sides for the kinds whose orientation is free.
 *
 * A single-port node (rail, output marker) faces whatever it is wired to. A
 * two-terminal element picks the axis its two ends are actually separated
 * along — vertical when the rise between them beats the run — and then puts
 * `a` and `b` on opposite ends of it. Ties resolve to the horizontal default,
 * so an element with nothing to go on keeps the symbol's usual look.
 */
export function portSides(graph: CircuitGraph): SidesByNode {
  const pos = new Map<string, Position>(graph.nodes.map((n) => [n.id, n.position]));
  const out: SidesByNode = new Map();

  for (const node of graph.nodes) {
    if (node.kind === "rail" || node.kind === "output") {
      const port = node.ports[0];
      if (!port) continue;
      const here = pos.get(node.id)!;
      const there = neighbourCentre(node.id, port.id, graph.edges, pos);
      if (!there) continue;
      const dx = there[0] - here[0];
      const dy = there[1] - here[1];
      const side: PortSide = Math.abs(dy) > Math.abs(dx)
        ? (dy > 0 ? "bottom" : "top")
        : (dx > 0 ? "right" : "left");
      out.set(node.id, { [port.id]: side });
      continue;
    }

    if (node.kind !== "resistor" && node.kind !== "capacitor") continue;
    const ca = neighbourCentre(node.id, "a", graph.edges, pos);
    const cb = neighbourCentre(node.id, "b", graph.edges, pos);
    if (!ca || !cb) continue;
    const dx = cb[0] - ca[0];
    const dy = cb[1] - ca[1];
    out.set(node.id, Math.abs(dy) > Math.abs(dx)
      ? (dy > 0 ? { a: "top", b: "bottom" } : { a: "bottom", b: "top" })
      : (dx > 0 ? { a: "left", b: "right" } : { a: "right", b: "left" }));
  }
  return out;
}
