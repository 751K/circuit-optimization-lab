/**
 * Rail node: a named fixed-potential net. Shows the net name and its bias value
 * (resolved numeric, or the bias-key string).
 *
 * The single wire leaves on whichever side faces what the rail feeds (see
 * sides.ts), and the symbol's stem follows it: a supply drawn above the circuit
 * hangs its bar over a downward stem, a ground drawn below it stands the stem
 * up under the bar. Both are the conventional drawing, and both stop the wire
 * having to double back around the node body.
 */
import { Handle, type NodeProps } from "@xyflow/react";
import type { RfNodeData } from "../adapter";
import type { RailNode as RailDomain } from "../../model";
import { RF_SIDE } from "../sides";
import Junction from "./Junction";

export default function RailNode({ data, selected }: NodeProps) {
  const d = data as RfNodeData;
  const node = d.node as RailDomain;
  const j = new Set(d.junctions ?? []);
  const side = d.sides?.["net"] ?? "bottom";
  const shown =
    node.biasValue !== undefined
      ? `${node.railValue} = ${node.biasValue}`
      : String(node.railValue);
  // The bar sits on the far side from the wire; the stem runs from it toward the
  // handle. `[bar, stem]` per side, in the 44x30 viewBox.
  const [bar, stem] = {
    top: [{ x1: 4, y1: 26, x2: 40, y2: 26 }, { x1: 22, y1: 26, x2: 22, y2: 8 }],
    bottom: [{ x1: 4, y1: 6, x2: 40, y2: 6 }, { x1: 22, y1: 6, x2: 22, y2: 24 }],
    left: [{ x1: 38, y1: 4, x2: 38, y2: 26 }, { x1: 38, y1: 15, x2: 6, y2: 15 }],
    right: [{ x1: 6, y1: 4, x2: 6, y2: 26 }, { x1: 6, y1: 15, x2: 38, y2: 15 }],
  }[side];
  return (
    <div className={`cnode rail ${selected ? "selected" : ""}`}>
      <svg width="44" height="30" viewBox="0 0 44 30" className="sym">
        <line {...bar} stroke="currentColor" strokeWidth="2.5" />
        <line {...stem} stroke="currentColor" strokeWidth="2" />
      </svg>
      <div className="cnode-body">
        <div className="cnode-name">{node.net}</div>
        <div className="cnode-sub">{shown}</div>
      </div>
      <Handle type="source" position={RF_SIDE[side]} id="net" className="handle" />
      <Junction active={j.has("net")} side={side} />
    </div>
  );
}
