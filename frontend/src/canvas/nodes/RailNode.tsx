/**
 * Rail node: a named fixed-potential net. Single handle (bottom). Shows the net
 * name and its bias value (resolved numeric, or the bias-key string).
 */
import { Handle, Position, type NodeProps } from "@xyflow/react";
import type { RfNodeData } from "../adapter";
import type { RailNode as RailDomain } from "../../model";
import Junction from "./Junction";

export default function RailNode({ data, selected }: NodeProps) {
  const d = data as RfNodeData;
  const node = d.node as RailDomain;
  const j = new Set(d.junctions ?? []);
  const shown =
    node.biasValue !== undefined
      ? `${node.railValue} = ${node.biasValue}`
      : String(node.railValue);
  return (
    <div className={`cnode rail ${selected ? "selected" : ""}`}>
      <svg width="44" height="30" viewBox="0 0 44 30" className="sym">
        <line x1="4" y1="10" x2="40" y2="10" stroke="currentColor" strokeWidth="2.5" />
        <line x1="22" y1="10" x2="22" y2="26" stroke="currentColor" strokeWidth="2" />
      </svg>
      <div className="cnode-body">
        <div className="cnode-name">{node.net}</div>
        <div className="cnode-sub">{shown}</div>
      </div>
      <Handle type="source" position={Position.Bottom} id="net" className="handle" />
      <Junction active={j.has("net")} side="bottom" />
    </div>
  );
}
