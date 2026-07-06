/**
 * Output marker: single handle (left). Shows the +/- order for differential
 * outputs ([VOP, VON] -> order 0 is "+", order 1 is "-"), plus the resolved net.
 */
import { Handle, Position, type NodeProps } from "@xyflow/react";
import type { RfNodeData } from "../adapter";
import type { OutputNode as OutputDomain } from "../../model";
import Junction from "./Junction";

export default function OutputNode({ data, selected }: NodeProps) {
  const d = data as RfNodeData;
  const node = d.node as OutputDomain;
  const sign = node.order === 0 ? "+" : node.order === 1 ? "−" : `#${node.order}`;
  const net = d.portNets["out"];
  const j = new Set(d.junctions ?? []);
  return (
    <div className={`cnode output ${selected ? "selected" : ""}`}>
      <Handle type="source" position={Position.Left} id="out" className="handle" />
      <Junction active={j.has("out")} side="left" />
      <svg width="34" height="34" viewBox="0 0 34 34" className="sym">
        <circle cx="17" cy="17" r="12" fill="none" stroke="currentColor" strokeWidth="2" />
        <text x="17" y="22" textAnchor="middle" fontSize="15" fill="currentColor">
          {sign}
        </text>
      </svg>
      <div className="cnode-body">
        <div className="cnode-name">out {sign}</div>
        {net && <div className="cnode-sub">{net}</div>}
      </div>
    </div>
  );
}
