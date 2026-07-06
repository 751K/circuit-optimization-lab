/**
 * Resistor node: two handles (a left, b right). Shows name and value.
 */
import { Handle, Position, type NodeProps } from "@xyflow/react";
import type { RfNodeData } from "../adapter";
import type { ResistorNode as ResistorDomain } from "../../model";
import { fmtValue } from "../polarity";
import Junction from "./Junction";

export default function ResistorNode({ data, selected }: NodeProps) {
  const d = data as RfNodeData;
  const node = d.node as ResistorDomain;
  const j = new Set(d.junctions ?? []);
  return (
    <div className={`cnode twot ${selected ? "selected" : ""}`}>
      <Handle type="source" position={Position.Left} id="a" className="handle" />
      <Junction active={j.has("a")} side="left" />
      {d.portNets["a"] && <span className="netlbl netlbl-left">{d.portNets["a"]}</span>}
      <svg width="60" height="26" viewBox="0 0 60 26" className="sym">
        <polyline
          points="2,13 12,13 16,5 24,21 32,5 40,21 48,13 58,13"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
        />
      </svg>
      <Handle type="source" position={Position.Right} id="b" className="handle" />
      <Junction active={j.has("b")} side="right" />
      {d.portNets["b"] && <span className="netlbl netlbl-right">{d.portNets["b"]}</span>}
      <div className="cnode-body">
        <div className="cnode-name">{node.name}</div>
        <div className="cnode-sub">{fmtValue(node.R)}Ω</div>
      </div>
    </div>
  );
}
