/**
 * Capacitor node: two handles (a left, b right). Shows name (or "load" for a
 * synthetic load_cap) and value.
 */
import { Handle, Position, type NodeProps } from "@xyflow/react";
import type { RfNodeData } from "../adapter";
import type { CapacitorNode as CapacitorDomain } from "../../model";
import { fmtValue } from "../polarity";
import Junction from "./Junction";

export default function CapacitorNode({ data, selected }: NodeProps) {
  const d = data as RfNodeData;
  const node = d.node as CapacitorDomain;
  const label = node.origin === "load_caps" ? "load" : node.name;
  const j = new Set(d.junctions ?? []);
  return (
    <div className={`cnode twot ${selected ? "selected" : ""}`}>
      <Handle type="source" position={Position.Left} id="a" className="handle" />
      <Junction active={j.has("a")} side="left" />
      {d.portNets["a"] && <span className="netlbl netlbl-left">{d.portNets["a"]}</span>}
      <svg width="44" height="30" viewBox="0 0 44 30" className="sym">
        <line x1="2" y1="15" x2="19" y2="15" stroke="currentColor" strokeWidth="2" />
        <line x1="19" y1="3" x2="19" y2="27" stroke="currentColor" strokeWidth="2.5" />
        <line x1="25" y1="3" x2="25" y2="27" stroke="currentColor" strokeWidth="2.5" />
        <line x1="25" y1="15" x2="42" y2="15" stroke="currentColor" strokeWidth="2" />
      </svg>
      <Handle type="source" position={Position.Right} id="b" className="handle" />
      <Junction active={j.has("b")} side="right" />
      {d.portNets["b"] && <span className="netlbl netlbl-right">{d.portNets["b"]}</span>}
      <div className="cnode-body">
        <div className="cnode-name">{label}</div>
        <div className="cnode-sub">{fmtValue(node.C)}F</div>
      </div>
    </div>
  );
}
