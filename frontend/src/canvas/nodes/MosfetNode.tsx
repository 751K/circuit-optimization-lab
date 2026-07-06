/**
 * Mosfet device node. Three handles: D (top), G (left), S (bottom). One symbol
 * for both polarities; a P-type gets a bubble on the gate and a "P" tag. Shows
 * name, W/L, and the short model name. Net labels sit next to each handle.
 */
import { Handle, Position, type NodeProps } from "@xyflow/react";
import type { RfNodeData } from "../adapter";
import type { MosfetNode as MosfetDomain } from "../../model";
import { polarityOf, shortModel } from "../polarity";
import Junction from "./Junction";

export default function MosfetNode({ data, selected }: NodeProps) {
  const d = data as RfNodeData;
  const node = d.node as MosfetDomain;
  const pol = polarityOf(node.model);
  const isP = pol === "pmos";
  const model = shortModel(node.model);
  const j = new Set(d.junctions ?? []);

  return (
    <div className={`cnode mosfet ${selected ? "selected" : ""}`}>
      {/* Drain — top */}
      <Handle type="source" position={Position.Top} id="D" className="handle handle-d" />
      <Junction active={j.has("D")} side="top" />
      {d.portNets["D"] && <span className="netlbl netlbl-top">{d.portNets["D"]}</span>}

      {/* Gate — left */}
      <Handle type="source" position={Position.Left} id="G" className="handle handle-g" />
      <Junction active={j.has("G")} side="left" />
      {d.portNets["G"] && <span className="netlbl netlbl-left">{d.portNets["G"]}</span>}

      {/* Source — bottom */}
      <Handle type="source" position={Position.Bottom} id="S" className="handle handle-s" />
      <Junction active={j.has("S")} side="bottom" />
      {d.portNets["S"] && <span className="netlbl netlbl-bottom">{d.portNets["S"]}</span>}

      <svg width="56" height="56" viewBox="0 0 56 56" className="sym">
        {/* channel bar */}
        <line x1="22" y1="14" x2="22" y2="42" stroke="currentColor" strokeWidth="2" />
        {/* gate bar */}
        <line x1="16" y1="16" x2="16" y2="40" stroke="currentColor" strokeWidth="2" />
        {/* gate lead + optional PMOS bubble */}
        {isP ? (
          <>
            <circle cx="11" cy="28" r="3" fill="none" stroke="currentColor" strokeWidth="1.6" />
            <line x1="8" y1="28" x2="0" y2="28" stroke="currentColor" strokeWidth="2" />
          </>
        ) : (
          <line x1="16" y1="28" x2="0" y2="28" stroke="currentColor" strokeWidth="2" />
        )}
        {/* drain lead (top) */}
        <line x1="22" y1="18" x2="40" y2="18" stroke="currentColor" strokeWidth="2" />
        <line x1="40" y1="18" x2="40" y2="0" stroke="currentColor" strokeWidth="2" />
        {/* source lead (bottom) */}
        <line x1="22" y1="38" x2="40" y2="38" stroke="currentColor" strokeWidth="2" />
        <line x1="40" y1="38" x2="40" y2="56" stroke="currentColor" strokeWidth="2" />
        {/* channel arrow (direction hints polarity) */}
        {isP ? (
          <polygon points="28,38 34,35 34,41" fill="currentColor" />
        ) : (
          <polygon points="34,38 28,35 28,41" fill="currentColor" />
        )}
      </svg>

      <div className="cnode-body">
        <div className="cnode-name">
          {node.name}
          <span className={`pol-tag ${isP ? "p" : "n"}`}>{isP ? "P" : "N"}</span>
        </div>
        <div className="cnode-sub">
          W/L {node.W}/{node.L}
        </div>
        {model && <div className="cnode-model">{model}</div>}
      </div>
    </div>
  );
}
