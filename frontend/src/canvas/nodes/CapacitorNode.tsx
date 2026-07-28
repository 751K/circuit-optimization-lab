/**
 * Capacitor node. Shows name (or "load" for a synthetic load_cap) and value.
 *
 * Drawn along the axis its two plates are actually separated by (see sides.ts):
 * a coupling capacitor between two stages stays horizontal, while a load or
 * compensation capacitor hanging from a signal node down to ground turns
 * vertical — plates across the current, not along it.
 */
import { type NodeProps } from "@xyflow/react";
import type { RfNodeData } from "../adapter";
import type { CapacitorNode as CapacitorDomain } from "../../model";
import { fmtValue } from "../polarity";
import TwoTerminal from "./TwoTerminal";

export default function CapacitorNode({ data, selected }: NodeProps) {
  const d = data as RfNodeData;
  const node = d.node as CapacitorDomain;
  return (
    <TwoTerminal
      data={d}
      selected={selected === true}
      name={node.origin === "load_caps" ? "load" : node.name}
      value={`${fmtValue(node.C)}F`}
      symbol={(vertical) => (vertical ? (
        <svg width="30" height="44" viewBox="0 0 30 44" className="sym">
          <line x1="15" y1="2" x2="15" y2="19" stroke="currentColor" strokeWidth="2" />
          <line x1="3" y1="19" x2="27" y2="19" stroke="currentColor" strokeWidth="2.5" />
          <line x1="3" y1="25" x2="27" y2="25" stroke="currentColor" strokeWidth="2.5" />
          <line x1="15" y1="25" x2="15" y2="42" stroke="currentColor" strokeWidth="2" />
        </svg>
      ) : (
        <svg width="44" height="30" viewBox="0 0 44 30" className="sym">
          <line x1="2" y1="15" x2="19" y2="15" stroke="currentColor" strokeWidth="2" />
          <line x1="19" y1="3" x2="19" y2="27" stroke="currentColor" strokeWidth="2.5" />
          <line x1="25" y1="3" x2="25" y2="27" stroke="currentColor" strokeWidth="2.5" />
          <line x1="25" y1="15" x2="42" y2="15" stroke="currentColor" strokeWidth="2" />
        </svg>
      ))}
    />
  );
}
