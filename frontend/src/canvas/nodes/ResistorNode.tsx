/**
 * Resistor node. Drawn along the axis its two ends are separated by (see
 * sides.ts): a degeneration or ladder resistor in the supply stack turns
 * vertical, a series or feedback resistor between stages stays horizontal.
 */
import { type NodeProps } from "@xyflow/react";
import type { RfNodeData } from "../adapter";
import type { ResistorNode as ResistorDomain } from "../../model";
import { fmtValue } from "../polarity";
import TwoTerminal from "./TwoTerminal";

export default function ResistorNode({ data, selected }: NodeProps) {
  const d = data as RfNodeData;
  const node = d.node as ResistorDomain;
  return (
    <TwoTerminal
      data={d}
      selected={selected === true}
      name={node.name}
      value={`${fmtValue(node.R)}Ω`}
      symbol={(vertical) => (vertical ? (
        <svg width="26" height="60" viewBox="0 0 26 60" className="sym">
          <polyline
            points="13,2 13,12 5,16 21,24 5,32 21,40 13,48 13,58"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
          />
        </svg>
      ) : (
        <svg width="60" height="26" viewBox="0 0 60 26" className="sym">
          <polyline
            points="2,13 12,13 16,5 24,21 32,5 40,21 48,13 58,13"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
          />
        </svg>
      ))}
    />
  );
}
