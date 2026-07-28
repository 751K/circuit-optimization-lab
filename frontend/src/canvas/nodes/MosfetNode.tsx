/**
 * Mosfet device node. Three handles: drain, gate (always left), and source.
 *
 * The symbol is drawn *source-down for an N-type and source-up for a P-type* —
 * the schematic convention, and the one that matters now that the auto-layout
 * ranks devices by supply altitude (see model/layout.ts). A PMOS load sits below
 * VDD with its source facing it, so flipping the symbol turns what would be a
 * wire looping around the body into a short vertical drop. The whole SVG is
 * mirrored, which also flips the channel arrow to its correct direction, and the
 * D/S handles swap ends with it. Each terminal carries a small D/G/S letter so
 * the orientation is never something you have to infer.
 */
import { Fragment } from "react";
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
  // Source faces the supply it returns to: down for an N-type, up for a P-type.
  const sideOf = (port: "D" | "S"): "top" | "bottom" =>
    (port === "S") === isP ? "top" : "bottom";
  const posOf = (port: "D" | "S"): Position =>
    sideOf(port) === "top" ? Position.Top : Position.Bottom;
  // Mirrored: gate on the right. The two halves of a differential pair face
  // each other that way, which is how you tell them apart at a glance.
  const mirror = node.mirrored === true;
  const gSide = mirror ? "right" : "left";

  return (
    <div className={`cnode mosfet ${mirror ? "mirror" : ""} ${selected ? "selected" : ""}`}>
      {(["D", "S"] as const).map((port) => {
        const side = sideOf(port);
        return (
          // A Fragment, not an element: `.cnode` is a flex row, and a wrapper
          // would become a flex item between the symbol and the label block.
          <Fragment key={port}>
            <Handle
              type="source"
              position={posOf(port)}
              id={port}
              className={`handle handle-${port.toLowerCase()}`}
            />
            <Junction active={j.has(port)} side={side} />
            <span className={`portlbl portlbl-${side}`}>{port}</span>
            {d.portNets[port] && (
              <span className={`netlbl netlbl-${side}`}>{d.portNets[port]}</span>
            )}
          </Fragment>
        );
      })}

      {/* Gate — left, or right when the device is mirrored */}
      <Handle
        type="source"
        position={mirror ? Position.Right : Position.Left}
        id="G"
        className="handle handle-g"
      />
      <Junction active={j.has("G")} side={gSide} />
      <span className={`portlbl portlbl-${gSide}`}>G</span>
      {d.portNets["G"] && <span className={`netlbl netlbl-${gSide}`}>{d.portNets["G"]}</span>}

      <svg width="56" height="56" viewBox="0 0 56 56" className="sym">
        {/* Two independent flips. The P-type is mirrored vertically so its
            source lead sits on top (and the channel arrow turns with it); a
            mirrored device is flipped horizontally so the gate lead leaves on
            the right. Composing them is just both transforms. */}
        <g transform={[
          isP ? "translate(0,56) scale(1,-1)" : "",
          mirror ? "translate(56,0) scale(-1,1)" : "",
        ].filter(Boolean).join(" ") || undefined}>
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
          {/* drain lead */}
          <line x1="22" y1="18" x2="40" y2="18" stroke="currentColor" strokeWidth="2" />
          <line x1="40" y1="18" x2="40" y2="0" stroke="currentColor" strokeWidth="2" />
          {/* source lead */}
          <line x1="22" y1="38" x2="40" y2="38" stroke="currentColor" strokeWidth="2" />
          <line x1="40" y1="38" x2="40" y2="56" stroke="currentColor" strokeWidth="2" />
          {/* channel arrow (direction hints polarity) */}
          {isP ? (
            <polygon points="28,38 34,35 34,41" fill="currentColor" />
          ) : (
            <polygon points="34,38 28,35 28,41" fill="currentColor" />
          )}
        </g>
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
