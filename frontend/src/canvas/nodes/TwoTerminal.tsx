/**
 * The shared body of a two-terminal element (resistor, capacitor).
 *
 * Only the symbol differs between them; the handles, junction dots, net labels
 * and the a/b orientation are identical, and the orientation in particular is
 * fiddly enough that two copies would drift. `symbol` is called with whether the
 * element is being drawn vertically so each kind can supply its own pair.
 */
import { Handle } from "@xyflow/react";
import { Fragment, type ReactNode } from "react";
import type { RfNodeData } from "../adapter";
import { RF_SIDE, type PortSide } from "../sides";
import Junction from "./Junction";

export default function TwoTerminal({
  data,
  selected,
  name,
  value,
  symbol,
}: {
  data: RfNodeData;
  selected: boolean;
  name: string;
  value: string;
  symbol: (vertical: boolean) => ReactNode;
}) {
  const j = new Set(data.junctions ?? []);
  const sideA: PortSide = data.sides?.["a"] ?? "left";
  const sideB: PortSide = data.sides?.["b"] ?? "right";
  const vertical = sideA === "top" || sideA === "bottom";
  return (
    <div className={`cnode twot ${vertical ? "vert" : ""} ${selected ? "selected" : ""}`}>
      {(["a", "b"] as const).map((port) => {
        const side = port === "a" ? sideA : sideB;
        return (
          <Fragment key={port}>
            <Handle type="source" position={RF_SIDE[side]} id={port} className="handle" />
            <Junction active={j.has(port)} side={side} />
            {data.portNets[port] && (
              <span className={`netlbl netlbl-${side}`}>{data.portNets[port]}</span>
            )}
          </Fragment>
        );
      })}
      {symbol(vertical)}
      <div className="cnode-body">
        <div className="cnode-name">{name}</div>
        <div className="cnode-sub">{value}</div>
      </div>
    </div>
  );
}
