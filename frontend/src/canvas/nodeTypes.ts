/** RF nodeTypes registry — maps a domain NodeKind to its custom component. */
import type { NodeTypes } from "@xyflow/react";
import MosfetNode from "./nodes/MosfetNode";
import RailNode from "./nodes/RailNode";
import ResistorNode from "./nodes/ResistorNode";
import CapacitorNode from "./nodes/CapacitorNode";
import OutputNode from "./nodes/OutputNode";

export const nodeTypes: NodeTypes = {
  mosfet: MosfetNode,
  rail: RailNode,
  resistor: ResistorNode,
  capacitor: CapacitorNode,
  output: OutputNode,
};
