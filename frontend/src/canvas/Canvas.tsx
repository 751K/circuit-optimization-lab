/**
 * The React Flow canvas. It is a *controlled projection* of the store's domain
 * graph: on every render it derives RF nodes/edges via the adapter, and pipes RF
 * callbacks back into store actions. RF never owns the source of truth.
 *
 * Interactions:
 *  - drag a node          -> moveNode(id, [x,y]) on drag stop
 *  - drag handle -> handle -> connect(source, target) with the port (handle) ids
 *  - select nodes/edges   -> setSelection
 *  - Delete/Backspace     -> deleteSelection
 *  - drop from palette     -> onDropNode(kind, position)  (wired by parent)
 */
import { useCallback, useMemo, useState } from "react";
import {
  Background,
  ConnectionMode,
  Controls,
  ReactFlow,
  applyNodeChanges,
  useReactFlow,
  type Connection,
  type Edge,
  type EdgeChange,
  type NodeChange,
  type OnSelectionChangeParams,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { useEditor } from "../store";
import { domainToRf, netClass, type RfEdge, type RfNode } from "./adapter";
import { nodeTypes } from "./nodeTypes";
import type { GraphNode } from "../model";

export default function Canvas({
  onDropNode,
}: {
  onDropNode?: (kind: GraphNode["kind"], position: { x: number; y: number }) => void;
}) {
  const graph = useEditor((s) => s.graph);
  const netError = useEditor((s) => s.netError);
  const moveNode = useEditor((s) => s.moveNode);
  const connect = useEditor((s) => s.connect);
  const setSelection = useEditor((s) => s.setSelection);
  const deleteNodes = useEditor((s) => s.deleteNodes);
  const deleteEdges = useEditor((s) => s.deleteEdges);
  const rf = useReactFlow();

  const conflictSet = useMemo(
    () => new Set(netError?.edgeIds ?? []),
    [netError],
  );

  const { nodes, edges } = useMemo(
    () => domainToRf(graph, conflictSet),
    [graph, conflictSet],
  );

  // Net-level hover highlight. We keep only the hovered net *name* in local
  // state and drive the visual via a single CSS class on the wrapper
  // (`highlight-<safe>`), so hovering never rebuilds the (memoized) edge array —
  // no per-hover re-projection, no jank on large nets.
  const [hoveredNet, setHoveredNet] = useState<string | undefined>(undefined);
  const onEdgeEnter = useCallback(
    (_e: React.MouseEvent, edge: Edge) =>
      setHoveredNet((edge as RfEdge).data?.net),
    [],
  );
  const onEdgeLeave = useCallback(() => setHoveredNet(undefined), []);
  // A single scoped CSS rule targeting the hovered net's shared class. Because
  // every edge on a net already carries `net-<safe>` (from the adapter),
  // toggling one rule highlights *all* of them at once with zero edge rebuilds.
  const hoverStyle = useMemo(() => {
    const c = netClass(hoveredNet);
    if (!c) return null;
    return `.canvas-wrap .react-flow__edge.${c} .react-flow__edge-path{stroke:var(--accent)!important;stroke-width:2.6px!important;stroke-opacity:1!important;stroke-dasharray:none!important;}
.canvas-wrap .react-flow__edge.${c} .react-flow__edge-text{fill:var(--accent)!important;font-weight:700;}
.canvas-wrap .react-flow__edge.${c} .react-flow__edge-textbg{fill:var(--panel)!important;}`;
  }, [hoveredNet]);

  const onNodesChange = useCallback(
    (changes: NodeChange[]) => {
      // Persist a drag only when it finishes (position change with dragging:false).
      for (const c of changes) {
        if (c.type === "position" && c.dragging === false && c.position) {
          moveNode(c.id, [c.position.x, c.position.y]);
        }
        if (c.type === "remove") {
          deleteNodes([c.id]);
        }
      }
      // Live drag preview is handled by RF internally on the derived nodes via
      // applyNodeChanges; we don't store transient positions.
      void applyNodeChanges(changes, nodes as RfNode[]);
    },
    [moveNode, deleteNodes, nodes],
  );

  const onEdgesChange = useCallback(
    (changes: EdgeChange[]) => {
      for (const c of changes) {
        if (c.type === "remove") deleteEdges([c.id]);
      }
    },
    [deleteEdges],
  );

  const onConnect = useCallback(
    (c: Connection) => {
      if (!c.source || !c.target || !c.sourceHandle || !c.targetHandle) return;
      connect(
        { node: c.source, port: c.sourceHandle },
        { node: c.target, port: c.targetHandle },
      );
    },
    [connect],
  );

  const onSelectionChange = useCallback(
    (params: OnSelectionChangeParams) => {
      setSelection({
        nodes: params.nodes.map((n) => n.id),
        edges: params.edges.map((e) => e.id),
      });
    },
    [setSelection],
  );

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      const kind = e.dataTransfer.getData("application/circuitopt-node") as
        | GraphNode["kind"]
        | "";
      if (!kind || !onDropNode) return;
      const position = rf.screenToFlowPosition({ x: e.clientX, y: e.clientY });
      onDropNode(kind, position);
    },
    [onDropNode, rf],
  );

  const onDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = "copy";
  }, []);

  return (
    <div className="canvas-wrap" onDrop={onDrop} onDragOver={onDragOver}>
      {hoverStyle && <style>{hoverStyle}</style>}
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onSelectionChange={onSelectionChange}
        onEdgeMouseEnter={onEdgeEnter}
        onEdgeMouseLeave={onEdgeLeave}
        connectionMode={ConnectionMode.Loose}
        deleteKeyCode={["Delete", "Backspace"]}
        fitView
        proOptions={{ hideAttribution: true }}
      >
        <Background gap={16} />
        <Controls />
      </ReactFlow>
    </div>
  );
}
