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
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
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

/**
 * A ranked schematic is far larger than the kind columns it replaced — a
 * 35-device amplifier spans several thousand units — and React Flow's default
 * `minZoom` of 0.5 silently clamps `fitView`, so the drawing simply arrives
 * cropped rather than framed. The floor has to clear the largest circuit in the
 * corpus, not a typical one.
 */
const MIN_ZOOM = 0.04;
const FIT = { padding: 0.12, minZoom: MIN_ZOOM } as const;

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

  // React Flow is fed from `live`, not straight from the projection.
  //
  // The store only learns a new position when the drag *ends*, so a fully
  // controlled node list has nothing to show in between: the node stays put
  // under the cursor and jumps to its new home on mouse-up. RF's own
  // `applyNodeChanges` output was being computed and thrown away, which is the
  // same thing. Holding the changes here gives RF the in-flight positions to
  // render, and the effect below resyncs from the store — including right after
  // the drag commits, where the two agree and nothing moves.
  const [live, setLive] = useState<RfNode[]>(nodes);
  useEffect(() => setLive(nodes), [nodes]);

  // Re-fit whenever the document is replaced wholesale. `fitView` on the
  // component only runs at mount, so loading a second circuit used to drop it
  // wherever the previous viewport happened to be — which the schematic layout
  // made worse, since a ranked drawing is far taller than the old kind columns.
  // Keyed on viewEpoch and not on `graph`, so an ordinary edit never yanks the
  // view out from under the user mid-drag.
  const viewEpoch = useEditor((s) => s.viewEpoch);
  useEffect(() => {
    if (graph.nodes.length === 0) return;
    // One frame later: RF measures the new nodes before it can frame them.
    const id = requestAnimationFrame(() => rf.fitView(FIT));
    return () => cancelAnimationFrame(id);
    // `graph` and `rf` are read but deliberately not dependencies: this must
    // fire on viewEpoch alone.
  }, [viewEpoch]); // eslint-disable-line react-hooks/exhaustive-deps

  // Re-frame when the canvas itself changes size. Collapsing the results dock
  // roughly doubles the pane height, and React Flow keeps the transform it had —
  // so the drawing stayed at its old scale with half the canvas left empty, and
  // even its own Fit View button measured against the stale size. Watching the
  // element is the only signal: the dock is resized by layout, not by a window
  // resize, so nothing else fires.
  const wrapRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    let last: { w: number; h: number } | null = null;
    let pending = 0;
    const obs = new ResizeObserver(([entry]) => {
      const { width: w, height: h } = entry!.contentRect;
      if (w === 0 || h === 0) return;
      // The first observation is the initial layout, which `fitView` on the
      // component already handled; and a change of a few pixels is a scrollbar
      // or a rounding wobble, not a resize the user would want re-framed over
      // whatever they had zoomed into.
      const material = last !== null
        && (Math.abs(w - last.w) > last.w * 0.05 || Math.abs(h - last.h) > last.h * 0.05);
      last = { w, h };
      if (!material) return;
      cancelAnimationFrame(pending);
      pending = requestAnimationFrame(() => rf.fitView(FIT));
    });
    obs.observe(el);
    return () => {
      cancelAnimationFrame(pending);
      obs.disconnect();
    };
  }, [rf]);

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
      // Render every change immediately — this is what makes a drag visible.
      setLive((current) => applyNodeChanges(changes, current) as RfNode[]);
      // …but only commit to the document when the drag finishes, so the undo
      // stack gets one entry per drag rather than one per mouse-move.
      for (const c of changes) {
        if (c.type === "position" && c.dragging === false && c.position) {
          moveNode(c.id, [c.position.x, c.position.y]);
        }
        if (c.type === "remove") {
          deleteNodes([c.id]);
        }
      }
    },
    [moveNode, deleteNodes],
  );

  // Shift+H flips the whole selection. The Inspector button only reaches one
  // device at a time, and mirroring is usually something you want for a pair.
  const mirrorNodes = useEditor((s) => s.mirrorNodes);
  const selectedNodes = useEditor((s) => s.selection.nodes);
  const onKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key !== "H" && e.key !== "h") return;
      if (!e.shiftKey || e.metaKey || e.ctrlKey || e.altKey) return;
      if (selectedNodes.length === 0) return;
      e.preventDefault();
      mirrorNodes(selectedNodes);
    },
    [mirrorNodes, selectedNodes],
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
    <div className="canvas-wrap" ref={wrapRef} onDrop={onDrop} onDragOver={onDragOver} onKeyDown={onKeyDown}>
      {hoverStyle && <style>{hoverStyle}</style>}
      <ReactFlow
        nodes={live}
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
        minZoom={MIN_ZOOM}
        fitView
        fitViewOptions={FIT}
        proOptions={{ hideAttribution: true }}
      >
        <Background gap={16} />
        <Controls />
      </ReactFlow>
    </div>
  );
}
