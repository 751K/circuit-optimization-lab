/**
 * F2 editor shell. Three-column flex layout (palette | canvas | inspector) with
 * a top toolbar and a bottom status bar. State lives in the zustand editor store
 * (src/store); the canvas is a controlled projection of that store (src/canvas).
 *
 * On mount we pull backend capabilities once (models/analyses/corners feed the
 * dropdowns). Failure surfaces as a retryable banner in the toolbar; the editor
 * stays fully usable offline (validation just can't run).
 */
import { useEffect } from "react";
import { ReactFlowProvider } from "@xyflow/react";
import { Canvas } from "./canvas";
import { useEditor } from "./store";
import { Inspector, Palette, RunPanel, StatusBar, Toolbar } from "./panels";
import type { GraphNode } from "./model";
import "./App.css";

export default function App() {
  const fetchCapabilities = useEditor((s) => s.fetchCapabilities);
  const caps = useEditor((s) => s.caps);
  const addNode = useEditor((s) => s.addNode);

  useEffect(() => {
    void fetchCapabilities();
  }, [fetchCapabilities]);

  const onDropNode = (kind: GraphNode["kind"], position: { x: number; y: number }): void => {
    const defaultModel = caps ? Object.keys(caps.models)[0] : undefined;
    addNode(kind, [position.x, position.y], { defaultModel });
  };

  return (
    <ReactFlowProvider>
      <div className="app">
        <Toolbar />
        <div className="columns">
          <Palette />
          <main className="center">
            <Canvas onDropNode={onDropNode} />
          </main>
          <div className="right">
            <Inspector />
            <RunPanel />
          </div>
        </div>
        <StatusBar />
      </div>
    </ReactFlowProvider>
  );
}
