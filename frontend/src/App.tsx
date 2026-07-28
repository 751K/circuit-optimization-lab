/**
 * Editor shell.
 *
 * Layout: a toolbar, then three columns (palette | canvas | right rail), then a
 * full-width results dock, then the status bar. Results sit along the bottom
 * rather than in the right rail because a frequency sweep needs width — in the
 * rail a Bode plot was 250 px across, which is not enough to read a decade.
 *
 * The right rail carries what *drives* a run: the inspector for the selected
 * element, the analysis picker, and the background sweeps. Controls stay narrow;
 * output gets the window.
 *
 * State: the document lives in the editor store (src/store/store.ts), the run and
 * any background job in the session store (src/store/session.ts). The canvas is a
 * controlled projection of the former.
 *
 * On mount we pull backend capabilities once (models/analyses feed the dropdowns).
 * Failure surfaces as a retryable banner in the toolbar; the editor stays fully
 * usable offline — only validation and solving need the backend.
 */
import { useEffect, useState } from "react";
import { ReactFlowProvider } from "@xyflow/react";
import { Canvas } from "./canvas";
import { useEditor } from "./store";
import {
  Inspector,
  Palette,
  ResultsDock,
  SimulatePanel,
  StatusBar,
  SweepPanel,
  Toolbar,
} from "./panels";
import type { GraphNode } from "./model";
import "./App.css";

type RailTab = "inspect" | "simulate" | "sweep";

const RAIL_TABS: { id: RailTab; label: string }[] = [
  { id: "inspect", label: "Inspect" },
  { id: "simulate", label: "Simulate" },
  { id: "sweep", label: "Sweeps" },
];

export default function App() {
  const fetchCapabilities = useEditor((s) => s.fetchCapabilities);
  const caps = useEditor((s) => s.caps);
  const addNode = useEditor((s) => s.addNode);
  const selectedNodes = useEditor((s) => s.selection.nodes);
  const [rail, setRail] = useState<RailTab>("simulate");

  useEffect(() => {
    void fetchCapabilities();
  }, [fetchCapabilities]);

  // Selecting an element on the canvas is a request to look at it.
  useEffect(() => {
    if (selectedNodes.length > 0) setRail("inspect");
  }, [selectedNodes]);

  const onDropNode = (
    kind: GraphNode["kind"],
    position: { x: number; y: number },
  ): void => {
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
            <div className="rail-tabs">
              {RAIL_TABS.map((tab) => (
                <button
                  key={tab.id}
                  className={`rail-tab${rail === tab.id ? " on" : ""}`}
                  onClick={() => setRail(tab.id)}
                >
                  {tab.label}
                </button>
              ))}
            </div>
            <div className="rail-body">
              {rail === "inspect" && <Inspector />}
              {rail === "simulate" && <SimulatePanel />}
              {rail === "sweep" && <SweepPanel />}
            </div>
          </div>
        </div>
        <ResultsDock />
        <StatusBar />
      </div>
    </ReactFlowProvider>
  );
}
