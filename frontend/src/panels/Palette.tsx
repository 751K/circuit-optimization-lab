/**
 * Left column: the palette. Five node kinds (click to add at a default spot, or
 * drag onto the canvas), a "load example" dropdown over the built-in fixtures, a
 * "new empty circuit" button, and an "import JSON" paste box.
 *
 * New mosfets default to the first capabilities.models key so the model dropdown
 * starts on something real.
 */
import { useState } from "react";
import { useEditor } from "../store";
import type { CircuitJson, GraphNode } from "../model";
import { FIXTURES } from "./fixtures";

const KINDS: { kind: GraphNode["kind"]; label: string; hint: string }[] = [
  { kind: "mosfet", label: "MOSFET", hint: "3-terminal device (D/G/S)" },
  { kind: "rail", label: "Rail", hint: "fixed-potential net" },
  { kind: "resistor", label: "Resistor", hint: "2-terminal R" },
  { kind: "capacitor", label: "Capacitor", hint: "2-terminal C" },
  { kind: "output", label: "Output", hint: "output marker" },
];

export default function Palette() {
  const addNode = useEditor((s) => s.addNode);
  const loadCircuit = useEditor((s) => s.loadCircuit);
  const newCircuit = useEditor((s) => s.newCircuit);
  const caps = useEditor((s) => s.caps);
  const [pasteOpen, setPasteOpen] = useState(false);
  const [pasteText, setPasteText] = useState("");
  const [pasteError, setPasteError] = useState<string | null>(null);

  const defaultModel = caps ? Object.keys(caps.models)[0] : undefined;

  const add = (kind: GraphNode["kind"]): void => {
    // Drop new nodes in a loose cascade near the top-left of the flow space.
    const jitter = Math.round(Math.random() * 80);
    addNode(kind, [120 + jitter, 120 + jitter], { defaultModel });
  };

  const onDragStart = (e: React.DragEvent, kind: GraphNode["kind"]): void => {
    e.dataTransfer.setData("application/circuitopt-node", kind);
    e.dataTransfer.effectAllowed = "copy";
  };

  const onLoadFixture = (key: string): void => {
    if (!key) return;
    const f = FIXTURES.find((x) => x.key === key);
    if (f) loadCircuit(structuredClone(f.json));
  };

  const doImport = (): void => {
    setPasteError(null);
    try {
      const json = JSON.parse(pasteText) as CircuitJson;
      if (typeof json !== "object" || json === null || !("solved" in json) || !("rails" in json)) {
        throw new Error("not a circuit JSON (missing solved/rails)");
      }
      loadCircuit(json);
      setPasteText("");
      setPasteOpen(false);
    } catch (e) {
      setPasteError(e instanceof Error ? e.message : String(e));
    }
  };

  return (
    <aside className="panel palette">
      <h2>Palette</h2>
      <div className="palette-grid">
        {KINDS.map((k) => (
          <button
            key={k.kind}
            className={`pal-item pal-${k.kind}`}
            title={`${k.hint} — click to add, or drag onto canvas`}
            draggable
            onDragStart={(e) => onDragStart(e, k.kind)}
            onClick={() => add(k.kind)}
          >
            {k.label}
          </button>
        ))}
      </div>

      <h3>Load</h3>
      <label className="field">
        <span>From example</span>
        <select defaultValue="" onChange={(e) => onLoadFixture(e.target.value)}>
          <option value="">— choose —</option>
          {FIXTURES.map((f) => (
            <option key={f.key} value={f.key}>
              {f.label}
            </option>
          ))}
        </select>
      </label>

      <button className="btn" onClick={() => newCircuit("untitled")}>
        New empty circuit
      </button>

      <button className="btn" onClick={() => setPasteOpen((v) => !v)}>
        {pasteOpen ? "Cancel import" : "Import JSON…"}
      </button>
      {pasteOpen && (
        <div className="import-box">
          <textarea
            value={pasteText}
            onChange={(e) => setPasteText(e.target.value)}
            placeholder='{"name": "...", "solved": [...], "rails": {...}, "devices": [...]}'
            rows={8}
            spellCheck={false}
          />
          {pasteError && <div className="err">{pasteError}</div>}
          <button className="btn primary" onClick={doImport} disabled={!pasteText.trim()}>
            Load pasted JSON
          </button>
        </div>
      )}
    </aside>
  );
}
