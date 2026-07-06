/**
 * Top toolbar: title, undo/redo, and — when the capabilities probe failed — a
 * "backend not connected" banner with a retry button.
 */
import { useEditor } from "../store";

export default function Toolbar() {
  const undo = useEditor((s) => s.undo);
  const redo = useEditor((s) => s.redo);
  const canUndo = useEditor((s) => s.past.length > 0);
  const canRedo = useEditor((s) => s.future.length > 0);
  const capsError = useEditor((s) => s.capsError);
  const capsLoading = useEditor((s) => s.capsLoading);
  const caps = useEditor((s) => s.caps);
  const fetchCapabilities = useEditor((s) => s.fetchCapabilities);

  return (
    <header className="toolbar">
      <div className="brand">circuitopt builder</div>
      <div className="toolbar-actions">
        <button className="btn" onClick={() => undo()} disabled={!canUndo} title="Undo (last edit)">
          ↶ Undo
        </button>
        <button className="btn" onClick={() => redo()} disabled={!canRedo} title="Redo">
          ↷ Redo
        </button>
      </div>
      {!caps && (capsError || !capsLoading) && (
        <div className="banner">
          {capsError ? (
            <>
              backend not connected ({capsError}) — start it with{" "}
              <code>circuit-opt serve --port 8341</code>{" "}
            </>
          ) : (
            "backend not connected "
          )}
          <button className="btn tiny" onClick={() => fetchCapabilities()} disabled={capsLoading}>
            {capsLoading ? "retrying…" : "retry"}
          </button>
        </div>
      )}
    </header>
  );
}
