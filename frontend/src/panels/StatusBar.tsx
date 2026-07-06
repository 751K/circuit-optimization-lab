/**
 * Bottom bar: backend connection state + live validation.
 *
 * Validation flow: on any graph/rest change, debounce 800ms, then export
 * graph->JSON and POST /validate, rendering `valid ✓` or the raw error list. A
 * local net-conflict (double-rail short from resolveNets) takes priority and is
 * shown in red without hitting the backend.
 */
import { useEffect, useRef, useState } from "react";
import { useEditor } from "../store";
import { validate, type ValidateResponse } from "../api/client";

type ValState =
  | { kind: "idle" }
  | { kind: "checking" }
  | { kind: "ok" }
  | { kind: "errors"; errors: string[] }
  | { kind: "unreachable"; message: string };

const DEBOUNCE_MS = 800;

export default function StatusBar() {
  const graph = useEditor((s) => s.graph);
  const rest = useEditor((s) => s.rest);
  const netError = useEditor((s) => s.netError);
  const caps = useEditor((s) => s.caps);
  const capsError = useEditor((s) => s.capsError);
  const exportJson = useEditor((s) => s.exportJson);
  const [state, setState] = useState<ValState>({ kind: "idle" });
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const seq = useRef(0);

  useEffect(() => {
    // Local net conflict: skip the backend, show it immediately.
    if (netError) {
      setState({ kind: "errors", errors: [netError.message] });
      return;
    }
    if (graph.nodes.length === 0) {
      setState({ kind: "idle" });
      return;
    }
    setState({ kind: "checking" });
    if (timer.current) clearTimeout(timer.current);
    const mySeq = ++seq.current;
    timer.current = setTimeout(() => {
      let circuit;
      try {
        circuit = exportJson();
      } catch (e) {
        if (mySeq === seq.current) {
          setState({ kind: "errors", errors: [e instanceof Error ? e.message : String(e)] });
        }
        return;
      }
      validate(circuit)
        .then((r: ValidateResponse) => {
          if (mySeq !== seq.current) return;
          setState(r.valid ? { kind: "ok" } : { kind: "errors", errors: r.errors ?? ["invalid"] });
        })
        .catch((e: unknown) => {
          if (mySeq !== seq.current) return;
          setState({ kind: "unreachable", message: e instanceof Error ? e.message : String(e) });
        });
    }, DEBOUNCE_MS);
    return () => {
      if (timer.current) clearTimeout(timer.current);
    };
  }, [graph, rest, netError, exportJson]);

  const connected = caps !== null;

  return (
    <footer className="statusbar">
      <span className={`conn ${connected ? "up" : "down"}`}>
        {connected ? "● backend connected" : capsError ? "● backend offline" : "● connecting…"}
      </span>
      <span className="sep" />
      <ValView state={state} />
    </footer>
  );
}

function ValView({ state }: { state: ValState }) {
  switch (state.kind) {
    case "idle":
      return <span className="muted">no circuit</span>;
    case "checking":
      return <span className="muted">validating…</span>;
    case "ok":
      return <span className="ok">valid ✓</span>;
    case "unreachable":
      return <span className="warn">cannot validate (backend): {state.message}</span>;
    case "errors":
      return (
        <span className="err">
          {state.errors.length} error{state.errors.length === 1 ? "" : "s"}:{" "}
          {state.errors.join("; ")}
        </span>
      );
  }
}
