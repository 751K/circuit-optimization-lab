/**
 * F3 — the Run / results panel.
 *
 * Controls: analysis multi-select (from capabilities.analyses; `ac` on by
 * default), a corner dropdown ("(default)" + the three corner families as
 * <optgroup>s), a Run button, and an Export JSON button. Running calls the
 * async `solve()` (the endpoint is synchronous server-side but fetch is async,
 * so a slow chopper PSS never freezes the UI — the button just shows a spinner
 * and disables).
 *
 * Results live in **component state**, not the store: they are large, transient,
 * and specific to this panel. The panel stays mounted (only its body is hidden
 * when collapsed), so a collapse/expand round-trip preserves the last run.
 *
 * Errors: an ApiError renders its {stage, message} inline — a `parse`/validate
 * error nudges the user to the status-bar validation; any other stage shows the
 * raw solver message (e.g. a DC-non-convergence). A network failure (backend
 * down) shows a retry hint.
 *
 * Analyses the circuit doesn't configure get a client default injected into
 * the request payload only (see runConfig.ts) — Export JSON and the editor
 * state are never touched; a note under the results says which sweeps were
 * defaulted. A selected analysis with no config and no default is intercepted
 * client-side (no request).
 */
import { useState } from "react";
import { useEditor } from "../store";
import { ApiError, solve } from "../api/client";
import { ResultView, downloadJson } from "../results";
import {
  DEFAULT_SWEEP_LABEL,
  missingConfigMessage,
  prepareSolveCircuit,
} from "./runConfig";

/** Flatten the three corner families into grouped options. */
interface CornerGroup {
  family: string;
  corners: string[];
}

export default function RunPanel() {
  const [open, setOpen] = useState(false);
  const caps = useEditor((s) => s.caps);
  const exportJson = useEditor((s) => s.exportJson);

  const analysisKeys = caps ? Object.keys(caps.analyses) : [];
  const cornerGroups: CornerGroup[] = caps
    ? Object.entries(caps.corners).map(([family, corners]) => ({ family, corners }))
    : [];

  // ── controls ──────────────────────────────────────────────────────────
  const [selected, setSelected] = useState<Set<string>>(new Set(["ac"]));
  const [corner, setCorner] = useState<string>(""); // "" = default

  // ── run state ─────────────────────────────────────────────────────────
  const [running, setRunning] = useState(false);
  const [results, setResults] = useState<Record<string, unknown> | null>(null);
  const [elapsed, setElapsed] = useState<number | null>(null);
  const [error, setError] = useState<{ stage: string; message: string } | null>(null);
  const [networkError, setNetworkError] = useState(false);
  /** Analyses whose config was defaulted for the last run (request-only). */
  const [defaulted, setDefaulted] = useState<string[]>([]);

  const toggle = (key: string): void =>
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });

  const run = async (): Promise<void> => {
    if (running) return;
    setError(null);
    setNetworkError(false);
    setOpen(true);
    const sel = analysisKeys.filter((k) => selected.has(k));

    // Inject request-only defaults for unconfigured analyses; intercept the
    // ones that have neither a config nor a default (no request sent).
    const prep = prepareSolveCircuit(exportJson(), sel);
    if (prep.missing.length > 0) {
      setResults(null);
      setElapsed(null);
      setDefaulted([]);
      setError({ stage: "config", message: missingConfigMessage(prep.missing) });
      return;
    }

    setRunning(true);
    try {
      const res = await solve(
        prep.circuit,
        sel.length ? sel : undefined,
        corner || undefined,
      );
      setResults(res.results);
      setElapsed(res.elapsed_s);
      setDefaulted(prep.injected);
    } catch (e) {
      setResults(null);
      setElapsed(null);
      if (e instanceof ApiError) {
        setError({ stage: e.stage, message: e.message });
      } else {
        // fetch() rejects (backend down / CORS / DNS) → not an ApiError.
        setNetworkError(true);
        setError({ stage: "network", message: e instanceof Error ? e.message : String(e) });
      }
    } finally {
      setRunning(false);
    }
  };

  const doExport = (): void => {
    try {
      downloadJson(exportJson());
    } catch {
      /* download blocked (very rare) — nothing actionable to show */
    }
  };

  const selCount = analysisKeys.filter((k) => selected.has(k)).length;

  return (
    <section className={`runpanel ${open ? "open" : ""}`}>
      <button className="runpanel-tab" onClick={() => setOpen((v) => !v)}>
        Run {open ? "▾" : "▸"}
        {running && <span className="spinner" aria-label="running" />}
      </button>
      {open && (
        <div className="runpanel-body">
          {!caps && (
            <p className="muted small">
              Backend offline — start it, then use the toolbar “retry”. Analyses and
              corners load from capabilities.
            </p>
          )}

          {/* ── analysis multi-select ── */}
          {caps && (
            <>
              <h3>Analyses</h3>
              <div className="analysis-list">
                {analysisKeys.map((k) => (
                  <label className="analysis-item" key={k}>
                    <input
                      type="checkbox"
                      checked={selected.has(k)}
                      onChange={() => toggle(k)}
                    />
                    <span>{k}</span>
                  </label>
                ))}
              </div>

              {/* ── corner ── */}
              <label className="field">
                <span>Corner</span>
                <select value={corner} onChange={(e) => setCorner(e.target.value)}>
                  <option value="">(default)</option>
                  {cornerGroups.map((g) => (
                    <optgroup key={g.family} label={g.family}>
                      {g.corners.map((c) => (
                        <option key={`${g.family}:${c}`} value={c}>
                          {c}
                        </option>
                      ))}
                    </optgroup>
                  ))}
                </select>
              </label>

              {/* ── actions ── */}
              <div className="run-actions">
                <button
                  className="btn primary"
                  onClick={() => void run()}
                  disabled={running || selCount === 0}
                  title={selCount === 0 ? "Select at least one analysis" : "Run solve"}
                >
                  {running ? "Running…" : "Run"}
                </button>
                <button className="btn" onClick={doExport} title="Download circuit JSON">
                  Export JSON
                </button>
              </div>
              {elapsed !== null && !error && (
                <p className="muted small">Solved in {elapsed.toFixed(4)} s.</p>
              )}
            </>
          )}

          {/* ── error ── */}
          {error && (
            <div className="run-error">
              <div className="run-error-stage">{error.stage} error</div>
              <div className="run-error-msg">{error.message}</div>
              {error.stage === "parse" && (
                <div className="muted small">
                  See the status bar for validation details.
                </div>
              )}
              {networkError && (
                <button className="btn tiny" onClick={() => void run()} disabled={running}>
                  Retry
                </button>
              )}
            </div>
          )}

          {/* ── results ── */}
          {results && !error && (
            <div className="run-results">
              {defaulted.length > 0 && (
                <p className="muted small">
                  {defaulted.map((n) => `${n}: ${DEFAULT_SWEEP_LABEL}`).join("; ")} — injected
                  for this run only; Export JSON is unchanged.
                </p>
              )}
              {Object.entries(results).map(([name, res]) => (
                <div className="result-block" key={name}>
                  <h3>{name}</h3>
                  <ResultView name={name} result={res} />
                </div>
              ))}
              {Object.keys(results).length === 0 && (
                <p className="muted small">No results — the selected analyses produced nothing.</p>
              )}
            </div>
          )}
        </div>
      )}
    </section>
  );
}
