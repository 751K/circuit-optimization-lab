/**
 * The results dock: a full-width, resizable strip under the canvas.
 *
 * Results used to render inside the 300 px right column, which left a Bode plot
 * about 250 px wide — too narrow to read a decade of frequency, let alone
 * compare two traces. Width is the whole reason this is a bottom dock: it gets
 * the full window, and the drag handle trades canvas height for plot height
 * as needed.
 *
 * One tab per analysis that produced a result, plus Signoff when the circuit
 * declares one and Sweep while a background job is live. The tab strip is the
 * run summary as well — elapsed time and the corner sit in it, so the numbers
 * below are never unattributed.
 */
import { useCallback, useEffect, useRef, useState } from "react";
import { useEditor, useSession } from "../store";
import { ResultView } from "../results";
import { formatDuration } from "../results/format";
import { McView, type McResult } from "../results/McView";
import { PvtView, type PvtResult } from "../results/PvtView";
import { SignoffView, type SignoffPayload } from "../results/SignoffView";
import { DEFAULT_SWEEP_LABEL } from "./runConfig";

const MIN_HEIGHT = 140;
const DEFAULT_HEIGHT = 340;

/** Elapsed wall-clock for a running job, ticking once a second. */
function useElapsed(since: number | null, active: boolean): string | null {
  const [, setTick] = useState(0);
  useEffect(() => {
    if (!active) return;
    const timer = setInterval(() => setTick((value) => value + 1), 1000);
    return () => clearInterval(timer);
  }, [active]);
  if (since === null) return null;
  return formatDuration((Date.now() - since) / 1000);
}

function SweepTab() {
  const sweep = useSession((s) => s.sweep);
  const stopSweep = useSession((s) => s.stopSweep);
  const watchSweep = useSession((s) => s.watchSweep);
  const running = sweep.status === "queued" || sweep.status === "running";
  const elapsed = useElapsed(sweep.startedAt, running);

  // One subscription, owned here, torn down when the job id changes or the dock
  // unmounts. The store deliberately does not hold the socket.
  useEffect(() => {
    if (!sweep.jobId) return;
    return watchSweep();
  }, [sweep.jobId, watchSweep]);

  if (!sweep.kind) {
    return <p className="muted small">No sweep has been started.</p>;
  }

  const progress = sweep.progress;
  const label = sweep.kind === "pvt" ? "PVT corner sweep" : "Mismatch Monte-Carlo";

  return (
    <div className="sweep-tab">
      <div className="sweep-head">
        <strong>{label}</strong>
        <span className={`pill ${sweep.status === "done" ? "ok"
          : sweep.status === "failed" ? "bad" : "muted"}`}>
          {sweep.status}
        </span>
        {elapsed && running && <span className="muted small">{elapsed}</span>}
        {running && (
          <button className="btn tiny" onClick={() => void stopSweep()}>
            Cancel
          </button>
        )}
      </div>

      {running && progress && (
        <div className="progress-wrap">
          <div className="progress-bar">
            <div
              className="progress-fill"
              style={{ width: `${Math.round((progress.frac ?? 0) * 100)}%` }}
            />
          </div>
          <span className="muted small">
            {progress.done} / {progress.total} {progress.unit ?? ""}
          </span>
        </div>
      )}
      {running && !progress && (
        <p className="muted small">Queued…</p>
      )}

      {sweep.error && (
        <div className="run-error">
          <div className="run-error-stage">{sweep.error.stage} error</div>
          <div className="run-error-msg">{sweep.error.message}</div>
        </div>
      )}

      {sweep.status === "cancelled" && (
        <p className="muted small">
          Cancelled. Cancellation is cooperative, so any point already in flight
          finished; partial results are shown when the driver produced them.
        </p>
      )}

      {sweep.result && sweep.kind === "pvt" && (
        <PvtView result={sweep.result as unknown as PvtResult} />
      )}
      {sweep.result && sweep.kind === "mc" && (
        <McView result={sweep.result as unknown as McResult} />
      )}
    </div>
  );
}

export default function ResultsDock() {
  const run = useSession((s) => s.run);
  const sweep = useSession((s) => s.sweep);
  const capsError = useEditor((s) => s.capsError);

  const [height, setHeight] = useState(DEFAULT_HEIGHT);
  const [collapsed, setCollapsed] = useState(false);
  const [active, setActive] = useState<string>("");
  const dragging = useRef<{ y: number; height: number } | null>(null);

  const onPointerDown = useCallback((event: React.PointerEvent) => {
    dragging.current = { y: event.clientY, height };
    (event.target as HTMLElement).setPointerCapture(event.pointerId);
  }, [height]);

  const onPointerMove = useCallback((event: React.PointerEvent) => {
    const start = dragging.current;
    if (!start) return;
    // Dragging up grows the dock, so the delta is inverted.
    const next = start.height + (start.y - event.clientY);
    setHeight(Math.max(MIN_HEIGHT, Math.min(next, window.innerHeight - 220)));
  }, []);

  const onPointerUp = useCallback((event: React.PointerEvent) => {
    dragging.current = null;
    (event.target as HTMLElement).releasePointerCapture(event.pointerId);
  }, []);

  const analyses = run.results ? Object.keys(run.results) : [];
  const hasSignoff = run.signoff !== null && run.signoff.status !== undefined;
  const tabs = [
    ...analyses,
    ...(hasSignoff ? ["signoff"] : []),
    ...(sweep.kind ? ["sweep"] : []),
  ];

  // Follow the work: a fresh run lands on its first analysis, a started sweep on
  // the sweep tab. Only when the current tab has gone away — re-selecting on
  // every render would fight the user's own clicks.
  useEffect(() => {
    const first = tabs[0];
    if (first === undefined) {
      if (active !== "") setActive("");
      return;
    }
    if (!tabs.includes(active)) setActive(first);
  }, [tabs, active]);

  useEffect(() => {
    if (sweep.kind && sweep.status !== "idle") setActive("sweep");
  }, [sweep.kind, sweep.jobId, sweep.status]);

  useEffect(() => {
    if (run.status === "done" && run.results) {
      const first = Object.keys(run.results)[0];
      if (first) setActive(first);
    }
  }, [run.status, run.results]);

  const body = (): React.ReactNode => {
    if (active === "sweep") return <SweepTab />;
    if (active === "signoff" && run.signoff) {
      return <SignoffView signoff={run.signoff as unknown as SignoffPayload} />;
    }
    if (active && run.results && active in run.results) {
      return <ResultView name={active} result={run.results[active]} />;
    }
    if (run.status === "running") return <p className="muted small">Solving…</p>;
    if (run.error) {
      return (
        <div className="run-error">
          <div className="run-error-stage">{run.error.stage} error</div>
          <div className="run-error-msg">{run.error.message}</div>
          {run.error.stage === "parse" && (
            <div className="muted small">
              See the status bar for the validation detail.
            </div>
          )}
          {run.error.stage === "network" && capsError && (
            <div className="muted small">
              The backend is not reachable. Start it with{" "}
              <code>circuit-opt serve</code>.
            </div>
          )}
        </div>
      );
    }
    return (
      <p className="muted small">
        No results yet. Pick analyses in <strong>Simulate</strong> and press Run,
        or start a sweep.
      </p>
    );
  };

  return (
    <section
      className={`dock${collapsed ? " collapsed" : ""}`}
      style={{ height: collapsed ? undefined : height }}
    >
      <div
        className="dock-grip"
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        role="separator"
        aria-orientation="horizontal"
        aria-label="Resize results"
      />
      <div className="dock-tabs">
        {tabs.map((tab) => (
          <button
            key={tab}
            className={`dock-tab${tab === active ? " on" : ""}`}
            onClick={() => { setActive(tab); setCollapsed(false); }}
          >
            {tab}
            {tab === "sweep"
              && (sweep.status === "queued" || sweep.status === "running") && (
              <span className="spinner" aria-label="running" />
            )}
          </button>
        ))}
        {tabs.length === 0 && <span className="dock-tab empty">Results</span>}

        <div className="dock-meta">
          {run.status === "running" && <span className="spinner" aria-label="solving" />}
          {run.elapsed !== null && run.status === "done" && (
            <span className="muted small">solved in {formatDuration(run.elapsed)}</span>
          )}
          {run.corner && <span className="pill muted">{run.corner}</span>}
          {run.defaulted.length > 0 && (
            <span
              className="muted small"
              title={`${run.defaulted.join(", ")}: ${DEFAULT_SWEEP_LABEL}. Injected for `
                + "this run only — the circuit and its export are unchanged."}
            >
              {run.defaulted.length} default sweep
              {run.defaulted.length > 1 ? "s" : ""}
            </span>
          )}
          <button
            className="btn tiny"
            onClick={() => setCollapsed((value) => !value)}
            title={collapsed ? "Expand results" : "Collapse results"}
          >
            {collapsed ? "▴" : "▾"}
          </button>
        </div>
      </div>

      {!collapsed && <div className="dock-body">{body()}</div>}
    </section>
  );
}
