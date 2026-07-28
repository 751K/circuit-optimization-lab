/**
 * Background sweeps: PVT corners and mismatch Monte-Carlo.
 *
 * Both are long enough to need a job rather than a request, so this panel only
 * *starts* them — progress and results live in the results dock, which has the
 * width to show a grid.
 *
 * The temperature and supply axes are silicon-only. Rather than let a user fill
 * them in and receive a 422, they are disabled with the reason stated, driven by
 * the `silicon` flag the same `/validate` call already returns.
 */
import { useState } from "react";
import { useEditor, useSession } from "../store";

/** Parse "0, 27, 85" into numbers, ignoring blanks and junk. */
export function parseAxis(text: string): number[] {
  return text
    .split(/[,\s]+/)
    .map((part) => part.trim())
    .filter((part) => part.length > 0)
    .map(Number)
    .filter((value) => Number.isFinite(value));
}

export default function SweepPanel() {
  const exportJson = useEditor((s) => s.exportJson);
  const silicon = useEditor((s) => s.circuitSilicon);
  const circuitCorners = useEditor((s) => s.circuitCorners);
  const nodeCount = useEditor((s) => s.graph.nodes.length);

  const startPvt = useSession((s) => s.startPvt);
  const startMc = useSession((s) => s.startMc);
  const sweep = useSession((s) => s.sweep);

  const [tempText, setTempText] = useState("0, 27, 85");
  const [vddText, setVddText] = useState("");
  const [useTemps, setUseTemps] = useState(false);
  const [useVdd, setUseVdd] = useState(false);
  const [samples, setSamples] = useState(64);
  const [seed, setSeed] = useState(0);
  const [workers, setWorkers] = useState(4);

  const busy = sweep.status === "queued" || sweep.status === "running";
  const empty = nodeCount === 0;

  const temps = parseAxis(tempText);
  const vdds = parseAxis(vddText);
  const gridPoints =
    (circuitCorners?.length ?? 0)
    * (useTemps && temps.length ? temps.length : 1)
    * (useVdd && vdds.length ? vdds.length : 1);

  const runPvt = (): void => {
    void startPvt(exportJson(), {
      workers,
      ...(useTemps && temps.length ? { temps } : {}),
      ...(useVdd && vdds.length ? { vdd_scale: vdds } : {}),
    });
  };

  const runMc = (): void => {
    void startMc(exportJson(), { n: samples, seed, workers });
  };

  return (
    <section className="panel sweep-panel">
      <h2>Sweeps</h2>

      <h3>PVT corners</h3>
      <p className="muted small">
        Sweeps {circuitCorners?.length ?? "?"} corner
        {circuitCorners?.length === 1 ? "" : "s"} for this circuit
        {gridPoints > 0 && ` — ${gridPoints} grid point${gridPoints === 1 ? "" : "s"}`}.
      </p>

      <label className="check-row" title={silicon ? undefined
        : "Temperature is a silicon-only axis; this circuit's model family has none."}>
        <input
          type="checkbox"
          checked={useTemps && silicon}
          disabled={!silicon}
          onChange={(e) => setUseTemps(e.target.checked)}
        />
        <span>Temperature (°C)</span>
      </label>
      {useTemps && silicon && (
        <input
          className="axis-input"
          value={tempText}
          onChange={(e) => setTempText(e.target.value)}
          placeholder="0, 27, 85"
          spellCheck={false}
        />
      )}

      <label className="check-row" title={silicon ? undefined
        : "Supply scaling is a silicon-only axis; this circuit's model family has none."}>
        <input
          type="checkbox"
          checked={useVdd && silicon}
          disabled={!silicon}
          onChange={(e) => setUseVdd(e.target.checked)}
        />
        <span>Supply scale (×)</span>
      </label>
      {useVdd && silicon && (
        <input
          className="axis-input"
          value={vddText}
          onChange={(e) => setVddText(e.target.value)}
          placeholder="0.9, 1.0, 1.1"
          spellCheck={false}
        />
      )}

      {!silicon && (
        <p className="muted small">
          Temperature and supply axes need an all-silicon circuit; this one sweeps
          process corners only.
        </p>
      )}

      <button
        className="btn primary"
        onClick={runPvt}
        disabled={busy || empty}
        title={empty ? "The canvas is empty" : "Run the PVT corner sweep"}
      >
        {busy && sweep.kind === "pvt" ? "Running…" : "Run PVT"}
      </button>

      <h3>Mismatch Monte-Carlo</h3>
      <div className="inline-fields">
        <label className="field">
          <span>Samples</span>
          <input
            type="number"
            min={2}
            value={samples}
            onChange={(e) => setSamples(Math.max(2, Number(e.target.value) || 2))}
          />
        </label>
        <label className="field">
          <span>Seed</span>
          <input
            type="number"
            value={seed}
            onChange={(e) => setSeed(Number(e.target.value) || 0)}
          />
        </label>
        <label className="field">
          <span>Workers</span>
          <input
            type="number"
            min={1}
            value={workers}
            onChange={(e) => setWorkers(Math.max(1, Number(e.target.value) || 1))}
          />
        </label>
      </div>
      <button
        className="btn primary"
        onClick={runMc}
        disabled={busy || empty}
        title={empty ? "The canvas is empty" : "Run a per-device mismatch Monte-Carlo"}
      >
        {busy && sweep.kind === "mc" ? "Running…" : "Run MC"}
      </button>

      <p className="muted small hint">
        Both run in the background; progress and results appear in the Sweep tab
        below.
      </p>
    </section>
  );
}
