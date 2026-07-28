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
import { formatValue } from "../results/format";
import { NumberField } from "./fields";

/** Parse "0, 27, 85" into numbers, ignoring blanks and junk. */
export function parseAxis(text: string): number[] {
  return text
    .split(/[,\s]+/)
    .map((part) => part.trim())
    .filter((part) => part.length > 0)
    .map(Number)
    .filter((value) => Number.isFinite(value));
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

export interface SweepRange {
  start: number;
  stop: number;
  bandLo: number;
  bandHi: number;
}

/**
 * The measurement range a sweep should use, taken from the circuit when it says.
 *
 * Both sweeps default to `corner_table`'s AFE range of 0.01 Hz – 10 kHz, which
 * on a silicon amplifier reports the top of its own grid as the bandwidth. The
 * service already inherits `analyses.ac.freqs` when a circuit has one — but many
 * decks (every chopper here, the plain 5T OTAs) have no `analyses` block at all,
 * and those fall back to the AFE range with no way to say otherwise. So the
 * panel prefills from the circuit where it can and exposes the range where it
 * cannot.
 */
export function rangeForCircuit(circuit: unknown): SweepRange {
  const fallback: SweepRange = {
    start: 1e3, stop: 1e10, bandLo: 1e4, bandHi: 1e7,
  };
  if (!isRecord(circuit)) return fallback;
  const analyses = isRecord(circuit.analyses) ? circuit.analyses : {};

  const ac = isRecord(analyses.ac) ? analyses.ac : null;
  const freqs = ac && isRecord(ac.freqs) ? ac.freqs : null;
  const start = freqs && typeof freqs.start === "number" ? freqs.start : fallback.start;
  const stop = freqs && typeof freqs.stop === "number" ? freqs.stop : fallback.stop;

  const noise = isRecord(analyses.noise) ? analyses.noise : null;
  const band = noise && Array.isArray(noise.band) ? noise.band : null;
  const bandLo = band && typeof band[0] === "number" ? band[0] : fallback.bandLo;
  const bandHi = band && typeof band[1] === "number" ? band[1] : fallback.bandHi;

  return { start, stop, bandLo, bandHi };
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

  // Prefilled from the circuit on first render; the circuit is loaded before
  // this panel is ever opened, so a lazy initializer is enough.
  const [range, setRange] = useState<SweepRange>(() => rangeForCircuit(exportJson()));
  const [showRange, setShowRange] = useState(false);

  const busy = sweep.status === "queued" || sweep.status === "running";
  const empty = nodeCount === 0;

  const temps = parseAxis(tempText);
  const vdds = parseAxis(vddText);
  const gridPoints =
    (circuitCorners?.length ?? 0)
    * (useTemps && temps.length ? temps.length : 1)
    * (useVdd && vdds.length ? vdds.length : 1);

  /** The measurement range both sweeps send, so their numbers are comparable. */
  const rangeArgs = {
    freqs: { start: range.start, stop: range.stop, num: 121, scale: "log" },
    band: [range.bandLo, range.bandHi] as [number, number],
  };

  const resetRange = (): void => setRange(rangeForCircuit(exportJson()));

  const runPvt = (): void => {
    void startPvt(exportJson(), {
      workers,
      ...rangeArgs,
      ...(useTemps && temps.length ? { temps } : {}),
      ...(useVdd && vdds.length ? { vdd_scale: vdds } : {}),
    });
  };

  const runMc = (): void => {
    void startMc(exportJson(), { n: samples, seed, workers, ...rangeArgs });
  };

  return (
    <section className="panel sweep-panel">
      <h2>Sweeps</h2>

      <div className="sub-config">
        <button className="sub-config-tab" onClick={() => setShowRange((v) => !v)}>
          {showRange ? "▾" : "▸"} Measurement range
        </button>
        <p className="muted small">
          {formatValue(range.start, "Hz", 3)} – {formatValue(range.stop, "Hz", 3)},
          noise band {formatValue(range.bandLo, "Hz", 3)} –{" "}
          {formatValue(range.bandHi, "Hz", 3)}
        </p>
        {showRange && (
          <>
            <div className="inline-fields">
              <NumberField
                label="Sweep from" unit="Hz" value={range.start}
                onCommit={(v) => v !== undefined && setRange({ ...range, start: v })}
              />
              <NumberField
                label="to" unit="Hz" value={range.stop}
                onCommit={(v) => v !== undefined && setRange({ ...range, stop: v })}
              />
            </div>
            <div className="inline-fields">
              <NumberField
                label="Noise band" unit="Hz" value={range.bandLo}
                onCommit={(v) => v !== undefined && setRange({ ...range, bandLo: v })}
              />
              <NumberField
                label="to" unit="Hz" value={range.bandHi}
                onCommit={(v) => v !== undefined && setRange({ ...range, bandHi: v })}
              />
            </div>
            <button className="btn tiny" onClick={resetRange}>
              Reset from circuit
            </button>
            <p className="muted small">
              Both sweeps measure gain, bandwidth and noise over this range. A
              bandwidth above the sweep top cannot be measured — it comes back as
              the top of the grid, identical for every sample.
            </p>
          </>
        )}
      </div>

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
