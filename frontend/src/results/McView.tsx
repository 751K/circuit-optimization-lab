/**
 * Mismatch Monte-Carlo: distribution per metric, plus the latch screen.
 *
 * A mismatch run answers a spread question, so the summary statistics lead and
 * the histogram supports them. Mean and σ alone would not be enough — a
 * bimodal distribution (half the samples latching into the wrong state) has a
 * perfectly ordinary mean, and that is exactly the failure MC is run to find. So
 * the histogram is drawn from the raw per-sample array rather than from the
 * summary, and the latch count is stated whether or not it is zero.
 */
import { useMemo, useState } from "react";
import { Chart } from "./Chart";
import { formatValue } from "./format";
import { NO_STIMULUS_HINT } from "./metrics";
import type { PlotSpec } from "./transform";

export interface McStats {
  mean: number;
  std: number;
  p5: number;
  p95: number;
}

export interface McResult {
  arrays: Record<string, number[]>;
  latched: boolean[];
  summary: {
    n: number;
    latched: number;
    latch_rate: number;
    noise_evaluated?: number;
    [metric: string]: unknown;
  };
  stopped_early?: boolean;
  freq_range_hz?: [number, number] | null;
  freq_source?: string;
  noise_band_hz?: [number, number] | null;
}

/** Metric key → label and unit. `irn_uV` is carried in µV by the solver. */
const METRIC_UNITS: Record<string, { label: string; unit: string; scale: number }> = {
  gain_peak_dB: { label: "Peak gain", unit: "dB", scale: 1 },
  bw_Hz: { label: "Bandwidth", unit: "Hz", scale: 1 },
  irn_uV: { label: "Input noise", unit: "V", scale: 1e-6 },
  latch_dV: { label: "Latch ΔV", unit: "V", scale: 1 },
};

function isStats(value: unknown): value is McStats {
  return (
    typeof value === "object" && value !== null
    && typeof (value as McStats).mean === "number"
  );
}

/**
 * Whether a metric is identical across every sample *because* it sat on the top
 * of the swept range rather than because the design is insensitive.
 *
 * `bw_Hz` is found by scanning the swept response, so a circuit whose corner
 * lies above the grid reports the last grid point — for every sample, giving a
 * spread of exactly zero. Read as a statistic that is an extraordinarily robust
 * design; read correctly it is a measurement that never happened. Exported for
 * test.
 */
export function metricIsGridPinned(result: McResult, key: string): boolean {
  const values = result.arrays?.[key];
  const top = result.freq_range_hz?.[1];
  if (!Array.isArray(values) || values.length < 2 || typeof top !== "number") {
    return false;
  }
  const first = values[0];
  if (typeof first !== "number") return false;
  if (!values.every((v) => v === first)) return false;
  return first >= top * 0.999;
}

/** Bin a sample array into a histogram plot. Exported for test. */
export function histogram(values: number[], bins = 24): PlotSpec | null {
  const finite = values.filter((value) => Number.isFinite(value));
  if (finite.length < 2) return null;
  const min = Math.min(...finite);
  const max = Math.max(...finite);
  if (min === max) return null;

  const width = (max - min) / bins;
  const counts = new Array(bins).fill(0);
  for (const value of finite) {
    // The maximum sample belongs in the last bin, not one past the end.
    const index = Math.min(bins - 1, Math.floor((value - min) / width));
    counts[index] += 1;
  }
  const centres = counts.map((_, index) => min + width * (index + 0.5));

  return {
    x: centres,
    xLabel: "",
    xLog: false,
    yLabel: "Samples",
    yLog: false,
    kind: "bar",
    yFromZero: true,     // a bar's height is only meaningful from zero
    series: [{ name: "count", values: counts }],
  };
}

export function McView({ result }: { result: McResult }) {
  const metricKeys = Object.keys(result.arrays ?? {});
  const [selected, setSelected] = useState(metricKeys[0] ?? "");
  const active = metricKeys.includes(selected) ? selected : (metricKeys[0] ?? "");

  const plot = useMemo(() => {
    const values = active ? result.arrays?.[active] : undefined;
    if (!values) return null;
    const spec = histogram(values);
    if (!spec) return null;
    const meta = METRIC_UNITS[active];
    return {
      ...spec,
      title: `${meta?.label ?? active} distribution`,
      xLabel: meta ? `${meta.label} (${meta.unit})` : active,
    };
  }, [result, active]);

  const { summary } = result;
  const latchRate = summary?.latch_rate ?? 0;
  const pinned = active === "bw_Hz" && metricIsGridPinned(result, active);
  // One check that explains all three rows at once: with no AC stimulus the gain
  // is the numerical floor and everything derived from it is meaningless.
  const gainStats = summary?.gain_peak_dB;
  const noStimulus = isStats(gainStats) && gainStats.mean < -120;

  return (
    <div className="mc-view">
      {noStimulus && (
        <div className="health-flag error">
          <strong>This sweep had no AC stimulus.</strong> {NO_STIMULUS_HINT}
        </div>
      )}
      <div className="sweep-meta">
        <span>{summary?.n ?? 0} samples</span>
        {result.stopped_early && <span className="warn">stopped early</span>}
        <span className={latchRate > 0 ? "err" : ""}>
          latched: {summary?.latched ?? 0} ({(latchRate * 100).toFixed(1)}%)
        </span>
        {typeof summary?.noise_evaluated === "number"
          && summary.noise_evaluated < (summary.n ?? 0) && (
          <span className="warn">
            noise evaluated on {summary.noise_evaluated} of {summary.n}
          </span>
        )}
        {result.freq_range_hz && (
          <span title={`Sweep range taken from: ${result.freq_source}`}>
            swept {formatValue(result.freq_range_hz[0], "Hz", 3)} –{" "}
            {formatValue(result.freq_range_hz[1], "Hz", 3)}
          </span>
        )}
        {result.noise_band_hz && (
          <span>
            noise band {formatValue(result.noise_band_hz[0], "Hz", 3)} –{" "}
            {formatValue(result.noise_band_hz[1], "Hz", 3)}
          </span>
        )}
      </div>

      <div className="table-scroll">
        <table className="dtable">
          <thead>
            <tr>
              <th>Metric</th>
              <th className="num">Mean</th>
              <th className="num">σ</th>
              <th className="num">p5</th>
              <th className="num">p95</th>
              <th className="num" title="σ as a fraction of the mean">σ/mean</th>
            </tr>
          </thead>
          <tbody>
            {metricKeys.map((key) => {
              const stats = summary?.[key];
              if (!isStats(stats)) return null;
              const meta = METRIC_UNITS[key];
              const scale = meta?.scale ?? 1;
              const unit = meta?.unit ?? "";
              const spread = stats.mean === 0 ? null : Math.abs(stats.std / stats.mean);
              return (
                <tr
                  key={key}
                  className={key === active ? "row-active" : ""}
                  onClick={() => setSelected(key)}
                >
                  <td className="mono">{meta?.label ?? key}</td>
                  <td className="num">{formatValue(stats.mean * scale, unit)}</td>
                  <td className="num">{formatValue(stats.std * scale, unit)}</td>
                  <td className="num">{formatValue(stats.p5 * scale, unit)}</td>
                  <td className="num">{formatValue(stats.p95 * scale, unit)}</td>
                  <td className="num">
                    {spread === null ? "—" : `${(spread * 100).toFixed(2)}%`}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div className="metric-picker">
        {metricKeys.map((key) => (
          <button
            key={key}
            className={`chip${key === active ? " on" : ""}`}
            onClick={() => setSelected(key)}
          >
            {METRIC_UNITS[key]?.label ?? key}
          </button>
        ))}
      </div>

      {plot ? (
        <Chart plot={plot} height={260} />
      ) : pinned ? (
        <div className="health-flag warn">
          <strong>{METRIC_UNITS[active]?.label ?? active} was not measured.</strong>{" "}
          Every sample returned{" "}
          {formatValue(
            (result.arrays[active]?.[0] ?? 0) * (METRIC_UNITS[active]?.scale ?? 1),
            METRIC_UNITS[active]?.unit ?? "",
          )}
          , which is the top of the swept range — the real value lies above it, so
          the sweep reported its own ceiling instead. Widen{" "}
          <strong>Measurement range</strong> in the Sweeps panel and run again.
          A zero spread here means the metric never varied because it was never
          resolved, not that the design is unusually tight.
        </div>
      ) : (
        <p className="muted small">
          Every sample of {METRIC_UNITS[active]?.label ?? active} landed on the
          same value, so there is no distribution to bin.
        </p>
      )}
    </div>
  );
}
