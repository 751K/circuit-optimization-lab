/**
 * One analysis result, rendered in reading order:
 *
 *   health flags → headline metrics → operating point → plots → raw JSON
 *
 * The order is deliberate. A flag saying the shooting iteration never converged
 * has to be read *before* the numbers it invalidates, and the raw tree comes
 * last because it is for the question the curated view did not anticipate.
 */
import { useState } from "react";
import { Chart } from "./Chart";
import { formatQuantity } from "./format";
import {
  bandwidthIsGridLimited,
  extractHealth,
  extractMetrics,
  responseIsAtGainFloor,
  timeDomainIsFlat,
} from "./metrics";
import { OperatingPoint, hasOperatingPoint } from "./OperatingPoint";
import { buildPlots } from "./transform";

interface ResultViewProps {
  name: string;
  result: unknown;
}

function JsonNode({ label, value }: { label?: string; value: unknown }) {
  if (Array.isArray(value)) {
    const preview = value.length > 8 ? `${value.length} items` : JSON.stringify(value);
    return (
      <details className="jt-node">
        <summary className="jt-summary">
          {label && <span className="jt-key">{label}: </span>}
          [{preview}]
        </summary>
        <div className="jt-children">
          {value.slice(0, 200).map((item, index) => (
            <JsonNode key={index} label={String(index)} value={item} />
          ))}
          {value.length > 200 && (
            <div className="jt-leaf muted">… {value.length - 200} more</div>
          )}
        </div>
      </details>
    );
  }

  if (typeof value === "object" && value !== null) {
    const entries = Object.entries(value);
    return (
      <details className="jt-node">
        <summary className="jt-summary">
          {label && <span className="jt-key">{label}: </span>}
          {`{${entries.length} fields}`}
        </summary>
        <div className="jt-children">
          {entries.map(([key, item]) => (
            <JsonNode key={key} label={key} value={item} />
          ))}
        </div>
      </details>
    );
  }

  return (
    <div className="jt-leaf">
      {label && <span className="jt-key">{label}: </span>}
      <span className="jt-val">{JSON.stringify(value)}</span>
    </div>
  );
}

export function ResultView({ name, result }: ResultViewProps) {
  const [showRaw, setShowRaw] = useState(false);
  const metrics = extractMetrics(name, result);
  const health = extractHealth(result);
  const plots = buildPlots(name, result);
  const showOp = hasOperatingPoint(result);
  const gridLimited = bandwidthIsGridLimited(result);
  const flat = (name === "transient" || name === "pss") && timeDomainIsFlat(result);
  const noDrive = (name === "ac" || name === "pac") && responseIsAtGainFloor(result);

  return (
    <div className="result-view">
      {health.map((flag) => (
        <div key={flag.key} className={`health-flag ${flag.severity}`}>
          <strong>{flag.label}</strong> — {flag.hint}
        </div>
      ))}

      {noDrive && (
        <div className="health-flag warn">
          <strong>The response is on the numerical gain floor.</strong> A gain near
          −180 dB across the whole sweep means the circuit was never driven, not
          that it has no gain: add an <code>ac_drives</code> entry naming the
          source and its magnitude. Check the stimulus before looking at devices.
        </div>
      )}

      {flat && (
        <div className="health-flag warn">
          <strong>Every trace is constant.</strong> The run converged, so this is
          almost always a missing stimulus rather than a broken circuit — a
          transient needs a <code>drives</code> entry (or a <code>periodic</code>
          block) to have anything to respond to. A balanced amplifier with no
          input holds its output at exactly zero.
        </div>
      )}

      {metrics.length > 0 && (
        <div className="metric-row">
          {metrics.map((metric) => {
            const quantity = formatQuantity(metric.value, metric.unit);
            const suspect = metric.key === "bw_Hz" && gridLimited;
            return (
              <div
                className={`metric-card${suspect ? " suspect" : ""}`}
                key={metric.key}
                title={
                  suspect
                    ? "This value sits at the top of the swept frequency range, so "
                      + "it is the sweep limit rather than a measured corner. Widen "
                      + "the sweep to measure it."
                    : metric.hint
                }
              >
                <div className="metric-key">
                  {metric.label}
                  {suspect && <span className="metric-warn"> ⚠ sweep limit</span>}
                </div>
                <div className="metric-val">
                  {quantity.value}
                  {quantity.unit && <span className="metric-unit"> {quantity.unit}</span>}
                </div>
              </div>
            );
          })}
        </div>
      )}

      {showOp && <OperatingPoint result={result} />}

      {plots.map((plot, index) => (
        <Chart key={plot.title ?? index} plot={plot} />
      ))}

      <button className="btn tiny raw-toggle" onClick={() => setShowRaw((v) => !v)}>
        {showRaw ? "Hide raw result" : "Raw result…"}
      </button>
      {showRaw && (
        <div className="jt-wrap">
          <JsonNode value={result} />
        </div>
      )}
    </div>
  );
}
