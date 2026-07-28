/**
 * The signoff verdict a `/solve` already returns and the UI used to discard.
 *
 * Measurements come back unit-bearing and individually statused, so a spec that
 * could not be measured (`status: "invalid"`) is shown as exactly that rather
 * than as a failure or, worse, as a pass. `invalid` outranks `fail`: a number
 * nobody could compute is a stronger finding than a number that missed.
 */
import { formatValue } from "./format";

interface Measurement {
  value?: number | null;
  unit?: string;
  status?: string;
  [key: string]: unknown;
}

export interface SignoffPayload {
  status: "pass" | "fail" | "not_configured" | string;
  measurements: Record<string, Measurement>;
  constraints: Record<string, unknown>;
  passed: boolean | null;
  worst_case: Record<string, unknown> | null;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/** Describe how a measurement sits against its declared limit. */
function verdictFor(
  measurement: Measurement,
  constraint: unknown,
): { label: string; tone: "ok" | "bad" | "muted"; limit: string } {
  if (measurement.status && measurement.status !== "valid") {
    return { label: measurement.status, tone: "bad", limit: "—" };
  }
  if (!isRecord(constraint) || typeof measurement.value !== "number") {
    return { label: "—", tone: "muted", limit: "—" };
  }
  const value = measurement.value;
  const unit = measurement.unit ?? "";
  const min = typeof constraint.min === "number" ? constraint.min : null;
  const max = typeof constraint.max === "number" ? constraint.max : null;

  const parts: string[] = [];
  if (min !== null) parts.push(`≥ ${formatValue(min, unit)}`);
  if (max !== null) parts.push(`≤ ${formatValue(max, unit)}`);
  const limit = parts.length ? parts.join(", ") : "—";

  const ok = (min === null || value >= min) && (max === null || value <= max);
  return { label: ok ? "pass" : "fail", tone: ok ? "ok" : "bad", limit };
}

export function SignoffView({ signoff }: { signoff: SignoffPayload }) {
  if (signoff.status === "not_configured") {
    return (
      <p className="muted small">
        This circuit declares no <code>signoff</code> block, so there is nothing to
        judge the measurements against. Add one to get a pass/fail verdict — phase
        margin, settling time and saturation checks are only produced when asked
        for, never inferred from an AC response.
      </p>
    );
  }

  const rows = Object.entries(signoff.measurements ?? {});
  const tone = signoff.status === "pass" ? "ok"
    : signoff.status === "fail" ? "bad" : "warn";

  return (
    <div className="signoff-view">
      <div className={`verdict ${tone}`}>
        <span className="verdict-word">{signoff.status}</span>
        {signoff.worst_case && isRecord(signoff.worst_case) && (
          <span className="verdict-note">
            worst case: {String(signoff.worst_case.name ?? "")}
          </span>
        )}
      </div>

      <div className="table-scroll">
        <table className="dtable">
          <thead>
            <tr>
              <th>Measurement</th>
              <th className="num">Value</th>
              <th className="num">Limit</th>
              <th>Verdict</th>
            </tr>
          </thead>
          <tbody>
            {rows.map(([key, measurement]) => {
              const verdict = verdictFor(measurement, signoff.constraints?.[key]);
              return (
                <tr key={key} className={verdict.tone === "bad" ? "row-bad" : ""}>
                  <td className="mono">{key}</td>
                  <td className="num">
                    {formatValue(measurement.value, measurement.unit ?? "")}
                  </td>
                  <td className="num">{verdict.limit}</td>
                  <td>
                    <span className={`pill ${verdict.tone}`}>{verdict.label}</span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
