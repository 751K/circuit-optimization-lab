/**
 * The PVT corner grid.
 *
 * `corner_table` nests its result by whichever axes are active — flat by corner,
 * then temperature, then supply scale — so the first job here is flattening that
 * back into one row per grid point. {@link flattenPvt} is exported for test.
 *
 * Two presentation decisions carry weight:
 *
 *  - A corner whose cards select **zero bins** for this geometry comes back as
 *    `null`. It is drawn as an explicit "not evaluated" row, never omitted: a
 *    missing row reads as a grid that was smaller than it was, which would let a
 *    corner silently escape review.
 *  - The **worst** cell in each column is marked. Ranking is the point of running
 *    a grid, and scanning 45 numbers for the smallest gain is not something a
 *    reader should be asked to do.
 */
import { formatValue } from "./format";

export interface PvtMetrics {
  gain_peak_dB?: number;
  bw_Hz?: number;
  irn_uV?: number;
  latch_dV?: number;
  [key: string]: unknown;
}

export interface PvtResult {
  table: Record<string, unknown>;
  corners: string[];
  temps: number[] | null;
  vdd_scale: number[] | null;
  silicon?: boolean;
  slices?: number;
  freq_range_hz?: [number, number] | null;
  freq_source?: string;
  noise_band_hz?: [number, number] | null;
}

export interface PvtRow {
  corner: string;
  temp: number | null;
  vdd: number | null;
  metrics: PvtMetrics | null;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * Flatten a nested corner table into one row per grid point.
 *
 * The nesting depth is implied by which axes are active, so it is read from
 * `temps`/`vdd_scale` rather than guessed from the object shape — a metrics dict
 * and an axis level are both plain objects, and telling them apart structurally
 * would break on the first metrics key that happened to look like a number.
 */
export function flattenPvt(result: PvtResult): PvtRow[] {
  const rows: PvtRow[] = [];
  const hasTemp = Array.isArray(result.temps) && result.temps.length > 0;
  const hasVdd = Array.isArray(result.vdd_scale) && result.vdd_scale.length > 0;

  const asMetrics = (value: unknown): PvtMetrics | null =>
    isRecord(value) ? (value as PvtMetrics) : null;

  for (const [corner, node] of Object.entries(result.table ?? {})) {
    if (!hasTemp && !hasVdd) {
      rows.push({ corner, temp: null, vdd: null, metrics: asMetrics(node) });
      continue;
    }
    if (!isRecord(node)) continue;

    if (hasTemp && hasVdd) {
      for (const [temp, inner] of Object.entries(node)) {
        if (!isRecord(inner)) continue;
        for (const [vdd, cell] of Object.entries(inner)) {
          rows.push({
            corner,
            temp: Number(temp),
            vdd: Number(vdd),
            metrics: asMetrics(cell),
          });
        }
      }
    } else {
      for (const [key, cell] of Object.entries(node)) {
        rows.push({
          corner,
          temp: hasTemp ? Number(key) : null,
          vdd: hasVdd ? Number(key) : null,
          metrics: asMetrics(cell),
        });
      }
    }
  }
  return rows;
}

/** Column definitions: which way is "worse" decides what gets marked. */
const COLUMNS: {
  key: keyof PvtMetrics & string;
  label: string;
  unit: string;
  worst: "min" | "max";
}[] = [
  { key: "gain_peak_dB", label: "Gain", unit: "dB", worst: "min" },
  { key: "bw_Hz", label: "BW", unit: "Hz", worst: "min" },
  { key: "irn_uV", label: "Input noise", unit: "V", worst: "max" },
  { key: "latch_dV", label: "Latch ΔV", unit: "V", worst: "max" },
];

/** The row index holding the worst value of each column. */
export function worstRows(rows: PvtRow[]): Record<string, number> {
  const out: Record<string, number> = {};
  for (const column of COLUMNS) {
    let best: number | null = null;
    let index = -1;
    rows.forEach((row, i) => {
      const raw = row.metrics?.[column.key];
      if (typeof raw !== "number" || !Number.isFinite(raw)) return;
      if (best === null
        || (column.worst === "min" ? raw < best : raw > best)) {
        best = raw;
        index = i;
      }
    });
    if (index >= 0) out[column.key] = index;
  }
  return out;
}

function cellValue(row: PvtRow, key: string, unit: string): string {
  const raw = row.metrics?.[key];
  if (typeof raw !== "number") return "—";
  // irn_uV is reported in microvolts; carry it as volts so the SI formatter
  // picks the prefix rather than printing "2188604 uV".
  return formatValue(key === "irn_uV" ? raw * 1e-6 : raw, unit, 4);
}

export function PvtView({ result }: { result: PvtResult }) {
  const rows = flattenPvt(result);
  const worst = worstRows(rows);
  const skipped = rows.filter((row) => row.metrics === null);
  const hasTemp = Array.isArray(result.temps) && result.temps.length > 0;
  const hasVdd = Array.isArray(result.vdd_scale) && result.vdd_scale.length > 0;

  return (
    <div className="pvt-view">
      <div className="sweep-meta">
        <span>{rows.length} grid point{rows.length === 1 ? "" : "s"}</span>
        <span>corners: {result.corners.join(", ")}</span>
        {hasTemp && <span>temps: {result.temps!.join(", ")} °C</span>}
        {hasVdd && <span>supply scale: {result.vdd_scale!.join(", ")}×</span>}
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

      {skipped.length > 0 && (
        <div className="health-flag warn">
          <strong>{skipped.length} of {rows.length} grid points not evaluated.</strong>{" "}
          Those corners select zero model bins for this geometry, so they were
          skipped rather than solved. They are listed below as “not evaluated” —
          they are not passes.
        </div>
      )}

      <div className="table-scroll">
        <table className="dtable">
          <thead>
            <tr>
              <th>Corner</th>
              {hasTemp && <th>Temp</th>}
              {hasVdd && <th>Supply</th>}
              {COLUMNS.map((column) => (
                <th key={column.key}>{column.label}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, index) => (
              <tr
                key={`${row.corner}-${row.temp}-${row.vdd}`}
                className={row.metrics === null ? "row-skipped" : ""}
              >
                <td className="mono">{row.corner}</td>
                {hasTemp && <td className="num">{row.temp} °C</td>}
                {hasVdd && <td className="num">{row.vdd}×</td>}
                {row.metrics === null ? (
                  <td className="muted" colSpan={COLUMNS.length}>
                    not evaluated — no model bin for this geometry at this corner
                  </td>
                ) : (
                  COLUMNS.map((column) => (
                    <td
                      key={column.key}
                      className={`num${worst[column.key] === index ? " worst" : ""}`}
                      title={worst[column.key] === index ? "worst in grid" : undefined}
                    >
                      {cellValue(row, column.key, column.unit)}
                    </td>
                  ))
                )}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
