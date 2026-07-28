/**
 * Engineering-notation value formatting.
 *
 * Solver results are raw SI scalars: a bandwidth is `79034.5`, an
 * input-referred noise is `8.386e-5`. Rendered verbatim those are unreadable and,
 * worse, hard to compare — the eye cannot rank `7.9e4` against `1.02e8` at a
 * glance. Everything on screen therefore goes through {@link formatQuantity},
 * which picks the SI prefix that puts the mantissa in [1, 1000).
 *
 * Two deliberate exceptions to prefixing:
 *  - **dB, degrees and dimensionless counts** are never prefixed. "k dB" is not a
 *    unit, and a step count of 2001 must not read as "2.001 k".
 *  - **Temperature in °C** is absolute, not a magnitude; 0.05 °C is not "50 m°C".
 */

/** SI prefixes from 1e-15 to 1e12, the range solver results actually span. */
const PREFIXES: { exp: number; symbol: string }[] = [
  { exp: 12, symbol: "T" },
  { exp: 9, symbol: "G" },
  { exp: 6, symbol: "M" },
  { exp: 3, symbol: "k" },
  { exp: 0, symbol: "" },
  { exp: -3, symbol: "m" },
  { exp: -6, symbol: "µ" },
  { exp: -9, symbol: "n" },
  { exp: -12, symbol: "p" },
  { exp: -15, symbol: "f" },
];

/** Units that carry no magnitude prefix. */
const UNPREFIXED = new Set(["dB", "dBc", "deg", "°", "°C", "", "x", "bit", "V/V"]);

export interface Quantity {
  /** The number with its prefix applied, e.g. "79.03". */
  value: string;
  /** The prefixed unit, e.g. "kHz". Empty when the quantity is dimensionless. */
  unit: string;
  /** `value` and `unit` joined for single-line display. */
  text: string;
}

function significant(value: number, digits: number): string {
  // toPrecision keeps trailing zeros ("79.030"); strip them so columns of
  // numbers stay narrow, but never strip a leading integer digit.
  const fixed = value.toPrecision(digits);
  if (!fixed.includes(".") || fixed.includes("e")) return fixed;
  return fixed.replace(/\.?0+$/, "");
}

/**
 * Format `value` in engineering notation against `unit`.
 *
 * A non-finite value is *not* rendered as a number: NaN and ±Infinity mean a
 * measurement did not happen, and printing "NaN" beside real results invites
 * reading it as one. They come back as an em dash with the reason in `unit`.
 */
export function formatQuantity(
  value: number | null | undefined,
  unit = "",
  digits = 4,
): Quantity {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    const why = value === null || value === undefined ? "not measured"
      : Number.isNaN(value) ? "NaN" : "infinite";
    return { value: "—", unit: why, text: `— (${why})` };
  }
  if (value === 0) return { value: "0", unit, text: unit ? `0 ${unit}` : "0" };

  if (UNPREFIXED.has(unit)) {
    const text = significant(value, digits);
    return { value: text, unit, text: unit ? `${text} ${unit}` : text };
  }

  const magnitude = Math.abs(value);
  // Below the smallest prefix (femto) the value keeps that prefix rather than
  // falling off the scale — "0.001 f" is still readable, "1e-18" beside a table
  // of prefixed numbers is not.
  const smallest = PREFIXES[PREFIXES.length - 1] as { exp: number; symbol: string };
  const chosen = PREFIXES.find((p) => magnitude >= Math.pow(10, p.exp)) ?? smallest;
  const scaled = value / Math.pow(10, chosen.exp);
  const shown = significant(scaled, digits);
  const prefixed = `${chosen.symbol}${unit}`;
  return {
    value: shown,
    unit: prefixed,
    text: prefixed ? `${shown} ${prefixed}` : shown,
  };
}

/** {@link formatQuantity} reduced to its display string. */
export function formatValue(
  value: number | null | undefined,
  unit = "",
  digits = 4,
): string {
  return formatQuantity(value, unit, digits).text;
}

/**
 * Format a wall-clock duration in seconds. Sub-second runs are the common case
 * here, so they get milliseconds rather than "0.0043 s".
 */
export function formatDuration(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return "—";
  if (seconds < 1) return `${(seconds * 1000).toPrecision(3)} ms`;
  if (seconds < 60) return `${seconds.toPrecision(3)} s`;
  const minutes = Math.floor(seconds / 60);
  return `${minutes} min ${Math.round(seconds - minutes * 60)} s`;
}

/** A signed percentage, for margins. Zero keeps its sign off. */
export function formatPercent(fraction: number | null | undefined): string {
  if (fraction === null || fraction === undefined || !Number.isFinite(fraction)) {
    return "—";
  }
  const pct = fraction * 100;
  const sign = pct > 0 ? "+" : "";
  return `${sign}${Math.abs(pct) >= 100 ? pct.toFixed(0) : pct.toFixed(1)}%`;
}
