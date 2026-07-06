/**
 * Client-side analysis-config defaults for the Run panel (F3 follow-up).
 *
 * The backend's `selected` parameter only *filters* the circuit's own
 * `analyses` block — a circuit with no block (e.g. sky130_5t_ota, or anything
 * drawn from scratch) solves to "No analyses configured", and `ac: {}` is also
 * rejected ("frequency grid is required"). So before sending a solve request
 * the panel injects a sensible default config for each selected analysis the
 * circuit doesn't configure — **into the request payload only**. The editor
 * state and Export JSON are never touched.
 *
 * Only `ac` and `noise` have a safe universal default (a wide log frequency
 * sweep; verified against the live backend — noise works with `freqs` alone,
 * its band-integrated scalars just come back null). transient/pss/pac/pnoise
 * need circuit-specific settings (tstop, a `periodic` block, drives …), so a
 * selected-but-unconfigured one is intercepted client-side with a clear
 * message instead of a request.
 */
import type { CircuitJson } from "../model/circuit";

/**
 * The default AC/noise sweep: 10 mHz – 1 GHz, 10 pts/decade (111 points, log).
 * Wide enough to cover both sub-kHz OTFT AFEs and ~100 MHz silicon OTAs.
 */
export const DEFAULT_FREQS = {
  start: 1e-2,
  stop: 1e9,
  num: 111,
  scale: "log",
} as const;

/** Human-readable description of the injected sweep, for the results note. */
export const DEFAULT_SWEEP_LABEL = "default sweep 10 mHz – 1 GHz (log, 111 pts)";

/**
 * The config to inject for `name` when the circuit doesn't configure it, or
 * `null` when no sensible client-side default exists.
 */
export function defaultAnalysisConfig(name: string): Record<string, unknown> | null {
  switch (name) {
    case "ac":
    case "noise":
      return { freqs: { ...DEFAULT_FREQS } };
    default:
      return null;
  }
}

/** What `prepareSolveCircuit` decided. */
export interface SolvePrep {
  /**
   * The circuit to POST. A shallow-patched copy when defaults were injected;
   * the original object (same reference, untouched) when nothing was.
   */
  circuit: CircuitJson;
  /** Selected analyses that received an injected default config. */
  injected: string[];
  /**
   * Selected analyses with neither a circuit config nor a client default —
   * the caller must block the run and explain, not send the request.
   */
  missing: string[];
}

function isPlainObject(x: unknown): x is Record<string, unknown> {
  return typeof x === "object" && x !== null && !Array.isArray(x);
}

/**
 * Build the request-only circuit for a solve of `selected`:
 *  - an analysis already configured in `circuit.analyses` is never overridden;
 *  - an unconfigured one with a default gets it injected (on a copied
 *    `analyses` object — the input circuit is never mutated);
 *  - an unconfigured one without a default lands in `missing`.
 *
 * A malformed non-object `analyses` value is passed through untouched (the
 * backend reports it) — we never rewrite bytes we don't understand.
 */
export function prepareSolveCircuit(circuit: CircuitJson, selected: string[]): SolvePrep {
  const raw = circuit.analyses;
  if (raw !== undefined && !isPlainObject(raw)) {
    return { circuit, injected: [], missing: [] };
  }
  const existing: Record<string, unknown> = raw ?? {};

  const injections: Record<string, Record<string, unknown>> = {};
  const injected: string[] = [];
  const missing: string[] = [];
  for (const name of selected) {
    if (name in existing) continue; // circuit's own config always wins
    const def = defaultAnalysisConfig(name);
    if (def) {
      injections[name] = def;
      injected.push(name);
    } else {
      missing.push(name);
    }
  }

  if (injected.length === 0) {
    return { circuit, injected, missing };
  }
  return {
    circuit: { ...circuit, analyses: { ...existing, ...injections } },
    injected,
    missing,
  };
}

/** The panel's user-facing message for blocked (missing-config) analyses. */
export function missingConfigMessage(missing: string[]): string {
  const list = missing.join(", ");
  return (
    `${list} ${missing.length > 1 ? "have" : "has"} no configuration in the ` +
    `circuit's "analyses" block and no client-side default exists ` +
    `(transient needs tstop/n_points, pss/pac/pnoise need a "periodic" block ` +
    `and drives). Add the config to the circuit JSON or uncheck ${list}.`
  );
}
