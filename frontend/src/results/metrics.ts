/**
 * Which result fields are *measurements* and what they mean.
 *
 * A solver result is a flat bag mixing three very different kinds of field:
 * engineering quantities (`bw_Hz`), solver telemetry (`pnoise_hb_sparse_nnz`,
 * `shooting_jacobian_reuses`), and health flags (`diverged`, `pnoise_degraded`).
 * A PNoise result has 33 scalars, of which four are things an engineer reads.
 *
 * Rendering them undifferentiated is not merely cluttered — it flattens the
 * distinction between a number that describes the circuit and a number that
 * describes the solver, and it buries the flags that say the measurement should
 * not be trusted at all. So this module splits them explicitly:
 *
 *   - {@link METRICS} — the headline quantities, with a label and a unit.
 *   - {@link HEALTH} — flags that qualify or invalidate the numbers beside them.
 *   - everything else stays in the raw JSON tree, one click away.
 */

export interface MetricSpec {
  /** Human label, e.g. "DC gain". */
  label: string;
  /** SI unit passed to formatQuantity; "" for dimensionless. */
  unit: string;
  /** Longer explanation shown on hover. */
  hint?: string;
}

/**
 * Headline metrics by result key. Keys are shared across analyses where the
 * quantity is the same (`bw_Hz` means the same thing in AC and PAC).
 */
export const METRICS: Record<string, MetricSpec> = {
  // ── AC / PAC ──
  Av_dc_dB: { label: "DC gain", unit: "dB", hint: "Low-frequency magnitude" },
  peak_dB: { label: "Peak gain", unit: "dB", hint: "Maximum magnitude over the sweep" },
  bw_Hz: {
    label: "Bandwidth",
    unit: "Hz",
    hint: "−3 dB corner relative to the DC value. Bounded by the sweep: a value "
      + "sitting exactly at the top of the frequency grid is a grid limit, not a "
      + "measurement.",
  },
  pacmag: { label: "PAC drive", unit: "", hint: "Small-signal excitation magnitude" },

  // ── noise / pnoise ──
  irn_uV_band: {
    label: "Input noise",
    unit: "V",
    hint: "Input-referred RMS over the integration band",
  },
  out_uV_band: {
    label: "Output noise",
    unit: "V",
    hint: "Output-referred RMS over the integration band",
  },
  f_chop: { label: "Chop frequency", unit: "Hz" },
  fundamental: { label: "Fundamental", unit: "Hz" },
  max_sideband: { label: "Sidebands", unit: "", hint: "Harmonics folded into the result" },

  // ── transient ──
  // The step count is the length of the time grid, not a scalar in the result.
  // `nsubsteps` counts retry *subdivisions* and is 0 for a clean run — printing
  // it as "Steps: 0" beside a solved 801-point waveform is the same mistake the
  // CLI summary made when it printed the node count instead.
  step_count: { label: "Steps", unit: "", hint: "Points on the solved time grid" },
  nfail: { label: "Failed steps", unit: "", hint: "Steps rejected and retried" },
  nretry: { label: "Retries", unit: "" },
  nsubsteps: {
    label: "Retry subdivisions",
    unit: "",
    hint: "Extra steps inserted when a step had to be halved; 0 on a clean run",
  },

  // ── PSS ──
  period: { label: "Period", unit: "s" },
  shooting_iters: { label: "Shooting iters", unit: "" },
  residual_norm: {
    label: "Residual",
    unit: "",
    hint: "Periodic-boundary mismatch; converged when below the tolerance",
  },
  residual_tol: { label: "Residual tol", unit: "" },
  rail_margin: {
    label: "Rail margin",
    unit: "V",
    hint: "Closest approach to a supply rail over the period",
  },
};

/** The metrics worth showing for a given analysis, in reading order. */
const ORDER: Record<string, string[]> = {
  ac: ["Av_dc_dB", "peak_dB", "bw_Hz"],
  noise: ["irn_uV_band", "out_uV_band"],
  transient: ["step_count", "nfail", "nretry", "nsubsteps"],
  pss: ["period", "step_count", "shooting_iters", "residual_norm", "residual_tol",
        "rail_margin"],
  pac: ["Av_dc_dB", "bw_Hz", "pacmag"],
  pnoise: ["irn_uV_band", "out_uV_band", "f_chop", "fundamental", "max_sideband"],
};

/**
 * A flag that qualifies the numbers next to it.
 *
 * `severity` drives how loudly it is shown. `bad` is the value that means
 * trouble: `converged` is alarming when false, `diverged` when true.
 */
export interface HealthSpec {
  label: string;
  bad: boolean;
  severity: "error" | "warn";
  hint: string;
}

export const HEALTH: Record<string, HealthSpec> = {
  converged: {
    label: "Shooting converged",
    bad: false,
    severity: "error",
    hint: "The periodic steady state was not reached; every PSS-derived number "
      + "below describes an unconverged orbit.",
  },
  diverged: {
    label: "Diverged",
    bad: true,
    severity: "error",
    hint: "The shooting iteration ran away rather than settling.",
  },
  stabilization_runaway: {
    label: "Stabilization runaway",
    bad: true,
    severity: "error",
    hint: "The warm-up transient did not stay bounded before shooting began.",
  },
  pnoise_degraded: {
    label: "PNoise degraded",
    bad: true,
    severity: "warn",
    hint: "The harmonic-balance solve fell back from its intended path; the "
      + "noise numbers are approximate.",
  },
  adaptive_grid_frozen: {
    label: "Adaptive grid frozen",
    bad: true,
    severity: "warn",
    hint: "Adaptive timestepping stopped refining and ran on a fixed grid.",
  },
};

export interface Metric {
  key: string;
  label: string;
  unit: string;
  value: number;
  hint?: string;
}

export interface HealthFlag {
  key: string;
  label: string;
  ok: boolean;
  severity: "error" | "warn";
  hint: string;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * The headline metrics for one analysis result.
 *
 * Only finite numbers are returned: a NaN `irn_uV_band` means the band
 * integration did not produce a value, and a metric card reading "NaN" beside
 * real ones invites reading it as a measurement. Those surface as health context
 * or in the raw tree instead.
 */
export function extractMetrics(analysis: string, result: unknown): Metric[] {
  if (!isRecord(result)) return [];
  const keys = ORDER[analysis] ?? Object.keys(METRICS);
  const out: Metric[] = [];
  for (const key of keys) {
    const spec = METRICS[key];
    // `step_count` is derived: the solved grid's length, not a field the solver
    // reports. See the METRICS entry for why it is not read from a scalar.
    const raw = key === "step_count"
      ? (Array.isArray(result.t) ? result.t.length : undefined)
      : result[key];
    if (!spec || typeof raw !== "number" || !Number.isFinite(raw)) continue;
    // Band-integrated noise is reported in µV but carries a volt unit here, so
    // the SI formatter can pick its own prefix from the true magnitude.
    const value = key.endsWith("_uV_band") ? raw * 1e-6 : raw;
    out.push({ key, label: spec.label, unit: spec.unit, value, hint: spec.hint });
  }
  return out;
}

/** Health flags present in a result, worst first. */
export function extractHealth(result: unknown): HealthFlag[] {
  if (!isRecord(result)) return [];
  const out: HealthFlag[] = [];
  for (const [key, spec] of Object.entries(HEALTH)) {
    const raw = result[key];
    if (typeof raw !== "boolean") continue;
    out.push({
      key,
      label: spec.label,
      ok: raw !== spec.bad,
      severity: spec.severity,
      hint: spec.hint,
    });
  }
  // Healthy flags are not worth screen space; only what is wrong is shown.
  return out
    .filter((flag) => !flag.ok)
    .sort((a, b) => (a.severity === b.severity ? 0 : a.severity === "error" ? -1 : 1));
}

/**
 * Whether a frequency response is sitting on the numerical gain floor.
 *
 * An AC run with no `ac_drives` has no stimulus. It does not error — it solves
 * and returns a response of essentially zero, which prints as roughly −180 dB
 * across the whole sweep. That reads as a catastrophically broken amplifier and
 * sends the reader to the devices, when the circuit was never driven.
 *
 * −120 dB is the threshold: no real amplifier this codebase builds sits there,
 * and the floor itself is far below it.
 */
export function responseIsAtGainFloor(result: unknown): boolean {
  if (!isRecord(result)) return false;
  const peak = result.peak_dB ?? result.Av_dc_dB;
  return typeof peak === "number" && Number.isFinite(peak) && peak < -120;
}

/** The shared explanation for a sweep whose AC measurement had no stimulus. */
export const NO_STIMULUS_HINT =
  "Every metric here is derived from an AC measurement that had no stimulus, so "
  + "none of them describe the circuit: the gain is the numerical floor, and the "
  + "input-referred noise is the output noise divided by that floor. Add an "
  + "`ac_drives` entry naming the source and its magnitude. A chopper or "
  + "switched-capacitor testbench driven only through its `periodic` block has "
  + "no AC path by construction — sweep it with PSS/PAC/PNoise instead.";

/**
 * Whether every trace in a time-domain result is constant.
 *
 * A transient with no stimulus solves cleanly and returns a flat line — for a
 * balanced amplifier, an output of exactly zero. That plot looks like a dead
 * circuit, and the reflex is to go looking at devices. It is the time-domain
 * twin of the documented AC trap where a missing `ac_drives` returns the
 * −180 dB gain floor instead of an error. Naming it costs one comparison.
 */
export function timeDomainIsFlat(result: unknown): boolean {
  if (!isRecord(result)) return false;
  const t = result.t;
  if (!Array.isArray(t) || t.length < 2) return false;

  const traces: unknown[] = [result.output, result.vout];
  if (isRecord(result.nodes)) traces.push(...Object.values(result.nodes));

  let seen = false;
  for (const trace of traces) {
    if (!Array.isArray(trace) || trace.length < 2) continue;
    seen = true;
    const first = trace[0];
    if (typeof first !== "number") continue;
    // A span wider than a nanovolt is deliberate signal, not solver noise.
    if (trace.some((v) => typeof v === "number" && Math.abs(v - first) > 1e-9)) {
      return false;
    }
  }
  return seen;
}

/**
 * Whether a bandwidth sits at the top of its own frequency sweep.
 *
 * `bw_Hz` is found by scanning the swept response, so a circuit whose corner is
 * above the sweep reports the last grid point. That number reads exactly like a
 * measurement — it is the failure that made a 79 kHz amplifier look like a
 * 10 kHz one — so the view flags it rather than printing it plainly.
 */
export function bandwidthIsGridLimited(result: unknown): boolean {
  if (!isRecord(result)) return false;
  const bw = result.bw_Hz;
  const freqs = result.freqs;
  if (typeof bw !== "number" || !Array.isArray(freqs) || freqs.length === 0) {
    return false;
  }
  const top = freqs[freqs.length - 1];
  if (typeof top !== "number") return false;
  return bw >= top * 0.999;
}
