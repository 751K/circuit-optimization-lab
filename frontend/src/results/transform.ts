/**
 * Turn a solver result into plot specs.
 *
 * One analysis can warrant more than one plot — an AC run is a magnitude *and* a
 * phase curve, and squeezing both onto one axis makes neither readable — so this
 * returns a list. Each spec is axis-complete (labels, log flags, units) because
 * the chart component should not have to know what analysis produced it.
 */

export interface PlotSeries {
  name: string;
  values: number[];
  /** Draw on the secondary y-axis (phase against magnitude). */
  secondary?: boolean;
}

export interface PlotMarker {
  /** x position in data coordinates. */
  x: number;
  label: string;
}

export interface PlotSpec {
  /** Short title when a result yields several plots. */
  title?: string;
  x: number[];
  xLabel: string;
  xLog: boolean;
  yLabel: string;
  yLog: boolean;
  /** Right-hand axis label, present only when a series sets `secondary`. */
  y2Label?: string;
  series: PlotSeries[];
  markers?: PlotMarker[];
  /** "bar" for binned counts, where the gap to zero is meaningful. */
  kind?: "line" | "bar";
  /**
   * Force the y-axis to include zero. Correct for counts (a bar whose baseline
   * is not zero misstates its own height) and wrong for a measured quantity,
   * where clipping to the data range is what makes a small variation visible.
   */
  yFromZero?: boolean;
}

type JsonRecord = Record<string, unknown>;

function isRecord(value: unknown): value is JsonRecord {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function numberArray(value: unknown, length?: number): number[] | null {
  if (!Array.isArray(value) || value.length === 0) return null;
  const out: number[] = [];
  for (const item of value) {
    // A non-finite sample is a hole in the trace, not a reason to drop the whole
    // series: NaN renders as a gap in the line, which is the honest picture.
    out.push(typeof item === "number" ? item : Number.NaN);
  }
  return length === undefined || out.length === length ? out : null;
}

interface Complex { re: number; im: number }

function complexArray(value: unknown, length: number): Complex[] | null {
  if (!Array.isArray(value) || value.length !== length) return null;
  const out: Complex[] = [];
  for (const item of value) {
    if (!isRecord(item)) return null;
    const re = item.re;
    const im = item.im;
    if (typeof re !== "number" || typeof im !== "number") return null;
    out.push({ re, im });
  }
  return out;
}

function dB(magnitude: number): number {
  return 20 * Math.log10(Math.max(magnitude, 1e-300));
}

/** Unwrapped phase in degrees, so a Bode plot does not jump by 360°. */
function unwrappedPhaseDeg(response: Complex[]): number[] {
  const out: number[] = [];
  let turns = 0;
  let previous = 0;
  response.forEach((c, index) => {
    const raw = (Math.atan2(c.im, c.re) * 180) / Math.PI;
    if (index > 0) {
      const delta = raw - previous;
      if (delta > 180) turns -= 1;
      else if (delta < -180) turns += 1;
    }
    previous = raw;
    out.push(raw + turns * 360);
  });
  return out;
}

function positive(values: number[]): boolean {
  return values.every((value) => Number.isFinite(value) && value > 0);
}

/** Linear interpolation of the first crossing of `level`, or null. */
function firstCrossing(x: number[], y: number[], level: number): number | null {
  for (let i = 1; i < y.length; i += 1) {
    const a = y[i - 1];
    const b = y[i];
    const xa = x[i - 1];
    const xb = x[i];
    if (a === undefined || b === undefined || xa === undefined || xb === undefined) {
      continue;
    }
    if (!Number.isFinite(a) || !Number.isFinite(b)) continue;
    if ((a - level) * (b - level) <= 0 && a !== b) {
      const t = (level - a) / (b - a);
      return xa + t * (xb - xa);
    }
  }
  return null;
}

/**
 * AC / PAC: magnitude in dB plus unwrapped phase, with the unity-gain crossing
 * marked. Unity gain is where a feedback loop's phase margin is read, so its
 * position is worth a marker rather than a squint.
 */
function frequencyResponsePlots(result: JsonRecord): PlotSpec[] {
  const x = numberArray(result.freqs);
  if (!x) return [];

  const response = complexArray(result.response, x.length);
  const gains = numberArray(result.gains, x.length)
    ?? numberArray(result.Hmag, x.length);

  let magnitude: number[] | null = null;
  if (gains) magnitude = gains.map(dB);
  else if (response) magnitude = response.map((c) => dB(Math.hypot(c.re, c.im)));
  if (!magnitude) return [];

  const series: PlotSeries[] = [{ name: "Magnitude", values: magnitude }];
  let y2Label: string | undefined;
  if (response) {
    series.push({
      name: "Phase",
      values: unwrappedPhaseDeg(response),
      secondary: true,
    });
    y2Label = "Phase (deg)";
  }

  const markers: PlotMarker[] = [];
  const unity = firstCrossing(x, magnitude, 0);
  if (unity !== null) markers.push({ x: unity, label: "unity gain" });

  return [{
    title: "Frequency response",
    x,
    xLabel: "Frequency (Hz)",
    xLog: positive(x),
    yLabel: "Magnitude (dB)",
    yLog: false,
    y2Label,
    series,
    markers,
  }];
}

/** Noise / PNoise: the input- and output-referred spectral densities. */
function noisePlots(result: JsonRecord): PlotSpec[] {
  const x = numberArray(result.freqs);
  if (!x) return [];

  const series: PlotSeries[] = [];
  const outPsd = numberArray(result.out_psd, x.length);
  const irnPsd = numberArray(result.irn_psd, x.length);
  if (irnPsd) series.push({ name: "Input-referred", values: irnPsd });
  if (outPsd) series.push({ name: "Output", values: outPsd });
  if (series.length === 0) return [];

  return [{
    title: "Noise spectral density",
    x,
    xLabel: "Frequency (Hz)",
    xLog: positive(x),
    yLabel: "PSD (V²/Hz)",
    yLog: series.every((item) => positive(item.values)),
    series,
  }];
}

/**
 * Transient / PSS: every solved node against time.
 *
 * The designated output goes first so it holds the first colour, then the
 * internal nodes. All of them are included — hiding nodes is what the trace
 * picker in the view is for, and a node that is missing entirely cannot be
 * asked about.
 */
function timePlots(result: JsonRecord): PlotSpec[] {
  const x = numberArray(result.t);
  if (!x) return [];

  const series: PlotSeries[] = [];
  const output = numberArray(result.output, x.length)
    ?? numberArray(result.vout, x.length);
  if (output) series.push({ name: "output", values: output });

  if (isRecord(result.nodes)) {
    for (const [name, value] of Object.entries(result.nodes)) {
      const values = numberArray(value, x.length);
      if (values) series.push({ name, values });
    }
  }
  if (series.length === 0) return [];

  return [{
    title: "Waveforms",
    x,
    xLabel: "Time (s)",
    xLog: false,
    yLabel: "Voltage (V)",
    yLog: false,
    series,
  }];
}

/**
 * PSS convergence history: the shooting residual per iteration. A PSS run that
 * converged and one that stalled look identical in the final numbers and utterly
 * different here.
 */
function shootingPlots(result: JsonRecord): PlotSpec[] {
  const history = numberArray(result.shooting_history);
  if (!history || history.length < 2) return [];
  return [{
    title: "Shooting convergence",
    x: history.map((_, index) => index + 1),
    xLabel: "Iteration",
    xLog: false,
    yLabel: "Residual norm",
    yLog: positive(history),
    series: [{ name: "residual", values: history }],
  }];
}

/** Every plot worth drawing for one analysis result, in reading order. */
export function buildPlots(analysis: string, result: unknown): PlotSpec[] {
  if (!isRecord(result)) return [];
  switch (analysis) {
    case "ac":
    case "pac":
      return frequencyResponsePlots(result);
    case "noise":
    case "pnoise":
      return noisePlots(result);
    case "transient":
      return timePlots(result);
    case "pss":
      return [...timePlots(result), ...shootingPlots(result)];
    default:
      // An analysis this module has not been taught still gets whichever shape
      // its arrays match, rather than nothing.
      return [
        ...frequencyResponsePlots(result),
        ...noisePlots(result),
        ...timePlots(result),
      ];
  }
}
