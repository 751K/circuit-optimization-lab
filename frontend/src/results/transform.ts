export interface PlotSeries {
  name: string;
  values: number[];
}

export interface PlotSpec {
  x: number[];
  xLabel: string;
  xLog: boolean;
  yLabel: string;
  yLog: boolean;
  series: PlotSeries[];
}

type JsonRecord = Record<string, unknown>;

function isRecord(value: unknown): value is JsonRecord {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function numberArray(value: unknown, length?: number): number[] | null {
  if (!Array.isArray(value) || value.length === 0) return null;
  const out: number[] = [];
  for (const item of value) {
    if (typeof item !== "number" || !Number.isFinite(item)) return null;
    out.push(item);
  }
  return length === undefined || out.length === length ? out : null;
}

function complexMagnitudeArray(value: unknown, length: number): number[] | null {
  if (!Array.isArray(value) || value.length !== length) return null;
  const out: number[] = [];
  for (const item of value) {
    if (!isRecord(item)) return null;
    const re = item.re;
    const im = item.im;
    if (typeof re !== "number" || typeof im !== "number") return null;
    out.push(Math.hypot(re, im));
  }
  return out;
}

function positive(values: number[]): boolean {
  return values.every((value) => value > 0);
}

function frequencyPlot(result: JsonRecord): PlotSpec | null {
  const x = numberArray(result.freqs);
  if (!x) return null;

  const series: PlotSeries[] = [];
  const outPsd = numberArray(result.out_psd, x.length);
  const irnPsd = numberArray(result.irn_psd, x.length);
  if (outPsd) series.push({ name: "Output PSD", values: outPsd });
  if (irnPsd) series.push({ name: "Input-referred PSD", values: irnPsd });

  const gains = numberArray(result.gains, x.length) ?? numberArray(result.Hmag, x.length);
  if (!outPsd && !irnPsd && gains) {
    series.push({
      name: "Gain",
      values: gains.map((value) => 20 * Math.log10(Math.max(value, 1e-300))),
    });
  } else if (!outPsd && !irnPsd) {
    const response = complexMagnitudeArray(result.response, x.length);
    if (response) {
      series.push({
        name: "Response",
        values: response.map((value) => 20 * Math.log10(Math.max(value, 1e-300))),
      });
    }
  }

  if (series.length === 0) return null;

  const isNoise = Boolean(outPsd || irnPsd);
  return {
    x,
    xLabel: "Frequency (Hz)",
    xLog: positive(x),
    yLabel: isNoise ? "Noise PSD (V^2/Hz)" : "Magnitude (dB)",
    yLog: isNoise && series.every((item) => positive(item.values)),
    series,
  };
}

function timePlot(result: JsonRecord): PlotSpec | null {
  const x = numberArray(result.t);
  if (!x) return null;

  const series: PlotSeries[] = [];
  const output = numberArray(result.output, x.length) ?? numberArray(result.vout, x.length);
  if (output) series.push({ name: "Output", values: output });

  if (isRecord(result.nodes)) {
    for (const [name, value] of Object.entries(result.nodes)) {
      const values = numberArray(value, x.length);
      if (values && series.length < 9) series.push({ name, values });
    }
  }
  if (series.length === 0) return null;

  return {
    x,
    xLabel: "Time (s)",
    xLog: false,
    yLabel: "Voltage (V)",
    yLog: false,
    series,
  };
}

export function buildPlot(result: unknown): PlotSpec | null {
  if (!isRecord(result)) return null;
  return frequencyPlot(result) ?? timePlot(result);
}

export interface Metric {
  key: string;
  value: string;
}

const METRIC_KEYS = [
  "Av_dc_dB",
  "peak_dB",
  "bw_Hz",
  "irn_uV_band",
  "out_uV_band",
  "residual_norm",
  "shooting_iters",
  "nfail",
  "nretry",
] as const;

function formatNumber(value: number): string {
  const abs = Math.abs(value);
  if (abs !== 0 && (abs >= 1e5 || abs < 1e-3)) return value.toExponential(4);
  return Number.isInteger(value) ? String(value) : value.toPrecision(6);
}

export function extractMetrics(result: unknown): Metric[] {
  if (!isRecord(result)) return [];
  const metrics: Metric[] = [];
  for (const key of METRIC_KEYS) {
    const value = result[key];
    if (typeof value === "number" && Number.isFinite(value)) {
      metrics.push({ key, value: formatNumber(value) });
    }
  }
  return metrics;
}
