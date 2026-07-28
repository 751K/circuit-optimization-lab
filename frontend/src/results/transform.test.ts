import { buildPlots, type PlotSpec } from "./transform";
import {
  bandwidthIsGridLimited,
  extractHealth,
  extractMetrics,
  responseIsAtGainFloor,
  timeDomainIsFlat,
} from "./metrics";
import { formatDuration, formatPercent, formatQuantity, formatValue } from "./format";
import { histogram } from "./McView";

/** The single plot a result was expected to produce. Fails loudly otherwise. */
function only(plots: PlotSpec[]): PlotSpec {
  expect(plots).toHaveLength(1);
  return plots[0] as PlotSpec;
}

/** Indexed access under `noUncheckedIndexedAccess`, asserting presence. */
function at<T>(items: T[], index: number): T {
  const item = items[index];
  expect(item).toBeDefined();
  return item as T;
}

describe("plot building", () => {
  it("plots AC magnitude in dB against a log frequency axis", () => {
    const plot = only(buildPlots("ac", { freqs: [1, 10, 100], gains: [10, 2, 1] }));

    expect(plot.xLog).toBe(true);
    expect(plot.yLabel).toBe("Magnitude (dB)");
    expect(at(at(plot.series, 0).values, 0)).toBeCloseTo(20);
  });

  it("adds an unwrapped phase trace on a second axis when the response is complex", () => {
    const plot = only(buildPlots("ac", {
      freqs: [1, 10, 100],
      response: [
        { re: 10, im: 0 },
        { re: 0, im: -5 },
        { re: -1, im: -0.1 },
      ],
    }));

    const phase = plot.series.find((series) => series.name === "Phase");
    expect(phase?.secondary).toBe(true);
    expect(plot.y2Label).toBe("Phase (deg)");
    // -90 deg then past -180: unwrapping keeps it descending rather than
    // snapping to +180, which would draw a cliff that is not in the circuit.
    expect(at(phase!.values, 1)).toBeCloseTo(-90);
    expect(at(phase!.values, 2)).toBeLessThan(-90);
  });

  it("marks the unity-gain crossing", () => {
    // 20 dB down to -20 dB: 0 dB is crossed between the middle two points.
    const plot = only(buildPlots("ac", {
      freqs: [1, 10, 100, 1000],
      gains: [10, 2, 0.5, 0.1],
    }));

    const unity = plot.markers?.find((marker) => marker.label === "unity gain");
    expect(unity).toBeDefined();
    expect(unity!.x).toBeGreaterThan(10);
    expect(unity!.x).toBeLessThan(100);
  });

  it("keeps a non-finite sample as a gap instead of dropping the trace", () => {
    // A single failed step must not erase the whole waveform: the rest of the
    // trace is what tells you where it failed.
    const plot = only(buildPlots("transient", {
      t: [0, 1e-6, 2e-6],
      output: [0.1, Number.NaN, 0.3],
    }));

    const values = at(plot.series, 0).values;
    expect(values).toHaveLength(3);
    expect(Number.isNaN(at(values, 1))).toBe(true);
    expect(at(values, 2)).toBe(0.3);
  });

  it("plots every solved node with the designated output first", () => {
    const plot = only(buildPlots("transient", {
      t: [0, 1e-6],
      output: [0.1, 0.2],
      nodes: { OUTP: [0.5, 0.6], OUTN: [0.4, 0.3] },
    }));

    expect(plot.series.map((series) => series.name)).toEqual(["output", "OUTP", "OUTN"]);
  });

  it("gives PSS both its waveform and its shooting-convergence history", () => {
    const plots = buildPlots("pss", {
      t: [0, 1e-6],
      output: [0.1, 0.2],
      shooting_history: [1e-2, 1e-5, 1e-9],
    });

    expect(plots.map((plot) => plot.title)).toEqual([
      "Waveforms",
      "Shooting convergence",
    ]);
    expect(at(plots, 1).yLog).toBe(true);
  });

  it("uses a linear noise axis when a PSD touches zero", () => {
    // A log axis silently drops non-positive points, which would hide a null.
    const plot = only(buildPlots("noise", { freqs: [1, 10], out_psd: [1e-18, 0] }));

    expect(plot.yLog).toBe(false);
  });
});

describe("mismatch histogram", () => {
  it("bins samples across the observed range, not from zero", () => {
    // A mismatch spread is a narrow band far from the origin (63.71–63.77 dB).
    // Binning or plotting from zero collapses the whole distribution into one
    // pixel and hides exactly what the run was for.
    const values = Array.from({ length: 64 }, (_, i) => 63.71 + (i / 63) * 0.06);
    const plot = histogram(values, 8);

    expect(plot).not.toBeNull();
    expect(at(plot!.x, 0)).toBeGreaterThan(63.7);
    expect(at(plot!.x, plot!.x.length - 1)).toBeLessThan(63.78);
    // Bars, counted from zero — a bar whose baseline floats misstates its height.
    expect(plot!.kind).toBe("bar");
    expect(plot!.yFromZero).toBe(true);
  });

  it("puts every sample in a bin, including the maximum", () => {
    // The largest sample sits exactly on the top edge; a naive floor() drops it
    // into a bin one past the end and it vanishes from the count.
    const values = [1, 2, 3, 4, 5];
    const plot = histogram(values, 4);
    const counts = at(plot!.series, 0).values;

    expect(counts.reduce((sum, count) => sum + count, 0)).toBe(values.length);
    expect(at(counts, counts.length - 1)).toBeGreaterThan(0);
  });

  it("declines to plot a degenerate distribution rather than drawing a fake one", () => {
    expect(histogram([2.5, 2.5, 2.5])).toBeNull();
    expect(histogram([1])).toBeNull();
  });
});

describe("metric extraction", () => {
  it("labels metrics and rescales band noise out of microvolts", () => {
    const metrics = extractMetrics("noise", { irn_uV_band: 84.2, out_uV_band: 1200 });

    expect(metrics.map((metric) => metric.label)).toEqual([
      "Input noise",
      "Output noise",
    ]);
    // Reported in µV; carried as volts so the formatter picks its own prefix.
    const first = at(metrics, 0);
    expect(first.value).toBeCloseTo(84.2e-6);
    expect(formatValue(first.value, first.unit)).toBe("84.2 µV");
  });

  it("counts transient steps from the time grid, not from nsubsteps", () => {
    // `nsubsteps` counts retry subdivisions and is 0 on a clean run. Showing it
    // as "Steps: 0" next to a solved 801-point waveform is the same class of
    // mistake the CLI made when it printed the node count as the step count.
    const metrics = extractMetrics("transient", {
      t: [0, 1e-6, 2e-6, 3e-6],
      nsubsteps: 0,
      nfail: 0,
    });
    const steps = metrics.find((metric) => metric.key === "step_count");

    expect(steps?.label).toBe("Steps");
    expect(steps?.value).toBe(4);
    // nsubsteps still appears, under a name that says what it counts.
    expect(metrics.find((m) => m.key === "nsubsteps")?.label)
      .toBe("Retry subdivisions");
  });

  it("drops non-finite metrics rather than printing NaN beside real ones", () => {
    const metrics = extractMetrics("ac", { Av_dc_dB: 40.2, bw_Hz: Number.NaN });

    expect(metrics.map((metric) => metric.key)).toEqual(["Av_dc_dB"]);
  });

  it("ignores solver telemetry that is not a measurement", () => {
    const metrics = extractMetrics("pnoise", {
      irn_uV_band: 5,
      pnoise_hb_sparse_nnz: 4096,
      pnoise_hb_solve_count: 12,
    });

    expect(metrics.map((metric) => metric.key)).toEqual(["irn_uV_band"]);
  });

  it("surfaces only the health flags that are wrong, worst first", () => {
    const flags = extractHealth({
      converged: false,      // error
      diverged: false,       // fine
      pnoise_degraded: true, // warn
    });

    expect(flags.map((flag) => flag.key)).toEqual(["converged", "pnoise_degraded"]);
    expect(at(flags, 0).severity).toBe("error");
  });

  it("reports nothing when a run is healthy", () => {
    expect(extractHealth({ converged: true, diverged: false })).toEqual([]);
  });

  it("detects an AC run sitting on the gain floor", () => {
    // No ac_drives means no stimulus. The solve succeeds and returns ~-180 dB,
    // which reads as a catastrophically broken amplifier.
    expect(responseIsAtGainFloor({ peak_dB: -180.2, Av_dc_dB: -180.2 })).toBe(true);
    expect(responseIsAtGainFloor({ peak_dB: 63.77, Av_dc_dB: 40.2 })).toBe(false);
    // A genuinely heavy attenuator is still above the floor by a wide margin.
    expect(responseIsAtGainFloor({ peak_dB: -60 })).toBe(false);
  });

  it("detects a transient with no stimulus", () => {
    // A balanced amplifier with no drive solves cleanly to exactly zero. That
    // plot reads as a dead circuit; it is a missing stimulus.
    expect(timeDomainIsFlat({
      t: [0, 1e-6, 2e-6],
      output: [0, 0, 0],
      nodes: { OUTP: [0.9, 0.9, 0.9] },
    })).toBe(true);

    // One trace moving is enough to make it a real response.
    expect(timeDomainIsFlat({
      t: [0, 1e-6, 2e-6],
      output: [0, 0, 0],
      nodes: { OUTP: [0.9, 0.95, 1.0] },
    })).toBe(false);

    // Not a time-domain result at all.
    expect(timeDomainIsFlat({ freqs: [1, 10], gains: [1, 1] })).toBe(false);
  });

  it("detects a bandwidth pinned at the top of its own sweep", () => {
    // The failure that made a 79 kHz amplifier read as a 10 kHz one.
    expect(bandwidthIsGridLimited({ freqs: [1, 100, 10000], bw_Hz: 10000 })).toBe(true);
    expect(bandwidthIsGridLimited({ freqs: [1, 100, 10000], bw_Hz: 79 })).toBe(false);
  });
});

describe("engineering formatting", () => {
  it("picks the SI prefix that keeps the mantissa readable", () => {
    expect(formatValue(79034.5, "Hz")).toBe("79.03 kHz");
    expect(formatValue(1.018e8, "Hz")).toBe("101.8 MHz");
    expect(formatValue(8.386e-5, "V")).toBe("83.86 µV");
    expect(formatValue(6.67e-5, "W")).toBe("66.7 µW");
    expect(formatValue(3.9e-15, "F")).toBe("3.9 fF");
  });

  it("never prefixes dB, degrees or bare counts", () => {
    expect(formatValue(-40.229, "dB")).toBe("-40.23 dB");
    expect(formatValue(2001, "")).toBe("2001");
    expect(formatValue(-135.4, "deg")).toBe("-135.4 deg");
  });

  it("renders a non-measurement as a dash, never as a number", () => {
    expect(formatValue(Number.NaN, "Hz")).toBe("— (NaN)");
    expect(formatValue(Number.POSITIVE_INFINITY, "Hz")).toBe("— (infinite)");
    expect(formatValue(null, "Hz")).toBe("— (not measured)");
  });

  it("keeps zero unscaled", () => {
    expect(formatValue(0, "V")).toBe("0 V");
  });

  it("splits value from unit for column alignment", () => {
    expect(formatQuantity(79034.5, "Hz")).toEqual({
      value: "79.03",
      unit: "kHz",
      text: "79.03 kHz",
    });
  });

  it("shows sub-second runs in milliseconds", () => {
    expect(formatDuration(0.0043)).toBe("4.30 ms");
    expect(formatDuration(12.5)).toBe("12.5 s");
    expect(formatDuration(135)).toBe("2 min 15 s");
  });

  it("signs a margin percentage", () => {
    expect(formatPercent(0.152)).toBe("+15.2%");
    expect(formatPercent(-0.031)).toBe("-3.1%");
    expect(formatPercent(null)).toBe("—");
  });
});
