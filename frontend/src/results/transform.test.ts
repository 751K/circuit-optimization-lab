import { buildPlot, extractMetrics } from "./transform";

describe("result transforms", () => {
  it("builds a dB frequency plot from AC gains", () => {
    const plot = buildPlot({
      freqs: [1, 10, 100],
      gains: [10, 2, 1],
    });

    expect(plot?.xLog).toBe(true);
    expect(plot?.yLabel).toBe("Magnitude (dB)");
    expect(plot?.series[0]?.values[0]).toBeCloseTo(20);
  });

  it("builds a transient plot with output and node traces", () => {
    const plot = buildPlot({
      t: [0, 1e-6],
      output: [0.1, 0.2],
      nodes: { OUTP: [0.5, 0.6] },
    });

    expect(plot?.xLabel).toBe("Time (s)");
    expect(plot?.series.map((series) => series.name)).toEqual(["Output", "OUTP"]);
  });

  it("extracts stable scalar metrics and ignores arrays", () => {
    expect(
      extractMetrics({
        Av_dc_dB: 81.25,
        bw_Hz: 1.2e8,
        gains: [1, 2],
      }),
    ).toEqual([
      { key: "Av_dc_dB", value: "81.2500" },
      { key: "bw_Hz", value: "1.2000e+8" },
    ]);
  });
});
