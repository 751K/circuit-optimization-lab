/**
 * ECharts wrapper for a {@link PlotSpec}.
 *
 * Traces are toggleable through the legend, which is the whole reason the
 * transform emits every solved node rather than a truncated set: a 30-node
 * transient is unreadable all at once and useless if the node you need was
 * dropped upstream. Legend selection answers both.
 */
import { useMemo } from "react";
import ReactECharts from "echarts-for-react";
import type { PlotSpec } from "./transform";

/**
 * Axis tick label for a value that may span many decades.
 *
 * ECharts writes a noise PSD tick of 1e-18 as "0.000000000000000001", which is
 * unreadable and pushes the axis area across the plot. Anything outside a
 * comfortable reading range becomes `1e-18`; values inside it keep their plain
 * form, because "1000" reads better than "1e3".
 */
export function axisTick(value: number): string {
  if (!Number.isFinite(value)) return "";
  if (value === 0) return "0";
  const magnitude = Math.abs(value);
  if (magnitude >= 1e-3 && magnitude < 1e5) {
    // Trim the float noise a log axis produces (0.30000000000000004).
    return String(Number(value.toPrecision(6)));
  }
  const exponent = Math.floor(Math.log10(magnitude));
  const mantissa = value / Math.pow(10, exponent);
  const shown = Number(mantissa.toPrecision(3));
  return shown === 1 ? `1e${exponent}`
    : shown === -1 ? `-1e${exponent}`
    : `${shown}e${exponent}`;
}

/**
 * Traces shown by default. Beyond this the plot is drawn with only the first few
 * enabled and the rest one legend click away — a wall of overlapping lines
 * conveys less than four.
 */
const DEFAULT_VISIBLE = 6;

export function Chart({ plot, height = 300 }: { plot: PlotSpec; height?: number }) {
  const option = useMemo(() => {
    const hasSecondary = plot.series.some((series) => series.secondary);
    const selected: Record<string, boolean> = {};
    plot.series.forEach((series, index) => {
      selected[series.name] = index < DEFAULT_VISIBLE;
    });

    const markLine = plot.markers?.length
      ? {
          silent: true,
          symbol: "none",
          lineStyle: { type: "dashed" as const, color: "#94a3b8", width: 1 },
          label: {
            formatter: (params: { name?: string }) => params.name ?? "",
            fontSize: 10,
            color: "#64748b",
            position: "insideEndTop" as const,
          },
          data: plot.markers.map((marker) => ({ xAxis: marker.x, name: marker.label })),
        }
      : undefined;

    return {
      animation: false,
      tooltip: {
        trigger: "axis",
        axisPointer: { type: "cross" },
        confine: true,
        textStyle: { fontSize: 11 },
        valueFormatter: (v: unknown) =>
          typeof v === "number" ? axisTick(v) : String(v),
      },
      legend: {
        type: "scroll",
        top: 0,
        itemWidth: 16,
        itemHeight: 8,
        textStyle: { fontSize: 11 },
        selected,
      },
      grid: {
        left: 66,
        right: hasSecondary ? 62 : 22,
        top: 30,
        bottom: 44,
      },
      xAxis: {
        type: plot.xLog ? "log" : "value",
        name: plot.xLabel,
        nameLocation: "middle",
        nameGap: 26,
        nameTextStyle: { fontSize: 11 },
        axisLabel: { fontSize: 10, formatter: axisTick },
        // A *value* axis includes zero by default, which collapses data
        // clustered far from it — a mismatch histogram spanning 63.71–63.77 dB
        // — into one pixel at the right edge. A log axis never includes zero
        // and rejects `scale`, so it must not be set there.
        ...(plot.xLog ? {} : { scale: true }),
      },
      yAxis: [
        {
          type: plot.yLog ? "log" : "value",
          name: plot.yLabel,
          nameLocation: "middle",
          nameGap: 48,
          nameTextStyle: { fontSize: 11 },
          axisLabel: { fontSize: 10, formatter: axisTick },
          scale: !plot.yFromZero,
        },
        ...(hasSecondary
          ? [{
              type: "value" as const,
              name: plot.y2Label ?? "",
              nameLocation: "middle" as const,
              nameGap: 46,
              nameTextStyle: { fontSize: 11 },
              axisLabel: { fontSize: 10, formatter: axisTick },
              splitLine: { show: false },
              scale: true,
            }]
          : []),
      ],
      series: plot.series.map((series, index) => ({
        name: series.name,
        type: plot.kind === "bar" ? "bar" : "line",
        ...(plot.kind === "bar"
          ? { barWidth: "90%", barCategoryGap: "5%" }
          : {
              symbol: "none",
              lineStyle: {
                width: series.secondary ? 1.2 : 1.6,
                type: series.secondary ? "dashed" : "solid",
              },
              // `connectNulls: false` is the point of keeping NaN samples: a gap
              // in the line is where the solve failed, and joining across it
              // would hide that.
              connectNulls: false,
            }),
        yAxisIndex: series.secondary ? 1 : 0,
        data: plot.x.map((x, i) => [x, series.values[i]]),
        ...(index === 0 && markLine ? { markLine } : {}),
      })),
    };
  }, [plot]);

  return (
    <div className="chart-block">
      {plot.title && <h4 className="chart-title">{plot.title}</h4>}
      <ReactECharts
        option={option}
        notMerge
        style={{ width: "100%", height }}
        opts={{ renderer: "canvas" }}
        aria-label={plot.title ?? "result plot"}
      />
    </div>
  );
}
