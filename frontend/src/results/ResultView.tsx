import ReactECharts from "echarts-for-react";
import { buildPlot, extractMetrics } from "./transform";

interface ResultViewProps {
  name: string;
  result: unknown;
}

function JsonNode({ label, value }: { label?: string; value: unknown }) {
  if (Array.isArray(value)) {
    const preview = value.length > 8 ? `${value.length} items` : JSON.stringify(value);
    return (
      <details className="jt-node">
        <summary className="jt-summary">
          {label && <span className="jt-key">{label}: </span>}
          [{preview}]
        </summary>
        <div className="jt-children">
          {value.map((item, index) => (
            <JsonNode key={index} label={String(index)} value={item} />
          ))}
        </div>
      </details>
    );
  }

  if (typeof value === "object" && value !== null) {
    const entries = Object.entries(value);
    return (
      <details className="jt-node">
        <summary className="jt-summary">
          {label && <span className="jt-key">{label}: </span>}
          {`{${entries.length} fields}`}
        </summary>
        <div className="jt-children">
          {entries.map(([key, item]) => (
            <JsonNode key={key} label={key} value={item} />
          ))}
        </div>
      </details>
    );
  }

  return (
    <div className="jt-leaf">
      {label && <span className="jt-key">{label}: </span>}
      <span className="jt-val">{JSON.stringify(value)}</span>
    </div>
  );
}

export function ResultView({ name, result }: ResultViewProps) {
  const metrics = extractMetrics(result);
  const plot = buildPlot(result);
  const option = plot
    ? {
        animation: false,
        tooltip: { trigger: "axis" },
        legend: { type: "scroll", top: 0 },
        grid: { left: 62, right: 18, top: 36, bottom: 48 },
        xAxis: {
          type: plot.xLog ? "log" : "value",
          name: plot.xLabel,
          nameLocation: "middle",
          nameGap: 30,
        },
        yAxis: {
          type: plot.yLog ? "log" : "value",
          name: plot.yLabel,
          nameLocation: "middle",
          nameGap: 46,
        },
        series: plot.series.map((series) => ({
          name: series.name,
          type: "line",
          symbol: "none",
          data: plot.x.map((x, index) => [x, series.values[index]]),
        })),
      }
    : null;

  return (
    <>
      {metrics.length > 0 && (
        <div className="metric-row">
          {metrics.map((metric) => (
            <div className="metric-card" key={metric.key}>
              <div className="metric-key">{metric.key}</div>
              <div className="metric-val">{metric.value}</div>
            </div>
          ))}
        </div>
      )}
      {option && (
        <ReactECharts
          option={option}
          style={{ width: "100%", height: 280 }}
          opts={{ renderer: "canvas" }}
          aria-label={`${name} result plot`}
        />
      )}
      <div className="jt-wrap">
        <JsonNode value={result} />
      </div>
    </>
  );
}
