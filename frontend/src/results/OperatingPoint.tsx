/**
 * The DC operating point: the first thing anyone asks of an amplifier.
 *
 * Every AC result already carries this — `dc_op` (node voltages),
 * `operating_regions` (per-device saturation with Vds/Vdsat/headroom), `ss`
 * (small-signal gm/gds/gmb/caps and channel current), `source_power`, and
 * `device_bindings` (which model bin each geometry actually selected). None of it
 * was being displayed.
 *
 * Two derived columns are what make the table a *design* table rather than a
 * dump:
 *  - **gm/Id** — transconductance efficiency, the knob that trades current for
 *    gain. It is how a device's operating region is chosen in the first place.
 *  - **gm/gds** — intrinsic gain, the per-stage ceiling.
 * Both come from `ss`, so neither costs an extra solve.
 *
 * A device out of saturation is called out in red. That single fact explains
 * more failed amplifiers than any other, and it is invisible in a gain number.
 */
import { formatValue } from "./format";

interface Region {
  status: string;
  saturated: boolean | null;
  vds_v?: number;
  vdsat_v?: number;
  headroom_v?: number;
}

interface SmallSignal {
  gm?: number;
  gds?: number;
  gmb?: number;
  Cgs?: number;
  Cgd?: number;
  Ich?: number;
}

interface Binding {
  pdk?: string;
  model?: string;
  section?: string;
  bin?: string;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function numbers(value: unknown): Record<string, number> {
  const out: Record<string, number> = {};
  if (!isRecord(value)) return out;
  for (const [key, item] of Object.entries(value)) {
    if (typeof item === "number") out[key] = item;
  }
  return out;
}

/** True when this result carries an operating point worth rendering. */
export function hasOperatingPoint(result: unknown): boolean {
  if (!isRecord(result)) return false;
  return isRecord(result.operating_regions) || isRecord(result.dc_op);
}

function ratio(a: number | undefined, b: number | undefined): number | null {
  if (typeof a !== "number" || typeof b !== "number") return null;
  if (!Number.isFinite(a) || !Number.isFinite(b) || b === 0) return null;
  return Math.abs(a / b);
}

export function OperatingPoint({ result }: { result: unknown }) {
  if (!isRecord(result)) return null;

  const regions = isRecord(result.operating_regions)
    ? (result.operating_regions as Record<string, Region>)
    : {};
  const ss = isRecord(result.ss) ? (result.ss as Record<string, SmallSignal>) : {};
  const bindings = isRecord(result.device_bindings)
    ? (result.device_bindings as Record<string, Binding>)
    : {};
  const nodes = numbers(result.dc_op);
  const power = isRecord(result.source_power) ? result.source_power : null;

  const deviceNames = Array.from(
    new Set([...Object.keys(regions), ...Object.keys(ss)]),
  );
  const unsaturated = deviceNames.filter((d) => regions[d]?.saturated === false);
  const totalW = power && typeof power.total_w === "number" ? power.total_w : null;
  const currents = numbers(power?.source_currents_a);
  const voltages = numbers(power?.source_voltages_v);
  const perSource = numbers(power?.per_source_w);

  return (
    <div className="op-view">
      {unsaturated.length > 0 && (
        <div className="op-alert">
          <strong>{unsaturated.length} device{unsaturated.length > 1 ? "s" : ""} out of
          saturation:</strong> {unsaturated.join(", ")}. Gain and bandwidth measured
          here describe a circuit biased into the triode region.
        </div>
      )}

      {deviceNames.length > 0 && (
        <>
          <h4 className="op-head">Devices</h4>
          <div className="table-scroll">
            <table className="dtable">
              <thead>
                <tr>
                  <th>Device</th>
                  <th>Region</th>
                  <th className="num" title="Drain-source voltage">V<sub>ds</sub></th>
                  <th className="num" title="Saturation voltage">V<sub>dsat</sub></th>
                  <th className="num" title="|Vds| − |Vdsat|; negative means triode">Headroom</th>
                  <th className="num" title="Channel current">I<sub>d</sub></th>
                  <th className="num" title="Transconductance">g<sub>m</sub></th>
                  <th className="num" title="Transconductance efficiency — the current-for-gain trade">
                    g<sub>m</sub>/I<sub>d</sub>
                  </th>
                  <th className="num" title="Intrinsic gain available from this device">
                    g<sub>m</sub>/g<sub>ds</sub>
                  </th>
                  <th title="Model bin the geometry selected">Bin</th>
                </tr>
              </thead>
              <tbody>
                {deviceNames.map((name) => {
                  const region = regions[name];
                  const small = ss[name] ?? {};
                  const saturated = region?.saturated;
                  const gmId = ratio(small.gm, small.Ich);
                  const gmGds = ratio(small.gm, small.gds);
                  return (
                    <tr key={name} className={saturated === false ? "row-bad" : ""}>
                      <td className="mono">{name}</td>
                      <td>
                        {saturated === true && <span className="pill ok">sat</span>}
                        {saturated === false && <span className="pill bad">triode</span>}
                        {saturated === null || saturated === undefined ? (
                          <span className="pill muted" title={region?.status}>
                            {region?.status ?? "—"}
                          </span>
                        ) : null}
                      </td>
                      <td className="num">{formatValue(region?.vds_v, "V", 3)}</td>
                      <td className="num">{formatValue(region?.vdsat_v, "V", 3)}</td>
                      <td className="num">{formatValue(region?.headroom_v, "V", 3)}</td>
                      <td className="num">{formatValue(small.Ich, "A", 3)}</td>
                      <td className="num">{formatValue(small.gm, "S", 3)}</td>
                      <td className="num">
                        {gmId === null ? "—" : `${gmId.toPrecision(3)} 1/V`}
                      </td>
                      <td className="num">
                        {gmGds === null ? "—" : gmGds.toPrecision(3)}
                      </td>
                      <td className="mono tiny" title={bindings[name]?.bin}>
                        {bindings[name]?.section ?? "—"}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </>
      )}

      <div className="op-columns">
        {Object.keys(nodes).length > 0 && (
          <div className="op-col">
            <h4 className="op-head">Node voltages</h4>
            <div className="table-scroll short">
              <table className="dtable">
                <tbody>
                  {Object.entries(nodes).map(([node, value]) => (
                    <tr key={node}>
                      <td className="mono">{node}</td>
                      <td className="num">{formatValue(value, "V", 4)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {power && (
          <div className="op-col">
            <h4 className="op-head">
              Supplies{totalW !== null && ` — ${formatValue(totalW, "W", 4)} total`}
            </h4>
            <div className="table-scroll short">
              <table className="dtable">
                <thead>
                  <tr><th>Source</th><th className="num">V</th><th className="num">I</th><th className="num">P</th></tr>
                </thead>
                <tbody>
                  {Object.keys(currents).map((name) => (
                    <tr key={name}>
                      <td className="mono">{name}</td>
                      <td className="num">{formatValue(voltages[name], "V", 3)}</td>
                      <td className="num">{formatValue(currents[name], "A", 3)}</td>
                      <td className="num">{formatValue(perSource[name], "W", 3)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
