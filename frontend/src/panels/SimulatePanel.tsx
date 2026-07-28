/**
 * The Simulate section: pick analyses, pick a corner, run.
 *
 * Two things differ from a plain checkbox list.
 *
 * **The corner menu is per-circuit.** It comes from `/validate`, which reports
 * the corner set this circuit's model family admits, not from the server's
 * capabilities menu of every family. A circuit belongs to exactly one family and
 * the name spaces are disjoint, so a union menu would offer corners that cannot
 * resolve.
 *
 * **Transient is configurable inline.** It has no universal default sweep
 * (tstop depends entirely on the circuit), and previously that meant the only
 * way to run one was to hand-edit the JSON. Its config is a real document edit,
 * so it exports and undoes like any other change — unlike the AC/noise sweep
 * defaults, which are injected into the request only.
 */
import { useEffect, useMemo, useState } from "react";
import { useEditor, useSession } from "../store";
import { DEFAULT_SWEEP_LABEL, missingConfigMessage, prepareSolveCircuit } from "./runConfig";
import { NumberField } from "./fields";
import {
  buildStimulus,
  stimulusOptions,
  stimulusReport,
  type StimulusPort,
} from "./stimulus";
import { formatValue } from "../results/format";

/** Analyses in the order they are usually reached, with a one-line purpose. */
const HINTS: Record<string, string> = {
  ac: "Small-signal gain and bandwidth, plus the DC operating point",
  noise: "Input- and output-referred noise spectral density",
  transient: "Time-domain response; needs a stop time",
  pss: "Periodic steady state via shooting; needs a periodic block",
  pac: "Small-signal response about a periodic orbit",
  pnoise: "Cyclostationary noise about a periodic orbit",
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * Choose a source for an analysis the circuit does not excite.
 *
 * The options are built from the circuit's own rails, so every entry names a net
 * that exists and can legitimately be driven — the alternative to this menu is
 * hand-editing `input_drives` or a `periodic` block in the JSON, which is what
 * made a silent run the path of least resistance.
 */
function StimulusPicker({
  analysis,
  options,
  value,
  onPick,
  onApply,
}: {
  analysis: string;
  options: ReturnType<typeof stimulusOptions>;
  value: string | undefined;
  onPick: (id: string) => void;
  onApply: () => void;
}) {
  if (options.length === 0) {
    return (
      <p className="muted small">
        No rail in this circuit could take a stimulus — only a fixed-potential
        net can be driven, and the only rails here are the supplies. Add an input
        rail, or write the source into the circuit JSON by hand.
      </p>
    );
  }
  const selected = options.some((o) => o.id === value) ? value : options[0]!.id;
  return (
    <span className="stim-pick">
      <select
        value={selected}
        aria-label={`Stimulus source for ${analysis}`}
        onChange={(e) => onPick(e.target.value)}
      >
        {options.map((o) => (
          <option key={o.id} value={o.id}>{o.label}</option>
        ))}
      </select>
      <button className="btn tiny" onClick={onApply}>Use this source</button>
    </span>
  );
}

/** The resolved AC input ports, so "what drives this" is a net name, not a block name. */
function PortList({ ports }: { ports: StimulusPort[] }) {
  if (ports.length === 0) return null;
  return (
    <span className="stim-ports">
      {ports.map((p) => (
        <span className="stim-port" key={`${p.via}:${p.device ?? p.net}`}>
          <code>{p.net}</code>
          {p.magnitude >= 0 ? " +" : " −"}
          {formatValue(Math.abs(p.magnitude), "")}
          {p.device && <span className="muted"> ({p.device}.G)</span>}
        </span>
      ))}
    </span>
  );
}

export default function SimulatePanel() {
  const caps = useEditor((s) => s.caps);
  const rest = useEditor((s) => s.rest);
  const exportJson = useEditor((s) => s.exportJson);
  const circuitCorners = useEditor((s) => s.circuitCorners);
  const setAnalysisConfig = useEditor((s) => s.setAnalysisConfig);
  const nodeCount = useEditor((s) => s.graph.nodes.length);

  const runSolve = useSession((s) => s.runSolve);
  const running = useSession((s) => s.run.status === "running");

  const analysisKeys = caps ? Object.keys(caps.analyses) : [];
  const [selected, setSelected] = useState<string[]>(["ac"]);
  const [corner, setCorner] = useState("");
  const [blocked, setBlocked] = useState<string | null>(null);
  const [showTransient, setShowTransient] = useState(false);

  // A corner that is no longer offered (the circuit's family changed) must not
  // stay silently selected — it would be sent to a solver that cannot resolve it.
  useEffect(() => {
    if (corner && circuitCorners && !circuitCorners.includes(corner)) setCorner("");
  }, [circuitCorners, corner]);

  const configured = isRecord(rest.analyses) ? rest.analyses : {};
  const transientCfg = isRecord(configured.transient) ? configured.transient : null;

  // Stimulus is read from the exported circuit, not from `rest`: the AC drives
  // live on the devices, which only exist once the graph is serialised.
  const [stimFreq, setStimFreq] = useState(1e3);
  const [stimAmp, setStimAmp] = useState(1e-3);
  const circuit = useMemo(
    () => (nodeCount > 0 ? exportJson() : null),
    // exportJson reads the live graph; recompute whenever the document changes.
    [nodeCount, rest, exportJson],
  );
  const reports = useMemo(
    () => (circuit
      ? selected.map((name) => ({ name, ...stimulusReport(circuit, name) }))
      : []),
    [circuit, selected],
  );
  const setAcStimulus = useEditor((s) => s.setAcStimulus);
  // Which source the user picked, per analysis. Reset is not needed on circuit
  // change: an id that no longer resolves falls back to the first option.
  const [picked, setPicked] = useState<Record<string, string>>({});

  const applyStimulus = (analysis: string): void => {
    if (!circuit) return;
    const options = stimulusOptions(circuit, analysis);
    const option = options.find((o) => o.id === picked[analysis]) ?? options[0];
    if (!option) return;
    const patch = buildStimulus(circuit, option, analysis, {
      frequency: stimFreq,
      amplitude: stimAmp,
    });
    if (patch.target === "ac") {
      setAcStimulus(patch.inputDrives, patch.acDrives);
      return;
    }
    const prior = configured[patch.owner];
    const owner: Record<string, unknown> = isRecord(prior) ? prior : {};
    setAnalysisConfig(patch.owner, { ...owner, periodic: patch.periodic });
  };

  const toggle = (key: string): void =>
    setSelected((prev) =>
      prev.includes(key) ? prev.filter((k) => k !== key) : [...prev, key]);

  const run = (): void => {
    setBlocked(null);
    const prep = prepareSolveCircuit(exportJson(), selected);
    if (prep.missing.length > 0) {
      setBlocked(missingConfigMessage(prep.missing));
      return;
    }
    void runSolve(prep.circuit, selected, corner, prep.injected);
  };

  const patchTransient = (patch: Record<string, unknown>): void => {
    setAnalysisConfig("transient", { ...(transientCfg ?? {}), ...patch });
  };

  const empty = nodeCount === 0;

  return (
    <section className="panel sim-panel">
      <h2>Simulate</h2>

      {!caps && (
        <p className="muted small">
          Backend offline — the analysis list comes from the server. Use “retry” in
          the toolbar.
        </p>
      )}

      {caps && (
        <>
          <div className="analysis-list">
            {analysisKeys.map((key) => {
              const isConfigured = key in configured;
              return (
                <label className="analysis-item" key={key} title={HINTS[key]}>
                  <input
                    type="checkbox"
                    checked={selected.includes(key)}
                    onChange={() => toggle(key)}
                  />
                  <span>{key}</span>
                  {isConfigured && (
                    <span className="cfg-dot" title="configured by this circuit" />
                  )}
                </label>
              );
            })}
          </div>

          {reports.length > 0 && (
            <div className="stimulus">
              <div className="subhead">Stimulus</div>
              {reports.map((r) => (
                <div className={`stim-row${r.silent ? " silent" : ""}`} key={r.name}>
                  <span className="stim-analysis">{r.name}</span>
                  <span className="stim-detail">
                    {r.detail}
                    {r.kind === "ac" && <PortList ports={r.ports} />}
                    {r.silent && circuit && (
                      <StimulusPicker
                        analysis={r.name}
                        options={stimulusOptions(circuit, r.name)}
                        value={picked[r.name]}
                        onPick={(id) => setPicked((p) => ({ ...p, [r.name]: id }))}
                        onApply={() => applyStimulus(r.name)}
                      />
                    )}
                  </span>
                </div>
              ))}
              {reports.some((r) => r.silent) && (
                <div className="stim-params">
                  <NumberField
                    label="Frequency"
                    unit="Hz"
                    value={stimFreq}
                    onCommit={(v) => v !== undefined && v > 0 && setStimFreq(v)}
                  />
                  <NumberField
                    label="Amplitude"
                    unit="V"
                    value={stimAmp}
                    onCommit={(v) => v !== undefined && setStimAmp(v)}
                  />
                  <p className="muted small">
                    A waveform is centred on the driven rail’s own DC level, so the
                    run starts from the operating point the circuit was designed
                    around; a differential pair’s two halves are the same waveform
                    inverted. Frequency applies to the time-domain sources only.
                  </p>
                </div>
              )}
            </div>
          )}

          {selected.includes("transient") && (
            <div className="sub-config">
              <button
                className="sub-config-tab"
                onClick={() => setShowTransient((v) => !v)}
              >
                {showTransient ? "▾" : "▸"} Transient setup
                {!transientCfg && <span className="needed"> needed</span>}
              </button>
              {showTransient && (
                <>
                  <NumberField
                    label="Stop time"
                    unit="s"
                    value={typeof transientCfg?.tstop === "number"
                      ? transientCfg.tstop : undefined}
                    onCommit={(v) => v !== undefined && patchTransient({ tstop: v })}
                  />
                  <NumberField
                    label="Points"
                    value={typeof transientCfg?.n_points === "number"
                      ? transientCfg.n_points : undefined}
                    onCommit={(v) => v !== undefined && patchTransient({ n_points: v })}
                  />
                  <p className="muted small">
                    Accepts <code>20u</code>, <code>1n</code>, <code>2e-5</code>.
                    Written into the circuit’s <code>analyses</code> block, so it
                    exports and undoes with the rest of the design.
                  </p>
                </>
              )}
            </div>
          )}

          <label className="field">
            <span>
              Corner
              {circuitCorners && (
                <span className="muted"> — {circuitCorners.length} for this circuit</span>
              )}
            </span>
            <select value={corner} onChange={(e) => setCorner(e.target.value)}>
              <option value="">(circuit default)</option>
              {(circuitCorners ?? []).map((name) => (
                <option key={name} value={name}>{name}</option>
              ))}
            </select>
          </label>
          {!circuitCorners && !empty && (
            <p className="muted small">
              Corner list loads with the next validation pass.
            </p>
          )}

          <button
            className="btn primary run-btn"
            onClick={run}
            disabled={running || selected.length === 0 || empty}
            title={
              empty ? "The canvas is empty"
                : selected.length === 0 ? "Select at least one analysis"
                : "Run the selected analyses"
            }
          >
            {running ? "Running…" : `Run ${selected.length || ""}`}
          </button>

          {blocked && <div className="run-error"><div className="run-error-msg">{blocked}</div></div>}

          <p className="muted small hint">
            Unconfigured <code>ac</code>/<code>noise</code> get {DEFAULT_SWEEP_LABEL} for
            the request only.
          </p>
        </>
      )}
    </section>
  );
}
