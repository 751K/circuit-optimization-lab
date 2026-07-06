/**
 * Right column: the inspector. Renders a property editor for the single selected
 * node (or the edge's read-only net, or — when nothing is selected — circuit-
 * level stats). Field edits go straight to store actions; renames go through
 * renameNode so edge endpoints follow.
 *
 * Numeric fields accept scientific notation (e.g. 2e-12); an unparseable entry
 * is left as the previous value on blur.
 */
import { useEditor } from "../store";
import {
  resolveNets,
  type CapacitorNode,
  type GraphNode,
  type MosfetNode,
  type OutputNode,
  type RailNode,
  type ResistorNode,
} from "../model";
import { NumberField, TextField } from "./fields";

const MODEL_KWARG_KEYS = ["vb", "corner", "extract_w", "temperature", "NF"] as const;

export default function Inspector() {
  const graph = useEditor((s) => s.graph);
  const selection = useEditor((s) => s.selection);
  const caps = useEditor((s) => s.caps);
  const rest = useEditor((s) => s.rest);

  const single =
    selection.nodes.length === 1 && selection.edges.length === 0
      ? graph.nodes.find((n) => n.id === selection.nodes[0])
      : undefined;
  const singleEdge =
    selection.edges.length === 1 && selection.nodes.length === 0
      ? graph.edges.find((e) => e.id === selection.edges[0])
      : undefined;

  return (
    <aside className="panel inspector">
      <h2>Inspector</h2>
      {single ? (
        <NodeEditor node={single} models={caps ? Object.keys(caps.models) : []} bias={rest.bias} />
      ) : singleEdge ? (
        <EdgeInfo edgeId={singleEdge.id} />
      ) : selection.nodes.length + selection.edges.length > 1 ? (
        <p className="muted">
          {selection.nodes.length} nodes, {selection.edges.length} edges selected.
        </p>
      ) : (
        <CircuitInfo />
      )}
    </aside>
  );
}

function NodeEditor({
  node,
  models,
  bias,
}: {
  node: GraphNode;
  models: string[];
  bias: unknown;
}) {
  switch (node.kind) {
    case "mosfet":
      return <MosfetEditor node={node} models={models} />;
    case "rail":
      return <RailEditor node={node} bias={bias} />;
    case "resistor":
      return <ResistorEditor node={node} />;
    case "capacitor":
      return <CapacitorEditor node={node} />;
    case "output":
      return <OutputEditor node={node} />;
  }
}

function MosfetEditor({ node, models }: { node: MosfetNode; models: string[] }) {
  const updateNodeProps = useEditor((s) => s.updateNodeProps);
  const renameNode = useEditor((s) => s.renameNode);
  const kwargs = (node.modelKwargs ?? {}) as Record<string, unknown>;

  const setKwarg = (key: string, value: unknown): void => {
    const next = { ...kwargs };
    if (value === undefined || value === "" || value === null) delete next[key];
    else next[key] = value;
    updateNodeProps(node.id, { modelKwargs: next });
  };

  return (
    <div className="editor">
      <div className="badge">MOSFET</div>
      <TextField label="name" value={node.name} onCommit={(v) => renameNode(node.id, v)} />
      <NumberField label="W" value={node.W} onCommit={(v) => updateNodeProps(node.id, { W: v })} />
      <NumberField label="L" value={node.L} onCommit={(v) => updateNodeProps(node.id, { L: v })} />
      <NumberField
        label="nf"
        value={node.nf}
        allowEmpty
        onCommit={(v) => updateNodeProps(node.id, { nf: v })}
      />
      <label className="field">
        <span>model</span>
        <select
          value={node.model ?? ""}
          onChange={(e) => updateNodeProps(node.id, { model: e.target.value || undefined })}
        >
          <option value="">(none)</option>
          {models.map((m) => (
            <option key={m} value={m}>
              {m}
            </option>
          ))}
          {node.model && !models.includes(node.model) && (
            <option value={node.model}>{node.model} (custom)</option>
          )}
        </select>
      </label>
      <NumberField
        label="input_drive"
        value={node.inputDrive}
        allowEmpty
        onCommit={(v) => updateNodeProps(node.id, { inputDrive: v })}
      />

      <div className="subhead">model kwargs</div>
      {MODEL_KWARG_KEYS.map((k) =>
        k === "corner" ? (
          <TextField
            key={k}
            label={k}
            value={typeof kwargs[k] === "string" ? (kwargs[k] as string) : ""}
            onCommit={(v) => setKwarg(k, v || undefined)}
          />
        ) : (
          <NumberField
            key={k}
            label={k}
            value={typeof kwargs[k] === "number" ? (kwargs[k] as number) : undefined}
            allowEmpty
            onCommit={(v) => setKwarg(k, v)}
          />
        ),
      )}
    </div>
  );
}

function RailEditor({ node, bias }: { node: RailNode; bias: unknown }) {
  const updateNodeProps = useEditor((s) => s.updateNodeProps);
  const renameNode = useEditor((s) => s.renameNode);
  const biasKeys = bias && typeof bias === "object" ? Object.keys(bias as object) : [];
  const isKey = typeof node.railValue === "string";

  const setRailValue = (raw: string): void => {
    // Prefer a bias-key string; else a numeric constant.
    const asNum = Number(raw);
    const value: string | number = raw !== "" && !Number.isNaN(asNum) && !biasKeys.includes(raw)
      ? asNum
      : raw;
    const patch: Partial<RailNode> = { railValue: value };
    // Resolve biasValue for display when the value is a known bias key.
    if (typeof value === "string" && bias && typeof bias === "object" && value in (bias as object)) {
      patch.biasValue = (bias as Record<string, number>)[value];
    } else {
      patch.biasValue = undefined;
    }
    updateNodeProps(node.id, patch);
  };

  return (
    <div className="editor">
      <div className="badge">RAIL</div>
      <TextField label="net name" value={node.net} onCommit={(v) => renameNode(node.id, v)} />
      <label className="field">
        <span>value</span>
        <input
          type="text"
          defaultValue={String(node.railValue)}
          onBlur={(e) => setRailValue(e.target.value)}
          list={biasKeys.length ? `bias-keys-${node.id}` : undefined}
        />
        {biasKeys.length > 0 && (
          <datalist id={`bias-keys-${node.id}`}>
            {biasKeys.map((k) => (
              <option key={k} value={k} />
            ))}
          </datalist>
        )}
      </label>
      <p className="muted small">
        {isKey
          ? `bias key${node.biasValue !== undefined ? ` = ${node.biasValue}` : " (unresolved)"}`
          : "numeric constant"}
      </p>
    </div>
  );
}

function ResistorEditor({ node }: { node: ResistorNode }) {
  const updateNodeProps = useEditor((s) => s.updateNodeProps);
  const renameNode = useEditor((s) => s.renameNode);
  return (
    <div className="editor">
      <div className="badge">RESISTOR</div>
      <TextField label="name" value={node.name} onCommit={(v) => renameNode(node.id, v)} />
      <NumberField label="R (Ω)" value={node.R} onCommit={(v) => updateNodeProps(node.id, { R: v })} />
    </div>
  );
}

function CapacitorEditor({ node }: { node: CapacitorNode }) {
  const updateNodeProps = useEditor((s) => s.updateNodeProps);
  const renameNode = useEditor((s) => s.renameNode);
  const isLoad = node.origin === "load_caps";
  return (
    <div className="editor">
      <div className="badge">CAPACITOR{isLoad ? " (load)" : ""}</div>
      {!isLoad && (
        <TextField label="name" value={node.name} onCommit={(v) => renameNode(node.id, v)} />
      )}
      <NumberField label="C (F)" value={node.C} onCommit={(v) => updateNodeProps(node.id, { C: v })} />
      <label className="field">
        <span>origin</span>
        <select
          value={node.origin}
          onChange={(e) =>
            updateNodeProps(node.id, { origin: e.target.value as CapacitorNode["origin"] })
          }
        >
          <option value="capacitors">capacitors</option>
          <option value="load_caps">load_caps</option>
        </select>
      </label>
    </div>
  );
}

function OutputEditor({ node }: { node: OutputNode }) {
  const updateNodeProps = useEditor((s) => s.updateNodeProps);
  return (
    <div className="editor">
      <div className="badge">OUTPUT</div>
      <NumberField
        label="order"
        value={node.order}
        onCommit={(v) => updateNodeProps(node.id, { order: v ?? 0 })}
      />
      <p className="muted small">order 0 = "+", 1 = "−" (differential [VOP, VON]).</p>
    </div>
  );
}

function EdgeInfo({ edgeId }: { edgeId: string }) {
  const graph = useEditor((s) => s.graph);
  let net: string | undefined;
  try {
    const { portNet } = resolveNets(graph);
    const e = graph.edges.find((x) => x.id === edgeId);
    if (e) {
      const SEP = String.fromCharCode(31);
      net = portNet.get(`${e.source.node}${SEP}${e.source.port}`);
    }
  } catch {
    net = undefined;
  }
  return (
    <div className="editor">
      <div className="badge">WIRE</div>
      <p>
        net: <strong>{net ?? "(unresolved — net conflict)"}</strong>
      </p>
      <p className="muted small">Wires are read-only; they only tie ports onto one net.</p>
    </div>
  );
}

function CircuitInfo() {
  const graph = useEditor((s) => s.graph);
  const rest = useEditor((s) => s.rest);
  const name = typeof rest.name === "string" ? rest.name : "(unnamed)";
  const devices = graph.nodes.filter((n) => n.kind === "mosfet").length;
  const rails = graph.nodes.filter((n) => n.kind === "rail").length;
  const passives = graph.nodes.filter(
    (n) => n.kind === "resistor" || n.kind === "capacitor",
  ).length;
  let netCount = 0;
  try {
    netCount = new Set(resolveNets(graph).portNet.values()).size;
  } catch {
    netCount = -1;
  }
  return (
    <div className="editor">
      <div className="badge">CIRCUIT</div>
      <p>
        name: <strong>{name}</strong>
      </p>
      <ul className="stats">
        <li>devices: {devices}</li>
        <li>rails: {rails}</li>
        <li>passives (R/C): {passives}</li>
        <li>nodes total: {graph.nodes.length}</li>
        <li>nets: {netCount < 0 ? "— (net conflict)" : netCount}</li>
      </ul>
      <p className="muted small">Select an element to edit it.</p>
    </div>
  );
}
