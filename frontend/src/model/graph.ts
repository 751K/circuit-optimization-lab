/**
 * Editor-graph domain types (pure TS, zero React / React Flow dependency).
 *
 * F2 will adapt these to React Flow's Node/Edge, but the mapping core deals
 * only in these domain types so it stays testable and framework-free.
 *
 * ── Net model ────────────────────────────────────────────────────────────
 * The circuit JSON names nets directly on element terminals (a device's
 * drain/gate/source hold a net name; a resistor's a/b hold net names). The
 * graph re-expresses this as nodes with *ports* plus *edges* that join ports
 * onto the same electrical net. On export, the connected components of
 * (ports + edges) each resolve to one net name (see toJson.ts):
 *   - a component touching a rail port  -> that rail's net name
 *   - a component whose ports remember a reserved original net name -> reuse it
 *   - otherwise                          -> an auto name n1, n2, ...
 * Every port therefore records the net name it came from (`originalNet`) so a
 * hand-authored name ("tail", "vout") survives a round-trip.
 */

/** Canvas position; `[x, y]`. */
export type Position = [number, number];

export type NodeKind =
  | "mosfet"
  | "rail"
  | "resistor"
  | "capacitor"
  | "output";

/** A connection point on a node. `originalNet` is the net name it was imported
 *  from (undefined for a freshly-created port that has never been in JSON). */
export interface Port {
  /** Port id, unique within its node (e.g. "D", "G", "S", "a", "b", "in"). */
  id: string;
  /** Net name this port carried in the source JSON, if any. */
  originalNet?: string;
}

interface BaseNode {
  /** Node id, unique in the graph. For imported nodes this is the element name
   *  (device/resistor/capacitor name, rail name, or "out:<net>"), which is what
   *  ui.positions is keyed by. */
  id: string;
  kind: NodeKind;
  ports: Port[];
  position: Position;
}

/**
 * A three-terminal transistor. `model` + `modelKwargs` mirror a `models` entry
 * (PDK type and forwarded ctor kwargs: vb / corner / extract_w / temperature /
 * NF). `inputDrive` mirrors an `input_drives[name]` AC gate drive. Ports are
 * always exactly D / G / S.
 */
export interface MosfetNode extends BaseNode {
  kind: "mosfet";
  name: string;
  W: number;
  L: number;
  /**
   * Whether W/L should be re-emitted embedded on the device object. False only
   * when the source W/L came purely from a top-level `sizes` override (which is
   * preserved separately via `rest`), so export must not add an embedded slot.
   * Defaults to true for graph-authored (F2) devices.
   */
  hasEmbeddedWL?: boolean;
  /** Number of fingers, from an embedded device-object `NF`. */
  nf?: number;
  /** PDK model-type key from a `models` entry (e.g. "sky130.nmos"). */
  model?: string;
  /** Forwarded `models` kwargs (vb, corner, extract_w, temperature, NF, ...). */
  modelKwargs?: Record<string, unknown>;
  /** AC small-signal gate drive from `input_drives`. */
  inputDrive?: number;
}

/**
 * A named fixed-potential net. `railValue` is the rails-map value: a bias-key
 * string (looked up in `bias`) or a numeric constant. `biasValue` is the
 * resolved numeric bias when `railValue` is a key present in `bias`. A rail
 * node has a single port; its net name is the rail's key (== node id).
 */
export interface RailNode extends BaseNode {
  kind: "rail";
  /** The rail's net name (its key in the rails map). */
  net: string;
  /** bias-key string or numeric constant (the rails-map value). */
  railValue: string | number;
  /** Resolved numeric value from `bias`, when railValue is a bias key. */
  biasValue?: number;
}

/** A two-terminal resistor (ports a / b). */
export interface ResistorNode extends BaseNode {
  kind: "resistor";
  name: string;
  R: number;
}

/**
 * A two-terminal capacitor (ports a / b). `origin` records which JSON block it
 * came from so export puts it back where it belongs:
 *  - "capacitors" -> a named `capacitors` entry
 *  - "load_caps"  -> a nameless `load_caps` entry (name is synthetic, not exported)
 */
export interface CapacitorNode extends BaseNode {
  kind: "capacitor";
  name: string;
  C: number;
  origin: "capacitors" | "load_caps";
}

/** An output marker (single port), mapping one entry of the `outputs` array. */
export interface OutputNode extends BaseNode {
  kind: "output";
  /** Position of this output in the `outputs` array (preserves +/- ordering
   *  for differential outputs `[VOP, VON]`). */
  order: number;
}

export type GraphNode =
  | MosfetNode
  | RailNode
  | ResistorNode
  | CapacitorNode
  | OutputNode;

/** An edge joins two ports (identified by node id + port id) onto one net. */
export interface GraphEdge {
  id: string;
  source: { node: string; port: string };
  target: { node: string; port: string };
}

/**
 * The editor graph. `rest` carries every circuit-JSON block the graph does not
 * model (analyses, explore, periodic, vsources, controlled sources, aliases,
 * dc_guesses, transient_inputs, name, sizes, nf, ...) verbatim, so export can
 * merge it back and lose nothing.
 */
export interface CircuitGraph {
  nodes: GraphNode[];
  edges: GraphEdge[];
}
