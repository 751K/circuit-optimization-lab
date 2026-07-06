/**
 * TypeScript types for the circuitopt circuit-JSON format.
 *
 * Authoritative format doc: docs/json_circuit_format.md.
 * Authoritative parser:      circuitopt/circuit_loader.py `circuit_from_dict`.
 *
 * The circuit JSON is the single source of truth for a circuit. These types
 * model *precisely* the blocks the graph-mapping layer (toGraph / toJson)
 * touches, and leave everything else to an index signature so an unknown or
 * not-yet-modeled block survives a round-trip untouched (the passthrough
 * principle: "you may not drop a single byte you don't understand").
 *
 * Value conventions (mirrored from the loader):
 *  - A `rails` value is a bias-key string (looked up in `bias`) or a numeric
 *    constant (e.g. `"GND": 0.0`).
 *  - Two-terminal elements accept an object form or a 4-tuple array shorthand;
 *    devices accept an object or a `[name, drain, gate, source]` shorthand.
 *  - `bias` values are numbers.
 */

// A device port / element terminal is always a net name (string).
export type NetName = string;

/** A rail value: a bias-key reference (string) or a numeric constant. */
export type RailValue = string | number;

/**
 * Device object form. W/L may be embedded here or supplied via top-level
 * `sizes`; NF may be embedded or via top-level `nf`. Extra keys are preserved.
 */
export interface DeviceObject {
  name: string;
  drain: NetName;
  gate: NetName;
  source: NetName;
  W?: number;
  L?: number;
  NF?: number;
  [key: string]: unknown;
}

/** Device array shorthand: `[name, drain, gate, source]`. */
export type DeviceArray = [string, NetName, NetName, NetName];

export type Device = DeviceObject | DeviceArray;

/** `sizes`: per-device `[W, L]`. */
export type Sizes = Record<string, [number, number]>;

/** `nf`: a global finger count, or per-device. */
export type Nf = number | Record<string, number>;

/** A `models` entry: a PDK model-type key plus forwarded ctor kwargs. */
export interface ModelEntry {
  type?: string;
  vb?: number;
  corner?: string;
  extract_w?: number;
  temperature?: number;
  NF?: number;
  [key: string]: unknown;
}

/** Two-terminal element object forms. */
export interface ResistorObject {
  name: string;
  a: NetName;
  b: NetName;
  R: number;
  [key: string]: unknown;
}
export interface CapacitorObject {
  name: string;
  a: NetName;
  b: NetName;
  C: number;
  [key: string]: unknown;
}
export type ResistorArray = [string, NetName, NetName, number];
export type CapacitorArray = [string, NetName, NetName, number];
export type Resistor = ResistorObject | ResistorArray;
export type Capacitor = CapacitorObject | CapacitorArray;

/** `load_caps` object / array forms (a nameless capacitor between two nets). */
export interface LoadCapObject {
  a: NetName;
  b: NetName;
  C: number;
  [key: string]: unknown;
}
export type LoadCapArray = [NetName, NetName, number];
export type LoadCap = LoadCapObject | LoadCapArray;

/**
 * UI layout + ordering metadata — a top-level block the backend loader ignores
 * (it silently drops unknown top-level keys). Besides canvas positions it
 * records the source ordering of order-significant blocks so a round-trip
 * reproduces it exactly:
 *   - `solved` order defines the MNA/DAE vector ordering, and
 *   - `devices` order is author-meaningful.
 * Both are reconstructed from resolved nets / nodes, which alphabetize for a
 * stable diff; the recorded order replays the original when present.
 */
export interface CircuitUi {
  /** node id -> [x, y] canvas position. */
  positions?: Record<string, [number, number]>;
  /** original ordering hints (block name -> ordered key list). */
  order?: {
    /** original `solved` net order. */
    solved?: string[];
    /** original `devices` name order. */
    devices?: string[];
    /** original `rails` key order. */
    rails?: string[];
    /** original `resistors` name order. */
    resistors?: string[];
    /** original `capacitors` name order. */
    capacitors?: string[];
    /** original `outputs` net order (usually implied by output-node `order`). */
    outputs?: string[];
  };
  [key: string]: unknown;
}

/**
 * The circuit JSON object. Only the blocks the mapping layer consumes are
 * typed explicitly; everything else (analyses, explore, periodic, vsources,
 * controlled sources, aliases, dc_guesses, transient_inputs, name, sizes,
 * nf, ...) is carried through the index signature and preserved verbatim.
 */
export interface CircuitJson {
  name?: string;
  solved: NetName[];
  rails: Record<string, RailValue>;
  devices: Device[];
  bias?: Record<string, number>;
  outputs?: NetName[];
  sizes?: Sizes;
  nf?: Nf;
  models?: Record<string, ModelEntry>;
  resistors?: Resistor[];
  capacitors?: Capacitor[];
  load_caps?: LoadCap[];
  input_drives?: Record<string, number>;
  ui?: CircuitUi;
  [key: string]: unknown;
}
