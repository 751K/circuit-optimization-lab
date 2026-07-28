/**
 * The repository's `examples/` directory, as seen by the editor and its tests.
 *
 * There used to be a curated copy under `src/model/__fixtures__/`. It drifted:
 * three of its twelve files no longer matched their `examples/` originals, and
 * twenty circuits added since — every TSMC28 design, the MDAC testbenches, the
 * SAR decks — never appeared in the palette at all. A duplicated corpus is a
 * corpus that goes stale, so `examples/` is now the only source and both the
 * palette and the round-trip tests read it.
 *
 * Not every file there is a circuit: `explore` configs and signoff manifests
 * live alongside them and carry no `solved` array. {@link isCircuitJson} selects
 * by shape rather than by an allow-list, so a new deck appears automatically and
 * a new manifest does not.
 */
import type { CircuitJson } from "./circuit";

/**
 * Whether a parsed JSON value is a circuit rather than a campaign manifest or
 * an explore config. The line-format loader requires `solved` (the node order)
 * and `rails` (the fixed-potential nets); nothing else in `examples/` has both.
 */
export function isCircuitJson(value: unknown): value is CircuitJson {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    return false;
  }
  const candidate = value as Record<string, unknown>;
  return Array.isArray(candidate.solved)
    && typeof candidate.rails === "object"
    && candidate.rails !== null;
}

export interface ExampleEntry {
  /** Base filename without extension, e.g. "sky130_fd_ota". */
  key: string;
  /** The circuit's declared name, or the key when absent. */
  label: string;
  json: CircuitJson;
}

/** Group an example by the PDK family its key names, for a grouped menu. */
export function exampleFamily(key: string): string {
  if (key.startsWith("tsmc28")) return "TSMC28HPC+";
  if (key.startsWith("sky130")) return "SKY130";
  if (key.startsWith("freepdk45")) return "FreePDK45";
  if (key.startsWith("afe")) return "AFE / OTFT";
  return "Generic";
}

/** Build the sorted, labelled entry list from a {path: json} map. */
export function toEntries(modules: Record<string, unknown>): ExampleEntry[] {
  return Object.entries(modules)
    .filter(([, json]) => isCircuitJson(json))
    .map(([path, json]) => {
      const key = path.slice(path.lastIndexOf("/") + 1).replace(/\.json$/, "");
      const circuit = json as CircuitJson;
      return {
        key,
        label: circuit.name && circuit.name !== key ? `${key} — ${circuit.name}` : key,
        json: circuit,
      };
    })
    .sort((a, b) => a.key.localeCompare(b.key));
}
