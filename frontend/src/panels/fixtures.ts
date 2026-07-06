/**
 * Built-in example circuits for the "load from example" dropdown. These are the
 * same 12 JSON fixtures F1's round-trip test pins; import.meta.glob eagerly
 * bundles them so the palette can load one with no backend round-trip.
 */
import type { CircuitJson } from "../model";

const modules = import.meta.glob<CircuitJson>(
  "../model/__fixtures__/*.json",
  { eager: true, import: "default" },
);

export interface FixtureEntry {
  /** Base filename without extension, e.g. "sky130_5t_ota". */
  key: string;
  /** The circuit's declared name, or the key when absent. */
  label: string;
  json: CircuitJson;
}

export const FIXTURES: FixtureEntry[] = Object.entries(modules)
  .map(([path, json]) => {
    const key = path.slice(path.lastIndexOf("/") + 1).replace(/\.json$/, "");
    return { key, label: json.name ? `${key} (${json.name})` : key, json };
  })
  .sort((a, b) => a.key.localeCompare(b.key));
