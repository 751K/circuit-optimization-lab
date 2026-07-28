/**
 * Built-in example circuits for the palette's "load an example" menu.
 *
 * Read straight from the repository's `examples/` directory — see
 * `model/examples.ts` for why that is the only copy. `import.meta.glob` bundles
 * them eagerly so the palette can load one with no backend round-trip; Vite
 * needs `server.fs.allow` to reach outside `frontend/` (see vite.config.ts).
 */
import { exampleFamily, toEntries, type ExampleEntry } from "../model/examples";

const modules = import.meta.glob<unknown>(
  "../../../examples/*.json",
  { eager: true, import: "default" },
);

export type { ExampleEntry } from "../model/examples";
export { exampleFamily } from "../model/examples";

export const FIXTURES: ExampleEntry[] = toEntries(modules);

/** The examples grouped by PDK family, for an <optgroup> menu. */
export const FIXTURE_GROUPS: { family: string; entries: ExampleEntry[] }[] =
  Object.entries(
    FIXTURES.reduce<Record<string, ExampleEntry[]>>((groups, entry) => {
      const family = exampleFamily(entry.key);
      (groups[family] ??= []).push(entry);
      return groups;
    }, {}),
  )
    .map(([family, entries]) => ({ family, entries }))
    .sort((a, b) => a.family.localeCompare(b.family));
