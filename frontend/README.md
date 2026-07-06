# circuitopt browser builder (frontend)

A React + React Flow circuit editor for the `circuitopt` solver stack. Draw a
circuit on a canvas, validate it live against the local service, run analyses,
and see Bode / noise / transient plots — all in the browser. The circuit JSON
is the single source of truth; the editor round-trips it losslessly.

## Prerequisites: the backend

The editor talks to the `circuitopt` FastAPI service (`/api/v1/*`). Start it
from the repo root:

```sh
pip install -e ".[serve]"      # installs the FastAPI service extras
circuit-opt serve --port 8341  # or: python -m circuitopt.service --port 8341
```

The frontend defaults to `http://127.0.0.1:8341`. Point it elsewhere with the
`VITE_API_BASE` env var (e.g. `VITE_API_BASE=http://host:9000 npm run dev`).
The editor is fully usable offline — only `/validate` and `/solve` need the
backend; a "backend not connected" banner offers a retry.

## Develop

```sh
npm install     # once
npm run dev     # Vite dev server (hot reload)
npm test        # vitest (pure-function unit tests)
npm run build   # tsc -b (strict) + vite production build
```

An optional live smoke check (needs the backend up) exercises the real
solve→transform pipeline:

```sh
node_modules/.bin/vite-node scripts/smoke_solve.ts
```

## Layout

- **`src/model/`** — the graph ⇄ circuit-JSON mapping core (F1): parses circuit
  JSON into an editable node/edge graph and serializes it back losslessly. This
  is the format contract; do not change its semantics.
- **`src/canvas/`** — the React Flow projection (F2): node components, the
  store↔React-Flow adapter, per-port net-label overlay.
- **`src/store/`** — the zustand editor state (F2): the document graph, undo/redo,
  selection, and the cached backend capabilities probe.
- **`src/panels/`** — the app shell UI: palette, inspector, toolbar, status bar,
  and the Run panel.
- **`src/results/`** — the plotting layer (F3): pure transforms from `/solve`
  responses to plot-ready data (`transform.ts`, unit-tested against real
  response fixtures under `__fixtures__/`) and the ECharts views
  (`charts.tsx`, `ResultView.tsx`) plus JSON export (`download.ts`).
- **`src/api/client.ts`** — the typed fetch client for the service routes.
