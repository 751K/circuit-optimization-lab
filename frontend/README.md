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

## Desktop app (Tauri)

The same editor ships as a native macOS app under `src-tauri/`. The app is a
thin **shell**: it does not bundle Python. Instead it discovers and manages the
`circuitopt` service you install yourself, then points the webview at it.

**Prerequisites**

- [Rust](https://rustup.rs/) (stable) and Xcode command-line tools.
- The backend installed on your machine:
  `pip install "circuit-optimization[serve]"` (provides `circuit-opt serve`).

**Develop / build**

```sh
npm run tauri:dev     # Vite dev server + native window, hot reload
npm run tauri:build   # release .app + .dmg under src-tauri/target/release/bundle/
```

The build is unsigned/un-notarized — first launch needs right-click → **Open**
(or `xattr -dr com.apple.quarantine "CircuitOpt Builder.app"`).

**Backend discovery order** (on launch, in `src-tauri/src/backend.rs`):

1. **Adopt** — if `GET /api/v1/health` on the default port **8341** answers, the
   app uses that service as-is and never kills it (it's yours).
2. **Config file** — `~/Library/Application Support/com.circuitopt.builder/backend.json`,
   shape `{"command": ["/path/to/python", "-m", "circuitopt.service"]}` (or
   `["/path/to/circuit-opt", "serve"]`). The app appends `--port <n>`. On first
   run a template with `command: null` and a `_hint` is written here.
3. **Login-shell PATH** — `zsh -lc "command -v circuit-opt"`. This runs through
   a *login* shell on purpose: a Finder-launched GUI app has a bare PATH, so
   conda/homebrew/pyenv shims are invisible without sourcing your profile.

If all three fail, the app opens in offline mode with the toolbar's "backend
not connected" banner (validation/solve disabled; editing still works).

When the app *spawns* the backend it picks a free port at/after 8341, waits for
`/health` (up to 15 s), injects the URL as `window.__CIRCUITOPT_API_BASE__`
(highest-priority tier in `src/api/client.ts`), and **kills that process on
quit** — via both the normal quit path and a SIGTERM/SIGINT handler, so
`kill <app>` reaps it too. An adopted service (step 1) is always left running.

**Logs**: `~/Library/Logs/com.circuitopt.builder/backend.log` — the discovery
decisions plus the backend's own stdout/stderr, for troubleshooting.

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
- **`src/api/client.ts`** — the typed fetch client for the service routes. Its
  `API_BASE` resolves in three tiers: the Tauri-injected
  `window.__CIRCUITOPT_API_BASE__`, then `VITE_API_BASE`, then the 8341 default.
- **`src-tauri/`** — the macOS desktop shell (see *Desktop app* above).
  `src/backend.rs` is the pure, unit-tested backend-discovery/port logic;
  `src/lib.rs` is the Tauri glue (spawn, health-wait, inject, quit-cleanup).
