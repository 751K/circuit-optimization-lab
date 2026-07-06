/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_BASE?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}

/**
 * Globals injected by the Tauri desktop shell (src-tauri/). Absent in plain
 * web mode. `__CIRCUITOPT_API_BASE__` is the negotiated backend URL (highest
 * precedence for client.ts); `__TAURI__` is Tauri's own presence marker,
 * enabled via `withGlobalTauri` — used to tailor the offline banner copy.
 */
interface Window {
  __CIRCUITOPT_API_BASE__?: string;
  __TAURI__?: unknown;
}
