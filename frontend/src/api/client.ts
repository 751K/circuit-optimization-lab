/**
 * Typed client for the circuitopt local FastAPI service (see docs/service_api.md).
 *
 * Every route is a thin adapter over the same solver stack the CLI drives. This
 * client covers the synchronous endpoints (health / capabilities / validate /
 * solve) and the background-job endpoints (PVT sweep / mismatch MC), including
 * their WebSocket progress stream and cooperative cancellation.
 *
 * Base URL, resolved once at load time in three tiers (highest first):
 *   1. `window.__CIRCUITOPT_API_BASE__` — injected by the Tauri desktop shell,
 *      which negotiates the backend port at launch (see src-tauri/).
 *   2. `import.meta.env.VITE_API_BASE` — the plain-web override env var.
 *   3. `http://127.0.0.1:8341` — the service's default port.
 * Pure web mode (vite dev/build with no Tauri) never sees tier 1, so its
 * behaviour is byte-for-byte what it was before the desktop shell existed.
 */
import type { CircuitJson } from "../model/circuit";

/** Read the shell-injected base if present; guarded for non-browser (test) envs. */
function injectedBase(): string | undefined {
  if (typeof window === "undefined") return undefined;
  const b = (window as { __CIRCUITOPT_API_BASE__?: unknown })
    .__CIRCUITOPT_API_BASE__;
  return typeof b === "string" && b.length > 0 ? b : undefined;
}

export const API_BASE: string =
  injectedBase() ?? import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8341";

// ── Response shapes (docs/service_api.md) ────────────────────────────────

/** `GET /api/v1/health` */
export interface HealthResponse {
  status: string; // "ok"
  version: string;
  api: string; // "v1"
}

/**
 * `GET /api/v1/capabilities` — the single source of truth for a GUI's
 * dropdowns. `models` maps a registered model-type key to its class's fully
 * qualified name; `analyses` maps each analysis name to its sorted list of
 * legal option keys; `corners` lists the three process-corner families; `jobs`
 * the background job kinds.
 */
export interface CapabilitiesResponse {
  version: string;
  api: string;
  models: Record<string, string>;
  analyses: Record<string, string[]>;
  corners: {
    otft: string[];
    sky130: string[];
    freepdk45: string[];
    [family: string]: string[];
  };
  jobs: string[];
}

/**
 * `POST /api/v1/validate` — always HTTP 200; the outcome is the payload.
 *
 * `corners` is the corner set **this circuit** admits, and is absent when the
 * circuit does not parse. It is not a subset of the capabilities menu: a circuit
 * belongs to exactly one model family, and the OTFT process names
 * (typical/slow/fast) and the silicon card corners (tt/ss/ff/sf/fs) are disjoint.
 * Offering the union would offer corners that cannot resolve, so the corner
 * dropdown is driven from here, not from capabilities. `silicon` additionally
 * says whether the temperature / supply PVT axes exist for this circuit.
 */
export interface ValidateResponse {
  valid: boolean;
  errors?: string[];
  corners?: string[];
  silicon?: boolean;
  corner_error?: string;
}

/** `POST /api/v1/solve` success (HTTP 200). Results are JSON-safe per to_jsonable. */
export interface SignoffResponse {
  status: "pass" | "fail" | "not_configured";
  measurements: Record<string, unknown>;
  constraints: Record<string, unknown>;
  passed: boolean | null;
  worst_case: Record<string, unknown> | null;
}

export interface SolveResponse {
  status: "valid";
  results: Record<string, unknown>;
  signoff: SignoffResponse;
  elapsed_s: number;
}

/**
 * The `{stage, message}` error envelope shared by the 422 `detail` and the
 * WS terminal frames. `stage` is "parse" | "solve" | "job" (open-ended).
 */
export interface ErrorEnvelope {
  stage: string;
  message: string;
}

/** Thrown when a route returns a non-2xx status carrying an ErrorEnvelope. */
export class ApiError extends Error {
  readonly stage: string;
  readonly status: number;
  constructor(status: number, envelope: ErrorEnvelope) {
    super(envelope.message);
    this.name = "ApiError";
    this.status = status;
    this.stage = envelope.stage;
  }
}

// ── internals ────────────────────────────────────────────────────────────

function isErrorEnvelope(x: unknown): x is ErrorEnvelope {
  return (
    typeof x === "object" &&
    x !== null &&
    typeof (x as Record<string, unknown>).stage === "string" &&
    typeof (x as Record<string, unknown>).message === "string"
  );
}

/**
 * Parse a non-OK response into an ApiError. FastAPI wraps HTTPException detail
 * as `{"detail": {stage, message}}`; fall back to a generic envelope for
 * anything else (e.g. a plain string detail or a 500 with no JSON body).
 */
async function toApiError(res: Response): Promise<ApiError> {
  let body: unknown = undefined;
  try {
    body = await res.json();
  } catch {
    // non-JSON body
  }
  const detail =
    body && typeof body === "object" && "detail" in body
      ? (body as { detail: unknown }).detail
      : body;
  if (isErrorEnvelope(detail)) {
    return new ApiError(res.status, detail);
  }
  const message =
    typeof detail === "string"
      ? detail
      : `HTTP ${res.status} ${res.statusText}`;
  return new ApiError(res.status, { stage: "http", message });
}

async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) throw await toApiError(res);
  return (await res.json()) as T;
}

async function postJson<T>(path: string, payload: unknown): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw await toApiError(res);
  return (await res.json()) as T;
}

// ── public API ─────────────────────────────────────────────────────────

export function health(): Promise<HealthResponse> {
  return getJson<HealthResponse>("/api/v1/health");
}

export function capabilities(): Promise<CapabilitiesResponse> {
  return getJson<CapabilitiesResponse>("/api/v1/capabilities");
}

/**
 * `POST /api/v1/validate` — the request body is the raw circuit JSON object
 * (not wrapped in an envelope). Always returns HTTP 200; a broken circuit is
 * reported via `valid:false` + `errors`, not an exception.
 */
export function validate(circuit: CircuitJson): Promise<ValidateResponse> {
  return postJson<ValidateResponse>("/api/v1/validate", circuit);
}

/**
 * `POST /api/v1/solve` — runs the analysis suite synchronously.
 * `selected` restricts which analyses run (omit to run everything the
 * circuit's `analyses` block configures); `corner` is a process-corner
 * override. A parse or solve failure surfaces as an {@link ApiError} carrying
 * the `stage`.
 */
export function solve(
  circuit: CircuitJson,
  selected?: string[],
  corner?: string,
): Promise<SolveResponse> {
  const payload: {
    circuit: CircuitJson;
    selected?: string[];
    corner?: string;
  } = { circuit };
  if (selected !== undefined) payload.selected = selected;
  if (corner !== undefined) payload.corner = corner;
  return postJson<SolveResponse>("/api/v1/solve", payload);
}

// ── background jobs ──────────────────────────────────────────────────────

/** A job's lifecycle state. The last three are terminal. */
export type JobStatus = "queued" | "running" | "done" | "failed" | "cancelled";

/** `GET /api/v1/jobs/{id}` — snapshot plus, once terminal, result or error. */
export interface JobSnapshot {
  job_id: string;
  kind: string;
  status: JobStatus;
  created: number;
  started: number | null;
  finished: number | null;
  progress: JobProgress | null;
  result?: Record<string, unknown>;
  error?: ErrorEnvelope;
}

/** A progress frame. `unit` names what `done`/`total` count (samples, slices). */
export interface JobProgress {
  type: "progress";
  done: number;
  total: number;
  frac: number;
  unit?: string;
  partial?: unknown;
}

export function isTerminal(status: JobStatus): boolean {
  return status === "done" || status === "failed" || status === "cancelled";
}

/** `POST /api/v1/jobs/pvt` — a corner sweep, optionally gridded over PVT axes. */
export interface PvtRequest {
  /** Omit to sweep the circuit's own family (see {@link ValidateResponse}). */
  corners?: string[];
  /** Temperature axis in °C. Silicon only; a 422 otherwise. */
  temps?: number[];
  /** Uniform bias multipliers. Silicon only; a 422 otherwise. */
  vdd_scale?: number[];
  workers?: number;
  /** Omit to inherit the circuit's own `analyses.ac.freqs`. */
  freqs?: { start: number; stop: number; num: number; scale: string };
  /** Omit to inherit the circuit's own `analyses.noise.band`. */
  band?: [number, number];
}

export function submitPvt(
  circuit: CircuitJson,
  options: PvtRequest = {},
): Promise<JobSnapshot> {
  return postJson<JobSnapshot>("/api/v1/jobs/pvt", { circuit, ...options });
}

export function submitMc(
  circuit: CircuitJson,
  options: { n?: number; seed?: number; corner?: string; workers?: number } = {},
): Promise<JobSnapshot> {
  return postJson<JobSnapshot>("/api/v1/jobs/mc", { circuit, ...options });
}

export function getJob(jobId: string): Promise<JobSnapshot> {
  return getJson<JobSnapshot>(`/api/v1/jobs/${encodeURIComponent(jobId)}`);
}

export function listJobs(): Promise<{ jobs: JobSnapshot[] }> {
  return getJson<{ jobs: JobSnapshot[] }>("/api/v1/jobs");
}

/**
 * Request cooperative cancellation. Cancellation is not a hard kill: work already
 * in flight runs to completion, so the job stays non-terminal for a while after
 * this resolves. A 409 (already terminal) is swallowed — the caller's intent is
 * satisfied either way.
 */
export async function cancelJob(jobId: string): Promise<void> {
  const res = await fetch(`${API_BASE}/api/v1/jobs/${encodeURIComponent(jobId)}`, {
    method: "DELETE",
  });
  if (!res.ok && res.status !== 409) throw await toApiError(res);
}

/** The frame the WS sends once a job reaches a terminal state. */
export interface JobTerminalFrame {
  type: "terminal";
  status: JobStatus;
  error?: ErrorEnvelope;
}

type JobEvent = JobProgress | JobTerminalFrame | { type: "error"; message: string };

/**
 * Stream a job's progress over the WebSocket, resolving when it terminates.
 *
 * The socket carries progress only; the *result* is fetched over HTTP once the
 * terminal frame lands, because it can be large and the caller usually wants it
 * as one JSON body rather than streamed. Returns a `cancel()` that closes the
 * socket without cancelling the job — closing a viewer is not stopping the work.
 *
 * If the socket cannot be opened at all (some proxies, or a browser that blocks
 * it), `onFallback` is invoked so the caller can degrade to polling rather than
 * hanging forever on a stream that will never arrive.
 */
export function watchJob(
  jobId: string,
  handlers: {
    onProgress?: (progress: JobProgress) => void;
    onTerminal?: (frame: JobTerminalFrame) => void;
    onFallback?: (reason: string) => void;
  },
): () => void {
  const wsBase = API_BASE.replace(/^http/, "ws");
  let socket: WebSocket;
  try {
    socket = new WebSocket(`${wsBase}/api/v1/jobs/${encodeURIComponent(jobId)}/events`);
  } catch (e) {
    handlers.onFallback?.(e instanceof Error ? e.message : String(e));
    return () => {};
  }

  let settled = false;
  socket.onmessage = (ev) => {
    let event: JobEvent;
    try {
      event = JSON.parse(String(ev.data)) as JobEvent;
    } catch {
      return; // a frame we cannot parse is not a reason to tear the stream down
    }
    if (event.type === "progress") handlers.onProgress?.(event);
    else if (event.type === "terminal") {
      settled = true;
      handlers.onTerminal?.(event);
    } else if (event.type === "error") {
      settled = true;
      handlers.onFallback?.(event.message);
    }
  };
  socket.onerror = () => {
    if (!settled) handlers.onFallback?.("websocket error");
  };
  socket.onclose = () => {
    // A close before any terminal frame means we lost the stream, not that the
    // job ended: fall back so the caller can poll for the real outcome.
    if (!settled) handlers.onFallback?.("websocket closed early");
  };

  return () => {
    settled = true; // a deliberate close must not look like a lost stream
    socket.close();
  };
}
