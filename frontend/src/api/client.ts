/**
 * Typed client for the circuitopt local FastAPI service (see docs/service_api.md).
 *
 * Every route is a thin adapter over the same solver stack the CLI drives. This
 * client models the synchronous endpoints (health / capabilities / validate /
 * solve) that the F1..F3 browser builder needs; background-job endpoints
 * (explore / mc) are intentionally out of scope for now.
 *
 * Base URL: `import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8341"`.
 */
import type { CircuitJson } from "../model/circuit";

export const API_BASE: string =
  import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8341";

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

/** `POST /api/v1/validate` — always HTTP 200; the outcome is the payload. */
export interface ValidateResponse {
  valid: boolean;
  errors?: string[];
}

/** `POST /api/v1/solve` success (HTTP 200). Results are JSON-safe per to_jsonable. */
export interface SolveResponse {
  results: Record<string, unknown>;
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
