/**
 * Shared helpers for the graph<->JSON mapping: element-shape normalization,
 * deterministic auto-layout, and a semantic deep-equal used by the round-trip
 * invariant test (never a string comparison).
 */
import type {
  Capacitor,
  Device,
  DeviceObject,
  LoadCap,
  Resistor,
} from "./circuit";

// ── element-shape normalization ──────────────────────────────────────────
// The loader accepts both an object form and an array shorthand for devices
// and two-terminal elements. We normalize to a common shape on import and do
// not try to reproduce the original (object vs array) form on export — the
// object form is always emitted, which is semantically identical for the
// backend and keeps output deterministic. This is an allowed round-trip
// difference (equivalent representation), verified against the real loader.

export function normalizeDevice(d: Device): {
  name: string;
  drain: string;
  gate: string;
  source: string;
  W?: number;
  L?: number;
  NF?: number;
  extra: Record<string, unknown>;
} {
  if (Array.isArray(d)) {
    const [name, drain, gate, source] = d;
    return { name, drain, gate, source, extra: {} };
  }
  const { name, drain, gate, source, W, L, NF, ...extra } = d;
  const out: {
    name: string;
    drain: string;
    gate: string;
    source: string;
    W?: number;
    L?: number;
    NF?: number;
    extra: Record<string, unknown>;
  } = { name, drain, gate, source, extra };
  if (W !== undefined) out.W = W;
  if (L !== undefined) out.L = L;
  if (NF !== undefined) out.NF = NF;
  return out;
}

export function normalizeTwoTerminal(
  el: Resistor | Capacitor,
): { name: string; a: string; b: string; value: number; extra: Record<string, unknown> } {
  if (Array.isArray(el)) {
    const [name, a, b, value] = el;
    return { name, a, b, value, extra: {} };
  }
  // Resistor uses R, Capacitor uses C — read whichever exists.
  const { name, a, b, R, C, ...extra } = el as Record<string, unknown> & {
    name: string;
    a: string;
    b: string;
    R?: number;
    C?: number;
  };
  const value = (R ?? C) as number;
  return { name, a, b, value, extra };
}

export function normalizeLoadCap(el: LoadCap): { a: string; b: string; C: number } {
  if (Array.isArray(el)) {
    const [a, b, C] = el;
    return { a, b, C };
  }
  return { a: el.a, b: el.b, C: el.C };
}

/** Rebuild a device object in a stable key order (name, ports, W, L, NF, extra). */
export function deviceToObject(d: {
  name: string;
  drain: string;
  gate: string;
  source: string;
  W?: number;
  L?: number;
  NF?: number;
  extra?: Record<string, unknown>;
}): DeviceObject {
  const out: DeviceObject = {
    name: d.name,
    drain: d.drain,
    gate: d.gate,
    source: d.source,
  };
  if (d.W !== undefined) out.W = d.W;
  if (d.L !== undefined) out.L = d.L;
  if (d.NF !== undefined) out.NF = d.NF;
  if (d.extra) Object.assign(out, d.extra);
  return out;
}

// ── deterministic auto-layout ────────────────────────────────────────────
// A dependency-free, purely deterministic column layout: rails on the left,
// then mosfets, then resistors, then capacitors, then outputs, each kind in a
// vertical stack, ordered by (already-sorted) node id. Same input -> same
// layout, which is all the invariant needs; F2 replaces this the moment a user
// drags a node (positions then persist through ui.positions).

const COL_X: Record<string, number> = {
  rail: 0,
  mosfet: 240,
  resistor: 480,
  capacitor: 660,
  output: 840,
};
const ROW_DY = 120;
const ROW_Y0 = 40;

export function autoPosition(kind: string, indexInKind: number): [number, number] {
  const x = COL_X[kind] ?? 480;
  return [x, ROW_Y0 + indexInKind * ROW_DY];
}

/**
 * One deterministic barycenter sweep to shorten total wire length in the
 * auto-layout. Within each column (nodes sharing an x), nodes are re-ordered by
 * the mean y of their connected neighbors (the classic one-pass barycenter
 * heuristic), then re-slotted onto the column's existing y grid — so spacing is
 * unchanged and only the *order* within a column shifts.
 *
 * Constraints honored:
 *  - Only nodes in `autoIds` (those without an explicit ui.positions entry) may
 *    move. Columns that contain a pinned node are left entirely untouched, so a
 *    stored layout is never perturbed.
 *  - Fully deterministic: same graph in -> same layout out. Ties in barycenter
 *    fall back to node id, and a node with no neighbors keeps its current slot.
 *
 * `adj` maps node id -> connected neighbor node ids (from the synthesized
 * edges). Mutates node positions in place.
 */
export function barycenterReorder(
  nodes: { id: string; position: [number, number] }[],
  adj: Map<string, Set<string>>,
  autoIds: Set<string>,
): void {
  // Snapshot every node's y for barycenter reference (uses the *current* layout
  // as the reference frame; a single sweep, so no iteration-order feedback).
  const yOf = new Map<string, number>(nodes.map((n) => [n.id, n.position[1]]));

  // Bucket nodes by column (x). Round to guard against float drift, though
  // autoPosition uses integers.
  const cols = new Map<number, { id: string; position: [number, number] }[]>();
  for (const n of nodes) {
    const x = Math.round(n.position[0]);
    let arr = cols.get(x);
    if (!arr) {
      arr = [];
      cols.set(x, arr);
    }
    arr.push(n);
  }

  for (const members of cols.values()) {
    // Skip any column touched by a pinned (non-auto) node — don't disturb a
    // stored layout.
    if (members.some((n) => !autoIds.has(n.id))) continue;
    if (members.length < 2) continue;

    // Existing y-slots for this column, ascending — we permute nodes onto them.
    const slots = members.map((n) => n.position[1]).sort((a, b) => a - b);

    const key = (n: { id: string }): number => {
      const nbrs = adj.get(n.id);
      if (!nbrs || nbrs.size === 0) return yOf.get(n.id) ?? 0;
      let sum = 0;
      let cnt = 0;
      for (const nb of nbrs) {
        const y = yOf.get(nb);
        if (y !== undefined) {
          sum += y;
          cnt += 1;
        }
      }
      return cnt === 0 ? (yOf.get(n.id) ?? 0) : sum / cnt;
    };

    const ordered = [...members].sort((a, b) => {
      const ka = key(a);
      const kb = key(b);
      if (ka !== kb) return ka - kb;
      return a.id < b.id ? -1 : a.id > b.id ? 1 : 0;
    });

    ordered.forEach((n, i) => {
      n.position[1] = slots[i]!;
    });
  }
}

// ── semantic deep-equal ──────────────────────────────────────────────────
// Compares two JSON-like values for structural + numeric equality. Object key
// order is ignored (an allowed round-trip difference). Numbers compare with a
// small relative tolerance so equivalent representations (2e-12 vs 2.0e-12,
// 40 vs 40.0) match. NaN === NaN here (arrays of results never appear in
// circuit descriptions, so this is a convenience, not a solver concern).

const REL_TOL = 1e-9;
const ABS_TOL = 1e-30;

function numbersClose(a: number, b: number): boolean {
  if (a === b) return true;
  if (Number.isNaN(a) && Number.isNaN(b)) return true;
  const diff = Math.abs(a - b);
  if (diff <= ABS_TOL) return true;
  return diff <= REL_TOL * Math.max(Math.abs(a), Math.abs(b));
}

export interface DeepEqualOptions {
  /** Top-level keys to ignore on both sides (e.g. "ui" — added by export). */
  ignoreTopLevelKeys?: string[];
}

export function deepEqual(
  a: unknown,
  b: unknown,
  opts: DeepEqualOptions = {},
  path = "$",
): { equal: boolean; diff?: string } {
  if (typeof a === "number" && typeof b === "number") {
    return numbersClose(a, b)
      ? { equal: true }
      : { equal: false, diff: `${path}: ${a} != ${b}` };
  }
  if (Array.isArray(a) || Array.isArray(b)) {
    if (!Array.isArray(a) || !Array.isArray(b)) {
      return { equal: false, diff: `${path}: array vs non-array` };
    }
    if (a.length !== b.length) {
      return { equal: false, diff: `${path}: array length ${a.length} != ${b.length}` };
    }
    for (let i = 0; i < a.length; i++) {
      const r = deepEqual(a[i], b[i], {}, `${path}[${i}]`);
      if (!r.equal) return r;
    }
    return { equal: true };
  }
  if (a !== null && b !== null && typeof a === "object" && typeof b === "object") {
    const ao = a as Record<string, unknown>;
    const bo = b as Record<string, unknown>;
    const ignore = new Set(path === "$" ? opts.ignoreTopLevelKeys ?? [] : []);
    const aKeys = Object.keys(ao).filter((k) => !ignore.has(k));
    const bKeys = Object.keys(bo).filter((k) => !ignore.has(k));
    const aSet = new Set(aKeys);
    const bSet = new Set(bKeys);
    for (const k of aKeys) {
      if (!bSet.has(k)) return { equal: false, diff: `${path}: key '${k}' missing on right` };
    }
    for (const k of bKeys) {
      if (!aSet.has(k)) return { equal: false, diff: `${path}: key '${k}' missing on left` };
    }
    for (const k of aKeys) {
      const r = deepEqual(ao[k], bo[k], {}, `${path}.${k}`);
      if (!r.equal) return r;
    }
    return { equal: true };
  }
  // primitives (string, boolean, null, undefined)
  return a === b
    ? { equal: true }
    : { equal: false, diff: `${path}: ${JSON.stringify(a)} != ${JSON.stringify(b)}` };
}
