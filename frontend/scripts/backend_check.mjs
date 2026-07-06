/**
 * One-shot backend cross-check for the F1 graph<->JSON mapping.
 *
 * For every example fixture:
 *   1. round-trip it: graphToCircuitJson(circuitJsonToGraph(x))
 *   2. POST the exported JSON to /api/v1/validate  ->  must be {valid:true}
 * Then, as the ultimate loss-free proof:
 *   3. POST the ORIGINAL sky130_5t_ota to /api/v1/solve (selected=["ac"])
 *   4. POST the ROUND-TRIPPED sky130_5t_ota to /api/v1/solve (selected=["ac"])
 *   5. deep-compare the two "ac" result blocks  ->  must be identical
 *
 * Usage (with the service running on 127.0.0.1:8341):
 *   node scripts/backend_check.mjs
 *
 * The script imports the mapping core straight from src/ via vite-node-free
 * dynamic import of the .ts through tsx? No — we import the compiled behavior by
 * running under `npx vite-node` OR node with a tiny inline JS reimplementation
 * is avoided; instead run this file with `npx vite-node scripts/backend_check.mjs`.
 */
import { readFileSync, readdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { circuitJsonToGraph } from "../src/model/toGraph.ts";
import { graphToCircuitJson } from "../src/model/toJson.ts";
import { deepEqual } from "../src/model/util.ts";

const BASE = process.env.VITE_API_BASE ?? "http://127.0.0.1:8341";
const FIX_DIR = join(
  dirname(fileURLToPath(import.meta.url)),
  "..",
  "src",
  "model",
  "__fixtures__",
);

function roundtrip(json) {
  const { graph, rest } = circuitJsonToGraph(json);
  return graphToCircuitJson(graph, rest);
}

async function post(path, body) {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const text = await res.text();
  let parsed;
  try {
    parsed = JSON.parse(text);
  } catch {
    parsed = text;
  }
  return { status: res.status, body: parsed };
}

async function main() {
  // liveness
  try {
    const h = await fetch(`${BASE}/api/v1/health`);
    const hb = await h.json();
    console.log(`health: ${JSON.stringify(hb)}`);
  } catch (e) {
    console.error(`FATAL: cannot reach service at ${BASE} — is it running?`, String(e));
    process.exit(2);
  }

  const files = readdirSync(FIX_DIR)
    .filter((f) => f.endsWith(".json"))
    .sort();

  let pass = 0;
  const failures = [];
  for (const f of files) {
    const orig = JSON.parse(readFileSync(join(FIX_DIR, f), "utf-8"));
    const rt = roundtrip(orig);
    const { status, body } = await post("/api/v1/validate", rt);
    const ok = status === 200 && body && body.valid === true;
    if (ok) {
      pass += 1;
      console.log(`  validate  ${f.padEnd(28)} valid:true`);
    } else {
      failures.push({ f, status, body });
      console.log(`  validate  ${f.padEnd(28)} FAIL (status ${status}) ${JSON.stringify(body)}`);
    }
  }
  console.log(`\nvalidate: ${pass}/${files.length} round-tripped fixtures valid\n`);

  // ── solve parity on sky130_5t_ota ─────────────────────────────────────
  // The 5T OTA fixture ships without an `analyses` block, so we inject a small
  // `ac` sweep into BOTH the original and the round-tripped circuit (identical
  // injection on each side) — the point is that the mapping is loss-free, so
  // the two solves must produce byte-identical AC results.
  const target = "sky130_5t_ota.json";
  const orig = JSON.parse(readFileSync(join(FIX_DIR, target), "utf-8"));
  const rt = roundtrip(orig);
  const AC = {
    analyses: {
      ac: { freqs: { start: 1000.0, stop: 100000000.0, num: 41, scale: "log" } },
    },
  };
  const origAc = { ...orig, ...AC };
  const rtAc = { ...rt, ...AC };

  const origSolve = await post("/api/v1/solve", { circuit: origAc, selected: ["ac"] });
  const rtSolve = await post("/api/v1/solve", { circuit: rtAc, selected: ["ac"] });

  let solveParity = false;
  if (origSolve.status === 200 && rtSolve.status === 200) {
    const a = origSolve.body?.results?.ac;
    const b = rtSolve.body?.results?.ac;
    const r = deepEqual(a, b);
    solveParity = r.equal;
    if (solveParity) {
      const g = a?.Av_dc_dB ?? a?.gain_dB;
      const bw = a?.bw_Hz;
      console.log(
        `solve parity (${target}, ac): IDENTICAL  ` +
          `(gain=${JSON.stringify(g)} dB, bw=${JSON.stringify(bw)} Hz)`,
      );
    } else {
      console.log(`solve parity (${target}, ac): DIVERGED at ${r.diff}`);
    }
  } else {
    console.log(
      `solve parity (${target}): could not run — ` +
        `orig status ${origSolve.status}, rt status ${rtSolve.status}. ` +
        `orig: ${JSON.stringify(origSolve.body).slice(0, 200)}`,
    );
  }

  const allGood = failures.length === 0 && solveParity;
  console.log(`\n=== ${allGood ? "ALL GOOD" : "PROBLEMS"} ===`);
  process.exit(allGood ? 0 : 1);
}

main();
