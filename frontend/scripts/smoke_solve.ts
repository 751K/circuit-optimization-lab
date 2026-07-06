/**
 * F3 end-to-end smoke: exercise the real solve → transform pipeline the Run
 * panel drives, against the live backend. Not part of `npm test`; run with:
 *
 *   node_modules/.bin/vite-node scripts/smoke_solve.ts
 *
 * (needs the backend on 127.0.0.1:8341). Fetches an ac+noise solve of the
 * periodic_rc example and asserts our transforms produce finite Bode / PSD
 * series on the actual response.
 */
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { toBode, toNoiseSpectra, isFiniteNum } from "../src/results/transform";
import { prepareSolveCircuit } from "../src/panels/runConfig";
import type { CircuitJson } from "../src/model/circuit";
import type { AcResult, NoiseResult } from "../src/results/types";

const BASE = process.env.VITE_API_BASE ?? "http://127.0.0.1:8341";
const root = join(dirname(fileURLToPath(import.meta.url)), "..");

function assert(cond: boolean, msg: string): void {
  if (!cond) throw new Error(`ASSERT FAILED: ${msg}`);
}

async function main(): Promise<void> {
  // The panel's exportJson() shape — reuse the model fixture as a stand-in.
  const circuit = JSON.parse(
    readFileSync(join(root, "src/model/__fixtures__/periodic_rc.json"), "utf8"),
  );

  const res = await fetch(`${BASE}/api/v1/solve`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ circuit, selected: ["ac", "noise"] }),
  });
  assert(res.ok, `solve HTTP ${res.status}`);
  const { results, elapsed_s } = (await res.json()) as {
    results: Record<string, unknown>;
    elapsed_s: number;
  };
  console.log("top-level result keys:", Object.keys(results), "elapsed_s:", elapsed_s);

  // ── ac → Bode ──
  const { magnitude, phase } = toBode(results.ac as AcResult);
  assert(magnitude.length > 0, "empty Bode magnitude");
  assert(magnitude.every(([, y]) => y === null || isFiniteNum(y)), "non-finite Bode mag");
  assert(phase.every(([, y]) => y === null || isFiniteNum(y)), "non-finite Bode phase");
  assert(isFiniteNum(magnitude[0]![1]), "first Bode point not finite");
  console.log(
    `Bode OK: ${magnitude.length} pts, mag[0]=${magnitude[0]![1]!.toFixed(4)} dB, phase[0]=${phase[0]![1]!.toFixed(2)}°`,
  );

  // ── noise → PSD ──
  const { series } = toNoiseSpectra(results.noise as NoiseResult);
  assert(series.length > 0, "empty noise spectra");
  const out = series.find((s) => s.key === "out_psd");
  assert(!!out, "no out_psd series");
  assert(out!.points.every(([, y]) => y === null || (isFiniteNum(y) && y > 0)), "bad PSD value");
  console.log(
    `PSD OK: ${series.length} curves (${series.map((s) => s.key).join(", ")}), out_psd[0]=${out!.points[0]![1]}`,
  );

  // ── no-analyses-block circuit → default-injection path (runConfig.ts) ──
  const ota = JSON.parse(
    readFileSync(join(root, "src/model/__fixtures__/sky130_5t_ota.json"), "utf8"),
  ) as CircuitJson;
  assert(!("analyses" in ota), "expected sky130_5t_ota to have no analyses block");
  const prep = prepareSolveCircuit(ota, ["ac"]);
  assert(prep.injected.length === 1 && prep.injected[0] === "ac", "ac default not injected");
  assert(prep.missing.length === 0, "unexpected missing analyses");
  assert(!("analyses" in ota), "input circuit was mutated (Export JSON polluted)");
  const res2 = await fetch(`${BASE}/api/v1/solve`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ circuit: prep.circuit, selected: ["ac"] }),
  });
  assert(res2.ok, `injected-default solve HTTP ${res2.status}`);
  const body2 = (await res2.json()) as { results: Record<string, unknown> };
  const bode2 = toBode(body2.results.ac as AcResult);
  assert(bode2.magnitude.length === 111, `expected 111 Bode pts, got ${bode2.magnitude.length}`);
  assert(
    bode2.magnitude.every(([f, y]) => f > 0 && (y === null || isFiniteNum(y))),
    "non-finite Bode point on injected sweep",
  );
  const peak = Math.max(...bode2.magnitude.map(([, y]) => (y === null ? -Infinity : y)));
  console.log(
    `Injected-default OK: sky130_5t_ota (no analyses block) → ${bode2.magnitude.length} Bode pts, peak ${peak.toFixed(3)} dB`,
  );

  console.log("\nSMOKE PASS");
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
