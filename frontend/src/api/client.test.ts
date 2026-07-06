/**
 * client.ts resolves API_BASE once, at module load, in three tiers:
 *   1. window.__CIRCUITOPT_API_BASE__  (Tauri shell injection)
 *   2. import.meta.env.VITE_API_BASE   (web override)
 *   3. http://127.0.0.1:8341           (default)
 *
 * Because the value is captured at import time, each case sets up the
 * environment first, then does a fresh dynamic import via `vi.resetModules()`.
 * Globals are restored after every case so nothing leaks between tests (or into
 * the rest of the suite).
 */
import { afterEach, describe, expect, it, vi } from "vitest";

const DEFAULT = "http://127.0.0.1:8341";

async function loadApiBase(): Promise<string> {
  vi.resetModules();
  const mod = await import("./client");
  return mod.API_BASE;
}

afterEach(() => {
  // Drop any injected global and clear the env override we may have set.
  delete (globalThis as { window?: unknown }).window;
  vi.unstubAllEnvs();
  vi.resetModules();
});

describe("API_BASE three-tier resolution", () => {
  it("prefers the Tauri-injected window global over everything else", async () => {
    vi.stubEnv("VITE_API_BASE", "http://env:9000");
    (globalThis as { window?: unknown }).window = {
      __CIRCUITOPT_API_BASE__: "http://injected:7777",
    };
    expect(await loadApiBase()).toBe("http://injected:7777");
  });

  it("falls back to VITE_API_BASE when no window global is set", async () => {
    vi.stubEnv("VITE_API_BASE", "http://env:9000");
    // No window at all (pure Node/test context).
    expect(await loadApiBase()).toBe("http://env:9000");
  });

  it("uses VITE_API_BASE when the injected global is present but empty", async () => {
    vi.stubEnv("VITE_API_BASE", "http://env:9000");
    (globalThis as { window?: unknown }).window = {
      __CIRCUITOPT_API_BASE__: "",
    };
    expect(await loadApiBase()).toBe("http://env:9000");
  });

  it("falls back to the default when neither is set", async () => {
    expect(await loadApiBase()).toBe(DEFAULT);
  });

  it("ignores a non-string injected value and uses the default", async () => {
    (globalThis as { window?: unknown }).window = {
      __CIRCUITOPT_API_BASE__: 12345,
    };
    expect(await loadApiBase()).toBe(DEFAULT);
  });
});
