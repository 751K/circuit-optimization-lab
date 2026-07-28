/**
 * Simulation session state: the current run, and any background sweeps.
 *
 * This lives in the store rather than in the Run panel because the controls and
 * the results are now in different places on screen — buttons on the right, the
 * results dock along the bottom. Component state cannot span that.
 *
 * A job is tracked here from submission to terminal state. The WebSocket
 * subscription that feeds it is *not* stored: sockets are not serialisable state
 * and there is exactly one owner (the dock), which unsubscribes on unmount.
 */
import { create } from "zustand";
import {
  ApiError,
  cancelJob,
  getJob,
  isTerminal,
  solve,
  submitMc,
  submitPvt,
  watchJob,
  type JobProgress,
  type JobSnapshot,
  type JobStatus,
  type PvtRequest,
  type SignoffResponse,
} from "../api/client";
import type { CircuitJson } from "../model/circuit";

export interface RunError {
  stage: string;
  message: string;
}

export interface RunState {
  status: "idle" | "running" | "done" | "error";
  /** Per-analysis results from the last successful solve. */
  results: Record<string, unknown> | null;
  signoff: SignoffResponse | null;
  elapsed: number | null;
  error: RunError | null;
  /** Analyses whose config was defaulted for the request only. */
  defaulted: string[];
  /** The corner the last run used, for labelling the results. */
  corner: string;
}

export interface SweepState {
  jobId: string | null;
  kind: "pvt" | "mc" | null;
  status: JobStatus | "idle";
  progress: JobProgress | null;
  result: Record<string, unknown> | null;
  error: RunError | null;
  /** Wall-clock start, for an elapsed readout while it runs. */
  startedAt: number | null;
}

const IDLE_RUN: RunState = {
  status: "idle",
  results: null,
  signoff: null,
  elapsed: null,
  error: null,
  defaulted: [],
  corner: "",
};

const IDLE_SWEEP: SweepState = {
  jobId: null,
  kind: null,
  status: "idle",
  progress: null,
  result: null,
  error: null,
  startedAt: null,
};

export interface SessionState {
  run: RunState;
  sweep: SweepState;

  runSolve: (
    circuit: CircuitJson,
    selected: string[],
    corner: string,
    defaulted: string[],
  ) => Promise<void>;
  clearRun: () => void;

  startPvt: (circuit: CircuitJson, options: PvtRequest) => Promise<void>;
  startMc: (
    circuit: CircuitJson,
    options: { n?: number; seed?: number; corner?: string; workers?: number },
  ) => Promise<void>;
  stopSweep: () => Promise<void>;
  clearSweep: () => void;
  /** Attach the progress stream for the running job; returns an unsubscribe. */
  watchSweep: () => () => void;
}

function toRunError(e: unknown): RunError {
  if (e instanceof ApiError) return { stage: e.stage, message: e.message };
  return {
    stage: "network",
    message: e instanceof Error ? e.message : String(e),
  };
}

export const useSession = create<SessionState>((set, get) => {
  /** Pull a terminal job's full payload over HTTP and land it in the store. */
  const settle = async (jobId: string): Promise<void> => {
    try {
      const snapshot: JobSnapshot = await getJob(jobId);
      if (get().sweep.jobId !== jobId) return;   // superseded by a newer sweep
      set({
        sweep: {
          ...get().sweep,
          status: snapshot.status,
          result: snapshot.result ?? null,
          error: snapshot.error
            ? { stage: snapshot.error.stage, message: snapshot.error.message }
            : null,
        },
      });
    } catch (e) {
      if (get().sweep.jobId !== jobId) return;
      set({ sweep: { ...get().sweep, status: "failed", error: toRunError(e) } });
    }
  };

  const submit = async (
    kind: "pvt" | "mc",
    request: () => Promise<JobSnapshot>,
  ): Promise<void> => {
    set({
      sweep: {
        ...IDLE_SWEEP,
        kind,
        status: "queued",
        startedAt: Date.now(),
      },
    });
    try {
      const job = await request();
      set({ sweep: { ...get().sweep, jobId: job.job_id, status: job.status } });
    } catch (e) {
      set({ sweep: { ...get().sweep, status: "failed", error: toRunError(e) } });
    }
  };

  return {
    run: IDLE_RUN,
    sweep: IDLE_SWEEP,

    runSolve: async (circuit, selected, corner, defaulted) => {
      set({ run: { ...IDLE_RUN, status: "running", corner } });
      try {
        const response = await solve(
          circuit,
          selected.length ? selected : undefined,
          corner || undefined,
        );
        set({
          run: {
            status: "done",
            results: response.results,
            signoff: response.signoff ?? null,
            elapsed: response.elapsed_s,
            error: null,
            defaulted,
            corner,
          },
        });
      } catch (e) {
        set({ run: { ...IDLE_RUN, status: "error", error: toRunError(e), corner } });
      }
    },

    clearRun: () => set({ run: IDLE_RUN }),

    startPvt: (circuit, options) =>
      submit("pvt", () => submitPvt(circuit, options)),

    startMc: (circuit, options) =>
      submit("mc", () => submitMc(circuit, options)),

    stopSweep: async () => {
      const { jobId } = get().sweep;
      if (!jobId) return;
      try {
        await cancelJob(jobId);
      } catch {
        // Cancellation is best-effort; a failure to request it is not worth
        // replacing the sweep's own state with a network error.
      }
    },

    clearSweep: () => set({ sweep: IDLE_SWEEP }),

    watchSweep: () => {
      const { jobId, status } = get().sweep;
      if (!jobId || isTerminal(status as JobStatus)) return () => {};

      let polling: ReturnType<typeof setInterval> | null = null;
      const stopPolling = (): void => {
        if (polling !== null) clearInterval(polling);
        polling = null;
      };

      // The stream carries progress; the outcome is always read back over HTTP,
      // so a dropped socket degrades to slower updates rather than a lost result.
      const startPolling = (): void => {
        if (polling !== null) return;
        polling = setInterval(() => {
          void (async () => {
            if (get().sweep.jobId !== jobId) return stopPolling();
            try {
              const snapshot = await getJob(jobId);
              if (get().sweep.jobId !== jobId) return stopPolling();
              set({
                sweep: {
                  ...get().sweep,
                  status: snapshot.status,
                  progress: snapshot.progress ?? get().sweep.progress,
                },
              });
              if (isTerminal(snapshot.status)) {
                stopPolling();
                await settle(jobId);
              }
            } catch {
              /* transient; the next tick retries */
            }
          })();
        }, 1000);
      };

      const unwatch = watchJob(jobId, {
        onProgress: (progress) => {
          if (get().sweep.jobId !== jobId) return;
          set({ sweep: { ...get().sweep, status: "running", progress } });
        },
        onTerminal: (frame) => {
          if (get().sweep.jobId !== jobId) return;
          set({ sweep: { ...get().sweep, status: frame.status } });
          void settle(jobId);
        },
        onFallback: startPolling,
      });

      return () => {
        stopPolling();
        unwatch();
      };
    },
  };
});
