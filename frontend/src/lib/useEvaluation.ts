"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { ApiError, getDataset, getEvaluation, getLatestEvaluation, startEvaluation } from "./api";
import type { Dataset, EvaluateRequest, EvaluationRun } from "./types";

/**
 * The evaluation dataset and the newest run: the last completed one when the page opens, then
 * the runs started here, followed with long polling until they end.
 */
export function useEvaluation(onFinished: () => void) {
  const [dataset, setDataset] = useState<Dataset | null>(null);
  const [run, setRun] = useState<EvaluationRun | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);
  const controller = useRef<AbortController | null>(null);

  useEffect(() => {
    let active = true;
    Promise.all([getDataset(), getLatestEvaluation()])
      .then(([loaded, latest]) => {
        if (!active) return;
        setDataset(loaded);
        setRun((current) => current ?? latest);
      })
      .catch((failure) => {
        if (active) setError(messageOf(failure, "Could not load the evaluation dataset."));
      });
    return () => {
      active = false;
      controller.current?.abort();
    };
  }, []);

  const follow = useCallback(
    async (started: EvaluationRun) => {
      const abort = new AbortController();
      controller.current = abort;
      let current = started;
      while (current.status === "running" && !abort.signal.aborted) {
        try {
          current = await getEvaluation(current.id, current.version, abort.signal);
          setRun(current);
        } catch (failure) {
          if (abort.signal.aborted) return;
          setError(messageOf(failure, "Lost track of the evaluation."));
          return;
        }
      }
      if (current.status === "failed") setError(current.error ?? "The evaluation failed.");
      onFinished();
    },
    [onFinished],
  );

  const start = useCallback(
    async (request: EvaluateRequest) => {
      controller.current?.abort();
      setError(null);
      setStarting(true);
      try {
        const started = await startEvaluation(request);
        setRun(started);
        void follow(started);
      } catch (failure) {
        setError(messageOf(failure, "Could not start the evaluation."));
      } finally {
        setStarting(false);
      }
    },
    [follow],
  );

  const running = starting || run?.status === "running";
  return { dataset, run, error, running, start };
}

function messageOf(error: unknown, fallback: string): string {
  return error instanceof ApiError ? error.message : fallback;
}
