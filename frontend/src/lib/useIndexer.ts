"use client";

import { useCallback, useState } from "react";

import { ApiError, indexFiles } from "./api";
import { batches, type Candidate, readSelection, type Selection } from "./files";
import type { IndexResult } from "./types";

export interface IndexSummary {
  codebase: string;
  files: number;
  chunks: number;
  embedded: number;
  reused: number;
  removed: number;
  durationMs: number;
}

export type IndexerState =
  | { phase: "idle" }
  | { phase: "reading" }
  | { phase: "ready"; selection: Selection }
  | { phase: "indexing"; selection: Selection; batch: number; batches: number }
  | { phase: "done"; selection: Selection; summary: IndexSummary }
  | { phase: "failed"; selection: Selection | null; error: string };

/**
 * Reads the chosen files, then indexes them in batches. With `replace`, the last batch removes
 * the codebase's files that the upload did not send.
 */
export function useIndexer(onIndexed: () => void) {
  const [state, setState] = useState<IndexerState>({ phase: "idle" });

  const choose = useCallback(async (candidates: Candidate[]) => {
    setState({ phase: "reading" });
    try {
      setState({ phase: "ready", selection: await readSelection(candidates) });
    } catch {
      setState({ phase: "failed", selection: null, error: "Could not read the chosen files." });
    }
  }, []);

  const index = useCallback(
    async (selection: Selection, codebase: string, replace: boolean) => {
      const groups = batches(selection.files);
      const results: IndexResult[] = [];
      try {
        for (const [number, files] of groups.entries()) {
          setState({ phase: "indexing", selection, batch: number + 1, batches: groups.length });
          const last = number === groups.length - 1;
          const keep =
            last && replace
              ? groups
                  .slice(0, -1)
                  .flat()
                  .map((file) => file.path)
              : [];
          results.push(await indexFiles({ codebase, files, replace: last && replace, keep }));
        }
        setState({ phase: "done", selection, summary: summarize(codebase, results) });
      } catch (error) {
        const message = error instanceof ApiError ? error.message : "Indexing failed.";
        const done = results.length ? ` (${results.length} of ${groups.length} batches done)` : "";
        setState({ phase: "failed", selection, error: `${message}${done}` });
      } finally {
        if (results.length) onIndexed();
      }
    },
    [onIndexed],
  );

  const reset = useCallback(() => setState({ phase: "idle" }), []);

  return { state, choose, index, reset };
}

function summarize(codebase: string, results: IndexResult[]): IndexSummary {
  const total = (pick: (result: IndexResult) => number) =>
    results.reduce((sum, result) => sum + pick(result), 0);
  return {
    codebase,
    files: total((result) => result.files),
    chunks: total((result) => result.chunks),
    embedded: total((result) => result.embedded),
    reused: total((result) => result.reused),
    removed: total((result) => result.removed),
    durationMs: total((result) => result.duration_ms),
  };
}
