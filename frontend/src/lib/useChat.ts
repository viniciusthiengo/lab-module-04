"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { ApiError, query, search } from "./api";
import type { Mode, QueryResult, SearchResult, Turn } from "./types";

export interface AskOptions {
  mode: Mode;
  k: number;
  rerank: boolean;
  /** Claude answers from the retrieved code; otherwise the code is only retrieved (free). */
  answer: boolean;
  /** Codebases to search; every indexed one when empty. */
  codebases: string[];
}

export type Exchange = {
  id: number;
  question: string;
  options: AskOptions;
} & (
  | { status: "pending" }
  | { status: "answered"; result: QueryResult }
  | { status: "retrieved"; result: SearchResult }
  | { status: "failed"; error: string }
);

/** Turns of conversation sent with a question, so that follow-ups can refer to them. */
export const HISTORY_EXCHANGES = 3;

/** The conversation: questions and their answers, or the code retrieved for them. */
export function useChat() {
  const [exchanges, setExchanges] = useState<Exchange[]>([]);
  const controller = useRef<AbortController | null>(null);
  const nextId = useRef(1);

  useEffect(() => () => controller.current?.abort(), []);

  const ask = useCallback(
    async (question: string, options: AskOptions) => {
      const id = nextId.current++;
      const history = historyOf(exchanges);
      setExchanges((current) => [...current, { id, question, options, status: "pending" }]);
      const abort = new AbortController();
      controller.current = abort;
      const request = {
        question,
        k: options.k,
        mode: options.mode,
        rerank: options.rerank,
        ...(options.codebases.length ? { codebases: options.codebases } : {}),
      };
      let update: Exchange;
      try {
        update = options.answer
          ? {
              id,
              question,
              options,
              status: "answered",
              result: await query({ ...request, history }, abort.signal),
            }
          : {
              id,
              question,
              options,
              status: "retrieved",
              result: await search(request, abort.signal),
            };
      } catch (error) {
        if (abort.signal.aborted) return;
        const message = error instanceof ApiError ? error.message : "The question failed.";
        update = { id, question, options, status: "failed", error: message };
      }
      setExchanges((current) =>
        current.map((exchange) => (exchange.id === id ? update : exchange)),
      );
    },
    [exchanges],
  );

  const clear = useCallback(() => {
    controller.current?.abort();
    setExchanges([]);
  }, []);

  const busy = exchanges.some((exchange) => exchange.status === "pending");
  return { exchanges, busy, ask, clear };
}

/** The last answered exchanges as conversation turns, oldest first. */
export function historyOf(exchanges: Exchange[]): Turn[] {
  return exchanges
    .filter((exchange) => exchange.status === "answered")
    .slice(-HISTORY_EXCHANGES)
    .flatMap((exchange) => [
      { role: "user" as const, content: exchange.question },
      { role: "assistant" as const, content: exchange.result.answer.slice(0, 4_000) },
    ]);
}
