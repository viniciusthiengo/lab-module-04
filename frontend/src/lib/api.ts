// The browser's calls to this app's /api proxy, which forwards them to the backend.

import type {
  Codebase,
  Dataset,
  EvaluateRequest,
  EvaluationRun,
  Health,
  IndexRequest,
  IndexResult,
  QueryRequest,
  QueryResult,
  SearchRequest,
  SearchResult,
} from "./types";

/** A failed call, with a message that can be shown to the user. */
export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

type Fetch = typeof fetch;

/** Long polling: the server answers as soon as the run changes, or after this long. */
export const WAIT_SECONDS = 20;

export function getHealth(fetcher: Fetch = fetch) {
  return send<Health>("/api/health", { method: "GET" }, fetcher);
}

export function getCodebases(fetcher: Fetch = fetch) {
  return send<Codebase[]>("/api/codebases", { method: "GET" }, fetcher);
}

export function deleteCodebase(name: string, fetcher: Fetch = fetch) {
  return send<null>(`/api/codebases/${encodeURIComponent(name)}`, { method: "DELETE" }, fetcher);
}

export function indexFiles(request: IndexRequest, fetcher: Fetch = fetch) {
  return send<IndexResult>("/api/index/files", { method: "POST", body: request }, fetcher);
}

export function indexDemo(fetcher: Fetch = fetch) {
  return send<IndexResult[]>("/api/index/demo", { method: "POST", body: {} }, fetcher);
}

/** The chunks that best match the question, without an answer (free without reranking). */
export function search(request: SearchRequest, signal?: AbortSignal, fetcher: Fetch = fetch) {
  return send<SearchResult>("/api/search", { method: "POST", body: request, signal }, fetcher);
}

/** Claude's answer from the retrieved code, with its sources. */
export function query(request: QueryRequest, signal?: AbortSignal, fetcher: Fetch = fetch) {
  return send<QueryResult>("/api/query", { method: "POST", body: request, signal }, fetcher);
}

export function getDataset(fetcher: Fetch = fetch) {
  return send<Dataset>("/api/evaluation/dataset", { method: "GET" }, fetcher);
}

/** Starts the evaluation suite in the background. */
export function startEvaluation(request: EvaluateRequest, fetcher: Fetch = fetch) {
  return send<EvaluationRun>("/api/evaluations", { method: "POST", body: request }, fetcher);
}

/** The newest completed evaluation, or `null` when there is none yet. */
export async function getLatestEvaluation(fetcher: Fetch = fetch): Promise<EvaluationRun | null> {
  try {
    return await send<EvaluationRun>("/api/evaluations/latest", { method: "GET" }, fetcher);
  } catch (error) {
    if (error instanceof ApiError && error.status === 404) return null;
    throw error;
  }
}

/** The run, once its version is newer than `since` (or after `WAIT_SECONDS`). */
export function getEvaluation(
  id: string,
  since: number | null,
  signal?: AbortSignal,
  fetcher: Fetch = fetch,
) {
  const query = since === null ? "" : `?since=${since}&wait=${WAIT_SECONDS}`;
  return send<EvaluationRun>(
    `/api/evaluations/${encodeURIComponent(id)}${query}`,
    { method: "GET", signal },
    fetcher,
  );
}

/** The message to show for a failed answer with `status` and `body`. */
export function errorMessage(status: number, body: unknown): string {
  if (body && typeof body === "object" && "detail" in body && typeof body.detail === "string") {
    return body.detail;
  }
  if (status === 404) return "Not found on the server.";
  if (status === 413) return "The request is too large; send fewer files at a time.";
  if (status === 504) return "The server took too long to answer. Try again.";
  if (status >= 500) return "The RAG service is unavailable. Try again in a moment.";
  return "Something went wrong. Try again.";
}

async function send<T>(
  path: string,
  {
    method,
    body,
    signal,
  }: { method: "GET" | "POST" | "DELETE"; body?: unknown; signal?: AbortSignal },
  fetcher: Fetch,
): Promise<T> {
  let response: Response;
  try {
    response = await fetcher(path, {
      method,
      headers: body === undefined ? undefined : { "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
      signal,
      cache: "no-store",
    });
  } catch (error) {
    if (signal?.aborted) throw error;
    throw new ApiError("Could not reach the server. Check your connection and try again.", 0);
  }
  if (response.status === 204) return null as T;
  const answer: unknown = await response.json().catch(() => null);
  if (!response.ok) throw new ApiError(errorMessage(response.status, answer), response.status);
  return answer as T;
}
