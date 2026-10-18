// Server-side calls from the /api route to the FastAPI backend (BACKEND_URL). The browser only
// talks to this app, so the backend needs no CORS and its address stays private. Only the
// routes below are forwarded.

type Method = "GET" | "POST" | "DELETE";

interface Route {
  method: Method;
  /** The path after /api/. */
  path: RegExp;
  /** Seconds the backend may take to answer. */
  timeout: number;
  /** Accepts `?since=<version>&wait=<seconds>` (long polling). */
  longPoll?: boolean;
}

export const MAX_WAIT_SECONDS = 25;

export const ROUTES: readonly Route[] = [
  { method: "GET", path: /^health$/, timeout: 15 },
  { method: "GET", path: /^codebases$/, timeout: 15 },
  { method: "DELETE", path: /^codebases\/[a-z0-9][a-z0-9._-]{0,47}$/, timeout: 30 },
  { method: "POST", path: /^index\/files$/, timeout: 240 },
  { method: "POST", path: /^index\/demo$/, timeout: 120 },
  { method: "POST", path: /^search$/, timeout: 60 },
  { method: "POST", path: /^query$/, timeout: 120 },
  { method: "GET", path: /^evaluation\/dataset$/, timeout: 15 },
  { method: "POST", path: /^evaluations$/, timeout: 60 },
  { method: "GET", path: /^evaluations\/latest$/, timeout: 15 },
  { method: "GET", path: /^evaluations\/[A-Za-z0-9_-]{8,32}$/, timeout: 15, longPoll: true },
];

/** The backend's base URL without a trailing slash, or `null` when it is not configured. */
export function backendUrl(): string | null {
  const value = process.env.BACKEND_URL?.trim();
  return value ? value.replace(/\/+$/, "") : null;
}

/** The visitor's address, forwarded so that the backend can rate-limit per person. */
export function clientIp(request: Request): string | null {
  const forwarded = request.headers.get("x-forwarded-for")?.split(",")[0]?.trim();
  return forwarded || request.headers.get("x-real-ip");
}

/** A `{ detail }` error answer, like the backend's. */
export function failure(status: number, detail: string): Response {
  return Response.json({ detail }, { status });
}

/** Forwards `request` for the API path `segments` when it is an allowed route. */
export async function proxy(request: Request, segments: string[]): Promise<Response> {
  const path = segments.join("/");
  const method = request.method as Method;
  const matching = ROUTES.filter((route) => route.path.test(path));
  if (matching.length === 0) return failure(404, "Unknown API route.");
  const route = matching.find((candidate) => candidate.method === method);
  if (!route) return failure(405, `${request.method} is not allowed here.`);

  let query = "";
  let timeout = route.timeout;
  if (route.longPoll) {
    const params = new URL(request.url).searchParams;
    const since = params.get("since");
    const wait = Number(params.get("wait") ?? 0);
    if (since !== null && !/^\d+$/.test(since)) {
      return failure(400, "'since' must be a version number.");
    }
    if (!Number.isFinite(wait) || wait < 0 || wait > MAX_WAIT_SECONDS) {
      return failure(400, `'wait' must be between 0 and ${MAX_WAIT_SECONDS} seconds.`);
    }
    if (since !== null) {
      query = `?since=${since}&wait=${wait}`;
      timeout += wait;
    }
  }

  let body: unknown;
  if (method === "POST") {
    body = await jsonBody(request);
    if (body === undefined) return failure(400, "Send a JSON body.");
  }
  return forward(`/${path}${query}`, request, { method, body, timeoutMs: timeout * 1000 });
}

/** The JSON body of `request`, or `undefined` when it is not JSON. */
async function jsonBody(request: Request): Promise<unknown> {
  try {
    return await request.json();
  } catch {
    return undefined;
  }
}

/**
 * Calls `path` on the backend and returns its JSON answer with the same status. Configuration,
 * network, timeout and format problems become `{ detail }` errors.
 */
async function forward(
  path: string,
  request: Request,
  { method, body, timeoutMs }: { method: Method; body?: unknown; timeoutMs: number },
): Promise<Response> {
  const base = backendUrl();
  if (!base) return failure(500, "The server is not configured: BACKEND_URL is missing.");
  const ip = clientIp(request);
  let response: Response;
  try {
    response = await fetch(`${base}${path}`, {
      method,
      headers: {
        Accept: "application/json",
        ...(body === undefined ? {} : { "Content-Type": "application/json" }),
        ...(ip ? { "X-Client-IP": ip } : {}),
      },
      body: body === undefined ? undefined : JSON.stringify(body),
      cache: "no-store",
      signal: AbortSignal.timeout(timeoutMs),
    });
  } catch (error) {
    const timedOut = error instanceof DOMException && error.name === "TimeoutError";
    return timedOut
      ? failure(504, "The RAG service took too long to answer. Try again.")
      : failure(502, "The RAG service is unreachable. Try again in a moment.");
  }
  if (response.status === 204) return new Response(null, { status: 204 });
  const answer: unknown = await response.json().catch(() => null);
  if (answer === null) return failure(502, "The RAG service sent an unexpected answer.");
  return Response.json(answer, { status: response.status });
}
