const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";
const TIMEOUT_MS = 10_000;

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
  }
}

let accessToken: string | null = null;

export function setAccessToken(token: string | null): void {
  accessToken = token;
}

async function refreshAccessToken(): Promise<string | null> {
  const response = await fetch(`${API_URL}/auth/refresh`, { method: "POST", credentials: "include" });
  if (!response.ok) return null;
  const { access_token } = await response.json();
  setAccessToken(access_token);
  return access_token;
}

/**
 * Calls the API with the bearer token. On a 401 it refreshes the token once and retries the
 * request; every request aborts after 10 seconds.
 */
export async function request<T>(path: string, init: RequestInit = {}, retried = false): Promise<T> {
  const headers = new Headers(init.headers);
  if (accessToken) headers.set("Authorization", `Bearer ${accessToken}`);
  const response = await fetch(`${API_URL}${path}`, {
    ...init,
    headers,
    signal: AbortSignal.timeout(TIMEOUT_MS),
  });
  if (response.status === 401 && !retried && (await refreshAccessToken())) {
    return request<T>(path, init, true);
  }
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new ApiError(body.detail ?? body.error ?? response.statusText, response.status);
  }
  return response.json() as Promise<T>;
}
