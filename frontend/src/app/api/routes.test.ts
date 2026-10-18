import { beforeEach, describe, expect, it, vi } from "vitest";

import { DELETE, GET, POST } from "./[...path]/route";

const backend = vi.fn<typeof fetch>();
const RUN = "AbC123_def";

beforeEach(() => {
  vi.stubEnv("BACKEND_URL", "https://api.example.com/");
  vi.stubGlobal("fetch", backend);
  backend.mockReset();
});

function call(handler: typeof GET, path: string, init: RequestInit = {}): Promise<Response> {
  const url = new URL(`http://localhost/api/${path}`);
  const segments = url.pathname.slice("/api/".length).split("/");
  return handler(new Request(url, init), { params: Promise.resolve({ path: segments }) });
}

function sent(index = 0): { url: string; init: RequestInit; headers: Record<string, string> } {
  const [url, init = {}] = backend.mock.calls[index];
  return { url: String(url), init, headers: init.headers as Record<string, string> };
}

describe("the /api proxy", () => {
  it("forwards a question with the visitor's address", async () => {
    backend.mockResolvedValue(Response.json({ answer: "Hashed [1]." }));

    const response = await call(POST, "query", {
      method: "POST",
      body: '{"question":"How?"}',
      headers: { "x-forwarded-for": "203.0.113.7, 10.0.0.1" },
    });

    expect(response.status).toBe(200);
    expect(await response.json()).toEqual({ answer: "Hashed [1]." });
    const { url, init, headers } = sent();
    expect(url).toBe("https://api.example.com/query");
    expect(init.method).toBe("POST");
    expect(init.body).toBe('{"question":"How?"}');
    expect(headers["X-Client-IP"]).toBe("203.0.113.7");
    expect(headers["Content-Type"]).toBe("application/json");
  });

  it("forwards reads and deletions", async () => {
    backend.mockResolvedValueOnce(Response.json([{ name: "shop" }]));
    backend.mockResolvedValueOnce(new Response(null, { status: 204 }));

    const list = await call(GET, "codebases");
    const removed = await call(DELETE, "codebases/shop-api.v2", { method: "DELETE" });

    expect(await list.json()).toEqual([{ name: "shop" }]);
    expect(removed.status).toBe(204);
    expect(sent(1).url).toBe("https://api.example.com/codebases/shop-api.v2");
    expect(sent(1).init.body).toBeUndefined();
  });

  it("only forwards the allowed routes and methods", async () => {
    const unknown = await call(GET, "admin/secrets");
    const badName = await call(DELETE, "codebases/Bad%20Name", { method: "DELETE" });
    const wrongMethod = await call(GET, "query");

    expect(unknown.status).toBe(404);
    expect(badName.status).toBe(404);
    expect(wrongMethod.status).toBe(405);
    expect(backend).not.toHaveBeenCalled();
  });

  it("refuses a body that is not JSON without calling the backend", async () => {
    const response = await call(POST, "index/files", { method: "POST", body: "nope" });

    expect(response.status).toBe(400);
    expect((await response.json()).detail).toBe("Send a JSON body.");
    expect(backend).not.toHaveBeenCalled();
  });

  it("long polls an evaluation run, and gives the backend the time it waits", async () => {
    backend.mockImplementation(async () => Response.json({ id: RUN, status: "running" }));

    await call(GET, `evaluations/${RUN}?since=3&wait=20`);
    await call(GET, `evaluations/${RUN}?ignored=1`);

    expect(sent(0).url).toBe(`https://api.example.com/evaluations/${RUN}?since=3&wait=20`);
    expect(sent(1).url).toBe(`https://api.example.com/evaluations/${RUN}`);
    const latest = await call(GET, "evaluations/latest?since=1");
    expect(latest.status).toBe(200);
    expect(sent(2).url).toBe("https://api.example.com/evaluations/latest");
  });

  it.each([
    ["since=abc", "'since' must be a version number."],
    ["since=1&wait=99", "'wait' must be between 0 and 25 seconds."],
    ["since=1&wait=-1", "'wait' must be between 0 and 25 seconds."],
  ])("refuses an invalid long poll (%s)", async (query, message) => {
    const response = await call(GET, `evaluations/${RUN}?${query}`);

    expect(response.status).toBe(400);
    expect((await response.json()).detail).toBe(message);
  });

  it("passes the backend's errors through", async () => {
    backend.mockResolvedValue(Response.json({ detail: "Add at least one file." }, { status: 422 }));

    const response = await call(POST, "index/files", { method: "POST", body: "{}" });

    expect(response.status).toBe(422);
    expect((await response.json()).detail).toBe("Add at least one file.");
  });

  it("explains timeouts, an unreachable backend and a missing configuration", async () => {
    const post = () => call(POST, "search", { method: "POST", body: "{}" });

    backend.mockRejectedValueOnce(new DOMException("The operation timed out.", "TimeoutError"));
    expect((await post()).status).toBe(504);

    backend.mockRejectedValueOnce(new TypeError("fetch failed"));
    expect((await post()).status).toBe(502);

    backend.mockResolvedValueOnce(new Response("<html>oops</html>", { status: 500 }));
    expect((await post()).status).toBe(502);

    vi.stubEnv("BACKEND_URL", "");
    const missing = await post();
    expect(missing.status).toBe(500);
    expect((await missing.json()).detail).toContain("BACKEND_URL");
  });
});
