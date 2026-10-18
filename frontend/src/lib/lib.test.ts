import { describe, expect, it, vi } from "vitest";

import { ApiError, errorMessage, getLatestEvaluation, indexFiles, search } from "./api";
import { citedSource, linkCitations } from "./citations";
import {
  batches,
  type Candidate,
  cleanPath,
  codebaseName,
  MAX_BATCH_FILES,
  readSelection,
  skipReason,
  skipSummary,
  splitRoot,
} from "./files";
import { duration, lines, percent, plural, score, splitKey, usd } from "./format";
import { best, estimateCost, firstRelevantRank, isRelevant, matches } from "./metrics";
import type { EvaluationReport, QueryResult } from "./types";
import { type Exchange, historyOf } from "./useChat";

function candidate(path: string, content: string, size = content.length): Candidate {
  return { path, size, text: () => Promise.resolve(content) };
}

describe("choosing the files to index", () => {
  it.each([
    ["src/app.py", null],
    ["Dockerfile", null],
    ["web/src/App.TSX", null],
    ["node_modules/react/index.js", "dependency or build output"],
    ["src/__pycache__/app.py", "dependency or build output"],
    ["package-lock.json", "lock file"],
    ["logo.png", "unsupported type"],
    [".env", "unsupported type"],
    ["src/my file (1).py", "unsupported name"],
  ])("skipReason(%s) is %s", (path, reason) => {
    expect(skipReason(path)).toBe(reason);
  });

  it("cleans paths", () => {
    expect(cleanPath(".\\src\\app.py")).toBe("src/app.py");
    expect(cleanPath("/shop//src/app.py")).toBe("shop/src/app.py");
  });

  it("finds the folder that was picked", () => {
    expect(splitRoot(["shop/a.py", "shop/src/b.py"])).toEqual({
      root: "shop",
      paths: ["a.py", "src/b.py"],
    });
    expect(splitRoot(["a.py", "shop/b.py"]).root).toBeNull();
    expect(splitRoot(["shop/a.py", "web/b.py"]).root).toBeNull();
  });

  it("reads the supported files and says why the others are skipped", async () => {
    const selection = await readSelection([
      candidate("shop/src/app.py", "print('hi')\n"),
      candidate("shop/README.md", "# Shop\n"),
      candidate("shop/node_modules/x/index.js", "x"),
      candidate("shop/logo.png", "png"),
      candidate("shop/empty.py", "  \n"),
      candidate("shop/data.json", "a\0b"),
      candidate("shop/huge.txt", "x", 10_000_000),
      candidate("shop/./src/app.py", "again"),
    ]);

    expect(selection.root).toBe("shop");
    expect(selection.files).toEqual([
      { path: "src/app.py", content: "print('hi')\n" },
      { path: "README.md", content: "# Shop\n" },
    ]);
    expect(selection.skipped.map(({ reason }) => reason)).toEqual([
      "dependency or build output",
      "unsupported type",
      "empty",
      "binary",
      "too large",
      "duplicate",
    ]);
    expect(skipSummary(selection.skipped)).toBe(
      "1 dependency or build output, 1 unsupported type, 1 empty, 1 binary, 1 too large, 1 duplicate",
    );
  });

  it("does not read files bigger than the limit", async () => {
    const text = vi.fn(() => Promise.resolve("x"));

    await readSelection([{ path: "big.py", size: 900_000, text }]);

    expect(text).not.toHaveBeenCalled();
  });

  it("splits large uploads into requests that fit", () => {
    const many = Array.from({ length: 450 }, (_, n) => ({ path: `f${n}.py`, content: "x" }));
    const big = Array.from({ length: 5 }, (_, n) => ({
      path: `big${n}.py`,
      content: "x".repeat(1_000_000),
    }));

    expect(batches(many).map((batch) => batch.length)).toEqual([MAX_BATCH_FILES, 200, 50]);
    expect(batches(big).map((batch) => batch.length)).toEqual([3, 2]); // 3.5 MB at most
    expect(batches([])).toEqual([]);
  });

  it.each([
    ["My Shop API", "my-shop-api"],
    ["__private.repo", "private.repo"],
    ["Ünïcode", "n-code"],
    ["!!!", "my-codebase"],
    ["a".repeat(60), "a".repeat(48)],
  ])("codebaseName(%s) is %s", (raw, name) => {
    expect(codebaseName(raw)).toBe(name);
  });
});

describe("citations", () => {
  it("links the citations of known sources, outside code", () => {
    const answer =
      "Hashed with PBKDF2 [1], see `items[1]` and [2, 3].\n\n```py\nx = a[2]\n```\n[9]";

    expect(linkCitations(answer, 3)).toBe(
      "Hashed with PBKDF2 [1](#source-1), see `items[1]` and [2](#source-2) [3](#source-3).\n\n```py\nx = a[2]\n```\n[9]",
    );
  });

  it("leaves Markdown links alone", () => {
    expect(linkCitations("[1](https://example.com)", 3)).toBe("[1](https://example.com)");
  });

  it("reads the source of a citation link", () => {
    expect(citedSource("#source-2")).toBe(2);
    expect(citedSource("#source-0")).toBeNull();
    expect(citedSource("https://example.com")).toBeNull();
    expect(citedSource(undefined)).toBeNull();
  });
});

describe("formatting", () => {
  it("formats numbers, costs, durations and lines", () => {
    expect(percent(0.8433)).toBe("84%");
    expect(score(0.77777)).toBe("0.778");
    expect(usd(0)).toBe("$0");
    expect(usd(0.00421)).toBe("$0.0042");
    expect(usd(0.3456)).toBe("$0.346");
    expect(usd(1.2)).toBe("$1.20");
    expect(duration(850)).toBe("850 ms");
    expect(duration(12_300)).toBe("12.3 s");
    expect(duration(95_000)).toBe("1 min 35 s");
    expect(lines(18, 22)).toBe("L18–22");
    expect(lines(7, 7)).toBe("L7");
    expect(plural(1, "file")).toBe("1 file");
    expect(plural(1200, "chunk")).toBe("1,200 chunks");
  });

  it("splits a chunk key", () => {
    expect(splitKey("bookshop-api:bookshop/cart.py::Cart.apply_coupon")).toEqual({
      codebase: "bookshop-api",
      path: "bookshop/cart.py",
      name: "Cart.apply_coupon",
    });
  });
});

describe("metrics", () => {
  it("matches chunks to the code a question is about, like the backend", () => {
    const target = "shop:cart.py::Cart.apply_coupon";

    expect(matches("shop:cart.py::Cart", target)).toBe(true);
    expect(matches("shop:cart.py::Cart.apply_coupon", target)).toBe(true);
    expect(matches("shop:cart.py::Cartography", target)).toBe(false);
    expect(matches("shop:other.py::Cart", target)).toBe(false);
    expect(isRelevant("shop:cart.py::Cart", ["x:y::z", target])).toBe(true);
  });

  it("finds the best settings of a metric, ties included", () => {
    const report = {
      settings: ["vector", "bm25", "hybrid"],
      metrics: {
        vector: { precision: 0.3, recall: 0.8, mrr: 0.7, ndcg: 0.7, hit_rate: 0.9 },
        bm25: { precision: 0.3, recall: 0.9, mrr: 0.6, ndcg: 0.6, hit_rate: 1 },
        hybrid: { precision: 0.2, recall: 0.9, mrr: 0.8, ndcg: 0.75, hit_rate: 0.9 },
      },
    } as unknown as EvaluationReport;

    expect(best(report, "precision")).toEqual(["vector", "bm25"]);
    expect(best(report, "mrr")).toEqual(["hybrid"]);
  });

  it("estimates the cost of an evaluation", () => {
    const options = { rerank: false, judge: false, check: false };

    expect(estimateCost(18, options)).toBe(0);
    expect(estimateCost(18, { ...options, rerank: true })).toBeCloseTo(18 * 0.0042);
    expect(estimateCost(18, { ...options, rerank: true, judge: true })).toBeCloseTo(18 * 0.0119);
    expect(estimateCost(18, { rerank: true, judge: true, check: true })).toBeCloseTo(18 * 0.0169);
    expect(estimateCost(18, { ...options, check: true })).toBe(0); // the check needs answers
  });

  it("turns a reciprocal rank into a rank", () => {
    expect(firstRelevantRank(1)).toBe(1);
    expect(firstRelevantRank(0.3333)).toBe(3);
    expect(firstRelevantRank(0)).toBeNull();
  });
});

describe("the API client", () => {
  it("sends JSON and returns the answer", async () => {
    const fetcher = vi.fn<typeof fetch>().mockResolvedValue(Response.json({ hits: [] }));

    const result = await search(
      { question: "q", k: 5, mode: "bm25", rerank: false },
      undefined,
      fetcher,
    );

    expect(result).toEqual({ hits: [] });
    const [path, init] = fetcher.mock.calls[0];
    expect(path).toBe("/api/search");
    expect(JSON.parse(String(init?.body))).toEqual({
      question: "q",
      k: 5,
      mode: "bm25",
      rerank: false,
    });
  });

  it("turns failures into messages", async () => {
    const refused = vi
      .fn<typeof fetch>()
      .mockResolvedValue(Response.json({ detail: "Add at least one file." }, { status: 422 }));
    const offline = vi.fn<typeof fetch>().mockRejectedValue(new TypeError("Failed to fetch"));

    await expect(indexFiles({ codebase: "x", files: [], replace: false }, refused)).rejects.toEqual(
      new ApiError("Add at least one file.", 422),
    );
    await expect(indexFiles({ codebase: "x", files: [], replace: false }, offline)).rejects.toThrow(
      "Could not reach the server",
    );
    expect(errorMessage(504, null)).toBe("The server took too long to answer. Try again.");
    expect(errorMessage(503, "<html>")).toBe(
      "The RAG service is unavailable. Try again in a moment.",
    );
  });

  it("answers no latest evaluation when the server has none", async () => {
    const none = vi
      .fn<typeof fetch>()
      .mockResolvedValue(Response.json({ detail: "Evaluation run not found." }, { status: 404 }));

    expect(await getLatestEvaluation(none)).toBeNull();
  });
});

describe("the conversation history", () => {
  it("keeps the last answered exchanges as turns", () => {
    const answered = (id: number): Exchange => ({
      id,
      question: `Q${id}`,
      options: { mode: "hybrid", k: 5, rerank: true, answer: true, codebases: [] },
      status: "answered",
      result: { answer: `A${id}` } as QueryResult,
    });
    const failed: Exchange = { ...answered(9), status: "failed", error: "x" } as Exchange;

    const history = historyOf([answered(1), answered(2), failed, answered(3), answered(4)]);

    expect(history.map((turn) => turn.content)).toEqual(["Q2", "A2", "Q3", "A3", "Q4", "A4"]);
    expect(history[0].role).toBe("user");
  });
});
