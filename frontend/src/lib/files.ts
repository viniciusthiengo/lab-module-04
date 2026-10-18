// Choosing the files to index: supported types only, no dependencies or build output, sizes
// the backend accepts, and requests small enough for serverless hosts.

import type { FileIn } from "./types";

/** The file types the backend indexes (its `chunking.LANGUAGES`). */
export const EXTENSIONS = new Set([
  ".py",
  ".js",
  ".jsx",
  ".mjs",
  ".cjs",
  ".ts",
  ".mts",
  ".tsx",
  ".md",
  ".markdown",
  ".go",
  ".java",
  ".kt",
  ".rb",
  ".php",
  ".rs",
  ".c",
  ".h",
  ".cc",
  ".cpp",
  ".hpp",
  ".cs",
  ".swift",
  ".scala",
  ".sql",
  ".sh",
  ".json",
  ".yaml",
  ".yml",
  ".toml",
  ".html",
  ".css",
  ".txt",
  ".rst",
  ".ini",
  ".cfg",
]);
const FILE_NAMES = new Set(["Dockerfile", "Makefile"]);

/** Directories that hold dependencies, build output, caches or tool settings. */
export const IGNORED_DIRECTORIES = new Set([
  ".git",
  ".hg",
  ".svn",
  ".idea",
  ".vscode",
  ".next",
  ".nuxt",
  ".venv",
  "venv",
  ".tox",
  "__pycache__",
  ".mypy_cache",
  ".pytest_cache",
  ".ruff_cache",
  "node_modules",
  "bower_components",
  "vendor",
  "dist",
  "build",
  "out",
  "target",
  "coverage",
]);
const IGNORED_FILES = new Set([
  "package-lock.json",
  "yarn.lock",
  "pnpm-lock.yaml",
  "poetry.lock",
  "Pipfile.lock",
  "Cargo.lock",
  "composer.lock",
]);

export const MAX_FILE_CHARS = 200_000;
/** Files of one indexing; they are sent in batches. */
export const MAX_FILES = 2_000;
export const MAX_BATCH_FILES = 200;
/** Under the 4.5 MB request limit of serverless hosts such as Vercel. */
export const MAX_BATCH_BYTES = 3_500_000;
export const CODEBASE_NAME = /^[a-z0-9][a-z0-9._-]{0,47}$/;
/** The paths the backend accepts (its `paths.normalize_path`). */
const SAFE_PATH = /^[\p{L}\p{N}_@+\-. /]{1,300}$/u;

export type SkipReason =
  | "unsupported type"
  | "unsupported name"
  | "dependency or build output"
  | "lock file"
  | "too large"
  | "binary"
  | "empty"
  | "duplicate"
  | "over the limit";

/** A file to read: from a file input or a dropped folder. */
export interface Candidate {
  /** Its path in what was picked, such as `shop/src/app.py`. */
  path: string;
  size: number;
  text: () => Promise<string>;
}

export interface Selection {
  files: FileIn[];
  skipped: { path: string; reason: SkipReason }[];
  /** The folder that was picked, when every path starts with it. */
  root: string | null;
}

/** `path` with forward slashes, without empty or `.` parts and a leading `/`. */
export function cleanPath(path: string): string {
  return path
    .replaceAll("\\", "/")
    .split("/")
    .filter((part) => part !== "" && part !== ".")
    .join("/");
}

/** Why `path` is not indexed, or `null` when it is. */
export function skipReason(path: string): SkipReason | null {
  const parts = path.split("/");
  const name = parts[parts.length - 1];
  if (parts.slice(0, -1).some((part) => IGNORED_DIRECTORIES.has(part))) {
    return "dependency or build output";
  }
  if (IGNORED_FILES.has(name)) return "lock file";
  const dot = name.lastIndexOf(".");
  const supported =
    FILE_NAMES.has(name) || (dot > 0 && EXTENSIONS.has(name.slice(dot).toLowerCase()));
  if (!supported) return "unsupported type";
  if (!SAFE_PATH.test(path) || parts.some((part) => part === "" || part === "." || part === "..")) {
    return "unsupported name";
  }
  return null;
}

/** The common first folder of `paths` (what was picked), and the paths inside it. */
export function splitRoot(paths: string[]): { root: string | null; paths: string[] } {
  const first = paths[0]?.split("/")[0];
  const shared =
    first !== undefined &&
    paths.every((path) => path.includes("/") && path.split("/")[0] === first);
  if (!shared) return { root: null, paths };
  return { root: first, paths: paths.map((path) => path.slice(first.length + 1)) };
}

/** Reads the files worth indexing among `candidates`, and says why the others are skipped. */
export async function readSelection(candidates: Candidate[]): Promise<Selection> {
  const { root, paths } = splitRoot(candidates.map((candidate) => cleanPath(candidate.path)));
  const files: FileIn[] = [];
  const skipped: Selection["skipped"] = [];
  const seen = new Set<string>();
  for (const [index, candidate] of candidates.entries()) {
    const path = paths[index];
    const skip = (reason: SkipReason) => skipped.push({ path, reason });
    const reason = skipReason(path);
    if (reason) {
      skip(reason);
      continue;
    }
    if (seen.has(path)) {
      skip("duplicate");
      continue;
    }
    if (files.length >= MAX_FILES) {
      skip("over the limit");
      continue;
    }
    // A character takes at most 4 bytes: bigger files cannot fit, so they are not read.
    if (candidate.size > MAX_FILE_CHARS * 4) {
      skip("too large");
      continue;
    }
    const content = await candidate.text();
    if (content.includes("\0")) skip("binary");
    else if (content.length > MAX_FILE_CHARS) skip("too large");
    else if (!content.trim()) skip("empty");
    else {
      seen.add(path);
      files.push({ path, content });
    }
  }
  return { files, skipped, root };
}

/** `files` in requests of at most `MAX_BATCH_FILES` files and `MAX_BATCH_BYTES` bytes. */
export function batches(files: FileIn[]): FileIn[][] {
  const encoder = new TextEncoder();
  const result: FileIn[][] = [];
  let current: FileIn[] = [];
  let bytes = 0;
  for (const file of files) {
    const size = encoder.encode(JSON.stringify(file)).length;
    if (
      current.length > 0 &&
      (current.length >= MAX_BATCH_FILES || bytes + size > MAX_BATCH_BYTES)
    ) {
      result.push(current);
      current = [];
      bytes = 0;
    }
    current.push(file);
    bytes += size;
  }
  if (current.length > 0) result.push(current);
  return result;
}

/** A valid codebase name made from `raw` (such as the folder's name). */
export function codebaseName(raw: string): string {
  const name = raw
    .toLowerCase()
    .replace(/[^a-z0-9._-]+/g, "-")
    .replace(/^[^a-z0-9]+/, "")
    .slice(0, 48)
    .replace(/[-._]+$/, "");
  return name || "my-codebase";
}

/** Counts of the skip reasons, most frequent first: "12 unsupported type, 3 too large". */
export function skipSummary(skipped: Selection["skipped"]): string {
  const counts = new Map<SkipReason, number>();
  for (const { reason } of skipped) counts.set(reason, (counts.get(reason) ?? 0) + 1);
  return [...counts.entries()]
    .sort((a, b) => b[1] - a[1])
    .map(([reason, count]) => `${count} ${reason}`)
    .join(", ");
}
