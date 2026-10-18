// Files chosen in the browser: from a file input, or dropped (folders included). Dropped
// folders are walked with the entries API, skipping the directories that are never indexed
// (node_modules, .git...) without reading them.

import { type Candidate, IGNORED_DIRECTORIES } from "./files";

/** A file of an input or a drop; `path` is relative to what was chosen. */
export function fileCandidate(file: File, path = file.webkitRelativePath || file.name): Candidate {
  return { path, size: file.size, text: () => file.text() };
}

export function inputCandidates(files: FileList): Candidate[] {
  return [...files].map((file) => fileCandidate(file));
}

export async function droppedCandidates(items: DataTransferItemList): Promise<Candidate[]> {
  const entries = [...items]
    .map((item) => item.webkitGetAsEntry())
    .filter((entry): entry is FileSystemEntry => entry !== null);
  const candidates: Candidate[] = [];
  for (const entry of entries) await walk(entry, candidates);
  return candidates;
}

async function walk(entry: FileSystemEntry, into: Candidate[]): Promise<void> {
  if (isFile(entry)) {
    const file = await new Promise<File>((resolve, reject) => entry.file(resolve, reject));
    into.push(fileCandidate(file, entry.fullPath));
  } else if (isDirectory(entry) && !IGNORED_DIRECTORIES.has(entry.name)) {
    const reader = entry.createReader();
    // readEntries answers in chunks (100 entries in Chrome): read until it answers none.
    for (;;) {
      const chunk = await new Promise<FileSystemEntry[]>((resolve, reject) =>
        reader.readEntries(resolve, reject),
      );
      if (chunk.length === 0) break;
      for (const child of chunk) await walk(child, into);
    }
  }
}

function isFile(entry: FileSystemEntry): entry is FileSystemFileEntry {
  return entry.isFile;
}

function isDirectory(entry: FileSystemEntry): entry is FileSystemDirectoryEntry {
  return entry.isDirectory;
}
