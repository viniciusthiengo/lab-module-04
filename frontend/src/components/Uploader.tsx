"use client";

import { type DragEvent, useEffect, useId, useRef, useState } from "react";

import { cx } from "@/lib/dom";
import { droppedCandidates, inputCandidates } from "@/lib/dropped";
import { batches, CODEBASE_NAME, codebaseName, type Selection, skipSummary } from "@/lib/files";
import { count, duration, plural } from "@/lib/format";
import type { Codebase } from "@/lib/types";
import { useIndexer } from "@/lib/useIndexer";

import styles from "./Indexing.module.css";
import ui from "./ui.module.css";

interface Props {
  codebases: Codebase[];
  onIndexed: () => void;
}

/** Drop or pick a folder (or files), name the codebase, index. */
export function Uploader({ codebases, onIndexed }: Props) {
  const { state, choose, index, reset } = useIndexer(onIndexed);
  const folderInput = useRef<HTMLInputElement>(null);
  const filesInput = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  useEffect(() => {
    // Not in React's attribute types: picks a whole folder, subfolders included.
    folderInput.current?.setAttribute("webkitdirectory", "");
  }, []);

  async function onDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setDragging(false);
    if (event.dataTransfer.items.length)
      await choose(await droppedCandidates(event.dataTransfer.items));
  }

  function onPicked(input: HTMLInputElement | null) {
    if (input?.files?.length) void choose(inputCandidates(input.files));
    if (input) input.value = "";
  }

  const selection = "selection" in state ? state.selection : null;
  const busy = state.phase === "reading" || state.phase === "indexing";

  return (
    <div className={styles.uploader}>
      <div
        className={cx(styles.dropZone, dragging && styles.dragging)}
        onDragOver={(event) => {
          event.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(event) => void onDrop(event)}
      >
        <p className={styles.dropTitle}>Drop a folder or files here</p>
        <p className={ui.hint}>
          Python, JavaScript, TypeScript, Markdown and 25 more types. Dependencies, build output and
          lock files are skipped.
        </p>
        <div className={styles.pickers}>
          <button
            type="button"
            className={ui.secondary}
            disabled={busy}
            onClick={() => folderInput.current?.click()}
          >
            Choose a folder
          </button>
          <button
            type="button"
            className={ui.ghost}
            disabled={busy}
            onClick={() => filesInput.current?.click()}
          >
            Choose files
          </button>
        </div>
        <input
          ref={folderInput}
          type="file"
          hidden
          multiple
          onChange={(event) => onPicked(event.currentTarget)}
        />
        <input
          ref={filesInput}
          type="file"
          hidden
          multiple
          onChange={(event) => onPicked(event.currentTarget)}
        />
      </div>

      {state.phase === "reading" && (
        <p className={ui.muted} aria-live="polite">
          <span className={ui.spinner} aria-hidden /> Reading the files…
        </p>
      )}
      {selection && state.phase !== "done" && (
        <SelectionForm
          key={selection.root ?? selection.files[0]?.path ?? "files"}
          selection={selection}
          codebases={codebases}
          state={state}
          onIndex={(name, replace) => void index(selection, name, replace)}
          onCancel={reset}
        />
      )}
      {state.phase === "failed" && (
        <p className={ui.alert} role="alert">
          {state.error}
        </p>
      )}
      {state.phase === "done" && (
        <div className={styles.done} role="status">
          <p>
            Indexed {plural(state.summary.files, "file")} into <b>{state.summary.codebase}</b>:{" "}
            {plural(state.summary.chunks, "chunk")} ({count(state.summary.embedded)} embedded,{" "}
            {count(state.summary.reused)} unchanged
            {state.summary.removed ? `, ${count(state.summary.removed)} removed` : ""}) in{" "}
            {duration(state.summary.durationMs)}.
          </p>
          <button type="button" className={ui.link} onClick={reset}>
            Index more files
          </button>
        </div>
      )}
    </div>
  );
}

interface FormProps {
  selection: Selection;
  codebases: Codebase[];
  state: ReturnType<typeof useIndexer>["state"];
  onIndex: (name: string, replace: boolean) => void;
  onCancel: () => void;
}

function SelectionForm({ selection, codebases, state, onIndex, onCancel }: FormProps) {
  const id = useId();
  const [name, setName] = useState(() => codebaseName(selection.root ?? "my-codebase"));
  const [replace, setReplace] = useState(false);
  const existing = codebases.find((codebase) => codebase.name === name);
  const problem = !CODEBASE_NAME.test(name)
    ? "Use lower-case letters, digits, '.', '_' or '-' (48 at most), starting with a letter or a digit."
    : existing?.demo
      ? `'${name}' is a demo codebase of the evaluation; choose another name.`
      : null;
  const indexing = state.phase === "indexing";
  const requests = batches(selection.files).length;

  return (
    <form
      className={styles.form}
      onSubmit={(event) => {
        event.preventDefault();
        if (!problem && selection.files.length) onIndex(name, replace && Boolean(existing));
      }}
    >
      <p className={styles.counts}>
        <b>{plural(selection.files.length, "file")}</b> to index
        {selection.skipped.length > 0 && (
          <span className={ui.muted}>
            {" "}
            · {count(selection.skipped.length)} skipped ({skipSummary(selection.skipped)})
          </span>
        )}
      </p>
      {selection.files.length > 0 && (
        <details className={styles.fileList}>
          <summary>Show the files</summary>
          <ul>
            {selection.files.slice(0, 300).map((file) => (
              <li key={file.path}>{file.path}</li>
            ))}
            {selection.files.length > 300 && (
              <li className={ui.muted}>…and {count(selection.files.length - 300)} more</li>
            )}
          </ul>
        </details>
      )}
      <label className={ui.field} htmlFor={`${id}-name`}>
        Codebase name
        <input
          id={`${id}-name`}
          className={styles.input}
          value={name}
          maxLength={48}
          spellCheck={false}
          autoComplete="off"
          aria-invalid={problem ? true : undefined}
          aria-describedby={`${id}-name-hint`}
          onChange={(event) => setName(event.currentTarget.value.trim().toLowerCase())}
        />
        <span id={`${id}-name-hint`} className={problem ? styles.invalid : ui.hint}>
          {problem ??
            (existing
              ? `Adds to '${name}': unchanged chunks keep their embeddings.`
              : "A new codebase.")}
        </span>
      </label>
      {existing && !existing.demo && (
        <label className={styles.check}>
          <input
            type="checkbox"
            checked={replace}
            onChange={(event) => setReplace(event.currentTarget.checked)}
          />
          Remove the files of &apos;{name}&apos; that are not in this upload
        </label>
      )}
      <div className={styles.actions}>
        <button
          type="submit"
          className={ui.primary}
          disabled={indexing || Boolean(problem) || selection.files.length === 0}
        >
          {indexing ? (
            <>
              <span className={ui.spinner} aria-hidden />
              {state.batches > 1 ? `Batch ${state.batch} of ${state.batches}…` : "Indexing…"}
            </>
          ) : (
            "Index"
          )}
        </button>
        <button type="button" className={ui.ghost} disabled={indexing} onClick={onCancel}>
          Cancel
        </button>
        {requests > 1 && !indexing && (
          <span className={ui.hint}>Sent in {count(requests)} requests.</span>
        )}
      </div>
    </form>
  );
}
