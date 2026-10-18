"use client";

import { useEffect, useState } from "react";

import { ApiError } from "@/lib/api";
import { cx } from "@/lib/dom";
import { count, dateTime, plural } from "@/lib/format";
import type { Codebase } from "@/lib/types";

import styles from "./Indexing.module.css";
import ui from "./ui.module.css";

interface Props {
  codebases: Codebase[] | null;
  scope: string[];
  onScopeChange: (scope: string[]) => void;
  onDelete: (name: string) => Promise<void>;
}

/** The indexed codebases; the checked ones are the ones questions search. */
export function CodebaseList({ codebases, scope, onScopeChange, onDelete }: Props) {
  if (codebases === null) {
    return (
      <p className={ui.muted}>
        <span className={ui.spinner} aria-hidden /> Loading…
      </p>
    );
  }
  if (codebases.length === 0) {
    return <p className={ui.empty}>Nothing is indexed yet. Upload a folder above.</p>;
  }

  function toggle(name: string, checked: boolean) {
    onScopeChange(checked ? [...scope, name] : scope.filter((item) => item !== name));
  }

  return (
    <>
      <p className={ui.hint}>
        {scope.length
          ? `Questions search ${scope.join(", ")}.`
          : "Questions search every codebase; check some to search only those."}
      </p>
      <ul className={styles.codebases}>
        {codebases.map((codebase) => (
          <CodebaseItem
            key={codebase.name}
            codebase={codebase}
            checked={scope.includes(codebase.name)}
            onCheck={(checked) => toggle(codebase.name, checked)}
            onDelete={() => onDelete(codebase.name)}
          />
        ))}
      </ul>
    </>
  );
}

interface ItemProps {
  codebase: Codebase;
  checked: boolean;
  onCheck: (checked: boolean) => void;
  onDelete: () => Promise<void>;
}

function CodebaseItem({ codebase, checked, onCheck, onDelete }: ItemProps) {
  const [confirming, setConfirming] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!confirming) return;
    const timer = setTimeout(() => setConfirming(false), 4000);
    return () => clearTimeout(timer);
  }, [confirming]);

  async function remove() {
    if (!confirming) {
      setConfirming(true);
      return;
    }
    setDeleting(true);
    setError(null);
    try {
      await onDelete();
    } catch (failure) {
      setError(failure instanceof ApiError ? failure.message : "Could not delete it.");
      setDeleting(false);
      setConfirming(false);
    }
  }

  return (
    <li className={cx(styles.codebase, checked && styles.checked)}>
      <div className={styles.codebaseHeader}>
        <label className={styles.codebaseName}>
          <input
            type="checkbox"
            checked={checked}
            onChange={(event) => onCheck(event.currentTarget.checked)}
          />
          {codebase.name}
        </label>
        {codebase.demo ? (
          <span className={ui.chip} title="A demo codebase of the evaluation dataset (read-only)">
            demo
          </span>
        ) : (
          <button
            type="button"
            className={cx(ui.link, confirming && styles.danger)}
            disabled={deleting}
            onClick={() => void remove()}
          >
            {deleting ? "Deleting…" : confirming ? "Confirm delete" : "Delete"}
          </button>
        )}
      </div>
      <p className={styles.codebaseMeta}>
        {plural(codebase.files.length, "file")} · {plural(codebase.chunks, "chunk")} ·{" "}
        {codebase.languages.join(", ")}
      </p>
      <details className={styles.fileList}>
        <summary>Files</summary>
        <ul>
          {codebase.files.map((file) => (
            <li key={file.path}>
              {file.path}{" "}
              <span className={ui.muted}>
                {count(file.lines)} lines · {plural(file.chunks, "chunk")}
              </span>
            </li>
          ))}
        </ul>
        <p className={ui.hint}>Indexed {dateTime(codebase.updated_at)}</p>
      </details>
      {error && (
        <p className={ui.error} role="alert">
          {error}
        </p>
      )}
    </li>
  );
}
