"use client";

import { useState } from "react";

import { cx } from "@/lib/dom";
import { lines } from "@/lib/format";
import type { Hit } from "@/lib/types";

import styles from "./Chat.module.css";
import ui from "./ui.module.css";

/** Lines of code shown before "Show all". */
const PREVIEW_LINES = 12;

interface Props {
  /** The element id, which citations scroll to. */
  id: string;
  number: number;
  hit: Hit;
  cited: boolean;
  highlighted: boolean;
  expanded: boolean;
  onExpandedChange: (expanded: boolean) => void;
}

/** A retrieved chunk: where it is, how it ranked, and its code with line numbers. */
export function SourceCard({
  id,
  number,
  hit,
  cited,
  highlighted,
  expanded,
  onExpandedChange,
}: Props) {
  const [copied, setCopied] = useState(false);
  const code = hit.content.split("\n");
  const shown = expanded ? code : code.slice(0, PREVIEW_LINES);

  async function copy() {
    try {
      await navigator.clipboard.writeText(hit.content);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      // The clipboard can be unavailable (insecure context): nothing to do.
    }
  }

  return (
    <li id={id} className={cx(styles.source, highlighted && styles.highlighted)}>
      <div className={styles.sourceHeader}>
        <span className={styles.sourceNumber}>{number}</span>
        <div className={styles.sourceTitle}>
          <span className={styles.sourcePath}>
            {hit.path} <span className={ui.muted}>{lines(hit.start_line, hit.end_line)}</span>
          </span>
          <span className={styles.sourceName}>
            {hit.kind} <code>{hit.name}</code> · {hit.codebase}
          </span>
        </div>
      </div>
      <div className={styles.ranks}>
        {cited && <span className={cx(ui.chip, ui.chipSuccess)}>cited</span>}
        {hit.relevance !== null && (
          <span className={ui.chip} title="The reranker's grade, from 0 to 3">
            relevance {hit.relevance}/3
          </span>
        )}
        {hit.vector_rank !== null && <span className={ui.chip}>vector #{hit.vector_rank}</span>}
        {hit.bm25_rank !== null && <span className={ui.chip}>BM25 #{hit.bm25_rank}</span>}
      </div>
      <pre className={styles.code} aria-label={`Code of ${hit.name}`}>
        {shown.map((line, index) => (
          <span key={index} className={styles.line}>
            <span className={styles.lineNumber} aria-hidden>
              {hit.start_line + index}
            </span>
            {line || " "}
          </span>
        ))}
      </pre>
      <div className={styles.sourceActions}>
        {code.length > PREVIEW_LINES && (
          <button type="button" className={ui.link} onClick={() => onExpandedChange(!expanded)}>
            {expanded ? "Show less" : `Show all ${code.length} lines`}
          </button>
        )}
        <button type="button" className={ui.link} onClick={() => void copy()}>
          {copied ? "Copied" : "Copy"}
        </button>
      </div>
    </li>
  );
}
