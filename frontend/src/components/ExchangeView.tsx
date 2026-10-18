"use client";

import { useState } from "react";

import { cx } from "@/lib/dom";
import { duration, usd } from "@/lib/format";
import type { Hit, Usage } from "@/lib/types";
import type { AskOptions, Exchange } from "@/lib/useChat";

import styles from "./Chat.module.css";
import { Markdown } from "./Markdown";
import { SourceCard } from "./SourceCard";
import ui from "./ui.module.css";

/** One question and its answer (or the code retrieved for it), with the sources. */
export function ExchangeView({ exchange }: { exchange: Exchange }) {
  const sources = useSources(`exchange-${exchange.id}-source`);
  return (
    <li className={styles.exchange}>
      <div className={styles.question}>
        <p>{exchange.question}</p>
        <span className={styles.questionOptions}>{describe(exchange.options)}</span>
      </div>
      <div className={styles.reply}>
        {exchange.status === "pending" && (
          <p className={ui.muted} aria-live="polite">
            <span className={ui.spinner} aria-hidden />{" "}
            {exchange.options.answer ? "Retrieving the code and answering…" : "Retrieving…"}
          </p>
        )}
        {exchange.status === "failed" && (
          <p className={ui.alert} role="alert">
            {exchange.error}
          </p>
        )}
        {exchange.status === "answered" && (
          <>
            <Markdown sources={exchange.result.sources.length} onCite={sources.show}>
              {exchange.result.answer}
            </Markdown>
            <Meta
              grounded={exchange.result.grounded}
              cached={exchange.result.cached}
              usage={exchange.result.usage}
              durationMs={exchange.result.duration_ms}
            />
            <SourceList
              hits={exchange.result.sources}
              cited={exchange.result.sources_used}
              {...sources}
            />
          </>
        )}
        {exchange.status === "retrieved" && (
          <>
            <p className={ui.hint}>
              {exchange.result.hits.length
                ? `The ${exchange.result.hits.length} chunks that best match, out of ${exchange.result.candidates} candidates (no answer generated).`
                : "No indexed code matches this question."}{" "}
              <Meta usage={exchange.result.usage} durationMs={exchange.result.duration_ms} />
            </p>
            <SourceList hits={exchange.result.hits} cited={[]} {...sources} />
          </>
        )}
      </div>
    </li>
  );
}

/** Which source cards are expanded or highlighted; `show` brings one into view (a citation). */
function useSources(prefix: string) {
  const [expanded, setExpanded] = useState<ReadonlySet<number>>(new Set());
  const [highlighted, setHighlighted] = useState<number | null>(null);

  function toggle(source: number, open: boolean) {
    setExpanded((current) => {
      const next = new Set(current);
      if (open) next.add(source);
      else next.delete(source);
      return next;
    });
  }

  function show(source: number) {
    toggle(source, true);
    setHighlighted(source);
    document
      .getElementById(`${prefix}-${source}`)
      ?.scrollIntoView({ behavior: "smooth", block: "nearest" });
    setTimeout(() => setHighlighted((current) => (current === source ? null : current)), 1600);
  }

  return { prefix, expanded, highlighted, show, toggle };
}

interface SourceListProps extends ReturnType<typeof useSources> {
  hits: Hit[];
  cited: number[];
}

function SourceList({ hits, cited, prefix, expanded, highlighted, toggle }: SourceListProps) {
  if (hits.length === 0) return null;
  return (
    <details className={styles.sources} open>
      <summary>
        Sources <span className={ui.tabBadge}>{hits.length}</span>
      </summary>
      <ol className={styles.sourceList}>
        {hits.map((hit, index) => {
          const number = index + 1;
          return (
            <SourceCard
              key={hit.id}
              id={`${prefix}-${number}`}
              number={number}
              hit={hit}
              cited={cited.includes(number)}
              highlighted={highlighted === number}
              expanded={expanded.has(number)}
              onExpandedChange={(open) => toggle(number, open)}
            />
          );
        })}
      </ol>
    </details>
  );
}

interface MetaProps {
  grounded?: boolean;
  cached?: boolean;
  usage: Usage;
  durationMs: number;
}

function Meta({ grounded, cached, usage, durationMs }: MetaProps) {
  const calls = usage.model_calls;
  return (
    <span className={styles.meta}>
      {grounded === false && (
        <span
          className={cx(ui.chip, ui.chipWarning)}
          title="The retrieved code did not contain the whole answer"
        >
          not grounded
        </span>
      )}
      {cached && <span className={cx(ui.chip, ui.chipSuccess)}>cached</span>}
      <span>
        {duration(durationMs)}
        {calls > 0 &&
          ` · ${calls} model ${calls === 1 ? "call" : "calls"} · ${usd(usage.cost_usd)}`}
      </span>
    </span>
  );
}

function describe(options: AskOptions): string {
  const parts = [
    options.mode === "bm25" ? "BM25" : options.mode,
    `top ${options.k}`,
    ...(options.rerank ? ["reranked"] : []),
    options.answer ? "answer" : "retrieval only",
    options.codebases.length ? options.codebases.join(", ") : "all codebases",
  ];
  return parts.join(" · ");
}
