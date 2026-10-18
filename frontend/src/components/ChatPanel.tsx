"use client";

import { type KeyboardEvent, useEffect, useId, useRef, useState } from "react";

import type { Codebase, Health, Mode } from "@/lib/types";
import { useChat } from "@/lib/useChat";

import styles from "./Chat.module.css";
import { ExchangeView } from "./ExchangeView";
import ui from "./ui.module.css";

/** Questions about the demo codebases, to try the system out. */
const SUGGESTIONS = [
  "How are passwords hashed?",
  "Are declined cards retried?",
  "How does the web client refresh an expired access token?",
  "How is the shipping cost of a cart computed?",
];

const MODES: { value: Mode; label: string; description: string }[] = [
  { value: "hybrid", label: "Hybrid", description: "Vector and BM25 rankings fused (RRF)" },
  { value: "vector", label: "Vector", description: "Embedding similarity only" },
  { value: "bm25", label: "BM25", description: "Keyword search only" },
];

const MAX_QUESTION = 1000;

interface Props {
  health: Health | null;
  codebases: Codebase[];
  scope: string[];
  onAnswered: () => void;
}

/** The chat: questions about the indexed code, answered with their sources. */
export function ChatPanel({ health, codebases, scope, onAnswered }: Props) {
  const id = useId();
  const { exchanges, busy, ask, clear } = useChat();
  const [question, setQuestion] = useState("");
  const [mode, setMode] = useState<Mode>("hybrid");
  const [k, setK] = useState(5);
  const [rerank, setRerank] = useState(true);
  const [answer, setAnswer] = useState(true);
  const list = useRef<HTMLOListElement>(null);
  const ready = health?.ready ?? false;
  const nothingIndexed = codebases.length === 0;
  const last = exchanges.at(-1);

  useEffect(() => {
    // Shows the newest exchange from its question, inside the conversation's own scroll.
    const container = list.current;
    const item = container?.lastElementChild as HTMLElement | null | undefined;
    if (container && item) {
      container.scrollTo({ top: item.offsetTop - container.offsetTop, behavior: "smooth" });
    }
  }, [last?.id, last?.status]);

  async function submit(text: string) {
    const trimmed = text.trim();
    if (!trimmed || busy || nothingIndexed) return;
    setQuestion("");
    await ask(trimmed, {
      mode,
      k,
      rerank: rerank && ready,
      answer: answer && ready,
      codebases: scope,
    });
    onAnswered();
  }

  function onKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault();
      void submit(question);
    }
  }

  return (
    <section className={`${ui.panel} ${styles.chat}`} aria-labelledby="querying-title">
      <div className={ui.panelHeader}>
        <h2 id="querying-title">Querying</h2>
        {exchanges.length > 0 && (
          <button type="button" className={ui.link} onClick={clear}>
            New conversation
          </button>
        )}
      </div>

      <div className={styles.options}>
        <div className={ui.segmented} role="group" aria-label="Retrieval">
          {MODES.map((item) => (
            <button
              key={item.value}
              type="button"
              title={item.description}
              aria-pressed={mode === item.value}
              onClick={() => setMode(item.value)}
            >
              {item.label}
            </button>
          ))}
        </div>
        <label className={styles.option}>
          Top
          <select value={k} onChange={(event) => setK(Number(event.currentTarget.value))}>
            {[3, 5, 8, 10].map((value) => (
              <option key={value} value={value}>
                {value}
              </option>
            ))}
          </select>
        </label>
        <label className={styles.option} title="Claude grades the 15 best candidates">
          <input
            type="checkbox"
            checked={rerank && ready}
            disabled={!ready}
            onChange={(event) => setRerank(event.currentTarget.checked)}
          />
          Rerank
        </label>
        <label className={styles.option} title="Off: only retrieve the code (free)">
          <input
            type="checkbox"
            checked={answer && ready}
            disabled={!ready}
            onChange={(event) => setAnswer(event.currentTarget.checked)}
          />
          Answer with Claude
        </label>
      </div>
      {health && !ready && (
        <p className={ui.hint}>
          The server has no Claude API key: questions retrieve the matching code without an answer.
        </p>
      )}

      <ol ref={list} className={styles.conversation} aria-live="polite">
        {exchanges.length === 0 && (
          <li className={styles.welcome}>
            <p>
              {nothingIndexed
                ? "Index a codebase to ask questions about it."
                : "Ask about the indexed code. Answers cite the code they use; click a citation to see it."}
            </p>
            {!nothingIndexed && (
              <div className={styles.suggestions}>
                {SUGGESTIONS.map((suggestion) => (
                  <button
                    key={suggestion}
                    type="button"
                    className={styles.suggestion}
                    disabled={busy}
                    onClick={() => void submit(suggestion)}
                  >
                    {suggestion}
                  </button>
                ))}
              </div>
            )}
          </li>
        )}
        {exchanges.map((exchange) => (
          <ExchangeView key={exchange.id} exchange={exchange} />
        ))}
      </ol>

      <form
        className={styles.composer}
        onSubmit={(event) => {
          event.preventDefault();
          void submit(question);
        }}
      >
        <label htmlFor={`${id}-question`} className="visually-hidden">
          Question
        </label>
        <textarea
          id={`${id}-question`}
          className={styles.input}
          value={question}
          rows={2}
          maxLength={MAX_QUESTION}
          placeholder={nothingIndexed ? "Nothing indexed yet" : "How does the cart apply a coupon?"}
          disabled={nothingIndexed}
          onChange={(event) => setQuestion(event.currentTarget.value)}
          onKeyDown={onKeyDown}
        />
        <div className={styles.composerActions}>
          <span className={ui.hint}>
            Enter to send · Shift+Enter for a new line
            {question.length > MAX_QUESTION - 100 && ` · ${question.length}/${MAX_QUESTION}`}
          </span>
          <button
            type="submit"
            className={ui.primary}
            disabled={busy || nothingIndexed || !question.trim()}
          >
            {busy ? <span className={ui.spinner} aria-label="Answering" /> : "Ask"}
          </button>
        </div>
      </form>
    </section>
  );
}
