"use client";

import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";

import { citedSource, linkCitations } from "@/lib/citations";

import styles from "./Markdown.module.css";

interface Props {
  children: string;
  /** How many sources the text may cite as [1], [2]...; none by default. */
  sources?: number;
  onCite?: (source: number) => void;
}

/** Markdown from the model; its citations become buttons that show the source. */
export function Markdown({ children, sources = 0, onCite }: Props) {
  const components: Components = {
    a({ href, children: text }) {
      const source = citedSource(href);
      if (source !== null) {
        return (
          <button
            type="button"
            className={styles.citation}
            aria-label={`Source ${source}`}
            onClick={() => onCite?.(source)}
          >
            {source}
          </button>
        );
      }
      return (
        <a href={href} target="_blank" rel="noopener noreferrer">
          {text}
        </a>
      );
    },
  };
  return (
    <div className={styles.markdown}>
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
        {sources ? linkCitations(children, sources) : children}
      </ReactMarkdown>
    </div>
  );
}
