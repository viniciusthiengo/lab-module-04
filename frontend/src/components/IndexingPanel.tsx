"use client";

import type { Codebase } from "@/lib/types";

import { CodebaseList } from "./CodebaseList";
import styles from "./Indexing.module.css";
import ui from "./ui.module.css";
import { Uploader } from "./Uploader";

interface Props {
  codebases: Codebase[] | null;
  scope: string[];
  onScopeChange: (scope: string[]) => void;
  onChanged: () => void;
  onDelete: (name: string) => Promise<void>;
}

/** Uploading code into a codebase, and the codebases already indexed. */
export function IndexingPanel({ codebases, scope, onScopeChange, onChanged, onDelete }: Props) {
  return (
    <section className={ui.panel} aria-labelledby="indexing-title">
      <div className={ui.panelHeader}>
        <h2 id="indexing-title">Indexing</h2>
      </div>
      <Uploader codebases={codebases ?? []} onIndexed={onChanged} />
      <h3 className={styles.subtitle}>Indexed codebases</h3>
      <CodebaseList
        codebases={codebases}
        scope={scope}
        onScopeChange={onScopeChange}
        onDelete={onDelete}
      />
    </section>
  );
}
