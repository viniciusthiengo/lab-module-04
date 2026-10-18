"use client";

import { useState } from "react";

import { useServer } from "@/lib/useServer";

import { ChatPanel } from "./ChatPanel";
import { EvaluationPanel } from "./EvaluationPanel";
import { IndexingPanel } from "./IndexingPanel";
import styles from "./RagApp.module.css";
import { StatusBar } from "./StatusBar";

/** The page: indexing and querying side by side, the evaluation below. */
export function RagApp() {
  const server = useServer();
  const [checked, setChecked] = useState<string[]>([]);
  const indexed = server.codebases ?? [];
  // Questions search the checked codebases that still exist, or every one when none is.
  const scope = checked.filter((name) => indexed.some((codebase) => codebase.name === name));

  return (
    <>
      <StatusBar health={server.health} error={server.error} />
      <div className={styles.layout}>
        <IndexingPanel
          codebases={server.codebases}
          scope={scope}
          onScopeChange={setChecked}
          onChanged={server.refresh}
          onDelete={server.remove}
        />
        <ChatPanel
          health={server.health}
          codebases={indexed}
          scope={scope}
          onAnswered={server.refresh}
        />
      </div>
      <EvaluationPanel health={server.health} onFinished={server.refresh} />
    </>
  );
}
