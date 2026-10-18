import { cx } from "@/lib/dom";
import { count, usd } from "@/lib/format";
import type { Health } from "@/lib/types";

import styles from "./StatusBar.module.css";
import ui from "./ui.module.css";

/** What the server runs: models, size of the index, today's spending. */
export function StatusBar({ health, error }: { health: Health | null; error: string | null }) {
  if (error && !health) {
    return (
      <p className={cx(ui.alert, styles.bar)} role="alert">
        {error}
      </p>
    );
  }
  if (!health) {
    return (
      <p className={cx(styles.bar, ui.muted)} aria-live="polite">
        <span className={ui.spinner} aria-hidden /> Connecting to the server…
      </p>
    );
  }
  return (
    <ul className={styles.bar} aria-label="Server status">
      <li className={cx(ui.chip, health.ready ? ui.chipSuccess : ui.chipWarning)}>
        {health.ready ? `Claude: ${health.model}` : "Claude not configured: retrieval only"}
      </li>
      {health.judge_model && <li className={ui.chip}>Judge: {health.judge_model}</li>}
      <li className={ui.chip}>Embeddings: {health.embedding_model}</li>
      <li className={ui.chip}>
        {count(health.codebases)} codebases · {count(health.chunks)} chunks
      </li>
      <li className={ui.chip}>
        Spent today: {usd(health.spent_today_usd)} of {usd(health.daily_budget_usd)}
      </li>
    </ul>
  );
}
