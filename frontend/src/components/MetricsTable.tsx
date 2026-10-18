import type { CSSProperties } from "react";

import { cx } from "@/lib/dom";
import { percent, score } from "@/lib/format";
import { best, METRICS, SETTINGS } from "@/lib/metrics";
import type { EvaluationReport } from "@/lib/types";

import styles from "./Evaluation.module.css";

/** Retrieval metrics: one row per setting, the best value of each metric in bold. */
export function MetricsTable({ report }: { report: EvaluationReport }) {
  const winners = Object.fromEntries(
    METRICS.map((metric) => [metric.key, best(report, metric.key)]),
  );
  return (
    <div className={styles.tableWrap}>
      <table className={styles.metrics}>
        <caption className="visually-hidden">
          Retrieval metrics of each setting at K={report.k}
        </caption>
        <thead>
          <tr>
            <th scope="col">Setting</th>
            {METRICS.map((metric) => (
              <th key={metric.key} scope="col" title={metric.description}>
                {metric.name.replace("K", String(report.k))}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {SETTINGS.filter((setting) => report.settings.includes(setting.label)).map((setting) => {
            const values = report.metrics[setting.label];
            if (!values) return null;
            return (
              <tr key={setting.label}>
                <th scope="row">
                  {setting.name}
                  <span className={styles.settingNote}>{setting.description}</span>
                </th>
                {METRICS.map((metric) => {
                  const value = values[metric.key];
                  const top = winners[metric.key].includes(setting.label);
                  return (
                    <td key={metric.key} className={cx(styles.value, top && styles.best)}>
                      <span className={styles.cell}>
                        <span
                          className={styles.cellMeter}
                          style={{ "--value": value } as CSSProperties}
                          aria-hidden
                        />
                        <span className={styles.number}>
                          {metric.as === "percent" ? percent(value) : score(value)}
                        </span>
                      </span>
                    </td>
                  );
                })}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
