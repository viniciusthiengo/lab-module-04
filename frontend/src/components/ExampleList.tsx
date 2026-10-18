import { cx } from "@/lib/dom";
import { firstRelevantRank, isRelevant, SETTINGS } from "@/lib/metrics";
import type { EvaluationReport, ExampleResult } from "@/lib/types";

import styles from "./Evaluation.module.css";
import { Markdown } from "./Markdown";
import ui from "./ui.module.css";

/** Each question: the rank of the first relevant chunk per setting, the judge's grades, and the
 * details (reference, answer, rationale, what each setting retrieved). */
export function ExampleList({ report }: { report: EvaluationReport }) {
  const settings = SETTINGS.filter((setting) => report.settings.includes(setting.label));
  return (
    <>
      <p className={ui.hint}>
        The numbers are the rank of the first relevant chunk for{" "}
        {settings.map((setting) => setting.name).join(", ")}; a dash means none in the top{" "}
        {report.k}.
      </p>
      <ul className={styles.examples}>
        {report.examples.map((example) => (
          <ExampleItem key={example.id} example={example} report={report} />
        ))}
      </ul>
    </>
  );
}

function ExampleItem({ example, report }: { example: ExampleResult; report: EvaluationReport }) {
  const settings = SETTINGS.filter((setting) => report.settings.includes(setting.label));
  const judgement = example.answer?.judgement ?? null;
  return (
    <li>
      <details className={styles.example}>
        <summary>
          <span className={styles.exampleId}>{example.id}</span>
          <span className={styles.exampleQuestion}>{example.question}</span>
          <span className={styles.ranksRow}>
            {settings.map((setting) => {
              const rank = firstRelevantRank(
                example.retrieval[setting.label]?.reciprocal_rank ?? 0,
              );
              return (
                <span
                  key={setting.label}
                  className={cx(
                    styles.rank,
                    rank === 1 ? styles.rankTop : rank === null ? styles.rankMiss : styles.rankLow,
                  )}
                  title={`${setting.name}: ${rank === null ? `no relevant chunk in the top ${report.k}` : `first relevant chunk at rank ${rank}`}`}
                >
                  {rank ?? "–"}
                </span>
              );
            })}
          </span>
          {judgement && (
            <span
              className={cx(ui.chip, judgement.passed ? ui.chipSuccess : ui.chipDanger)}
              title="Faithfulness, relevance and correctness, from 1 to 5"
            >
              {judgement.faithfulness}·{judgement.relevance}·{judgement.correctness}
            </span>
          )}
        </summary>
        <div className={styles.exampleBody}>
          <section>
            <h4>Reference answer</h4>
            <p>{example.expected_answer}</p>
          </section>
          {example.answer && (
            <section>
              <h4>Generated answer{example.answer.grounded ? "" : " (not grounded)"}</h4>
              <Markdown>{example.answer.answer}</Markdown>
            </section>
          )}
          {judgement && (
            <section>
              <h4>
                Judge: faithfulness {judgement.faithfulness}, relevance {judgement.relevance},
                correctness {judgement.correctness}
              </h4>
              <p>{judgement.rationale}</p>
            </section>
          )}
          <section>
            <h4>Relevant code</h4>
            <ul className={styles.keys}>
              {example.relevant.map((target) => (
                <li key={target}>
                  <code>{target}</code>
                </li>
              ))}
            </ul>
          </section>
          <section className={styles.retrieved}>
            {settings.map((setting) => (
              <div key={setting.label}>
                <h4>{setting.name}</h4>
                <ol className={styles.keys}>
                  {(example.retrieval[setting.label]?.retrieved ?? []).map((key, index) => (
                    <li
                      key={`${key}-${index}`}
                      className={isRelevant(key, example.relevant) ? styles.relevant : undefined}
                    >
                      <code>{key}</code>
                    </li>
                  ))}
                </ol>
              </div>
            ))}
          </section>
        </div>
      </details>
    </li>
  );
}
