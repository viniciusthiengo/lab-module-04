"use client";

import { useId, useState } from "react";

import { downloadText } from "@/lib/dom";
import { dateTime, duration, plural, usd } from "@/lib/format";
import { estimateCost } from "@/lib/metrics";
import type { Dataset, EvaluationReport, Health, SettingLabel } from "@/lib/types";
import { useEvaluation } from "@/lib/useEvaluation";

import styles from "./Evaluation.module.css";
import { ExampleList } from "./ExampleList";
import { JudgeScores } from "./JudgeScores";
import { MetricsTable } from "./MetricsTable";
import ui from "./ui.module.css";

interface Props {
  health: Health | null;
  onFinished: () => void;
}

/** Runs the evaluation suite on the dataset and shows its report. */
export function EvaluationPanel({ health, onFinished }: Props) {
  const id = useId();
  const { dataset, run, error, running, start } = useEvaluation(onFinished);
  const [k, setK] = useState(5);
  const [judge, setJudge] = useState(true);
  const [check, setCheck] = useState(false);
  const ready = health?.ready ?? false;
  const questions = dataset?.examples.length ?? 0;
  const judging = judge && ready;
  const checking = check && judging;
  const report = run?.report ?? null;
  const cost = estimateCost(questions, { rerank: ready, judge: judging, check: checking });

  function begin() {
    const settings: SettingLabel[] = ["vector", "bm25", "hybrid"];
    if (ready) settings.push("hybrid+rerank");
    void start({
      k,
      settings,
      answer_setting: judging ? "hybrid+rerank" : null,
      judge_check: checking,
    });
  }

  return (
    <section className={ui.panel} aria-labelledby="evaluation-title">
      <div className={ui.panelHeader}>
        <h2 id="evaluation-title">Evaluation</h2>
        {report && (
          <span className={ui.hint}>
            Last run {dateTime(report.created_at)} · K={report.k} ·{" "}
            {plural(report.examples.length, "question")}
          </span>
        )}
      </div>
      {dataset && <DatasetSummary dataset={dataset} />}

      <div className={styles.controls}>
        <label className={styles.control} htmlFor={`${id}-k`}>
          K
          <select
            id={`${id}-k`}
            value={k}
            disabled={running}
            onChange={(event) => setK(Number(event.currentTarget.value))}
          >
            {[3, 5, 10].map((value) => (
              <option key={value} value={value}>
                {value}
              </option>
            ))}
          </select>
        </label>
        <label className={styles.control}>
          <input
            type="checkbox"
            checked={judging}
            disabled={!ready || running}
            onChange={(event) => setJudge(event.currentTarget.checked)}
          />
          Generate and judge the answers
        </label>
        <label
          className={styles.control}
          title="Grades each question again with the answer to another question: a judge worth trusting fails them"
        >
          <input
            type="checkbox"
            checked={checking}
            disabled={!judging || running}
            onChange={(event) => setCheck(event.currentTarget.checked)}
          />
          Check the judge
        </label>
        <button type="button" className={ui.primary} disabled={running || !dataset} onClick={begin}>
          {running ? (
            <>
              <span className={ui.spinner} aria-hidden /> Running…
            </>
          ) : (
            "Run the evaluation"
          )}
        </button>
        <span className={ui.hint}>
          {ready
            ? `About ${usd(cost)} of Claude calls.`
            : "Retrieval metrics only: the server has no Claude API key (free)."}
        </span>
      </div>

      {running && run && run.total > 0 && (
        <div
          className={styles.progress}
          role="progressbar"
          aria-valuemin={0}
          aria-valuemax={run.total}
          aria-valuenow={run.done}
        >
          <span style={{ width: `${(run.done / run.total) * 100}%` }} />
          <p>
            {run.done} of {plural(run.total, "question")}
          </p>
        </div>
      )}
      {error && (
        <p className={ui.alert} role="alert">
          {error}
        </p>
      )}
      {report ? (
        <Report report={report} />
      ) : (
        !running && (
          <p className={ui.empty}>
            No evaluation has run on this server yet. Run one to measure the retrieval settings and
            the answers.
          </p>
        )
      )}
    </section>
  );
}

function DatasetSummary({ dataset }: { dataset: Dataset }) {
  return (
    <details className={styles.dataset}>
      <summary>
        Dataset <b>{dataset.name}</b>: {plural(dataset.examples.length, "question")} about{" "}
        {dataset.codebases.join(" and ")}
      </summary>
      <p className={ui.hint}>{dataset.description}</p>
      <ol>
        {dataset.examples.map((example) => (
          <li key={example.id}>
            <b>{example.question}</b>
            <p>{example.expected_answer}</p>
            <p className={styles.targets}>
              {example.relevant.map((target) => (
                <code key={target}>{target}</code>
              ))}
            </p>
          </li>
        ))}
      </ol>
    </details>
  );
}

function Report({ report }: { report: EvaluationReport }) {
  return (
    <div className={styles.report}>
      <p className={styles.runMeta}>
        {duration(report.duration_ms)} · {usd(report.usage.cost_usd)} ·{" "}
        {plural(report.usage.model_calls, "model call")}
        {report.generator_model && ` · answers: ${report.generator_model}`}
        {report.judge_model && ` · judge: ${report.judge_model}`}{" "}
        <button
          type="button"
          className={ui.link}
          onClick={() =>
            downloadText(
              `evaluation-${report.dataset}-k${report.k}.json`,
              JSON.stringify(report, null, 2),
              "application/json",
            )
          }
        >
          Download the report (JSON)
        </button>
      </p>
      <h3 className={styles.subtitle}>Retrieval</h3>
      <MetricsTable report={report} />
      {report.generation && (
        <>
          <h3 className={styles.subtitle}>Answers, graded by the judge</h3>
          <JudgeScores report={report} generation={report.generation} />
        </>
      )}
      <h3 className={styles.subtitle}>Questions</h3>
      <ExampleList report={report} />
    </div>
  );
}
