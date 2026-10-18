import type { CSSProperties } from "react";

import { grade, percent } from "@/lib/format";
import { settingName } from "@/lib/metrics";
import type { EvaluationReport, GenerationMetrics } from "@/lib/types";

import styles from "./Evaluation.module.css";
import ui from "./ui.module.css";

const CRITERIA: {
  key: "faithfulness" | "relevance" | "correctness";
  name: string;
  text: string;
}[] = [
  { key: "faithfulness", name: "Faithfulness", text: "Claims supported by the retrieved code." },
  { key: "relevance", name: "Relevance", text: "Answers the question that was asked." },
  { key: "correctness", name: "Correctness", text: "Agrees with the reference answer." },
];

interface Props {
  report: EvaluationReport;
  generation: GenerationMetrics;
}

/** The judge's mean grades (1 to 5) and the share of answers that pass. */
export function JudgeScores({ report, generation }: Props) {
  return (
    <>
      <div className={styles.scores}>
        {CRITERIA.map((criterion) => (
          <div key={criterion.key} className={styles.score}>
            <span className={styles.scoreName}>{criterion.name}</span>
            <span className={styles.scoreValue}>
              {grade(generation[criterion.key])}
              <small>/5</small>
            </span>
            <span
              className={styles.meter}
              style={{ "--value": (generation[criterion.key] - 1) / 4 } as CSSProperties}
              aria-hidden
            />
            <span className={ui.hint}>{criterion.text}</span>
          </div>
        ))}
        <div className={styles.score}>
          <span className={styles.scoreName}>Pass rate</span>
          <span className={styles.scoreValue}>{percent(generation.pass_rate)}</span>
          <span
            className={styles.meter}
            style={{ "--value": generation.pass_rate } as CSSProperties}
            aria-hidden
          />
          <span className={ui.hint}>Every grade 4 or 5.</span>
        </div>
        <div className={styles.score}>
          <span className={styles.scoreName}>Grounded</span>
          <span className={styles.scoreValue}>{percent(generation.grounded_rate)}</span>
          <span
            className={styles.meter}
            style={{ "--value": generation.grounded_rate } as CSSProperties}
            aria-hidden
          />
          <span className={ui.hint}>Answers the model found in the code.</span>
        </div>
      </div>
      {generation.check && (
        <p className={styles.check}>
          <b>Judge check:</b> graded with the answer to another question,{" "}
          {percent(generation.check.pass_rate)} of {generation.check.judged} answers pass
          (correctness {grade(generation.check.correctness)}/5, faithfulness{" "}
          {grade(generation.check.faithfulness)}/5).{" "}
          {generation.check.pass_rate <= 0.1
            ? "The judge fails wrong answers, so its grades above are meaningful."
            : "The judge passes too many wrong answers: read its grades with care."}
        </p>
      )}
      <p className={ui.hint}>
        {generation.judged} answers by {report.generator_model ?? "Claude"} from the{" "}
        {report.answer_setting ? settingName(report.answer_setting) : ""} retrieval, graded by{" "}
        {report.judge_model ?? "Claude"} against the retrieved code and each question&apos;s
        reference answer.
      </p>
    </>
  );
}
