// The evaluation's settings and metrics, and how a retrieved chunk is matched to the code a
// question is about (the backend's `metrics.matches`).

import type { EvaluationReport, SettingLabel, SettingMetrics } from "./types";

export const SETTINGS: readonly { label: SettingLabel; name: string; description: string }[] = [
  { label: "vector", name: "Vector", description: "Embedding similarity (all-MiniLM-L6-v2)." },
  { label: "bm25", name: "BM25", description: "Keyword search with a tokenizer for code." },
  { label: "hybrid", name: "Hybrid", description: "Vector and BM25 rankings fused with RRF." },
  {
    label: "hybrid+rerank",
    name: "Hybrid + rerank",
    description: "The 15 best hybrid candidates graded by Claude.",
  },
];

export const METRICS: readonly {
  key: keyof SettingMetrics;
  /** With "K" for the number of results, such as "Precision@K". */
  name: string;
  description: string;
  /** Shares are shown as percentages, means of ranks as scores. */
  as: "percent" | "score";
}[] = [
  {
    key: "precision",
    name: "Precision@K",
    description: "Share of the top K results that is relevant code.",
    as: "percent",
  },
  {
    key: "recall",
    name: "Recall@K",
    description: "Share of the relevant code found in the top K.",
    as: "percent",
  },
  {
    key: "mrr",
    name: "MRR",
    description: "Mean of 1 / rank of the first relevant result.",
    as: "score",
  },
  {
    key: "ndcg",
    name: "nDCG@K",
    description: "Relevant results count more the higher they rank (1 is the best ranking).",
    as: "score",
  },
  {
    key: "hit_rate",
    name: "Hit@K",
    description: "Share of the questions with relevant code in the top K.",
    as: "percent",
  },
];

/** Cost of an evaluation per question, in US dollars, measured on real runs. */
export const COST_PER_QUESTION = { rerank: 0.0042, answerAndJudge: 0.0077, judgeCheck: 0.005 };

/** What an evaluation of `questions` questions should cost. */
export function estimateCost(
  questions: number,
  { rerank, judge, check }: { rerank: boolean; judge: boolean; check: boolean },
): number {
  // Judging answers the hybrid+rerank setting, so it implies reranking.
  const perQuestion =
    (rerank || judge ? COST_PER_QUESTION.rerank : 0) +
    (judge ? COST_PER_QUESTION.answerAndJudge : 0) +
    (judge && check ? COST_PER_QUESTION.judgeCheck : 0);
  return questions * perQuestion;
}

/** The rank of the first relevant result (1-based), from its reciprocal; `null` for none. */
export function firstRelevantRank(reciprocalRank: number): number | null {
  return reciprocalRank > 0 ? Math.round(1 / reciprocalRank) : null;
}

export function settingName(label: SettingLabel): string {
  return SETTINGS.find((setting) => setting.label === label)?.name ?? label;
}

/** The settings of `report` with the best value of `metric` (several when tied). */
export function best(report: EvaluationReport, metric: keyof SettingMetrics): SettingLabel[] {
  const values = report.settings
    .map((label) => [label, report.metrics[label]?.[metric]] as const)
    .filter((entry): entry is readonly [SettingLabel, number] => entry[1] !== undefined);
  const top = Math.max(...values.map(([, value]) => value));
  return values.filter(([, value]) => value === top).map(([label]) => label);
}

/** Whether the chunk `key` is the `target` code, contains it or is part of it. */
export function matches(key: string, target: string): boolean {
  const [keyFile, keyName = ""] = key.split("::");
  const [targetFile, targetName = ""] = target.split("::");
  if (keyFile !== targetFile) return false;
  return (
    keyName === targetName ||
    keyName.startsWith(`${targetName}.`) ||
    targetName.startsWith(`${keyName}.`)
  );
}

/** Whether `key` is relevant to a question about `targets`. */
export function isRelevant(key: string, targets: string[]): boolean {
  return targets.some((target) => matches(key, target));
}
