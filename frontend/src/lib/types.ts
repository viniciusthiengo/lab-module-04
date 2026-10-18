// The backend's API, as the browser sees it through the /api proxy.

export type Mode = "vector" | "bm25" | "hybrid";
export type SettingLabel = "vector" | "bm25" | "hybrid" | "hybrid+rerank";

export interface Health {
  status: "ok";
  /** Whether Claude is configured: answers, reranking and judging need it. */
  ready: boolean;
  model: string | null;
  judge_model: string | null;
  embedding_model: string;
  codebases: number;
  chunks: number;
  embedding_cache: { hits: number; misses: number };
  query_cache: { hits: number; misses: number };
  daily_budget_usd: number;
  spent_today_usd: number;
}

export interface FileView {
  path: string;
  language: string;
  lines: number;
  chunks: number;
}

export interface Codebase {
  name: string;
  /** A demo codebase of the evaluation dataset: read-only. */
  demo: boolean;
  files: FileView[];
  chunks: number;
  languages: string[];
  version: number;
  updated_at: number;
}

export interface FileIn {
  path: string;
  content: string;
}

export interface IndexRequest {
  codebase: string;
  files: FileIn[];
  /** Remove the codebase's files that are neither in this request nor in `keep`. */
  replace: boolean;
  /** With `replace`: the paths sent in the other requests of the same indexing. */
  keep?: string[];
}

export interface IndexResult {
  codebase: Codebase;
  files: number;
  chunks: number;
  /** New or changed chunks, embedded now. */
  embedded: number;
  /** Unchanged chunks, whose embeddings were kept. */
  reused: number;
  removed: number;
  kinds: Record<string, number>;
  duration_ms: number;
}

/** A retrieved chunk of code. */
export interface Hit {
  id: string;
  /** `codebase:path::name` */
  key: string;
  codebase: string;
  path: string;
  language: string;
  kind: string;
  name: string;
  start_line: number;
  end_line: number;
  content: string;
  score: number;
  vector_rank: number | null;
  bm25_rank: number | null;
  /** The reranker's grade, 0 to 3. */
  relevance: number | null;
}

export interface Usage {
  model_calls: number;
  input_tokens: number;
  output_tokens: number;
  cost_usd: number;
  models: string[];
}

export interface SearchRequest {
  question: string;
  /** Codebases to search; every indexed one when omitted. */
  codebases?: string[];
  k: number;
  mode: Mode;
  rerank: boolean;
}

export interface Turn {
  role: "user" | "assistant";
  content: string;
}

export interface QueryRequest extends SearchRequest {
  history: Turn[];
}

export interface SearchResult {
  hits: Hit[];
  codebases: string[];
  mode: Mode;
  rerank: boolean;
  k: number;
  candidates: number;
  usage: Usage;
  duration_ms: number;
}

export interface QueryResult {
  question: string;
  /** Markdown that cites the sources as [1], [2]... */
  answer: string;
  grounded: boolean;
  /** 1-based numbers of the sources the answer cites. */
  sources_used: number[];
  sources: Hit[];
  codebases: string[];
  mode: Mode;
  rerank: boolean;
  k: number;
  cached: boolean;
  usage: Usage;
  duration_ms: number;
}

export interface RetrievalScore {
  /** Keys of the top K chunks. */
  retrieved: string[];
  precision: number;
  recall: number;
  reciprocal_rank: number;
  ndcg: number;
  hit: number;
}

export interface Judgement {
  faithfulness: number;
  relevance: number;
  correctness: number;
  passed: boolean;
  rationale: string;
}

export interface AnswerScore {
  answer: string;
  grounded: boolean;
  sources_used: number[];
  judgement: Judgement | null;
}

export interface ExampleResult {
  id: string;
  question: string;
  expected_answer: string;
  relevant: string[];
  retrieval: Partial<Record<SettingLabel, RetrievalScore>>;
  answer: AnswerScore | null;
}

export interface SettingMetrics {
  precision: number;
  recall: number;
  mrr: number;
  ndcg: number;
  hit_rate: number;
}

/** The judge grading answers given to other questions: they should fail. */
export interface JudgeCheck {
  judged: number;
  pass_rate: number;
  faithfulness: number;
  relevance: number;
  correctness: number;
}

export interface GenerationMetrics {
  faithfulness: number;
  relevance: number;
  correctness: number;
  pass_rate: number;
  grounded_rate: number;
  judged: number;
  check: JudgeCheck | null;
}

export interface EvaluationReport {
  dataset: string;
  codebases: string[];
  k: number;
  settings: SettingLabel[];
  answer_setting: SettingLabel | null;
  metrics: Partial<Record<SettingLabel, SettingMetrics>>;
  generation: GenerationMetrics | null;
  examples: ExampleResult[];
  usage: Usage;
  generator_model: string | null;
  judge_model: string | null;
  duration_ms: number;
  created_at: number;
}

export interface EvaluationRun {
  id: string;
  status: "running" | "completed" | "failed";
  done: number;
  total: number;
  error: string | null;
  created_at: number;
  finished_at: number | null;
  /** Grows with every change: the long-polling cursor. */
  version: number;
  report: EvaluationReport | null;
}

export interface EvaluateRequest {
  k: number;
  settings: SettingLabel[];
  /** The setting whose answers are generated and judged; `null` for retrieval only. */
  answer_setting: SettingLabel | null;
  /** Also grade each question with the answer to another question (doubles the judge calls). */
  judge_check?: boolean;
}

export interface DatasetExample {
  id: string;
  question: string;
  expected_answer: string;
  relevant: string[];
}

export interface Dataset {
  name: string;
  description: string;
  codebases: string[];
  examples: DatasetExample[];
}
