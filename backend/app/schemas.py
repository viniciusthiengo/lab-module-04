"""Request bodies and answers of the API."""

from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.chunking import language_of
from app.paths import normalize_path
from app.retrieval import Mode

MAX_FILES = 200
MAX_KEEP = 10_000
MAX_FILE_CHARS = 200_000
MAX_REQUEST_CHARS = 3_000_000
MAX_QUESTION_CHARS = 1_000
MAX_HISTORY = 6
CODEBASE_NAME = r"^[a-z0-9][a-z0-9._-]{0,47}$"

CodebaseName = Annotated[str, Field(pattern=CODEBASE_NAME)]
SettingLabel = Literal["vector", "bm25", "hybrid", "hybrid+rerank"]

# --- Requests -----------------------------------------------------------------------------


class FileIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(examples=["src/cart.py"])
    content: str

    @field_validator("path")
    @classmethod
    def _check_path(cls, value: str) -> str:
        path = normalize_path(value)
        if language_of(path) is None:
            raise ValueError(f"'{path}' is not a supported file type.")
        return path

    @field_validator("content")
    @classmethod
    def _check_content(cls, value: str) -> str:
        if len(value) > MAX_FILE_CHARS:
            raise ValueError(f"A file is too long ({MAX_FILE_CHARS:,} characters at most).")
        if "\0" in value:
            raise ValueError("Binary files cannot be indexed.")
        return value


class IndexRequest(BaseModel):
    """Body of ``POST /index/files``."""

    model_config = ConfigDict(extra="forbid")

    codebase: str = Field(
        pattern=CODEBASE_NAME, examples=["shop-api"], description="Name of the codebase."
    )
    files: list[FileIn] = Field(min_length=1, max_length=MAX_FILES)
    replace: bool = Field(
        default=False,
        description="Remove the codebase's files that are neither in this request nor in `keep`.",
    )
    keep: list[str] = Field(
        default_factory=list,
        max_length=MAX_KEEP,
        description="With `replace`: paths sent in the other requests of the same indexing "
        "(a large upload is split into several), which are kept.",
    )

    @field_validator("keep")
    @classmethod
    def _check_keep(cls, value: list[str]) -> list[str]:
        return [normalize_path(path) for path in value]

    @model_validator(mode="after")
    def _check(self) -> Self:
        paths = [file.path for file in self.files]
        duplicates = sorted({path for path in paths if paths.count(path) > 1})
        if duplicates:
            raise ValueError(f"Two files have the same path: {duplicates[0]}.")
        if sum(len(file.content) for file in self.files) > MAX_REQUEST_CHARS:
            raise ValueError(
                f"The files are too long together ({MAX_REQUEST_CHARS:,} characters at most)."
            )
        return self


class Turn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    content: str = Field(max_length=4_000)


class SearchRequest(BaseModel):
    """Body of ``POST /search``."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    codebases: list[CodebaseName] | None = Field(
        default=None,
        max_length=20,
        description="Codebases to search; all indexed ones when omitted.",
    )
    k: int = Field(default=5, ge=1, le=10, description="How many chunks to retrieve.")
    mode: Mode = "hybrid"
    rerank: bool = Field(default=False, description="Rerank the candidates with Claude.")

    @field_validator("question")
    @classmethod
    def _check_question(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("The question is empty.")
        return value.strip()


class QueryRequest(SearchRequest):
    """Body of ``POST /query``."""

    rerank: bool = True
    history: list[Turn] = Field(
        default_factory=list,
        max_length=MAX_HISTORY,
        description="Earlier turns of the conversation, oldest first.",
    )


class EvaluateRequest(BaseModel):
    """Body of ``POST /evaluate`` and ``POST /evaluations``: runs the demo dataset."""

    model_config = ConfigDict(extra="forbid")

    k: int = Field(default=5, ge=1, le=10)
    settings: list[SettingLabel] = Field(
        default=["vector", "bm25", "hybrid", "hybrid+rerank"],
        min_length=1,
        description="Retrieval settings whose metrics to compute.",
    )
    answer_setting: SettingLabel | None = Field(
        default="hybrid+rerank",
        description="Setting whose answers are generated and judged; none to skip.",
    )
    judge_check: bool = Field(
        default=False,
        description="Also grade each question with the answer to another question: the judge "
        "should fail those (doubles the judge's calls).",
    )


# --- Answers ------------------------------------------------------------------------------


class HitView(BaseModel):
    id: str
    key: str = Field(description="codebase:path::name")
    codebase: str
    path: str
    language: str
    kind: str
    name: str
    start_line: int
    end_line: int
    content: str
    score: float
    vector_rank: int | None
    bm25_rank: int | None
    relevance: int | None = Field(description="The reranker's grade, 0-3.")


class UsageView(BaseModel):
    model_calls: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    models: list[str]


class SearchResponse(BaseModel):
    hits: list[HitView]
    codebases: list[str]
    mode: Mode
    rerank: bool
    k: int
    candidates: int
    usage: UsageView
    duration_ms: int


class QueryResponse(BaseModel):
    question: str
    answer: str = Field(description="Markdown, citing the sources as [1], [2]...")
    grounded: bool
    sources_used: list[int] = Field(description="Numbers (1-based) of the sources cited.")
    sources: list[HitView]
    codebases: list[str]
    mode: Mode
    rerank: bool
    k: int
    cached: bool
    usage: UsageView
    duration_ms: int


class FileView(BaseModel):
    path: str
    language: str
    lines: int
    chunks: int


class CodebaseView(BaseModel):
    name: str
    demo: bool = Field(description="A demo codebase of the evaluation dataset (read-only).")
    files: list[FileView]
    chunks: int
    languages: list[str]
    version: int
    updated_at: float


class IndexResponse(BaseModel):
    codebase: CodebaseView
    files: int
    chunks: int
    embedded: int = Field(description="New or changed chunks, embedded now.")
    reused: int = Field(description="Unchanged chunks, whose embeddings were kept.")
    removed: int
    kinds: dict[str, int]
    duration_ms: int


class RetrievalScoreView(BaseModel):
    retrieved: list[str]
    precision: float
    recall: float
    reciprocal_rank: float
    ndcg: float
    hit: float


class JudgementView(BaseModel):
    faithfulness: int
    relevance: int
    correctness: int
    passed: bool
    rationale: str


class AnswerScoreView(BaseModel):
    answer: str
    grounded: bool
    sources_used: list[int]
    judgement: JudgementView | None


class ExampleView(BaseModel):
    id: str
    question: str
    expected_answer: str
    relevant: list[str]
    retrieval: dict[str, RetrievalScoreView]
    answer: AnswerScoreView | None


class SettingMetricsView(BaseModel):
    precision: float = Field(description="Mean Precision@K.")
    recall: float = Field(description="Mean Recall@K.")
    mrr: float = Field(description="Mean Reciprocal Rank.")
    ndcg: float = Field(description="Mean nDCG@K.")
    hit_rate: float = Field(description="Questions with a relevant chunk in the top K.")


class JudgeCheckView(BaseModel):
    judged: int
    pass_rate: float = Field(description="Answers to other questions that passed: near 0.")
    faithfulness: float
    relevance: float
    correctness: float


class GenerationMetricsView(BaseModel):
    faithfulness: float
    relevance: float
    correctness: float
    pass_rate: float = Field(description="Answers graded 4 or 5 on every criterion.")
    grounded_rate: float
    judged: int
    check: JudgeCheckView | None = Field(
        default=None, description="The judge on answers swapped between questions."
    )


class EvaluationReportView(BaseModel):
    dataset: str
    codebases: list[str]
    k: int
    settings: list[str]
    answer_setting: str | None
    metrics: dict[str, SettingMetricsView]
    generation: GenerationMetricsView | None
    examples: list[ExampleView]
    usage: UsageView
    generator_model: str | None
    judge_model: str | None
    duration_ms: int
    created_at: float


class EvaluationRunView(BaseModel):
    id: str
    status: Literal["running", "completed", "failed"]
    done: int
    total: int
    error: str | None
    created_at: float
    finished_at: float | None
    version: int
    report: EvaluationReportView | None


class DatasetExampleView(BaseModel):
    id: str
    question: str
    expected_answer: str
    relevant: list[str]


class DatasetView(BaseModel):
    name: str
    description: str
    codebases: list[str]
    examples: list[DatasetExampleView]


class ErrorResponse(BaseModel):
    detail: str
