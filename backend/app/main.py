"""FastAPI application of the codebase RAG system."""

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Callable
from concurrent.futures import Executor, ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, Query, Request
from fastapi import Path as PathParam
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.cache import TTLCache
from app.config import Settings
from app.embeddings import CachedEmbedder, Embedder, MiniLMEmbedder
from app.evaluation.dataset import load_dataset
from app.evaluation.judge import Judge
from app.evaluation.runner import EvaluationReport, Evaluator, Progress
from app.generation import AnswerGenerator
from app.jobs import EvaluationRun, EvaluationRuns, RunNotFoundError
from app.limits import BudgetExceededError, DailyBudget, RateLimitedError, RateLimiter
from app.llm.anthropic_client import AnthropicClient
from app.llm.base import LLMClient, LLMError
from app.llm.metering import MeteredLLM
from app.rag import (
    CodeRag,
    GenerationUnavailableError,
    IndexFullError,
    NothingIndexedError,
    SourceFile,
)
from app.retrieval import Reranker, Retriever
from app.schemas import (
    CODEBASE_NAME,
    CodebaseView,
    DatasetView,
    ErrorResponse,
    EvaluateRequest,
    EvaluationReportView,
    EvaluationRunView,
    IndexRequest,
    IndexResponse,
    QueryRequest,
    QueryResponse,
    SearchRequest,
    SearchResponse,
)
from app.store import ChunkStore
from app.views import (
    codebase_view,
    dataset_view,
    index_view,
    query_view,
    report_view,
    run_view,
    search_view,
)

log = logging.getLogger("rag")

DEMO_DIR = Path(__file__).resolve().parent.parent / "demo"
MAX_WAIT_SECONDS = 25.0
POLL_SECONDS = 0.2
MAX_RUNNING_EVALUATIONS = 2

DESCRIPTION = """
Questions and answers over codebases, with an evaluation suite.

- **Indexing**: files are split into functions, classes and methods (Python, JavaScript,
  TypeScript) or sections (Markdown), embedded with all-MiniLM-L6-v2 and stored in ChromaDB.
  Re-indexing only embeds the chunks that changed.
- **Querying**: vector, BM25 or hybrid retrieval (Reciprocal Rank Fusion), optionally reranked
  by Claude, then an answer from Claude that cites the code it used. A query can span several
  codebases; answers are cached.
- **Evaluation**: Precision@K, Recall@K, MRR and nDCG@K of each retrieval setting on the demo
  dataset, and an LLM judge that grades the answers. `POST /evaluate` answers with the report;
  `POST /evaluations` runs it in the background, followed with
  `GET /evaluations/{id}?since=<version>&wait=20`.
"""

ERRORS = {
    404: {"model": ErrorResponse, "description": "Unknown codebase or evaluation run."},
    409: {"model": ErrorResponse, "description": "Not indexed, read-only or index full."},
    422: {"model": ErrorResponse, "description": "Invalid request."},
    429: {"model": ErrorResponse, "description": "Rate limit or daily budget reached."},
    502: {"model": ErrorResponse, "description": "The AI provider failed."},
    503: {"model": ErrorResponse, "description": "No AI provider configured, or server busy."},
}

CodebaseName = Annotated[str, PathParam(pattern=CODEBASE_NAME)]
RunId = Annotated[str, PathParam(pattern=r"^[A-Za-z0-9_-]{8,32}$")]


class ModelRequiredError(Exception):
    """The request needs Claude, and no API key is configured."""


class CodebaseNotFoundError(Exception):
    """No indexed codebase has this name."""


class DemoCodebaseError(Exception):
    """The demo codebases belong to the evaluation dataset: they cannot be changed."""


class ServerBusyError(Exception):
    """Too many evaluations are running."""


def build_llms(settings: Settings) -> tuple[LLMClient | None, LLMClient | None]:
    """Claude for the answers and the reranking, and for the judge; ``None`` without an API
    key. Thinking is disabled: the costs assume answers without thinking tokens."""
    key = settings.anthropic_api_key
    if not key:
        return None, None
    llm = AnthropicClient(key, settings.llm_model, thinking="disabled")
    if settings.judge_model == settings.llm_model:
        return llm, llm
    return llm, AnthropicClient(key, settings.judge_model, thinking="disabled")


def create_app(
    settings: Settings | None = None,
    llm: LLMClient | None = None,
    *,
    judge_llm: LLMClient | None = None,
    embedder: Embedder | None = None,
    runner: Executor | None = None,
    rate_clock: Callable[[], float] = time.monotonic,
) -> FastAPI:
    """Builds the application; tests pass their own models (``llm`` is also the judge by
    default), embedder, runner and clock."""
    settings = settings or Settings.from_env()
    if llm is None:
        llm, judge_llm = build_llms(settings)
    judge_llm = judge_llm or llm
    budget = DailyBudget(settings.daily_budget_usd)
    metered = MeteredLLM(llm, budget) if llm else None
    embeddings = CachedEmbedder(embedder or MiniLMEmbedder(settings.model_dir))
    store = ChunkStore(settings.data_dir, embeddings)
    rag = CodeRag(
        store,
        Retriever(store, embeddings, Reranker(metered) if metered else None),
        AnswerGenerator(metered) if metered else None,
        TTLCache(settings.query_cache_size, settings.query_cache_seconds),
        max_codebases=settings.max_codebases,
        max_chunks=settings.max_chunks,
    )
    judge = Judge(MeteredLLM(judge_llm, budget)) if judge_llm else None
    evaluator = Evaluator(rag, judge)
    dataset = load_dataset(DEMO_DIR / "dataset.json")
    demo = set(dataset.codebases)
    latest_file = settings.data_dir / "latest-evaluation.json" if settings.data_dir else None

    def save_latest(run: EvaluationRun) -> None:
        """Keeps the newest evaluation across restarts, next to the index."""
        if latest_file is not None:
            temporary = latest_file.with_suffix(".tmp")
            temporary.write_text(run_view(run).model_dump_json(), encoding="utf-8")
            temporary.replace(latest_file)

    def saved_latest() -> EvaluationRunView | None:
        if latest_file is None or not latest_file.exists():
            return None
        try:
            return EvaluationRunView.model_validate_json(latest_file.read_text(encoding="utf-8"))
        except ValueError:  # written by another version: ignored
            log.warning("ignoring unreadable %s", latest_file)
            return None

    own_runner = runner is None
    runner = runner or ThreadPoolExecutor(MAX_RUNNING_EVALUATIONS, thread_name_prefix="eval")
    runs = EvaluationRuns(runner, on_complete=save_latest)
    limiters = {
        "query": RateLimiter(settings.queries_per_hour, 3600, rate_clock),
        "index": RateLimiter(settings.index_per_hour, 3600, rate_clock),
        "evaluation": RateLimiter(settings.evaluations_per_hour, 3600, rate_clock),
    }
    log.info("starting with %r", settings)

    def index_demo() -> list[IndexResponse]:
        responses = []
        for name in dataset.codebases:
            report = rag.index_directory(name, DEMO_DIR / name)
            responses.append(index_view(report, demo=True))
        return responses

    def ensure_demo() -> None:
        """The evaluation dataset asks about the demo codebases: index them when missing."""
        if any(store.codebase(name) is None for name in dataset.codebases):
            index_demo()

    def require_model(needed: bool = True) -> None:
        if needed and metered is None:
            raise ModelRequiredError

    def evaluation(body: EvaluateRequest) -> Callable[[Progress | None], EvaluationReport]:
        """Checks the request, then answers the function that runs it."""
        require_model(body.answer_setting is not None or "hybrid+rerank" in body.settings)
        ensure_demo()
        labels = list(dict.fromkeys(body.settings))

        def run(progress: Progress | None) -> EvaluationReport:
            return evaluator.run(
                dataset,
                k=body.k,
                settings=labels,
                answer_setting=body.answer_setting,
                judge_check=body.judge_check,
                progress=progress,
            )

        return run

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if settings.index_demo_on_start:
            await asyncio.to_thread(ensure_demo)
        yield
        if own_runner:
            runner.shutdown(wait=False, cancel_futures=True)

    app = FastAPI(
        title="Codebase RAG API", version="1.0.0", description=DESCRIPTION, lifespan=lifespan
    )
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.cors_origins),
            allow_methods=["GET", "POST", "DELETE"],
            allow_headers=["Content-Type"],
        )
    _register_error_handlers(app)

    def limit(kind: str) -> Callable[[Request], None]:
        def check(request: Request) -> None:
            limiters[kind].check(_client_id(request))

        return check

    QueryLimit = Annotated[None, Depends(limit("query"))]
    IndexLimit = Annotated[None, Depends(limit("index"))]
    EvaluationLimit = Annotated[None, Depends(limit("evaluation"))]

    @app.get("/health", tags=["meta"])
    def health() -> dict[str, object]:
        """Liveness, models, size of the index, caches and today's spending (never the key)."""
        codebases = store.codebases()
        return {
            "status": "ok",
            "ready": metered is not None,
            "model": metered.model if metered else None,
            "judge_model": judge.model if judge else None,
            "embedding_model": embeddings.name,
            "codebases": len(codebases),
            "chunks": sum(info.chunks for info in codebases),
            "embedding_cache": {"hits": embeddings.hits, "misses": embeddings.misses},
            "query_cache": {"hits": rag.cache.hits, "misses": rag.cache.misses},
            "daily_budget_usd": settings.daily_budget_usd,
            "spent_today_usd": round(budget.spent_usd, 4),
        }

    # --- Indexing -------------------------------------------------------------------------

    @app.get("/codebases", response_model=list[CodebaseView], tags=["indexing"])
    def list_codebases() -> list[CodebaseView]:
        """The indexed codebases and their files."""
        return [codebase_view(info, info.name in demo) for info in store.codebases()]

    @app.delete("/codebases/{name}", status_code=204, tags=["indexing"], responses=ERRORS)
    def delete_codebase(name: CodebaseName, _: IndexLimit) -> None:
        """Removes a codebase and its chunks (not the demo ones)."""
        if name in demo:
            raise DemoCodebaseError(name)
        if not store.remove_codebase(name):
            raise CodebaseNotFoundError(name)

    @app.post("/index/files", response_model=IndexResponse, tags=["indexing"], responses=ERRORS)
    def index_files(body: IndexRequest, _: IndexLimit) -> IndexResponse:
        """Chunks, embeds and stores files in a codebase (created when new). Unchanged chunks
        keep their embeddings. With `replace`, the codebase's files missing from the request
        (and from `keep`) are removed: a large upload sends its batches without it, then the
        last one with `replace` and the paths of the others in `keep`."""
        if body.codebase in demo:
            raise DemoCodebaseError(body.codebase)
        files = [SourceFile(file.path, file.content) for file in body.files]
        report = rag.index(body.codebase, files, replace_all=body.replace, keep=body.keep)
        return index_view(report)

    @app.post(
        "/index/demo", response_model=list[IndexResponse], tags=["indexing"], responses=ERRORS
    )
    def index_demo_codebases(_: IndexLimit) -> list[IndexResponse]:
        """Indexes the demo codebases of the evaluation dataset (`bookshop-api`, a Python
        service, and `bookshop-web`, its TypeScript client)."""
        return index_demo()

    # --- Querying -------------------------------------------------------------------------

    @app.post("/search", response_model=SearchResponse, tags=["querying"], responses=ERRORS)
    def search(body: SearchRequest, _: QueryLimit) -> SearchResponse:
        """The code that best matches the question, without an answer. Free unless `rerank`
        is set."""
        require_model(body.rerank)
        result = rag.search(body.question, body.codebases, body.k, body.mode, body.rerank)
        return search_view(result, body.mode, body.rerank, body.k)

    @app.post("/query", response_model=QueryResponse, tags=["querying"], responses=ERRORS)
    def query(body: QueryRequest, _: QueryLimit) -> QueryResponse:
        """Answers a question about the indexed code, citing the sources it used."""
        require_model()
        history = [(turn.role, turn.content) for turn in body.history]
        result = rag.ask(body.question, body.codebases, body.k, body.mode, body.rerank, history)
        return query_view(result, body.mode, body.rerank, body.k)

    # --- Evaluation -----------------------------------------------------------------------

    @app.get("/evaluation/dataset", response_model=DatasetView, tags=["evaluation"])
    def evaluation_dataset() -> DatasetView:
        """The evaluation dataset: questions, reference answers and the code that answers
        them."""
        return dataset_view(dataset)

    @app.post(
        "/evaluate", response_model=EvaluationReportView, tags=["evaluation"], responses=ERRORS
    )
    def evaluate(body: EvaluateRequest, _: EvaluationLimit) -> EvaluationReportView:
        """Runs the evaluation suite and answers with the report (about a minute with
        answers; seconds without)."""
        return report_view(evaluation(body)(None))

    @app.post(
        "/evaluations",
        status_code=202,
        response_model=EvaluationRunView,
        tags=["evaluation"],
        responses=ERRORS,
    )
    def start_evaluation(body: EvaluateRequest, _: EvaluationLimit) -> EvaluationRunView:
        """Starts the evaluation suite in the background and answers at once."""
        if runs.running() >= MAX_RUNNING_EVALUATIONS:
            raise ServerBusyError
        return run_view(runs.start(evaluation(body)))

    @app.get(
        "/evaluations/latest",
        response_model=EvaluationRunView,
        tags=["evaluation"],
        responses=ERRORS,
    )
    def latest_evaluation() -> EvaluationRunView:
        """The newest completed evaluation, kept across restarts when the index is."""
        run = runs.latest()
        if run is not None:
            return run_view(run)
        saved = saved_latest()
        if saved is None:
            raise RunNotFoundError("latest")
        return saved

    @app.get(
        "/evaluations/{run_id}",
        response_model=EvaluationRunView,
        tags=["evaluation"],
        responses=ERRORS,
    )
    async def get_evaluation(
        run_id: RunId,
        since: Annotated[int | None, Query(ge=0, description="Last version you saw.")] = None,
        wait: Annotated[
            float, Query(ge=0, le=MAX_WAIT_SECONDS, description="Seconds to wait for a change.")
        ] = 0,
    ) -> EvaluationRunView:
        """An evaluation run. With `since` and `wait`, answers as soon as its version is newer
        than `since`, or after `wait` seconds (long polling)."""
        run = runs.get(run_id)
        if since is not None:
            deadline = time.monotonic() + wait
            while run.version <= since and time.monotonic() < deadline:
                await asyncio.sleep(POLL_SECONDS)
                run = runs.get(run_id)
        return run_view(run)

    return app


def _client_id(request: Request) -> str:
    """The client's address: from the frontend's proxy header, the platform's proxy, or the
    connection."""
    forwarded = request.headers.get("x-client-ip") or request.headers.get("x-forwarded-for", "")
    first = forwarded.split(",")[0].strip()
    return first or (request.client.host if request.client else "unknown")


def _register_error_handlers(app: FastAPI) -> None:
    """Every error answers ``{"detail": message}``."""

    def error(status: int, message: str) -> JSONResponse:
        return JSONResponse({"detail": message}, status_code=status)

    @app.exception_handler(RequestValidationError)
    async def invalid(_: Request, exc: RequestValidationError) -> JSONResponse:
        return error(422, _validation_message(exc))

    @app.exception_handler(RateLimitedError)
    async def rate_limited(_: Request, __: RateLimitedError) -> JSONResponse:
        return error(429, "Too many requests from this client; try again later.")

    @app.exception_handler(BudgetExceededError)
    async def over_budget(_: Request, exc: BudgetExceededError) -> JSONResponse:
        return error(429, str(exc))

    @app.exception_handler(ModelRequiredError)
    @app.exception_handler(GenerationUnavailableError)
    async def no_model(_: Request, __: Exception) -> JSONResponse:
        return error(
            503,
            "Answers, reranking and judging need ANTHROPIC_API_KEY on the server; search "
            "without reranking works without it.",
        )

    @app.exception_handler(NothingIndexedError)
    async def nothing_indexed(_: Request, exc: NothingIndexedError) -> JSONResponse:
        return error(409, f"{exc} Index a codebase first.")

    @app.exception_handler(IndexFullError)
    async def index_full(_: Request, exc: IndexFullError) -> JSONResponse:
        return error(409, str(exc))

    @app.exception_handler(DemoCodebaseError)
    async def demo_codebase(_: Request, exc: DemoCodebaseError) -> JSONResponse:
        return error(
            409,
            f"'{exc}' is a demo codebase of the evaluation dataset and cannot be changed; "
            "use another name.",
        )

    @app.exception_handler(CodebaseNotFoundError)
    async def codebase_not_found(_: Request, exc: CodebaseNotFoundError) -> JSONResponse:
        return error(404, f"No codebase named '{exc}' is indexed.")

    @app.exception_handler(RunNotFoundError)
    async def run_not_found(_: Request, __: RunNotFoundError) -> JSONResponse:
        return error(404, "Evaluation run not found.")

    @app.exception_handler(ServerBusyError)
    async def busy(_: Request, __: ServerBusyError) -> JSONResponse:
        return error(503, "Two evaluations are already running; try again in a minute.")

    @app.exception_handler(LLMError)
    async def provider_failed(_: Request, exc: LLMError) -> JSONResponse:
        log.warning("provider error: %s", exc)
        return error(503 if exc.retryable else 502, str(exc))


def _validation_message(exc: RequestValidationError) -> str:
    """The first validation error, as a sentence."""
    first = exc.errors()[0] if exc.errors() else {}
    kind = first.get("type")
    location = [str(part) for part in first.get("loc", ())]
    field = ".".join(location[1:])
    context = first.get("ctx") or {}
    if kind == "missing":
        return f"The field '{field}' is required." if field else "Send a JSON body."
    if kind == "extra_forbidden":
        return f"Unknown field '{field}'."
    if kind == "too_short" and field == "files":
        return "Add at least one file."
    if kind == "too_long" and field == "files":
        return f"Send at most {context.get('max_length')} files at a time."
    if kind == "string_pattern_mismatch" and location[:1] == ["path"]:
        return "Invalid codebase name or run id."
    if kind == "string_pattern_mismatch" and location[1:2] in (["codebase"], ["codebases"]):
        return (
            "A codebase name has lower-case letters, digits, '.', '_' or '-' (48 at most) and "
            "starts with a letter or a digit."
        )
    if kind == "string_too_long":
        return f"'{field}' is too long ({context.get('max_length')} characters at most)."
    if kind in ("less_than_equal", "greater_than_equal") and field:
        return f"'{field}' {str(first.get('msg', 'is out of range')).lower()}."
    if kind == "literal_error" and field:
        return f"Invalid value for '{field}': {first.get('msg', '')}."
    return str(first.get("msg", "Invalid request.")).removeprefix("Value error, ")
