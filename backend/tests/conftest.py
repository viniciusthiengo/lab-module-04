from collections.abc import Callable, Iterator
from concurrent.futures import Executor, Future
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.cache import TTLCache
from app.config import Settings
from app.embeddings import CachedEmbedder
from app.evaluation.dataset import EvalDataset, load_dataset
from app.generation import AnswerGenerator
from app.llm.base import LLMClient
from app.llm.fake import FakeLLMClient
from app.main import DEMO_DIR, create_app
from app.rag import CodeRag, QueryResult, SourceFile
from app.retrieval import Reranker, Retriever
from app.store import ChunkStore
from tests.fakes import HashEmbedder, ScriptedClaude


class InlineExecutor(Executor):
    """Runs submitted work at once, in the caller's thread: a background evaluation has
    ended when the request that started it returns."""

    def submit(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Future[Any]:
        future: Future[Any] = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except BaseException as error:  # pragma: no cover - the runs report their errors
            future.set_exception(error)
        return future


class HeldExecutor(Executor):
    """Keeps submitted work without running it: evaluations stay running."""

    def __init__(self) -> None:
        self.held: list[Callable[[], Any]] = []

    def submit(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Future[Any]:
        self.held.append(lambda: fn(*args, **kwargs))
        return Future()

    def run_all(self) -> None:
        while self.held:
            self.held.pop(0)()


@dataclass
class Rig:
    """A RAG over an in-memory store, the hashing embedder and the scripted Claude."""

    rag: CodeRag
    store: ChunkStore
    embedder: HashEmbedder
    llm: FakeLLMClient | None

    def index(self, codebase: str, files: dict[str, str], replace_all: bool = False):
        return self.rag.index(
            codebase, [SourceFile(path, content) for path, content in files.items()], replace_all
        )

    def index_demo(self) -> "Rig":
        for name in DEMO_CODEBASES:
            self.rag.index_directory(name, DEMO_DIR / name)
        return self


DEMO_CODEBASES = ("bookshop-api", "bookshop-web")


def make_rig(
    llm: FakeLLMClient | str | None = "fake",
    *,
    data_dir: Path | None = None,
    embedder: HashEmbedder | None = None,
    **options: Any,
) -> Rig:
    model = FakeLLMClient(ScriptedClaude()) if llm == "fake" else llm
    hashing = embedder or HashEmbedder()
    embeddings = CachedEmbedder(hashing)
    store = ChunkStore(data_dir, embeddings)
    retriever = Retriever(store, embeddings, Reranker(model) if model else None)
    generator = AnswerGenerator(model) if model else None
    cache: TTLCache[QueryResult] = TTLCache(100, 3600)
    rag = CodeRag(store, retriever, generator, cache, **options)
    return Rig(rag, store, hashing, model)  # type: ignore[arg-type]


@pytest.fixture
def rig() -> Rig:
    return make_rig()


@pytest.fixture
def demo_rig() -> Rig:
    """The rig with the two demo codebases indexed."""
    return make_rig().index_demo()


@pytest.fixture(scope="session")
def dataset() -> EvalDataset:
    return load_dataset(DEMO_DIR / "dataset.json")


@pytest.fixture
def make_client() -> Iterator[Callable[..., TestClient]]:
    """A client of the app over the scripted Claude (or ``llm``; ``None`` is a server
    without an API key), with the demo codebases indexed at startup unless ``demo=False``.
    Background evaluations run inline unless ``runner`` says otherwise; keywords override
    settings."""
    clients: list[TestClient] = []

    def make(
        *,
        llm: LLMClient | str | None = "fake",
        judge_llm: LLMClient | None = None,
        runner: Executor | None = None,
        demo: bool = True,
        rate_clock: Callable[[], float] | None = None,
        **overrides: Any,
    ) -> TestClient:
        model = FakeLLMClient(ScriptedClaude()) if llm == "fake" else llm
        app = create_app(
            Settings(index_demo_on_start=demo, **overrides),
            llm=model,  # type: ignore[arg-type]
            judge_llm=judge_llm,
            embedder=HashEmbedder(),
            runner=runner or InlineExecutor(),
            **({"rate_clock": rate_clock} if rate_clock else {}),
        )
        client = TestClient(app)
        client.__enter__()  # runs the lifespan: the demo is indexed at startup
        clients.append(client)
        return client

    yield make
    for client in clients:
        client.__exit__(None, None, None)


@pytest.fixture
def client(make_client: Callable[..., TestClient]) -> TestClient:
    return make_client()
