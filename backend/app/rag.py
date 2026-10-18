"""The RAG pipeline: index code, retrieve the chunks that answer a question, answer it.

    indexing:  files → chunks → embeddings → ChromaDB (unchanged chunks are not re-embedded)
    querying:  question → retrieval (vector, BM25 or hybrid, then reranking) → answer

Answers are cached: the cache key includes each codebase's version, so re-indexing a
codebase makes its old answers unreachable.
"""

import hashlib
import json
import logging
import threading
import time
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

from app.cache import TTLCache
from app.chunking import Chunk, chunk_file, language_of
from app.generation import AnswerGenerator
from app.llm.base import Usage, cost_usd
from app.prompting import PROMPT_VERSION
from app.retrieval import Hit, Mode, Retriever
from app.store import ChunkStore, CodebaseInfo

log = logging.getLogger("rag")

NO_MATCH = "No indexed code matches this question. Try other words, or index more files."


class NothingIndexedError(Exception):
    """The codebases to search are not indexed."""


class GenerationUnavailableError(Exception):
    """Answering needs a model, and none is configured."""


class IndexFullError(Exception):
    """Indexing would go over the number of codebases or chunks the index holds."""


@dataclass(frozen=True)
class SourceFile:
    path: str
    content: str


@dataclass(frozen=True)
class IndexReport:
    codebase: str
    files: int
    chunks: int
    embedded: int  # new or changed chunks
    reused: int  # unchanged chunks, not embedded again
    removed: int
    kinds: dict[str, int]
    info: CodebaseInfo  # the codebase after the indexing
    duration_ms: int


@dataclass(frozen=True)
class Cost:
    """Model usage of one request."""

    calls: int = 0
    usage: Usage = field(default_factory=Usage)
    cost_usd: float = 0.0
    models: tuple[str, ...] = ()

    def add(self, model: str | None, usage: Usage) -> "Cost":
        """This cost plus one call of ``model`` (nothing when no model was called)."""
        if model is None:
            return self
        models = self.models if model in self.models else (*self.models, model)
        return Cost(
            self.calls + 1, self.usage + usage, self.cost_usd + cost_usd(model, usage), models
        )

    def merged(self, other: "Cost") -> "Cost":
        models = self.models + tuple(model for model in other.models if model not in self.models)
        return Cost(
            self.calls + other.calls,
            self.usage + other.usage,
            self.cost_usd + other.cost_usd,
            models,
        )


@dataclass(frozen=True)
class SearchResult:
    hits: list[Hit]
    codebases: list[str]
    candidates: int
    cost: Cost
    duration_ms: int


@dataclass(frozen=True)
class QueryResult:
    question: str
    answer: str
    grounded: bool
    sources_used: list[int]
    search: SearchResult
    cost: Cost
    cached: bool
    duration_ms: int


class CodeRag:
    def __init__(
        self,
        store: ChunkStore,
        retriever: Retriever,
        generator: AnswerGenerator | None,
        cache: TTLCache[QueryResult],
        *,
        max_codebases: int = 20,
        max_chunks: int = 20_000,
    ) -> None:
        self._store = store
        self._retriever = retriever
        self._generator = generator
        self._cache = cache
        self._max_codebases = max_codebases
        self._max_chunks = max_chunks
        self._index_lock = threading.Lock()  # one indexing at a time

    @property
    def store(self) -> ChunkStore:
        return self._store

    @property
    def can_answer(self) -> bool:
        return self._generator is not None

    @property
    def answer_model(self) -> str | None:
        return self._generator.model if self._generator else None

    @property
    def can_rerank(self) -> bool:
        return self._retriever.can_rerank

    @property
    def cache(self) -> TTLCache[QueryResult]:
        return self._cache

    # --- Indexing -----------------------------------------------------------------------

    def index(
        self,
        codebase: str,
        files: Sequence[SourceFile],
        replace_all: bool = False,
        keep: Iterable[str] = (),
    ) -> IndexReport:
        """Indexes ``files`` into ``codebase``. ``replace_all`` also removes the codebase's
        files that are neither in ``files`` nor in ``keep`` (the paths an indexing sent in
        other requests). Raises ``IndexFullError`` before changing anything when the index
        would get too big."""
        started = time.perf_counter()
        chunked = [(file, chunk_file(codebase, file.path, file.content)) for file in files]
        kept = {file.path for file in files} | set(keep)
        with self._index_lock:
            self._check_limits(codebase, chunked, kept if replace_all else None)
            totals: Counter[str] = Counter()
            kinds: Counter[str] = Counter()
            current = self._store.codebase(codebase)
            if replace_all and current is not None:
                for path in [path for path in current.files if path not in kept]:
                    totals.update(removed=self._store.remove_file(codebase, path), files_removed=1)
            for file, chunks in chunked:
                language = language_of(file.path) or "text"
                change = self._store.index_file(
                    codebase, file.path, language, len(file.content.splitlines()), chunks
                )
                totals.update(
                    embedded=change.added, reused=change.unchanged, removed=change.removed
                )
                kinds.update(chunk.kind for chunk in chunks)
            changed = any(totals[name] for name in ("embedded", "removed", "files_removed"))
            info = self._store.commit(codebase, changed)
        report = IndexReport(
            codebase=codebase,
            files=len(files),
            chunks=sum(kinds.values()),
            embedded=totals["embedded"],
            reused=totals["reused"],
            removed=totals["removed"],
            kinds=dict(kinds),
            info=info,
            duration_ms=_ms(started),
        )
        log.info(
            "indexed codebase=%s files=%d chunks=%d embedded=%d reused=%d ms=%d",
            codebase,
            report.files,
            report.chunks,
            report.embedded,
            report.reused,
            report.duration_ms,
        )
        return report

    def _check_limits(
        self,
        codebase: str,
        chunked: Sequence[tuple[SourceFile, list[Chunk]]],
        kept: set[str] | None,
    ) -> None:
        """``kept``: the paths that stay when the codebase is replaced (``None``: all)."""
        codebases = self._store.codebases()
        current = next((info for info in codebases if info.name == codebase), None)
        if current is None and len(codebases) >= self._max_codebases:
            raise IndexFullError(
                f"The index holds {self._max_codebases} codebases at most; delete one first."
            )
        sent = {file.path for file, _ in chunked}
        staying = (
            sum(
                info.chunks
                for path, info in current.files.items()
                if path not in sent and (kept is None or path in kept)
            )
            if current is not None
            else 0
        )
        others = sum(info.chunks for info in codebases if info.name != codebase)
        total = others + staying + sum(len(chunks) for _, chunks in chunked)
        if total > self._max_chunks:
            raise IndexFullError(
                f"The index would hold {total:,} chunks ({self._max_chunks:,} at most); "
                "delete a codebase or send fewer files."
            )

    def index_directory(self, codebase: str, root: Path) -> IndexReport:
        """Indexes every supported file under ``root`` (the demo codebases)."""
        files = [
            SourceFile(str(path.relative_to(root)), path.read_text(encoding="utf-8"))
            for path in sorted(root.rglob("*"))
            if path.is_file() and language_of(path.name)
        ]
        return self.index(codebase, files, replace_all=True)

    # --- Querying -----------------------------------------------------------------------

    def search(
        self,
        question: str,
        codebases: Sequence[str] | None,
        k: int,
        mode: Mode,
        rerank: bool,
    ) -> SearchResult:
        """The ``k`` chunks that best answer ``question`` (no answer is generated)."""
        started = time.perf_counter()
        names = self._resolve(codebases)
        retrieval = self._retriever.retrieve(question, names, k, mode, rerank)
        cost = Cost().add(retrieval.model, retrieval.usage)
        return SearchResult(retrieval.hits, names, retrieval.candidates, cost, _ms(started))

    def ask(
        self,
        question: str,
        codebases: Sequence[str] | None,
        k: int,
        mode: Mode,
        rerank: bool,
        history: Sequence[tuple[str, str]] = (),
    ) -> QueryResult:
        """Answers ``question`` from the code, citing the chunks it used."""
        if self._generator is None:
            raise GenerationUnavailableError
        started = time.perf_counter()
        names = self._resolve(codebases)
        key = self._cache_key(question, names, k, mode, rerank, history)
        cached = self._cache.get(key)
        if cached is not None:
            return replace(cached, cached=True, cost=Cost(), duration_ms=_ms(started))
        search = self.search(question, names, k, mode, rerank)
        if not search.hits:
            result = QueryResult(
                question, NO_MATCH, False, [], search, search.cost, False, _ms(started)
            )
        else:
            generated = self._generator.generate(
                question, [hit.chunk for hit in search.hits], history
            )
            cost = search.cost.add(generated.model, generated.usage)
            result = QueryResult(
                question,
                generated.answer,
                generated.grounded,
                generated.sources_used,
                search,
                cost,
                False,
                _ms(started),
            )
        self._cache.put(key, result)
        log.info(
            "query codebases=%s mode=%s rerank=%s hits=%d cost_usd=%.5f ms=%d",
            ",".join(names),
            mode,
            rerank,
            len(search.hits),
            result.cost.cost_usd,
            result.duration_ms,
        )
        return result

    def _resolve(self, codebases: Sequence[str] | None) -> list[str]:
        """The codebases to search: the requested ones, or every indexed one."""
        indexed = [info.name for info in self._store.codebases() if info.chunks]
        if codebases:
            missing = [name for name in codebases if name not in indexed]
            if missing:
                raise NothingIndexedError(f"Not indexed: {', '.join(missing)}.")
            return sorted(set(codebases))
        if not indexed:
            raise NothingIndexedError("No codebase is indexed yet.")
        return indexed

    def _cache_key(
        self,
        question: str,
        codebases: Sequence[str],
        k: int,
        mode: Mode,
        rerank: bool,
        history: Sequence[tuple[str, str]],
    ) -> str:
        payload = {
            "question": " ".join(question.split()),
            "codebases": self._store.versions(codebases),
            "k": k,
            "mode": mode,
            "rerank": rerank,
            "history": list(history),
            "model": self._generator.model if self._generator else None,
            "prompts": PROMPT_VERSION,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _ms(started: float) -> int:
    return round((time.perf_counter() - started) * 1000)
