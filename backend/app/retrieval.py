"""Retrieval: vector search, BM25 keyword search, hybrid search and reranking.

- ``vector``: the question's embedding against the chunks' embeddings (cosine).
- ``bm25``: keyword search with a tokenizer for code (``bm25.py``).
- ``hybrid``: both lists fused with Reciprocal Rank Fusion: each chunk scores
  ``1 / (60 + rank)`` in each list it appears in. Ranks, not scores, are fused, so the two
  scales never need to be made comparable.
- ``rerank``: the best candidates (15 by default) go to Claude, which grades each one's
  relevance to the question; the final order follows those grades.
"""

import threading
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Literal

from app.bm25 import BM25, tokenize
from app.chunking import Chunk
from app.embeddings import Embedder
from app.llm.base import LLMClient, Message, Usage
from app.outputs import Ranking
from app.prompting import rerank_prompt, system_prompt
from app.store import ChunkStore

Mode = Literal["vector", "bm25", "hybrid"]
MODES: tuple[Mode, ...] = ("vector", "bm25", "hybrid")
RRF_K = 60


@dataclass(frozen=True)
class Hit:
    """A retrieved chunk and how it got there (ranks are 1-based)."""

    chunk: Chunk
    score: float
    vector_rank: int | None = None
    bm25_rank: int | None = None
    relevance: int | None = None  # the reranker's grade, 0-3


@dataclass(frozen=True)
class Retrieval:
    hits: list[Hit]
    candidates: int  # chunks considered before keeping the top k
    usage: Usage  # model usage of the reranking
    model: str | None = None


class Reranker:
    """Asks the model to grade the relevance of each candidate (structured output)."""

    def __init__(self, llm: LLMClient, *, max_chars: int = 1_200) -> None:
        self._llm = llm
        self._max_chars = max_chars

    @property
    def model(self) -> str:
        return self._llm.model

    def rerank(self, question: str, hits: Sequence[Hit]) -> tuple[list[Hit], Usage, str]:
        """``hits`` by decreasing relevance; ties keep their previous order."""
        if not hits:
            return [], Usage(), self._llm.model
        prompt = rerank_prompt(question, [hit.chunk for hit in hits], self._max_chars)
        reply = self._llm.respond(
            system=system_prompt("rerank"),
            messages=[Message("user", text=prompt)],
            tools=(),
            output_type=Ranking,
            max_tokens=1_024,
        )
        grades: dict[int, int] = {}
        if isinstance(reply.output, Ranking):
            for item in reply.output.scores:
                if 1 <= item.index <= len(hits):
                    grades[item.index] = max(0, min(3, item.relevance))
        order = sorted(range(len(hits)), key=lambda i: (-grades.get(i + 1, 0), i))
        reranked = [
            replace(hits[i], relevance=grades.get(i + 1, 0), score=float(grades.get(i + 1, 0)))
            for i in order
        ]
        return reranked, reply.usage, reply.model


class Retriever:
    """Finds the chunks that best answer a question, in one codebase or several."""

    def __init__(
        self,
        store: ChunkStore,
        embedder: Embedder,
        reranker: Reranker | None = None,
        *,
        candidates: int = 30,
        rerank_candidates: int = 15,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._reranker = reranker
        self._candidates = candidates
        self._rerank_candidates = rerank_candidates
        self._bm25: OrderedDict[tuple, tuple[list[Chunk], BM25]] = OrderedDict()
        self._lock = threading.Lock()

    @property
    def can_rerank(self) -> bool:
        return self._reranker is not None

    def retrieve(
        self,
        question: str,
        codebases: Sequence[str],
        k: int,
        mode: Mode = "hybrid",
        rerank: bool = False,
    ) -> Retrieval:
        if rerank and self._reranker is None:
            raise ValueError("reranking needs a model")
        n = max(self._candidates, k, self._rerank_candidates if rerank else 0)
        vector = self._vector(question, codebases, n) if mode != "bm25" else []
        keyword = self._keyword(question, codebases, n) if mode != "vector" else []
        if mode == "vector":
            fused = vector
        elif mode == "bm25":
            fused = keyword
        else:
            fused = fuse(vector, keyword)
        if not rerank or self._reranker is None:
            return Retrieval(fused[:k], len(fused), Usage())
        top = fused[: self._rerank_candidates]
        reranked, usage, model = self._reranker.rerank(question, top)
        return Retrieval(reranked[:k], len(top), usage, model)

    def _vector(self, question: str, codebases: Sequence[str], n: int) -> list[Hit]:
        [vector] = self._embedder.embed([question])
        return [
            Hit(chunk, similarity, vector_rank=rank)
            for rank, (chunk, similarity) in enumerate(
                self._store.search(vector, codebases, n), start=1
            )
        ]

    def _keyword(self, question: str, codebases: Sequence[str], n: int) -> list[Hit]:
        chunks, index = self._index(codebases)
        return [
            Hit(chunks[i], score, bm25_rank=rank)
            for rank, (i, score) in enumerate(index.top(tokenize(question), n), start=1)
        ]

    def _index(self, codebases: Sequence[str]) -> tuple[list[Chunk], BM25]:
        """The BM25 index of ``codebases``, rebuilt only when one of them changed."""
        key = self._store.versions(codebases)
        with self._lock:
            if key in self._bm25:
                self._bm25.move_to_end(key)
                return self._bm25[key]
        chunks = self._store.chunks(codebases)
        index = BM25([tokenize(f"{chunk.path} {chunk.name}\n{chunk.content}") for chunk in chunks])
        with self._lock:
            self._bm25[key] = (chunks, index)
            while len(self._bm25) > 8:
                self._bm25.popitem(last=False)
        return chunks, index


def fuse(vector: Sequence[Hit], keyword: Sequence[Hit], k: int = RRF_K) -> list[Hit]:
    """Reciprocal Rank Fusion of two rankings of chunks."""
    merged: dict[str, Hit] = {}
    for hit in vector:
        merged[hit.chunk.id] = Hit(hit.chunk, 1 / (k + (hit.vector_rank or 0)), hit.vector_rank)
    for hit in keyword:
        rank = hit.bm25_rank or 0
        found = merged.get(hit.chunk.id)
        if found is None:
            merged[hit.chunk.id] = Hit(hit.chunk, 1 / (k + rank), bm25_rank=rank)
        else:
            merged[hit.chunk.id] = replace(
                found, score=found.score + 1 / (k + rank), bm25_rank=rank
            )
    return sorted(merged.values(), key=lambda hit: -hit.score)
