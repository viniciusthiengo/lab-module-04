"""The real embedding model, and the retrieval quality it gives on the demo dataset.

The model (79 MB) is part of the Docker image, where these tests run; elsewhere they are
skipped.
"""

import math
import os
from pathlib import Path

import pytest

from app.cache import TTLCache
from app.embeddings import CachedEmbedder, MiniLMEmbedder
from app.evaluation.dataset import EvalDataset
from app.evaluation.runner import Evaluator
from app.main import DEMO_DIR
from app.rag import CodeRag
from app.retrieval import Retriever
from app.store import ChunkStore

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/nonexistent"))

pytestmark = pytest.mark.skipif(
    not (MODEL_DIR / MiniLMEmbedder.name / "onnx" / "model.onnx").exists(),
    reason="the embedding model is only in the Docker image",
)


@pytest.fixture(scope="module")
def model() -> MiniLMEmbedder:
    return MiniLMEmbedder(MODEL_DIR)


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True)) / math.hypot(*a) / math.hypot(*b)


def test_related_code_is_closer(model: MiniLMEmbedder) -> None:
    question, hashing, shipping = model.embed(
        [
            "How are passwords hashed?",
            "def hash_password(password):\n    return pbkdf2_hmac('sha256', password, salt)",
            "def shipping(cart):\n    return Decimal('4.99')",
        ]
    )

    assert len(question) == 384
    assert cosine(question, hashing) > cosine(question, shipping)


def test_long_texts_are_truncated(model: MiniLMEmbedder) -> None:
    [vector] = model.embed(["word " * 2_000])

    assert len(vector) == 384


def test_retrieval_quality_on_the_demo_dataset(model: MiniLMEmbedder, dataset: EvalDataset) -> None:
    embedder = CachedEmbedder(model)
    store = ChunkStore(None, embedder)
    rag = CodeRag(store, Retriever(store, embedder), None, TTLCache(10, 60))
    for name in dataset.codebases:
        rag.index_directory(name, DEMO_DIR / name)

    report = Evaluator(rag, None).run(
        dataset, k=5, settings=["vector", "bm25", "hybrid"], answer_setting=None
    )

    hybrid, bm25 = report.metrics["hybrid"], report.metrics["bm25"]
    assert hybrid.hit_rate >= 0.85
    assert hybrid.recall >= 0.8
    assert hybrid.mrr >= 0.7
    assert hybrid.mrr > bm25.mrr
