"""Text embeddings: sentence-transformers' all-MiniLM-L6-v2, run with ONNX Runtime.

ChromaDB ships the model as ONNX, so no PyTorch is needed: 384 dimensions, a few
milliseconds per chunk on a CPU, nothing to pay. Its inputs are truncated at 256 tokens,
which is why every chunk starts with a header that says what it is.
"""

import hashlib
import threading
from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol


class Embedder(Protocol):
    @property
    def name(self) -> str: ...

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class MiniLMEmbedder:
    """all-MiniLM-L6-v2 through ChromaDB's ONNX embedding function. The model (79 MB) is
    downloaded into ``model_dir`` on first use; the Docker image downloads it at build time."""

    name = "all-MiniLM-L6-v2"

    def __init__(self, model_dir: Path | None = None) -> None:
        from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

        self._function = ONNXMiniLM_L6_V2()
        if model_dir is not None:
            self._function.DOWNLOAD_PATH = model_dir / ONNXMiniLM_L6_V2.MODEL_NAME
        self._lock = threading.Lock()

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        with self._lock:  # one ONNX session, used by one thread at a time
            vectors = self._function(list(texts))
        return [[float(value) for value in vector] for vector in vectors]


class CachedEmbedder:
    """Remembers the vectors of the last ``max_entries`` texts, so repeated questions and
    re-indexed chunks are not embedded again."""

    def __init__(self, embedder: Embedder, max_entries: int = 20_000) -> None:
        self._embedder = embedder
        self._max = max_entries
        self._vectors: OrderedDict[str, list[float]] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    @property
    def name(self) -> str:
        return self._embedder.name

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        keys = [hashlib.sha256(text.encode()).hexdigest() for text in texts]
        with self._lock:
            found = {key: self._vectors[key] for key in keys if key in self._vectors}
            for key in found:
                self._vectors.move_to_end(key)
        missing = [(key, text) for key, text in zip(keys, texts, strict=True) if key not in found]
        unique = list(dict(missing).items())
        computed = self._embedder.embed([text for _, text in unique]) if unique else []
        with self._lock:
            for (key, _), vector in zip(unique, computed, strict=True):
                found[key] = vector
                self._vectors[key] = vector
            while len(self._vectors) > self._max:
                self._vectors.popitem(last=False)
            self.hits += len(keys) - len(missing)
            self.misses += len(missing)
        return [found[key] for key in keys]
