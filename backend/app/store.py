"""The vector store: chunks and their embeddings in ChromaDB, and a registry of codebases.

All codebases share one Chroma collection (cosine distance); each chunk carries its codebase,
path, kind, name and lines as metadata, so one query can search one codebase or several.
Re-indexing a file only embeds the chunks that changed: a chunk's id is a hash of its
content and position, and ids that already exist are kept as they are.
"""

import json
import threading
import time
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.chunking import Chunk
from app.embeddings import Embedder

COLLECTION = "code_chunks"


@dataclass(frozen=True)
class FileInfo:
    path: str
    language: str
    lines: int
    chunks: int


@dataclass
class CodebaseInfo:
    name: str
    files: dict[str, FileInfo] = field(default_factory=dict)
    version: int = 0
    updated_at: float = 0.0

    @property
    def chunks(self) -> int:
        return sum(file.chunks for file in self.files.values())

    @property
    def languages(self) -> list[str]:
        return sorted({file.language for file in self.files.values()})

    def snapshot(self) -> "CodebaseInfo":
        """A copy that later changes to the registry do not affect."""
        return CodebaseInfo(self.name, dict(self.files), self.version, self.updated_at)


@dataclass(frozen=True)
class FileChange:
    """What indexing one file did."""

    path: str
    added: int
    unchanged: int
    removed: int


class _Registry:
    """The codebases and their files, saved as JSON next to the Chroma data."""

    def __init__(self, path: Path | None) -> None:
        self._path = path
        self.codebases: dict[str, CodebaseInfo] = {}
        if path is not None and path.exists():
            for name, raw in json.loads(path.read_text()).items():
                files = {p: FileInfo(**info) for p, info in raw["files"].items()}
                self.codebases[name] = CodebaseInfo(name, files, raw["version"], raw["updated_at"])

    def save(self) -> None:
        if self._path is None:
            return
        data = {
            name: {
                "files": {path: asdict(info) for path, info in codebase.files.items()},
                "version": codebase.version,
                "updated_at": codebase.updated_at,
            }
            for name, codebase in self.codebases.items()
        }
        temporary = self._path.with_suffix(".tmp")
        temporary.write_text(json.dumps(data, indent=1))
        temporary.replace(self._path)


class ChunkStore:
    """Chunks in ChromaDB: persistent under ``data_dir``, in memory when it is ``None``."""

    def __init__(self, data_dir: Path | None, embedder: Embedder) -> None:
        settings = ChromaSettings(anonymized_telemetry=False)
        if data_dir is None:
            # In-memory clients of one process share their data: a collection of its own
            # keeps this store apart from any other.
            client = chromadb.EphemeralClient(settings=settings)
            collection, registry_path = f"{COLLECTION}-{uuid.uuid4().hex[:12]}", None
        else:
            data_dir.mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(path=str(data_dir / "chroma"), settings=settings)
            collection, registry_path = COLLECTION, data_dir / "codebases.json"
        self._client = client
        self._collection = client.get_or_create_collection(
            collection, metadata={"hnsw:space": "cosine"}, embedding_function=None
        )
        self._embedder = embedder
        self._registry = _Registry(registry_path)
        self._lock = threading.RLock()

    # --- Reading ------------------------------------------------------------------------

    def codebases(self) -> list[CodebaseInfo]:
        with self._lock:
            infos = sorted(self._registry.codebases.values(), key=lambda info: info.name)
            return [info.snapshot() for info in infos]

    def codebase(self, name: str) -> CodebaseInfo | None:
        with self._lock:
            info = self._registry.codebases.get(name)
            return info.snapshot() if info is not None else None

    def versions(self, names: Iterable[str]) -> tuple[tuple[str, int], ...]:
        """Identifies the content of ``names``: it changes whenever one is re-indexed."""
        with self._lock:
            return tuple(
                (name, self._registry.codebases[name].version)
                for name in sorted(names)
                if name in self._registry.codebases
            )

    def chunks(self, codebases: Sequence[str]) -> list[Chunk]:
        """Every chunk of ``codebases``, in a stable order (codebase, path, line)."""
        if not codebases:
            return []
        result = self._collection.get(
            where=_in_codebases(codebases), include=["documents", "metadatas"]
        )
        chunks = [
            _chunk(document, metadata)
            for document, metadata in zip(
                result["documents"] or [], result["metadatas"] or [], strict=True
            )
        ]
        return sorted(chunks, key=lambda chunk: (chunk.codebase, chunk.path, chunk.start_line))

    def search(
        self, vector: list[float], codebases: Sequence[str], n: int
    ) -> list[tuple[Chunk, float]]:
        """The ``n`` chunks nearest to ``vector``, with their cosine similarity."""
        available = sum(info.chunks for info in self.codebases() if info.name in codebases)
        if not codebases or available == 0:
            return []
        result = self._collection.query(
            query_embeddings=[vector],
            n_results=min(n, available),
            where=_in_codebases(codebases),
            include=["documents", "metadatas", "distances"],
        )
        documents = (result["documents"] or [[]])[0]
        metadatas = (result["metadatas"] or [[]])[0]
        distances = (result["distances"] or [[]])[0]
        return [
            (_chunk(document, metadata), 1.0 - distance)
            for document, metadata, distance in zip(documents, metadatas, distances, strict=True)
        ]

    # --- Writing ------------------------------------------------------------------------

    def index_file(
        self, codebase: str, path: str, language: str, lines: int, chunks: Sequence[Chunk]
    ) -> FileChange:
        """Stores the chunks of one file, replacing the file's previous chunks."""
        with self._lock:
            existing = set(self._collection.get(where=_file(codebase, path), include=[])["ids"])
            new = [chunk for chunk in chunks if chunk.id not in existing]
            stale = existing - {chunk.id for chunk in chunks}
            if new:
                vectors = self._embedder.embed([chunk.embedding_text for chunk in new])
                self._collection.add(
                    ids=[chunk.id for chunk in new],
                    embeddings=vectors,  # type: ignore[arg-type]
                    documents=[chunk.content for chunk in new],
                    metadatas=[_metadata(chunk) for chunk in new],
                )
            if stale:
                self._collection.delete(ids=list(stale))
            info = self._registry.codebases.setdefault(codebase, CodebaseInfo(codebase))
            info.files[path] = FileInfo(path, language, lines, len(chunks))
            return FileChange(path, len(new), len(chunks) - len(new), len(stale))

    def remove_file(self, codebase: str, path: str) -> int:
        """Removes a file and its chunks; answers how many chunks it had."""
        with self._lock:
            ids = self._collection.get(where=_file(codebase, path), include=[])["ids"]
            if ids:
                self._collection.delete(ids=ids)
            info = self._registry.codebases.get(codebase)
            if info is not None:
                info.files.pop(path, None)
            return len(ids)

    def commit(self, codebase: str, changed: bool = True) -> CodebaseInfo:
        """Ends a round of changes to ``codebase`` and saves the registry. The version only
        grows when its chunks ``changed``, so cached answers stay valid otherwise."""
        with self._lock:
            info = self._registry.codebases.setdefault(codebase, CodebaseInfo(codebase))
            if changed or info.version == 0:
                info.version += 1
                info.updated_at = time.time()
            self._registry.save()
            return info.snapshot()

    def remove_codebase(self, name: str) -> bool:
        with self._lock:
            if name not in self._registry.codebases:
                return False
            self._collection.delete(where={"codebase": name})
            del self._registry.codebases[name]
            self._registry.save()
            return True


def _in_codebases(codebases: Sequence[str]) -> dict[str, Any]:
    return {"codebase": {"$in": list(codebases)}}


def _file(codebase: str, path: str) -> dict[str, Any]:
    return {"$and": [{"codebase": codebase}, {"path": path}]}


def _metadata(chunk: Chunk) -> dict[str, Any]:
    return {
        "codebase": chunk.codebase,
        "path": chunk.path,
        "language": chunk.language,
        "kind": chunk.kind,
        "name": chunk.name,
        "start_line": chunk.start_line,
        "end_line": chunk.end_line,
    }


def _chunk(document: str | None, metadata: Any) -> Chunk:
    return Chunk(
        codebase=str(metadata["codebase"]),
        path=str(metadata["path"]),
        language=str(metadata["language"]),
        kind=str(metadata["kind"]),
        name=str(metadata["name"]),
        start_line=int(metadata["start_line"]),
        end_line=int(metadata["end_line"]),
        content=document or "",
    )
