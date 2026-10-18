from pathlib import Path

from app.chunking import chunk_file
from app.store import ChunkStore
from tests.fakes import HashEmbedder

CART = '''def add_item(cart, book):
    """Adds a book to the cart."""
    cart.append(book)


def total(cart):
    return sum(book.price for book in cart)
'''

AUTH = """def hash_password(password):
    return pbkdf2(password)
"""


def index(store: ChunkStore, codebase: str, path: str, content: str):
    chunks = chunk_file(codebase, path, content)
    change = store.index_file(codebase, path, "python", len(content.splitlines()), chunks)
    store.commit(codebase)
    return change


def query(store: ChunkStore, embedder: HashEmbedder, text: str, codebases: list[str], n: int = 5):
    [vector] = embedder.embed([text])
    return store.search(vector, codebases, n)


class TestIndexing:
    def test_stores_chunks_and_registers_the_file(self) -> None:
        store = ChunkStore(None, HashEmbedder())

        change = index(store, "shop", "cart.py", CART)

        assert (change.added, change.unchanged, change.removed) == (2, 0, 0)
        info = store.codebase("shop")
        assert info is not None
        assert (info.chunks, info.version, info.languages) == (2, 1, ["python"])
        assert info.files["cart.py"].lines == 7
        assert [chunk.name for chunk in store.chunks(["shop"])] == ["add_item", "total"]

    def test_unchanged_chunks_are_not_embedded_again(self) -> None:
        embedder = HashEmbedder()
        store = ChunkStore(None, embedder)
        index(store, "shop", "cart.py", CART)
        embedded = embedder.embedded

        change = index(store, "shop", "cart.py", CART)

        assert (change.added, change.unchanged, change.removed) == (0, 2, 0)
        assert embedder.embedded == embedded

    def test_a_changed_chunk_replaces_the_old_one(self) -> None:
        embedder = HashEmbedder()
        store = ChunkStore(None, embedder)
        index(store, "shop", "cart.py", CART)
        embedded = embedder.embedded

        change = index(store, "shop", "cart.py", CART.replace("book.price", "book.cost"))

        assert (change.added, change.unchanged, change.removed) == (1, 1, 1)
        assert embedder.embedded == embedded + 1
        assert "book.cost" in store.chunks(["shop"])[1].content
        assert len(store.chunks(["shop"])) == 2

    def test_remove_file_and_codebase(self) -> None:
        store = ChunkStore(None, HashEmbedder())
        index(store, "shop", "cart.py", CART)
        index(store, "shop", "auth.py", AUTH)

        assert store.remove_file("shop", "cart.py") == 2
        assert [chunk.path for chunk in store.chunks(["shop"])] == ["auth.py"]
        assert list(store.codebase("shop").files) == ["auth.py"]  # type: ignore[union-attr]
        assert store.remove_codebase("shop") is True
        assert store.remove_codebase("shop") is False
        assert store.codebases() == []
        assert store.chunks(["shop"]) == []

    def test_the_version_grows_only_when_chunks_changed(self) -> None:
        store = ChunkStore(None, HashEmbedder())
        index(store, "shop", "cart.py", CART)

        unchanged = store.commit("shop", changed=False)
        changed = store.commit("shop", changed=True)

        assert unchanged.version == 1
        assert changed.version == 2
        assert store.versions(["shop", "unknown"]) == (("shop", 2),)


class TestSearch:
    def test_finds_the_nearest_chunks_in_the_requested_codebases(self) -> None:
        embedder = HashEmbedder()
        store = ChunkStore(None, embedder)
        index(store, "shop", "cart.py", CART)
        index(store, "auth", "auth.py", AUTH)

        only_shop = query(store, embedder, "hash the password", ["shop"])
        both = query(store, embedder, "hash the password", ["shop", "auth"])

        assert {chunk.codebase for chunk, _ in only_shop} == {"shop"}
        best, similarity = both[0]
        assert best.key == "auth:auth.py::hash_password"
        assert 0 < similarity <= 1
        assert [s for _, s in both] == sorted((s for _, s in both), reverse=True)

    def test_asks_for_no_more_chunks_than_there_are(self) -> None:
        embedder = HashEmbedder()
        store = ChunkStore(None, embedder)
        index(store, "shop", "cart.py", CART)

        assert len(query(store, embedder, "cart", ["shop"], n=50)) == 2
        assert query(store, embedder, "cart", ["unknown"]) == []
        assert query(store, embedder, "cart", []) == []

    def test_in_memory_stores_are_isolated(self) -> None:
        first = ChunkStore(None, HashEmbedder())
        second = ChunkStore(None, HashEmbedder())
        index(first, "shop", "cart.py", CART)

        assert second.chunks(["shop"]) == []


def test_a_persistent_store_keeps_its_index(tmp_path: Path) -> None:
    embedder = HashEmbedder()
    store = ChunkStore(tmp_path, embedder)
    index(store, "shop", "cart.py", CART)

    reopened = ChunkStore(tmp_path, embedder)

    info = reopened.codebase("shop")
    assert info is not None
    assert (info.version, info.chunks) == (1, 2)
    assert [chunk.name for chunk in reopened.chunks(["shop"])] == ["add_item", "total"]
    assert query(reopened, embedder, "add a book", ["shop"])[0][0].name == "add_item"
    assert (tmp_path / "codebases.json").exists()
