import pytest

from app.llm.fake import FakeLLMClient
from app.main import DEMO_DIR
from app.outputs import Answer
from app.rag import (
    NO_MATCH,
    GenerationUnavailableError,
    IndexFullError,
    NothingIndexedError,
    SourceFile,
)
from tests.conftest import Rig, make_rig
from tests.fakes import ScriptedClaude

CART = '''"""Cart."""


def add_item(cart, book):
    cart.append(book)


def total(cart):
    return sum(book.price for book in cart)
'''

AUTH = """def hash_password(password):
    return pbkdf2(password)
"""

README = "# Shop\n\nA shop.\n\n## Configuration\n\nSet `SHOP_SECRET`.\n"


class TestIndex:
    def test_reports_what_was_indexed(self, rig: Rig) -> None:
        report = rig.index("shop", {"cart.py": CART, "auth.py": AUTH, "README.md": README})

        assert (report.files, report.chunks, report.embedded, report.reused) == (3, 6, 6, 0)
        assert report.kinds == {"module": 1, "function": 3, "section": 2}
        assert report.info.languages == ["markdown", "python"]
        assert rig.store.codebase("shop").version == 1  # type: ignore[union-attr]

    def test_indexing_the_same_files_again_embeds_nothing(self, rig: Rig) -> None:
        rig.index("shop", {"cart.py": CART})

        report = rig.index("shop", {"cart.py": CART})

        assert (report.embedded, report.reused, report.removed) == (0, 3, 0)
        assert rig.store.codebase("shop").version == 1  # type: ignore[union-attr]

    def test_replace_removes_the_files_missing_from_the_request(self, rig: Rig) -> None:
        rig.index("shop", {"cart.py": CART, "auth.py": AUTH})

        kept = rig.index("shop", {"cart.py": CART})
        replaced = rig.index("shop", {"cart.py": CART}, replace_all=True)

        assert kept.removed == 0
        assert replaced.removed == 1
        assert list(rig.store.codebase("shop").files) == ["cart.py"]  # type: ignore[union-attr]
        assert rig.store.codebase("shop").version == 2  # type: ignore[union-attr]

    def test_replace_keeps_the_paths_sent_in_other_requests(self, rig: Rig) -> None:
        rig.index("shop", {"cart.py": CART, "auth.py": AUTH, "README.md": README})
        rig.index("shop", {"cart.py": CART})  # first batch of a new upload

        report = rig.rag.index(
            "shop", [SourceFile("auth.py", AUTH)], replace_all=True, keep=["cart.py"]
        )

        assert report.removed == 2  # the README's sections
        assert sorted(rig.store.codebase("shop").files) == ["auth.py", "cart.py"]  # type: ignore[union-attr]

    def test_too_many_codebases_are_refused_before_any_change(self) -> None:
        rig = make_rig(max_codebases=1)
        rig.index("shop", {"cart.py": CART})
        embedded = rig.embedder.embedded

        with pytest.raises(IndexFullError, match="1 codebases at most"):
            rig.index("auth", {"auth.py": AUTH})

        assert [info.name for info in rig.store.codebases()] == ["shop"]
        assert rig.embedder.embedded == embedded
        rig.index("shop", {"auth.py": AUTH})  # an existing codebase can still grow

    def test_too_many_chunks_are_refused_before_any_change(self) -> None:
        rig = make_rig(max_chunks=4)
        rig.index("shop", {"cart.py": CART})

        with pytest.raises(IndexFullError, match="would hold 6 chunks"):
            rig.index("other", {"README.md": README, "auth.py": AUTH})

        assert rig.store.codebase("other") is None
        rig.index("shop", {"auth.py": AUTH, "cart.py": CART}, replace_all=True)  # 4 in all

    def test_index_directory_reads_the_supported_files(self, rig: Rig) -> None:
        report = rig.rag.index_directory("bookshop-web", DEMO_DIR / "bookshop-web")

        assert report.files == 7
        assert "src/api/client.ts" in rig.store.codebase("bookshop-web").files  # type: ignore[union-attr]


class TestAsk:
    def test_answers_from_the_retrieved_code_and_cites_it(self, demo_rig: Rig) -> None:
        result = demo_rig.rag.ask("How are passwords hashed?", None, 5, "hybrid", True)

        assert result.answer.startswith("It is done in `hash_password`")
        assert (result.grounded, result.sources_used, result.cached) == (True, [1], False)
        assert result.search.hits[0].chunk.name == "hash_password"
        assert result.search.codebases == ["bookshop-api", "bookshop-web"]
        assert result.cost.calls == 2  # reranking and answer
        assert result.cost.cost_usd > 0
        answer_request = demo_rig.llm.calls_for(Answer)[0]  # type: ignore[union-attr]
        assert '<source id="1" codebase="bookshop-api" path="bookshop/auth.py"' in (
            answer_request.prompt
        )
        assert answer_request.prompt.endswith("Question: How are passwords hashed?")

    def test_the_same_question_is_answered_from_the_cache(self, demo_rig: Rig) -> None:
        first = demo_rig.rag.ask("How are passwords hashed?", None, 5, "hybrid", False)
        calls = len(demo_rig.llm.requests)  # type: ignore[union-attr]

        again = demo_rig.rag.ask("  How are   passwords hashed? ", None, 5, "hybrid", False)

        assert len(demo_rig.llm.requests) == calls  # type: ignore[union-attr]
        assert (again.cached, again.answer, again.cost.calls) == (True, first.answer, 0)
        assert demo_rig.rag.cache.hits == 1

    def test_changing_the_code_invalidates_cached_answers(self, rig: Rig) -> None:
        rig.index("shop", {"cart.py": CART})
        rig.rag.ask("total of the cart", None, 3, "hybrid", False)

        rig.index("shop", {"cart.py": CART})  # unchanged: the answer stays valid
        assert rig.rag.ask("total of the cart", None, 3, "hybrid", False).cached
        rig.index("shop", {"cart.py": CART.replace("book.price", "book.cost")})
        assert not rig.rag.ask("total of the cart", None, 3, "hybrid", False).cached

    def test_other_settings_and_history_are_other_questions(self, demo_rig: Rig) -> None:
        question = "How are passwords hashed?"
        demo_rig.rag.ask(question, None, 5, "hybrid", False)

        assert not demo_rig.rag.ask(question, None, 3, "hybrid", False).cached
        assert not demo_rig.rag.ask(question, None, 5, "bm25", False).cached
        assert not demo_rig.rag.ask(question, ["bookshop-api"], 5, "hybrid", False).cached
        history = [("user", "Hi"), ("assistant", "Hello")]
        with_history = demo_rig.rag.ask(question, None, 5, "hybrid", False, history)
        assert not with_history.cached
        prompt = demo_rig.llm.calls_for(Answer)[-1].prompt  # type: ignore[union-attr]
        assert "<conversation>\nUser: Hi\nAssistant: Hello\n</conversation>" in prompt

    def test_no_matching_code_answers_without_the_model(self, demo_rig: Rig) -> None:
        result = demo_rig.rag.ask("zebra quantum", None, 5, "bm25", False)

        assert (result.answer, result.grounded, result.search.hits) == (NO_MATCH, False, [])
        assert demo_rig.llm.requests == []  # type: ignore[union-attr]

    def test_only_valid_source_numbers_are_kept(self) -> None:
        def answer(_: str, __: object) -> dict[str, object]:
            return {"answer": "See [2].", "sources_used": [0, 2, 2, 99], "grounded": True}

        rig = make_rig(FakeLLMClient(ScriptedClaude(answer=answer))).index_demo()

        result = rig.rag.ask("How are passwords hashed?", None, 5, "hybrid", False)

        assert result.sources_used == [2]

    def test_without_a_model_answers_are_unavailable(self) -> None:
        rig = make_rig(None).index_demo()

        with pytest.raises(GenerationUnavailableError):
            rig.rag.ask("How are passwords hashed?", None, 5, "hybrid", False)
        assert rig.rag.search("How are passwords hashed?", None, 5, "hybrid", False).hits


class TestCodebases:
    def test_every_indexed_codebase_by_default(self, demo_rig: Rig) -> None:
        result = demo_rig.rag.search("cart", None, 10, "hybrid", False)

        assert result.codebases == ["bookshop-api", "bookshop-web"]

    def test_unknown_codebases_are_refused(self, demo_rig: Rig) -> None:
        with pytest.raises(NothingIndexedError, match="Not indexed: shop"):
            demo_rig.rag.search("cart", ["bookshop-api", "shop"], 5, "hybrid", False)

    def test_an_empty_index_is_refused(self, rig: Rig) -> None:
        with pytest.raises(NothingIndexedError, match="No codebase is indexed yet"):
            rig.rag.ask("cart", None, 5, "hybrid", False)
