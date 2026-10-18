import pytest

from app.chunking import Chunk
from app.llm.base import Usage
from app.llm.fake import FakeLLMClient, FakeRequest, ScriptedTurn
from app.outputs import Ranking
from app.retrieval import Hit, Reranker, fuse
from tests.conftest import Rig, make_rig
from tests.fakes import ScriptedClaude, excerpts

DEMO = ["bookshop-api", "bookshop-web"]


def chunk(name: str) -> Chunk:
    return Chunk("shop", "cart.py", "python", "function", name, 1, 2, f"def {name}(): ...")


def names(hits: list[Hit]) -> list[str]:
    return [hit.chunk.name for hit in hits]


class TestFuse:
    def test_reciprocal_rank_fusion_adds_one_over_60_plus_rank_per_list(self) -> None:
        a, b, c = chunk("a"), chunk("b"), chunk("c")
        vector = [Hit(a, 0.9, vector_rank=1), Hit(b, 0.8, vector_rank=2)]
        keyword = [Hit(b, 7.0, bm25_rank=1), Hit(c, 3.0, bm25_rank=2)]

        fused = fuse(vector, keyword)

        assert names(fused) == ["b", "a", "c"]
        assert fused[0].score == pytest.approx(1 / 62 + 1 / 61)
        assert (fused[0].vector_rank, fused[0].bm25_rank) == (2, 1)
        assert (fused[1].score, fused[1].bm25_rank) == (pytest.approx(1 / 61), None)
        assert (fused[2].score, fused[2].vector_rank) == (pytest.approx(1 / 62), None)

    def test_fusing_empty_lists(self) -> None:
        assert fuse([], []) == []


class TestReranker:
    def test_orders_by_grade_and_keeps_the_order_of_ties(self) -> None:
        grades = {"a": 1, "b": 3, "c": 1, "d": 0}
        llm = FakeLLMClient(ScriptedClaude(grade=lambda _, e: grades[e.name]))
        hits = [Hit(chunk(name), 0.5) for name in "abcd"]

        reranked, usage, model = Reranker(llm).rerank("question?", hits)

        assert names(reranked) == ["b", "a", "c", "d"]
        assert [hit.relevance for hit in reranked] == [3, 1, 1, 0]
        assert (usage, model) == (Usage(1000, 400), "fake-model")
        assert llm.requests[0].output_type is Ranking

    def test_ignores_unknown_excerpts_and_clamps_grades(self) -> None:
        def ranking(_: FakeRequest) -> ScriptedTurn:
            return ScriptedTurn(
                output={
                    "scores": [
                        {"index": 2, "relevance": 9},
                        {"index": 7, "relevance": 3},
                        {"index": 1, "relevance": -2},
                    ]
                }
            )

        hits = [Hit(chunk(name), 0.5) for name in "ab"]

        reranked, _, _ = Reranker(FakeLLMClient(ranking)).rerank("q", hits)

        assert [(hit.chunk.name, hit.relevance) for hit in reranked] == [("b", 3), ("a", 0)]

    def test_long_chunks_are_cut_in_the_prompt(self) -> None:
        llm = FakeLLMClient(ScriptedClaude())
        long = Chunk("shop", "big.py", "python", "function", "big", 1, 300, "x" * 5_000)

        Reranker(llm, max_chars=100).rerank("q", [Hit(long, 1.0)])

        assert "x" * 100 + "\n…" in llm.requests[0].prompt
        assert "x" * 101 not in llm.requests[0].prompt

    def test_no_candidates_no_call(self) -> None:
        llm = FakeLLMClient(ScriptedClaude())

        assert Reranker(llm).rerank("q", [])[0] == []
        assert llm.requests == []


class TestRetriever:
    def test_bm25_finds_the_function_named_in_the_question(self, demo_rig: Rig) -> None:
        hits = demo_rig.rag.search("How are passwords hashed?", DEMO, 3, "bm25", False).hits

        assert hits[0].chunk.key == "bookshop-api:bookshop/auth.py::hash_password"
        assert all(hit.bm25_rank and hit.vector_rank is None for hit in hits)

    def test_vector_search_ranks_by_similarity(self, demo_rig: Rig) -> None:
        hits = demo_rig.rag.search("hash password", DEMO, 5, "vector", False).hits

        assert len(hits) == 5
        assert [hit.vector_rank for hit in hits] == [1, 2, 3, 4, 5]
        assert [hit.score for hit in hits] == sorted((hit.score for hit in hits), reverse=True)

    def test_hybrid_search_fuses_both_rankings(self, demo_rig: Rig) -> None:
        result = demo_rig.rag.search("How are passwords hashed?", DEMO, 5, "hybrid", False)

        assert result.hits[0].chunk.name == "hash_password"
        assert result.hits[0].vector_rank and result.hits[0].bm25_rank
        assert result.candidates > 5
        assert result.cost.calls == 0

    def test_reranking_reorders_the_candidates(self) -> None:
        llm = FakeLLMClient(
            ScriptedClaude(grade=lambda _, e: 3 if e.name == "verify_password" else 1)
        )
        rig = make_rig(llm).index_demo()

        result = rig.rag.search("How are passwords hashed?", DEMO, 3, "hybrid", True)

        assert result.hits[0].chunk.name == "verify_password"
        assert result.hits[0].relevance == 3
        assert result.candidates == 15
        assert (result.cost.calls, result.cost.models) == (1, ("fake-model",))
        assert len(excerpts(llm.requests[0].prompt)) == 15

    def test_one_codebase_or_several(self, demo_rig: Rig) -> None:
        api = demo_rig.rag.search("access token", ["bookshop-api"], 10, "hybrid", False)
        both = demo_rig.rag.search("access token", DEMO, 10, "hybrid", False)

        assert {hit.chunk.codebase for hit in api.hits} == {"bookshop-api"}
        assert {hit.chunk.codebase for hit in both.hits} == set(DEMO)

    def test_the_keyword_index_is_rebuilt_only_when_a_codebase_changes(self, rig: Rig) -> None:
        rig.index("shop", {"cart.py": "def total(cart):\n    return 1\n"})
        calls = 0
        chunks = rig.store.chunks

        def counting(codebases: list[str]) -> list[Chunk]:
            nonlocal calls
            calls += 1
            return chunks(codebases)

        rig.store.chunks = counting  # type: ignore[method-assign]

        rig.rag.search("total", ["shop"], 3, "bm25", False)
        rig.rag.search("cart", ["shop"], 3, "bm25", False)
        assert calls == 1
        rig.index("shop", {"cart.py": "def total(cart):\n    return 2\n"})
        hits = rig.rag.search("total", ["shop"], 3, "bm25", False).hits
        assert calls == 2
        assert "return 2" in hits[0].chunk.content

    def test_reranking_needs_a_model(self) -> None:
        rig = make_rig(None).index_demo()

        with pytest.raises(ValueError, match="needs a model"):
            rig.rag.search("q", DEMO, 3, "hybrid", True)
