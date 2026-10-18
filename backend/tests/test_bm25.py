import pytest

from app.bm25 import BM25, stem, tokenize


class TestTokenize:
    @pytest.mark.parametrize(
        ("text", "tokens"),
        [
            ("refreshAccessToken", ["refreshaccesstoken", "refresh", "access", "token"]),
            ("hash_password", ["hash_password", "hash", "password"]),
            ("HTTPServer", ["httpserver", "http", "server"]),
            ("How are passwords hashed?", ["password", "hash"]),
            ("a b x1 _", ["x1"]),
        ],
    )
    def test_splits_identifiers_and_drops_stop_words(self, text: str, tokens: list[str]) -> None:
        assert tokenize(text) == tokens

    @pytest.mark.parametrize(
        ("word", "stemmed"),
        [
            ("passwords", "password"),
            ("hashed", "hash"),
            ("classes", "class"),
            ("access", "access"),
            ("environment", "environ"),
            ("expiration", "expir"),
            ("expires", "expir"),
        ],
    )
    def test_stem_folds_word_forms(self, word: str, stemmed: str) -> None:
        assert stem(word) == stemmed

    def test_queries_and_code_meet_on_the_same_tokens(self) -> None:
        assert set(tokenize("Which tokens are refreshed?")) <= set(
            tokenize("async function refreshAccessToken(tokens)")
        )


class TestBM25:
    DOCUMENTS = (
        ["cart", "total"],
        ["payment", "retry", "retry", "gateway"],
        ["cart", "coupon", "discount", "total", "price"],
    )

    def test_only_documents_with_a_query_term_score(self) -> None:
        index = BM25(self.DOCUMENTS)

        assert [i for i, _ in index.top(["retry"], 5)] == [1]
        assert index.top(["unknown"], 5) == []

    def test_shorter_documents_score_higher_for_the_same_term(self) -> None:
        index = BM25(self.DOCUMENTS)

        assert [i for i, _ in index.top(["cart"], 5)] == [0, 2]

    def test_rare_terms_weigh_more_than_common_ones(self) -> None:
        index = BM25(self.DOCUMENTS)

        scores = index.scores(["total", "coupon"])

        assert scores[2] > scores[0] > 0
        assert scores[1] == 0

    def test_top_keeps_the_k_best(self) -> None:
        index = BM25(self.DOCUMENTS)

        assert len(index.top(["cart", "retry", "total"], 2)) == 2

    def test_an_empty_corpus_finds_nothing(self) -> None:
        index = BM25([])

        assert len(index) == 0
        assert index.top(["cart"], 3) == []
