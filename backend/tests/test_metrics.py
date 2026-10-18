import math

import pytest

from app.evaluation import metrics

TARGET = "shop:cart.py::Cart.apply_coupon"


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("shop:cart.py::Cart.apply_coupon", True),
        ("shop:cart.py::Cart", True),  # the whole class contains the method
        ("shop:cart.py::Cart.apply_coupon.inner", True),  # part of it
        ("shop:cart.py::Cart.apply", False),
        ("shop:cart.py::Cartography", False),
        ("shop:other.py::Cart.apply_coupon", False),
        ("web:cart.py::Cart.apply_coupon", False),
    ],
)
def test_matches(key: str, expected: bool) -> None:
    assert metrics.matches(key, TARGET) is expected


class TestRankingMetrics:
    RETRIEVED = ("shop:a.py::x", "shop:b.py::target_one", "shop:c.py::y", "shop:d.py::target_two")
    TARGETS = ("shop:b.py::target_one", "shop:d.py::target_two", "shop:e.py::missed")

    def test_precision_is_the_share_of_relevant_results_in_the_top_k(self) -> None:
        assert metrics.precision_at_k(self.RETRIEVED, self.TARGETS, 2) == 0.5
        assert metrics.precision_at_k(self.RETRIEVED, self.TARGETS, 4) == 0.5

    def test_precision_counts_missing_results_as_irrelevant(self) -> None:
        assert metrics.precision_at_k(self.RETRIEVED[:2], self.TARGETS, 5) == pytest.approx(0.2)

    def test_recall_is_the_share_of_targets_found(self) -> None:
        assert metrics.recall_at_k(self.RETRIEVED, self.TARGETS, 2) == pytest.approx(1 / 3)
        assert metrics.recall_at_k(self.RETRIEVED, self.TARGETS, 4) == pytest.approx(2 / 3)

    def test_recall_counts_a_target_once_even_when_two_chunks_match_it(self) -> None:
        retrieved = ["shop:b.py::target_one", "shop:b.py::target_one"]

        assert metrics.recall_at_k(retrieved, ["shop:b.py::target_one"], 2) == 1.0

    def test_reciprocal_rank_is_one_over_the_first_relevant_rank(self) -> None:
        assert metrics.reciprocal_rank(self.RETRIEVED, self.TARGETS) == 0.5
        assert metrics.reciprocal_rank(["shop:a.py::x"], self.TARGETS) == 0.0

    def test_ndcg_rewards_relevant_results_ranked_higher(self) -> None:
        ndcg = metrics.ndcg_at_k(self.RETRIEVED, self.TARGETS, 4)

        dcg = 1 / math.log2(3) + 1 / math.log2(5)
        ideal = 1 + 1 / math.log2(3) + 1 / math.log2(4)
        assert ndcg == pytest.approx(dcg / ideal)
        best = metrics.ndcg_at_k(self.RETRIEVED[1::2], self.TARGETS[:2], 2)
        assert best == pytest.approx(1.0)

    def test_hit_is_one_when_any_relevant_result_is_in_the_top_k(self) -> None:
        assert metrics.hit_at_k(self.RETRIEVED, self.TARGETS, 1) == 0.0
        assert metrics.hit_at_k(self.RETRIEVED, self.TARGETS, 2) == 1.0

    def test_no_targets_score_zero(self) -> None:
        assert metrics.recall_at_k(self.RETRIEVED, [], 3) == 0.0
        assert metrics.ndcg_at_k(self.RETRIEVED, [], 3) == 0.0

    def test_mean(self) -> None:
        assert metrics.mean([1.0, 0.0, 0.5]) == 0.5
        assert metrics.mean([]) == 0.0
