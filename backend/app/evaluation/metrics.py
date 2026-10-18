"""Retrieval metrics: Precision@K, Recall@K, reciprocal rank (averaged into MRR), nDCG@K
and hit rate.

The dataset names the code that answers each question with targets such as
``bookshop-api:bookshop/cart.py::Cart.apply_coupon``. A retrieved chunk is relevant when it
is that code, contains it or is part of it: names match when one is a prefix of the other at
a ``.`` boundary, so a chunk of the whole ``Cart`` class finds ``Cart.apply_coupon``.
"""

import math
from collections.abc import Sequence


def matches(key: str, target: str) -> bool:
    """Whether the chunk ``key`` and the ``target`` refer to the same code."""
    key_file, _, key_name = key.partition("::")
    target_file, _, target_name = target.partition("::")
    if key_file != target_file:
        return False
    return (
        key_name == target_name
        or key_name.startswith(target_name + ".")
        or target_name.startswith(key_name + ".")
    )


def _relevant(keys: Sequence[str], targets: Sequence[str]) -> list[bool]:
    return [any(matches(key, target) for target in targets) for key in keys]


def precision_at_k(keys: Sequence[str], targets: Sequence[str], k: int) -> float:
    """The fraction of the top ``k`` results that are relevant."""
    return sum(_relevant(keys[:k], targets)) / k if k > 0 else 0.0


def recall_at_k(keys: Sequence[str], targets: Sequence[str], k: int) -> float:
    """The fraction of the targets found in the top ``k`` results."""
    if not targets:
        return 0.0
    found = sum(1 for target in targets if any(matches(key, target) for key in keys[:k]))
    return found / len(targets)


def reciprocal_rank(keys: Sequence[str], targets: Sequence[str]) -> float:
    """1 / the rank of the first relevant result (0 when there is none)."""
    for rank, relevant in enumerate(_relevant(keys, targets), start=1):
        if relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(keys: Sequence[str], targets: Sequence[str], k: int) -> float:
    """Normalized discounted cumulative gain with binary relevance: relevant results count
    more the higher they rank, relative to the best possible ranking."""
    gains = _relevant(keys[:k], targets)
    dcg = sum(1 / math.log2(rank + 1) for rank, relevant in enumerate(gains, start=1) if relevant)
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(k, len(targets)) + 1))
    return dcg / ideal if ideal else 0.0


def hit_at_k(keys: Sequence[str], targets: Sequence[str], k: int) -> float:
    """1 when at least one of the top ``k`` results is relevant."""
    return 1.0 if any(_relevant(keys[:k], targets)) else 0.0


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0
