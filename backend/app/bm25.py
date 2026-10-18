"""BM25 (Okapi) keyword search over chunks, with a tokenizer for code.

Identifiers are split as well as kept whole, so ``refreshAccessToken`` matches the query
words "refresh", "access" and "token"; the Snowball (Porter2) English stemmer folds word
forms ("passwords" and "hashed" match ``hash_password``, "environment" matches
``os.environ``); common English words are ignored.
"""

import math
import re
import threading
from collections import Counter
from collections.abc import Sequence
from functools import lru_cache

import snowballstemmer

_WORD = re.compile(r"[A-Za-z0-9_]+")
_PART = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")
# Common English words, which carry no meaning for a search.
_STOP_WORDS_TEXT = """
a an and are as at be but by can do does for from has have how if in into is it its of on
or that the their then there these this to was what when where which who why will with
"""
_STOP_WORDS = frozenset(_STOP_WORDS_TEXT.split())
_stemmers = threading.local()  # a stemmer keeps state while it works: one per thread


@lru_cache(maxsize=100_000)
def stem(token: str) -> str:
    """The Snowball English stem: "passwords" → "password", "expiration" → "expir"."""
    stemmer = getattr(_stemmers, "english", None)
    if stemmer is None:
        stemmer = _stemmers.english = snowballstemmer.stemmer("english")
    return str(stemmer.stemWord(token))


def tokenize(text: str) -> list[str]:
    """Lower-case, stemmed tokens: whole identifiers plus their camelCase and snake_case
    parts, without stop words and single characters."""
    tokens: list[str] = []
    for word in _WORD.findall(text):
        parts = [part for piece in word.split("_") for part in _PART.findall(piece)]
        candidates = [word.lower()] + ([part.lower() for part in parts] if len(parts) > 1 else [])
        tokens += [
            stem(token) for token in candidates if len(token) > 1 and token not in _STOP_WORDS
        ]
    return tokens


class BM25:
    """Okapi BM25 over tokenized documents (``k1`` term saturation, ``b`` length
    normalization)."""

    def __init__(self, documents: Sequence[Sequence[str]], k1: float = 1.5, b: float = 0.75):
        self._k1, self._b = k1, b
        self._frequencies = [Counter(document) for document in documents]
        self._lengths = [len(document) for document in documents]
        self._average_length = sum(self._lengths) / len(documents) if documents else 0.0
        counts = Counter(term for document in documents for term in set(document))
        total = len(documents)
        # The "+ 1" keeps every idf positive, even for a term in most documents.
        self._idf = {
            term: math.log((total - count + 0.5) / (count + 0.5) + 1)
            for term, count in counts.items()
        }

    def __len__(self) -> int:
        return len(self._lengths)

    def scores(self, query: Sequence[str]) -> list[float]:
        scores = [0.0] * len(self._lengths)
        for term in set(query):
            idf = self._idf.get(term)
            if idf is None:
                continue
            for index, frequencies in enumerate(self._frequencies):
                frequency = frequencies.get(term, 0)
                if frequency:
                    norm = 1 - self._b + self._b * self._lengths[index] / self._average_length
                    scores[index] += (
                        idf * frequency * (self._k1 + 1) / (frequency + self._k1 * norm)
                    )
        return scores

    def top(self, query: Sequence[str], k: int) -> list[tuple[int, float]]:
        """The ``k`` best documents with a positive score: ``(index, score)``, best first."""
        ranked = sorted(enumerate(self.scores(query)), key=lambda item: -item[1])
        return [(index, score) for index, score in ranked[:k] if score > 0]
