"""Test doubles: an embedder that needs no model, and a scripted Claude.

``HashEmbedder`` hashes the words of a text into a vector, so texts that share words are
close: retrieval behaves sensibly without the 79 MB model. ``ScriptedClaude`` answers each
kind of request (answer, reranking, judging) the way the model would.
"""

import hashlib
import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from app.bm25 import tokenize
from app.llm.fake import FakeRequest, ScriptedTurn
from app.outputs import Answer, Ranking, Verdict

_SOURCE = re.compile(r'<source id="(\d+)" codebase="[^"]*" path="([^"]*)" [^>]*name="([^"]*)">')


class HashEmbedder:
    """A bag of words hashed into ``dimensions`` and normalized (cosine-friendly)."""

    name = "hash-embedder"

    def __init__(self, dimensions: int = 256) -> None:
        self.dimensions = dimensions
        self.calls: list[list[str]] = []

    @property
    def embedded(self) -> int:
        """How many texts were embedded so far."""
        return sum(len(call) for call in self.calls)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [self._vector(text) for text in texts]

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        vector[0] = 0.01  # never the zero vector
        for token in tokenize(text):
            vector[int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dimensions] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector]


@dataclass(frozen=True)
class Excerpt:
    """A ``<source>`` block of a prompt."""

    number: int
    path: str
    name: str


def excerpts(prompt: str) -> list[Excerpt]:
    return [Excerpt(int(n), path, name) for n, path, name in _SOURCE.findall(prompt)]


def answer_of(prompt: str) -> str:
    """The answer a judge prompt asks to grade."""
    match = re.search(r"<answer>\n(.*?)\n</answer>", prompt, re.DOTALL)
    return match.group(1) if match else ""


def question_of(prompt: str) -> str:
    match = re.search(r"Question: (.+)", prompt) or re.search(r"<question>(.+?)</question>", prompt)
    return match.group(1) if match else ""


def by_shared_words(question: str, excerpt: Excerpt) -> int:
    """Relevance 3 when the excerpt's name shares a word with the question, else 0."""
    words = set(tokenize(question))
    return 3 if words & set(tokenize(f"{excerpt.name}")) else 0


class ScriptedClaude:
    """Responder of ``FakeLLMClient``: answers, rerankings and verdicts.

    - answers cite the first excerpt, or what ``answer`` returns;
    - the reranker grades each excerpt with ``grade`` (default: shared words with the name);
    - the judge answers ``verdict``, or what it returns for the question, the answer and the
      excerpts.
    """

    def __init__(
        self,
        *,
        answer: Callable[[str, list[Excerpt]], dict[str, Any]] | None = None,
        grade: Callable[[str, Excerpt], int] = by_shared_words,
        verdict: dict[str, Any] | Callable[[str, str, list[Excerpt]], dict[str, Any]] | None = None,
    ) -> None:
        self._answer = answer
        self._grade = grade
        self._verdict = verdict or {
            "faithfulness": 5,
            "relevance": 5,
            "correctness": 4,
            "rationale": "Supported by the excerpts and agrees with the reference.",
        }

    def __call__(self, request: FakeRequest) -> ScriptedTurn:
        prompt = request.prompt
        found = excerpts(prompt)
        question = question_of(prompt)
        if request.output_type is Answer:
            if self._answer is not None:
                return ScriptedTurn(output=self._answer(question, found))
            first = found[0]
            return ScriptedTurn(
                output={
                    "answer": f"It is done in `{first.name}` ({first.path}) [1].",
                    "sources_used": [1],
                    "grounded": True,
                }
            )
        if request.output_type is Ranking:
            scores = [
                {"index": excerpt.number, "relevance": self._grade(question, excerpt)}
                for excerpt in found
            ]
            return ScriptedTurn(output={"scores": scores})
        if request.output_type is Verdict:
            if callable(self._verdict):
                return ScriptedTurn(output=self._verdict(question, answer_of(prompt), found))
            return ScriptedTurn(output=self._verdict)
        raise AssertionError(f"unexpected output type {request.output_type}")
