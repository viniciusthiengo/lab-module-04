"""LLM-as-judge: a model grades each generated answer for faithfulness to the retrieved
code, relevance to the question and correctness against the reference answer."""

import re
from collections.abc import Sequence
from dataclasses import dataclass

from app.chunking import Chunk
from app.llm.base import LLMClient, Message, Usage
from app.outputs import Verdict
from app.prompting import judge_prompt, system_prompt

PASSING_GRADE = 4
# The judge's model sometimes ends a structured-output string with zero-width spaces or a
# couple of stray characters after the last sentence ("...the reference.ăn").
_INVISIBLE = re.compile("[\u200b-\u200d\u2060\ufeff]")
_STRAY_TAIL = re.compile(r"(?<=[.!?])(?:['\"]|[^\W\d_]{1,2})$")


@dataclass(frozen=True)
class Judgement:
    faithfulness: int
    relevance: int
    correctness: int
    rationale: str
    usage: Usage
    model: str

    @property
    def passed(self) -> bool:
        """Every criterion graded 4 or 5."""
        return min(self.faithfulness, self.relevance, self.correctness) >= PASSING_GRADE


class Judge:
    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    @property
    def model(self) -> str:
        return self._llm.model

    def grade(
        self, question: str, chunks: Sequence[Chunk], answer: str, reference: str
    ) -> Judgement:
        reply = self._llm.respond(
            system=system_prompt("judge"),
            messages=[Message("user", text=judge_prompt(question, chunks, answer, reference))],
            tools=(),
            output_type=Verdict,
            max_tokens=1_024,
        )
        if not isinstance(reply.output, Verdict):
            raise ValueError("the judge did not return a verdict")
        verdict = reply.output
        return Judgement(
            faithfulness=_grade(verdict.faithfulness),
            relevance=_grade(verdict.relevance),
            correctness=_grade(verdict.correctness),
            rationale=_clean(verdict.rationale),
            usage=reply.usage,
            model=reply.model,
        )


def _grade(value: int) -> int:
    return max(1, min(5, value))


def _clean(text: str) -> str:
    return _STRAY_TAIL.sub("", _INVISIBLE.sub("", text).strip())
