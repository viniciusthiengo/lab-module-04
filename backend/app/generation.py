"""Answer generation: Claude answers from the retrieved chunks and cites them."""

from collections.abc import Sequence
from dataclasses import dataclass

from app.chunking import Chunk
from app.llm.base import LLMClient, Message, Usage
from app.outputs import Answer
from app.prompting import answer_prompt, system_prompt


@dataclass(frozen=True)
class Generated:
    answer: str
    sources_used: list[int]  # 1-based numbers of the chunks the answer cites
    grounded: bool
    usage: Usage
    model: str


class AnswerGenerator:
    def __init__(self, llm: LLMClient, *, max_tokens: int = 2_048) -> None:
        self._llm = llm
        self._max_tokens = max_tokens

    @property
    def model(self) -> str:
        return self._llm.model

    def generate(
        self, question: str, chunks: Sequence[Chunk], history: Sequence[tuple[str, str]] = ()
    ) -> Generated:
        reply = self._llm.respond(
            system=system_prompt("answer"),
            messages=[Message("user", text=answer_prompt(question, chunks, history))],
            tools=(),
            output_type=Answer,
            max_tokens=self._max_tokens,
        )
        if not isinstance(reply.output, Answer):
            raise ValueError("the model did not return an answer")
        used = sorted({n for n in reply.output.sources_used if 1 <= n <= len(chunks)})
        return Generated(
            reply.output.answer.strip(), used, reply.output.grounded, reply.usage, reply.model
        )
