"""System prompts (``prompts/*.md``) and the user prompts built from retrieved chunks."""

from collections.abc import Sequence
from functools import cache
from pathlib import Path

from app.chunking import Chunk

PROMPTS = Path(__file__).parent / "prompts"
PROMPT_VERSION = "1"  # part of the query cache key: change it when a prompt changes


@cache
def system_prompt(name: str) -> str:
    """The system prompt ``answer``, ``rerank`` or ``judge``."""
    return (PROMPTS / f"{name}.md").read_text(encoding="utf-8").strip()


def excerpt(number: int, chunk: Chunk, max_chars: int | None = None) -> str:
    """A chunk as a numbered ``<source>`` block; long code is cut at ``max_chars``."""
    code = chunk.content
    if max_chars is not None and len(code) > max_chars:
        code = code[:max_chars] + "\n…"
    return (
        f'<source id="{number}" codebase="{chunk.codebase}" path="{chunk.path}" '
        f'lines="{chunk.start_line}-{chunk.end_line}" kind="{chunk.kind}" name="{chunk.name}">\n'
        f"{code}\n</source>"
    )


def excerpts(chunks: Sequence[Chunk], max_chars: int | None = None) -> str:
    return "\n\n".join(excerpt(number, chunk, max_chars) for number, chunk in enumerate(chunks, 1))


def answer_prompt(
    question: str, chunks: Sequence[Chunk], history: Sequence[tuple[str, str]]
) -> str:
    parts = [f"<context>\n{excerpts(chunks)}\n</context>"]
    if history:
        turns = "\n".join(f"{role.capitalize()}: {text}" for role, text in history)
        parts.append(f"<conversation>\n{turns}\n</conversation>")
    parts.append(f"Question: {question}")
    return "\n\n".join(parts)


def rerank_prompt(question: str, chunks: Sequence[Chunk], max_chars: int) -> str:
    return (
        f"Question: {question}\n\n<candidates>\n{excerpts(chunks, max_chars)}\n</candidates>\n\n"
        f"Give the relevance of each of the {len(chunks)} excerpts."
    )


def judge_prompt(question: str, chunks: Sequence[Chunk], answer: str, reference: str) -> str:
    return (
        f"<question>{question}</question>\n\n"
        f"<retrieved_excerpts>\n{excerpts(chunks, 2_000)}\n</retrieved_excerpts>\n\n"
        f"<answer>\n{answer}\n</answer>\n\n"
        f"<reference_answer>\n{reference}\n</reference_answer>"
    )
