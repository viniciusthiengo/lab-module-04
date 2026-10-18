"""Evaluation datasets: questions, reference answers and the code that answers them."""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EvalExample:
    id: str
    question: str
    expected_answer: str
    relevant: tuple[str, ...]  # targets: codebase:path::name


@dataclass(frozen=True)
class EvalDataset:
    name: str
    description: str
    codebases: tuple[str, ...]
    examples: tuple[EvalExample, ...]


def load_dataset(path: Path) -> EvalDataset:
    data = json.loads(path.read_text(encoding="utf-8"))
    return EvalDataset(
        name=data["name"],
        description=data.get("description", ""),
        codebases=tuple(data["codebases"]),
        examples=tuple(
            EvalExample(
                id=example["id"],
                question=example["question"],
                expected_answer=example["expected_answer"],
                relevant=tuple(example["relevant"]),
            )
            for example in data["examples"]
        ),
    )
