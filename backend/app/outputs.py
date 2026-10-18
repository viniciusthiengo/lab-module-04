"""What the model answers: structured outputs constrained to these schemas.

The Messages API makes the model's answer follow the JSON schema of these models, and
Pydantic validates it again. The field descriptions are part of the schema the model sees.
"""

from pydantic import BaseModel, Field


class Answer(BaseModel):
    """The answer to a question about the code."""

    answer: str = Field(
        description="Markdown. Cite the excerpts you rely on as [1], [2]... Use code blocks."
    )
    sources_used: list[int] = Field(description="Numbers of the excerpts the answer relies on.")
    grounded: bool = Field(
        description="True when the excerpts contain what the answer says; false when the "
        "answer says what is missing."
    )


class Relevance(BaseModel):
    index: int = Field(description="Number of the excerpt.")
    relevance: int = Field(
        description="3: answers the question; 2: directly related; 1: loosely related; "
        "0: unrelated."
    )


class Ranking(BaseModel):
    """Relevance of each candidate excerpt to the question."""

    scores: list[Relevance]


class Verdict(BaseModel):
    """The judge's grades of one answer, from 1 (worst) to 5 (best)."""

    faithfulness: int = Field(
        description="1-5: are the answer's claims supported by the retrieved excerpts?"
    )
    relevance: int = Field(description="1-5: does the answer address the question directly?")
    correctness: int = Field(description="1-5: does the answer agree with the reference answer?")
    rationale: str = Field(description="Why, in one or two sentences.")
