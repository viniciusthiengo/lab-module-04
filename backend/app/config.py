"""Settings read from environment variables."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Self

from app.llm.anthropic_client import DEFAULT_MODEL

DEFAULT_JUDGE_MODEL = "claude-sonnet-5"


@dataclass(frozen=True)
class Settings:
    """Runtime configuration.

    Attributes:
        anthropic_api_key: key of the Anthropic API (never logged). Without it, indexing,
            search and retrieval metrics work; answers, reranking and judging answer 503.
        llm_model: Claude model of the answers and the reranking.
        judge_model: Claude model of the LLM-as-judge: a stronger model grades the answers.
        data_dir: where ChromaDB and the codebase registry live (``None``: in memory).
        model_dir: where the embedding model is cached (``None``: ChromaDB's default).
        index_demo_on_start: index the two demo codebases at startup when they are missing.
        daily_budget_usd: spending cap per UTC day, for every model call.
        queries_per_hour, index_per_hour, evaluations_per_hour: limits per client and hour.
        query_cache_seconds, query_cache_size: reuse of answers to identical questions.
        max_codebases, max_chunks: size of the index.
        cors_origins: origins allowed to call the API directly from a browser.
    """

    anthropic_api_key: str | None = None
    llm_model: str = DEFAULT_MODEL
    judge_model: str = DEFAULT_JUDGE_MODEL
    data_dir: Path | None = None
    model_dir: Path | None = None
    index_demo_on_start: bool = False
    daily_budget_usd: float = 1.0
    queries_per_hour: int = 30
    index_per_hour: int = 30
    evaluations_per_hour: int = 3
    query_cache_seconds: int = 3600
    query_cache_size: int = 500
    max_codebases: int = 20
    max_chunks: int = 20_000
    cors_origins: tuple[str, ...] = ()

    @classmethod
    def from_env(cls) -> Self:
        def number(name: str, default: float, kind: type = int) -> float:
            raw = os.environ.get(name, "").strip()
            try:
                return kind(raw) if raw else default
            except ValueError:
                raise ValueError(f"{name} must be a number, not '{raw}'") from None

        def path(name: str, default: str | None) -> Path | None:
            raw = os.environ.get(name, "").strip() or default
            return Path(raw) if raw else None

        defaults = cls()
        return cls(
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY", "").strip() or None,
            llm_model=os.environ.get("LLM_MODEL", "").strip() or DEFAULT_MODEL,
            judge_model=os.environ.get("JUDGE_MODEL", "").strip() or DEFAULT_JUDGE_MODEL,
            data_dir=path("DATA_DIR", "data"),
            model_dir=path("MODEL_DIR", None),
            index_demo_on_start=os.environ.get("INDEX_DEMO_ON_START", "true").lower() != "false",
            daily_budget_usd=number("DAILY_BUDGET_USD", defaults.daily_budget_usd, float),
            queries_per_hour=int(number("QUERIES_PER_HOUR", defaults.queries_per_hour)),
            index_per_hour=int(number("INDEX_PER_HOUR", defaults.index_per_hour)),
            evaluations_per_hour=int(number("EVALUATIONS_PER_HOUR", defaults.evaluations_per_hour)),
            query_cache_seconds=int(number("QUERY_CACHE_SECONDS", defaults.query_cache_seconds)),
            query_cache_size=int(number("QUERY_CACHE_SIZE", defaults.query_cache_size)),
            max_codebases=int(number("MAX_CODEBASES", defaults.max_codebases)),
            max_chunks=int(number("MAX_CHUNKS", defaults.max_chunks)),
            cors_origins=tuple(
                origin.strip()
                for origin in os.environ.get("CORS_ORIGINS", "").split(",")
                if origin.strip()
            ),
        )

    def __repr__(self) -> str:
        return (
            f"Settings(llm_model={self.llm_model!r}, judge_model={self.judge_model!r}, "
            f"api_key={'set' if self.anthropic_api_key else 'missing'}, "
            f"data_dir={str(self.data_dir)!r}, daily_budget_usd={self.daily_budget_usd})"
        )
