"""Caches, limits, paths and settings."""

from datetime import date

import pytest

from app.cache import TTLCache
from app.config import Settings
from app.embeddings import CachedEmbedder
from app.limits import BudgetExceededError, DailyBudget, RateLimitedError, RateLimiter
from app.paths import InvalidPathError, normalize_path
from tests.fakes import HashEmbedder


class TestCachedEmbedder:
    def test_each_text_is_embedded_once(self) -> None:
        inner = HashEmbedder()
        cached = CachedEmbedder(inner)

        first = cached.embed(["a cart", "a book", "a cart"])
        again = cached.embed(["a book", "a coupon"])

        assert inner.calls == [["a cart", "a book"], ["a coupon"]]
        assert first[0] == first[2]
        assert again[0] == first[1]
        assert (cached.hits, cached.misses) == (1, 4)
        assert cached.name == "hash-embedder"

    def test_the_least_recently_used_texts_are_dropped(self) -> None:
        inner = HashEmbedder()
        cached = CachedEmbedder(inner, max_entries=2)
        cached.embed(["one", "two"])
        cached.embed(["one"])  # "two" is now the least recently used
        cached.embed(["three"])

        cached.embed(["one", "two"])

        assert inner.calls[-1] == ["two"]

    def test_nothing_to_embed(self) -> None:
        inner = HashEmbedder()

        assert CachedEmbedder(inner).embed([]) == []
        assert inner.calls == []


class TestTTLCache:
    def test_entries_expire(self) -> None:
        now = [0.0]
        cache: TTLCache[str] = TTLCache(10, 60, clock=lambda: now[0])
        cache.put("k", "v")

        now[0] = 59.0
        assert cache.get("k") == "v"
        now[0] = 60.0
        assert cache.get("k") is None
        assert (cache.hits, cache.misses, len(cache)) == (1, 1, 0)

    def test_the_least_recently_used_entry_is_dropped(self) -> None:
        cache: TTLCache[int] = TTLCache(2, 60)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.get("a")
        cache.put("c", 3)

        assert (cache.get("a"), cache.get("b"), cache.get("c")) == (1, None, 3)


class TestLimits:
    def test_rate_limiter_uses_a_sliding_window_per_client(self) -> None:
        now = [0.0]
        limiter = RateLimiter(2, 60, clock=lambda: now[0])
        limiter.check("a")
        limiter.check("a")

        with pytest.raises(RateLimitedError):
            limiter.check("a")
        limiter.check("b")
        now[0] = 60.0
        limiter.check("a")

    def test_daily_budget_resets_every_day(self) -> None:
        today = [date(2026, 10, 17)]
        budget = DailyBudget(1.0, today=lambda: today[0])
        budget.add(1.0)

        with pytest.raises(BudgetExceededError, match="used up"):
            budget.ensure_available()
        today[0] = date(2026, 10, 18)
        budget.ensure_available()
        assert budget.spent_usd == 0


@pytest.mark.parametrize(
    ("raw", "path"),
    [("src/app.py", "src/app.py"), ("./src\\app.py", "src/app.py"), (" a.md ", "a.md")],
)
def test_normalize_path(raw: str, path: str) -> None:
    assert normalize_path(raw) == path


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ("", "empty"),
        ("/etc/passwd.py", "relative"),
        ("C:/x.py", "relative"),
        ("a/../b.py", "'..'"),
        ("a//b.py", "empty"),
        ("a/<b>.py", "not allowed"),
        ("a" * 301, "too long"),
    ],
)
def test_invalid_paths(raw: str, message: str) -> None:
    with pytest.raises(InvalidPathError, match=message):
        normalize_path(raw)


class TestSettings:
    def test_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", " sk-test ")
        monkeypatch.setenv("JUDGE_MODEL", "claude-opus-5")
        monkeypatch.setenv("DAILY_BUDGET_USD", "2.5")
        monkeypatch.setenv("INDEX_DEMO_ON_START", "false")
        monkeypatch.setenv("CORS_ORIGINS", "https://a.example, https://b.example")
        monkeypatch.delenv("DATA_DIR", raising=False)

        settings = Settings.from_env()

        assert settings.anthropic_api_key == "sk-test"
        assert (settings.llm_model, settings.judge_model) == ("claude-haiku-4-5", "claude-opus-5")
        assert settings.daily_budget_usd == 2.5
        assert settings.index_demo_on_start is False
        assert settings.cors_origins == ("https://a.example", "https://b.example")
        assert str(settings.data_dir) == "data"

    def test_invalid_numbers_are_reported(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("QUERIES_PER_HOUR", "many")

        with pytest.raises(ValueError, match="QUERIES_PER_HOUR must be a number, not 'many'"):
            Settings.from_env()

    def test_the_key_is_never_printed(self) -> None:
        text = repr(Settings(anthropic_api_key="sk-secret"))

        assert "sk-secret" not in text
        assert "api_key=set" in text
