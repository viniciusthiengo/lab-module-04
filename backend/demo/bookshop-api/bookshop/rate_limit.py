"""Token-bucket rate limiting per client."""

import time
from dataclasses import dataclass, field


@dataclass
class TokenBucket:
    """Allows bursts of up to ``capacity`` requests, refilled at ``rate`` tokens per second."""

    capacity: int = 20
    rate: float = 5.0
    tokens: float = field(init=False)
    updated: float = field(init=False)

    def __post_init__(self) -> None:
        self.tokens = float(self.capacity)
        self.updated = time.monotonic()

    def allow(self) -> bool:
        now = time.monotonic()
        self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.rate)
        self.updated = now
        if self.tokens >= 1:
            self.tokens -= 1
            return True
        return False

    def retry_after(self) -> float:
        """Seconds until the next token."""
        return max(0.0, (1 - self.tokens) / self.rate)


class RateLimiter:
    """One bucket per client key (the client's IP address)."""

    def __init__(self, capacity: int = 20, rate: float = 5.0):
        self._buckets: dict[str, TokenBucket] = {}
        self._capacity, self._rate = capacity, rate

    def check(self, key: str) -> tuple[bool, float]:
        bucket = self._buckets.setdefault(key, TokenBucket(self._capacity, self._rate))
        return bucket.allow(), bucket.retry_after()
