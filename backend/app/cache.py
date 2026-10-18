"""A least-recently-used cache with a time to live, for query results."""

import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from typing import Generic, TypeVar

ValueT = TypeVar("ValueT")


class TTLCache(Generic[ValueT]):
    def __init__(
        self, max_entries: int, ttl_seconds: float, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self._max = max_entries
        self._ttl = ttl_seconds
        self._clock = clock
        self._entries: OrderedDict[str, tuple[float, ValueT]] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:
        return len(self._entries)

    def get(self, key: str) -> ValueT | None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None or self._clock() - entry[0] >= self._ttl:
                self._entries.pop(key, None)
                self.misses += 1
                return None
            self._entries.move_to_end(key)
            self.hits += 1
            return entry[1]

    def put(self, key: str, value: ValueT) -> None:
        with self._lock:
            self._entries[key] = (self._clock(), value)
            self._entries.move_to_end(key)
            while len(self._entries) > self._max:
                self._entries.popitem(last=False)
