import time
from threading import Lock
from typing import Any, Callable, Hashable


class TTLCache:
    """Simple thread-safe cache that re-computes a value once its TTL expires."""

    def __init__(self, ttl_seconds: float):
        self._ttl = ttl_seconds
        self._store: dict[Hashable, tuple[float, Any]] = {}
        self._lock = Lock()

    def get_or_set(self, key: Hashable, compute: Callable[[], Any]) -> Any:
        now = time.monotonic()
        with self._lock:
            cached = self._store.get(key)
            if cached is not None and cached[0] > now:
                return cached[1]

        value = compute()

        with self._lock:
            self._store[key] = (now + self._ttl, value)
        return value
