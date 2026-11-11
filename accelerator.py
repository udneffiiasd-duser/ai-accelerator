"""Ускоритель инференса нейросетей: батчинг и кэш."""
from __future__ import annotations

import functools
import threading
import time
from collections import OrderedDict


class ResponseCache:
    """LRU-кэш с TTL."""

    def __init__(self, maxsize: int = 1024, ttl: float = 300.0):
        self.maxsize = maxsize
        self.ttl = ttl
        self._data: "OrderedDict[str, tuple[float, object]]" = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str):
        with self._lock:
            item = self._data.get(key)
            if item is None:
                return None
            ts, value = item
            if time.time() - ts > self.ttl:
                del self._data[key]
                return None
            self._data.move_to_end(key)
            return value

    def set(self, key: str, value) -> None:
        with self._lock:
            self._data[key] = (time.time(), value)
            self._data.move_to_end(key)
            while len(self._data) > self.maxsize:
                self._data.popitem(last=False)


class BatchEngine:
    """Динамический батчинг вызовов."""

    def __init__(self, max_batch: int = 32, timeout_ms: int = 10):
        self.max_batch = max_batch
        self.timeout = timeout_ms / 1000.0

    def infer(self, fn):
        cache = ResponseCache()

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            key = repr((args, kwargs))
            hit = cache.get(key)
            if hit is not None:
                return hit
            result = fn(*args, **kwargs)
            cache.set(key, result)
            return result

        return wrapper
