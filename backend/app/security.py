"""Request guards: image sniffing, in-memory rate limiter, PII-safe helpers."""
from __future__ import annotations

import hashlib
import threading
import time
from collections import defaultdict, deque

from ml.common.privacy import anonymize

RATE_WINDOW_S = 60.0


def sniff_image_type(data: bytes) -> str | None:
    """'png' | 'jpeg' | 'webp' from magic bytes (never trust the client's Content-Type), else None."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if data.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    return None


def image_hash(data: bytes) -> str:
    """sha256 hex digest (the only trace of an uploaded image we keep)."""
    return hashlib.sha256(data).hexdigest()


def mask(text: str | None, limit: int | None = None) -> str | None:
    """Anonymise (phones/e-mails/IDs) and optionally truncate text before it is stored or listed."""
    if text is None:
        return None
    out = anonymize(text)
    return out[:limit] if limit else out


class RateLimiter:
    """Sliding-window limiter keyed by client id. Thread-safe, in-memory (single process)."""

    def __init__(self, limit: int, window_s: float = RATE_WINDOW_S) -> None:
        self.limit, self.window = limit, window_s
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        """Record a hit; False if ``key`` exceeded the limit in the window."""
        now = time.monotonic()
        with self._lock:
            q = self._hits[key]
            while q and now - q[0] > self.window:
                q.popleft()
            if len(q) >= self.limit:
                return False
            q.append(now)
            return True
