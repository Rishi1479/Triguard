"""Backend settings (environment-driven, with safe defaults)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from ml.common import config


def _env_list(name: str, default: list[str]) -> list[str]:
    raw = os.getenv(name)
    return [x.strip() for x in raw.split(",") if x.strip()] if raw else default


@dataclass(frozen=True)
class Settings:
    """Runtime configuration. Override via TRIGUARD_* environment variables."""

    db_url: str = field(default_factory=lambda: os.getenv("TRIGUARD_DB_URL", f"sqlite:///{config.DATA_DIR / 'triguard.db'}"))
    cors_origins: list[str] = field(default_factory=lambda: _env_list(
        "TRIGUARD_CORS", ["http://localhost:5173", "http://127.0.0.1:5173"]))
    max_image_bytes: int = config.MAX_IMAGE_BYTES
    max_text_chars: int = config.MAX_TEXT_CHARS
    max_url_chars: int = config.MAX_URL_CHARS
    max_request_bytes: int = config.MAX_IMAGE_BYTES + 2 * config.MAX_TEXT_CHARS + 16_384
    rate_limit_per_minute: int = int(os.getenv("TRIGUARD_RATE_LIMIT", "30"))
    analysis_timeout_s: float = float(os.getenv("TRIGUARD_TIMEOUT_S", "90"))
    workers: int = int(os.getenv("TRIGUARD_WORKERS", "2"))
    enable_live_checks: bool = os.getenv("TRIGUARD_LIVE_CHECKS", "1") != "0"
    history_size: int = 20
    stored_text_chars: int = 2000
