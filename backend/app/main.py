"""TriGuard FastAPI application entrypoint."""
from __future__ import annotations

from fastapi import FastAPI

app = FastAPI(title="TriGuard API", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}
