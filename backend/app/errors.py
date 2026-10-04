"""Consistent error responses: {"error": {"code", "message", "request_id"}}."""
from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

log = logging.getLogger(__name__)


class ApiError(HTTPException):
    """HTTP error with a machine-readable ``code``."""

    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(status_code=status, detail=message)
        self.code = code


def error_body(request: Request, code: str, message: str) -> dict:
    """Standard error payload."""
    return {"error": {"code": code, "message": message, "request_id": getattr(request.state, "request_id", None)}}


def install_handlers(app: FastAPI) -> None:
    """Register handlers so every failure uses the same JSON shape."""

    @app.exception_handler(ApiError)
    async def _api(request: Request, exc: ApiError):
        return JSONResponse(error_body(request, exc.code, str(exc.detail)), status_code=exc.status_code)

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, exc: StarletteHTTPException):
        return JSONResponse(error_body(request, f"http_{exc.status_code}", str(exc.detail)), status_code=exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError):
        first = exc.errors()[0] if exc.errors() else {}
        loc = ".".join(str(x) for x in first.get("loc", []) if x != "body")
        return JSONResponse(error_body(request, "validation_error", f"{loc}: {first.get('msg', 'invalid request')}".strip(": ")),
                            status_code=422)

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        log.exception("unhandled error")
        return JSONResponse(error_body(request, "internal_error", "Something went wrong. Please try again."), status_code=500)
