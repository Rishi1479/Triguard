"""TriGuard FastAPI application: pipeline loaded once at startup, inference in a thread pool."""
from __future__ import annotations

import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.db import init_db, make_engine
from backend.app.errors import error_body, install_handlers
from backend.app.logging_conf import log_event, setup_logging
from backend.app.routes import router
from backend.app.security import RateLimiter
from backend.app.settings import Settings

log = logging.getLogger("triguard.api")


def create_app(pipeline=None, settings: Settings | None = None) -> FastAPI:
    """Build the app. Pass ``pipeline`` to inject a prebuilt/mock pipeline (otherwise loaded at startup)."""
    settings = settings or Settings()
    setup_logging()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        engine = make_engine(settings.db_url)
        app.state.sessions = init_db(engine)
        app.state.executor = ThreadPoolExecutor(max_workers=settings.workers, thread_name_prefix="analysis")
        if pipeline is not None:
            app.state.pipeline = pipeline
        else:
            from ml.pipeline import TriGuardPipeline
            app.state.pipeline = TriGuardPipeline(enable_live=settings.enable_live_checks)
        log_event(log, "startup complete", text_model=getattr(getattr(app.state.pipeline, "text", None), "model_name", None))
        yield
        app.state.executor.shutdown(wait=False, cancel_futures=True)
        engine.dispose()

    app = FastAPI(title="TriGuard API", version="0.2.0", lifespan=lifespan)
    app.state.settings, app.state.limiter, app.state.pipeline = settings, RateLimiter(settings.rate_limit_per_minute), None
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["GET", "POST"],
                       allow_headers=["*"], expose_headers=["X-Request-ID"])

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request.state.request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        length = request.headers.get("content-length", "")
        too_big = length.isdigit() and int(length) > settings.max_request_bytes
        t0 = time.perf_counter()
        if too_big:
            response = JSONResponse(error_body(request, "payload_too_large", "Request body is too large."), status_code=413)
        else:
            response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        log_event(log, "request", method=request.method, path=request.url.path, status=response.status_code,
                  ms=int((time.perf_counter() - t0) * 1000), request_id=request.state.request_id)
        return response

    install_handlers(app)
    app.include_router(router)
    return app


app = create_app()
