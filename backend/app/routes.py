"""API routes."""
from __future__ import annotations

import asyncio
import time

from fastapi import APIRouter, File, Form, Request, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from starlette.datastructures import UploadFile as StarletteUploadFile

from app.db import Analysis, Feedback
from app.errors import ApiError
from app.security import image_hash, mask, sniff_image_type
from ml.common.schemas import FinalResult
from ml.pipeline import NoUsableInputError
from ml.url.features import parse_host, registered_domain

router = APIRouter()
SENSITIVE_EXTRACTED_KEYS = ("ocr_text",)


class AnalyzeResponse(FinalResult):
    """FinalResult plus bookkeeping fields."""

    id: int
    latency_ms: int


class FeedbackIn(BaseModel):
    """"Was this correct?" payload."""

    analysis_id: int
    correct: bool
    comment: str | None = Field(default=None, max_length=500)


def _client_key(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _check_rate(request: Request) -> None:
    if not request.app.state.limiter.allow(_client_key(request)):
        raise ApiError(429, "rate_limited", "Too many requests. Please wait a minute and try again.")


def _domain(url: str | None) -> str | None:
    try:
        return registered_domain(parse_host(url)[0]) if url else None
    except ValueError:
        return None


async def _read_image(upload: UploadFile | str | None, max_bytes: int) -> bytes | None:
    if not isinstance(upload, StarletteUploadFile) or (not upload.filename and not upload.size):  # absent or empty form field
        return None
    data = await upload.read(max_bytes + 1)
    if not data:
        return None
    if len(data) > max_bytes:
        raise ApiError(413, "image_too_large", f"Image must be at most {max_bytes // (1024 * 1024)} MB.")
    if sniff_image_type(data) is None:
        raise ApiError(415, "unsupported_image_type", "Only PNG, JPEG or WebP images are accepted.")
    return data


@router.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze(request: Request, text: str | None = Form(default=None), url: str | None = Form(default=None),
                  image: UploadFile | str | None = File(default=None)) -> AnalyzeResponse:
    """Analyze any combination of text, URL and image (multipart/form-data)."""
    _check_rate(request)
    st = request.app.state
    s = st.settings
    text = text.strip() if text else None
    url = url.strip() if url else None
    if text and len(text) > s.max_text_chars:
        raise ApiError(422, "text_too_long", f"Text must be at most {s.max_text_chars} characters.")
    if url and len(url) > s.max_url_chars:
        raise ApiError(422, "url_too_long", f"URL must be at most {s.max_url_chars} characters.")
    data = await _read_image(image, s.max_image_bytes)
    if not (text or url or data):
        raise ApiError(400, "no_input", "Provide at least one of text, url or image.")

    t0 = time.perf_counter()
    loop = asyncio.get_running_loop()
    try:
        result: FinalResult = await asyncio.wait_for(
            loop.run_in_executor(st.executor, lambda: st.pipeline.analyze(text=text, image=data, url=url)),
            timeout=s.analysis_timeout_s)
    except NoUsableInputError as exc:
        raise ApiError(422, "unusable_input", str(exc))
    except asyncio.TimeoutError:
        raise ApiError(504, "timeout", "The analysis took too long. Try a smaller input.")
    latency = int((time.perf_counter() - t0) * 1000)

    for r in result.modality_results:  # never return/persist raw OCR text
        for k in SENSITIVE_EXTRACTED_KEYS:
            r.extracted.pop(k, None)
    present = "+".join(n for n, v in (("text", text), ("image", data), ("url", url)) if v)
    row = Analysis(inputs_present=present, anonymized_text=mask(text, s.stored_text_chars), url_domain=_domain(url),
                   image_hash=image_hash(data) if data else None, label=result.label, risk_score=result.risk_score,
                   reasons=[mask(x) for x in result.reasons], latency_ms=latency)
    with st.sessions() as db:
        db.add(row)
        db.commit()
    return AnalyzeResponse(**result.model_dump(), id=row.id, latency_ms=latency)


@router.post("/api/feedback", status_code=201)
def feedback(body: FeedbackIn, request: Request) -> dict:
    """Store feedback for an analysis."""
    _check_rate(request)
    with request.app.state.sessions() as db:
        if db.get(Analysis, body.analysis_id) is None:
            raise ApiError(404, "analysis_not_found", "No such analysis.")
        fb = Feedback(analysis_id=body.analysis_id, correct=body.correct, comment=mask(body.comment))
        db.add(fb)
        db.commit()
        return {"id": fb.id, "status": "recorded"}


@router.get("/api/history")
def history(request: Request) -> list[dict]:
    """Last analyses (anonymised)."""
    n = request.app.state.settings.history_size
    with request.app.state.sessions() as db:
        rows = db.scalars(select(Analysis).order_by(Analysis.id.desc()).limit(n)).all()
        return [{"id": r.id, "created_at": r.created_at.isoformat(), "inputs_present": r.inputs_present,
                 "label": r.label, "risk_score": r.risk_score, "url_domain": r.url_domain,
                 "text_preview": (r.anonymized_text or "")[:120] or None, "reasons": r.reasons[:3],
                 "latency_ms": r.latency_ms} for r in rows]


@router.get("/api/stats")
def stats(request: Request) -> dict:
    """Aggregate statistics."""
    with request.app.state.sessions() as db:
        total = db.scalar(select(func.count(Analysis.id))) or 0
        labels = dict(db.execute(select(Analysis.label, func.count()).group_by(Analysis.label)).all())
        combos = dict(db.execute(select(Analysis.inputs_present, func.count()).group_by(Analysis.inputs_present)).all())
        avg_risk, avg_lat = db.execute(select(func.avg(Analysis.risk_score), func.avg(Analysis.latency_ms))).one()
        fb_total = db.scalar(select(func.count(Feedback.id))) or 0
        fb_ok = db.scalar(select(func.count(Feedback.id)).where(Feedback.correct.is_(True))) or 0
    return {"total_analyses": total, "label_counts": labels, "input_combinations": combos,
            "avg_risk_score": round(avg_risk, 1) if avg_risk is not None else None,
            "avg_latency_ms": round(avg_lat) if avg_lat is not None else None,
            "feedback": {"total": fb_total, "marked_correct": fb_ok,
                         "accuracy": round(fb_ok / fb_total, 3) if fb_total else None}}


@router.get("/health")
def health(request: Request) -> dict:
    """Liveness + which models are loaded."""
    p = request.app.state.pipeline
    return {"status": "ok", "pipeline_loaded": p is not None,
            "text_model": getattr(getattr(p, "text", None), "model_name", None),
            "fusion_mode": getattr(getattr(p, "fuser", None), "mode", None)}
