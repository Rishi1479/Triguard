"""SQLAlchemy models and session helpers (SQLite). Only anonymised data is stored; images only as a hash."""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text, create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from sqlalchemy.pool import StaticPool


def utcnow() -> datetime:
    """Timezone-aware UTC now."""
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    """Declarative base."""


class Analysis(Base):
    """One analysis request (anonymised)."""

    __tablename__ = "analyses"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    inputs_present: Mapped[str] = mapped_column(String(32))  # e.g. "text+url"
    anonymized_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    url_domain: Mapped[str | None] = mapped_column(String(255), nullable=True)
    image_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)  # sha256; the image itself is never stored
    label: Mapped[str] = mapped_column(String(16), index=True)
    risk_score: Mapped[float] = mapped_column(Float)
    reasons: Mapped[list] = mapped_column(JSON, default=list)
    latency_ms: Mapped[int] = mapped_column(Integer)


class Feedback(Base):
    """User feedback on an analysis ("Was this correct?")."""

    __tablename__ = "feedback"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    analysis_id: Mapped[int] = mapped_column(ForeignKey("analyses.id"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    correct: Mapped[bool] = mapped_column(Boolean)
    comment: Mapped[str | None] = mapped_column(String(500), nullable=True)


def make_engine(url: str) -> Engine:
    """SQLite engine safe for use from worker threads (in-memory DBs share one connection)."""
    kw: dict = {"connect_args": {"check_same_thread": False}}
    if url in ("sqlite://", "sqlite:///:memory:"):
        kw["poolclass"] = StaticPool
    return create_engine(url, **kw)


def init_db(engine: Engine) -> sessionmaker:
    """Create tables and return a session factory."""
    Base.metadata.create_all(engine)
    return sessionmaker(engine, expire_on_commit=False)
