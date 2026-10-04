import io
import sqlite3
import threading
import time

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.settings import Settings
from ml.common import config
from ml.common.schemas import FinalResult, ModalityResult, RedFlag
from ml.pipeline import NoUsableInputError

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64  # sniffed by magic bytes only (the mock pipeline never decodes it)


class FakePipeline:
    text = type("T", (), {"model_name": "fake"})()
    fuser = type("F", (), {"mode": "weighted_average"})()

    def __init__(self):
        self.calls, self.thread, self.delay, self.error = [], None, 0, None

    def analyze(self, text=None, image=None, url=None):
        self.thread = threading.current_thread().name
        self.calls.append({"text": text, "image": image, "url": url})
        time.sleep(self.delay)
        if self.error:
            raise self.error
        img = ModalityResult(modality="image", probability=.5, extracted={"ocr_text": "SECRET 9876543210", "urls": []})
        return FinalResult(label="Suspicious", risk_score=55.5, modality_results=[ModalityResult(modality="text", probability=.6), img],
                           reasons=["Warning – Text: mail hr@gmail.com or call 9876543210"],
                           cross_modal_flags=[RedFlag(code="X", message="m", severity="low")])


def make_client(tmp_path, pipe=None, **kw):
    pipe = pipe or FakePipeline()
    s = Settings(db_url=f"sqlite:///{tmp_path / 't.db'}", rate_limit_per_minute=kw.pop("rate", 1000), **kw)
    return TestClient(create_app(pipeline=pipe, settings=s), raise_server_exceptions=False), pipe, tmp_path / "t.db"


@pytest.fixture
def ctx(tmp_path):
    c, pipe, db = make_client(tmp_path)
    with c:
        yield c, pipe, db


def err(r):
    return r.json()["error"]


def test_no_input_400(ctx):
    c, _, _ = ctx
    for data in ({}, {"text": "   "}, {"url": ""}):
        r = c.post("/api/analyze", data=data)
        assert r.status_code == 400 and err(r)["code"] == "no_input" and err(r)["request_id"]


def test_text_url_ok_response_shape_and_runs_in_worker_thread(ctx):
    c, pipe, _ = ctx
    r = c.post("/api/analyze", data={"text": "Pay fee now", "url": "http://tcs-hiring.online"})
    body = r.json()
    assert r.status_code == 200 and body["label"] == "Suspicious" and body["risk_score"] == 55.5
    assert body["id"] >= 1 and body["latency_ms"] >= 0 and {"modality_results", "reasons", "cross_modal_flags"} <= set(body)
    assert pipe.thread.startswith("analysis") and r.headers["x-request-id"]
    assert all("ocr_text" not in m["extracted"] for m in body["modality_results"])  # raw OCR text never returned
    assert pipe.calls[0] == {"text": "Pay fee now", "image": None, "url": "http://tcs-hiring.online"}


def test_image_validation(ctx):
    c, pipe, _ = ctx
    ok = c.post("/api/analyze", files={"image": ("a.png", PNG, "image/png")})
    assert ok.status_code == 200 and pipe.calls[-1]["image"] == PNG
    bad = c.post("/api/analyze", files={"image": ("a.png", b"hello this is text", "image/png")})
    assert bad.status_code == 415 and err(bad)["code"] == "unsupported_image_type"
    big = c.post("/api/analyze", files={"image": ("a.png", PNG + b"0" * (5 * 1024 * 1024), "image/png")})
    assert big.status_code == 413 and err(big)["code"] == "image_too_large"
    for name, head in (("a.jpg", b"\xff\xd8\xff\xe0"), ("a.webp", b"RIFF\x00\x00\x00\x00WEBPVP8 ")):
        assert c.post("/api/analyze", files={"image": (name, head + b"0" * 20, "image/x")}).status_code == 200


def test_empty_file_field_is_ignored(ctx):
    c, _, _ = ctx
    r = c.post("/api/analyze", data={"text": "hello"}, files={"image": ("", b"", "application/octet-stream")})
    assert r.status_code == 200


def test_length_limits(ctx):
    c, _, _ = ctx
    assert err(c.post("/api/analyze", data={"text": "a" * 20_001}))["code"] == "text_too_long"
    assert err(c.post("/api/analyze", data={"url": "http://a.com/" + "x" * 3000}))["code"] == "url_too_long"


def test_unusable_input_and_timeout_and_crash(tmp_path):
    c, pipe, _ = make_client(tmp_path, analysis_timeout_s=0.2)
    with c:
        pipe.error = NoUsableInputError("The image could not be read.")
        r = c.post("/api/analyze", data={"text": "x"})
        assert r.status_code == 422 and err(r)["code"] == "unusable_input"
        pipe.error, pipe.delay = None, 1.0
        assert c.post("/api/analyze", data={"text": "x"}).status_code == 504
        pipe.delay, pipe.error = 0, RuntimeError("kaboom secret")
        r = c.post("/api/analyze", data={"text": "x"})
        assert r.status_code == 500 and "secret" not in r.text and err(r)["code"] == "internal_error"


def test_storage_is_anonymised_and_image_hashed(ctx):
    c, _, db = ctx
    c.post("/api/analyze", data={"text": "Call 9876543210 or mail john@gmail.com. Aadhaar 1234 5678 9012", "url": "https://x.tcs-hiring.online/p?a=1"},
           files={"image": ("a.png", PNG, "image/png")})
    row = sqlite3.connect(db).execute("select anonymized_text, url_domain, image_hash, reasons from analyses").fetchone()
    assert "9876543210" not in row[0] and "john@gmail.com" not in row[0] and "1234 5678" not in row[0]
    assert "[PHONE]" in row[0] and "[EMAIL]" in row[0] and "[AADHAAR]" in row[0]
    assert row[1] == "tcs-hiring.online" and len(row[2]) == 64
    assert "9876543210" not in row[3] and "hr@gmail.com" not in row[3]
    assert b"\x89PNG" not in open(db, "rb").read()  # image bytes are nowhere in the DB file


def test_history_stats_feedback(ctx):
    c, _, _ = ctx
    ids = [c.post("/api/analyze", data={"text": f"posting number {i} call 9876543210"}).json()["id"] for i in range(25)]
    h = c.get("/api/history").json()
    assert len(h) == 20 and h[0]["id"] == ids[-1] and "9876543210" not in str(h) and h[0]["inputs_present"] == "text"
    assert c.post("/api/feedback", json={"analysis_id": ids[0], "correct": True}).status_code == 201
    assert c.post("/api/feedback", json={"analysis_id": ids[1], "correct": False, "comment": "mail me a@b.com"}).status_code == 201
    assert err(c.post("/api/feedback", json={"analysis_id": 9999, "correct": True}))["code"] == "analysis_not_found"
    assert c.post("/api/feedback", json={"analysis_id": 1}).status_code == 422
    assert c.post("/api/feedback", json={"analysis_id": 1, "correct": True, "comment": "x" * 501}).status_code == 422
    s = c.get("/api/stats").json()
    assert s["total_analyses"] == 25 and s["label_counts"] == {"Suspicious": 25} and s["input_combinations"] == {"text": 25}
    assert s["feedback"] == {"total": 2, "marked_correct": 1, "accuracy": 0.5} and s["avg_risk_score"] == 55.5


def test_empty_stats(ctx):
    c, _, _ = ctx
    s = c.get("/api/stats").json()
    assert s["total_analyses"] == 0 and s["avg_risk_score"] is None and s["feedback"]["accuracy"] is None
    assert c.get("/api/history").json() == []


def test_rate_limit(tmp_path):
    c, _, _ = make_client(tmp_path, rate=3)
    with c:
        codes = [c.post("/api/analyze", data={"text": "x"}).status_code for _ in range(5)]
        assert codes == [200, 200, 200, 429, 429]
        assert err(c.post("/api/analyze", data={"text": "x"}))["code"] == "rate_limited"
        assert c.get("/health").status_code == 200  # health is not limited


def test_cors_and_error_shapes(ctx):
    c, _, _ = ctx
    r = c.options("/api/analyze", headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST"})
    assert r.headers["access-control-allow-origin"] == "http://localhost:5173"
    assert c.get("/nope").json()["error"]["code"] == "http_404"
    assert "x-request-id" in c.get("/health").headers
    assert c.post("/api/analyze", content=b"x" * 7_000_000, headers={"content-type": "multipart/form-data; boundary=a"}).status_code == 413


def test_health_reports_models(ctx):
    c, _, _ = ctx
    assert c.get("/health").json() == {"status": "ok", "pipeline_loaded": True, "text_model": "fake", "fusion_mode": "weighted_average"}


# ---------------------------------------------------------------- one real integration test
def test_real_pipeline_integration(tmp_path):
    import random

    import numpy as np

    from ml.image import synth_letters
    from ml.pipeline import TriGuardPipeline

    buf = io.BytesIO()
    synth_letters.make_letter(True, random.Random(2), np.random.default_rng(2))[0].save(buf, format="PNG")
    pipe = TriGuardPipeline(enable_live=False, ocr_backend="tesseract")
    c, _, _ = make_client(tmp_path, pipe=pipe)
    with c:
        r = c.post("/api/analyze", data={"text": "TCS is hiring. Pay a registration fee of Rs 2,000. No interview.",
                                         "url": "http://tcs-hiring.online/apply"},
                   files={"image": ("l.png", buf.getvalue(), "image/png")})
        assert r.status_code == 200
        body = r.json()
        assert body["label"] in ("Suspicious", "Likely Scam") and len(body["reasons"]) >= 1
        assert {m["modality"] for m in body["modality_results"] if m["available"]} >= {"text", "image", "url"}
        assert c.get("/health").json()["text_model"] in (*config.TEXT_MODEL_PRIORITY, "rules_only")


# ---------------------------------------------------------------- history detail
def test_history_detail_masked_and_complete(ctx):
    c, _, db = ctx
    aid = c.post("/api/analyze", data={"text": "Call 9876543210 now", "url": "https://x.tcs-hiring.online/p?token=SECRET123"}).json()["id"]
    c.post("/api/feedback", json={"analysis_id": aid, "correct": False, "comment": "wrong, mail a@b.com"})
    d = c.get(f"/api/history/{aid}").json()
    assert d["id"] == aid and d["detail_available"] and d["url_domain"] == "tcs-hiring.online"
    assert "9876543210" not in d["submitted_text"] and "[PHONE]" in d["submitted_text"]
    assert d["result"]["label"] == "Suspicious" and len(d["result"]["modality_results"]) == 2 and d["result"]["cross_modal_flags"]
    assert d["feedback"][0]["correct"] is False and "a@b.com" not in str(d["feedback"])
    blob = str(d)
    assert "9876543210" not in blob and "hr@gmail.com" not in blob
    assert "SECRET123" not in sqlite3.connect(db).execute("select result_json from analyses").fetchone()[0]


def test_history_detail_404_and_legacy_rows(tmp_path):
    c, _, db = make_client(tmp_path)
    with c:
        assert err(c.get("/api/history/999"))["code"] == "analysis_not_found"
        aid = c.post("/api/analyze", data={"text": "x"}).json()["id"]
    con = sqlite3.connect(db)
    con.execute("update analyses set result_json = NULL")
    con.commit()
    c2, _, _ = make_client(tmp_path)
    with c2:
        d = c2.get(f"/api/history/{aid}").json()
        assert d["detail_available"] is False and d["result"]["label"] == "Suspicious" and d["result"]["modality_results"] == []


def test_old_database_is_migrated(tmp_path):
    db = tmp_path / "old.db"
    con = sqlite3.connect(db)
    con.execute("create table analyses (id integer primary key, created_at datetime, inputs_present varchar(32), anonymized_text text, "
                "url_domain varchar(255), image_hash varchar(64), label varchar(16), risk_score float, reasons json, latency_ms integer)")
    con.commit()
    from app.db import init_db, make_engine
    init_db(make_engine(f"sqlite:///{db}"))
    assert "result_json" in [r[1] for r in sqlite3.connect(db).execute("pragma table_info(analyses)")]
