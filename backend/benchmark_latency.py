"""Measure average API latency per input combination -> reports/backend_latency.md.

Uses the real pipeline (Tesseract OCR, live network checks off) through FastAPI's TestClient.
"""
from __future__ import annotations

import itertools
import statistics
import tempfile
import time
from pathlib import Path

from fastapi.testclient import TestClient

from backend.app.main import create_app
from backend.app.settings import Settings
from ml.common import config
from ml.pipeline import TriGuardPipeline

REPEATS = 5
TEXT = "TCS is hiring a Data Analyst. Pay a registration fee of Rs 2,000 on UPI. No interview. Contact tcshr@gmail.com"
URL = "http://tcs-hiring.online/apply"


def run(repeats: int = REPEATS) -> str:
    """Time every non-empty combination of text/image/url and write the report."""
    letters = sorted((config.IMAGES_DIR / "scam").glob("*.png")) or sorted((config.IMAGES_DIR / "genuine").glob("*.png"))
    image = letters[0].read_bytes() if letters else None
    tmp = Path(tempfile.mkdtemp())
    pipe = TriGuardPipeline(enable_live=False, ocr_backend="tesseract")
    app = create_app(pipeline=pipe, settings=Settings(db_url=f"sqlite:///{tmp / 'b.db'}", rate_limit_per_minute=10_000))
    rows = []
    with TestClient(app) as c:
        c.post("/api/analyze", data={"text": "warm up"})
        for n in (1, 2, 3):
            for combo in itertools.combinations(("text", "image", "url"), n):
                if "image" in combo and image is None:
                    continue
                data = {k: v for k, v in (("text", TEXT), ("url", URL)) if k in combo}
                files = {"image": ("l.png", image, "image/png")} if "image" in combo else None
                ms = []
                for _ in range(repeats):
                    t0 = time.perf_counter()
                    assert c.post("/api/analyze", data=data, files=files).status_code == 200
                    ms.append((time.perf_counter() - t0) * 1000)
                rows.append(("+".join(combo), statistics.mean(ms), statistics.median(ms), max(ms)))
    lines = ["# Backend latency per input combination", "",
             f"Real pipeline via TestClient, {repeats} runs each, Tesseract OCR, live WHOIS/TLS checks disabled "
             "(add roughly the 3 s timeout per slow lookup when enabled; EasyOCR adds ~2-3 s per image on CPU).", "",
             "| Inputs | Mean ms | Median ms | Max ms |", "|---|---|---|---|"]
    lines += [f"| {n} | {m:.0f} | {md:.0f} | {mx:.0f} |" for n, m, md, mx in rows]
    out = "\n".join(lines) + "\n"
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORTS_DIR / "backend_latency.md").write_text(out)
    return out


if __name__ == "__main__":
    print(run())
