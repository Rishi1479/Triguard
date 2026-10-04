"""Edge cases across analyzers, pipeline and API: nothing may crash or return a 5xx."""
import io
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.app.main import create_app
from backend.app.settings import Settings
from ml.fusion.fuser import Fuser
from ml.image.analyzer import ImageAnalyzer
from ml.image.preprocess import ImageDecodeError, decode_image
from ml.pipeline import NoUsableInputError, TriGuardPipeline
from ml.text.analyzer import TextAnalyzer
from ml.url.analyzer import URLAnalyzer


class FakeOCR:
    name = "fake"

    def read(self, img):
        return "Offer for Data Analyst. Pay registration fee Rs 1500."


@pytest.fixture(scope="module")
def pipe():
    # real trained models if present in models/, otherwise rules only; no network, fake OCR
    text = TextAnalyzer()
    return TriGuardPipeline(text=text, image=ImageAnalyzer(text, ocr=FakeOCR(), visual=None),
                            url=URLAnalyzer(enable_live=False), fuser=Fuser())


def png(w=64, h=64, mode="RGB") -> bytes:
    buf = io.BytesIO()
    Image.new(mode, (w, h), "white" if mode != "1" else 1).save(buf, format="PNG")
    return buf.getvalue()


NON_ENGLISH = [
    "नौकरी के लिए पंजीकरण शुल्क ₹2000 जमा करें, कोई इंटरव्यू नहीं", "عمل من المنزل ادفع رسوم التسجيل الآن",
    "招聘：日本語のテキスト 登録料を支払ってください", "Привет! Работа дома, заплатите взнос 😀🔥", "வேலை வாய்ப்பு – பதிவு கட்டணம் செலுத்துங்கள்",
    "mixed: Pay Rs 2,000 फीस जमा करें now", "\u0000\u0007 control \x1b[31m chars", "😀" * 500, "a" * 5,
]


@pytest.mark.parametrize("text", NON_ENGLISH)
def test_non_english_and_odd_text(pipe, text):
    r = pipe.analyze(text=text)
    assert 0 <= r.risk_score <= 100 and r.label in ("Safe", "Suspicious", "Likely Scam") and r.reasons


def test_huge_text_is_bounded(pipe):
    big = ("We are hiring engineers. Pay registration fee Rs 2,000. " * 20_000)  # ~1.1 MB
    t0 = time.time()
    r = pipe.analyze(text=big)
    assert time.time() - t0 < 30 and r.label != "Safe"


@pytest.mark.parametrize("text", ["", "   ", "\n\t"])
def test_empty_text_alone_is_rejected_but_ok_with_url(pipe, text):
    with pytest.raises(NoUsableInputError):
        pipe.analyze(text=text)
    assert pipe.analyze(text=text, url="https://www.tcs.com/careers").modality_results[0].available is False


MALFORMED_URLS = ["http://", "http://[::1", "javascript:alert(1)", "data:text/html,<script>", "   ", "ht!tp://bad", "http://256.256.256.256/x",
                  "http://exa mple.com", "://nohost", "http://" + "a" * 5000 + ".com", "http://пример.рф/вакансия", "http://user:pw@evil.com@tcs.com/",
                  "http://xn--80ak6aa92e.com", "file:///etc/passwd", "http://localhost:8000/admin", "\x00http://a.com"]


@pytest.mark.parametrize("url", MALFORMED_URLS)
def test_malformed_urls_never_crash(pipe, url):
    try:
        r = pipe.analyze(text="Hiring a Data Analyst, apply now", url=url)
        assert 0 <= r.risk_score <= 100
    except NoUsableInputError:
        pytest.fail("text was valid, so the analysis must still succeed")
    try:
        pipe.analyze(url=url)
    except NoUsableInputError:
        pass  # unusable alone is fine


def _bad_images() -> dict[str, bytes]:
    good = png()
    return {"truncated": good[: len(good) // 2], "zero": b"", "garbage": b"\x89PNG\r\n\x1a\n" + b"junk" * 50,
            "jpeg-header-only": b"\xff\xd8\xff\xe0" + b"\x00" * 30, "text-file": b"hello world " * 100,
            "random": np.random.default_rng(0).bytes(5000)}


@pytest.mark.parametrize("name", list(_bad_images()))
def test_corrupt_images(pipe, name):
    data = _bad_images()[name]
    if not data:
        with pytest.raises(NoUsableInputError):
            pipe.analyze(image=data)
        return
    with pytest.raises(NoUsableInputError):
        pipe.analyze(image=data)
    r = pipe.analyze(text="Hiring analyst", image=data)  # a bad image must not block the other inputs
    assert r.modality_results[1].available is False and r.modality_results[1].red_flags[0].code == "UNREADABLE_IMAGE"


def test_decompression_bomb_rejected_before_decode():
    bomb = png(30_000, 30_000, mode="1")  # tiny file, 900 megapixels
    assert len(bomb) < 1_000_000
    with pytest.raises(ImageDecodeError, match="dimensions"):
        decode_image(bomb)


def test_tiny_and_blank_images_are_fine(pipe):
    for data in (png(1, 1), png(5, 3), png(2000, 40)):
        assert pipe.analyze(text="Hiring analyst", image=data).risk_score >= 0


@pytest.fixture(scope="module")
def client(pipe, tmp_path_factory):
    s = Settings(db_url=f"sqlite:///{tmp_path_factory.mktemp('db') / 'e.db'}", rate_limit_per_minute=10_000)
    with TestClient(create_app(pipeline=pipe, settings=s), raise_server_exceptions=False) as c:
        yield c


@pytest.mark.parametrize("data,files", [
    ({"text": "नौकरी शुल्क"}, None), ({"text": "x" * 20_000}, None), ({"url": "http://"}, None), ({"url": "javascript:alert(1)"}, None),
    ({"text": "hi"}, {"image": ("a.png", png()[:40] + b"\x00" * 10, "image/png")}),
    ({}, {"image": ("a.png", b"\x89PNG\r\n\x1a\n" + b"x" * 100, "image/png")}),
    ({"text": "a\x00b"}, None),
])
def test_api_never_5xx_on_odd_input(client, data, files):
    r = client.post("/api/analyze", data=data, files=files)
    assert r.status_code < 500, r.text
    if r.status_code != 200:
        assert "error" in r.json() and r.json()["error"]["request_id"]
