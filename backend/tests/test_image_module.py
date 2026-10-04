import shutil
import tempfile

import cv2
import numpy as np
import pytest
from PIL import Image

from ml.common import config
from ml.common.schemas import ModalityResult
from ml.image import ocr as ocr_mod
from ml.image.analyzer import ImageAnalyzer
from ml.image.compare_ocr import cer
from ml.image.preprocess import ImageDecodeError, adaptive_threshold, decode_image, deskew, estimate_skew, preprocess, resize_for_ocr
from ml.image.qr import decode_qr
from ml.text.analyzer import TextAnalyzer
import random

import io

from ml.image import synth_letters

SCAM_TEXT = ("CONGRATULATIONS! You are SELECTED as Data Analyst. No interview required. Pay refundable registration fee "
             "Rs 1,500 via UPI to confirm your seat. Email hrjobs@gmail.com")


class FakeOCR:
    name = "fake"

    def __init__(self, text):
        self.text, self.calls = text, 0

    def read(self, img):
        self.calls += 1
        return self.text


class FakeVisual:
    def __init__(self, p):
        self.p = p

    def predict_proba(self, imgs):
        return np.array([self.p])


def png_bytes(arr_or_img) -> bytes:
    img = arr_or_img if isinstance(arr_or_img, Image.Image) else Image.fromarray(arr_or_img)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def qr_array(payload: str) -> np.ndarray:
    qr = cv2.QRCodeEncoder.create().encode(payload)
    qr = cv2.copyMakeBorder(qr, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
    return cv2.resize(qr, None, fx=6, fy=6, interpolation=cv2.INTER_NEAREST)


def letter(scam: bool, seed: int = 1) -> Image.Image:
    return synth_letters.make_letter(scam, random.Random(seed), np.random.default_rng(seed))[0]


def blank_with_qr(payload: str) -> bytes:
    canvas = np.full((900, 900), 255, np.uint8)
    q = qr_array(payload)
    canvas[100: 100 + q.shape[0], 100: 100 + q.shape[1]] = q
    return png_bytes(canvas)


# ---------------------------------------------------------------- preprocessing
def test_decode_rejects_bad_input():
    for bad in (b"", b"not an image", b"\x89PNG\r\n\x1a\n" + b"0" * 20):
        with pytest.raises(ImageDecodeError):
            decode_image(bad)
    with pytest.raises(ImageDecodeError):
        decode_image(b"0" * (config.MAX_IMAGE_BYTES + 1))


def test_preprocess_shapes_and_resize():
    small = np.full((100, 200, 3), 255, np.uint8)
    assert min(resize_for_ocr(small).shape[:2]) >= config.OCR_TARGET_MIN_SIDE
    huge = np.full((4000, 5000, 3), 255, np.uint8)
    assert max(resize_for_ocr(huge).shape[:2]) <= config.OCR_MAX_SIDE
    out = preprocess(small)
    assert out.ndim == 2
    assert set(np.unique(preprocess(small, binarize=True))) <= {0, 255}


def test_deskew_reduces_skew():
    img = np.array(letter(False))
    h, w = img.shape
    rot = cv2.warpAffine(img, cv2.getRotationMatrix2D((w / 2, h / 2), 6, 1), (w, h), borderValue=255)
    assert abs(estimate_skew(rot)) > 3
    assert abs(estimate_skew(deskew(rot))) < abs(estimate_skew(rot))
    assert estimate_skew(np.full((50, 50), 255, np.uint8)) == 0.0


# ---------------------------------------------------------------- QR
@pytest.mark.parametrize("payload", ["https://tcs-hiring.online/pay", "upi://pay?pa=hr@upi&am=1500"])
def test_qr_roundtrip(payload):
    assert decode_qr(qr_array(payload)) == [payload]


def test_no_qr_in_plain_letter():
    assert decode_qr(np.array(letter(True))) == []


# ---------------------------------------------------------------- analyzer (fake OCR)
@pytest.fixture
def text_an(tmp_path):
    return TextAnalyzer(models_dir=tmp_path)  # rules only: fast and deterministic


def test_analyzer_scam_text_via_ocr(text_an):
    fake = FakeOCR(SCAM_TEXT)
    r = ImageAnalyzer(text_an, ocr=fake, visual=None).analyze(png_bytes(letter(True)))
    assert r.available and r.modality == "image" and fake.calls >= 1
    assert {"PAYMENT_REQUEST", "NO_INTERVIEW"} <= {f.code for f in r.red_flags}
    assert "hrjobs@gmail.com" in r.extracted["emails"] and "ocr_text" in r.extracted
    assert r.explanation["weights"]["visual"] == 0.0  # no CLIP model -> weight 0
    assert r.probability > config.RULES_ONLY_PRIOR


def test_analyzer_genuine_text_low(text_an):
    r = ImageAnalyzer(text_an, ocr=FakeOCR("We are pleased to offer you the position of Software Engineer. Please sign and return."),
                      visual=None).analyze(png_bytes(letter(False)))
    assert r.probability == pytest.approx(config.RULES_ONLY_PRIOR) and not r.red_flags


def test_visual_score_is_weighted(text_an):
    an = ImageAnalyzer(text_an, ocr=FakeOCR("We are pleased to offer you a position at our firm."), visual=FakeVisual(1.0))
    r = an.analyze(png_bytes(letter(False)))
    w_v, w_t = config.IMAGE_WEIGHT_VISUAL, config.IMAGE_WEIGHT_OCR_TEXT
    expected = (w_v * 1.0 + w_t * config.RULES_ONLY_PRIOR) / (w_v + w_t)
    assert r.probability == pytest.approx(expected) and r.explanation["visual_probability"] == 1.0


def test_qr_link_and_upi_payment(text_an):
    an = ImageAnalyzer(text_an, ocr=FakeOCR(""), visual=None)
    r = an.analyze(blank_with_qr("https://tcs-hiring.online/apply"))
    assert r.extracted["qr_links"] == ["https://tcs-hiring.online/apply"]
    assert "https://tcs-hiring.online/apply" in r.extracted["urls"] and "QR_LINK" in {f.code for f in r.red_flags}
    p = an.analyze(blank_with_qr("upi://pay?pa=hr@upi&am=1500"))
    assert "QR_PAYMENT_REQUEST" in {f.code for f in p.red_flags} and p.probability >= 0.5


def test_no_text_found_flag(text_an):
    r = ImageAnalyzer(text_an, ocr=FakeOCR(""), visual=None).analyze(png_bytes(np.full((200, 200), 255, np.uint8)))
    assert r.available and "NO_TEXT_FOUND" in {f.code for f in r.red_flags}


def test_none_corrupt_and_no_ocr(text_an):
    an = ImageAnalyzer(text_an, ocr=FakeOCR("x"), visual=None)
    assert an.analyze(None) == ModalityResult.unavailable("image")
    assert an.analyze(b"") == ModalityResult.unavailable("image")
    bad = an.analyze(b"garbage bytes")
    assert not bad.available and bad.red_flags[0].code == "UNREADABLE_IMAGE"
    assert ImageAnalyzer(text_an, ocr=None, visual=None).analyze(png_bytes(letter(True))).available


def test_never_writes_to_disk(text_an, monkeypatch):
    qr_img, letter_img = blank_with_qr("https://a.com/x"), png_bytes(letter(True))  # build inputs before patching

    def boom(*a, **k):
        raise AssertionError("image analysis must not create temp files")

    for name in ("mkstemp", "NamedTemporaryFile", "TemporaryFile", "mkdtemp"):
        monkeypatch.setattr(tempfile, name, boom)
    monkeypatch.setattr(cv2, "imwrite", boom)
    monkeypatch.setattr(Image.Image, "save", boom)
    an = ImageAnalyzer(text_an, ocr=FakeOCR(SCAM_TEXT), visual=None)
    assert an.analyze(qr_img).available
    if shutil.which("tesseract"):  # the real Tesseract path pipes through stdin/stdout
        ImageAnalyzer(text_an, ocr=ocr_mod.TesseractEngine(), visual=None).analyze(letter_img)


# ---------------------------------------------------------------- real OCR end-to-end
@pytest.mark.skipif(not shutil.which("tesseract"), reason="tesseract not installed")
def test_real_tesseract_reads_scam_letter(text_an):
    r = ImageAnalyzer(text_an, ocr=ocr_mod.TesseractEngine(), visual=None).analyze(png_bytes(letter(True, seed=3)))
    assert r.explanation["ocr_chars"] > 100
    assert {"NO_INTERVIEW", "SENSITIVE_INFO_REQUEST"} & {f.code for f in r.red_flags}


@pytest.mark.skipif(not shutil.which("tesseract"), reason="tesseract not installed")
def test_real_ocr_plus_qr_in_one_image(text_an):
    base = np.array(letter(True, seed=4))
    q = qr_array("https://tcs-hiring.online/apply")
    q = cv2.resize(q, (200, 200), interpolation=cv2.INTER_NEAREST)
    base[760:960, 560:760] = q
    r = ImageAnalyzer(text_an, ocr=ocr_mod.TesseractEngine(), visual=None).analyze(png_bytes(base))
    assert r.extracted["qr_links"] == ["https://tcs-hiring.online/apply"] and r.explanation["ocr_chars"] > 100


def test_get_engine_none_when_unavailable(monkeypatch):
    monkeypatch.setattr(ocr_mod, "EasyOCREngine", lambda: (_ for _ in ()).throw(RuntimeError("no")))
    monkeypatch.setattr(shutil, "which", lambda n: None)
    assert ocr_mod.get_engine("auto") is None


def test_cer():
    assert cer("hello world", "hello  world") == 0.0
    assert cer("abcd", "abxd") == pytest.approx(0.25)
    assert cer("", "abc") == 3.0
