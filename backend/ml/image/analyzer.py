"""ImageAnalyzer: OCR -> TextAnalyzer -> QR decode (+ optional CLIP visual score) -> ModalityResult.

The image is handled purely in memory and never written to disk.
"""
from __future__ import annotations

import logging
import re

from ml.common import config
from ml.common.red_flags import extract_entities
from ml.common.schemas import ModalityResult, RedFlag
from ml.image.ocr import OCREngine, get_engine
from ml.image.preprocess import ImageDecodeError, decode_image, preprocess
from ml.image.qr import decode_qr
from ml.image.visual import load_visual_scorer
from ml.text.analyzer import TextAnalyzer

log = logging.getLogger(__name__)
_UPI_RE = re.compile(r"^upi://", re.I)
_URL_PREFIX_RE = re.compile(r"^(?:https?://|www\.)", re.I)
_UNSET = object()


def _unreadable(msg: str) -> ModalityResult:
    return ModalityResult(modality="image", available=False, extracted={"error": msg},
                          red_flags=[RedFlag(code="UNREADABLE_IMAGE", message="The image could not be read.",
                                             severity="low", evidence=msg)])


class ImageAnalyzer:
    """Analyze an offer letter / poster / screenshot given as raw bytes."""

    def __init__(self, text_analyzer: TextAnalyzer | None = None, ocr: OCREngine | None | object = _UNSET,
                 visual: object = _UNSET) -> None:
        self.text = text_analyzer or TextAnalyzer()
        self.ocr = get_engine("auto") if ocr is _UNSET else ocr
        self.visual = load_visual_scorer() if visual is _UNSET else visual

    @property
    def weights(self) -> tuple[float, float]:
        """(visual, ocr_text) weights; visual is 0 when no CLIP model is available."""
        return (config.IMAGE_WEIGHT_VISUAL if self.visual else 0.0, config.IMAGE_WEIGHT_OCR_TEXT)

    def _ocr_text(self, img) -> str:
        if self.ocr is None:
            return ""
        text = self.ocr.read(preprocess(img))
        if len(text.strip()) < config.MIN_OCR_CHARS:  # retry on the binarised version
            alt = self.ocr.read(preprocess(img, binarize=True))
            text = alt if len(alt.strip()) > len(text.strip()) else text
        return re.sub(r"[ \t]+", " ", text).strip()

    def analyze(self, image_bytes: bytes | None, explain: bool = True) -> ModalityResult:
        """Score an image. None/empty -> unavailable; corrupt -> unavailable + UNREADABLE_IMAGE flag."""
        if not image_bytes:
            return ModalityResult.unavailable("image")
        try:
            img = decode_image(image_bytes)
        except ImageDecodeError as exc:
            return _unreadable(str(exc))

        text = self._ocr_text(img)
        flags: list[RedFlag] = []
        t_res = self.text.analyze(text, explain=explain) if len(text) >= config.MIN_OCR_CHARS else None
        if t_res is not None:
            flags += t_res.red_flags
        elif self.ocr is not None:
            flags.append(RedFlag(code="NO_TEXT_FOUND", message="No readable text found in the image.",
                                 severity="low", evidence=f"{len(text)} characters recognised"))

        qr_payloads = decode_qr(img)
        qr_links = [q for q in qr_payloads if _URL_PREFIX_RE.match(q)]
        for q in qr_payloads:
            if _UPI_RE.match(q):
                flags.append(RedFlag(code="QR_PAYMENT_REQUEST", message="The QR code is a payment request — genuine "
                                     "employers never ask candidates to pay.", severity="high", evidence=q[:80]))
        if qr_links:
            flags.append(RedFlag(code="QR_LINK", message="The image hides a link in a QR code; verify it before scanning.",
                                 severity="low", evidence=qr_links[0][:120]))

        w_vis, w_txt = self.weights
        p_vis = float(self.visual.predict_proba([img[:, :, ::-1]])[0]) if w_vis else None
        parts = [(w_txt, t_res.probability)] if t_res is not None else []
        if p_vis is not None:
            parts.append((w_vis, p_vis))
        prob = sum(w * p for w, p in parts) / sum(w for w, _ in parts) if parts else 0.0
        if any(f.code == "QR_PAYMENT_REQUEST" for f in flags):
            prob = max(prob, 0.5 + prob / 2)

        ents = t_res.extracted if t_res is not None else extract_entities(text)
        extracted = {**ents, "urls": list(dict.fromkeys([*ents.get("urls", []), *qr_links])), "qr_links": qr_links,
                     "qr_payloads": qr_payloads, "ocr_text": text[: config.OCR_TEXT_KEEP_CHARS],
                     "n_chars": len(text)}  # OCR yield: drives the fuser's evidence-strength check
        explanation = {"ocr_backend": getattr(self.ocr, "name", None), "ocr_chars": len(text),
                       "visual_probability": p_vis, "text_probability": t_res.probability if t_res else None,
                       "weights": {"visual": w_vis, "ocr_text": w_txt}}
        if t_res is not None and "lime_top_words" in t_res.explanation:
            explanation["lime_top_words"] = t_res.explanation["lime_top_words"]
        return ModalityResult(modality="image", probability=min(max(prob, 0.0), 1.0), red_flags=flags,
                              extracted=extracted, explanation=explanation)
