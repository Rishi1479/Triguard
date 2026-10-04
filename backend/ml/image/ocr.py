"""OCR backends: EasyOCR (primary) and Tesseract (alternative). Images are passed in memory only."""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
from typing import Protocol

import cv2
import numpy as np

from ml.common import config

log = logging.getLogger(__name__)


class OCREngine(Protocol):
    """Anything that turns an image array into text."""

    name: str

    def read(self, img: np.ndarray) -> str: ...


class EasyOCREngine:
    """EasyOCR (English, optional Hindi). Models download on first use; CPU only."""

    name = "easyocr"

    def __init__(self, languages: tuple[str, ...] = ("en",)) -> None:
        try:  # python.org builds on macOS often lack CA certs, which breaks the model download
            import certifi
            os.environ.setdefault("SSL_CERT_FILE", certifi.where())
        except ImportError:
            pass
        import easyocr
        self._reader = easyocr.Reader(list(languages), gpu=False, verbose=False)

    def read(self, img: np.ndarray) -> str:
        """Concatenate detected lines in reading order."""
        return " ".join(self._reader.readtext(img, detail=0, paragraph=False))


class TesseractEngine:
    """Tesseract via the CLI, piping a PNG through stdin/stdout so nothing is written to disk."""

    name = "tesseract"

    def __init__(self, lang: str = "eng", psm: int = 6) -> None:
        self.exe = shutil.which("tesseract")
        if not self.exe:
            raise RuntimeError("tesseract binary not found")
        self.lang, self.psm = lang, psm

    def read(self, img: np.ndarray) -> str:
        """OCR ``img``; returns '' on failure."""
        ok, buf = cv2.imencode(".png", img)
        if not ok:
            return ""
        try:
            proc = subprocess.run([self.exe, "stdin", "stdout", "-l", self.lang, "--psm", str(self.psm)],
                                  input=buf.tobytes(), capture_output=True, timeout=config.TESSERACT_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            log.warning("tesseract timed out")
            return ""
        return proc.stdout.decode("utf-8", errors="ignore")


def get_engine(backend: str = "auto") -> OCREngine | None:
    """Build an OCR engine. 'auto' tries EasyOCR then Tesseract and returns None if neither works."""
    order = {"auto": ("easyocr", "tesseract"), "easyocr": ("easyocr",), "tesseract": ("tesseract",)}[backend]
    for name in order:
        try:
            return EasyOCREngine() if name == "easyocr" else TesseractEngine()
        except Exception as exc:
            log.warning("OCR backend %s unavailable: %s", name, exc)
    return None
