"""QR / barcode decoding: pyzbar when the zbar library is present, OpenCV's QR detector otherwise."""
from __future__ import annotations

import ctypes.util
import logging
import os

import cv2
import numpy as np

log = logging.getLogger(__name__)
_HOMEBREW_ZBAR = "/opt/homebrew/lib/libzbar.dylib"
QR_UPSCALE = 2.0

_pyzbar = None
try:  # pyzbar cannot find Homebrew's arm64 libzbar on its own
    _orig_find = ctypes.util.find_library
    if os.path.exists(_HOMEBREW_ZBAR):
        ctypes.util.find_library = lambda n: _HOMEBREW_ZBAR if n == "zbar" else _orig_find(n)
    from pyzbar import pyzbar as _pyzbar
except Exception as exc:  # ImportError, or OSError when libzbar is missing
    log.info("pyzbar unavailable (%s); using OpenCV QR detector", exc)


def _decode_pyzbar(gray: np.ndarray) -> list[str]:
    if _pyzbar is None:
        return []
    return [r.data.decode("utf-8", errors="ignore") for r in _pyzbar.decode(gray)]


def _decode_opencv(gray: np.ndarray) -> list[str]:
    det = cv2.QRCodeDetector()
    try:
        ok, texts, _pts, _ = det.detectAndDecodeMulti(gray)
        if ok:
            return [t for t in texts if t]
    except cv2.error:
        pass
    text = det.detectAndDecode(gray)[0]
    return [text] if text else []


def decode_qr(img: np.ndarray) -> list[str]:
    """Decoded payloads of all QR/bar codes found in ``img`` (deduplicated, order kept)."""
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    found: list[str] = []
    variants = [gray, cv2.resize(gray, None, fx=QR_UPSCALE, fy=QR_UPSCALE, interpolation=cv2.INTER_CUBIC)]
    for v in variants:
        found = _decode_pyzbar(v) or _decode_opencv(v)
        if found:
            break
    return list(dict.fromkeys(found))
