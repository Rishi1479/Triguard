"""OpenCV preprocessing for OCR: grayscale, denoise, deskew, adaptive threshold, resize. All in memory."""
from __future__ import annotations

import cv2
import numpy as np

from ml.common import config

MAX_DESKEW_DEG = 15.0
MIN_TEXT_PIXELS = 50


class ImageDecodeError(ValueError):
    """Raised when bytes are not a decodable, reasonably sized image."""


def decode_image(data: bytes) -> np.ndarray:
    """Decode bytes to a BGR array without touching the disk. Rejects corrupt/oversized images."""
    if not data or len(data) > config.MAX_IMAGE_BYTES:
        raise ImageDecodeError("empty or too large image")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ImageDecodeError("not a valid image")
    if img.shape[0] * img.shape[1] > config.MAX_IMAGE_PIXELS:
        raise ImageDecodeError("image dimensions too large")
    return img


def to_gray(img: np.ndarray) -> np.ndarray:
    """Grayscale copy (no-op for 2-D input)."""
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def denoise(gray: np.ndarray) -> np.ndarray:
    """Non-local-means denoising for 8-bit grayscale."""
    return cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)


def estimate_skew(gray: np.ndarray) -> float:
    """Skew angle in degrees (positive = counter-clockwise text), 0 if unreliable."""
    inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(inv > 0))
    if len(coords) < MIN_TEXT_PIXELS:
        return 0.0
    angle = cv2.minAreaRect(coords[:, ::-1].astype(np.float32))[-1]
    angle = angle - 90 if angle > 45 else angle  # normalise to (-45, 45]
    return float(angle) if abs(angle) <= MAX_DESKEW_DEG else 0.0


def deskew(gray: np.ndarray) -> np.ndarray:
    """Rotate to level the text lines."""
    angle = estimate_skew(gray)
    if abs(angle) < 0.3:
        return gray
    h, w = gray.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(gray, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def adaptive_threshold(gray: np.ndarray) -> np.ndarray:
    """Gaussian adaptive binarisation (robust to uneven lighting)."""
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15)


def resize_for_ocr(img: np.ndarray) -> np.ndarray:
    """Upscale tiny images (min side >= OCR_TARGET_MIN_SIDE) and cap huge ones (max side <= OCR_MAX_SIDE)."""
    h, w = img.shape[:2]
    scale = 1.0
    if min(h, w) < config.OCR_TARGET_MIN_SIDE:
        scale = config.OCR_TARGET_MIN_SIDE / min(h, w)
    if max(h, w) * scale > config.OCR_MAX_SIDE:
        scale = config.OCR_MAX_SIDE / max(h, w)
    if abs(scale - 1.0) < 0.05:
        return img
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)


def preprocess(img: np.ndarray, binarize: bool = False) -> np.ndarray:
    """Full pipeline: resize -> grayscale -> denoise -> deskew [-> adaptive threshold]."""
    gray = deskew(denoise(to_gray(resize_for_ocr(img))))
    return adaptive_threshold(gray) if binarize else gray
