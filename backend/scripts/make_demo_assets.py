"""Create the demo offer-letter image (fake letter with a QR code) used in docs/demo_script.md (repo root).

The image carries a visible "DEMO SAMPLE – NOT A REAL OFFER" footer. Run: python -m scripts.make_demo_assets
"""
from __future__ import annotations

import random

import cv2
import numpy as np
from PIL import Image, ImageDraw

from ml.common import config
from ml.image import synth_letters

OUT = config.ROOT_DIR / "demo_assets" / "fake_offer_letter_with_qr.png"
QR_URL = "http://tcs-hiring.online/pay-registration"
QR_SIZE = 220


def build() -> Image.Image:
    """Scam-style letter naming TCS, with a QR code that leads to a look-alike domain."""
    rng, np_rng = random.Random(7), np.random.default_rng(7)
    img, _ = synth_letters.make_letter(True, rng, np_rng, company="TCS", role="Data Analyst", salary="600,000")
    qr = cv2.QRCodeEncoder.create().encode(QR_URL)
    qr = cv2.copyMakeBorder(qr, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=255)
    qr = cv2.resize(qr, (QR_SIZE, QR_SIZE), interpolation=cv2.INTER_NEAREST)
    canvas = img.convert("L")
    canvas.paste(Image.fromarray(qr), (synth_letters.CANVAS[0] - QR_SIZE - 50, synth_letters.CANVAS[1] - QR_SIZE - 110))
    d = ImageDraw.Draw(canvas)
    d.text((60, synth_letters.CANVAS[1] - 50), "DEMO SAMPLE - NOT A REAL OFFER (TriGuard test asset)", fill=90)
    return canvas


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    build().save(OUT)
    print("wrote", OUT)
