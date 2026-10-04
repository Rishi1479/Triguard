"""Generate SYNTHETIC offer-letter images (genuine-style and scam-style) for pipeline testing."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ml.common import config

FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
    "/System/Library/Fonts/Supplemental/Courier New.ttf", "/System/Library/Fonts/Supplemental/Verdana.ttf",
    "/System/Library/Fonts/Supplemental/Georgia.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]
COMPANIES = ["Northwind Technologies", "Helios Systems", "BlueOrbit Solutions", "Kestrel Labs", "Aurora Retail", "Vertex Analytics"]
ROLES = ["Software Engineer", "Data Analyst", "Marketing Intern", "HR Executive", "Business Analyst", "Content Writer"]
CANVAS = (800, 1000)

GENUINE_BODY = [
    "We are pleased to offer you the position of {role} at {co}.",
    "Your annual compensation will be Rs {sal} as per the attached annexure.",
    "Please report to our office on {date} with original documents for verification.",
    "This offer is subject to successful background verification.",
    "Kindly sign and return a copy of this letter within 7 days.",
    "For queries write to careers@{dom}.com",
]
SCAM_BODY = [
    "CONGRATULATIONS! You are SELECTED as {role} at {co} - no interview required.",
    "Monthly salary Rs {sal}. Joining IMMEDIATELY, offer valid for 24 hours only!",
    "Pay refundable registration fee Rs {fee} via UPI to confirm your seat.",
    "Send Aadhaar and PAN card copy to complete verification.",
    "Contact HR on WhatsApp only: +91 9{ph}",
    "Email: {dom}hr{n}@gmail.com",
]


def available_fonts() -> list[str]:
    """Paths of TrueType fonts present on this machine."""
    return [f for f in FONT_CANDIDATES if Path(f).exists()]


def _font(rng: random.Random, size: int, fonts: list[str]) -> ImageFont.ImageFont:
    return ImageFont.truetype(rng.choice(fonts), size) if fonts else ImageFont.load_default(size)


def render_letter(lines: list[str], rng: random.Random, np_rng: np.random.Generator,
                  header: str, noise_sigma: float = 6.0) -> Image.Image:
    """Render lines on a paper-like canvas with varied font/layout, rotation and noise."""
    fonts = available_fonts()
    bg = rng.randint(235, 255)
    img = Image.new("L", CANVAS, bg)
    draw = ImageDraw.Draw(img)
    x0, y = rng.randint(50, 90), rng.randint(50, 90)
    draw.text((x0, y), header, fill=rng.randint(0, 60), font=_font(rng, rng.randint(30, 38), fonts))
    y += 90
    body_font = _font(rng, rng.randint(22, 27), fonts)
    for line in lines:
        words, cur = line.split(), ""
        for w in words:  # naive word wrap
            if draw.textlength(cur + " " + w, font=body_font) > CANVAS[0] - 2 * x0 - 20:
                draw.text((x0, y), cur, fill=rng.randint(0, 50), font=body_font)
                y, cur = y + 38, w
            else:
                cur = (cur + " " + w).strip()
        draw.text((x0, y), cur, fill=rng.randint(0, 50), font=body_font)
        y += rng.randint(50, 70)
    img = img.rotate(rng.uniform(-2.0, 2.0), fillcolor=bg, resample=Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) + np_rng.normal(0, noise_sigma, CANVAS[::-1])
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def make_letter(scam: bool, rng: random.Random, np_rng: np.random.Generator,
                company: str | None = None, role: str | None = None, salary: str | None = None) -> tuple[Image.Image, str]:
    """Build one letter image; returns (image, ground-truth text). ``company``/``role`` override the random pick."""
    co, role = company or rng.choice(COMPANIES), role or rng.choice(ROLES)
    ctx = dict(co=co, role=role, sal=salary or rng.choice(["450,000", "600,000", "35,000"]), fee=rng.choice(["1,500", "2,999"]),
               date="15 July", dom=co.split()[0].lower(), ph=rng.randint(10**8, 10**9 - 1), n=rng.randint(1, 99))
    pool = SCAM_BODY if scam else GENUINE_BODY
    lines = [s.format(**ctx) for s in rng.sample(pool, rng.randint(4, len(pool)))]
    header = f"{co} - OFFER LETTER" if not scam else f"{co.upper()} HIRING!!"
    return render_letter(lines, rng, np_rng, header), header + "\n" + "\n".join(lines)


def generate(n_each: int = 200, seed: int = config.SEED, out_dir: Path | None = None) -> int:
    """Write n_each genuine + n_each scam SYNTHETIC letters to data/images/{genuine,scam}/."""
    out_dir = out_dir or config.IMAGES_DIR
    rng, np_rng = random.Random(seed), np.random.default_rng(seed)
    for label, scam in (("genuine", False), ("scam", True)):
        folder = out_dir / label
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(n_each):
            img, _ = make_letter(scam, rng, np_rng)
            img.save(folder / f"{config.SYNTHETIC_PREFIX}{label}_{i:04d}.png")
    return 2 * n_each


if __name__ == "__main__":
    print(f"SYNTHETIC letters written: {generate()}")
