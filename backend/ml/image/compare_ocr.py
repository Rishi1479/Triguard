"""Compare OCR backends/preprocessing by character error rate (CER) -> reports/ocr_compare.md.

Ground truth: data/images/ocr_gt.csv (path,text). If missing, a SYNTHETIC set is generated.
"""
from __future__ import annotations

import random
import re
import time

import cv2
import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein

from ml.common import config
from ml.image import synth_letters
from ml.image.ocr import EasyOCREngine, TesseractEngine
from ml.image.preprocess import preprocess

N_SYNTH = 20


def cer(ref: str, hyp: str) -> float:
    """Character error rate after whitespace normalisation."""
    norm = lambda s: re.sub(r"\s+", " ", s).strip()
    ref, hyp = norm(ref), norm(hyp)
    return Levenshtein.distance(ref, hyp) / max(len(ref), 1)


def ensure_ground_truth() -> tuple[pd.DataFrame, bool]:
    """Load ocr_gt.csv or create SYNTHETIC letters + ground truth. Returns (df, synthetic)."""
    if config.OCR_GT_FILE.exists() and not config.OCR_GT_FILE.name.startswith(config.SYNTHETIC_PREFIX):
        return pd.read_csv(config.OCR_GT_FILE), False
    rng, np_rng = random.Random(config.SEED + 7), np.random.default_rng(config.SEED + 7)
    folder = config.IMAGES_DIR / "ocr"
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(N_SYNTH):
        img, text = synth_letters.make_letter(i % 2 == 1, rng, np_rng)
        path = folder / f"{config.SYNTHETIC_PREFIX}ocr_{i:03d}.png"
        img.save(path)
        rows.append({"path": str(path), "text": text})
    df = pd.DataFrame(rows)
    df.to_csv(config.OCR_GT_FILE, index=False)
    return df, True


def run() -> str:
    """Evaluate every available backend x {raw gray, preprocessed, preprocessed+binarised}."""
    gt, synthetic = ensure_ground_truth()
    engines = []
    for cls in (EasyOCREngine, TesseractEngine):
        try:
            engines.append(cls())
        except Exception as exc:
            print(f"skip {cls.__name__}: {exc}")
    modes = {"raw grayscale": lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), "preprocessed": preprocess,
             "preprocessed + binarised": lambda im: preprocess(im, binarize=True)}
    rows = []
    for eng in engines:
        for mode, fn in modes.items():
            errs, t0 = [], time.time()
            for _, r in gt.iterrows():
                img = cv2.imread(r["path"])
                errs.append(cer(r["text"], eng.read(fn(img))))
            rows.append((eng.name, mode, float(np.mean(errs)), float(np.median(errs)), (time.time() - t0) / len(gt)))
    lines = ["# OCR comparison (character error rate, lower is better)", ""]
    if synthetic:
        lines += ["> **SYNTHETIC images.** Clean rendered letters; real phone photos will be much harder. "
                  "Replace `data/images/ocr_gt.csv` (path,text) with real annotated images and re-run.", ""]
    lines += [f"Images: {len(gt)}", "", "| Backend | Preprocessing | Mean CER | Median CER | s/image |", "|---|---|---|---|---|"]
    lines += [f"| {b} | {m} | {mean:.3f} | {med:.3f} | {t:.2f} |" for b, m, mean, med, t in rows]
    out = "\n".join(lines) + "\n"
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORTS_DIR / "ocr_compare.md").write_text(out)
    return out


if __name__ == "__main__":
    print(run())
