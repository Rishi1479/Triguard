"""Compare OCR backends/preprocessing by character error rate (CER) -> reports/ocr_compare.md.

Ground truth: data/images/ocr_gt.csv (path,text) of real annotated images.
"""
from __future__ import annotations

import re
import time

import cv2
import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein

from ml.common import config
from ml.image.ocr import EasyOCREngine, TesseractEngine
from ml.image.preprocess import preprocess

def cer(ref: str, hyp: str) -> float:
    """Character error rate after whitespace normalisation."""
    norm = lambda s: re.sub(r"\s+", " ", s).strip()
    ref, hyp = norm(ref), norm(hyp)
    return Levenshtein.distance(ref, hyp) / max(len(ref), 1)


def load_ground_truth() -> pd.DataFrame:
    """Load ocr_gt.csv. Raises FileNotFoundError if it is missing."""
    if not config.OCR_GT_FILE.exists():
        raise FileNotFoundError(f"{config.OCR_GT_FILE} not found; add real images with a path,text CSV")
    return pd.read_csv(config.OCR_GT_FILE)


def run() -> str:
    """Evaluate every available backend x {raw gray, preprocessed, preprocessed+binarised}."""
    gt = load_ground_truth()
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
    lines += [f"Images: {len(gt)}", "", "| Backend | Preprocessing | Mean CER | Median CER | s/image |", "|---|---|---|---|---|"]
    lines += [f"| {b} | {m} | {mean:.3f} | {med:.3f} | {t:.2f} |" for b, m, mean, med, t in rows]
    out = "\n".join(lines) + "\n"
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORTS_DIR / "ocr_compare.md").write_text(out)
    return out


if __name__ == "__main__":
    print(run())
