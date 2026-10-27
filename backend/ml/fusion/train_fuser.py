"""Train the fusion meta-classifier on data/aligned/aligned.csv -> models/fusion/fuser.joblib.

Runs the modality analyzers over every aligned case (Tesseract OCR, live checks off, for speed), caches the
per-case outputs in data/processed/aligned_cache.joblib (reused by the ablation study), then:
  train split  -> logistic-regression meta-classifier
  val split    -> Platt calibration + Safe/Scam thresholds (recall-favouring)
  test split   -> reported metrics for weighted-average vs meta
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ml.common import config
from ml.common.bootstrap import bootstrap_metrics
from ml.common.calibration import PlattCalibrator
from ml.common.metrics import evaluate, expected_calibration_error, find_best_threshold
from ml.common.plots import reliability_diagram
from ml.common.textprep import stratified_split
from ml.fusion.fuser import FEATURE_NAMES, FuserArtifact, feature_vector, tune_thresholds, weighted_average

CACHE = config.PROCESSED_DIR / "aligned_cache.joblib"


def load_aligned() -> pd.DataFrame:
    """The real aligned set. Raises FileNotFoundError if data/aligned/aligned.csv is missing."""
    real = config.ALIGNED_DIR / config.ALIGNED_FILE
    if not real.exists():
        raise FileNotFoundError(f"{real} not found; see data/aligned/README.md for the format")
    return pd.read_csv(real).fillna("")


def compute_cache(df: pd.DataFrame, pipeline) -> list[dict]:
    """Run the pipeline's analyzers on each case; returns serialisable per-case outputs."""
    cases = []
    for _, r in df.iterrows():
        img = (config.ROOT_DIR / r["image_path"]).read_bytes() if r["image_path"] else None
        results, cross = pipeline.analyze_modalities(r["text"] or None, img, r["url"] or None, explain=False)
        cases.append({"case_id": int(r["case_id"]), "label": int(r["label"]),
                      "results": [x.model_dump() for x in results], "cross": [f.model_dump() for f in cross]})
    return cases


def build_default_pipeline():
    """Fast pipeline for batch work: Tesseract OCR, no live network checks."""
    from ml.pipeline import TriGuardPipeline
    return TriGuardPipeline(enable_live=False, ocr_backend="tesseract")


def run(pipeline=None, models_dir: Path = config.MODELS_DIR, reports_dir: Path = config.REPORTS_DIR) -> dict:
    """Train/evaluate the fuser and persist artifacts."""
    from ml.common.schemas import ModalityResult, RedFlag

    df = load_aligned()
    pipeline = pipeline or build_default_pipeline()
    cases = compute_cache(df, pipeline)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"cases": cases}, CACHE)

    def unpack(c):
        return [ModalityResult(**r) for r in c["results"]], [RedFlag(**f) for f in c["cross"]]

    y = np.array([c["label"] for c in cases])
    # Absent modalities are imputed at the observed scam prior, not 0.0 (which reads as
    # "confidently genuine"). The prior is stored in the artifact so scoring matches training.
    missing_prior = float(y.mean()) if len(y) else config.MISSING_MODALITY_PRIOR
    X = np.array([feature_vector(*unpack(c), missing_prior) for c in cases])
    avg = np.array([weighted_average(*unpack(c)) for c in cases])
    idx = pd.DataFrame({"i": np.arange(len(y)), "label": y})
    out: dict = {"n_cases": len(y), "feature_names": FEATURE_NAMES,
                 "missing_modality_prior": missing_prior}
    art = FuserArtifact(n_train=0, missing_prior=missing_prior)
    if len(y) >= config.FUSION_MIN_ALIGNED and len(set(y)) == 2:
        tr, va, te = (part["i"].to_numpy() for part in stratified_split(idx))
        meta = LogisticRegression(class_weight="balanced", max_iter=2000, C=1.0, random_state=config.SEED).fit(X[tr], y[tr])
        raw = lambda ix: meta.predict_proba(X[ix])[:, 1]
        meta_cal = PlattCalibrator().fit(raw(va), y[va])
        avg_cal = PlattCalibrator().fit(avg[va], y[va])
        t_low, t_high = tune_thresholds(y[va], meta_cal.predict(raw(va)))
        art = FuserArtifact(meta=meta, meta_cal=meta_cal, avg_cal=avg_cal, t_low=t_low, t_high=t_high,
                            n_train=len(tr), missing_prior=missing_prior)
        avg_thr = find_best_threshold(y[va], avg_cal.predict(avg[va]), "f1")
        for name, p, thr in (("meta", meta_cal.predict(raw(te)), t_high),
                             ("weighted_average", avg_cal.predict(avg[te]), avg_thr)):
            m = evaluate(y[te], p, thr)
            m["ece"] = expected_calibration_error(y[te], p)
            m["ci"] = bootstrap_metrics(y[te], p, thr)  # n is small here: always report the interval
            out[name] = m
        out["thresholds"] = {"t_low": t_low, "t_high": t_high}
        out["coefficients"] = dict(zip(FEATURE_NAMES, map(float, meta.coef_[0])))
        reliability_diagram(y[te], meta_cal.predict(raw(te)), config.FIGURES_DIR / "reliability_fusion_meta.png",
                            "Fusion (meta) reliability")
    else:
        out["note"] = f"fewer than {config.FUSION_MIN_ALIGNED} aligned rows: weighted-average fallback in use"
    folder = models_dir / "fusion"
    folder.mkdir(parents=True, exist_ok=True)
    joblib.dump(art, folder / "fuser.joblib")
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "fusion_results.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    r = run()
    for k in ("weighted_average", "meta"):
        if k in r:
            m = r[k]
            print(f"{k}: F1={m['f1']:.3f} recall={m['recall']:.3f} PR-AUC={m['pr_auc']:.3f} ECE={m['ece']:.3f}")
    print("thresholds:", r.get("thresholds"))
