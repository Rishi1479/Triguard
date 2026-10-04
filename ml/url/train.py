"""Train the LightGBM URL classifier on lexical features -> models/url_lgbm/, reports/url_results.json."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from ml.common import config
from ml.common.calibration import PlattCalibrator
from ml.common.metrics import evaluate, expected_calibration_error, find_best_threshold
from ml.url.features import FEATURE_NAMES, features_matrix


def load(split: str) -> tuple[np.ndarray, np.ndarray]:
    """Feature matrix and labels for a processed URL split."""
    df = pd.read_csv(config.PROCESSED_DIR / f"url_{split}.csv")
    return np.array(features_matrix(df["url"].tolist())), df["label"].to_numpy()


def run(models_dir: Path = config.MODELS_DIR, reports_dir: Path = config.REPORTS_DIR) -> dict:
    """Fit on train (early stopping on val), calibrate + tune threshold on val, report on test."""
    (xtr, ytr), (xva, yva), (xte, yte) = load("train"), load("val"), load("test")
    clf = lgb.LGBMClassifier(**config.LGBM_PARAMS)
    clf.fit(xtr, ytr, eval_set=[(xva, yva)], eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(config.LGBM_EARLY_STOP_ROUNDS, verbose=False)])
    raw_val = clf.predict_proba(xva)[:, 1]
    cal = PlattCalibrator().fit(raw_val, yva)
    thr = find_best_threshold(yva, cal.predict(raw_val), "f1")
    p_test = cal.predict(clf.predict_proba(xte)[:, 1])
    test = evaluate(yte, p_test, thr)
    test["ece"] = expected_calibration_error(yte, p_test)
    folder = models_dir / "url_lgbm"
    folder.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": clf, "calibrator": cal, "threshold": thr, "features": FEATURE_NAMES}, folder / "model.joblib")
    out = {"synthetic": (config.PROCESSED_DIR / "SYNTHETIC_URL.flag").exists(), "threshold": thr,
           "best_iteration": int(clf.best_iteration_ or clf.n_estimators), "test": test,
           "top_features": sorted(zip(FEATURE_NAMES, map(int, clf.feature_importances_)), key=lambda t: -t[1])[:10]}
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "url_results.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    r = run()
    t = r["test"]
    print(f"URL LightGBM test: F1={t['f1']:.3f} recall={t['recall']:.3f} PR-AUC={t['pr_auc']:.3f} "
          f"ROC-AUC={t['roc_auc']:.3f} MCC={t['mcc']:.3f} (synthetic={r['synthetic']})")
