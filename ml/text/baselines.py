"""Train and evaluate the text baselines: TF-IDF+LR and the base paper's Bi-LSTM reproduction.

Trains on text_train.csv, calibrates + tunes the threshold on text_val.csv, reports on text_test.csv
(and the custom Indian set when present) -> reports/text_baselines.json. Artifacts -> models/.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import pandas as pd

from ml.common import config
from ml.common.calibration import PlattCalibrator
from ml.common.metrics import evaluate, expected_calibration_error, find_best_threshold
from ml.text.models import BiLSTMClassifier, TfidfLR


def load_splits() -> dict[str, pd.DataFrame]:
    """Read the processed text splits (run prepare_emscad first)."""
    out = {n: pd.read_csv(config.PROCESSED_DIR / f"text_{n}.csv") for n in ("train", "val", "test")}
    custom = config.PROCESSED_DIR / "custom_text_test.csv"
    if custom.exists():
        out["custom"] = pd.read_csv(custom)
    return out


def report_for(name: str, predict, splits: dict[str, pd.DataFrame]) -> tuple[dict, PlattCalibrator, float]:
    """Calibrate on val, tune the F1 threshold on val, evaluate on the held-out sets."""
    val_raw = predict(splits["val"]["text"].tolist())
    cal = PlattCalibrator().fit(val_raw, splits["val"]["label"])
    thr = find_best_threshold(splits["val"]["label"], cal.predict(val_raw), "f1")
    res: dict = {"model": name, "threshold": thr}
    for split in ("test", "custom"):
        if split in splits:
            p = cal.predict(predict(splits[split]["text"].tolist()))
            m = evaluate(splits[split]["label"], p, thr)
            m["ece"] = expected_calibration_error(splits[split]["label"], p)
            res[split] = m
    return res, cal, thr


def run(models_dir: Path = config.MODELS_DIR, reports_dir: Path = config.REPORTS_DIR) -> dict:
    """Train both baselines, save artifacts and write the JSON report."""
    splits = load_splits()
    train = splits["train"]
    results: dict = {"synthetic": (config.PROCESSED_DIR / "SYNTHETIC_TEXT.flag").exists()}

    t0 = time.time()
    lr = TfidfLR().fit(train["text"].tolist(), train["label"])
    res, cal, thr = report_for("tfidf_lr", lr.predict_proba, splits)
    res["train_seconds"] = round(time.time() - t0, 1)
    results["tfidf_lr"] = res
    folder = models_dir / "text_tfidf_lr"
    folder.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": lr, "calibrator": cal, "threshold": thr}, folder / "model.joblib")

    t0 = time.time()
    bl = BiLSTMClassifier().fit(train["text"].tolist(), train["label"],
                                splits["val"]["text"].tolist(), splits["val"]["label"])
    res, cal, thr = report_for("bilstm", bl.predict_proba, splits)
    res["train_seconds"] = round(time.time() - t0, 1)
    results["bilstm"] = res
    bl.save(models_dir / "text_bilstm")
    joblib.dump({"calibrator": cal, "threshold": thr}, models_dir / "text_bilstm" / "calib.joblib")

    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "text_baselines.json").write_text(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    out = run()
    for k in ("tfidf_lr", "bilstm"):
        t = out[k]["test"]
        print(f"{k}: test F1={t['f1']:.3f} recall={t['recall']:.3f} PR-AUC={t['pr_auc']:.3f} ROC-AUC={t['roc_auc']:.3f} MCC={t['mcc']:.3f}")
