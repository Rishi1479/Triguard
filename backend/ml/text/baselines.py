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
from ml.common.bootstrap import bootstrap_metrics
from ml.common.calibration import PlattCalibrator
from ml.common.metrics import evaluate, expected_calibration_error, find_best_threshold
from ml.common.prior_shift import calibration_holdout, fit_transfer
from ml.text.models import BiLSTMClassifier, TfidfLR, TfidfRulesLGBM


def load_splits() -> dict[str, pd.DataFrame]:
    """Read the processed text splits (run prepare_emscad first)."""
    out = {n: pd.read_csv(config.PROCESSED_DIR / f"text_{n}.csv") for n in ("train", "val", "test")}
    custom = config.PROCESSED_DIR / "custom_text_test.csv"
    if custom.exists():
        out["custom"] = pd.read_csv(custom)
    return out


def _with_ci(y, p, thr: float) -> dict:
    """Metrics at ``thr`` plus ECE and bootstrap confidence intervals."""
    m = evaluate(y, p, thr)
    m["ece"] = expected_calibration_error(y, p)
    m["ci"] = bootstrap_metrics(y, p, thr)
    return m


def domain_transfer_report(val_prob, val_y, target_prob, target_y) -> dict:
    """Naive threshold transfer vs. prior-corrected transfer on a target-domain set.

    Quantifies the failure in reports/RESULTS.md section 4: a source threshold carried over
    unchanged collapses to "predict everything scam". Corrected metrics are reported on the half of
    the target set that recalibration did **not** see.
    """
    tr = fit_transfer(val_prob, val_y, target_prob, target_y)
    hold = calibration_holdout(target_y)
    y_h = pd.Series(target_y).to_numpy()[hold]
    p_naive = pd.Series(target_prob).to_numpy()[hold]
    naive_thr = find_best_threshold(val_y, val_prob, "f1")
    return {
        "method": tr.method,
        "source_prior": tr.source_prior,
        "target_prior": tr.target_prior,
        "logit_shift": tr.diagnostics.get("logit_shift"),
        "n_calibration": tr.n_calibration,
        "n_holdout": int(len(hold)),
        "naive_threshold": float(naive_thr),
        "adapted_threshold": float(tr.threshold),
        "naive": _with_ci(y_h, p_naive, naive_thr),
        "adapted": _with_ci(y_h, tr.transform(p_naive), tr.threshold),
    }


def report_for(name: str, predict, splits: dict[str, pd.DataFrame]) -> tuple[dict, PlattCalibrator, float]:
    """Calibrate on val, tune the F1 threshold on val, evaluate on the held-out sets."""
    val_raw = predict(splits["val"]["text"].tolist())
    cal = PlattCalibrator().fit(val_raw, splits["val"]["label"])
    val_p = cal.predict(val_raw)
    thr = find_best_threshold(splits["val"]["label"], val_p, "f1")
    res: dict = {"model": name, "threshold": thr}
    for split in ("test", "custom"):
        if split in splits:
            p = cal.predict(predict(splits[split]["text"].tolist()))
            res[split] = _with_ci(splits[split]["label"], p, thr)
    if "custom" in splits:
        cp = cal.predict(predict(splits["custom"]["text"].tolist()))
        res["domain_transfer"] = domain_transfer_report(
            val_p, splits["val"]["label"].to_numpy(), cp, splits["custom"]["label"].to_numpy())
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

    # Hybrid: TF-IDF(SVD) + behavioural rule features -> LightGBM. Rule signals become features the
    # model can weigh, instead of a fixed post-hoc probability boost.
    if len(train) >= config.HYBRID_MIN_TRAIN_ROWS:
        t0 = time.time()
        hy = TfidfRulesLGBM().fit(train["text"].tolist(), train["label"],
                                  splits["val"]["text"].tolist(), splits["val"]["label"])
        res, cal, thr = report_for("hybrid_lgbm", hy.predict_proba, splits)
        res["train_seconds"] = round(time.time() - t0, 1)
        res["behavioural_importance"] = [{"feature": n, "gain": round(g, 2)}
                                         for n, g in hy.behavioural_importance(10)]
        results["hybrid_lgbm"] = res
        folder = models_dir / "text_hybrid_lgbm"
        folder.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": hy, "calibrator": cal, "threshold": thr}, folder / "model.joblib")
    else:
        results["hybrid_lgbm"] = {"skipped": f"fewer than {config.HYBRID_MIN_TRAIN_ROWS} training rows"}

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
    for k in ("tfidf_lr", "hybrid_lgbm", "bilstm"):
        if "test" not in out.get(k, {}):
            continue
        t = out[k]["test"]
        print(f"{k}: test F1={t['f1']:.3f} recall={t['recall']:.3f} PR-AUC={t['pr_auc']:.3f} "
              f"ROC-AUC={t['roc_auc']:.3f} MCC={t['mcc']:.3f}")
        dt = out[k].get("domain_transfer")
        if dt:
            print(f"   target transfer ({dt['method']}): naive F1={dt['naive']['f1']:.3f} MCC={dt['naive']['mcc']:.3f}"
                  f"  ->  adapted F1={dt['adapted']['f1']:.3f} MCC={dt['adapted']['mcc']:.3f}")
