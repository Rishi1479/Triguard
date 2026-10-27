"""Train and evaluate the text models, one at a time or all together.

Models (artifacts -> models/text_<name>/): ``hybrid_lgbm``, ``tfidf_lr``, ``bilstm``. Each model trains on text_train.csv, is calibrated + threshold-tuned on text_val.csv and reported on
text_test.csv (and the custom Indian set when present). Results are merged into
reports/text_baselines.json, so training one model keeps the others' numbers.

    python -m ml.text.baselines                       # everything
    python -m ml.text.baselines --model hybrid_lgbm   # one model
    python -m ml.text.baselines --model bilstm tfidf_lr  # several
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd

from ml.common import config
from ml.common.bootstrap import bootstrap_metrics
from ml.common.calibration import PlattCalibrator
from ml.common.metrics import evaluate, expected_calibration_error, find_best_threshold
from ml.common.novelty import NoveltyDetector
from ml.common.prior_shift import calibration_holdout, fit_transfer
from ml.text.models import BiLSTMClassifier, TfidfLR, TfidfRulesLGBM

Predict = Callable[[pd.DataFrame], np.ndarray]


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


def report_for(name: str, predict: Predict, splits: dict[str, pd.DataFrame]) -> tuple[dict, PlattCalibrator, float]:
    """Calibrate on val, tune the F1 threshold on val, evaluate on the held-out sets.

    ``predict`` maps a split DataFrame to raw P(scam).
    """
    val_raw = predict(splits["val"])
    cal = PlattCalibrator().fit(val_raw, splits["val"]["label"])
    val_p = cal.predict(val_raw)
    thr = find_best_threshold(splits["val"]["label"], val_p, "f1")
    res: dict = {"model": name, "threshold": thr}
    for split in ("test", "custom"):
        if split in splits:
            res[split] = _with_ci(splits[split]["label"], cal.predict(predict(splits[split])), thr)
    if "custom" in splits:
        cp = cal.predict(predict(splits["custom"]))
        res["domain_transfer"] = domain_transfer_report(
            val_p, splits["val"]["label"].to_numpy(), cp, splits["custom"]["label"].to_numpy())
    return res, cal, thr


def _texts(df: pd.DataFrame) -> list[str]:
    return df["text"].tolist()


def train_tfidf_lr(splits: dict[str, pd.DataFrame], models_dir: Path) -> dict:
    """TF-IDF word+char -> balanced LogisticRegression -> models/text_tfidf_lr/."""
    train = splits["train"]
    lr = TfidfLR().fit(_texts(train), train["label"])
    res, cal, thr = report_for("tfidf_lr", lambda df: lr.predict_proba(_texts(df)), splits)
    folder = models_dir / "text_tfidf_lr"
    folder.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": lr, "calibrator": cal, "threshold": thr}, folder / "model.joblib")
    return res


def train_hybrid_lgbm(splits: dict[str, pd.DataFrame], models_dir: Path) -> dict:
    """TF-IDF(SVD) + behavioural rule features -> LightGBM -> models/text_hybrid_lgbm/.

    Rule signals become features the model can weigh, instead of a fixed post-hoc probability boost.
    """
    train = splits["train"]
    if len(train) < config.HYBRID_MIN_TRAIN_ROWS:
        return {"skipped": f"fewer than {config.HYBRID_MIN_TRAIN_ROWS} training rows"}
    hy = TfidfRulesLGBM().fit(_texts(train), train["label"], _texts(splits["val"]), splits["val"]["label"])
    res, cal, thr = report_for("hybrid_lgbm", lambda df: hy.predict_proba(_texts(df)), splits)
    res["behavioural_importance"] = [{"feature": n, "gain": round(g, 2)} for n, g in hy.behavioural_importance(10)]
    folder = models_dir / "text_hybrid_lgbm"
    folder.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": hy, "calibrator": cal, "threshold": thr}, folder / "model.joblib")
    return res


def train_bilstm(splits: dict[str, pd.DataFrame], models_dir: Path) -> dict:
    """Text-only Bi-LSTM (the paper's text branch) -> models/text_bilstm/."""
    train, val = splits["train"], splits["val"]
    bl = BiLSTMClassifier().fit(_texts(train), train["label"], _texts(val), val["label"])
    res, cal, thr = report_for("bilstm", lambda df: bl.predict_proba(_texts(df)), splits)
    bl.save(models_dir / "text_bilstm")
    joblib.dump({"calibrator": cal, "threshold": thr}, models_dir / "text_bilstm" / "calib.joblib")
    return res


MODELS: dict[str, Callable[[dict, Path], dict]] = {
    "hybrid_lgbm": train_hybrid_lgbm, "tfidf_lr": train_tfidf_lr, "bilstm": train_bilstm,
}


def resolve(names: list[str] | None) -> list[str]:
    """Expand ``all`` and drop a ``text_`` prefix; unknown names raise ValueError."""
    out: list[str] = []
    for n in names or ["all"]:
        n = n.removeprefix("text_")
        for g in list(MODELS) if n == "all" else [n]:
            if g not in MODELS:
                raise ValueError(f"unknown model {n!r}; choose from: all, {', '.join(MODELS)}")
            if g not in out:
                out.append(g)
    return out


def _previous(path: Path) -> dict:
    """Earlier results to merge into; legacy results computed on SYNTHETIC data are discarded."""
    if not path.exists():
        return {}
    old = json.loads(path.read_text())
    return {} if old.get("synthetic") else old


def run(models: list[str] | None = None, models_dir: Path = config.MODELS_DIR,
        reports_dir: Path = config.REPORTS_DIR) -> dict:
    """Train ``models`` (default: all), save their artifacts and merge results into text_baselines.json."""
    names = resolve(models)
    splits = load_splits()
    out_file = reports_dir / "text_baselines.json"
    results: dict = _previous(out_file)

    # Novelty guard, fit on the training split only. Lets the analyzer shrink scores toward the
    # prior on registers it has no training evidence for, instead of guessing confidently.
    nov = NoveltyDetector().fit(_texts(splits["train"]))
    if nov.enabled:
        nov.save(models_dir / config.NOVELTY_DIR_SUFFIX)
        results["novelty"] = {"threshold": nov.threshold, "vocab_size": len(nov.vocab),
                              "n_train": nov.n_train}

    for name in names:
        t0 = time.time()
        res = MODELS[name](splits, models_dir)
        res["train_seconds"] = round(time.time() - t0, 1)
        results[name] = res
        print(summary_line(name, res), flush=True)

    reports_dir.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(results, indent=2))
    return results


def summary_line(name: str, res: dict) -> str:
    """One-line test summary for the console."""
    if "test" not in res:
        return f"{name}: {res.get('skipped', 'no test result')}"
    t = res["test"]
    line = (f"{name}: test F1={t['f1']:.3f} recall={t['recall']:.3f} PR-AUC={t['pr_auc']:.3f} "
            f"ROC-AUC={t['roc_auc']:.3f} MCC={t['mcc']:.3f}  ({res['train_seconds']}s)")
    dt = res.get("domain_transfer")
    if dt:
        line += (f"\n   target transfer ({dt['method']}): naive F1={dt['naive']['f1']:.3f} MCC={dt['naive']['mcc']:.3f}"
                 f"  ->  adapted F1={dt['adapted']['f1']:.3f} MCC={dt['adapted']['mcc']:.3f}")
    return line


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train TriGuard's text models, all or only the ones named.")
    ap.add_argument("--model", nargs="+", default=["all"], metavar="NAME",
                    help=f"all | {' | '.join(MODELS)} (default: all)")
    args = ap.parse_args()
    try:
        selected = resolve(args.model)
    except ValueError as exc:
        ap.error(str(exc))
    run(selected)
