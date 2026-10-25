"""Full evaluation: CV, base-paper comparison, ablation, generalisation, calibration, errors -> reports/RESULTS.md.

Run: python -m reports.evaluate_all [--with-bilstm-cv]
Every section is flagged when it was computed on SYNTHETIC data.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from ml.common import config
from ml.common import base_paper
from ml.common.bootstrap import CI_METRICS, bootstrap_metrics, format_ci
from ml.common.calibration import PlattCalibrator
from ml.common.metrics import evaluate, expected_calibration_error, find_best_threshold
from ml.common.plots import confusion_matrix_plot, pr_curve, reliability_diagram
from ml.common.privacy import anonymize
from ml.common.schemas import ModalityResult, RedFlag
from ml.fusion.fuser import feature_vector, weighted_average
from ml.text.analyzer import TextAnalyzer
from ml.text.evaluate import score_set
from ml.text.models import BiLSTMClassifier, TfidfLR
from ml.url.features import features_matrix

FIG = config.FIGURES_DIR
METRICS = ("precision", "recall", "f1", "pr_auc", "roc_auc", "mcc")
ABLATION: dict[str, tuple[set[str], bool]] = {
    "Text only": ({"text"}, False), "URL only": ({"url"}, False), "Image only": ({"image"}, False),
    "Text + URL": ({"text", "url"}, False), "Text + Image": ({"text", "image"}, False),
    "Image + URL": ({"image", "url"}, False), "All three": ({"text", "image", "url"}, False),
    "All three + cross-modal checks": ({"text", "image", "url"}, True),
}


def flag(name: str) -> bool:
    """True if the dataset behind ``name`` ('TEXT'|'URL') is SYNTHETIC."""
    return (config.PROCESSED_DIR / f"SYNTHETIC_{name}.flag").exists()


def md_table(header: list[str], rows: list[list]) -> str:
    """GitHub markdown table."""
    fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
                     + ["| " + " | ".join(fmt(v) for v in r) + " |" for r in rows])


def mean_std(rows: list[dict]) -> dict[str, str]:
    """'mean ± std' per metric."""
    return {m: f"{np.mean([r[m] for r in rows]):.3f} ± {np.std([r[m] for r in rows]):.3f}" for m in METRICS}


# ------------------------------------------------------------------------------- 1. cross-validation
def _fold_metrics(fit_predict, X, y, n_splits: int = 5) -> list[dict]:
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=config.SEED)
    out = []
    for tr, te in skf.split(np.zeros(len(y)), y):
        tr2, va = train_test_split(tr, test_size=config.VAL_RATIO, stratify=y[tr], random_state=config.SEED)
        raw_va, raw_te = fit_predict(tr2, va, te)
        cal = PlattCalibrator().fit(raw_va, y[va])
        thr = find_best_threshold(y[va], cal.predict(raw_va), "f1")
        out.append(evaluate(y[te], cal.predict(raw_te), thr))
    return out


def cv_text(texts: list[str], y: np.ndarray, bilstm: bool) -> dict[str, dict]:
    """5-fold CV for TF-IDF+LR (and optionally the Bi-LSTM reproduction)."""
    def lr(tr, va, te):
        m = TfidfLR().fit([texts[i] for i in tr], y[tr])
        return m.predict_proba([texts[i] for i in va]), m.predict_proba([texts[i] for i in te])

    def bl(tr, va, te):
        m = BiLSTMClassifier().fit([texts[i] for i in tr], y[tr], [texts[i] for i in va], y[va])
        return m.predict_proba([texts[i] for i in va]), m.predict_proba([texts[i] for i in te])

    out = {"TF-IDF + LR": mean_std(_fold_metrics(lr, texts, y))}
    if bilstm:
        out["Bi-LSTM (base-paper reproduction)"] = mean_std(_fold_metrics(bl, texts, y))
    return out


def cv_url(urls: list[str], y: np.ndarray) -> dict[str, dict]:
    """5-fold CV for the LightGBM URL model."""
    import lightgbm as lgb

    X = np.array(features_matrix(urls))

    def gbm(tr, va, te):
        clf = lgb.LGBMClassifier(**config.LGBM_PARAMS).fit(
            X[tr], y[tr], eval_set=[(X[va], y[va])], callbacks=[lgb.early_stopping(config.LGBM_EARLY_STOP_ROUNDS, verbose=False)])
        return clf.predict_proba(X[va])[:, 1], clf.predict_proba(X[te])[:, 1]

    return {"LightGBM (lexical)": mean_std(_fold_metrics(gbm, X, y))}


# ------------------------------------------------------------------------------- 2. base paper
def base_paper_table(splits: dict[str, pd.DataFrame]) -> tuple[str, dict]:
    """The paper's reported figures vs our models; AUROC from probabilities and from hard labels."""
    test = splits["test"]
    y = test["label"].to_numpy()
    rows, extra = [], {}
    for name, st in config.BASE_PAPER_MODELS.items():
        label = f"Pillai (2023) – {name}" + (" ★" if name == "Bi-LSTM" else "")
        rows.append([label, st["f1"], st["recall"], "n/a†", st["auroc"], "n/a"])
    folder = config.MODELS_DIR / "text_bilstm"
    cands = {}
    if (folder / "calib.joblib").exists():
        calib = joblib.load(folder / "calib.joblib")
        bl = BiLSTMClassifier.load(folder)
        cands["Ours: text_bilstm"] = (calib["calibrator"].predict(bl.predict_proba(test["text"].tolist())), calib["threshold"])
    an = TextAnalyzer(use_rules_boost=False)
    if an.model_name not in ("rules_only", "text_bilstm"):
        cands[f"Ours: {an.model_name} (served)"] = (an.model_proba(test["text"].tolist()), an.threshold)
    for name, (p, thr) in cands.items():
        m = evaluate(y, p, thr)
        hard = float(roc_auc_score(y, (p >= thr).astype(int)))
        rows.append([name, m["f1"], m["recall"], m["roc_auc"], hard, m["pr_auc"]])
        extra[name] = {**m, "roc_auc_hard_labels": hard}
    table = md_table(["Model", "F1", "Recall", "ROC-AUC (probabilities)", "ROC-AUC (hard labels)", "PR-AUC"], rows)
    return table, extra


def base_paper_derivation() -> tuple[str, dict]:
    """Show, rather than assert, that the base paper's AUROC comes from hard labels.

    Reconstructs each published confusion matrix and recomputes accuracy and the
    single-operating-point AUC, so a reader can check the claim instead of taking it on trust.
    """
    recs = base_paper.reconcile_all()
    if not recs:
        return "_no published confusion matrix to reconcile._", {}

    rows = [[r.name, f"{r.tp}/{r.tn}/{r.fp}/{r.fn}", r.positives, f"{100 * r.positive_rate:.2f}%",
             f"{r.accuracy:.4f}", f"{r.accuracy_reported:.4f}",
             f"{r.hard_label_auc:.4f}", f"{r.auroc_reported:.2f}",
             "yes" if r.auc_matches_hard_labels else "no"]
            for r in recs]
    table = md_table(["Model", "TP/TN/FP/FN", "positives in test", "positive rate",
                      "accuracy (recomputed)", "accuracy (reported)",
                      "(1+TPR−FPR)/2", "AUROC (reported)", "match"], rows)

    expected = base_paper.expected_fakes_in_test()
    stated, actual = config.BASE_PAPER_DATASET, config.EMSCAD_ACTUAL
    # Quote the benchmark model (the Bi-LSTM) rather than whichever happens to be lowest.
    headline = next((r for r in recs if r.name == "Bi-LSTM"), recs[0])
    lines = [
        table, "",
        "**(a) The reported AUROC is a hard-label AUROC.** The area under an ROC curve drawn through "
        "a single operating point is `(1 + TPR − FPR) / 2`. That expression reproduces the published "
        "AUROC for every model with a confusion matrix, which an AUROC integrated over probability "
        "thresholds would not. Our own AUROCs are therefore reported in both forms above — the "
        "columns are not comparable.", "",
        f"**(b) The paper misstates its own dataset.** It reports {stated['fake']:,} fake postings of "
        f"{stated['total']:,} ({100 * base_paper.dataset_positive_rate():.2f}%); the dataset holds "
        f"{actual['fake']:,} ({100 * base_paper.actual_positive_rate():.2f}%). The reconstructed test "
        f"matrix holds {headline.positives} positives in {config.BASE_PAPER_TEST_N:,} rows — a representative "
        f"split would hold {expected:.0f} — so the split matches the real class balance and the "
        f"{stated['fake']:,} in the paper's text is the error. The paper also does not deduplicate: "
        "EMSCAD contains exact and near-duplicate postings that can sit on both sides of a random split. "
        "This project deduplicates before splitting (`ml/common/textprep.py`).", "",
        f"**(c) Accuracy sits close to the majority-class floor.** For the {headline.name}, at a "
        f"{100 * headline.positive_rate:.2f}% positive rate, always predicting \"genuine\" already "
        f"scores {100 * headline.majority_baseline:.2f}% against its reported "
        f"{100 * headline.accuracy_reported:.2f}% — a margin of "
        f"{100 * (headline.accuracy_reported - headline.majority_baseline):.2f} points. Accuracy is "
        "reported here only as a secondary metric; PR-AUC and MCC are the headline numbers.",
    ]
    disc = base_paper.f1_discrepancies()
    if disc:
        items = "; ".join(f"{n}: reported {rep:.2f}, implied {imp:.2f}" for n, rep, imp in disc)
        lines += ["", f"**(d) An arithmetic slip in the paper's Table 3.** The published F1 does not "
                      f"match the published precision and recall — {items}."]
    out = {r.name: {"tp": r.tp, "tn": r.tn, "fp": r.fp, "fn": r.fn, "positives": r.positives,
                    "positive_rate": r.positive_rate, "accuracy_recomputed": r.accuracy,
                    "hard_label_auc": r.hard_label_auc, "auroc_reported": r.auroc_reported,
                    "majority_baseline": r.majority_baseline,
                    "auc_matches_hard_labels": r.auc_matches_hard_labels} for r in recs}
    out["expected_fakes_if_representative"] = expected
    out["dataset_fake_count"] = {"stated": stated["fake"], "actual": actual["fake"]}
    out["f1_discrepancies"] = disc
    return "\n".join(lines), out


# ------------------------------------------------------------------------------- 3. ablation
def load_cases() -> tuple[list[dict], bool]:
    """Cached per-case modality outputs written by ml.fusion.train_fuser."""
    cache = joblib.load(config.PROCESSED_DIR / "aligned_cache.joblib")
    cases = [{"case_id": c["case_id"], "label": c["label"], "results": [ModalityResult(**r) for r in c["results"]],
              "cross": [RedFlag(**f) for f in c["cross"]]} for c in cache["cases"]]
    return cases, cache["synthetic"]


def ablation(cases: list[dict]) -> tuple[list[list], dict[str, np.ndarray], np.ndarray]:
    """Out-of-fold results for every input configuration + a CV'd meta-classifier. Returns (rows, scores, y)."""
    from sklearn.linear_model import LogisticRegression

    y = np.array([c["label"] for c in cases])
    skf = list(StratifiedKFold(5, shuffle=True, random_state=config.SEED).split(np.zeros(len(y)), y))
    scores: dict[str, np.ndarray] = {}
    preds: dict[str, np.ndarray] = {}
    coverage: dict[str, float] = {}
    for name, (mods, use_cross) in ABLATION.items():
        s, cov = [], []
        for c in cases:
            res = [r for r in c["results"] if r.modality in mods]
            s.append(weighted_average(res, c["cross"] if use_cross else []))
            cov.append(any(r.available for r in res))
        scores[name], coverage[name] = np.array(s), float(np.mean(cov))
        preds[name] = np.zeros(len(y), dtype=int)
        for tr, te in skf:
            thr = find_best_threshold(y[tr], scores[name][tr], "f1")
            preds[name][te] = (scores[name][te] >= thr).astype(int)
    missing_prior = float(y.mean())  # absent modality imputed at the prior, never 0.0
    X = np.array([feature_vector(c["results"], c["cross"], missing_prior) for c in cases])
    name = "Meta-classifier (all three + cross-modal)"
    scores[name], preds[name], coverage[name] = np.zeros(len(y)), np.zeros(len(y), dtype=int), 1.0
    for tr, te in skf:
        tr2, va = train_test_split(tr, test_size=config.VAL_RATIO, stratify=y[tr], random_state=config.SEED)
        m = LogisticRegression(class_weight="balanced", max_iter=2000).fit(X[tr2], y[tr2])
        cal = PlattCalibrator().fit(m.predict_proba(X[va])[:, 1], y[va])
        thr = find_best_threshold(y[va], cal.predict(m.predict_proba(X[va])[:, 1]), "f1")
        scores[name][te] = cal.predict(m.predict_proba(X[te])[:, 1])
        preds[name][te] = (scores[name][te] >= thr).astype(int)
    from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
    rows, ci_rows = [], []
    for n, s in scores.items():
        p = preds[n]
        pr_auc = float(average_precision_score(y, s)) if s.std() > 0 else float("nan")
        roc = float(roc_auc_score(y, s)) if s.std() > 0 else float("nan")
        rows.append([n, f"{100 * coverage[n]:.0f}%", float(precision_score(y, p, zero_division=0)),
                     float(recall_score(y, p, zero_division=0)), float(f1_score(y, p, zero_division=0)), pr_auc, roc])
        # n is only a couple of hundred rows here, so a point estimate alone is not reportable.
        # A degenerate (constant) score has no usable ranking, so the hard predictions are
        # bootstrapped instead and the AUC intervals come back NaN.
        ci = (bootstrap_metrics(y, p.astype(float), 0.5) if s.std() == 0
              else bootstrap_metrics(y, s, find_best_threshold(y, s, "f1")))
        ci_rows.append([n, *(format_ci(ci[m]) for m in CI_METRICS)])
    return rows, scores, y, ci_rows


# ------------------------------------------------------------------------------- 4/5. generalisation, calibration
def generalisation(an: TextAnalyzer, splits: dict[str, pd.DataFrame], custom: pd.DataFrame | None) -> tuple[str, dict]:
    """EMSCAD test vs Indian set at the EMSCAD-tuned threshold."""
    if custom is None:
        return "_Indian custom set not available._", {}
    rows, raw = [], {}
    for set_name, df in (("EMSCAD test", splits["test"]), ("Indian custom", custom)):
        for variant, rules in (("model only", False), ("model + rules", True)):
            _, m = score_set(an, df, rules)
            rows.append([set_name, variant, m["precision"], m["recall"], m["f1"], m["pr_auc"], m["roc_auc"], m["mcc"]])
            raw[f"{set_name} / {variant}"] = m
    return md_table(["Set", "Variant", "Precision", "Recall", "F1", "PR-AUC", "ROC-AUC", "MCC"], rows), raw


def calibration_section(splits: dict[str, pd.DataFrame]) -> tuple[str, dict]:
    """ECE before/after Platt scaling for text and URL on their test sets + reliability diagrams."""
    rows, out = [], {}
    tf = joblib.load(config.MODELS_DIR / "text_tfidf_lr" / "model.joblib") if (config.MODELS_DIR / "text_tfidf_lr" / "model.joblib").exists() else None
    if tf:
        y, raw = splits["test"]["label"].to_numpy(), tf["model"].predict_proba(splits["test"]["text"].tolist())
        cal = tf["calibrator"].predict(raw)
        rows.append(["Text (TF-IDF+LR)", expected_calibration_error(y, raw), expected_calibration_error(y, cal)])
        reliability_diagram(y, cal, FIG / "reliability_text.png", "Text reliability (test)")
        out["text"] = rows[-1][1:]
    up = config.MODELS_DIR / "url_lgbm" / "model.joblib"
    if up.exists():
        blob = joblib.load(up)
        df = pd.read_csv(config.PROCESSED_DIR / "url_test.csv")
        raw = blob["model"].predict_proba(np.array(features_matrix(df["url"].tolist())))[:, 1]
        cal = blob["calibrator"].predict(raw)
        y = df["label"].to_numpy()
        rows.append(["URL (LightGBM)", expected_calibration_error(y, raw), expected_calibration_error(y, cal)])
        reliability_diagram(y, cal, FIG / "reliability_url.png", "URL reliability (test)")
        out["url"] = rows[-1][1:]
    return md_table(["Model", "ECE before calibration", "ECE after Platt scaling"], rows), out


def transfer_section() -> tuple[str, dict]:
    """Naive vs prior-corrected threshold transfer to the Indian set (from text_baselines.json)."""
    f = config.REPORTS_DIR / "text_baselines.json"
    if not f.exists():
        return "_run `python -m ml.text.baselines` first._", {}
    base = json.loads(f.read_text())
    rows, out = [], {}
    for model in ("tfidf_lr", "hybrid_lgbm", "bilstm"):
        dt = base.get(model, {}).get("domain_transfer")
        if not dt:
            continue
        out[model] = dt
        for variant in ("naive", "adapted"):
            m = dt[variant]
            thr = dt["naive_threshold"] if variant == "naive" else dt["adapted_threshold"]
            rows.append([model, variant, f"{thr:.3f}", m["precision"], m["recall"], m["f1"],
                         m["pr_auc"], m["roc_auc"], m["mcc"]])
    if not rows:
        return "_no labelled target-domain set available._", {}
    any_dt = next(iter(out.values()))
    table = md_table(["Model", "Transfer", "Threshold", "Precision", "Recall", "F1", "PR-AUC", "ROC-AUC", "MCC"], rows)
    note = (f"Source scam prior {any_dt['source_prior']:.3f} -> target {any_dt['target_prior']:.3f} "
            f"(logit shift {any_dt['logit_shift']:+.3f}); method `{any_dt['method']}`, "
            f"{any_dt['n_calibration']} target rows used for recalibration, "
            f"metrics reported on the {any_dt['n_holdout']} held-out target rows.")
    return table + "\n\n" + note, out


def robustness_section() -> tuple[str, dict]:
    """Adversarial obfuscation table (from reports/robustness.json)."""
    f = config.REPORTS_DIR / "robustness.json"
    if not f.exists():
        return "_run `python -m reports.robustness`._", {}
    payload = json.loads(f.read_text())
    cols = ("rule_recall", "recall", "precision", "f1", "mcc")
    parts = []
    for set_name, rows in payload["sets"].items():
        parts += [f"**{set_name}**", "",
                  md_table(["Attack", *cols], [[r["attack"], *(r[c] for c in cols)] for r in rows]), ""]
    return "\n".join(parts), payload


# ------------------------------------------------------------------------------- 6. figures
def make_figures(splits: dict[str, pd.DataFrame], scores: dict[str, np.ndarray], y_al: np.ndarray, an: TextAnalyzer) -> list[str]:
    """Confusion matrices and PR curves."""
    made = []
    y = splits["test"]["label"].to_numpy()
    p = an.model_proba(splits["test"]["text"].tolist())
    confusion_matrix_plot(evaluate(y, p, an.threshold)["confusion_matrix"], FIG / "cm_text.png", "Text (test)")
    pr_curve({"Text model": (y, p)}, FIG / "pr_text.png", "Text PR curve (test)")
    up = config.MODELS_DIR / "url_lgbm" / "model.joblib"
    if up.exists():
        blob, df = joblib.load(up), pd.read_csv(config.PROCESSED_DIR / "url_test.csv")
        pu = blob["calibrator"].predict(blob["model"].predict_proba(np.array(features_matrix(df["url"].tolist())))[:, 1])
        confusion_matrix_plot(evaluate(df["label"], pu, blob["threshold"])["confusion_matrix"], FIG / "cm_url.png", "URL (test)")
        pr_curve({"URL model": (df["label"].to_numpy(), pu)}, FIG / "pr_url.png", "URL PR curve (test)")
    best = "Meta-classifier (all three + cross-modal)"
    confusion_matrix_plot(evaluate(y_al, scores[best], find_best_threshold(y_al, scores[best], "f1"))["confusion_matrix"],
                          FIG / "cm_fusion.png", "Fusion (out-of-fold)")
    pr_curve({k: (y_al, v) for k, v in scores.items() if v.std() > 0}, FIG / "pr_ablation.png", "Ablation PR curves (out-of-fold)")
    return sorted(f.name for f in FIG.glob("*.png"))


# ------------------------------------------------------------------------------- 7. error analysis
def _note(is_fn: bool, flags: list[str], near_miss: bool) -> str:
    if near_miss:
        return "Correct but low margin (no hard errors available) – the decision here is close to the threshold."
    if is_fn:
        return ("Missed scam: rule flags fired (" + ", ".join(flags) + ") but the model score stayed low." if flags else
                "Missed scam: no explicit cue (fee, urgency, off-platform contact); reads like an ordinary posting.")
    return ("False alarm: rules fired on wording that can also appear in genuine postings (" + ", ".join(flags) + ")."
            if flags else "False alarm: the model over-weights generic job vocabulary.")


def error_analysis(an: TextAnalyzer, splits: dict[str, pd.DataFrame], custom: pd.DataFrame | None,
                   cases: list[dict], aligned_text: dict[int, str], fusion_scores: np.ndarray, n: int = 10) -> str:
    """Collect the most confident mistakes (padded with near misses) from three evaluation sets."""
    from ml.common.red_flags import detect_red_flags

    pool = []
    def add(set_name, ids, texts, y, p, thr, flags_of):
        for i, t, yy, pp in zip(ids, texts, y, p):
            wrong = int(pp >= thr) != int(yy)
            pool.append({"set": set_name, "id": i, "text": t, "y": int(yy), "p": float(pp), "thr": thr, "wrong": wrong,
                         "margin": abs(pp - thr), "flags": flags_of(t)})
    flags_text = lambda t: [f.code for f in detect_red_flags(t)]
    for name, df in (("EMSCAD test", splits["test"]), ("Indian custom", custom)):
        if df is None:
            continue
        p, _ = score_set(an, df, True)
        add(name, df.index, df["text"], df["label"], p, an.threshold, flags_text)
    thr = find_best_threshold([c["label"] for c in cases], fusion_scores, "f1")
    ft = {c["case_id"]: [f.code for r in c["results"] if r.available for f in r.red_flags] + [f.code for f in c["cross"]] for c in cases}
    add("Aligned (fusion, out-of-fold)", [c["case_id"] for c in cases], [aligned_text.get(c["case_id"], "") or "(no text; image/URL only)" for c in cases],
        [c["label"] for c in cases], fusion_scores, thr, lambda t: [])
    for e in pool:
        if e["set"].startswith("Aligned"):
            e["flags"] = sorted(set(ft[e["id"]]))[:4]
    wrong = sorted([e for e in pool if e["wrong"]], key=lambda e: -abs(e["p"] - e["y"]))
    chosen, per_set = [], {}
    for e in wrong:  # round-robin over sets so one set does not dominate
        if per_set.get(e["set"], 0) < 4 and len(chosen) < n:
            chosen.append(e); per_set[e["set"]] = per_set.get(e["set"], 0) + 1
    near = sorted([e for e in pool if not e["wrong"]], key=lambda e: e["margin"])
    near_ids = {id(e) for e in near[: max(0, n - len(chosen))]}
    chosen += near[: max(0, n - len(chosen))]
    lines = ["# Error analysis", ""]
    if flag("TEXT") or flag("URL"):
        lines += ["> **SYNTHETIC data:** examples below are generated text; the failure modes shown are artefacts of the generator, not real-world findings.", ""]
    lines += [f"{sum(not id(e) in near_ids for e in chosen)} real error(s) and {len(near_ids)} near miss(es) shown "
              f"(total errors available: {len(wrong)}).", ""]
    for k, e in enumerate(chosen, 1):
        is_fn = e["y"] == 1
        nm = id(e) in near_ids
        text = re.sub(r"\s+", " ", anonymize(str(e["text"])))[:260]
        kind = "near miss" if nm else ("false negative" if is_fn else "false positive")
        lines += [f"## {k}. {e['set']} – {kind}", "", f"- True label: **{'scam' if e['y'] else 'genuine'}** · score {e['p']:.2f} (threshold {e['thr']:.2f})",
                  f"- Rule flags: {', '.join(e['flags']) or 'none'}", f"- Text: “{text}”", f"- Likely cause: {_note(is_fn, e['flags'], nm)}", ""]
    return "\n".join(lines)


# ------------------------------------------------------------------------------- driver
def run(with_bilstm_cv: bool = False) -> str:
    """Compute everything, write figures, error_analysis.md, evaluation.json and RESULTS.md."""
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    splits = {n: pd.read_csv(config.PROCESSED_DIR / f"text_{n}.csv") for n in ("train", "val", "test")}
    custom_f = config.PROCESSED_DIR / "custom_text_test.csv"
    custom = pd.read_csv(custom_f) if custom_f.exists() else None
    all_text = pd.concat(splits.values(), ignore_index=True)
    all_url = pd.concat([pd.read_csv(config.PROCESSED_DIR / f"url_{n}.csv") for n in ("train", "val", "test")], ignore_index=True)
    an = TextAnalyzer()
    res: dict = {}

    cv = {**cv_text(all_text["text"].tolist(), all_text["label"].to_numpy(), with_bilstm_cv),
          **cv_url(all_url["url"].tolist(), all_url["label"].to_numpy())}
    res["cv"] = cv
    paper_tbl, res["base_paper"] = base_paper_table(splits)
    deriv_tbl, res["base_paper_derivation"] = base_paper_derivation()
    cases, aligned_synth = load_cases()
    abl_rows, scores, y_al, abl_ci_rows = ablation(cases)
    gen_tbl, res["generalisation"] = generalisation(an, splits, custom)
    tr_tbl, res["domain_transfer"] = transfer_section()
    rob_tbl, res["robustness"] = robustness_section()
    cal_tbl, res["calibration"] = calibration_section(splits)
    figs = make_figures(splits, scores, y_al, an)
    aligned_df = pd.read_csv(next(iter([config.ALIGNED_DIR / config.ALIGNED_FILE] if (config.ALIGNED_DIR / config.ALIGNED_FILE).exists()
                                      else [config.ALIGNED_DIR / f"{config.SYNTHETIC_PREFIX}aligned.csv"]))).fillna("")
    err_md = error_analysis(an, splits, custom, cases, dict(zip(aligned_df["case_id"], aligned_df["text"])),
                            scores["Meta-classifier (all three + cross-modal)"])
    (config.REPORTS_DIR / "error_analysis.md").write_text(err_md)
    res["ablation"] = [dict(zip(["config", "coverage", "precision", "recall", "f1", "pr_auc", "roc_auc"], r)) for r in abl_rows]
    (config.REPORTS_DIR / "evaluation.json").write_text(json.dumps(res, indent=2, default=str))

    # The Indian set writes no flag file; its rows carry source=SYNTHETIC instead.
    custom_synth = custom is not None and bool((custom["source"] == "SYNTHETIC").any())
    syn = {"text": flag("TEXT"), "url": flag("URL"), "aligned": aligned_synth, "indian": custom_synth}
    ocr = (config.REPORTS_DIR / "ocr_compare.md")
    md = ["# TriGuard – results", ""]
    if any(syn.values()):
        md += ["> **WARNING – SYNTHETIC DATA.** Sections marked 🧪 were computed on generated stand-in data "
               f"(text: {syn['text']}, Indian set: {syn['indian']}, URL: {syn['url']}, aligned set: {syn['aligned']}). Do not quote them as scientific results; "
               "rerun `python -m reports.evaluate_all` after adding the real datasets (see PROGRESS.md → WAITING FOR USER).", ""]
    tag = lambda s: " 🧪 SYNTHETIC" if s else ""
    cv_rows = [[k, *[v[m] for m in METRICS]] for k, v in cv.items()]
    md += [f"## 1. 5-fold stratified cross-validation (mean ± std){tag(syn['text'] or syn['url'])}", "",
           md_table(["Model", *METRICS], cv_rows), "",
           "Thresholds and Platt calibration are fitted inside each fold on a 15% validation split; vectorizers/tokenizers fit on the training part only.", "",
           f"## 2. Comparison with the base paper (EMSCAD test split){tag(syn['text'])}", "", paper_tbl, "",
           "★ = the model this project is benchmarked against. † the paper publishes no "
           "probability-based AUROC; its AUROC column is reconciled in section 2b below, and for our "
           "models both forms are shown so like is compared with like. `text_bilstm` reads text only; the "
           "paper's Bi-LSTM also takes EMSCAD's structured form fields (company logo, employment type, "
           "industry …), which TriGuard's inputs never carry.", "",
           "### 2b. Reconciling the base paper's reported figures", "", deriv_tbl, "",
           "Computed by `ml/common/base_paper.py` from the figures in the paper's Tables 2–3 and "
           "Figures 5–6, and covered by `tests/test_base_paper.py` — the derivation can be re-run, "
           "not just read.", "",
           f"## 3. Ablation on the aligned multimodal set (out-of-fold, n={len(cases)}){tag(syn['aligned'])}", "",
           md_table(["Configuration", "Coverage", "Precision", "Recall", "F1", "PR-AUC", "ROC-AUC"], abl_rows), "",
           "Coverage = share of cases where the configuration has at least one input. Thresholds are tuned on each fold's training part. ![PR curves](figures/pr_ablation.png)", "",
           f"## 4. Generalisation: EMSCAD-trained text model → Indian set{tag(syn['text'] or syn['indian'])}", "", gen_tbl, "",
           "Threshold fixed from EMSCAD validation; a gap between ROC/PR-AUC (ranking) and F1 (threshold) indicates calibration shift rather than lost signal.", "",
           f"### 4b. Fixing the transfer: prior correction + target recalibration{tag(syn['text'] or syn['indian'])}", "", tr_tbl, "",
           "`naive` carries the EMSCAD threshold over unchanged — the MCC of ~0 means the model labels *every* "
           "target row as scam, so ranking is perfect but the decision is worthless. `adapted` shifts the logit by "
           "the prior log-odds difference and re-fits Platt scaling on half the target rows, then scores the other "
           "half. See `ml/common/prior_shift.py`.", ""]
    md += ["## 5. OCR comparison (character error rate)", "",
           "> **Do not quote the backend ranking from this table.** The images are rendered from fonts, so OCR "
           "reads back the exact glyphs that were drawn; real offer letters arrive as recompressed phone photos "
           "with skew and glare, where CER is typically 10-30x worse and the EasyOCR/Tesseract ordering commonly "
           "inverts. Re-run on real annotated images before drawing any conclusion.", "", ocr.read_text().split("\n", 2)[2] if ocr.exists() else "_run `python -m ml.image.compare_ocr`_", "",
           f"## 6. Calibration (ECE){tag(syn['text'] or syn['url'])}", "", cal_tbl, "",
           "![text](figures/reliability_text.png) ![url](figures/reliability_url.png) ![fusion](figures/reliability_fusion_meta.png)", "",
           "## 7. Confusion matrices & PR curves", "", " ".join(f"![{f}](figures/{f})" for f in figs if f.startswith(("cm_", "pr_"))), "",
           "## 8. Error analysis", "", "See [error_analysis.md](error_analysis.md) (10 examples).", "",
           f"## 9. Adversarial robustness{tag(syn['text'] or syn['indian'])}", "", rob_tbl,
           "Scam rows only are rewritten by label-preserving obfuscations (spacing, homoglyphs, contact "
           "obfuscation, spelled-out amounts, euphemisms). `rule_recall` isolates the regex layer: where it "
           "collapses, keyword matching alone would have missed the scam. Full table: "
           "[robustness.md](robustness.md).", "",
           f"## 10. Ablation with 95% bootstrap confidence intervals{tag(syn['aligned'])}", "",
           md_table(["Configuration", *CI_METRICS], abl_ci_rows), "",
           f"Stratified percentile bootstrap, {config.BOOTSTRAP_SAMPLES} resamples. With n={len(cases)} "
           "the intervals are wide enough that most configurations in section 3 are not separable; quote "
           "intervals, not point estimates.", ""]
    out = "\n".join(md)
    (config.REPORTS_DIR / "RESULTS.md").write_text(out)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-bilstm-cv", action="store_true", help="also cross-validate the Bi-LSTM (slow on CPU)")
    print(run(ap.parse_args().with_bilstm_cv))
