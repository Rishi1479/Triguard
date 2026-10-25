"""Evaluate the TextAnalyzer on EMSCAD test and the custom Indian set -> reports/text_results.md."""
from __future__ import annotations

import json

import pandas as pd

from ml.common import config
from ml.common.metrics import evaluate, expected_calibration_error
from ml.text.analyzer import TextAnalyzer, boost_with_rules
from ml.common.red_flags import detect_red_flags

COLS = ("precision", "recall", "f1", "pr_auc", "roc_auc", "mcc", "accuracy")


def score_set(an: TextAnalyzer, df: pd.DataFrame, rules: bool) -> tuple[pd.Series, dict]:
    """Probabilities (optionally with rule boost) and metrics at the analyzer threshold."""
    p_model = an.model_proba(df["text"].tolist())
    p = [boost_with_rules(float(x), detect_red_flags(t)) if rules else float(x) for x, t in zip(p_model, df["text"])]
    m = evaluate(df["label"], p, an.threshold)
    m["ece"] = expected_calibration_error(df["label"], p)
    return pd.Series(p, index=df.index), m


def fmt(m: dict) -> str:
    return " | ".join(f"{m[c]:.3f}" for c in (*COLS, "ece"))


def run() -> str:
    """Write reports/text_results.md and return its text."""
    an = TextAnalyzer()
    synth = (config.PROCESSED_DIR / "SYNTHETIC_TEXT.flag").exists()
    base = json.loads((config.REPORTS_DIR / "text_baselines.json").read_text())
    lines = ["# Text module results", ""]
    if synth:
        lines += ["> **WARNING: SYNTHETIC DATA.** These numbers come from generated postings, not EMSCAD. "
                  "They only prove the pipeline runs; do not quote them as results.", ""]
    lines += [f"Analyzer model in use: `{an.model_name}` (threshold {an.threshold:.3f}, tuned on validation).", "",
              "| Set | Variant | " + " | ".join(COLS) + " | ECE |", "|---|---|" + "---|" * (len(COLS) + 1)]
    for name, path in (("EMSCAD test", "text_test.csv"), ("Indian custom", "custom_text_test.csv")):
        f = config.PROCESSED_DIR / path
        if not f.exists():
            continue
        df = pd.read_csv(f)
        if "source" in df.columns and (df["source"] == "SYNTHETIC").any():
            name += " (SYNTHETIC)"
        for variant, rules in (("model only", False), ("model + rule boost", True)):
            lines.append(f"| {name} | {variant} | {fmt(score_set(an, df, rules)[1])} |")
    lines += ["", "## Baselines (from reports/text_baselines.json, EMSCAD test)", "",
              "| Model | " + " | ".join(COLS) + " | ECE |", "|---|" + "---|" * (len(COLS) + 1)]
    for k in ("hybrid_lgbm", "tfidf_lr", "bilstm"):
        if "test" in base.get(k, {}):
            lines.append(f"| {k} | {fmt(base[k]['test'])} |")
    bp = config.BASE_PAPER
    lines += ["", "## Comparison with the base paper (Pillai 2023, Bi-LSTM, EMSCAD)", "",
              "| Model | F1 | Recall |", "|---|---|---|", f"| Pillai (reported) | {bp['f1']:.2f} | {bp['recall']:.2f} |"]
    for k in ("bilstm", "hybrid_lgbm", "tfidf_lr"):
        if "test" in base.get(k, {}):
            t = base[k]["test"]
            lines.append(f"| ours: {k} | {t['f1']:.3f} | {t['recall']:.3f} |")
    lines += ["", "Note: the base paper's AUROC is a hard-label AUROC — `(1+TPR-FPR)/2` reproduces it from its "
              "own confusion matrices (see RESULTS.md section 2b). All AUCs here use probabilities."]
    out = "\n".join(lines) + "\n"
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORTS_DIR / "text_results.md").write_text(out)
    return out


if __name__ == "__main__":
    print(run())
