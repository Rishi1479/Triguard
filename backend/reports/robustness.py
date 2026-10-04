"""Adversarial robustness evaluation -> reports/robustness.md.

Measures how much scam-class recall survives label-preserving obfuscation of the scam texts only
(genuine texts are left untouched, so precision stays interpretable). Reports, per attack:

* rule recall  -- share of scam rows where at least one red flag still fires,
* model recall / F1 / MCC at the analyzer's threshold,
* mean probability drop on the scam rows.

Run: ``python -m reports.robustness``
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ml.common import config
from ml.common.metrics import evaluate
from ml.common.red_flags import detect_red_flags
from ml.common.robustness import attack_suite, perturb_many
from ml.text.analyzer import TextAnalyzer

COLS = ("rule_recall", "recall", "precision", "f1", "mcc", "mean_prob_drop")


def _rule_recall(texts: list[str]) -> float:
    """Share of texts where at least one red flag fires."""
    if not texts:
        return float("nan")
    return float(np.mean([bool(detect_red_flags(t)) for t in texts]))


def evaluate_attack(an: TextAnalyzer, df: pd.DataFrame, name: str | None,
                    clean_scam_prob: np.ndarray | None = None) -> dict:
    """Score the set with the scam rows perturbed by ``name`` (None = clean baseline)."""
    texts = df["text"].tolist()
    y = df["label"].to_numpy()
    scam_idx = np.flatnonzero(y == 1)

    if name is not None:
        perturbed = perturb_many(name, [texts[i] for i in scam_idx])
        texts = list(texts)
        for i, t in zip(scam_idx, perturbed):
            texts[i] = t

    prob = np.asarray(an.model_proba(texts), dtype=float)
    # Rule flags are recomputed on the (possibly perturbed) scam rows only.
    boosted = []
    for t, p in zip(texts, prob):
        from ml.text.analyzer import boost_with_rules
        boosted.append(boost_with_rules(float(p), detect_red_flags(t)))
    boosted_arr = np.asarray(boosted, dtype=float)

    m = evaluate(y, boosted_arr, an.threshold)
    scam_prob = prob[scam_idx]
    row = {
        "attack": name or "none (clean)",
        "rule_recall": _rule_recall([texts[i] for i in scam_idx]),
        "recall": m["recall"],
        "precision": m["precision"],
        "f1": m["f1"],
        "mcc": m["mcc"],
        "mean_prob_drop": 0.0 if clean_scam_prob is None
        else float(np.mean(clean_scam_prob - scam_prob)),
    }
    return {"row": row, "scam_prob": scam_prob}


def run() -> str:
    """Write reports/robustness.md (and robustness.json) and return the markdown."""
    an = TextAnalyzer()
    synth = (config.PROCESSED_DIR / "SYNTHETIC_TEXT.flag").exists()
    frames: dict[str, pd.DataFrame] = {}
    for label, fname in (("EMSCAD test", "text_test.csv"), ("Indian custom", "custom_text_test.csv")):
        path = config.PROCESSED_DIR / fname
        if path.exists():
            df = pd.read_csv(path).dropna(subset=["text"])
            if df["label"].nunique() == 2:
                frames[label] = df

    lines = ["# Adversarial robustness (text)", ""]
    if synth:
        lines += ["> **WARNING: SYNTHETIC DATA.** Re-run after adding the real datasets.", ""]
    lines += [
        "Scam rows are rewritten by label-preserving obfuscations (the kind real scammers use to",
        "dodge keyword filters); genuine rows are untouched. `rule_recall` is the share of scam rows",
        "where at least one red flag still fires, so it isolates the regex layer from the model.", "",
    ]

    payload: dict = {"synthetic": synth, "model": an.model_name, "sets": {}}
    for label, df in frames.items():
        base = evaluate_attack(an, df, None)
        rows = [base["row"]]
        for name, _ in attack_suite():
            rows.append(evaluate_attack(an, df, name, base["scam_prob"])["row"])
        payload["sets"][label] = rows
        lines += [f"## {label} (n={len(df)}, scam={int(df['label'].sum())})", "",
                  "| Attack | " + " | ".join(COLS) + " |", "|---|" + "---|" * len(COLS)]
        for r in rows:
            lines.append("| " + r["attack"] + " | "
                         + " | ".join(f"{r[c]:.3f}" for c in COLS) + " |")
        lines.append("")
        worst = min(rows[1:], key=lambda r: r["rule_recall"]) if len(rows) > 1 else None
        if worst:
            lines += [f"Weakest point: `{worst['attack']}` leaves the rule layer firing on only "
                      f"{worst['rule_recall']:.1%} of scam rows "
                      f"(clean baseline {rows[0]['rule_recall']:.1%}).", ""]

    out = "\n".join(lines) + "\n"
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORTS_DIR / "robustness.md").write_text(out)
    (config.REPORTS_DIR / "robustness.json").write_text(json.dumps(payload, indent=2, default=float))
    return out


if __name__ == "__main__":
    print(run())
