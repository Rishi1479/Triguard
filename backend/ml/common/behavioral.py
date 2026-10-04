"""Rule firings and contact-channel structure as named numeric features.

The red-flag rules were previously used only to boost a probability after the fact
(``boost_with_rules``), with hand-set weights. Feeding the same signals to the model as features
lets it learn *when* a flag is actually predictive, and keeps explanations readable: SHAP over
``FEATURE_NAMES`` names a behaviour ("asks for a fee") rather than a character n-gram.

These features are behavioural rather than lexical -- money asked before work, identity documents
requested, contact pushed to a messenger, a free e-mail domain on a corporate-branded posting --
so they transfer across language and geography far better than TF-IDF n-grams.
"""
from __future__ import annotations

import re

import numpy as np

from ml.common import config
from ml.common.red_flags import (
    FREE_EMAIL_DOMAINS,
    detect_red_flags,
    extract_entities,
)

_SEVERITY_SCORE = {"high": 1.0, "medium": 0.5, "low": 0.2}
_MESSENGER_LINK = re.compile(r"\b(?:wa\.me|t\.me|chat\.whatsapp\.com|telegram\.me)\b", re.I)
_SHORTENER = re.compile(r"\b(?:bit\.ly|tinyurl\.com|goo\.gl|rb\.gy|cutt\.ly|forms\.gle|is\.gd|t\.co)\b", re.I)
_MONEY = re.compile(r"(?:₹|rs\.?|inr|usd|\$|€|£)\s?\d[\d,]*", re.I)
_DIGITS = re.compile(r"\d")

# Order matters: this is the model's feature layout.
FEATURE_NAMES: tuple[str, ...] = (
    *(f"flag_{c.lower()}" for c in config.BEHAVIOURAL_FEATURE_CODES),
    "n_flags",
    "flag_severity_sum",
    "has_high_flag",
    "n_urls",
    "n_emails",
    "n_phones",
    "n_companies",
    "free_email_share",
    "has_messenger_link",
    "has_shortener",
    "n_money_mentions",
    "digit_share",
    "upper_share",
    "exclaim_count",
    "text_len_log",
)


def _share(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def behavioural_features(text: str) -> dict[str, float]:
    """Named behavioural features for one text. Empty text -> all zeros."""
    feats = {name: 0.0 for name in FEATURE_NAMES}
    if not text or not text.strip():
        return feats

    flags = detect_red_flags(text)
    by_code = {f.code for f in flags}
    for code in config.BEHAVIOURAL_FEATURE_CODES:
        feats[f"flag_{code.lower()}"] = float(code in by_code)
    feats["n_flags"] = float(len(flags))
    feats["flag_severity_sum"] = float(sum(_SEVERITY_SCORE[f.severity] for f in flags))
    feats["has_high_flag"] = float(any(f.severity == "high" for f in flags))

    ents = extract_entities(text)
    emails = ents["emails"]
    feats["n_urls"] = float(len(ents["urls"]))
    feats["n_emails"] = float(len(emails))
    feats["n_phones"] = float(len(ents["phones"]))
    feats["n_companies"] = float(len(ents["companies"]))
    free = sum(e.rsplit("@", 1)[-1].lower() in FREE_EMAIL_DOMAINS for e in emails)
    feats["free_email_share"] = _share(free, len(emails))

    feats["has_messenger_link"] = float(bool(_MESSENGER_LINK.search(text)))
    feats["has_shortener"] = float(bool(_SHORTENER.search(text)))
    feats["n_money_mentions"] = float(len(_MONEY.findall(text)))
    feats["digit_share"] = _share(len(_DIGITS.findall(text)), len(text))
    feats["upper_share"] = _share(sum(c.isupper() for c in text), len(text))
    feats["exclaim_count"] = float(text.count("!"))
    feats["text_len_log"] = float(np.log1p(len(text)))
    return feats


def feature_matrix(texts: list[str]) -> np.ndarray:
    """``(n_texts, len(FEATURE_NAMES))`` behavioural feature matrix."""
    rows = [behavioural_features(t or "") for t in texts]
    return np.array([[r[n] for n in FEATURE_NAMES] for r in rows], dtype=float)
