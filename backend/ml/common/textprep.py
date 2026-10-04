"""Text cleaning, near-duplicate removal and stratified splitting shared by all data prep scripts."""
from __future__ import annotations

import hashlib
import html
import re

import pandas as pd
from sklearn.model_selection import train_test_split

from ml.common import config

_TAG_RE = re.compile(r"<[^>]+>")
_PLACEHOLDER_RE = re.compile(r"#(?:URL|EMAIL|PHONE)_[A-Za-z0-9]+#")
_WS_RE = re.compile(r"\s+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def clean_text(raw: object) -> str:
    """Strip HTML, ``#URL_xxx#`` placeholders and collapse whitespace. NaN -> ''."""
    if not isinstance(raw, str):
        return ""
    text = html.unescape(_TAG_RE.sub(" ", raw))
    text = _PLACEHOLDER_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def merge_columns(df: pd.DataFrame, columns: tuple[str, ...] = config.EMSCAD_TEXT_COLUMNS) -> pd.Series:
    """Join the given columns (cleaned) into one text string per row."""
    parts = [df[c].map(clean_text) for c in columns if c in df.columns]
    merged = pd.concat(parts, axis=1).apply(lambda r: " ".join(p for p in r if p), axis=1)
    return merged.map(lambda s: _WS_RE.sub(" ", s).strip())


def normalized_hash(text: str) -> str:
    """Hash of the lower-cased alphanumeric content: equal for exact and near-exact duplicates."""
    return hashlib.md5(_NON_ALNUM_RE.sub("", text.lower()).encode()).hexdigest()


def dedupe(df: pd.DataFrame, text_col: str = "text", label_col: str = "label") -> pd.DataFrame:
    """Drop duplicates by normalised-text hash. On a label conflict the scam (1) row wins."""
    out = df.assign(_h=df[text_col].map(normalized_hash))
    out = out.sort_values(label_col, ascending=False, kind="stable")
    out = out.drop_duplicates("_h", keep="first").drop(columns="_h")
    return out.sort_index()


def stratified_split(
    df: pd.DataFrame, label_col: str = "label"
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified 70/15/15 split with the project seed."""
    holdout = config.VAL_RATIO + config.TEST_RATIO
    train, rest = train_test_split(
        df, test_size=holdout, stratify=df[label_col], random_state=config.SEED
    )
    test_share = config.TEST_RATIO / holdout
    val, test = train_test_split(
        rest, test_size=test_share, stratify=rest[label_col], random_state=config.SEED
    )
    return train, val, test


def class_balance(df: pd.DataFrame, label_col: str = "label") -> str:
    """Human-readable class balance, e.g. ``n=100 scam=12 (12.0%)``."""
    n, pos = len(df), int(df[label_col].sum())
    return f"n={n} scam={pos} ({100 * pos / max(n, 1):.1f}%)"
