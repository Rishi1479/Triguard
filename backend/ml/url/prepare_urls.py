"""Prepare URL datasets (PhishTank-style and Kaggle malicious_phish) into train/val/test CSVs."""
from __future__ import annotations

import logging

import pandas as pd

from ml.common import config
from ml.common.textprep import class_balance, stratified_split
from ml.url.features import canonical_url


def normalize_url(url: str) -> str:
    """Key for dedupe: the lower-cased form features are computed on (no scheme/www/trailing slash)."""
    return canonical_url(url).lower()


def to_standard(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Map a raw frame to (url,label,source). Detects malicious_phish (url,type) or PhishTank (url[,verified])."""
    cols = {c.lower(): c for c in df.columns}
    if "url" not in cols:
        raise ValueError(f"{source}: no 'url' column (found {list(df.columns)})")
    url = df[cols["url"]].astype(str)
    if "type" in cols:  # malicious_phish: benign / defacement / phishing / malware
        label = (df[cols["type"]].astype(str).str.lower() != "benign").astype(int)
    elif "label" in cols:
        label = df[cols["label"]].astype(int)
    else:  # PhishTank: every listed URL is a verified phish
        label = pd.Series(1, index=df.index)
    return pd.DataFrame({"url": url, "label": label, "source": source})


def load_raw() -> pd.DataFrame:
    """Read all CSVs in data/raw/urls/. Raises FileNotFoundError if there are none."""
    files = sorted(config.URL_RAW_DIR.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"no URL CSVs in {config.URL_RAW_DIR}; add e.g. Kaggle malicious_phish.csv")
    frames = [to_standard(pd.read_csv(f), f.stem) for f in files]
    return pd.concat(frames, ignore_index=True)


def run() -> dict[str, pd.DataFrame]:
    """Load -> map -> dedupe (scam wins conflicts) -> stratified split -> data/processed/url_*.csv."""
    df = load_raw()
    n_in = len(df)
    df = df.assign(_k=df["url"].map(normalize_url)).sort_values("label", ascending=False, kind="stable")
    df = df.drop_duplicates("_k").drop(columns="_k").reset_index(drop=True)
    print(f"urls in: {n_in}  duplicates removed: {n_in - len(df)}  balance: {class_balance(df)}")
    splits = dict(zip(("train", "val", "test"), stratified_split(df)))
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for name, part in splits.items():
        part.to_csv(config.PROCESSED_DIR / f"url_{name}.csv", index=False)
        print(f"  {name}: {class_balance(part)}")
    return splits


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
