"""Prepare URL datasets (PhishTank-style and Kaggle malicious_phish) into train/val/test CSVs."""
from __future__ import annotations

import logging

import pandas as pd

from ml.common import config
from ml.common.synthetic import make_urls
from ml.common.textprep import class_balance, stratified_split
from ml.url.features import canonical_url

log = logging.getLogger(__name__)


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


def load_raw() -> tuple[pd.DataFrame, bool]:
    """Read all CSVs in data/raw/urls/ (or generate a SYNTHETIC file). Returns (df, synthetic)."""
    config.URL_RAW_DIR.mkdir(parents=True, exist_ok=True)
    files = [f for f in sorted(config.URL_RAW_DIR.glob("*.csv")) if not f.name.startswith(config.SYNTHETIC_PREFIX)]
    synthetic = not files
    if synthetic:
        synth = config.URL_RAW_DIR / f"{config.SYNTHETIC_PREFIX}malicious_phish.csv"
        log.warning("SYNTHETIC: no CSVs in %s; generating %s", config.URL_RAW_DIR, synth.name)
        make_urls().to_csv(synth, index=False)
        files = [synth]
    frames = [to_standard(pd.read_csv(f), f.stem) for f in files]
    return pd.concat(frames, ignore_index=True), synthetic


def run() -> dict[str, pd.DataFrame]:
    """Load -> map -> dedupe (scam wins conflicts) -> stratified split -> data/processed/url_*.csv."""
    df, synthetic = load_raw()
    n_in = len(df)
    df = df.assign(_k=df["url"].map(normalize_url)).sort_values("label", ascending=False, kind="stable")
    df = df.drop_duplicates("_k").drop(columns="_k").reset_index(drop=True)
    print(f"urls in: {n_in}  duplicates removed: {n_in - len(df)}  balance: {class_balance(df)}")
    splits = dict(zip(("train", "val", "test"), stratified_split(df)))
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for name, part in splits.items():
        part.to_csv(config.PROCESSED_DIR / f"url_{name}.csv", index=False)
        print(f"  {name}: {class_balance(part)}")
    flag = config.PROCESSED_DIR / "SYNTHETIC_URL.flag"
    if synthetic:
        flag.write_text("url_*.csv were generated from SYNTHETIC data\n")
    elif flag.exists():
        flag.unlink()
    return splits


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
