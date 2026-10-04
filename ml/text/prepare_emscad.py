"""Prepare the EMSCAD / Kaggle fake_job_postings.csv into stratified train/val/test CSVs."""
from __future__ import annotations

import logging

import pandas as pd

from ml.common import config
from ml.common.synthetic import make_emscad_like
from ml.common.textprep import class_balance, clean_text, dedupe, merge_columns, stratified_split

log = logging.getLogger(__name__)


def load_raw() -> tuple[pd.DataFrame, bool]:
    """Load the real CSV, or fall back to a SYNTHETIC one (written to data/raw/). Returns (df, synthetic)."""
    real = config.RAW_DIR / config.EMSCAD_FILE
    if real.exists():
        return pd.read_csv(real), False
    synth = config.RAW_DIR / (config.SYNTHETIC_PREFIX + config.EMSCAD_FILE)
    log.warning("SYNTHETIC: %s missing; generating %s", real.name, synth.name)
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    df = make_emscad_like()
    df.to_csv(synth, index=False)
    return df, True


def build_dataset(raw: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Merge text columns, clean, drop empties and duplicates. Returns (df[text,label], n_removed)."""
    df = pd.DataFrame({"text": merge_columns(raw), "label": raw["fraudulent"].astype(int)})
    df = df[df["text"].str.len() >= config.MIN_TEXT_CHARS].reset_index(drop=True)
    before = len(df)
    df = dedupe(df).reset_index(drop=True)
    return df, before - len(df)


def run() -> dict[str, pd.DataFrame]:
    """Full pipeline: load -> clean -> dedupe -> split -> write data/processed/text_*.csv."""
    raw, synthetic = load_raw()
    df, removed = build_dataset(raw)
    df["source"] = "SYNTHETIC" if synthetic else "EMSCAD"
    print(f"rows in: {len(raw)}  duplicates/empties removed: {len(raw) - len(df)} (dupes: {removed})")
    print("class balance after dedupe:", class_balance(df))
    splits = dict(zip(("train", "val", "test"), stratified_split(df)))
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for name, part in splits.items():
        part.to_csv(config.PROCESSED_DIR / f"text_{name}.csv", index=False)
        print(f"  {name}: {class_balance(part)}")
    flag = config.PROCESSED_DIR / "SYNTHETIC_TEXT.flag"
    if synthetic:
        flag.write_text("text_*.csv were generated from SYNTHETIC data\n")
    elif flag.exists():
        flag.unlink()
    return splits


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
