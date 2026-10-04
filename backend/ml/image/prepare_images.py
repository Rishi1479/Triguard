"""Build data/processed/image_manifest.csv (path,label,source,split) from data/images/{scam,genuine}/."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml.common import config
from ml.common.textprep import stratified_split

EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def build_manifest(images_dir: Path | None = None) -> pd.DataFrame:
    """Scan class folders and return a manifest with a stratified split column."""
    images_dir = images_dir or config.IMAGES_DIR
    rows = []
    for label_name, label in (("genuine", 0), ("scam", 1)):
        (images_dir / label_name).mkdir(parents=True, exist_ok=True)
        for p in sorted((images_dir / label_name).iterdir()):
            if p.suffix.lower() in EXTS:
                src = "SYNTHETIC" if p.name.startswith(config.SYNTHETIC_PREFIX) else "real"
                rows.append({"path": str(p.relative_to(config.ROOT_DIR)) if p.is_relative_to(config.ROOT_DIR) else str(p),
                             "label": label, "source": src})
    df = pd.DataFrame(rows, columns=["path", "label", "source"])
    if df.empty or df["label"].nunique() < 2 or df["label"].value_counts().min() < 4:
        return df.assign(split=pd.Series(dtype=str))
    parts = stratified_split(df)
    return pd.concat([p.assign(split=n) for n, p in zip(("train", "val", "test"), parts)]).sort_index()


def run() -> pd.DataFrame:
    """Write the manifest and print counts."""
    df = build_manifest()
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.PROCESSED_DIR / "image_manifest.csv", index=False)
    print(f"manifest rows: {len(df)}")
    if len(df):
        print(df.groupby(["split", "label"]).size().unstack(fill_value=0))
    return df


if __name__ == "__main__":
    run()
