"""Prepare the separate Indian-scam text test set (data/raw/custom_text.csv: text,label,source)."""
from __future__ import annotations

import logging

import pandas as pd

from ml.common import config
from ml.common.privacy import anonymize
from ml.common.synthetic import make_custom_indian
from ml.common.textprep import class_balance, clean_text, dedupe

log = logging.getLogger(__name__)
__all__ = ["anonymize", "write_template", "run"]

TEMPLATE_ROWS = [
    ("Pay Rs 2,500 registration fee on UPI to confirm your internship offer letter. No interview.", 1, "whatsapp_forward"),
    ("Urgent! Send Aadhaar and PAN to hr.jobs99@gmail.com to receive joining letter today.", 1, "telegram"),
    ("Earn Rs 5000 daily from home by liking videos. Join our Telegram group. Share OTP to start.", 1, "sms"),
    ("Infosys is hiring Systems Engineers. Apply on the official careers page. Online test and two interviews.", 0, "company_site"),
    ("Summer internship 2 months, stipend 20,000 per month. Selection via interview. Apply at company website.", 0, "linkedin"),
]


def write_template() -> None:
    """Write data/raw/custom_text_template.csv with 5 example rows."""
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(TEMPLATE_ROWS, columns=["text", "label", "source"]).to_csv(
        config.RAW_DIR / "custom_text_template.csv", index=False)


def run() -> pd.DataFrame:
    """Load, clean, anonymise, dedupe -> data/processed/custom_text_test.csv."""
    write_template()
    real = config.RAW_DIR / config.CUSTOM_TEXT_FILE
    if real.exists():
        df = pd.read_csv(real)
    else:
        synth = config.RAW_DIR / (config.SYNTHETIC_PREFIX + config.CUSTOM_TEXT_FILE)
        log.warning("SYNTHETIC: %s missing; generating %s", real.name, synth.name)
        df = make_custom_indian()
        df.to_csv(synth, index=False)
    df = df.assign(text=df["text"].map(clean_text).map(anonymize), label=df["label"].astype(int))
    df = dedupe(df[df["text"].str.len() >= config.MIN_TEXT_CHARS]).reset_index(drop=True)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.PROCESSED_DIR / "custom_text_test.csv", index=False)
    print("custom Indian test set:", class_balance(df))
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
