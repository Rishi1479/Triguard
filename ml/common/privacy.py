"""Anonymisation helpers: mask phones, e-mails and Aadhaar/PAN-like IDs before storing text."""
from __future__ import annotations

import re

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
AADHAAR_RE = re.compile(r"(?<!\d)\d{4}[\s-]\d{4}[\s-]\d{4}(?!\d)")
PAN_RE = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")
PHONE_RE = re.compile(
    r"(?<![\w.])(?:\+?91[\s-]?)?[6-9]\d{4}[\s-]?\d{5}(?!\d)"  # Indian mobile
    r"|(?<![\w.])\+\d{1,3}[\s-]?\(?\d{2,4}\)?[\s-]?\d{3,4}[\s-]?\d{3,4}(?!\d)"  # intl
)


def anonymize(text: str) -> str:
    """Mask e-mails, Aadhaar-like numbers, PAN-like codes and phone numbers."""
    text = EMAIL_RE.sub("[EMAIL]", text)
    text = AADHAAR_RE.sub("[AADHAAR]", text)
    text = PAN_RE.sub("[PAN]", text)
    return PHONE_RE.sub("[PHONE]", text)
