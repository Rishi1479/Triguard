"""Adversarial obfuscations used to measure how much scam recall survives evasion.

Real scammers rewrite the exact tokens a keyword filter looks for: ``Aadhaar`` becomes
``aadhar no.``, ``@`` becomes ``(at)``, ``Rs 1,500`` becomes ``fifteen hundred rupees``. A model or
rule set evaluated only on clean text overstates its field performance, so every attack here is a
*label-preserving* rewrite: the posting is still a scam, only its surface form changed.

Each attack is a ``str -> str`` function; :func:`apply_attack` dispatches by name and
:func:`attack_suite` yields every configured attack for reporting.
"""
from __future__ import annotations

import random
import re
from typing import Callable, Iterator

from ml.common import config

# Cyrillic/Greek lookalikes for Latin letters -- visually identical, different code points.
_HOMOGLYPHS = {"a": "а", "e": "е", "o": "о", "p": "р", "c": "с", "y": "у", "x": "х", "i": "і"}

_CONTACT_SUBS = (
    (re.compile(r"@", re.I), "(at)"),
    (re.compile(r"\.com\b", re.I), " dot com"),
    (re.compile(r"\bwhatsapp\b", re.I), "W-hats App"),
    (re.compile(r"\btelegram\b", re.I), "Tele gram"),
)

_AMOUNT_WORDS = {
    "999": "nine hundred ninety nine", "1,500": "fifteen hundred", "1500": "fifteen hundred",
    "2,000": "two thousand", "2000": "two thousand", "2,999": "two thousand nine hundred",
    "3,500": "three thousand five hundred", "5000": "five thousand", "50,000": "fifty thousand",
}

_SYNONYMS = (
    (re.compile(r"\bregistration fee\b", re.I), "onboarding contribution"),
    (re.compile(r"\bsecurity deposit\b", re.I), "refundable amount"),
    (re.compile(r"\btraining fee\b", re.I), "course contribution"),
    (re.compile(r"\bno interview\b", re.I), "direct onboarding"),
    (re.compile(r"\bguaranteed\b", re.I), "assured"),
    (re.compile(r"\baadhaar\b", re.I), "aadhar no."),
    (re.compile(r"\bpan card\b", re.I), "pan no."),
    (re.compile(r"\bbank details\b", re.I), "acc info"),
    (re.compile(r"\bOTP\b", re.I), "one time code"),
    (re.compile(r"\burgent\b", re.I), "time-sensitive"),
    (re.compile(r"\blimited seats\b", re.I), "few slots left"),
)

_KEYWORDS = re.compile(
    r"\b(?:fee|fees|deposit|charges|aadhaar|aadhar|pan|otp|whatsapp|telegram|urgent|guaranteed|"
    r"registration|interview|salary|earn)\b", re.I)


def spacing(text: str, rng: random.Random | None = None) -> str:
    """Insert a dot or space inside trigger keywords (``fee`` -> ``f.ee``)."""
    rng = rng or random.Random(config.ROBUSTNESS_SEED)

    def sub(m: re.Match) -> str:
        w = m.group(0)
        if len(w) < 3:
            return w
        i = rng.randrange(1, len(w) - 1)
        return w[:i] + rng.choice([".", " ", "-"]) + w[i:]

    return _KEYWORDS.sub(sub, text)


def homoglyph(text: str, rng: random.Random | None = None) -> str:
    """Swap Latin letters inside trigger keywords for Cyrillic lookalikes."""
    rng = rng or random.Random(config.ROBUSTNESS_SEED)

    def sub(m: re.Match) -> str:
        return "".join(_HOMOGLYPHS.get(ch.lower(), ch) if rng.random() < 0.5 else ch for ch in m.group(0))

    return _KEYWORDS.sub(sub, text)


def obfuscate_contact(text: str, rng: random.Random | None = None) -> str:
    """Break up e-mail addresses and messenger names the way real scam posts do."""
    for pat, repl in _CONTACT_SUBS:
        text = pat.sub(repl, text)
    return text


def spell_amounts(text: str, rng: random.Random | None = None) -> str:
    """Replace numeric money amounts with words, dropping the currency symbol."""
    for num, words in sorted(_AMOUNT_WORDS.items(), key=lambda kv: -len(kv[0])):
        text = re.sub(rf"(?:₹|rs\.?|inr|usd|\$)\s?{re.escape(num)}\b", words + " rupees", text, flags=re.I)
        text = re.sub(rf"\b{re.escape(num)}\b", words, text)
    return text


def synonym_swap(text: str, rng: random.Random | None = None) -> str:
    """Paraphrase the scam cues into euphemisms a keyword list does not contain."""
    for pat, repl in _SYNONYMS:
        text = pat.sub(repl, text)
    return text


def all_combined(text: str, rng: random.Random | None = None) -> str:
    """Worst case: synonyms, then contact obfuscation, then spelled amounts, then spacing."""
    rng = rng or random.Random(config.ROBUSTNESS_SEED)
    for fn in (synonym_swap, obfuscate_contact, spell_amounts, spacing):
        text = fn(text, rng)
    return text


ATTACKS: dict[str, Callable[..., str]] = {
    "spacing": spacing,
    "homoglyph": homoglyph,
    "obfuscate_contact": obfuscate_contact,
    "spell_amounts": spell_amounts,
    "synonym_swap": synonym_swap,
    "all_combined": all_combined,
}


def apply_attack(name: str, text: str, seed: int = config.ROBUSTNESS_SEED) -> str:
    """Apply the named attack to ``text``."""
    if name not in ATTACKS:
        raise ValueError(f"unknown attack {name!r}; expected one of {sorted(ATTACKS)}")
    return ATTACKS[name](text, random.Random(seed))


def perturb_many(name: str, texts: list[str], seed: int = config.ROBUSTNESS_SEED) -> list[str]:
    """Apply the named attack to each text (same RNG stream, so runs are reproducible)."""
    rng = random.Random(seed)
    return [ATTACKS[name](t or "", rng) for t in texts]


def attack_suite() -> Iterator[tuple[str, Callable[..., str]]]:
    """Yield ``(name, fn)`` for every attack in ``config.ROBUSTNESS_ATTACKS``."""
    for name in config.ROBUSTNESS_ATTACKS:
        yield name, ATTACKS[name]
