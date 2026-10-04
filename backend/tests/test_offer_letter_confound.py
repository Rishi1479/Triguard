"""Regression tests for the offer-letter confound.

A real genuine offer letter was scored 0.998 P(scam) because offer/selection wording existed only
in the scam class of every training set: the models had learned "offer letter => scam" and were
classifying the document *type* rather than its legitimacy. These tests pin the data property that
caused it, so the confound cannot silently return.
"""
from __future__ import annotations

import pandas as pd
import pytest

from ml.common.synthetic import make_custom_indian, make_emscad_like

# Wording that belongs to the offer-letter register rather than to job postings.
OFFER_REGISTER = ("pleased to offer", "offer letter", "selected", "congratulations",
                  "offer of employment")

GENUINE_OFFER = (
    "We are pleased to offer you the Relationship Manager position at Wealth Bank, starting "
    "10th May. Your skills and experience will be a valuable asset to our team. In this full-time "
    "role you will develop client relationships and report to Samuel Boateng, Senior Branch "
    "Manager. Your annual compensation will be USD 72,000, along with benefits per our policies."
)
FAKE_OFFER = (
    "Congratulations! You are selected as Relationship Manager at Wealth Bank. No interview "
    "required. Pay a refundable registration fee of Rs 1,500 on UPI to confirm your seat. Send "
    "your Aadhaar and PAN. Contact HR on WhatsApp only or wealthbankhr22@gmail.com"
)


@pytest.fixture(scope="module")
def emscad_text() -> tuple[pd.Series, pd.Series]:
    """(text, label) for the EMSCAD-shaped synthetic set."""
    df = make_emscad_like()
    text = df["title"].astype(str) + " " + df["description"].astype(str)
    return text, df["fraudulent"]


@pytest.mark.parametrize("phrase", OFFER_REGISTER)
def test_offer_register_appears_in_both_classes_emscad(emscad_text, phrase):
    """Neither class may own this register, or it becomes a label shortcut."""
    text, label = emscad_text
    genuine = text[label == 0].str.contains(phrase, case=False, regex=False).sum()
    scam = text[label == 1].str.contains(phrase, case=False, regex=False).sum()
    assert genuine > 0, f"{phrase!r} never appears in a genuine row"
    assert scam > 0, f"{phrase!r} never appears in a scam row"


@pytest.mark.parametrize("phrase", ["offer", "selected", "congratulations"])
def test_offer_register_appears_in_both_classes_custom(phrase):
    df = make_custom_indian()
    genuine = df[df.label == 0]["text"].str.contains(phrase, case=False, regex=False).sum()
    scam = df[df.label == 1]["text"].str.contains(phrase, case=False, regex=False).sum()
    assert genuine > 0 and scam > 0


def test_genuine_offer_letters_exist_in_the_training_data(emscad_text):
    text, label = emscad_text
    genuine_offers = text[label == 0].str.contains("pleased to offer", case=False, regex=False).sum()
    assert genuine_offers >= 50, "too few genuine offer letters to learn the register"


def test_rules_do_not_fire_on_a_genuine_offer_letter():
    """The red-flag layer must stay silent: nothing here is a scam cue."""
    from ml.common.red_flags import detect_red_flags

    assert detect_red_flags(GENUINE_OFFER) == []


def test_rules_fire_on_a_fake_offer_letter():
    from ml.common.red_flags import detect_red_flags

    codes = {f.code for f in detect_red_flags(FAKE_OFFER)}
    assert "PAYMENT_REQUEST" in codes
    assert "NO_INTERVIEW" in codes


def test_behavioural_features_separate_the_two_offer_letters():
    """Behaviour, not register, is what distinguishes them."""
    from ml.common.behavioral import behavioural_features

    genuine = behavioural_features(GENUINE_OFFER)
    fake = behavioural_features(FAKE_OFFER)
    assert genuine["has_high_flag"] == 0.0
    assert fake["has_high_flag"] == 1.0
    assert fake["flag_severity_sum"] > genuine["flag_severity_sum"]


@pytest.mark.parametrize("text,expect_flags", [(GENUINE_OFFER, False), (FAKE_OFFER, True)])
def test_offer_letters_are_not_novel_to_the_detector(text, expect_flags):
    """Offer letters must be in-distribution; otherwise the guard masks the real fix."""
    from ml.common.novelty import NoveltyDetector

    df = make_emscad_like()
    train = (df["title"].astype(str) + " " + df["description"].astype(str)).tolist()
    det = NoveltyDetector().fit(train)
    assert det.enabled
    assert not det.is_novel([text])[0]
