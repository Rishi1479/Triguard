"""Tests for the behavioural (rule-derived) feature extractor."""
from __future__ import annotations

import numpy as np
import pytest

from ml.common import config
from ml.common.behavioral import FEATURE_NAMES, behavioural_features, feature_matrix

SCAM = ("Congratulations! You are selected. No interview required. Pay refundable registration fee "
        "Rs 1,500 on UPI. Send your Aadhaar and PAN card. Contact HR on WhatsApp only or hr99@gmail.com. "
        "Urgent, limited seats!")
GENUINE = ("Northwind Technologies is hiring a Data Analyst in Pune. Two interview rounds followed by an "
           "HR discussion. Competitive salary as per industry standards. Apply via careers@northwind.com.")


def test_every_declared_feature_is_produced():
    feats = behavioural_features(SCAM)
    assert set(feats) == set(FEATURE_NAMES)
    assert all(isinstance(v, float) for v in feats.values())


def test_empty_text_is_all_zeros():
    for text in ("", "   ", None):
        feats = behavioural_features(text)
        assert set(feats) == set(FEATURE_NAMES)
        assert all(v == 0.0 for v in feats.values())


def test_scam_text_fires_the_expected_flags():
    feats = behavioural_features(SCAM)
    assert feats["flag_payment_request"] == 1.0
    assert feats["flag_no_interview"] == 1.0
    assert feats["flag_sensitive_info_request"] == 1.0
    assert feats["flag_messenger_only_contact"] == 1.0
    assert feats["flag_free_email_hr"] == 1.0
    assert feats["has_high_flag"] == 1.0
    assert feats["n_flags"] >= 4


def test_genuine_text_fires_no_high_severity_flag():
    feats = behavioural_features(GENUINE)
    assert feats["has_high_flag"] == 0.0
    assert feats["flag_payment_request"] == 0.0
    assert feats["free_email_share"] == 0.0  # corporate domain


def test_severity_sum_discriminates():
    assert behavioural_features(SCAM)["flag_severity_sum"] > behavioural_features(GENUINE)["flag_severity_sum"]


def test_free_email_share_is_a_proportion():
    mixed = "Write to hr@gmail.com or to careers@acme.com about the hiring process."
    assert behavioural_features(mixed)["free_email_share"] == pytest.approx(0.5)


def test_messenger_link_and_shortener_detection():
    feats = behavioural_features("Apply here https://bit.ly/abc or ping https://wa.me/919999999999")
    assert feats["has_messenger_link"] == 1.0
    assert feats["has_shortener"] == 1.0


def test_money_mentions_counted():
    assert behavioural_features("Pay Rs 500 then ₹1,000 more")["n_money_mentions"] == 2.0


def test_shares_are_bounded_and_length_is_log_scaled():
    feats = behavioural_features(SCAM)
    assert 0.0 <= feats["digit_share"] <= 1.0
    assert 0.0 <= feats["upper_share"] <= 1.0
    assert feats["text_len_log"] == pytest.approx(float(np.log1p(len(SCAM))))


def test_negation_does_not_fire_the_fee_flag():
    """The rules are negation-aware; the features must inherit that."""
    safe = "We never ask for any registration fee from candidates at any stage."
    assert behavioural_features(safe)["flag_payment_request"] == 0.0


def test_feature_matrix_shape_and_order():
    X = feature_matrix([SCAM, GENUINE, "", None])
    assert X.shape == (4, len(FEATURE_NAMES))
    assert X.dtype == float
    # Row order must match input order and column order must match FEATURE_NAMES.
    idx = FEATURE_NAMES.index("flag_payment_request")
    assert X[0, idx] == 1.0 and X[1, idx] == 0.0 and X[2, idx] == 0.0


def test_feature_names_cover_every_configured_code():
    for code in config.BEHAVIOURAL_FEATURE_CODES:
        assert f"flag_{code.lower()}" in FEATURE_NAMES
