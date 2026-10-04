"""Tests for the adversarial obfuscation attacks."""
from __future__ import annotations

import pytest

from ml.common import config
from ml.common.red_flags import detect_red_flags
from ml.common.robustness import ATTACKS, apply_attack, attack_suite, perturb_many

SCAM = ("Pay refundable registration fee Rs 1,500 on UPI. Send Aadhaar and PAN card. "
        "Contact HR on WhatsApp only or hr12@gmail.com. Urgent, limited seats!")


def test_every_configured_attack_is_implemented():
    for name in config.ROBUSTNESS_ATTACKS:
        assert name in ATTACKS
    assert [n for n, _ in attack_suite()] == list(config.ROBUSTNESS_ATTACKS)


@pytest.mark.parametrize("name", list(config.ROBUSTNESS_ATTACKS))
def test_attack_changes_the_text_but_keeps_it_readable(name):
    out = apply_attack(name, SCAM)
    assert out != SCAM
    assert len(out) > 0.5 * len(SCAM)  # not truncated into nothing


@pytest.mark.parametrize("name", list(config.ROBUSTNESS_ATTACKS))
def test_attacks_are_deterministic(name):
    assert apply_attack(name, SCAM) == apply_attack(name, SCAM)


def test_unknown_attack_raises():
    with pytest.raises(ValueError, match="unknown attack"):
        apply_attack("not_an_attack", SCAM)


def test_empty_text_survives_every_attack():
    for name in config.ROBUSTNESS_ATTACKS:
        assert apply_attack(name, "") == ""


def test_combined_attack_defeats_the_rule_layer():
    """The headline robustness finding: regex matching alone is evadable."""
    before = {f.code for f in detect_red_flags(SCAM)}
    after = {f.code for f in detect_red_flags(apply_attack("all_combined", SCAM))}
    assert len(before) >= 4
    assert len(after) < len(before)


def test_spacing_breaks_keyword_matches():
    out = apply_attack("spacing", "registration fee")
    assert "fee" not in out or "registration" not in out


def test_homoglyph_introduces_non_ascii_lookalikes():
    out = apply_attack("homoglyph", "urgent fee aadhaar")
    assert any(ord(ch) > 127 for ch in out)


def test_obfuscate_contact_hides_the_email_shape():
    out = apply_attack("obfuscate_contact", "mail hr@gmail.com now")
    assert "@" not in out
    assert "(at)" in out


def test_spell_amounts_removes_the_digits():
    out = apply_attack("spell_amounts", "Pay Rs 1,500 today")
    assert "1,500" not in out
    assert "fifteen hundred" in out


def test_synonym_swap_removes_the_trigger_phrase():
    out = apply_attack("synonym_swap", "A registration fee and no interview, guaranteed.")
    assert "registration fee" not in out.lower()
    assert "no interview" not in out.lower()


def test_perturb_many_matches_length_and_handles_none():
    out = perturb_many("synonym_swap", [SCAM, "", None])
    assert len(out) == 3
    assert out[1] == "" and out[2] == ""


def test_attacks_preserve_genuine_text_without_triggers():
    """Nothing to obfuscate in clean corporate wording -> most attacks are near no-ops."""
    genuine = "We conduct two interview rounds and share the outcome by email."
    out = apply_attack("spell_amounts", genuine)
    assert out == genuine
