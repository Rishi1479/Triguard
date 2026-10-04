"""Tests for missing-modality imputation, abstention and the Safe-threshold floor."""
from __future__ import annotations

import numpy as np
import pytest

from ml.common import config
from ml.common.schemas import ModalityResult, RedFlag
from ml.fusion.fuser import (
    FEATURE_NAMES,
    Fuser,
    evidence_strength,
    feature_vector,
    should_abstain,
    tune_thresholds,
)


def mr(modality: str, prob: float, available: bool = True, n_chars: int | None = None,
       flags: list[RedFlag] | None = None) -> ModalityResult:
    """Build a ModalityResult for the fuser tests."""
    extracted = {} if n_chars is None else {"n_chars": n_chars}
    return ModalityResult(modality=modality, probability=prob, available=available,
                          extracted=extracted, red_flags=flags or [])


HIGH_FLAG = RedFlag(code="PAYMENT_REQUEST", message="fee", severity="high", evidence="fee")


# ----------------------------------------------------------------- missing-modality imputation
def test_absent_modality_is_imputed_at_the_prior_not_zero():
    results = [mr("text", 0.8, n_chars=500), mr("image", 0.0, available=False)]
    vec = feature_vector(results, [], missing_prior=0.3)
    assert vec[FEATURE_NAMES.index("p_image")] == pytest.approx(0.3)
    assert vec[FEATURE_NAMES.index("has_image")] == 0.0


def test_present_and_clean_is_distinguishable_from_absent():
    """p=0.0 with has=1 (clean) must not look like the absent case (p=prior, has=0)."""
    clean = feature_vector([mr("image", 0.0, n_chars=500)], [], missing_prior=0.4)
    absent = feature_vector([mr("image", 0.0, available=False)], [], missing_prior=0.4)
    i_p, i_has = FEATURE_NAMES.index("p_image"), FEATURE_NAMES.index("has_image")
    assert clean[i_p] == 0.0 and clean[i_has] == 1.0
    assert absent[i_p] == pytest.approx(0.4) and absent[i_has] == 0.0


def test_feature_vector_length_matches_feature_names():
    vec = feature_vector([mr("text", 0.5, n_chars=200)], [])
    assert len(vec) == len(FEATURE_NAMES)


def test_default_missing_prior_comes_from_config():
    vec = feature_vector([mr("text", 0.5, n_chars=200)], [])
    assert vec[FEATURE_NAMES.index("p_url")] == pytest.approx(config.MISSING_MODALITY_PRIOR)


def test_cross_modal_flag_counts_are_split_by_severity():
    flags = [HIGH_FLAG, RedFlag(code="X", message="m", severity="medium", evidence="e")]
    vec = feature_vector([mr("text", 0.5, n_chars=200)], flags)
    assert vec[FEATURE_NAMES.index("n_cross_high")] == 1.0
    assert vec[FEATURE_NAMES.index("n_cross_other")] == 1.0


# ----------------------------------------------------------------- evidence strength
def test_evidence_strength_sums_available_modality_weights():
    s = evidence_strength([mr("text", 0.5, n_chars=500), mr("url", 0.5)])
    assert s == pytest.approx(config.EVIDENCE_WEIGHTS["text"] + config.EVIDENCE_WEIGHTS["url"])


def test_absent_modalities_contribute_nothing():
    assert evidence_strength([mr("text", 0.5, available=False)]) == 0.0


def test_very_short_text_is_discounted():
    short = evidence_strength([mr("text", 0.5, n_chars=5)])
    full = evidence_strength([mr("text", 0.5, n_chars=500)])
    assert short == pytest.approx(full * 0.5)


# ----------------------------------------------------------------- abstention
def test_abstains_on_thin_evidence():
    assert should_abstain(0.9, [mr("image", 0.9, n_chars=3)], []) is True


def test_abstains_inside_the_uncertain_band():
    lo, hi = config.ABSTAIN_BAND
    mid = (lo + hi) / 2
    assert should_abstain(mid, [mr("text", mid, n_chars=500), mr("url", mid)], []) is True


def test_does_not_abstain_outside_the_band():
    lo, hi = config.ABSTAIN_BAND
    strong = [mr("text", 0.95, n_chars=500), mr("url", 0.95)]
    assert should_abstain(hi + 0.05, strong, []) is False
    assert should_abstain(max(lo - 0.05, 0.0), strong, []) is False


def test_a_high_severity_flag_never_abstains():
    """A fee/Aadhaar/OTP demand is decisive evidence, whatever the probability."""
    lo, hi = config.ABSTAIN_BAND
    mid = (lo + hi) / 2
    results = [mr("text", mid, n_chars=500, flags=[HIGH_FLAG])]
    assert should_abstain(mid, results, []) is False


def test_a_high_severity_cross_modal_flag_never_abstains():
    lo, hi = config.ABSTAIN_BAND
    mid = (lo + hi) / 2
    assert should_abstain(mid, [mr("text", mid, n_chars=500)], [HIGH_FLAG]) is False


# ----------------------------------------------------------------- thresholds
def test_t_low_respects_the_configured_floor():
    """A recall target near 1.0 used to drive t_low to ~0, making 'Safe' unreachable."""
    rng = np.random.default_rng(0)
    y = np.array([0] * 80 + [1] * 20)
    p = np.clip(np.where(y == 1, rng.uniform(0.01, 0.9, len(y)), rng.uniform(0.0, 0.3, len(y))), 0, 1)
    t_low, t_high = tune_thresholds(y, p)
    assert t_low >= config.SAFE_THRESHOLD_FLOOR
    assert t_high >= t_low + 0.1


def test_tune_thresholds_falls_back_on_single_class():
    t_low, t_high = tune_thresholds(np.zeros(10, dtype=int), np.full(10, 0.2))
    assert (t_low, t_high) == (config.SAFE_THRESHOLD, config.SCAM_THRESHOLD)


# ----------------------------------------------------------------- decide_full
def test_decide_full_reports_abstention(tmp_path):
    fuser = Fuser(models_dir=tmp_path)  # no artifact -> weighted-average defaults
    prob, label, abstained = fuser.decide_full([mr("image", 0.5, n_chars=2)], [])
    assert label in (config.LABEL_SAFE, config.LABEL_SUSPICIOUS, config.LABEL_SCAM)
    assert abstained is True
    assert 0.0 <= prob <= 1.0


def test_decide_matches_decide_full(tmp_path):
    fuser = Fuser(models_dir=tmp_path)
    results, cross = [mr("text", 0.8, n_chars=500)], []
    assert fuser.decide(results, cross) == fuser.decide_full(results, cross)[:2]


def test_forced_suspicious_still_applies(tmp_path):
    """A payment request must never come back as 'Safe'."""
    fuser = Fuser(models_dir=tmp_path)
    results = [mr("text", 0.01, n_chars=500, flags=[HIGH_FLAG]), mr("url", 0.01)]
    _, label, _ = fuser.decide_full(results, [])
    assert label != config.LABEL_SAFE
