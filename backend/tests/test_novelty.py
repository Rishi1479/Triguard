"""Tests for the out-of-distribution (novelty) guard."""
from __future__ import annotations

import numpy as np
import pytest

from ml.common import config
from ml.common.novelty import NoveltyDetector, shrink_toward_prior, tokenize

# A small "training corpus" in one register: job postings.
TRAIN = [
    "We are hiring a data analyst to join our growing team in Pune with hybrid flexibility.",
    "We are hiring a software engineer to join our growing team in Chennai with hybrid flexibility.",
    "Responsibilities include planning, execution and reporting of key deliverables each quarter.",
    "Candidates will go through two interview rounds followed by a technical assessment.",
    "Competitive salary as per industry standards and a performance based annual bonus.",
    "This is a full time position based in our Bengaluru office reporting to the department head.",
] * 30  # enough rows to clear NOVELTY_MIN_TRAIN_ROWS

IN_DIST = "We are hiring a business analyst to join our team in Pune with hybrid flexibility."
OUT_OF_DIST = "Heat the oil, add cumin seeds and chopped onions, then simmer the tomatoes with rice."


@pytest.fixture
def detector() -> NoveltyDetector:
    return NoveltyDetector().fit(TRAIN)


def test_tokenize_lowercases_and_keeps_alphanumerics():
    assert tokenize("Pay Rs 1,500 NOW!") == ["pay", "rs", "1", "500", "now"]


def test_fit_builds_a_vocabulary_and_enables_the_detector(detector):
    assert detector.enabled
    assert "hiring" in detector.vocab
    assert detector.n_train == len(TRAIN)


def test_stays_disabled_below_the_minimum_training_rows():
    d = NoveltyDetector().fit(["one posting only"])
    assert not d.enabled
    assert d.novelty([OUT_OF_DIST]).tolist() == [0.0]  # no opinion rather than a wrong one


def test_min_doc_freq_excludes_hapax_words():
    d = NoveltyDetector(min_doc_freq=2).fit([*TRAIN, "a wholly unprecedented zygomorphic utterance"])
    assert "zygomorphic" not in d.vocab  # appeared in a single document


def test_in_distribution_text_scores_low(detector):
    assert detector.novelty([IN_DIST])[0] < config.NOVELTY_THRESHOLD


def test_out_of_distribution_text_scores_high(detector):
    assert detector.novelty([OUT_OF_DIST])[0] > config.NOVELTY_THRESHOLD


def test_separation_between_registers(detector):
    """The whole point: the two must be separable, not merely ordered."""
    assert detector.novelty([OUT_OF_DIST])[0] > detector.novelty([IN_DIST])[0] + 0.2


def test_another_language_is_maximally_novel(detector):
    german = "Wir freuen uns, Ihnen die Stelle als Kundenberater anzubieten."
    assert detector.novelty([german])[0] > 0.8


def test_training_text_itself_is_not_novel(detector):
    assert detector.novelty([TRAIN[0]])[0] == pytest.approx(0.0)


def test_empty_text_is_treated_as_fully_novel(detector):
    assert detector.novelty([""])[0] == 1.0
    assert detector.novelty([None])[0] == 1.0


def test_novelty_is_bounded(detector):
    vals = detector.novelty([IN_DIST, OUT_OF_DIST, "", "zzz qqq xxx"])
    assert np.all((vals >= 0.0) & (vals <= 1.0))


def test_is_novel_mask_matches_threshold(detector):
    mask = detector.is_novel([IN_DIST, OUT_OF_DIST])
    assert mask.tolist() == [False, True]


def test_save_and_load_round_trip(detector, tmp_path):
    detector.save(tmp_path / "text_novelty")
    loaded = NoveltyDetector.load(tmp_path / "text_novelty")
    assert loaded is not None
    assert loaded.vocab == detector.vocab
    assert loaded.novelty([OUT_OF_DIST])[0] == pytest.approx(detector.novelty([OUT_OF_DIST])[0])


def test_load_returns_none_when_absent(tmp_path):
    assert NoveltyDetector.load(tmp_path / "nothing_here") is None


def test_load_returns_none_for_a_disabled_artifact(tmp_path):
    NoveltyDetector().fit(["too few rows"]).save(tmp_path / "text_novelty")
    assert NoveltyDetector.load(tmp_path / "text_novelty") is None


# ----------------------------------------------------------------- shrinkage
def test_no_shrinkage_below_the_threshold():
    assert shrink_toward_prior(0.9, novelty=0.1, prior=0.1, threshold=0.35) == pytest.approx(0.9)


def test_full_shrinkage_at_maximum_novelty():
    assert shrink_toward_prior(0.9, novelty=1.0, prior=0.1, threshold=0.35) == pytest.approx(0.1)


def test_partial_shrinkage_is_monotone():
    vals = [shrink_toward_prior(0.9, n, prior=0.1, threshold=0.35) for n in (0.4, 0.6, 0.8, 1.0)]
    assert vals == sorted(vals, reverse=True)
    assert all(0.1 <= v <= 0.9 for v in vals)


def test_shrinkage_pulls_low_scores_up_toward_the_prior():
    """Shrinkage is toward the prior, in whichever direction that lies."""
    assert shrink_toward_prior(0.0, novelty=1.0, prior=0.3, threshold=0.35) == pytest.approx(0.3)
