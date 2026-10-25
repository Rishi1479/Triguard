"""Tests for the base-paper reconciliation.

These pin the claims made in reports/RESULTS.md section 2b against the figures printed in
Pillai (2023). If the recorded figures in config are ever edited, these fail rather than letting the
report quietly assert something the numbers no longer support.
"""
from __future__ import annotations

import pytest

from ml.common import config
from ml.common.base_paper import (
    actual_positive_rate,
    dataset_positive_rate,
    expected_fakes_in_test,
    f1_discrepancies,
    f1_from,
    hard_label_auc,
    reconcile_all,
    reconstruct,
)


def test_f1_from_matches_the_harmonic_mean():
    assert f1_from(0.89, 0.83) == pytest.approx(0.859, abs=5e-4)
    assert f1_from(1.0, 1.0) == pytest.approx(1.0)
    assert f1_from(0.0, 0.0) == 0.0  # no division by zero


def test_hard_label_auc_of_a_perfect_classifier_is_one():
    assert hard_label_auc(tp=10, tn=10, fp=0, fn=0) == pytest.approx(1.0)


def test_hard_label_auc_of_a_coin_flip_is_one_half():
    assert hard_label_auc(tp=5, tn=5, fp=5, fn=5) == pytest.approx(0.5)


# ----------------------------------------------------------------- confusion-matrix reconstruction
def test_bilstm_matrix_reconstructs_to_the_published_accuracy():
    r = reconstruct("Bi-LSTM")
    assert (r.tp, r.tn, r.fp, r.fn) == (144, 3386, 18, 29)
    assert r.tp + r.tn == 3530  # "correctly identified labels for 3530 records"
    assert r.accuracy == pytest.approx(r.accuracy_reported, abs=5e-4)


def test_lightgbm_matrix_reconstructs_to_the_published_accuracy():
    r = reconstruct("LightGBM")
    assert (r.tp, r.tn) == (115, 3396)
    assert r.tp + r.tn == 3511  # "correctly identified labels for 3511 records"
    assert r.accuracy == pytest.approx(r.accuracy_reported, abs=5e-4)


def test_models_without_a_published_matrix_return_none():
    assert reconstruct("Random Forest") is None
    assert reconstruct("Gradient Boosting") is None


# ----------------------------------------------------------------- claim (a): hard-label AUROC
@pytest.mark.parametrize("name", ["Bi-LSTM", "LightGBM"])
def test_published_auroc_is_a_single_operating_point_auc(name):
    """The headline methodological claim: their AUROC came from 0/1 predictions."""
    r = reconstruct(name)
    assert r.hard_label_auc == pytest.approx(r.auroc_reported, abs=0.005)
    assert r.auc_matches_hard_labels
    assert any("hard-label AUROC" in n for n in r.notes)


def test_every_reconciled_model_supports_the_claim():
    recs = reconcile_all()
    assert len(recs) == 2
    assert all(r.auc_matches_hard_labels for r in recs)


# ----------------------------------------------------------------- claim (b): paper misstates the dataset
def test_paper_states_a_positive_rate_the_dataset_does_not_have():
    assert dataset_positive_rate() == pytest.approx(0.0915, abs=5e-4)  # 1,636 / 17,880 as printed
    assert actual_positive_rate() == pytest.approx(0.0484, abs=5e-4)  # 866 / 17,880 as counted


def test_test_split_matches_the_real_class_balance():
    """173 positives in 3,576 rows is what a representative split of the real dataset holds."""
    r = reconstruct("Bi-LSTM")
    assert expected_fakes_in_test() == pytest.approx(173.2, abs=0.1)
    assert abs(r.positives - expected_fakes_in_test()) < 2
    assert r.positive_rate == pytest.approx(actual_positive_rate(), abs=5e-4)


def test_expected_fakes_under_the_papers_stated_counts():
    assert expected_fakes_in_test(counts=config.BASE_PAPER_DATASET) == pytest.approx(327, abs=1)


def test_actual_counts_match_the_dataset_when_present():
    for name in config.EMSCAD_FILES:
        f = config.RAW_DIR / name
        if f.exists():
            import pandas as pd
            from ml.text.prepare_emscad import to_binary
            y = to_binary(pd.read_csv(f, usecols=["fraudulent"])["fraudulent"])
            assert {"total": len(y), "genuine": int((y == 0).sum()), "fake": int(y.sum())} == config.EMSCAD_ACTUAL
            return
    pytest.skip("real EMSCAD CSV not present")


# ----------------------------------------------------------------- claim (c): accuracy floor
def test_accuracy_is_close_to_the_majority_class_floor():
    r = reconstruct("Bi-LSTM")
    assert r.majority_baseline == pytest.approx(0.9519, abs=5e-4)
    # The headline 98.71% is under four points above doing nothing.
    assert r.accuracy_reported - r.majority_baseline < 0.04


# ----------------------------------------------------------------- claim (d): Table 3 arithmetic
def test_gradient_boosting_f1_disagrees_with_its_precision_and_recall():
    disc = dict((n, (rep, imp)) for n, rep, imp in f1_discrepancies())
    assert "Gradient Boosting" in disc
    reported, implied = disc["Gradient Boosting"]
    assert reported == 0.62
    assert implied == pytest.approx(0.517, abs=5e-3)


def test_the_other_models_f1_values_are_self_consistent():
    flagged = {n for n, _, _ in f1_discrepancies()}
    assert flagged == {"Gradient Boosting"}


# ----------------------------------------------------------------- config wiring
def test_base_paper_alias_points_at_the_bilstm():
    assert config.BASE_PAPER is config.BASE_PAPER_MODELS["Bi-LSTM"]
    assert config.BASE_PAPER["f1"] == 0.86
    assert config.BASE_PAPER["recall"] == 0.83


def test_dataset_counts_are_internally_consistent():
    for d in (config.BASE_PAPER_DATASET, config.EMSCAD_ACTUAL):
        assert d["genuine"] + d["fake"] == d["total"] == 17_880
