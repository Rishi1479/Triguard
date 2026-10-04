"""Tests for cross-domain prior/threshold transfer."""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import matthews_corrcoef, roc_auc_score

from ml.common import config
from ml.common.prior_shift import (
    DomainTransfer,
    calibration_holdout,
    correct_for_prior,
    estimate_prior,
    fit_transfer,
    prior_log_odds,
)


def test_prior_log_odds_symmetry():
    assert prior_log_odds(0.5) == pytest.approx(0.0)
    assert prior_log_odds(0.1) == pytest.approx(-prior_log_odds(0.9))


def test_prior_log_odds_clips_degenerate_values():
    assert np.isfinite(prior_log_odds(0.0)) and np.isfinite(prior_log_odds(1.0))


def test_identical_priors_are_a_no_op():
    p = np.array([0.1, 0.5, 0.9])
    assert correct_for_prior(p, 0.2, 0.2) == pytest.approx(p, abs=1e-9)


def test_raising_target_prior_raises_probabilities():
    p = np.array([0.1, 0.3, 0.7])
    up = correct_for_prior(p, 0.05, 0.5)
    assert np.all(up > p)


def test_correction_is_monotone_so_ranking_is_preserved():
    """The whole point: prior correction moves probabilities, never the ordering (hence AUC)."""
    rng = np.random.default_rng(0)
    y = np.array([0] * 80 + [1] * 20)
    p = np.clip(y * 0.4 + rng.uniform(0.05, 0.5, len(y)), 0.01, 0.99)
    corrected = correct_for_prior(p, 0.2, 0.6)
    assert np.array_equal(np.argsort(p), np.argsort(corrected))
    assert roc_auc_score(y, corrected) == pytest.approx(roc_auc_score(y, p))


def test_estimate_prior():
    assert estimate_prior(np.array([0, 0, 1, 1])) == pytest.approx(0.5)
    assert estimate_prior(np.array([])) == pytest.approx(config.EMSCAD_SCAM_PRIOR)


@pytest.fixture
def shifted_domains():
    """Source: 10% scam. Target: 55% scam, same score distribution -> pure label shift."""
    rng = np.random.default_rng(42)

    def make(n: int, prior: float):
        y = (rng.random(n) < prior).astype(int)
        p = np.clip(np.where(y == 1, rng.normal(0.72, 0.10, n), rng.normal(0.45, 0.10, n)), 0.01, 0.99)
        return p, y

    return make(600, 0.10), make(160, 0.55)


def test_naive_transfer_degenerates_and_adaptation_fixes_it(shifted_domains):
    """Reproduces RESULTS.md section 4: a carried-over threshold collapses to all-positive."""
    (sp, sy), (tp, ty) = shifted_domains
    # A source threshold tuned at a 10% prior sits far below the target's useful cut-point.
    naive_pred = (tp >= 0.5).astype(int)
    naive_mcc = matthews_corrcoef(ty, naive_pred)

    tr = fit_transfer(sp, sy, tp, ty)
    hold = calibration_holdout(ty)
    adapted_mcc = matthews_corrcoef(ty[hold], tr.predict(tp[hold]))

    assert tr.method == "prior_plus_target_platt"
    assert adapted_mcc > naive_mcc
    assert adapted_mcc > 0.3


def test_prior_only_when_target_labels_are_scarce(shifted_domains):
    (sp, sy), (tp, ty) = shifted_domains
    tiny = slice(0, config.MIN_TARGET_CALIBRATION_ROWS - 1)
    tr = fit_transfer(sp, sy, tp[tiny], ty[tiny])
    assert tr.method == "prior_only"
    assert tr.recalibrator is None
    assert 0.0 < tr.threshold < 1.0


def test_prior_only_with_no_target_data_at_all(shifted_domains):
    (sp, sy), _ = shifted_domains
    tr = fit_transfer(sp, sy)
    assert tr.method == "prior_only"
    assert tr.target_prior == pytest.approx(tr.source_prior)


def test_explicit_target_prior_is_honoured(shifted_domains):
    (sp, sy), _ = shifted_domains
    tr = fit_transfer(sp, sy, target_prior=0.4)
    assert tr.target_prior == pytest.approx(0.4)
    assert tr.diagnostics["logit_shift"] > 0  # 0.4 > source 0.10


def test_calibration_holdout_is_disjoint_from_fitting_rows(shifted_domains):
    """Metrics must be reported on rows recalibration never saw."""
    (sp, sy), (tp, ty) = shifted_domains
    tr = fit_transfer(sp, sy, tp, ty)
    hold = calibration_holdout(ty)
    assert len(hold) + tr.n_calibration == len(ty)
    assert len(np.unique(ty[hold])) == 2  # both classes survive the split


def test_holdout_is_everything_when_set_is_too_small():
    ty = np.array([0, 1] * 5)
    assert np.array_equal(calibration_holdout(ty), np.arange(len(ty)))


def test_transform_accepts_scalar_and_array():
    tr = DomainTransfer(source_prior=0.1, target_prior=0.5)
    assert tr.transform(0.3).shape == (1,)
    assert tr.transform([0.3, 0.6]).shape == (2,)


def test_single_class_target_falls_back_to_prior_only(shifted_domains):
    (sp, sy), (tp, _) = shifted_domains
    tr = fit_transfer(sp, sy, tp, np.ones(len(tp), dtype=int))
    assert tr.method == "prior_only"
