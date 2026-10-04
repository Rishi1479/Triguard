"""Tests for bootstrap confidence intervals."""
from __future__ import annotations

import numpy as np
import pytest

from ml.common import config
from ml.common.bootstrap import CI_METRICS, bootstrap_metrics, ci_table_row, format_ci


@pytest.fixture
def separable() -> tuple[np.ndarray, np.ndarray]:
    """A well-separated set of 100 rows (40 positive)."""
    rng = np.random.default_rng(0)
    y = np.array([0] * 60 + [1] * 40)
    p = np.where(y == 1, rng.uniform(0.6, 1.0, len(y)), rng.uniform(0.0, 0.4, len(y)))
    return y, p


def test_point_estimate_inside_interval(separable):
    y, p = separable
    ci = bootstrap_metrics(y, p, 0.5, n_samples=200)
    for m in CI_METRICS:
        assert ci[m]["lo"] <= ci[m]["value"] <= ci[m]["hi"], m


def test_interval_ordering_and_width(separable):
    y, p = separable
    ci = bootstrap_metrics(y, p, 0.5, n_samples=200)
    for m in CI_METRICS:
        assert ci[m]["lo"] <= ci[m]["hi"]
        assert ci[m]["width"] == pytest.approx(ci[m]["hi"] - ci[m]["lo"])


def test_wider_interval_for_smaller_sample():
    """A 30-row set must give a wider recall interval than a 400-row set with the same signal."""
    rng = np.random.default_rng(1)

    def ci_width(n: int) -> float:
        y = np.array([0] * (n // 2) + [1] * (n - n // 2))
        p = np.clip(y * 0.55 + rng.normal(0, 0.25, n) + 0.2, 0, 1)
        return bootstrap_metrics(y, p, 0.5, n_samples=300, seed=7)["recall"]["width"]

    assert ci_width(30) > ci_width(400)


def test_small_sample_returns_nan_interval():
    y = np.array([0, 1, 0, 1])
    ci = bootstrap_metrics(y, np.array([0.1, 0.9, 0.2, 0.8]), 0.5)
    assert np.isnan(ci["f1"]["lo"]) and np.isnan(ci["f1"]["hi"])
    assert ci["f1"]["value"] == pytest.approx(1.0)


def test_single_class_returns_nan_interval():
    y = np.zeros(50, dtype=int)
    ci = bootstrap_metrics(y, np.full(50, 0.2), 0.5)
    assert np.isnan(ci["precision"]["lo"])


def test_resamples_preserve_class_balance(separable):
    """Stratified resampling must never produce a single-class draw (which would NaN the AUCs)."""
    y, p = separable
    ci = bootstrap_metrics(y, p, 0.5, n_samples=100)
    assert not np.isnan(ci["roc_auc"]["lo"])  # would be NaN if draws collapsed to one class


def test_deterministic_given_seed(separable):
    y, p = separable
    a = bootstrap_metrics(y, p, 0.5, n_samples=100, seed=3)
    b = bootstrap_metrics(y, p, 0.5, n_samples=100, seed=3)
    assert a == b


def test_format_ci_and_table_row(separable):
    y, p = separable
    ci = bootstrap_metrics(y, p, 0.5, n_samples=100)
    text = format_ci(ci["f1"])
    assert "[" in text and "–" in text
    assert format_ci({"value": 0.5, "lo": float("nan"), "hi": float("nan"), "width": float("nan")}) == "0.500"
    row = ci_table_row("Text only", ci)
    assert row[0] == "Text only" and len(row) == 1 + len(CI_METRICS)


def test_alpha_controls_interval_width():
    """A 50% interval must be narrower than a 99% one on the same (noisy) data."""
    assert config.BOOTSTRAP_SAMPLES > 0 and 0 < config.BOOTSTRAP_ALPHA < 1
    rng = np.random.default_rng(11)
    y = np.array([0] * 60 + [1] * 40)
    p = np.clip(y * 0.3 + rng.normal(0.35, 0.3, len(y)), 0, 1)  # overlapping, so recall varies
    wide = bootstrap_metrics(y, p, 0.5, n_samples=400, alpha=0.5, seed=5)["recall"]["width"]
    narrow = bootstrap_metrics(y, p, 0.5, n_samples=400, alpha=0.01, seed=5)["recall"]["width"]
    assert 0 < wide < narrow
