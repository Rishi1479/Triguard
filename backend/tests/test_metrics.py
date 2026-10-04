import math

import numpy as np
import pytest

from ml.common import config
from ml.common.metrics import evaluate, find_best_threshold


def test_perfect_classifier():
    m = evaluate([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert m["precision"] == m["recall"] == m["f1"] == m["accuracy"] == 1.0
    assert m["pr_auc"] == m["roc_auc"] == m["mcc"] == 1.0
    assert m["confusion_matrix"] == [[2, 0], [0, 2]]


def test_known_confusion_matrix_and_values():
    y = [0, 0, 0, 1, 1, 1]
    p = [0.1, 0.6, 0.2, 0.9, 0.4, 0.8]
    m = evaluate(y, p, threshold=0.5)
    assert m["confusion_matrix"] == [[2, 1], [1, 2]]
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["accuracy"] == pytest.approx(4 / 6)
    assert m["mcc"] == pytest.approx(1 / 3)


def test_aucs_use_probabilities_not_threshold():
    y, p = [0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]
    a, b = evaluate(y, p, 0.2), evaluate(y, p, 0.9)
    assert a["roc_auc"] == b["roc_auc"] == pytest.approx(0.75)
    assert a["pr_auc"] == b["pr_auc"]


def test_no_positive_predictions_is_zero_not_error():
    m = evaluate([0, 1, 1], [0.1, 0.2, 0.3])
    assert m["precision"] == 0.0 and m["recall"] == 0.0 and m["f1"] == 0.0


def test_single_class_gives_nan_aucs():
    m = evaluate([1, 1, 1], [0.6, 0.7, 0.9])
    assert math.isnan(m["roc_auc"]) and math.isnan(m["pr_auc"])


@pytest.mark.parametrize(
    "y,p",
    [([0, 1], [0.5]), ([0, 2], [0.1, 0.9]), ([0, 1], [0.1, 1.5]), ([], []), ([0, 1], [np.nan, 0.5])],
)
def test_invalid_inputs(y, p):
    with pytest.raises(ValueError):
        evaluate(y, p)


def test_find_best_threshold_separable():
    y = [0, 0, 0, 1, 1, 1]
    p = [0.05, 0.1, 0.2, 0.6, 0.7, 0.9]
    t = find_best_threshold(y, p, "f1")
    assert evaluate(y, p, t)["f1"] == 1.0
    assert 0.2 < t <= 0.6


def test_find_best_threshold_beats_default_on_skewed_scores():
    rng = np.random.default_rng(config.SEED)
    y = np.r_[np.zeros(200), np.ones(50)].astype(int)
    p = np.r_[rng.uniform(0, 0.3, 200), rng.uniform(0.2, 0.45, 50)]
    t = find_best_threshold(y, p, "f1")
    assert evaluate(y, p, t)["f1"] > evaluate(y, p, 0.5)["f1"]


@pytest.mark.parametrize("metric", ["precision", "recall", "mcc"])
def test_other_metrics_supported(metric):
    t = find_best_threshold([0, 0, 1, 1], [0.1, 0.3, 0.6, 0.9], metric)
    assert 0.0 < t < 1.0


def test_unknown_metric():
    with pytest.raises(ValueError):
        find_best_threshold([0, 1], [0.1, 0.9], "auc")
