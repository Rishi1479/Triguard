"""Evaluation metrics for the scam (positive) class, computed from probabilities."""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ml.common import config

_THRESHOLD_METRICS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "f1": lambda y, p: f1_score(y, p, zero_division=0),
    "precision": lambda y, p: precision_score(y, p, zero_division=0),
    "recall": lambda y, p: recall_score(y, p, zero_division=0),
    "mcc": lambda y, p: matthews_corrcoef(y, p),
}


def _validate(y_true: Any, y_prob: Any) -> tuple[np.ndarray, np.ndarray]:
    """Convert inputs to arrays and check shape and value ranges."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_prob, dtype=float)
    if y.ndim != 1 or y.shape != p.shape:
        raise ValueError("y_true and y_prob must be 1-D arrays of equal length")
    if y.size == 0:
        raise ValueError("inputs must not be empty")
    if not set(np.unique(y)) <= {0, 1}:
        raise ValueError("y_true must contain only 0/1 labels")
    if np.isnan(p).any() or p.min() < 0.0 or p.max() > 1.0:
        raise ValueError("y_prob must be probabilities in [0, 1]")
    return y, p


def evaluate(
    y_true: Any, y_prob: Any, threshold: float = config.DEFAULT_CLASS_THRESHOLD
) -> dict[str, Any]:
    """Compute scam-class metrics. PR-AUC and ROC-AUC use probabilities; the rest use ``threshold``.

    AUCs are NaN when only one class is present in ``y_true``.
    """
    y, p = _validate(y_true, y_prob)
    y_hat = (p >= threshold).astype(int)
    single_class = len(np.unique(y)) < 2
    cm = confusion_matrix(y, y_hat, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in cm.ravel())
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y, y_hat, zero_division=0)),
        "recall": float(recall_score(y, y_hat, zero_division=0)),
        "f1": float(f1_score(y, y_hat, zero_division=0)),
        "pr_auc": float("nan") if single_class else float(average_precision_score(y, p)),
        "roc_auc": float("nan") if single_class else float(roc_auc_score(y, p)),
        "mcc": float(matthews_corrcoef(y, y_hat)),
        "accuracy": float(accuracy_score(y, y_hat)),
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def find_best_threshold(y_true: Any, y_prob: Any, metric: str = "f1") -> float:
    """Return the threshold in (0, 1) maximising ``metric`` (f1, precision, recall or mcc).

    Ties go to the threshold closest to the default (0.5). Tune on validation data only.
    """
    if metric not in _THRESHOLD_METRICS:
        raise ValueError(f"metric must be one of {sorted(_THRESHOLD_METRICS)}, got {metric!r}")
    y, p = _validate(y_true, y_prob)
    score_fn = _THRESHOLD_METRICS[metric]
    grid = np.linspace(0.0, 1.0, config.THRESHOLD_GRID_SIZE + 2)[1:-1]
    scores = np.array([score_fn(y, (p >= t).astype(int)) for t in grid])
    best = np.flatnonzero(scores >= scores.max() - 1e-12)
    closest = best[np.argmin(np.abs(grid[best] - config.DEFAULT_CLASS_THRESHOLD))]
    return float(grid[closest])
