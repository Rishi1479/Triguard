"""Probability calibration (Platt scaling) fitted on validation data only."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from ml.common import config


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), config.PROB_EPS, 1 - config.PROB_EPS)
    return np.log(p / (1 - p)).reshape(-1, 1)


class PlattCalibrator:
    """Sigmoid calibration on the logit of a raw score. Identity until ``fit`` sees both classes."""

    def __init__(self) -> None:
        self._lr: LogisticRegression | None = None

    def fit(self, raw_prob: np.ndarray, y: np.ndarray) -> "PlattCalibrator":
        """Fit on validation scores. A single-class ``y`` leaves the calibrator as identity."""
        y = np.asarray(y).astype(int)
        if len(np.unique(y)) == 2:
            self._lr = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(raw_prob), y)
        return self

    def predict(self, raw_prob: np.ndarray | float) -> np.ndarray:
        """Calibrated P(scam) for one or many raw probabilities."""
        arr = np.atleast_1d(np.asarray(raw_prob, dtype=float))
        if self._lr is None:
            return arr
        return self._lr.predict_proba(_logit(arr))[:, 1]
