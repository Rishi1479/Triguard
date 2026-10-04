"""Cross-domain threshold and prior transfer.

Motivation (reports/RESULTS.md section 4): the EMSCAD-trained text model scores ROC-AUC ~1.0 on the
Indian scam set yet lands at precision 0.54 / recall 1.00 / MCC 0.00 -- it labels *everything* scam.
The ranking transfers perfectly; only the decision cut-point does not. Two corrections are applied,
cheapest first:

1. **Prior correction** (no labels needed). Under label shift the class-conditional densities are
   unchanged and only P(y=1) moves, so the correct target posterior is obtained by shifting the
   logit by the difference of the prior log-odds::

       logit_t = logit_s - log(pi_s / (1 - pi_s)) + log(pi_t / (1 - pi_t))

2. **Target recalibration** (needs a small labelled sample). Platt scaling has two parameters, so
   40-100 labelled target rows are enough to re-fit it; this also absorbs mild covariate shift that
   the prior correction alone cannot.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ml.common import config
from ml.common.calibration import PlattCalibrator
from ml.common.metrics import find_best_threshold


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), config.PROB_EPS, 1 - config.PROB_EPS)
    return np.log(p / (1 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def prior_log_odds(prior: float) -> float:
    """log(pi / (1 - pi)), with ``prior`` clipped away from 0 and 1."""
    pi = float(np.clip(prior, config.PROB_EPS, 1 - config.PROB_EPS))
    return float(np.log(pi / (1 - pi)))


def correct_for_prior(p: np.ndarray | float, source_prior: float, target_prior: float) -> np.ndarray:
    """Re-express source-domain probabilities under the target scam base rate.

    Pure label-shift correction: monotone in ``p``, so ranking (ROC-AUC, PR-AUC) is untouched and
    only the probabilities -- hence the usable threshold -- move.
    """
    shift = prior_log_odds(target_prior) - prior_log_odds(source_prior)
    return _sigmoid(_logit(np.atleast_1d(np.asarray(p, dtype=float))) + shift)


def estimate_prior(y: np.ndarray) -> float:
    """Observed scam share of a labelled set."""
    arr = np.asarray(y).astype(int)
    return float(arr.mean()) if arr.size else config.EMSCAD_SCAM_PRIOR


@dataclass
class DomainTransfer:
    """A fitted source->target adaptation: prior correction, optional Platt re-fit, new threshold."""

    source_prior: float
    target_prior: float
    recalibrator: PlattCalibrator | None = None
    threshold: float = config.DEFAULT_CLASS_THRESHOLD
    n_calibration: int = 0
    method: str = "prior_only"
    diagnostics: dict[str, float] = field(default_factory=dict)

    def transform(self, p: np.ndarray | float) -> np.ndarray:
        """Source-domain probabilities -> target-domain calibrated probabilities."""
        out = correct_for_prior(p, self.source_prior, self.target_prior)
        if self.recalibrator is not None:
            out = self.recalibrator.predict(out)
        return np.atleast_1d(np.asarray(out, dtype=float))

    def predict(self, p: np.ndarray | float) -> np.ndarray:
        """Hard target-domain predictions at the adapted threshold."""
        return (self.transform(p) >= self.threshold).astype(int)


def fit_transfer(
    source_prob: np.ndarray,
    source_y: np.ndarray,
    target_prob: np.ndarray | None = None,
    target_y: np.ndarray | None = None,
    target_prior: float | None = None,
    metric: str = "f1",
    seed: int = config.SEED,
) -> DomainTransfer:
    """Fit the cheapest adaptation the available data supports.

    ``source_prob``/``source_y`` are the source validation split (used for the fallback threshold).
    When ``target_prob``/``target_y`` hold at least ``config.MIN_TARGET_CALIBRATION_ROWS`` labelled
    target rows, half are used to re-fit Platt and tune the threshold and half are held out, so the
    caller can still report an honest target-domain score. Otherwise only the prior correction is
    applied and the threshold is re-tuned on prior-corrected source scores.
    """
    s_prior = estimate_prior(source_y)
    have_target = (target_prob is not None and target_y is not None
                   and len(np.asarray(target_y)) >= config.MIN_TARGET_CALIBRATION_ROWS
                   and len(np.unique(np.asarray(target_y))) == 2)

    if target_prior is None:
        target_prior = estimate_prior(np.asarray(target_y)) if have_target else s_prior

    tr = DomainTransfer(source_prior=s_prior, target_prior=float(target_prior))

    if not have_target:
        corrected = correct_for_prior(source_prob, s_prior, tr.target_prior)
        tr.threshold = find_best_threshold(source_y, corrected, metric)
        tr.method = "prior_only"
        tr.diagnostics = {"source_prior": s_prior, "target_prior": tr.target_prior,
                          "logit_shift": prior_log_odds(tr.target_prior) - prior_log_odds(s_prior)}
        return tr

    tp = np.asarray(target_prob, dtype=float)
    ty = np.asarray(target_y).astype(int)
    fit_idx, _ = _stratified_half(ty, seed)
    corrected = correct_for_prior(tp, s_prior, tr.target_prior)
    tr.recalibrator = PlattCalibrator().fit(corrected[fit_idx], ty[fit_idx])
    tr.threshold = find_best_threshold(ty[fit_idx], tr.recalibrator.predict(corrected[fit_idx]), metric)
    tr.n_calibration = int(len(fit_idx))
    tr.method = "prior_plus_target_platt"
    tr.diagnostics = {"source_prior": s_prior, "target_prior": tr.target_prior,
                      "logit_shift": prior_log_odds(tr.target_prior) - prior_log_odds(s_prior),
                      "n_calibration": float(len(fit_idx))}
    return tr


def _stratified_half(y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Split indices into (calibration, holdout) keeping the class balance of ``y``."""
    rng = np.random.default_rng(seed)
    fit_parts, hold_parts = [], []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        k = max(1, int(round(len(idx) * config.TARGET_CALIBRATION_HOLDOUT)))
        k = min(k, len(idx) - 1) if len(idx) > 1 else len(idx)
        fit_parts.append(idx[:k])
        hold_parts.append(idx[k:])
    return np.concatenate(fit_parts), np.concatenate(hold_parts)


def calibration_holdout(target_y: np.ndarray, seed: int = config.SEED) -> np.ndarray:
    """Indices NOT used by :func:`fit_transfer` for recalibration (report metrics on these)."""
    ty = np.asarray(target_y).astype(int)
    if len(ty) < config.MIN_TARGET_CALIBRATION_ROWS or len(np.unique(ty)) < 2:
        return np.arange(len(ty))
    return _stratified_half(ty, seed)[1]
