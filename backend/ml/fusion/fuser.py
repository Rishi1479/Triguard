"""Fusion: weighted average (fallback) or logistic-regression meta-classifier over modality outputs.

Final calibration and the 3-level label thresholds are tuned on validation data, favouring recall
(the "Safe" cut-off keeps scam recall >= ``RECALL_TARGET_LOW``).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

from ml.common import config
from ml.common.schemas import ModalityResult, RedFlag

log = logging.getLogger(__name__)
MODALITIES = ("text", "image", "url")
FEATURE_NAMES = ["p_text", "p_image", "p_url", "has_text", "has_image", "has_url", "n_cross_high", "n_cross_other"]


def modality_probs(results: list[ModalityResult]) -> dict[str, float | None]:
    """Per-modality P(scam); several URL results are merged conservatively with max. None = unavailable."""
    out: dict[str, float | None] = {m: None for m in MODALITIES}
    for r in results:
        if r.available:
            out[r.modality] = r.probability if out[r.modality] is None else max(out[r.modality], r.probability)
    return out


def feature_vector(results: list[ModalityResult], cross_flags: list[RedFlag],
                   missing_prior: float = config.MISSING_MODALITY_PRIOR) -> list[float]:
    """Meta-classifier features: probabilities, presence flags, cross-modal flag counts.

    An absent modality is imputed at ``missing_prior`` (the training-set scam prior), never 0.0:
    0.0 reads as "confidently genuine" and would pull every case without an image towards Safe.
    The ``has_*`` indicators let the model separate "absent" from "present and clean".
    """
    p = modality_probs(results)
    high = sum(f.severity == "high" for f in cross_flags)
    return [*(missing_prior if p[m] is None else p[m] for m in MODALITIES),
            *(float(p[m] is not None) for m in MODALITIES),
            float(high), float(len(cross_flags) - high)]


def evidence_strength(results: list[ModalityResult]) -> float:
    """Weighted amount of usable evidence behind a verdict (0 = none).

    Used for abstention: a verdict resting only on a weak modality, or on a handful of characters
    of text, is reported as low-confidence rather than presented as a firm label.
    """
    total = 0.0
    for r in results:
        if not r.available:
            continue
        w = config.EVIDENCE_WEIGHTS.get(r.modality, 0.0)
        n_chars = r.extracted.get("n_chars")
        if isinstance(n_chars, int) and n_chars < config.MIN_TEXT_CHARS_FOR_VERDICT:
            w *= 0.5  # a few characters of text/OCR is thin evidence
        total += w
    return float(total)


def should_abstain(prob: float, results: list[ModalityResult], cross_flags: list[RedFlag]) -> bool:
    """True when the inputs do not justify a firm verdict.

    Abstains on thin evidence, or when the fused probability sits inside ``ABSTAIN_BAND`` with no
    high-severity flag to anchor it. A high-severity flag is decisive evidence, so it never abstains.
    """
    if any(f.severity == "high" for r in results if r.available for f in r.red_flags):
        return False
    if any(f.severity == "high" for f in cross_flags):
        return False
    # An input unlike anything in training has no model evidence behind its score.
    if any(r.extracted.get("out_of_distribution") for r in results if r.available):
        return True
    if evidence_strength(results) < config.MIN_EVIDENCE_FOR_VERDICT:
        return True
    lo, hi = config.ABSTAIN_BAND
    return bool(lo <= prob < hi)


def weighted_average(results: list[ModalityResult], cross_flags: list[RedFlag]) -> float:
    """Weighted mean over AVAILABLE modalities, then a noisy-or boost per cross-modal flag."""
    p = {m: v for m, v in modality_probs(results).items() if v is not None}
    if not p:
        return 0.0
    w = {m: config.FUSION_WEIGHTS[m] for m in p}
    avg = sum(w[m] * p[m] for m in p) / sum(w.values())
    keep = 1.0 - avg
    for f in cross_flags:
        keep *= 1.0 - config.CROSS_BOOST[f.severity]
    return float(1.0 - keep)


@dataclass
class FuserArtifact:
    """Everything learnt on aligned data (any field may be None -> that step is skipped)."""

    meta: object | None = None  # LogisticRegression on FEATURE_NAMES
    meta_cal: object | None = None  # Platt for meta output
    avg_cal: object | None = None  # Platt for the weighted average
    t_low: float = config.SAFE_THRESHOLD
    t_high: float = config.SCAM_THRESHOLD
    n_train: int = 0
    missing_prior: float = config.MISSING_MODALITY_PRIOR  # imputation value used at training time


def tune_thresholds(y_val, p_val) -> tuple[float, float]:
    """(t_low, t_high): t_low = largest threshold keeping scam recall >= target; t_high = best-F1 threshold.

    Falls back to config defaults when validation lacks both classes. Guarantees t_high >= t_low + 0.1.
    """
    from ml.common.metrics import find_best_threshold

    y, p = np.asarray(y_val), np.asarray(p_val)
    if len(np.unique(y)) < 2:
        return config.SAFE_THRESHOLD, config.SCAM_THRESHOLD
    pos = np.sort(p[y == 1])
    k = int(np.floor((1 - config.RECALL_TARGET_LOW) * len(pos)))  # scams allowed below t_low
    t_low = float(pos[k]) if k < len(pos) else float(pos[-1])
    # Floor t_low: a recall target near 1.0 otherwise drives it to ~0 and nothing is ever "Safe".
    t_low = min(max(t_low - 1e-9, config.SAFE_THRESHOLD_FLOOR), 0.5)
    t_high = max(find_best_threshold(y, p, "f1"), t_low + 0.1)
    return float(t_low), float(min(t_high, 0.95))


class Fuser:
    """Combine modality results into a calibrated probability and a 3-level label."""

    def __init__(self, models_dir: Path | None = None) -> None:
        folder = (Path(models_dir) / "fusion") if models_dir else config.FUSION_DIR
        self.art = FuserArtifact()
        if (folder / "fuser.joblib").exists():
            try:
                self.art = joblib.load(folder / "fuser.joblib")
            except Exception as exc:
                log.warning("could not load fuser (%s); using defaults", exc)

    @property
    def mode(self) -> str:
        """'meta' when a trained meta-classifier is loaded, else 'weighted_average'."""
        return "meta" if self.art.meta is not None else "weighted_average"

    def probability(self, results: list[ModalityResult], cross_flags: list[RedFlag]) -> float:
        """Calibrated fused P(scam)."""
        if self.art.meta is not None:
            x = np.array([feature_vector(results, cross_flags, self.art.missing_prior)])
            raw = float(self.art.meta.predict_proba(x)[:, 1][0])
            cal = self.art.meta_cal
        else:
            raw, cal = weighted_average(results, cross_flags), self.art.avg_cal
        return float(cal.predict(raw)[0]) if cal is not None else raw

    def label(self, prob: float) -> str:
        """Safe < t_low <= Suspicious < t_high <= Likely Scam."""
        if prob >= self.art.t_high:
            return config.LABEL_SCAM
        return config.LABEL_SAFE if prob < self.art.t_low else config.LABEL_SUSPICIOUS

    def decide(self, results: list[ModalityResult], cross_flags: list[RedFlag]) -> tuple[float, str]:
        """(probability, label) with the hard rule: payment / Aadhaar / OTP requests never yield 'Safe'."""
        prob, label, _ = self.decide_full(results, cross_flags)
        return prob, label

    def decide_full(self, results: list[ModalityResult],
                    cross_flags: list[RedFlag]) -> tuple[float, str, bool]:
        """(probability, label, abstained).

        The hard rule stands: payment / Aadhaar / OTP requests never yield 'Safe'. ``abstained``
        reports that the label is provisional -- see :func:`should_abstain`.
        """
        prob = self.probability(results, cross_flags)
        label = self.label(prob)
        forced = any(f.severity == "high" and f.code in config.FORCE_SUSPICIOUS_CODES
                     for r in results if r.available for f in r.red_flags)
        if forced and label == config.LABEL_SAFE:
            label, prob = config.LABEL_SUSPICIOUS, max(prob, self.art.t_low)
        return prob, label, should_abstain(prob, results, cross_flags)
