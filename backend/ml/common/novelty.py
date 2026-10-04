"""Out-of-distribution detection for text inputs.

A classifier asked about a document register it never saw in training still returns a confident
number. That is how a genuine offer letter scored 0.998 while offer-letter wording existed only in
the scam class: the score reflected model bias, not evidence (LIME showed the top drivers were
function words at weights of ±0.02).

Rebalancing the training data removes that specific confound, but the general failure remains --
resumes, SMS forwards, other languages and unseen formats keep arriving. This module measures how
far an input sits from the training distribution so the caller can shrink the score back toward the
prior and abstain instead of guessing.

Signal: the share of word tokens absent from the training vocabulary. It is directly
interpretable ("two thirds of these words never appeared in training"), language-agnostic and
degrades gracefully, which a cosine distance in a truncated-SVD space did not -- that variant
flagged ordinary job postings as novel while missing a cooking recipe, because the leading
components were dominated by function words.

The cut-off is an absolute configured value, deliberately **not** a percentile of the training
distribution: training documents are near-duplicates of each other (a templated corpus has a 99th
percentile OOV share of ~0.015), so a self-calibrated threshold would flag almost every real input.
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import joblib
import numpy as np

from ml.common import config

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lower-case alphanumeric tokens."""
    return _TOKEN_RE.findall(str(text).lower())


class NoveltyDetector:
    """Out-of-vocabulary novelty scorer. Fit on the training split only."""

    def __init__(self, min_doc_freq: int = config.NOVELTY_MIN_DOC_FREQ,
                 threshold: float = config.NOVELTY_THRESHOLD) -> None:
        self.min_doc_freq = min_doc_freq
        self.threshold = threshold
        self.vocab: frozenset[str] = frozenset()
        self.n_train: int = 0

    def fit(self, texts: list[str]) -> "NoveltyDetector":
        """Build the training vocabulary (words appearing in >= ``min_doc_freq`` documents)."""
        texts = [t for t in (texts or []) if t and str(t).strip()]
        self.n_train = len(texts)
        if len(texts) < config.NOVELTY_MIN_TRAIN_ROWS:
            self.vocab = frozenset()  # stays disabled
            return self
        df: Counter[str] = Counter()
        for t in texts:
            df.update(set(tokenize(t)))
        self.vocab = frozenset(w for w, c in df.items() if c >= self.min_doc_freq)
        return self

    @property
    def enabled(self) -> bool:
        """False when the detector was never fitted on enough data."""
        return bool(self.vocab)

    def novelty(self, texts: list[str]) -> np.ndarray:
        """Share of out-of-vocabulary tokens per text, in [0, 1]. All zeros when disabled."""
        if not self.enabled:
            return np.zeros(len(texts))
        out = []
        for t in texts:
            toks = tokenize(t or "")
            out.append(float(np.mean([w not in self.vocab for w in toks])) if toks else 1.0)
        return np.asarray(out, dtype=float)

    def is_novel(self, texts: list[str]) -> np.ndarray:
        """Boolean mask of inputs past the novelty cut-off."""
        return self.novelty(texts) > self.threshold

    def save(self, folder: Path) -> None:
        """Persist to ``folder/novelty.joblib``."""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, folder / "novelty.joblib")

    @classmethod
    def load(cls, folder: Path) -> "NoveltyDetector | None":
        """Restore a detector, or None when the artifact is missing, stale or unreadable."""
        path = Path(folder) / "novelty.joblib"
        if not path.exists():
            return None
        try:
            obj = joblib.load(path)
        except Exception:
            return None
        return obj if isinstance(obj, cls) and obj.enabled else None


def shrink_toward_prior(p: float, novelty: float, prior: float, threshold: float) -> float:
    """Pull ``p`` toward ``prior`` in proportion to how far past ``threshold`` the novelty sits.

    At the threshold nothing changes; at novelty 1.0 the score collapses to the prior, because the
    model has no comparable training evidence to justify any other number.
    """
    if novelty <= threshold:
        return float(p)
    span = max(1.0 - threshold, 1e-9)
    w = min(max((novelty - threshold) / span, 0.0), 1.0)
    return float((1.0 - w) * p + w * prior)
