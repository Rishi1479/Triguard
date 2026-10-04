"""Bootstrap confidence intervals for the scam-class metrics.

With an aligned set of a few hundred rows a point estimate like "precision 1.000" is one fold's
luck: the 95% interval on 14 positives spans roughly 0.75-1.00. Every reported table should carry
an interval, so :func:`bootstrap_metrics` wraps :func:`ml.common.metrics.evaluate` in a stratified
percentile bootstrap.
"""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from ml.common import config
from ml.common.metrics import evaluate

CI_METRICS: tuple[str, ...] = ("precision", "recall", "f1", "pr_auc", "roc_auc", "mcc")


def _stratified_indices(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One resample that keeps the per-class counts of ``y`` (avoids all-one-class draws)."""
    parts = [rng.choice(np.flatnonzero(y == c), size=int((y == c).sum()), replace=True)
             for c in np.unique(y)]
    return np.concatenate(parts)


def bootstrap_metrics(
    y_true: Any,
    y_prob: Any,
    threshold: float = config.DEFAULT_CLASS_THRESHOLD,
    metrics: Iterable[str] = CI_METRICS,
    n_samples: int = config.BOOTSTRAP_SAMPLES,
    alpha: float = config.BOOTSTRAP_ALPHA,
    seed: int = config.SEED,
) -> dict[str, dict[str, float]]:
    """Point estimate plus a percentile CI per metric.

    Returns ``{metric: {"value", "lo", "hi", "width"}}``. Resampling is stratified on ``y_true``,
    so the class balance of every resample matches the original set. Intervals are NaN when the
    set is smaller than ``config.BOOTSTRAP_MIN_ROWS`` or has a single class.
    """
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_prob, dtype=float)
    point = evaluate(y, p, threshold)
    names = list(metrics)
    too_small = len(y) < config.BOOTSTRAP_MIN_ROWS or len(np.unique(y)) < 2
    if too_small:
        return {m: {"value": float(point[m]), "lo": float("nan"), "hi": float("nan"),
                    "width": float("nan")} for m in names}

    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {m: [] for m in names}
    for _ in range(n_samples):
        idx = _stratified_indices(y, rng)
        res = evaluate(y[idx], p[idx], threshold)
        for m in names:
            v = res[m]
            if not np.isnan(v):
                draws[m].append(float(v))

    out: dict[str, dict[str, float]] = {}
    for m in names:
        vals = np.asarray(draws[m], dtype=float)
        if vals.size == 0:
            lo = hi = float("nan")
        else:
            lo, hi = (float(x) for x in np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)]))
        out[m] = {"value": float(point[m]), "lo": lo, "hi": hi,
                  "width": float(hi - lo) if not np.isnan(hi) else float("nan")}
    return out


def format_ci(ci: dict[str, float], digits: int = 3) -> str:
    """``'0.812 [0.74-0.88]'``, or just the value when the interval is unavailable."""
    v = ci["value"]
    if np.isnan(ci["lo"]) or np.isnan(ci["hi"]):
        return f"{v:.{digits}f}"
    return f"{v:.{digits}f} [{ci['lo']:.2f}–{ci['hi']:.2f}]"


def ci_table_row(label: str, ci: dict[str, dict[str, float]],
                 metrics: Iterable[str] = CI_METRICS) -> list[str]:
    """Markdown table row: label followed by 'value [lo-hi]' per metric."""
    return [label, *(format_ci(ci[m]) for m in metrics)]
