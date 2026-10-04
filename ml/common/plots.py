"""Matplotlib figures for reports: reliability diagram, PR curve, confusion matrix."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve

from ml.common import config


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def reliability_diagram(y_true, y_prob, path: Path, title: str = "Reliability diagram", n_bins: int = config.ECE_BINS) -> Path:
    """Mean predicted probability vs observed scam frequency per bin."""
    y, p = np.asarray(y_true), np.asarray(y_prob)
    idx = np.minimum((p * n_bins).astype(int), n_bins - 1)
    xs, ys, ns = [], [], []
    for b in range(n_bins):
        m = idx == b
        if m.any():
            xs.append(p[m].mean()); ys.append(y[m].mean()); ns.append(m.sum())
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], "--", color="grey", label="perfect")
    ax.plot(xs, ys, "o-", color="#0b2545", label="model")
    for x, yv, n in zip(xs, ys, ns):
        ax.annotate(str(n), (x, yv), textcoords="offset points", xytext=(4, -10), fontsize=7)
    ax.set(xlabel="Predicted P(scam)", ylabel="Observed frequency", title=title, xlim=(0, 1), ylim=(0, 1))
    ax.legend()
    return _save(fig, path)


def pr_curve(curves: dict[str, tuple], path: Path, title: str = "Precision-recall") -> Path:
    """``curves``: name -> (y_true, y_prob)."""
    fig, ax = plt.subplots(figsize=(5, 4.5))
    for name, (y, p) in curves.items():
        pr, rc, _ = precision_recall_curve(y, p)
        ax.plot(rc, pr, label=name)
    ax.set(xlabel="Recall", ylabel="Precision", title=title, xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(fontsize=8)
    return _save(fig, path)


def confusion_matrix_plot(cm, path: Path, title: str = "Confusion matrix") -> Path:
    """``cm`` = [[tn, fp], [fn, tp]]."""
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, int(v), ha="center", va="center", color="white" if v > cm.max() / 2 else "black")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["genuine", "scam"], yticklabels=["genuine", "scam"],
           xlabel="Predicted", ylabel="Actual", title=title)
    return _save(fig, path)
