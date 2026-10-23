"""Reconciliation of the base paper's published figures.

Pillai (2023) reports, per model, an AUROC, accuracy, precision, recall and F1, plus a confusion
matrix for the Bi-LSTM and LightGBM. Those are over-determined: from TP, TN, precision and recall
the whole matrix follows, and accuracy and AUROC can then be recomputed and compared with what was
printed. Doing that recomputation here turns three claims in reports/RESULTS.md from assertions
into something a reader can run:

1. **The reported AUROC is a hard-label AUROC.** The area under an ROC curve through a *single*
   operating point is ``(1 + TPR - FPR) / 2``. For both models with a published matrix that
   expression reproduces the paper's AUROC to two decimals, which an AUROC integrated over
   probability thresholds would not.
2. **The test split is not stratified.** EMSCAD is 9.15% fraudulent, so a 20% test split should hold
   ~327 fake postings; the reconstructed matrices contain ~173.
3. **Accuracy is close to the majority-class floor.** At ~4.8% positives, predicting "never fake"
   already scores ~95.2%.

Nothing here is a criticism of the Bi-LSTM architecture, which this project reproduces as a
baseline; it is about how the numbers were measured.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ml.common import config


@dataclass(frozen=True)
class Reconciliation:
    """A base-paper model's published figures alongside the recomputed ones."""

    name: str
    tp: int
    tn: int
    fp: int
    fn: int
    positives: int
    negatives: int
    n: int
    accuracy: float
    accuracy_reported: float
    f1: float
    f1_reported: float
    hard_label_auc: float
    auroc_reported: float
    positive_rate: float
    majority_baseline: float
    notes: list[str] = field(default_factory=list)

    @property
    def auc_matches_hard_labels(self) -> bool:
        """True when the published AUROC equals the single-operating-point AUC (2 dp)."""
        return round(self.hard_label_auc, 2) == round(self.auroc_reported, 2)

    @property
    def f1_consistent(self) -> bool:
        """True when the published F1 equals the F1 implied by the published precision/recall."""
        return abs(self.f1 - self.f1_reported) <= 0.015


def f1_from(precision: float, recall: float) -> float:
    """Harmonic mean of precision and recall (0 when both are 0)."""
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def hard_label_auc(tp: int, tn: int, fp: int, fn: int) -> float:
    """Area under the ROC curve through one operating point: ``(1 + TPR - FPR) / 2``.

    This is what you get by passing 0/1 predictions where probabilities are expected.
    """
    tpr = tp / (tp + fn) if tp + fn else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return (1.0 + tpr - fpr) / 2.0


def reconstruct(name: str, stats: dict[str, float] | None = None,
                n: int = config.BASE_PAPER_TEST_N) -> Reconciliation | None:
    """Rebuild one model's confusion matrix. Returns None when no matrix was published.

    ``fn`` follows from recall (``TP/(TP+FN)``) and ``fp`` from precision (``TP/(TP+FP)``).
    """
    stats = stats if stats is not None else config.BASE_PAPER_MODELS[name]
    if "tp" not in stats or "tn" not in stats:
        return None

    tp, tn = int(stats["tp"]), int(stats["tn"])
    precision, recall = stats["precision"], stats["recall"]
    fn = round(tp / recall - tp) if recall else 0
    fp = round(tp / precision - tp) if precision else 0
    positives, negatives = tp + fn, tn + fp

    rec = Reconciliation(
        name=name, tp=tp, tn=tn, fp=fp, fn=fn, positives=positives, negatives=negatives, n=n,
        accuracy=(tp + tn) / n, accuracy_reported=stats["accuracy"],
        f1=f1_from(precision, recall), f1_reported=stats["f1"],
        hard_label_auc=hard_label_auc(tp, tn, fp, fn), auroc_reported=stats["auroc"],
        positive_rate=positives / n, majority_baseline=negatives / n,
    )
    notes = []
    if rec.auc_matches_hard_labels:
        notes.append("published AUROC reproduces (1+TPR-FPR)/2, i.e. it is a hard-label AUROC")
    if not rec.f1_consistent:
        notes.append(f"published F1 {rec.f1_reported:.2f} disagrees with P/R, which imply {rec.f1:.2f}")
    return Reconciliation(**{**rec.__dict__, "notes": notes})


def expected_fakes_in_test(n: int = config.BASE_PAPER_TEST_N) -> float:
    """Fake postings a *stratified* test split of ``n`` rows would contain."""
    d = config.BASE_PAPER_DATASET
    return n * d["fake"] / d["total"]


def dataset_positive_rate() -> float:
    """Fraudulent share of the full EMSCAD dataset as the paper describes it."""
    d = config.BASE_PAPER_DATASET
    return d["fake"] / d["total"]


def reconcile_all() -> list[Reconciliation]:
    """Reconciliation for every base-paper model that published a confusion matrix."""
    out = [reconstruct(name) for name in config.BASE_PAPER_MODELS]
    return [r for r in out if r is not None]


def f1_discrepancies() -> list[tuple[str, float, float]]:
    """``(model, published_f1, f1_implied_by_precision_and_recall)`` where the two disagree."""
    out = []
    for name, s in config.BASE_PAPER_MODELS.items():
        implied = f1_from(s["precision"], s["recall"])
        if abs(implied - s["f1"]) > 0.015:
            out.append((name, s["f1"], implied))
    return out
