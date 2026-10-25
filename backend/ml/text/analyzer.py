"""TextAnalyzer: chosen or best available text model (TF-IDF+LR > hybrid LightGBM > Bi-LSTM > rules only) -> ModalityResult."""
from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np

from ml.common import config
from ml.common.calibration import PlattCalibrator
from ml.common.novelty import NoveltyDetector, shrink_toward_prior
from ml.common.red_flags import detect_red_flags, extract_entities
from ml.common.schemas import ModalityResult, RedFlag

log = logging.getLogger(__name__)


class _TfidfScorer:
    """Wraps models/text_tfidf_lr/model.joblib."""

    def __init__(self, folder: Path) -> None:
        blob = joblib.load(folder / "model.joblib")
        self.model, self.calibrator, self.threshold = blob["model"], blob["calibrator"], blob["threshold"]

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Calibrated P(scam)."""
        return self.calibrator.predict(self.model.predict_proba(texts))


class _HybridScorer:
    """Wraps models/text_hybrid_lgbm/model.joblib (TF-IDF+SVD+behavioural features, LightGBM)."""

    def __init__(self, folder: Path) -> None:
        blob = joblib.load(folder / "model.joblib")
        self.model, self.calibrator, self.threshold = blob["model"], blob["calibrator"], blob["threshold"]

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Calibrated P(scam)."""
        return self.calibrator.predict(self.model.predict_proba(texts))

    def behavioural_importance(self, top_k: int = config.SHAP_TOP_K) -> list[tuple[str, float]]:
        """Gain importance of the behavioural features (passthrough for explanations)."""
        return self.model.behavioural_importance(top_k)


class _BiLSTMScorer:
    """Wraps models/text_bilstm/ (weights + vocab from ``BiLSTMClassifier.save``, Platt from ``calib.joblib``).

    The base paper's architecture (Pillai 2023), served through the same interface as every other
    scorer so a demo can be run on it directly instead of only comparing it offline in the report.
    """

    def __init__(self, folder: Path) -> None:
        from ml.text.models import BiLSTMClassifier

        blob = joblib.load(folder / "calib.joblib")
        self.model = BiLSTMClassifier.load(folder)
        self.calibrator, self.threshold = blob["calibrator"], blob["threshold"]

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Calibrated P(scam)."""
        return self.calibrator.predict(self.model.predict_proba(texts))


_LOADERS = {
    "text_hybrid_lgbm": ("model.joblib", _HybridScorer),
    "text_tfidf_lr": ("model.joblib", _TfidfScorer),
    # calib.joblib is written after model.pt/vocab.json, so it marks a *complete* Bi-LSTM artifact.
    "text_bilstm": ("calib.joblib", _BiLSTMScorer),
}


def search_order(prefer: str | None) -> list[str]:
    """``config.TEXT_MODEL_PRIORITY`` with ``prefer`` moved to the front; unknown names are ignored."""
    order = list(config.TEXT_MODEL_PRIORITY)
    if not prefer:
        return order
    if prefer not in _LOADERS:
        log.warning("unknown text model %r (known: %s); using the default order", prefer, ", ".join(_LOADERS))
        return order
    return [prefer, *(n for n in order if n != prefer)]


def boost_with_rules(p: float, flags: list[RedFlag]) -> float:
    """Raise P(scam) for rule flags: p' = 1 - (1-p) * prod(1 - w[severity])."""
    keep = 1.0 - p
    for f in flags:
        keep *= 1.0 - config.RULE_BOOST[f.severity]
    return float(1.0 - keep)


class TextAnalyzer:
    """Analyze job/internship text. Picks the best model present in ``models_dir`` automatically.

    ``prefer`` forces a specific model (e.g. ``"text_bilstm"``) to be tried first; the rest of the
    priority order still acts as a fallback, so a missing artifact degrades instead of failing. The
    model actually loaded is reported by ``model_name`` and by ``GET /health``.
    """

    def __init__(self, models_dir: Path | None = None, use_rules_boost: bool = True,
                 prefer: str | None = None) -> None:
        self.models_dir = Path(models_dir) if models_dir else config.MODELS_DIR
        self.use_rules_boost = use_rules_boost
        self.model_name = "rules_only"
        self.scorer = None
        for name in search_order(prefer):
            if name not in _LOADERS:
                log.warning("no loader registered for %s; skipping", name)
                continue
            marker, cls = _LOADERS[name]
            folder = self.models_dir / name
            if not (folder / marker).exists():
                continue
            try:
                scorer = cls(folder)
                scorer.predict_proba(["self-test"])  # catches artifacts saved by an incompatible library version
                self.scorer, self.model_name = scorer, name
                break
            except Exception as exc:  # corrupt/incomplete artifact -> try the next model
                log.warning("could not load %s (%s); falling back", name, exc)
        if self.scorer is None:
            log.warning("no trained text model in %s; using rules only", self.models_dir)
        elif prefer and self.model_name != prefer:
            log.warning("preferred text model %r is unavailable; using %r instead", prefer, self.model_name)
        self.novelty = NoveltyDetector.load(self.models_dir / config.NOVELTY_DIR_SUFFIX)

    @property
    def threshold(self) -> float:
        """Validation-tuned decision threshold of the loaded model."""
        return float(self.scorer.threshold) if self.scorer else config.DEFAULT_CLASS_THRESHOLD

    def model_proba(self, texts: list[str]) -> np.ndarray:
        """Calibrated model-only P(scam) for a batch (no rules)."""
        if self.scorer is None:
            return np.full(len(texts), config.RULES_ONLY_PRIOR)
        return self.scorer.predict_proba([t[: config.MAX_TEXT_CHARS] for t in texts])

    def _lime_words(self, text: str) -> list[dict]:
        from lime.lime_text import LimeTextExplainer

        def fn(texts: list[str]) -> np.ndarray:
            p = self.model_proba(list(texts))
            return np.column_stack([1 - p, p])

        exp = LimeTextExplainer(class_names=["genuine", "scam"], random_state=config.SEED).explain_instance(
            text, fn, num_features=config.LIME_TOP_WORDS, num_samples=config.LIME_NUM_SAMPLES)
        return [{"word": w, "weight": round(float(v), 4)} for w, v in exp.as_list()]

    def analyze(self, text: str | None, explain: bool = True) -> ModalityResult:
        """Score ``text``. Empty/None input -> ``available=False``."""
        if text is None or not text.strip():
            return ModalityResult.unavailable("text")
        text = text[: config.MAX_TEXT_CHARS]
        flags = detect_red_flags(text)
        p_raw = float(self.model_proba([text])[0])
        p_model, ood, nov = p_raw, False, None
        if self.novelty is not None and self.scorer is not None:
            nov = float(self.novelty.novelty([text])[0])
            ood = nov > self.novelty.threshold
            # Far from anything seen in training, the score carries no evidence: shrink to the prior.
            p_model = shrink_toward_prior(p_raw, nov, config.RULES_ONLY_PRIOR, self.novelty.threshold)
        prob = boost_with_rules(p_model, flags) if self.use_rules_boost else p_model
        explanation: dict = {"model": self.model_name, "model_probability": round(p_model, 4),
                             "threshold": self.threshold}
        if nov is not None:
            explanation["novelty"] = round(nov, 4)
            explanation["out_of_distribution"] = ood
            if ood:
                explanation["raw_model_probability"] = round(p_raw, 4)
        if explain and self.scorer is not None:
            try:
                explanation["lime_top_words"] = self._lime_words(text)
            except Exception as exc:  # explanation must never break scoring
                log.warning("LIME failed: %s", exc)
        extracted = extract_entities(text)
        extracted["n_chars"] = len(text)
        if ood:
            # Picked up by the fuser's abstention check.
            extracted["out_of_distribution"] = True
        return ModalityResult(modality="text", probability=min(max(prob, 0.0), 1.0), red_flags=flags,
                              extracted=extracted, explanation=explanation)
