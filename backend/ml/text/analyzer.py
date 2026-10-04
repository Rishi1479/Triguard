"""TextAnalyzer: best available text model (DistilBERT > TF-IDF+LR > rules only) -> ModalityResult."""
from __future__ import annotations

import json
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


class _TransformerScorer:
    """Wraps a HuggingFace sequence-classification folder exported by a Colab notebook.

    Covers MuRIL, XLM-R and DistilBERT: the tokenizer and config travel with the folder, so the
    only model-specific detail is that DistilBERT's ``forward()`` rejects ``token_type_ids``.
    """

    def __init__(self, folder: Path) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        meta = json.loads((folder / "meta.json").read_text())
        self.tok = AutoTokenizer.from_pretrained(folder)
        self.model = AutoModelForSequenceClassification.from_pretrained(folder).eval()
        self.max_len = int(meta.get("max_len", config.TRANSFORMER_MAX_LEN))
        self.threshold = float(meta.get("threshold", config.DEFAULT_CLASS_THRESHOLD))
        self.coef, self.intercept = meta.get("platt_coef"), meta.get("platt_intercept")
        self.accepts_token_type = "distilbert" not in type(self.model).__name__.lower()
        self._torch = torch

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Softmax P(scam), Platt-scaled with the parameters fitted on validation in Colab."""
        torch = self._torch
        out = []
        for i in range(0, len(texts), 16):
            enc = self.tok(texts[i: i + 16], truncation=True, max_length=self.max_len, padding=True,
                           return_tensors="pt")
            if not self.accepts_token_type:
                enc.pop("token_type_ids", None)
            with torch.no_grad():
                out.append(torch.softmax(self.model(**enc).logits, dim=-1)[:, 1].numpy())
        p = np.concatenate(out)
        if self.coef is None:
            return p
        clipped = np.clip(p, config.PROB_EPS, 1 - config.PROB_EPS)
        z = np.log(clipped / (1 - clipped))
        return 1 / (1 + np.exp(-(self.coef * z + self.intercept)))


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


_LOADERS = {
    "text_muril": ("meta.json", _TransformerScorer),
    "text_xlmr": ("meta.json", _TransformerScorer),
    "text_distilbert": ("meta.json", _TransformerScorer),
    "text_hybrid_lgbm": ("model.joblib", _HybridScorer),
    "text_tfidf_lr": ("model.joblib", _TfidfScorer),
}


def boost_with_rules(p: float, flags: list[RedFlag]) -> float:
    """Raise P(scam) for rule flags: p' = 1 - (1-p) * prod(1 - w[severity])."""
    keep = 1.0 - p
    for f in flags:
        keep *= 1.0 - config.RULE_BOOST[f.severity]
    return float(1.0 - keep)


class TextAnalyzer:
    """Analyze job/internship text. Picks the best model present in ``models_dir`` automatically."""

    def __init__(self, models_dir: Path | None = None, use_rules_boost: bool = True) -> None:
        self.models_dir = Path(models_dir) if models_dir else config.MODELS_DIR
        self.use_rules_boost = use_rules_boost
        self.model_name = "rules_only"
        self.scorer = None
        for name in config.TEXT_MODEL_PRIORITY:
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
