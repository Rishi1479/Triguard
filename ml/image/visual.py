"""CLIP visual scorer: ViT-B/32 image embeddings + Logistic Regression (genuine vs scam).

Training is meant for Colab (see ml/notebooks/image_clip_colab.ipynb). At inference this module is used
only if models/image_clip/model.joblib exists; otherwise ``load_visual_scorer`` returns None and the
analyzer gives the visual score weight 0.
"""
from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np

from ml.common import config

log = logging.getLogger(__name__)


class ClipScorer:
    """Embeds an image with open_clip and applies the saved calibrated classifier."""

    def __init__(self, folder: Path) -> None:
        blob = joblib.load(folder / "model.joblib")
        self.clf, self.calibrator, self.threshold = blob["clf"], blob["calibrator"], blob["threshold"]
        import open_clip
        import torch

        self._torch = torch
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            blob.get("model_name", config.CLIP_MODEL_NAME), pretrained=blob.get("pretrained", config.CLIP_PRETRAINED))
        self._model.eval()

    def embed(self, rgb_images: list[np.ndarray]) -> np.ndarray:
        """L2-normalised CLIP embeddings for RGB uint8 arrays."""
        from PIL import Image

        torch = self._torch
        batch = torch.stack([self._preprocess(Image.fromarray(a)) for a in rgb_images])
        with torch.no_grad():
            e = self._model.encode_image(batch)
            e = e / e.norm(dim=-1, keepdim=True)
        return e.numpy()

    def predict_proba(self, rgb_images: list[np.ndarray]) -> np.ndarray:
        """Calibrated P(scam) for each image."""
        return self.calibrator.predict(self.clf.predict_proba(self.embed(rgb_images))[:, 1])


def load_visual_scorer(folder: Path | None = None) -> ClipScorer | None:
    """The CLIP scorer, or None if its files are missing or fail to load."""
    folder = folder or config.IMAGE_CLIP_DIR
    if not (folder / "model.joblib").exists():
        return None
    try:
        return ClipScorer(folder)
    except Exception as exc:
        log.warning("CLIP scorer failed to load (%s); visual weight set to 0", exc)
        return None
