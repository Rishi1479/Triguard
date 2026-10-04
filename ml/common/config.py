"""Central configuration for TriGuard: paths, seed, split ratios and decision thresholds."""
from __future__ import annotations

from pathlib import Path

ROOT_DIR: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = ROOT_DIR / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
IMAGES_DIR: Path = DATA_DIR / "images"
ALIGNED_DIR: Path = DATA_DIR / "aligned"
MODELS_DIR: Path = ROOT_DIR / "models"
REPORTS_DIR: Path = ROOT_DIR / "reports"

SEED: int = 42

TRAIN_RATIO: float = 0.70
VAL_RATIO: float = 0.15
TEST_RATIO: float = 0.15

# Default decision thresholds on the fused scam probability (to be tuned on validation).
SAFE_THRESHOLD: float = 0.30  # risk < this  -> Safe
SCAM_THRESHOLD: float = 0.70  # risk >= this -> Likely Scam; in between -> Suspicious

# Default decision threshold for binary classification metrics.
DEFAULT_CLASS_THRESHOLD: float = 0.5

# Number of candidate thresholds scanned by find_best_threshold.
THRESHOLD_GRID_SIZE: int = 199

LABEL_SAFE: str = "Safe"
LABEL_SUSPICIOUS: str = "Suspicious"
LABEL_SCAM: str = "Likely Scam"

# --- Data files ---------------------------------------------------------------
EMSCAD_FILE: str = "fake_job_postings.csv"
CUSTOM_TEXT_FILE: str = "custom_text.csv"
URL_RAW_DIR: Path = RAW_DIR / "urls"
ALIGNED_FILE: str = "aligned.csv"
SYNTHETIC_PREFIX: str = "SYNTHETIC_"
EMSCAD_TEXT_COLUMNS: tuple[str, ...] = (
    "title", "company_profile", "description", "requirements", "benefits",
)
MIN_TEXT_CHARS: int = 20  # shorter merged texts are dropped as empty

# --- Text module -----------------------------------------------------------------
TEXT_MODEL_PRIORITY: tuple[str, ...] = ("text_distilbert", "text_tfidf_lr")  # best first
TFIDF_WORD_NGRAMS: tuple[int, int] = (1, 2)
TFIDF_CHAR_NGRAMS: tuple[int, int] = (3, 5)
TFIDF_MAX_FEATURES: int = 200_000
BILSTM_VOCAB_SIZE: int = 20_000
BILSTM_MAX_LEN: int = 300
BILSTM_EMBED_DIM: int = 100
BILSTM_HIDDEN: int = 64
BILSTM_EPOCHS: int = 6
BILSTM_PATIENCE: int = 2
BILSTM_BATCH: int = 64
BILSTM_LR: float = 1e-3
DISTILBERT_MAX_LEN: int = 256
LIME_NUM_SAMPLES: int = 300
LIME_TOP_WORDS: int = 8
# Probability boost from rule flags: p' = 1 - (1-p) * prod(1 - w[severity]).
RULE_BOOST: dict[str, float] = {"high": 0.30, "medium": 0.10, "low": 0.02}
BASE_PAPER: dict[str, float] = {"f1": 0.86, "recall": 0.83}  # Pillai (2023) Bi-LSTM, EMSCAD

# --- Calibration -------------------------------------------------------------------
ECE_BINS: int = 10
PROB_EPS: float = 1e-6  # clip before logit
MAX_TEXT_CHARS: int = 20_000  # longer inputs are truncated before modelling
RULES_ONLY_PRIOR: float = 0.10  # P(scam) before rule boosts when no trained text model exists

# --- URL module --------------------------------------------------------------------
COMPANY_DOMAINS_FILE: Path = DATA_DIR / "company_domains.csv"
URL_MODEL_DIR: Path = MODELS_DIR / "url_lgbm"
MAX_URL_CHARS: int = 2048
LIVE_TIMEOUT_S: float = 3.0
LIVE_CACHE_TTL_S: int = 3600
LOOKALIKE_MIN_SCORE: float = 86.0  # rapidfuzz ratio (0-100) for typosquat detection
NEW_DOMAIN_DAYS_HIGH: int = 30
NEW_DOMAIN_DAYS_MEDIUM: int = 180
URL_RULE_BOOST: dict[str, float] = {"high": 0.40, "medium": 0.12, "low": 0.03}
LGBM_PARAMS: dict = {
    "n_estimators": 400, "learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 10,
    "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8, "class_weight": "balanced",
    "random_state": SEED, "verbose": -1, "n_jobs": 1,  # n_jobs=1: avoids libomp clash with torch (segfault on macOS)
}
LGBM_EARLY_STOP_ROUNDS: int = 30
SHAP_TOP_K: int = 6

# --- Image module ------------------------------------------------------------------
MAX_IMAGE_BYTES: int = 5 * 1024 * 1024
MAX_IMAGE_PIXELS: int = 40_000_000  # decompression-bomb guard
OCR_TARGET_MIN_SIDE: int = 1000  # upscale small images so the shorter side is at least this
OCR_MAX_SIDE: int = 3000
MIN_OCR_CHARS: int = 15
OCR_TEXT_KEEP_CHARS: int = 4000  # OCR text kept in ModalityResult.extracted
IMAGE_WEIGHT_VISUAL: float = 0.4  # CLIP visual score weight; auto-0 when models/image_clip is absent
IMAGE_WEIGHT_OCR_TEXT: float = 0.6
IMAGE_CLIP_DIR: Path = MODELS_DIR / "image_clip"
CLIP_MODEL_NAME: str = "ViT-B-32"
CLIP_PRETRAINED: str = "openai"
OCR_GT_FILE: Path = IMAGES_DIR / "ocr_gt.csv"
TESSERACT_TIMEOUT_S: int = 30
