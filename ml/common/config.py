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
