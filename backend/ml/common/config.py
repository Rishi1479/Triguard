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
# Accepted names for the real EMSCAD CSV, first match wins: the original University of the Aegean
# release (t/f labels) and the Kaggle copy (0/1 labels). Both hold the same 17,880 postings.
EMSCAD_FILES: tuple[str, ...] = ("DataSet.csv", EMSCAD_FILE)
CUSTOM_TEXT_FILE: str = "custom_text.csv"
URL_RAW_DIR: Path = RAW_DIR / "urls"
ALIGNED_FILE: str = "aligned.csv"
SYNTHETIC_PREFIX: str = "SYNTHETIC_"
EMSCAD_TEXT_COLUMNS: tuple[str, ...] = (
    "title", "company_profile", "description", "requirements", "benefits",
)
MIN_TEXT_CHARS: int = 20  # shorter merged texts are dropped as empty

# --- Text module -----------------------------------------------------------------
# The three servable text models, best first. Each is trained on its own with
# ``python -m ml.text.baselines --model <name>`` and one can be forced with TRIGUARD_TEXT_MODEL;
# otherwise the first one present in models/ is served. ``text_bilstm`` is the base paper's
# architecture and is kept LAST, so it is never chosen over a stronger model just because its
# artifact happens to be present.
TEXT_MODEL_PRIORITY: tuple[str, ...] = ("text_hybrid_lgbm", "text_tfidf_lr", "text_bilstm")
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
LIME_NUM_SAMPLES: int = 300
LIME_TOP_WORDS: int = 8
# Probability boost from rule flags: p' = 1 - (1-p) * prod(1 - w[severity]).
RULE_BOOST: dict[str, float] = {"high": 0.30, "medium": 0.10, "low": 0.02}
# --- Base paper -------------------------------------------------------------------
# Pillai (2023), "Detecting Fake Job Postings Using Bidirectional LSTM", IRJMETS 5(3), 3883-3890.
# Every figure below is as printed in the paper's Tables 2 and 3; `tp`/`tn` come from its confusion
# matrices (Figures 5 and 6). ml/common/base_paper.py reconstructs the full matrices from these and
# checks them against the reported accuracy and AUROC -- which is how the hard-label AUROC claim in
# reports/RESULTS.md section 2 is derived rather than asserted.
BASE_PAPER_DATASET: dict[str, int] = {"total": 17_880, "genuine": 16_244, "fake": 1_636}
BASE_PAPER_TEST_N: int = 3_576  # "Out of the 3576 records in the test set"
BASE_PAPER_MODELS: dict[str, dict[str, float]] = {
    "Bi-LSTM": {"auroc": 0.91, "accuracy": 0.9871, "precision": 0.89, "recall": 0.83, "f1": 0.86,
                "tp": 144, "tn": 3386},
    "LightGBM": {"auroc": 0.83, "accuracy": 0.9818, "precision": 0.94, "recall": 0.66, "f1": 0.77,
                 "tp": 115, "tn": 3396},
    # Random Forest and Gradient Boosting have no published confusion matrix, so no tp/tn.
    "Random Forest": {"auroc": 0.70, "accuracy": 0.9704, "precision": 0.97, "recall": 0.40, "f1": 0.57},
    "Gradient Boosting": {"auroc": 0.68, "accuracy": 0.9667, "precision": 0.86, "recall": 0.37, "f1": 0.62},
}
BASE_PAPER: dict[str, float] = BASE_PAPER_MODELS["Bi-LSTM"]  # the model we are compared against
# The dataset as it actually is, counted from data/raw/DataSet.csv (EMSCAD v1). The paper's 1,636
# fake postings are not in it; tests/test_base_paper.py re-counts when the CSV is present.
EMSCAD_ACTUAL: dict[str, int] = {"total": 17_880, "genuine": 17_014, "fake": 866}


# --- Behavioural (rule-derived) features -------------------------------------------
# Rule firings are fed to the model as named numeric features instead of only boosting the
# probability afterwards: behavioural cues transfer across language/geography far better than
# lexical n-grams, and SHAP over named features is readable in the report.
BEHAVIOURAL_FEATURE_CODES: tuple[str, ...] = (
    "PAYMENT_REQUEST", "NO_INTERVIEW", "SENSITIVE_INFO_REQUEST", "OTP_REQUEST", "URGENCY",
    "URGENCY_MILD", "UNREALISTIC_PAY", "MESSENGER_ONLY_CONTACT", "FREE_EMAIL_HR",
)
HYBRID_SVD_COMPONENTS: int = 150  # TF-IDF -> SVD before concatenating behavioural features
HYBRID_MIN_TRAIN_ROWS: int = 200  # below this the hybrid model is not trained

# --- Calibration -------------------------------------------------------------------
ECE_BINS: int = 10
BOOTSTRAP_SAMPLES: int = 2000  # resamples for metric confidence intervals
BOOTSTRAP_ALPHA: float = 0.05  # -> 95% percentile intervals
BOOTSTRAP_MIN_ROWS: int = 20  # below this a CI is reported as NaN
PROB_EPS: float = 1e-6  # clip before logit
MAX_TEXT_CHARS: int = 20_000  # longer inputs are truncated before modelling
RULES_ONLY_PRIOR: float = 0.10  # P(scam) before rule boosts when no trained text model exists

# --- Prior shift / cross-domain transfer -------------------------------------------
# A source-domain threshold transferred unchanged to a target domain with a different scam
# base rate collapses to "predict everything positive" (see reports/RESULTS.md section 4).
# Correct the logit by the log-odds difference of the two priors, then re-fit Platt on a small
# labelled target sample when one is available.
MIN_TARGET_CALIBRATION_ROWS: int = 40  # fewer labelled target rows -> prior correction only
TARGET_CALIBRATION_HOLDOUT: float = 0.5  # share of the target sample kept out of recalibration
EMSCAD_SCAM_PRIOR: float = 0.0484  # EMSCAD fraudulent share, used as the source prior default

# --- Novelty / out-of-distribution guard -------------------------------------------
# A model asked about a register absent from training still returns a confident number. When an
# input sits far from the training manifold its score is shrunk back toward the prior and the
# verdict abstains, rather than presenting a guess as evidence.
NOVELTY_MIN_TRAIN_ROWS: int = 100  # below this the detector stays disabled
NOVELTY_MIN_DOC_FREQ: int = 2  # a word counts as "seen" from this many training documents
# Absolute cut-off on the out-of-vocabulary token share, NOT a percentile of the training
# distribution: a templated corpus sits at ~0.015, which would flag nearly every real input.
# Measured separation: real job content 0.08-0.15, recipe 0.68, Hinglish SMS 0.56, German 0.92.
NOVELTY_THRESHOLD: float = 0.35
NOVELTY_DIR_SUFFIX: str = "text_novelty"

# --- URL module --------------------------------------------------------------------
COMPANY_DOMAINS_FILE: Path = DATA_DIR / "company_domains.csv"
URL_MODEL_DIR: Path = MODELS_DIR / "url_lgbm"
MAX_URL_CHARS: int = 2048
LIVE_TIMEOUT_S: float = 3.0
LIVE_CACHE_TTL_S: int = 3600
LOOKALIKE_MIN_SCORE: float = 86.0  # rapidfuzz ratio (0-100) for typosquat detection
# A domain older than this is treated as established, so "we cannot recognise this employer" is not
# worth reporting for it; below it (or when WHOIS gives no age) an unknown domain is worth a note.
ESTABLISHED_DOMAIN_DAYS: int = 730
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

# --- Fusion --------------------------------------------------------------------------
FUSION_DIR: Path = MODELS_DIR / "fusion"
FUSION_WEIGHTS: dict[str, float] = {"text": 0.40, "image": 0.25, "url": 0.35}  # renormalised over available
CROSS_BOOST: dict[str, float] = {"high": 0.25, "medium": 0.08, "low": 0.02}
FORCE_SUSPICIOUS_CODES: frozenset[str] = frozenset(
    {"PAYMENT_REQUEST", "SENSITIVE_INFO_REQUEST", "OTP_REQUEST", "QR_PAYMENT_REQUEST"})
FUSION_MIN_ALIGNED: int = 100  # fewer labelled aligned rows -> weighted-average fallback
# "Safe" cut-off keeps scam recall >= this on validation. 0.95 drove t_low to its floor, so almost
# nothing could ever be labelled Safe; 0.90 with the floor below keeps the band usable.
RECALL_TARGET_LOW: float = 0.90
SAFE_THRESHOLD_FLOOR: float = 0.10  # t_low is never tuned below this, whatever the recall target

# --- Missing modalities / abstention -------------------------------------------------
# An absent modality must not be encoded as p=0.0 ("confidently genuine"). It is imputed at the
# training-set prior and marked by its has_* indicator so the model can tell "absent" from "clean".
MISSING_MODALITY_PRIOR: float = 0.5  # neutral fallback until train_fuser learns the real prior
# Evidence weight of each modality for the abstention decision (text carries the most signal).
EVIDENCE_WEIGHTS: dict[str, float] = {"text": 1.0, "url": 0.6, "image": 0.5}
MIN_EVIDENCE_FOR_VERDICT: float = 0.6  # below this the verdict is reported as low-confidence
ABSTAIN_BAND: tuple[float, float] = (0.35, 0.65)  # probabilities this close to the fence abstain
MIN_TEXT_CHARS_FOR_VERDICT: int = 40  # very short text alone is not enough evidence
MAX_REASONS: int = 6

# --- Adversarial robustness ---------------------------------------------------------
# Scammers evade keyword matching; recall is re-measured under these obfuscations.
ROBUSTNESS_ATTACKS: tuple[str, ...] = (
    "spacing", "homoglyph", "obfuscate_contact", "spell_amounts", "synonym_swap", "all_combined",
)
ROBUSTNESS_SEED: int = SEED
MAX_EXTRA_URLS: int = 3
SALARY_MISMATCH_RATIO: float = 1.25
ROLE_MISMATCH_JACCARD: float = 0.2
FIGURES_DIR: Path = REPORTS_DIR / "figures"
ATS_DOMAINS: frozenset[str] = frozenset({
    "lever.co", "greenhouse.io", "myworkdayjobs.com", "smartrecruiters.com", "ashbyhq.com", "linkedin.com",
    "naukri.com", "indeed.com", "glassdoor.com", "internshala.com", "workable.com", "icims.com", "taleo.net",
    "successfactors.com", "wellfound.com"})
PLATT_C: float = 1.0  # L2 strength of Platt scaling; keeps probabilities finite when validation is separable
