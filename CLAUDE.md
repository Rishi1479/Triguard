# Project: TriGuard — Multimodal AI System for Detecting Fake Job & Internship Scams

## What we are building
A system that takes any combination of (1) job/internship TEXT, (2) an IMAGE (offer letter,
poster, screenshot), and (3) an application URL, and returns:
- a label: Safe / Suspicious / Likely Scam
- a calibrated risk score (0–100%)
- human-readable reasons (red flags)
It is a final-year academic project (Team 3, Data Mining & Bigdata MRG). The base paper is
Pillai (2023), "Detecting Fake Job Postings Using Bidirectional LSTM", which used text only
on the EMSCAD dataset. We must beat and extend it.

## Tech stack (do not change without asking)
- Python 3.11, FastAPI, Pydantic v2, SQLAlchemy + SQLite
- ML: scikit-learn, LightGBM, PyTorch, HuggingFace transformers, SHAP, LIME
- OCR/Image: OpenCV, EasyOCR (Tesseract as comparison), pyzbar, open_clip
- URL: tldextract, python-whois, rapidfuzz
- Frontend: React + Vite + TypeScript + Tailwind CSS
- Tests: pytest (backend), Vitest (frontend)

## Folder structure
triguard/
  backend/                     # self-contained: run Python commands from here
    app/                       # FastAPI app
    ml/
      common/                  # shared interface, metrics, red-flag rules
      text/  url/  image/  fusion/
      notebooks/               # Colab training notebooks
    data/raw/  data/processed/  data/images/  data/aligned/
    models/                    # saved model artifacts (gitignored)
    reports/                   # metrics, figures, ablation tables
    tests/  scripts/  requirements.txt  Dockerfile
  frontend/                    # React app (self-contained)
  docs/  docker-compose.yml  CLAUDE.md  PROGRESS.md  README.md

## Shared module interface (EVERY modality must follow this)
Each modality analyzer exposes: analyze(input) -> ModalityResult
ModalityResult (Pydantic):
  modality: "text" | "image" | "url"
  probability: float         # calibrated P(scam), 0..1
  red_flags: list[RedFlag]   # RedFlag = {code, message, severity: low|medium|high, evidence}
  extracted: dict            # e.g. urls/emails/phones/company found (used for cross-checks)
  explanation: dict          # e.g. top SHAP features or LIME tokens
  available: bool            # False if this input was not provided

## Rules
- Never fit vectorizers/scalers/tokenizers on anything except the training split.
- Deduplicate data BEFORE splitting. Use stratified 70/15/15 splits with a fixed seed (42).
- Handle class imbalance (class weights + threshold tuning on validation).
- Report for the scam class: precision, recall, F1, PR-AUC, ROC-AUC (from probabilities,
  never hard labels), MCC. Accuracy is secondary.
- Never open/render suspicious URLs; metadata lookups only, with timeouts.
- Never store uploaded images permanently; anonymize stored text (mask phones/emails/IDs).
- Type hints, docstrings, small functions, config in ml/common/config.py (no magic numbers).
- Every phase: write tests, run them, fix failures, then append a summary to PROGRESS.md
  (what was built, files, how to run, metrics, known issues).
- If something is ambiguous, ask me before guessing.