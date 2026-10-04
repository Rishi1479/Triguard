# PROGRESS

## Phase 1 — Project setup (2026-10-04)

**Built**
- Folder structure per CLAUDE.md, `__init__.py` in all Python packages, `.gitkeep` in data/models/reports dirs.
- `requirements.txt` (pinned; `pip install --dry-run` resolves on Python 3.12) and `.gitignore`.
- `ml/common/schemas.py`: `RedFlag`, `ModalityResult` (with `.unavailable()` helper), `FinalResult`.
- `ml/common/config.py`: paths, `SEED=42`, 70/15/15 split ratios, thresholds Safe<0.30 / Likely Scam>=0.70 (defaults, to tune), labels.
- `ml/common/metrics.py`: `evaluate(y_true, y_prob, threshold)` (precision, recall, F1, PR-AUC, ROC-AUC, MCC, accuracy,
  confusion matrix `[[tn, fp], [fn, tp]]`; AUCs from probabilities) and `find_best_threshold(y_true, y_prob, metric)` (f1/precision/recall/mcc).
- `backend/app/main.py`: FastAPI with `GET /health`.
- `frontend/`: Vite + React + TS + Tailwind v4 placeholder page, Vitest configured with one smoke test.
- `README.md`.

**Run**
- `.venv/bin/python -m pytest` — backend tests; `cd frontend && npm test && npm run build`.

**Results**: 28 backend tests pass; 1 frontend test passes; frontend production build succeeds.

**Known issues / notes**
- Machine has Python 3.12.4, not 3.11 (CLAUDE.md target). Phase 1 verified on 3.12; heavy ML deps not installed yet (only dry-run resolved).
- `evaluate` on a single-class `y_true` returns NaN AUCs and sklearn emits an MCC/confusion-matrix warning.
- Starlette warns that `httpx` TestClient is deprecated in favour of `httpx2` (harmless for now).
- Not a git repo yet.
