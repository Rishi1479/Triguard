# PROGRESS

## WAITING FOR USER
_(Updated every phase; final consolidated list in Phase 10.)_
- [ ] `data/raw/fake_job_postings.csv` (Kaggle "Real / Fake Job Posting Prediction") — currently SYNTHETIC stand-in `data/raw/SYNTHETIC_fake_job_postings.csv`.
- [ ] `data/raw/custom_text.csv` (text,label,source; template in `data/raw/custom_text_template.csv`) — currently SYNTHETIC (68 rows).
- [ ] `data/raw/urls/*.csv` (PhishTank and/or Kaggle malicious_phish) — currently SYNTHETIC.
- [ ] Real offer-letter/poster images in `data/images/{scam,genuine}/` — currently 400 SYNTHETIC letters.
- [ ] `data/aligned/aligned.csv` (see `data/aligned/README.md`) — none yet.
- [ ] Re-run after adding data: `python -m ml.text.prepare_emscad && python -m ml.text.prepare_custom && python -m ml.url.prepare_urls && python -m ml.image.prepare_images`.

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

## Phase 2 — Data collection & preprocessing
**Built**: `ml/common/{privacy,textprep,synthetic}.py` (anonymize, clean/dedupe/split helpers, SYNTHETIC generators);
`ml/text/prepare_emscad.py`, `ml/text/prepare_custom.py`, `ml/url/prepare_urls.py`, `ml/image/synth_letters.py`,
`ml/image/prepare_images.py`; `data/aligned/{aligned_template.csv,README.md}`; `tests/test_phase2.py` (14 tests).
**Run**: `python -m ml.text.prepare_emscad` (also `prepare_custom`, `prepare_urls`, `synth_letters`, `prepare_images`).
Scripts fall back to SYNTHETIC data (files prefixed `SYNTHETIC_`, flag files in data/processed/) when real data is absent.
**Output (SYNTHETIC)**: text n=2300 (13% scam) -> 1610/345/345, 40 near-dups removed before the split; URLs n=2910 (51.5% scam);
custom Indian set 68 rows; 400 letters (200/200) with 70/15/15 manifest.
**Known issues**: all numbers above are SYNTHETIC. Dedupe keeps the scam row on label conflicts. URL split is random,
not domain-grouped (possible host leakage on real data). Custom synthetic set collapses to 68 rows after anonymised dedupe.

## Phase 3 — Text module
**Built**: `ml/common/{red_flags,calibration}.py` (+ `expected_calibration_error` in metrics.py); `ml/text/{models,baselines,analyzer,evaluate}.py`;
`ml/notebooks/text_distilbert_colab.ipynb` (weighted loss, early stopping on val PR-AUC, Platt+threshold on val, zip export, optional MuRIL cell);
tests `tests/test_red_flags.py` (40), `tests/test_text_module.py` (13).
**Run**: `python -m ml.text.baselines` (trains TF-IDF+LR and Bi-LSTM -> `models/text_{tfidf_lr,bilstm}/`, `reports/text_baselines.json`);
`python -m ml.text.evaluate` -> `reports/text_results.md`. `TextAnalyzer().analyze(text)` auto-selects `models/text_distilbert/` (needs `meta.json`)
over TF-IDF+LR, else rules only.
**Analyzer**: calibrated model prob (Platt on val) boosted by rule flags (`config.RULE_BOOST`), LIME top words, entity extraction.
**Metrics (SYNTHETIC data — meaningless)**: EMSCAD-like test F1 0.989 for both models (synthetic text is nearly template-separable).
Indian custom set: ROC-AUC 1.0 but precision 0.54 at the EMSCAD-tuned threshold — shows the domain-shift issue the real data should be used to study.
**Known issues**: all reported numbers must be regenerated with real EMSCAD. DistilBERT not trained (needs Colab/GPU); loader tested only with a tiny random model.
Rule boost makes final probabilities slightly less calibrated (ECE reported both ways). Bi-LSTM reproduction is architecture-faithful, not a bit-exact copy of the paper.

## Phase 4 — URL module
**Built**: `ml/url/{features,impersonation,live_checks,train,analyzer}.py`, `data/company_domains.csv` (52 employers),
`tests/test_url_module.py` (30 tests, all network calls mocked).
**Run**: `python -m ml.url.train` (-> `models/url_lgbm/`, `reports/url_results.json`); `URLAnalyzer().analyze(url)`.
**Design**: 33 lexical features (no network). LightGBM (balanced weights, early stopping on val logloss) + Platt calibration + F1 threshold on val.
Impersonation via rapidfuzz (leet-normalised lookalikes, brand in domain/subdomain/path) vs company_domains.csv.
Live checks (WHOIS age, TLS cert, shortener expansion with HEAD only, SSRF guard, 3s timeouts, TTL cache) are **inference-time rules only** and
become flags (NEW_DOMAIN, SSL_INVALID, URL_SHORTENER); never training features. Pages are never fetched or rendered.
Probability = calibrated model prob boosted by rule flags (`config.URL_RULE_BOOST`); SHAP top-6 features in `explanation`.
Malformed/unsupported-scheme input -> `available=False` + `MALFORMED_URL` flag.
**Metrics (SYNTHETIC URLs — meaningless)**: test F1 1.000 (synthetic URL patterns are trivially separable).
**Known issues**: needs real PhishTank/malicious_phish data for honest numbers; random (not domain-grouped) split may leak hosts; `brand_in_path` is low severity to avoid flagging job aggregators.
`brew install libomp zbar` was needed on this Mac for LightGBM and pyzbar.
