# PROGRESS

## WAITING FOR USER
_(Updated every phase; final consolidated list in Phase 10.)_
- [ ] `data/raw/fake_job_postings.csv` (Kaggle "Real / Fake Job Posting Prediction") — currently SYNTHETIC stand-in `data/raw/SYNTHETIC_fake_job_postings.csv`.
- [ ] `data/raw/custom_text.csv` (text,label,source; template in `data/raw/custom_text_template.csv`) — currently SYNTHETIC (68 rows).
- [ ] `data/raw/urls/*.csv` (PhishTank and/or Kaggle malicious_phish) — currently SYNTHETIC.
- [ ] Real offer-letter/poster images in `data/images/{scam,genuine}/` — currently 400 SYNTHETIC letters.
- [ ] `data/aligned/aligned.csv` (see `data/aligned/README.md`) — none yet.
- [ ] Colab (GPU): run `ml/notebooks/text_distilbert_colab.ipynb` -> unzip into `models/text_distilbert/`; run `ml/notebooks/image_clip_colab.ipynb` -> unzip into `models/image_clip/` (both auto-detected, no code change).
- [ ] `data/images/ocr_gt.csv` (path,text) with real annotated images — currently SYNTHETIC (20 letters).
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
Note (Phase 4): torch + LightGBM in one process segfaulted on macOS (duplicate libomp). Mitigated with `KMP_DUPLICATE_LIB_OK=TRUE` in `ml/__init__.py`, `n_jobs=1` for LightGBM and `tests/conftest.py`. 125 tests pass.

## Phase 5 — Image & OCR module
**Built**: `ml/image/{preprocess,ocr,qr,visual,analyzer,compare_ocr}.py`, `ml/notebooks/image_clip_colab.ipynb`, `tests/test_image_module.py` (17 tests).
**Run**: `python -m ml.image.compare_ocr` -> `reports/ocr_compare.md`; `ImageAnalyzer().analyze(image_bytes)`.
**Flow**: decode (in memory; size/pixel limits) -> preprocess -> OCR (EasyOCR, Tesseract fallback; Tesseract is piped via stdin so nothing touches disk)
-> TextAnalyzer on the OCR text -> QR decode (pyzbar, OpenCV fallback; UPI QR => `QR_PAYMENT_REQUEST` high flag, URL QR => `QR_LINK` + added to `extracted.urls`)
-> weighted combine of OCR-text prob (0.6) and CLIP prob (0.4, forced to 0 until `models/image_clip/` exists). A test asserts no temp files / imwrite / PIL save during analysis.
**Metrics (SYNTHETIC letters, 20 images, CER)**: Tesseract 0.002, EasyOCR 0.039-0.042; preprocessing did not help on clean synthetic renders (expect it to matter on real photos).
**Bug found & fixed**: negation window leaked across sentences ("No interview required. Pay a fee..." suppressed PAYMENT_REQUEST); now sentence-bounded (test added).
**Known issues**: CLIP not trained (needs Colab); `extracted.ocr_text` holds OCR text in memory — the API layer must not persist it unmasked.
EasyOCR first-run model download needed certifi's CA bundle on this Mac (handled via SSL_CERT_FILE default). Hindi OCR not enabled by default.

## Phase 6 — Fusion, calibration & explanation
**Built**: `ml/fusion/{cross_modal,fuser,explain,make_synthetic_aligned,train_fuser}.py`, `ml/pipeline.py` (`TriGuardPipeline`, `NoUsableInputError`),
`ml/common/plots.py` (reliability / PR / confusion figures), `tests/test_fusion.py` (24 tests incl. all 7 input combinations).
**Run**: `python -m ml.fusion.make_synthetic_aligned` (only if you have no real aligned set), `python -m ml.fusion.train_fuser`
(-> `models/fusion/fuser.joblib`, `reports/fusion_results.json`, `reports/figures/reliability_fusion_meta.png`, cache `data/processed/aligned_cache.joblib`).
`TriGuardPipeline().analyze(text=None, image=None, url=None) -> FinalResult`.
**Design**: cross-modal checks = company vs URL domain, free/mismatched HR e-mail vs named company, QR domain vs stated URL, text-vs-image company/salary/role mismatch.
URLs found in text/image/QR are auto-analysed (max 3, tagged `extracted.source="found_in_content"`). Fuser: weighted average over available modalities
(+ noisy-or cross-modal boost) or, once `train_fuser` has run on >=100 aligned rows, a logistic-regression meta-classifier on
[p_text,p_image,p_url,presence x3,#cross high,#cross other] with Platt calibration; Safe/Scam thresholds tuned on validation
(t_low keeps scam recall >=0.95; t_high = best F1). Payment / Aadhaar / OTP / UPI-QR high flags force at least "Suspicious".
Reasons: <=6, severity-ranked, topped up with LIME words / SHAP phrases. One failing analyzer degrades to unavailable instead of crashing.
**Metrics (SYNTHETIC aligned, 240 cases, test n~36 — NOT evidence)**: meta F1 1.000 vs weighted-average F1 0.50 (cross-modal flags perfectly separate synthetic cases).
**Bug fixed**: OpenCV `convexHull` assertion in QR detector crashed image analysis on some images (now guarded; regression test).
**Known issues**: all fusion numbers need the real aligned set (`data/aligned/aligned.csv`). Batch training uses Tesseract OCR while the API defaults to EasyOCR.
`explanation`/`extracted.ocr_text` stay in memory only; the API layer must strip or mask before storing.

## Phase 7 — Backend (FastAPI)
**Built**: `backend/app/{main,routes,db,security,errors,logging_conf,settings}.py`, `backend/benchmark_latency.py`, `tests/test_backend.py` (13 tests) + updated `tests/test_health.py`.
**Run**: `uvicorn backend.app.main:app --reload` (env: `TRIGUARD_DB_URL`, `TRIGUARD_CORS`, `TRIGUARD_RATE_LIMIT`, `TRIGUARD_TIMEOUT_S`, `TRIGUARD_LIVE_CHECKS=0`);
`python -m backend.benchmark_latency` -> `reports/backend_latency.md`. Tests: `pytest tests/test_backend.py`.
**API**: `POST /api/analyze` (multipart text?/url?/image?; 400 `no_input`, 413 too large, 415 bad image type by magic bytes, 422 validation/unusable input, 429 rate limit, 504 timeout);
`POST /api/feedback`; `GET /api/history` (last 20, anonymised); `GET /api/stats`; `GET /health` (also reports text model + fusion mode).
All errors share `{"error": {"code","message","request_id"}}`; JSON structured logs (no request content); `X-Request-ID`; CORS for Vite; Content-Length guard.
Pipeline is built once in the lifespan (injectable for tests); inference runs in a `ThreadPoolExecutor`. SQLite tables `analyses` (anonymised text, URL registered domain only,
sha256 image hash, masked reasons) and `feedback`; a test asserts raw phones/e-mails/Aadhaar and image bytes are nowhere in the DB. `ocr_text` is stripped from responses.
**Latency (real pipeline, Tesseract, live checks off)**: text ~20 ms, url ~130 ms (first call incl. SHAP init), image-containing combos ~600 ms. With EasyOCR ~2-4 s/image on CPU.
**Important fixes found while smoke-testing the real server**: (1) torch+LightGBM OpenMP clash on macOS made EasyOCR hang / LightGBM segfault depending on import order;
`ml/__init__.py` now imports LightGBM then torch and sets `torch.set_num_threads(1)` (verified: server starts, EasyOCR 2 s/image). (2) Platt calibration now L2-regularised (`PLATT_C`).
(3) A fuser trained on SYNTHETIC aligned data overfit template cross-modal flags (obvious scam -> 50%), so `train_fuser` saves it as `SYNTHETIC_fuser.joblib` and the pipeline keeps the weighted average until real `aligned.csv` exists.
**Known issues**: rate limiter and history are in-memory/global (single process, no auth) — fine for a demo, not multi-tenant. Models trained on SYNTHETIC data are brittle on out-of-domain text
(e.g. a short genuine posting still scores high), so live demo behaviour will only be meaningful after training on real EMSCAD. torch is single-threaded, so Bi-LSTM training/DistilBERT CPU inference are slower.

## Phase 8 — Frontend (React + Vite + TS + Tailwind)
**Built** (`frontend/src`): `api/{types,client}.ts` (typed client, timeouts, cancellation, backend error codes), `lib/{validation,reasons,useAnimatedNumber}.ts`,
components `Header, InputCards (text / drag-drop image with preview / URL), LoadingSteps, RiskGauge (animated, role=meter), ResultView (verdict badge, "Why?" reasons with severity icons + highlighted evidence,
cross-modal warning banner, per-modality cards, feedback, safety tips incl. cybercrime.gov.in / 1930, disclaimer)`, pages `Analyze, History, About` (method + usage stats), hash router, Vite dev proxy to the backend.
Branding: "TriGuard — Verify before you apply", navy base, green/amber/red verdicts (always paired with icon + text), mobile-first, skip link, aria-live progress, reduced-motion support.
**Run**: `cd frontend && npm install && npm run dev` (backend on :8000; override with `VITE_BACKEND` / `VITE_API_BASE`); `npm test`; `npm run build`.
**Tests**: 39 Vitest tests (validation, reason parsing, API client incl. timeout/cancel/errors, ResultView rendering + feedback, AnalyzePage gating/validation/submit/error). Build passes; `oxlint` shows 3 non-blocking warnings.
**Known issues**: no end-to-end browser test (checked one headless-Chrome screenshot only); history is global (no accounts); the About page's stats need the backend running.
