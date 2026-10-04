# PROGRESS

> **Layout change (after Phase 10):** the repo is now split into two self-contained folders, `backend/` and `frontend/`.
> Everything that was at the repo root (`ml/`, `tests/`, `reports/`, `scripts/`, `data/`, `models/`, `requirements.txt`, `pytest.ini`, `Dockerfile`, `.venv`) now lives in `backend/`;
> `backend/app/` is the FastAPI package (import `app.*`, no longer `backend.app.*`). Run all Python commands from `backend/`. Older entries below use the pre-split paths.

## WAITING FOR USER (final, consolidated)
Everything below currently runs on SYNTHETIC stand-ins (files prefixed `SYNTHETIC_`, flag files `data/processed/SYNTHETIC_*.flag`). Do these in order:

**A. Data to add**
- [ ] `data/raw/fake_job_postings.csv` — Kaggle "Real / Fake Job Posting Prediction" (EMSCAD).
- [ ] `data/raw/custom_text.csv` — Indian-scam test set `text,label,source` (template: `data/raw/custom_text_template.csv`). Currently 68 synthetic rows.
- [ ] `data/raw/urls/*.csv` — PhishTank and/or Kaggle `malicious_phish` CSVs.
- [ ] Real offer letters / posters in `data/images/scam/` and `data/images/genuine/` (delete the `SYNTHETIC_*.png` files first).
- [ ] `data/aligned/aligned.csv` — real multimodal cases (`case_id,text,image_path,url,label,notes`, see `data/aligned/README.md`; ≥100 labelled rows for the meta-classifier, more is better).
- [ ] `data/images/ocr_gt.csv` — `path,text` ground truth for real images (OCR comparison). Delete `data/images/ocr/SYNTHETIC_*` and the synthetic `ocr_gt.csv`.
- [ ] Team member names / supervisor in `README.md` (placeholder under "Team").

**B. Colab (GPU) models**
- [ ] `ml/notebooks/text_distilbert_colab.ipynb` → upload `data/processed/text_{train,val,test}.csv`, download `text_distilbert.zip`, unzip to `models/text_distilbert/` (auto-selected over TF-IDF+LR).
- [ ] `ml/notebooks/image_clip_colab.ipynb` → upload zipped `data/images/` + `data/processed/image_manifest.csv`, unzip result to `models/image_clip/` (visual weight switches from 0 to 0.4 automatically).
- [ ] MuRIL / XLM-R cells in the text notebook. `text_muril` and `text_xlmr` are already first in
  `TEXT_MODEL_PRIORITY` and are picked up automatically once `models/text_muril/` (or `text_xlmr/`)
  contains a `meta.json`; both load through the shared `_TransformerScorer`. Indian scam text is
  Hinglish/transliterated, so these should beat DistilBERT — this is now a priority, not optional.
- [ ] 150-200 real hand-labelled Indian scam texts are the highest-value single addition: they unlock
  the prior-shift transfer result (RESULTS.md section 4b), which is the project's main contribution.

**C. Re-run after A and B (in this order)**
```bash
rm -f data/processed/SYNTHETIC_*.flag models/fusion/SYNTHETIC_fuser.joblib
python -m ml.text.prepare_emscad && python -m ml.text.prepare_custom && python -m ml.url.prepare_urls && python -m ml.image.prepare_images
python -m ml.text.baselines && python -m ml.url.train
python -m ml.fusion.train_fuser          # real aligned.csv -> installs models/fusion/fuser.joblib (meta-classifier)
python -m ml.image.compare_ocr
python -m reports.robustness             # adversarial obfuscation table (RESULTS.md section 9)
python -m ml.text.evaluate && python -m reports.evaluate_all --with-bilstm-cv
python -m scripts.make_demo_assets       # then re-run the 4 demo cases and update docs/demo_script.md "Recorded" lines
```
**D. Verify**: Docker images were written but never built (daemon was off) — run `docker compose up --build` once. `frontend` end-to-end browser test not done.

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

## Phase 9 — Evaluation & ablation
**Built**: `reports/evaluate_all.py` (+ `reports/__init__.py`), `ml/common/plots.py` usage, `BiLSTMClassifier.load`, `tests/test_evaluation.py` (7 tests).
**Run**: `python -m reports.evaluate_all [--with-bilstm-cv]` -> `reports/RESULTS.md`, `reports/error_analysis.md`, `reports/evaluation.json`, `reports/figures/*.png`
(reliability_{text,url,fusion_meta}, cm_{text,url,fusion}, pr_{text,url,ablation}). Prerequisites: `ml.text.baselines`, `ml.url.train`, `ml.fusion.train_fuser`, `ml.image.compare_ocr`.
**Contents of RESULTS.md**: (1) 5-fold CV mean±std for TF-IDF+LR, Bi-LSTM (optional flag) and LightGBM; (2) base-paper comparison with ROC-AUC from probabilities vs from hard labels;
(3) out-of-fold ablation on the aligned set (text, URL, image, pairs, all three, all three + cross-modal, CV'd meta-classifier) with coverage; (4) EMSCAD -> Indian generalisation;
(5) OCR CER table; (6) ECE before/after Platt; (7) confusion matrices / PR curves; (8) error analysis (10 examples; padded with labelled near-misses if fewer real errors exist).
**Results (ALL SYNTHETIC — do not quote)**: CV F1 TF-IDF+LR 0.981±0.010, Bi-LSTM 0.969±0.014, LightGBM 1.000. Ablation F1: text 0.50, URL 0.84, image 0.41, text+URL 0.78, all three 0.78, +cross-modal 0.78, meta 0.99.
The only qualitative finding worth keeping: ranking quality (PR/ROC-AUC) holds on the Indian set while F1 collapses at the EMSCAD-tuned threshold, i.e. a calibration/threshold shift, which is exactly what real Indian data should be used to study.
**Known issues**: every number needs regenerating after real EMSCAD / URL / aligned data are added (DistilBERT row stays "pending" until its Colab model is placed in `models/text_distilbert/`). Hard-label AUROC comparison is computed for our models, not for the paper (its predictions are unavailable).

## Phase 10 — Testing, deployment & docs
**Built**: `tests/test_edge_cases.py` (44 cases: empty / 1 MB text, non-English & control characters, 16 malformed URLs, 6 corrupt images, decompression bomb, API never 5xx);
decompression-bomb guard in `ml/image/preprocess.py` (reads PNG/JPEG header before decoding); `TRIGUARD_OCR_BACKEND` setting;
`Dockerfile` (python:3.11-slim, CPU torch, tesseract, zbar, non-root), `frontend/Dockerfile` + `nginx.conf`, `docker-compose.yml` (models mounted read-only, SQLite + EasyOCR cache volumes), `.dockerignore`;
`docs/deploy.md` (HF Spaces / Render + Vercel / Netlify), `docs/demo_script.md`, `scripts/make_demo_assets.py` + `docs/demo_assets/fake_offer_letter_with_qr.png`, final `README.md` (mermaid architecture, setup, training, API, results, limitations, ethics, team).
**Final test run**: `pytest` 232 passed; `frontend` Vitest 39 passed, production build OK.
**Known issues**: Docker build not verified (daemon not running here; only `docker compose config` validated). Demo case 4 (genuine posting) scores "Suspicious ~53%" with the synthetic stand-in text model — documented in the demo script.

## Delivered-items checklist
- [x] Phase 1 setup: structure, schemas, config, metrics, FastAPI health, Vite/React/Tailwind scaffold
- [x] Phase 2 data prep: EMSCAD/custom/URL/image prep, dedupe-before-split, anonymiser, synthetic letters, aligned template (+ tests)
- [x] Phase 3 text: red-flag rules, TF-IDF+LR, Bi-LSTM reproduction, DistilBERT Colab notebook, TextAnalyzer (calibrated, LIME), text_results.md
- [x] Phase 4 URL: lexical features, impersonation (50 companies), live checks (rules only, mocked in tests), LightGBM, SHAP analyzer
- [x] Phase 5 image: preprocessing, EasyOCR + Tesseract, QR (pyzbar + OpenCV), CLIP notebook + loader (weight 0 until trained), ImageAnalyzer (in-memory), OCR comparison
- [x] Phase 6 fusion: cross-modal checks, weighted-average + meta-classifier fuser, recall-tuned thresholds, reliability diagrams, explanations, pipeline (7 combinations tested)
- [x] Phase 7 backend: analyze/feedback/history/stats/health, SQLite (anonymised), rate limit, JSON logs, error shape, latency report
- [x] Phase 8 frontend: analyze/result/history/about, gauge, evidence highlighting, feedback, safety tips, a11y, Vitest
- [x] Phase 9 evaluation: CV, base-paper comparison, ablation, generalisation, ECE, figures, error analysis, RESULTS.md (synthetic-flagged)
- [x] Phase 10 hardening & docs: edge cases, Docker, deploy guide, README, demo script
- [ ] Real data, GPU models and real metrics — see WAITING FOR USER above

## Post-split improvements (user-requested)
**Why**: the user asked for the "Why?" reasons to be explained clearly and for the History page to show full detail on click.
**Built**
- `frontend/src/lib/flagHelp.ts`: a plain-English explanation (title / what we found / why it matters / what to do) for each of the ~30 red-flag codes,
  plus two fallbacks for model-driven notes (LIME words, SHAP URL features) and a generic one for unknown codes.
- `frontend/src/components/ReasonDetails.tsx`: each reason now has a "What does this mean?" toggle (`aria-expanded`/`aria-controls`) showing that explanation;
  `ResultView.ReasonList` delegates to it and exports `allFlags()`.
- `GET /api/history/{id}` (`app/routes.py`): full masked record — submitted text, URL domain, whether an image was analysed, verdict, every reason,
  per-modality breakdown, cross-modal flags, feedback given, latency.
- `Analysis.result_json` column (`app/db.py`) storing the masked `FinalResult`, plus `_add_missing_columns()`, a forward-only
  `ALTER TABLE ... ADD COLUMN` migration so existing SQLite files keep working (old rows report `detail_available: false`).
- `app/security.py: mask_json()` — recursive PII masking that also strips URL query strings (UPI/`token=` parameters) before anything is stored.
- `frontend/src/pages/HistoryPage.tsx`: list items are now buttons opening a full report view (gauge, what was submitted, expandable reasons, breakdown, feedback) with a back link.
- Clearer wording for `FORM_OR_MESSENGER_LINK` and for the model-evidence notes ("the AI model's score was driven mostly by the words … Single words are only hints, not proof.").
**Tests**: backend 237 passed (3 new: history detail masking, 404 + legacy rows, DB migration); frontend 45 passed (6 new: explanation toggle, fallbacks, history detail view).
**Verified live**: `POST /api/analyze` then `GET /api/history/1` returns the full masked record (`[EMAIL]` in place of the address, `forms.gle` as the domain).

## Methodology hardening — prediction quality, transfer & robustness (2026-10-04)

Addresses the flaws found while reviewing the project rather than adding features. Six changes, all
backend-side plus a small frontend surface. Nothing here needs real data to run, but every number it
produces is still 🧪 SYNTHETIC until `data/raw/` is filled in (see WAITING FOR USER).

### What was built

**1. Cross-domain prior/threshold transfer — `ml/common/prior_shift.py` (new)**
The biggest real finding was already sitting in `reports/RESULTS.md` §4: on the Indian set the text
model scored ROC-AUC 1.000 but precision 0.544 / recall 1.000 / **MCC 0.000** — it was labelling
*every* row scam. The ranking transfers; only the cut-point does not. Two corrections, cheapest first:
- `correct_for_prior()` — shifts the logit by the prior log-odds difference
  `log(πt/(1-πt)) - log(πs/(1-πs))`. Monotone, so AUCs are provably unchanged (tested).
- `fit_transfer()` — when ≥ `MIN_TARGET_CALIBRATION_ROWS` (40) labelled target rows exist, re-fits
  Platt on half and tunes the threshold there, holding the other half back for honest reporting.

Result (new §4b of RESULTS.md), naive → adapted MCC: tfidf_lr 0.000 → 1.000, bilstm 0.000 → 0.675,
hybrid_lgbm 0.000 → 0.521. This is the strongest contribution over the base paper and is now a
first-class report section.

**2. Rules as model features — `ml/common/behavioral.py` (new) + `TfidfRulesLGBM`**
24 named behavioural features (rule firings, severity sum, contact-channel structure, free-email
share, messenger/shortener links, money mentions, digit/upper share) now feed the model instead of
only boosting the probability afterwards with hand-set weights. New `TfidfRulesLGBM` model in
`ml/text/models.py` = TF-IDF → TruncatedSVD(150) ⊕ behavioural features → LightGBM, trained to
`models/text_hybrid_lgbm/` and auto-selected by `TextAnalyzer`. `behavioural_importance()` gives
report-ready gain per named behaviour.
> On the synthetic data every behavioural feature gets **zero** gain — `svd_0` alone separates the
> templates (only 12 of 174 features are used at all). That is further evidence of the circularity
> noted below, not a defect of the feature set; it should be re-read on real data.

**3. Adversarial robustness — `ml/common/robustness.py` + `reports/robustness.py` (new)**
Six label-preserving obfuscations (`spacing`, `homoglyph`, `obfuscate_contact`, `spell_amounts`,
`synonym_swap`, `all_combined`) applied to scam rows only. Headline result, now §9 of RESULTS.md:
on the Indian set the rule layer fires on **100% of scams clean but only 27% under `all_combined`**
(5 flags → 0 on a single example). Regex-only detection is evadable; this justifies learned features.

**4. Missing-modality imputation + abstention — `ml/fusion/fuser.py`**
- `feature_vector()` previously substituted `p = 0.0` for an absent modality, which reads as
  "confidently genuine" and dragged every image-less case toward Safe. It now imputes at the
  training-set scam prior, stored on `FuserArtifact.missing_prior` so scoring matches training.
- `evidence_strength()` + `should_abstain()` + `Fuser.decide_full()` → new non-breaking
  `FinalResult.abstained` / `evidence_strength` fields. A high-severity flag never abstains.
- `RECALL_TARGET_LOW` 0.95 → 0.90 and a new `SAFE_THRESHOLD_FLOOR` (0.10): the old target drove
  `t_low` to its floor so almost nothing could ever be labelled "Safe".

**5. Bootstrap confidence intervals — `ml/common/bootstrap.py` (new)**
Stratified percentile bootstrap (2000 resamples). The aligned set is n=240 with ~14 positives per
test fold, so point estimates like "precision 1.000" were not reportable. New §10 of RESULTS.md
carries intervals for every ablation configuration; `text_baselines.json` and `fusion_results.json`
now embed a `ci` block.

**6. Code-mixed text encoders + OCR caveat**
`TEXT_MODEL_PRIORITY` is now `text_muril → text_xlmr → text_distilbert → text_hybrid_lgbm →
text_tfidf_lr`. Indian scam text is Hinglish/transliterated, which DistilBERT handles poorly.
`_DistilBertScorer` was generalised to `_TransformerScorer` (handles all three; only DistilBERT's
`token_type_ids` quirk is special-cased), and unknown priority entries are skipped with a warning
instead of raising. The OCR table in RESULTS.md §5 now carries an explicit "do not quote the backend
ranking" warning — those images are rendered from fonts, so OCR reads back the glyphs that were
drawn, and the EasyOCR/Tesseract ordering commonly inverts on real photos.

### Files
New: `ml/common/{prior_shift,behavioral,robustness,bootstrap}.py`, `reports/robustness.py`,
`tests/test_{prior_shift,behavioral,robustness,bootstrap,fusion_abstain}.py`.
Changed: `ml/common/{config,schemas}.py`, `ml/text/{models,analyzer,baselines}.py`,
`ml/image/analyzer.py`, `ml/fusion/{fuser,train_fuser}.py`, `ml/pipeline.py`,
`reports/evaluate_all.py`, `frontend/src/api/types.ts`,
`frontend/src/components/ResultView.tsx` (new `ProvisionalNotice`), `frontend/src/pages/HistoryPage.tsx`.

### How to run
```bash
cd backend
python -m ml.text.baselines        # also trains models/text_hybrid_lgbm + the transfer report
python -m reports.robustness       # -> reports/robustness.{md,json}
python -m ml.fusion.train_fuser    # learns + stores missing_modality_prior
python -m ml.text.evaluate && python -m reports.evaluate_all   # -> RESULTS.md §4b, §9, §10
```

**Tests**: backend 313 passed (76 new across 5 files); frontend 49 passed (4 new for
`ProvisionalNotice`). Three existing tests were updated to the new intended behaviour:
`test_fusion.py` (prior imputation replaces 0.0), `test_evaluation.py` (`ablation()` now also
returns CI rows), `test_backend.py` (`/health` may report any model in `TEXT_MODEL_PRIORITY`).

### Known issues / still open
- **The synthetic metrics remain circular, and no amount of synthetic data fixes that.**
  `ml/fusion/make_synthetic_aligned.py` branches on the label in all three modalities: scam URLs
  always get `.online/.xyz/.top/.site/.work/.info` or a typo-squat while genuine URLs always get the
  real company domain, and `synth_letters.make_letter(bool(label), …)` branches on the label too. The
  label is independently recoverable from each modality, which is the whole explanation for
  "URL only → precision 1.000" and the near-perfect meta-classifier. Do not invest further in the
  generators; §1–§4 and §10 only become meaningful with real data.
- **Rules and synthetic data share an author.** `S_DESC_OBVIOUS` in `ml/common/synthetic.py` maps
  almost 1:1 onto the regexes in `ml/common/red_flags.py`, so "model + rules" gains are unmeasurable
  on synthetic text.
- **The URL module solves a different problem than it is deployed in.** It trains on PhishTank /
  `malicious_phish` (credential-harvesting URLs), but real fake-job operations use WhatsApp numbers,
  Telegram invites, Google Forms and clean-looking `.in` domains. Expect near-zero real recall plus
  false alarms on legitimate ATS redirects. `suspicious_tld` will carry far less weight than §3 implies.
- Prior correction assumes **label shift** (class-conditionals fixed, only P(y=1) moves). Indian
  WhatsApp scam text vs. EMSCAD corporate postings is partly *covariate* shift, which only the
  target-recalibration branch absorbs — so collecting ~50–100 labelled Indian rows matters more than
  any further modelling.
- Data priority unchanged and now sharper: **150–200 real hand-labelled Indian scam texts** unlock
  §4/§4b, which is the actual thesis. Real aligned multimodal cases are the most expensive and least
  necessary — the fusion can honestly be reported as a calibrated weighted average with ablations.
- `MISSING_MODALITY_PRIOR` default (0.5) is only used until `train_fuser` runs; the learned prior is
  what ships in the artifact.
- Transformer artifacts (`text_muril`, `text_xlmr`) are still Colab work — the notebooks currently
  only cover DistilBERT.

## Fix: genuine offer letters scored as scams (2026-10-04)

**Reported bug.** A real, clean offer letter ("We are pleased to offer you the Relationship Manager
position at Wealth Bank … annual compensation USD 72,000") was scored **P(scam) = 0.715** by the
hybrid model and **0.998** by TF-IDF+LR. Zero red flags fired, so this was entirely the trained model.

**Root cause — a label confound, not a tuning problem.** Offer/selection wording existed *only* in
the scam class of every training set:

| contains | genuine | scam |
|---|---|---|
| "offer letter" (Indian set) | **0** | 12 |
| "selected" | **0** | 12 |
| "congratulations" | **0** | 22 |
| offer-letter phrasing (EMSCAD-shaped set) | **0** | 0 |

Every offer/selection document in the data was labelled scam, so the models learned
`offer letter ⇒ scam` and were classifying **document type, not legitimacy**. Proof: the same job,
company and salary written as a *job posting* scored 0.040 while written as an *offer letter* it
scored 0.998 — and LIME's top drivers on the offer letter were function words ("In", "Your", "We")
at weights of ±0.02, i.e. no real evidence at all.

This mattered more than the synthetic-data problem because (a) EMSCAD is job-postings-only too, so
real data would **not** have fixed it, and (b) the product's flagship input is an offer-letter image,
which OCRs straight into this model.

### Fix 1 — balance the register across both classes (`ml/common/synthetic.py`)
Offer-letter register now appears in *both* classes, so it carries no label information and only
*behaviour* (a fee, an identity-document demand, a free-mail recruiter, no interview) separates them:
- `G_OFFER` + `GENUINE_OFFER_SHARE` (0.18) — genuine offer/appointment letters with manager, CTC,
  start date, background-check and signature clauses.
- `S_OFFER` + `SCAM_OFFER_SHARE` (0.35) — fake offer letters that mimic the genuine register closely
  and differ only in the behavioural cues.
- Four genuine offer letters added to `_CUSTOM_GENUINE` for the Indian set.
- `_fill()` gained `{co} {dom} {date} {mgr} {mgr_title} {ctc}` placeholders.

Balance after the fix (EMSCAD-shaped, genuine / scam): "pleased to offer" 365/115,
"offer letter" 241/94, "selected" 148/42, "congratulations" 141/115.

> A first attempt only added *genuine* offer letters, which simply flipped the confound
> (385/0 the other way). Both classes are needed — that is the actual invariant, and it is now
> pinned by tests.

### Fix 2 — out-of-distribution guard (`ml/common/novelty.py`, new)
Rebalancing fixes this confound; the general failure (a confident score on a register never seen)
stays. `NoveltyDetector` measures the **share of out-of-vocabulary word tokens**, and
`shrink_toward_prior()` pulls the score back toward the prior in proportion to how far past the
cut-off it sits; `TextAnalyzer` marks `extracted["out_of_distribution"]` and the fuser abstains on it.

The cut-off is an **absolute** 0.35, deliberately not a percentile of the training distribution: a
templated corpus has a 99th-percentile OOV share of ~0.015, so a self-calibrated threshold flags
nearly every real input. Measured separation: real job content 0.08–0.15, cooking recipe 0.68,
Hinglish SMS 0.56, German 0.92.

> A first implementation used cosine distance in a TruncatedSVD space. It was discarded: it flagged
> an ordinary job posting as novel (0.34) while *missing* a cooking recipe (0.21), because the
> leading components were dominated by function words. The OOV share is simpler, interpretable and
> separates cleanly.

### Result
| Input | before | after |
|---|---|---|
| Genuine offer letter (the reported case) | 0.715 / 0.998 | **0.022** → verdict **Safe, risk 2.2** |
| Fake offer letter (same company/role) | — | 0.899 → **Likely Scam, risk 97.2** |
| Genuine job posting | 0.040 | Safe, risk 13.4 |
| Cooking recipe | confident score | **abstains** (novelty 0.68) |
| German offer letter | confident score | **abstains** (novelty 0.94) |

### Files
New: `ml/common/novelty.py`, `tests/test_novelty.py`, `tests/test_offer_letter_confound.py`.
Changed: `ml/common/{synthetic,config}.py`, `ml/text/{analyzer,baselines}.py`, `ml/fusion/fuser.py`.

### How to run
```bash
cd backend
python -m ml.text.prepare_emscad && python -m ml.text.prepare_custom   # regenerate balanced data
python -m ml.text.baselines        # retrains models + fits models/text_novelty/
python -m ml.fusion.train_fuser && python -m reports.robustness
python -m ml.text.evaluate && python -m reports.evaluate_all
```

**Tests**: backend 346 passed (33 new: 19 novelty, 14 offer-letter confound). The confound tests
assert the *data property* ("neither class may own the offer-letter register"), so regenerating the
corpus cannot silently reintroduce the bug.

### Known issues
- Target-transfer MCC moved with the rebalanced data (tfidf_lr naive 0.164 → adapted 0.721;
  hybrid 0.000 → 1.000; bilstm 0.236 → 0.557). Still synthetic, so treat the direction, not the value.
- The novelty vocabulary is only 429 words because the synthetic corpus is templated. On real
  EMSCAD it will be tens of thousands and the margin between real job content and true OOD widens;
  `NOVELTY_THRESHOLD` should be re-checked once real data lands.
- **The same confound audit has not been done for the other classes of input.** Worth checking
  before the viva whether any other register (resume, internship certificate, poster) is
  single-class in the data. `tests/test_offer_letter_confound.py` is the template for that check.
- The image path inherits this fix automatically (OCR text feeds `TextAnalyzer`), but
  `ml/image/synth_letters.py` still renders scam and genuine letters from label-branched templates,
  so the *visual* channel retains the circularity noted in the previous entry.

## URL detector: placeholder domains & employer recognition (2026-10-04)

### Why
Reviewing the URL modality against the signals it ought to cover showed two gaps. Reserved
placeholder addresses (`.example`, `.test`, `.invalid`, `.localhost`) were almost invisible:
`http://hiring-portal.example/apply` scored 0.127 with a single low `NO_HTTPS` flag, because
`tldextract` returns an empty public suffix for these names, so the `suspicious_tld` feature —
the one check that should have caught them — evaluated to False. Separately, there was no signal
at all for "this domain belongs to no employer we can recognise": `COMPANY_URL_MISMATCH` only
fires when the text or image names a company that is already in `company_domains.csv`, so a
standalone link to an unknown domain produced nothing.

### What was built
- `is_reserved_domain(host)` in `ml/url/features.py` (RFC 2606 / RFC 6761 names, plus the reserved
  `example.com|net|org`) → **`PLACEHOLDER_DOMAIN`** (high). A link that cannot resolve to any real
  site is either a template left unfilled or a fabricated address.
- `official_domains()` in `ml/url/impersonation.py`: the set of every registered domain in
  `company_domains.csv`, cached.
- `recognition_flags()` in `ml/url/analyzer.py` → **`UNRECOGNIZED_EMPLOYER_DOMAIN`** (low). Kept low
  on purpose: most legitimate small employers are not in the CSV either, so this is corroborating
  evidence, not a verdict. It is suppressed when a more specific flag already explains the host
  (impersonation, placeholder, raw IP, shortener, form/chat link, ATS/job board, official domain)
  and when WHOIS shows the domain is at least `ESTABLISHED_DOMAIN_DAYS` (730) old — without that
  gate it would fire on nearly every result and become noise.
- Plain-English entries for both codes in `frontend/src/lib/flagHelp.ts` (what / why / action).
- `features._is_ip` renamed to the public `features.is_ip`, since the analyzer now needs it.

Both are **rule signals only**; `FEATURE_NAMES` is unchanged, so saved models stay loadable and no
retraining is needed. A reserved TLD would be an all-zero column in training anyway — no real-world
URL corpus contains one.

### Files
Changed: `ml/url/{features,impersonation,analyzer}.py`, `ml/common/config.py`
(+`ESTABLISHED_DOMAIN_DAYS`), `tests/test_url_module.py`, `frontend/src/lib/flagHelp.ts`.

### Effect
`http://hiring-portal.example/apply` 0.127 → 0.476 (high flag).
`https://randomcorp-hr-team.online/apply?id=22` 0.127 → 0.153 (the low note).
`https://www.tcs.com/careers`, `bit.ly`, `forms.gle` and IP hosts are unchanged — no new flag fires
where one already applied.

### How to run
```bash
cd backend && .venv/bin/python -m pytest tests/test_url_module.py -q
cd frontend && npx tsc --noEmit && npm test -- --run
```

**Tests**: backend 358 passed (12 new: reserved-domain parametrisation, placeholder severity,
employer recognition, the suppression matrix, CSV coverage). Frontend 49 passed, clean typecheck.

### Known issues
- `UNRECOGNIZED_EMPLOYER_DOMAIN` is only as good as `company_domains.csv`. A real employer outside
  that list gets the note; that is why it is low severity and why the message says "could not be
  matched" rather than asserting the domain is fake.
- Employer recognition is still list-based. Nothing checks whether a domain *behaves* like a company
  site (MX records, age + traffic, certificate organisation name). The TLS certificate's
  `organizationName` is already fetched into `LiveSignals.ssl_issuer` and would be the cheapest next
  signal — comparing it to the named company needs no extra lookup.
- Redirect following is still limited to the ~15 known shorteners (HEAD only, 3 hops). A redirect
  chain on an arbitrary domain is never followed, by design: the project rule is never to open a
  suspicious URL.

## Frontend redesign (2026-10-04)

Full visual rebuild of the React app. Backend untouched. Light theme only, and the
"How it works" page was dropped on request.

### Design foundation
- `src/index.css` is now a token system: surfaces, ink, lines, accent, risk status and a separate
  `--viz-*` group for chart colour, so data colour never drifts with a brand tweak.
- `src/components/ui.tsx` — shared primitives (`Card`, `Button`, `SectionTitle`, `Chip`, `StatTile`,
  `Accordion`, `Skeleton`, `EmptyState`, `ErrorNote`), replacing inline
  `rounded-xl border border-slate-200 bg-white p-4 shadow-sm` repeated across pages.
- `src/lib/riskTokens.ts` — risk/severity tone maps + `LABEL_SUMMARY`, kept out of component files.

### Charts — inline SVG, no chart library
`src/components/charts.tsx`: `RiskBandMeter`, `ModalityBars`, `DivergingWords`, `LabelDistribution`,
`CountBars`, `DataTable`. Five simple forms did not justify a dependency; the app still ships with
only `react` + `react-dom`.

Colour was **validated, not eyeballed** (`validate_palette.js`): the diverging pair
blue `#2a78d6` ↔ red `#d03b3b` passes all six checks on the `#ffffff` surface. Rules held:
- one hue for every bar of a single measure — never a value-ramp on nominal categories;
- status colours only where they mean status, and **always with an icon + the verdict name**:
  status red↔green sit at CVD ΔE 4.1, so colour alone could not distinguish them;
- diverging pair with a neutral gray midpoint, solid hairline rules (never dashed);
- 2px surface gaps between fills, 4px rounded data-ends anchored to the baseline;
- selective direct labels (≤3 series are each labelled), and a table view for every chart.

### Pages
- **Analyze** — hero, "Try an example" chips, input-progress checklist, clearer disabled state.
  The examples are the confound pair from the previous entry: *Genuine offer letter* and
  *Fake offer letter* are the same company, role and salary and differ only in behaviour, so the
  demo shows the model judging behaviour rather than document type.
- **Result** — verdict hero (gauge + badge + summary) beside a **band meter that shows where the
  decision thresholds are**, which a bare gauge cannot; per-input bar chart with a table toggle;
  key facts (domain, domain age) inline; a new **Confidence & coverage** panel surfacing
  `abstained`, inputs used and the novelty/out-of-distribution signal; LIME words as a diverging
  chart.
- **History** — verdict filter chips with counts, search over text/domain/reason, input-type icons,
  relative timestamps.
- **Stats** — NEW. Wires up `getStats()`, which existed in `api/client.ts` but no page ever called.
  KPI tiles, verdict distribution, input combinations, plus a note that these are usage counts and
  not model accuracy.

### Bugs found and fixed while rendering the pages
Screenshots were taken with headless Chrome at several widths rather than trusting the markup:
- `InputCards.tsx`, `ReasonDetails.tsx` and `LoadingSteps.tsx` still hardcoded `bg-white` /
  `slate-*`; the "Why?" reasons were dark-on-white and unreadable. All three are tokenised now, and
  no hardcoded light colour remains in any component.
- `useAnimatedNumber` could return **negative** values: `t` was capped at 1 but never floored at 0,
  so a rAF timestamp preceding the captured `performance.now()` start eased through negatives and
  the gauge rendered "-3%". Now clamped both ways, with a regression test.
- `RiskGauge` painted a stray dot at zero progress (a rounded cap on a zero-length dash).
- The Analyze grid left the link card stranded in a one-third column; it now spans the row.
- A disabled "Analyze" button still read as the call to action.
- The Stats latency tile said "Median response" while the backend returns a mean.

Checked at 390px in a real 390px-wide iframe: no horizontal overflow (an apparent overflow in
`--window-size=390` screenshots is a headless artifact — macOS enforces a minimum window width).

**Tests**: frontend 82 passed (`charts.test.tsx` 17 new, `StatsPage.test.tsx` 7 new,
`useAnimatedNumber.test.ts` 3 new, plus example-chip and provisional-verdict cases). `tsc` clean,
`vite build` clean, oxlint clean apart from the pre-existing fast-refresh warnings. Backend: 358,
unchanged by this work.

### Known gaps
- **Light theme only** — the dark palette was built and validated against a `#0f1b2d` surface, then
  removed on request. Restoring it means re-adding the two dark scopes in `index.css`; the
  `--viz-*` dark steps are recorded in this entry's git history.
- **The "How it works" page is gone**, and with it the pipeline diagram and the written
  limitations/privacy section. The synthetic-data caveat survives on the Stats page and the
  disclaimer survives on the Result page, but the fuller honest-limitations text no longer has a
  home in the UI — worth relocating before the viva.
- Charts have hover/focus affordances but no floating tooltip; with ≤8 marks every value is
  direct-labelled or in the table view instead.
