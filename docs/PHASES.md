Read CLAUDE.md and PROGRESS.md first. Phase 1 (setup) is already complete.

## How to work
1. First, save this entire message verbatim to docs/PHASES.md. Add a status line under
   every phase heading: "Status: TODO". This file is the source of truth.
2. Then work through Phases 2 → 10 strictly in order. For EACH phase:
   a. Set its status to "IN PROGRESS" in docs/PHASES.md.
   b. Build everything listed for that phase.
   c. Write tests, run them, fix failures until they pass.
   d. Append a summary to PROGRESS.md: what was built, files, how to run, metrics,
      known issues.
   e. Git commit with message "Phase N: <name>".
   f. Set status to "DONE" and move straight on to the next phase without waiting for me.
3. Do NOT stop between phases. Stop only if something truly blocks you and no
   workaround exists.
4. Missing data rule: if a dataset I'm supposed to provide is not in data/raw/ yet, generate
   a small realistic synthetic sample (clearly marked SYNTHETIC in filenames and logs) so
   the code runs end to end, add it to a "WAITING FOR USER" list at the top of PROGRESS.md,
   and continue.
5. GPU rule: you cannot train on a GPU here. For DistilBERT and CLIP, create the Colab
   notebook, then continue using the CPU fallback model (TF-IDF+LR for text; skip the CLIP
   visual score with weight 0 for images). Everything must work with the fallbacks and
   automatically use the better model once its files appear in models/.
6. If you resume after an interruption, read docs/PHASES.md and PROGRESS.md, then continue
   from the first phase that is not DONE.
7. Keep the shared ModalityResult / FinalResult interface from CLAUDE.md unchanged.

---------------------------------------------------------------------------------------

## Phase 2: Data collection & preprocessing
Status: DONE

TEXT
- ml/text/prepare_emscad.py: load data/raw/fake_job_postings.csv (Kaggle "Real / Fake Job
  Posting Prediction"). Merge title, company_profile, description, requirements, benefits
  into "text"; strip HTML, #URL_...# placeholders, extra whitespace; remove exact and
  near-duplicates (normalized-text hash) BEFORE splitting; stratified 70/15/15, seed 42
  -> data/processed/text_{train,val,test}.csv; print class balance and duplicates removed.
- ml/text/prepare_custom.py: load data/raw/custom_text.csv (text,label,source) as a separate
  Indian-scam test set; include a template CSV with 5 example rows; anonymize() masks phones,
  emails, Aadhaar/PAN-like patterns.
URL
- ml/url/prepare_urls.py: load CSVs in data/raw/urls/ (PhishTank-style and Kaggle
  malicious_phish formats), map to scam=1/benign=0, dedupe, stratified split
  -> data/processed/url_{train,val,test}.csv.
IMAGE
- data/images/{scam,genuine}/ and ml/image/prepare_images.py building a manifest
  (path,label,source) with a stratified split.
- ml/image/synth_letters.py: generate 200 genuine-style and 200 scam-style synthetic offer
  letters (fee request, gmail contact, urgency) with varied fonts, layouts and noise.
ALIGNED SET
- data/aligned/aligned_template.csv (case_id,text,image_path,url,label,notes) + README.
TESTS: cleaning, dedupe, anonymize, split stratification.

## Phase 3: Text module
Status: DONE

- ml/common/red_flags.py: rule-based RedFlags (code, message, severity, evidence) for
  payment/registration/training fee or deposit (incl. ₹/Rs amounts), urgency phrases,
  "no interview"/guaranteed selection, unrealistic pay, WhatsApp/Telegram-only contact,
  free email domains for HR, requests for Aadhaar/PAN/bank details/OTP. Also extract urls,
  emails, phones, company names. Thorough positive/negative unit tests.
- ml/text/baselines.py: (a) TF-IDF word 1-2 + char 3-5 grams + Logistic Regression
  (class_weight=balanced); (b) reproduction of the base paper's Bi-LSTM on text only.
  Train on train, tune threshold on val, report test -> reports/text_baselines.json.
- ml/notebooks/text_distilbert_colab.ipynb: fine-tune distilbert-base-uncased, weighted
  loss, early stopping on val PR-AUC, export zip for models/text_distilbert/; optional
  MuRIL cell.
- ml/text/analyzer.py: TextAnalyzer.analyze(text) -> ModalityResult; best available model
  with fallback, calibration fitted on val, red flags, LIME top words.
- Evaluate on EMSCAD test and the custom Indian test set; reports/text_results.md with a
  comparison vs the base paper (Bi-LSTM F1 0.86, recall 0.83).

## Phase 4: URL module
Status: IN PROGRESS

- ml/url/features.py: length, counts of . - _ / ? = @ digits, subdomain depth, IP host,
  suspicious TLDs, entropy, shortener detection, keyword hits (login, verify, job, hiring,
  apply, offer), platform links (forms.gle, docs.google.com/forms, t.me, wa.me).
- ml/url/impersonation.py + data/company_domains.csv (~50 Indian/global employers);
  rapidfuzz lookalike detection (e.g. tcs-hiring.online vs tcs.com).
- ml/url/live_checks.py: WHOIS domain age, SSL presence/issuer, shortener expansion via
  HEAD only; 3s timeouts, caching; used as inference-time RULE signals only, never as
  training features. Never render pages.
- LightGBM on lexical features (class weights, val threshold, calibration)
  -> models/url_lgbm/, reports/url_results.json.
- ml/url/analyzer.py: URLAnalyzer.analyze(url) -> ModalityResult with red flags and SHAP
  top features. Mock all network calls in tests.

## Phase 5: Image & OCR module
Status: TODO

- ml/image/preprocess.py: grayscale, denoise, deskew, adaptive threshold, resize.
- ml/image/ocr.py: EasyOCR (en, optional hi) with Tesseract as an alternative backend;
  ml/image/compare_ocr.py measures character error rate on data/images/ocr_gt.csv
  -> reports/ocr_compare.md.
- ml/image/qr.py: pyzbar decoding with OpenCV fallback.
- ml/image/visual.py + ml/notebooks/image_clip_colab.ipynb: CLIP ViT-B-32 embeddings +
  Logistic Regression genuine-vs-scam -> models/image_clip/.
- ml/image/analyzer.py: ImageAnalyzer.analyze(image_bytes) -> ModalityResult: OCR ->
  TextAnalyzer -> red flags -> QR decode -> weighted combine of visual and OCR-text
  probabilities (weights in config); put urls/emails/phones/company/QR links in `extracted`.
  Never write the image to disk. Tests with generated images, including one with a QR code.

## Phase 6: Fusion, calibration & explanation
Status: TODO

- ml/fusion/cross_modal.py: company in text/image vs URL domain; HR email domain vs
  company's real domain; QR link domain vs stated URL; salary/role mismatch between text
  and image; any URL found in text or image is auto-sent to URLAnalyzer.
- ml/fusion/fuser.py: (a) weighted average over AVAILABLE modalities; (b) logistic-
  regression meta-classifier on [p_text, p_image, p_url, presence flags, cross-modal flag
  counts] trained on data/aligned/, falling back to (a) if too few rows. High-severity flags
  (payment, Aadhaar/OTP request) force at least "Suspicious".
- Final calibration and 3-level labels with thresholds tuned on validation favouring
  recall; reliability diagram -> reports/figures/.
- ml/fusion/explain.py: rank flags by severity, add top SHAP/LIME items, plain-English
  reasons, max 6.
- ml/pipeline.py: TriGuardPipeline.analyze(text=None, image=None, url=None) -> FinalResult;
  load models once. End-to-end tests for all 7 input combinations.

## Phase 7: Backend (FastAPI)
Status: TODO

- POST /api/analyze (multipart text?, url?, image? ≤5 MB png/jpg/webp; 400 if no input),
  POST /api/feedback, GET /api/history (last 20, anonymized), GET /api/stats, GET /health.
- Pipeline loaded once in lifespan; inference in a thread pool.
- SQLAlchemy + SQLite: Analysis (id, created_at, inputs_present, anonymized_text,
  url_domain, label, risk_score, reasons JSON, latency_ms), Feedback. Store only an image hash.
- CORS for Vite, size limits, validation, simple rate limit, structured logging,
  consistent error responses.
- pytest with TestClient (mocked pipeline + one real integration test); report average
  latency per input combination.

## Phase 8: Frontend (React + Vite + TS + Tailwind)
Status: TODO

- Branding: "TriGuard", tagline "Verify before you apply"; navy base, green/amber/red verdicts;
  clean, trustworthy, mobile-first.
- Analyze page: three input cards (text textarea; image drag-and-drop with preview and
  validation; URL input with validation); any combination allowed; button disabled until
  one input exists; step-by-step loading messages.
- Result view: verdict badge, animated 0–100% risk gauge, "Why?" reasons with severity
  icons and highlighted evidence, per-modality breakdown cards, prominent cross-modal
  warnings, "Was this correct?" feedback, safety tips (never pay fees, verify on official
  careers pages, report at cybercrime.gov.in / 1930), disclaimer.
- History page and Stats/About page explaining the method.
- Typed API client matching FinalResult, error handling, timeouts; accessibility and
  responsive layout; Vitest tests for validation and result rendering.

## Phase 9: Evaluation & ablation
Status: TODO

- reports/evaluate_all.py: 5-fold stratified CV (mean ± std) for text and URL; base paper
  comparison (reported Bi-LSTM vs our reproduction vs DistilBERT), noting the base paper's
  AUROC appears computed from hard labels; ablation on the aligned set (text, URL, image,
  text+URL, text+image, all three, all three + cross-modal checks: F1, recall, PR-AUC);
  generalization EMSCAD -> Indian set; OCR comparison; calibration (ECE); confusion
  matrices and PR curves -> reports/figures/.
- reports/error_analysis.md with 10 misclassified examples.
- reports/RESULTS.md with all tables ready for slides. Clearly mark any results that
  came from SYNTHETIC data.

## Phase 10: Testing, deployment & docs
Status: TODO

- Full test run; add edge cases (empty text, huge text, corrupt image, malformed URL,
  non-English text).
- Backend Dockerfile, frontend Dockerfile (nginx), docker-compose.yml with models as a volume.
- docs/deploy.md: free deployment (Render or Hugging Face Spaces + Vercel or Netlify).
- Final README: overview, mermaid architecture diagram, setup, training steps, run
  instructions, API docs, results, limitations, ethics and privacy, team.
- docs/demo_script.md: 5-minute demo with 4 cases (scam text, fake offer letter with QR,
  lookalike URL, genuine posting).
- Final PROGRESS.md summary, a delivered-items checklist, and the final
  "WAITING FOR USER" list (data to add, Colab models to train, then which scripts to rerun).
