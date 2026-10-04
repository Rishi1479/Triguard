# TriGuard — Multimodal Fake Job & Internship Scam Detector

Takes job/internship **text**, an **image** (offer letter, poster, screenshot) and/or an **application URL**
and returns a label (Safe / Suspicious / Likely Scam), a calibrated risk score (0–100%) and human-readable red flags.
Final-year project (Team 3, Data Mining & Bigdata MRG); extends Pillai (2023) beyond text-only BiLSTM on EMSCAD.

See [CLAUDE.md](CLAUDE.md) for project rules and [PROGRESS.md](PROGRESS.md) for the phase log.

## Setup

### Backend / ML (Python 3.11 target; 3.12 also works for Phase 1)
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # full stack (large: torch, easyocr, ...)
# Lightweight set, enough for Phase 1 tests:
# pip install pytest numpy scikit-learn "pydantic>=2" fastapi httpx
pytest                                  # run backend tests
uvicorn backend.app.main:app --reload   # http://127.0.0.1:8000/health
```

### Frontend
```bash
cd frontend
npm install
npm run dev      # http://localhost:5173
npm test         # Vitest
npm run build
```

## Layout
```
data/{raw,processed,images,aligned}   datasets (gitignored where large)
ml/{common,text,url,image,fusion}     modality analyzers + shared schemas/config/metrics
ml/notebooks                          Colab training notebooks
models/                               saved artifacts (gitignored)
backend/app                           FastAPI app
frontend                              React + Vite + TS + Tailwind
reports  tests
```
