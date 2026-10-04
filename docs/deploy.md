# Deploying TriGuard on free tiers

Architecture: a **static frontend** (Vercel or Netlify) calling a **Docker backend** (Hugging Face Spaces or Render).
Free-tier limits change often — check each provider's current limits before you commit to a plan.

> The backend needs RAM for PyTorch + EasyOCR (roughly 1.5–2 GB when loaded). On a small instance set
> `TRIGUARD_OCR_BACKEND=tesseract` — Tesseract is installed in the image and is much lighter (PyTorch is still loaded for the text models).

## 0. Before you deploy
1. Train/export models and keep them out of git (`models/` is gitignored): `python -m ml.text.baselines`, `python -m ml.url.train`,
   `python -m ml.fusion.train_fuser`, plus the Colab models (`models/text_distilbert/`, `models/image_clip/`).
2. Decide how the container gets `models/`: mount a volume (compose/VPS), or copy the folder into the deployment repo with Git LFS.
3. `docker compose up --build` locally first (UI http://localhost:8080, API http://localhost:8000/health).
   (The Dockerfiles were written but could not be built in the authoring environment because the Docker daemon was not running.)

## 1a. Backend on Hugging Face Spaces (recommended: more free RAM)
1. Create a Space → **SDK: Docker** (blank template).
2. Add this front-matter at the top of the Space's `README.md`:
   ```yaml
   ---
   title: TriGuard API
   sdk: docker
   app_port: 8000
   ---
   ```
3. Push the repo contents (root `Dockerfile`, `ml/`, `backend/`, `data/company_domains.csv`, `requirements.txt`) and your `models/` folder (Git LFS) to the Space.
4. Space → Settings → Variables: `TRIGUARD_CORS=https://<your-frontend-domain>`, `TRIGUARD_LIVE_CHECKS=1`, optionally `TRIGUARD_OCR_BACKEND=tesseract`.
   Note Spaces' disk is ephemeral: history/feedback in SQLite is lost on restart (acceptable for a demo).
5. API base URL: `https://<user>-<space>.hf.space`.

## 1b. Backend on Render
1. New → **Web Service** → connect the repo → Runtime **Docker**, health-check path `/health`.
2. Environment: `TRIGUARD_CORS=https://<your-frontend-domain>`; optionally a disk mounted at `/app/db` for SQLite.
3. Free instances sleep when idle, so the first request can take a minute (model loading).

## 2. Frontend on Vercel or Netlify
1. Import the repo, set **root directory** `frontend`, build command `npm run build`, output `dist`.
2. Environment variable `VITE_API_BASE=https://<backend-url>` (no trailing slash) — it is baked in at build time.
3. SPA routing uses hashes (`#/history`), so no rewrite rules are needed.
4. Add the frontend's final URL to the backend's `TRIGUARD_CORS` and redeploy the backend.

## 3. Smoke test
```bash
curl https://<backend-url>/health
curl -F "text=Pay Rs 2000 registration fee, no interview" https://<backend-url>/api/analyze
```

## Production hardening (not done here)
Authentication and per-user history, a shared rate limiter (Redis) instead of the in-memory one, HTTPS-only CORS origins, log shipping, and a
scheduled job to prune the SQLite tables.
