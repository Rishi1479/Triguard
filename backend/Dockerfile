# TriGuard backend (FastAPI + ML pipeline). Models are mounted at /app/models (see docker-compose.yml).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 KMP_DUPLICATE_LIB_OK=TRUE

# tesseract (OCR fallback), zbar (QR), libgomp (LightGBM), libglib/libgl (OpenCV runtime)
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr libzbar0 libgomp1 libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch first (the default Linux wheel pulls ~2 GB of CUDA libraries); requirements.txt pin is then already satisfied.
RUN pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY ml ./ml
COPY app ./app
COPY data/company_domains.csv ./data/company_domains.csv

# non-root user; SQLite lives in /app/db (a volume), EasyOCR model cache in /home/app/.EasyOCR (a volume)
RUN useradd -m app && mkdir -p /app/db /app/models /home/app/.EasyOCR && chown -R app /app /home/app
USER app

ENV TRIGUARD_DB_URL=sqlite:////app/db/triguard.db TRIGUARD_CORS=http://localhost:8080,http://localhost:5173
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
