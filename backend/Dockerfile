# backend/Dockerfile
# Container for the FastAPI + CPU-ML backend. Runs on Google Cloud Run
# (honors the injected $PORT, default 8080) and also on an HF Docker Space
# (port 7860) or locally. See backend/README.md + DEPLOY.md.
FROM python:3.12-slim

# System deps: docTR's OpenCV backend and PyMuPDF's rendering need libGL /
# glib even headless; libgomp1 is OpenMP, used by torch/faiss-cpu on CPU.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces containers run as uid 1000 with most of the filesystem read-only
# outside $HOME/the workdir -- route every model-download cache (HF hub,
# sentence-transformers, torch) through one writable dir so the weights baked
# in at build time (see below) are the same ones found at runtime, and so a
# cold container never falls back to hitting the network for them.
ENV HF_HOME=/app/.cache \
    SENTENCE_TRANSFORMERS_HOME=/app/.cache \
    TORCH_HOME=/app/.cache \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# CPU-only torch FIRST, pinned, before requirements.txt. If requirements.txt
# were installed first, pip could resolve a CUDA build from PyPI's default
# index (multi-GB) and blow the free Space's disk/build budget. Keep this
# version in sync with the `torch==` pin in requirements.txt.
RUN pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cpu

# requirements.txt pins the same torch version already installed above, so
# pip sees it satisfied and does not re-resolve/re-download it from PyPI.
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app/ ./app/
# /eval/report reads eval/report.json at runtime (app/main.py); ship the
# whole eval/ dir (gold set + metrics + committed report, if any). The large
# corpus/*.png images are gitignored (not present in the build context) and
# excluded via .dockerignore regardless.
COPY eval/ ./eval/

# Bake all 4 sentence-transformers models + docTR's OCR weights into the
# image so the Space has a warm cold-start (no multi-GB download blocking
# the first request). Loaded through app.config.settings so these can never
# drift from what app/index/embedders.py, app/index/rerank.py,
# app/verify/nli.py and app/ingest/ocr.py actually load at request time.
RUN python -c "\
from app.config import settings; \
from sentence_transformers import SentenceTransformer, CrossEncoder; \
from doctr.models import ocr_predictor; \
SentenceTransformer(settings.text_model, device='cpu'); \
SentenceTransformer(settings.clip_model, device='cpu'); \
CrossEncoder(settings.reranker_model, device='cpu'); \
CrossEncoder(settings.nli_model, device='cpu'); \
ocr_predictor(pretrained=True)"

# Offline at RUNTIME only (weights are now baked above). Prevents cold-start
# network metadata checks. Must come AFTER the bake step, which needs the
# network to download the weights.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# Non-root, per HF Spaces Docker convention (containers run as uid 1000).
# Own the whole app dir (including .cache) so runtime cache lookups/writes
# and in-memory session handling (app/session.py) never hit a permission
# error.
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8080

# Shell form so ${PORT} expands: Cloud Run injects PORT (8080); falls back to
# 7860 for local/HF Space use.
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}
