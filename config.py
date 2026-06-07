"""Application configuration (environment-driven, no secrets in code)."""

from __future__ import annotations

import os
from pathlib import Path

# Project root (directory containing this file)
BASE_DIR = Path(__file__).resolve().parent

# Folders — created on startup if missing
UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", BASE_DIR / "uploads"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", BASE_DIR / "outputs"))

# Model — prefer lean ONNX deploy checkpoint; fall back to PyTorch and training checkpoints
_default_model = BASE_DIR / "models" / "deploy" / "model.onnx"
_fallback_model = BASE_DIR / "models" / "deploy" / "model.pt"
MODEL_PATH = Path(os.environ.get("MODEL_PATH", _default_model))

# Optional: download checkpoint at startup (e.g. GitHub release / Hugging Face)
# Accept either uppercase or lowercase env variable names just in case.
MODEL_URL = os.environ.get("MODEL_URL", os.environ.get("model_url", "")).strip()

# Inference
try:
    MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "50"))
except (ValueError, TypeError):
    MAX_UPLOAD_MB = 50
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}
DEVICE = os.environ.get("DEVICE", "cpu")

# Flask
SECRET_KEY = os.environ.get("SECRET_KEY", "dev-change-me-in-production")
DEBUG = os.environ.get("FLASK_DEBUG", "0") == "1"

# Gunicorn / Render
try:
    PORT = int(os.environ.get("PORT", "5000"))
except (ValueError, TypeError):
    PORT = 5000
