"""Application configuration (environment-driven, no secrets in code)."""

from __future__ import annotations

import os
from pathlib import Path

# Project root (directory containing this file)
BASE_DIR = Path(__file__).resolve().parent

# Folders — created on startup if missing
UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", BASE_DIR / "uploads"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", BASE_DIR / "outputs"))

# Model — prefer lean deploy checkpoint; fall back to training checkpoint
_default_model = BASE_DIR / "models" / "deploy" / "model.pt"
_fallback_model = BASE_DIR / "models" / "overfit" / "best_model.pth"
MODEL_PATH = Path(
    os.environ.get(
        "MODEL_PATH",
        _default_model if _default_model.exists() else _fallback_model,
    )
)

# Optional: download checkpoint at startup (e.g. GitHub release / Hugging Face)
MODEL_URL = os.environ.get("MODEL_URL", "").strip()

# Inference
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "50"))
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}
DEVICE = os.environ.get("DEVICE", "cpu")

# Flask
SECRET_KEY = os.environ.get("SECRET_KEY", "dev-change-me-in-production")
DEBUG = os.environ.get("FLASK_DEBUG", "0") == "1"

# Gunicorn / Render
PORT = int(os.environ.get("PORT", "5000"))
