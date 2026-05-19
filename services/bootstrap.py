"""Startup helpers: directories, optional model download."""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path

import config

logger = logging.getLogger(__name__)


def ensure_directories() -> None:
    """Create upload/output folders used at runtime."""
    config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)


def download_model_if_configured() -> None:
    """Download MODEL_URL to MODEL_PATH when the file is not present."""
    if not config.MODEL_URL:
        return
    if config.MODEL_PATH.exists():
        logger.info("Model already present at %s", config.MODEL_PATH)
        return

    logger.info("Downloading model from MODEL_URL to %s ...", config.MODEL_PATH)
    config.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(config.MODEL_URL, config.MODEL_PATH)
        logger.info("Model download complete.")
    except Exception as exc:
        logger.error("Model download failed: %s", exc)
        raise
