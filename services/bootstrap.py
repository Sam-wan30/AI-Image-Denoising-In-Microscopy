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
    logger.info("MODEL_URL: %s", config.MODEL_URL if config.MODEL_URL else "Not set")
    logger.info("MODEL_PATH: %s", config.MODEL_PATH)

    if not config.MODEL_URL:
        logger.warning("MODEL_URL not set - model must be present in repository")
        return

    if config.MODEL_PATH.exists():
        logger.info("Model already present at %s (%.2f MB)", config.MODEL_PATH, config.MODEL_PATH.stat().st_size / (1024 * 1024))
        return

    logger.info("Downloading model from MODEL_URL to %s ...", config.MODEL_PATH)
    config.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(config.MODEL_URL, config.MODEL_PATH)
        if not config.MODEL_PATH.exists() or config.MODEL_PATH.stat().st_size < 1024:
            raise RuntimeError(
                f"Downloaded model file is invalid or too small: {config.MODEL_PATH}"
            )
        logger.info("Model download complete (%.2f MB).", config.MODEL_PATH.stat().st_size / (1024 * 1024))
    except Exception as exc:
        if config.MODEL_PATH.exists():
            try:
                config.MODEL_PATH.unlink()
            except Exception:
                pass
        logger.error("Model download failed: %s", exc)
        raise
