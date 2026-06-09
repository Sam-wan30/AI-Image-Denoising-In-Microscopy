"""Startup helpers: directories."""

from __future__ import annotations

import logging
from pathlib import Path

import config

logger = logging.getLogger(__name__)


def ensure_directories() -> None:
    """Create upload/output folders used at runtime."""
    config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
