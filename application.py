"""
Flask production app for AI Microscopy Image Denoising (Render / gunicorn).

Local dev:  python application.py
Production: gunicorn application:app
"""

from __future__ import annotations

import base64
import logging
import threading
from io import BytesIO
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template, request, send_from_directory
from PIL import Image
from werkzeug.exceptions import RequestEntityTooLarge

import config
from services.bootstrap import download_model_if_configured, ensure_directories
from services.denoiser import ModelNotReadyError, get_denoiser_service

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def create_app() -> Flask:
    ensure_directories()

    try:
        download_model_if_configured()
    except Exception:
        logger.warning("Model download skipped or failed; will retry on first request.")

    app = Flask(
        __name__,
        static_folder="static",
        template_folder="templates",
    )
    app.config["SECRET_KEY"] = config.SECRET_KEY
    app.config["MAX_CONTENT_LENGTH"] = config.MAX_UPLOAD_MB * 1024 * 1024

    @app.errorhandler(RequestEntityTooLarge)
    def file_too_large(_exc):
        return jsonify(
            {"success": False, "error": f"File too large (max {config.MAX_UPLOAD_MB} MB)."}
        ), 413

    @app.errorhandler(Exception)
    def handle_unexpected(exc):
        logger.exception("Unhandled error")
        return jsonify({"success": False, "error": "Internal server error."}), 500

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/health")
    def health():
        """Render health check — always 200 if the web process is up."""
        svc = get_denoiser_service()
        return jsonify(
            {
                "status": "ok",
                "service": "neuroscope-denoising",
                "model": svc.status,
            }
        ), 200

    @app.route("/api/status")
    def api_status():
        return jsonify(get_denoiser_service().status)

    @app.route("/api/denoise", methods=["POST"])
    def api_denoise():
        svc = get_denoiser_service()
        if not svc.is_ready:
            try:
                svc.warm_up()
            except Exception as exc:
                return jsonify({"success": False, "error": str(exc)}), 503

        if not svc.is_ready:
            return jsonify(
                {
                    "success": False,
                    "error": svc.status.get("error") or "Model is still loading. Try again shortly.",
                }
            ), 503

        if "image" not in request.files:
            return jsonify({"success": False, "error": "No image uploaded."}), 400

        file = request.files["image"]
        if not file or not file.filename:
            return jsonify({"success": False, "error": "Empty file."}), 400

        ext = Path(file.filename).suffix.lower()
        if ext not in config.ALLOWED_EXTENSIONS:
            return jsonify(
                {
                    "success": False,
                    "error": f"Unsupported format. Allowed: {', '.join(sorted(config.ALLOWED_EXTENSIONS))}",
                }
            ), 400

        mode = request.form.get("mode", "auto").lower()
        if mode not in ("auto", "unet", "salt_pepper", "brightfield"):
            mode = "auto"

        try:
            raw = file.read()
            upload_path = config.UPLOAD_DIR / f"upload_{Path(file.filename).name}"
            upload_path.write_bytes(raw)

            result = svc.process_upload(raw, file.filename, mode=mode)

            pil = Image.open(BytesIO(raw))
            original = np.array(pil)
            denoised_path = config.OUTPUT_DIR / result["output_filename"]
            denoised = np.array(Image.open(denoised_path))

            def to_b64(arr: np.ndarray) -> str:
                if arr.ndim == 2:
                    img = Image.fromarray(arr)
                else:
                    img = Image.fromarray(arr)
                buf = BytesIO()
                img.save(buf, format="PNG")
                return base64.b64encode(buf.getvalue()).decode("ascii")

            return jsonify(
                {
                    "success": True,
                    "psnr": result["psnr"],
                    "ssim": result["ssim"],
                    "download_url": f"/api/download/{result['output_filename']}",
                    "original_b64": to_b64(original),
                    "denoised_b64": to_b64(denoised),
                    "width": result["width"],
                    "height": result["height"],
                }
            )
        except ModelNotReadyError as exc:
            return jsonify({"success": False, "error": str(exc)}), 503
        except Exception as exc:
            logger.exception("Denoise failed")
            return jsonify({"success": False, "error": str(exc)}), 500

    @app.route("/api/download/<filename>")
    def api_download(filename: str):
        safe = Path(filename).name
        path = config.OUTPUT_DIR / safe
        if not path.is_file():
            return jsonify({"error": "File not found."}), 404
        return send_from_directory(config.OUTPUT_DIR, safe, as_attachment=True)

    return app


app = create_app()


def _warm_model_background() -> None:
    try:
        get_denoiser_service().warm_up()
    except Exception:
        pass


# Pre-load model in background so first user request is faster
threading.Thread(target=_warm_model_background, daemon=True).start()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=config.PORT, debug=config.DEBUG)
