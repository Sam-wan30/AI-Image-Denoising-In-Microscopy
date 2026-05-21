"""Thread-safe lazy-loaded denoising service for production."""

from __future__ import annotations

import logging
import threading
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any, Union

import numpy as np
import torch
from PIL import Image

import config
from services.model_utils import detect_model_type
from src.unet_model import create_unet_model
from utils.metrics import calculate_psnr, calculate_ssim
from utils.preprocessing import IMAGE_SIZE, load_grayscale, postprocess_tensor, preprocess_tensor
from utils.salt_pepper import denoise_salt_pepper, estimate_salt_pepper_ratio
from utils.brightfield import brightfield_object_mask

logger = logging.getLogger(__name__)

_service: "DenoiserService | None" = None
_service_lock = threading.Lock()


class ModelNotReadyError(RuntimeError):
    """Raised when the model cannot be loaded or is still loading."""


class DenoiserService:
    """Loads the U-Net once and runs inference on CPU (Render-friendly)."""

    def __init__(self) -> None:
        self.device = torch.device(config.DEVICE)
        self.model: torch.nn.Module | None = None
        self.model_info: dict[str, Any] = {}
        self._load_error: str | None = None
        self._model_lock = threading.Lock()

    @property
    def is_ready(self) -> bool:
        return self.model is not None

    @property
    def status(self) -> dict[str, Any]:
        return {
            "ready": self.is_ready,
            "error": self._load_error,
            "model_path": str(config.MODEL_PATH),
            "device": str(self.device),
            **self.model_info,
        }

    def warm_up(self) -> None:
        """Load model weights (call from a background thread on startup)."""
        try:
            self._load_model()
        except Exception as exc:
            self._load_error = str(exc)
            logger.exception("Model warm-up failed")

    def _load_model(self) -> None:
        with self._model_lock:
            if self.model is not None:
                return

            path = Path(config.MODEL_PATH)
            if not path.is_file():
                raise FileNotFoundError(
                    f"Model not found at {path}. "
                    "Run scripts/export_inference_checkpoint.py or set MODEL_URL."
                )

            logger.info("Loading model from %s", path)
            torch.set_num_threads(1)

            load_kwargs: dict[str, Any] = {"map_location": self.device}
            try:
                checkpoint = torch.load(path, weights_only=False, **load_kwargs)
            except TypeError:
                checkpoint = torch.load(path, **load_kwargs)

            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model_state_dict = checkpoint["model_state_dict"]
                model_type = checkpoint.get("model_type") or detect_model_type(
                    list(model_state_dict.keys())
                )
            else:
                model_state_dict = checkpoint
                model_type = detect_model_type(list(model_state_dict.keys()))

            state_dict_keys = list(model_state_dict.keys())
            if model_type == "residual":
                first_conv_key = [k for k in state_dict_keys if "unet.inc.double_conv.0.weight" in k][0]
                last_conv_key = [k for k in state_dict_keys if "unet.outc.conv.weight" in k][0]
            else:
                first_conv_key = [k for k in state_dict_keys if "inc.double_conv.0.weight" in k][0]
                last_conv_key = [k for k in state_dict_keys if "outc.conv.weight" in k][0]

            in_channels = model_state_dict[first_conv_key].shape[1]
            out_channels = model_state_dict[last_conv_key].shape[0]

            model = create_unet_model(
                model_type=model_type,
                in_channels=in_channels,
                out_channels=out_channels,
            )
            missing, unexpected = model.load_state_dict(model_state_dict, strict=False)
            if missing or unexpected:
                raise RuntimeError(
                    f"Incompatible checkpoint: missing={len(missing)}, unexpected={len(unexpected)}"
                )

            model.to(self.device)
            model.eval()

            self.model = model
            self.model_info = {
                "type": model_type,
                "parameters": sum(p.numel() for p in model.parameters()),
            }
            self._load_error = None
            logger.info("Model loaded (%s U-Net)", model_type)

    def _preprocess(self, image: np.ndarray) -> tuple[torch.Tensor, tuple[int, int]]:
        gray = load_grayscale(image)
        original_shape = gray.shape[:2]
        tensor = preprocess_tensor(gray, IMAGE_SIZE)
        return tensor, original_shape

    def denoise(
        self,
        image: Union[np.ndarray, Image.Image],
        mode: str = "unet",
    ) -> np.ndarray:
        if isinstance(image, Image.Image):
            image = image.convert("RGB")

        if mode == "brightfield":
            return brightfield_object_mask(image)
        if mode == "salt_pepper":
            return denoise_salt_pepper(image)
        if mode == "auto":
            if estimate_salt_pepper_ratio(np.asarray(image)) >= 0.08:
                return denoise_salt_pepper(image)
            mode = "unet"

        if not self.is_ready:
            if self._load_error:
                raise ModelNotReadyError(self._load_error)
            self._load_model()
        if self.model is None:
            raise ModelNotReadyError("Model is not loaded.")

        input_tensor, original_shape = self._preprocess(image)
        input_tensor = input_tensor.to(self.device)

        with torch.inference_mode():
            output_tensor = self.model(input_tensor)

        return postprocess_tensor(output_tensor, original_shape, IMAGE_SIZE)

    def process_upload(
        self,
        file_bytes: bytes,
        filename: str,
        mode: str = "auto",
    ) -> dict[str, Any]:
        """Run denoising and return metrics + saved output paths."""
        pil = Image.open(BytesIO(file_bytes)).convert("RGB")
        image_array = np.asarray(pil)

        denoised = self.denoise(pil, mode=mode)

        orig_tensor = torch.from_numpy(image_array).float() / 255.0
        den_tensor = torch.from_numpy(denoised).float() / 255.0
        psnr = float(calculate_psnr(den_tensor, orig_tensor, max_val=1.0))
        ssim = float(calculate_ssim(den_tensor, orig_tensor, max_val=1.0))

        job_id = uuid.uuid4().hex[:12]
        out_name = f"denoised_{job_id}.png"
        out_path = config.OUTPUT_DIR / out_name
        Image.fromarray(denoised).save(out_path, format="PNG")

        return {
            "job_id": job_id,
            "psnr": round(psnr, 2),
            "ssim": round(ssim, 4),
            "output_filename": out_name,
            "width": int(image_array.shape[1]) if image_array.ndim >= 2 else 0,
            "height": int(image_array.shape[0]) if image_array.ndim >= 2 else 0,
        }


def get_denoiser_service() -> DenoiserService:
    global _service
    if _service is None:
        with _service_lock:
            if _service is None:
                _service = DenoiserService()
    return _service
