"""Thread-safe lazy-loaded denoising service for production.

This module prefers ONNXRuntime for inference when an `.onnx` model is present
so the web service can run without a PyTorch install. PyTorch (`.pt`) checkpoints
are still supported as a fallback when `torch` is available.
"""

from __future__ import annotations

import logging
import os
import threading
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any, Union

import numpy as np
from PIL import Image

import config
from services.model_utils import detect_model_type
from utils.metrics import calculate_psnr, calculate_ssim
from utils.preprocessing import IMAGE_SIZE, load_grayscale, postprocess_tensor, preprocess_tensor
from utils.salt_pepper import denoise_salt_pepper, estimate_salt_pepper_ratio
from utils.brightfield import brightfield_object_mask

# Configure ONNX Runtime to use CPU-only execution and suppress GPU detection warnings
# This prevents GPU provider detection errors on CPU-only environments like Render
os.environ["ORT_TENSORRT_ENGINE_CACHE_ENABLE"] = "0"
os.environ["ORT_DISABLE_PROVIDER_TYPE_STRING"] = "TensorrtExecutionProvider,CUDAExecutionProvider,ROCmExecutionProvider,CoreMLExecutionProvider"

# Optional backends: prefer ONNXRuntime to avoid a hard PyTorch dependency in lightweight deploys.
try:
    import onnxruntime as ort
except Exception:
    ort = None

try:
    # Do NOT import torch at module import time; import lazily when a .pt checkpoint is loaded.
    import torch  # type: ignore
except Exception:
    torch = None

logger = logging.getLogger(__name__)

_service: "DenoiserService | None" = None
_service_lock = threading.Lock()


class ModelNotReadyError(RuntimeError):
    """Raised when the model cannot be loaded or is still loading."""


class DenoiserService:
    """Loads the U-Net once and runs inference on CPU (Render-friendly).

    This service supports two backends:
    - ONNXRuntime when `MODEL_PATH` points to an `.onnx` file (preferred for Render)
    - PyTorch when `MODEL_PATH` points to a `.pt` checkpoint (requires `torch` installed)
    """

    def __init__(self) -> None:
        self.device = config.DEVICE
        self.torch = None
        self.onnx_session = None
        self.model: Any | None = None
        self.model_info: dict[str, Any] = {}
        self._load_error: str | None = None
        self._model_lock = threading.Lock()

    @property
    def is_ready(self) -> bool:
        return self.model is not None

    @property
    def status(self) -> dict[str, Any]:
        try:
            return {
                "ready": self.is_ready,
                "error": self._load_error,
                "model_path": str(config.MODEL_PATH),
                "device": str(self.device),
                **self.model_info,
            }
        except Exception:
            return {
                "ready": False,
                "error": "Status check failed",
                "model_path": str(config.MODEL_PATH),
                "device": "unknown",
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

            # Startup diagnostics
            logger.info("=" * 70)
            logger.info("MODEL LOADING DIAGNOSTICS")
            logger.info("=" * 70)
            logger.info("MODEL_URL: %s", config.MODEL_URL if config.MODEL_URL else "Not set")
            logger.info("MODEL_PATH: %s", path)
            logger.info("File exists: %s", path.is_file())
            if path.is_file():
                logger.info("File size: %.2f MB", path.stat().st_size / (1024 * 1024))
            logger.info("BASE_DIR: %s", config.BASE_DIR)
            logger.info("=" * 70)

            if not path.is_file():
                raise FileNotFoundError(
                    f"Model not found at {path}. Set MODEL_URL environment variable or ensure model is included in deployment."
                )

            logger.info("Loading model from %s", path)

            # ONNX path (preferred for lightweight deploys)
            if path.suffix.lower() == ".onnx":
                if ort is None:
                    raise RuntimeError("onnxruntime is not installed; cannot load .onnx model.")
                session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
                self.onnx_session = session
                self.model = None
                self.model_info = {"type": "ONNX U-Net", "parameters": None, "device": "cpu", "path": str(path)}
                self._load_error = None
                logger.info("Loaded ONNX model: %s (Execution Provider: CPUExecutionProvider)", path)
                return

            # PyTorch fallback
            if torch is None:
                raise RuntimeError(
                    "PyTorch is not installed in this environment. To use .pt checkpoints, install torch or provide an ONNX model (.onnx)."
                )

            # Import model constructor lazily
            from src.unet_model import create_unet_model  # local import

            torch.set_num_threads(1)
            load_kwargs: dict[str, Any] = {"map_location": config.DEVICE}
            try:
                checkpoint = torch.load(path, weights_only=False, **load_kwargs)
            except TypeError:
                checkpoint = torch.load(path, **load_kwargs)

            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model_state_dict = checkpoint["model_state_dict"]
                model_type = checkpoint.get("model_type") or detect_model_type(list(model_state_dict.keys()))
            else:
                model_state_dict = checkpoint
                model_type = detect_model_type(list(model_state_dict.keys()))

            state_dict_keys = list(model_state_dict.keys())
            if model_type == "residual":
                first_conv_key = next(k for k in state_dict_keys if "unet.inc.double_conv.0.weight" in k)
                last_conv_key = next(k for k in state_dict_keys if "unet.outc.conv.weight" in k)
            else:
                first_conv_key = next(k for k in state_dict_keys if "inc.double_conv.0.weight" in k)
                last_conv_key = next(k for k in state_dict_keys if "outc.conv.weight" in k)

            in_channels = model_state_dict[first_conv_key].shape[1]
            out_channels = model_state_dict[last_conv_key].shape[0]

            model = create_unet_model(model_type=model_type, in_channels=in_channels, out_channels=out_channels)
            missing, unexpected = model.load_state_dict(model_state_dict, strict=False)
            if missing or unexpected:
                raise RuntimeError(f"Incompatible checkpoint: missing={len(missing)}, unexpected={len(unexpected)}")

            model.to(config.DEVICE)
            model.eval()

            self.model = model
            self.model_info = {"type": model_type, "parameters": sum(p.numel() for p in model.parameters()), "device": config.DEVICE}
            self._load_error = None
            logger.info("Model loaded (%s U-Net)", model_type)

    def _preprocess(self, image: np.ndarray) -> tuple[Any, tuple[int, int]]:
        gray = load_grayscale(image)
        original_shape = gray.shape[:2]
        # Return a torch.Tensor when PyTorch backend is used; otherwise a numpy array for ONNX.
        tensor = preprocess_tensor(gray, IMAGE_SIZE, as_tensor=True) if torch is not None else preprocess_tensor(gray, IMAGE_SIZE, as_tensor=False)
        return tensor, original_shape

    def _preprocess_numpy(self, image: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
        gray = load_grayscale(image)
        original_shape = gray.shape[:2]
        arr = preprocess_tensor(gray, IMAGE_SIZE, as_tensor=False)
        # ONNXRuntime expects shape (N, C, H, W)
        inp = np.expand_dims(np.expand_dims(arr.astype(np.float32), axis=0), axis=0)
        return inp, original_shape

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
            # If ONNX session loaded, run it
            if self.onnx_session is not None:
                inp, original_shape = self._preprocess_numpy(image)
                input_name = self.onnx_session.get_inputs()[0].name
                out = self.onnx_session.run(None, {input_name: inp})[0]
                return postprocess_tensor(out, original_shape, IMAGE_SIZE)
            raise ModelNotReadyError("Model is not loaded.")

        # PyTorch inference path (model is a torch.nn.Module)
        input_tensor, original_shape = self._preprocess(image)
        if hasattr(input_tensor, "to"):
            input_tensor = input_tensor.to(config.DEVICE)

        if torch is not None:
            with torch.inference_mode():
                output_tensor = self.model(input_tensor)
        else:
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

        # Use numpy-based metrics to avoid requiring torch during runtime
        try:
            orig_norm = image_array.astype(np.float32) / 255.0
            den_norm = denoised.astype(np.float32) / 255.0
            psnr = float(calculate_psnr(den_norm, orig_norm, max_val=1.0))
            ssim = float(calculate_ssim(den_norm, orig_norm, max_val=1.0))
        except Exception:
            psnr = 0.0
            ssim = 0.0

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
