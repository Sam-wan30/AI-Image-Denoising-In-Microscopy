#!/usr/bin/env python3
"""
Streamlit web app for microscopy image denoising.

This interactive web application allows users to upload noisy microscopy images,
apply AI-based denoising, and compare results side by side.
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Union

import config
import numpy as np
import streamlit as st
import torch
from PIL import Image, UnidentifiedImageError

from inference import detect_model_type
from services.bootstrap import download_model_if_configured
from src.unet_model import create_unet_model
from ui.components import inject_global_styles
from utils.brightfield import brightfield_object_mask
from utils.metrics import calculate_psnr, calculate_ssim
from utils.preprocessing import (
    IMAGE_SIZE,
    load_grayscale,
    postprocess_tensor,
    preprocess_tensor,
)
from utils.salt_pepper import denoise_salt_pepper, estimate_salt_pepper_ratio


DEFAULT_MODEL_PATH = config.MODEL_PATH
FALLBACK_MODEL_PATHS = [
    Path("models/overfit_residual_blocks/best_model.pth"),
    Path("models/overfit/best_model.pth"),
]
MAX_UPLOAD_MB = config.MAX_UPLOAD_MB


st.set_page_config(
    page_title="NeuroScope · AI Microscopy Denoising",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

inject_global_styles()


def resolve_default_model_path() -> Path:
    """Return the first available model path or the configured default."""
    if DEFAULT_MODEL_PATH.exists():
        return DEFAULT_MODEL_PATH
    for fallback in FALLBACK_MODEL_PATHS:
        if fallback.exists():
            return fallback
    return DEFAULT_MODEL_PATH


@st.cache_resource
def load_model_resource(model_path: str, device: str) -> tuple[torch.nn.Module, dict]:
    """Load and cache the model checkpoint for Streamlit session reuse."""
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)
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
        first_conv_key = next(
            k for k in state_dict_keys if "unet.inc.double_conv.0.weight" in k
        )
        last_conv_key = next(
            k for k in state_dict_keys if "unet.outc.conv.weight" in k
        )
    else:
        first_conv_key = next(
            k for k in state_dict_keys if "inc.double_conv.0.weight" in k
        )
        last_conv_key = next(
            k for k in state_dict_keys if "outc.conv.weight" in k
        )

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
            "Checkpoint is not compatible with the current model code: "
            f"missing={len(missing)}, unexpected={len(unexpected)}"
        )

    model.to(device)
    model.eval()

    info = {
        "type": model_type,
        "parameters": sum(p.numel() for p in model.parameters()),
        "device": device,
        "path": model_path,
    }
    return model, info


class StreamlitDenoisingApp:
    """Main class for the Streamlit denoising application."""

    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self, model_path: Union[str, Path] | None = None) -> bool:
        """Load the trained denoising model from a checkpoint path."""
        model_path = Path(model_path or resolve_default_model_path())

        if not model_path.exists():
            st.error(
                f"Model file not found: {model_path}. "
                "Place a deploy checkpoint at models/deploy/model.pt "
                "or update MODEL_PATH in your environment."
            )
            return False

        with st.spinner("Loading model..."):
            try:
                model, info = load_model_resource(str(model_path), self.device)
            except Exception as exc:
                st.error(f"Error loading model: {exc}")
                return False

        st.session_state.model = model
        st.session_state.model_loaded = True
        st.session_state.model_info = info
        st.session_state.loaded_model_path = str(model_path)

        st.success("✅ Model loaded successfully!")
        st.info(
            f"**Model Details:**\n"
            f"- Type: {info['type']} U-Net\n"
            f"- Parameters: {info['parameters']:,}\n"
            f"- Device: {info['device']}"
        )
        return True

    def preprocess_image(
        self, image: Union[np.ndarray, Image.Image]
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        """Preprocess image using shared pipeline (same as training)."""
        gray = load_grayscale(image)
        original_shape = gray.shape[:2]
        tensor = preprocess_tensor(gray, IMAGE_SIZE)
        return tensor, original_shape

    def denoise_image(
        self, image: Union[np.ndarray, Image.Image], mode: str = "Microscopy U-Net"
    ) -> np.ndarray | None:
        """Perform denoising on the input image."""
        if mode == "Brightfield object mask":
            return brightfield_object_mask(image)

        if mode == "Salt-and-pepper filter":
            return denoise_salt_pepper(image)

        if mode == "Auto":
            if isinstance(image, Image.Image):
                image_to_test = np.asarray(image.convert("RGB"))
            else:
                image_to_test = image

            impulse_ratio = estimate_salt_pepper_ratio(image_to_test)
            if impulse_ratio >= 0.08:
                st.info(
                    f"Detected salt-and-pepper noise ({impulse_ratio:.1%}). "
                    "Using median filter."
                )
                return denoise_salt_pepper(image_to_test)

        if not st.session_state.get("model_loaded", False):
            st.error("Model not loaded. Please load a checkpoint before denoising.")
            return None

        try:
            input_tensor, original_shape = self.preprocess_image(image)
            input_tensor = input_tensor.to(self.device)

            with torch.inference_mode():
                output_tensor = st.session_state.model(input_tensor)

            return postprocess_tensor(output_tensor, original_shape, IMAGE_SIZE)
        except Exception as exc:
            st.error(f"Error during inference: {exc}")
            return None

    def calculate_metrics(self, original: np.ndarray, denoised: np.ndarray) -> dict[str, float]:
        """Calculate PSNR and SSIM metrics."""
        try:
            orig_tensor = torch.from_numpy(original).float() / 255.0
            den_tensor = torch.from_numpy(denoised).float() / 255.0
            return {
                "psnr": float(calculate_psnr(den_tensor, orig_tensor, max_val=1.0)),
                "ssim": float(calculate_ssim(den_tensor, orig_tensor, max_val=1.0)),
            }
        except Exception as exc:
            st.error(f"Error calculating metrics: {exc}")
            return {}

    def run(self) -> None:
        """Render the Streamlit UI."""
        from ui.run_layout import render_main_layout

        render_main_layout(self)



def main() -> None:
    """Main function to run the Streamlit app."""
    # Ensure model checkpoint is available when MODEL_URL is configured.
    if config.MODEL_URL and not config.MODEL_PATH.exists():
        try:
            download_model_if_configured()
        except Exception as exc:
            st.error(f"Unable to download model from MODEL_URL: {exc}")

    # Initialize session state
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
        st.session_state.model = None
        st.session_state.model_info = None
        st.session_state.loaded_model_path = None
    
    # Auto-load model if not loaded and model file exists
    if (
        not st.session_state.model_loaded
        and resolve_default_model_path().exists()
    ):
        app = StreamlitDenoisingApp()
        app.load_model(resolve_default_model_path())

    app = StreamlitDenoisingApp()
    app.run()


if __name__ == "__main__":
    main()
