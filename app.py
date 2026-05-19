#!/usr/bin/env python3
"""
Streamlit web app for microscopy image denoising.

This interactive web application allows users to upload noisy microscopy images,
apply AI-based denoising, and compare results side by side.
"""

import streamlit as st
import torch
import numpy as np
import os

# Import our custom modules
from src.unet_model import create_unet_model
from inference import detect_model_type
from utils.metrics import calculate_psnr, calculate_ssim
from utils.preprocessing import (
    IMAGE_SIZE,
    load_grayscale,
    postprocess_tensor,
    preprocess_tensor,
)
from utils.salt_pepper import (
    denoise_salt_pepper,
    estimate_salt_pepper_ratio,
)
from utils.brightfield import brightfield_object_mask
from ui.components import inject_global_styles


DEFAULT_MODEL_PATH = "models/overfit_residual_blocks/best_model.pth"


st.set_page_config(
    page_title="NeuroScope · AI Microscopy Denoising",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

inject_global_styles()


class StreamlitDenoisingApp:
    """
    Main class for the Streamlit denoising application.
    """
    
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def load_model(self, model_path: str):
        """Load the trained denoising model."""
        try:
            if not os.path.exists(model_path):
                st.error(f"Model file not found: {model_path}")
                return False
            
            with st.spinner("Loading model..."):
                # Load checkpoint
                checkpoint = torch.load(model_path, map_location=self.device)
                
                # Extract model state dict
                if 'model_state_dict' in checkpoint:
                    model_state_dict = checkpoint['model_state_dict']
                else:
                    model_state_dict = checkpoint
                
                state_dict_keys = list(model_state_dict.keys())
                model_type = (
                    checkpoint.get("model_type")
                    if isinstance(checkpoint, dict)
                    else None
                ) or detect_model_type(state_dict_keys)

                if model_type == "residual":
                    first_conv_key = [
                        k for k in state_dict_keys if "unet.inc.double_conv.0.weight" in k
                    ][0]
                    last_conv_key = [
                        k for k in state_dict_keys if "unet.outc.conv.weight" in k
                    ][0]
                else:
                    first_conv_key = [
                        k for k in state_dict_keys if "inc.double_conv.0.weight" in k
                    ][0]
                    last_conv_key = [
                        k for k in state_dict_keys if "outc.conv.weight" in k
                    ][0]

                in_channels = model_state_dict[first_conv_key].shape[1]
                out_channels = model_state_dict[last_conv_key].shape[0]
                self.model = create_unet_model(
                    model_type=model_type,
                    in_channels=in_channels,
                    out_channels=out_channels
                )
                
                missing, unexpected = self.model.load_state_dict(
                    model_state_dict, strict=False
                )
                if missing or unexpected:
                    st.error(
                        "Checkpoint is not compatible with the current model code. "
                        f"Missing keys: {len(missing)}, unexpected keys: {len(unexpected)}. "
                        f"Use {DEFAULT_MODEL_PATH} or retrain with the current train.py."
                    )
                    return False

                self.model.to(self.device)
                self.model.eval()
                
                # Store model in session state
                st.session_state.model = self.model
                st.session_state.model_loaded = True
                st.session_state.model_info = {
                    'type': model_type,
                    'parameters': sum(p.numel() for p in self.model.parameters()),
                    'device': self.device,
                    'path': model_path,
                }
                st.session_state.loaded_model_path = model_path
                
                # Display model info
                st.success(f"✅ Model loaded successfully!")
                st.info(f"**Model Details:**\n"
                       f"- Type: {model_type} U-Net\n"
                       f"- Parameters: {sum(p.numel() for p in self.model.parameters()):,}\n"
                       f"- Device: {self.device}\n"
                       f"- Checkpoint compatibility: missing={len(missing)}, unexpected={len(unexpected)}")
                
                return True
                
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return False
    
    def preprocess_image(self, image: np.ndarray) -> tuple:
        """Preprocess image using shared pipeline (same as training)."""
        gray = load_grayscale(image)
        original_shape = gray.shape[:2]
        tensor = preprocess_tensor(gray, IMAGE_SIZE)
        return tensor, original_shape

    def denoise_image(self, image: np.ndarray, mode: str = "Microscopy U-Net") -> np.ndarray:
        """Perform denoising on the input image."""
        if mode == "Brightfield object mask":
            return brightfield_object_mask(image)

        if mode == "Salt-and-pepper filter":
            return denoise_salt_pepper(image)

        if mode == "Auto":
            impulse_ratio = estimate_salt_pepper_ratio(image)
            if impulse_ratio >= 0.08:
                st.info(
                    f"Detected salt-and-pepper style noise ({impulse_ratio:.1%} impulse pixels). "
                    "Using median filter."
                )
                return denoise_salt_pepper(image)

        if not st.session_state.get('model_loaded', False):
            st.error("Model not loaded!")
            return None
        
        try:
            input_tensor, original_shape = self.preprocess_image(image)
            input_tensor = input_tensor.to(self.device)

            with torch.no_grad():
                model = st.session_state.model
                if self.device == "cuda":
                    with torch.cuda.amp.autocast():
                        output_tensor = model(input_tensor)
                else:
                    output_tensor = model(input_tensor)

            return postprocess_tensor(output_tensor, original_shape, IMAGE_SIZE)
            
        except Exception as e:
            st.error(f"Error during denoising: {str(e)}")
            return None
    
    def calculate_metrics(self, original: np.ndarray, denoised: np.ndarray) -> dict:
        """Calculate quality metrics."""
        try:
            # Convert to tensors
            orig_tensor = torch.from_numpy(original).float() / 255.0
            den_tensor = torch.from_numpy(denoised).float() / 255.0
            
            # Calculate metrics
            psnr = calculate_psnr(den_tensor, orig_tensor, max_val=1.0)
            ssim = calculate_ssim(den_tensor, orig_tensor, max_val=1.0)
            
            return {
                'psnr': psnr,
                'ssim': ssim,
            }
        except Exception as e:
            st.error(f"Error calculating metrics: {str(e)}")
            return {}
    
    def run(self):
        """Run the Streamlit app."""
        from ui.run_layout import render_main_layout

        self.DEFAULT_MODEL_PATH = DEFAULT_MODEL_PATH
        render_main_layout(self)



def main():
    """Main function to run the Streamlit app."""
    # Initialize session state
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
        st.session_state.model = None
        st.session_state.model_info = None
    
    # Auto-load model if not loaded and model file exists
    loaded_path = st.session_state.get("loaded_model_path")
    if (
        os.path.exists(DEFAULT_MODEL_PATH)
        and (not st.session_state.model_loaded or loaded_path != DEFAULT_MODEL_PATH)
    ):
        app = StreamlitDenoisingApp()
        app.load_model(DEFAULT_MODEL_PATH)
    
    app = StreamlitDenoisingApp()
    app.run()


if __name__ == "__main__":
    main()
