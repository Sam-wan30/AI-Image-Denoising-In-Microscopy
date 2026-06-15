"""Streamlit main-page layout (frontend only)."""

from __future__ import annotations

import io
import os
from datetime import datetime

import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError

from ui.components import (
    render_architecture_section,
    render_comparison_labels,
    render_footer,
    render_hero,
    render_loading_banner,
    render_metric_cards_row,
    render_navbar,
    render_section_heading,
    render_upload_shell_close,
    render_upload_shell_open,
)


def _resolution_label(shape) -> str:
    if shape is None or len(shape) < 2:
        return "—"
    h, w = int(shape[0]), int(shape[1])
    return f"{w} × {h}"


def _render_compare_image(column, image) -> None:
    with column:
        st.markdown('<div class="ns-glass ns-compare-body">', unsafe_allow_html=True)
        if image is None:
            st.markdown(
                """
                <div class="ns-placeholder">
                    <div class="ns-placeholder-icon">🖼</div>
                    <div>No image uploaded</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.image(image, width='stretch')
        st.markdown("</div>", unsafe_allow_html=True)


def render_main_layout(app) -> None:
    """Render the FluoClean AI-themed main page; delegates inference to `app`."""
    model_online = st.session_state.get("model_loaded", False)
    model_label = "Residual U-Net"
    if info := st.session_state.get("model_info"):
        model_label = f"{info['type']} U-Net"

    render_navbar(model_online=model_online, model_label=model_label)
    render_hero()

    with st.sidebar:
        st.markdown("### ⚙ Settings")
        denoise_mode = st.selectbox(
            "Denoising Mode",
            ["Auto", "Microscopy U-Net", "Salt-and-pepper filter", "Brightfield object mask"],
            help="Use Brightfield object mask for dark structures on noisy gray/white backgrounds.",
        )
        st.session_state.denoise_mode = denoise_mode

        model_path = st.text_input(
            "Model Path",
            value=app.DEFAULT_MODEL_PATH if hasattr(app, "DEFAULT_MODEL_PATH") else "models/overfit_residual_blocks/best_model.pth",
        )

        if st.button("Load Model", type="primary"):
            app.load_model(model_path)

        if model_online:
            st.success("Model loaded")
            if info:
                parameter_count = info.get("parameters")
                parameter_label = (
                    f"{parameter_count:,}" if isinstance(parameter_count, int) else "N/A"
                )
                st.caption(f"{info['type']} · {parameter_label} params")
        else:
            st.warning("Load a model to enable U-Net denoising.")

    render_section_heading("01", "Upload", "DROP A SAMPLE")
    render_upload_shell_open()
    uploaded_file = st.file_uploader(
        "Upload microscopy image",
        type=["png", "jpg", "jpeg", "webp", "tif", "tiff", "bmp"],
        label_visibility="collapsed",
        key="ns_uploader",
    )
    render_upload_shell_close()

    image_array = None
    image_pil = None
    upload_name = None
    if uploaded_file is not None:
        if uploaded_file.size > int(os.environ.get("MAX_UPLOAD_MB", "50")) * 1024 * 1024:
            st.error(f"Upload too large. Please use files smaller than {os.environ.get('MAX_UPLOAD_MB', '50')} MB.")
        else:
            upload_key = f"upload::{uploaded_file.name}::{uploaded_file.size}"
            if st.session_state.get("last_upload_key") != upload_key:
                st.session_state["last_upload_key"] = upload_key
                st.session_state.pop("denoised_result", None)
                st.session_state.pop("result_metrics", None)

            image_bytes = uploaded_file.getvalue()
            try:
                image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                image_array = np.array(image_pil)
                upload_name = uploaded_file.name
                st.session_state["upload_shape"] = image_array.shape
            except UnidentifiedImageError:
                st.error("Unsupported or corrupted image file. Please upload a valid PNG, JPG, TIFF, or BMP image.")
            except Exception as exc:
                st.error(f"Unable to open uploaded image: {exc}")

    denoised_image = st.session_state.get("denoised_result")
    shape = image_array.shape if image_array is not None else st.session_state.get("upload_shape")
    render_section_heading("02", "Comparison", "SIDE-BY-SIDE")
    render_comparison_labels(
        original_res=_resolution_label(shape),
        denoised_res=_resolution_label(denoised_image.shape if denoised_image is not None else None),
        has_original=image_array is not None,
        has_denoised=denoised_image is not None,
    )

    col_orig, col_den = st.columns(2)
    _render_compare_image(col_orig, image_array)
    _render_compare_image(col_den, denoised_image)

    if image_array is not None:
        st.markdown('<div class="ns-spacer-md"></div>', unsafe_allow_html=True)
        if st.button("Start denoising →", type="primary", width='stretch'):
            if not model_online and st.session_state.get("denoise_mode", "Auto") in (
                "Auto",
                "Microscopy U-Net",
            ):
                st.error("Load a model from the sidebar before running U-Net denoising.")
            else:
                render_loading_banner()
                progress = st.progress(0)
                mode = st.session_state.get("denoise_mode", "Auto")
                progress.progress(20)
                result = app.denoise_image(image_pil or image_array, mode=mode)
                progress.progress(90)
                progress.progress(100)

                if result is not None:
                    # Always calculate metrics, even if using non-U-Net modes
                    try:
                        metrics = app.calculate_metrics(image_array, result)
                    except Exception as exc:
                        st.warning(f"Metrics calculation encountered an issue: {exc}")
                        metrics = {"psnr": None, "ssim": None}
                    
                    st.session_state["denoised_result"] = result
                    st.session_state["result_metrics"] = metrics
                    st.session_state["upload_name"] = upload_name
                    st.rerun()

    metrics = st.session_state.get("result_metrics")
    if denoised_image is not None:
        render_section_heading("03", "Quality Metrics", "QUANTITATIVE FIDELITY")
        # Display metrics even if calculation failed, showing fallback values
        render_metric_cards_row(
            psnr=metrics.get("psnr") if metrics else None,
            ssim=metrics.get("ssim") if metrics else None,
        )

        st.markdown('<div class="ns-spacer-sm"></div>', unsafe_allow_html=True)
        denoised_pil = Image.fromarray(denoised_image)
        img_byte_arr = io.BytesIO()
        denoised_pil.save(img_byte_arr, format="PNG")
        img_byte_arr.seek(0)
        
        # Generate timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"denoised-image-{timestamp}.png"
        
        try:
            st.download_button(
                label="⬇ Download denoised image",
                data=img_byte_arr,
                file_name=filename,
                mime="image/png",
                width='stretch',
            )
        except Exception as exc:
            st.error(f"Download failed: {exc}")

    render_section_heading("04", "Architecture", "UNDER THE HOOD")
    render_architecture_section()
    render_footer()
