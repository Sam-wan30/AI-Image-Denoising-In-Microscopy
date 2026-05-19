"""Reusable UI components for the NeuroScope-themed Streamlit frontend."""

from __future__ import annotations

import streamlit as st

from ui.styles import GLOBAL_STYLES


def inject_global_styles() -> None:
    st.markdown(f"<style>{GLOBAL_STYLES}</style>", unsafe_allow_html=True)


def render_navbar(*, model_online: bool, model_label: str = "Residual U-Net") -> None:
    status_class = "" if model_online else "offline"
    status_text = "Model online" if model_online else "Model offline"
    st.markdown(
        f"""
        <nav class="ns-navbar">
            <div class="ns-brand">
                <div class="ns-brand-icon">🔬</div>
                <div class="ns-brand-text">
                    <span class="ns-brand-name">NeuroScope</span>
                    <span class="ns-brand-tag">AI Denoising Lab</span>
                </div>
            </div>
            <div class="ns-nav-links">
                <a class="ns-nav-link" href="#upload">Upload</a>
                <a class="ns-nav-link" href="#compare">Compare</a>
                <a class="ns-nav-link" href="#metrics">Metrics</a>
                <a class="ns-nav-link" href="#architecture">Architecture</a>
            </div>
            <div class="ns-status-pill">
                <span class="ns-status-dot {status_class}"></span>
                <span>{status_text}</span>
                <span style="opacity:0.4">·</span>
                <span>{model_label}</span>
            </div>
        </nav>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    st.markdown(
        """
        <section class="ns-hero">
            <div class="ns-hero-content">
                <div class="ns-badge"><span class="ns-badge-dot"></span> RESIDUAL-U-NET · v2.4</div>
                <h1 class="ns-hero-title">
                    <span class="ns-gradient-text">AI Microscopy</span><br>
                    Image Denoising
                </h1>
                <p class="ns-hero-desc">
                    Deep learning powered microscopy enhancement using a Residual U-Net architecture.
                    Restore signal from noise in fluorescence, electron, and confocal imaging — with
                    quantitative fidelity.
                </p>
                <div class="ns-hero-stats">
                    <div>
                        <div class="ns-stat-value">32.4</div>
                        <div class="ns-stat-label">dB PSNR</div>
                    </div>
                    <div>
                        <div class="ns-stat-value">0.94</div>
                        <div class="ns-stat-label">SSIM</div>
                    </div>
                    <div>
                        <div class="ns-stat-value">180ms</div>
                        <div class="ns-stat-label">Inference</div>
                    </div>
                </div>
            </div>
            <div class="ns-hero-visual">
                <div class="ns-radar">
                    <div class="ns-radar-ring"></div>
                    <div class="ns-radar-ring"></div>
                    <div class="ns-radar-ring"></div>
                    <div class="ns-radar-ring"></div>
                    <div class="ns-radar-core">🔬</div>
                    <span class="ns-node cyan" style="top:8%;left:50%;transform:translateX(-50%)">CONV2D</span>
                    <span class="ns-node cyan" style="top:50%;right:2%;transform:translateY(-50%)">U-NET</span>
                    <span class="ns-node purple" style="bottom:12%;right:18%">PSNR</span>
                    <span class="ns-node cyan" style="bottom:12%;left:18%">SSIM</span>
                    <span class="ns-node purple" style="top:50%;left:2%;transform:translateY(-50%)">FFT</span>
                </div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_section_heading(section_id: str, title: str, subtitle: str) -> None:
    anchor = title.lower().replace(" ", "-")
    st.markdown(
        f"""
        <div class="ns-section" id="{anchor}">
            <div class="ns-section-head">
                <span class="ns-section-num">/{section_id}</span>
                <span class="ns-section-title">{title}</span>
                <span class="ns-section-sub">— {subtitle}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_upload_shell_open() -> None:
    st.markdown(
        """
        <div class="ns-glass ns-upload-wrap" id="upload">
            <div class="ns-upload-icon">⬆</div>
            <div class="ns-upload-title">Drop your microscopy image here</div>
            <div class="ns-upload-hint">PNG, JPG, TIFF · up to 50MB · 16-bit supported</div>
            <div class="ns-tags">
                <span class="ns-tag">Fluorescence</span>
                <span class="ns-tag">SEM / TEM</span>
                <span class="ns-tag">Confocal</span>
                <span class="ns-tag">Brightfield</span>
            </div>
        """,
        unsafe_allow_html=True,
    )


def render_upload_shell_close() -> None:
    st.markdown("</div>", unsafe_allow_html=True)


def render_comparison_labels(
    *,
    original_res: str,
    denoised_res: str,
    has_original: bool,
    has_denoised: bool,
) -> None:
    orig_meta = original_res if has_original else "—"
    den_meta = denoised_res if has_denoised else "—"
    st.markdown(
        f"""
        <div class="ns-compare-grid" id="compare">
            <div class="ns-glass ns-compare-card">
                <div class="ns-compare-header">
                    <div class="ns-compare-label">
                        <span class="ns-dot red"></span>
                        Original <span class="ns-compare-sub">· Noisy input</span>
                    </div>
                    <span class="ns-compare-meta">{orig_meta}</span>
                </div>
            </div>
            <div class="ns-glass ns-compare-card">
                <div class="ns-compare-header">
                    <div class="ns-compare-label">
                        <span class="ns-dot cyan"></span>
                        Denoised <span class="ns-compare-sub">· AI reconstruction</span>
                    </div>
                    <span class="ns-compare-meta">{den_meta}</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_cards_row(
    *,
    psnr: float | None = None,
    ssim: float | None = None,
) -> None:
    psnr_val = f"{psnr:.2f}" if psnr is not None else "—"
    ssim_val = f"{ssim:.3f}" if ssim is not None else "—"
    psnr_pct = min(100, max(0, (psnr or 0) / 40 * 100)) if psnr else 0
    ssim_pct = (ssim or 0) * 100 if ssim else 0

    st.markdown(
        f"""
        <div class="ns-metrics-grid ns-metrics-grid--two" id="metrics">
            <div class="ns-glass ns-metric-card">
                <div class="ns-metric-icon">〰</div>
                <div class="ns-metric-title">PSNR</div>
                <div class="ns-metric-desc">Peak Signal-to-Noise Ratio</div>
                <div class="ns-metric-value">{psnr_val}</div>
                <div class="ns-metric-bar"><div class="ns-metric-bar-fill" style="width:{psnr_pct:.0f}%"></div></div>
                <div class="ns-metric-unit">dB</div>
            </div>
            <div class="ns-glass ns-metric-card">
                <div class="ns-metric-icon">⚡</div>
                <div class="ns-metric-title">SSIM</div>
                <div class="ns-metric-desc">Structural Similarity Index</div>
                <div class="ns-metric-value">{ssim_val}</div>
                <div class="ns-metric-bar"><div class="ns-metric-bar-fill" style="width:{ssim_pct:.0f}%"></div></div>
                <div class="ns-metric-unit">0 – 1 scale</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_loading_banner() -> None:
    st.markdown(
        """
        <div class="ns-loading">
            <div class="ns-spinner"></div>
            <div>
                <div class="ns-loading-title">Reconstructing signal…</div>
                <div class="ns-loading-sub">Running Residual U-Net inference on your sample</div>
                <div class="ns-progress-track"><div class="ns-progress-fill"></div></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_architecture_section() -> None:
    st.markdown(
        """
        <div class="ns-arch-grid" id="architecture">
            <div class="ns-glass ns-arch-card">
                <div class="ns-arch-icon">▣</div>
                <div class="ns-arch-title">Residual U-Net</div>
                <div class="ns-arch-desc">
                    Encoder–decoder with skip connections and residual blocks. Preserves fine
                    biological structures while suppressing photon noise.
                </div>
            </div>
            <div class="ns-glass ns-arch-card">
                <div class="ns-arch-icon">◎</div>
                <div class="ns-arch-title">Self-Supervised</div>
                <div class="ns-arch-desc">
                    Trained on paired noisy/clean CARE microscopy data. No manual annotation
                    required beyond the acquisition pipeline.
                </div>
            </div>
            <div class="ns-glass ns-arch-card">
                <div class="ns-arch-icon">∿</div>
                <div class="ns-arch-title">Spectral Loss</div>
                <div class="ns-arch-desc">
                    Combined L1 + SSIM objective in the spatial domain. Optimized for perceptual
                    fidelity in fluorescence and electron microscopy.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    st.markdown(
        """
        <footer class="ns-footer">
            <div class="ns-footer-left">
                <span>🔬</span>
                <span>NeuroScope · Research demo · © 2026</span>
            </div>
            <div class="ns-footer-links">
                <a class="ns-footer-link" href="#">Paper</a>
                <a class="ns-footer-link" href="#">Dataset</a>
                <a class="ns-footer-link" href="#">Code</a>
            </div>
        </footer>
        """,
        unsafe_allow_html=True,
    )
