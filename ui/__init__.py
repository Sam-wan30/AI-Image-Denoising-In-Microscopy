"""Frontend UI package for the microscopy denoising Streamlit app."""

from ui.components import (
    inject_global_styles,
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
from ui.styles import GLOBAL_STYLES

__all__ = [
    "GLOBAL_STYLES",
    "inject_global_styles",
    "render_navbar",
    "render_hero",
    "render_section_heading",
    "render_upload_shell_open",
    "render_upload_shell_close",
    "render_comparison_labels",
    "render_metric_cards_row",
    "render_loading_banner",
    "render_architecture_section",
    "render_footer",
]
