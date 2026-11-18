"""
PixelLab – Image Processing Studio
Main Streamlit Application
A modern, desktop-grade web application for professional digital image processing.
Designed with a clean, high-precision interface inspired by Linear and Raycast.
"""

from typing import Optional
from pathlib import Path
import io
from PIL import Image, ImageOps
import streamlit as st

# Import PixelLab modules
from modules.image_utils import (
    validate_image_file,
    get_image_info,
    format_bytes,
    ensure_rgb,
    image_to_bytes,
)
from modules.filters import (
    apply_grayscale,
    apply_sepia,
    apply_black_and_white,
    apply_vintage,
    apply_gaussian_blur,
    apply_median_blur,
    apply_sharpen,
    apply_emboss,
    apply_invert_colors,
    apply_edge_detection_filter,
)
from modules.adjustments import (
    adjust_brightness,
    adjust_contrast,
    adjust_saturation,
    adjust_sharpness,
    adjust_gamma,
    apply_adjustments_pipeline,
)
from modules.transformations import (
    resize_image,
    crop_image,
    rotate_90_clockwise,
    rotate_180,
    rotate_270_clockwise,
    rotate_custom,
    flip_horizontal,
    flip_vertical,
)
from modules.enhancement import (
    enhance_sharpen,
    enhance_noise_reduction,
    enhance_gaussian_smoothing,
    enhance_histogram_equalization,
    enhance_clahe,
)
from modules.edge_detection import (
    detect_canny,
    detect_sobel_x,
    detect_sobel_y,
    detect_sobel_combined,
    detect_laplacian,
)
from modules.analysis import (
    get_image_statistics,
    plot_rgb_histogram,
    plot_separate_channels,
    plot_grayscale_histogram,
)
from modules.compression import (
    compress_image,
    convert_image_format,
)
from modules.watermark import add_text_watermark

# ==========================================
# STREAMLIT CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="PixelLab – Image Processing Studio",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================
# BESPOKE DESIGN SYSTEM & HUGEICONS INTEGRATION
# Clean, high-density, professional developer/creator UI (Linear / Raycast)
# ==========================================
st.markdown(
    """
    <!-- Hugeicons Font CDN -->
    <link rel="stylesheet" href="https://cdn.hugeicons.com/font/hgi-stroke-rounded.css">
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">

    <style>
    /* Base Reset & Typography */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
        -webkit-font-smoothing: antialiased;
    }

    /* Studio Dark Foundation */
    .stApp {
        background-color: #090a0f !important;
        color: #f4f4f5 !important;
    }

    /* Top Studio Header */
    .studio-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 14px 20px;
        background: #11131a;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        margin-bottom: 20px;
    }

    .studio-brand {
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .brand-mark {
        width: 32px;
        height: 32px;
        border-radius: 8px;
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        color: #ffffff;
        font-size: 16px;
        box-shadow: 0 2px 8px rgba(99, 102, 241, 0.3);
    }

    .brand-title {
        font-size: 18px;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #ffffff;
    }

    .studio-chip {
        font-size: 11px;
        font-weight: 600;
        padding: 3px 8px;
        border-radius: 6px;
        background: rgba(99, 102, 241, 0.15);
        color: #a5b4fc;
        border: 1px solid rgba(99, 102, 241, 0.3);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .file-chip {
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
        padding: 4px 10px;
        border-radius: 6px;
        background: #181b24;
        color: #94a3b8;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }

    /* High Precision Panels */
    .studio-panel {
        background: #11131a;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 16px;
    }

    .panel-header {
        font-size: 14px;
        font-weight: 600;
        color: #f4f4f5;
        margin-bottom: 14px;
        display: flex;
        align-items: center;
        gap: 8px;
        letter-spacing: -0.01em;
    }

    /* Metric Badges Grid */
    .metrics-bar {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
        gap: 10px;
        margin-bottom: 18px;
    }

    .metric-card {
        background: #11131a;
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 8px;
        padding: 10px 14px;
    }

    .metric-label {
        font-size: 11px;
        font-weight: 500;
        color: #71717a;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 3px;
    }

    .metric-val {
        font-size: 15px;
        font-weight: 600;
        color: #f4f4f5;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Upload Hero Section */
    .upload-hero {
        background: #11131a;
        border: 1px dashed rgba(255, 255, 255, 0.15);
        border-radius: 12px;
        padding: 40px 24px;
        text-align: center;
        margin-bottom: 20px;
    }

    .upload-hero:hover {
        border-color: #6366f1;
    }

    /* Sidebar Refinements */
    section[data-testid="stSidebar"] {
        background-color: #0c0d14 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.07) !important;
    }

    /* Streamlit Button Restyle */
    .stButton > button {
        background-color: #181b24;
        color: #e4e4e7;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 6px;
        font-weight: 500;
        font-size: 13px;
        padding: 6px 14px;
        transition: all 0.15s ease;
    }

    .stButton > button:hover {
        background-color: #202430;
        border-color: #6366f1;
        color: #ffffff;
    }

    .stButton > button:focus {
        border-color: #6366f1;
        box-shadow: 0 0 0 1px #6366f1;
    }

    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        background: #11131a;
        padding: 4px;
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.07);
        gap: 4px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        padding: 6px 14px;
        font-size: 13px;
        color: #a1a1aa;
    }

    .stTabs [aria-selected="true"] {
        background: #1f2330 !important;
        color: #ffffff !important;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================================
# SESSION STATE MANAGEMENT
# ==========================================
def init_session_state():
    """Initialize persistent session variables."""
    if "original_image" not in st.session_state:
        st.session_state.original_image = None
    if "processed_image" not in st.session_state:
        st.session_state.processed_image = None
    if "image_filename" not in st.session_state:
        st.session_state.image_filename = "No file loaded"
    if "raw_bytes_len" not in st.session_state:
        st.session_state.raw_bytes_len = 0
    if "last_operation" not in st.session_state:
        st.session_state.last_operation = "None"
    if "active_category" not in st.session_state:
        st.session_state.active_category = "Image Filters"


init_session_state()


def load_image_into_state(pil_image: Image.Image, filename: str, raw_size: int = 0):
    """Load an image into session state as both original and working copy."""
    st.session_state.original_image = pil_image.copy()
    st.session_state.processed_image = pil_image.copy()
    st.session_state.image_filename = filename
    st.session_state.raw_bytes_len = raw_size or (pil_image.size[0] * pil_image.size[1] * 3)
    st.session_state.last_operation = "Loaded image"


def reset_to_original():
    """Reset the processed working image back to the original upload."""
    if st.session_state.original_image is not None:
        st.session_state.processed_image = st.session_state.original_image.copy()
        st.session_state.last_operation = "Reset to original"
        st.toast("Reverted back to original image", icon="🔄")


def commit_processed_as_base():
    """Commit current processed image as the new base image for chaining operations."""
    if st.session_state.processed_image is not None:
        st.session_state.original_image = st.session_state.processed_image.copy()
        st.session_state.last_operation = "Committed as new base"
        st.toast("Changes applied as new working base! You can chain another effect.", icon="✅")


def clear_image():
    """Clear image state completely."""
    st.session_state.original_image = None
    st.session_state.processed_image = None
    st.session_state.image_filename = "No file loaded"
    st.session_state.raw_bytes_len = 0
    st.session_state.last_operation = "None"


def load_sample_file(sample_path: Path):
    """Safely load a built-in sample file directly from disk into state."""
    try:
        img = Image.open(sample_path)
        img.load()
        img = ImageOps.exif_transpose(img)
        file_size = sample_path.stat().st_size
        load_image_into_state(img, sample_path.name, file_size)
    except Exception as e:
        st.error(f"Failed to load sample image: {e}")


# ==========================================
# SIDEBAR NAVIGATION
# ==========================================
with st.sidebar:
    # Studio Brand Header with Hugeicons
    st.markdown(
        """
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 6px;">
            <div class="brand-mark">
                <i class="hgi-stroke hgi-image-01" style="font-size: 18px;"></i>
            </div>
            <div>
                <div style="font-size: 17px; font-weight: 700; color: #ffffff; letter-spacing: -0.02em;">PixelLab</div>
                <div style="font-size: 11px; color: #818cf8; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase;">Studio v1.0</div>
            </div>
        </div>
        <div style="font-size: 12px; color: #71717a; margin-bottom: 14px;">Professional Image Processing Studio</div>
        <hr style="border-color: rgba(255,255,255,0.07); margin: 0 0 14px 0;"/>
        """,
        unsafe_allow_html=True,
    )

    # Tool Navigation Menu
    st.markdown("<div style='font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; color: #71717a; margin-bottom: 6px;'>Studio Modules</div>", unsafe_allow_html=True)

    tool_names = [
        "Image Filters",
        "Image Adjustments",
        "Transformations",
        "Image Enhancement",
        "Edge Detection",
        "Image Analysis",
        "Compression & Export",
        "Text Watermark",
    ]

    selected_category = st.radio(
        "Tools",
        tool_names,
        index=0,
        label_visibility="collapsed",
    )
    st.session_state.active_category = selected_category

    st.markdown("<hr style='border-color: rgba(255,255,255,0.07); margin: 16px 0;'/>", unsafe_allow_html=True)

    # Active Image Card
    if st.session_state.original_image is not None:
        info = get_image_info(st.session_state.processed_image, st.session_state.raw_bytes_len)
        st.markdown(
            f"""
            <div class="studio-panel" style="padding: 14px 16px; margin-bottom: 12px;">
                <div style="font-size: 11px; font-weight: 600; color: #71717a; text-transform: uppercase; margin-bottom: 6px;">LOADED IMAGE</div>
                <div style="font-size: 13px; font-weight: 600; color: #ffffff; word-break: break-all; margin-bottom: 8px;">{st.session_state.image_filename}</div>
                <div style="font-size: 12px; color: #a1a1aa; display: flex; flex-direction: column; gap: 4px;">
                    <div>Dimensions: <b style="color: #f4f4f5;">{info['width']} × {info['height']} px</b></div>
                    <div>Aspect Ratio: <b style="color: #f4f4f5;">{info['aspect_ratio']}</b></div>
                    <div>File Size: <b style="color: #f4f4f5;">{info['file_size_formatted']}</b></div>
                    <div>Color Mode: <b style="color: #f4f4f5;">{info['mode']} ({info['channels']} ch)</b></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_side_a, col_side_b = st.columns(2)
        with col_side_a:
            if st.button("🔄 Reset", use_container_width=True, help="Revert to original uploaded image"):
                reset_to_original()
                st.rerun()
        with col_side_b:
            if st.button("➕ Stack", use_container_width=True, help="Set current result as base to apply another filter"):
                commit_processed_as_base()
                st.rerun()

        with st.expander("📁 Upload Another Image"):
            sidebar_upload = st.file_uploader(
                "Upload image file",
                type=["jpg", "jpeg", "png", "webp"],
                key="sidebar_uploader",
                label_visibility="collapsed",
            )
            if sidebar_upload is not None:
                sb_bytes = sidebar_upload.getvalue()
                valid, err, sb_img = validate_image_file(sb_bytes)
                if valid and sb_img:
                    load_image_into_state(sb_img, sidebar_upload.name, len(sb_bytes))
                    st.rerun()
                elif err:
                    st.error(err)

        if st.button("🗑️ Clear Canvas", use_container_width=True, help="Remove image and return to upload screen"):
            clear_image()
            st.rerun()
    else:
        st.info("💡 Upload an image or select a sample preset below to start editing.")

    st.markdown("<hr style='border-color: rgba(255,255,255,0.07); margin: 16px 0;'/>", unsafe_allow_html=True)

    # Preset Sample Images in Sidebar
    st.markdown("<div style='font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; color: #71717a; margin-bottom: 8px;'>Sample Presets</div>", unsafe_allow_html=True)
    sample_dir = Path("assets/sample_images")
    if sample_dir.exists():
        sample_files = sorted(list(sample_dir.glob("*.*")))
        for sfile in sample_files:
            btn_title = sfile.stem.replace('_', ' ').title()
            if st.button(f"🖼️ {btn_title}", key=f"sidebar_sample_{sfile.name}", use_container_width=True):
                load_sample_file(sfile)
                st.rerun()


# ==========================================
# MAIN CONTENT WORKSPACE
# ==========================================

# Top Minimalist Navigation Bar
header_html = f"""
<div class="studio-header">
    <div class="studio-brand">
        <span class="brand-title">PixelLab Studio</span>
        <span class="studio-chip">Ready</span>
        <span class="file-chip">{st.session_state.image_filename}</span>
    </div>
    <div style="font-size: 12px; color: #71717a;">
        Active Module: <b style="color: #f4f4f5;">{selected_category}</b>
    </div>
</div>
"""
st.markdown(header_html, unsafe_allow_html=True)

# ----------------------------------------------------
# EMPTY STATE: No Image Loaded Yet
# ----------------------------------------------------
if st.session_state.original_image is None:
    st.markdown(
        """
        <div class="upload-hero">
            <div style="font-size: 36px; margin-bottom: 10px; color: #818cf8;">
                <i class="hgi-stroke hgi-upload-04"></i>
            </div>
            <h2 style="font-size: 20px; font-weight: 600; color: #ffffff; margin-bottom: 6px;">Select or Drop Image to Open</h2>
            <p style="color: #71717a; max-width: 480px; margin: 0 auto 18px auto; font-size: 13px;">
                Supports JPG, JPEG, PNG (with full transparency), and WEBP.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
        key="main_uploader",
    )

    if uploaded_file is not None:
        try:
            file_bytes = uploaded_file.getvalue()
            is_valid, err_msg, loaded_img = validate_image_file(file_bytes)
            if not is_valid:
                st.error(f"❌ Upload Failed: {err_msg}")
            else:
                load_image_into_state(loaded_img, uploaded_file.name, len(file_bytes))
                st.rerun()
        except Exception as upload_err:
            st.error(f"Upload error: {upload_err}")

    # Instant Sample Presets
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size: 12px; font-weight: 600; color: #71717a; text-transform: uppercase; margin-bottom: 10px;'>Or open a built-in demo image with 1 click:</div>", unsafe_allow_html=True)
    scol1, scol2, scol3 = st.columns(3)

    sample_dir = Path("assets/sample_images")
    if sample_dir.exists():
        samples = sorted(list(sample_dir.glob("*.*")))
        cols = [scol1, scol2, scol3]
        for idx, sfile in enumerate(samples[:3]):
            with cols[idx]:
                title = sfile.stem.replace('_', ' ').title()
                if st.button(f"🖼️ {title}", key=f"hero_sample_{sfile.name}", use_container_width=True):
                    load_sample_file(sfile)
                    st.rerun()

    # Studio Capabilities Cards
    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
    st.markdown("<div class='panel-header'>Studio Capabilities</div>", unsafe_allow_html=True)
    fcol1, fcol2, fcol3 = st.columns(3)
    with fcol1:
        st.markdown(
            """
            <div class="studio-panel">
                <div class="panel-header"><i class="hgi-stroke hgi-filter"></i> 10 Creative Filters</div>
                <p style="font-size: 13px; color: #71717a; margin: 0;">Grayscale, Warm Sepia, Vintage film look, Gaussian & Median blur, Emboss, Inversion, and Sharpening.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with fcol2:
        st.markdown(
            """
            <div class="studio-panel">
                <div class="panel-header"><i class="hgi-stroke hgi-magic-wand-01"></i> Computer Vision CLAHE</div>
                <p style="font-size: 13px; color: #71717a; margin: 0;">Bilateral edge-preserving denoising, Unsharp Masking, and localized contrast equalization.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with fcol3:
        st.markdown(
            """
            <div class="studio-panel">
                <div class="panel-header"><i class="hgi-stroke hgi-scan"></i> Multi-Operator Edges</div>
                <p style="font-size: 13px; color: #71717a; margin: 0;">Interactive Canny with hysteresis thresholds, Sobel X & Y gradients, and Laplacian filters.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.stop()


# ----------------------------------------------------
# ACTIVE WORKSPACE: Image Loaded
# ----------------------------------------------------

# Quick Metrics Bar
img_info = get_image_info(st.session_state.processed_image, st.session_state.raw_bytes_len)
st.markdown(
    f"""
    <div class="metrics-bar">
        <div class="metric-card">
            <div class="metric-label">Dimensions</div>
            <div class="metric-val">{img_info['width']} × {img_info['height']}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Aspect Ratio</div>
            <div class="metric-val">{img_info['aspect_ratio']}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Resolution</div>
            <div class="metric-val">{img_info['megapixels']} MP</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Channels & Mode</div>
            <div class="metric-val">{img_info['mode']} ({img_info['channels']} ch)</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">File Size</div>
            <div class="metric-val">{img_info['file_size_formatted']}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Workspace Layout: Left Column = Preview & Comparison; Right Column = Controls
col_preview, col_controls = st.columns([1.15, 0.85], gap="large")

# ----------------------------------------------------
# RIGHT COLUMN: TOOL CONTROLS BY CATEGORY
# ----------------------------------------------------
with col_controls:
    st.markdown(f"<div class='panel-header'>{selected_category}</div>", unsafe_allow_html=True)

    # 1. FILTERS
    if selected_category == "Image Filters":
        st.caption("Apply color matrices, frequency blurs, and artistic convolutions.")

        filter_choice = st.selectbox(
            "Select Filter",
            [
                "Grayscale",
                "Sepia",
                "Black and White",
                "Vintage",
                "Gaussian Blur",
                "Median Blur",
                "Sharpen",
                "Emboss",
                "Invert Colors",
                "Edge Detection Filter",
            ],
        )

        preview_result = None

        if filter_choice == "Grayscale":
            intensity = st.slider("Desaturation Intensity", 0.0, 1.0, 1.0, 0.05)
            preview_result = apply_grayscale(st.session_state.original_image, intensity=intensity)

        elif filter_choice == "Sepia":
            intensity = st.slider("Sepia Warmth", 0.0, 1.0, 1.0, 0.05)
            preview_result = apply_sepia(st.session_state.original_image, intensity=intensity)

        elif filter_choice == "Black and White":
            threshold = st.slider("Binarization Cutoff Threshold", 0, 255, 128, 1)
            preview_result = apply_black_and_white(st.session_state.original_image, threshold=threshold)

        elif filter_choice == "Vintage":
            intensity = st.slider("Vintage Effect Strength", 0.0, 1.0, 0.85, 0.05)
            preview_result = apply_vintage(st.session_state.original_image, intensity=intensity)

        elif filter_choice == "Gaussian Blur":
            k_size = st.slider("Blur Kernel Radius (Odd Pixels)", 1, 51, 9, 2)
            preview_result = apply_gaussian_blur(st.session_state.original_image, kernel_size=k_size)

        elif filter_choice == "Median Blur":
            k_size = st.slider("Median Aperture (Odd Pixels)", 1, 31, 5, 2)
            preview_result = apply_median_blur(st.session_state.original_image, kernel_size=k_size)

        elif filter_choice == "Sharpen":
            strength = st.slider("Sharpen Factor", 0.0, 3.0, 1.2, 0.1)
            preview_result = apply_sharpen(st.session_state.original_image, strength=strength)

        elif filter_choice == "Emboss":
            intensity = st.slider("Emboss Depth", 0.0, 1.0, 1.0, 0.05)
            preview_result = apply_emboss(st.session_state.original_image, intensity=intensity)

        elif filter_choice == "Invert Colors":
            st.info("Inverts all color channels: R' = 255 - R, G' = 255 - G, B' = 255 - B.")
            preview_result = apply_invert_colors(st.session_state.original_image)

        elif filter_choice == "Edge Detection Filter":
            intensity = st.slider("Edge Highlight Intensity", 0.0, 1.0, 1.0, 0.05)
            preview_result = apply_edge_detection_filter(st.session_state.original_image, intensity=intensity)

        if preview_result is not None:
            st.session_state.processed_image = preview_result

    # 2. IMAGE ADJUSTMENTS
    elif selected_category == "Image Adjustments":
        st.caption("Fine-tune tonality, contrast, color saturation, and nonlinear gamma curves.")

        brightness_val = st.slider("Brightness", 0.1, 2.5, 1.0, 0.05)
        contrast_val = st.slider("Contrast", 0.1, 2.5, 1.0, 0.05)
        saturation_val = st.slider("Saturation", 0.0, 2.5, 1.0, 0.05)
        sharpness_val = st.slider("Sharpness", 0.0, 3.0, 1.0, 0.1)
        gamma_val = st.slider("Gamma Correction", 0.2, 2.5, 1.0, 0.05, help="Gamma < 1.0 darkens midtones; Gamma > 1.0 boosts shadow details.")

        if st.button("Reset Sliders to Default (1.0)"):
            st.session_state.processed_image = st.session_state.original_image.copy()
            st.rerun()

        st.session_state.processed_image = apply_adjustments_pipeline(
            st.session_state.original_image,
            brightness=brightness_val,
            contrast=contrast_val,
            saturation=saturation_val,
            sharpness=sharpness_val,
            gamma=gamma_val,
        )

    # 3. TRANSFORMATIONS
    elif selected_category == "Transformations":
        st.caption("Resize, crop, rotate, and flip image dimensions.")

        transform_tab1, transform_tab2, transform_tab3 = st.tabs(["Resize", "Crop", "Rotate & Flip"])

        # Tab 1: Resize
        with transform_tab1:
            orig_w, orig_h = st.session_state.original_image.size
            st.write(f"Original size: `{orig_w} × {orig_h} px`")

            maintain_aspect = st.checkbox("Lock Aspect Ratio", value=True)
            new_w = st.number_input("Target Width (px)", min_value=1, max_value=8000, value=orig_w, step=10)

            if maintain_aspect:
                calc_h = max(1, int(orig_h * (new_w / orig_w)))
                st.write(f"Computed Height: `{calc_h} px`")
                new_h = calc_h
            else:
                new_h = st.number_input("Target Height (px)", min_value=1, max_value=8000, value=orig_h, step=10)

            if st.button("Apply Resize", use_container_width=True):
                try:
                    resized_img = resize_image(
                        st.session_state.original_image,
                        target_width=new_w,
                        target_height=new_h,
                        maintain_aspect_ratio=False,
                    )
                    st.session_state.processed_image = resized_img
                    st.toast(f"Resized to {new_w} × {new_h} px", icon="📐")
                except Exception as e:
                    st.error(f"Resize failed: {e}")

        # Tab 2: Crop
        with transform_tab2:
            img_w, img_h = st.session_state.original_image.size
            st.write("Specify crop boundaries in pixels:")

            col_c1, col_c2 = st.columns(2)
            with col_c1:
                crop_left = st.number_input("Left X", min_value=0, max_value=img_w - 2, value=0, step=10)
                crop_top = st.number_input("Top Y", min_value=0, max_value=img_h - 2, value=0, step=10)
            with col_c2:
                crop_right = st.number_input("Right X", min_value=crop_left + 1, max_value=img_w, value=img_w, step=10)
                crop_bottom = st.number_input("Bottom Y", min_value=crop_top + 1, max_value=img_h, value=img_h, step=10)

            crop_w = crop_right - crop_left
            crop_h = crop_bottom - crop_top
            st.caption(f"Resulting Crop Area: `{crop_w} × {crop_h} px`")

            if st.button("Apply Crop", use_container_width=True):
                try:
                    cropped_img = crop_image(
                        st.session_state.original_image,
                        left=crop_left,
                        top=crop_top,
                        right=crop_right,
                        bottom=crop_bottom,
                    )
                    st.session_state.processed_image = cropped_img
                    st.toast(f"Cropped to {crop_w} × {crop_h} px", icon="✂️")
                except Exception as e:
                    st.error(f"Crop failed: {e}")

        # Tab 3: Rotate & Flip
        with transform_tab3:
            st.write("Fixed Rotations:")
            rcol1, rcol2, rcol3 = st.columns(3)
            with rcol1:
                if st.button("↷ 90° CW", use_container_width=True):
                    st.session_state.processed_image = rotate_90_clockwise(st.session_state.processed_image)
                    st.toast("Rotated 90° Clockwise", icon="↷")
            with rcol2:
                if st.button("↻ 180°", use_container_width=True):
                    st.session_state.processed_image = rotate_180(st.session_state.processed_image)
                    st.toast("Rotated 180°", icon="↻")
            with rcol3:
                if st.button("↶ 270° CW", use_container_width=True):
                    st.session_state.processed_image = rotate_270_clockwise(st.session_state.processed_image)
                    st.toast("Rotated 270° Clockwise", icon="↶")

            st.write("Flips:")
            fcol1, fcol2 = st.columns(2)
            with fcol1:
                if st.button("⇄ Flip Horizontal", use_container_width=True):
                    st.session_state.processed_image = flip_horizontal(st.session_state.processed_image)
                    st.toast("Flipped Horizontally", icon="⇄")
            with fcol2:
                if st.button("⇅ Flip Vertical", use_container_width=True):
                    st.session_state.processed_image = flip_vertical(st.session_state.processed_image)
                    st.toast("Flipped Vertically", icon="⇅")

            st.write("Arbitrary Rotation:")
            custom_angle = st.slider("Rotation Angle (°)", -180.0, 180.0, 0.0, 1.0)
            expand_canvas = st.checkbox("Expand Canvas to Fit Corners", value=True)
            if st.button("Apply Arbitrary Rotation", use_container_width=True):
                st.session_state.processed_image = rotate_custom(
                    st.session_state.original_image,
                    angle=custom_angle,
                    expand=expand_canvas,
                )
                st.toast(f"Rotated {custom_angle}°", icon="🔄")

    # 4. ENHANCEMENT
    elif selected_category == "Image Enhancement":
        st.caption("Computer Vision algorithms for sharpness, noise attenuation, and adaptive contrast.")

        enhance_op = st.selectbox(
            "Select Enhancement Algorithm",
            [
                "Unsharp Masking (USM) Sharpen",
                "Bilateral Noise Reduction",
                "Gaussian Smoothing",
                "Global Histogram Equalization",
                "CLAHE (Adaptive Local Equalization)",
            ],
        )

        if enhance_op == "Unsharp Masking (USM) Sharpen":
            st.caption("Subtracts a Gaussian low-pass blurred copy to amplify high-frequency textural detail.")
            usm_amount = st.slider("Sharpen Amount", 0.1, 3.0, 1.2, 0.1)
            usm_radius = st.slider("Blur Kernel Radius", 1, 15, 3, 2)
            st.session_state.processed_image = enhance_sharpen(
                st.session_state.original_image,
                amount=usm_amount,
                radius=usm_radius,
            )

        elif enhance_op == "Bilateral Noise Reduction":
            st.caption("Edge-preserving filter: averages pixels in spatial and color space, preserving sharp boundaries.")
            diameter = st.slider("Neighborhood Diameter", 3, 21, 9, 2)
            sigma_c = st.slider("Color Sigma (Tolerance)", 10.0, 150.0, 75.0, 5.0)
            sigma_s = st.slider("Space Sigma", 10.0, 150.0, 75.0, 5.0)
            st.session_state.processed_image = enhance_noise_reduction(
                st.session_state.original_image,
                diameter=diameter,
                sigma_color=sigma_c,
                sigma_space=sigma_s,
            )

        elif enhance_op == "Gaussian Smoothing":
            k_size = st.slider("Smoothing Kernel Size", 3, 31, 7, 2)
            sigma = st.slider("Gaussian Sigma", 0.5, 10.0, 1.5, 0.5)
            st.session_state.processed_image = enhance_gaussian_smoothing(
                st.session_state.original_image,
                kernel_size=k_size,
                sigma=sigma,
            )

        elif enhance_op == "Global Histogram Equalization":
            st.caption("Converts image to YCrCb space and redistributes the Luminance (Y) channel across dynamic range.")
            st.session_state.processed_image = enhance_histogram_equalization(st.session_state.original_image)

        elif enhance_op == "CLAHE (Adaptive Local Equalization)":
            st.caption("Divides the Lightness channel into contextual tiles and limits peak amplification to prevent noise.")
            clip_limit = st.slider("Contrast Clip Limit", 1.0, 10.0, 2.5, 0.5)
            grid_size = st.slider("Tile Grid Size (N × N)", 2, 24, 8, 2)
            st.session_state.processed_image = enhance_clahe(
                st.session_state.original_image,
                clip_limit=clip_limit,
                tile_grid_size=grid_size,
            )

    # 5. EDGE DETECTION
    elif selected_category == "Edge Detection":
        st.caption("Compute first and second spatial derivatives to extract structural outlines and boundaries.")

        edge_op = st.selectbox(
            "Select Edge Detector",
            [
                "Canny Edge Detector",
                "Sobel X (Vertical Edges)",
                "Sobel Y (Horizontal Edges)",
                "Sobel Combined (Gradient Magnitude)",
                "Laplacian (Second Derivative)",
            ],
        )

        if edge_op == "Canny Edge Detector":
            st.caption("Multi-stage detector with Gaussian smoothing, gradient computation, non-max suppression, and hysteresis.")
            t1 = st.slider("Threshold 1 (Lower Hysteresis)", 0, 400, 100, 5)
            t2 = st.slider("Threshold 2 (Upper Hysteresis)", 0, 500, 200, 5)
            aperture = st.selectbox("Sobel Aperture Size", [3, 5, 7], index=0)
            st.session_state.processed_image = detect_canny(
                st.session_state.original_image,
                threshold1=t1,
                threshold2=t2,
                aperture_size=aperture,
            )

        elif edge_op == "Sobel X (Vertical Edges)":
            st.caption("Highlights vertical edges by calculating the horizontal gradient derivative Gx.")
            k = st.selectbox("Kernel Size", [1, 3, 5, 7], index=1)
            st.session_state.processed_image = detect_sobel_x(st.session_state.original_image, ksize=k)

        elif edge_op == "Sobel Y (Horizontal Edges)":
            st.caption("Highlights horizontal edges by calculating the vertical gradient derivative Gy.")
            k = st.selectbox("Kernel Size", [1, 3, 5, 7], index=1)
            st.session_state.processed_image = detect_sobel_y(st.session_state.original_image, ksize=k)

        elif edge_op == "Sobel Combined (Gradient Magnitude)":
            st.caption("Computes Euclidean gradient magnitude: sqrt(Gx² + Gy²) across all directions.")
            k = st.selectbox("Kernel Size", [1, 3, 5, 7], index=1)
            st.session_state.processed_image = detect_sobel_combined(st.session_state.original_image, ksize=k)

        elif edge_op == "Laplacian (Second Derivative)":
            st.caption("Isolates rapid intensity inflection points and zero-crossings using 2nd order derivatives.")
            k = st.selectbox("Aperture Size", [1, 3, 5, 7], index=1)
            st.session_state.processed_image = detect_laplacian(st.session_state.original_image, ksize=k)

    # 6. IMAGE ANALYSIS
    elif selected_category == "Image Analysis":
        st.caption("Statistical metrics and channel intensity distributions.")

        analysis_tab1, analysis_tab2 = st.tabs(["Histograms", "Channel Statistics"])

        with analysis_tab1:
            hist_type = st.radio(
                "Histogram View",
                ["Overlaid RGB", "Separate Channels (R, G, B)", "Grayscale Luminance"],
                horizontal=True,
            )

            if hist_type == "Overlaid RGB":
                fig = plot_rgb_histogram(st.session_state.processed_image)
                st.pyplot(fig, use_container_width=True)
            elif hist_type == "Separate Channels (R, G, B)":
                fig = plot_separate_channels(st.session_state.processed_image)
                st.pyplot(fig, use_container_width=True)
            else:
                fig = plot_grayscale_histogram(st.session_state.processed_image)
                st.pyplot(fig, use_container_width=True)

        with analysis_tab2:
            stats = get_image_statistics(st.session_state.processed_image)
            st.write("Channel Distribution Statistics:")
            col_s1, col_s2, col_s3 = st.columns(3)
            with col_s1:
                st.markdown(
                    f"""
                    <div class="metric-card" style="border-left: 3px solid #ef4444;">
                        <div class="metric-label" style="color: #ef4444;">RED CHANNEL</div>
                        <div>Mean: <b>{stats['red']['mean']:.1f}</b></div>
                        <div>Std: <b>{stats['red']['std']:.1f}</b></div>
                        <div>Range: <b>{stats['red']['min']} - {stats['red']['max']}</b></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with col_s2:
                st.markdown(
                    f"""
                    <div class="metric-card" style="border-left: 3px solid #22c55e;">
                        <div class="metric-label" style="color: #22c55e;">GREEN CHANNEL</div>
                        <div>Mean: <b>{stats['green']['mean']:.1f}</b></div>
                        <div>Std: <b>{stats['green']['std']:.1f}</b></div>
                        <div>Range: <b>{stats['green']['min']} - {stats['green']['max']}</b></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with col_s3:
                st.markdown(
                    f"""
                    <div class="metric-card" style="border-left: 3px solid #3b82f6;">
                        <div class="metric-label" style="color: #3b82f6;">BLUE CHANNEL</div>
                        <div>Mean: <b>{stats['blue']['mean']:.1f}</b></div>
                        <div>Std: <b>{stats['blue']['std']:.1f}</b></div>
                        <div>Range: <b>{stats['blue']['min']} - {stats['blue']['max']}</b></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # 7. COMPRESSION & EXPORT
    elif selected_category == "Compression & Export":
        st.caption("Convert between PNG, JPEG, and WEBP formats with tunable lossy compression and size verification.")

        export_format = st.selectbox("Target File Format", ["PNG", "JPEG", "WEBP"], index=1)
        quality = 95
        if export_format in ("JPEG", "WEBP"):
            quality = st.slider("Encoding Quality (1 = Smallest, 100 = Best)", 5, 100, 80, 5)

        # Compress and analyze
        comp_bytes, stats = compress_image(
            st.session_state.processed_image,
            target_format=export_format,
            quality=quality,
            original_size_bytes=st.session_state.raw_bytes_len,
        )

        st.markdown(
            f"""
            <div class="studio-panel" style="margin-top: 14px;">
                <div class="panel-header">Compression Analytics</div>
                <div style="font-size: 13px; line-height: 1.8;">
                    <div>• Original Upload Size: <b>{stats['original_size_formatted']}</b></div>
                    <div>• Compressed File Size: <b style="color: #60a5fa;">{stats['new_size_formatted']}</b></div>
                    <div>• Output Format: <b>{stats['format']}</b> (Quality: {stats['quality']})</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if stats["original_size_bytes"] > 0:
            if stats["is_reduced"]:
                st.success(f"🎉 Reduced by {stats['reduction_percentage']}% ({format_bytes(stats['original_size_bytes'] - stats['new_size_bytes'])} saved)")
            else:
                pct_increase = abs(stats["reduction_percentage"])
                st.warning(f"ℹ️ File size increased by {pct_increase}%. (Lossless format or high quality setting produces larger payload).")

        base_stem = Path(st.session_state.image_filename).stem
        download_name = f"{base_stem}_pixellab{stats['extension']}"
        st.download_button(
            label=f"⬇️ Download {stats['format']} ({stats['new_size_formatted']})",
            data=comp_bytes,
            file_name=download_name,
            mime=stats["mime_type"],
            use_container_width=True,
        )

    # 8. WATERMARK STUDIO
    elif selected_category == "Text Watermark":
        st.caption("Add an anti-tamper or branded text watermark with adjustable opacity, positioning, and color.")

        wm_text = st.text_input("Watermark Text", value="© PixelLab Studio")
        wm_size = st.slider("Font Size (px)", 12, 120, 36, 2)
        wm_opacity = st.slider("Opacity", 0.05, 1.0, 0.6, 0.05)
        wm_pos = st.selectbox(
            "Position",
            ["Bottom-Right", "Bottom-Left", "Top-Right", "Top-Left", "Center", "Custom"],
        )

        custom_x, custom_y = 20, 20
        if wm_pos == "Custom":
            img_w, img_h = st.session_state.original_image.size
            col_x, col_y = st.columns(2)
            with col_x:
                custom_x = st.number_input("X Position", 0, img_w, 40)
            with col_y:
                custom_y = st.number_input("Y Position", 0, img_h, 40)

        wm_color = st.color_picker("Text Color", "#FFFFFF")

        watermarked_img = add_text_watermark(
            st.session_state.original_image,
            text=wm_text,
            font_size=wm_size,
            opacity=wm_opacity,
            position=wm_pos,
            color=wm_color,
            custom_coords=(custom_x, custom_y),
        )
        st.session_state.processed_image = watermarked_img


# ----------------------------------------------------
# LEFT COLUMN: PREVIEW WORKSPACE & COMPARISON
# ----------------------------------------------------
with col_preview:
    st.markdown("<div class='panel-header'>Canvas & Preview</div>", unsafe_allow_html=True)

    view_mode = st.radio(
        "View Mode",
        ["Processed Output", "Side-by-Side Comparison", "Original Image"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if view_mode == "Processed Output":
        st.image(
            st.session_state.processed_image,
            use_container_width=True,
            caption=f"Processed Image ({st.session_state.processed_image.size[0]} × {st.session_state.processed_image.size[1]} px)",
        )

    elif view_mode == "Side-by-Side Comparison":
        cmp_col1, cmp_col2 = st.columns(2)
        with cmp_col1:
            st.markdown("<div style='font-size: 11px; font-weight: 600; color: #71717a; text-transform: uppercase; margin-bottom: 4px;'>ORIGINAL</div>", unsafe_allow_html=True)
            st.image(st.session_state.original_image, use_container_width=True)
        with cmp_col2:
            st.markdown("<div style='font-size: 11px; font-weight: 600; color: #818cf8; text-transform: uppercase; margin-bottom: 4px;'>PROCESSED</div>", unsafe_allow_html=True)
            st.image(st.session_state.processed_image, use_container_width=True)

    elif view_mode == "Original Image":
        st.image(
            st.session_state.original_image,
            use_container_width=True,
            caption=f"Original Upload ({st.session_state.original_image.size[0]} × {st.session_state.original_image.size[1]} px)",
        )

    # Fast Download Action Bar
    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
    dcol1, dcol2, dcol3 = st.columns([1, 1, 1])
    with dcol1:
        png_data = image_to_bytes(st.session_state.processed_image, format="PNG")
        st.download_button(
            label="⬇️ PNG",
            data=png_data,
            file_name=f"{Path(st.session_state.image_filename).stem}_pixellab.png",
            mime="image/png",
            use_container_width=True,
        )
    with dcol2:
        jpg_data = image_to_bytes(st.session_state.processed_image, format="JPEG", quality=90)
        st.download_button(
            label="⬇️ JPG (90%)",
            data=jpg_data,
            file_name=f"{Path(st.session_state.image_filename).stem}_pixellab.jpg",
            mime="image/jpeg",
            use_container_width=True,
        )
    with dcol3:
        webp_data = image_to_bytes(st.session_state.processed_image, format="WEBP", quality=85)
        st.download_button(
            label="⬇️ WEBP (85%)",
            data=webp_data,
            file_name=f"{Path(st.session_state.image_filename).stem}_pixellab.webp",
            mime="image/webp",
            use_container_width=True,
        )
