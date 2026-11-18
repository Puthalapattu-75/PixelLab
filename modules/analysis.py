"""
PixelLab - Image Analysis Module
Provides statistical profiling and histogram visualizations using Matplotlib:
- Comprehensive metadata and channel statistics
- Combined RGB Histogram (with dark modern creative UI theme)
- Separate Red, Green, and Blue Channel Histograms
- Grayscale / Luminance Histogram
"""

from typing import Dict, Any
from PIL import Image
import numpy as np
import matplotlib
# Use non-interactive Agg backend for thread-safe rendering
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from modules.image_utils import ensure_rgb, get_image_info

# Dark theme palette inspired by modern developer creative tools
THEME_BG = "#0f1117"
THEME_CARD_BG = "#161922"
THEME_TEXT = "#e2e8f0"
THEME_MUTED = "#94a3b8"
THEME_GRID = "#262b3d"

COLOR_RED = "#ef4444"
COLOR_GREEN = "#22c55e"
COLOR_BLUE = "#3b82f6"
COLOR_GRAY = "#cbd5e1"


def get_image_statistics(image: Image.Image) -> Dict[str, Any]:
    """
    Compute channel-level statistical metrics (mean, std dev, min, max).

    Args:
        image (Image.Image): Input PIL Image.

    Returns:
        Dict[str, Any]: Dictionary of statistical metrics per channel and overall.
    """
    rgb_img = ensure_rgb(image)
    arr = np.array(rgb_img)

    r_channel = arr[:, :, 0]
    g_channel = arr[:, :, 1]
    b_channel = arr[:, :, 2]

    # Luminance approximation: 0.299 R + 0.587 G + 0.114 B
    luminance = 0.299 * r_channel + 0.587 * g_channel + 0.114 * b_channel

    return {
        "red": {
            "mean": float(np.mean(r_channel)),
            "std": float(np.std(r_channel)),
            "min": int(np.min(r_channel)),
            "max": int(np.max(r_channel)),
            "median": float(np.median(r_channel)),
        },
        "green": {
            "mean": float(np.mean(g_channel)),
            "std": float(np.std(g_channel)),
            "min": int(np.min(g_channel)),
            "max": int(np.max(g_channel)),
            "median": float(np.median(g_channel)),
        },
        "blue": {
            "mean": float(np.mean(b_channel)),
            "std": float(np.std(b_channel)),
            "min": int(np.min(b_channel)),
            "max": int(np.max(b_channel)),
            "median": float(np.median(b_channel)),
        },
        "luminance": {
            "mean": float(np.mean(luminance)),
            "std": float(np.std(luminance)),
            "min": float(np.min(luminance)),
            "max": float(np.max(luminance)),
        },
    }


def _apply_dark_theme(fig, ax):
    """Internal helper to apply consistent dark styling to Matplotlib axes."""
    fig.patch.set_facecolor(THEME_BG)
    ax.set_facecolor(THEME_CARD_BG)
    ax.tick_params(colors=THEME_MUTED, labelsize=9)
    ax.spines["bottom"].set_color(THEME_GRID)
    ax.spines["top"].set_color(THEME_GRID)
    ax.spines["left"].set_color(THEME_GRID)
    ax.spines["right"].set_color(THEME_GRID)
    ax.xaxis.label.set_color(THEME_TEXT)
    ax.yaxis.label.set_color(THEME_TEXT)
    ax.title.set_color(THEME_TEXT)
    ax.grid(True, linestyle="--", linewidth=0.5, color=THEME_GRID, alpha=0.7)


def plot_rgb_histogram(image: Image.Image) -> plt.Figure:
    """
    Generate an overlaid RGB color distribution histogram.

    How it works:
    Counts pixel frequency for each of the 256 intensity bins across the Red, Green,
    and Blue channels, rendering smooth translucent curves for direct tonal comparison.

    Args:
        image (Image.Image): Input PIL Image.

    Returns:
        plt.Figure: Matplotlib figure.
    """
    rgb_img = ensure_rgb(image)
    arr = np.array(rgb_img)

    fig, ax = plt.subplots(figsize=(7, 3.5), dpi=100)
    _apply_dark_theme(fig, ax)

    bins = np.arange(257)

    # Compute histogram for each channel
    hist_r, _ = np.histogram(arr[:, :, 0], bins=bins)
    hist_g, _ = np.histogram(arr[:, :, 1], bins=bins)
    hist_b, _ = np.histogram(arr[:, :, 2], bins=bins)

    x = np.arange(256)
    ax.plot(x, hist_r, color=COLOR_RED, label="Red", linewidth=1.5, alpha=0.85)
    ax.fill_between(x, hist_r, color=COLOR_RED, alpha=0.15)

    ax.plot(x, hist_g, color=COLOR_GREEN, label="Green", linewidth=1.5, alpha=0.85)
    ax.fill_between(x, hist_g, color=COLOR_GREEN, alpha=0.15)

    ax.plot(x, hist_b, color=COLOR_BLUE, label="Blue", linewidth=1.5, alpha=0.85)
    ax.fill_between(x, hist_b, color=COLOR_BLUE, alpha=0.15)

    ax.set_xlim([0, 255])
    ax.set_title("RGB Channel Distribution", fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel("Pixel Intensity (0 - 255)", fontsize=9)
    ax.set_ylabel("Pixel Count", fontsize=9)

    legend = ax.legend(facecolor=THEME_CARD_BG, edgecolor=THEME_GRID, fontsize=9)
    for text in legend.get_texts():
        text.set_color(THEME_TEXT)

    fig.tight_layout()
    return fig


def plot_separate_channels(image: Image.Image) -> plt.Figure:
    """
    Generate individual subplots for Red, Green, and Blue channel distributions.

    Args:
        image (Image.Image): Input PIL Image.

    Returns:
        plt.Figure: Matplotlib figure with 3 side-by-side subplots.
    """
    rgb_img = ensure_rgb(image)
    arr = np.array(rgb_img)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), dpi=100, sharey=True)
    fig.patch.set_facecolor(THEME_BG)

    channels = [
        ("Red Channel", arr[:, :, 0], COLOR_RED),
        ("Green Channel", arr[:, :, 1], COLOR_GREEN),
        ("Blue Channel", arr[:, :, 2], COLOR_BLUE),
    ]

    bins = np.arange(257)
    x = np.arange(256)

    for ax, (title, ch_data, color) in zip(axes, channels):
        _apply_dark_theme(fig, ax)
        hist, _ = np.histogram(ch_data, bins=bins)
        ax.plot(x, hist, color=color, linewidth=1.5)
        ax.fill_between(x, hist, color=color, alpha=0.25)
        ax.set_xlim([0, 255])
        ax.set_title(title, fontsize=10, fontweight="bold", color=color)
        ax.set_xlabel("Intensity", fontsize=8)

    axes[0].set_ylabel("Pixel Count", fontsize=8)
    fig.tight_layout()
    return fig


def plot_grayscale_histogram(image: Image.Image) -> plt.Figure:
    """
    Generate Grayscale Luminance distribution histogram with mean indicator.

    Args:
        image (Image.Image): Input PIL Image.

    Returns:
        plt.Figure: Matplotlib figure.
    """
    rgb_img = ensure_rgb(image)
    gray_img = rgb_img.convert("L")
    arr = np.array(gray_img)

    fig, ax = plt.subplots(figsize=(7, 3.5), dpi=100)
    _apply_dark_theme(fig, ax)

    bins = np.arange(257)
    hist, _ = np.histogram(arr, bins=bins)
    x = np.arange(256)

    mean_val = float(np.mean(arr))

    ax.plot(x, hist, color=COLOR_GRAY, label="Luminance", linewidth=1.5)
    ax.fill_between(x, hist, color=COLOR_GRAY, alpha=0.2)

    # Highlight mean luminance line
    ax.axvline(
        mean_val,
        color="#fbbf24",
        linestyle="--",
        linewidth=1.5,
        label=f"Mean: {mean_val:.1f}",
    )

    ax.set_xlim([0, 255])
    ax.set_title("Grayscale / Luminance Distribution", fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel("Pixel Luminosity (0 = Black, 255 = White)", fontsize=9)
    ax.set_ylabel("Pixel Count", fontsize=9)

    legend = ax.legend(facecolor=THEME_CARD_BG, edgecolor=THEME_GRID, fontsize=9)
    for text in legend.get_texts():
        text.set_color(THEME_TEXT)

    fig.tight_layout()
    return fig
