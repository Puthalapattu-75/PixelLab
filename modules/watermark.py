"""
PixelLab - Watermark Module
Provides customizable text watermark rendering with opacity, positioning,
and cross-platform font discovery.
"""

from typing import Tuple, Union
import os
from PIL import Image, ImageDraw, ImageFont
from modules.image_utils import ensure_rgba, ensure_rgb

# Standard system font candidates for Windows, Linux, and macOS
FONT_CANDIDATES = [
    # Windows
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/segoeui.ttf",
    "C:/Windows/Fonts/calibri.ttf",
    # Linux / macOS
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "arial.ttf",
    "DejaVuSans.ttf",
]


def _get_truetype_font(font_size: int = 36) -> ImageFont.ImageFont:
    """
    Locate and load a high-quality TrueType font, falling back to default if unavailable.

    Args:
        font_size (int): Font size in pixels.

    Returns:
        ImageFont: Loaded PIL ImageFont instance.
    """
    for font_path in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(font_path, size=font_size)
        except (IOError, OSError):
            continue

    # Try bare name lookup via PIL's internal font directory
    try:
        return ImageFont.truetype("arial.ttf", size=font_size)
    except (IOError, OSError):
        pass

    # Safe fallback (default bitmap font)
    return ImageFont.load_default()


def _hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert hex color string (e.g. '#ffffff' or 'ffffff') to RGB tuple."""
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 3:
        hex_color = "".join([c * 2 for c in hex_color])
    if len(hex_color) != 6:
        return (255, 255, 255)  # Default white
    try:
        return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        return (255, 255, 255)


def add_text_watermark(
    image: Image.Image,
    text: str,
    font_size: int = 36,
    opacity: float = 0.5,
    position: str = "Bottom-Right",
    color: Union[str, Tuple[int, int, int]] = "#FFFFFF",
    custom_coords: Tuple[int, int] = (20, 20),
    padding: int = 24,
) -> Image.Image:
    """
    Apply a customizable text watermark with alpha blending.

    How it works:
    1. Converts base image to RGBA.
    2. Creates a completely transparent overlay matching the image dimensions.
    3. Computes the text bounding box for exact pixel placement.
    4. Renders text with alpha = (opacity * 255).
    5. Performs alpha compositing: Output = Over(Overlay, Base).

    Args:
        image (Image.Image): Input PIL Image.
        text (str): Watermark text.
        font_size (int): Size of font in points/pixels (default 36).
        opacity (float): Opacity from 0.0 (invisible) to 1.0 (fully opaque, default 0.5).
        position (str): 'Top-Left', 'Top-Right', 'Bottom-Left', 'Bottom-Right', 'Center', or 'Custom'.
        color (Union[str, Tuple]): Hex string ('#ffffff') or RGB tuple.
        custom_coords (Tuple[int, int]): (x, y) coordinates if position is 'Custom'.
        padding (int): Pixel margin from image edges (default 24).

    Returns:
        Image.Image: Watermarked PIL Image.
    """
    if not text or not text.strip():
        return image.copy()

    # Parse color
    if isinstance(color, str):
        rgb_color = _hex_to_rgb(color)
    else:
        rgb_color = color[:3]

    alpha_val = max(0, min(255, int(opacity * 255)))
    fill_rgba = (rgb_color[0], rgb_color[1], rgb_color[2], alpha_val)

    # Convert base to RGBA
    orig_mode = image.mode
    base_rgba = ensure_rgba(image)
    img_w, img_h = base_rgba.size

    # Create transparent drawing layer
    overlay = Image.new("RGBA", (img_w, img_h), (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    # Load font
    font = _get_truetype_font(max(10, font_size))

    # Calculate text bounding dimensions
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    # Compute (x, y) coordinates
    pos_lower = position.lower().replace(" ", "").replace("-", "")
    if "topleft" in pos_lower:
        x, y = padding, padding
    elif "topright" in pos_lower:
        x = max(padding, img_w - text_w - padding)
        y = padding
    elif "bottomleft" in pos_lower:
        x = padding
        y = max(padding, img_h - text_h - padding)
    elif "center" in pos_lower:
        x = max(0, (img_w - text_w) // 2)
        y = max(0, (img_h - text_h) // 2)
    elif "custom" in pos_lower:
        x, y = custom_coords
        x = max(0, min(int(x), img_w - 1))
        y = max(0, min(int(y), img_h - 1))
    else:  # Default: Bottom-Right
        x = max(padding, img_w - text_w - padding)
        y = max(padding, img_h - text_h - padding)

    # Draw text onto overlay
    draw.text((x, y), text, font=font, fill=fill_rgba)

    # Composite overlay over base image
    watermarked = Image.alpha_composite(base_rgba, overlay)

    # Restore original RGB mode if source was RGB
    if orig_mode == "RGB":
        return watermarked.convert("RGB")

    return watermarked
