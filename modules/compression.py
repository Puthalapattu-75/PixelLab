"""
PixelLab - Compression & Format Conversion Module
Handles multi-format image export, quality-tuning compression, transparency
compositing for JPEG, and accurate size reduction analytics.
"""

from typing import Tuple, Dict, Any, Optional
import io
from PIL import Image
from modules.image_utils import ensure_rgb, format_bytes

# Supported target export formats and their MIME types
FORMAT_CONFIG = {
    "PNG": {
        "extension": ".png",
        "mime": "image/png",
        "supports_quality": False,
        "lossless": True,
    },
    "JPEG": {
        "extension": ".jpg",
        "mime": "image/jpeg",
        "supports_quality": True,
        "lossless": False,
    },
    "WEBP": {
        "extension": ".webp",
        "mime": "image/webp",
        "supports_quality": True,
        "lossless": False,
    },
}


def compress_image(
    image: Image.Image,
    target_format: str = "JPEG",
    quality: int = 80,
    original_size_bytes: int = 0,
) -> Tuple[bytes, Dict[str, Any]]:
    """
    Compress an image into the chosen format and compute accurate size analytics.

    How it works:
    - JPEG: Uses Discrete Cosine Transform (DCT) lossy compression with standard quantization tables.
      Alpha channel transparency is safely flattened onto pure white.
    - WEBP: Modern predictive and entropy coding format offering superior compression efficiency.
    - PNG: DEFLATE lossless compression with line filtering and optimization pass.

    Args:
        image (Image.Image): Input PIL Image.
        target_format (str): 'PNG', 'JPEG', or 'WEBP'.
        quality (int): Compression quality (1-100) for JPEG and WEBP.
        original_size_bytes (int): Original file size in bytes to compare against.

    Returns:
        Tuple[bytes, Dict[str, Any]]:
            - bytes: The encoded compressed image data.
            - Dict: Analytics dictionary with original size, new size, percentage reduction, etc.
    """
    fmt = target_format.upper()
    if fmt not in FORMAT_CONFIG:
        fmt = "JPEG"

    quality = max(1, min(100, int(quality)))
    buffer = io.BytesIO()

    if fmt == "JPEG":
        # Ensure RGBA is safely flattened onto white to avoid black background artifacts
        rgb_img = ensure_rgb(image)
        rgb_img.save(buffer, format="JPEG", quality=quality, optimize=True)
    elif fmt == "WEBP":
        image.save(buffer, format="WEBP", quality=quality, method=6)
    else:  # PNG
        image.save(buffer, format="PNG", optimize=True)

    buffer.seek(0)
    compressed_bytes = buffer.getvalue()
    new_size_bytes = len(compressed_bytes)

    # Compute comparative metrics
    has_orig = original_size_bytes > 0
    if has_orig:
        size_delta = original_size_bytes - new_size_bytes
        reduction_percentage = (size_delta / original_size_bytes) * 100.0
        is_reduced = new_size_bytes < original_size_bytes
    else:
        size_delta = 0
        reduction_percentage = 0.0
        is_reduced = False

    stats = {
        "format": fmt,
        "quality": quality,
        "original_size_bytes": original_size_bytes,
        "original_size_formatted": format_bytes(original_size_bytes) if has_orig else "N/A",
        "new_size_bytes": new_size_bytes,
        "new_size_formatted": format_bytes(new_size_bytes),
        "is_reduced": is_reduced,
        "reduction_percentage": round(reduction_percentage, 2),
        "mime_type": FORMAT_CONFIG[fmt]["mime"],
        "extension": FORMAT_CONFIG[fmt]["extension"],
    }

    return compressed_bytes, stats


def convert_image_format(
    image: Image.Image,
    target_format: str = "PNG",
    quality: int = 95,
) -> Tuple[bytes, str, str]:
    """
    Convert image to a specific format (PNG, JPEG, WEBP) ready for download.

    Args:
        image (Image.Image): Input PIL Image.
        target_format (str): 'PNG', 'JPEG', or 'WEBP'.
        quality (int): Quality level (default 95).

    Returns:
        Tuple[bytes, str, str]:
            - bytes: Image binary data.
            - str: MIME type (e.g. 'image/png').
            - str: File extension (e.g. '.png').
    """
    fmt = target_format.upper()
    if fmt not in FORMAT_CONFIG:
        fmt = "PNG"

    data, stats = compress_image(image, target_format=fmt, quality=quality, original_size_bytes=0)
    return data, stats["mime_type"], stats["extension"]
