"""
PixelLab - Image Utilities Module
Provides helper functions for image validation, color format conversions (PIL <-> OpenCV),
channel management, metadata extraction, and buffer export.
"""

from typing import Tuple, Optional, Dict, Any
import io
import math
from PIL import Image, ImageOps
import numpy as np
import cv2

# Supported image extensions and formats
SUPPORTED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
SUPPORTED_FORMATS = {"JPEG", "JPG", "PNG", "WEBP", "MPO"}


def format_bytes(size_in_bytes: int) -> str:
    """
    Format a byte count into a human-readable string (B, KB, MB).

    Args:
        size_in_bytes (int): Size in bytes.

    Returns:
        str: Human-friendly string representation, e.g. '245.20 KB'.
    """
    if size_in_bytes <= 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB"]
    digit_group = min(int(math.log(size_in_bytes, 1024)), len(units) - 1)
    formatted_value = size_in_bytes / (1024 ** digit_group)
    return f"{formatted_value:.2f} {units[digit_group]}"


def calculate_aspect_ratio(width: int, height: int) -> str:
    """
    Calculate simplified aspect ratio or decimal ratio string.

    Args:
        width (int): Image width in pixels.
        height (int): Image height in pixels.

    Returns:
        str: Aspect ratio string (e.g. '16:9' or '4:3' or '1.78:1').
    """
    if width <= 0 or height <= 0:
        return "N/A"
    gcd = math.gcd(width, height)
    simplified_w = width // gcd
    simplified_h = height // gcd
    # If ratio numbers are small and clean, return simplified (e.g. 16:9, 4:3, 1:1)
    if simplified_w <= 32 and simplified_h <= 32:
        return f"{simplified_w}:{simplified_h}"
    decimal_ratio = width / height
    return f"{decimal_ratio:.2f}:1"


def validate_image_file(file_obj_or_bytes) -> Tuple[bool, Optional[str], Optional[Image.Image]]:
    """
    Validate that an uploaded file or byte stream contains a valid, supported image.

    Args:
        file_obj_or_bytes: A file-like object, Streamlit UploadedFile, or bytes.

    Returns:
        Tuple[bool, Optional[str], Optional[Image.Image]]:
            - bool: True if valid, False otherwise.
            - Optional[str]: Error message if invalid, None if valid.
            - Optional[Image.Image]: Opened and verified PIL Image if valid, None if invalid.
    """
    try:
        # Convert any input type (bytes, UploadedFile, file-like) into a clean in-memory BytesIO stream
        if isinstance(file_obj_or_bytes, (bytes, bytearray)):
            stream = io.BytesIO(file_obj_or_bytes)
        elif hasattr(file_obj_or_bytes, "read"):
            if hasattr(file_obj_or_bytes, "seek"):
                file_obj_or_bytes.seek(0)
            data = file_obj_or_bytes.read()
            stream = io.BytesIO(data)
            if hasattr(file_obj_or_bytes, "seek"):
                file_obj_or_bytes.seek(0)
        else:
            return False, "Invalid file object provided.", None

        # Check for empty file
        raw_bytes = stream.getvalue()
        if len(raw_bytes) == 0:
            return False, "The uploaded file is empty (0 bytes).", None

        # Open and verify image integrity without reading entire data
        stream.seek(0)
        img = Image.open(stream)
        img.verify()

        # Reopen image because verify() marks the stream/image as closed for reading
        stream.seek(0)
        img = Image.open(stream)

        # Check format
        img_format = (img.format or "").upper()
        if img_format not in SUPPORTED_FORMATS:
            return (
                False,
                f"Unsupported format '{img_format}'. PixelLab supports JPG, JPEG, PNG, and WEBP.",
                None,
            )

        # Force load image data into memory to catch truncation or corruption errors
        img.load()

        # Auto-orient based on EXIF if present
        img = ImageOps.exif_transpose(img)

        return True, None, img

    except (Image.UnidentifiedImageError, OSError, ValueError) as err:
        return False, f"Invalid or corrupted image file: {str(err)}", None
    except Exception as err:
        return False, f"Unexpected error reading image: {str(err)}", None


def get_image_info(img: Image.Image, raw_bytes_len: Optional[int] = None) -> Dict[str, Any]:
    """
    Extract comprehensive image metadata for display and analysis.

    Args:
        img (Image.Image): PIL Image instance.
        raw_bytes_len (Optional[int]): File size in bytes if known.

    Returns:
        Dict[str, Any]: Dictionary containing image dimensions, format, mode, channels, etc.
    """
    width, height = img.size
    mode = img.mode

    # Determine channel count based on mode
    channel_map = {
        "1": 1,      # 1-bit pixels, black and white
        "L": 1,      # 8-bit pixels, grayscale
        "P": 1,      # 8-bit pixels, mapped to any other mode using a color palette
        "RGB": 3,    # 3x8-bit pixels, true color
        "RGBA": 4,   # 4x8-bit pixels, true color with transparency mask
        "CMYK": 4,   # 4x8-bit pixels, color separation
        "YCbCr": 3,  # 3x8-bit pixels, color video format
        "LAB": 3,    # 3x8-bit pixels, the L*a*b color space
        "HSV": 3,    # 3x8-bit pixels, Hue, Saturation, Value color space
        "I": 1,      # 32-bit signed integer pixels
        "F": 1,      # 32-bit floating point pixels
    }
    channels = channel_map.get(mode, len(img.getbands()) if hasattr(img, "getbands") else 3)
    aspect_ratio = calculate_aspect_ratio(width, height)

    # Estimate memory footprint
    memory_footprint_bytes = width * height * channels

    return {
        "width": width,
        "height": height,
        "aspect_ratio": aspect_ratio,
        "format": img.format or "RAW / In-Memory",
        "mode": mode,
        "channels": channels,
        "file_size_bytes": raw_bytes_len or memory_footprint_bytes,
        "file_size_formatted": format_bytes(raw_bytes_len or memory_footprint_bytes),
        "megapixels": round((width * height) / 1_000_000, 2),
    }


def ensure_rgb(image: Image.Image) -> Image.Image:
    """
    Safely convert any PIL Image to standard RGB mode (3 channels).
    If the image has an Alpha transparency channel (RGBA), it composites onto
    a clean white background to prevent dark artifacts.

    Args:
        image (Image.Image): Source PIL image.

    Returns:
        Image.Image: RGB mode PIL image.
    """
    if image.mode == "RGB":
        return image.copy()

    if image.mode == "RGBA":
        # Create a white background and composite alpha over it
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])  # Split index 3 is Alpha channel
        return background

    if image.mode == "LA":
        # Grayscale with Alpha
        rgba = image.convert("RGBA")
        return ensure_rgb(rgba)

    # Grayscale, palette, etc.
    return image.convert("RGB")


def ensure_rgba(image: Image.Image) -> Image.Image:
    """
    Safely convert any PIL Image to RGBA mode (4 channels with alpha).

    Args:
        image (Image.Image): Source PIL image.

    Returns:
        Image.Image: RGBA mode PIL image.
    """
    if image.mode == "RGBA":
        return image.copy()
    return image.convert("RGBA")


def pil_to_cv2(image: Image.Image) -> np.ndarray:
    """
    Convert a PIL Image to an OpenCV NumPy array.
    Accurately handles RGB -> BGR, RGBA -> BGRA, and Grayscale (L).

    Args:
        image (Image.Image): PIL Image.

    Returns:
        np.ndarray: OpenCV compatible uint8 array.
    """
    # Ensure standard mode
    if image.mode == "RGBA":
        arr = np.array(image, dtype=np.uint8)
        # Convert RGBA to BGRA
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)
    elif image.mode == "L":
        # Grayscale single channel
        return np.array(image, dtype=np.uint8)
    else:
        # Standard RGB
        rgb_img = ensure_rgb(image)
        arr = np.array(rgb_img, dtype=np.uint8)
        # Convert RGB to BGR for OpenCV
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def cv2_to_pil(image_array: np.ndarray) -> Image.Image:
    """
    Convert an OpenCV NumPy array back to a PIL Image.
    Accurately handles BGR -> RGB, BGRA -> RGBA, and Grayscale.

    Args:
        image_array (np.ndarray): OpenCV array.

    Returns:
        Image.Image: PIL Image ready for display or further processing.
    """
    if image_array is None or not isinstance(image_array, np.ndarray):
        raise ValueError("Invalid image array provided to cv2_to_pil.")

    # Ensure array is contiguous and uint8
    if image_array.dtype != np.uint8:
        # Clip and convert to uint8 if float or int32
        image_array = np.clip(image_array, 0, 255).astype(np.uint8)

    # Check dimensions
    if len(image_array.shape) == 2:
        # Single channel grayscale
        return Image.fromarray(image_array, mode="L")
    elif len(image_array.shape) == 3:
        channels = image_array.shape[2]
        if channels == 1:
            return Image.fromarray(image_array.squeeze(axis=2), mode="L")
        elif channels == 3:
            # OpenCV BGR -> PIL RGB
            rgb_arr = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            return Image.fromarray(rgb_arr, mode="RGB")
        elif channels == 4:
            # OpenCV BGRA -> PIL RGBA
            rgba_arr = cv2.cvtColor(image_array, cv2.COLOR_BGRA2RGBA)
            return Image.fromarray(rgba_arr, mode="RGBA")

    raise ValueError(f"Unsupported image array shape: {image_array.shape}")


def image_to_bytes(image: Image.Image, format: str = "PNG", quality: int = 95) -> bytes:
    """
    Export a PIL Image to in-memory bytes for download or streaming.

    Args:
        image (Image.Image): PIL Image.
        format (str): 'PNG', 'JPEG', or 'WEBP'.
        quality (int): Compression quality (1-100) for JPEG and WEBP.

    Returns:
        bytes: Compressed image binary data.
    """
    format = format.upper()
    if format not in SUPPORTED_FORMATS:
        format = "PNG"

    buffer = io.BytesIO()

    # Handle JPEG transparency compatibility
    if format == "JPEG":
        rgb_img = ensure_rgb(image)
        rgb_img.save(buffer, format="JPEG", quality=quality, optimize=True)
    elif format == "WEBP":
        image.save(buffer, format="WEBP", quality=quality, method=6)
    else:  # PNG
        # PNG is lossless, optimize=True for smaller footprint
        image.save(buffer, format="PNG", optimize=True)

    buffer.seek(0)
    return buffer.getvalue()
