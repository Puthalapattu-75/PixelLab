"""
PixelLab - Image Transformations Module
Provides geometric transformations:
- Resizing (with aspect ratio preservation and dimension validation)
- Cropping (with boundary validation)
- Fixed rotations (90°, 180°, 270°)
- Custom arbitrary rotation (with canvas expansion toggle)
- Horizontal flip (left-to-right mirror)
- Vertical flip (top-to-bottom flip)
"""

from typing import Tuple
from PIL import Image


def resize_image(
    image: Image.Image,
    target_width: int,
    target_height: int,
    maintain_aspect_ratio: bool = False,
) -> Image.Image:
    """
    Resize an image with boundary validation and optional aspect ratio locking.

    How it works:
    Uses Lanczos resampling (high-quality sinc interpolation) to smoothly interpolate
    pixels when downscaling or upscaling, preserving sharp details.

    Args:
        image (Image.Image): Input PIL Image.
        target_width (int): Desired width in pixels (must be >= 1).
        target_height (int): Desired height in pixels (must be >= 1).
        maintain_aspect_ratio (bool): If True, re-calculates dimensions to preserve original ratio.

    Returns:
        Image.Image: Resized PIL Image.

    Raises:
        ValueError: If width or height is less than 1.
    """
    orig_w, orig_h = image.size

    # Validate input dimensions
    if target_width < 1 or target_height < 1:
        raise ValueError(
            f"Invalid dimensions: width ({target_width}) and height ({target_height}) must be >= 1 pixel."
        )

    # Sanity limit to prevent memory crash from exorbitant dimensions
    MAX_DIM = 10000
    if target_width > MAX_DIM or target_height > MAX_DIM:
        raise ValueError(f"Target dimension exceeds safety limit of {MAX_DIM} pixels.")

    if maintain_aspect_ratio:
        aspect = orig_w / orig_h
        # Determine whether width or height should drive scaling
        scale_w = target_width / orig_w
        scale_h = target_height / orig_h
        scale = min(scale_w, scale_h)

        new_w = max(1, int(orig_w * scale))
        new_h = max(1, int(orig_h * scale))
    else:
        new_w = int(target_width)
        new_h = int(target_height)

    return image.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)


def crop_image(
    image: Image.Image,
    left: int,
    top: int,
    right: int,
    bottom: int,
) -> Image.Image:
    """
    Crop an image to the specified bounding box coordinates.

    Coordinates format: (left, top, right, bottom) where:
    - left: X coordinate of top-left corner
    - top: Y coordinate of top-left corner
    - right: X coordinate of bottom-right corner
    - bottom: Y coordinate of bottom-right corner

    Args:
        image (Image.Image): Input PIL Image.
        left (int): Left boundary pixel.
        top (int): Top boundary pixel.
        right (int): Right boundary pixel.
        bottom (int): Bottom boundary pixel.

    Returns:
        Image.Image: Cropped PIL Image.

    Raises:
        ValueError: If crop box is invalid or outside image boundaries.
    """
    w, h = image.size

    # Validate coordinate ordering
    if int(left) >= int(right):
        raise ValueError(f"Invalid crop box: left ({left}) must be strictly less than right ({right}).")
    if int(top) >= int(bottom):
        raise ValueError(f"Invalid crop box: top ({top}) must be strictly less than bottom ({bottom}).")

    # Clamp coordinates to valid image bounds [0, w] and [0, h]
    l = max(0, min(int(left), w - 1))
    t = max(0, min(int(top), h - 1))
    r = max(l + 1, min(int(right), w))
    b = max(t + 1, min(int(bottom), h))

    if l >= r or t >= b:
        raise ValueError(
            f"Invalid crop bounds after clamping: left ({l}) must be < right ({r}), and top ({t}) must be < bottom ({b})."
        )

    return image.crop((l, t, r, b))


def rotate_90_clockwise(image: Image.Image) -> Image.Image:
    """
    Rotate image 90 degrees clockwise.

    Args:
        image (Image.Image): Input PIL Image.

    Returns:
        Image.Image: 90° clockwise rotated image.
    """
    return image.transpose(Image.Transpose.ROTATE_270)


def rotate_180(image: Image.Image) -> Image.Image:
    """
    Rotate image 180 degrees.

    Args:
        image (Image.Image): Input PIL Image.

    Returns:
        Image.Image: 180° rotated image.
    """
    return image.transpose(Image.Transpose.ROTATE_180)


def rotate_270_clockwise(image: Image.Image) -> Image.Image:
    """
    Rotate image 270 degrees clockwise (equivalent to 90 degrees counter-clockwise).

    Args:
        image (Image.Image): Input PIL Image.

    Returns:
        Image.Image: 270° clockwise rotated image.
    """
    return image.transpose(Image.Transpose.ROTATE_90)


def rotate_custom(image: Image.Image, angle: float, expand: bool = True) -> Image.Image:
    """
    Rotate image by any arbitrary angle in degrees (clockwise positive).

    How it works:
    Rotates around the image center using Bicubic interpolation.
    If expand=True, the canvas is resized to accommodate all rotated corners
    without clipping.

    Args:
        image (Image.Image): Input PIL Image.
        angle (float): Angle in degrees (clockwise positive, -180 to +180).
        expand (bool): Whether to expand the canvas to fit the full rotated image.

    Returns:
        Image.Image: Rotated PIL Image.
    """
    if angle == 0 or angle == 360:
        return image.copy()

    # PIL rotate uses counter-clockwise positive, so we negate angle for intuitive clockwise UI
    return image.rotate(-float(angle), resample=Image.Resampling.BICUBIC, expand=expand)


def flip_horizontal(image: Image.Image) -> Image.Image:
    """
    Mirror the image horizontally along the vertical axis (left-to-right).

    Args:
        image (Image.Image): Input PIL Image.

    Returns:
        Image.Image: Horizontally flipped image.
    """
    return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)


def flip_vertical(image: Image.Image) -> Image.Image:
    """
    Flip the image vertically along the horizontal axis (upside down).

    Args:
        image (Image.Image): Input PIL Image.

    Returns:
        Image.Image: Vertically flipped image.
    """
    return image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
